"""`verify`: what did conversion cost?

Runs standalone -- no job spec, no run directory, no prior stage, on an artifact
litetune did not produce. Given a `.litertlm` file, a float reference and a
held-out split, it produces a manifest from which the release decision can be
reconstructed months later.

The order of operations is the design:

1. read the held-out split (its content, not its path, identifies it)
2. generate with the candidate
3. run the label-free liveness tier on it -- and stop here if it does not pass,
   because every comparison after this point would be over output whose
   production is not established
4. only then generate with the reference, and run the same tier on *it*: a
   reference that did not generate makes the comparison invalid, which is a
   harness result and not a verdict about the candidate
5. refuse the comparison outright if the two sides were not measured the same
   way -- prompt mode, decoding, split
6. score both, and report training gain and conversion cost as separate
   quantities or as explicitly unavailable

Statuses distinguish "the model is bad" from "we could not tell". A run with no
labelled data is `unmeasured`, never `passed`: the liveness tier once passed a
model scoring nine times worse than its own base, so it can establish that a
model is alive and can never establish that it is right.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from litetune import models
from litetune.checks import Check, Outcome, guard
from litetune.evaluate import (
    GREEDY,
    DecodeConfig,
    GenerationBackend,
    HuggingFaceBackend,
    LiteRtLmBackend,
    PromptMode,
    PromptModeDecision,
    Split,
    evaluate,
    harness_mismatch,
    load_split,
    resolve_prompt_mode,
)
from litetune.events import EventStream
from litetune.liveness import (
    DEFAULT_THRESHOLDS,
    LivenessResult,
    LivenessThresholds,
    SkippedCheck,
    divergence_check,
    liveness_tier,
    unterminated_count,
)
from litetune.metrics import (
    Difference,
    Proportion,
    QualityMetrics,
    Unavailable,
    agreement,
    paired_difference,
    score,
)

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "litetune.verify/1"

# Below this the interval swamps what the tool exists to measure: at n=200 the
# Wald half-width on a 0.91 proportion is ±0.0398, against a measured recipe
# effect of 0.024. At n=64 it is ±0.0703, which is where three conclusions were
# drawn and later overturned. This is a warning, not a gate -- a small split is
# still worth running, as long as nobody reads a difference off it.
MIN_HELDOUT_EXAMPLES = 200

# A baseline at or below this scores so close to zero that any improvement over
# it is expected by construction rather than evidence.
NEAR_ZERO = 0.05

UNMEASURED_REASON = (
    "no labelled held-out data was supplied: quality is unmeasured, which is not the same as "
    "verified. The liveness tier establishes only that the model is alive -- a run scoring nine "
    "times worse than its own base passed every check in it."
)


class Status(str, Enum):
    PASSED = "passed"
    FAILED_SMOKE = "failed_smoke"
    FAILED_GATE = "failed_gate"
    INCONCLUSIVE = "inconclusive"
    FAILED_HARNESS = "failed_harness"
    # Not a pass. The run completed and the quantity in question was never
    # established -- either quality was not measured at all, or it was
    # measured and could not be attributed to a cause.
    UNMEASURED = "unmeasured"
    ERROR = "error"


# Exit codes keep "could not check" distinguishable from "failed" at the shell,
# which is the whole thesis of this tool applied to its own interface.
EXIT_CODES: dict[Status, int] = {
    Status.PASSED: 0,
    Status.FAILED_SMOKE: 1,
    Status.FAILED_GATE: 1,
    Status.INCONCLUSIVE: 2,
    Status.UNMEASURED: 3,
    Status.FAILED_HARNESS: 4,
    Status.ERROR: 4,
}


class ReferenceRole(str, Enum):
    """What the reference model is, which decides what can be attributed.

    The two are not interchangeable. Against the float twin -- the same weights
    before conversion -- the difference *is* the conversion cost. Against an
    untuned base it confounds training with conversion, and neither quantity can
    be separated out of a single number.
    """

    FLOAT_TWIN = "float_twin"
    UNTUNED_BASE = "untuned_base"


@dataclass(frozen=True)
class VerifyRequest:
    model: Path
    reference: str
    data: Path
    limit: int | None = None
    reference_role: ReferenceRole = ReferenceRole.FLOAT_TWIN
    decode: DecodeConfig = GREEDY
    thresholds: LivenessThresholds = DEFAULT_THRESHOLDS
    max_conversion_cost: float | None = None
    # How the prompt reaching the model is built. Declared by the caller wins;
    # otherwise `contract` -- the bundle the artifact shipped with, which is
    # where `tune`'s decision is written down -- and otherwise it is inferred
    # from the split and reported as inferred. `run_verify` fills this in before
    # `build_backends` is called, so a backend never has to fall back to a
    # default nobody chose.
    prompt_mode: PromptMode | None = None
    contract: Path | None = None


@dataclass(frozen=True)
class BackendPair:
    candidate: GenerationBackend
    reference: GenerationBackend


def build_backends(request: VerifyRequest) -> BackendPair:
    """The real backends. Tests pass their own pair to `run_verify` instead.

    `request.prompt_mode` has been resolved by the time this is called, so both
    sides are configured from one decision: the runtime gets `--no-template`
    only when the prompts are pre-rendered, and the reference applies its chat
    template only when they are not.
    """
    return BackendPair(
        candidate=LiteRtLmBackend(
            model=request.model, decode=request.decode, declared_prompt_mode=request.prompt_mode
        ),
        reference=HuggingFaceBackend(
            model=request.reference, decode=request.decode, declared_prompt_mode=request.prompt_mode
        ),
    )


CONTRACT_CHECK = "prompt-rendering mode is known"

INFERRED_PROMPT_MODE = (
    "the prompt-rendering mode was not declared and no bundle contract was supplied, so it was "
    "inferred from the held-out prompts: {evidence}. This is a guess about a calling convention. "
    "`--no-template` is narrow, not general -- it routes the runtime to create_session() "
    "instead of create_conversation(), bypassing the chat template, the <|turn>model anchor, "
    "tool handling and channel extraction, and its own help says 'the input should include "
    "all control tokens for the "
    "model expected'. It is right only when the caller built the whole prompt, which is what "
    "training decided. Pass the mode explicitly, or point --contract at the bundle this model "
    "shipped with"
)


def contract_prompt_mode(path: Path) -> PromptMode:
    """The mode recorded in a bundle contract. Raises if it is not readable.

    `tune` decides the mode, `bundle` writes it here, and this reads it back:
    the whole point of that field is that nothing downstream has to guess.
    """
    # Imported here rather than at module scope: `bundle` is downstream of
    # `verify` in the pipeline, and a top-level import would point the
    # dependency backwards.
    from litetune.bundle import Contract

    payload = json.loads(path.read_text(encoding="utf-8"))
    return Contract.read(payload).prompt_mode


def _resolve_mode(request: VerifyRequest, split: Split) -> PromptModeDecision:
    """Declared, then the contract, then the prompts. Never a silent default."""
    contract_mode = contract_prompt_mode(request.contract) if request.contract is not None else None
    return resolve_prompt_mode(split.prompts, declared=request.prompt_mode, contract=contract_mode)


def _model_rules(request: VerifyRequest) -> tuple[str, models.ModelRules | None]:
    """Which model the family rules were looked up for, and what they are.

    The reference is asked first: it is a checkpoint id, where the family is
    written down, while the artifact under test is a `.litertlm` path that may be
    called anything at all.
    """
    for candidate in (request.reference, str(request.model)):
        rules = models.identify(candidate)
        if rules is not None:
            return candidate, rules
    return request.reference, None


@dataclass
class VerifyResult:
    status: Status
    manifest: dict[str, Any]

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]


def _tool_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    from litetune._version import __version__

    try:
        installed = version("litetune")
    except PackageNotFoundError:
        # Running from a checkout, a vendored copy or a zipapp. The module knows
        # its own version; only the distribution metadata is missing. Recording
        # "unknown" here put an unusable value in the field the whole manifest
        # exists to make trustworthy.
        return __version__
    if installed != __version__:
        logger.warning(
            "installed litetune distribution reports %s, package says %s; recording the package",
            installed,
            __version__,
        )
    return __version__


def _model_provenance(path: Path) -> dict[str, Any]:
    """Identify the artifact under test well enough to recognise it later."""
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not record["exists"]:
        return record
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        record["bytes"] = path.stat().st_size
        record["sha256"] = digest.hexdigest()
    except OSError as exc:
        # Provenance is what lets a disputed result be reconstructed, so a
        # failure to read it is recorded rather than silently omitted.
        logger.exception("could not hash %s", path)
        record["sha256"] = None
        record["provenance_error"] = f"{type(exc).__name__}: {exc}"
    return record


@dataclass
class _Run:
    """Mutable state of one verification, assembled into a manifest at the end."""

    request: VerifyRequest
    events: EventStream
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool_version": _tool_version(),
            "status": None,
            "model": _model_provenance(self.request.model),
            "reference": {
                "ref": self.request.reference,
                "role": self.request.reference_role.value,
            },
            "harness": {
                "decode_requested": self.request.decode.as_dict(),
                "liveness_thresholds": self.request.thresholds.as_dict(),
            },
            "checks": [],
            "measurements": {},
            "liveness": {},
            "quality": Unavailable("verification did not reach the quality tier").as_dict(),
            "attribution": {},
            "gates": [],
            "limitations": [],
        }

    def record(self, check: Check) -> Check:
        self.manifest["checks"].append(check.as_dict())
        self.events.check(check)
        return check

    def limitation(self, text: str) -> None:
        self.manifest["limitations"].append(text)

    def finish(self, status: Status) -> VerifyResult:
        self.manifest["status"] = status.value
        self.events.stage_finished(status.value)
        return VerifyResult(status=status, manifest=self.manifest)


def run_verify(
    request: VerifyRequest,
    events: EventStream | None = None,
    backends: BackendPair | None = None,
) -> VerifyResult:
    """Verify one converted artifact. Returns a status and a manifest; never raises."""
    events = events or EventStream(echo_json=False)
    events.stage_started("verify", model=str(request.model), reference=request.reference)
    run = _Run(request=request, events=events)

    # -- the split ---------------------------------------------------------
    with guard("held-out data") as sink:
        split = load_split(request.data, limit=request.limit)
        sink.append(
            Check.passed(
                "held-out data",
                f"{split.n} examples ({len(split.labelled)} labelled), content {split.id}",
                observed=split.as_dict(),
            )
        )
    if not run.record(sink[0]).conclusive:
        return run.finish(Status.ERROR)

    run.manifest["data"] = split.as_dict()
    if split.n < MIN_HELDOUT_EXAMPLES:
        run.limitation(
            f"held-out split has {split.n} examples, below {MIN_HELDOUT_EXAMPLES}: intervals here "
            "are wide enough to swamp the effects this tool measures, so treat differences as "
            "unresolved unless the interval says otherwise"
        )

    # -- which calling convention is this measurement taken in? ------------
    # Resolved before the backends are built, so both sides are configured from
    # one decision and neither falls back to a default nobody made.
    with guard(CONTRACT_CHECK) as sink:
        decision = _resolve_mode(request, split)
    if sink:
        # A contract was named and could not be read. Falling back to inference
        # here would be a silent substitution for a mode the caller had told us
        # where to find.
        run.record(sink[0])
        return run.finish(Status.FAILED_HARNESS)

    run.manifest["harness"]["prompt_mode_decision"] = decision.as_dict()
    events.note(
        f"prompt mode {decision.mode.value} ({decision.source}): {decision.evidence}",
        prompt_mode=decision.mode.value,
        source=decision.source,
    )
    if decision.inferred:
        run.limitation(INFERRED_PROMPT_MODE.format(evidence=decision.evidence))
    request = replace(request, prompt_mode=decision.mode)

    # -- what does litetune know about this model family? ------------------
    rules_for, rules = _model_rules(request)
    run.manifest["model_rules"] = models.report(rules_for)
    for text in rules.limitations if rules is not None else ():
        run.limitation(text)

    pair = backends or build_backends(request)

    # -- the candidate -----------------------------------------------------
    # An empty sink means the evaluator returned; anything in it is the reason
    # it did not, and that is `could not check`, not a verdict.
    with guard("run candidate model") as sink:
        candidate = evaluate(pair.candidate, split, label="candidate", events=events)
    if sink:
        run.record(sink[0])
        return run.finish(Status.FAILED_HARNESS)
    run.manifest["measurements"]["candidate"] = candidate.as_dict()
    if candidate.prompt_mode is not decision.mode:
        # Only reachable when the caller supplied their own backends:
        # `build_backends` configures both sides from the decision. The
        # measurement's own mode is the one that describes the numbers, so the
        # manifest carries both rather than letting them silently disagree.
        run.limitation(
            f"the prompt mode resolved as {decision.mode.value} ({decision.source}) but the "
            f"candidate was measured {candidate.prompt_mode.value}: the backend was supplied by "
            "the caller and did not take the resolved mode. The mode recorded against the "
            "measurement is the one the numbers were produced in"
        )
    run.limitation(
        f"measured on the {candidate.engine.get('backend', 'unknown')} backend of "
        f"{candidate.engine.get('engine', 'unknown')}; published reports put the GPU backend "
        "materially below CPU on identical artifacts, so this is an optimistic estimate of "
        "on-device behaviour"
    )

    # Check 5 (divergence) is deferred: the caller decides whether it applies,
    # and generating the reference before the candidate is known alive would pay
    # for a comparison that must not be made.
    live = liveness_tier(
        candidate,
        thresholds=request.thresholds,
        baseline=None,
        baseline_absent_reason=None,
        events=events,
    )
    run.manifest["liveness"]["candidate"] = live.as_dict()
    if live.outcome is Outcome.UNCHECKED:
        return run.finish(Status.FAILED_HARNESS)
    if live.outcome is Outcome.FAILED:
        return run.finish(Status.FAILED_SMOKE)

    # -- do we need the reference? ----------------------------------------
    labelled = split.labelled
    needs_reference = bool(labelled) or request.reference_role is ReferenceRole.UNTUNED_BASE
    if not needs_reference:
        _skip_divergence(live)
        run.manifest["liveness"]["candidate"] = live.as_dict()
        run.manifest["measurements"]["reference"] = Unavailable(
            "no labelled examples, so the reference was not run"
        ).as_dict()
        run.manifest["quality"] = Unavailable(UNMEASURED_REASON).as_dict()
        run.manifest["attribution"] = _unattributable("quality was not measured on either side")
        return run.finish(Status.UNMEASURED)

    # -- can the reference's environment even load this model? -------------
    # Checked here rather than earlier because it is a fact about the reference
    # side: a version too old raises `AttributeError: 'list' object has no
    # attribute 'keys'` from inside a tokenizer load, and a named check is worth
    # more than that traceback.
    if rules is not None and rules.min_transformers:
        version_check = models.transformers_check(
            rules_for,
            rules,
            models.declared_version(pair.reference.describe().get("requirements") or ()),
            f"the reference backend ({pair.reference.name})",
            unknown_reason="it declares no pinned transformers version",
        )
        run.record(version_check)
        if version_check.outcome is Outcome.FAILED:
            return run.finish(Status.FAILED_HARNESS)
        if not version_check.conclusive:
            run.limitation(version_check.detail)

    with guard("run reference model") as sink:
        reference = evaluate(pair.reference, split, label="reference", events=events)
    if sink:
        run.record(sink[0])
        return run.finish(Status.FAILED_HARNESS)
    run.manifest["measurements"]["reference"] = reference.as_dict()

    # The reference is one side of the comparison; if it did not generate, the
    # comparison is unavailable rather than the candidate being at fault.
    ref_live = liveness_tier(
        reference,
        thresholds=request.thresholds,
        baseline=None,
        baseline_absent_reason="the reference is the baseline; it is not compared against itself",
        events=events,
    )
    run.manifest["liveness"]["reference"] = ref_live.as_dict()
    if not ref_live.passed:
        run.record(
            Check.unchecked(
                "reference is comparable",
                "the reference model's own liveness tier did not pass "
                f"({ref_live.outcome.value}); the comparison was not attempted",
                observed=ref_live.as_dict(),
            )
        )
        return run.finish(Status.FAILED_HARNESS)

    # -- were both sides measured the same way? ---------------------------
    mismatch = harness_mismatch(candidate, reference)
    if mismatch is not None:
        run.record(Check.failed("comparable measurement", mismatch))
        run.manifest["harness"]["equivalent"] = False
        run.manifest["harness"]["mismatch"] = mismatch
        run.manifest["attribution"] = _unattributable(mismatch)
        return run.finish(Status.FAILED_HARNESS)
    run.record(
        Check.passed(
            "comparable measurement",
            f"both sides: {candidate.prompt_mode.value} prompts, split {candidate.split_id}, "
            f"{candidate.decode.fingerprint}",
        )
    )
    for point in (candidate, reference):
        if point.batch_failures:
            run.limitation(
                f"{point.batch_failures} of {point.n} {point.label} generations came from a "
                f"process that exited non-zero after writing its results; the outputs exist and "
                "are scored, but the run that produced them did not end cleanly"
            )
    run.manifest["harness"]["equivalent"] = True
    run.manifest["harness"]["prompt_mode"] = candidate.prompt_mode.value
    # The typed field, not the engine dict: read from `describe()` this
    # compared `None != None` for any backend that omitted the key, so the
    # limitation silently vanished for exactly the third-party backend that
    # most needs it.
    if candidate.decode_enforced != reference.decode_enforced:
        # Both sides declare the same decoding, but only one of them was handed
        # it: litetune passes none to the runtime, so it runs on its own
        # defaults. They are greedy on both sides, which is what makes the
        # comparison admissible at all -- but a differing token limit would show
        # as truncation on one side only, so it is named rather than assumed
        # away.
        # How far the unverified limit could reach is measurable even though the
        # limit is not: a generation that ends with a terminator stopped on its
        # own and no bound touched it. The count is not proof -- a runtime may
        # strip the terminator before we see the text -- but it turns a standing
        # caveat into a number that moves, and a run where it is large is a run
        # whose model is not stopping.
        loose = unterminated_count(candidate)
        run.limitation(
            f"decoding was passed explicitly to {reference.backend} "
            f"({reference.decode.as_dict()}) but not to {candidate.backend}, which uses the "
            "pinned runtime's own defaults; both are greedy, and the token limit is unverified "
            f"on the runtime side. {loose} of {candidate.n} {candidate.backend} generations end "
            "without a terminator"
        )

    # -- liveness check 5, now that a baseline exists ----------------------
    if request.reference_role is ReferenceRole.UNTUNED_BASE:
        check = divergence_check(candidate, reference.texts, "the untuned base", request.thresholds)
        live.checks.add(check)
        events.check(check)
        run.manifest["liveness"]["candidate"] = live.as_dict()
        if check.outcome is Outcome.FAILED:
            return run.finish(Status.FAILED_SMOKE)
        if check.outcome is Outcome.UNCHECKED:
            return run.finish(Status.FAILED_HARNESS)
    else:
        _skip_divergence(live)
        run.manifest["liveness"]["candidate"] = live.as_dict()

    if not labelled:
        run.manifest["quality"] = Unavailable(UNMEASURED_REASON).as_dict()
        run.manifest["attribution"] = _unattributable(
            "the split carries no targets, so nothing was scored"
        )
        return run.finish(Status.UNMEASURED)

    # -- quality -----------------------------------------------------------
    with guard("quality measured") as sink:
        indices = [e.index for e in labelled]
        targets = [e.target for e in labelled if e.target is not None]
        candidate_metrics = score(targets, [candidate.generations[i].text for i in indices])
        reference_metrics = score(targets, [reference.generations[i].text for i in indices])
        agreed = agreement(candidate.texts, reference.texts)
        sink.append(
            Check.passed(
                "quality measured",
                f"scored {candidate_metrics.n} labelled examples on both sides",
                observed={"n": candidate_metrics.n},
            )
        )
    if not run.record(sink[0]).conclusive:
        return run.finish(Status.FAILED_HARNESS)

    run.manifest["quality"] = {
        "available": True,
        "candidate": candidate_metrics.as_dict(),
        "reference": reference_metrics.as_dict(),
        "agreement_with_reference": agreed.as_dict(),
        "note": (
            "agreement is label-free and is not a quality claim on its own; it is reported "
            "only because both sides passed the liveness tier first"
        ),
    }
    _emit_metrics(events, candidate_metrics, reference_metrics)

    if reference_metrics.exact_match.value <= NEAR_ZERO:
        run.limitation(
            f"the reference scores {reference_metrics.exact_match.value:.4f}, at or near zero: "
            "a difference against it is expected by construction and is not evidence about the "
            "candidate's absolute quality"
        )

    cost, gain = _attribute(request, candidate_metrics, reference_metrics)
    run.manifest["attribution"] = {
        "conversion_cost": cost.as_dict(),
        "training_gain": gain.as_dict(),
        "sign": "positive conversion_cost means the converted model scores below the reference",
    }
    if isinstance(cost, Difference):
        events.metric("conversion_cost", cost.value, ci95=cost.ci95, resolved=cost.resolved)
        if not cost.resolved:
            run.limitation(cost.detail)

    return run.finish(_gate(run, request, cost))


def _skip_divergence(live: LivenessResult) -> None:
    live.skipped.append(
        SkippedCheck(
            "divergence from baseline",
            "the reference is the float twin of the model under test; agreement with it is the "
            "desired outcome of a lossless conversion, so divergence from it is not evidence of "
            f"life (pass --reference-role {ReferenceRole.UNTUNED_BASE.value} to check it)",
        )
    )


def _unattributable(reason: str) -> dict[str, Any]:
    return {
        "conversion_cost": Unavailable(reason).as_dict(),
        "training_gain": Unavailable(reason).as_dict(),
    }


def _attribute(
    request: VerifyRequest,
    candidate: QualityMetrics,
    reference: QualityMetrics,
) -> tuple[Difference | Unavailable, Difference | Unavailable]:
    """Split the observed difference into what conversion cost and what training gained.

    Standalone `verify` has two measurement points, not three, so exactly one of
    these can ever be a number. Deriving the other from the points that exist is
    the mistake this returns `Unavailable` to avoid.
    """
    if request.reference_role is ReferenceRole.FLOAT_TWIN:
        # Paired, because both sides answered the same prompts: see
        # metrics.paired_difference for why the unpaired interval is too blunt
        # to resolve a conversion cost of the size this tool reports.
        cost = paired_difference(reference.correct, candidate.correct)
        gain = Unavailable(
            "no untuned baseline was measured; supply the base model's float score to attribute "
            "training gain"
        )
        return cost, gain

    confounded = (
        "the reference is the untuned base, so the difference between it and the converted "
        "model confounds training with conversion; measure the float twin of the model under "
        "test to separate them"
    )
    return Unavailable(confounded), Unavailable(confounded)


def _emit_metrics(
    events: EventStream, candidate: QualityMetrics, reference: QualityMetrics
) -> None:
    for label, metrics in (("candidate", candidate), ("reference", reference)):
        for name in ("exact_match", "name_accuracy", "argument_accuracy", "parse_rate"):
            value = getattr(metrics, name)
            if isinstance(value, Proportion):
                events.metric(f"{label}.{name}", value.value, ci95=value.ci95, n=value.n)
            else:
                events.note(f"{label}.{name}: not available — {value.reason}")


def _gate(run: _Run, request: VerifyRequest, cost: Difference | Unavailable) -> Status:
    """Apply the configured gate, if any."""
    if request.max_conversion_cost is None:
        if isinstance(cost, Unavailable):
            # Nothing was attributed, so there is nothing for a threshold to
            # have judged. Returning PASSED here made the presence of a
            # threshold decide whether the run counted as established: the same
            # `--reference-role untuned_base` run exited 0 without one and
            # FAILED_HARNESS with one. `unmeasured` is the status for "the run
            # completed and the quantity was never established", and it is what
            # this is.
            run.manifest["gates"].append(
                {
                    "name": "conversion_cost",
                    "outcome": Outcome.UNCHECKED.value,
                    "detail": f"nothing to gate: {cost.reason}",
                }
            )
            return Status.UNMEASURED
        run.manifest["gates"].append(
            {
                "name": "conversion_cost",
                "outcome": Outcome.UNCHECKED.value,
                "detail": "no threshold configured; 'passed' here means the model was measured, "
                "not that it met a bar",
            }
        )
        return Status.PASSED

    if isinstance(cost, Unavailable):
        run.record(
            Check.unchecked(
                "conversion cost within threshold",
                f"threshold {request.max_conversion_cost} could not be applied: {cost.reason}",
            )
        )
        return Status.FAILED_HARNESS

    threshold = request.max_conversion_cost
    low, high = cost.value - cost.ci95, cost.value + cost.ci95
    observed = {"threshold": threshold, "value": cost.value, "ci95": cost.ci95}
    name = "conversion cost within threshold"
    # The comparison is between the *interval* and the threshold, not between
    # the interval's width and the threshold. Those differ, and the difference
    # matters in both directions: a 0.98 cost measured to ±0.24 clears a 0.05
    # threshold decisively even though the interval is five times wider than it,
    # while a 0.03 cost measured to ±0.04 does not settle a 0.01 threshold at
    # all. The second case is the n=64 mistake; the first would be its mirror
    # image, refusing to fail an obviously failing model.
    if low > threshold:
        check = Check.failed(
            name,
            f"conversion cost {cost.value:+.4f} ±{cost.ci95:.4f} is entirely above "
            f"{threshold:.4f}",
            observed=observed,
        )
        status = Status.FAILED_GATE
    elif high <= threshold:
        check = Check.passed(
            name,
            f"conversion cost {cost.value:+.4f} ±{cost.ci95:.4f} is within {threshold:.4f}",
            observed=observed,
        )
        status = Status.PASSED
    else:
        # The instrument is blunter than the question. Reporting a verdict here
        # is how a 0.172 estimate at n=64 became three conclusions that n=640
        # overturned.
        check = Check.unchecked(
            name,
            f"the interval {low:+.4f}..{high:+.4f} straddles the threshold {threshold:.4f}; "
            f"this split cannot resolve the gate",
            observed=observed,
        )
        status = Status.INCONCLUSIVE
    run.record(check)
    run.manifest["gates"].append(
        {"name": "conversion_cost", "outcome": check.outcome.value, "detail": check.detail}
        | observed
    )
    return status
