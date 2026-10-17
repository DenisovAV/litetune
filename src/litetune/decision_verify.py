"""`verify` for a decision checkpoint: option scores on both sides, calibration, the shipped file.

A decision model never generates in use. flutter_edge_ai's `createDecider`
scores every option's log-probability through the runtime and answers with the
best one, so that is what this measures, as `decision_scripts.py` describes:
- **the reference**, the tuned checkpoint in PyTorch float32, scores every
  option on every calibration and held-out row;
- **the calibration** (`calibration.py`) is fitted on the reference's
  calibration scores: a temperature, and for Choice the abstain threshold;
- **the runtime** scores the shipped `.litertlm` on the first `engine_rows`
  held-out rows, in the mode its self-check picks, and is compared with the
  reference on the same rows at the same calibration.

Scoring the runtime on every row costs too much: at 151 options a decision took
7.2 s on four threads of a server CPU (measured 2026-10-16), so 3,000 rows
would take six hours.

**Refused:** a runtime whose answers agree with the reference's on fewer than
95 % of the rows, a gross-failure check; a float twin whose per-token scores
differ from the reference's by more than 1e-3 nats; and a Choice question that
clearly misses its own rule on the held-out rows, by a one-sided 95 %
Clopper-Pearson bound.

**The section.** The calibration and the decision travel inside the bundle as
the `litetune.decision` section; this stage writes its payload into the
manifest, and `bundle` writes it into the file.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import subprocess
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import calibration, envs
from litetune.checks import Check, CheckSet, Outcome
from litetune.decision_scripts import ENGINE_SCRIPT, REFERENCE_SCRIPT
from litetune.decision_section import SECTION_NAME, payload_bytes
from litetune.decisions import CONTRACT_VERSION, Decision, DecisionError, decision_from
from litetune.events import EventStream
from litetune.metrics import paired_difference
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

SCORER = "option-scoring"
SCORER_MEANS = (
    "the option whose shortest unique prefix has the highest summed log-probability, as "
    "flutter_edge_ai's createDecider scores it; a Choice answer abstains below the fitted "
    "threshold or on the none_option"
)
SCORING = "unique_prefix"

DEFAULT_ENGINE_ROWS = 200
DEFAULT_THREADS = 4
PROBE_ROWS = 4
AGREEMENT_FLOOR = 0.95
TWIN_TOLERANCE = 1e-3
DEFAULT_TERMINATOR = "<end_of_turn>"
# The runtime the method was measured correct on, against PyTorch, on a float32
# bundle (2026-10-16). On another, only a float twin shows it still is.
MEASURED_RUNTIME = "0.18.0"

# Seconds: a start-up allowance, then a budget per row. At 151 options the
# runtime took 7.2 s a decision on a server CPU; a phone-class CPU is slower.
STARTUP_TIMEOUT_S = 900
REFERENCE_ROW_TIMEOUT_S = 30
ENGINE_ROW_TIMEOUT_S = 120

ROWS_CHECK = "rows are the decision's prompts"
REFERENCE_CHECK = "reference option scores"
CALIBRATION_CHECK = "calibration"
RUNTIME_CHECK = "runtime option scores"
TARGETS_CHECK = "both sides score the same targets"
AGREEMENT_CHECK = "runtime answers agree with the reference"
TWIN_CHECK = "float twin matches the reference"
RULE_CHECK = "the abstain rule holds on held-out rows"

NO_TWIN = (
    "no --float-twin was given, so the runtime's per-token scores were not compared with the "
    "reference's on a float32 export of this checkpoint. The method the self-check picked was "
    "measured within 8.2e-4 nats of PyTorch on LiteRT-LM {runtime} on 2026-10-16; convert with "
    "--recipe none as well and pass that file to check it on this checkpoint"
)
TEMPERATURE_CARRIED = (
    "the temperature and the threshold were fitted on the reference's calibration scores and "
    "are applied to the runtime's. The runtime's option distribution was within the mean KL "
    "reported here of the reference's; whether a temperature fitted on the runtime would differ "
    "is not measured"
)
TYPE_UNMEASURED = (
    "this is a {kind} question; only Choice questions have been trained and measured by option "
    "scoring"
)


class DecisionRowsError(ValueError):
    """A split that is not the decision's prompts. Names the file and line."""


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionRows:
    """One split's prompts and the option index each is trained to answer."""

    path: Path
    prompts: tuple[str, ...]
    targets: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "rows": len(self.prompts), "sha256": hash_file(self.path)}


def read_decision_rows(path: Path, decision: Decision) -> DecisionRows:
    """`prepare`'s split rows, each checked to be the decision rendered around a state."""
    suffix = decision.render("")
    prompts: list[str] = []
    targets: list[int] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DecisionRowsError(f"{path} could not be read: {type(exc).__name__}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DecisionRowsError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise DecisionRowsError(f"{path}:{lineno}: expected an object with a 'prompt'")
        prompt, target = row.get("prompt"), row.get("target", row.get("completion"))
        if not isinstance(prompt, str) or not prompt.endswith(suffix) or prompt == suffix:
            raise DecisionRowsError(
                f"{path}:{lineno}: the prompt is not a state rendered with the decision this "
                "checkpoint records. Measure the split `prepare` wrote for it"
            )
        if target not in decision.ids:
            raise DecisionRowsError(
                f"{path}:{lineno}: the target {target!r} is not one of the decision's options"
            )
        prompts.append(prompt)
        targets.append(decision.ids.index(target))
    if not prompts:
        raise DecisionRowsError(f"{path}: no rows")
    return DecisionRows(Path(path), tuple(prompts), tuple(targets))


# ---------------------------------------------------------------------------
# Running a scoring script
# ---------------------------------------------------------------------------


@dataclass
class ScoreRun:
    """What one script produced: per-row token scores, per-row errors, its report."""

    scores: dict[int, list[list[float]]] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    # Milliseconds per row, as the script measured it.
    ms: dict[int, float] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    failure: str | None = None


def run_script(
    env: envs.StageEnv,
    source: str,
    spec: dict[str, Any],
    timeout: int,
    auto_provision: bool = True,
    events: EventStream | None = None,
) -> ScoreRun:
    """Write `source` and `spec`, run it in `env`, and read back what it wrote."""
    run = ScoreRun()
    if auto_provision:
        try:
            env.provision(events=events)
        except Exception as exc:  # noqa: BLE001 - a failed environment is a harness failure
            logger.exception("could not provision environment %r", env.name)
            run.failure = f"the {env.name} environment could not be provisioned: {exc}"
            return run
    with tempfile.TemporaryDirectory(prefix="litetune-decision-") as tmp:
        work = Path(tmp)
        script = work / "score.py"
        script.write_text(source, encoding="utf-8")
        out, report = work / "scores.jsonl", work / "report.json"
        spec_path = work / "spec.json"
        spec_path.write_text(
            json.dumps({**spec, "out": str(out), "report": str(report)}), encoding="utf-8"
        )
        proc = None
        try:
            proc = env.run(["python", str(script), str(spec_path)], timeout=timeout)
        except subprocess.TimeoutExpired:
            run.failure = f"no result after {timeout} s (timeout)"
        if out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a kill
                if "error" in rec:
                    run.errors[rec["index"]] = str(rec["error"])
                elif "token_scores" in rec:
                    run.scores[rec["index"]] = rec["token_scores"]
                    run.ms[rec["index"]] = (
                        float(rec["ms"]) if "ms" in rec else float(rec.get("seconds", 0.0)) * 1000
                    )
        if report.exists():
            try:
                run.report = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                run.report = {}
        if proc is not None and proc.returncode != 0 and run.failure is None:
            said = run.report.get("error") or (proc.stderr or "").strip()[-2000:]
            run.failure = f"the script exited {proc.returncode}: {said}"
    return run


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


def unique_lengths(cont_ids: Sequence[Sequence[int]]) -> list[int]:
    """Per option, the fewest leading ids no other option shares."""
    ks = []
    for i, ids in enumerate(cont_ids):
        k = 1
        while any(j != i and list(o[:k]) == list(ids[:k]) for j, o in enumerate(cont_ids)):
            k += 1
            if k > len(ids):
                raise ValueError(f"option {i}'s ids {list(ids)} are a prefix of another option's")
        ks.append(k)
    return ks


def summed(token_scores: Sequence[Sequence[float]], lengths: Sequence[int]) -> list[float]:
    """Each option's score: its first `lengths[j]` token log-probabilities, summed."""
    return [math.fsum(t[:k]) for t, k in zip(token_scores, lengths, strict=True)]


def _kl(p: Sequence[float], q: Sequence[float]) -> float:
    return math.fsum(
        a * (math.log(a) - math.log(max(b, 1e-300))) for a, b in zip(p, q, strict=True) if a > 0
    )


@dataclass(frozen=True)
class Answers:
    """One side's answers on a set of rows, at a calibration."""

    choice: tuple[int, ...]
    top: tuple[float, ...]
    probabilities: tuple[tuple[float, ...], ...]


def answers(scores: Sequence[Sequence[float]], temperature: float) -> Answers:
    probs = [calibration.softmax(row, temperature) for row in scores]
    return Answers(
        choice=tuple(calibration.argmax(row) for row in scores),
        top=tuple(max(p) for p in probs),
        probabilities=tuple(tuple(p) for p in probs),
    )


def outcome_metrics(
    side: Answers,
    targets: Sequence[int],
    fit: calibration.Fit,
    none_index: int | None,
    kind: str,
) -> dict[str, Any]:
    """Accuracy over in-scope rows, the majority answer, and for Choice the abstain rates."""
    in_scope = [i for i, t in enumerate(targets) if t != none_index]
    outside = [i for i, t in enumerate(targets) if t == none_index]
    correct = [side.choice[i] == targets[i] for i in in_scope]
    most = Counter(side.choice).most_common(1)[0]
    out: dict[str, Any] = {
        "rows": len(targets),
        "in_scope_rows": len(in_scope),
        "out_of_scope_rows": len(outside),
        "accuracy": sum(correct) / len(correct) if correct else None,
        "majority_answer": {"option": most[0], "share": most[1] / len(targets)},
    }
    if outside:
        out["out_of_scope_sent_to_none"] = sum(side.choice[i] == none_index for i in outside) / len(
            outside
        )
    if kind != "choice" or fit.rule is None:
        return out

    def answered(i: int) -> bool:
        return calibration.answers(side.top[i], side.choice[i] == none_index, fit.abstain_below)

    answered_in = [i for i in in_scope if answered(i)]
    wrong = sum(side.choice[i] != targets[i] for i in answered_in)
    out["coverage"] = len(answered_in) / len(in_scope) if in_scope else None
    out["answered"] = len(answered_in)
    out["error_on_answered"] = wrong / len(answered_in) if answered_in else None
    out["error_upper_bound_95"] = (
        calibration.upper_bound(wrong, len(answered_in)) if answered_in else None
    )
    out["error_lower_bound_95"] = (
        calibration.lower_bound(wrong, len(answered_in)) if answered_in else None
    )
    if outside:
        rejected = sum(not answered(i) for i in outside)
        out["out_of_scope_rejected"] = rejected / len(outside)
        out["out_of_scope_rejected_upper_bound_95"] = calibration.upper_bound(
            rejected, len(outside)
        )
    return out


def section_payload(decision: Decision, fit: calibration.Fit) -> dict[str, Any]:
    """What the bundle carries in its `litetune.decision` section, and flutter_edge_ai reads."""
    return {
        "contract_version": CONTRACT_VERSION,
        "type": decision.type,
        "question": decision.question,
        "options": [option.as_dict() for option in decision.options],
        "none_option": decision.none_option,
        "scoring": SCORING,
        "temperature": fit.temperature,
        "abstain_below": fit.abstain_below,
    }


def section_bytes(payload: dict[str, Any]) -> bytes:
    """The section's bytes, as `bundle` writes them."""
    return payload_bytes(payload)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class DecisionVerify:
    """Everything one decision verification needs from `verify`'s request."""

    model: Path
    reference: str
    heldout: Path
    calibration: Path | None
    float_twin: Path | None = None
    engine_rows: int = DEFAULT_ENGINE_ROWS
    threads: int = DEFAULT_THREADS
    backend: str = "cpu"
    train_env: envs.StageEnv = envs.TRAIN
    runtime_env: envs.StageEnv = envs.RUNTIME
    auto_provision: bool = True


@dataclass
class _State:
    checks: CheckSet = field(default_factory=lambda: CheckSet(name="verify:decision"))
    limitations: list[str] = field(default_factory=list)
    harness_failed: bool = False
    gate_failed: bool = False
    unchecked: bool = False

    def add(self, check: Check, events: EventStream, harness: bool = False) -> None:
        self.checks.add(check)
        events.check(check)
        if check.outcome is Outcome.FAILED:
            if harness:
                self.harness_failed = True
            else:
                self.gate_failed = True
        elif check.outcome is Outcome.UNCHECKED:
            self.unchecked = True


def verify_decision(
    job: DecisionVerify, record: dict[str, Any] | None, events: EventStream
) -> tuple[str, dict[str, Any]]:
    """Measure a decision checkpoint and its bundle. Returns (status, manifest fields).

    The status is one of `verify.Status`'s values; `verify` wraps the fields in
    its manifest. Raises nothing about the model.
    """
    state = _State()
    fields: dict[str, Any] = {"scorer": {"name": SCORER, "means": SCORER_MEANS}}

    def finish(status: str) -> tuple[str, dict[str, Any]]:
        fields["checks"] = state.checks.as_dict()
        fields["limitations"] = state.limitations
        return status, fields

    def refuse(name: str, detail: str, observed: Any = None) -> tuple[str, dict[str, Any]]:
        state.add(Check.failed(name, detail, observed=observed), events, harness=True)
        return finish("failed_harness")

    # -- what the checkpoint answers -------------------------------------------
    if record is None or not isinstance(record.get("decision"), dict):
        return refuse(
            ROWS_CHECK,
            "--calibration or --float-twin was given, but the reference records no decision. A "
            "decision checkpoint is one `tune --decision` trained; its litetune.json holds it",
        )
    try:
        decision = decision_from(record["decision"], "the reference's litetune.json")
    except DecisionError as exc:
        return refuse(ROWS_CHECK, str(exc))
    if record.get("decision_sha256") not in (None, decision.digest()):
        return refuse(
            ROWS_CHECK,
            "the reference's litetune.json records a decision whose digest is not its own: "
            f"{record.get('decision_sha256')} against {decision.digest()}",
        )
    fields["decision"] = {"decision": decision.as_dict(), "decision_sha256": decision.digest()}
    if decision.type != "choice":
        state.limitations.append(TYPE_UNMEASURED.format(kind=decision.type))
    if job.backend != "cpu":
        return refuse(
            RUNTIME_CHECK,
            f"option scoring is measured on the CPU backend only; --backend {job.backend} is "
            "refused rather than measured with a method nobody has checked there",
        )
    if job.calibration is None:
        return refuse(
            CALIBRATION_CHECK,
            "a decision checkpoint needs its calibration rows: pass --calibration with the "
            "calibration.jsonl `prepare` wrote beside the held-out split",
        )
    try:
        heldout = read_decision_rows(job.heldout, decision)
        calib = read_decision_rows(job.calibration, decision)
    except DecisionRowsError as exc:
        return refuse(ROWS_CHECK, str(exc))
    state.add(
        Check.passed(
            ROWS_CHECK,
            f"{len(calib.prompts)} calibration and {len(heldout.prompts)} held-out rows, each "
            f"the decision's prompt over {len(decision.ids)} options",
        ),
        events,
    )
    fields["data"] = {"heldout": heldout.as_dict(), "calibration": calib.as_dict()}
    none_index = decision.ids.index(decision.none_option) if decision.none_option else None
    terminator = record.get("turn_terminator") or DEFAULT_TERMINATOR

    # -- the reference -----------------------------------------------------------
    prompts = [*calib.prompts, *heldout.prompts]
    events.note(f"reference: scoring {len(decision.ids)} options on {len(prompts)} rows")
    ref = run_script(
        job.train_env,
        REFERENCE_SCRIPT,
        {
            "model": job.reference,
            "rows": [{"index": i, "prompt": p} for i, p in enumerate(prompts)],
            "options": list(decision.ids),
            "terminator": terminator,
        },
        timeout=STARTUP_TIMEOUT_S + REFERENCE_ROW_TIMEOUT_S * len(prompts),
        auto_provision=job.auto_provision,
        events=events,
    )
    fields["harness"] = {"reference": {k: v for k, v in ref.report.items() if k != "cont_ids"}}
    missing = [i for i in range(len(prompts)) if i not in ref.scores]
    if ref.failure or missing:
        return refuse(
            REFERENCE_CHECK,
            ref.failure or f"{len(missing)} of {len(prompts)} rows have no reference scores",
        )
    cont_ids = ref.report.get("cont_ids")
    try:
        ks = unique_lengths(cont_ids or [])
    except ValueError as exc:
        return refuse(REFERENCE_CHECK, str(exc))
    if len(ks) != len(decision.ids):
        return refuse(REFERENCE_CHECK, "the reference reported no tokenisation of the options")
    state.add(
        Check.passed(
            REFERENCE_CHECK,
            f"{len(prompts)} rows on {ref.report.get('device')}, "
            f"{ref.report.get('cache_variant')} scoring",
        ),
        events,
    )
    ref_scores = [summed(ref.scores[i], ks) for i in range(len(prompts))]
    calib_scores, held_scores = ref_scores[: len(calib.prompts)], ref_scores[len(calib.prompts) :]

    # -- the calibration --------------------------------------------------------
    try:
        fit = calibration.fit(decision.type, calib_scores, calib.targets, none_index)
    except calibration.CalibrationError as exc:
        return refuse(CALIBRATION_CHECK, str(exc))
    state.add(
        Check.passed(
            CALIBRATION_CHECK,
            f"temperature {fit.temperature:.4f}"
            + (
                f", abstain below {fit.abstain_below:.4f} ({fit.rule})"
                if fit.rule is not None and fit.abstain_below is not None
                else ""
            ),
            observed=fit.as_dict(),
        ),
        events,
    )
    if fit.rule == calibration.C1:
        state.limitations.append(calibration.C1_NOTE)
    payload = section_payload(decision, fit)
    fields["decision"].update(
        {
            "fit": fit.as_dict(),
            "k": ks,
            "option_tokens": cont_ids,
            "section": {
                "name": SECTION_NAME,
                "payload": payload,
                "bytes": len(section_bytes(payload)),
            },
        }
    )
    ref_answers = answers(held_scores, fit.temperature)
    ref_metrics = outcome_metrics(ref_answers, heldout.targets, fit, none_index, decision.type)
    fields["measurements"] = {"reference": ref_metrics}

    # -- does the rule hold on held-out rows? (Choice) ----------------------------
    if decision.type == "choice" and fit.rule is not None:
        state.add(_rule_check(fit, ref_metrics), events)

    # -- the runtime ---------------------------------------------------------------
    n_engine = min(job.engine_rows, len(heldout.prompts))
    engine_prompts = list(heldout.prompts[:n_engine])
    events.note(f"runtime: scoring {len(decision.ids)} options on {n_engine} held-out rows")
    eng = run_script(
        job.runtime_env,
        ENGINE_SCRIPT,
        {
            "model": str(job.model),
            "rows": [{"index": i, "prompt": p} for i, p in enumerate(engine_prompts)],
            "options": list(decision.ids),
            "terminator": terminator,
            "threads": job.threads,
            "targets": "unique",
            "mode": "auto",
        },
        timeout=STARTUP_TIMEOUT_S + ENGINE_ROW_TIMEOUT_S * n_engine,
        auto_provision=job.auto_provision,
        events=events,
    )
    fields["harness"]["runtime"] = {
        k: v for k, v in eng.report.items() if k not in ("cont_ids", "k", "scored_lengths")
    }
    if "selfcheck" in eng.report:
        fields["decision"]["selfcheck"] = eng.report["selfcheck"]
    missing = [i for i in range(n_engine) if i not in eng.scores]
    if eng.failure or missing:
        detail = eng.failure or (
            f"{len(missing)} of {n_engine} rows have no runtime scores"
            + (f": {next(iter(eng.errors.values()))}" if eng.errors else "")
        )
        return refuse(RUNTIME_CHECK, detail)
    mode = eng.report.get("mode")
    if eng.report.get("cont_ids") != cont_ids or eng.report.get("k") != ks:
        return refuse(
            TARGETS_CHECK,
            "the runtime tokenises the options differently from the reference, so the two sides "
            "would score different targets",
            observed={"reference_k": ks, "runtime_k": eng.report.get("k")},
        )
    mismatched = eng.report.get("mismatches") or []
    state.add(
        Check.passed(
            TARGETS_CHECK,
            f"both sides tokenise all {len(ks)} options alike; {sum(ks)} scored tokens per "
            f"decision" + (f", {len(mismatched)} scored in full" if mismatched else ""),
        ),
        events,
    )
    eng_lengths = eng.report.get("scored_lengths") or ks
    eng_scores = [summed(eng.scores[i], eng_lengths) for i in range(n_engine)]
    state.add(
        Check.passed(
            RUNTIME_CHECK,
            f"{n_engine} rows in mode {mode!r}, picked by the runtime self-check "
            f"({(eng.report.get('selfcheck') or {}).get('verdict')})",
            observed=eng.report.get("selfcheck"),
        ),
        events,
    )

    # -- runtime against reference, on the same rows -------------------------------
    sub_ref = answers(held_scores[:n_engine], fit.temperature)
    sub_eng = answers(eng_scores, fit.temperature)
    targets = heldout.targets[:n_engine]
    agree = sum(a == b for a, b in zip(sub_ref.choice, sub_eng.choice, strict=True)) / n_engine
    kl = [_kl(p, q) for p, q in zip(sub_ref.probabilities, sub_eng.probabilities, strict=True)]
    in_scope = [i for i, t in enumerate(targets) if t != none_index]
    ms = [eng.ms[i] for i in range(n_engine)]
    fields["measurements"]["candidate"] = outcome_metrics(
        sub_eng, targets, fit, none_index, decision.type
    )
    # In the shape the generative path records, so `bundle` reads the runtime
    # and the CPU these numbers were taken on the same way.
    fields["measurements"]["candidate"]["engine"] = {
        "engine": "litert-lm",
        "backend": "cpu",
        "runtime_version": (eng.report.get("runtime") or {}).get("litert-lm"),
        "cpu": {"model": eng.report.get("cpu")},
        "threads": job.threads,
    }
    fields["measurements"]["reference_on_runtime_rows"] = outcome_metrics(
        sub_ref, targets, fit, none_index, decision.type
    )
    fields["quality"] = {
        "available": True,
        "rows": n_engine,
        "argmax_agreement": agree,
        "mean_kl": math.fsum(kl) / len(kl),
        "max_kl": max(kl),
        "accuracy_difference": (
            paired_difference(
                [sub_eng.choice[i] == targets[i] for i in in_scope],
                [sub_ref.choice[i] == targets[i] for i in in_scope],
            ).as_dict()
            if in_scope
            else None
        ),
    }
    fields["cost"] = {
        "ms_per_decision_median": statistics.median(ms),
        "ms_per_decision_max": max(ms),
        "options": len(decision.ids),
        "scored_tokens_per_decision": sum(eng_lengths),
        "mode": mode,
        "threads": job.threads,
        "cpu": eng.report.get("cpu"),
        "runtime": eng.report.get("runtime"),
        "rows": n_engine,
    }
    state.add(
        (Check.passed if agree >= AGREEMENT_FLOOR else Check.failed)(
            AGREEMENT_CHECK,
            f"the runtime's answer is the reference's on {agree:.4f} of {n_engine} rows "
            f"(refused below {AGREEMENT_FLOOR}); mean KL {fields['quality']['mean_kl']:.2e}",
            observed={"agreement": agree, "rows": n_engine},
        ),
        events,
    )
    if n_engine < len(heldout.prompts):
        state.limitations.append(
            f"the runtime scored the first {n_engine} of {len(heldout.prompts)} held-out rows; "
            "the reference scored them all. Runtime accuracy and agreement are on those rows"
        )
    state.limitations.append(TEMPERATURE_CARRIED)

    # -- the float twin -----------------------------------------------------------------
    runtime_version = (eng.report.get("runtime") or {}).get("litert-lm")
    if job.float_twin is None:
        if runtime_version == MEASURED_RUNTIME:
            state.limitations.append(NO_TWIN.format(runtime=MEASURED_RUNTIME))
        else:
            state.add(
                Check.unchecked(
                    TWIN_CHECK,
                    f"the runtime is litert-lm {runtime_version}, not {MEASURED_RUNTIME}, on which "
                    "the method was measured against PyTorch, and no --float-twin was given to "
                    "measure it on this one",
                ),
                events,
            )
    else:
        state.add(
            _twin_check(job, decision, terminator, heldout, ref, mode, events),
            events,
            harness=True,
        )

    if state.harness_failed:
        return finish("failed_harness")
    if state.gate_failed:
        return finish("failed_gate")
    if state.unchecked:
        return finish("inconclusive")
    return finish("passed")


def _rule_check(fit: calibration.Fit, held: dict[str, Any]) -> Check:
    """Whether the held-out rows clearly contradict the rule the threshold was fitted by."""
    if fit.rule == calibration.C2 and held.get("out_of_scope_rows"):
        bound = held["out_of_scope_rejected_upper_bound_95"]
        detail = (
            f"{held['out_of_scope_rejected']:.4f} of {held['out_of_scope_rows']} out-of-scope "
            f"rows rejected; the 95 % upper bound {bound:.4f} against the "
            f"{calibration.REJECT_TARGET} target"
        )
        ok = bound >= calibration.REJECT_TARGET
    elif fit.rule == calibration.C1 and held.get("answered"):
        bound = held["error_lower_bound_95"]
        detail = (
            f"{held['error_on_answered']:.4f} error on {held['answered']} answered rows; the 95 % "
            f"lower bound {bound:.4f} against the {calibration.ERROR_BUDGET} budget"
        )
        ok = bound <= calibration.ERROR_BUDGET
    else:
        return Check.unchecked(
            RULE_CHECK, "the held-out rows have none of the rows the rule is about"
        )
    return (Check.passed if ok else Check.failed)(RULE_CHECK, detail, observed=held)


def _twin_check(
    job: DecisionVerify,
    decision: Decision,
    terminator: str,
    heldout: DecisionRows,
    ref: ScoreRun,
    mode: str | None,
    events: EventStream,
) -> Check:
    """The float twin's per-token scores against the reference's, on a few held-out rows."""
    n = min(PROBE_ROWS, len(heldout.prompts))
    calib_n = len(ref.scores) - len(heldout.prompts)
    twin = run_script(
        job.runtime_env,
        ENGINE_SCRIPT,
        {
            "model": str(job.float_twin),
            "rows": [{"index": i, "prompt": heldout.prompts[i]} for i in range(n)],
            "options": list(decision.ids),
            "terminator": terminator,
            "threads": job.threads,
            "targets": "full",
            "mode": mode or "auto",
        },
        timeout=STARTUP_TIMEOUT_S + ENGINE_ROW_TIMEOUT_S * n,
        auto_provision=False,
        events=events,
    )
    if twin.failure or len(twin.scores) < n:
        return Check.failed(TWIN_CHECK, twin.failure or "the float twin left rows unscored")
    worst = 0.0
    for i in range(n):
        for a, b in zip(twin.scores[i], ref.scores[calib_n + i], strict=True):
            if len(a) != len(b):
                return Check.failed(TWIN_CHECK, "the float twin scored targets of another length")
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b, strict=True)))
    detail = (
        f"largest per-token difference {worst:.2e} nats on {n} rows, every option, in mode "
        f"{mode!r} (refused above {TWIN_TOLERANCE})"
    )
    observed = {"max_abs_token": worst, "rows": n, "mode": mode}
    return (Check.passed if worst <= TWIN_TOLERANCE else Check.failed)(
        TWIN_CHECK, detail, observed=observed
    )
