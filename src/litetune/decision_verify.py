"""`verify` for a decision checkpoint: option scores on both sides, calibration, the shipped file.

A decision model never generates in use. flutter_edge_ai's proposed
`createDecider` (the 2026-10-16 design) is to score every option's
log-probability through the runtime and answer with the best one, so that is
what this measures, as `decision_scripts.py` describes:
- **the reference**, the tuned checkpoint in PyTorch float32, scores every
  option on every calibration and held-out row;
- **the calibration** (`calibration.py`) is fitted on the reference's
  calibration scores: a temperature, and for Choice the abstain threshold;
- **the runtime** scores the shipped `.litertlm` on `engine_rows` held-out
  rows spread through the split, in the mode its self-check picks, and is
  compared with the reference on the same rows, prompt ids first, at the same
  calibration.

Scoring the runtime on every row costs too much: at 151 options a decision took
7.2 s on four threads of a server CPU (measured 2026-10-16), so 3,000 rows
would take six hours.

**Refused:** a runtime that scored another prompt than the reference; a
runtime whose answers agree with the reference's on fewer than 95 % of the
rows, a gross-failure check; a float twin whose per-token scores differ from
the reference's by more than 1e-3 nats; and a Choice question that answers no
in-scope held-out row, or clearly misses its own rule on them, by a one-sided
95 % Clopper-Pearson bound.

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
import sys
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
from litetune.decisions import (
    CONTRACT_KEYS,
    TYPE_MEASURED,
    Decision,
    DecisionError,
    decision_from,
)
from litetune.evaluate import gpu_observed, gpu_unused
from litetune.events import EventStream
from litetune.metrics import paired_difference
from litetune.prepare import DECISION_FAMILY, DECISION_MEASURED_ONCE
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

SCORER = "option-scoring"
SCORER_MEANS = (
    "the option whose shortest unique prefix has the highest summed log-probability, as "
    "flutter_edge_ai's proposed createDecider would score it; a Choice answer abstains below "
    "the fitted threshold or on the none_option"
)
SCORING = "unique_prefix"
# How the runtime scores one decision's options, which the section carries so
# that a decider scores them the same way: `rewind`, one session prefilled once
# and rewound to a checkpoint after each option; `fresh_engine`, an Engine and
# a session of their own for each option, never rewound, for a family whose
# rule gives every prompt an Engine (`models.ModelRules.engine_per_prompt_reason`).
METHOD_REWIND = "rewind"
METHOD_FRESH_ENGINE = "fresh_engine"

DEFAULT_ENGINE_ROWS = 200
DEFAULT_THREADS = 4
PROBE_ROWS = 4
AGREEMENT_FLOOR = 0.95
TWIN_TOLERANCE = 1e-3
# The runtime the method was measured correct on, against PyTorch, on a float32
# bundle (2026-10-16). On another, only a float twin shows it still is.
MEASURED_RUNTIME = "0.18.0"

# Seconds: a start-up allowance, then a budget per row. At 151 options the
# runtime took 7.2 s a decision on a server CPU; a phone-class CPU is slower.
STARTUP_TIMEOUT_S = 900
REFERENCE_ROW_TIMEOUT_S = 30
ENGINE_ROW_TIMEOUT_S = 120
# Per Engine load, under `fresh_engine`: one per option per row, and at most
# twelve for the self-check (two prompts, three targets, twice each). Creating
# an Engine took 1.17 s for a gemma-3-270m bundle on LiteRT-LM 0.18.0 (E7,
# 2026-10-16); a bigger model takes longer.
ENGINE_LOAD_TIMEOUT_S = 60
SELFCHECK_LOADS = 12

ROWS_CHECK = "rows are the decision's prompts"
REFERENCE_CHECK = "reference option scores"
CALIBRATION_CHECK = "calibration"
RUNTIME_CHECK = "runtime option scores"
TARGETS_CHECK = "both sides score the same targets"
AGREEMENT_CHECK = "runtime answers agree with the reference"
TWIN_CHECK = "float twin matches the reference"
RULE_CHECK = "the abstain rule holds on held-out rows"
PROMPT_CHECK = "both sides score the same prompts"
# Below this many out-of-scope calibration rows, the rejection rule lets none of
# them answer (`calibration.max_answered`), so one row sets the threshold.
THIN_OUT_OF_SCOPE = 10

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
TYPE_LESS_MEASURED = (
    "this is a {kind} question; option scoring has trained and measured Choice on several tasks, "
    "and {kind} {measured}"
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
    # The prompt's token ids, for the rows the script was asked to give them for.
    prompt_ids: dict[int, list[int]] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    failure: str | None = None


def _tail(text: str | bytes | None, limit: int = 2000) -> str:
    """The end of a child's stderr, as text: where a Python traceback says what went wrong."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return (text or "").strip()[-limit:]


GPU_IDLE = (
    "the GPU backend was asked for and the kernel shows the {run} run did no GPU work: its "
    "scores came from somewhere else"
)
# Runs of the GPU scoring, in all; the last scores each row still in question this many
# times in one process.
GPU_MAX_RUNS = 3
GPU_COPIES_IN_LAST_RUN = 3
GPU_REPEAT = (
    "on the GPU every decision was scored twice, and each row the two runs scored differently "
    "three more times in a third run, keeping the scores two of its runs or copies agreed on: "
    "through litert-lm's package on macOS, which runs LiteRT's WebGPU accelerator, a few rows of "
    "300 got another prompt's scores in a run, at different rows each time, and flutter_edge_ai "
    "on a Mac loads LiteRT's Metal accelerator, whose scores repeated (MEASUREMENTS.md, \"The "
    'CPU and the GPU on a Mac"); {rows} of {n} rows needed the third run'
)


def settle(runs: Sequence[dict[int, Any]], rows: Sequence[int]) -> tuple[dict[int, Any], list[int]]:
    """Per row, the scores at least two of `runs` gave, and the rows where no two agree."""
    settled: dict[int, Any] = {}
    unresolved = []
    for i in rows:
        seen = [run[i] for run in runs if i in run]
        agreed = next((a for k, a in enumerate(seen) if a in seen[k + 1 :]), None)
        if agreed is None:
            unresolved.append(i)
        else:
            settled[i] = agreed
    return settled, unresolved


def _repeat_on_gpu(
    job: Any,
    spec: dict[str, Any],
    first: ScoreRun,
    n_rows: int,
    n_options: int,
    method: str,
    events: EventStream,
) -> dict[str, Any] | str:
    """Score every row a second time, and the rows the two runs differ on in a third run,
    `GPU_COPIES_IN_LAST_RUN` copies of each.

    Replaces `first.scores` with the scores two runs agree on. Returns what was
    done, or why it could not be: a run that failed, or a row no two runs agreed
    on. The repeats take the mode the first run's self-check chose. A row's
    right scores repeat bit for bit and the wrong ones are another prompt's,
    different each time, so two runs that agree have the right ones.
    """
    again_spec = {**spec, "mode": first.report.get("mode") or spec["mode"]}
    events.note(f"runtime: scoring the {n_rows} rows again, on the GPU")
    second = run_script(
        job.runtime_env,
        ENGINE_SCRIPT,
        again_spec,
        timeout=_engine_timeout(n_rows, n_options, method),
        auto_provision=False,
        events=events,
    )
    if second.failure or len(second.scores) < n_rows:
        return "the second GPU run left rows unscored: " + (
            second.failure or f"{n_rows - len(second.scores)} rows"
        )
    if gpu_unused("gpu", second.report.get("device")):
        return GPU_IDLE.format(run="second")
    differ = [i for i in range(n_rows) if second.scores.get(i) != first.scores.get(i)]
    runs: list[dict[int, Any]] = [first.scores, second.scores]
    if differ:
        events.note(f"runtime: {len(differ)} rows scored differently twice; a third GPU run")
        # Each row in question, `GPU_COPIES_IN_LAST_RUN` times: a run's wrong
        # scores fall on a few rows, so copies in one process mostly agree.
        by_index = {r["index"]: r for r in spec["rows"]}
        copies = [
            {"index": n_rows * (c + 1) + i, "prompt": by_index[i]["prompt"]}
            for c in range(GPU_COPIES_IN_LAST_RUN)
            for i in differ
        ]
        third = run_script(
            job.runtime_env,
            ENGINE_SCRIPT,
            {**again_spec, "rows": copies},
            timeout=_engine_timeout(len(copies), n_options, method),
            auto_provision=False,
            events=events,
        )
        if third.failure or any(c["index"] not in third.scores for c in copies):
            return "the third GPU run left rows unscored: " + (third.failure or "some rows")
        if gpu_unused("gpu", third.report.get("device")):
            return GPU_IDLE.format(run="third")
        for c in range(GPU_COPIES_IN_LAST_RUN):
            runs.append({i: third.scores[n_rows * (c + 1) + i] for i in differ})
    settled, unresolved = settle(runs, differ)
    if unresolved:
        return (
            f"{len(unresolved)} rows got no score twice in {GPU_MAX_RUNS} GPU runs (rows "
            f"{unresolved[:10]}): no two runs agree, so there is no score to measure"
        )
    first.scores.update(settled)
    return {"runs": 3 if differ else 2, "rows_rescored": len(differ), "rows": differ[:50]}


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
        except subprocess.TimeoutExpired as expired:
            said = _tail(expired.stderr)
            run.failure = f"no result after {timeout} s (timeout)" + (f": {said}" if said else "")
        except OSError as exc:
            run.failure = f"the script could not be started: {type(exc).__name__}: {exc}"
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
                    if "ms" in rec:
                        run.ms[rec["index"]] = float(rec["ms"])
                if "prompt_ids" in rec:
                    run.prompt_ids[rec["index"]] = rec["prompt_ids"]
        if report.exists():
            try:
                run.report = json.loads(report.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                run.failure = run.failure or f"the script's report could not be read: {exc}"
        # A non-zero exit first: a script that died before its report says why
        # on stderr, and "wrote no report" alone would throw that away.
        if proc is not None and proc.returncode != 0 and run.failure is None:
            said = run.report.get("error") or _tail(proc.stderr)
            run.failure = f"the script exited {proc.returncode}: {said}"
        elif proc is not None and not report.exists() and run.failure is None:
            said = _tail(proc.stderr)
            run.failure = "the script wrote no report" + (f": {said}" if said else "")
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
    """Each option's score: its first `lengths[j]` token log-probabilities, summed.

    Raises `ValueError` when an option has fewer token scores than its length:
    a shorter sum is a score for another target.
    """
    for j, (t, k) in enumerate(zip(token_scores, lengths, strict=True)):
        if len(t) < k:
            raise ValueError(f"option {j} has {len(t)} token score(s), and {k} are scored")
    return [math.fsum(t[:k]) for t, k in zip(token_scores, lengths, strict=True)]


def spread(total: int, count: int) -> list[int]:
    """`count` row indices out of `total`, evenly through them, in order.

    `prepare` keeps the source file's order within a split, and a file grouped
    by label would give its first rows to one or two options. Each index is the
    middle of its stretch of `total / count` rows, so the last stretch is
    sampled as the first is: 2 of 3 rows are 0 and 2, not 0 and 1.
    """
    count = min(count, total)
    return [((2 * i + 1) * total) // (2 * count) for i in range(count)]


def _not_finite(scores: dict[int, list[list[float]]]) -> int | None:
    """The first row holding a token score that is not a finite number, if any."""
    for index, per in sorted(scores.items()):
        if not all(isinstance(x, int | float) and math.isfinite(x) for t in per for x in t):
            return index
    return None


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


def section_payload(
    decision: Decision,
    fit: calibration.Fit,
    scored_lengths: Sequence[int],
    terminator: str,
    method: str,
) -> dict[str, Any]:
    """What the bundle carries in its `litetune.decision` section, for flutter_edge_ai to read.

    The decision as `tune` recorded it, then how its options were scored: each
    listed option's id followed by `terminator`, on its first `scored_lengths[i]`
    tokens (the shortest unique prefix, or the whole target where that prefix
    does not tokenise back to itself), the `none_option` last. The temperature
    and the threshold were fitted on exactly those scores, so a reader that
    tokenises the options differently is scoring something else. `method` is how
    the runtime scored them (`METHOD_REWIND` or `METHOD_FRESH_ENGINE`): a
    decider that rewinds where `fresh_engine` is written scores options on a
    state the runtime was never measured on.
    """
    return {
        **decision.as_dict(),
        "scoring": SCORING,
        "method": method,
        "terminator": terminator,
        "scored_lengths": list(scored_lengths),
        "temperature": fit.temperature,
        "abstain_rule": fit.rule,
        "abstain_below": fit.abstain_below,
    }


def decision_of_section(payload: dict[str, Any]) -> Decision:
    """The decision a section payload carries. Raises `DecisionError` naming what is wrong."""
    return decision_from(
        {key: value for key, value in payload.items() if key in CONTRACT_KEYS},
        "the decision section",
    )


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
    # The family rule's `engine_per_prompt_reason`: option scoring rewinds
    # within one session, which no Engine of its own per prompt can make safe.
    engine_per_prompt_reason: str = ""
    # The family rule's `decision_twin_tolerance`, and why, for a family whose
    # float32 bundle is measured further from its model than `TWIN_TOLERANCE`.
    twin_tolerance: float = TWIN_TOLERANCE
    twin_tolerance_reason: str = ""
    # The family rule that claims the checkpoint, `None` for none. Only
    # `prepare.DECISION_FAMILY` has had the scoring method measured on it without
    # a float twin.
    family: str | None = None
    # What `verify` was asked that the decision path does not do: refused rather
    # than ignored, so a manifest never records a role or a limit nobody applied.
    reference_role: str = "float_twin"
    limit: int | None = None
    # The other verify flags given that do not apply here, by their CLI names.
    not_applicable: tuple[str, ...] = ()
    # Where the reference runs: the operator's LITETUNE_DEVICE ("auto" or
    # "cpu"); "auto" is CUDA when the training environment's torch reaches one.
    device: str = "auto"
    # A LITETUNE_DEVICE litetune does not accept, as `devices.device_setting` said.
    device_error: str | None = None
    # Why the reference's litetune.json could not be read, when it could not.
    record_error: str | None = None
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
    if job.record_error is not None:
        return refuse(
            ROWS_CHECK,
            f"the reference's litetune.json could not be read ({job.record_error}), so whether it "
            "is a decision checkpoint, and which decision it answers, is unknown",
        )
    if record is None or record.get("decision") is None:
        return refuse(
            ROWS_CHECK,
            "--calibration or --float-twin was given, but the reference records no decision. A "
            "decision checkpoint is one `tune --decision` trained; its litetune.json holds it",
        )
    if not isinstance(record.get("decision"), dict):
        return refuse(
            ROWS_CHECK,
            "the reference's litetune.json has a 'decision' that is not an object: "
            f"{json.dumps(record.get('decision'))[:200]}",
        )
    if job.reference_role != "float_twin":
        return refuse(
            ROWS_CHECK,
            f"--reference-role {job.reference_role}: a decision checkpoint is measured against "
            "itself in float32, the role float_twin; an untuned base answers a question it was "
            "never trained on",
        )
    if job.limit is not None:
        return refuse(
            ROWS_CHECK,
            "--limit does not apply to a decision checkpoint: the reference scores every held-out "
            "row, and --engine-rows sets how many the runtime scores",
        )
    if job.device_error is not None:
        return refuse(REFERENCE_CHECK, f"verification was not attempted: {job.device_error}")
    if job.not_applicable:
        return refuse(
            ROWS_CHECK,
            f"{', '.join(job.not_applicable)} do(es) not apply to a decision checkpoint, which is "
            "measured by option scoring in the prompt mode tune trained it in, against its own "
            "reference, and gated on their agreement; refused rather than ignored",
        )
    try:
        decision = decision_from(record["decision"], "the reference's litetune.json")
    except DecisionError as exc:
        return refuse(ROWS_CHECK, str(exc))
    if record.get("decision_sha256") != decision.digest():
        return refuse(
            ROWS_CHECK,
            "the reference's litetune.json records a decision whose digest is not its own: "
            f"{record.get('decision_sha256')} against {decision.digest()}. `tune --decision` "
            "records both; a record without its digest was not written by it",
        )
    fields["decision"] = {"decision": decision.as_dict(), "decision_sha256": decision.digest()}
    if decision.type != "choice":
        state.limitations.append(
            TYPE_LESS_MEASURED.format(kind=decision.type, measured=TYPE_MEASURED[decision.type])
        )
    if job.backend not in ("cpu", "gpu"):
        return refuse(
            RUNTIME_CHECK,
            f"--backend {job.backend}: option scoring runs on LiteRT-LM's CPU or GPU backend",
        )
    if job.backend == "gpu" and sys.platform != "darwin":
        return refuse(
            RUNTIME_CHECK,
            f"--backend gpu: option scoring on the GPU is measured on macOS only, and this is "
            f"{sys.platform}; on Linux litert-lm's GPU path needs a Vulkan driver, and Windows "
            'is not measured (MEASUREMENTS.md, "The CPU and the GPU on a Mac")',
        )
    method = METHOD_FRESH_ENGINE if job.engine_per_prompt_reason else METHOD_REWIND
    if method == METHOD_FRESH_ENGINE:
        state.limitations.append(
            "this family's rule gives every prompt an Engine of its own: "
            f"{job.engine_per_prompt_reason}. Option scoring rewinds a session to the prompt "
            "for every option, so here each option is scored on an Engine and a session of "
            f"its own, never rewound (method {METHOD_FRESH_ENGINE}, which the section carries "
            "for the decider): an Engine load per option per decision, here and on a device. "
            "Creating an Engine took 1.17 s for a gemma-3-270m bundle on LiteRT-LM 0.18.0 "
            "(2026-10-16); this family's load time is not measured"
        )
    # The method -- a rewind that resets only the step while the re-prefilled
    # cell stays (decision_scripts.ENGINE_SCRIPT, mode `once`) -- was measured
    # against PyTorch on Gemma 3's caches only, on 2026-10-16. On another family
    # nothing has shown it holds, and a float twin is what shows it.
    if job.family != DECISION_FAMILY:
        if job.float_twin is None:
            return refuse(
                TWIN_CHECK,
                f"this checkpoint's family is {job.family or 'one litetune has no rule for'}, "
                "and the runtime's option-scoring method was measured against PyTorch without "
                f"a float twin on {DECISION_FAMILY} only (2026-10-16). On another family a "
                "decision is verified with --float-twin, which checks the method on this "
                "checkpoint: convert with --recipe none as well and pass that file",
            )
        measured = DECISION_MEASURED_ONCE.get(job.family or "")
        if measured is None:
            state.limitations.append(
                f"decisions by option scoring have not been trained or measured on {job.family}; "
                "the float twin checks the runtime's method on this checkpoint, and nothing here "
                "measures how well the family answers such a question"
            )
        else:
            state.limitations.append(
                f"decisions by option scoring on {job.family} were trained and verified for one "
                f"task only: {' and '.join(measured)}, on an emotion question with six options "
                "(2026-10-20); the float twin checks the runtime's method on this checkpoint"
            )
    if job.twin_tolerance != TWIN_TOLERANCE:
        state.limitations.append(
            f"the float twin is held to {job.twin_tolerance} nats per token, not "
            f"{TWIN_TOLERANCE}: {job.twin_tolerance_reason}"
        )
    if job.calibration is None:
        return refuse(
            CALIBRATION_CHECK,
            "a decision checkpoint needs its calibration rows: pass --calibration with the "
            "calibration.jsonl `prepare` wrote beside the held-out split",
        )
    # Before the reference, which scores every row first and takes minutes.
    for flag, path in (("--model", job.model), ("--float-twin", job.float_twin)):
        if path is not None and not Path(path).is_file():
            return refuse(
                RUNTIME_CHECK,
                f"{flag} {path} is not a file; the runtime scores a converted .litertlm",
            )
    try:
        heldout = read_decision_rows(job.heldout, decision)
        calib = read_decision_rows(job.calibration, decision)
    except DecisionRowsError as exc:
        return refuse(ROWS_CHECK, str(exc))
    if hash_file(job.heldout) == hash_file(job.calibration):
        return refuse(
            CALIBRATION_CHECK,
            f"--calibration and --data are the same rows ({job.calibration}); a threshold fitted "
            "on the rows it is then measured on shows nothing about new ones",
        )
    shared = len(set(calib.prompts) & set(heldout.prompts))
    if shared:
        state.limitations.append(
            f"{shared} prompt(s) appear in both the calibration and the held-out rows; a "
            "dataset's own duplicates do this, and a split made by `prepare` from one file keeps "
            "every row on one side"
        )
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
    # The scorers' terminator is the first token of the end of turn `tune`
    # recorded (`tune.turn_terminator`: the ids the chat template puts after an
    # assistant answer, and their text), as measured on Gemma 3, where it is
    # `<end_of_turn>` of `<end_of_turn>\n`. The reference reads that token's text
    # off the checkpoint's tokenizer, and the runtime scores the same text.
    turn = _recorded_turn(record.get("turn_terminator"))
    if isinstance(turn, str):
        return refuse(ROWS_CHECK, turn)
    turn_ids, turn_text = turn

    if job.engine_rows < 1:
        return refuse(
            RUNTIME_CHECK, f"engine_rows is {job.engine_rows}; the runtime has to score a row"
        )

    # The held-out rows the runtime scores, chosen now so that the reference
    # also returns their prompt's token ids.
    engine_rows = spread(len(heldout.prompts), job.engine_rows)
    calib_n = len(calib.prompts)

    # -- the reference -----------------------------------------------------------
    prompts = [*calib.prompts, *heldout.prompts]
    # Besides its first rows, the longest: a cache gone wrong past Gemma 3's
    # sliding window would not show on short ones.
    longest = max(range(len(prompts)), key=lambda i: len(prompts[i]))
    events.note(f"reference: scoring {len(decision.ids)} options on {len(prompts)} rows")
    ref = run_script(
        job.train_env,
        REFERENCE_SCRIPT,
        {
            "model": job.reference,
            "rows": [{"index": i, "prompt": p} for i, p in enumerate(prompts)],
            "options": list(decision.ids),
            "turn_ids": turn_ids,
            "turn_text": turn_text,
            "check_indices": [longest],
            "ids_for": [calib_n + i for i in engine_rows],
            # The rows the float twin is compared on, scored alone with no
            # cache: the twin's 1e-3 then measures the bundle, not batching.
            "alone_for": [calib_n + i for i in engine_rows[:PROBE_ROWS]],
            "device": job.device,
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
    unusable = _not_finite(ref.scores)
    if unusable is not None:
        return refuse(REFERENCE_CHECK, f"row {unusable} has a reference score that is not finite")
    terminator = ref.report.get("terminator")
    if not isinstance(terminator, str) or not terminator or not turn_text.startswith(terminator):
        return refuse(
            REFERENCE_CHECK,
            f"the reference reported {terminator!r} as the first token of the recorded end of "
            f"turn {turn_text!r}, which it does not begin with",
        )
    cont_ids = ref.report.get("cont_ids")
    try:
        ks = unique_lengths(cont_ids or [])
    except ValueError as exc:
        return refuse(REFERENCE_CHECK, str(exc))
    if len(ks) != len(decision.ids):
        return refuse(REFERENCE_CHECK, "the reference reported no tokenisation of the options")
    row_batch = ref.report.get("row_batch")
    batching = f", {row_batch} rows at a time" if isinstance(row_batch, int) else ""
    fell_back = ref.report.get("batch_trial")
    if fell_back:
        # Slower, not wrong: said here so a sixfold slowdown has a reason.
        batching += f"; batching fell back to one row at a time ({fell_back})"
    state.add(
        Check.passed(
            REFERENCE_CHECK,
            f"{len(prompts)} rows on {ref.report.get('device')}, "
            f"{ref.report.get('cache_variant')} scoring{batching}",
        ),
        events,
    )

    # -- the runtime ---------------------------------------------------------------
    # Before the calibration: the runtime decides how many tokens each option is
    # scored on. A unique prefix that does not tokenise back to its own ids is
    # scored in full there, and then on both sides, so that the calibration, the
    # comparison and the section all describe the scores an application gets.
    n_engine = len(engine_rows)
    engine_prompts = [heldout.prompts[i] for i in engine_rows]
    events.note(f"runtime: scoring {len(decision.ids)} options on {n_engine} held-out rows")
    engine_spec = {
        "model": str(job.model),
        "rows": [{"index": i, "prompt": p} for i, p in enumerate(engine_prompts)],
        "options": list(decision.ids),
        "terminator": terminator,
        "threads": job.threads,
        "targets": "unique",
        "mode": "auto",
        "method": method,
        "backend": job.backend,
    }
    eng = run_script(
        job.runtime_env,
        ENGINE_SCRIPT,
        engine_spec,
        timeout=_engine_timeout(n_engine, len(decision.ids), method),
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
    if job.backend == "gpu":
        if gpu_unused("gpu", eng.report.get("device")):
            return refuse(
                RUNTIME_CHECK,
                "the GPU backend was asked for and the kernel shows this process did no GPU "
                "work: the scores came from somewhere else",
            )
        repeated = _repeat_on_gpu(
            job, engine_spec, eng, n_engine, len(decision.ids), method, events
        )
        if isinstance(repeated, str):
            return refuse(RUNTIME_CHECK, repeated)
        fields["harness"]["gpu_repeat"] = repeated
        state.limitations.append(GPU_REPEAT.format(rows=repeated["rows_rescored"], n=n_engine))
    mode = eng.report.get("mode")
    if eng.report.get("method") != method:
        return refuse(
            RUNTIME_CHECK,
            f"the runtime scored by method {eng.report.get('method')!r}, and {method!r} was "
            "asked for: the scores would describe another decider",
        )
    if eng.report.get("cont_ids") != cont_ids or eng.report.get("k") != ks:
        return refuse(
            TARGETS_CHECK,
            "the runtime tokenises the options differently from the reference, so the two sides "
            "would score different targets",
            observed={"reference_k": ks, "runtime_k": eng.report.get("k")},
        )
    lengths = eng.report.get("scored_lengths")
    if (
        not isinstance(lengths, list)
        or len(lengths) != len(ks)
        or any(not k <= n <= len(c) for k, n, c in zip(ks, lengths, cont_ids or [], strict=False))
    ):
        return refuse(TARGETS_CHECK, f"the runtime reported unusable scored lengths: {lengths}")
    mismatched = eng.report.get("mismatches") or []
    fields["decision"].update({"k": ks, "scored_lengths": lengths, "option_tokens": cont_ids})
    state.add(
        Check.passed(
            TARGETS_CHECK,
            f"both sides tokenise all {len(ks)} options alike and score {sum(lengths)} tokens per "
            "decision"
            + (
                f"; {len(mismatched)} option(s) whose unique prefix does not tokenise back to its "
                "own ids are scored in full on both sides"
                if mismatched
                else ""
            ),
            observed={"k": ks, "scored_lengths": lengths, "mismatches": mismatched},
        ),
        events,
    )
    state.add(
        Check.passed(
            RUNTIME_CHECK,
            f"{n_engine} rows by method {method!r} in mode {mode!r}, picked by the runtime "
            f"self-check ({(eng.report.get('selfcheck') or {}).get('verdict')})",
            observed=eng.report.get("selfcheck"),
        ),
        events,
    )
    # The same prompt on both sides: the reference's ids, as its chat template
    # renders them, against what the runtime's session reads -- the ids it puts
    # first on a first turn (`session_prefix`: the BOS, for a bundle whose
    # metadata has a start token), then the rendering's. Which of the two
    # writes a BOS differs by family: Gemma 3's template prints it and LiteRT-LM
    # hands its template none, so the session adds it; Qwen's tokenizers have
    # none (`bos_token` null in Qwen3-0.6B's and Qwen2.5-0.5B-Instruct's
    # tokenizer_config.json). An empty render, a BOS on one side only, a second
    # BOS or another tokenisation of the text would otherwise show only as
    # disagreement, if at all.
    prefix = eng.report.get("session_prefix")
    if not isinstance(prefix, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in prefix
    ):
        return refuse(
            PROMPT_CHECK,
            "the runtime did not report the ids its session puts before the prompt "
            f"(session_prefix: {json.dumps(prefix)[:100]}), so its prompt cannot be compared "
            "with the reference's",
        )
    differing = [
        j
        for j, i in enumerate(engine_rows)
        if eng.prompt_ids.get(j) is None
        or ref.prompt_ids.get(calib_n + i) != [*prefix, *eng.prompt_ids[j]]
    ]
    if differing:
        j = differing[0]
        want, got = ref.prompt_ids.get(calib_n + engine_rows[j]), eng.prompt_ids.get(j)
        return refuse(
            PROMPT_CHECK,
            f"on {len(differing)} of {n_engine} rows the runtime's prompt is not the "
            f"reference's. First, held-out row {engine_rows[j]}: the reference's "
            f"{_ids_text(want)}, the runtime's session prefix {prefix} and then "
            f"{_ids_text(got)}",
        )
    fields["decision"]["session_prefix"] = prefix
    state.add(
        Check.passed(
            PROMPT_CHECK,
            f"on all {n_engine} rows the runtime's prompt ids, after "
            + (f"the ids {prefix} its session puts first" if prefix else "nothing put first")
            + ", are the reference's",
        ),
        events,
    )
    try:
        ref_scores = [summed(ref.scores[i], lengths) for i in range(len(prompts))]
    except ValueError as exc:
        return refuse(REFERENCE_CHECK, f"the reference scored too few tokens: {exc}")
    try:
        eng_scores = [summed(eng.scores[i], lengths) for i in range(n_engine)]
    except ValueError as exc:
        return refuse(RUNTIME_CHECK, f"the runtime scored too few tokens: {exc}")
    calib_scores, held_scores = ref_scores[:calib_n], ref_scores[calib_n:]

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
    if fit.rule == calibration.C2 and fit.out_of_scope_rows < THIN_OUT_OF_SCOPE:
        state.limitations.append(
            f"the calibration rows hold {fit.out_of_scope_rows} out-of-scope row(s); with fewer "
            f"than {THIN_OUT_OF_SCOPE}, the rejection rule lets none of them be answered, so the "
            "most confidently misread one sets the threshold"
        )
    payload = section_payload(decision, fit, lengths, terminator, method)
    fields["decision"].update(
        {
            "fit": fit.as_dict(),
            "section": {
                "name": SECTION_NAME,
                "payload": payload,
                "bytes": len(payload_bytes(payload)),
            },
        }
    )
    ref_answers = answers(held_scores, fit.temperature)
    ref_metrics = outcome_metrics(ref_answers, heldout.targets, fit, none_index, decision.type)
    fields["measurements"] = {"reference": ref_metrics}

    # -- does the rule hold on held-out rows? (Choice) ----------------------------
    if decision.type == "choice" and fit.rule is not None:
        state.add(_rule_check(fit, ref_metrics), events)

    # -- runtime against reference, on the same rows -------------------------------
    sub_ref = answers([held_scores[i] for i in engine_rows], fit.temperature)
    sub_eng = answers(eng_scores, fit.temperature)
    targets = tuple(heldout.targets[i] for i in engine_rows)
    agree = sum(a == b for a, b in zip(sub_ref.choice, sub_eng.choice, strict=True)) / n_engine
    # What the application returns: the option, or an abstention. A runtime
    # that keeps the reference's top option but crosses the threshold the other
    # way gives another answer, which the top option alone does not show.
    given_ref = [_given(sub_ref, i, fit, none_index) for i in range(n_engine)]
    given_eng = [_given(sub_eng, i, fit, none_index) for i in range(n_engine)]
    answer_agree = sum(a == b for a, b in zip(given_ref, given_eng, strict=True)) / n_engine
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
        "backend": job.backend,
        # Whether the kernel showed the GPU work (macOS); False on the CPU.
        "backend_observed": gpu_observed(job.backend, eng.report.get("device")),
        "runtime_version": (eng.report.get("runtime") or {}).get("litert-lm"),
        "cpu": {"model": eng.report.get("cpu")},
        "threads": job.threads if job.backend == "cpu" else None,
    }
    fields["measurements"]["reference_on_runtime_rows"] = outcome_metrics(
        sub_ref, targets, fit, none_index, decision.type
    )
    fields["quality"] = {
        "available": True,
        "rows": n_engine,
        "argmax_agreement": agree,
        "answer_agreement": answer_agree,
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
        "measures": (
            "per decision: the session's prefill, the re-prefill of mode 'once', and every "
            "option's scoring; not rendering or tokenising the prompt"
            if method == METHOD_REWIND
            else "per decision, for every option: an Engine load, a session, the prompt's "
            "prefill, the re-prefill of mode 'once' and the option's scoring; not rendering "
            "or tokenising the prompt"
        ),
        "method": method,
        "ms_per_decision_median": statistics.median(ms),
        "ms_per_decision_max": max(ms),
        "options": len(decision.ids),
        "scored_tokens_per_decision": sum(lengths),
        "mode": mode,
        "threads": job.threads,
        "cpu": eng.report.get("cpu"),
        "runtime": eng.report.get("runtime"),
        "rows": n_engine,
    }
    worst = min(agree, answer_agree)
    state.add(
        (Check.passed if worst >= AGREEMENT_FLOOR else Check.failed)(
            AGREEMENT_CHECK,
            f"the runtime's top option is the reference's on {agree:.4f} and its answer, "
            f"abstentions included, on {answer_agree:.4f} of {n_engine} rows (refused below "
            f"{AGREEMENT_FLOOR}); mean KL {fields['quality']['mean_kl']:.2e}",
            observed={"argmax": agree, "answer": answer_agree, "rows": n_engine},
        ),
        events,
    )
    if n_engine < len(heldout.prompts):
        state.limitations.append(
            f"the runtime scored {n_engine} of the {len(heldout.prompts)} held-out rows, spread "
            "evenly through the file; the reference scored them all. Runtime accuracy and "
            "agreement are on those rows"
        )
    state.limitations.append(TEMPERATURE_CARRIED)

    # -- the float twin -----------------------------------------------------------------
    runtime_version = (eng.report.get("runtime") or {}).get("litert-lm")
    if job.float_twin is None:
        if runtime_version == MEASURED_RUNTIME and mode == "once":
            state.limitations.append(NO_TWIN.format(runtime=MEASURED_RUNTIME))
        elif runtime_version == MEASURED_RUNTIME:
            state.add(
                Check.unchecked(
                    TWIN_CHECK,
                    f"the self-check picked mode {mode!r} on litert-lm {MEASURED_RUNTIME}, where "
                    "the measured method is 'once', and no --float-twin was given to show which "
                    "is right on this build",
                ),
                events,
            )
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
            _twin_check(
                job, decision, terminator, heldout, engine_rows, calib_n, ref, mode, method, events
            ),
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


def _recorded_turn(recorded: Any) -> tuple[list[int], str] | str:
    """The end of turn `tune` recorded, as (ids, text), or why it cannot be scored with."""
    if not isinstance(recorded, dict):
        return (
            "the checkpoint records no end of turn in the shape `tune` writes ({ids, source, "
            f"text}}), got {json.dumps(recorded)[:200]}; option scoring scores each option "
            "followed by the first token of the turn's end, and nothing else says which it is"
        )
    ids, text, source = recorded.get("ids"), recorded.get("text"), recorded.get("source")
    if source != "chat_template":
        return (
            f"`tune` took this checkpoint's end of turn from {source!r}, not from its chat "
            "template; a decision is trained and scored through the template's turn, and the "
            "token that closes it is what each option is scored with"
        )
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids)
        or not isinstance(text, str)
        or not text
    ):
        return (
            "the checkpoint's recorded end of turn has no ids or no text: "
            f"{json.dumps(recorded)[:200]}"
        )
    return list(ids), text


def _engine_timeout(rows: int, options: int, method: str) -> int:
    """Seconds for the runtime script: start-up, the self-check's loads, and every row's."""
    if method == METHOD_FRESH_ENGINE:
        per_row = ENGINE_ROW_TIMEOUT_S + ENGINE_LOAD_TIMEOUT_S * options
        return STARTUP_TIMEOUT_S + ENGINE_LOAD_TIMEOUT_S * SELFCHECK_LOADS + per_row * rows
    return STARTUP_TIMEOUT_S + ENGINE_ROW_TIMEOUT_S * rows


def _ids_text(ids: list[int] | None) -> str:
    if ids is None:
        return "no ids"
    return f"{len(ids)} ids, {ids[:6]}{' ...' if len(ids) > 6 else ''}"


def _given(side: Answers, i: int, fit: calibration.Fit, none_index: int | None) -> int | None:
    """The answer row `i` gets: its top option, or `None` where it abstains."""
    if fit.rule is None:
        return side.choice[i]
    forced = side.choice[i] == none_index
    return side.choice[i] if calibration.answers(side.top[i], forced, fit.abstain_below) else None


def _rule_check(fit: calibration.Fit, held: dict[str, Any]) -> Check:
    """Whether the held-out rows clearly contradict the rule the threshold was fitted by.

    Under either rule, a threshold that answers no in-scope held-out row fails:
    the rejection rule alone is met by a decider that abstains on everything.
    """
    if not held.get("in_scope_rows"):
        return Check.unchecked(
            RULE_CHECK, "the held-out rows have no in-scope row to measure answers on"
        )
    if not held.get("answered"):
        return Check.failed(
            RULE_CHECK,
            f"the model abstained on every one of the {held['in_scope_rows']} in-scope held-out "
            "rows at the fitted threshold: as calibrated, it would answer nothing",
            observed=held,
        )
    if fit.rule == calibration.C2 and held.get("out_of_scope_rows"):
        bound = held["out_of_scope_rejected_upper_bound_95"]
        detail = (
            f"{held['out_of_scope_rejected']:.4f} of {held['out_of_scope_rows']} out-of-scope "
            f"rows rejected; the 95 % upper bound {bound:.4f} against the "
            f"{calibration.REJECT_TARGET} target"
        )
        ok = bound >= calibration.REJECT_TARGET
    elif fit.rule == calibration.C1:
        bound = held["error_lower_bound_95"]
        detail = (
            f"{held['error_on_answered']:.4f} error on {held['answered']} answered rows; the 95 % "
            f"lower bound {bound:.4f} against the {calibration.ERROR_BUDGET} budget"
        )
        ok = bound <= calibration.ERROR_BUDGET
    else:
        return Check.unchecked(
            RULE_CHECK, "the held-out rows have no out-of-scope row to measure rejection on"
        )
    return (Check.passed if ok else Check.failed)(RULE_CHECK, detail, observed=held)


def _twin_check(
    job: DecisionVerify,
    decision: Decision,
    terminator: str,
    heldout: DecisionRows,
    engine_rows: Sequence[int],
    calib_n: int,
    ref: ScoreRun,
    mode: str | None,
    method: str,
    events: EventStream,
) -> Check:
    """The float twin's per-token scores against the reference's, on a few of the runtime's rows."""
    rows = list(engine_rows[:PROBE_ROWS])
    n = len(rows)
    twin = run_script(
        job.runtime_env,
        ENGINE_SCRIPT,
        {
            "model": str(job.float_twin),
            "rows": [{"index": j, "prompt": heldout.prompts[i]} for j, i in enumerate(rows)],
            "options": list(decision.ids),
            "terminator": terminator,
            "threads": job.threads,
            "targets": "full",
            "mode": mode or "auto",
            "method": method,
        },
        timeout=_engine_timeout(n, len(decision.ids), method),
        auto_provision=False,
        events=events,
    )
    if twin.failure or len(twin.scores) < n:
        said = twin.failure or (
            "the float twin left rows unscored"
            + (f": {next(iter(twin.errors.values()))}" if twin.errors else "")
        )
        return Check.failed(TWIN_CHECK, said)
    unusable = _not_finite(twin.scores)
    if unusable is not None:
        return Check.failed(TWIN_CHECK, f"row {unusable} has a twin score that is not finite")
    worst = 0.0
    for j, i in enumerate(rows):
        for a, b in zip(twin.scores[j], ref.scores[calib_n + i], strict=True):
            if len(a) != len(b):
                return Check.failed(TWIN_CHECK, "the float twin scored targets of another length")
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b, strict=True)))
    detail = (
        f"largest per-token difference {worst:.2e} nats on {n} rows, every option, in mode "
        f"{mode!r} (refused above {job.twin_tolerance})"
    )
    observed = {"max_abs_token": worst, "rows": n, "mode": mode}
    return (Check.passed if worst <= job.twin_tolerance else Check.failed)(
        TWIN_CHECK, detail, observed=observed
    )
