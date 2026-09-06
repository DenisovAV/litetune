"""The label-free tier: is this model alive?

Five checks, in this order, short-circuiting on the first that is not `passed`:

1. every generation exited zero
2. output is non-empty
3. no special or padding token leaked into the output
4. no degenerate repetition
5. output diverges materially from a named baseline model's on the same prompts

The order is the point. Comparison is meaningless until generation is known to
have succeeded: three identical *crashes* compare equal exactly as well as three
identical generations, and during the work behind this tool that false pass was
read as "decoding is deterministic". So check 1 gates everything after it, and
`Generation.harness_error` -- a process that never started -- produces
`could_not_check` rather than `failed`, because a missing shared library says
nothing about the model.

**This tier is never a quality claim.** A LoRA run scoring 0.0625 against a
0.5625 base -- nine times worse than doing nothing -- passed all five checks
including divergence from base. It was alive, fluent and wrong. Only held-out
measurement caught it. `verify` therefore uses this tier as a gate on quality
measurement and never as a substitute for one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.evaluate import MeasurementPoint
from litetune.events import EventStream
from litetune.metrics import TERMINATORS, comparable_form, trim_terminator


@dataclass(frozen=True)
class LivenessThresholds:
    """Where each label-free check draws its line.

    The shares exist because a single flake among 640 prompts is not a dead
    model, and the repetition threshold is set far from both observed
    populations: a correct single call scores near 0, an observed degenerate
    decode scores above 0.9.
    """

    max_empty_share: float = 0.0
    max_leak_share: float = 0.0
    max_degenerate_share: float = 0.05
    repetition_ratio: float = 0.5
    # Below this many tokens a text cannot be degenerate in any useful sense; a
    # correct single call is about ten tokens long, and scoring repetition on it
    # would fail every correct output.
    repetition_min_tokens: int = 12
    repetition_ngram: int = 4
    min_divergence_share: float = 0.10
    # A decoder asked to keep special tokens ends every generation that stopped
    # on its own with one. A generation without one either ran to the token
    # bound or closed its turn with a marker this tool's vocabulary does not
    # list -- and the second is indistinguishable from the first in the text
    # alone, which is why this is a share and not a per-row verdict. The
    # observed rate of hitting the bound on a healthy run is about 1% (8 of
    # 640, README); this leaves an order of magnitude above it.
    max_unterminated_share: float = 0.10

    def as_dict(self) -> dict:
        return {
            "max_empty_share": self.max_empty_share,
            "max_leak_share": self.max_leak_share,
            "max_degenerate_share": self.max_degenerate_share,
            "repetition_ratio": self.repetition_ratio,
            "repetition_min_tokens": self.repetition_min_tokens,
            "repetition_ngram": self.repetition_ngram,
            "min_divergence_share": self.min_divergence_share,
            "max_unterminated_share": self.max_unterminated_share,
        }


DEFAULT_THRESHOLDS = LivenessThresholds()

# Special tokens that must not appear in decoded output. `<escape>` is
# deliberately absent: it is part of FunctionGemma's wire format, and flagging
# it would fail every correct call.
_SPECIAL_TOKEN_RE = re.compile(
    r"<(pad|unk|bos|s|/s|\|endoftext\|)>|<unused\d+>|<start_of_turn>|<0x[0-9A-Fa-f]{2}>"
)


def ends_with_terminator(text: str) -> bool:
    """Did this generation stop on its own?

    The distinction matters because a generation that emitted a terminator
    cannot have been cut by a token bound, whichever side's bound it was --
    which is what lets two points with different bounds still be compared.
    """
    return text.strip() != "" and text.strip().endswith(TERMINATORS)


def unterminated_count(point: MeasurementPoint) -> int:
    """Generations that do not end with a terminator.

    One that emitted a terminator stopped on its own, so no token bound --
    whichever side's -- can have altered it. One that did not was still running
    when something cut it, and two sides with different bounds may have cut it
    in different places. That is what decides whether a decoding difference
    could have mattered, as opposed to whether one was declared.

    A free function here rather than a property on `MeasurementPoint`: reading a
    terminator out of generated text is interpretation, which is what this
    module is for, and putting it on the record of what happened made
    `evaluate` and `liveness` import each other.

    The vocabulary checked against is `metrics.TERMINATORS`, recorded verbatim
    at `harness.terminators` in the manifest; two runs' counts are comparable
    only when that field agrees between them.
    """
    return sum(1 for g in point.generations if g.ok and not ends_with_terminator(g.text))


def leaked_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _SPECIAL_TOKEN_RE.finditer(trim_terminator(text))]


def repetition_ratio(text: str, ngram: int = 4, min_tokens: int = 12) -> float:
    """Share of n-grams that are repeats. 0.0 for text too short to judge."""
    tokens = trim_terminator(text).split()
    if len(tokens) < max(min_tokens, ngram + 1):
        return 0.0
    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def divergence_share(candidate: Sequence[str], baseline: Sequence[str]) -> float:
    """Share of prompts where the two models produced different calls."""
    if len(candidate) != len(baseline):
        raise ValueError(f"{len(candidate)} outputs against {len(baseline)}: not the same prompts")
    if not candidate:
        raise ValueError("no outputs to compare")
    differing = sum(
        1
        for a, b in zip(candidate, baseline, strict=True)
        if comparable_form(a) != comparable_form(b)
    )
    return differing / len(candidate)


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def _checked(name: str, body: Callable[[], Check]) -> Check:
    """Run one check body. An exception in it means the check did not run."""
    with guard(name) as sink:
        sink.append(body())
    return sink[0]


def exit_status_check(point: MeasurementPoint) -> Check:
    name = "exit status"
    if point.n == 0:
        return Check.unchecked(name, "no generations were attempted")
    not_performed = [g for g in point.generations if g.harness_error is not None]
    if not_performed:
        first = not_performed[0]
        # The process never ran. That is a statement about this machine, not
        # about the model, and reporting it as a failure is the exact mistake
        # checks.py exists to prevent.
        return Check.unchecked(
            name,
            f"{len(not_performed)}/{point.n} generations were not performed: "
            f"{first.harness_error}",
            observed={"not_performed": len(not_performed), "n": point.n},
        )
    failed = [g for g in point.generations if not g.ok]
    if failed:
        first = failed[0]
        return Check.failed(
            name,
            f"{len(failed)}/{point.n} generations exited non-zero "
            f"(first: rc={first.returncode}, {first.stderr.strip()[-200:] or 'no stderr'})",
            observed={"failed": len(failed), "n": point.n, "first_returncode": first.returncode},
        )
    return Check.passed(
        name, f"{point.n}/{point.n} generations exited zero", observed={"n": point.n}
    )


def non_empty_check(point: MeasurementPoint, thresholds: LivenessThresholds) -> Check:
    name = "non-empty output"
    empty = [g.index for g in point.generations if not trim_terminator(g.text)]
    share = len(empty) / point.n
    observed = {"empty": len(empty), "n": point.n, "share": round(share, 6)}
    if share > thresholds.max_empty_share:
        return Check.failed(
            name,
            f"{len(empty)}/{point.n} generations were empty after decoding "
            f"(share {share:.4f} over {thresholds.max_empty_share:.4f})",
            observed=observed,
        )
    return Check.passed(name, f"all {point.n} generations produced text", observed=observed)


def token_hygiene_check(point: MeasurementPoint, thresholds: LivenessThresholds) -> Check:
    name = "no special-token leakage"
    leaks = {g.index: leaked_tokens(g.text) for g in point.generations}
    leaking = {i: t for i, t in leaks.items() if t}
    share = len(leaking) / point.n
    observed = {"leaking": len(leaking), "n": point.n, "share": round(share, 6)}
    if share > thresholds.max_leak_share:
        first_index, first_tokens = next(iter(leaking.items()))
        return Check.failed(
            name,
            f"{len(leaking)}/{point.n} generations leaked special tokens "
            f"(first at index {first_index}: {', '.join(sorted(set(first_tokens)))})",
            observed=observed | {"examples": sorted({t for ts in leaking.values() for t in ts})},
        )
    return Check.passed(
        name, f"no padding or special tokens in {point.n} generations", observed=observed
    )


def repetition_check(point: MeasurementPoint, thresholds: LivenessThresholds) -> Check:
    name = "no degenerate repetition"
    ratios = [
        repetition_ratio(g.text, thresholds.repetition_ngram, thresholds.repetition_min_tokens)
        for g in point.generations
    ]
    degenerate = [i for i, r in enumerate(ratios) if r > thresholds.repetition_ratio]
    share = len(degenerate) / point.n
    worst = max(ratios) if ratios else 0.0
    observed = {
        "degenerate": len(degenerate),
        "n": point.n,
        "share": round(share, 6),
        "worst_ratio": round(worst, 6),
    }
    if share > thresholds.max_degenerate_share:
        return Check.failed(
            name,
            f"{len(degenerate)}/{point.n} generations repeat themselves above "
            f"{thresholds.repetition_ratio:.2f} (worst {worst:.4f})",
            observed=observed,
        )
    return Check.passed(
        name,
        f"worst repetition ratio {worst:.4f}, threshold {thresholds.repetition_ratio:.2f}",
        observed=observed,
    )


def divergence_check(
    point: MeasurementPoint,
    baseline: Sequence[str],
    baseline_label: str,
    thresholds: LivenessThresholds,
) -> Check:
    """Does the candidate say anything different from the baseline model?

    Only meaningful against a *different* model -- an untuned base. Against the
    float twin of the same weights, agreement is the desired outcome of a
    lossless conversion, so `verify` skips this check there and records why.
    """
    name = "divergence from baseline"
    share = divergence_share(point.texts, baseline)
    observed = {
        "divergence_share": round(share, 6),
        "n": point.n,
        "baseline": baseline_label,
        "threshold": thresholds.min_divergence_share,
    }
    if share < thresholds.min_divergence_share:
        return Check.failed(
            name,
            f"outputs differ from {baseline_label} on only {share:.4f} of prompts "
            f"(threshold {thresholds.min_divergence_share:.2f}); the model under test may be "
            "the baseline",
            observed=observed,
        )
    return Check.passed(
        name, f"differs from {baseline_label} on {share:.4f} of prompts", observed=observed
    )


@dataclass(frozen=True)
class SkippedCheck:
    name: str
    reason: str

    def as_dict(self) -> dict:
        return {"name": self.name, "reason": self.reason}


@dataclass
class LivenessResult:
    """The tier's checks plus an explicit record of any that did not apply.

    `skipped` is not a fourth outcome. It records that a check was never in
    scope for this run, so a reader of the manifest can see that four checks ran
    rather than assuming five did.
    """

    checks: CheckSet
    skipped: list[SkippedCheck] = field(default_factory=list)

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASSED

    def as_dict(self) -> dict:
        return self.checks.as_dict() | {"skipped": [s.as_dict() for s in self.skipped]}


def liveness_tier(
    point: MeasurementPoint,
    thresholds: LivenessThresholds = DEFAULT_THRESHOLDS,
    baseline: Sequence[str] | None = None,
    baseline_label: str = "baseline",
    baseline_absent_reason: str | None = "no baseline model outputs were supplied",
    events: EventStream | None = None,
) -> LivenessResult:
    """Run the label-free tier in order, stopping at the first non-pass.

    Passing `baseline=None` with `baseline_absent_reason=None` means the caller
    will handle the divergence check itself -- `verify` does, so that it does
    not pay for baseline generation before the candidate is known to be alive.
    """
    result = LivenessResult(checks=CheckSet(name=f"liveness:{point.label}"))

    bodies: list[tuple[str, Callable[[], Check]]] = [
        ("exit status", lambda: exit_status_check(point)),
        ("non-empty output", lambda: non_empty_check(point, thresholds)),
        ("no special-token leakage", lambda: token_hygiene_check(point, thresholds)),
        ("no degenerate repetition", lambda: repetition_check(point, thresholds)),
    ]
    if baseline is not None:
        bodies.append(
            (
                "divergence from baseline",
                lambda: divergence_check(point, baseline, baseline_label, thresholds),
            )
        )

    for position, (name, body) in enumerate(bodies):
        check = _checked(name, body)
        result.checks.add(check)
        if events:
            events.check(check)
        if check.outcome is not Outcome.PASSED:
            # Short-circuit. Every later check would be comparing outputs whose
            # production is not established, and a comparison over unestablished
            # output is how three crashes once read as determinism.
            for skipped_name, _ in bodies[position + 1 :]:
                result.skipped.append(
                    SkippedCheck(skipped_name, f"not reached: {name} did not pass")
                )
            return result

    if baseline is None and baseline_absent_reason is not None:
        result.skipped.append(SkippedCheck("divergence from baseline", baseline_absent_reason))
    return result
