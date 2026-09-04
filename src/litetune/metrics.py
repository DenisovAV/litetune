"""Metrics over parsed structured outputs, with their uncertainty attached.

Two rules here were measured rather than chosen.

**Operation name and arguments are reported separately**, because they degrade
separately. Across the runs behind this tool the selected operation stayed at
0.995 while argument accuracy absorbed essentially all of the conversion loss.
A single exact-match number hides which of the two moved, and therefore hides
whether the fix is a different quantization recipe or more training data.

**Every proportion carries `n` and an interval**, because point estimates from
small held-out sets were repeatedly wrong: at n=64 a recipe comparison read as
0.172; at n=640 the same comparison was 0.024, and three conclusions drawn at
the smaller size were overturned. `difference()` therefore refuses to call a gap
real when the interval covers it, and says so in words the report can print.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

# 95% two-sided normal quantile.
Z95 = 1.959963985


# ---------------------------------------------------------------------------
# The wire format
# ---------------------------------------------------------------------------
#
# FunctionGemma emits `call:NAME{key:<escape>value<escape>,...}`. Values are
# delimited by a literal `<escape>` marker rather than quoted, so a value may
# contain commas and braces; the parser below therefore scans argument pairs
# instead of splitting the body on punctuation. `<escape>` is part of the
# format, not a leaked special token -- see liveness.py, where flagging it would
# fail every correct output.

_HEAD_RE = re.compile(r"call:\s*(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)\s*\{")
_ARG_RE = re.compile(
    r"\s*(?P<key>[A-Za-z_][A-Za-z0-9_.\-]*)\s*:\s*<escape>(?P<value>.*?)<escape>\s*",
    re.DOTALL,
)
_CLOSE_RE = re.compile(r"\s*\}")


def _stringify(value: Any) -> str:
    """Render a target value the way the wire format would carry it.

    The format is untyped: a target of `3` and an emitted `3` are the same
    answer, so targets are stringified before comparison. Booleans and null go
    to their JSON spellings because that is the shape the training data used;
    `str(True)` would compare `"True"` against an emitted `"true"` and score a
    correct answer wrong.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value).strip()


@dataclass(frozen=True)
class ToolCall:
    """One parsed call. `args` values are always strings; see `_stringify`."""

    name: str
    args: dict[str, str]

    def __post_init__(self) -> None:
        # Normalise on construction so that equality means "same answer"
        # regardless of which side of the comparison a value came from.
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "args", {str(k): _stringify(v) for k, v in self.args.items()})

    @classmethod
    def from_target(cls, obj: Any) -> ToolCall | None:
        """Build from a held-out label. Returns None for an unlabelled example."""
        if obj is None:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            raise ValueError(
                "target must be an object with a 'name' field (a tool call) or a string "
                f"(the answer itself), got {obj!r}"
            )
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"target 'args' must be an object, got {args!r}")
        return cls(name=str(obj["name"]), args=args)

    def as_dict(self) -> dict:
        return {"name": self.name, "args": dict(self.args)}


def read_target(obj: Any) -> ToolCall | str | None:
    """A held-out label, in either shape the shipped scorers understand.

    An object with a `name` is a tool call and is scored by `tool-call`; a bare
    string is the answer itself and is scored by `exact-text`. `None` is an
    unlabelled example, which is a documented state and not an error.

    Two shapes rather than a schema, because the shape *is* the declaration of
    what the task is, and asking for both a target and a `--target-kind` would
    let them disagree.
    """
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return ToolCall.from_target(obj)


def parse_call(text: str) -> ToolCall | None:
    """Parse the first call in `text`.

    Returns None when the text contains no well-formed call. That is a
    documented result -- "the model did not emit the format" -- and is counted
    by `parse_rate`, not raised: a malformed generation is data.
    """
    head = _HEAD_RE.search(text)
    if head is None:
        return None
    pos = head.end()
    args: dict[str, str] = {}
    while True:
        if _CLOSE_RE.match(text, pos) is not None:
            return ToolCall(name=head.group("name"), args=args)
        arg = _ARG_RE.match(text, pos)
        if arg is None:
            # A body that starts but never closes cleanly is not half a call.
            return None
        args[arg.group("key")] = arg.group("value")
        pos = arg.end()
        if text[pos : pos + 1] == ",":
            pos += 1


# ---------------------------------------------------------------------------
# Estimates and their uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Unavailable:
    """A quantity that was not measured, and why.

    Distinct from a zero and from a failure: the caller asked for a number and
    is being told it does not exist, which is the only honest answer when a
    measurement point is missing.
    """

    reason: str

    @property
    def available(self) -> bool:
        return False

    def as_dict(self) -> dict:
        return {"available": False, "reason": self.reason}


@dataclass(frozen=True)
class Proportion:
    """A proportion with its sample size and 95% Wald half-width."""

    value: float
    n: int
    ci95: float

    @property
    def available(self) -> bool:
        return True

    @classmethod
    def of(cls, successes: int, n: int) -> Proportion:
        if n <= 0:
            # No denominator means no estimate. Returning 0.0 here would print
            # as a score, which is exactly the confident-answer-without-a-
            # measurement failure this tool exists to prevent.
            raise ValueError("cannot form a proportion from an empty sample")
        if not 0 <= successes <= n:
            raise ValueError(f"successes={successes} outside 0..{n}")
        p = successes / n
        # Wald. At p exactly 0 or 1 the half-width collapses to zero, which
        # overstates certainty; Wilson would not. Kept because the reports and
        # thresholds behind this tool were all computed as Wald, and a metric
        # whose definition changes silently is worse than a known-conservative
        # one. Interpret a zero-width interval as "degenerate", not "certain".
        half = Z95 * math.sqrt(p * (1.0 - p) / n)
        return cls(value=p, n=n, ci95=half)

    @property
    def low(self) -> float:
        return max(0.0, self.value - self.ci95)

    @property
    def high(self) -> float:
        return min(1.0, self.value + self.ci95)

    def as_dict(self) -> dict:
        return {
            "available": True,
            "value": round(self.value, 6),
            "n": self.n,
            "ci95": round(self.ci95, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
        }


@dataclass(frozen=True)
class Difference:
    """A difference between two measurements, and whether the sample resolves it."""

    value: float
    ci95: float
    n_a: int
    n_b: int
    detail: str
    method: str = "unpaired"
    # Examples on which the two sides disagreed. Only the paired form knows it,
    # and it is what makes that form the sharper instrument.
    discordant: int | None = None

    @property
    def available(self) -> bool:
        return True

    @property
    def resolved(self) -> bool:
        return abs(self.value) > self.ci95

    def as_dict(self) -> dict:
        return {
            "available": True,
            "value": round(self.value, 6),
            "ci95": round(self.ci95, 6),
            "n": [self.n_a, self.n_b],
            "resolved": self.resolved,
            "method": self.method,
            "discordant": self.discordant,
            "detail": self.detail,
        }


def _difference_detail(value: float, half: float, n: int) -> str:
    if abs(value) > half:
        return f"difference {value:+.4f} exceeds the interval ±{half:.4f} at n={n}"
    return (
        f"the sample does not resolve this difference: {value:+.4f} lies inside "
        f"±{half:.4f} at n={n}"
    )


def difference(a: Proportion, b: Proportion) -> Difference:
    """`a - b` from two proportions alone, treating the samples as independent.

    Use `paired_difference` when the per-example outcomes are in hand, which for
    this tool is always: both sides run the same prompts. This form is kept for
    comparing measurements that were not paired, and it is deliberately the
    conservative one -- on the measured recipe numbers (0.9094 against 0.8906 at
    n=640) it reports ±0.0329 and calls a real 0.019 effect unresolved.
    """
    value = a.value - b.value
    half = math.sqrt(a.ci95**2 + b.ci95**2)
    return Difference(
        value=value,
        ci95=half,
        n_a=a.n,
        n_b=b.n,
        detail=_difference_detail(value, half, min(a.n, b.n)),
    )


def paired_difference(a: Sequence[bool], b: Sequence[bool]) -> Difference:
    """`a - b` over per-example outcomes on the *same* prompts.

    Both sides answer the same held-out examples, so the samples are paired and
    only the examples where they disagree carry information -- the McNemar
    standard error, `sqrt(discordant)/n`. This is not a refinement for its own
    sake: on the measured 0.9094-against-0.8906 comparison at n=640 the unpaired
    interval is ±0.0329 and the paired one is ±0.0162, so the unpaired form
    reports the effect this tool exists to measure as unresolved and the paired
    one resolves it.
    """
    if len(a) != len(b):
        raise ValueError(f"{len(a)} outcomes against {len(b)}: not the same examples")
    n = len(a)
    if n == 0:
        raise ValueError("cannot difference an empty sample")
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    discordant = a_only + b_only
    value = (a_only - b_only) / n
    half = Z95 * math.sqrt(discordant) / n
    if discordant == 0:
        detail = f"the two sides agreed on all {n} examples; the difference is exactly zero"
    else:
        detail = (
            f"{_difference_detail(value, half, n)} " f"({discordant} of {n} examples discordant)"
        )
    return Difference(
        value=value,
        ci95=half,
        n_a=n,
        n_b=n,
        detail=detail,
        method="paired (McNemar)",
        discordant=discordant,
    )


# ---------------------------------------------------------------------------
# Task metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityMetrics:
    """Scores for one measurement point.

    `exact_match` is the one field every scorer produces: the proportion of
    examples answered correctly, and the only one the statistics downstream
    consume. Everything else here is a *decomposition*, and a scorer that has
    none says so rather than inventing one.

    For tool calls the decomposition is `name_accuracy * argument_accuracy`,
    because argument accuracy is conditioned on the operation having been
    selected correctly. Reporting it that way is what makes "the recipe cost
    0.024" attributable -- 0.995 of the operations survived conversion and the
    loss lived entirely in the arguments. A task with no such split -- one right
    string, and you either produced it or did not -- leaves both `Unavailable`.
    """

    n: int
    parse_rate: Proportion
    name_accuracy: Proportion | Unavailable
    argument_accuracy: Proportion | Unavailable
    exact_match: Proportion
    # Per-example exact-match outcome, in split order. Kept because the two
    # sides of a comparison answer the same prompts, and `paired_difference`
    # needs the pairing to produce an interval sharp enough to resolve the
    # effects this tool reports.
    correct: tuple[bool, ...] = ()

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "parse_rate": self.parse_rate.as_dict(),
            "name_accuracy": self.name_accuracy.as_dict(),
            "argument_accuracy": self.argument_accuracy.as_dict(),
            "argument_accuracy_denominator": "examples whose operation name was correct",
            "exact_match": self.exact_match.as_dict(),
        }


class Scorer(Protocol):
    """Turns generated text and held-out labels into `QualityMetrics`.

    The seam that makes the rest of this package task-agnostic. Everything
    downstream -- the paired comparison, the intervals, whether a difference
    resolves, the exit code -- consumes `QualityMetrics.correct`, a tuple of
    booleans. Only the way a boolean is arrived at is specific to a task, so
    that is the only thing worth replacing.

    Implementations raise rather than returning a degenerate score when the
    inputs cannot be scored at all: callers run them inside `checks.guard`, so
    the result is `could not check` rather than a fabricated number.
    """

    @property
    def name(self) -> str:
        """How this scorer is named in the manifest and on the command line."""
        ...

    @property
    def describes(self) -> str:
        """What "correct" means here, in one line, for the report to carry."""
        ...

    def __call__(self, targets: Sequence[Any], outputs: Sequence[str]) -> QualityMetrics: ...


def _require_alignment(targets: Sequence[Any], outputs: Sequence[str]) -> int:
    if len(targets) != len(outputs):
        raise ValueError(f"{len(targets)} targets against {len(outputs)} outputs")
    if not targets:
        raise ValueError("cannot score an empty split")
    return len(targets)


def score(targets: Sequence[ToolCall], outputs: Sequence[str]) -> QualityMetrics:
    """Score generated text against held-out labels.

    Raises rather than returning a degenerate score when the inputs cannot be
    scored at all; callers run this inside `checks.guard`, so the result is
    `could not check` rather than a fabricated number.
    """
    n = _require_alignment(targets, outputs)
    parsed = [parse_call(text) for text in outputs]
    n_parsed = sum(p is not None for p in parsed)
    name_hits = [p is not None and p.name == t.name for p, t in zip(parsed, targets, strict=True)]
    n_name = sum(name_hits)
    correct = [
        bool(name_ok and p is not None and p.args == t.args)
        for p, t, name_ok in zip(parsed, targets, name_hits, strict=True)
    ]
    n_exact = sum(correct)

    if n_name:
        argument_accuracy: Proportion | Unavailable = Proportion.of(n_exact, n_name)
    else:
        argument_accuracy = Unavailable(
            "no generation selected the correct operation, so argument accuracy "
            "has no denominator on this split"
        )

    return QualityMetrics(
        n=n,
        parse_rate=Proportion.of(n_parsed, n),
        name_accuracy=Proportion.of(n_name, n),
        argument_accuracy=argument_accuracy,
        exact_match=Proportion.of(n_exact, n),
        correct=tuple(correct),
    )


def score_exact_text(targets: Sequence[Any], outputs: Sequence[str]) -> QualityMetrics:
    """Correct means the generation equals the target text, once normalised.

    For every task where there is one right answer and no structure inside it:
    a label, an extracted field, a rewritten sentence. Whitespace is collapsed
    and the ends stripped, because a trailing newline is not a wrong answer;
    nothing else is forgiven, since deciding that "almost" counts is a judgement
    about your task that this package has no way to make.

    `name_accuracy` and `argument_accuracy` are unavailable rather than zero.
    There is nothing here to decompose, and a zero would read as a measurement.
    """
    n = _require_alignment(targets, outputs)
    want = [" ".join(str(t).split()) for t in targets]
    got = [" ".join(text.split()) for text in outputs]
    correct = [w == g for w, g in zip(want, got, strict=True)]
    no_split = Unavailable(
        "exact-text scoring has no operation/argument split: the answer is one "
        "string, and it either matches or does not"
    )
    return QualityMetrics(
        n=n,
        # Every generation is "parsed": the text is the answer.
        parse_rate=Proportion.of(n, n),
        name_accuracy=no_split,
        argument_accuracy=no_split,
        exact_match=Proportion.of(sum(correct), n),
        correct=tuple(correct),
    )


@dataclass(frozen=True)
class NamedScorer:
    """A scoring function with the two things a report needs to say about it."""

    name: str
    describes: str
    fn: Callable[[Sequence[Any], Sequence[str]], QualityMetrics]

    def __call__(self, targets: Sequence[Any], outputs: Sequence[str]) -> QualityMetrics:
        return self.fn(targets, outputs)


SCORERS: dict[str, NamedScorer] = {
    "tool-call": NamedScorer(
        name="tool-call",
        describes=(
            "correct means the parsed call's operation name and every argument value match "
            "the target"
        ),
        fn=score,
    ),
    "exact-text": NamedScorer(
        name="exact-text",
        describes=(
            "correct means the generation equals the target text after collapsing whitespace"
        ),
        fn=score_exact_text,
    ),
}
"""The scorers that ship. `verify` names the one it used in the manifest.

Two, not one, because `tune` and `convert` were never specific to tool calls --
only the measurement was. Two, not a plugin system, because which metrics people
actually want is not yet known, and inventing an extension point for it would be
guessing in public.
"""


def comparable_form(text: str) -> tuple:
    """Normalise one output for equality against another model's output."""
    call = parse_call(text)
    if call is None:
        return ("raw", text.strip())
    return ("call", call.name, tuple(sorted(call.args.items())))


def agreement(a: Sequence[str], b: Sequence[str]) -> Proportion:
    """Share of prompts on which two models produced the same call.

    Label-free, and therefore never a quality claim on its own: three identical
    empty outputs agree perfectly. Callers must establish liveness on both sides
    first -- see liveness.py.
    """
    if len(a) != len(b):
        raise ValueError(f"{len(a)} outputs against {len(b)}: not the same prompts")
    same = sum(1 for x, y in zip(a, b, strict=True) if comparable_form(x) == comparable_form(y))
    return Proportion.of(same, len(a))
