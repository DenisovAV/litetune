"""The calibration of a decision model's option scores: a temperature, and for Choice a threshold.

Pure standard library. The parent process has no numpy, and `verify` fits on
the reference's scores and then measures the engine's at the same values, so
this is the one implementation of both.

**Scores** are one per option, each the sum of the log-probabilities of the
option's shortest unique prefix (`decisions.py`, contract version 1). The
probabilities are `softmax(scores / T)`.

**The temperature** is fitted by NLL on the in-scope calibration rows: a grid
of 241 points over log T in [0.01, 1000], then 60 ternary steps between the
grid point's neighbours. That is the fit the option-scoring measurements of
2026-10-15 and 2026-10-16 used, and `tests/test_calibration.py` holds it to
their recorded temperatures.

**Abstention, Choice only.** A row abstains when its top option is the
`none_option`, or when its top probability is below the threshold. The
threshold comes from one of two rules, out-of-scope rejection measured on
2026-10-15:
- with labelled out-of-scope calibration rows, the rejection target (C2): the
  lowest observed top probability at which at most `n - ceil(0.9 n)` of the
  `n` out-of-scope rows answer;
- without them, the error budget (C1): the lowest top probability at which the
  answered in-scope calibration rows err at most 5 %. It does not reject
  out-of-scope inputs reliably, and the report says so.

Noul and Score never abstain: Jev has no abstention for them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

T_LOW = 0.01
T_HIGH = 1000.0
T_POINTS = 241
T_REFINEMENTS = 60
NLL_FLOOR = 1e-300

REJECT_TARGET = 0.9
ERROR_BUDGET = 0.05

C2 = "reject_target"
C1 = "error_budget"

C1_NOTE = (
    "the calibration rows have no out-of-scope rows, so the threshold caps the error on answered "
    "in-scope rows instead (the error budget). It does not reject out-of-scope inputs reliably: "
    "measured on 2026-10-15, its held-out error exceeded the budget it kept on calibration rows"
)


class CalibrationError(ValueError):
    """Scores that cannot be calibrated. Says which rows and why."""


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    """`softmax(scores / temperature)`, with the maximum subtracted first."""
    scaled = [x / temperature for x in scores]
    top = max(scaled)
    exps = [math.exp(x - top) for x in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


def argmax(scores: Sequence[float]) -> int:
    """The first maximum, as numpy's `argmax`."""
    best = 0
    for i, x in enumerate(scores):
        if x > scores[best]:
            best = i
    return best


def _check(rows: Sequence[Sequence[float]], where: str) -> None:
    width = len(rows[0]) if rows else 0
    for i, row in enumerate(rows):
        if len(row) != width:
            raise CalibrationError(f"{where}: row {i} has {len(row)} scores, row 0 has {width}")
        if not all(math.isfinite(x) for x in row):
            raise CalibrationError(f"{where}: row {i} has a score that is not finite")


def mean_nll(rows: Sequence[Sequence[float]], targets: Sequence[int], temperature: float) -> float:
    """The mean of `-log p[target]` at `temperature`."""
    if not rows:
        raise CalibrationError("no rows to compute an NLL on")
    losses = []
    for row, target in zip(rows, targets, strict=True):
        p = softmax(row, temperature)[target]
        losses.append(-math.log(max(p, NLL_FLOOR)))
    return math.fsum(losses) / len(losses)


def fit_temperature(rows: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """The temperature with the lowest mean NLL: a log grid, then a ternary refinement."""
    _check(rows, "temperature")
    low, high = math.log(T_LOW), math.log(T_HIGH)
    step = (high - low) / (T_POINTS - 1)
    grid = [math.exp(low + i * step) for i in range(T_POINTS)]
    values = [mean_nll(rows, targets, t) for t in grid]
    best = min(range(len(grid)), key=lambda i: (values[i], i))
    lo, hi = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
    for _ in range(T_REFINEMENTS):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if mean_nll(rows, targets, m1) < mean_nll(rows, targets, m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def max_answered(n: int, reject_target: float = REJECT_TARGET) -> int:
    """How many of `n` out-of-scope rows may answer: `n - ceil(reject_target * n)`, exactly.

    In floating point `100 * (1 - 0.9)` floors to 9, where the rule's budget for
    100 rows at 0.9 is 10; and `0.55 * 100` ceils to 56, not 55.
    """
    return n - math.ceil(Fraction(repr(reject_target)) * n)


def threshold_reject_target(
    top: Sequence[float], forced: Sequence[bool], reject_target: float = REJECT_TARGET
) -> float:
    """The lowest observed top probability at which at most `max_answered` rows answer.

    `top` and `forced` are per out-of-scope calibration row: its top
    probability, and whether its top option is the `none_option`, which
    abstains whatever the threshold. When no observed value is high enough, the
    next float above the highest, so that none answers. When the rows that
    could answer are already few enough, 0.0: every top probability is at or
    above it, so every row may answer -- the measured rule's `-inf`, written as
    a number JSON has. The section is read by a JSON decoder, and `-Infinity`
    is not JSON.
    """
    allowed = max_answered(len(top), reject_target)
    free = sorted(p for p, f in zip(top, forced, strict=True) if not f)
    if len(free) <= allowed:
        return 0.0
    for i, value in enumerate(free):
        if i > 0 and value == free[i - 1]:
            continue
        if len(free) - i <= allowed:
            return value
    return math.nextafter(free[-1], math.inf)


def threshold_error_budget(
    top: Sequence[float],
    correct: Sequence[bool],
    forced: Sequence[bool],
    target_error: float = ERROR_BUDGET,
) -> float | None:
    """The lowest top probability at which the answered in-scope rows err at most `target_error`.

    Rows by descending top probability, a running error count over the rows that
    can answer, and the value of the last prefix within the budget. `None` when
    no prefix is. Rows with the same top probability are taken together, since
    a threshold at that value answers all of them: counted one at a time, two
    rows at 0.9, one right and one wrong, gave 0.9 under a 5 % budget and then
    answered both. A saturated softmax puts many rows at exactly 1.0.
    """
    rows = [(p, ok) for p, ok, f in zip(top, correct, forced, strict=True) if not f]
    rows.sort(key=lambda r: -r[0])
    errors = 0
    found: float | None = None
    for size, (value, ok) in enumerate(rows, 1):
        errors += not ok
        last_of_its_value = size == len(rows) or rows[size][0] != value
        if last_of_its_value and errors / size <= target_error:
            found = value
    return found


def answers(top: float, forced: bool, threshold: float | None) -> bool:
    """Whether a row answers rather than abstains."""
    return not forced and threshold is not None and top >= threshold


@dataclass(frozen=True)
class Fit:
    """What `verify` fits on calibration rows, and what the bundle carries."""

    temperature: float
    # Choice only; `None` for Noul and Score.
    rule: str | None
    abstain_below: float | None
    in_scope_rows: int
    out_of_scope_rows: int
    nll: float
    nll_at_1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "rule": self.rule,
            "abstain_below": self.abstain_below,
            "in_scope_rows": self.in_scope_rows,
            "out_of_scope_rows": self.out_of_scope_rows,
            "nll": self.nll,
            "nll_at_temperature_1": self.nll_at_1,
            "note": C1_NOTE if self.rule == C1 else None,
        }


def fit(
    kind: str,
    scores: Sequence[Sequence[float]],
    targets: Sequence[int],
    none_index: int | None,
) -> Fit:
    """Fit the temperature and, for Choice, the threshold on calibration rows.

    `targets` are option indices; a row whose target is `none_index` is out of
    scope. The temperature is fitted on the in-scope rows alone, as measured.
    """
    _check(scores, "calibration")
    if len(scores) != len(targets):
        raise CalibrationError(f"{len(scores)} calibration rows but {len(targets)} targets")
    in_scope = [i for i, t in enumerate(targets) if t != none_index]
    outside = [i for i, t in enumerate(targets) if t == none_index]
    if not in_scope:
        raise CalibrationError("the calibration rows have no in-scope row to fit a temperature on")
    rows = [scores[i] for i in in_scope]
    wanted = [targets[i] for i in in_scope]
    temperature = fit_temperature(rows, wanted)
    nll, nll_at_1 = mean_nll(rows, wanted, temperature), mean_nll(rows, wanted, 1.0)
    rule: str | None = None
    threshold: float | None = None
    if kind == "choice":

        def signals(indices: Sequence[int]) -> tuple[list[float], list[bool]]:
            probs = [softmax(scores[i], temperature) for i in indices]
            return [max(p) for p in probs], [argmax(scores[i]) == none_index for i in indices]

        if outside:
            rule = C2
            top, forced = signals(outside)
            threshold = threshold_reject_target(top, forced)
        else:
            rule = C1
            top, forced = signals(in_scope)
            correct = [argmax(scores[i]) == targets[i] for i in in_scope]
            threshold = threshold_error_budget(top, correct, forced)
            if threshold is None:
                raise CalibrationError(
                    f"no threshold keeps the error on answered calibration rows within "
                    f"{ERROR_BUDGET:.0%}: even the most confident row is wrong"
                )
    return Fit(
        temperature=temperature,
        rule=rule,
        abstain_below=threshold,
        in_scope_rows=len(in_scope),
        out_of_scope_rows=len(outside),
        nll=nll,
        nll_at_1=nll_at_1,
    )


# ---------------------------------------------------------------------------
# One-sided Clopper-Pearson bounds, for the held-out check
# ---------------------------------------------------------------------------


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * log_p
        + (n - i) * log_q
        for i in range(k + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


def _bisect(f, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def upper_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """The one-sided (1 - alpha) Clopper-Pearson upper bound on a proportion seen `k` of `n`."""
    if n <= 0:
        raise CalibrationError("a bound needs at least one row")
    if k >= n:
        return 1.0
    return _bisect(lambda p: _binomial_cdf(k, n, p) <= alpha, k / n, 1.0)


def lower_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """The one-sided (1 - alpha) Clopper-Pearson lower bound on a proportion seen `k` of `n`."""
    if n <= 0:
        raise CalibrationError("a bound needs at least one row")
    if k <= 0:
        return 0.0
    # P(X >= k) = 1 - P(X <= k - 1) rises with p; the bound is where it reaches alpha.
    return _bisect(lambda p: 1.0 - _binomial_cdf(k - 1, n, p) >= alpha, 0.0, k / n)
