"""Temperature, thresholds and bounds for option scores, in pure Python."""

from __future__ import annotations

import math
import random

import pytest

from litetune import calibration as C


def test_softmax_is_shift_invariant_and_sums_to_one():
    p = C.softmax([1000.0, 999.0, 998.0])
    assert math.isclose(sum(p), 1.0)
    assert p == C.softmax([2.0, 1.0, 0.0])


def test_a_higher_temperature_flattens():
    assert max(C.softmax([2.0, 0.0], 2.0)) < max(C.softmax([2.0, 0.0], 1.0))


def test_argmax_takes_the_first_maximum():
    assert C.argmax([1.0, 3.0, 3.0]) == 1


def _rows_at(temperature: float, n: int, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    """Scores whose targets are drawn from softmax(scores / temperature)."""
    rng = random.Random(seed)
    rows, targets = [], []
    for _ in range(n):
        row = [rng.gauss(0.0, 3.0) for _ in range(6)]
        p = C.softmax(row, temperature)
        targets.append(rng.choices(range(6), weights=p)[0])
        rows.append(row)
    return rows, targets


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.5])
def test_the_fit_recovers_the_temperature_the_targets_were_drawn_at(temperature):
    rows, targets = _rows_at(temperature, 4000)
    fitted = C.fit_temperature(rows, targets)
    assert abs(math.log(fitted / temperature)) < 0.1


def test_the_fitted_temperature_is_no_worse_than_its_neighbours():
    rows, targets = _rows_at(1.7, 500, seed=3)
    t = C.fit_temperature(rows, targets)
    at = C.mean_nll(rows, targets, t)
    assert at <= C.mean_nll(rows, targets, t * 1.01)
    assert at <= C.mean_nll(rows, targets, t / 1.01)


def test_the_out_of_scope_budget_is_counted_exactly():
    """In floating point 100 * (1 - 0.9) floors to 9; the rule allows 10."""
    assert C.max_answered(100) == 10
    assert C.max_answered(101) == 10
    assert C.max_answered(9) == 0


def test_the_reject_target_threshold_lets_at_most_the_budget_answer():
    top = [i / 100 for i in range(100)]
    threshold = C.threshold_reject_target(top, [False] * 100)
    assert threshold == 0.9
    assert sum(C.answers(p, False, threshold) for p in top) == 10


def test_rows_sent_to_the_none_option_count_as_rejected():
    top = [i / 100 for i in range(100)]
    forced = [i >= 50 for i in range(100)]  # the confident half already abstains
    threshold = C.threshold_reject_target(top, forced)
    assert threshold == 0.4
    assert sum(C.answers(p, f, threshold) for p, f in zip(top, forced, strict=True)) == 10


def test_when_few_enough_could_answer_every_one_may():
    forced = [True] * 95 + [False] * 5
    # 0.0, not -inf: every probability is at or above it, and it is JSON.
    assert C.threshold_reject_target([0.5] * 100, forced) == 0.0


def test_ties_at_the_top_push_the_threshold_past_them():
    top = [0.99] * 20 + [0.1] * 80
    threshold = C.threshold_reject_target(top, [False] * 100)
    assert threshold > 0.99
    assert not any(C.answers(p, False, threshold) for p in top)


def test_the_error_budget_threshold_is_the_last_prefix_within_it():
    top = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, False]
    # 1/4 = 0.25 > 0.2, 2/5 too; 0/3 is the last prefix within 20 %.
    assert C.threshold_error_budget(top, correct, [False] * 5, 0.2) == 0.7


def test_an_unreachable_error_budget_is_none():
    assert C.threshold_error_budget([0.9, 0.8], [False, False], [False, False]) is None


def test_rows_sent_to_the_none_option_are_left_out_of_the_error_budget():
    top = [0.95, 0.9, 0.8]
    correct = [False, True, True]
    forced = [True, False, False]
    assert C.threshold_error_budget(top, correct, forced, 0.05) == 0.8


def test_a_choice_fit_with_out_of_scope_rows_uses_the_reject_target():
    rows, targets = _rows_at(1.0, 300)
    oos = [[0.0, 0.0, 0.0, 0.0, 0.0, -float(i % 7)] for i in range(40)]
    fit = C.fit("choice", rows + oos, targets + [5] * 40, none_index=5)
    assert fit.rule == C.C2
    assert fit.in_scope_rows == len([t for t in targets if t != 5])
    assert fit.out_of_scope_rows == 40 + targets.count(5)
    assert fit.abstain_below is not None
    assert fit.as_dict()["note"] is None


def test_a_choice_fit_without_them_uses_the_error_budget_and_says_what_it_cannot_do():
    rows, targets = _rows_at(1.0, 300)
    fit = C.fit("choice", rows, targets, none_index=None)
    assert fit.rule == C.C1
    assert fit.as_dict()["note"] == C.C1_NOTE


@pytest.mark.parametrize("kind", ["noul", "score"])
def test_noul_and_score_get_a_temperature_and_no_threshold(kind):
    rows, targets = _rows_at(1.0, 100)
    fit = C.fit(kind, rows, targets, none_index=None)
    assert fit.rule is None and fit.abstain_below is None


def test_the_temperature_is_fitted_on_in_scope_rows_alone():
    rows, targets = _rows_at(1.0, 200)
    noise = [[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 50
    with_noise = C.fit("choice", rows + noise, targets + [5] * 50, none_index=5)
    alone = C.fit(
        "choice",
        [r for r, t in zip(rows, targets, strict=True) if t != 5],
        [t for t in targets if t != 5],
        none_index=None,
    )
    assert with_noise.temperature == alone.temperature


def test_scores_that_are_not_finite_are_refused():
    with pytest.raises(C.CalibrationError, match="not finite"):
        C.fit("choice", [[0.0, math.nan]], [0], None)


def test_rows_of_different_widths_are_refused():
    with pytest.raises(C.CalibrationError, match="row 1 has 3 scores"):
        C.fit("choice", [[0.0, 1.0], [0.0, 1.0, 2.0]], [0, 1], None)


def test_calibration_with_no_in_scope_row_is_refused():
    with pytest.raises(C.CalibrationError, match="no in-scope row"):
        C.fit("choice", [[0.0, 1.0]], [1], none_index=1)


def test_an_error_budget_nothing_meets_is_refused():
    with pytest.raises(C.CalibrationError, match="even the most confident row is wrong"):
        C.fit("choice", [[5.0, 0.0], [5.0, 0.0]], [1, 1], none_index=None)


@pytest.mark.parametrize("n", [1, 10, 59, 300])
def test_with_no_successes_the_upper_bound_is_its_closed_form(n):
    assert math.isclose(C.upper_bound(0, n), 1 - 0.05 ** (1 / n), rel_tol=1e-9)


@pytest.mark.parametrize("n", [1, 10, 59, 300])
def test_with_all_successes_the_lower_bound_is_its_closed_form(n):
    assert math.isclose(C.lower_bound(n, n), 0.05 ** (1 / n), rel_tol=1e-9)


@pytest.mark.parametrize(("k", "n"), [(5, 100), (1, 3), (40, 50), (300, 1000)])
def test_each_bound_is_where_the_tail_reaches_alpha(k, n):
    upper, lower = C.upper_bound(k, n), C.lower_bound(k, n)
    assert lower < k / n < upper
    assert math.isclose(C._binomial_cdf(k, n, upper), 0.05, rel_tol=1e-6)
    assert math.isclose(1 - C._binomial_cdf(k - 1, n, lower), 0.05, rel_tol=1e-6)


def test_fifty_nine_clean_rows_are_the_fewest_that_show_a_five_percent_error():
    assert C.upper_bound(0, 59) <= 0.05 < C.upper_bound(0, 58)


def test_the_budget_is_exact_where_floating_point_is_not():
    """0.55 * 100 is 55.00000000000001 in floating point, whose ceiling is 56."""
    assert C.max_answered(100, 0.55) == 45


def test_the_refinement_finds_the_minimum_between_grid_points():
    rows, targets = _rows_at(1.7, 500, seed=3)
    t = C.fit_temperature(rows, targets)
    dense = [t * math.exp(i / 2000 - 0.05) for i in range(201)]
    best = min(C.mean_nll(rows, targets, x) for x in dense)
    assert C.mean_nll(rows, targets, t) <= best + 1e-9


def test_rows_tied_at_a_value_are_answered_together_in_the_error_budget():
    """A threshold at 0.9 answers both rows at 0.9, one of them wrong: 50 % error."""
    top = [0.95, 0.9, 0.9, 0.5]
    correct = [True, True, False, True]
    assert C.threshold_error_budget(top, correct, [False] * 4, 0.05) == 0.95


def test_a_saturated_softmax_does_not_hide_an_error_at_one():
    top = [1.0] * 30 + [0.99]
    correct = [True] * 29 + [False] + [True]
    threshold = C.threshold_error_budget(top, correct, [False] * 31, 0.01)
    assert threshold is None
