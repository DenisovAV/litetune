"""Parsing, and the interval that goes with every number.

The sample-size tests carry the measured numbers from the README on purpose: a
refactor that drops the interval, or swaps the paired difference for the
unpaired one, changes what those tests conclude about a real comparison.
"""

import math

import pytest

from litetune.metrics import (
    Proportion,
    QualityMetrics,
    ToolCall,
    Unavailable,
    agreement,
    difference,
    paired_difference,
    parse_call,
    score,
)

# -- the wire format --------------------------------------------------------


def test_parses_a_single_call():
    call = parse_call("call:change_background_color{color:<escape>red<escape>}")
    assert call == ToolCall("change_background_color", {"color": "red"})


def test_parses_multiple_arguments():
    call = parse_call("call:set{a:<escape>1<escape>,b:<escape>two<escape>}")
    assert call is not None
    assert call.args == {"a": "1", "b": "two"}


def test_parses_a_call_with_no_arguments():
    assert parse_call("call:refresh{}") == ToolCall("refresh", {})


def test_value_may_contain_commas_and_braces():
    # Values are delimited by <escape>, not quoted, so splitting the body on
    # punctuation would truncate this one.
    call = parse_call("call:say{text:<escape>hello, {world}<escape>}")
    assert call is not None
    assert call.args["text"] == "hello, {world}"


def test_call_surrounded_by_other_text_still_parses():
    call = parse_call("sure!\ncall:open{app:<escape>maps<escape>}\n<end_of_turn>")
    assert call is not None
    assert call.name == "open"


def test_a_second_call_does_not_leak_into_the_first():
    """A model that emits two calls must score as its first one, not a merge.

    An earlier parser used a greedy `call:(\\w+)\\{(.*)\\}` over the whole
    text, which returned the first call's NAME with BOTH calls' arguments. On a
    dataset where 33% of training rows carry two calls and every scored row
    carries one, a model that learned to append a second call scored zero on an
    otherwise perfect answer -- 640 of 640 such cases -- and that read as a
    17-point fine-tuning regression for most of a day.

    This parser walks arguments from the head and stops at the first close, so
    it is correct by construction. The test exists so it stays that way: the
    invariant is invisible on single-call data, which is all the reference
    fixtures contain.
    """
    one = "call:set_alarm{hour:<escape>7<escape>}"
    two = one + "call:send_message{to:<escape>Ann<escape>,text:<escape>hi<escape>}"
    assert parse_call(two) == ToolCall(name="set_alarm", args={"hour": "7"})


def test_trailing_prose_after_a_call_is_ignored():
    """Chat-tuned models like to explain themselves after the call."""
    assert parse_call("call:mute{}\n\nI've muted the device for you.") == ToolCall(
        name="mute", args={}
    )


def test_text_without_a_call_is_not_a_call():
    assert parse_call("I would open the maps app.") is None


def test_unterminated_body_is_not_half_a_call():
    assert parse_call("call:open{app:<escape>maps<escape>") is None


def test_targets_are_stringified_because_the_format_is_untyped():
    target = ToolCall.from_target({"name": "wait", "args": {"seconds": 3, "loud": True, "x": None}})
    assert target is not None
    assert target.args == {"seconds": "3", "loud": "true", "x": "null"}


def test_an_integer_target_matches_the_string_the_model_emits():
    target = ToolCall.from_target({"name": "wait", "args": {"seconds": 3}})
    assert parse_call("call:wait{seconds:<escape>3<escape>}") == target


def test_unlabelled_target_is_none_not_an_error():
    assert ToolCall.from_target(None) is None


def test_malformed_target_is_rejected_loudly():
    with pytest.raises(ValueError):
        ToolCall.from_target({"args": {"a": 1}})


# -- estimates --------------------------------------------------------------


def test_proportion_carries_n_and_an_interval():
    p = Proportion.of(87, 100)
    assert p.value == pytest.approx(0.87)
    assert p.n == 100
    assert p.ci95 == pytest.approx(1.959963985 * math.sqrt(0.87 * 0.13 / 100))


def test_an_empty_sample_has_no_estimate():
    # Returning 0.0 here would print as a score for a measurement never made.
    with pytest.raises(ValueError):
        Proportion.of(0, 0)


def test_the_interval_shrinks_with_the_sample():
    assert Proportion.of(64, 100).ci95 > Proportion.of(640, 1000).ci95


def test_a_difference_inside_the_interval_is_not_resolved():
    # n=64 is where three conclusions were drawn and later overturned.
    small_a = Proportion.of(56, 64)  # 0.875
    small_b = Proportion.of(54, 64)  # 0.844
    d = difference(small_a, small_b)
    assert not d.resolved
    assert "does not resolve" in d.detail


def test_paired_difference_resolves_what_the_unpaired_one_cannot():
    # The README's measured pair: 0.9094 float against 0.8906 converted, n=640.
    # The two models agree on most examples, so the unpaired interval is far too
    # blunt for the effect and the paired one is not.
    n = 640
    reference = [i < 557 for i in range(n)]
    candidate = [i < 538 for i in range(n)]  # every candidate error is a subset

    unpaired = difference(Proportion.of(557, n), Proportion.of(538, n))
    paired = paired_difference(reference, candidate)

    assert unpaired.value == pytest.approx(paired.value, abs=1e-9)
    assert not unpaired.resolved
    assert paired.resolved
    assert paired.discordant == 19
    assert paired.ci95 < unpaired.ci95


def test_paired_difference_of_identical_outcomes_is_exactly_zero():
    d = paired_difference([True, False, True], [True, False, True])
    assert d.value == 0.0
    assert d.discordant == 0
    assert not d.resolved
    assert "agreed on all 3" in d.detail


# -- task metrics -----------------------------------------------------------


def _targets(names, args):
    return [ToolCall(n, a) for n, a in zip(names, args, strict=False)]


def test_operation_and_argument_accuracy_are_reported_separately():
    # The measured shape of conversion loss: the operation survives, the
    # arguments do not. A single exact-match number would hide which moved.
    targets = _targets(["open", "open"], [{"app": "maps"}, {"app": "mail"}])
    outputs = [
        "call:open{app:<escape>maps<escape>}",
        "call:open{app:<escape>music<escape>}",
    ]
    m = score(targets, outputs)
    assert m.name_accuracy.value == 1.0
    assert isinstance(m.argument_accuracy, Proportion)
    assert m.argument_accuracy.value == 0.5
    assert m.exact_match.value == 0.5


def test_exact_match_factorises_into_name_times_argument_accuracy():
    targets = _targets(["a", "a", "b", "b"], [{"x": "1"}, {"x": "2"}, {"x": "3"}, {"x": "4"}])
    outputs = [
        "call:a{x:<escape>1<escape>}",
        "call:a{x:<escape>9<escape>}",
        "call:b{x:<escape>3<escape>}",
        "call:zzz{x:<escape>4<escape>}",
    ]
    m = score(targets, outputs)
    assert isinstance(m.argument_accuracy, Proportion)
    assert m.exact_match.value == pytest.approx(m.name_accuracy.value * m.argument_accuracy.value)


def test_argument_accuracy_has_no_denominator_when_no_name_is_right():
    m = score(_targets(["a"], [{"x": "1"}]), ["call:b{x:<escape>1<escape>}"])
    assert isinstance(m.argument_accuracy, Unavailable)
    assert "no denominator" in m.argument_accuracy.reason


def test_unparseable_output_counts_against_the_parse_rate():
    m = score(_targets(["a", "a"], [{}, {}]), ["call:a{}", "I'd rather not."])
    assert m.parse_rate.value == 0.5
    assert m.exact_match.value == 0.5


def test_score_refuses_mismatched_inputs():
    with pytest.raises(ValueError):
        score(_targets(["a"], [{}]), ["call:a{}", "call:a{}"])


def test_quality_metrics_serialise_with_their_denominator_named():
    m: QualityMetrics = score(_targets(["a"], [{}]), ["call:a{}"])
    d = m.as_dict()
    assert d["exact_match"]["n"] == 1
    assert "operation name was correct" in d["argument_accuracy_denominator"]


def test_agreement_ignores_formatting_differences():
    a = ["call:x{p:<escape>1<escape>}", "call:y{}"]
    b = ["call:x{ p : <escape>1<escape> }", "call:z{}"]
    assert agreement(a, b).value == 0.5
