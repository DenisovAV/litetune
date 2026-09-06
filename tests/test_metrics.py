"""Parsing, and the interval that goes with every number.

The sample-size tests carry the measured numbers from the README on purpose: a
refactor that drops the interval, or swaps the paired difference for the
unpaired one, changes what those tests conclude about a real comparison.
"""

import math

import pytest

from litetune.metrics import (
    TERMINATORS,
    Proportion,
    QualityMetrics,
    ToolCall,
    Unavailable,
    _reproduces_target,
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


def test_exact_text_scores_one_right_answer_and_claims_no_decomposition():
    """A task with one right string has nothing to split into name and arguments.

    Both are `Unavailable` rather than zero: a zero would read as a measurement
    of something, and there is nothing there to measure.
    """
    from litetune.metrics import Unavailable, score_exact_text

    metrics = score_exact_text(
        ["red", "blue", "green"],
        ["red\n", "  blue  ", "yellow"],
    )

    # Whitespace is forgiven; nothing else is.
    assert metrics.correct == (True, True, False)
    assert metrics.exact_match.value == pytest.approx(2 / 3)
    assert metrics.parse_rate.value == 1.0
    assert isinstance(metrics.name_accuracy, Unavailable)
    assert isinstance(metrics.argument_accuracy, Unavailable)


def test_a_scorer_refuses_rather_than_returning_a_degenerate_number():
    from litetune.metrics import score_exact_text

    with pytest.raises(ValueError, match="empty split"):
        score_exact_text([], [])
    with pytest.raises(ValueError, match="2 targets against 1 outputs"):
        score_exact_text(["a", "b"], ["a"])


def test_both_shipped_scorers_carry_their_own_name_and_meaning():
    """The manifest records which one ran; two manifests scored differently are
    not comparable, and nothing else in the file would say so."""
    from litetune.metrics import SCORERS

    assert set(SCORERS) == {"tool-call", "exact-text"}
    for key, scorer in SCORERS.items():
        assert scorer.name == key
        assert scorer.describes.startswith("correct means")


def test_the_statistics_below_a_scorer_do_not_know_the_task():
    """The claim the whole design rests on, asserted rather than assumed.

    `paired_difference` consumes the per-example booleans and nothing else, so
    a tool-call run and an exact-text run with the same hit pattern must produce
    the same difference, interval and verdict. If that ever stops being true,
    swapping the scorer stops being safe.
    """
    from litetune.metrics import SCORERS, paired_difference, score_exact_text

    # Same pattern of agreement, reached two different ways.
    text_a = score_exact_text(["x"] * 8, ["x"] * 6 + ["no", "no"])
    text_b = score_exact_text(["x"] * 8, ["x"] * 4 + ["no"] * 4)
    assert text_a.correct == (True,) * 6 + (False, False)
    assert text_b.correct == (True,) * 4 + (False,) * 4

    difference = paired_difference(text_a.correct, text_b.correct)
    assert difference.value == pytest.approx(2 / 8)
    assert difference.discordant == 2
    assert difference.method == "paired (McNemar)"
    # It reports the interval and the verdict without ever asking what the task
    # was -- which is the property that makes swapping the scorer safe.
    assert "does not resolve" in difference.detail
    assert SCORERS["tool-call"].name == "tool-call"


# -- the terminator vocabulary ----------------------------------------------


def test_every_stop_token_a_family_declares_is_a_terminator_here():
    """`models.RULES` establishes a family's stop token from evidence;
    `TERMINATORS` is where scoring learns to ignore it. Nothing linked them, so
    a family added there with a marker missing here silently reproduced the
    0.0000 defect -- which is what FunctionGemma's `<start_function_response>`
    did until this was written down.
    """
    from litetune import models

    declared = {token for rules in models.RULES for token in rules.extra_stop_tokens}
    assert declared, "no family declares a stop token; this test has stopped testing anything"
    assert declared <= set(TERMINATORS), sorted(declared - set(TERMINATORS))


def test_the_vocabulary_lists_every_marker_a_supported_family_uses():
    """Spelled out rather than parametrised over `TERMINATORS`, because a test
    that iterates the thing under test cannot notice a deletion: removing an
    entry removes its own case and the suite stays green. Each string here is a
    family that would silently go back to scoring 0.0000 without it.
    """
    for marker, family in [
        ("<eos>", "Gemma, and the tokenizer default for most others"),
        ("<end_of_turn>", "Gemma chat templates"),
        ("</s>", "Mistral and Llama-2"),
        ("<|im_end|>", "ChatML: Qwen and everything that copied it"),
        ("<|endoftext|>", "GPT-2 lineage"),
        ("<|eot_id|>", "Llama-3"),
        ("<|end_of_text|>", "Llama-3 raw completions"),
        ("<|end|>", "Phi-3 and Phi-4"),
        ("<start_function_response>", "FunctionGemma, per models.RULES"),
    ]:
        assert marker in TERMINATORS, f"{marker} ({family}) is no longer trimmed"
    # `test_a_reference_whose_terminator_is_unknown_is_not_a_conversion_cost`
    # (test_verify.py) depends on this marker staying unrecognised.
    assert "<|assistant_end|>" not in TERMINATORS


@pytest.mark.parametrize("terminator", TERMINATORS)
def test_each_terminator_comes_off_the_end_and_only_the_end(terminator):
    """Every entry in the vocabulary comes off the end, comes off however many
    times it repeats, and is left alone anywhere else -- a marker in the
    middle of a generation is content, not termination.
    """
    from litetune.metrics import trim_terminator

    assert trim_terminator(f"answer{terminator}") == "answer"
    assert trim_terminator(f"answer{terminator}{terminator}") == "answer"
    assert (
        trim_terminator(f"say {terminator} then stop{terminator}") == f"say {terminator} then stop"
    )


def test_strip_terminators_reports_markers_in_text_order():
    """`markers` lists innermost first -- the order the markers appear in the
    text, not the order stripping removes them (which is outermost first).
    """
    from litetune.metrics import _strip_terminators

    assert _strip_terminators("a</s><eos>") == ("a", ("</s>", "<eos>"))
    assert _strip_terminators("label_3<end_of_turn>\n<eos>") == (
        "label_3",
        ("<end_of_turn>", "<eos>"),
    )
    assert _strip_terminators("a") == ("a", ())
    assert _strip_terminators("<eos>") == ("", ("<eos>",))


def test_stacked_terminators_separated_by_whitespace_all_come_off():
    """A chat-template close and the tokenizer's own EOS, separated by a newline.

    Constructed, not observed: measured against the real checkpoints, both
    supported families carry their template close in `eos_token_id`, so
    generation stops there and returns one marker. The stack belongs to a
    family whose eos set excludes its own close, where generation runs past it
    -- and this tool is used on models it has no rules for. A mutation
    dropping the inner `.strip()` between removals survived the whole suite;
    this pins the newline coming off too.
    """
    from litetune.metrics import terminators_trimmed, trim_terminator

    text = "label_3<end_of_turn>\n<eos>"
    assert trim_terminator(text) == "label_3"
    assert terminators_trimmed(text) == 2


def test_a_terminator_followed_by_trailing_whitespace_still_comes_off():
    """The outer twin of `test_stacked_terminators_separated_by_whitespace_all_come_off`:
    that test pins the `.strip()` *between* removals; this one pins the first
    `.strip()`, before any marker has come off at all. Mutating
    `out = text.strip()` to `out = text` passes the whole suite unnoticed,
    because every existing fixture puts its marker at the very end of the
    string with nothing trailing it. `trim_terminator("<eos>\\n")` would then
    return `"<eos>\\n"` instead of `""`, so a model that emits nothing but its
    terminator plus a trailing newline stops being caught by
    `liveness.non_empty_check` (`max_empty_share = 0.0`), and any generation
    ending `<eos>\\n` scores 0 on every row under `exact-text` -- the exact
    0.0000 defect this branch exists to fix, one newline away.
    """
    from litetune.metrics import terminators_trimmed, trim_terminator

    assert trim_terminator("label_3<end_of_turn>\n<eos>\n") == "label_3"
    assert terminators_trimmed("answer<eos>  \n") == 1
    assert _reproduces_target("positive", "positive<eos>\n")


def test_exact_text_holds_a_target_to_its_own_markers_and_forgives_the_decoders():
    """One assertion per branch of the contract: a target's own trailing
    markers are part of the answer and must be reproduced; anything a
    generation carries beyond them is a decoder's convention and is ignored.
    """
    from litetune.metrics import score_exact_text

    def hit(target: str, output: str) -> bool:
        return score_exact_text([target], [output]).correct[0]

    assert hit("positive", "positive<eos>")  # plain target: decoder EOS is forgiven
    assert hit("<b>x</s>", "<b>x</s><eos>")  # target's marker kept, decoder EOS beyond it forgiven
    assert hit("<b>x</s>", "<b>x</s>")  # target's own marker reproduced exactly
    assert not hit("<b>x</s>", "<b>x")  # target's own marker dropped: wrong
    assert not hit("<b>x</s>", "<b>x<eos>")  # decoder EOS is not the target's own marker
    assert not hit("</s>", "")  # empty output cannot reproduce a marker-only target
    assert not hit("</s>", "<eos>")  # a different marker is not the target's marker
    assert hit("a b", "a  b<eos>")  # whitespace collapsed, decoder EOS forgiven
    # The template's close (<end_of_turn>) sits in front of the target's own
    # marker rather than in place of it, so the target's <eos> is still among
    # the generation's markers. Constructed: measured, both supported families
    # stop at their close and never reach the eos.
    assert hit("x<eos>", "x<end_of_turn>\n<eos>")
    assert not hit("x<eos><eos>", "x<eos>")  # multiplicity: target needs two, generation has one
    assert hit("x<eos>", "x<eos><eos>")  # target needs one; a second beyond it is forgiven


def test_agreement_and_exact_text_share_one_whitespace_rule():
    """Label-free `agreement` and labelled `score_exact_text` must collapse
    whitespace the same way, or the two could disagree about what counts as
    the same answer on the very same pair of outputs.
    """
    from litetune.metrics import score_exact_text

    assert agreement(["a b"], ["a  b"]).value == 1.0
    assert score_exact_text(["a b"], ["a  b"]).exact_match.value == 1.0


def test_whitespace_is_collapsed_on_the_targets_side_too():
    """The test above only ever puts the extra whitespace on the output side,
    and every fixture target elsewhere in this file is single-spaced, so
    weakening `same_answer` from `" ".join(target_core.split()) == ...` to
    `target_core == ...` passes the whole suite. A held-out target with an
    internal double space would then score 0 forever, contradicting the
    scorer's own docstring, which claims whitespace is "collapsed... on both
    sides".
    """
    from litetune.metrics import score_exact_text

    assert score_exact_text(["a  b"], ["a b<eos>"]).exact_match.value == 1.0


def test_terminators_trimmed_counts_what_trimming_hides():
    from litetune.metrics import terminators_trimmed

    assert terminators_trimmed("answer") == 0
    assert terminators_trimmed("answer<eos>") == 1
    assert terminators_trimmed("answer" + "<eos>" * 7) == 7


def test_the_order_of_the_markers_a_target_ends_with_is_part_of_the_answer():
    """`<eos></s>` and `</s><eos>` are different endings.

    The generation's markers have to contain the target's *as a subsequence*,
    not as a bag. A `Counter` or a set would call these two equal, which would
    let a model that closed its turn in the wrong order score correct -- while
    a subsequence still forgives what the decoder inserts around the target's
    own markers, which is the reason any of this is forgiven at all.
    """
    assert _reproduces_target("x<eos></s>", "x<eos></s>")
    assert not _reproduces_target("x<eos></s>", "x</s><eos>")
    # The forgiveness the subsequence keeps: the chat template's close sits in
    # front of the target's own marker rather than in place of it.
    assert _reproduces_target("x<eos>", "x<end_of_turn>\n<eos>")
    assert not _reproduces_target("x<eos><eos>", "x<eos>")


def test_the_vocabulary_cannot_contain_an_empty_or_nested_marker():
    """Two invariants `_strip_terminators` depends on and nothing enforces.

    An empty entry would make the stripper loop forever: `out[:-0]` is `""`,
    so `out.endswith("")` is always true and the removal never terminates. A
    marker that is a suffix of another (say `"eos>"` alongside `"<eos>"`) would
    make the reported core -- and therefore `_reproduces_target` -- depend on
    which one `TERMINATORS` happens to try first, since the shorter one would
    strip first and leave the rest of the longer one behind as content.
    Neither is triggered here; both are just true of every entry, always.
    """
    assert all(marker for marker in TERMINATORS), "an empty entry would hang _strip_terminators"
    for marker in TERMINATORS:
        others = [other for other in TERMINATORS if other != marker]
        assert not any(
            marker.endswith(other) for other in others
        ), f"{marker!r} is a suffix of another entry: order-dependent stripping"
