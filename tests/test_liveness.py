"""The label-free tier, and the order it runs in.

The ordering tests are the ones that matter. Three identical *crashes* compare
equal exactly as well as three identical generations, and that false pass
happened during the measurement work -- it was read as "decoding is
deterministic". So there is a test asserting that no comparison is computed
before the generations are known to have succeeded.
"""

import dataclasses

import pytest

from litetune.checks import Outcome
from litetune.evaluate import GREEDY, Generation, MeasurementPoint, PromptMode
from litetune.liveness import (
    DEFAULT_THRESHOLDS,
    LivenessThresholds,
    divergence_check,
    divergence_share,
    ends_with_terminator,
    leaked_tokens,
    liveness_tier,
    repetition_ratio,
    unterminated_count,
)
from litetune.metrics import TERMINATORS, trim_terminator


def make_point(texts, returncode: int = 0, harness_error: str | None = None) -> MeasurementPoint:
    generations = tuple(
        Generation(
            index=i,
            prompt=f"p{i}",
            text=text,
            returncode=None if harness_error else returncode,
            harness_error=harness_error,
        )
        for i, text in enumerate(texts)
    )
    return MeasurementPoint(
        label="candidate",
        model_ref="m",
        backend="fake",
        prompt_mode=PromptMode.PRERENDERED,
        decode=GREEDY,
        split_id="split",
        engine={},
        generations=generations,
        # Stated, not defaulted: the record refuses to guess whether the
        # decoding parameters it carries actually governed the run.
        decode_enforced=True,
    )


ALIVE = ["call:open{app:<escape>maps<escape>}", "call:close{app:<escape>mail<escape>}"]


def test_a_live_model_passes_the_four_label_free_checks():
    result = liveness_tier(make_point(ALIVE))
    assert result.outcome is Outcome.PASSED
    assert [c.name for c in result.checks.checks] == [
        "exit status",
        "non-empty output",
        "no special-token leakage",
        "no degenerate repetition",
    ]


def test_identical_crashes_are_never_compared():
    # Three runs that all died produce identical output. Reporting that as
    # agreement is the false pass this ordering exists to prevent.
    result = liveness_tier(make_point(["", "", ""], returncode=1), baseline=["", "", ""])
    assert result.outcome is Outcome.FAILED
    assert [c.name for c in result.checks.checks] == ["exit status"]
    assert [s.name for s in result.skipped] == [
        "non-empty output",
        "no special-token leakage",
        "no degenerate repetition",
        "divergence from baseline",
    ]
    assert all("not reached" in s.reason for s in result.skipped)


def test_a_generation_that_never_ran_is_not_a_model_failure():
    result = liveness_tier(make_point(["", ""], harness_error="libvulkan.so.1 missing"))
    assert result.outcome is Outcome.UNCHECKED
    assert result.checks.checks[0].outcome is Outcome.UNCHECKED
    assert "libvulkan" in result.checks.checks[0].detail


def test_empty_output_fails_before_anything_is_compared():
    result = liveness_tier(make_point(["call:a{}", ""]))
    assert result.outcome is Outcome.FAILED
    assert result.checks.first_failure.name == "non-empty output"


def test_the_wire_format_escape_marker_is_not_leakage():
    # <escape> delimits every argument value; flagging it would fail every
    # correct output this tool is built to score.
    assert leaked_tokens("call:a{x:<escape>1<escape>}") == []


def test_a_trailing_end_of_turn_is_termination_not_leakage():
    assert leaked_tokens("call:a{}<end_of_turn>") == []
    assert trim_terminator("call:a{}<eos><eos>") == "call:a{}"


def test_padding_token_leakage_is_caught():
    assert leaked_tokens("call:a{}<pad><pad>") == ["<pad>", "<pad>"]
    result = liveness_tier(make_point(["call:a{}<pad>", "call:b{}"]))
    assert result.outcome is Outcome.FAILED
    assert result.checks.first_failure.name == "no special-token leakage"


def test_a_short_correct_call_is_not_degenerate_repetition():
    # A correct single call is about ten tokens; scoring repetition on it would
    # fail every good output.
    assert repetition_ratio("call:open{app:<escape>maps<escape>}") == 0.0


def test_a_looping_decode_is_degenerate():
    looping = "open the app open the app open the app open the app open the app"
    assert repetition_ratio(looping) > 0.5
    result = liveness_tier(make_point([looping, looping]))
    assert result.outcome is Outcome.FAILED
    assert result.checks.first_failure.name == "no degenerate repetition"


def test_one_flake_among_many_does_not_condemn_the_run():
    texts = ["call:a{}"] * 99 + ["open the app " * 12]
    assert liveness_tier(make_point(texts)).outcome is Outcome.PASSED


def test_the_shipped_failure_ratio_still_trips_the_degenerate_share_gate():
    """This fixture's own construction, not a figure this repo publishes
    anywhere: 571 looping generations against 29 correct ones, chosen so the
    degenerate share (571/600 = 0.9517) sits far above the 0.05 default with
    room in both directions -- `max_degenerate_share` survives being set to
    0.99 in a way nothing else in this file proves, since every other
    fixture here sits at an extreme (about 1% or effectively 100%).

    Asserted on `observed`, not on `.2f`-formatted text: a `"0.50" in detail`
    assertion pins `repetition_ratio` (also 0.5 by default), not
    `max_degenerate_share`, the constant this test is named for -- and it
    would not notice a 0.05 -> 0.50 mutation of that constant, since 0.9517
    clears either threshold. `test_default_liveness_thresholds_are_pinned`
    below pins the value itself.
    """
    looping = "open the app open the app open the app open the app open the app"
    correct = "call:open{app:<escape>maps<escape>}"
    texts = [looping] * 571 + [correct] * 29
    result = liveness_tier(make_point(texts))
    assert result.outcome is Outcome.FAILED
    check = result.checks.first_failure
    assert check.name == "no degenerate repetition"
    assert check.observed["share"] == pytest.approx(571 / 600)
    assert check.observed["degenerate"] == 571
    assert check.observed["n"] == 600


def test_degenerate_share_exactly_at_the_default_threshold_still_passes():
    """`>` not `>=`: a share landing exactly on `max_degenerate_share`'s
    default must not fail the check.
    """
    looping = "open the app open the app open the app open the app open the app"
    correct = "call:open{app:<escape>maps<escape>}"
    texts = [looping] * 5 + [correct] * 95
    result = liveness_tier(make_point(texts))
    assert result.outcome is Outcome.PASSED


def test_repetition_ratio_exactly_at_the_default_threshold_still_passes():
    """`>` not `>=`: a per-generation ratio landing exactly on
    `repetition_ratio`'s default must not fail `no degenerate repetition`.
    Distinct from `max_degenerate_share`'s boundary above, which is a
    share-of-generations bar computed *over* this per-generation ratio, not
    the ratio itself -- a fixture sitting at an extreme for one says nothing
    about the other.
    """
    text = "a a a a a a a a a b a a a"
    assert repetition_ratio(text) == 0.5
    result = liveness_tier(make_point([text]))
    assert result.outcome is Outcome.PASSED


def test_default_liveness_thresholds_are_pinned():
    """Each of these is free to drift silently unless the value itself is
    pinned, the way `verify.NEAR_ZERO` is pinned in `test_verify.py`: a
    fixture that only ever sits at an extreme -- 1% or 100%, 0.09 or 0.995 --
    cannot tell 0.05 from 0.50, or 4 from 2.

    `max_empty_share` and `max_leak_share` are pinned at `0.0` here, not as
    shares with room to slide, but as switches: zero tolerance means a single
    empty or leaking generation is enough to fail its check, and the
    behavioural tests below prove that at the smallest share above zero a
    fixture's construction can produce.
    """
    assert DEFAULT_THRESHOLDS.max_empty_share == 0.0
    assert DEFAULT_THRESHOLDS.max_leak_share == 0.0
    assert DEFAULT_THRESHOLDS.max_degenerate_share == 0.05
    assert DEFAULT_THRESHOLDS.repetition_ratio == 0.5
    assert DEFAULT_THRESHOLDS.repetition_min_tokens == 12
    assert DEFAULT_THRESHOLDS.repetition_ngram == 4
    assert DEFAULT_THRESHOLDS.min_divergence_share == 0.10
    assert DEFAULT_THRESHOLDS.max_unterminated_share == 0.10


def test_one_empty_generation_among_many_fails_at_the_zero_tolerance_default():
    """`max_empty_share` is `0.0`: any share above it -- not just a run that
    is entirely empty -- must fail. One empty row out of many is the smallest
    non-zero share a fixture here can produce, and it is also the point that
    a share free to drift to 0.40 would still pass in silence.
    """
    texts = ["call:a{}"] * 99 + [""]
    result = liveness_tier(make_point(texts))
    assert result.outcome is Outcome.FAILED
    assert result.checks.first_failure.name == "non-empty output"


def test_one_leaking_generation_among_many_fails_at_the_zero_tolerance_default():
    """`max_leak_share` is `0.0`: the same one-row argument as the empty-share
    test above, for the other zero-tolerance switch.
    """
    texts = ["call:a{}"] * 99 + ["call:b{}<pad>"]
    result = liveness_tier(make_point(texts))
    assert result.outcome is Outcome.FAILED
    assert result.checks.first_failure.name == "no special-token leakage"


def test_liveness_thresholds_as_dict_carries_every_field():
    """`harness.liveness_thresholds` is what a reader checks to see which
    threshold decided a verdict; a field silently dropped from `as_dict`
    would leave a manifest missing the number that caused a refusal, and a
    field carried at the wrong value would leave a manifest lying about which
    number decided it. Pinned against the dataclass's own field set *and* its
    own field values, the way `test_cli.py` pins `EXIT_CODES` whole rather
    than by membership, so neither a field added, renamed or dropped, nor a
    default changed without `as_dict` following it, can pass unnoticed.
    Compared against `DEFAULT_THRESHOLDS`'s own attributes rather than a
    hand-written literal, so this test cannot itself go stale the way a
    literal `{"max_empty_share": 0.0, ...}` would the day a default changes.
    """
    fields = {f.name for f in dataclasses.fields(LivenessThresholds)}
    as_dict = DEFAULT_THRESHOLDS.as_dict()
    assert set(as_dict) == fields
    assert as_dict == {name: getattr(DEFAULT_THRESHOLDS, name) for name in fields}


def test_a_model_identical_to_its_baseline_fails_divergence():
    point = make_point(ALIVE)
    check = divergence_check(point, list(ALIVE), "the untuned base", LivenessThresholds())
    assert check.outcome is Outcome.FAILED
    assert "may be the baseline" in check.detail


def test_a_model_that_says_something_else_diverges():
    point = make_point(ALIVE)
    check = divergence_check(point, ["call:zzz{}", "call:yyy{}"], "base", LivenessThresholds())
    assert check.outcome is Outcome.PASSED


def test_divergence_share_exactly_at_the_default_threshold_still_passes():
    """`<` not `<=`: a divergence share landing exactly on
    `min_divergence_share`'s default must not fail the check.
    """
    baseline = [f"answer {i}" for i in range(100)]
    candidate = list(baseline)
    for i in range(10):
        candidate[i] = f"different {i}"
    check = divergence_check(
        make_point(candidate), baseline, "the untuned base", LivenessThresholds()
    )
    assert check.observed["divergence_share"] == pytest.approx(0.10)
    assert check.outcome is Outcome.PASSED


def test_the_stripped_terminator_mechanism_the_old_comment_rested_on():
    """The mechanism `min_divergence_share`'s comment used to lean on: two
    answers differing only in which *recognised* terminator they close with
    compare equal, because `comparable_form` strips every member of
    `TERMINATORS` before comparing. That much is true; see the next test for
    what it does not cover.
    """
    assert divergence_share(["x<eos>"] * 100, ["x<end_of_turn>"] * 100) == 0.0


def test_an_unrecognised_marker_can_supply_the_divergence_this_check_requires():
    """The comment's false half, made concrete. `comparable_form` strips only
    `TERMINATORS` members, and a residue of unrecognised markers is by
    definition not among them: a candidate byte-identical to the baseline
    except for one on 10% of rows clears this check's own default threshold.
    The candidate could be the baseline.
    """
    baseline = [f"label_{i}" for i in range(100)]
    candidate = list(baseline)
    for i in range(10):
        candidate[i] = baseline[i] + "<|assistant_end|>"
    check = divergence_check(
        make_point(candidate), baseline, "the untuned base", LivenessThresholds()
    )
    assert check.outcome is Outcome.PASSED
    assert check.observed["divergence_share"] == pytest.approx(0.10)


def test_divergence_over_mismatched_prompts_cannot_be_checked():
    # An exception inside a check body means the check did not run, not that
    # the model failed it.
    result = liveness_tier(make_point(ALIVE), baseline=["only one output"])
    assert result.outcome is Outcome.UNCHECKED
    assert result.checks.first_unchecked.name == "divergence from baseline"


def test_a_skipped_divergence_check_is_recorded_not_assumed():
    result = liveness_tier(make_point(ALIVE), baseline_absent_reason="no baseline was supplied")
    assert result.outcome is Outcome.PASSED
    assert [s.name for s in result.skipped] == ["divergence from baseline"]
    assert result.as_dict()["skipped"][0]["reason"] == "no baseline was supplied"


def test_no_generations_means_nothing_was_established():
    assert liveness_tier(make_point([])).outcome is Outcome.UNCHECKED


@pytest.mark.parametrize("marker", TERMINATORS)
def test_every_marker_in_the_shared_vocabulary_counts_as_termination(marker):
    """This branch widened `ends_with_terminator` from a hand-rolled five-marker
    tuple to `metrics.TERMINATORS`'s nine, so that liveness and scoring share
    one vocabulary. Mutating `endswith(TERMINATORS)` to `endswith(TERMINATORS[:1])`
    still passes if nothing here exercises a generation ending in any of the
    other eight -- `<|eot_id|>`, `<|end|>`, `<|im_end|>` and
    `<start_function_response>` among them, the last being FunctionGemma's own
    declared stop token, the family this tool is measured on.
    """
    assert ends_with_terminator(f"answer{marker}")


def test_unterminated_count_recognises_the_markers_the_vocabulary_gained():
    """A point whose generations end in the newer markers -- not just `<eos>`,
    which every other fixture in this file uses -- must still count as fully
    terminated. `<start_function_response>` is FunctionGemma's declared stop
    token per `models.RULES`.
    """
    point = make_point(
        [
            "call:a{}<|eot_id|>",
            "call:b{}<|end|>",
            "call:c{}<|im_end|>",
            "call:d{}<start_function_response>",
        ]
    )
    assert unterminated_count(point) == 0
