"""The label-free tier, and the order it runs in.

The ordering tests are the ones that matter. Three identical *crashes* compare
equal exactly as well as three identical generations, and that false pass
happened during the measurement work -- it was read as "decoding is
deterministic". So there is a test asserting that no comparison is computed
before the generations are known to have succeeded.
"""

import pytest

from litetune.checks import Outcome
from litetune.evaluate import GREEDY, Generation, MeasurementPoint, PromptMode
from litetune.liveness import (
    LivenessThresholds,
    divergence_check,
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


def test_a_model_identical_to_its_baseline_fails_divergence():
    point = make_point(ALIVE)
    check = divergence_check(point, list(ALIVE), "the untuned base", LivenessThresholds())
    assert check.outcome is Outcome.FAILED
    assert "may be the baseline" in check.detail


def test_a_model_that_says_something_else_diverges():
    point = make_point(ALIVE)
    check = divergence_check(point, ["call:zzz{}", "call:yyy{}"], "base", LivenessThresholds())
    assert check.outcome is Outcome.PASSED


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
