"""End-to-end `verify`, with faked backends.

Each test below is a failure that happened, or a claim the tool must refuse to
make. The three load-bearing ones: a dead model never reaches the quality tier,
a model that merely runs is never reported as verified, and a comparison whose
two sides were not measured the same way is refused rather than annotated.
"""

from pathlib import Path

import pytest
from conftest import FakeBackend, call_text, correct_texts, labelled_rows

from litetune.evaluate import Generation, PromptMode
from litetune.verify import (
    BackendPair,
    ReferenceRole,
    Status,
    VerifyRequest,
    run_verify,
)


def text_rows(n: int) -> list[dict]:
    """`n` examples whose target is a plain string, for the `exact-text` scorer.

    A string target, not `labelled_rows`' call dict, because the tests below
    append a terminator to it -- which is what they are about.
    """
    return [{"prompt": f"classify {i}", "target": f"label_{i % 5}"} for i in range(n)]


def markup_rows(n: int) -> list[dict]:
    """`n` examples whose target ends in `</s>` as content, not as termination.

    A strikethrough close tag is also a terminator, which is what makes these
    the rows where trimming the target rather than holding the generation to it
    would be wrong.
    """
    return [{"prompt": f"restore {i}", "target": f"<b>bold {i}</s>"} for i in range(n)]


def wrong_texts(rows, n_wrong: int) -> list[str]:
    """Correct calls, except the first `n_wrong`, which get the wrong argument."""
    texts = correct_texts(rows)
    for i in range(n_wrong):
        texts[i] = call_text(rows[i]["target"]["name"], color="wrong")
    return texts


def verify(
    write_split,
    rows,
    candidate: FakeBackend,
    reference: FakeBackend,
    **request_kwargs,
):
    request = VerifyRequest(
        model=write_split(rows).parent / "model.litertlm",
        reference="org/reference",
        data=write_split(rows),
        **request_kwargs,
    )
    return run_verify(request, backends=BackendPair(candidate=candidate, reference=reference))


# -- the ordinary case ------------------------------------------------------


def test_conversion_cost_is_measured_and_attributed(write_split):
    rows = labelled_rows(400)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=wrong_texts(rows, 20)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert result.status is Status.PASSED
    quality = result.manifest["quality"]
    assert quality["available"]
    assert quality["candidate"]["exact_match"]["value"] == pytest.approx(0.95)
    assert quality["reference"]["exact_match"]["value"] == pytest.approx(1.0)

    cost = result.manifest["attribution"]["conversion_cost"]
    assert cost["available"]
    assert cost["value"] == pytest.approx(0.05)
    assert cost["resolved"]
    assert cost["discordant"] == 20


def test_training_gain_is_unavailable_rather_than_inferred(write_split):
    # Standalone verify has two measurement points. Deriving the third from
    # them is the attribution mistake this reports as unavailable instead.
    rows = labelled_rows(20)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    gain = result.manifest["attribution"]["training_gain"]
    assert gain["available"] is False
    assert "untuned baseline" in gain["reason"]


def test_operation_and_argument_accuracy_reach_the_manifest(write_split):
    rows = labelled_rows(20)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=wrong_texts(rows, 4)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    candidate = result.manifest["quality"]["candidate"]
    assert candidate["name_accuracy"]["value"] == pytest.approx(1.0)
    assert candidate["argument_accuracy"]["value"] == pytest.approx(0.8)


# -- liveness gates quality -------------------------------------------------


def test_a_dead_model_never_reaches_the_quality_tier(write_split):
    rows = labelled_rows(8)
    reference = FakeBackend(model="org/reference", texts=correct_texts(rows))
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[""], returncode=1),
        reference=reference,
    )
    assert result.status is Status.FAILED_SMOKE
    assert result.manifest["quality"]["available"] is False
    # Not merely unscored: the reference was never even run, because a
    # comparison against output that failed to be produced is meaningless.
    assert reference.prompts_seen == []
    assert "reference" not in result.manifest["measurements"]


def test_a_model_that_could_not_run_is_not_reported_as_failed(write_split):
    rows = labelled_rows(8)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[""], harness_error="libvulkan.so.1 missing"),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert result.status is Status.FAILED_HARNESS
    assert result.status is not Status.FAILED_SMOKE
    assert result.manifest["quality"]["available"] is False


def test_a_reference_that_did_not_generate_invalidates_the_comparison(write_split):
    rows = labelled_rows(8)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=[""], harness_error="env missing"),
    )
    assert result.status is Status.FAILED_HARNESS
    # The candidate itself was alive; the failure belongs to the harness.
    assert result.manifest["liveness"]["candidate"]["outcome"] == "passed"
    assert result.manifest["quality"]["available"] is False


def test_backend_that_raises_is_could_not_check_not_a_crash(write_split):
    class ExplodingBackend(FakeBackend):
        def generate(self, prompts, events=None):
            raise RuntimeError("the runtime went away")

    rows = labelled_rows(4)
    result = verify(
        write_split,
        rows,
        candidate=ExplodingBackend(),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert result.status is Status.FAILED_HARNESS
    assert any(c["outcome"] == "could_not_check" for c in result.manifest["checks"])


# -- liveness is never a quality claim -------------------------------------


def test_a_live_but_wrong_model_is_measured_not_waved_through(write_split):
    # The LoRA run that scored 0.0625 against a 0.5625 base passed every
    # label-free check. Only held-out measurement caught it.
    rows = labelled_rows(64)
    candidate_texts = [call_text("change_background_color", color="always-this") for _ in rows]
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
        max_conversion_cost=0.05,
    )
    assert result.manifest["liveness"]["candidate"]["outcome"] == "passed"
    assert result.status is Status.FAILED_GATE
    assert result.manifest["quality"]["candidate"]["exact_match"]["value"] < 0.05


def test_without_labels_quality_is_unmeasured_never_verified(write_split):
    rows = [{"prompt": f"do thing {i}"} for i in range(8)]
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[call_text("a", x="1")]),
        reference=FakeBackend(model="org/reference", texts=[call_text("a", x="1")]),
    )
    assert result.status is Status.UNMEASURED
    assert result.exit_code != 0
    assert result.manifest["quality"]["available"] is False
    assert "not the same as verified" in result.manifest["quality"]["reason"]
    assert result.manifest["attribution"]["conversion_cost"]["available"] is False


# -- the comparison must be like for like ----------------------------------


def test_a_comparison_across_prompt_modes_is_refused(write_split):
    rows = labelled_rows(8)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(
            model="org/reference",
            texts=correct_texts(rows),
            prompt_mode=PromptMode.RUNTIME_RENDERED,
        ),
    )
    assert result.status is Status.FAILED_HARNESS
    assert "mode" in result.manifest["harness"]["mismatch"]
    assert result.manifest["attribution"]["conversion_cost"]["available"] is False
    assert result.manifest["quality"]["available"] is False


# -- reference roles --------------------------------------------------------


def test_against_an_untuned_base_neither_quantity_is_attributed(write_split):
    rows = labelled_rows(16)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/base", texts=[call_text("something_else", x="1")]),
        reference_role=ReferenceRole.UNTUNED_BASE,
    )
    # Not PASSED. Quality was measured on both sides, but neither difference
    # could be attributed, so there is nothing a threshold would have judged --
    # and exit 0 for that made the *presence* of a threshold decide whether the
    # run counted as established: the same run exited 0 without one and
    # FAILED_HARNESS with one.
    assert result.status is Status.UNMEASURED
    for name in ("conversion_cost", "training_gain"):
        assert result.manifest["attribution"][name]["available"] is False
        assert "confounds" in result.manifest["attribution"][name]["reason"]


def test_a_threshold_does_not_change_whether_anything_was_established(write_split):
    """The same run, with and without a gate, must agree on that question.

    It did not: without `--max-conversion-cost` it exited 0, with one it exited
    4. A gate judges a measured quantity; it cannot decide whether one exists.
    """
    rows = labelled_rows(16)
    backends = dict(
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/base", texts=[call_text("something_else", x="1")]),
        reference_role=ReferenceRole.UNTUNED_BASE,
    )

    without = verify(write_split, rows, **backends)
    with_gate = verify(write_split, rows, **backends, max_conversion_cost=0.5)

    assert without.status is Status.UNMEASURED
    assert with_gate.status is Status.FAILED_HARNESS
    # Different statuses, but neither claims the model passed anything.
    assert Status.PASSED not in {without.status, with_gate.status}


def test_a_candidate_indistinguishable_from_its_base_fails_divergence(write_split):
    rows = labelled_rows(16)
    same = correct_texts(rows)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=same),
        reference=FakeBackend(model="org/base", texts=same),
        reference_role=ReferenceRole.UNTUNED_BASE,
    )
    assert result.status is Status.FAILED_SMOKE
    names = [c["name"] for c in result.manifest["liveness"]["candidate"]["checks"]]
    assert names[-1] == "divergence from baseline"


def test_against_the_float_twin_divergence_is_skipped_with_a_reason(write_split):
    # Agreement with the float twin is the desired outcome of a lossless
    # conversion, so requiring divergence from it would fail a perfect result.
    rows = labelled_rows(16)
    same = correct_texts(rows)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=same),
        reference=FakeBackend(model="org/reference", texts=same),
    )
    assert result.status is Status.PASSED
    skipped = result.manifest["liveness"]["candidate"]["skipped"]
    assert skipped[0]["name"] == "divergence from baseline"
    assert "float twin" in skipped[0]["reason"]


# -- gates ------------------------------------------------------------------


def test_a_threshold_finer_than_the_interval_is_inconclusive(write_split):
    # The n=64 mistake, mechanised: the instrument is blunter than the question,
    # so the answer is "cannot tell", not "fails".
    rows = labelled_rows(64)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=wrong_texts(rows, 2)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
        max_conversion_cost=0.01,
    )
    assert result.status is Status.INCONCLUSIVE
    assert result.exit_code == 2
    assert result.manifest["gates"][0]["outcome"] == "could_not_check"


def test_a_cost_within_the_threshold_passes(write_split):
    rows = labelled_rows(400)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
        max_conversion_cost=0.03,
    )
    assert result.status is Status.PASSED
    assert result.exit_code == 0


def test_without_a_threshold_passed_means_measured_not_good(write_split):
    rows = labelled_rows(16)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert result.status is Status.PASSED
    assert result.manifest["gates"][0]["outcome"] == "could_not_check"
    assert "not that it met a bar" in result.manifest["gates"][0]["detail"]


# -- the manifest -----------------------------------------------------------


def test_manifest_records_what_produced_the_numbers(write_split, tmp_path):
    rows = labelled_rows(16)
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"not really a model")
    request = VerifyRequest(model=model, reference="org/reference", data=write_split(rows))
    result = run_verify(
        request,
        backends=BackendPair(
            candidate=FakeBackend(texts=correct_texts(rows)),
            reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
        ),
    )
    manifest = result.manifest
    assert manifest["schema"] == "litetune.verify/1"
    assert manifest["model"]["sha256"]
    assert manifest["model"]["bytes"] == len(b"not really a model")
    assert manifest["data"]["id"]
    assert manifest["harness"]["prompt_mode"] == "prerendered"
    assert manifest["harness"]["liveness_thresholds"]["repetition_ratio"]
    assert manifest["measurements"]["candidate"]["engine"]["engine"] == "fake"


def test_a_small_split_says_so(write_split):
    rows = labelled_rows(16)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert any("below 200" in note for note in result.manifest["limitations"])


def test_a_near_zero_reference_is_flagged_as_unremarkable(write_split):
    rows = labelled_rows(16)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=[call_text("nothing_like_it", x="1")]),
    )
    assert any("at or near zero" in note for note in result.manifest["limitations"])


def test_every_run_reports_which_backend_measured_it(write_split):
    rows = labelled_rows(4)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert any("optimistic estimate" in note for note in result.manifest["limitations"])


def test_missing_held_out_file_is_an_error_not_a_verdict(tmp_path):
    request = VerifyRequest(
        model=tmp_path / "model.litertlm",
        reference="org/reference",
        data=tmp_path / "absent.jsonl",
    )
    result = run_verify(
        request,
        backends=BackendPair(candidate=FakeBackend(), reference=FakeBackend()),
    )
    assert result.status is Status.ERROR
    assert result.manifest["checks"][0]["outcome"] == "could_not_check"


def test_limit_reaches_the_backends(write_split):
    rows = labelled_rows(20)
    candidate = FakeBackend(texts=[call_text("change_background_color", color="c0")])
    reference = FakeBackend(model="org/reference", texts=[call_text("x", y="1")])
    data = write_split(rows)
    result = run_verify(
        VerifyRequest(model=Path("m.litertlm"), reference="r", data=data, limit=5),
        backends=BackendPair(candidate=candidate, reference=reference),
    )
    assert len(candidate.prompts_seen[0]) == 5
    assert result.manifest["data"]["limit"] == 5


def test_a_decoding_parameter_only_one_side_received_is_named(write_split):
    # The pinned runtime CLI takes no decoding flags, so the two sides declare
    # the same decoding while only one of them was handed it. That gap belongs
    # in the report rather than in a comment nobody reads.
    # Declared as a field, not as a key in `describe()`. Read from the dict this
    # defaulted to None on both sides for any backend that omitted it, so the
    # two compared equal and the limitation disappeared.
    class RuntimeLikeBackend(FakeBackend):
        def describe(self) -> dict:
            return {"engine": "fake-runtime", "backend": "cpu"}

    class LibraryLikeBackend(FakeBackend):
        def describe(self) -> dict:
            return {"engine": "fake-library", "backend": "cpu"}

    rows = labelled_rows(8)
    result = verify(
        write_split,
        rows,
        candidate=RuntimeLikeBackend(texts=correct_texts(rows), decode_enforced=False),
        reference=LibraryLikeBackend(
            model="org/reference", texts=correct_texts(rows), decode_enforced=True
        ),
    )
    assert any("token limit is unverified" in note for note in result.manifest["limitations"])


def test_an_unknown_scorer_is_refused_before_anything_runs():
    """The scorer decides what "correct" means; litetune will not guess it.

    Refused in `__post_init__` rather than at the scoring call, which happens
    after both sides have generated — several minutes and a provisioned
    environment later, for a typo.
    """
    from litetune.evaluate import DataError
    from litetune.verify import VerifyRequest

    with pytest.raises(DataError, match="unknown scorer 'bleu'"):
        VerifyRequest(
            model=Path("m.litertlm"),
            reference="checkpoint",
            data=Path("held.jsonl"),
            scorer="bleu",
        )


def test_the_default_scorer_is_named_not_assumed():
    from litetune.verify import VerifyRequest

    request = VerifyRequest(
        model=Path("m.litertlm"),
        reference="checkpoint",
        data=Path("held.jsonl"),
    )
    assert request.scorer == "tool-call"


# -- a trailing terminator is termination, not part of the answer ----------


def test_exact_text_forgives_the_terminator_the_reference_backend_leaves_on(write_split):
    """The transformers reference decodes with skip_special_tokens=False so the
    liveness tier can see leakage. Under `exact-text` that trailing marker is
    not part of the answer -- liveness already treats one as normal termination
    -- and it must not turn a right answer into a wrong one.

    Found by the first end-to-end exact-text measurement: a reference that
    scores 0.6933 measured 0.0000 because every generation ended in `<eos>`,
    while the runtime side, which strips it, measured 0.6767. The tool reported
    a *resolved* conversion cost of -0.6767 across 406 discordant pairs; after
    the fix the same artifacts give an unresolved +0.0167 ±0.0201 across 38.

    The two sides carry *different* markers here on purpose. They are different
    decoders, and the normalisation has to be invariant to which one strips
    what -- with the candidate left bare, half of it is untested.
    """
    rows = text_rows(40)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] + "<end_of_turn>" for r in rows]),
        reference=FakeBackend(model="org/reference", texts=[r["target"] + "<eos>" for r in rows]),
        scorer="exact-text",
    )
    quality = result.manifest["quality"]
    assert quality["reference"]["exact_match"]["value"] == pytest.approx(1.0)
    assert quality["candidate"]["exact_match"]["value"] == pytest.approx(1.0)
    # Label-free, and silently wrong before this: it read 0.0 for two models
    # that agreed on every prompt.
    assert quality["agreement_with_reference"]["value"] == pytest.approx(1.0)
    assert result.manifest["attribution"]["conversion_cost"]["value"] == pytest.approx(0.0)


def test_a_target_that_ends_in_a_terminator_is_held_to_the_same_convention(write_split):
    """A constructed case, not a measured one: `</s>` is a strikethrough close
    tag as well as a terminator, so a markup task carries one at the END of a
    legitimate answer. Trimming generations but not targets would mirror the
    bug onto that axis. The reference additionally leaks `<eos>` the way the
    real transformers decoder does, so the "beyond the target's own markers"
    branch is exercised end-to-end rather than at the scorer alone.
    """
    rows = markup_rows(40)
    answers = [r["target"] for r in rows]
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=answers),
        reference=FakeBackend(model="org/reference", texts=[a + "<eos>" for a in answers]),
        scorer="exact-text",
    )
    assert result.manifest["quality"]["candidate"]["exact_match"]["value"] == pytest.approx(1.0)
    assert result.manifest["quality"]["reference"]["exact_match"]["value"] == pytest.approx(1.0)


def test_a_model_that_drops_the_targets_own_marker_is_scored_wrong(write_split):
    """The mirror of the case above, end-to-end: a candidate that drops the
    target's own `</s>` is wrong on every row, even though the reference --
    which reproduces the target and leaks its own decoder's `<eos>` beyond it
    -- is right on all of them. Quality is measured, not refused: `available`
    is True, and the run fails the gate on a real difference rather than
    coming back could-not-check.
    """
    rows = markup_rows(40)
    candidate_texts = [r["target"].removesuffix("</s>") for r in rows]
    reference_texts = [r["target"] + "<eos>" for r in rows]
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=FakeBackend(model="org/reference", texts=reference_texts),
        scorer="exact-text",
        max_conversion_cost=0.05,
    )
    quality = result.manifest["quality"]
    assert quality["available"] is True
    assert quality["candidate"]["exact_match"]["value"] == pytest.approx(0.0)
    assert quality["reference"]["exact_match"]["value"] == pytest.approx(1.0)
    assert result.status is Status.FAILED_GATE


def test_two_sides_that_genuinely_differ_are_scored_not_refused(write_split):
    """A real disagreement between the two sides must still be scored.

    A reference that emits a constant one-character tail on every row is not a
    decoder leaking its own marker -- it is a model that differs from the
    candidate. Quality is measured (`available` is True); a tool that refused
    here would be asserting a cause -- "this is a decoding convention, not a
    model difference" -- it never established.

    The reference is `TransformersLikeBackend`, not the plain `FakeBackend`:
    with `describe()["engine"] == "fake"` the terminator-recognition check is
    unreachable by scoping alone, which is exactly what
    `test_a_fake_reference_backend_does_not_trip_the_terminator_check` already
    covers, and this test's name would otherwise be true for the wrong reason.
    The tail is a genuine `<eos>`-terminated difference -- the reference is
    recognisably terminated -- so what is scored, and not refused, is the
    disagreement itself.
    """
    rows = text_rows(40)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=TransformersLikeBackend(
            model="org/reference", texts=[r["target"] + ".<eos>" for r in rows]
        ),
        scorer="exact-text",
    )
    quality = result.manifest["quality"]
    assert quality["available"] is True
    assert quality["candidate"]["exact_match"]["value"] == pytest.approx(1.0)
    assert quality["reference"]["exact_match"]["value"] == pytest.approx(0.0)
    assert result.status is not Status.FAILED_HARNESS


class TransformersLikeBackend(FakeBackend):
    """A reference that describes itself the way `HuggingFaceBackend` does.

    The refusal below is scoped to `describe()["engine"] == "transformers"`, so
    triggering or dodging it needs a fake that carries that engine name --
    `FakeBackend` itself reports "fake" and must never trip it.
    """

    def describe(self) -> dict:
        return {"engine": "transformers", "backend": "cpu"}


def test_a_reference_whose_terminator_is_unknown_is_not_a_conversion_cost(write_split):
    """The gate this check exists to close.

    Before it: a reference whose only defect is ending in a marker
    `harness.terminators` does not list scored 0.0000 under `exact-text` --
    the unrecognised suffix stayed attached to its "core" and never matched a
    bare target -- while a genuinely mediocre candidate, right on 60% of rows,
    scored above it. `paired_difference` read that as a *resolved* conversion
    cost of -0.6000 +/-0.1074, and a 0.05 gate read "within 0.0500" and exited
    0: a model that lost 40% of its accuracy, shipped clean.
    """
    rows = text_rows(200)
    candidate_texts = [r["target"] for r in rows[:120]] + ["not the label"] * 80
    reference_texts = [r["target"] + "<|assistant_end|>" for r in rows]
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=TransformersLikeBackend(model="org/reference", texts=reference_texts),
        scorer="exact-text",
        max_conversion_cost=0.05,
    )
    assert result.status is Status.FAILED_HARNESS
    assert result.manifest["quality"]["available"] is False
    assert not any(gate.get("outcome") == "passed" for gate in result.manifest["gates"])


def test_the_refusal_records_its_evidence_and_withdraws_attribution(write_split):
    """Four independent mutants survive without this: `first_unrecognised_tail`
    forced to `""`, `generations_ran` forced to `0`, `observed["terminators"]`
    forced to `[]`, and deleting the line that sets
    `run.manifest["attribution"] = _unattributable(detail)` -- which makes
    `attribution` *absent* from the manifest rather than present-and-unavailable,
    so a consumer reading it gets a `KeyError` instead of an explicit "not
    measured".
    """
    from litetune.metrics import TERMINATORS

    rows = text_rows(40)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=TransformersLikeBackend(
            model="org/reference", texts=[r["target"] + "<|assistant_end|>" for r in rows]
        ),
        scorer="exact-text",
    )
    assert result.status is Status.FAILED_HARNESS
    check = next(
        c for c in result.manifest["checks"] if c["name"] == "reference terminator recognised"
    )
    assert "<|assistant_end|>" in check["observed"]["first_unrecognised_tail"]
    assert check["observed"]["generations_ran"] == 40
    assert check["observed"]["terminators"] == list(TERMINATORS)
    assert result.manifest["attribution"]["conversion_cost"]["available"] is False


def test_a_reference_that_did_not_finish_is_reported_as_such_not_as_a_bad_vocabulary(
    write_split,
):
    """Ordering: the reference's own liveness tier judges before the vocabulary does.

    A generation that never ran carries no text, so it can never have had a
    terminator trimmed. If the unrecognised-terminator check ran first, a
    reference that simply failed to finish would be reported as one whose turn
    marker this tool does not know -- a claim about a vocabulary made from a
    run that produced nothing. The tier above rejects any failed generation, so
    at the check's own line every generation it sees has run; the `g.ok` filter
    there is what keeps its population the same as `terminators_trimmed`'s
    rather than a live guard.
    """
    rows = text_rows(40)

    class LastGenerationNeverRan(TransformersLikeBackend):
        def generate(self, prompts, events=None):
            generations = super().generate(prompts, events=events)
            last = generations[-1]
            return [
                *generations[:-1],
                Generation(index=last.index, prompt=last.prompt, harness_error="did not run"),
            ]

    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=LastGenerationNeverRan(
            model="org/reference",
            texts=[r["target"] + "<|assistant_end|>" for r in rows],
        ),
        scorer="exact-text",
    )
    assert result.status is Status.FAILED_HARNESS
    names = [check["name"] for check in result.manifest["checks"]]
    assert "reference is comparable" in names
    assert "reference terminator recognised" not in names


def test_one_recognised_generation_does_not_vouch_for_the_rest(write_split):
    """The evidence is per generation, not an all-or-nothing aggregate.

    An earlier form of this check refused only when *no* generation had a
    terminator trimmed. A reference ending in an unlisted `<|assistant_end|>`
    on 199 of 200 rows and, by accident, a recognised `<eos>` on the last one
    passed that test and reproduced the whole failure: reference 0.005,
    candidate 0.600, a resolved conversion cost of -0.595, exit 0. One row
    cannot speak for the other 199.
    """
    rows = text_rows(200)
    candidate_texts = [r["target"] for r in rows[:120]] + ["not the label"] * 80
    reference_texts = [r["target"] + "<|assistant_end|>" for r in rows[:-1]]
    reference_texts.append(rows[-1]["target"] + "<eos>")
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=TransformersLikeBackend(model="org/reference", texts=reference_texts),
        scorer="exact-text",
        max_conversion_cost=0.05,
    )
    assert result.status is Status.FAILED_HARNESS
    assert result.manifest["quality"]["available"] is False
    assert not any(gate.get("outcome") == "passed" for gate in result.manifest["gates"])
    observed = next(
        check["observed"]
        for check in result.manifest["checks"]
        if check["name"] == "reference terminator recognised"
    )
    assert observed["unterminated"] == 199
    assert observed["generations_ran"] == 200


def test_a_reference_that_sometimes_runs_to_the_token_bound_is_still_scored(write_split):
    """The threshold is what separates a short vocabulary from a long answer.

    A generation that hit the token bound carries no terminator either, and
    the text cannot tell the two apart -- which is why this is a share with a
    named threshold recorded beside the other liveness numbers rather than a
    per-row verdict. README's own run had 8 of 640 reach the bound; two of two
    hundred here must not cost the measurement.
    """
    rows = text_rows(200)
    reference_texts = [r["target"] + "<end_of_turn>\n<eos>" for r in rows[:-2]]
    reference_texts += ["an answer that was still going when the bound cut it"] * 2
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=TransformersLikeBackend(model="org/reference", texts=reference_texts),
        scorer="exact-text",
    )
    assert result.status is Status.PASSED
    assert not [
        check
        for check in result.manifest["checks"]
        if check["name"] == "reference terminator recognised"
    ]


def test_a_healthy_gemma_reference_is_not_refused(write_split):
    """The guard against the refusal firing on a healthy reference.

    Two markers here -- a chat template's close in front of the tokenizer's
    own eos. A reference that trimmed anything has a recognised terminator and
    must not be refused as if its marker were absent from the vocabulary.
    Measured, gemma-3-270m-it returns one marker rather than two, because
    `<end_of_turn>` is in its `eos_token_id` and generation stops there; the
    two-marker shape is the harder case and the one worth pinning.
    """
    rows = text_rows(40)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=TransformersLikeBackend(
            model="org/reference", texts=[r["target"] + "<end_of_turn>\n<eos>" for r in rows]
        ),
        scorer="exact-text",
    )
    assert result.status is Status.PASSED
    assert result.manifest["quality"]["available"] is True


def test_a_fake_reference_backend_does_not_trip_the_terminator_check(write_split):
    """Scoped to the `transformers` engine specifically.

    Without that scoping, every fixture in this file that uses the default
    `FakeBackend` -- none of which ends its generations in a terminator at all
    -- would have to grow one just to keep passing.
    """
    rows = text_rows(40)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=FakeBackend(model="org/reference", texts=[r["target"] for r in rows]),
        scorer="exact-text",
    )
    assert result.status is Status.PASSED
    assert result.manifest["quality"]["available"] is True


def test_targets_that_carry_a_turn_marker_are_named_as_a_limitation(write_split):
    """Observable from the targets alone, so it names a fact about the data,
    not a claim about either model."""
    rows = markup_rows(40)  # every target ends in </s>, a real terminator
    answers = [r["target"] for r in rows]
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=answers),
        reference=FakeBackend(model="org/reference", texts=answers),
        scorer="exact-text",
    )
    assert any(
        "40 of 40 held-out targets end in a turn marker" in note
        for note in result.manifest["limitations"]
    )


def test_a_candidate_that_emits_its_terminator_as_text_is_named_as_a_limitation(write_split):
    """On a litert-lm candidate the runtime consumes its own stop token before
    this tool ever sees the text, so a candidate generation ending in one is
    the model emitting a terminator as text and continuing -- named with both
    the count and the worst offender, not folded into a single number.
    """
    rows = text_rows(40)
    candidate_texts = [r["target"] for r in rows[:-1]]
    candidate_texts.append(rows[-1]["target"] + "<eos><eos><eos>")
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=FakeBackend(model="org/reference", texts=[r["target"] for r in rows]),
        scorer="exact-text",
    )
    assert any(
        "1 of 40 candidate generations end in a" in note and "carrying 3" in note
        for note in result.manifest["limitations"]
    )


def test_a_clean_exact_text_run_names_neither_terminator_limitation(write_split):
    """A false limitation is the same class of defect as a false number, for a
    tool whose product is a manifest that refuses to claim what it did not
    observe.

    Two mutants make one of these fire on *every* clean run regardless of what
    happened: `_strip_terminators(...)[1]` weakened to always return `[0]`
    makes every held-out target look marked, and the candidate-count condition
    weakened to always-true makes every candidate generation look like it kept
    a terminator. Both existing tests for these notes are presence-only, so
    neither would catch either mutant printing "40 of 40 held-out targets end
    in a turn marker" against plain, unmarked `label_3` targets.
    """
    rows = text_rows(40)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=FakeBackend(model="org/reference", texts=[r["target"] for r in rows]),
        scorer="exact-text",
    )
    notes = result.manifest["limitations"]
    assert not any("end in a turn marker" in note for note in notes), notes
    assert not any("candidate generations end in a" in note for note in notes), notes


def test_how_many_terminators_were_trimmed_reaches_the_manifest(write_split):
    """A generation needing one marker removed stopped normally; one needing
    seven never stopped, and both score the same once they are off. README
    leads "Known to be broken" with a model that does not stop, so the count
    has to survive the normalisation that hides it.

    Non-uniform on purpose, in both dimensions: a fixture where every
    generation carries a marker cannot tell "how many needed trimming" from
    `n`, and one where every generation carries the same count cannot tell
    `max` from `min`. Half the candidate's rows are bare, the way a runtime
    that consumes its own stop token returns them; one row carries seven.

    The reference's last generation never ran at all, which pins the
    ran-only denominator: it must not count in `generations_trimmed`, in
    `most_trimmed_from_one_generation`, or in the renamed third key -- the
    old `n` duplicated `generations.n`, written by `MeasurementPoint.as_dict()`
    two keys above it in the same manifest object.
    """
    rows = text_rows(40)
    candidate_texts = [r["target"] for r in rows[:20]]
    candidate_texts += [r["target"] + "<eos>" for r in rows[20:-1]]
    candidate_texts.append(rows[-1]["target"] + "<eos>" * 7)

    class LastReferenceGenerationNeverRan(FakeBackend):
        def generate(self, prompts, events=None):
            generations = super().generate(prompts, events=events)
            last = generations[-1]
            return [
                *generations[:-1],
                Generation(index=last.index, prompt=last.prompt, harness_error="did not run"),
            ]

    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=LastReferenceGenerationNeverRan(
            model="org/reference", texts=[r["target"] + "<eos>" for r in rows]
        ),
        scorer="exact-text",
    )
    candidate_trimmed = result.manifest["measurements"]["candidate"]["terminators_trimmed"]
    assert candidate_trimmed["generations_trimmed"] == 20
    assert candidate_trimmed["most_trimmed_from_one_generation"] == 7
    assert candidate_trimmed["over_generations_that_ran"] == 40
    reference_trimmed = result.manifest["measurements"]["reference"]["terminators_trimmed"]
    assert reference_trimmed["generations_trimmed"] == 39
    assert reference_trimmed["most_trimmed_from_one_generation"] == 1
    assert reference_trimmed["over_generations_that_ran"] == 39
    assert "terminators_trimmed" not in result.manifest["quality"]


def test_the_terminator_vocabulary_reaches_the_manifest(write_split):
    """Two runs scored under different vocabularies are not comparable, and
    nothing else in the file would say so -- the reason `liveness_thresholds`
    is recorded beside it. The whole list, not mere membership: a `[:1]` slice
    of the vocabulary would still contain `<eos>`.
    """
    from litetune.metrics import TERMINATORS

    rows = labelled_rows(20)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert result.manifest["harness"]["terminators"] == list(TERMINATORS)


def test_the_terminator_vocabulary_reaches_the_manifest_even_on_a_dead_run(write_split):
    """Written in `_Run.__post_init__`, before any check runs.

    The liveness tier's own checks close over `metrics.TERMINATORS` directly,
    not over this field -- `harness.terminators` is written for whoever reads
    the manifest later, and nothing in litetune reads it back. This test is
    what holds it there: a `failed_smoke` manifest still has to carry it,
    because a comparison between two runs scored under different vocabularies
    would not be valid, and this field is what a reader would check.
    """
    rows = labelled_rows(8)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[""], returncode=1),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )
    assert result.status is Status.FAILED_SMOKE
    assert "terminators" in result.manifest["harness"]


# -- a reference already at the floor cannot baseline a conversion cost -----


def _floored_reference_rows_and_texts():
    """200 rows, a candidate right on 60%, and a reference zeroed by a marker
    `harness.terminators` does not list. The reference's generation trims the
    listed `<eos>` -- so it reads as recognisably terminated and the
    unterminated-share refusal (G2 above) never fires -- but is left holding
    the unlisted `<|assistant_end|>` as content, so it reproduces no target at
    all: exact_match 0.0000. This is the case that slips past every other
    check to reach `_attribute`'s own guard.
    """
    rows = text_rows(200)
    candidate_texts = [r["target"] for r in rows[:120]] + ["not the label"] * 80
    reference_texts = [r["target"] + "<|assistant_end|><eos>" for r in rows]
    return rows, candidate_texts, reference_texts


def test_a_reference_zeroed_by_an_unrecognised_terminator_is_not_a_conversion_cost_baseline(
    write_split,
):
    """The guard in `_attribute`'s `FLOAT_TWIN` branch: `if
    reference.exact_match.value <= NEAR_ZERO`. It is deliberately blind to
    *why* the reference is on the floor -- an unrecognised turn terminator
    here, but the wrong prompt mode or targets in a format the reference was
    never trained for would land on the same floor -- only that a conversion
    cost measures how far a model fell below its own float twin, and nothing
    falls below a baseline already at zero.

    Before this guard existed: reference 0.0000, candidate 0.6000, a
    *resolved* conversion cost of -0.6000 against a 0.05 threshold, exited 0.
    Mutating the guard's condition to `if False:` reproduces exactly that.
    """
    rows, candidate_texts, reference_texts = _floored_reference_rows_and_texts()
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=FakeBackend(model="org/reference", texts=reference_texts),
        scorer="exact-text",
        max_conversion_cost=0.05,
    )
    assert result.status is Status.FAILED_HARNESS
    assert result.manifest["attribution"]["conversion_cost"]["available"] is False
    assert not any(gate.get("outcome") == "passed" for gate in result.manifest["gates"])


def test_the_same_floored_reference_still_leaves_quality_reported(write_split):
    """Only the *attribution* is meaningless here, not the measurement: both
    sides were genuinely scored, so `quality` must still carry both
    `exact_match` values even though nothing can be attributed from them.
    Without a threshold the run is `UNMEASURED`, not `PASSED` -- there is
    nothing for a gate to have judged.
    """
    rows, candidate_texts, reference_texts = _floored_reference_rows_and_texts()
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=candidate_texts),
        reference=FakeBackend(model="org/reference", texts=reference_texts),
        scorer="exact-text",
    )
    assert result.status is Status.UNMEASURED
    assert result.manifest["attribution"]["conversion_cost"]["available"] is False
    quality = result.manifest["quality"]
    assert quality["available"] is True
    assert quality["candidate"]["exact_match"]["value"] == pytest.approx(0.6)
    assert quality["reference"]["exact_match"]["value"] == pytest.approx(0.0)


def test_a_healthy_reference_is_not_floored_by_the_zero_guard(write_split):
    """The guard must not fire on the ordinary case: a reference scoring 1.0 is
    nowhere near the floor, and the run this branch exists for must come back
    exactly as every other healthy-reference test in this file does -- a real,
    numeric conversion cost, not withdrawn.
    """
    rows = text_rows(200)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=[r["target"] for r in rows]),
        reference=FakeBackend(
            model="org/reference", texts=[r["target"] + "<end_of_turn>\n<eos>" for r in rows]
        ),
        scorer="exact-text",
        max_conversion_cost=0.60,
    )
    assert result.status is Status.PASSED
    cost = result.manifest["attribution"]["conversion_cost"]
    assert cost["available"] is True
    assert cost["value"] == pytest.approx(0.0)


def test_a_genuinely_bad_conversion_is_still_gated_when_the_reference_is_healthy(write_split):
    """The guard against Change 1 firing on the case it must not excuse: a
    reference that is right and cleanly terminated on every row, against a
    candidate that is wrong on every row, is a real conversion cost -- not a
    reference at the floor -- and must still fail the gate.
    """
    rows = text_rows(200)
    result = verify(
        write_split,
        rows,
        candidate=FakeBackend(texts=["not the label"] * len(rows)),
        reference=FakeBackend(
            model="org/reference", texts=[r["target"] + "<end_of_turn>\n<eos>" for r in rows]
        ),
        scorer="exact-text",
        max_conversion_cost=0.05,
    )
    assert result.status is Status.FAILED_GATE
    cost = result.manifest["attribution"]["conversion_cost"]
    assert cost["available"] is True
    assert cost["value"] == pytest.approx(1.0)
