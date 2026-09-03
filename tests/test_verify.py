"""End-to-end `verify`, with faked backends.

Each test below is a failure that happened, or a claim the tool must refuse to
make. The three load-bearing ones: a dead model never reaches the quality tier,
a model that merely runs is never reported as verified, and a comparison whose
two sides were not measured the same way is refused rather than annotated.
"""

from pathlib import Path

import pytest
from conftest import FakeBackend, call_text, correct_texts, labelled_rows

from litetune.evaluate import PromptMode
from litetune.verify import (
    BackendPair,
    ReferenceRole,
    Status,
    VerifyRequest,
    run_verify,
)


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
