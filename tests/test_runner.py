"""Composition adds caching and sequencing; it is never a precondition.

The first group of tests is the important one: a stage runs on its own, with
explicit inputs, no run directory, no spec and no cache. Everything after it is
the layer built on top -- and each of those tests is a cache rule that was
observed to be wrong when it was left out.
"""

import json
from pathlib import Path

import pytest
from stage_fakes import DATA_B, FakeStage, collect_events, make_spec, pipeline

from litetune.manifest import CacheOutcome, RunStatus
from litetune.runner import (
    Artifact,
    CacheIndex,
    MissingInput,
    Runner,
    StageInput,
    StageResult,
    new_run_id,
    run_stage,
)
from litetune.storage import LocalStorage, hash_bytes


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "store")


@pytest.fixture
def cache(storage: LocalStorage) -> CacheIndex:
    return CacheIndex(storage=storage)


def build(storage, cache, spec, run_id, tmp_path, stages=None):
    return Runner(
        stages=stages if stages is not None else pipeline(),
        spec=spec,
        storage=storage,
        run_id=run_id,
        cache=cache,
        workspace_root=tmp_path / "work",
    )


def cache_outcomes(manifest) -> dict[str, str]:
    return {stage.name: stage.cache.value for stage in manifest.stages}


# ---------------------------------------------------------------------------
# A stage on its own
# ---------------------------------------------------------------------------


def test_a_stage_runs_with_no_run_directory_no_spec_and_no_prior_stage():
    # The product's entry point is verifying an artifact litetune did not
    # produce. If composition were a precondition, that entry point would not
    # exist.
    stage = FakeStage(name="verify")
    outcome = run_stage(stage)

    assert stage.call_count == 1
    assert outcome.status is RunStatus.PASSED
    assert stage.calls[0].spec is None
    assert stage.calls[0].storage is None


def test_a_directly_invoked_stage_produces_a_manifest_scaled_to_itself():
    outcome = run_stage(FakeStage(name="verify"))
    data = outcome.manifest.as_dict()

    assert data["scope"] == "stage"
    assert [s["name"] for s in data["stages"]] == ["verify"]
    # Marked, not omitted: a reader must be able to tell "ran outside a job"
    # from "this manifest is missing half its content".
    assert data["spec_hash"]["available"] is False
    assert data["stages"][0]["env_identity"]["available"] is False
    assert "no job spec" in data["stages"][0]["env_identity"]["reason"]


def test_a_directly_invoked_stage_records_its_artifacts_by_content():
    outcome = run_stage(FakeStage(name="verify"))
    artifact = outcome.record.artifacts[0]

    assert artifact.name == "verify.json"
    assert artifact.content_hash == hash_bytes(Path(artifact.key).read_bytes())
    assert artifact.bytes > 0


def test_a_directly_invoked_stage_takes_explicit_inputs(tmp_path):
    model = tmp_path / "their-model.litertlm"
    model.write_bytes(b"an artifact litetune did not produce")
    stage = FakeStage(name="verify", input_names=("model",))

    outcome = run_stage(stage, inputs=[StageInput.from_file("model", model)])

    assert stage.calls[0].input("model").locator == str(model)
    assert outcome.record.inputs[0].content_hash == hash_bytes(model.read_bytes())


def test_a_directly_invoked_stage_says_which_input_it_is_missing():
    stage = FakeStage(name="verify", input_names=("model",))
    with pytest.raises(MissingInput) as exc:
        run_stage(stage)
    assert "model" in str(exc.value)
    assert stage.call_count == 0


def test_a_directly_invoked_stage_executes_rather_than_consulting_a_cache(storage, cache, tmp_path):
    # The same stage, the same spec, the same inputs, and an entry sitting in a
    # cache it was not handed. It runs.
    spec = make_spec()
    train, _, _ = pipeline()
    build(storage, cache, spec, "r1", tmp_path, stages=[train]).run()
    assert train.call_count == 1

    again = FakeStage(
        name="train", spec_sections=("base_model", "dataset", "train"), env_names=("train",)
    )
    outcome = run_stage(again, spec=spec, storage=storage, run_id="direct-1")

    assert again.call_count == 1
    assert outcome.record.cache is CacheOutcome.NOT_CONSULTED
    assert outcome.record.cache_key.as_dict()["available"] is False


def test_a_directly_invoked_stage_with_a_spec_records_the_environment_it_ran_in(storage):
    spec = make_spec()
    stage = FakeStage(name="export", spec_sections=("base_model", "export"), env_names=("export",))
    outcome = run_stage(stage, spec=spec, storage=storage, run_id="direct-1")

    assert outcome.record.env_identity == spec.env_identity_for("export")
    assert outcome.manifest.as_dict()["environments"]["export"]


def test_a_stage_that_raises_reports_error_and_not_a_verdict():
    def explode(ctx):
        raise RuntimeError("libvulkan.so.1: cannot open shared object file")

    outcome = run_stage(FakeStage(name="measure", body=explode))

    # `failed` would say the model is bad. Nothing here observed the model.
    assert outcome.status is RunStatus.ERROR
    assert "RuntimeError" in outcome.record.detail
    assert "Traceback" in outcome.record.output


def test_a_declared_artifact_that_is_not_there_is_not_a_pass(tmp_path):
    def claim_without_writing(ctx):
        return StageResult(
            status=RunStatus.PASSED,
            artifacts=(Artifact(name="model.litertlm", path=ctx.workspace / "model.litertlm"),),
        )

    outcome = run_stage(FakeStage(name="export", body=claim_without_writing))

    # Exit code zero plus a claimed output file is the documented shape of a bad
    # conversion in this toolchain; a claimed file that is not even there cannot
    # be reported as a pass.
    assert outcome.status is RunStatus.FAILED_HARNESS
    assert "model.litertlm" in outcome.record.detail


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------


def test_a_stages_artifact_becomes_the_next_stages_input(storage, cache, tmp_path):
    train, export, measure = stages = pipeline()
    manifest = build(storage, cache, make_spec(), "r1", tmp_path, stages).run()

    assert manifest.status is RunStatus.PASSED
    assert [s.name for s in manifest.stages] == ["train", "export", "measure"]
    assert export.calls[0].input("train.json").content_hash == (
        manifest.stage("train").artifacts[0].content_hash
    )
    assert measure.call_count == 1


def test_the_run_manifest_is_persisted(storage, cache, tmp_path):
    build(storage, cache, make_spec(), "r1", tmp_path).run()
    written = json.loads(storage.read_text("runs/r1/manifest.json"))
    assert written["status"] == "passed"
    assert written["spec"]["base_model"]["revision"]


def test_a_stage_whose_input_nobody_produced_is_a_wiring_error(storage, cache, tmp_path):
    orphan = FakeStage(
        name="export",
        spec_sections=("base_model", "export"),
        input_names=("nope",),
        env_names=("export",),
    )
    manifest = build(storage, cache, make_spec(), "r1", tmp_path, [orphan]).run()

    assert manifest.status is RunStatus.ERROR
    assert "nope" in manifest.stage("export").detail
    assert orphan.call_count == 0


def test_the_events_a_consumer_needs_are_emitted(storage, cache, tmp_path):
    events, seen = collect_events()
    Runner(
        stages=pipeline(),
        spec=make_spec(),
        storage=storage,
        run_id="r1",
        events=events,
        cache=cache,
        workspace_root=tmp_path / "work",
    ).run()

    kinds = [event.kind for event in seen]
    assert kinds.count("stage_started") == 3
    assert kinds.count("stage_finished") == 3
    assert kinds.count("artifact_written") == 3
    started = next(e for e in seen if e.kind == "stage_started")
    assert started.stage == "train"
    finished = next(e for e in seen if e.kind == "stage_finished")
    assert finished.as_dict()["status"] == "passed"


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_an_unchanged_spec_resumes_every_stage(storage, cache, tmp_path):
    spec = make_spec()
    build(storage, cache, spec, "r1", tmp_path).run()

    second = pipeline()
    manifest = build(storage, cache, spec, "r2", tmp_path, second).run()

    assert cache_outcomes(manifest) == {"train": "hit", "export": "hit", "measure": "hit"}
    assert [stage.call_count for stage in second] == [0, 0, 0]
    assert manifest.status is RunStatus.PASSED


def test_replacing_the_dataset_content_re_runs_everything_downstream(storage, cache, tmp_path):
    # Same URI, different bytes. Keying on the location would resume the whole
    # pipeline and report a green manifest over a model trained on the old file.
    build(storage, cache, make_spec(), "r1", tmp_path).run()

    second = pipeline()
    manifest = build(
        storage, cache, make_spec(dataset={"content_sha256": DATA_B}), "r2", tmp_path, second
    ).run()

    assert cache_outcomes(manifest) == {"train": "miss", "export": "miss", "measure": "miss"}
    assert [stage.call_count for stage in second] == [1, 1, 1]


def test_moving_the_dataset_file_resumes(storage, cache, tmp_path):
    build(storage, cache, make_spec(), "r1", tmp_path).run()

    second = pipeline()
    manifest = build(
        storage, cache, make_spec(dataset={"uri": "s3://elsewhere.jsonl"}), "r2", tmp_path, second
    ).run()

    assert cache_outcomes(manifest) == {"train": "hit", "export": "hit", "measure": "hit"}


def test_tightening_a_gate_re_runs_no_measurement(storage, cache, tmp_path):
    # Hours of generation must not be spent to re-judge numbers already
    # recorded. This is why gates live in their own section.
    build(storage, cache, make_spec(gates={"max_conversion_cost": 0.05}), "r1", tmp_path).run()

    second = pipeline()
    manifest = build(
        storage, cache, make_spec(gates={"max_conversion_cost": 0.001}), "r2", tmp_path, second
    ).run()

    assert cache_outcomes(manifest) == {"train": "hit", "export": "hit", "measure": "hit"}
    assert [stage.call_count for stage in second] == [0, 0, 0]


def test_changing_the_environment_re_runs_the_stage_that_runs_in_it(storage, cache, tmp_path):
    # An unpinned nightly produced a working export on 2026-08-26 and
    # AttributeError: pad_token on 2026-08-30. Repinning must re-export.
    build(storage, cache, make_spec(), "r1", tmp_path).run()

    train, export, measure = second = pipeline()
    manifest = build(
        storage,
        cache,
        make_spec(
            toolchain={"export": ["litert-torch-nightly==0.10.0.dev20260830", "numpy==2.0.2"]}
        ),
        "r2",
        tmp_path,
        second,
    ).run()

    assert manifest.stage("train").cache is CacheOutcome.HIT
    assert manifest.stage("export").cache is CacheOutcome.MISS
    assert export.call_count == 1
    # `measure` still resumes, and that is the content keying paying for itself:
    # the re-export produced byte-identical output, so nothing downstream of it
    # changed. A key over the *location* of the export could not tell.
    assert manifest.stage("measure").cache is CacheOutcome.HIT
    assert measure.call_count == 0


def test_resume_can_be_turned_off(storage, cache, tmp_path):
    build(storage, cache, make_spec(), "r1", tmp_path).run()

    second = pipeline()
    manifest = build(storage, cache, make_spec(), "r2", tmp_path, second).run(resume=False)

    assert cache_outcomes(manifest) == {
        "train": "not_consulted",
        "export": "not_consulted",
        "measure": "not_consulted",
    }
    assert [stage.call_count for stage in second] == [1, 1, 1]


def test_an_artifact_edited_in_place_is_a_miss_not_a_hit(storage, cache, tmp_path):
    build(storage, cache, make_spec(), "r1", tmp_path).run()
    # Same key, different bytes: the dataset-at-a-URI failure, one directory
    # down. A cache entry naming it must not still read as a hit.
    storage.write_text("runs/r1/train/train.json", '{"tampered": true}')

    second = pipeline()
    manifest = build(storage, cache, make_spec(), "r2", tmp_path, second).run()

    assert manifest.stage("train").cache is CacheOutcome.MISS
    assert second[0].call_count == 1


def test_a_deleted_artifact_is_a_miss(storage, cache, tmp_path):
    build(storage, cache, make_spec(), "r1", tmp_path).run()
    storage.local_path("runs/r1/train/train.json").unlink()

    second = pipeline()
    manifest = build(storage, cache, make_spec(), "r2", tmp_path, second).run()
    assert manifest.stage("train").cache is CacheOutcome.MISS


def test_an_input_with_no_content_hash_makes_the_cache_unusable(storage, cache, tmp_path):
    stage = FakeStage(
        name="export",
        spec_sections=("base_model", "export"),
        input_names=("model",),
        env_names=("export",),
    )
    unidentified = StageInput(name="model", locator="hf://someone/model", content_hash=None)

    manifest = build(storage, cache, make_spec(), "r1", tmp_path, [stage]).run([unidentified])

    # A hit under a key that does not describe what the stage ran on would be
    # reuse of an artifact produced from unknown material.
    assert manifest.stage("export").cache is CacheOutcome.UNUSABLE
    assert stage.call_count == 1
    assert any("no content hash" in note for note in manifest.limitations)


def test_a_cache_index_that_will_not_parse_costs_time_and_not_correctness(storage, cache, tmp_path):
    storage.write_text("cache/index.json", "{not json")

    stages = pipeline()
    manifest = build(storage, cache, make_spec(), "r1", tmp_path, stages).run()

    assert manifest.status is RunStatus.PASSED
    assert [stage.call_count for stage in stages] == [1, 1, 1]
    # Dropped loudly: in the manifest, not only in a log nobody reads.
    assert any("cache index could not be read" in note for note in manifest.limitations)


def test_a_failed_stage_is_not_cached(storage, cache, tmp_path):
    calls = {"n": 0}

    def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            return StageResult(status=RunStatus.FAILED_HARNESS, detail="litert-lm would not start")
        return StageResult(status=RunStatus.PASSED, detail="second time")

    first = FakeStage(
        name="train",
        spec_sections=("base_model", "dataset", "train"),
        env_names=("train",),
        body=flaky,
    )
    build(storage, cache, make_spec(), "r1", tmp_path, [first]).run()

    second = FakeStage(
        name="train",
        spec_sections=("base_model", "dataset", "train"),
        env_names=("train",),
        body=flaky,
    )
    manifest = build(storage, cache, make_spec(), "r2", tmp_path, [second]).run()

    # A failure is at least as likely to be about this machine as about the
    # artifact; replaying it as a hit would make a transient failure permanent.
    assert manifest.stage("train").status is RunStatus.PASSED
    assert manifest.stage("train").cache is CacheOutcome.MISS


# ---------------------------------------------------------------------------
# Failure retains everything
# ---------------------------------------------------------------------------


def test_a_failure_retains_every_artifact_and_records_what_the_stage_printed(
    storage, cache, tmp_path
):
    def fail_after_writing(ctx):
        partial = ctx.workspace / "model.litertlm.partial"
        partial.write_bytes(b"half an export")
        raise RuntimeError("exporter died")

    train, export, measure = pipeline()
    export.body = fail_after_writing
    manifest = build(storage, cache, make_spec(), "r1", tmp_path, [train, export, measure]).run()

    assert manifest.status is RunStatus.ERROR
    # The completed stage keeps its artifact.
    assert storage.exists("runs/r1/train/train.json")
    # The failing stage keeps its partial output, and the manifest says where.
    workspace = Path(manifest.stage("export").workspace)
    assert (workspace / "model.litertlm.partial").read_bytes() == b"half an export"
    assert "exporter died" in manifest.stage("export").output
    # Nothing after it is claimed to have run.
    assert [s["name"] for s in manifest.as_dict()["not_reached"]] == ["measure"]
    assert measure.call_count == 0


def test_a_failing_stage_does_not_delete_an_earlier_runs_artifacts(storage, cache, tmp_path):
    build(storage, cache, make_spec(), "r1", tmp_path).run()
    before = storage.list("runs/r1/")

    train, export, measure = pipeline()
    export.body = lambda ctx: StageResult(status=RunStatus.FAILED_SMOKE, detail="degenerate output")
    build(
        storage,
        cache,
        make_spec(dataset={"content_sha256": DATA_B}),
        "r2",
        tmp_path,
        [train, export, measure],
    ).run()

    assert storage.list("runs/r1/") == before


def test_a_failed_run_still_writes_a_manifest(storage, cache, tmp_path):
    train, export, measure = pipeline()
    export.body = lambda ctx: StageResult(
        status=RunStatus.FAILED_GATE, detail="cost 0.09 over 0.05"
    )
    build(storage, cache, make_spec(), "r1", tmp_path, [train, export, measure]).run()

    written = json.loads(storage.read_text("runs/r1/manifest.json"))
    assert written["status"] == "failed_gate"
    assert written["stages"][1]["detail"] == "cost 0.09 over 0.05"


def test_a_run_id_names_the_spec_it_came_from():
    spec = make_spec()
    assert spec.hash in new_run_id(spec)


def test_a_stages_checks_reach_the_manifest_with_all_three_outcomes(storage, cache, tmp_path):
    from litetune.checks import Check

    def with_checks(ctx):
        return StageResult(
            status=RunStatus.FAILED_HARNESS,
            detail="the runtime never started",
            checks=(
                Check.passed("artifact exists", "285577392 bytes", 285577392),
                Check.unchecked("model runs", "libvulkan1 is not installed on this platform"),
            ),
        )

    stage = FakeStage(
        name="measure",
        spec_sections=("base_model", "eval"),
        env_names=("runtime",),
        body=with_checks,
    )
    manifest = build(storage, cache, make_spec(), "r1", tmp_path, [stage]).run()

    # `could_not_check` has to survive into the manifest as itself. Collapsing
    # it to a failure there would undo checks.py one layer up.
    outcomes = [check["outcome"] for check in manifest.as_dict()["stages"][0]["checks"]]
    assert outcomes == ["passed", "could_not_check"]


def test_an_artifact_named_like_a_path_is_a_recorded_problem_not_a_crash(storage, cache, tmp_path):
    def escape(ctx):
        out = ctx.workspace / "model.litertlm"
        out.write_bytes(b"weights")
        return StageResult(
            status=RunStatus.PASSED, artifacts=(Artifact(name="../../escape", path=out),)
        )

    stage = FakeStage(
        name="export", spec_sections=("base_model", "export"), env_names=("export",), body=escape
    )
    manifest = build(storage, cache, make_spec(), "r1", tmp_path, [stage]).run()

    assert manifest.status is RunStatus.FAILED_HARNESS
    assert "escape" in manifest.stage("export").detail
    assert storage.list("runs/r1/export/") == []


def test_a_caller_may_supply_the_workspace(tmp_path):
    given = tmp_path / "mine"
    given.mkdir()
    stage = FakeStage(name="verify")
    outcome = run_stage(stage, workspace=given)

    assert stage.calls[0].workspace == given
    assert (given / "verify.json").is_file()
    assert outcome.record.workspace == str(given)
