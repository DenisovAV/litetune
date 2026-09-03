"""The manifest's honesty rules.

A run that could not tell you is not a run that passed, and a field a single
stage could not know is marked rather than dropped -- an absent key reads as an
oversight, an explicit reason reads as a fact.
"""

import json

import pytest

from litetune.manifest import (
    ArtifactRecord,
    CacheOutcome,
    RunManifest,
    RunStatus,
    StageRecord,
    summarise,
    worst_status,
)
from litetune.metrics import Unavailable
from litetune.storage import LocalStorage


def record(name: str, status: RunStatus, **kwargs) -> StageRecord:
    return StageRecord(
        name=name,
        status=status,
        cache=kwargs.pop("cache", CacheOutcome.MISS),
        cache_key=kwargs.pop("cache_key", "deadbeefdeadbeef"),
        env_identity=kwargs.pop("env_identity", "env-1"),
        **kwargs,
    )


# -- aggregation ------------------------------------------------------------


def test_a_run_of_passes_passed():
    assert worst_status([RunStatus.PASSED, RunStatus.PASSED]) is RunStatus.PASSED


def test_could_not_tell_dominates_a_verdict():
    # Same rule as checks.CheckSet: a caller promised an answer and denied one
    # is worse off than a caller given a clear no.
    assert worst_status([RunStatus.FAILED_GATE, RunStatus.INCONCLUSIVE]) is RunStatus.INCONCLUSIVE
    assert (
        worst_status([RunStatus.FAILED_SMOKE, RunStatus.FAILED_HARNESS]) is RunStatus.FAILED_HARNESS
    )
    assert worst_status([RunStatus.PASSED, RunStatus.UNMEASURED]) is RunStatus.UNMEASURED


def test_an_error_dominates_everything():
    assert worst_status(list(RunStatus)) is RunStatus.ERROR


def test_a_manifest_with_no_stages_has_established_nothing():
    # Reporting PASSED for a run that never ran a stage is the exact mistake
    # checks.py exists to prevent, one level up.
    assert RunManifest(run_id="r1", spec_hash="abc").status is RunStatus.INCONCLUSIVE


def test_unmeasured_is_a_status_because_verify_can_return_it():
    verify = pytest.importorskip("litetune.verify")
    ours = {status.value for status in RunStatus}
    missing = {status.value for status in verify.Status} - ours
    assert not missing, f"verify can return statuses the manifest cannot record: {missing}"


def test_a_status_is_not_always_a_verdict_about_the_model():
    assert RunStatus.FAILED_GATE.conclusive
    assert not RunStatus.FAILED_HARNESS.conclusive
    assert not RunStatus.INCONCLUSIVE.conclusive


# -- the record ------------------------------------------------------------


def test_the_worst_stage_decides_the_run():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(record("train", RunStatus.PASSED))
    manifest.add(record("export", RunStatus.FAILED_HARNESS))
    assert manifest.status is RunStatus.FAILED_HARNESS


def test_stages_that_were_never_reached_are_listed_not_dropped():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(record("train", RunStatus.ERROR))
    manifest.not_reached.append({"name": "export", "reason": "train reported error"})
    data = manifest.as_dict()
    assert [s["name"] for s in data["not_reached"]] == ["export"]
    # Otherwise a run that stopped at stage one reads as a one-stage job.
    assert data["status"] == "error"


def test_a_single_stage_manifest_marks_what_it_cannot_know():
    manifest = RunManifest.for_stage(record("verify", RunStatus.PASSED), run_id="verify-1")
    data = manifest.as_dict()
    assert data["scope"] == "stage"
    assert data["spec_hash"]["available"] is False
    assert "without a job spec" in data["spec_hash"]["reason"]
    assert data["spec"]["available"] is False


def test_a_single_stage_manifest_keeps_a_spec_when_it_has_one():
    manifest = RunManifest.for_stage(
        record("export", RunStatus.PASSED),
        run_id="r1",
        spec_hash="abc123",
        environments={"export": "id-1"},
        spec={"schema": "litetune.spec/1"},
    )
    data = manifest.as_dict()
    assert data["spec_hash"] == "abc123"
    assert data["environments"] == {"export": "id-1"}


def test_an_input_without_a_content_hash_says_so_in_the_manifest():
    from litetune.manifest import InputRecord

    row = InputRecord(name="model", locator="./their-model.litertlm").as_dict()
    assert row["content_hash"]["available"] is False
    assert "cached" in row["content_hash"]["reason"]


def test_artifacts_carry_their_size_and_content_hash():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(
        record(
            "export",
            RunStatus.PASSED,
            artifacts=(
                ArtifactRecord(
                    name="model.litertlm",
                    key="runs/r1/export/model.litertlm",
                    bytes=285_577_392,
                    content_hash="sha256:" + "f" * 64,
                    stage="export",
                ),
            ),
        )
    )
    artifact = manifest.as_dict()["stages"][0]["artifacts"][0]
    assert artifact["bytes"] == 285_577_392
    assert artifact["content_hash"].startswith("sha256:")


def test_captured_output_is_bounded_and_says_when_it_was_cut():
    from litetune.manifest import MAX_CAPTURED_OUTPUT

    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(record("export", RunStatus.ERROR, output="x" * (MAX_CAPTURED_OUTPUT + 500)))
    stage = manifest.as_dict()["stages"][0]
    assert len(stage["output"]) == MAX_CAPTURED_OUTPUT
    assert stage["output_truncated"] is True


def test_the_workspace_is_recorded_so_a_failed_run_can_be_found():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(record("export", RunStatus.ERROR, workspace="/tmp/litetune-export-1"))
    assert manifest.as_dict()["stages"][0]["workspace"] == "/tmp/litetune-export-1"


def test_a_stage_that_did_not_run_has_no_workspace_and_says_so():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(record("export", RunStatus.PASSED))
    assert manifest.as_dict()["stages"][0]["workspace"]["available"] is False


# -- serialisation ---------------------------------------------------------


def test_the_manifest_is_json_and_round_trips(tmp_path):
    manifest = RunManifest(run_id="r1", spec_hash="abc", environments={"export": "id-1"})
    manifest.add(record("train", RunStatus.PASSED, metrics={"loss": 0.31}))
    manifest.limitation("measured on CPU; users run on a phone")
    storage = LocalStorage(tmp_path)
    key = manifest.write(storage)
    reloaded = json.loads(storage.read_text(key))
    assert key == "runs/r1/manifest.json"
    assert reloaded["status"] == "passed"
    assert reloaded["stages"][0]["metrics"] == {"loss": 0.31}
    assert reloaded["limitations"] == ["measured on CPU; users run on a phone"]


def test_an_unavailable_field_survives_serialisation():
    manifest = RunManifest(run_id="r1", spec_hash=Unavailable("no spec"))
    json.loads(manifest.as_json())


def test_a_limitation_is_recorded_once():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.limitation("same note")
    manifest.limitation("same note")
    assert manifest.limitations == ["same note"]


def test_summarise_names_every_stage_and_every_gap():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    manifest.add(record("train", RunStatus.PASSED, cache=CacheOutcome.HIT, detail="reused"))
    manifest.add(record("export", RunStatus.ERROR, detail="exporter raised"))
    manifest.not_reached.append({"name": "measure", "reason": "export reported error"})
    lines = "\n".join(summarise(manifest))
    assert "train: passed (hit)" in lines
    assert "export: error" in lines
    assert "measure: not reached" in lines


def test_looking_up_an_absent_stage_raises_rather_than_returning_none():
    manifest = RunManifest(run_id="r1", spec_hash="abc")
    with pytest.raises(KeyError):
        manifest.stage("export")


def test_the_package_and_the_packaging_agree_on_the_version():
    """One number, one place -- and this actually compares them.

    Read with `re`, not `tomllib`: `tomllib` is 3.11+ and this project declares
    3.10, so the first version of this test made CI red on the floor it exists
    to protect. A metadata test that cannot run on the declared minimum is a
    claim about the minimum, not a check of it.
    """
    import re
    from pathlib import Path

    from litetune._version import __version__

    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    # Structurally where the file is parseable, and by line where it is not:
    # `tomllib` is 3.11+ and this project declares 3.10, so the regexes are the
    # floor's version of the same three assertions rather than a weaker
    # substitute chosen for convenience. The regex form cannot tell which table
    # a key belongs to, which is exactly what the structural form checks.
    try:
        import tomllib
    except ModuleNotFoundError:
        assert re.search(r'^dynamic = \["version"\]', pyproject, re.M)
        assert re.search(r'^path = "src/litetune/_version\.py"', pyproject, re.M)
        assert not re.search(r'^version = "', pyproject, re.M)
    else:
        config = tomllib.loads(pyproject)
        assert config["project"]["dynamic"] == ["version"]
        assert "version" not in config["project"]
        assert config["tool"]["hatch"]["version"]["path"] == "src/litetune/_version.py"

    # And the module's value is what a manifest records.
    from litetune.manifest import _tool_version

    assert _tool_version() == __version__
