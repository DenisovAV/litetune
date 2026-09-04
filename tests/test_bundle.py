"""The deliverable: model + declarations + contract + report, and never less.

No network, no accelerator, no model load. The "model" here is a few bytes on
disk, because what is under test is what travels with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litetune import envs
from litetune.bundle import (
    CONTRACT_NAME,
    DECLARATIONS_NAME,
    MANIFEST_NAME,
    MEASUREMENT_POINTS,
    REPORT_NAME,
    BundleError,
    BundleRequest,
    Contract,
    ContractError,
    MissingRenderingMode,
    build_bundle,
    versions_from,
)
from litetune.checks import Outcome
from litetune.evaluate import PromptMode
from litetune.events import EventStream
from litetune.manifest import CacheOutcome, RunManifest, RunStatus, StageRecord
from litetune.storage import hash_file

REVISION = "0123456789abcdef0123456789abcdef01234567"


def a_contract(**overrides) -> Contract:
    params = {
        "prompt_mode": PromptMode.PRERENDERED,
        "established_against": {"litert-lm": "0.16.1", "transformers": "5.16.1"},
        "base_model": "google/functiongemma-270m-it",
        "base_model_revision": REVISION,
    }
    params.update(overrides)
    # A heterogeneous kwargs factory: the checker cannot narrow a `dict[str,
    # Collection[str]]` splat to seven differently-typed parameters, and
    # restructuring the helper to satisfy it would make every test that uses it
    # longer for no gain in what is actually verified.
    return Contract(**params)  # type: ignore[arg-type]


@pytest.fixture
def model_file(tmp_path) -> Path:
    path = tmp_path / "source" / "model.litertlm"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 4096)
    return path


@pytest.fixture
def declarations(tmp_path) -> Path:
    path = tmp_path / "source" / "tools.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {"name": "change_background_color", "parameters": {"color": "string"}},
                {"name": "open_app", "parameters": {"app": "string"}},
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def request_for(tmp_path, model_file, declarations):
    def _build(**overrides) -> BundleRequest:
        params = {
            "output_dir": tmp_path / "bundle",
            "model": model_file,
            "declarations": declarations,
            "contract": a_contract(),
        }
        params.update(overrides)
        return BundleRequest(**params)

    return _build


def all_measurements() -> dict:
    return {name: {"exact_match": {"value": 0.87, "n": 640}} for name in MEASUREMENT_POINTS}


def check_named(result, name):
    return next(c for c in result.checks.checks if c.name == name)


# ---------------------------------------------------------------------------
# The rendering mode has no default
# ---------------------------------------------------------------------------


def test_a_contract_with_no_rendering_mode_is_an_error():
    # A runtime cannot infer which of the two conventions a model expects, and
    # the wrong one produces a fluent wrong answer rather than an error. So
    # there is nothing to default to.
    with pytest.raises(MissingRenderingMode):
        a_contract(prompt_mode=None)


def test_reading_a_contract_without_a_mode_is_an_error():
    with pytest.raises(MissingRenderingMode) as exc:
        Contract.read(
            {
                "established_against": {"litert-lm": "0.16.1"},
                "base_model": "m",
                "base_model_revision": REVISION,
            }
        )

    assert "prompt_mode" in str(exc.value)
    assert "runtime_rendered" in str(exc.value) and "prerendered" in str(exc.value)


def test_an_unrecognised_mode_is_an_error_not_a_fallback():
    with pytest.raises(MissingRenderingMode):
        Contract.read({"prompt_mode": "chatml", "established_against": {"x": "1"}})
    with pytest.raises(MissingRenderingMode):
        a_contract(prompt_mode="runtime_rendered")  # the string, not the enum


def test_a_mode_recorded_against_no_versions_is_an_error():
    # Which prompt a runtime renders is a property of that runtime's release, so
    # a mode established against nothing is a claim that resolves differently
    # over time -- the failure envs.StageEnv refuses unpinned requirements for.
    with pytest.raises(ContractError):
        a_contract(established_against={})


def test_a_contract_without_its_starting_weights_is_an_error():
    with pytest.raises(ContractError):
        a_contract(base_model_revision="")


def test_both_modes_round_trip_and_carry_their_meaning():
    for mode in PromptMode:
        contract = a_contract(prompt_mode=mode)
        record = contract.as_dict()
        assert record["prompt_mode"] == mode.value
        assert Contract.read(record).prompt_mode is mode
    assert (
        "--no-template"
        in a_contract(prompt_mode=PromptMode.PRERENDERED).as_dict()["prompt_mode_meaning"]
    )


def test_versions_from_reads_the_pins_a_mode_was_established_against():
    versions = versions_from(envs.RUNTIME, envs.TRAIN)

    assert versions["litert-lm"] == "0.16.1"
    assert versions["transformers"] == "5.16.1"


# ---------------------------------------------------------------------------
# A bundle is four things
# ---------------------------------------------------------------------------


def test_a_bundle_is_model_declarations_contract_and_report(request_for):
    result = build_bundle(request_for(status=RunStatus.PASSED, measurements=all_measurements()))

    root = result.request.output_dir
    assert (root / "model.litertlm").is_file()
    assert (root / DECLARATIONS_NAME).is_file()
    assert (root / CONTRACT_NAME).is_file()
    assert (root / REPORT_NAME).is_file()
    assert (root / MANIFEST_NAME).is_file()
    assert result.complete is True
    assert result.status is RunStatus.PASSED
    names = {m.name for m in result.members}
    assert {"model.litertlm", DECLARATIONS_NAME, CONTRACT_NAME, MANIFEST_NAME} <= names


def test_the_bundle_owns_its_copy_of_the_model(request_for, model_file):
    result = build_bundle(request_for())

    copied = result.request.output_dir / "model.litertlm"
    assert copied.is_file() and copied != model_file
    # Identified by content: a bundle pointing outside itself is a bundle whose
    # model can be swapped without the bundle changing.
    member = next(m for m in result.members if m.name == "model.litertlm")
    assert member.content_sha256 == hash_file(copied)


def test_a_checkpoint_directory_is_copied_wholesale(tmp_path, request_for):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"\0" * 64)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")

    result = build_bundle(request_for(model=checkpoint))

    assert (result.request.output_dir / "model" / "model.safetensors").is_file()
    assert {"model/model.safetensors", "model/config.json"} <= {m.name for m in result.members}


def test_the_contract_travels_in_the_bundle(request_for):
    result = build_bundle(request_for())

    recorded = json.loads((result.request.output_dir / CONTRACT_NAME).read_text(encoding="utf-8"))
    assert Contract.read(recorded).prompt_mode is PromptMode.PRERENDERED
    assert recorded["established_against"]["litert-lm"] == "0.16.1"


def test_a_missing_model_is_a_failed_check_not_an_exception(tmp_path, request_for):
    result = build_bundle(request_for(model=tmp_path / "gone.litertlm"))

    assert check_named(result, "model included").outcome is Outcome.FAILED
    assert result.complete is False
    # litetune could not assemble a deliverable; that says nothing about the
    # model, so it is a harness result rather than a verdict.
    assert result.status is RunStatus.FAILED_HARNESS


def test_missing_declarations_are_a_failed_check(tmp_path, request_for):
    result = build_bundle(request_for(declarations=tmp_path / "absent.json"))

    check = check_named(result, "declarations included")
    assert check.outcome is Outcome.FAILED
    assert "no record of the tools" in check.detail


def test_declarations_that_do_not_parse_are_shipped_and_rejected(tmp_path, request_for):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    result = build_bundle(request_for(declarations=broken))

    check = check_named(result, "declarations included")
    assert check.outcome is Outcome.FAILED
    # Copied anyway, so whoever has to fix it can see what was there.
    assert (result.request.output_dir / DECLARATIONS_NAME).is_file()


def test_declarations_that_are_not_the_ones_the_contract_names_fail(request_for, declarations):
    contract = a_contract(declarations_sha256="f" * 64)

    result = build_bundle(request_for(contract=contract))

    check = check_named(result, "declarations included")
    assert check.outcome is Outcome.FAILED
    assert "different tool list" in check.detail


def test_matching_declarations_pass_the_hash_check(request_for, declarations):
    contract = a_contract(declarations_sha256=hash_file(declarations))

    result = build_bundle(request_for(contract=contract))

    assert check_named(result, "declarations included").outcome is Outcome.PASSED


# ---------------------------------------------------------------------------
# A non-passing run still bundles
# ---------------------------------------------------------------------------


def test_a_failed_run_still_produces_a_manifest_and_a_report(tmp_path, request_for):
    manifest = RunManifest(run_id="run-1", spec_hash="abc123")
    manifest.add(
        StageRecord(
            name="export",
            status=RunStatus.FAILED_GATE,
            cache=CacheOutcome.MISS,
            cache_key="k",
            env_identity="e",
            detail="conversion cost 0.09 exceeds the 0.05 threshold",
        )
    )

    result = build_bundle(
        request_for(
            status=RunStatus.FAILED_GATE,
            manifest=manifest,
            measurements={"tuned_converted": {"exact_match": {"value": 0.61}}},
        )
    )

    root = result.request.output_dir
    assert (root / MANIFEST_NAME).is_file()
    assert (root / REPORT_NAME).is_file()
    recorded = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert recorded["status"] == "failed_gate"
    assert recorded["stages"][0]["detail"].startswith("conversion cost")
    report = json.loads((root / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["status"] == "failed_gate"
    assert report["verified"] is False


def test_the_report_names_the_measurements_that_were_not_made(request_for):
    result = build_bundle(
        request_for(measurements={"tuned_converted": {"exact_match": {"value": 0.61}}})
    )

    assert result.missing_measurements == ["base_float", "tuned_float"]
    # Three-valued, and deliberately outside the packaging CheckSet: an
    # incomplete measurement is a fact about the run, not a bundle litetune
    # failed to assemble.
    assert result.coverage.outcome is Outcome.UNCHECKED
    assert "measurements present" not in {c.name for c in result.checks.checks}
    report = json.loads((result.request.output_dir / REPORT_NAME).read_text(encoding="utf-8"))
    not_made = {entry["point"] for entry in report["measurements_not_made"]}
    assert not_made == {"base_float", "tuned_float"}
    # Every one of them says what it was for, so a reader knows which
    # attribution is missing rather than only that something is.
    assert all(entry["reason"] for entry in report["measurements_not_made"])
    assert any("which stage to fix" in text for text in result.limitations)


def test_all_three_points_present_makes_the_attribution_separable(request_for):
    result = build_bundle(request_for(measurements=all_measurements()))

    assert result.missing_measurements == []
    assert result.coverage.outcome is Outcome.PASSED
    assert result.complete is True


def test_a_bundle_with_no_run_manifest_synthesises_one_and_says_so(request_for):
    result = build_bundle(request_for())

    recorded = json.loads((result.request.output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    # A field that could not be known is marked, not omitted.
    assert recorded["spec_hash"]["available"] is False
    assert recorded["stages"] == []
    assert any("no run manifest was supplied" in text for text in result.limitations)


def test_an_unknown_measurement_point_is_refused(request_for):
    with pytest.raises(BundleError):
        request_for(measurements={"tuned_gpu": {}})


# ---------------------------------------------------------------------------
# Bundling establishes nothing
# ---------------------------------------------------------------------------


def test_a_bundle_never_reports_itself_as_verified(request_for):
    result = build_bundle(request_for(status=RunStatus.PASSED, measurements=all_measurements()))

    assert result.verified is False
    report = json.loads((result.request.output_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["verified"] is False
    with pytest.raises(AttributeError):
        result.verified = True


def test_the_run_status_is_carried_in_not_decided_here(request_for):
    # Bundling copies files; it cannot re-measure anything, so a bundle that
    # decided its own verdict would be deciding it from the fact that a copy
    # succeeded.
    for status in (RunStatus.PASSED, RunStatus.UNMEASURED, RunStatus.INCONCLUSIVE):
        result = build_bundle(request_for(status=status))
        assert result.status is status


def test_nothing_is_printed_and_everything_is_an_event(request_for, capsys):
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    build_bundle(request_for(measurements=all_measurements()), events=events)

    assert capsys.readouterr().out == ""
    kinds = [event.kind for event in seen]
    assert kinds[0] == "stage_started" and kinds[-1] == "stage_finished"
    names = {e.data.get("name") for e in seen if e.kind == "check"}
    assert {"model included", "declarations included", "contract recorded"} <= names
    assert seen[0].data["prompt_mode"] == PromptMode.PRERENDERED.value


def test_a_rerun_does_not_inherit_the_previous_checkpoint(tmp_path):
    """`dirs_exist_ok=True` merged the new checkpoint over the old one.

    Files the new one no longer has survived, were hashed, and were reported as
    members — a manifest accurate about a checkpoint that never existed.
    """
    from litetune.bundle import _copy_model

    source = tmp_path / "ckpt"
    source.mkdir()
    (source / "model.safetensors").write_text("v2", encoding="utf-8")

    out = tmp_path / "bundle"
    out.mkdir()
    stale = out / "model" / "removed_in_v2.bin"
    stale.parent.mkdir(parents=True)
    stale.write_text("v1", encoding="utf-8")

    destination = _copy_model(source, out)

    assert not stale.exists()
    assert sorted(p.name for p in destination.iterdir()) == ["model.safetensors"]


def test_a_bundle_cannot_be_rebuilt_from_its_own_model_directory(tmp_path):
    """Refusing overlap, because the replace is destructive.

    `_copy_model` removes the destination before copying, so
    `--model ./b/model --output-dir ./b` -- rebuilding a bundle in place, a
    plausible command -- deleted the source before reading it. The refusal has
    to come first, and it has to be a refusal rather than a silent merge.
    """
    from litetune.bundle import BundleError, _copy_model

    out = tmp_path / "bundle"
    (out / "model").mkdir(parents=True)
    (out / "model" / "weights.bin").write_text("the only copy", encoding="utf-8")

    with pytest.raises(BundleError, match="overlaps the bundle directory"):
        _copy_model(out / "model", out)

    assert (out / "model" / "weights.bin").read_text(encoding="utf-8") == "the only copy"


def test_overlap_is_refused_in_the_other_direction_too(tmp_path):
    """The first version of the check only looked one way.

    It rejected `--model ./b/model --output-dir ./b` and still accepted
    `--model ./b --output-dir ./b`, which copied `./b/model` into staging and
    then deleted it -- the same data loss, through the fix for it.
    """
    from litetune.bundle import BundleError, _copy_model

    out = tmp_path / "bundle"
    (out / "model").mkdir(parents=True)
    (out / "model" / "weights.bin").write_text("the only copy", encoding="utf-8")

    with pytest.raises(BundleError, match="overlaps the bundle directory"):
        _copy_model(out, out)

    assert (out / "model" / "weights.bin").read_text(encoding="utf-8") == "the only copy"


def test_a_failed_swap_restores_the_previous_model(tmp_path, monkeypatch):
    """Removing the destination before the rename left no model when it raised."""
    from pathlib import Path

    from litetune.bundle import _copy_model

    out = tmp_path / "bundle"
    (out / "model").mkdir(parents=True)
    (out / "model" / "v1.bin").write_text("previous", encoding="utf-8")
    source = tmp_path / "ckpt"
    source.mkdir()
    (source / "v2.bin").write_text("incoming", encoding="utf-8")

    real_rename = Path.rename
    failed = []

    def fail_the_swap(self, target):
        # Only the first swap, so the restore -- which uses the same operation --
        # can run. If rename is broken outright the restore cannot work either,
        # and that case raises a BundleError naming where the checkpoint is.
        if Path(target).name == "model" and not failed:
            failed.append(True)
            raise OSError("rename failed")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_the_swap)

    with pytest.raises(OSError):
        _copy_model(source, out)

    assert (out / "model" / "v1.bin").read_text(encoding="utf-8") == "previous"


def test_a_failed_copy_leaves_the_previous_model_in_place(tmp_path, monkeypatch):
    """Staged and swapped, so a copy that dies partway is not a lost bundle."""
    import shutil

    from litetune.bundle import _copy_model

    out = tmp_path / "bundle"
    (out / "model").mkdir(parents=True)
    (out / "model" / "v1.bin").write_text("previous", encoding="utf-8")
    source = tmp_path / "ckpt"
    source.mkdir()
    (source / "v2.bin").write_text("incoming", encoding="utf-8")

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copytree", explode)

    with pytest.raises(OSError):
        _copy_model(source, out)

    assert (out / "model" / "v1.bin").read_text(encoding="utf-8") == "previous"


def test_when_the_restore_also_fails_the_checkpoint_is_where_the_message_says(
    tmp_path, monkeypatch
):
    """The branch that raises was the one with no test, and it lost the data.

    A `finally` removed the previous copy on the way out of the very error that
    told the operator it was "intact ... and must be moved back by hand". The
    invariant is that nothing is removed until a copy is known to exist
    elsewhere, so this path must not touch it at all.
    """
    from pathlib import Path

    from litetune.bundle import BundleError, _copy_model

    out = tmp_path / "bundle"
    (out / "model").mkdir(parents=True)
    (out / "model" / "v1.bin").write_text("the previous checkpoint", encoding="utf-8")
    source = tmp_path / "ckpt"
    source.mkdir()
    (source / "v2.bin").write_text("incoming", encoding="utf-8")

    real_rename = Path.rename

    def rename_is_broken(self, target):
        if Path(target).name == "model":
            raise OSError("rename broken")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", rename_is_broken)

    with pytest.raises(BundleError, match="must be moved back by hand") as raised:
        _copy_model(source, out)

    named = [word for word in str(raised.value).split() if ".model.previous." in word]
    assert named, "the message must name where the checkpoint is"
    recovered = Path(named[0])
    assert (recovered / "v1.bin").read_text(encoding="utf-8") == "the previous checkpoint"
    assert (source / "v2.bin").exists()


def test_a_scratch_symlink_does_not_survive_the_pre_clean(tmp_path):
    """`is_dir()` is True for a symlink to a directory, and `rmtree` refuses it.

    So the pre-clean silently left the link in place and the copy that follows
    failed on it. Removing the link must never touch what it points at.
    """
    from litetune.bundle import _clear

    target = tmp_path / "real"
    target.mkdir()
    (target / "keep.bin").write_text("live data", encoding="utf-8")

    to_a_directory = tmp_path / "link"
    to_a_directory.symlink_to(target)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent")

    _clear(to_a_directory)
    _clear(dangling)

    assert not to_a_directory.is_symlink()
    assert not dangling.is_symlink()
    assert (target / "keep.bin").read_text(encoding="utf-8") == "live data"


def test_a_rebuild_with_a_differently_named_model_leaves_no_stale_artifact(tmp_path):
    """Found by walking the CLI, not by a review.

    A single-file artifact keeps its own name, so the directory branch's
    "replace, never merge" did not apply: rebuilding with `model2.litertlm`
    left `model.litertlm` beside it — in the bundle, absent from `report.json`'s
    member list, in a module whose opening claim is that its four files travel
    together and are identified by content.
    """
    import json

    from litetune.bundle import BundleRequest, build_bundle
    from litetune.manifest import RunStatus

    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")
    out = tmp_path / "bundle"

    for name in ("model.litertlm", "model2.litertlm"):
        artifact = tmp_path / name
        artifact.write_text(name, encoding="utf-8")
        build_bundle(
            BundleRequest(
                output_dir=out,
                model=artifact,
                declarations=declarations,
                contract=a_contract(),
                status=RunStatus.INCONCLUSIVE,
            )
        )

    recorded = {m["name"] for m in json.loads((out / "report.json").read_text())["members"]}
    on_disk = {p.name for p in out.iterdir()} - {"report.json"}

    assert on_disk == recorded
    assert "model.litertlm" not in on_disk


def test_a_recorded_member_cannot_delete_outside_the_bundle(tmp_path):
    """`report.json` is the least trustworthy input in this module, not the most.

    The stale-artifact fix built paths by joining names read out of a report on
    disk and handed them to `_clear`. Nothing resolved them, so `../victim.txt`
    escaped `--output-dir` — and litetune deleted a file it had never written,
    then exited with its documented status. `_copy_model` sixty lines above
    refuses an overlapping `--model` for exactly this reason.
    """
    from litetune.bundle import _previously_recorded_model

    out = tmp_path / "bundle"
    out.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("important", encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps({"members": [{"name": "../victim.txt"}, {"name": "model.litertlm"}]}),
        encoding="utf-8",
    )

    recorded = _previously_recorded_model(out)

    assert all(out.resolve() in p.resolve().parents for p in recorded)
    assert victim.exists()


def test_an_unreadable_report_is_not_silently_read_as_no_previous_model(tmp_path, caplog):
    """The fallback said "no previous model" for a report it could not parse.

    That is the stale-artifact bug the function exists to fix, returning with no
    record — in the module whose `_clear` docstring had just named a silent
    swallow as the one place this package did that.
    """
    from litetune.bundle import _previously_recorded_model

    out = tmp_path / "bundle"
    out.mkdir()
    (out / "report.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level("WARNING"):
        assert _previously_recorded_model(out) == []

    assert "report.json" in caplog.text


@pytest.mark.parametrize(
    "first,second",
    [("file", "dir"), ("dir", "file"), ("file", "file"), ("dir", "dir")],
)
def test_no_rebuild_shape_leaves_a_stale_model(tmp_path, first, second):
    """ "Replace, never merge" held for file->file and no other shape.

    A directory rebuilt as a file left `model/` behind; a file rebuilt as a
    directory left the old `.litertlm` beside it. Both are in the bundle and
    absent from `report.json`'s member list -- the one thing this module says
    cannot happen.
    """
    import json

    from litetune.bundle import BundleRequest, build_bundle
    from litetune.manifest import RunStatus

    as_file = tmp_path / "a.litertlm"
    as_file.write_text("file model", encoding="utf-8")
    as_dir = tmp_path / "ckpt"
    as_dir.mkdir()
    (as_dir / "weights.bin").write_text("dir model", encoding="utf-8")
    shapes = {"file": as_file, "dir": as_dir}

    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")
    out = tmp_path / "bundle"

    for shape in (first, second):
        build_bundle(
            BundleRequest(
                output_dir=out,
                model=shapes[shape],
                declarations=declarations,
                contract=a_contract(),
                status=RunStatus.INCONCLUSIVE,
            )
        )

    recorded = {
        m["name"].split("/")[0]
        for m in json.loads((out / "report.json").read_text(encoding="utf-8"))["members"]
    }
    on_disk = {p.name for p in out.iterdir()} - {"report.json"}

    assert on_disk == recorded, f"{first}->{second} left {sorted(on_disk - recorded)}"


def test_the_adapter_travels_with_the_bundle(tmp_path):
    """`merge_and_unload()` is one-way, so a bundle without it loses it.

    `convert` is pointed at the merged checkpoint; nothing recovers the LoRA
    weights from it. Six rounds of code review missed this because it is a
    requirement in the result-bundle spec and not a defect in any file.
    """
    import json

    from litetune.bundle import BundleRequest, build_bundle
    from litetune.manifest import RunStatus

    model = tmp_path / "model.litertlm"
    model.write_text("{}", encoding="utf-8")
    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_text("lora", encoding="utf-8")

    out = tmp_path / "bundle"
    build_bundle(
        BundleRequest(
            output_dir=out,
            model=model,
            adapter=adapter,
            declarations=declarations,
            contract=a_contract(),
            status=RunStatus.INCONCLUSIVE,
        )
    )

    members = {m["name"] for m in json.loads((out / "report.json").read_text())["members"]}
    assert "adapter/adapter_model.safetensors" in members
    assert (out / "adapter" / "adapter_model.safetensors").read_text() == "lora"


def test_a_named_adapter_that_is_absent_is_a_failed_check_not_a_refusal(tmp_path):
    """The bundle is still worth writing; the report is where the gap is read."""
    from litetune.bundle import BundleRequest, build_bundle
    from litetune.checks import Outcome
    from litetune.manifest import RunStatus

    model = tmp_path / "model.litertlm"
    model.write_text("{}", encoding="utf-8")
    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")

    result = build_bundle(
        BundleRequest(
            output_dir=tmp_path / "bundle",
            model=model,
            adapter=tmp_path / "absent",
            declarations=declarations,
            contract=a_contract(),
            status=RunStatus.INCONCLUSIVE,
        )
    )

    recorded = {c["name"]: c for c in result.checks.as_dict()["checks"]}
    assert recorded["adapter included"]["outcome"] == Outcome.FAILED.value
    assert (tmp_path / "bundle" / "report.json").is_file()


def test_a_rebuild_without_an_adapter_removes_the_previous_one(tmp_path):
    """ "Replace, never merge" covers the adapter too — and pins an ordering.

    `_copy_model` clears what the last build recorded; `_copy_adapter` runs
    after it and re-copies. Swapping the two would copy an adapter and then
    delete it, and nothing else in the suite would notice.
    """
    import json

    from litetune.bundle import BundleRequest, build_bundle
    from litetune.manifest import RunStatus

    model = tmp_path / "model.litertlm"
    model.write_text("{}", encoding="utf-8")
    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_text("lora", encoding="utf-8")
    out = tmp_path / "bundle"

    def build(with_adapter: bool) -> set[str]:
        build_bundle(
            BundleRequest(
                output_dir=out,
                model=model,
                declarations=declarations,
                contract=a_contract(),
                status=RunStatus.INCONCLUSIVE,
                adapter=adapter if with_adapter else None,
            )
        )
        recorded = {
            m["name"].split("/")[0]
            for m in json.loads((out / "report.json").read_text(encoding="utf-8"))["members"]
        }
        on_disk = {p.name for p in out.iterdir()} - {"report.json"}
        assert on_disk == recorded, f"stale: {sorted(on_disk - recorded)}"
        return on_disk

    assert "adapter" in build(with_adapter=True)
    assert "adapter" in build(with_adapter=True)
    assert "adapter" not in build(with_adapter=False)


def test_an_adapter_inside_the_output_directory_is_refused_not_deleted(tmp_path):
    """`tune` writes the adapter to `<output>/adapter`, so this is the likely command.

    `_copy_adapter` was written twelve lines below the comment explaining why
    clear-then-copy loses data, and did clear-then-copy. Bundling with
    `--adapter out/adapter --output-dir out` removed the only copy of the LoRA
    weights and then reported that the adapter "could not be copied" -- the
    weights this function calls unrecoverable, destroyed by the function that
    exists to preserve them.
    """
    from litetune.bundle import BundleRequest, BundleResult, _copy_adapter

    out = tmp_path / "bundle"
    (out / "adapter").mkdir(parents=True)
    weights = out / "adapter" / "adapter_model.safetensors"
    weights.write_bytes(b"the only copy")
    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")

    request = BundleRequest(
        output_dir=out,
        model=out / "absent",
        declarations=declarations,
        contract={},
        adapter=out / "adapter",
    )
    check = _copy_adapter(request, BundleResult(request=request, checks=[]))

    assert weights.read_bytes() == b"the only copy", "the adapter must survive the refusal"
    assert check is not None
    assert check.outcome.value == "failed"
    assert "overlaps" in (check.detail or "")


def test_an_adapter_outside_the_output_directory_is_copied_and_left_intact(tmp_path):
    from litetune.bundle import BundleRequest, BundleResult, _copy_adapter

    source = tmp_path / "run" / "adapter"
    source.mkdir(parents=True)
    (source / "adapter_model.safetensors").write_bytes(b"weights")
    (source / "adapter_config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "bundle"
    out.mkdir()
    declarations = tmp_path / "declarations.json"
    declarations.write_text("[]", encoding="utf-8")

    request = BundleRequest(
        output_dir=out,
        model=out / "absent",
        declarations=declarations,
        contract={},
        adapter=source,
    )
    check = _copy_adapter(request, BundleResult(request=request, checks=[]))

    assert check is not None and check.outcome.value == "passed"
    assert (out / "adapter" / "adapter_model.safetensors").read_bytes() == b"weights"
    assert (source / "adapter_model.safetensors").exists(), "the source must not be consumed"


def test_the_wire_convention_is_recorded_or_declared_unknown():
    """`declarations_sha256` cannot carry this, which is why the field exists.

    The same declarations rendered in declaration order and in dictsort hash
    identically, so without this a bundle has no way to say which one it was
    built under. Measured on the base checkpoint twice, on disjoint samples:
    choosing wrong costs a resolved 0.019-0.036 exact match, and it fails as a
    *wrong argument value* -- argument dicts compare without regard to key
    order, so a merely reordered call would score the same.
    """
    from litetune.bundle import Contract, PromptMode, WireConvention

    versions = {"litert-lm": "0.16.1"}
    common = dict(
        prompt_mode=PromptMode.PRERENDERED,
        established_against=versions,
        base_model="google/functiongemma-270m-it",
        base_model_revision="a" * 40,
    )

    recorded = Contract(wire_convention=WireConvention.TEMPLATE_DICTSORT, **common).as_dict()
    assert recorded["wire_convention"] == "template_dictsort"
    assert "sorted by name" in recorded["wire_convention_meaning"]

    # Unset is "unrecorded", never a quiet default: a default here is a guess
    # with a measured, resolved cost.
    unknown = Contract(**common).as_dict()
    assert unknown["wire_convention"] is None
    assert "unrecorded" in unknown["wire_convention_meaning"]

    # And it survives the round trip, so a manifest can carry it forward.
    assert Contract.read(recorded).wire_convention is WireConvention.TEMPLATE_DICTSORT
    assert Contract.read(unknown).wire_convention is None
