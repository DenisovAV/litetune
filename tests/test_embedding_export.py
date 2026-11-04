"""`convert` of an embedding checkpoint: dispatch, inputs, refusals, the report.

Nothing is downloaded and nothing is grafted here. `FakeHub`, this file's own,
stands in for `StageEnv.run` on the download script and writes small files in
place of Google's files and the base checkpoint; `FakeGraft` stands in for
`embedding_convert.graft` (whose own tests run the real script on a synthetic
flatbuffer) and writes a file where the graft would; `FakeSplice` stands in for
`embedding_vision.splice` (test_embedding_vision.py runs the real script) and
writes a file where the splice would.
"""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned

from litetune import embedding_export, envs
from litetune.checks import Outcome
from litetune.cli import main
from litetune.decisions import Decision, Option
from litetune.embedding_convert import (
    BASE_MODEL,
    BASE_REVISION,
    GOOGLE_FILE,
    GOOGLE_REPO,
    EmbeddingConvertError,
    GraftResult,
)
from litetune.embedding_export import (
    ARTIFACT_NAME,
    BASE_FILE,
    DEFAULT_WEIGHTS,
    EMBEDDING_CONVERT_SCHEMA,
    GOOGLE_REVISION,
    REPORT_NAME,
    EmbeddingExportError,
    EmbeddingExportRequest,
    run_embedding_export,
)
from litetune.embedding_tune import (
    DECIDER_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    IMAGE_PREFIX,
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    read_embedding_record,
    write_embedding_record,
)
from litetune.embedding_vision import VISION_FILE, VISION_REPO, VISION_REVISION, SpliceResult
from litetune.events import EventStream
from litetune.storage import hash_file


def safetensors(tensors: dict[str, tuple[str, list[int], list[float]]]) -> bytes:
    """A safetensors file of (dtype, shape, values); BF16 as the upper half of each float32."""
    header: dict[str, Any] = {}
    data = b""
    for name, (dtype, shape, values) in tensors.items():
        packed = struct.pack(f"<{len(values)}f", *values)
        if dtype == "BF16":
            packed = b"".join(packed[i + 2 : i + 4] for i in range(0, len(packed), 4))
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(packed)],
        }
        data += packed
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + data


# Values a BF16 holds exactly, so the base's BF16 and a fine-tune's F32 of them agree.
FROZEN = {
    "vision_tower.encoder.weight": [1.0, -2.0],
    "embed_vision.embedding_projection.weight": [0.5, 4.0],
    "language_model.embed_tokens.weight": [3.0, -0.25, 8.0, 1.5],
}


def _weights(dtype: str, trained: float, **changed: list[float]) -> bytes:
    """Base-shaped weights: the frozen tensors (one per prefix in `changed` replaced)
    and one trained tensor of value `trained`."""
    shapes = {"language_model.embed_tokens.weight": [2, 2]}
    tensors = {
        name: (dtype, shapes.get(name, [len(v)]), changed.get(name.split(".")[0], v))
        for name, v in FROZEN.items()
    }
    tensors["language_model.layers.0.weight"] = (dtype, [1], [trained])
    return safetensors(tensors)


GOOGLE_BYTES = b"not google's file, a stand-in"
VISION_BYTES = b"not google's text-vision file, a stand-in"
BASE_BYTES = _weights("BF16", 1.0)
TUNED_BYTES = b"tuned weights"
# A checkpoint tuned on images: the frozen tensors as the base's, in float32.
TUNED_IMAGE_BYTES = _weights("F32", 1.25)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass
class FakeHub:
    """`StageEnv.run` on the download script: writes each file asked for under `root`."""

    root: Path
    fail: bool = False
    asked: list[list[dict[str, str]]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.asked.append(spec["files"])
        if self.fail:
            return subprocess.CompletedProcess(args, 1, "", "HTTPError: 401 gated repository")
        paths = []
        for f in spec["files"]:
            path = self.root / f["repo"].replace("/", "--") / f["revision"] / f["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            stand_in = {GOOGLE_FILE: GOOGLE_BYTES, VISION_FILE: VISION_BYTES}
            path.write_bytes(stand_in.get(f["filename"], BASE_BYTES))
            paths.append(str(path))
        Path(spec["out"]).write_text(
            json.dumps({"paths": paths, "huggingface_hub": "0.0-fake"}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, "", "")


@dataclass
class FakeGraft:
    """`embedding_convert.graft`: writes the output, or refuses with `cause`."""

    cause: str | None = None
    # An exception the graft is not expected to raise.
    raises: BaseException | None = None
    # (path, bytes): a file written over while the graft runs -- the original
    # checkpoint, standing for a change made beside the run.
    rewrite: tuple[Path, bytes] | None = None
    # What the graft read at the fine-tune path it was given.
    read_tuned: bytes | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, env, google, base, tuned, out_path, weights=DEFAULT_WEIGHTS, **kwargs
    ) -> GraftResult:
        self.calls.append(
            {"google": Path(google), "base": Path(base), "tuned": Path(tuned), "out": out_path}
            | {"weights": weights}
        )
        if self.raises is not None:
            raise self.raises
        if self.rewrite is not None:
            self.rewrite[0].write_bytes(self.rewrite[1])
        self.read_tuned = Path(tuned).read_bytes()
        if self.cause is not None:
            raise EmbeddingConvertError(self.cause, f"refused for {self.cause}")
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"grafted " + weights.encode())
        return GraftResult(
            output=out,
            weights=weights,
            bytes=out.stat().st_size,
            sha256="sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
            builder="0.18.0",
            parts={"embedder": {"bits": 8}, "text_encoder": {"bits": 8}},
            folds={"language_model.ple.per_layer_model_projection.weight": 0.001953125},
            norms_written=170,
            norms_skipped=("v_norm",),
            layer_scalars=24,
            report={},
        )


@pytest.fixture
def hub(monkeypatch, tmp_path) -> FakeHub:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeHub(tmp_path / "hub")
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    monkeypatch.setattr(embedding_export, "GOOGLE_SHA256", hashlib.sha256(GOOGLE_BYTES).hexdigest())
    monkeypatch.setattr(embedding_export, "BASE_SHA256", hashlib.sha256(BASE_BYTES).hexdigest())
    monkeypatch.setattr(embedding_export, "VISION_SHA256", hashlib.sha256(VISION_BYTES).hexdigest())
    return fake


@pytest.fixture
def grafter(monkeypatch) -> FakeGraft:
    fake = FakeGraft()
    monkeypatch.setattr(embedding_export, "graft", fake)
    return fake


@dataclass
class FakeSplice:
    """`embedding_vision.splice`: writes the output from the graft's file, or refuses."""

    cause: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, env, grafted, vision, out_path, **kwargs) -> SpliceResult:
        # What the splice reads, while it runs: the graft's file is still there.
        self.calls.append(
            {"grafted": Path(grafted), "grafted_bytes": Path(grafted).read_bytes()}
            | {"vision": Path(vision), "out": Path(out_path)}
        )
        if self.cause is not None:
            raise EmbeddingConvertError(self.cause, f"refused for {self.cause}")
        out = Path(out_path)
        out.write_bytes(b"spliced " + Path(grafted).read_bytes() + Path(vision).read_bytes())
        return SpliceResult(
            output=out,
            bytes=out.stat().st_size,
            sha256=hash_file(out),
            builder="0.18.0",
            sections=({"index": 2, "source": "graft"}, {"index": 4, "source": "google_vision"}),
            replaced=("TFLiteModel embedder",),
        )


@pytest.fixture
def splicer(monkeypatch) -> FakeSplice:
    fake = FakeSplice()
    monkeypatch.setattr(embedding_export, "splice", fake)
    return fake


def _decision() -> Decision:
    return Decision("choice", "Which?", (Option("a"), Option("b")))


def _record(**overrides: Any) -> EmbeddingRecord:
    fields: dict[str, Any] = {
        "base_model": EMBEDDING_MODEL,
        "base_model_revision": EMBEDDING_REVISION,
        "prompt_name": PROMPT_NAME,
        "prefix": DECIDER_PREFIX,
        "labels": ("a", "b"),
        "rows": {"train": 20, "heldout": 4},
        "steps_done": 6,
        "steps_planned": 6,
        "seconds": 1.0,
        "device": "cpu",
        "dtype": "float32",
        "head_heldout_accuracy": 0.5,
        "recipe": RECIPE.as_dict(),
        "decision": _decision().as_dict(),
        "decision_sha256": _decision().digest(),
    }
    fields.update(overrides)
    return EmbeddingRecord(**fields)


def _checkpoint(tmp_path: Path, record: EmbeddingRecord | None = None, weights=True) -> Path:
    model = tmp_path / "tuned" / "model"
    model.mkdir(parents=True)
    record = record or _record()
    if weights:
        (model / BASE_FILE).write_bytes(TUNED_IMAGE_BYTES if record.is_image else TUNED_BYTES)
    write_embedding_record(model, record)
    return model


def _convert(model: Path, out: Path, *extra: str) -> int:
    return main(["convert", "--model", str(model), "--output-dir", str(out), *extra])


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_export(monkeypatch):
    from litetune import cli

    seen: list[Any] = []

    def fake(request, events=None):
        seen.append(request)
        raise AssertionError("the LLM export ran")

    monkeypatch.setattr(cli, "run_export", fake)
    return seen


def test_an_embedding_checkpoint_is_grafted_not_exported(hub, grafter, llm_export, tmp_path):
    model = _checkpoint(tmp_path)
    assert _convert(model, tmp_path / "out") == 0
    assert llm_export == []
    call = grafter.calls[0]
    assert call["weights"] == DEFAULT_WEIGHTS == "int4"
    # A copy of the fine-tune, in a directory of the run's own beside the output.
    assert call["tuned"].name == BASE_FILE and call["tuned"].parent.name.startswith(
        ".litetune-checkpoint-"
    )
    assert call["out"] == tmp_path / "out" / DEFAULT_WEIGHTS / ARTIFACT_NAME
    assert call["google"].name == GOOGLE_FILE
    assert call["base"].name == BASE_FILE


def test_both_hub_files_are_asked_for_at_their_pinned_revisions(hub, grafter, tmp_path):
    _convert(_checkpoint(tmp_path), tmp_path / "out")
    assert hub.asked == [
        [
            {"repo": GOOGLE_REPO, "filename": GOOGLE_FILE, "revision": GOOGLE_REVISION},
            {"repo": BASE_MODEL, "filename": BASE_FILE, "revision": BASE_REVISION},
        ]
    ]


@pytest.mark.parametrize("weights", ["int8", "int4"])
def test_each_weight_mode_writes_its_own_directory(hub, grafter, tmp_path, weights):
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", "--weights", weights) == 0
    assert (tmp_path / "out" / weights / ARTIFACT_NAME).is_file()
    assert grafter.calls[0]["weights"] == weights


def test_an_unknown_weight_mode_is_refused_by_the_parser(hub, grafter, tmp_path):
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", "--weights", "int2") != 0
    assert grafter.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        ("--recipe", "dynamic_wi8_afp32"),
        ("--flag=--prefill_lengths=8",),
        ("--no-externalize-embedder",),
        ("--base-model", "google/embeddinggemma-2"),
    ],
)
def test_an_llm_flag_is_refused_by_name(hub, grafter, tmp_path, capsys, extra):
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", *extra) == 4
    assert grafter.calls == []
    assert extra[0].split("=")[0].replace("--no-", "--") in capsys.readouterr().err


def test_weights_for_an_llm_checkpoint_is_refused(hub, grafter, llm_export, tmp_path, capsys):
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m-it"}))
    assert _convert(llm, tmp_path / "out", "--weights", "int4") == 4
    assert llm_export == [] and grafter.calls == []
    assert "embedding checkpoint" in capsys.readouterr().err


def test_an_llm_checkpoint_still_goes_to_the_llm_export(monkeypatch, tmp_path):
    from litetune import cli

    seen: list[Any] = []

    def fake(request, events=None):
        seen.append(request)
        raise EmbeddingExportError("stop here")  # a refusal, exit 4, after dispatch

    monkeypatch.setattr(cli, "run_export", fake)
    _convert(tmp_path / "missing", tmp_path / "out", "--recipe", "dynamic_wi8_afp32")
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_lands_beside_the_file(hub, grafter, tmp_path):
    _convert(_checkpoint(tmp_path), tmp_path / "out", "--weights", "int4")
    report = json.loads((tmp_path / "out" / "int4" / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["schema"] == EMBEDDING_CONVERT_SCHEMA
    assert report["outcome"] == "passed"
    assert report["verified"] is False
    assert report["weights"] == "int4"
    assert report["prefix"] == DECIDER_PREFIX
    assert report["artifact"] == str(tmp_path / "out" / "int4" / ARTIFACT_NAME)
    assert report["graft"]["sha256"].startswith("sha256:")
    assert report["google_file"]["revision"] == GOOGLE_REVISION
    assert report["google_file"]["source"] == "downloaded"
    assert report["base_checkpoint"]["revision"] == BASE_REVISION
    assert any("has not been measured" in t for t in report["limitations"])
    # Each digest is of the file used, read from it.
    assert report["google_file"]["sha256"] == _digest(GOOGLE_BYTES)
    assert report["base_checkpoint"]["sha256"] == _digest(BASE_BYTES)
    assert report["checkpoint"]["sha256"] == _digest(TUNED_BYTES)
    assert report["checkpoint"]["decision_sha256"] == _decision().digest()


def test_the_json_output_is_the_report(hub, grafter, tmp_path, capsys):
    _convert(_checkpoint(tmp_path), tmp_path / "out", "--json")
    printed = json.loads(capsys.readouterr().out)
    assert printed["direction"] == "embedding" and printed["weights"] == DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Inputs and refusals
# ---------------------------------------------------------------------------


def _run(tmp_path: Path, **kwargs: Any):
    model = kwargs.pop("model", None) or _checkpoint(tmp_path)
    request = EmbeddingExportRequest(model=model, output_dir=tmp_path / "out", **kwargs)
    from litetune.embedding_tune import read_embedding_record

    record = read_embedding_record(model)
    assert record is not None
    return run_embedding_export(request, record)


def test_a_local_base_file_is_used_and_only_the_base_checkpoint_downloaded(hub, grafter, tmp_path):
    local = tmp_path / "google.litertlm"
    local.write_bytes(GOOGLE_BYTES)
    result = _run(tmp_path, base_litertlm=local)
    assert result.ok
    assert hub.asked == [[{"repo": BASE_MODEL, "filename": BASE_FILE, "revision": BASE_REVISION}]]
    assert grafter.calls[0]["google"] == local
    assert result.as_dict()["google_file"]["source"] == "--base-litertlm"


def test_a_local_base_file_with_another_sha256_is_a_failure(hub, grafter, tmp_path):
    local = tmp_path / "google.litertlm"
    local.write_bytes(b"someone else's file")
    result = _run(tmp_path, base_litertlm=local)
    assert result.outcome is Outcome.FAILED
    assert grafter.calls == []
    assert "not sha256:" in result.checks.checks[-1].detail
    report = json.loads(result.request.report_path.read_text(encoding="utf-8"))
    assert report["google_file"]["sha256"] == _digest(b"someone else's file")


def test_a_base_checkpoint_with_another_sha256_is_a_failure(hub, grafter, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_export, "BASE_SHA256", "0" * 64)
    result = _run(tmp_path)
    assert result.outcome is Outcome.FAILED
    assert grafter.calls == []
    last = result.checks.checks[-1]
    assert last.name == embedding_export.BASE_CHECKPOINT_CHECK
    assert f"{_digest(BASE_BYTES)}, not sha256:{'0' * 64}" in last.detail


def test_an_unexpected_graft_error_is_not_reported_as_passed(hub, grafter, tmp_path):
    grafter.raises = KeyError("parts")
    seen: list[Any] = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)
    model = _checkpoint(tmp_path)
    record = read_embedding_record(model)
    assert record is not None
    request = EmbeddingExportRequest(model=model, output_dir=tmp_path / "out")
    with pytest.raises(KeyError):
        run_embedding_export(request, record, events)
    report = json.loads(request.report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == Outcome.UNCHECKED.value
    assert report["checks"]["checks"][-1]["detail"] == "KeyError: 'parts'"
    finished = [e.data["status"] for e in seen if e.kind == "stage_finished"]
    assert finished == [Outcome.UNCHECKED.value]


def test_the_graft_reads_a_copy_and_the_report_names_the_copys_bytes(hub, grafter, tmp_path):
    # A checkpoint rewritten while the graft runs does not change what was read,
    # nor the digest `verify` binds the file to.
    model = _checkpoint(tmp_path)
    original = (model / "model.safetensors").read_bytes()
    grafter.rewrite = (model / "model.safetensors", b"another checkpoint")
    result = _run(tmp_path, model=model)
    assert (model / "model.safetensors").read_bytes() == b"another checkpoint"
    assert grafter.read_tuned == original
    assert result.outcome is Outcome.PASSED
    read = grafter.calls[0]["tuned"]
    assert read != model / "model.safetensors"
    assert result.checkpoint_sha256 == "sha256:" + hashlib.sha256(original).hexdigest()
    assert not read.exists()  # the copy goes with the run
    assert not list((tmp_path / "out" / DEFAULT_WEIGHTS).glob(".litetune-checkpoint-*"))


def test_a_missing_local_base_file_is_refused(hub, grafter, tmp_path):
    with pytest.raises(EmbeddingExportError, match="no such file"):
        _run(tmp_path, base_litertlm=tmp_path / "nowhere.litertlm")


def test_a_failed_download_is_could_not_check(hub, grafter, tmp_path):
    hub.fail = True
    result = _run(tmp_path)
    assert result.outcome is Outcome.UNCHECKED
    assert grafter.calls == []
    assert "gated repository" in result.checks.checks[-1].detail
    assert (tmp_path / "out" / DEFAULT_WEIGHTS / REPORT_NAME).is_file()


def test_a_graft_refusal_is_a_failure_naming_its_cause(hub, grafter, tmp_path):
    grafter.cause = "norm"
    result = _run(tmp_path)
    assert result.outcome is Outcome.FAILED
    assert result.checks.checks[-1].detail.startswith("norm:")
    assert result.artifact is None


def test_a_graft_that_could_not_run_is_could_not_check(hub, grafter, tmp_path):
    grafter.cause = "run"
    assert _run(tmp_path).outcome is Outcome.UNCHECKED


def test_an_existing_output_is_refused_before_anything_runs(hub, grafter, tmp_path):
    out = tmp_path / "out" / DEFAULT_WEIGHTS / ARTIFACT_NAME
    out.parent.mkdir(parents=True)
    out.write_bytes(b"earlier")
    with pytest.raises(EmbeddingExportError, match="already exists"):
        _run(tmp_path)
    assert hub.asked == [] and grafter.calls == []


def test_a_checkpoint_without_its_safetensors_is_refused(hub, grafter, tmp_path):
    with pytest.raises(EmbeddingExportError, match=f"no {BASE_FILE}"):
        _run(tmp_path, model=_checkpoint(tmp_path, weights=False))


def test_a_record_with_another_prefix_is_refused(hub, grafter, tmp_path):
    model = _checkpoint(tmp_path, _record(prefix="task: classification | query: "))
    with pytest.raises(EmbeddingExportError, match="prefix"):
        _run(tmp_path, model=model)


def test_a_record_of_another_revision_is_refused(hub, grafter, tmp_path):
    model = _checkpoint(tmp_path, _record(base_model_revision="main"))
    with pytest.raises(EmbeddingExportError, match="graft's mapping"):
        _run(tmp_path, model=model)


def test_a_malformed_embedding_record_is_refused_on_the_command_line(
    hub, grafter, tmp_path, capsys
):
    model = tmp_path / "broken"
    model.mkdir()
    (model / "litetune.json").write_text(json.dumps({"direction": "embedding", "labels": 3}))
    assert _convert(model, tmp_path / "out") == 4
    assert grafter.calls == []


def test_an_unknown_weight_mode_is_refused_by_the_request(tmp_path):
    with pytest.raises(EmbeddingExportError, match="--weights int2"):
        EmbeddingExportRequest(model=tmp_path, output_dir=tmp_path, weights="int2")


def test_a_tune_output_directory_is_pointed_at_its_model_directory(tmp_path):
    from litetune.embedding_tune import model_dir_hint

    assert model_dir_hint(tmp_path) == ""
    model = _checkpoint(tmp_path)
    assert model_dir_hint(model) == ""
    assert str(model) in model_dir_hint(model.parent)


def test_weights_on_a_tune_output_directory_name_its_model_directory(tmp_path, capsys):
    model = _checkpoint(tmp_path)
    assert _convert(model.parent, tmp_path / "out", "--weights", "int8") == 4
    assert f"{model} is" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A checkpoint tuned on images: the graft, then the splice into the 440m file
# ---------------------------------------------------------------------------


def _image_record(**overrides: Any) -> EmbeddingRecord:
    image = {
        "modality": "image",
        "vision_tokens": 70,
        "prefix": IMAGE_PREFIX,
        "soft_tokens": {"a" * 64: 63},
    }
    return _record(**(image | overrides))


def _image_checkpoint(tmp_path: Path, **overrides: Any) -> Path:
    return _checkpoint(tmp_path, _image_record(**overrides))


def _report(out: Path, weights: str) -> dict[str, Any]:
    return json.loads((out / weights / REPORT_NAME).read_text(encoding="utf-8"))


def test_an_image_checkpoint_is_grafted_then_spliced_at_int8(hub, grafter, splicer, tmp_path):
    out = tmp_path / "out"
    assert _convert(_image_checkpoint(tmp_path), out) == 0
    artifact = out / "int8" / ARTIFACT_NAME
    graft_call, splice_call = grafter.calls[0], splicer.calls[0]
    assert graft_call["weights"] == "int8"
    # The graft writes its text file into the run's scratch, not at the output...
    assert graft_call["out"] != artifact
    assert graft_call["out"].parent.name.startswith(".litetune-checkpoint-")
    # ...and the splice reads it there, with Google's text-vision file, into the output.
    assert splice_call["grafted"] == graft_call["out"]
    assert splice_call["grafted_bytes"] == b"grafted int8"
    assert splice_call["vision"].name == VISION_FILE
    assert splice_call["out"] == artifact
    # The text file goes with the scratch; the output is the splice's.
    assert not graft_call["out"].exists()
    assert sorted(p.name for p in (out / "int8").iterdir()) == sorted([ARTIFACT_NAME, REPORT_NAME])
    assert artifact.read_bytes() == b"spliced grafted int8" + VISION_BYTES


def test_an_image_checkpoint_asks_for_all_three_hub_files(hub, grafter, splicer, tmp_path):
    _convert(_image_checkpoint(tmp_path), tmp_path / "out")
    assert hub.asked == [
        [
            {"repo": GOOGLE_REPO, "filename": GOOGLE_FILE, "revision": GOOGLE_REVISION},
            {"repo": VISION_REPO, "filename": VISION_FILE, "revision": VISION_REVISION},
            {"repo": BASE_MODEL, "filename": BASE_FILE, "revision": BASE_REVISION},
        ]
    ]


def test_an_image_report_ties_to_the_spliced_file(hub, grafter, splicer, tmp_path):
    out = tmp_path / "out"
    _convert(_image_checkpoint(tmp_path, vision_tokens=140), out)
    report = _report(out, "int8")
    artifact = out / "int8" / ARTIFACT_NAME
    assert report["outcome"] == "passed"
    assert (report["modality"], report["vision_tokens"]) == ("image", 140)
    assert report["prefix"] == IMAGE_PREFIX
    assert report["artifact"] == str(artifact)
    assert report["splice"]["sha256"] == hash_file(artifact)
    assert report["splice"]["output"] == str(artifact)
    assert report["splice"]["replaced"] == ["TFLiteModel embedder"]
    # The graft's block describes the text file, which is not kept.
    assert report["graft"]["output"] is None
    assert report["graft"]["sha256"] == _digest(b"grafted int8") != report["splice"]["sha256"]
    assert report["google_vision_file"] == {
        "path": str(hub.root / VISION_REPO.replace("/", "--") / VISION_REVISION / VISION_FILE),
        "source": "downloaded",
        "repo": VISION_REPO,
        "filename": VISION_FILE,
        "revision": VISION_REVISION,
        "sha256": _digest(VISION_BYTES),
    }
    names = [c["name"] for c in report["checks"]["checks"]]
    assert names == [
        embedding_export.ENV_CHECK,
        embedding_export.BASE_FILE_CHECK,
        embedding_export.VISION_FILE_CHECK,
        embedding_export.BASE_CHECKPOINT_CHECK,
        embedding_export.FROZEN_CHECK,
        embedding_export.GRAFT_CHECK,
        embedding_export.SPLICE_CHECK,
    ]
    assert not any("int4 has not been measured" in t for t in report["limitations"])
    assert any("has not been measured" in t for t in report["limitations"])
    # What verify reads of this report ties the file to it.
    from litetune.embedding_verify import convert_facts

    facts, source = convert_facts(artifact)
    assert facts is not None and source == str(out / "int8" / REPORT_NAME)
    assert (facts.modality, facts.vision_tokens, facts.weights) == ("image", 140, "int8")
    assert facts.checkpoint_sha256 == report["checkpoint"]["sha256"]
    assert facts.decision_sha256 == _decision().digest()


@pytest.mark.parametrize(
    ("changed", "named"),
    [
        ({"language_model": [3.0, -0.25, 8.0, 1.75]}, "language_model.embed_tokens.weight"),
        ({"vision_tower": [1.0, -2.5]}, "vision_tower.encoder.weight"),
        ({"embed_vision": [0.5, 4.5]}, "embed_vision.embedding_projection.weight"),
    ],
)
def test_an_image_checkpoint_whose_frozen_tensors_changed_is_refused_before_the_graft(
    hub, grafter, splicer, tmp_path, changed, named
):
    model = _image_checkpoint(tmp_path)
    (model / BASE_FILE).write_bytes(_weights("F32", 1.25, **changed))
    result = _run(tmp_path, model=model, weights="int8")
    assert result.outcome is Outcome.FAILED
    last = result.checks.checks[-1]
    assert last.name == embedding_export.FROZEN_CHECK
    assert f"{named} (other values)" in last.detail and "1 of the tensors" in last.detail
    assert grafter.calls == [] and splicer.calls == []
    assert result.artifact is None


def test_an_image_checkpoint_that_is_not_safetensors_is_refused_by_the_frozen_check(
    hub, grafter, splicer, tmp_path
):
    model = _image_checkpoint(tmp_path)
    (model / BASE_FILE).write_bytes(b"tuned weights")
    result = _run(tmp_path, model=model, weights="int8")
    last = result.checks.checks[-1]
    assert last.name == embedding_export.FROZEN_CHECK and last.outcome is Outcome.FAILED
    assert last.detail.startswith("input:") and grafter.calls == []


def test_a_text_checkpoint_is_not_held_to_the_frozen_tensors(hub, grafter, tmp_path):
    # Its weights are no safetensors at all here, and nothing reads them but the graft.
    result = _run(tmp_path)
    assert result.outcome is Outcome.PASSED
    assert embedding_export.FROZEN_CHECK not in [c.name for c in result.checks.checks]


def test_a_text_report_carries_no_vision_and_ties_to_the_graft(hub, grafter, splicer, tmp_path):
    out = tmp_path / "out"
    _convert(_checkpoint(tmp_path), out)
    report = _report(out, DEFAULT_WEIGHTS)
    assert (report["modality"], report["vision_tokens"]) == ("text", None)
    assert report["google_vision_file"] is None and report["splice"] is None
    assert report["graft"]["output"] == str(out / DEFAULT_WEIGHTS / ARTIFACT_NAME)
    assert splicer.calls == []
    assert len(hub.asked[0]) == 2


def test_int4_for_an_image_checkpoint_says_it_was_not_measured(hub, grafter, splicer, tmp_path):
    out = tmp_path / "out"
    assert _convert(_image_checkpoint(tmp_path), out, "--weights", "int4") == 0
    assert grafter.calls[0]["weights"] == "int4"
    assert embedding_export.IMAGE_INT4_NOT_MEASURED in _report(out, "int4")["limitations"]


def test_int4_for_a_text_checkpoint_carries_no_image_limitation(hub, grafter, tmp_path):
    out = tmp_path / "out"
    _convert(_checkpoint(tmp_path), out, "--weights", "int4")
    assert embedding_export.IMAGE_INT4_NOT_MEASURED not in _report(out, "int4")["limitations"]


@pytest.mark.parametrize("weights", [(), ("--weights", "int8"), ("--weights", "int4")])
def test_npu_for_an_image_checkpoint_is_refused_before_anything_runs(
    hub, grafter, splicer, tmp_path, capsys, weights
):
    out = tmp_path / "out"
    assert _convert(_image_checkpoint(tmp_path), out, "--npu", "SM8750", *weights) == 4
    assert hub.asked == [] and grafter.calls == [] and splicer.calls == []
    assert "tuned on images" in capsys.readouterr().err
    assert not out.exists()


def test_npu_for_an_image_checkpoint_is_refused_by_the_run_too(hub, grafter, splicer, tmp_path):
    with pytest.raises(EmbeddingExportError, match="tuned on images"):
        _run(tmp_path, model=_image_checkpoint(tmp_path), weights="int8", npu=("SM8750",))
    assert hub.asked == [] and grafter.calls == []


def test_a_local_text_vision_file_is_used_and_not_downloaded(hub, grafter, splicer, tmp_path):
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_BYTES)
    result = _run(
        tmp_path, model=_image_checkpoint(tmp_path), weights="int8", base_litertlm_440m=local
    )
    assert result.ok
    assert [f["filename"] for f in hub.asked[0]] == [GOOGLE_FILE, BASE_FILE]
    assert splicer.calls[0]["vision"] == local
    vision = result.as_dict()["google_vision_file"]
    assert vision["source"] == "--base-litertlm-440m" and vision["path"] == str(local)


def test_both_local_files_leave_only_the_base_checkpoint_to_download(
    hub, grafter, splicer, tmp_path
):
    google, vision = tmp_path / "google.litertlm", tmp_path / "vision.litertlm"
    google.write_bytes(GOOGLE_BYTES)
    vision.write_bytes(VISION_BYTES)
    model = _image_checkpoint(tmp_path)
    out = tmp_path / "out"
    extra = ("--base-litertlm", str(google), "--base-litertlm-440m", str(vision))
    assert _convert(model, out, *extra) == 0
    assert hub.asked == [[{"repo": BASE_MODEL, "filename": BASE_FILE, "revision": BASE_REVISION}]]
    assert (grafter.calls[0]["google"], splicer.calls[0]["vision"]) == (google, vision)


def test_a_local_text_vision_file_with_another_sha256_is_a_failure(hub, grafter, splicer, tmp_path):
    local = tmp_path / "vision.litertlm"
    local.write_bytes(b"someone else's file")
    result = _run(
        tmp_path, model=_image_checkpoint(tmp_path), weights="int8", base_litertlm_440m=local
    )
    assert result.outcome is Outcome.FAILED
    assert grafter.calls == [] and splicer.calls == []
    last = result.checks.checks[-1]
    assert last.name == embedding_export.VISION_FILE_CHECK
    assert f"{_digest(b'someone else' + chr(39).encode() + b's file')}, not sha256:" in last.detail
    report = json.loads(result.request.report_path.read_text(encoding="utf-8"))
    assert report["google_vision_file"]["sha256"] == _digest(b"someone else's file")
    assert report["artifact"] is None


def test_a_downloaded_text_vision_file_with_another_sha256_is_a_failure(
    hub, grafter, splicer, tmp_path, monkeypatch
):
    monkeypatch.setattr(embedding_export, "VISION_SHA256", "0" * 64)
    result = _run(tmp_path, model=_image_checkpoint(tmp_path), weights="int8")
    assert result.outcome is Outcome.FAILED
    assert grafter.calls == []
    assert result.checks.checks[-1].name == embedding_export.VISION_FILE_CHECK
    assert f"not sha256:{'0' * 64}" in result.checks.checks[-1].detail


def test_a_missing_local_text_vision_file_is_refused(hub, grafter, splicer, tmp_path):
    with pytest.raises(EmbeddingExportError, match="--base-litertlm-440m .*no such file"):
        _run(
            tmp_path,
            model=_image_checkpoint(tmp_path),
            weights="int8",
            base_litertlm_440m=tmp_path / "nowhere.litertlm",
        )


def test_a_local_text_vision_file_for_a_text_checkpoint_is_refused(hub, grafter, tmp_path, capsys):
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_BYTES)
    assert (
        _convert(_checkpoint(tmp_path), tmp_path / "out", "--base-litertlm-440m", str(local)) == 4
    )
    assert hub.asked == [] and grafter.calls == []
    assert "tuned on text" in capsys.readouterr().err


def test_a_local_text_vision_file_for_an_llm_checkpoint_is_refused(
    hub, grafter, llm_export, tmp_path, capsys
):
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m-it"}))
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_BYTES)
    assert _convert(llm, tmp_path / "out", "--base-litertlm-440m", str(local)) == 4
    assert llm_export == [] and grafter.calls == []
    assert "--base-litertlm-440m" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("cause", "outcome"), [("readback", Outcome.FAILED), ("run", Outcome.UNCHECKED)]
)
def test_a_splice_refusal_leaves_no_file_and_no_text_file(
    hub, grafter, splicer, tmp_path, cause, outcome
):
    splicer.cause = cause
    result = _run(tmp_path, model=_image_checkpoint(tmp_path), weights="int8")
    assert result.outcome is outcome
    last = result.checks.checks[-1]
    assert last.name == embedding_export.SPLICE_CHECK and last.detail.startswith(f"{cause}:")
    assert result.artifact is None and result.graft is not None
    assert sorted(p.name for p in (tmp_path / "out" / "int8").iterdir()) == [REPORT_NAME]
    report = json.loads(result.request.report_path.read_text(encoding="utf-8"))
    assert report["artifact"] is None and report["splice"] is None
    assert not any("has not been measured" in t for t in report["limitations"])


def test_a_graft_refusal_on_an_image_checkpoint_never_splices(hub, grafter, splicer, tmp_path):
    grafter.cause = "norm"
    result = _run(tmp_path, model=_image_checkpoint(tmp_path), weights="int8")
    assert result.outcome is Outcome.FAILED
    assert splicer.calls == [] and result.artifact is None


def test_an_image_checkpoint_with_the_deciders_prefix_is_refused(hub, grafter, tmp_path):
    with pytest.raises(EmbeddingExportError, match="not the image input's ''"):
        _run(tmp_path, model=_image_checkpoint(tmp_path, prefix=DECIDER_PREFIX), weights="int8")


def test_the_summary_names_the_440m_file_and_the_vision_tokens(hub, grafter, splicer, tmp_path):
    result = _run(tmp_path, model=_image_checkpoint(tmp_path), weights="int8")
    lines = "\n".join(embedding_export.summarise(result))
    assert VISION_FILE in lines and "70 vision tokens per image" in lines
    assert str(tmp_path / "out" / "int8" / ARTIFACT_NAME) in lines
