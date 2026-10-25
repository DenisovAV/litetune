"""`convert` of an embedding checkpoint: dispatch, inputs, refusals, the report.

Nothing is downloaded and nothing is grafted here. `FakeHub`, this file's own,
stands in for `StageEnv.run` on the download script and writes small files in
place of Google's file and the base checkpoint; `FakeGraft` stands in for
`embedding_convert.graft` (whose own tests run the real script on a synthetic
flatbuffer) and writes a file where the graft would.
"""

from __future__ import annotations

import hashlib
import json
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
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    write_embedding_record,
)

GOOGLE_BYTES = b"not google's file, a stand-in"


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
            path.write_bytes(GOOGLE_BYTES if f["filename"] == GOOGLE_FILE else b"base weights")
            paths.append(str(path))
        Path(spec["out"]).write_text(
            json.dumps({"paths": paths, "huggingface_hub": "0.0-fake"}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, "", "")


@dataclass
class FakeGraft:
    """`embedding_convert.graft`: writes the output, or refuses with `cause`."""

    cause: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, env, google, base, tuned, out_path, weights="int8", **kwargs) -> GraftResult:
        self.calls.append(
            {"google": Path(google), "base": Path(base), "tuned": Path(tuned), "out": out_path}
            | {"weights": weights}
        )
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
    return fake


@pytest.fixture
def grafter(monkeypatch) -> FakeGraft:
    fake = FakeGraft()
    monkeypatch.setattr(embedding_export, "graft", fake)
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
    if weights:
        (model / BASE_FILE).write_bytes(b"tuned weights")
    write_embedding_record(model, record or _record())
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
    assert call["tuned"] == model / BASE_FILE
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
