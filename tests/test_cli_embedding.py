"""Command-line seams of the embedding path that no stage test reaches.

`convert`'s timeout default, an LLM flag refused, and `bundle` handed an embedding
checkpoint or the file `convert` wrote. The stages are faked at the `cli` module:
what is under test is what the command line forwards and refuses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from litetune import cli
from litetune.cli import main
from litetune.decisions import Decision, Option
from litetune.embedding_convert import GRAFT_TIMEOUT_S
from litetune.embedding_export import (
    ARTIFACT_NAME,
    EMBEDDING_CONVERT_SCHEMA,
    REPORT_NAME,
    EmbeddingExportError,
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
from litetune.export import ExportRequest

DECISION = Decision("choice", "What now?", (Option("a"), Option("b")))


def _embedding_checkpoint(root: Path) -> Path:
    model = root / "tuned" / "model"
    model.mkdir(parents=True)
    write_embedding_record(
        model,
        EmbeddingRecord(
            base_model=EMBEDDING_MODEL,
            base_model_revision=EMBEDDING_REVISION,
            prompt_name=PROMPT_NAME,
            prefix=DECIDER_PREFIX,
            labels=("a", "b"),
            rows={"train": 20, "heldout": 6},
            steps_done=6,
            steps_planned=6,
            seconds=1.0,
            device="cpu",
            dtype="float32",
            head_heldout_accuracy=0.5,
            recipe=RECIPE.as_dict(),
            decision=DECISION.as_dict(),
            decision_sha256=DECISION.digest(),
        ),
    )
    return model


@pytest.fixture
def seen(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def embedding(request, record, events=None):
        captured["embedding"] = request
        raise EmbeddingExportError("stop here")

    def llm(request, events=None):
        captured["llm"] = request
        raise EmbeddingExportError("stop here")

    monkeypatch.setattr(cli, "run_embedding_export", embedding)
    monkeypatch.setattr(cli, "run_export", llm)
    return captured


def _convert(model: Path, out: Path, *extra: str) -> int:
    return main(["convert", "--model", str(model), "--output-dir", str(out), *extra])


def _llm_checkpoint(root: Path) -> Path:
    llm = root / "llm"
    llm.mkdir()
    (llm / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m-it"}))
    return llm


def test_an_embedding_convert_uses_the_grafts_timeout_when_none_is_given(seen, tmp_path):
    _convert(_embedding_checkpoint(tmp_path), tmp_path / "out")
    assert seen["embedding"].timeout_s == GRAFT_TIMEOUT_S == 3600


def test_an_embedding_convert_takes_the_timeout_it_is_given(seen, tmp_path):
    _convert(_embedding_checkpoint(tmp_path), tmp_path / "out", "--timeout-s", "77")
    assert seen["embedding"].timeout_s == 77


def test_an_llm_convert_keeps_the_exports_timeout_default(seen, tmp_path):
    _convert(_llm_checkpoint(tmp_path), tmp_path / "out", "--recipe", "dynamic_wi8_afp32")
    assert seen["llm"].timeout_s == ExportRequest.timeout_s
    assert ExportRequest.timeout_s != GRAFT_TIMEOUT_S


def test_an_llm_convert_takes_the_timeout_it_is_given(seen, tmp_path):
    _convert(
        _llm_checkpoint(tmp_path),
        tmp_path / "out",
        "--recipe",
        "dynamic_wi8_afp32",
        "--timeout-s",
        "77",
    )
    assert seen["llm"].timeout_s == 77


def test_train_metrics_is_an_llm_flag_the_embedding_convert_refuses(seen, tmp_path, capsys):
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    code = _convert(
        _embedding_checkpoint(tmp_path), tmp_path / "out", "--train-metrics", str(metrics)
    )
    assert code == 4
    assert "embedding" not in seen
    assert "--train-metrics" in capsys.readouterr().err


def _bundle(model: Path, out: Path) -> int:
    return main(
        [
            "bundle",
            "--output-dir",
            str(out),
            "--model",
            str(model),
            "--base-model",
            EMBEDDING_MODEL,
            "--base-model-revision",
            EMBEDDING_REVISION,
        ]
    )


def test_bundle_refuses_an_embedding_checkpoint_and_points_at_convert_and_verify(tmp_path, capsys):
    code = _bundle(_embedding_checkpoint(tmp_path), tmp_path / "bundle")
    err = capsys.readouterr().err
    assert code == 4
    assert "ends at `convert` and `verify`" in err
    assert f"{ARTIFACT_NAME}" in err and "EmbeddingModel.createDecider" in err
    assert not (tmp_path / "bundle").exists()


def test_bundle_refuses_the_file_convert_wrote_for_an_embedding_checkpoint(tmp_path, capsys):
    artifact = tmp_path / "out" / "int4" / ARTIFACT_NAME
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"grafted")
    (artifact.parent / REPORT_NAME).write_text(json.dumps({"schema": EMBEDDING_CONVERT_SCHEMA}))
    code = _bundle(artifact, tmp_path / "bundle")
    err = capsys.readouterr().err
    assert code == 4
    assert "converted file" in err and "EmbeddingModel.createDecider" in err


def test_bundle_does_not_take_another_convert_report_for_an_embedding_one(tmp_path, capsys):
    artifact = tmp_path / "out" / ARTIFACT_NAME
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x")
    (artifact.parent / REPORT_NAME).write_text(json.dumps({"schema": "litetune.convert/1"}))
    _bundle(artifact, tmp_path / "bundle")
    assert "EmbeddingModel.createDecider" not in capsys.readouterr().err
