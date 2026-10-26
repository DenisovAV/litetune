"""`verify`'s routing of flags that belong to one kind of checkpoint.

Nothing here runs a model: every request is refused before any stage starts, which
is the property under test. A checkpoint is a directory with a `litetune.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from litetune.decisions import Decision, Option
from litetune.embedding_tune import (
    DECIDER_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    write_embedding_record,
)
from litetune.evaluate import DecodeConfig
from litetune.liveness import LivenessThresholds
from litetune.prompt_mode import PromptMode
from litetune.verify import Status, VerifyRequest, run_verify

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


def _decision_checkpoint(root: Path) -> Path:
    model = root / "decided"
    model.mkdir()
    (model / "litetune.json").write_text(
        json.dumps(
            {
                "base_model": "google/gemma-3-270m",
                "decision": DECISION.as_dict(),
                "decision_sha256": DECISION.digest(),
            }
        ),
        encoding="utf-8",
    )
    return model


def _files(root: Path) -> tuple[Path, Path]:
    model = root / "model.litertlm"
    model.write_bytes(b"x")
    heldout = root / "heldout.jsonl"
    heldout.write_text("", encoding="utf-8")
    return model, heldout


def _checks_text(result: Any) -> str:
    return json.dumps(result.manifest["checks"])


def _failed_checks(result: Any) -> list[dict[str, Any]]:
    return [c for c in result.manifest["checks"] if c["outcome"] == "failed"]


def test_one_embedding_flag_on_another_checkpoint_is_refused_in_the_singular(tmp_path):
    model, heldout = _files(tmp_path)
    other = tmp_path / "llm"
    other.mkdir()
    result = run_verify(
        VerifyRequest(model=model, reference=str(other), data=heldout, with_base=True)
    )
    assert result.status is Status.FAILED_HARNESS
    (check,) = _failed_checks(result)
    assert "--with-base does not apply here" in check["detail"]


def test_several_embedding_flags_on_another_checkpoint_are_refused_in_the_plural(tmp_path):
    model, heldout = _files(tmp_path)
    other = tmp_path / "llm"
    other.mkdir()
    result = run_verify(
        VerifyRequest(
            model=model,
            reference=str(other),
            data=heldout,
            with_base=True,
            train_data=tmp_path / "train.jsonl",
        )
    )
    (check,) = _failed_checks(result)
    assert "--train-data, --with-base do not apply here" in check["detail"]


def test_the_hint_for_a_tune_output_directory_reads_as_a_sentence(tmp_path):
    model, heldout = _files(tmp_path)
    inner = _embedding_checkpoint(tmp_path)
    result = run_verify(
        VerifyRequest(model=model, reference=str(inner.parent), data=heldout, with_base=True)
    )
    (check,) = _failed_checks(result)
    assert (
        f"none; {inner} is the model directory, where `tune` writes it. Refused"
        in (check["detail"])
    )


@pytest.mark.parametrize("flag", ["with_base", "train_data", "base_litertlm"])
def test_a_decision_checkpoint_given_an_embedding_flag_is_refused(tmp_path, flag):
    model, heldout = _files(tmp_path)
    reference = _decision_checkpoint(tmp_path)
    value: Any = True if flag == "with_base" else tmp_path / "given"
    result = run_verify(
        VerifyRequest(model=model, reference=str(reference), data=heldout, **{flag: value})
    )
    assert result.status is Status.FAILED_HARNESS
    cli_name = "--" + flag.replace("_", "-")
    assert cli_name in _checks_text(result)
    assert "not apply" in _checks_text(result)


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"prompt_mode": PromptMode.PRERENDERED}, "--prompt-mode"),
        ({"contract": Path("contract.json")}, "--contract"),
        ({"declarations": Path("tools.json")}, "--declarations"),
        ({"decode": DecodeConfig(max_tokens=7)}, "--max-tokens"),
        ({"thresholds": LivenessThresholds(max_empty_share=0.5)}, "the liveness thresholds"),
    ],
)
def test_a_generative_flag_on_an_embedding_checkpoint_is_refused_by_name(tmp_path, kwargs, flag):
    model, heldout = _files(tmp_path)
    reference = _embedding_checkpoint(tmp_path)
    result = run_verify(
        VerifyRequest(model=model, reference=str(reference), data=heldout, **kwargs)
    )
    assert result.status is Status.FAILED_HARNESS
    assert flag in _checks_text(result)
    assert "not apply to an embedding checkpoint" in _checks_text(result)
