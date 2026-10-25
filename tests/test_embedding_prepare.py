"""`prepare --base-model google/embeddinggemma-2`: a choice decision, measured as it is embedded.

No tokenizer runs here. `FakeEmbedTokenCounter`, this file's own, counts
whitespace words; the command-line tests replace `StageEnv.run` to see which
environment and which spec the real counter would have used.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned

from litetune import envs
from litetune.checks import Outcome
from litetune.cli import _token_counter, main
from litetune.embedding_tune import DECIDER_PREFIX, EMBEDDING_MODEL, EMBEDDING_REVISION, RECIPE
from litetune.prepare import (
    LENGTH_CHECK,
    HuggingFaceTokenCounter,
    PrepareError,
    PrepareRequest,
    prepare,
)

OPTIONS = ("refund", "card_lost", "balance")


@dataclass
class FakeEmbedTokenCounter:
    """One token per whitespace-separated word; no tokenizer, no subprocess."""

    calls: list[list[str]] = field(default_factory=list)

    name = "fake"

    def describe(self) -> dict[str, Any]:
        return {"tokenizer": "fake"}

    def count(self, texts: Sequence[str]) -> list[int]:
        self.calls.append(list(texts))
        return [len(t.split()) for t in texts]

    def first_special(self, texts: Sequence[str]) -> list[str | None]:
        return [None for _ in texts]


def _questions(tmp_path: Path, name: str = "questions.json", **extra: Any) -> Path:
    path = tmp_path / name
    body = {
        "type": "choice",
        "question": "What does the customer want?",
        "options": [{"id": o} for o in OPTIONS],
        **extra,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _data(
    tmp_path: Path, rows: list[dict[str, Any]] | None = None, name: str = "raw.jsonl"
) -> Path:
    rows = rows or [
        {"prompt": f"message {i} about {option}", "label": option}
        for option in OPTIONS
        for i in range(12)
    ]
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _request(tmp_path: Path, **kwargs: Any) -> PrepareRequest:
    params: dict[str, Any] = {
        "data": _data(tmp_path),
        "output_dir": tmp_path / "out",
        "context_length": 1024,
        "base_model": EMBEDDING_MODEL,
        "questions": _questions(tmp_path),
    }
    params.update(kwargs)
    return PrepareRequest(**params)


def _check(result, name):
    return next(c for c in result.checks.checks if c.name == name)


# ---------------------------------------------------------------------------
# Accepted, and refused
# ---------------------------------------------------------------------------


def test_a_choice_decision_for_embeddinggemma_2_is_split(tmp_path):
    result = prepare(_request(tmp_path))
    assert result.outcome is not Outcome.FAILED
    assert result.train is not None and result.heldout is not None
    assert (tmp_path / "out" / "decision.json").is_file()
    assert any("embedding path does not read it" in t for t in result.limitations)


def test_no_family_rule_is_asked_for_embeddinggemma_2(tmp_path):
    # Before this, `decision_model_refusal` refused the model for having no rule.
    result = prepare(_request(tmp_path))
    assert not any("no family rule" in t for t in result.limitations)


def test_a_none_option_is_refused(tmp_path):
    with pytest.raises(PrepareError, match="none_option \\('other'\\)"):
        prepare(
            _request(tmp_path, questions=_questions(tmp_path, "none.json", none_option="other"))
        )


@pytest.mark.parametrize("kind", ["noul", "score"])
def test_a_question_that_is_not_choice_is_refused(tmp_path, kind):
    options = (
        [{"id": "no"}, {"id": "uncertain"}, {"id": "yes"}]
        if kind == "noul"
        else [{"id": str(i)} for i in range(1, 6)]
    )
    path = tmp_path / f"{kind}.json"
    path.write_text(
        json.dumps({"type": kind, "question": "How sure?", "options": options}), encoding="utf-8"
    )
    rows = [{"prompt": f"m {i}", "label": o["id"]} for o in options for i in range(10)]
    with pytest.raises(PrepareError, match=f"{kind} question"):
        prepare(_request(tmp_path, data=_data(tmp_path, rows, "given.jsonl"), questions=path))


def test_rows_that_are_not_a_decision_are_refused(tmp_path):
    rows = [{"prompt": f"p {i}", "completion": "c"} for i in range(20)]
    with pytest.raises(PrepareError, match="on a decision only"):
        prepare(_request(tmp_path, data=_data(tmp_path, rows, "given.jsonl"), questions=None))


def test_another_model_without_a_rule_is_still_refused(tmp_path):
    with pytest.raises(PrepareError, match="no family rule"):
        prepare(_request(tmp_path, base_model="someone/unknown-model"))


# ---------------------------------------------------------------------------
# Lengths: the prefix and the state, against the recipe's window
# ---------------------------------------------------------------------------


def test_the_counter_is_given_prefix_and_state_only(tmp_path):
    counter = FakeEmbedTokenCounter()
    result = prepare(_request(tmp_path, tokens=counter))
    counted = counter.calls[0]
    assert len(counted) == 36
    assert all(t.startswith(DECIDER_PREFIX) for t in counted)
    assert DECIDER_PREFIX + "message 0 about refund" in counted
    # Neither the question nor an option list reaches the count.
    assert not any("What does the customer want?" in t for t in counted)
    assert _check(result, LENGTH_CHECK).outcome is Outcome.PASSED
    lengths = result.lengths.as_dict()
    assert lengths["context_length"] == RECIPE.max_seq_length
    assert "completion_tokens" not in lengths


def test_a_row_over_the_recipes_window_stops_the_split(tmp_path):
    rows = [
        {"prompt": f"message {i} about {option}", "label": option}
        for option in OPTIONS
        for i in range(12)
    ]
    rows[0]["prompt"] = " ".join(["word"] * 200)
    counter = FakeEmbedTokenCounter()
    result = prepare(_request(tmp_path, data=_data(tmp_path, rows, "given.jsonl"), tokens=counter))
    check = _check(result, LENGTH_CHECK)
    assert check.outcome is Outcome.FAILED
    assert f"{RECIPE.max_seq_length}-token" in check.detail
    assert result.train is None


def test_a_smaller_context_length_is_the_window(tmp_path):
    counter = FakeEmbedTokenCounter()
    result = prepare(_request(tmp_path, tokens=counter, context_length=8))
    # "task: sentence similarity | query: message N about X" is 10 words.
    assert _check(result, LENGTH_CHECK).outcome is Outcome.FAILED
    assert result.lengths.as_dict()["context_length"] == 8


def test_without_a_tokenizer_the_note_names_the_embedding_script(tmp_path):
    result = prepare(_request(tmp_path))
    note = next(t for t in result.limitations if "token lengths were not measured" in t)
    assert "no length check of its own" in note
    assert "refuses a row over max_seq_length" not in note


# ---------------------------------------------------------------------------
# The command line: which tokenizer, in which environment
# ---------------------------------------------------------------------------


def _args(**kw: Any):
    import argparse

    base = {"tokenizer": None, "tokenizer_revision": None, "base_model": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_embedding_tokenizer_counts_in_embed_train_with_its_specials():
    counter = _token_counter(_args(tokenizer=EMBEDDING_MODEL, base_model=EMBEDDING_MODEL))
    assert isinstance(counter, HuggingFaceTokenCounter)
    assert counter.env is envs.EMBED_TRAIN
    assert counter.revision == EMBEDDING_REVISION
    assert counter.special_tokens is True


def test_an_llm_tokenizer_still_counts_in_train_without_specials():
    counter = _token_counter(_args(tokenizer="google/gemma-3-270m-it"))
    assert isinstance(counter, HuggingFaceTokenCounter)
    assert counter.env is envs.TRAIN
    assert counter.special_tokens is False


def test_another_tokenizer_for_embeddinggemma_2_is_refused():
    with pytest.raises(PrepareError, match="its own tokenizer"):
        _token_counter(_args(tokenizer="google/gemma-3-270m-it", base_model=EMBEDDING_MODEL))


def test_another_revision_of_the_embedding_tokenizer_is_refused():
    with pytest.raises(PrepareError, match="tokenizer-revision"):
        _token_counter(
            _args(tokenizer=EMBEDDING_MODEL, tokenizer_revision="main", base_model=EMBEDDING_MODEL)
        )


@dataclass
class FakeCountingEnv:
    """`StageEnv.run` for the token-count script: records the spec, answers word counts."""

    specs: list[dict[str, Any]] = field(default_factory=list)
    envs_run: list[str] = field(default_factory=list)

    def __call__(self, env, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        self.envs_run.append(env.name)
        if spec.get("specials"):
            payload: dict[str, Any] = {"specials": [None for _ in spec["texts"]]}
        else:
            payload = {"counts": [len(t.split()) for t in spec["texts"]]}
        Path(spec["out"]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")


def test_the_command_line_counts_in_embed_train_with_add_special_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    fake = FakeCountingEnv()

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    def run(self, args, timeout: int = 3600, **kwargs):
        return fake(self, args, timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", run)
    code = main(
        [
            "prepare",
            "--data",
            str(_data(tmp_path)),
            "--output-dir",
            str(tmp_path / "out"),
            "--context-length",
            "1024",
            "--questions",
            str(_questions(tmp_path)),
            "--base-model",
            EMBEDDING_MODEL,
            "--tokenizer",
            EMBEDDING_MODEL,
        ]
    )
    assert code == 0
    assert set(fake.envs_run) == {envs.EMBED_TRAIN.name}
    counting = [s for s in fake.specs if not s.get("specials")]
    assert counting and all(s["add_special_tokens"] is True for s in counting)
    assert all(s["revision"] == EMBEDDING_REVISION for s in fake.specs)
