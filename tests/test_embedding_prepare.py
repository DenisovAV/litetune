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


# ---------------------------------------------------------------------------
# Fix round 1: the window's boundary, and options verify's gate cannot draw from
# ---------------------------------------------------------------------------


def _rows_with_one_of_length(words: int) -> list[dict[str, Any]]:
    rows = [
        {"prompt": f"message {i} about {option}", "label": option}
        for option in OPTIONS
        for i in range(12)
    ]
    rows[0]["prompt"] = " ".join(["word"] * words)
    return rows


def test_a_row_of_exactly_the_window_fits_and_one_token_more_does_not(tmp_path):
    prefix_words = len(DECIDER_PREFIX.split())
    exact = RECIPE.max_seq_length - prefix_words
    fits = prepare(
        _request(
            tmp_path,
            data=_data(tmp_path, _rows_with_one_of_length(exact), "fits.jsonl"),
            tokens=FakeEmbedTokenCounter(),
        )
    )
    check = _check(fits, LENGTH_CHECK)
    assert check.outcome is Outcome.PASSED, check.detail
    assert fits.lengths.total["max"] == RECIPE.max_seq_length

    over = prepare(
        _request(
            tmp_path,
            data=_data(tmp_path, _rows_with_one_of_length(exact + 1), "over.jsonl"),
            output_dir=tmp_path / "out-over",
            tokens=FakeEmbedTokenCounter(),
        )
    )
    assert _check(over, LENGTH_CHECK).outcome is Outcome.FAILED
    assert over.lengths.over_length[0].tokens == RECIPE.max_seq_length + 1


def test_measure_embedded_lengths_counts_the_limit_itself_as_fitting():
    from litetune.decisions import Decision, Option
    from litetune.prepare import Row, measure_embedded_lengths

    decision = Decision("choice", "Which?", (Option("a"), Option("b")))
    row = Row(lineno=1, prompt=decision.render("one two three"), completion="a", target="a")
    counter = FakeEmbedTokenCounter()
    words = len(DECIDER_PREFIX.split()) + 3
    assert measure_embedded_lengths([row], decision, counter, words).over_length == ()
    assert len(measure_embedded_lengths([row], decision, counter, words - 1).over_length) == 1


def _thin_rows(thin_count: int) -> list[dict[str, Any]]:
    return [
        {"prompt": f"message {i} about {option}", "label": option}
        for option, n in (("refund", 400), ("card_lost", 400), ("balance", thin_count))
        for i in range(n)
    ]


def test_an_option_with_fewer_training_rows_than_verifys_gate_draws_is_named(tmp_path):
    from litetune.embedding_verify import GATED_K

    result = prepare(
        _request(tmp_path, data=_data(tmp_path, _thin_rows(GATED_K - 1), "thin.jsonl"))
    )
    note = next(t for t in result.limitations if "will refuse this split" in t)
    assert "'balance'" in note
    assert "'refund'" not in note and "'card_lost'" not in note
    assert f"{GATED_K}-shot gate" in note


def test_an_option_with_exactly_the_gates_k_training_rows_is_not_named(tmp_path):
    from litetune.embedding_verify import GATED_K

    # The split is seeded by content, so find the size at which 'balance' keeps exactly k.
    for size in range(GATED_K, 4 * GATED_K):
        result = prepare(
            _request(
                tmp_path,
                data=_data(tmp_path, _thin_rows(size), f"k{size}.jsonl"),
                output_dir=tmp_path / f"out-{size}",
            )
        )
        if result.decision.train["balance"] == GATED_K:
            assert not any("will refuse this split" in t for t in result.limitations)
            return
    pytest.fail(f"no size in range kept exactly {GATED_K} training rows of 'balance'")


def test_no_such_note_when_every_option_has_enough_training_rows(tmp_path):
    result = prepare(_request(tmp_path, data=_data(tmp_path, _thin_rows(400), "ok.jsonl")))
    assert not any("will refuse this split" in t for t in result.limitations)
