"""`verify` of an embedding checkpoint: the deciders' arithmetic, the gate, the run, the manifest.

No model and no engine runs here. The evaluation script's helpers are `exec`'d
in this process against synthetic vectors (numpy, a dev dependency).
`FakeEmbedEnvs`, this file's own, stands in for `StageEnv.run`: for the two
embedding scripts it writes vectors made from each text's label, for the
evaluation script it runs those same helpers, and for the download script it
writes a stand-in for Google's file.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned

from litetune import embedding_export, embedding_verify, envs
from litetune.checks import Outcome
from litetune.cli import main
from litetune.decisions import Decision, Option
from litetune.embedding_export import EMBEDDING_CONVERT_SCHEMA, REPORT_NAME
from litetune.embedding_tune import (
    DECIDER_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    LabelledText,
    write_embedding_record,
)
from litetune.embedding_verify import (
    BASE_EMBED_CHECK,
    CANDIDATE_CHECK,
    DEFAULT_MAX_CONVERSION_COST,
    DRAWS,
    ENGINE_SCRIPT,
    ENGINES_SCRIPT,
    EVAL_SCRIPT,
    GATE_CHECK,
    GATED_K,
    KS,
    MAX_INPUT_TOKENS,
    RECORD_CHECK,
    REFERENCE_CHECK,
    REFERENCE_SCRIPT,
    ROWS_CHECK,
    SCORER,
    SCORER_MEANS,
    STOP_ON,
    centroid_cost,
    convert_facts,
    gate,
    kshot_costs,
    labels_short_of,
    side_by_side,
    usable_cpus,
    weights_of,
)
from litetune.metrics import Difference, paired_difference
from litetune.storage import hash_file
from litetune.verify import ReferenceRole, Status, VerifyRequest, run_verify

np: Any = pytest.importorskip("numpy")

OPTIONS = ("balance", "card_lost", "refund")
GOOGLE_BYTES = b"a stand-in for Google's file"


@pytest.fixture(scope="module")
def ev() -> dict[str, Any]:
    """The evaluation script's namespace, `main` not run."""
    namespace: dict[str, Any] = {"__name__": "embedding_eval"}
    exec(compile(EVAL_SCRIPT, "embedding_eval.py", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# The deciders, on synthetic vectors
# ---------------------------------------------------------------------------


def test_kshot_picks_draw_k_rows_of_each_label_as_eval_s5_does(ev):
    ytr = np.array([0, 1, 2] * 12)
    picks = ev["kshot_picks"](ytr, 3, 5, 0)
    assert len(picks) == 15
    for c in range(3):
        chosen = picks[c * 5 : (c + 1) * 5]
        assert (ytr[chosen] == c).all() and len(set(chosen.tolist())) == 5
        # One generator seeded with the draw, per label: eval_s5.py's expression.
        expected = np.random.default_rng(0).choice(np.flatnonzero(ytr == c), 5, replace=False)
        assert chosen.tolist() == expected.tolist()


def test_kshot_picks_are_the_same_for_the_same_draw_and_differ_across_draws(ev):
    ytr = np.array([0, 1] * 30)
    a, b = ev["kshot_picks"](ytr, 2, 10, 3), ev["kshot_picks"](ytr, 2, 10, 3)
    assert a.tolist() == b.tolist()
    assert a.tolist() != ev["kshot_picks"](ytr, 2, 10, 4).tolist()


def test_the_centroid_decider_answers_the_nearest_normalised_mean(ev):
    xtr = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]])
    ytr = np.array([0, 0, 1, 1])
    xte = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    yte = np.array([0, 1, 1])
    assert ev["centroid_correct"](xtr, ytr, xte, yte, 2).tolist() == [True, True, False]


def test_the_centroid_is_normalised_before_the_cosine(ev):
    # Label 0's rows are long, label 1's short: unnormalised, the dot product
    # would pull every input to label 0.
    xtr = np.array([[10.0, 0.0], [0.0, 0.1]])
    ytr = np.array([0, 1])
    xte = np.array([[0.4, 0.6]])
    assert ev["centroid_correct"](xtr, ytr, xte, np.array([1]), 2).tolist() == [True]


def _clusters(rng, labels: list[int], noise: float = 0.05, dim: int = 6):
    v = np.eye(dim)[labels] + noise * rng.standard_normal((len(labels), dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_evaluate_reports_every_decider_and_its_rows(ev):
    rng = np.random.default_rng(1)
    ytr = np.array([0, 1, 2] * 12)
    yte = np.array([0, 1, 2] * 4)
    v = _clusters(rng, [*ytr.tolist(), *yte.tolist()])
    acc, correct = ev["evaluate"](v, ytr, yte, 3, (5, 10), 5)
    assert acc["centroid"] == 1.0
    assert acc["centroid_k10"] == {"mean": 1.0, "min": 1.0, "draws": [1.0] * 5}
    assert len(correct["centroid"]) == 12
    assert len(correct["centroid_k5"]) == 5 and len(correct["centroid_k5"][0]) == 12
    assert "linear" not in acc


def test_evaluate_keeps_mean_and_min_of_the_draws(ev, monkeypatch):
    ytr = np.array([0, 1] * 10)
    yte = np.array([0, 1])
    v = np.zeros((22, 2))
    # The all-rows centroid first, then each draw.
    draws = iter([np.array([True, True])] + [np.array([True, True]), np.array([True, False])])
    monkeypatch.setitem(ev, "centroid_correct", lambda *a: next(draws))
    acc, _ = ev["evaluate"](v, ytr, yte, 2, (5,), 2)
    assert acc["centroid_k5"] == {"mean": 0.75, "min": 0.5, "draws": [1.0, 0.5]}


def test_the_kshot_decider_is_fitted_on_its_drawn_rows_alone(ev):
    # Noisy vectors: the centroid of every training row and the centroid of
    # five drawn rows per label answer differently on some held-out rows.
    rng = np.random.default_rng(11)
    ytr = np.array([0, 1, 2] * 20)
    yte = np.array([0, 1, 2] * 20)
    v = _clusters(rng, [*ytr.tolist(), *yte.tolist()], noise=1.2)
    xtr, xte = v[:60], v[60:]
    _, correct = ev["evaluate"](v, ytr, yte, 3, (5,), 3)
    differs = False
    for d in range(3):
        idx = ev["kshot_picks"](ytr, 3, 5, d)
        drawn = ev["centroid_correct"](xtr[idx], ytr[idx], xte, yte, 3).astype(int).tolist()
        assert correct["centroid_k5"][d] == drawn
        differs = differs or drawn != correct["centroid"]
    assert differs, "the data must tell the two deciders apart"


def test_unusable_rows_are_non_finite_or_zero(ev):
    v = np.array([[1.0, 0.0], [np.nan, 1.0], [0.0, 0.0], [np.inf, 0.0], [0.6, 0.8]])
    assert ev["unusable_rows"](v).tolist() == [1, 2, 3]


def test_cosine_rows_is_one_for_the_same_vectors_and_names_the_worst(ev):
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert ev["cosine_rows"](a, a) == {"mean": 1.0, "min": 1.0, "rows": 2}
    assert ev["cosine_rows"](a, b) == {"mean": 0.9, "min": 0.8, "rows": 2}


def test_the_linear_decider_separates_clusters(ev):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    ytr = np.array([0, 1, 2] * 10)
    yte = np.array([0, 1, 2] * 3)
    v = _clusters(rng, [*ytr.tolist(), *yte.tolist()])
    assert ev["linear_correct"](torch, v[:30], ytr, v[30:], yte, 3, 1e-5).all()


def test_the_evaluation_script_main_writes_what_the_host_reads(ev, tmp_path, monkeypatch):
    import sys
    import types

    rng = np.random.default_rng(5)
    ytr = [0, 1, 2] * 12
    yte = [0, 1, 2] * 4
    v = _clusters(rng, ytr + yte)
    paths = {}
    for name in ("reference", "candidate"):
        paths[name] = str(tmp_path / f"{name}.npy")
        np.save(paths[name], v.astype(np.float32))
    spec = tmp_path / "spec.json"
    report = tmp_path / "report.json"
    spec.write_text(
        json.dumps(
            {
                "vectors": paths,
                "train_labels": ytr,
                "heldout_labels": yte,
                "n_labels": 3,
                "ks": [5, 10],
                "draws": 5,
                "decay": 1e-5,
                "pairs": [["reference", "candidate"]],
                "report": str(report),
            }
        )
    )
    stub = types.ModuleType("torch")
    stub.__version__ = "stub"
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setitem(ev, "linear_correct", lambda torch, xtr, ytr, xte, yte, n, d: yte == yte)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(spec)])
    assert ev["main"]() == 0
    out = json.loads(report.read_text())
    assert out["accuracy"]["candidate"]["linear"] == 1.0
    assert out["accuracy"]["reference"]["centroid_k10"]["mean"] == 1.0
    assert len(out["correct"]["candidate"]["centroid_k10"]) == 5
    assert out["cosine"]["reference|candidate"]["mean"] == pytest.approx(1.0)
    assert out["versions"]["torch"] == "stub"
    correct = out["correct"]
    assert kshot_costs(correct, "reference", "candidate", 10)[0].value == 0.0


def test_the_evaluation_script_refuses_vectors_that_do_not_cover_the_rows(
    ev, tmp_path, monkeypatch
):
    import sys
    import types

    path = str(tmp_path / "v.npy")
    np.save(path, np.ones((3, 4), dtype=np.float32))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "vectors": {"reference": path},
                "train_labels": [0, 1],
                "heldout_labels": [0, 1],
                "n_labels": 2,
                "ks": [1],
                "draws": 1,
                "decay": 1e-5,
                "pairs": [],
                "report": str(tmp_path / "r.json"),
            }
        )
    )
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(spec)])
    assert ev["main"]() == 3
    assert "3 vectors for 4 rows" in (tmp_path / "r.json").read_text()


def _eval_main(ev, tmp_path, monkeypatch, vectors: dict[str, Any], ytr, yte, pairs):
    paths = {}
    for name, v in vectors.items():
        paths[name] = str(tmp_path / f"{name}.npy")
        np.save(paths[name], np.asarray(v, dtype=np.float32))
    spec = tmp_path / "spec.json"
    report = tmp_path / "report.json"
    spec.write_text(
        json.dumps(
            {
                "vectors": paths,
                "train_labels": ytr,
                "heldout_labels": yte,
                "n_labels": 3,
                "ks": [5],
                "draws": 1,
                "decay": 1e-5,
                "pairs": pairs,
                "report": str(report),
            }
        )
    )
    stub = types.ModuleType("torch")
    stub.__version__ = "stub"
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setitem(ev, "linear_correct", lambda torch, xtr, ytr, xte, yte, n, d: yte == yte)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(spec)])
    return ev["main"](), json.loads(report.read_text())


def test_the_cosine_between_sides_is_taken_on_the_held_out_rows_only(ev, tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    ytr, yte = [0, 1, 2] * 12, [0, 1, 2] * 4
    a = _clusters(rng, ytr + yte)
    b = a.copy()
    b[:36] = _clusters(np.random.default_rng(9), [2, 0, 1] * 12)  # other training rows
    code, out = _eval_main(
        ev,
        tmp_path,
        monkeypatch,
        {"reference": a, "candidate": b},
        ytr,
        yte,
        [["reference", "candidate"]],
    )
    assert code == 0
    assert out["cosine"]["reference|candidate"]["rows"] == 12
    assert out["cosine"]["reference|candidate"]["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0])
def test_the_evaluation_script_refuses_unusable_vectors(ev, tmp_path, monkeypatch, bad):
    rng = np.random.default_rng(6)
    ytr, yte = [0, 1, 2] * 12, [0, 1, 2] * 4
    v = _clusters(rng, ytr + yte)
    v[40] = bad if bad != 0.0 else np.zeros(v.shape[1])
    code, out = _eval_main(ev, tmp_path, monkeypatch, {"candidate": v}, ytr, yte, [])
    assert code == 3
    assert "candidate: 1 vectors have a non-finite value or a zero norm" in out["error"]


@pytest.mark.parametrize("source", [REFERENCE_SCRIPT, ENGINE_SCRIPT, ENGINES_SCRIPT, EVAL_SCRIPT])
def test_every_script_compiles(source):
    compile(source, "script.py", "exec")


def test_the_engine_script_uses_the_measured_engine_call():
    # emb_s5.py's call: EmbeddingEngine on Backend.CPU(thread_count=...), the
    # prefix in front of the text; and the settings flutter_edge_ai's
    # litert_lm_embedding_pass.dart pins, by the 0.18.0 wrapper's names.
    assert 'litert_lm.Backend.CPU(thread_count=spec["threads"])' in ENGINE_SCRIPT
    assert 'cache_dir=spec["cache_dir"]' in ENGINE_SCRIPT
    assert 'max_input_length=spec["max_input_length"]' in ENGINE_SCRIPT
    assert "normalize=True," in ENGINE_SCRIPT
    assert "insert_special_tokens=True," in ENGINE_SCRIPT
    assert "input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR" in ENGINE_SCRIPT
    assert 'spec["prefix"] + text' in ENGINE_SCRIPT
    assert "cpu_report()" in ENGINE_SCRIPT and "def cpu_report" in ENGINE_SCRIPT


def test_the_reference_script_encodes_as_emb_s5_does():
    assert 'prompt_name=spec["prompt_name"]' in REFERENCE_SCRIPT
    assert "normalize_embeddings=True" in REFERENCE_SCRIPT
    assert 'device="cpu"' in REFERENCE_SCRIPT
    assert '"dtype": torch.float32' in REFERENCE_SCRIPT


# ---------------------------------------------------------------------------
# The engine and reference scripts' main, against stand-in modules
# ---------------------------------------------------------------------------


def _stub_litert_lm(answers: list[Any]) -> tuple[types.ModuleType, dict[str, list[Any]]]:
    """litert_lm 0.18.0's names, with its keyword arguments and no `**kwargs`, so
    a misspelt setting fails here (the real `EmbeddingEngine` swallows them)."""
    calls: dict[str, list[Any]] = {"engine": [], "options": [], "texts": []}
    mod = types.ModuleType("litert_lm")
    replies = iter(answers)

    class Backend:
        @staticmethod
        def CPU(thread_count=None):  # noqa: N802 - the wrapper's name
            return ("cpu", thread_count)

    class InputOverflowStrategy:
        CHUNK_AND_AVERAGE, TRUNCATE, ERROR = "chunk", "truncate", "error"

    class EmbeddingOptions:
        def __init__(
            self,
            *,
            normalize=None,
            insert_special_tokens=None,
            input_overflow_strategy=None,
            output_size=None,
            vision_tokens_per_image=None,
        ):
            calls["options"].append(
                {
                    "normalize": normalize,
                    "insert_special_tokens": insert_special_tokens,
                    "input_overflow_strategy": input_overflow_strategy,
                    "output_size": output_size,
                }
            )

    class EmbeddingEngine:
        def __init__(
            self,
            model_path,
            backend=None,
            vision_backend=None,
            audio_backend=None,
            cache_dir=None,
            min_input_length=None,
            max_input_length=None,
            vision_tokens_per_image=None,
        ):
            calls["engine"].append(
                {
                    "model": model_path,
                    "backend": backend,
                    "cache_dir": cache_dir,
                    "max_input_length": max_input_length,
                    "min_input_length": min_input_length,
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def compute_embedding(self, contents, options=None):
            calls["texts"].append(contents)
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return types.SimpleNamespace(embedding=list(reply))

    mod.Backend = Backend
    mod.InputOverflowStrategy = InputOverflowStrategy
    mod.EmbeddingOptions = EmbeddingOptions
    mod.EmbeddingEngine = EmbeddingEngine
    return mod, calls


def _engine_main(tmp_path, monkeypatch, answers, texts):
    ns: dict[str, Any] = {"__name__": "engine_script"}
    exec(compile(ENGINE_SCRIPT, "engine.py", "exec"), ns)
    stub, calls = _stub_litert_lm(answers)
    monkeypatch.setitem(sys.modules, "litert_lm", stub)
    monkeypatch.setitem(ns, "cpu_report", lambda: {"model": "stub cpu", "count": 2})
    (tmp_path / "texts.json").write_text(json.dumps(texts))
    spec = {
        "model": "/models/m.litertlm",
        "prefix": DECIDER_PREFIX,
        "texts": str(tmp_path / "texts.json"),
        "threads": 3,
        "cache_dir": str(tmp_path / "cache-candidate"),
        "max_input_length": 1024,
        "out": str(tmp_path / "out.npy"),
        "report": str(tmp_path / "report.json"),
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["engine.py", str(tmp_path / "spec.json")])
    code = ns["main"]()
    return code, json.loads((tmp_path / "report.json").read_text()), calls


def test_the_engine_script_sets_the_apps_settings_and_its_own_cache(tmp_path, monkeypatch):
    code, report, calls = _engine_main(tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], ["a", "b"])
    assert code == 0
    assert calls["engine"] == [
        {
            "model": "/models/m.litertlm",
            "backend": ("cpu", 3),
            "cache_dir": str(tmp_path / "cache-candidate"),
            "max_input_length": 1024,
            "min_input_length": None,
        }
    ]
    assert calls["options"] == [
        {
            "normalize": True,
            "insert_special_tokens": True,
            "input_overflow_strategy": "error",
            "output_size": None,
        }
    ]
    assert calls["texts"] == [DECIDER_PREFIX + "a", DECIDER_PREFIX + "b"]
    assert report["settings"] == {
        "cache_dir": str(tmp_path / "cache-candidate"),
        "max_input_length": 1024,
        "normalize": True,
        "insert_special_tokens": True,
        "input_overflow_strategy": "ERROR",
    }
    assert report["cpu"] == {"model": "stub cpu", "count": 2}
    assert report["rows"] == 2 and report["dim"] == 2


def test_the_engine_script_normalises_what_the_engine_returns(tmp_path, monkeypatch):
    code, _, _ = _engine_main(tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], ["a", "b"])
    assert code == 0
    saved = np.load(tmp_path / "out.npy")
    np.testing.assert_allclose(saved, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)


@pytest.mark.parametrize("bad", [[np.nan, 1.0], [np.inf, 0.0], [0.0, 0.0]])
def test_the_engine_script_refuses_an_unusable_vector(tmp_path, monkeypatch, bad):
    code, report, _ = _engine_main(tmp_path, monkeypatch, [[1.0, 0.0], bad], ["a", "b"])
    assert code == 3
    assert (
        "1 of 2 engine vectors have a non-finite value or a zero norm (rows [1])"
        in (report["error"])
    )
    assert not (tmp_path / "out.npy").exists()


def test_the_engine_script_names_a_text_the_engine_refuses(tmp_path, monkeypatch):
    answers = [[1.0, 0.0], RuntimeError("Failed to compute embedding.")]
    code, report, _ = _engine_main(tmp_path, monkeypatch, answers, ["a", "bb"])
    assert code == 3
    assert "refused text 1 (2 characters): Failed to compute embedding." in report["error"]
    assert "longer than 1024 tokens" in report["error"]


def _reference_main(
    tmp_path, monkeypatch, *, prompt: str, vectors, lengths: list[int], max_seq: int = 128
):
    ns: dict[str, Any] = {"__name__": "reference_script"}
    exec(compile(REFERENCE_SCRIPT, "reference.py", "exec"), ns)
    seen: dict[str, Any] = {"tokenized": [], "encode": None, "model": None}
    counts = iter(lengths)

    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.set_num_threads = lambda n: None
    torch.get_num_threads = lambda: 2

    class SentenceTransformer:
        def __init__(self, model, device=None, model_kwargs=None, config_kwargs=None):
            seen["model"] = model
            self.prompts = {PROMPT_NAME: prompt}
            self.max_seq_length = max_seq

        def tokenizer(self, batch, add_special_tokens=None, truncation=None):
            # The count must see past max_seq_length: no truncation, with specials.
            assert add_special_tokens is True and truncation is False
            seen["tokenized"].append(list(batch))
            return {"input_ids": [[0] * next(counts) for _ in batch]}

        def tokenize(self, batch):
            raise AssertionError("st.tokenize truncates; the count must not use it")

        def encode(self, texts, **kwargs):
            seen["encode"] = kwargs
            return np.asarray(vectors, dtype=np.float32)

    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "stub")
    texts = [f"t{i}" for i in range(len(vectors))]
    (tmp_path / "texts.json").write_text(json.dumps(texts))
    spec = {
        "model": "/tuned/model",
        "prompt_name": PROMPT_NAME,
        "prefix": DECIDER_PREFIX,
        "texts": str(tmp_path / "texts.json"),
        "threads": 2,
        "batch_size": 2,
        "out": str(tmp_path / "out.npy"),
        "report": str(tmp_path / "report.json"),
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["reference.py", str(tmp_path / "spec.json")])
    code = ns["main"]()
    return code, json.loads((tmp_path / "report.json").read_text()), seen


def test_the_reference_script_counts_tokens_with_the_prompt(tmp_path, monkeypatch):
    vectors = [[1.0, 0.0], [0.0, 2.0], [0.6, 0.8]]
    code, report, seen = _reference_main(
        tmp_path, monkeypatch, prompt=DECIDER_PREFIX, vectors=vectors, lengths=[5, 129, 128]
    )
    assert code == 0
    assert seen["model"] == "/tuned/model"
    p = DECIDER_PREFIX
    assert seen["tokenized"] == [[p + "t0", p + "t1"], [p + "t2"]]
    assert seen["encode"]["prompt_name"] == PROMPT_NAME
    # 129 exceeds 128; exactly 128 fits, as prepare counts it.
    assert report["tokens"] == {"max": 129, "over_max_seq_length": 1}
    assert report["max_seq_length"] == 128
    np.testing.assert_allclose(
        np.load(tmp_path / "out.npy"), [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], rtol=1e-6
    )


def test_the_reference_script_refuses_a_checkpoint_with_another_prompt(tmp_path, monkeypatch):
    code, report, seen = _reference_main(
        tmp_path, monkeypatch, prompt="task: search result | query: ", vectors=[[1.0]], lengths=[1]
    )
    assert code == 3
    assert "is 'task: search result | query: ', not the decider's prefix" in report["error"]
    assert seen["encode"] is None and not (tmp_path / "out.npy").exists()


@pytest.mark.parametrize("bad", [[np.nan, 1.0], [0.0, 0.0]])
def test_the_reference_script_refuses_an_unusable_vector(tmp_path, monkeypatch, bad):
    code, report, _ = _reference_main(
        tmp_path, monkeypatch, prompt=DECIDER_PREFIX, vectors=[[1.0, 0.0], bad], lengths=[2, 2]
    )
    assert code == 3
    assert (
        "1 of 2 reference vectors have a non-finite value or a zero norm (rows [1])"
        in (report["error"])
    )
    assert not (tmp_path / "out.npy").exists()


# ---------------------------------------------------------------------------
# Pairing and the gate
# ---------------------------------------------------------------------------


def _correct(rows: int, wrong: int) -> list[int]:
    return [0] * wrong + [1] * (rows - wrong)


def test_kshot_costs_pair_each_draw_reference_minus_candidate():
    correct = {
        "reference": {"centroid_k10": [_correct(100, 0)] * DRAWS, "centroid": _correct(100, 0)},
        "candidate": {"centroid_k10": [_correct(100, 2)] * DRAWS, "centroid": _correct(100, 1)},
    }
    costs = kshot_costs(correct, "reference", "candidate", 10)
    assert len(costs) == DRAWS
    assert costs[0].value == pytest.approx(0.02) and costs[0].discordant == 2
    assert costs[0] == paired_difference([True] * 100, [False] * 2 + [True] * 98)
    assert centroid_cost(correct, "reference", "candidate").value == pytest.approx(0.01)


def test_kshot_costs_refuse_unequal_draws():
    correct = {
        "reference": {"centroid_k10": [[1]] * 5},
        "candidate": {"centroid_k10": [[1]] * 4},
    }
    with pytest.raises(ValueError, match="draws"):
        kshot_costs(correct, "reference", "candidate", 10)


def _diff(value: float, half: float) -> Difference:
    return Difference(value=value, ci95=half, n_a=100, n_b=100, detail="", method="paired")


def test_the_gate_passes_when_every_draw_is_within():
    status, check = gate([_diff(0.0, 0.005), _diff(0.004, 0.006)], 0.01)
    assert status == "passed" and check.outcome is Outcome.PASSED
    assert check.name == GATE_CHECK


def test_the_gate_passes_at_the_threshold_exactly():
    assert gate([_diff(0.005, 0.005)], 0.01)[0] == "passed"


def test_the_gate_boundaries_are_exact():
    # Binary fractions, so the sums are exact: an upper end at the threshold
    # passes, a lower end at it does not fail.
    assert gate([_diff(0.125, 0.125)], 0.25)[0] == "passed"
    assert gate([_diff(0.125, 0.1875)], 0.25)[0] == "inconclusive"
    assert gate([_diff(0.5, 0.25)], 0.25)[0] == "inconclusive"
    assert gate([_diff(0.5, 0.125)], 0.25)[0] == "failed_gate"


def test_the_gate_fails_when_any_draw_lies_above():
    status, check = gate([_diff(0.0, 0.001), _diff(0.05, 0.01)], 0.01)
    assert status == "failed_gate" and check.outcome is Outcome.FAILED
    assert "[1]" in check.detail


def test_the_gate_is_inconclusive_when_a_draw_straddles():
    status, check = gate([_diff(0.0, 0.001), _diff(0.008, 0.01)], 0.01)
    assert status == "inconclusive" and check.outcome is Outcome.UNCHECKED


def test_a_failure_wins_over_a_straddle():
    assert gate([_diff(0.008, 0.01), _diff(0.05, 0.01)], 0.01)[0] == "failed_gate"


def test_the_gate_refuses_no_draws():
    with pytest.raises(ValueError):
        gate([], 0.01)


def test_labels_short_of_k_are_named():
    rows = [LabelledText(f"t{i}", "a") for i in range(10)] + [LabelledText("t", "b")] * 9
    assert labels_short_of(rows, ["a", "b", "c"], 10) == ["b", "c"]


# ---------------------------------------------------------------------------
# A whole run, faked
# ---------------------------------------------------------------------------


def _decision(none_option: str | None = None) -> Decision:
    return Decision(
        "choice", "What does the customer want?", tuple(Option(o) for o in OPTIONS), none_option
    )


def _label_of(text: str) -> str:
    return text.rsplit(" ", 1)[1]


@dataclass
class FakeEmbedEnvs:
    """`StageEnv.run` for the embedding, evaluation and download scripts."""

    ev: dict[str, Any]
    candidate: str = "same"  # "same": the reference's vectors; "random": unrelated ones
    fail: set[str] = field(default_factory=set)
    runs: list[tuple[str, str]] = field(default_factory=list)
    specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    texts: dict[str, list[str]] = field(default_factory=dict)
    launches: list[dict[str, Any]] = field(default_factory=list)
    timeouts: dict[str, int] = field(default_factory=dict)
    # Sides the launcher reports as stopped (returncode null) without running them.
    stopped: set[str] = field(default_factory=set)
    # Sides that refuse as the real scripts do: {"error": ...} and exit 3.
    refuse: dict[str, str] = field(default_factory=dict)
    # Sides whose report counts one vector too many; sides that write no .npy.
    rows_off: set[str] = field(default_factory=set)
    no_npy: set[str] = field(default_factory=set)
    # The reference's count of texts at its max_seq_length; None: no count at all.
    over_max_seq_length: int | None = 0
    # Applied to the evaluation's report before it is written.
    edit_eval: Callable[[dict[str, Any]], None] | None = None
    # Each engine side's cache directory, and whether it existed when it ran.
    caches: dict[str, tuple[str, bool]] = field(default_factory=dict)
    hub: Path | None = None

    def vectors(self, texts: list[str], kind: str) -> Any:
        if kind == "random":
            rng = np.random.default_rng(7)
            v = rng.standard_normal((len(texts), 6))
        else:
            rng = np.random.default_rng(3)
            labels = [OPTIONS.index(_label_of(t)) for t in texts]
            v = np.eye(6)[labels] + 0.05 * rng.standard_normal((len(texts), 6))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed(self, name: str, spec: dict[str, Any]) -> int:
        texts = json.loads(Path(spec["texts"]).read_text(encoding="utf-8"))
        self.texts[name] = texts
        if name in self.refuse:
            Path(spec["report"]).write_text(json.dumps({"error": self.refuse[name]}))
            return 3
        kind = {"candidate": self.candidate, "base": "random"}.get(name, "same")
        if name not in self.no_npy:
            np.save(spec["out"], self.vectors(texts, kind).astype(np.float32))
        report: dict[str, Any] = {"rows": len(texts) + (name in self.rows_off), "dim": 6}
        if name == "reference":
            report["max_seq_length"] = 128
            if self.over_max_seq_length is not None:
                report["tokens"] = {"max": 9, "over_max_seq_length": self.over_max_seq_length}
        else:
            self.caches[name] = (spec["cache_dir"], Path(spec["cache_dir"]).is_dir())
            report |= {
                "runtime": {"litert-lm": "0.18.0"},
                "cpu": {"model": "fake", "count": 4},
                "settings": {"max_input_length": spec["max_input_length"]},
                "ms_median": 1.0,
            }
        Path(spec["report"]).write_text(json.dumps(report), encoding="utf-8")
        return 0

    def __call__(self, env, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        name = Path(args[1]).stem
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.timeouts[name] = timeout
        if name != "engines":  # the sides it runs are recorded one by one below
            self.runs.append((name, env.name))
            self.specs[name] = spec
        if name in self.fail:
            return subprocess.CompletedProcess(args, 1, "", f"{name}: RuntimeError: boom")
        if name == "download":
            assert self.hub is not None
            path = self.hub / spec["files"][0]["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(GOOGLE_BYTES)
            Path(spec["out"]).write_text(json.dumps({"paths": [str(path)]}), encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")
        report = Path(spec["report"])
        if name == "reference":
            code = self.embed(name, spec)
            return subprocess.CompletedProcess(args, code, "", "")
        if name == "engines":
            # ENGINES_SCRIPT's contract: a failed run named in stop_on stops
            # the ones after it; any other failure is only reported.
            self.launches.append(spec)
            results: dict[str, Any] = {}
            stopped = False
            for run in spec["runs"]:
                side = run["name"]
                if stopped:
                    results[side] = {"returncode": None, "seconds": None}
                    continue
                if side in self.stopped:
                    results[side] = {"returncode": None, "seconds": None}
                    continue
                sub = json.loads(Path(run["spec"]).read_text(encoding="utf-8"))
                self.runs.append((side, env.name))
                self.specs[side] = sub
                if side in self.fail:
                    Path(run["stderr"]).write_text(f"{side}: RuntimeError: boom", encoding="utf-8")
                    results[side] = {"returncode": 1, "seconds": 1.0}
                    stopped = side in spec["stop_on"]
                    continue
                code = self.embed(side, sub)
                results[side] = {"returncode": code, "seconds": 1.0}
                stopped = code != 0 and side in spec["stop_on"]
            report.write_text(
                json.dumps({"runs": results, "parallel": spec["parallel"]}), encoding="utf-8"
            )
            return subprocess.CompletedProcess(args, 0, "", "")
        assert name == "evaluate"
        ytr, yte = np.array(spec["train_labels"]), np.array(spec["heldout_labels"])
        out: dict[str, Any] = {"accuracy": {}, "correct": {}, "cosine": {}, "versions": {}}
        loaded = {k: np.load(p).astype(np.float64) for k, p in spec["vectors"].items()}
        for side, v in loaded.items():
            out["accuracy"][side], out["correct"][side] = self.ev["evaluate"](
                v, ytr, yte, spec["n_labels"], spec["ks"], spec["draws"]
            )
        for a, b in spec["pairs"]:
            out["cosine"][f"{a}|{b}"] = self.ev["cosine_rows"](
                loaded[a][len(ytr) :], loaded[b][len(ytr) :]
            )
        if self.edit_eval is not None:
            self.edit_eval(out)
        report.write_text(json.dumps(out), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def fake(monkeypatch, tmp_path, ev) -> FakeEmbedEnvs:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    instance = FakeEmbedEnvs(ev=ev, hub=tmp_path / "hub")

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    def run(self, args, timeout: int = 3600, **kwargs):
        return instance(self, args, timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", run)
    monkeypatch.setattr(embedding_export, "GOOGLE_SHA256", hashlib.sha256(GOOGLE_BYTES).hexdigest())
    return instance


def _write_split(path: Path, decision: Decision, per_label: int, stem: str = "message") -> Path:
    rows = [(f"{stem} {i} about {o}", o) for i in range(per_label) for o in OPTIONS]
    path.write_text(
        "".join(
            json.dumps({"prompt": decision.render(s), "completion": label, "target": label}) + "\n"
            for s, label in rows
        ),
        encoding="utf-8",
    )
    return path


@dataclass
class Setup:
    model: Path
    reference: Path
    heldout: Path
    train: Path


def _setup(
    tmp_path: Path,
    *,
    train_per_label: int = 12,
    record: dict[str, Any] | None = None,
    bind_train: bool = False,
) -> Setup:
    decision = _decision()
    split = tmp_path / "split"
    split.mkdir()
    train = _write_split(split / "train.jsonl", decision, train_per_label)
    heldout = _write_split(split / "heldout.jsonl", decision, 6, stem="held")
    reference = tmp_path / "tuned" / "model"
    reference.mkdir(parents=True)
    (reference / "model.safetensors").write_bytes(b"tuned weights")
    fields: dict[str, Any] = {
        "base_model": EMBEDDING_MODEL,
        "base_model_revision": EMBEDDING_REVISION,
        "prompt_name": PROMPT_NAME,
        "prefix": DECIDER_PREFIX,
        "labels": tuple(sorted(OPTIONS)),
        "rows": {"train": train_per_label * 3, "heldout": 18},
        "steps_done": 10,
        "steps_planned": 10,
        "seconds": 1.0,
        "device": "cpu",
        "dtype": "float32",
        "head_heldout_accuracy": 0.9,
        "recipe": RECIPE.as_dict(),
        "decision": decision.as_dict(),
        "decision_sha256": decision.digest(),
    }
    if bind_train:
        fields["train_sha256"] = hash_file(train)
    fields.update(record or {})
    write_embedding_record(reference, EmbeddingRecord(**fields))
    model = tmp_path / "out" / "int4" / "model.litertlm"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"grafted file")
    return Setup(model, reference, heldout, train)


def _verify(s: Setup, **kwargs: Any):
    return run_verify(
        VerifyRequest(model=s.model, reference=str(s.reference), data=s.heldout, **kwargs)
    )


def _checks(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in manifest["checks"]}


def test_a_faithful_file_passes_and_the_manifest_says_what_was_measured(fake, tmp_path):
    s = _setup(tmp_path)
    result = _verify(s)
    m = result.manifest
    assert result.status is Status.PASSED, m["checks"]
    assert m["scorer"]["name"] == SCORER
    assert m["embedding"]["prefix"] == DECIDER_PREFIX
    assert m["embedding"]["gated_on"] == f"centroid_k{GATED_K}"
    assert m["measurements"]["reference"]["centroid_k10"]["mean"] == 1.0
    assert m["measurements"]["candidate"]["engine"]["runtime_version"] == "0.18.0"
    cost = m["quality"]["conversion_cost"]["centroid_k10"]
    assert len(cost["draws"]) == DRAWS and cost["mean"] == 0.0
    assert m["quality"]["cosine"]["reference|candidate"]["mean"] == pytest.approx(1.0)
    gate_entry = m["gates"][0]
    assert gate_entry["outcome"] == "passed"
    assert gate_entry["threshold"] == DEFAULT_MAX_CONVERSION_COST
    assert m["data"]["train"]["rows"] == 36 and m["data"]["heldout"]["rows"] == 18
    notes = " ".join(m["limitations"])
    assert "CPU backend" in notes and "GPU" in notes
    assert "not the application's own build of the runtime on a device's CPU" in notes
    assert "k=5 and k=10" in notes
    assert "gain over Google's file was not measured" in notes
    assert "the gate used 0.01" in notes
    assert "EmbeddingDeciderCalibration.crossFit keeping 0.95" in notes
    assert "abstention is not measured here" in notes
    assert "fitted on every training row" in notes and "weight decay 1e-05" in notes
    assert "with no abstention" in SCORER_MEANS and m["scorer"]["means"] == SCORER_MEANS
    engine = m["measurements"]["candidate"]["engine"]
    assert engine["cpu"] == {"model": "fake", "count": 4}
    assert engine["settings"] == {"max_input_length": MAX_INPUT_TOKENS}


def test_the_sides_run_where_they_should_one_after_another(fake, tmp_path):
    _verify(_setup(tmp_path))
    assert fake.runs == [
        ("reference", envs.EMBED_TRAIN.name),
        ("candidate", envs.RUNTIME.name),
        ("evaluate", envs.EMBED_TRAIN.name),
    ]
    assert fake.specs["reference"]["prompt_name"] == PROMPT_NAME
    assert fake.specs["reference"]["prefix"] == DECIDER_PREFIX
    assert fake.specs["candidate"]["prefix"] == DECIDER_PREFIX
    assert fake.specs["evaluate"]["ks"] == list(KS) and fake.specs["evaluate"]["draws"] == DRAWS


def test_the_reference_side_embeds_with_the_reference_checkpoint(fake, tmp_path):
    s = _setup(tmp_path)
    _verify(s)
    assert fake.specs["reference"]["model"] == str(s.reference)
    assert fake.specs["candidate"]["model"] == str(s.model)


def test_both_sides_embed_the_states_train_then_heldout(fake, tmp_path):
    _verify(_setup(tmp_path))
    expected = [f"message {i} about {o}" for i in range(12) for o in OPTIONS] + [
        f"held {i} about {o}" for i in range(6) for o in OPTIONS
    ]
    assert fake.texts["candidate"] == expected
    assert fake.texts["reference"] == expected
    labels = sorted(OPTIONS)
    assert fake.specs["evaluate"]["heldout_labels"] == [
        labels.index(o) for _ in range(6) for o in OPTIONS
    ]


def test_each_engine_side_has_its_own_cache_directory_in_the_work_dir(fake, tmp_path):
    s = _setup(tmp_path)
    _verify(s, base_litertlm=_google(tmp_path))
    (cand, cand_existed), (base, base_existed) = fake.caches["candidate"], fake.caches["base"]
    assert cand_existed and base_existed and cand != base
    assert Path(cand).name == "cache-candidate" and Path(base).name == "cache-base"
    assert Path(cand).parent not in (s.model.parent, tmp_path)
    assert not Path(cand).exists()  # gone with the run's temporary directory
    assert fake.specs["candidate"]["max_input_length"] == MAX_INPUT_TOKENS == 1024
    assert fake.launches[0]["stop_on"] == list(STOP_ON) == ["candidate"]


def test_a_file_that_embeds_otherwise_fails_the_gate(fake, tmp_path):
    fake.candidate = "random"
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_GATE
    assert _checks(result.manifest)[GATE_CHECK]["outcome"] == "failed"
    assert result.manifest["quality"]["cosine"]["reference|candidate"]["mean"] < 0.9


def test_an_explicit_threshold_is_used(fake, tmp_path):
    result = _verify(_setup(tmp_path), max_conversion_cost=0.05)
    assert result.status is Status.PASSED
    assert result.manifest["gates"][0]["threshold"] == 0.05
    assert not any("the gate used" in t for t in result.manifest["limitations"])


def test_a_large_cost_on_few_rows_against_a_loose_threshold_is_inconclusive(fake, tmp_path):
    # 18 held-out rows cannot place a cost near 0.67 on either side of 0.9.
    fake.candidate = "random"
    assert _verify(_setup(tmp_path), max_conversion_cost=0.9).status is Status.INCONCLUSIVE


_UNSET: Any = object()


def _write_convert(s: Setup, *, checkpoint: Any = _UNSET, decision: Any = _UNSET) -> None:
    """convert.json for `s.model`, its checkpoint block bound to `s.reference` unless told."""
    block: dict[str, Any] = {"path": str(s.reference)}
    sha = hash_file(s.reference / "model.safetensors") if checkpoint is _UNSET else checkpoint
    dec = _decision().digest() if decision is _UNSET else decision
    if sha is not None:
        block["sha256"] = sha
    if dec is not None:
        block["decision_sha256"] = dec
    (s.model.parent / REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": EMBEDDING_CONVERT_SCHEMA,
                "graft": {"sha256": hash_file(s.model), "weights": "int4"},
                "checkpoint": block,
            }
        )
    )


def _not_tied(manifest: dict[str, Any]) -> list[str]:
    return [t for t in manifest["limitations"] if "not tied to the reference" in t]


def test_the_weight_mode_is_read_from_the_convert_report_of_this_file(fake, tmp_path):
    s = _setup(tmp_path)
    _write_convert(s)
    result = _verify(s)
    assert result.status is Status.PASSED
    assert result.manifest["embedding"]["weights"] == "int4"
    assert result.manifest["embedding"]["checkpoint_sha256"] == hash_file(
        s.reference / "model.safetensors"
    )
    assert _not_tied(result.manifest) == []


def test_a_file_converted_from_another_checkpoint_is_refused(fake, tmp_path):
    s = _setup(tmp_path)
    _write_convert(s, checkpoint="sha256:" + "0" * 64)
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == RECORD_CHECK
    assert "not converted from this reference" in failed[0]["detail"]
    assert fake.runs == []


def test_a_file_converted_for_another_decision_is_refused(fake, tmp_path):
    s = _setup(tmp_path)
    _write_convert(s, decision="sha256:" + "1" * 64)
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == RECORD_CHECK and "1111" in failed[0]["detail"]
    assert fake.runs == []


@pytest.mark.parametrize(
    "missing, said",
    [("checkpoint", "records no checkpoint sha256"), ("decision", "records no decision sha256")],
)
def test_a_convert_report_without_a_digest_leaves_the_file_untied(fake, tmp_path, missing, said):
    s = _setup(tmp_path)
    _write_convert(s, **{missing: None})
    result = _verify(s)
    assert result.status is Status.PASSED
    assert said in " ".join(_not_tied(result.manifest))


def test_a_reference_without_model_safetensors_leaves_the_file_untied(fake, tmp_path):
    s = _setup(tmp_path)
    _write_convert(s)
    (s.reference / "model.safetensors").unlink()
    result = _verify(s)
    assert result.status is Status.PASSED
    assert "model.safetensors to compare" in " ".join(_not_tied(result.manifest))


def test_without_a_convert_report_the_file_is_not_tied(fake, tmp_path):
    m = _verify(_setup(tmp_path)).manifest
    assert f"no {REPORT_NAME} beside" in " ".join(_not_tied(m))


def test_a_convert_report_that_is_not_an_object_is_no_report(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"this one")
    (tmp_path / REPORT_NAME).write_text("[1, 2]")
    facts, why = convert_facts(model)
    assert facts is None and "is not a JSON object" in why
    assert weights_of(model)[0] is None


def test_a_missing_model_is_refused_before_any_run(fake, tmp_path):
    s = _setup(tmp_path)
    _write_convert(s)
    s.model.unlink()
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == CANDIDATE_CHECK and "no such file" in failed[0]["detail"]
    assert fake.runs == []


def test_a_convert_report_of_another_file_is_not_believed(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"this one")
    (tmp_path / REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": EMBEDDING_CONVERT_SCHEMA,
                "graft": {"sha256": "sha256:00", "weights": "int8"},
            }
        )
    )
    weights, why = weights_of(model)
    assert weights is None and "another file" in why


def test_without_a_convert_report_the_weights_are_unknown_and_said(fake, tmp_path):
    m = _verify(_setup(tmp_path)).manifest
    assert m["embedding"]["weights"] is None
    assert any("weight mode of the file is not known" in t for t in m["limitations"])


def test_googles_file_gives_the_gain(fake, tmp_path):
    local = tmp_path / "google.litertlm"
    local.write_bytes(GOOGLE_BYTES)
    m = _verify(_setup(tmp_path), base_litertlm=local).manifest
    assert [r[0] for r in fake.runs] == ["reference", "candidate", "base", "evaluate"]
    gain = m["quality"]["gain_over_base"]["centroid_k10"]
    assert gain["mean"] > 0 and len(gain["draws"]) == DRAWS
    assert "candidate|base" in m["quality"]["cosine"]
    assert not any("gain over Google's file was not measured" in t for t in m["limitations"])


def test_googles_file_with_another_sha256_is_refused(fake, tmp_path):
    local = tmp_path / "google.litertlm"
    local.write_bytes(b"something else")
    result = _verify(_setup(tmp_path), base_litertlm=local)
    assert result.status is Status.FAILED_HARNESS
    assert fake.runs == []


def test_with_base_downloads_googles_file_in_the_export_env(fake, tmp_path):
    m = _verify(_setup(tmp_path), with_base=True).manifest
    assert fake.runs[0] == ("download", envs.EXPORT.name)
    assert "gain_over_base" in m["quality"]


@pytest.mark.parametrize("script", ["reference", "candidate", "evaluate"])
def test_a_script_that_fails_is_a_harness_failure_naming_it(fake, tmp_path, script):
    fake.fail = {script}
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert "boom" in failed[0]["detail"]


def _google(tmp_path: Path) -> Path:
    local = tmp_path / "google.litertlm"
    local.write_bytes(GOOGLE_BYTES)
    return local


def test_the_files_are_embedded_side_by_side_when_the_cpus_allow(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_verify, "usable_cpus", lambda: 8)
    monkeypatch.setattr(embedding_verify.os, "cpu_count", lambda: 1)  # not what is read
    m = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path)).manifest
    assert [launch["parallel"] for launch in fake.launches] == [True]
    assert [r["name"] for r in fake.launches[0]["runs"]] == ["candidate", "base"]
    assert fake.timeouts["engines"] == fake.timeouts["reference"]
    assert m["embedding"]["engines_side_by_side"] is True
    assert any("embedded at the same time" in t for t in m["limitations"])


def test_too_few_cpus_embed_the_files_one_after_another(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_verify, "usable_cpus", lambda: 7)
    monkeypatch.setattr(embedding_verify.os, "cpu_count", lambda: 64)  # not what is read
    m = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path)).manifest
    assert [launch["parallel"] for launch in fake.launches] == [False]
    # One after another, the launcher has a file's allowance per file.
    assert fake.timeouts["engines"] == 2 * fake.timeouts["reference"]
    assert m["embedding"]["engines_side_by_side"] is False
    assert not any("embedded at the same time" in t for t in m["limitations"])


def test_googles_file_failing_is_a_limitation_and_the_candidate_is_still_gated(fake, tmp_path):
    fake.fail = {"base"}
    result = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path))
    m = result.manifest
    assert result.status is Status.PASSED, m["checks"]
    checks = _checks(m)
    assert checks[BASE_EMBED_CHECK]["outcome"] == Outcome.UNCHECKED.value
    assert "boom" in checks[BASE_EMBED_CHECK]["detail"]
    assert checks[GATE_CHECK]["outcome"] == "passed"
    notes = " ".join(m["limitations"])
    assert "Google's file could not be embedded" in notes and "boom" in notes
    assert "gain_over_base" not in m["quality"]
    assert "base" not in m["measurements"] and "base" not in m["harness"]
    assert set(fake.specs["evaluate"]["vectors"]) == {"reference", "candidate"}
    assert fake.specs["evaluate"]["pairs"] == [["reference", "candidate"]]


def test_googles_file_refusing_its_vectors_is_a_limitation(fake, tmp_path):
    fake.refuse = {"base": "3 of 54 engine vectors have a non-finite value or a zero norm"}
    result = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path))
    assert result.status is Status.PASSED
    assert "non-finite" in " ".join(result.manifest["limitations"])


def test_the_failure_reported_is_the_side_that_failed_not_the_one_it_stopped(fake, tmp_path):
    fake.fail = {"candidate"}
    result = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    assert [r[0] for r in fake.runs] == ["reference", "candidate"]  # base stopped
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert [c["name"] for c in failed] == [CANDIDATE_CHECK]
    assert "boom" in failed[0]["detail"]
    assert BASE_EMBED_CHECK not in _checks(result.manifest)


def test_a_candidate_only_stopped_is_still_a_harness_failure(fake, tmp_path):
    fake.stopped = {"candidate"}
    result = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == CANDIDATE_CHECK and "stopped" in failed[0]["detail"]


def test_googles_file_only_stopped_is_a_limitation(fake, tmp_path):
    fake.stopped = {"base"}
    result = _verify(_setup(tmp_path), base_litertlm=_google(tmp_path))
    assert result.status is Status.PASSED
    assert "stopped" in _checks(result.manifest)[BASE_EMBED_CHECK]["detail"]


@pytest.mark.parametrize(
    "side, check", [("reference", REFERENCE_CHECK), ("candidate", CANDIDATE_CHECK)]
)
def test_unusable_vectors_on_a_side_are_a_harness_failure_of_that_side(fake, tmp_path, side, check):
    said = "2 of 54 vectors have a non-finite value or a zero norm (rows [3, 7])"
    fake.refuse = {side: said}
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == check and said in failed[0]["detail"]
    assert "evaluate" not in [r[0] for r in fake.runs]


def test_a_side_reporting_another_count_of_vectors_is_refused(fake, tmp_path):
    fake.rows_off = {"candidate"}
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == CANDIDATE_CHECK
    assert "reports 55 vectors for 54 texts" in failed[0]["detail"]


def test_a_side_that_writes_no_vectors_file_is_refused_and_said(fake, tmp_path):
    fake.no_npy = {"candidate"}
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    assert "wrote no candidate.npy" in json.dumps(result.manifest["checks"])


def test_reference_rows_over_max_seq_length_are_refused(fake, tmp_path):
    fake.over_max_seq_length = 3
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == REFERENCE_CHECK
    assert "3 of 54 texts exceed the reference's max_seq_length of 128" in failed[0]["detail"]
    assert "litetune prepare --tokenizer" in failed[0]["detail"]
    assert [r[0] for r in fake.runs] == ["reference"]


def test_a_reference_report_without_token_counts_is_refused(fake, tmp_path):
    fake.over_max_seq_length = None
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    assert "reports no token counts" in json.dumps(result.manifest["checks"])


def _set_all(rows: list[int], value: int) -> list[int]:
    return [value] * len(rows)


def test_the_gate_reads_the_ten_shot_decider_not_the_five_shot(fake, tmp_path):
    def worse_k5(out: dict[str, Any]) -> None:
        cand = out["correct"]["candidate"]
        cand["centroid_k5"] = [_set_all(d, 0) for d in cand["centroid_k5"]]

    fake.edit_eval = worse_k5
    result = _verify(_setup(tmp_path))
    assert result.status is Status.PASSED
    assert result.manifest["quality"]["conversion_cost"]["centroid_k10"]["mean"] == 0.0


def test_the_all_rows_cost_reads_the_all_rows_centroid(fake, tmp_path):
    def worse_centroid(out: dict[str, Any]) -> None:
        cand = out["correct"]["candidate"]
        cand["centroid"] = _set_all(cand["centroid"], 0)

    fake.edit_eval = worse_centroid
    result = _verify(_setup(tmp_path))
    cost = result.manifest["quality"]["conversion_cost"]
    assert cost["centroid"]["value"] == 1.0
    assert cost["centroid_k10"]["mean"] == 0.0
    assert result.status is Status.PASSED  # not gated


@pytest.mark.parametrize(
    "files, threads, cpus, expected",
    [
        (["a"], 4, 64, False),
        (["a", "b"], 4, 8, True),
        (["a", "b"], 4, 7, False),
        (["a", "b"], 4, None, False),
    ],
)
def test_side_by_side_needs_two_files_and_the_cpus_for_both(files, threads, cpus, expected):
    assert side_by_side([Path(f) for f in files], threads, cpus) is expected


def test_usable_cpus_prefers_the_process_count(monkeypatch):
    monkeypatch.setattr(os, "process_cpu_count", lambda: 3, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert usable_cpus() == 3


def test_usable_cpus_falls_back_to_the_affinity_mask(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert usable_cpus() == 2


def test_usable_cpus_falls_back_to_the_cpu_count(monkeypatch):
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 5)
    assert usable_cpus() == 5


# A stand-in for ENGINE_SCRIPT: records when it ran, sleeps, exits as told;
# with "ignore_term" it does not stop on SIGTERM.
_STAND_IN = r"""
import json, signal, sys, time
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
if spec.get("ignore_term"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(spec["mark"]).write_text(str(time.time()))
time.sleep(spec["sleep"])
Path(spec["mark"]).write_text(Path(spec["mark"]).read_text() + " " + str(time.time()))
sys.exit(spec["code"])
"""


def _launch(
    tmp_path: Path,
    parallel: bool,
    runs: dict[str, tuple[float, int]],
    *,
    stop_on: tuple[str, ...] = STOP_ON,
    stop_s: float = 5.0,
    stderr: dict[str, Path] | None = None,
    ignore_term: frozenset[str] = frozenset(),
    timeout: float = 60,
    expect_exit: int | None = 0,
):
    script = tmp_path / "stand_in.py"
    script.write_text(_STAND_IN)
    entries = []
    for name, (sleep, code) in runs.items():
        spec = tmp_path / f"{name}.json"
        spec.write_text(
            json.dumps(
                {
                    "mark": str(tmp_path / f"{name}.mark"),
                    "sleep": sleep,
                    "code": code,
                    "ignore_term": name in ignore_term,
                }
            )
        )
        err = (stderr or {}).get(name, tmp_path / f"{name}.err")
        entries.append({"name": name, "spec": str(spec), "stderr": str(err)})
    launcher_spec = tmp_path / "engines.json"
    report = tmp_path / "engines.out"
    launcher_spec.write_text(
        json.dumps(
            {
                "script": str(script),
                "runs": entries,
                "parallel": parallel,
                "stop_on": list(stop_on),
                "stop_s": stop_s,
                "report": str(report),
            }
        )
    )
    launcher = tmp_path / "engines.py"
    launcher.write_text(ENGINES_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(launcher), str(launcher_spec)],
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if expect_exit is None:
        return proc
    assert proc.returncode == expect_exit, proc.stderr
    return json.loads(report.read_text())["runs"]


def _span(tmp_path: Path, name: str) -> tuple[float, float]:
    start, end = (tmp_path / f"{name}.mark").read_text().split()
    return float(start), float(end)


def _finished(tmp_path: Path, name: str) -> bool:
    mark = tmp_path / f"{name}.mark"
    return mark.is_file() and len(mark.read_text().split()) == 2


def test_the_launcher_runs_side_by_side_when_told(tmp_path):
    runs = _launch(tmp_path, True, {"candidate": (1.5, 0), "base": (1.5, 0)})
    assert {r["returncode"] for r in runs.values()} == {0}
    (a0, a1), (b0, b1) = _span(tmp_path, "candidate"), _span(tmp_path, "base")
    assert b0 < a1 and a0 < b1  # the two runs overlapped


def test_the_launcher_runs_one_after_another_when_told(tmp_path):
    _launch(tmp_path, False, {"candidate": (0.3, 0), "base": (0.3, 0)})
    assert _span(tmp_path, "candidate")[1] <= _span(tmp_path, "base")[0]


def test_a_failed_critical_run_stops_the_one_beside_it(tmp_path):
    runs = _launch(tmp_path, True, {"base": (30, 0), "candidate": (0, 2)})
    assert runs["candidate"]["returncode"] == 2
    assert runs["base"]["returncode"] is None
    assert not _finished(tmp_path, "base")


def test_a_critical_run_failing_first_while_another_is_live_is_reported(tmp_path):
    # The candidate, started first, fails while the base still runs: the
    # base is stopped once and the launcher still writes its report.
    runs = _launch(tmp_path, True, {"candidate": (0, 1), "base": (30, 0)})
    assert runs["candidate"]["returncode"] == 1
    assert runs["base"] == {"returncode": None, "seconds": None}
    assert not _finished(tmp_path, "base")


def test_a_run_not_named_in_stop_on_does_not_stop_the_others(tmp_path):
    runs = _launch(tmp_path, True, {"candidate": (1.0, 0), "base": (0, 2)})
    assert runs["base"]["returncode"] == 2
    assert runs["candidate"]["returncode"] == 0
    assert _finished(tmp_path, "candidate")


def test_a_run_not_named_in_stop_on_does_not_keep_the_next_from_starting(tmp_path):
    runs = _launch(tmp_path, False, {"base": (0, 2), "candidate": (0, 0)})
    assert runs["base"]["returncode"] == 2 and runs["candidate"]["returncode"] == 0


def test_a_failed_critical_run_keeps_the_next_from_starting(tmp_path):
    runs = _launch(tmp_path, False, {"candidate": (0, 1), "base": (0, 0)})
    assert runs == {
        "candidate": {"returncode": 1, "seconds": runs["candidate"]["seconds"]},
        "base": {"returncode": None, "seconds": None},
    }
    assert not (tmp_path / "base.mark").exists()


def test_a_run_still_going_does_not_outlive_a_launcher_that_failed(tmp_path):
    # The second run cannot be started (its stderr is in no directory): the
    # launcher fails, and the first run is stopped with it, not orphaned.
    sleep = 2.0
    proc = _launch(
        tmp_path,
        True,
        {"candidate": (sleep, 0), "base": (0, 0)},
        stderr={"base": tmp_path / "missing" / "base.err"},
        expect_exit=None,
    )
    assert proc.returncode != 0 and "FileNotFoundError" in proc.stderr
    time.sleep(sleep + 1.5)
    assert not _finished(tmp_path, "candidate")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM cannot be ignored on Windows")
def test_a_run_that_ignores_termination_is_killed(tmp_path):
    began = time.monotonic()
    runs = _launch(
        tmp_path,
        True,
        {"candidate": (1.0, 1), "base": (30, 0)},
        stop_s=0.5,
        ignore_term=frozenset({"base"}),
        timeout=20,
    )
    assert time.monotonic() - began < 15
    assert runs["base"]["returncode"] is None
    assert not _finished(tmp_path, "base")


def test_too_few_training_rows_for_the_gated_k_is_refused_before_any_run(fake, tmp_path):
    result = _verify(_setup(tmp_path, train_per_label=9))
    assert result.status is Status.FAILED_HARNESS
    assert fake.runs == []
    assert "fewer than 10 training rows" in json.dumps(result.manifest["checks"])


@pytest.mark.parametrize(
    "kwargs, flag",
    [
        ({"calibration": Path("calibration.jsonl")}, "--calibration"),
        ({"float_twin": Path("twin.litertlm")}, "--float-twin"),
        ({"scorer": "exact-text"}, "--scorer"),
        ({"engine_rows": 7}, "--engine-rows"),
    ],
)
def test_a_flag_of_another_path_is_refused_not_ignored(fake, tmp_path, kwargs, flag):
    result = _verify(_setup(tmp_path), **kwargs)
    assert result.status is Status.FAILED_HARNESS
    assert flag in json.dumps(result.manifest["checks"])
    assert fake.runs == []


def test_the_gpu_backend_is_refused(fake, tmp_path):
    assert _verify(_setup(tmp_path), backend="gpu").status is Status.FAILED_HARNESS


def test_a_record_trained_with_another_prefix_is_refused(fake, tmp_path):
    s = _setup(tmp_path, record={"prefix": "task: classification | query: "})
    assert _verify(s).status is Status.FAILED_HARNESS
    assert fake.runs == []


def test_a_decision_with_a_none_option_is_refused(fake, tmp_path):
    decision = _decision(none_option="other")
    s = _setup(
        tmp_path,
        record={"decision": decision.as_dict(), "decision_sha256": decision.digest()},
    )
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    assert "none_option" in json.dumps(result.manifest["checks"])


def test_a_malformed_embedding_record_is_a_harness_failure(fake, tmp_path):
    s = _setup(tmp_path)
    (s.reference / "litetune.json").write_text(json.dumps({"direction": "embedding"}))
    assert _verify(s).status is Status.FAILED_HARNESS
    assert fake.runs == []


def test_embedding_flags_on_another_checkpoint_are_refused(fake, tmp_path):
    s = _setup(tmp_path)
    other = tmp_path / "llm"
    other.mkdir()
    result = run_verify(
        VerifyRequest(model=s.model, reference=str(other), data=s.heldout, with_base=True)
    )
    assert result.status is Status.FAILED_HARNESS
    assert "--with-base" in json.dumps(result.manifest["checks"])


@pytest.mark.parametrize(
    "record, kwargs, said",
    [
        ({"base_model_revision": "0" * 40}, {}, f"not {EMBEDDING_MODEL} at {EMBEDDING_REVISION}"),
        ({"prompt_name": "query"}, {}, "the decider's prefix"),
        ({}, {"reference_role": ReferenceRole.UNTUNED_BASE}, "--reference-role untuned_base"),
        ({}, {"limit": 5}, "--limit does not apply"),
        ({}, {"max_conversion_cost": 1.0}, "--max-conversion-cost 1.0 is not in [0, 1)"),
        ({}, {"max_conversion_cost": -0.01}, "--max-conversion-cost -0.01 is not in [0, 1)"),
        (
            {"labels": ("balance", "card_lost", "other")},
            {},
            "are not the decision's options",
        ),
        (
            {"decision_sha256": "sha256:" + "2" * 64},
            {},
            "records a decision whose digest is not its own",
        ),
    ],
)
def test_a_request_or_record_that_cannot_be_measured_is_refused_before_any_run(
    fake, tmp_path, record, kwargs, said
):
    result = _verify(_setup(tmp_path, record=record), **kwargs)
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert said in failed[0]["detail"], failed
    assert fake.runs == []


def test_a_held_out_split_with_the_training_splits_content_is_refused(fake, tmp_path):
    s = _setup(tmp_path)
    s.heldout.write_text(s.train.read_text(encoding="utf-8"), encoding="utf-8")
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == ROWS_CHECK and "are the same content" in failed[0]["detail"]
    assert fake.runs == []


def test_a_training_split_other_than_tunes_is_a_limitation_naming_both(fake, tmp_path):
    s = _setup(tmp_path, record={"train_sha256": "sha256:" + "3" * 64})
    result = _verify(s)
    assert result.status is Status.PASSED
    notes = " ".join(result.manifest["limitations"])
    assert hash_file(s.train) in notes and "sha256:" + "3" * 64 in notes


@pytest.mark.parametrize("bind", [True, False])
def test_the_training_split_tune_read_or_none_recorded_says_nothing(fake, tmp_path, bind):
    s = _setup(tmp_path, bind_train=bind)
    notes = " ".join(_verify(s).manifest["limitations"])
    assert "the one tune read" not in notes


def test_an_explicit_train_split_is_used(fake, tmp_path):
    s = _setup(tmp_path)
    moved = tmp_path / "elsewhere.jsonl"
    moved.write_text(s.train.read_text(encoding="utf-8"), encoding="utf-8")
    s.train.unlink()
    assert _verify(s, train_data=moved).manifest["data"]["train"]["path"] == str(moved)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_command_line_exits_with_the_verdict_and_prints_it(fake, tmp_path, capsys):
    s = _setup(tmp_path)
    argv = ["verify", "--model", str(s.model), "--reference", str(s.reference)]
    assert main([*argv, "--data", str(s.heldout)]) == 0
    out = capsys.readouterr().out
    assert "status: passed" in out
    assert "centroid_k10 mean 1.0" in out


def test_the_command_line_json_is_the_manifest(fake, tmp_path, capsys):
    s = _setup(tmp_path)
    fake.candidate = "random"
    code = main(
        [
            "verify",
            "--model",
            str(s.model),
            "--reference",
            str(s.reference),
            "--data",
            str(s.heldout),
            "--train-data",
            str(s.train),
            "--threads",
            "2",
            "--json",
        ]
    )
    assert code == 1
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["status"] == "failed_gate"
    assert manifest["embedding"]["threads"] == 2
    assert fake.specs["candidate"]["threads"] == 2
