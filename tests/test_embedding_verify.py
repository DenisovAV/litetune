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
    DEFAULT_MAX_CONVERSION_COST,
    DRAWS,
    ENGINE_SCRIPT,
    EVAL_SCRIPT,
    GATE_CHECK,
    GATED_K,
    KS,
    REFERENCE_SCRIPT,
    SCORER,
    centroid_cost,
    gate,
    kshot_costs,
    labels_short_of,
    weights_of,
)
from litetune.metrics import Difference, paired_difference
from litetune.verify import Status, VerifyRequest, run_verify

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


@pytest.mark.parametrize("source", [REFERENCE_SCRIPT, ENGINE_SCRIPT, EVAL_SCRIPT])
def test_every_script_compiles(source):
    compile(source, "script.py", "exec")


def test_the_engine_script_uses_the_measured_engine_call():
    # emb_s5.py's call: EmbeddingEngine on Backend.CPU(thread_count=...),
    # EmbeddingOptions(normalize=True), the prefix in front of the text.
    assert 'litert_lm.EmbeddingEngine(spec["model"], backend=backend)' in ENGINE_SCRIPT
    assert 'litert_lm.Backend.CPU(thread_count=spec["threads"])' in ENGINE_SCRIPT
    assert "litert_lm.EmbeddingOptions(normalize=True)" in ENGINE_SCRIPT
    assert 'spec["prefix"] + text' in ENGINE_SCRIPT


def test_the_reference_script_encodes_as_emb_s5_does():
    assert 'prompt_name=spec["prompt_name"]' in REFERENCE_SCRIPT
    assert "normalize_embeddings=True" in REFERENCE_SCRIPT
    assert 'device="cpu"' in REFERENCE_SCRIPT
    assert '"dtype": torch.float32' in REFERENCE_SCRIPT


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

    def __call__(self, env, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        name = Path(args[1]).stem
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
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
        if name in ("reference", "candidate", "base"):
            texts = json.loads(Path(spec["texts"]).read_text(encoding="utf-8"))
            self.texts[name] = texts
            kind = self.candidate if name == "candidate" else "same"
            if name == "base":
                kind = "random"
            v = self.vectors(texts, kind)
            np.save(spec["out"], v.astype(np.float32))
            report.write_text(
                json.dumps(
                    {
                        "rows": len(texts),
                        "dim": 6,
                        "runtime": {"litert-lm": "0.18.0"},
                        "cpu": "fake",
                        "ms_median": 1.0,
                    }
                ),
                encoding="utf-8",
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


def _write_split(path: Path, decision: Decision, per_label: int) -> Path:
    rows = [(f"message {i} about {o}", o) for i in range(per_label) for o in OPTIONS]
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
    tmp_path: Path, *, train_per_label: int = 12, record: dict[str, Any] | None = None
) -> Setup:
    decision = _decision()
    split = tmp_path / "split"
    split.mkdir()
    train = _write_split(split / "train.jsonl", decision, train_per_label)
    heldout = _write_split(split / "heldout.jsonl", decision, 6)
    reference = tmp_path / "tuned" / "model"
    reference.mkdir(parents=True)
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
    assert "k=5 and k=10" in notes
    assert "gain over Google's file was not measured" in notes
    assert "the gate used 0.01" in notes


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


def test_both_sides_embed_the_states_not_the_rendered_prompts(fake, tmp_path):
    _verify(_setup(tmp_path))
    texts = fake.texts["candidate"]
    assert texts[0] == "message 0 about balance"
    assert len(texts) == 36 + 18
    assert fake.texts["reference"] == texts


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


def test_the_weight_mode_is_read_from_the_convert_report_of_this_file(fake, tmp_path):
    s = _setup(tmp_path)
    digest = "sha256:" + hashlib.sha256(s.model.read_bytes()).hexdigest()
    (s.model.parent / REPORT_NAME).write_text(
        json.dumps(
            {"schema": EMBEDDING_CONVERT_SCHEMA, "graft": {"sha256": digest, "weights": "int4"}}
        )
    )
    assert _verify(s).manifest["embedding"]["weights"] == "int4"


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
