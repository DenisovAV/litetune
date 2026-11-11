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
from collections.abc import Callable, Sequence
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
    IMAGE_PREFIX,
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    EmbeddingTaskRecord,
    LabelledText,
    write_embedding_record,
)
from litetune.embedding_verify import (
    BASE_CHECK,
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
from litetune.embedding_vision import VISION_FILE
from litetune.metrics import Difference, paired_difference
from litetune.storage import hash_file
from litetune.verify import ReferenceRole, Status, VerifyRequest, run_verify

np: Any = pytest.importorskip("numpy")

OPTIONS = ("balance", "card_lost", "refund")
GOOGLE_BYTES = b"a stand-in for Google's file"
VISION_STAND_IN = b"a stand-in for Google's text-vision file"


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


def test_the_crossfit_threshold_holds_each_rank_out_against_the_other_folds(ev):
    # Two options, two inputs each (ranks 0 and 1, so folds 0 and 1). Worked by
    # hand: fold 0's inputs against the rank-1 centroids score 0.6 and 0.96,
    # fold 1's against the rank-0 centroids 0.8 and 0.96.
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.28, 0.96]])
    y = np.array([0, 1, 0, 1])
    assert ev["crossfit_threshold"](x, y, 2, 0.95) == pytest.approx(0.6)
    assert ev["crossfit_threshold"](x, y, 2, 0.75) == pytest.approx(0.8)
    assert ev["crossfit_threshold"](x, y, 2, 0.5) == pytest.approx(0.96)


def test_the_crossfit_ranks_follow_the_order_the_inputs_are_given(ev):
    # Worked by hand: the signals are 0.6, 0.936, 0.96 and 0.96, where the
    # order of the previous test gave 0.6, 0.8, 0.96 and 0.96.
    x = np.array([[1.0, 0.0], [0.28, 0.96], [0.6, 0.8], [0.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    assert ev["crossfit_threshold"](x, y, 2, 0.75) == pytest.approx(0.936)


def test_the_crossfit_has_the_apps_five_folds(ev):
    # Six inputs of one option: with five folds rank 5 shares fold 0 with rank
    # 0, so input 5 is held out together with input 0 and scored against ranks
    # 1-4 only. Any other fold count groups them otherwise.
    assert ev["CROSSFIT_FOLDS"] == 5
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    b = np.array([[-1.0, 0.0]] * 6)
    x, y = np.vstack([a, b]), np.array([0] * 6 + [1] * 6)
    # Ranks 0 and 5 of option 0 ([1, 0]) against the mean of ranks 1-4 ([0, 1]):
    # cosine 0 to option 0, -1 to option 1, so their signal is 0; every other
    # input of option 0 is 1 or above 0. Option 1's inputs are all 1.
    assert ev["crossfit_threshold"](x, y, 2, 0.95) == pytest.approx(0.0)


def test_the_crossfit_centroids_are_renormalised_as_the_apps_are(ev):
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.6, 0.8], [0.0, 1.0], [0.28, 0.96]])
    y = np.array([0, 1, 0, 1, 0, 1])
    c = ev["centroids"](x, y, 2)
    assert np.linalg.norm(c, axis=1) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "x, y, why",
    [
        # One input of option 1: the app's crossFit needs two.
        ([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], [0, 0, 1], "fewer than two inputs"),
        # Option 0's two inputs are opposite: their centroid has no direction.
        ([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], [0, 0, 1, 1], "cancel out"),
    ],
)
def test_the_crossfit_refuses_what_the_app_refuses(ev, x, y, why):
    with pytest.raises(ev["Uncalibrated"], match=why):
        ev["abstention"](np.array(x), np.array(y), np.array([[1.0, 0.0]]), np.array([0]), 2, 0.95)


def test_keep_threshold_picks_the_apps_index_including_the_float_edge(ev):
    signal = np.arange(10, dtype=float)
    # (1 - 0.9) * 10 is 0.9999999999999998; the app's + 1e-9 makes it index 1.
    assert ev["keep_threshold"](signal, 0.9) == 1.0
    assert ev["keep_threshold"](signal, 0.95) == 0.0
    assert ev["keep_threshold"](signal, 1.0) == 0.0
    assert ev["keep_threshold"](np.array([0.3]), 0.5) == 0.3


def test_a_row_at_the_threshold_is_answered_and_the_next_float_below_is_not(ev, monkeypatch):
    # Centroids [1, 0] and [0, 1] (exact in float32); a threshold of 0.8 set
    # directly. The first row's best cosine is 0.8, the second's the float
    # just below it.
    xs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    ys = np.array([0, 0, 1, 1])
    monkeypatch.setitem(ev, "crossfit_threshold", lambda *a: 0.8)
    monkeypatch.setitem(ev, "unit_rows", lambda x: np.asarray(x, dtype=np.float64))
    under = np.nextafter(0.8, 0.0)
    xte = np.array([[0.6, 0.8], [np.sqrt(1 - under * under), under]])
    t, answered, right = ev["abstention"](xs, ys, xte, np.array([1, 1]), 2, 0.95)
    assert t == 0.8
    assert answered.tolist() == [True, False]
    assert right.tolist() == [True, True]


def test_requests_are_scored_against_float32_centroids_as_the_app_stores_them(ev, monkeypatch):
    # A centroid whose float32 rounding moves a request's cosine across the
    # threshold: answered against the float32 centroid, which is what the app
    # scores against, whatever float64 would say.
    c64 = np.array([0.6000000001, 0.8], dtype=np.float64)
    c64 /= np.linalg.norm(c64)
    c32 = c64.astype(np.float32).astype(np.float64)
    req = c32  # the request points exactly along the stored centroid
    monkeypatch.setitem(ev, "centroids", lambda x, y, n: np.vstack([c64, [-c64[1], c64[0]]]))
    monkeypatch.setitem(ev, "crossfit_threshold", lambda *a: float(req @ c32))
    # The request as given: renormalising it would move it off the centroid.
    monkeypatch.setitem(ev, "unit_rows", lambda x: np.asarray(x, dtype=np.float64))
    _, answered, _ = ev["abstention"](
        np.eye(2), np.array([0, 1]), req[None, :], np.array([0]), 2, 0.95
    )
    assert float(req @ c64) < float(req @ c32)  # float64 alone would refuse it
    assert answered.tolist() == [True]


def test_rows_are_renormalised_before_the_crossfit_as_the_app_does(ev):
    # The app renormalises every engine vector (unitRowInto); rows a little
    # off unit length must give what their unit rows give.
    rng = np.random.default_rng(3)
    ytr = np.array([0, 1, 2] * 5)
    v = _clusters(rng, [*ytr.tolist(), 0, 1, 2, 0], noise=0.5)
    xs, xte, yte = v[:15], v[15:], np.array([0, 1, 2, 0])
    scale = rng.uniform(0.5, 2.0, size=(len(v), 1))
    unit = ev["abstention"](xs, ytr, xte, yte, 3, 0.95)
    scaled = ev["abstention"](xs * scale[:15], ytr, xte * scale[15:], yte, 3, 0.95)
    assert scaled[0] == pytest.approx(unit[0])
    assert scaled[1].tolist() == unit[1].tolist() and scaled[2].tolist() == unit[2].tolist()


def test_evaluate_reports_the_crossfit_abstention_per_k(ev):
    rng = np.random.default_rng(5)
    ytr = np.array([0, 1, 2] * 12)
    yte = np.array([0, 1, 2] * 6)
    v = _clusters(rng, [*ytr.tolist(), *yte.tolist()], noise=0.6)
    acc, _ = ev["evaluate"](v, ytr, yte, 3, (5, 10), 3, None, 0.95)
    masked = False
    for k in (5, 10):
        got = acc[f"crossfit_k{k}"]
        assert got["keep"] == 0.95 and got["folds"] == 5 and len(got["threshold"]) == 3
        for d in range(3):
            idx = ev["kshot_picks"](ytr, 3, k, d)
            xs, xte = v[:36][idx], v[36:]
            t, answered, right = ev["abstention"](xs, ytr[idx], xte, yte, 3, 0.95)
            # `right` checked on its own: the decider's answer is the nearest centroid.
            assert right.tolist() == ev["centroid_correct"](xs, ytr[idx], xte, yte, 3).tolist()
            assert got["threshold"][d] == round(t, 6)
            assert got["answered"]["draws"][d] == round(float(np.mean(answered)), 4)
            assert got["right_when_answered"]["draws"][d] == round(
                float(np.mean(right[answered])), 4
            )
            assert got["answered_and_right"]["draws"][d] == round(
                float(np.mean(answered & right)), 4
            )
            masked = masked or (not answered.all() and np.mean(right[answered]) != np.mean(right))
    assert masked, "the fixture must tell right-when-answered from plain accuracy"
    plain, _ = ev["evaluate"](v, ytr, yte, 3, (5,), 2)
    assert "crossfit_k5" not in plain


def test_a_draw_that_answers_nothing_reports_no_right_when_answered(ev, monkeypatch):
    rng = np.random.default_rng(5)
    ytr = np.array([0, 1, 2] * 12)
    yte = np.array([0, 1, 2] * 6)
    v = _clusters(rng, [*ytr.tolist(), *yte.tolist()])
    monkeypatch.setitem(ev, "crossfit_threshold", lambda *a: 2.0)  # above every cosine
    acc, _ = ev["evaluate"](v, ytr, yte, 3, (5,), 2, None, 0.95)
    got = acc["crossfit_k5"]
    assert got["answered"]["draws"] == [0.0, 0.0]
    assert got["right_when_answered"] == {
        "mean": None,
        "min": None,
        "draws": [None, None],
        "missing": 2,
    }


def test_a_crossfit_the_app_would_refuse_is_reported_not_raised(ev, monkeypatch):
    rng = np.random.default_rng(5)
    ytr = np.array([0, 1, 2] * 12)
    yte = np.array([0, 1, 2] * 6)
    v = _clusters(rng, [*ytr.tolist(), *yte.tolist()])

    def refuse(*a):
        raise ev["Uncalibrated"]("option 1's inputs cancel out")

    monkeypatch.setitem(ev, "crossfit_threshold", refuse)
    acc, correct = ev["evaluate"](v, ytr, yte, 3, (5,), 2, None, 0.95)
    assert acc["crossfit_k5"]["error"] == "draw 0: option 1's inputs cancel out"
    # The gated decider is untouched.
    assert acc["centroid_k5"]["mean"] == 1.0 and len(correct["centroid_k5"]) == 2


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
                "keep": 0.95,
                "pairs": [["reference", "candidate"]],
                "report": str(report),
            }
        )
    )
    stub: Any = types.ModuleType("torch")
    stub.__version__ = "stub"
    monkeypatch.setitem(sys.modules, "torch", stub)
    monkeypatch.setitem(ev, "linear_correct", lambda torch, xtr, ytr, xte, yte, n, d: yte == yte)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(spec)])
    assert ev["main"]() == 0
    out = json.loads(report.read_text())
    assert out["accuracy"]["candidate"]["linear"] == 1.0
    assert out["accuracy"]["reference"]["centroid_k10"]["mean"] == 1.0
    assert len(out["correct"]["candidate"]["centroid_k10"]) == 5
    # The real script fits the app's crossFit: keep from the spec, five folds.
    assert out["accuracy"]["candidate"]["crossfit_k10"]["keep"] == 0.95
    assert out["accuracy"]["candidate"]["crossfit_k10"]["folds"] == 5
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
                "keep": 0.95,
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
                "keep": 0.95,
                "pairs": pairs,
                "report": str(report),
            }
        )
    )
    stub: Any = types.ModuleType("torch")
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


@pytest.mark.parametrize(
    "source",
    [REFERENCE_SCRIPT, ENGINE_SCRIPT, ENGINES_SCRIPT, EVAL_SCRIPT],
    ids=["reference", "engine", "engines", "evaluation"],
)
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
    calls: dict[str, list[Any]] = {"engine": [], "options": [], "texts": [], "vision": []}
    mod: Any = types.ModuleType("litert_lm")
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
            calls["vision"].append(
                {
                    "vision_backend": vision_backend,
                    "vision_tokens_per_image": vision_tokens_per_image,
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

    @dataclass
    class ImageFile:
        absolute_path: str

    # litert_lm 0.18.0's `ImageBytes` is a dataclass of one field, `bytes`.
    @dataclass
    class ImageBytes:
        bytes: bytes

    mod.Content = types.SimpleNamespace(ImageFile=ImageFile, ImageBytes=ImageBytes)
    mod.Backend = Backend
    mod.InputOverflowStrategy = InputOverflowStrategy
    mod.EmbeddingOptions = EmbeddingOptions
    mod.EmbeddingEngine = EmbeddingEngine
    return mod, calls


def _engine_main(tmp_path, monkeypatch, answers, texts, backend=None, stderr=None):
    ns: dict[str, Any] = {"__name__": "engine_script"}
    exec(compile(ENGINE_SCRIPT, "engine.py", "exec"), ns)
    stub, calls = _stub_litert_lm(answers)
    monkeypatch.setitem(sys.modules, "litert_lm", stub)
    # `backend_for` imports these two from litert_lm.interfaces.
    interfaces: Any = types.ModuleType("litert_lm.interfaces")
    interfaces.CPU = lambda: ("cpu", None)
    interfaces.GPU = lambda: ("gpu", None)
    monkeypatch.setitem(sys.modules, "litert_lm.interfaces", interfaces)
    monkeypatch.setitem(ns, "cpu_report", lambda: {"model": "stub cpu", "count": 2})
    readings = iter([{"reading": "at engine"}, {"reading": "after"}])
    monkeypatch.setitem(ns, "device_report", lambda: next(readings))
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
    if backend is not None:
        spec["backend"] = backend
    if stderr is not None:
        (tmp_path / "side.err").write_text(stderr, encoding="utf-8")
        spec["stderr"] = str(tmp_path / "side.err")
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
    # The CPU run does not ask the kernel about the GPU.
    assert report["device"] == {}


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

    torch: Any = types.ModuleType("torch")
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

    st: Any = types.ModuleType("sentence_transformers")
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


def _reading(time_: int) -> dict[str, Any]:
    return {
        "platform": "darwin",
        "looked": True,
        "gpu_client": "pid 1, python3",
        "gpu_time": time_,
        "gpu_clients": {"0x1": time_},
    }


# The kernel's two readings around a run that worked on the GPU, and one that did not.
GPU_WORKED = {"at_engine": _reading(10), "after_generation": _reading(500)}
GPU_IDLE = {"at_engine": _reading(10), "after_generation": _reading(10)}
NOT_LOOKED = {
    "at_engine": {"platform": "linux", "looked": False},
    "after_generation": {"platform": "linux", "looked": False},
}


@dataclass
class FakeEmbedEnvs:
    """`StageEnv.run` for the embedding, evaluation and download scripts."""

    ev: dict[str, Any]
    candidate: str = "same"  # "same": the reference's vectors; "random": unrelated ones
    fail: set[str] = field(default_factory=set)
    runs: list[tuple[str, str]] = field(default_factory=list)
    provisioned: list[str] = field(default_factory=list)
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
    # What an engine side on the GPU reports from the kernel; the default shows GPU work.
    gpu_device: dict[str, Any] | None = None
    # The same for one side only, by its name.
    gpu_device_for: dict[str, dict[str, Any]] = field(default_factory=dict)
    # What an engine side writes to its stderr, by its name: where the runtime
    # names the adapter it selected.
    stderr_for: dict[str, str] = field(default_factory=dict)
    # An image split's files by absolute path, and the label each was written with.
    image_labels: dict[str, str] = field(default_factory=dict)
    # The soft tokens the fake reference makes of a picture, by path; else 63.
    soft_tokens: dict[str, int] = field(default_factory=dict)
    vision_backend_constraints: dict[str, str] = field(default_factory=dict)
    vision_backend_reason: str | None = None
    # What the candidate's second run embeds; None: what the candidate embeds.
    repeat: str | None = None
    # Each script's source as written to run, by script stem or engine side.
    sources: dict[str, str] = field(default_factory=dict)
    # The image reference's processor directory as it was when the script ran:
    # each processor file's bare sha256, by name.
    processor_seen: dict[str, str] | None = None

    def vectors(self, texts: list[str], kind: str) -> Any:
        if kind == "random":
            rng = np.random.default_rng(7)
            v = rng.standard_normal((len(texts), 6))
        else:
            rng = np.random.default_rng(3)
            labels = [OPTIONS.index(self.image_labels.get(t) or _label_of(t)) for t in texts]
            v = np.eye(6)[labels] + 0.05 * rng.standard_normal((len(texts), 6))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def embed(self, name: str, spec: dict[str, Any]) -> int:
        # An image script's spec names its inputs "images", a text script's "texts".
        texts = json.loads(Path(spec.get("texts") or spec["images"]).read_text(encoding="utf-8"))
        self.texts[name] = texts
        if name in self.refuse:
            Path(spec["report"]).write_text(json.dumps({"error": self.refuse[name]}))
            return 3
        kind = {
            "candidate": self.candidate,
            "candidate-repeat": self.repeat or self.candidate,
            "base": "random",
        }.get(name, "same")
        if name not in self.no_npy:
            np.save(spec["out"], self.vectors(texts, kind).astype(np.float32))
        report: dict[str, Any] = {"rows": len(texts) + (name in self.rows_off), "dim": 6}
        if name == "reference":
            report["max_seq_length"] = 128
            if "images" in spec:
                report["vision_tokens"] = spec["vision_tokens"]
                report["soft_tokens_per_row"] = [self.soft_tokens.get(t, 63) for t in texts]
            elif self.over_max_seq_length is not None:
                report["tokens"] = {"max": 9, "over_max_seq_length": self.over_max_seq_length}
        else:
            self.caches[name] = (spec["cache_dir"], Path(spec["cache_dir"]).is_dir())
            if spec.get("backend") == "gpu":
                default = self.gpu_device if self.gpu_device is not None else GPU_WORKED
                report["device"] = self.gpu_device_for.get(name, default)
            if "images" in spec:
                # IMAGE_ENGINE_SCRIPT's vision encoder, on either backend.
                report["vision_backend"] = spec.get("backend", "cpu")
                report["vision_backend_constraints"] = self.vision_backend_constraints
                report["vision_backend_reason"] = self.vision_backend_reason
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
        # A script runs in an environment a provision finished, as `StageEnv.run` needs.
        assert env.ready, f"{name} ran in {env.name}, which was never provisioned"
        if name != "engines":  # the sides it runs are recorded one by one below
            self.runs.append((name, env.name))
            self.specs[name] = spec
            self.sources[name] = Path(args[1]).read_text(encoding="utf-8")
        if name in self.fail:
            return subprocess.CompletedProcess(args, 1, "", f"{name}: RuntimeError: boom")
        if name == "download":
            assert self.hub is not None
            filename = spec["files"][0]["filename"]
            path = self.hub / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(VISION_STAND_IN if filename == VISION_FILE else GOOGLE_BYTES)
            Path(spec["out"]).write_text(json.dumps({"paths": [str(path)]}), encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")
        report = Path(spec["report"])
        if name == "reference":
            if "processor" in spec:
                self.processor_seen = {
                    file: hashlib.sha256((Path(spec["processor"]) / file).read_bytes()).hexdigest()
                    for file in embedding_verify.PROCESSOR_FILES
                }
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
                self.sources[side] = Path(spec["script"]).read_text(encoding="utf-8")
                said = self.stderr_for.get(side, "")
                if side in self.fail:
                    Path(run["stderr"]).write_text(
                        f"{said}\n{side}: RuntimeError: boom", encoding="utf-8"
                    )
                    results[side] = {"returncode": 1, "seconds": 1.0}
                    stopped = side in spec["stop_on"]
                    continue
                if said:
                    Path(run["stderr"]).write_text(said, encoding="utf-8")
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
                v,
                ytr,
                yte,
                spec["n_labels"],
                spec["ks"],
                spec["draws"],
                None,
                spec["keep"],
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
        instance.provisioned.append(self.name)
        mark_provisioned(self)
        return self.path

    def run(self, args, timeout: int = 3600, **kwargs):
        return instance(self, args, timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", run)
    monkeypatch.setattr(embedding_export, "GOOGLE_SHA256", hashlib.sha256(GOOGLE_BYTES).hexdigest())
    monkeypatch.setattr(
        embedding_export, "VISION_SHA256", hashlib.sha256(VISION_STAND_IN).hexdigest()
    )
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
    assert "is fitted here as the app fits it" in notes and "it is not gated" in notes
    assert m["measurements"]["candidate"]["crossfit_k10"]["keep"] == 0.95
    assert len(m["measurements"]["reference"]["crossfit_k5"]["answered"]["draws"]) == DRAWS
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


def test_with_base_downloads_googles_file_in_the_reference_env(fake, tmp_path):
    m = _verify(_setup(tmp_path), with_base=True).manifest
    assert fake.runs[0] == ("download", envs.EMBED_TRAIN.name)
    # The export environment needs litert-converter, which has no Windows wheel.
    assert envs.EXPORT.name not in fake.provisioned
    assert all(env != envs.EXPORT.name for _, env in fake.runs)
    assert "gain_over_base" in m["quality"]


def test_a_failed_download_of_googles_file_is_the_base_check_failing(fake, tmp_path):
    fake.fail = {"download"}
    result = _verify(_setup(tmp_path), with_base=True)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed["name"] == BASE_CHECK
    assert "could not be fetched" in failed["detail"] and "boom" in failed["detail"]


def test_with_base_on_a_host_without_the_reference_env_names_the_reference(
    fake, tmp_path, monkeypatch
):
    def cannot(self, events=None, force: bool = False) -> Path:
        raise RuntimeError(f"{self.name} cannot be installed here")

    monkeypatch.setattr(envs.StageEnv, "provision", cannot)
    result = _verify(_setup(tmp_path), with_base=True)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed["name"] == REFERENCE_CHECK
    assert f"{envs.EMBED_TRAIN.name} cannot be installed here" in failed["detail"]
    assert fake.runs == []


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


@pytest.fixture
def on_macos(monkeypatch) -> None:
    """The GPU path's kernel reading is macOS's; the fakes' readings are written for it."""
    monkeypatch.setattr(sys, "platform", "darwin")


def test_the_gpu_backend_runs_each_engine_file_on_the_gpu_one_at_a_time(fake, tmp_path, on_macos):
    result = _verify(_setup(tmp_path), backend="gpu", with_base=True)

    assert result.status is Status.PASSED, result.manifest["checks"]
    for side in ("candidate", "base"):
        assert fake.specs[side]["backend"] == "gpu"
    assert fake.specs["reference"].get("backend") is None
    # Two files on one GPU at once is not measured.
    assert fake.launches[-1]["parallel"] is False
    engine = result.manifest["measurements"]["candidate"]["engine"]
    assert engine["backend"] == "gpu"
    assert engine["backend_observed"] is True
    assert engine["threads"] is None
    assert engine["device"] == GPU_WORKED
    # macOS says what established it as it always has, end to end.
    assert engine["gpu_evidence"] == (
        "the kernel showed the candidate's process doing work on this host's GPU while it "
        "generated"
    )
    gain = result.manifest["quality"]["gain_over_base"]
    assert gain["base_backend"] == "gpu" and gain["base_backend_observed"] is True
    limitations = " ".join(result.manifest["limitations"])
    assert "GPU backend of LiteRT-LM's EmbeddingEngine" in limitations
    assert "deletes the keyword arguments" in limitations
    assert "CPU backend of LiteRT-LM's EmbeddingEngine" not in limitations
    assert "recorded as asked for" not in limitations


def test_the_cpu_backend_is_passed_and_stated_as_before(fake, tmp_path):
    result = _verify(_setup(tmp_path))

    assert result.status is Status.PASSED
    assert fake.specs["candidate"]["backend"] == "cpu"
    engine = result.manifest["measurements"]["candidate"]["engine"]
    assert engine["backend"] == "cpu" and engine["backend_observed"] is False
    assert engine["threads"] is not None
    limitations = " ".join(result.manifest["limitations"])
    assert "CPU backend of LiteRT-LM's EmbeddingEngine" in limitations
    assert "GPU backend of LiteRT-LM's EmbeddingEngine" not in limitations


@pytest.mark.parametrize("platform", ["freebsd14", "cygwin"])
def test_the_gpu_backend_is_refused_where_its_work_is_not_read(
    fake, tmp_path, monkeypatch, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    result = _verify(_setup(tmp_path), backend="gpu", with_base=True)

    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert [c["name"] for c in failed] == [CANDIDATE_CHECK]
    assert f"--backend gpu on {platform}" in failed[0]["detail"]
    assert "on macOS, Windows and Linux only" in failed[0]["detail"]
    assert "The GPU on Windows and Linux" in failed[0]["detail"]
    assert fake.runs == []


# The adapter lines litert-lm 0.18.0 printed on an NVIDIA L4 and, without its
# Vulkan driver, on llvmpipe (measurements/2026-11-06-gpu-linux-windows).
L4_LINE = (
    "I0000 environment.cc:532] Selected adapter: NVIDIA L4, arch=lovelace, vendor=nvidia, "
    "backend=Vulkan, adapterType=Discrete GPU"
)
LLVMPIPE_LINE = (
    "I0000 environment.cc:532] Selected adapter: llvmpipe (LLVM 15.0.7, 256 bits), "
    "arch=software, vendor=mesa, backend=Vulkan, adapterType=CPU / Software"
)
LINUX_UNREAD = {
    "at_engine": {"platform": "linux", "looked": False},
    "after_generation": {"platform": "linux", "looked": False},
}
# The same, with nvidia-smi listing the process after the work.
LINUX_LISTED = {
    "at_engine": {"platform": "linux", "looked": False},
    "after_generation": {
        "platform": "linux",
        "looked": False,
        "nvidia_smi": {"listed": True, "used_memory_mib": 450, "pids_hidden": False, "gpus": 1},
    },
}
ARC_LINE = (
    "I0000 environment.cc:532] Selected adapter: Intel(R) Arc(TM) B390 GPU, arch=xe-3lpg, "
    "vendor=intel, backend=Direct3D 12, adapterType=Integrated GPU"
)
ARC_LUID = "0x00000000_0x0000A62B"


def _windows_reading(time_: int) -> dict[str, Any]:
    return {
        "platform": "win32",
        "looked": True,
        "gpu_client": None,
        "gpu_time": time_,
        "gpu_clients": {f"luid_{ARC_LUID}_phys_0_eng_0_engtype_3D": time_},
        "dxgi_adapters": [
            {"description": "Intel(R) Arc(TM) B390 GPU", "luid": ARC_LUID, "software": False}
        ],
    }


@pytest.fixture
def nvidia_host(monkeypatch):
    from litetune import evaluate

    monkeypatch.setattr(evaluate, "_nvidia_driver_present", lambda: True)
    for name in evaluate.VULKAN_DRIVER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_on_linux_the_gpu_runs_and_the_adapter_with_its_process_listed_is_the_reading(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "platform", "linux")
    fake.gpu_device = LINUX_LISTED
    fake.stderr_for = {"candidate": L4_LINE, "base": L4_LINE}
    result = _verify(_setup(tmp_path), backend="gpu", with_base=True)

    assert result.status is Status.PASSED, result.manifest["checks"]
    engine = result.manifest["measurements"]["candidate"]["engine"]
    assert engine["backend_observed"] is True and engine["adapter_selected"] is False
    assert "NVIDIA L4" in engine["gpu_evidence"] and "nvidia-smi listed" in engine["gpu_evidence"]
    assert engine["device"]["adapters"][0]["adapter_type"] == "Discrete GPU"
    assert result.manifest["quality"]["gain_over_base"]["base_backend_observed"] is True
    said = embedding_verify.GPU_MEASURED.format(how=embedding_verify.GPU_SEEN)
    assert said in result.manifest["limitations"]


def test_on_linux_an_adapter_nvidia_smi_did_not_show_working_is_selected(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "platform", "linux")
    fake.gpu_device = LINUX_UNREAD
    fake.stderr_for = {"candidate": L4_LINE}
    result = _verify(_setup(tmp_path), backend="gpu")

    assert result.status is Status.PASSED, result.manifest["checks"]
    engine = result.manifest["measurements"]["candidate"]["engine"]
    assert engine["backend_observed"] is False and engine["adapter_selected"] is True
    said = embedding_verify.GPU_MEASURED.format(how=embedding_verify.GPU_ASKED_FOR)
    assert said in result.manifest["limitations"]
    assert any("said as selected there" in t for t in result.manifest["limitations"])


def test_on_linux_a_gpu_run_with_no_adapter_line_fails_the_harness(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    fake.gpu_device = LINUX_LISTED
    result = _verify(_setup(tmp_path), backend="gpu")

    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert "named no adapter on stderr" in failed["detail"]


def test_on_windows_the_gpu_runs_and_the_selected_adapters_counters_are_the_reading(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "platform", "win32")
    fake.gpu_device = {
        "at_engine": _windows_reading(60343),
        "after_generation": _windows_reading(30493406),
    }
    fake.stderr_for = {"candidate": ARC_LINE}
    result = _verify(_setup(tmp_path), backend="gpu")

    assert result.status is Status.PASSED, result.manifest["checks"]
    engine = result.manifest["measurements"]["candidate"]["engine"]
    assert engine["backend_observed"] is True
    assert "GPU engine counters" in engine["gpu_evidence"]
    assert engine["gpu_luids"] == {ARC_LUID.lower(): {"start": 60343, "end": 30493406}}


def test_on_windows_growth_without_an_adapter_line_is_not_observed(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    fake.gpu_device = {
        "at_engine": _windows_reading(60343),
        "after_generation": _windows_reading(30493406),
    }
    result = _verify(_setup(tmp_path), backend="gpu")
    assert result.status is Status.PASSED, result.manifest["checks"]
    assert result.manifest["measurements"]["candidate"]["engine"]["backend_observed"] is False


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_a_candidate_on_a_software_adapter_fails_the_harness(
    fake, tmp_path, monkeypatch, platform, nvidia_host
):
    """The runtime embeds on llvmpipe and says nothing; the kernel's reading, here
    one that shows GPU work, does not outweigh the adapter it named."""
    monkeypatch.setattr(sys, "platform", platform)
    fake.stderr_for = {"candidate": LLVMPIPE_LINE}
    result = _verify(_setup(tmp_path), backend="gpu", with_base=True)

    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert "selected a software adapter, llvmpipe (LLVM 15.0.7, 256 bits)" in failed["detail"]
    assert ("libegl1" in failed["detail"]) is (platform == "linux")


def test_google_s_file_on_a_software_adapter_is_not_compared(fake, tmp_path, monkeypatch):
    """As an idle one is not: its vectors feed the gain over it."""
    monkeypatch.setattr(sys, "platform", "linux")
    fake.gpu_device = LINUX_UNREAD
    fake.stderr_for = {"candidate": L4_LINE, "base": LLVMPIPE_LINE}
    result = _verify(_setup(tmp_path), backend="gpu", with_base=True)

    assert result.status is Status.PASSED, result.manifest["checks"]
    assert "selected a software adapter, llvmpipe" in json.dumps(result.manifest["checks"])
    assert "gain_over_base" not in result.manifest.get("quality", {})


def test_a_side_that_failed_on_a_software_adapter_names_the_adapter_first(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "platform", "linux")
    fake.stderr_for = {"candidate": LLVMPIPE_LINE}
    fake.fail = {"candidate"}
    result = _verify(_setup(tmp_path), backend="gpu")

    assert result.status is Status.FAILED_HARNESS
    detail = json.dumps(result.manifest["checks"])
    assert "selected a software adapter, llvmpipe" in detail
    assert "the run then failed: the script exited 1" in detail


def test_a_cpu_side_is_not_judged_by_an_adapter_line(fake, tmp_path):
    fake.stderr_for = {"candidate": LLVMPIPE_LINE}
    result = _verify(_setup(tmp_path))
    assert result.status is Status.PASSED, result.manifest["checks"]
    assert "adapters" not in result.manifest["measurements"]["candidate"]["engine"]["device"]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_the_cpu_backend_still_runs_off_macos(fake, tmp_path, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    result = _verify(_setup(tmp_path))

    assert result.status is Status.PASSED, result.manifest["checks"]
    assert fake.specs["candidate"]["backend"] == "cpu"


def test_a_gpu_run_the_kernel_shows_idle_is_refused(fake, tmp_path, on_macos):
    fake.gpu_device = GPU_IDLE
    result = _verify(_setup(tmp_path), backend="gpu")

    assert result.status is Status.FAILED_HARNESS
    assert "did no GPU work" in json.dumps(result.manifest["checks"])


def test_google_s_file_idle_on_the_gpu_is_not_compared(fake, tmp_path, on_macos):
    """Google's file is held to the kernel's reading as the candidate is: its
    vectors feed the gain over it."""
    fake.gpu_device_for = {"base": GPU_IDLE}
    result = _verify(_setup(tmp_path), backend="gpu", with_base=True)

    assert "did no GPU work" in json.dumps(result.manifest["checks"])
    assert "gain_over_base" not in result.manifest.get("quality", {})


def test_a_gpu_run_nobody_could_observe_is_recorded_as_asked_for(fake, tmp_path, on_macos):
    fake.gpu_device = NOT_LOOKED
    result = _verify(_setup(tmp_path), backend="gpu")

    assert result.status is Status.PASSED
    assert result.manifest["measurements"]["candidate"]["engine"]["backend_observed"] is False
    limitations = " ".join(result.manifest["limitations"])
    assert "asked of, and not observed on, the GPU backend" in limitations
    # Not both: a manifest that says the GPU was measured and was not observed.
    assert "measured on the GPU backend" not in limitations


def test_a_backend_other_than_cpu_or_gpu_is_refused(fake, tmp_path):
    result = _verify(_setup(tmp_path), backend="npu")

    assert result.status is Status.FAILED_HARNESS
    assert fake.runs == []


def test_the_engine_script_builds_a_gpu_engine_and_reads_the_kernel_around_the_work(
    tmp_path, monkeypatch
):
    code, report, calls = _engine_main(
        tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], ["a", "b"], backend="gpu"
    )

    assert code == 0
    assert calls["engine"][0]["backend"] == ("gpu", None)
    assert report["backend"] == "gpu"
    assert report["threads"] is None
    assert report["device"] == {
        "at_engine": {"reading": "at engine"},
        "after_generation": {"reading": "after"},
    }


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


def test_a_file_compiled_for_an_npu_is_refused_before_any_run(fake, tmp_path):
    """`convert --npu` writes a report of kind npu beside its file; LiteRT-LM's CPU and
    GPU cannot run its DISPATCH_OP, so verify says that instead of failing in the engine."""
    s = _setup(tmp_path)
    digest = hash_file(s.model).removeprefix("sha256:")
    (s.model.parent / REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": EMBEDDING_CONVERT_SCHEMA,
                "kind": "npu",
                "npu": {"soc": "SM8750", "sha256": digest},
            }
        )
    )
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    failed = [c for c in result.manifest["checks"] if c["outcome"] == "failed"]
    assert failed[0]["name"] == CANDIDATE_CHECK and "Qualcomm SM8750 NPU" in failed[0]["detail"]
    assert fake.runs == []


def test_an_image_file_compiled_for_an_npu_is_refused_before_any_run(fake, tmp_path):
    """An image checkpoint's `convert --npu` file: Google's text-vision file for the SoC
    with the compiled encoder in it, and a report of kind npu as a text one's."""
    s = _image_setup(tmp_path, fake)
    digest = hash_file(s.model).removeprefix("sha256:")
    report = json.loads((s.model.parent / REPORT_NAME).read_text(encoding="utf-8"))
    report |= {
        "kind": "npu",
        "npu": {"soc": "SM8650", "sha256": digest, "splice": {"sha256": hash_file(s.model)}},
        "google_vision_npu_file": {"filename": "x_Qualcomm_SM8650.litertlm"},
    }
    (s.model.parent / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    failed = _failed(result.manifest)
    assert failed[0]["name"] == CANDIDATE_CHECK and "Qualcomm SM8650 NPU" in failed[0]["detail"]
    assert fake.runs == []


# -- the engines' CPU threads on a CPU with SME2 -------------------------------


def _engine_thread_notes(m: dict[str, Any]) -> list[str]:
    return [t for t in m["limitations"] if "on one thread" in t]


def test_on_a_cpu_with_sme2_the_engines_run_one_thread_and_the_reference_keeps_its_own(
    fake, tmp_path, sme2_host
):
    result = _verify(_setup(tmp_path), threads=4, base_litertlm=_google(tmp_path))
    m = result.manifest

    assert result.status is Status.PASSED, m["checks"]
    assert fake.specs["reference"]["threads"] == 4
    assert fake.specs["candidate"]["threads"] == 1 and fake.specs["base"]["threads"] == 1
    embedding = m["embedding"]
    assert (embedding["threads"], embedding["engine_threads"]) == (4, 1)
    assert "this CPU has SME2" in embedding["engine_threads_reason"]
    engine = m["measurements"]["candidate"]["engine"]
    assert (engine["threads"], engine["threads_requested"]) == (1, 4)
    assert engine["threads_reason"] == embedding["engine_threads_reason"]
    (said,) = _engine_thread_notes(m)
    assert said.startswith(
        "each EmbeddingEngine side ran LiteRT-LM on the CPU on one thread (4 asked for): "
        "this CPU has SME2"
    )
    assert any("EmbeddingEngine, 1 thread, from the" in t for t in m["limitations"])


def test_one_thread_each_lets_two_files_run_side_by_side_on_two_cpus(
    fake, tmp_path, monkeypatch, sme2_host
):
    monkeypatch.setattr(embedding_verify, "usable_cpus", lambda: 2)
    m = _verify(_setup(tmp_path), threads=4, base_litertlm=_google(tmp_path)).manifest
    assert [launch["parallel"] for launch in fake.launches] == [True]
    assert any("each on its own 1 thread;" in t for t in m["limitations"])


def test_off_such_a_cpu_the_engines_get_the_threads_asked_for(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_verify, "usable_cpus", lambda: 8)
    m = _verify(_setup(tmp_path), threads=4, base_litertlm=_google(tmp_path)).manifest
    assert fake.specs["candidate"]["threads"] == 4 and fake.specs["base"]["threads"] == 4
    assert (m["embedding"]["engine_threads"], m["embedding"]["engine_threads_reason"]) == (4, None)
    engine = m["measurements"]["candidate"]["engine"]
    assert (engine["threads"], engine["threads_requested"], engine["threads_reason"]) == (
        4,
        4,
        None,
    )
    assert not _engine_thread_notes(m)
    assert any("EmbeddingEngine, 4 threads, from the" in t for t in m["limitations"])
    assert any("each on its own 4 threads;" in t for t in m["limitations"])


def test_on_the_gpu_the_engines_get_no_thread_count(fake, tmp_path, on_macos, sme2_host):
    result = _verify(_setup(tmp_path), backend="gpu", threads=4)
    m = result.manifest

    assert result.status is Status.PASSED, m["checks"]
    assert fake.specs["candidate"]["threads"] is None
    assert fake.specs["reference"]["threads"] == 4
    assert m["embedding"]["engine_threads"] is None
    assert m["embedding"]["engine_threads_reason"] is None
    engine = m["measurements"]["candidate"]["engine"]
    assert engine["threads"] is None and engine["threads_requested"] is None
    assert not _engine_thread_notes(m)


def _convert_report(model: Path, **fields: Any) -> None:
    (model.parent / REPORT_NAME).write_text(
        json.dumps({"schema": EMBEDDING_CONVERT_SCHEMA} | fields), encoding="utf-8"
    )


def test_an_image_convert_report_is_tied_by_the_splices_sha256(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"the spliced file")
    _convert_report(
        model,
        modality="image",
        vision_tokens=140,
        graft={"sha256": "sha256:" + "0" * 64, "weights": "int8", "output": None},
        splice={"sha256": hash_file(model)},
        checkpoint={"sha256": "sha256:" + "1" * 64, "decision_sha256": "sha256:" + "2" * 64},
    )
    facts, source = convert_facts(model)
    assert facts is not None and source == str(tmp_path / REPORT_NAME)
    assert (facts.modality, facts.vision_tokens, facts.weights) == ("image", 140, "int8")
    assert facts.checkpoint_sha256 == "sha256:" + "1" * 64


def test_an_image_convert_report_is_not_tied_by_the_grafts_sha256(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"the text file")
    graft = {"sha256": hash_file(model), "weights": "int8"}
    _convert_report(model, modality="image", graft=graft, splice={"sha256": "sha256:00"})
    facts, why = convert_facts(model)
    assert facts is None and "another file" in why
    _convert_report(model, modality="image", graft=graft)
    facts, why = convert_facts(model)
    assert facts is None and "without a splice" in why


def test_a_text_convert_report_is_tied_by_the_grafts_sha256_with_or_without_modality(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"the grafted file")
    graft = {"sha256": hash_file(model), "weights": "int4"}
    for modality in ({}, {"modality": "text", "vision_tokens": None, "splice": None}):
        _convert_report(model, graft=graft, **modality)
        facts, _ = convert_facts(model)
        assert facts is not None
        assert (facts.modality, facts.vision_tokens, facts.weights) == ("text", None, "int4")
    # A splice block in a text report ties nothing.
    _convert_report(model, graft={"sha256": "sha256:00"}, splice={"sha256": hash_file(model)})
    assert convert_facts(model)[0] is None


def test_a_convert_report_of_an_unknown_modality_is_not_believed(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"a file")
    _convert_report(model, modality="audio", graft={"sha256": hash_file(model)})
    facts, why = convert_facts(model)
    assert facts is None and "'audio'" in why


# -- a checkpoint tuned on images ------------------------------------------------

IMAGE_TOKENS = 70
IMAGE_PROCESSOR_FILES = embedding_verify.PROCESSOR_FILES


def _write_image_split(
    path: Path, decision: Decision, per_label: int, fake: FakeEmbedEnvs, stem: str
) -> Path:
    """An image split as prepare writes one: `images/<sha256>.png` beside it, rows naming them."""
    images = path.parent / "images"
    images.mkdir(exist_ok=True)
    lines: list[str] = []
    for i in range(per_label):
        for option in OPTIONS:
            data = f"{stem} picture {i} of {option}".encode()
            digest = hashlib.sha256(data).hexdigest()
            file = images / f"{digest}.png"
            file.write_bytes(data)
            fake.image_labels[str(file.resolve())] = option
            lines.append(
                json.dumps(
                    {
                        "image": f"images/{digest}.png",
                        "image_sha256": digest,
                        "completion": option,
                        "target": option,
                        "source_line": len(lines) + 1,
                    }
                )
                + "\n"
            )
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _processor_text(name: str) -> str:
    """A processor file's stand-in: different for each file."""
    return json.dumps({"file": name})


def _image_setup(
    tmp_path: Path,
    fake: FakeEmbedEnvs,
    *,
    record: dict[str, Any] | None = None,
    convert: dict[str, Any] | None = None,
    processor: bool = True,
) -> Setup:
    """An image checkpoint, its split and its spliced file, with the convert.json that ties them.

    `convert` updates that report; `{"skip": True}` writes none."""
    decision = _decision()
    split = tmp_path / "split"
    split.mkdir()
    train = _write_image_split(split / "train.jsonl", decision, 12, fake, "train")
    heldout = _write_image_split(split / "heldout.jsonl", decision, 6, fake, "held")
    reference = tmp_path / "tuned" / "model"
    reference.mkdir(parents=True)
    (reference / "model.safetensors").write_bytes(b"tuned image weights")
    if processor:
        for name in IMAGE_PROCESSOR_FILES:
            (reference / name).write_text(_processor_text(name), encoding="utf-8")
    fields: dict[str, Any] = {
        "base_model": EMBEDDING_MODEL,
        "base_model_revision": EMBEDDING_REVISION,
        "prompt_name": PROMPT_NAME,
        "prefix": IMAGE_PREFIX,
        "labels": tuple(sorted(OPTIONS)),
        "rows": {"train": 36, "heldout": 18},
        "steps_done": 10,
        "steps_planned": 10,
        "seconds": 1.0,
        "device": "cpu",
        "dtype": "float32",
        "head_heldout_accuracy": 0.9,
        "recipe": RECIPE.as_dict(),
        "decision": decision.as_dict(),
        "decision_sha256": decision.digest(),
        "modality": "image",
        "vision_tokens": IMAGE_TOKENS,
        # What tune would have recorded: the fake reference's count for every picture.
        "soft_tokens": dict.fromkeys(_image_digests(train) + _image_digests(heldout), 63),
        # What tune would have recorded of the processor files written above.
        "processor_sha256": {
            name: hashlib.sha256(_processor_text(name).encode()).hexdigest()
            for name in IMAGE_PROCESSOR_FILES
        },
    }
    fields.update(record or {})
    if fields.get("tasks"):
        for key in (
            "labels",
            "rows",
            "head_heldout_accuracy",
            "decision",
            "decision_sha256",
        ):
            fields.pop(key)
    write_embedding_record(reference, EmbeddingRecord(**fields))
    model = tmp_path / "out" / "int8" / "model.litertlm"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"spliced file")
    report: dict[str, Any] = {
        "schema": EMBEDDING_CONVERT_SCHEMA,
        "modality": "image",
        "vision_tokens": IMAGE_TOKENS,
        "graft": {"sha256": "sha256:" + "0" * 64, "weights": "int8", "output": None},
        "splice": {"sha256": hash_file(model)},
        "checkpoint": {
            "path": str(reference),
            "sha256": hash_file(reference / "model.safetensors"),
            "decision_sha256": decision.digest(),
        },
    }
    report.update(convert or {})
    if not report.pop("skip", False):
        (model.parent / REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    return Setup(model, reference, heldout, train)


def _image_paths(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [str((path.parent / row["image"]).resolve()) for row in rows]


def _image_digests(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["image_sha256"] for row in rows]


def _failed(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in manifest["checks"] if c["outcome"] == "failed"]


def test_a_multitask_record_without_groups_selects_and_verifies(fake, tmp_path):
    selected = _decision()
    other = Decision(
        "choice",
        "Another question?",
        tuple(Option(option) for option in OPTIONS),
    )
    tasks = (
        EmbeddingTaskRecord(
            "other",
            other.as_dict(),
            other.digest(),
            tuple(sorted(OPTIONS)),
            {"train": 36, "heldout": 18},
            "sha256:" + "1" * 64,
            0.1,
            False,
            (),
        ),
        EmbeddingTaskRecord(
            "selected",
            selected.as_dict(),
            selected.digest(),
            tuple(sorted(OPTIONS)),
            {"train": 36, "heldout": 18},
            "sha256:" + "2" * 64,
            0.9,
            False,
            (),
        ),
    )
    setup = _image_setup(tmp_path, fake, record={"tasks": tasks})
    (setup.heldout.parent / "decision.json").write_text(
        selected.canonical_text() + " \n", encoding="utf-8"
    )
    report_path = setup.model.parent / REPORT_NAME
    record_path = setup.reference / "litetune.json"
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    assert all("train_groups" not in task for task in recorded["tasks"])
    recorded["tasks"][1]["train_sha256"] = hash_file(setup.train)
    record_path.write_text(json.dumps(recorded), encoding="utf-8")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint"]["decision_sha256"] = None
    report["checkpoint"]["tasks"] = [
        {"name": task.name, "decision_sha256": task.decision_sha256} for task in tasks
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = _verify(setup)
    assert result.status is Status.PASSED, result.manifest["checks"]
    assert result.manifest["embedding"]["decision_sha256"] == selected.digest()


def test_a_multitask_reference_refuses_cross_task_training_picture_leakage(fake, tmp_path):
    selected = _decision()
    heldout_digest = ""
    task = EmbeddingTaskRecord(
        "selected",
        selected.as_dict(),
        selected.digest(),
        tuple(sorted(OPTIONS)),
        {"train": 36, "heldout": 18},
        "sha256:" + "1" * 64,
        0.9,
        False,
        (),
    )
    setup = _image_setup(tmp_path, fake, record={"tasks": (task,)})
    (setup.heldout.parent / "decision.json").write_text(selected.canonical_text(), encoding="utf-8")
    heldout_digest = _image_digests(setup.heldout)[0]
    record = json.loads((setup.reference / "litetune.json").read_text(encoding="utf-8"))
    record["tasks"][0]["train_pictures_sha256"] = [heldout_digest]
    (setup.reference / "litetune.json").write_text(json.dumps(record), encoding="utf-8")
    report_path = setup.model.parent / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint"]["decision_sha256"] = None
    report["checkpoint"]["tasks"] = [{"name": "selected", "decision_sha256": selected.digest()}]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = _verify(setup)
    assert result.status is Status.FAILED_HARNESS
    assert ROWS_CHECK in _checks(result.manifest), result.manifest["checks"]
    detail = _checks(result.manifest)[ROWS_CHECK]["detail"]
    assert heldout_digest in detail and "training pictures of a task" in detail


def test_a_multitask_reference_refuses_cross_task_training_group_leakage(fake, tmp_path):
    selected = _decision()
    other = Decision("choice", "Another question?", tuple(Option(option) for option in OPTIONS))
    tasks = (
        EmbeddingTaskRecord(
            "other",
            other.as_dict(),
            other.digest(),
            tuple(sorted(OPTIONS)),
            {"train": 36, "heldout": 18},
            "sha256:" + "1" * 64,
            0.1,
            False,
            (),
            train_groups=("shared-run",),
        ),
        EmbeddingTaskRecord(
            "selected",
            selected.as_dict(),
            selected.digest(),
            tuple(sorted(OPTIONS)),
            {"train": 36, "heldout": 18},
            "sha256:" + "2" * 64,
            0.9,
            False,
            (),
        ),
    )
    setup = _image_setup(tmp_path, fake, record={"tasks": tasks})
    (setup.heldout.parent / "decision.json").write_text(selected.canonical_text(), encoding="utf-8")
    heldout = [json.loads(line) for line in setup.heldout.read_text().splitlines()]
    heldout[0]["group"] = "shared-run"
    setup.heldout.write_text("".join(json.dumps(row) + "\n" for row in heldout))
    report_path = setup.model.parent / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint"]["decision_sha256"] = None
    report["checkpoint"]["tasks"] = [
        {"name": task.name, "decision_sha256": task.decision_sha256} for task in tasks
    ]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = _verify(setup)

    assert result.status is Status.FAILED_HARNESS
    detail = _checks(result.manifest)[ROWS_CHECK]["detail"]
    assert "shared-run" in detail and "training groups of a task" in detail


def test_a_multitask_reference_refuses_a_different_selected_training_split(fake, tmp_path):
    selected = _decision()
    task = EmbeddingTaskRecord(
        "selected",
        selected.as_dict(),
        selected.digest(),
        tuple(sorted(OPTIONS)),
        {"train": 36, "heldout": 18},
        "sha256:" + "1" * 64,
        0.9,
        False,
        (),
    )
    setup = _image_setup(tmp_path, fake, record={"tasks": (task,)})
    (setup.heldout.parent / "decision.json").write_text(selected.canonical_text(), encoding="utf-8")
    report_path = setup.model.parent / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint"]["decision_sha256"] = None
    report["checkpoint"]["tasks"] = [{"name": "selected", "decision_sha256": selected.digest()}]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    result = _verify(setup)
    assert result.status is Status.FAILED_HARNESS
    detail = _checks(result.manifest)[ROWS_CHECK]["detail"]
    assert hash_file(setup.train) in detail and task.train_sha256 in detail


def test_a_multitask_image_reference_refuses_an_untrained_decision(fake, tmp_path):
    trained = _decision()
    task = EmbeddingTaskRecord(
        "trained",
        trained.as_dict(),
        trained.digest(),
        tuple(sorted(OPTIONS)),
        {"train": 36, "heldout": 18},
        "sha256:" + "1" * 64,
        0.9,
        False,
        (),
    )
    setup = _image_setup(tmp_path, fake, record={"tasks": (task,)})
    unknown = Decision(
        "choice",
        "Question this checkpoint did not train?",
        tuple(Option(option) for option in OPTIONS),
    )
    (setup.heldout.parent / "decision.json").write_text(unknown.canonical_text(), encoding="utf-8")
    result = _verify(setup)
    assert result.status is Status.FAILED_HARNESS
    assert "matches none" in _checks(result.manifest)[RECORD_CHECK]["detail"]


def test_an_image_checkpoint_embeds_its_images_on_both_sides_and_passes(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    result = _verify(s)
    m = result.manifest

    assert result.status is Status.PASSED, m["checks"]
    assert fake.runs == [
        ("reference", envs.EMBED_TRAIN.name),
        ("candidate", envs.RUNTIME.name),
        ("evaluate", envs.EMBED_TRAIN.name),
    ]
    assert fake.sources["reference"] == embedding_verify.IMAGE_REFERENCE_SCRIPT
    assert fake.sources["candidate"] == embedding_verify.IMAGE_ENGINE_SCRIPT
    paths = _image_paths(s.train) + _image_paths(s.heldout)
    assert fake.texts["reference"] == paths and fake.texts["candidate"] == paths
    ref, cand = fake.specs["reference"], fake.specs["candidate"]
    assert ref["vision_tokens"] == IMAGE_TOKENS and cand["vision_tokens"] == IMAGE_TOKENS
    assert ref["model"] == str(s.reference) and ref["image_text"] == "<|image|>"
    assert Path(ref["processor"]).name == "processor" and ref["processor"] != ref["model"]
    assert "prefix" not in cand and "texts" not in cand and "prompt_name" not in ref
    # The budget allows for an image, not a text.
    assert fake.timeouts["reference"] == (
        embedding_verify.STARTUP_TIMEOUT_S + embedding_verify.IMAGE_ROW_TIMEOUT_S * len(paths)
    )

    embedding = m["embedding"]
    assert (embedding["modality"], embedding["vision_tokens"]) == ("image", IMAGE_TOKENS)
    assert embedding["weights"] == "int8" and embedding["processor"] == str(s.reference)
    assert embedding["vision_threads"] is None and embedding["candidate_repeated"] is False
    assert embedding["soft_tokens"] == {"compared": 54, "unrecorded": 0, "min": 63, "max": 63}
    assert embedding["engines_side_by_side"] is False
    engine = m["measurements"]["candidate"]["engine"]
    assert engine["ms_per_image_median"] == 1.0 and "ms_per_text_median" not in engine
    assert m["scorer"]["means"] == embedding_verify.SCORER_MEANS_IMAGES
    assert m["gates"][0]["outcome"] == "passed"
    assert _checks(m)[ROWS_CHECK]["detail"] == "36 training and 18 held-out images"
    notes = " ".join(m["limitations"])
    assert "each image given as its bytes (litert_lm.Content.ImageBytes)" in notes
    assert "at a budget of at most 70 vision tokens per image" in notes
    assert "the vision encoder was given the CPU backend" in notes
    assert "Nothing litetune reads shows which device" in notes
    assert "through transformers" in notes and "at most 70 soft tokens" in notes
    assert "of each of 54 pictures the soft tokens `tune` recorded" in notes
    assert "image path calibrates the same way was not checked" in notes
    assert "as text, beside image examples" in notes
    assert "sentence-transformers" not in notes
    assert "not tied" not in notes
    lines = embedding_verify.summarise(m)
    assert any("embedding decider on images, 3 options" in ln for ln in lines)
    assert any("at most 70 vision tokens per image" in ln for ln in lines)


def test_an_embeddinggemma_adapter_mismatch_is_a_manifest_limitation(fake, tmp_path, on_macos):
    fake.vision_backend_constraints = {"vision_adapter": "cpu"}
    fake.vision_backend_reason = (
        "the vision adapter constraint vision_adapter='cpu' excludes the gpu vision backend "
        "given to the engine, but adapter constraints are recorded and left to LiteRT-LM: "
        "measured on one Mac with litert-lm 0.18.0: an EmbeddingGemma 2 file whose vision "
        "adapter is constrained to cpu ran with a GPU vision backend; which device the adapter "
        "used was not observed"
    )
    result = _verify(_image_setup(tmp_path, fake), backend="gpu")
    assert result.status is Status.PASSED, result.manifest["checks"]
    assert fake.vision_backend_reason in result.manifest["limitations"]
    assert result.manifest["embedding"]["vision_backend"] == "gpu"
    assert result.manifest["embedding"]["vision_backend_constraints"] == {"vision_adapter": "cpu"}


def test_verify_reads_the_split_prepare_wrote(fake, tmp_path):
    from test_embedding_prepare_images import FakeImageInspector, _fake_image

    from litetune.prepare import PrepareRequest, prepare

    rows = []
    for option in OPTIONS:
        for i in range(20):
            _fake_image(tmp_path / "raw" / f"{option}-{i}.png", width=8 + i, height=4)
            rows.append({"image": f"raw/{option}-{i}.png", "label": option})
    # A picture given twice stays in one half.
    rows.append(dict(rows[0]))
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(_decision().canonical_text(), encoding="utf-8")
    out = tmp_path / "prepared"
    prepared = prepare(
        PrepareRequest(
            data=data,
            output_dir=out,
            context_length=None,
            base_model=EMBEDDING_MODEL,
            questions=questions,
            images=FakeImageInspector(),
        )
    )
    assert prepared.train is not None and prepared.heldout is not None
    s = _image_setup(tmp_path, fake)
    for path in (prepared.train.path, prepared.heldout.path):
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()):
            fake.image_labels[str((out / row["image"]).resolve())] = row["completion"]
    record = json.loads((s.reference / "litetune.json").read_text(encoding="utf-8"))
    pictures = _image_digests(prepared.train.path) + _image_digests(prepared.heldout.path)
    record["soft_tokens"] = dict.fromkeys(pictures, 63)
    record["rows"] = {"train": prepared.train.n, "heldout": prepared.heldout.n}
    (s.reference / "litetune.json").write_text(json.dumps(record), encoding="utf-8")
    result = run_verify(
        VerifyRequest(model=s.model, reference=str(s.reference), data=prepared.heldout.path)
    )
    assert result.status is Status.PASSED, result.manifest["checks"]
    paths = _image_paths(prepared.train.path) + _image_paths(prepared.heldout.path)
    assert fake.texts["reference"] == paths
    assert result.manifest["embedding"]["soft_tokens"]["unrecorded"] == 0


def test_an_image_checkpoint_at_140_gives_every_side_its_budget(fake, tmp_path):
    s = _image_setup(tmp_path, fake, record={"vision_tokens": 140}, convert={"vision_tokens": 140})
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_STAND_IN)
    result = _verify(s, base_litertlm=local)
    assert result.status is Status.PASSED, result.manifest["checks"]
    for side in ("reference", "candidate", "base"):
        assert fake.specs[side]["vision_tokens"] == 140, side
    assert result.manifest["embedding"]["vision_tokens"] == 140


def test_a_picture_the_reference_makes_other_soft_tokens_of_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    victim = _image_paths(s.heldout)[2]
    fake.soft_tokens[victim] = 54
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == REFERENCE_CHECK
    assert f"made 54 soft tokens of {victim}" in failed["detail"]
    assert "tune recorded 63 for it" in failed["detail"]
    assert [r[0] for r in fake.runs] == ["reference"]


def test_pictures_tune_did_not_record_are_counted_and_said(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    record = json.loads((s.reference / "litetune.json").read_text(encoding="utf-8"))
    for sha in _image_digests(s.heldout)[:4]:
        del record["soft_tokens"][sha]
    (s.reference / "litetune.json").write_text(json.dumps(record), encoding="utf-8")
    m = _verify(s).manifest
    assert m["embedding"]["soft_tokens"]["unrecorded"] == 4
    assert m["embedding"]["soft_tokens"]["compared"] == 50
    assert any(
        t.startswith("4 of the pictures here were not in the splits") for t in m["limitations"]
    )


@pytest.mark.parametrize("reported", [None, [63] * 3, ["63"] * 54])
def test_a_reference_that_reports_no_count_per_picture_is_refused(fake, tmp_path, reported):
    s = _image_setup(tmp_path, fake)
    real = fake.embed

    def embed(name: str, spec: dict[str, Any]) -> int:
        code = real(name, spec)
        if name == "reference":
            report = json.loads(Path(spec["report"]).read_text(encoding="utf-8"))
            report["soft_tokens_per_row"] = reported
            Path(spec["report"]).write_text(json.dumps(report), encoding="utf-8")
        return code

    fake.embed = embed
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == REFERENCE_CHECK and "not one count per picture" in failed["detail"]


def test_a_picture_in_both_splits_is_refused_before_any_run(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    train_rows = s.train.read_text(encoding="utf-8").splitlines()
    held_rows = s.heldout.read_text(encoding="utf-8").splitlines()
    s.heldout.write_text("\n".join([*held_rows, train_rows[0]]) + "\n", encoding="utf-8")
    result = _verify(s)
    (failed,) = _failed(result.manifest)
    assert failed["name"] == ROWS_CHECK
    assert "1 picture(s) are in both the training split" in failed["detail"]
    assert _image_digests(s.train)[0] in failed["detail"]
    assert fake.runs == []


def test_a_text_checkpoint_keeps_its_text_scripts_and_reports_text(fake, tmp_path):
    m = _verify(_setup(tmp_path)).manifest
    assert fake.sources["reference"] == REFERENCE_SCRIPT
    assert fake.sources["candidate"] == ENGINE_SCRIPT
    assert (m["embedding"]["modality"], m["embedding"]["vision_tokens"]) == ("text", None)
    assert "processor" not in m["embedding"]
    assert "ms_per_text_median" in m["measurements"]["candidate"]["engine"]
    assert m["scorer"]["means"] == SCORER_MEANS
    assert any(
        ln.startswith("  embedding decider, 3 options") for ln in embedding_verify.summarise(m)
    )


def test_a_missing_image_is_refused_naming_its_row_before_any_run(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    victim = _image_paths(s.heldout)[4]
    Path(victim).unlink()
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == ROWS_CHECK
    assert f"{s.heldout}:5:" in failed["detail"] and "does not exist" in failed["detail"]
    assert fake.runs == []


def test_an_image_changed_since_prepare_is_refused_naming_its_row(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    victim = Path(_image_paths(s.train)[2])
    victim.write_bytes(b"other pixels")
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == ROWS_CHECK
    assert f"{s.train}:3:" in failed["detail"]
    assert "changed after `prepare` wrote the split" in failed["detail"]
    assert hashlib.sha256(b"other pixels").hexdigest() in failed["detail"]
    assert fake.runs == []


@pytest.mark.parametrize(
    ("edit", "said"),
    [
        ({"image_sha256": "not a digest"}, "'image_sha256' is 'not a digest'"),
        ({"image_sha256": "A" * 64}, "not a sha256"),
        ({"image": ""}, "'image' must be the image's path in the split"),
        ({"completion": "elsewhere"}, "'elsewhere' is not one of the decision's options"),
    ],
)
def test_an_image_row_that_is_not_preparess_is_refused(fake, tmp_path, edit, said):
    s = _image_setup(tmp_path, fake)
    rows = [json.loads(line) for line in s.heldout.read_text(encoding="utf-8").splitlines()]
    rows[0].update(edit)
    s.heldout.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = _verify(s)
    (failed,) = _failed(result.manifest)
    assert failed["name"] == ROWS_CHECK and said in failed["detail"]
    assert f"{s.heldout}:1:" in failed["detail"]
    assert fake.runs == []


def _point_first_row_at(s: Setup, value: str) -> None:
    rows = [json.loads(line) for line in s.heldout.read_text(encoding="utf-8").splitlines()]
    rows[0]["image"] = value
    s.heldout.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.mark.parametrize("how", ["absolute inside", "absolute outside", "dot-dot"])
def test_an_image_path_that_leaves_the_split_or_is_absolute_is_refused(fake, tmp_path, how):
    s = _image_setup(tmp_path, fake)
    first = _image_paths(s.heldout)[0]
    outside = tmp_path / "outside.png"
    outside.write_bytes(Path(first).read_bytes())
    value = {
        "absolute inside": first,
        "absolute outside": str(outside),
        "dot-dot": "../outside.png",
    }[how]
    _point_first_row_at(s, value)
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == ROWS_CHECK
    assert f"{s.heldout}:1:" in failed["detail"] and "not inside the split" in failed["detail"]
    assert fake.runs == []


@pytest.mark.posix
def test_an_images_directory_linked_out_of_the_split_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    images = s.heldout.parent / "images"
    elsewhere = tmp_path / "elsewhere"
    images.rename(elsewhere)
    images.symlink_to(elsewhere, target_is_directory=True)
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == ROWS_CHECK and "not inside the split" in failed["detail"]
    assert fake.runs == []


def test_a_text_checkpoint_given_an_image_split_is_refused(fake, tmp_path):
    s = _setup(tmp_path)
    images = _write_image_split(tmp_path / "images.jsonl", _decision(), 6, fake, "held")
    result = run_verify(
        VerifyRequest(model=s.model, reference=str(s.reference), data=images, train_data=s.train)
    )
    (failed,) = _failed(result.manifest)
    assert failed["name"] == ROWS_CHECK
    assert "holds image rows and the reference was tuned on text" in failed["detail"]
    assert fake.runs == []


def test_an_image_checkpoint_given_a_text_split_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    text = _write_split(tmp_path / "text.jsonl", _decision(), 12)
    result = run_verify(
        VerifyRequest(model=s.model, reference=str(s.reference), data=s.heldout, train_data=text)
    )
    (failed,) = _failed(result.manifest)
    assert failed["name"] == ROWS_CHECK
    assert "holds text rows and the reference was tuned on images" in failed["detail"]
    assert fake.runs == []


@pytest.mark.parametrize("image_checkpoint", [True, False])
def test_a_row_with_both_an_image_and_a_prompt_is_refused_with_prepares_reason(
    fake, tmp_path, image_checkpoint
):
    s = _image_setup(tmp_path, fake) if image_checkpoint else _setup(tmp_path)
    rows = [json.loads(line) for line in s.heldout.read_text(encoding="utf-8").splitlines()]
    if image_checkpoint:
        rows[2]["prompt"] = "a road"
    else:
        # A text split whose every row also names a picture: read as text, the
        # picture would be dropped, so it is refused before the modality is decided.
        rows = [row | {"image": "images/x.png"} for row in rows]
    s.heldout.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == ROWS_CHECK
    line = 3 if image_checkpoint else 1
    said = failed["detail"]
    assert f"heldout.jsonl:{line}: this row gives both an 'image' and a 'prompt'" in said
    assert fake.runs == []


def test_a_file_converted_at_other_vision_tokens_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake, convert={"vision_tokens": 140})
    result = _verify(s)
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert "records a budget of 140 vision tokens per image" in failed["detail"]
    assert "tuned at 70" in failed["detail"]
    assert fake.runs == []


def test_an_image_report_without_vision_tokens_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake, convert={"vision_tokens": None})
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert "records a budget of None vision tokens" in failed["detail"]


def test_a_text_file_for_an_image_checkpoint_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake, convert={"skip": True})
    (s.model.parent / REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": EMBEDDING_CONVERT_SCHEMA,
                "graft": {"sha256": hash_file(s.model), "weights": "int8"},
                "checkpoint": {"sha256": hash_file(s.reference / "model.safetensors")},
            }
        ),
        encoding="utf-8",
    )
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert "tuned on text; the reference's litetune.json was tuned on image" in failed["detail"]
    assert fake.runs == []


def test_an_image_file_for_a_text_checkpoint_is_refused(fake, tmp_path):
    s = _setup(tmp_path)
    (s.model.parent / REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": EMBEDDING_CONVERT_SCHEMA,
                "modality": "image",
                "vision_tokens": 70,
                "graft": {"sha256": "sha256:00", "weights": "int8"},
                "splice": {"sha256": hash_file(s.model)},
                "checkpoint": {},
            }
        ),
        encoding="utf-8",
    )
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert "tuned on image; the reference's litetune.json was tuned on text" in failed["detail"]


def test_an_untied_image_file_is_still_measured_at_the_records_vision_tokens(fake, tmp_path):
    s = _image_setup(tmp_path, fake, convert={"skip": True})
    m = _verify(s).manifest
    assert fake.specs["candidate"]["vision_tokens"] == IMAGE_TOKENS
    assert any("not tied to the reference" in t for t in m["limitations"])


def test_an_image_checkpoint_on_the_gpu_puts_text_and_vision_there(fake, tmp_path, on_macos):
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_STAND_IN)
    result = _verify(_image_setup(tmp_path, fake), backend="gpu", threads=4, base_litertlm=local)
    m = result.manifest
    assert result.status is Status.PASSED, m["checks"]
    assert [r[0] for r in fake.runs] == [
        "reference",
        "candidate",
        "candidate-repeat",
        "base",
        "evaluate",
    ]
    for side in ("candidate", "candidate-repeat", "base"):
        assert fake.specs[side]["backend"] == "gpu"
        assert fake.specs[side]["threads"] is None
        assert fake.specs[side]["vision_tokens"] == m["embedding"]["vision_tokens"]
        assert fake.sources[side] == embedding_verify.IMAGE_ENGINE_SCRIPT
    assert fake.specs["reference"].get("backend") is None
    assert fake.specs["reference"]["threads"] == 4
    (launch,) = fake.launches
    assert launch["parallel"] is False and launch["stop_on"] == ["candidate", "candidate-repeat"]
    embedding = m["embedding"]
    assert embedding["vision_backend"] == "gpu" and embedding["candidate_repeated"] is True
    assert embedding["engine_threads"] is None
    engine = m["measurements"]["candidate"]["engine"]
    assert engine["backend"] == "gpu" and engine["backend_observed"] is True
    assert engine["vision_backend"] == "gpu"
    assert engine["threads"] is None and engine["device"] == GPU_WORKED
    gain = m["quality"]["gain_over_base"]
    assert gain["base_backend"] == "gpu" and gain["base_backend_observed"] is True
    said = embedding_verify.IMAGE_GPU_MEASURED.format(
        how=embedding_verify.GPU_SEEN, vision_tokens=embedding["vision_tokens"]
    )
    assert said in m["limitations"]
    assert embedding_verify.GPU_PRECISION in said
    assert embedding_verify.REPEATED_GPU in m["limitations"]
    limitations = " ".join(m["limitations"])
    # Not the text GPU run's words (no CPU in it), nor the CPU image run's.
    assert "the CPU and the NPU are not measured" not in limitations
    assert "CPU backend of LiteRT-LM's EmbeddingEngine, " not in limitations
    assert "applied to the text tower alone" not in limitations


def test_a_text_checkpoint_on_the_gpu_keeps_its_own_words(fake, tmp_path, on_macos):
    m = _verify(_setup(tmp_path), backend="gpu").manifest
    assert embedding_verify.GPU_MEASURED.format(how=embedding_verify.GPU_SEEN) in m["limitations"]
    assert not any("for the text tower, with the vision" in t for t in m["limitations"])
    assert "vision_backend" not in m["measurements"]["candidate"]["engine"]


@pytest.mark.parametrize("platform", ["freebsd14", "cygwin"])
def test_an_image_checkpoint_on_the_gpu_is_refused_where_its_work_is_not_read(
    fake, tmp_path, monkeypatch, platform
):
    s = _image_setup(tmp_path, fake)
    monkeypatch.setattr(sys, "platform", platform)
    result = _verify(s, backend="gpu", with_base=True)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert failed["detail"] == embedding_verify.GPU_NOT_MEASURED_HERE.format(platform=platform)
    assert fake.runs == []


def test_an_image_gpu_run_nobody_could_observe_is_recorded_as_asked_for(fake, tmp_path, on_macos):
    fake.gpu_device = NOT_LOOKED
    m = _verify(_image_setup(tmp_path, fake), backend="gpu").manifest
    assert m["measurements"]["candidate"]["engine"]["backend_observed"] is False
    said = embedding_verify.IMAGE_GPU_MEASURED.format(
        how=embedding_verify.GPU_ASKED_FOR, vision_tokens=m["embedding"]["vision_tokens"]
    )
    assert said in m["limitations"]


def test_an_image_gpu_run_the_kernel_shows_idle_is_refused(fake, tmp_path, on_macos):
    fake.gpu_device = GPU_IDLE
    result = _verify(_image_setup(tmp_path, fake), backend="gpu")
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK and "did no GPU work" in failed["detail"]


def test_on_a_cpu_with_sme2_an_image_gpu_run_repeats_for_the_gpu_rule(
    fake, tmp_path, on_macos, sme2_host
):
    """The repeat belongs to every GPU image run, independently of the host CPU."""
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_STAND_IN)
    result = _verify(_image_setup(tmp_path, fake), backend="gpu", threads=4, base_litertlm=local)
    m = result.manifest
    assert result.status is Status.PASSED, m["checks"]
    assert [r[0] for r in fake.runs] == [
        "reference",
        "candidate",
        "candidate-repeat",
        "base",
        "evaluate",
    ]
    (launch,) = fake.launches
    assert launch["parallel"] is False and launch["stop_on"] == ["candidate", "candidate-repeat"]
    assert m["embedding"]["candidate_repeated"] is True
    # The GPU takes no thread count, SME2 or not.
    assert m["embedding"]["engine_threads"] is None
    assert m["embedding"]["engine_threads_reason"] is None
    assert embedding_verify.REPEATED_GPU in m["limitations"]
    assert "candidate-repeat" not in fake.specs["evaluate"]["vectors"]


def test_an_image_gpu_run_refuses_when_the_repeat_differs(fake, tmp_path, on_macos, sme2_host):
    fake.repeat = "random"
    result = _verify(_image_setup(tmp_path, fake), backend="gpu")
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK
    assert (
        embedding_verify.NOT_REPEATABLE_GPU.format(
            differ="candidate.npy and candidate-repeat.npy differ"
        )
        in failed["detail"]
    )


def test_an_image_checkpoint_with_a_text_prefix_is_refused(fake, tmp_path):
    s = _image_setup(tmp_path, fake, record={"prefix": DECIDER_PREFIX})
    (failed,) = _failed(_verify(s).manifest)
    assert failed["name"] == RECORD_CHECK and "tuned on images with the prefix" in failed["detail"]
    assert fake.runs == []


def test_an_image_checkpoint_does_not_read_the_prompt_name(fake, tmp_path):
    s = _image_setup(tmp_path, fake, record={"prompt_name": "Anything"})
    assert _verify(s).status is Status.PASSED


def test_the_processor_files_are_what_transformers_saves():
    assert IMAGE_PROCESSOR_FILES == (
        "chat_template.jinja",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    assert "preprocessor_config.json" not in IMAGE_PROCESSOR_FILES


def test_a_reference_without_its_processor_is_refused_with_no_stand_in(fake, tmp_path):
    s = _image_setup(tmp_path, fake, processor=False)
    (s.reference / "tokenizer.json").write_text("{}", encoding="utf-8")
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == REFERENCE_CHECK
    said = failed["detail"]
    assert "has no chat_template.jinja, processor_config.json, tokenizer_config.json" in said
    assert "tokenizer.json," not in failed["detail"].split("has no")[1].split(":")[0]
    assert fake.runs == []


def test_a_reference_with_every_processor_file_is_embedded_with_its_own(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    assert _verify(s).status is Status.PASSED
    assert fake.specs["reference"]["model"] == str(s.reference)


@pytest.mark.parametrize("changed", IMAGE_PROCESSOR_FILES)
def test_a_processor_file_that_is_not_the_one_tune_recorded_is_refused(fake, tmp_path, changed):
    s = _image_setup(tmp_path, fake)
    (s.reference / changed).write_text('{"another": "processor"}', encoding="utf-8")
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == REFERENCE_CHECK
    said = failed["detail"]
    found = hashlib.sha256(b'{"another": "processor"}').hexdigest()
    recorded = hashlib.sha256(_processor_text(changed).encode()).hexdigest()
    assert f"the reference's {changed} hash to {found}, not the sha256" in said
    assert f"({recorded})" in said
    assert fake.runs == []


def _recorded_processor() -> dict[str, str]:
    return {
        name: hashlib.sha256(_processor_text(name).encode()).hexdigest()
        for name in IMAGE_PROCESSOR_FILES
    }


def test_a_processor_file_replaced_after_the_copy_does_not_reach_the_reference(
    fake, tmp_path, monkeypatch
):
    s = _image_setup(tmp_path, fake)
    copy = embedding_verify.copy_processor

    def copy_then_replace(reference: Path, dest: Path) -> None:
        copy(reference, dest)
        (reference / "tokenizer.json").write_text('{"another": "processor"}', encoding="utf-8")

    monkeypatch.setattr(embedding_verify, "copy_processor", copy_then_replace)
    assert _verify(s).status is Status.PASSED
    processor = Path(fake.specs["reference"]["processor"])
    assert processor != s.reference and s.reference not in processor.parents
    assert fake.processor_seen == _recorded_processor()


def test_a_processor_file_replaced_after_the_first_check_is_refused_in_the_copy(
    fake, tmp_path, monkeypatch
):
    s = _image_setup(tmp_path, fake)
    copy = embedding_verify.copy_processor

    def replace_then_copy(reference: Path, dest: Path) -> None:
        (reference / "tokenizer.json").write_text('{"another": "processor"}', encoding="utf-8")
        copy(reference, dest)

    monkeypatch.setattr(embedding_verify, "copy_processor", replace_then_copy)
    result = _verify(s)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == REFERENCE_CHECK
    found = hashlib.sha256(b'{"another": "processor"}').hexdigest()
    assert failed["detail"].startswith(
        "verify's copy of the processor files, taken after the first check: the reference's "
        f"tokenizer.json hash to {found}"
    )
    assert "reference" not in [name for name, _ in fake.runs]


def test_an_image_record_without_processor_digests_is_not_read(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    path = s.reference / "litetune.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["processor_sha256"]
    path.write_text(json.dumps(data), encoding="utf-8")
    result = _verify(s)
    assert result.status is not Status.PASSED
    said = json.dumps(result.manifest)
    assert "'processor_sha256' is None; an image record gives the sha256 of each of" in said
    assert fake.runs == []


def test_googles_text_vision_file_gives_the_gain_for_an_image_checkpoint(fake, tmp_path):
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_STAND_IN)
    m = _verify(_image_setup(tmp_path, fake), base_litertlm=local).manifest
    assert [r[0] for r in fake.runs] == ["reference", "candidate", "base", "evaluate"]
    assert fake.sources["base"] == embedding_verify.IMAGE_ENGINE_SCRIPT
    assert fake.specs["base"]["vision_tokens"] == IMAGE_TOKENS
    assert "gain_over_base" in m["quality"]


def test_googles_text_file_is_refused_as_the_base_of_an_image_checkpoint(fake, tmp_path):
    result = _verify(_image_setup(tmp_path, fake), base_litertlm=_google(tmp_path))
    (failed,) = _failed(result.manifest)
    assert failed["name"] == BASE_EMBED_CHECK.replace(" embeddings", "")
    assert VISION_FILE in failed["detail"]
    assert fake.runs == []


def test_with_base_downloads_the_text_vision_file_for_an_image_checkpoint(fake, tmp_path):
    m = _verify(_image_setup(tmp_path, fake), with_base=True).manifest
    assert fake.runs[0] == ("download", envs.EMBED_TRAIN.name)
    assert envs.EXPORT.name not in fake.provisioned
    assert fake.specs["download"]["files"] == [embedding_export.VISION_HUB_FILE.as_dict()]
    assert "gain_over_base" in m["quality"]


def test_the_text_path_still_downloads_the_text_file(fake, tmp_path):
    _verify(_setup(tmp_path), with_base=True)
    assert fake.specs["download"]["files"] == [embedding_export.GOOGLE_HUB_FILE.as_dict()]


def test_on_a_cpu_with_sme2_the_image_candidate_runs_one_thread_twice(fake, tmp_path, sme2_host):
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_STAND_IN)
    result = _verify(_image_setup(tmp_path, fake), threads=4, base_litertlm=local)
    m = result.manifest
    assert result.status is Status.PASSED, m["checks"]
    assert fake.specs["reference"]["threads"] == 4
    assert fake.specs["candidate"]["threads"] == 1
    assert [r[0] for r in fake.runs] == [
        "reference",
        "candidate",
        "candidate-repeat",
        "base",
        "evaluate",
    ]
    assert fake.specs["candidate-repeat"]["model"] == fake.specs["candidate"]["model"]
    assert fake.specs["candidate-repeat"]["cache_dir"] != fake.specs["candidate"]["cache_dir"]
    (launch,) = fake.launches
    assert launch["parallel"] is False and launch["stop_on"] == ["candidate", "candidate-repeat"]
    assert m["embedding"]["candidate_repeated"] is True
    assert any("EmbeddingEngine, 1 thread, each image" in t for t in m["limitations"])
    assert any("1 thread applied to the text tower alone" in t for t in m["limitations"])
    assert embedding_verify.REPEATED in m["limitations"]
    # The evaluation gets the first run's vectors, and no vectors of the second.
    assert "candidate-repeat" not in fake.specs["evaluate"]["vectors"]


def test_a_second_run_that_differs_is_a_harness_failure(fake, tmp_path, sme2_host):
    fake.repeat = "random"
    result = _verify(_image_setup(tmp_path, fake), threads=4)
    assert result.status is Status.FAILED_HARNESS
    (failed,) = _failed(result.manifest)
    assert failed["name"] == CANDIDATE_CHECK
    said = failed["detail"]
    assert "did not give the same vectors (candidate.npy and candidate-repeat.npy differ)" in said
    assert said == embedding_verify.NOT_REPEATABLE.format(
        differ="candidate.npy and candidate-repeat.npy differ"
    )
    assert "evaluate" not in [r[0] for r in fake.runs]


def test_a_second_run_that_fails_is_a_harness_failure(fake, tmp_path, sme2_host):
    fake.fail = {"candidate-repeat"}
    (failed,) = _failed(_verify(_image_setup(tmp_path, fake), threads=4).manifest)
    assert failed["name"] == CANDIDATE_CHECK
    said = failed["detail"]
    assert "the candidate's second run, which shows the first one repeatable, failed" in said


def test_off_such_a_cpu_an_image_run_embeds_the_candidate_once_and_never_side_by_side(
    fake, tmp_path, monkeypatch
):
    monkeypatch.setattr(embedding_verify, "usable_cpus", lambda: 64)
    local = tmp_path / "vision.litertlm"
    local.write_bytes(VISION_STAND_IN)
    m = _verify(_image_setup(tmp_path, fake), threads=4, base_litertlm=local).manifest
    assert [r[0] for r in fake.runs] == ["reference", "candidate", "base", "evaluate"]
    assert [launch["parallel"] for launch in fake.launches] == [False]
    assert not any(
        t.startswith("the candidate and Google's file were embedded at the same")
        for t in m["limitations"]
    )
    assert embedding_verify.REPEATED not in m["limitations"]


# -- the image engine script, run here against a stub of litert_lm ------------


PICTURES = (b"picture a", b"picture b")


def _pictures(tmp_path: Path, pictures: Sequence[bytes] = PICTURES) -> tuple[list[str], list[str]]:
    """Each picture written to its own file: (the paths, the bytes' sha256)."""
    folder = tmp_path / "pics"
    folder.mkdir(exist_ok=True)
    paths = []
    for i, data in enumerate(pictures):
        path = folder / f"{'ab'[i]}.png"
        path.write_bytes(data)
        paths.append(str(path))
    return paths, [hashlib.sha256(data).hexdigest() for data in pictures]


def _image_engine_main(
    tmp_path,
    monkeypatch,
    answers,
    images,
    digests=None,
    backend=None,
    stderr=None,
    constraints=None,
):
    ns: dict[str, Any] = {"__name__": "image_engine_script"}
    exec(compile(embedding_verify.IMAGE_ENGINE_SCRIPT, "engine.py", "exec"), ns)
    stub, calls = _stub_litert_lm(answers)
    monkeypatch.setitem(sys.modules, "litert_lm", stub)
    # `backend_for` imports these two from litert_lm.interfaces.
    interfaces: Any = types.ModuleType("litert_lm.interfaces")
    interfaces.CPU = lambda: ("cpu", None)
    interfaces.GPU = lambda: ("gpu", None)
    monkeypatch.setitem(sys.modules, "litert_lm.interfaces", interfaces)
    monkeypatch.setitem(ns, "cpu_report", lambda: {"model": "stub cpu", "count": 2})
    readings = iter([{"reading": "at engine"}, {"reading": "after"}])
    monkeypatch.setitem(ns, "device_report", lambda: next(readings))
    monkeypatch.setitem(ns, "bundle_vision_constraints", lambda path: constraints or {})
    (tmp_path / "images.json").write_text(json.dumps(images))
    spec = {
        "model": "/models/m.litertlm",
        "images": str(tmp_path / "images.json"),
        "image_sha256": digests,
        "threads": 1,
        "vision_tokens": 140,
        "cache_dir": str(tmp_path / "cache-candidate"),
        "max_input_length": 1024,
        "out": str(tmp_path / "out.npy"),
        "report": str(tmp_path / "report.json"),
    }
    if backend is not None:
        spec["backend"] = backend
    if stderr is not None:
        (tmp_path / "side.err").write_text(stderr, encoding="utf-8")
        spec["stderr"] = str(tmp_path / "side.err")
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["engine.py", str(tmp_path / "spec.json")])
    code = ns["main"]()
    return code, json.loads((tmp_path / "report.json").read_text()), calls


def test_the_image_engine_script_on_the_gpu_puts_the_vision_encoder_on_the_gpu(
    tmp_path, monkeypatch
):
    paths, digests = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], paths, digests, backend="gpu"
    )
    assert code == 0
    # The text and vision towers get the requested GPU backend.
    assert calls["engine"][0]["backend"] == ("gpu", None)
    assert calls["vision"] == [{"vision_backend": ("gpu", None), "vision_tokens_per_image": 140}]
    assert [c.bytes for c in calls["texts"]] == list(PICTURES)
    assert report["backend"] == "gpu" and report["vision_backend"] == "gpu"
    assert report["threads"] is None and report["vision_threads"] is None
    assert report["device"] == {
        "at_engine": {"reading": "at engine"},
        "after_generation": {"reading": "after"},
    }
    np.testing.assert_allclose(np.load(tmp_path / "out.npy"), [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)


def test_the_image_engine_script_leaves_an_adapter_constraint_to_litert_lm(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path,
        monkeypatch,
        [[3.0, 4.0], [0.0, 2.0]],
        paths,
        digests,
        backend="gpu",
        constraints={"vision_adapter": "cpu"},
    )
    assert code == 0
    assert calls["engine"][0]["backend"] == ("gpu", None)
    assert calls["vision"] == [{"vision_backend": ("gpu", None), "vision_tokens_per_image": 140}]
    assert report["vision_backend"] == "gpu"
    assert report["vision_backend_constraints"] == {"vision_adapter": "cpu"}
    assert "measured on one Mac with litert-lm 0.18.0" in report["vision_backend_reason"]
    assert "which device the adapter used was not observed" in report["vision_backend_reason"]


def test_the_image_engine_script_honours_a_cpu_encoder_constraint(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path,
        monkeypatch,
        [[3.0, 4.0], [0.0, 2.0]],
        paths,
        digests,
        backend="gpu",
        constraints={"vision_encoder": "cpu", "vision_adapter": "cpu"},
    )
    assert code == 0
    assert calls["vision"] == [{"vision_backend": ("cpu", None), "vision_tokens_per_image": 140}]
    assert report["vision_backend"] == "cpu"
    assert "vision_encoder='cpu'" in report["vision_backend_reason"]


def test_the_image_engine_script_refuses_cpu_text_with_gpu_only_vision(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path,
        monkeypatch,
        [[3.0, 4.0], [0.0, 2.0]],
        paths,
        digests,
        constraints={"vision_encoder": "gpu", "vision_adapter": "cpu,gpu"},
    )
    assert code == 3 and calls["engine"] == []
    assert "vision_encoder='gpu'" in report["error"]
    assert "Pass --backend gpu" in report["error"]
    assert report["vision_backend_constraints"] == {
        "vision_encoder": "gpu",
        "vision_adapter": "cpu,gpu",
    }


def test_the_image_engine_script_builds_a_vision_engine_and_gives_each_pictures_bytes(
    tmp_path, monkeypatch
):
    paths, digests = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], paths, digests
    )
    assert code == 0
    assert calls["engine"] == [
        {
            "model": "/models/m.litertlm",
            "backend": ("cpu", 1),
            "cache_dir": str(tmp_path / "cache-candidate"),
            "max_input_length": 1024,
            "min_input_length": None,
        }
    ]
    # No thread count for the vision backend: 0.18.0 would not apply one.
    assert calls["vision"] == [{"vision_backend": ("cpu", None), "vision_tokens_per_image": 140}]
    assert report["vision_threads"] is None
    # The bytes read and hashed are the bytes embedded: no path reaches the engine.
    assert [c.bytes for c in calls["texts"]] == list(PICTURES)
    assert calls["options"] == [
        {
            "normalize": True,
            "insert_special_tokens": True,
            "input_overflow_strategy": "error",
            "output_size": None,
        }
    ]
    np.testing.assert_allclose(np.load(tmp_path / "out.npy"), [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)
    assert report["rows"] == 2 and report["dim"] == 2
    assert report["backend"] == "cpu" and report["vision_backend"] == "cpu"
    assert report["threads"] == 1 and report["device"] == {}
    assert report["settings"]["vision_tokens_per_image"] == 140
    assert report["cpu"] == {"model": "stub cpu", "count": 2}
    assert isinstance(report["ms_median"], float)


def test_the_image_engine_script_refuses_an_image_the_engine_refuses(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    code, report, _ = _image_engine_main(
        tmp_path, monkeypatch, [[1.0, 0.0], RuntimeError("bad image")], paths, digests
    )
    assert code == 3
    assert report["error"] == f"the engine refused image 1 ({paths[1]}): bad image"
    assert not (tmp_path / "out.npy").exists()


def test_the_image_engine_script_refuses_a_zero_vector(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    code, report, _ = _image_engine_main(
        tmp_path, monkeypatch, [[1.0, 0.0], [0.0, 0.0]], paths, digests
    )
    assert code == 3 and "zero" in report["error"] and "[1]" in report["error"]


def test_the_image_engine_script_refuses_a_picture_changed_after_the_check(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    Path(paths[1]).write_bytes(b"another picture")
    code, report, calls = _image_engine_main(
        tmp_path, monkeypatch, [[1.0, 0.0], [0.0, 1.0]], paths, digests
    )
    assert code == 3
    found = hashlib.sha256(b"another picture").hexdigest()
    assert report["error"] == (
        f"{paths[1]} hashes to {found}, not the {digests[1]} its row names; the picture "
        "changed after verify checked it"
    )
    # The first picture was embedded; the changed one never reached the engine.
    assert [c.bytes for c in calls["texts"]] == [PICTURES[0]]
    assert not (tmp_path / "out.npy").exists()


@pytest.mark.parametrize("digests", [None, ["a" * 64], ["a" * 64] * 3])
def test_the_image_engine_script_refuses_digests_that_are_not_one_per_image(
    tmp_path, monkeypatch, digests
):
    paths, _ = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path, monkeypatch, [[1.0, 0.0], [0.0, 1.0]], paths, digests
    )
    assert code == 3 and "sha256 for 2 images" in report["error"]
    assert calls["engine"] == []


def test_the_image_scripts_share_one_picture_reader():
    for script in (embedding_verify.IMAGE_ENGINE_SCRIPT, embedding_verify.IMAGE_REFERENCE_SCRIPT):
        assert embedding_verify.PICTURE_BYTES_SOURCE in script
        assert script.count("def picture_bytes(") == 1
        assert "read_bytes()" not in script.replace(embedding_verify.PICTURE_BYTES_SOURCE, "")
    assert "ImageFile" not in embedding_verify.IMAGE_ENGINE_SCRIPT.split('"""', 2)[2]


def test_the_picture_reader_returns_the_bytes_it_hashed(tmp_path):
    ns: dict[str, Any] = {"__name__": "image_reference"}
    exec(compile(embedding_verify.IMAGE_REFERENCE_SCRIPT, "reference.py", "exec"), ns)
    picture = tmp_path / "p.png"
    picture.write_bytes(b"the picture")
    digest = hashlib.sha256(b"the picture").hexdigest()
    assert ns["picture_bytes"](str(picture), digest) == (b"the picture", None)
    data, problem = ns["picture_bytes"](str(picture), "0" * 64)
    assert data == b"the picture"
    assert problem == (
        f"{picture} hashes to {digest}, not the {'0' * 64} its row names; the picture changed "
        "after verify checked it"
    )


def _image_reference_main(tmp_path, monkeypatch, images, digests, loads=None):
    """The image reference script's `main` against stand-ins for torch, PIL and transformers.

    `loads`, when given, collects (class, path) for each `from_pretrained` call."""
    loads = [] if loads is None else loads

    def loader(cls: str, made: Callable[[], Any]) -> Any:
        def from_pretrained(path, *a, **k):
            loads.append((cls, path))
            return made()

        return types.SimpleNamespace(from_pretrained=from_pretrained)

    ns: dict[str, Any] = {"__name__": "image_reference"}
    exec(compile(embedding_verify.IMAGE_REFERENCE_SCRIPT, "reference.py", "exec"), ns)
    decoded: list[bytes] = []

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    class Model:
        def eval(self):
            return self

    torch: Any = types.ModuleType("torch")
    torch.set_num_threads = lambda n: None
    torch.float32 = "float32"
    torch.no_grad = NoGrad
    transformers: Any = types.ModuleType("transformers")
    transformers.AutoConfig = loader(
        "AutoConfig", lambda: types.SimpleNamespace(audio_config=object())
    )
    transformers.AutoModel = loader("AutoModel", Model)
    transformers.AutoProcessor = loader("AutoProcessor", object)
    pil: Any = types.ModuleType("PIL")
    pil.Image, pil.ImageOps = object(), object()
    for name, module in (("torch", torch), ("transformers", transformers), ("PIL", pil)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(ns, "strict_pillow", lambda image: None)

    def picture_of(image, ops, data, flipped):
        decoded.append(data)
        raise StopIteration("decoded")

    monkeypatch.setitem(ns, "picture_of", picture_of)
    (tmp_path / "images.json").write_text(json.dumps(images))
    spec = {
        "model": str(tmp_path / "ref"),
        "processor": str(tmp_path / "processor"),
        "images": str(tmp_path / "images.json"),
        "image_sha256": digests,
        "image_text": "<|image|>",
        "vision_tokens": 70,
        "threads": 1,
        "out": str(tmp_path / "ref.npy"),
        "report": str(tmp_path / "ref.json"),
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["reference.py", str(tmp_path / "spec.json")])
    try:
        code = ns["main"]()
    except StopIteration:
        code = None
    report = tmp_path / "ref.json"
    return code, json.loads(report.read_text()) if report.exists() else None, decoded


def test_the_image_reference_script_decodes_the_bytes_it_hashed(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    code, _, decoded = _image_reference_main(tmp_path, monkeypatch, paths, digests)
    # The stand-in decoder stops the run at the first picture, given its bytes.
    assert code is None and decoded == [PICTURES[0]]


def test_the_image_reference_script_loads_its_processor_from_verifys_copy(tmp_path, monkeypatch):
    paths, digests = _pictures(tmp_path)
    loads: list[tuple[str, str]] = []
    _image_reference_main(tmp_path, monkeypatch, paths, digests, loads)
    model, processor = str(tmp_path / "ref"), str(tmp_path / "processor")
    assert loads == [("AutoConfig", model), ("AutoModel", model), ("AutoProcessor", processor)]


def test_the_image_reference_script_refuses_a_picture_changed_after_the_check(
    tmp_path, monkeypatch
):
    paths, digests = _pictures(tmp_path)
    Path(paths[0]).write_bytes(b"another picture")
    code, report, decoded = _image_reference_main(tmp_path, monkeypatch, paths, digests)
    assert code == 3 and decoded == []
    assert report is not None
    found = hashlib.sha256(b"another picture").hexdigest()
    assert report["error"].startswith(f"{paths[0]} hashes to {found}, not the {digests[0]}")


@pytest.mark.parametrize("digests", [None, ["a" * 64]])
def test_the_image_reference_script_refuses_digests_that_are_not_one_per_image(
    tmp_path, monkeypatch, digests
):
    paths, _ = _pictures(tmp_path)
    code, report, decoded = _image_reference_main(tmp_path, monkeypatch, paths, digests)
    assert code == 3 and decoded == []
    assert report is not None and "sha256 for 2 images" in report["error"]


def test_verify_gives_each_script_the_rows_digests_in_the_inputs_order(fake, tmp_path):
    s = _image_setup(tmp_path, fake)
    assert _verify(s).status is Status.PASSED
    digests = _image_digests(s.train) + _image_digests(s.heldout)
    for side in ("reference", "candidate"):
        assert fake.specs[side]["image_sha256"] == digests
        assert fake.texts[side] == _image_paths(s.train) + _image_paths(s.heldout)


def test_the_image_reference_script_refuses_unusable_rows():
    ns: dict[str, Any] = {"__name__": "image_reference"}
    exec(compile(embedding_verify.IMAGE_REFERENCE_SCRIPT, "reference.py", "exec"), ns)
    v = np.array([[1.0, 0.0], [0.0, 0.0], [np.nan, 1.0]])
    assert ns["unusable_rows"](np, v).tolist() == [1, 2]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_an_image_checkpoint_runs_on_the_gpu_off_macos(fake, tmp_path, monkeypatch, platform):
    s = _image_setup(tmp_path, fake)
    monkeypatch.setattr(sys, "platform", platform)
    if platform == "linux":
        fake.gpu_device = LINUX_LISTED
        fake.stderr_for = {"candidate": L4_LINE}
    else:
        fake.stderr_for = {"candidate": ARC_LINE}
        fake.gpu_device = {
            "at_engine": _windows_reading(10),
            "after_generation": _windows_reading(99),
        }
    m = _verify(s, backend="gpu").manifest
    assert m["status"] == "passed", m["checks"]
    assert m["measurements"]["candidate"]["engine"]["backend_observed"] is True
    said = embedding_verify.IMAGE_GPU_MEASURED.format(
        how=embedding_verify.GPU_SEEN, vision_tokens=m["embedding"]["vision_tokens"]
    )
    assert said in m["limitations"]


def test_an_image_checkpoint_on_a_software_adapter_fails_the_harness(fake, tmp_path, monkeypatch):
    s = _image_setup(tmp_path, fake)
    monkeypatch.setattr(sys, "platform", "linux")
    fake.stderr_for = {"candidate": LLVMPIPE_LINE}
    result = _verify(s, backend="gpu")
    assert result.status is Status.FAILED_HARNESS
    assert "selected a software adapter, llvmpipe" in json.dumps(result.manifest["checks"])


def test_a_failed_cpu_side_is_not_blamed_on_an_adapter_line(fake, tmp_path):
    fake.stderr_for = {"candidate": LLVMPIPE_LINE}
    fake.fail = {"candidate"}
    result = _verify(_setup(tmp_path))
    assert result.status is Status.FAILED_HARNESS
    detail = json.dumps(result.manifest["checks"])
    assert "the script exited 1" in detail and "software adapter" not in detail


@pytest.mark.parametrize("line", [LLVMPIPE_LINE, L4_LINE])
def test_the_engine_script_stops_once_its_engine_is_on_a_software_adapter(
    tmp_path, monkeypatch, line
):
    """The launcher sends the side's stderr to a file the script is told of; the
    runtime has named its adapter there by the time the engine exists."""
    code, report, calls = _engine_main(
        tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], ["a", "b"], backend="gpu", stderr=line
    )
    if line is LLVMPIPE_LINE:
        assert code == 3 and calls["texts"] == []
        assert "selected a software adapter, llvmpipe (LLVM 15.0.7, 256 bits)" in report["error"]
    else:
        assert code == 0 and len(calls["texts"]) == 2


def test_the_engine_script_on_the_cpu_does_not_read_its_stderr(tmp_path, monkeypatch):
    code, _, calls = _engine_main(
        tmp_path, monkeypatch, [[3.0, 4.0], [0.0, 2.0]], ["a", "b"], stderr=LLVMPIPE_LINE
    )
    assert code == 0 and len(calls["texts"]) == 2


def test_the_image_engine_script_stops_once_its_engine_is_on_a_software_adapter(
    tmp_path, monkeypatch
):
    paths, digests = _pictures(tmp_path)
    code, report, calls = _image_engine_main(
        tmp_path,
        monkeypatch,
        [[3.0, 4.0], [0.0, 2.0]],
        paths,
        digests,
        backend="gpu",
        stderr=LLVMPIPE_LINE,
    )
    assert code == 3 and calls["texts"] == []
    assert "software adapter, llvmpipe" in report["error"]


def test_each_engine_side_is_told_where_its_stderr_goes(fake, tmp_path, on_macos):
    _verify(_setup(tmp_path), backend="gpu", with_base=True)
    for side in ("candidate", "base"):
        assert fake.specs[side]["stderr"].endswith(f"{side}.err")
    (launch,) = fake.launches
    assert [r["stderr"] for r in launch["runs"]] == [
        fake.specs[r["name"]]["stderr"] for r in launch["runs"]
    ]


def test_the_script_and_the_parent_tell_a_software_adapter_alike(tmp_path):
    """The rule is written twice, once for the script, which cannot import it."""
    from litetune.evaluate import RUNTIME_ENGINE_SOURCE, adapters_of, software_adapter

    ns: dict[str, Any] = {}
    exec(RUNTIME_ENGINE_SOURCE, ns)
    lines = [
        LLVMPIPE_LINE,
        L4_LINE,
        "Selected adapter: Some GPU, arch=software, vendor=x, backend=Vulkan, adapterType=Unknown",
        "Selected adapter: Some GPU, arch=x, vendor=x, backend=Vulkan, adapterType=CPU / Software",
        "Selected adapter: SwiftShader Device, arch=x, vendor=google, backend=Vulkan, "
        "adapterType=Unknown",
        "Selected adapter: Microsoft Basic Render Driver, arch=x, vendor=microsoft, "
        "backend=Direct3D 12, adapterType=Unknown",
    ]
    for i, line in enumerate(lines):
        path = tmp_path / f"{i}.err"
        path.write_text(line, encoding="utf-8")
        script = ns["software_adapter_in"](str(path))
        parent = software_adapter(adapters_of(line))
        assert (script is None) is (parent is None), line
        assert script is None or script == parent
    assert ns["software_adapter_in"](str(tmp_path / "missing.err")) is None


def test_run_script_refuses_a_name_that_shadows_a_standard_module(tmp_path):
    class NeverRun:
        def run(self, argv, timeout=None):
            raise AssertionError("ran a script that shadows a standard module")

    with pytest.raises(ValueError, match="shadow the module json"):
        embedding_verify.run_script(NeverRun(), "", {}, tmp_path, "json", 1)
    assert not (tmp_path / "json.py").exists()
    with pytest.raises(ValueError, match="shadow the module torch"):
        embedding_verify.run_script(NeverRun(), "", {}, tmp_path, "torch", 1)
