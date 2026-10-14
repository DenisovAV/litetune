"""`decide`: an embedding decider fitted here, held to the Dart code it ports.

The expected numbers are flutter_edge_ai's own: `tests/fixtures/decide/` was written by its
decider code at 77f0484e, run on fixed vectors (that directory's README.md says how). The fit
under test is the stage script's, exec'd from its one source (`decide._script()`), and the
stage runs that script for real against a fake `litert_lm` whose engine serves the fixtures'
vectors -- the only part not run is LiteRT-LM itself.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import mark_provisioned

from litetune import decide, envs
from litetune.cli import main
from litetune.decide import (
    ArtifactError,
    DecideError,
    DecideRequest,
    check_artifact,
    escape,
    identity,
    plan,
    read_options,
    read_rows,
    run_decide,
    space_id,
)

FIXTURES = Path(__file__).parent / "fixtures" / "decide"
GOLDEN_NAME = "embeddinggemma-2-text-270m.litertlm"
GOLDEN_BYTES = 164626432


def golden(name: str) -> dict[str, Any]:
    case: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    case["table"] = json.loads(
        (FIXTURES / f"vectors-{case['vectors']}.json").read_text(encoding="utf-8")
    )
    return case


CASES = sorted(
    p.stem
    for p in FIXTURES.glob("*.json")
    if not p.stem.startswith(("vectors-", "fingerprints", "space-ids"))
)
FITTED = [name for name in CASES if not name.startswith("refused-")]
REFUSED = [name for name in CASES if name.startswith("refused-")]
LINEAR = [name for name in FITTED if golden(name)["method"] == "linear"]


def script() -> dict[str, Any]:
    return decide._script()


def unit_rows(case: dict[str, Any], texts: list[str]) -> Any:
    unit = script()["unit_row"]
    return np.stack([unit(case["table"][t], case["dimension"], t) for t in texts])


def fit_case(case: dict[str, Any]) -> dict[str, Any]:
    """The script's fit on a golden case's inputs, in `createDecider`'s order."""
    ids = [o["id"] for o in case["options"]]
    texts, labels, ranks = [], [], []
    for o, option in enumerate(case["options"]):
        inputs = ([option["description"]] if "description" in option else []) + option["examples"]
        for rank, text in enumerate(inputs):
            texts.append(text)
            labels.append(o)
            ranks.append(rank)
    labelled = case["labelled"] or None
    return script()["fit_decider"](
        unit_rows(case, texts),
        labels,
        ranks,
        len(ids),
        ids,
        case["method"] == "linear",
        case["calibration"],
        labelled_x=None if labelled is None else unit_rows(case, [r["text"] for r in labelled]),
        labelled_labels=None if labelled is None else [ids.index(r["optionId"]) for r in labelled],
    )


def options_of(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": o["id"],
            "description": o.get("description"),
            "metadata": o.get("metadata"),
            "examples": o["examples"],
        }
        for o in case["options"]
    ]


def float32s(text: str) -> Any:
    return np.frombuffer(base64.b64decode(text), dtype="<f4")


def within_one_float32_step(a: Any, b: Any) -> bool:
    """Equal, or adjacent float32 values: a last-bit difference in the linear layer."""
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return bool(np.all((a == b) | (np.nextafter(a, b) == b)))


def same_artifact(ours: Any, dart: Any, where: str = "artifact") -> None:
    """Field for field: keys in order, every value equal -- the linear layer's float32 values
    within one step of each other, the temperature equal as a float32 (the grid value is
    `exp` of the platform's libm, decider_fit.dart L430)."""
    if isinstance(dart, dict):
        assert list(ours) == list(dart), where
        for key in dart:
            same_artifact(ours[key], dart[key], f"{where}.{key}")
    elif isinstance(dart, list):
        assert len(ours) == len(dart), where
        for i, (a, b) in enumerate(zip(ours, dart, strict=True)):
            same_artifact(a, b, f"{where}[{i}]")
    elif where.endswith(".weights"):
        assert within_one_float32_step(float32s(ours), float32s(dart)), where
    elif where.endswith(".bias"):
        assert within_one_float32_step(ours, dart), where
    elif where.endswith(".temperature"):
        assert np.float32(ours) == np.float32(dart), where
    else:
        assert type(ours) is type(dart) and ours == dart, (where, ours, dart)


# -- the fit against the Dart goldens --------------------------------------------------


@pytest.mark.parametrize("name", FITTED)
def test_centroids_and_the_cross_fit_are_the_darts_to_the_bit(name):
    case = golden(name)
    fitted = fit_case(case)
    internals = case["internals"]
    count = len(case["options"])
    expected = np.array(internals["centroids64"]).reshape(count, case["dimension"])
    assert np.array_equal(fitted["centroids64"], expected)
    assert (
        base64.b64encode(fitted["centroids"].astype("<f4").tobytes()).decode()
        == (case["fit"]["centroids"])
    )
    if "crossFitSignal" in internals:
        assert np.array_equal(fitted["signal"], np.array(internals["crossFitSignal"]))


@pytest.mark.parametrize("name", FITTED)
def test_the_threshold_and_temperature_are_the_darts(name):
    case = golden(name)
    fitted = fit_case(case)
    assert fitted["threshold"] == case["fit"]["threshold"]
    expected = case["fit"]["temperature"]
    if expected is None:
        assert fitted["temperature"] is None
    else:
        assert np.float32(fitted["temperature"]) == np.float32(expected)


@pytest.mark.parametrize("name", LINEAR)
def test_the_linear_layer_takes_the_darts_path(name):
    case = golden(name)
    linear = fit_case(case)["linear"]
    dart = case["internals"]["linear"]
    assert (linear["iterations"], linear["evaluations"]) == (
        dart["iterations"],
        dart["evaluations"],
    )
    assert linear["loss"] == pytest.approx(dart["loss"], rel=1e-9)
    weights = np.array(dart["weights64"]).reshape(linear["weights"].shape)
    scale = np.abs(weights).max()
    assert np.abs(linear["weights"] - weights).max() <= 1e-9 * scale
    assert np.abs(linear["bias"] - np.array(dart["bias64"])).max() <= 1e-9 * scale


@pytest.mark.parametrize("name", FITTED)
def test_the_artifact_is_the_darts_field_for_field(name):
    case = golden(name)
    artifact = script()["to_artifact"](
        case["artifact"]["spaceId"],
        case["dimension"],
        case["method"],
        case["calibration"],
        options_of(case),
        fit_case(case),
    )
    same_artifact(json.loads(json.dumps(artifact, allow_nan=False)), case["artifact"])


@pytest.mark.parametrize("name", [n for n in FITTED if golden(n)["probes"]])
def test_a_loaded_artifact_answers_as_the_darts_decider(name):
    case = golden(name)
    probes = case["probes"]
    top, similarity, abstain, scores = script()["decisions"](
        unit_rows(case, [p["text"] for p in probes]), case["artifact"]
    )
    ids = [o["id"] for o in case["options"]]
    for i, answer in enumerate(probes):
        assert similarity[i] == answer["similarity"]
        assert bool(abstain[i]) == (answer["abstainReason"] == "belowThreshold")
        assert ids[top[i]] == answer["ranked"][0]["id"]
        assert answer["choice"] == (None if abstain[i] else ids[top[i]])
        for ranked in answer["ranked"]:
            assert scores[i][ids.index(ranked["id"])] == pytest.approx(ranked["score"], abs=1e-12)


# The words each refusal shares with the Dart one.
REFUSED_SAYING = {
    "refused-temperature-floor": "the best softmax temperature is the end of the grid (1e-3",
    "refused-temperature-ceiling": "the best softmax temperature is the end of the grid (10",
    "refused-no-threshold": "no threshold keeps the error on the",
    "refused-cancelling-inputs": 'The inputs of option "cancel" cancel out',
}


@pytest.mark.parametrize("name", REFUSED)
def test_what_the_dart_fit_refuses_is_refused(name):
    case = golden(name)
    said = REFUSED_SAYING[name]
    assert said in case["refused"]["message"]
    with pytest.raises(script()["Refused"]) as raised:
        fit_case(case)
    assert said in str(raised.value)


def _xorshift(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def fingerprint_request(c: int) -> tuple[Any, list[int], list[int]]:
    """`fingerprintRequest` (test/decision/fit_fingerprint.dart): 10 unit rows per option
    near `c` directions, 768 dimensions, from a 32-bit xorshift."""
    d, k = 768, 10
    seq_sum = script()["seq_sum"]
    x = np.zeros((c * k, d))
    state = 0x9E3779B9 ^ c
    for i in range(c * k):
        row = []
        for j in range(d):
            state = _xorshift(state)
            row.append((1.0 if j % c == i // k else 0.0) + state / 4294967296.0 - 0.5)
        v = np.array(row)
        x[i] = v / math.sqrt(seq_sum(v * v))
    return x, [i // k for i in range(c * k)], [i % k for i in range(c * k)]


def weighted(values: Any) -> float:
    """`_weighted` (fit_fingerprint.dart): sum of value * (i % 13 + 1), in order."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    result: float = script()["seq_sum"](flat * (np.arange(flat.size) % 13 + 1))
    return result


FINGERPRINTS = json.loads((FIXTURES / "fingerprints.json").read_text(encoding="utf-8"))["fits"]


@pytest.mark.parametrize(
    "row",
    FINGERPRINTS,
    ids=[f"{r['options']}-{'linear' if r['linear'] else 'centroid'}" for r in FINGERPRINTS],
)
def test_flutter_edge_ais_own_fit_fingerprints(row):
    """`expectFingerprint` (fit_fingerprint.dart): the threshold, centroids and iterations
    exactly, the loss, weights and bias to 1e-9 relative."""
    c = row["options"]
    x, labels, ranks = fingerprint_request(c)
    assert weighted(x) == row["inputSum"]
    ids = [f"o{o}" for o in range(c)]
    fitted = script()["fit_decider"](
        x, labels, ranks, c, ids, row["linear"], {"kind": "crossFit", "keep": 0.95}
    )
    assert fitted["threshold"] == row["threshold"]
    assert weighted(fitted["centroids"]) == row["centroids"]
    if not row["linear"]:
        assert fitted["linear"] is None
    else:
        assert fitted["linear"]["iterations"] == row["iterations"]
        for key, value in (
            ("loss", fitted["linear"]["loss"]),
            ("weights", weighted(fitted["weights"])),
            ("bias", weighted(fitted["bias"])),
        ):
            assert value == pytest.approx(row[key], rel=1e-9), key
    if c == 10:
        assert np.array_equal(fitted["signal"], np.array(row["crossFitSignal"]))
        encoded = base64.b64encode(fitted["centroids"].astype("<f4").tobytes()).decode()
        assert encoded == row["centroidsB64"]
        if row["linear"]:
            assert fitted["linear"]["evaluations"] == row["evaluations"]
            assert within_one_float32_step(fitted["weights"].ravel(), float32s(row["weightsB64"]))
            assert within_one_float32_step(fitted["bias"], row["biasValues"])


# -- the pieces the goldens do not reach -----------------------------------------------


def test_keep_rounds_its_index_as_dart_does():
    keep = script()["keep_threshold"]
    signal = [float(i) for i in range(10)]
    # (1 - 0.9) * 10 is 0.9999999999999998: index 1, not 0.
    assert keep(signal, 0.9) == 1.0
    assert keep(signal, 1.0) == 0.0
    # Index n would be past the end: the last.
    assert keep([3.0, 1.0, 2.0], 1e-12) == 3.0


def test_a_negative_zero_sorts_first_as_dart_orders_doubles():
    keep, budget = script()["keep_threshold"], script()["error_budget_threshold"]
    assert math.copysign(1.0, keep([0.0, -0.0, 0.5], 1.0)) == -1.0
    # Descending, +0.0 leads its group of equal signals.
    assert math.copysign(1.0, budget([-0.0, 0.0], [True, True], 0.0)) == 1.0


def test_the_error_budget_counts_rows_of_one_signal_together():
    budget = script()["error_budget_threshold"]
    signal = [0.9, 0.8, 0.5, 0.5]
    # Row by row, the right 0.5 would be admitted before its wrong twin.
    assert budget(signal, [True, True, True, False], 0.2) == 0.8
    assert budget(signal, [True, True, True, False], 0.25) == 0.5
    assert budget(signal, [False, True, True, True], 0.0) == 1.0 + 1e-9


def test_a_sum_is_taken_in_order_from_positive_zero():
    seq_sum = script()["seq_sum"]
    terms = [1e16, 1.0, -1e16, 1.0]
    assert seq_sum(terms) == ((((0.0 + 1e16) + 1.0) - 1e16) + 1.0)
    assert seq_sum(terms) != math.fsum(terms)
    assert math.copysign(1.0, seq_sum([-0.0, -0.0])) == 1.0
    assert seq_sum([]) == 0.0


def test_an_embedded_row_is_refused_as_dart_refuses_it():
    unit, refused = script()["unit_row"], script()["Refused"]
    assert np.array_equal(unit([3.0, 4.0], 2, "x"), np.array([0.6, 0.8]))
    with pytest.raises(refused, match="returned 3 values for row 7; its dimension is 2"):
        unit([1.0, 2.0, 3.0], 2, "row 7")
    with pytest.raises(refused, match="non-finite value for row 7"):
        unit([1.0, math.nan], 2, "row 7")
    with pytest.raises(refused, match="zero vector for row 7"):
        unit([0.0, -0.0], 2, "row 7")


def test_a_fitted_value_that_is_not_finite_is_refused():
    check, refused = script()["check_fitted"], script()["Refused"]
    good = np.ones((2, 3), dtype=np.float32)
    check(good, threshold=0.5, temperature=1.0)
    with pytest.raises(refused, match="non-finite centroid"):
        check(np.array([[math.nan, 1.0]], dtype=np.float32))
    with pytest.raises(refused, match=r"zero centroid \(option 1\)"):
        check(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    with pytest.raises(refused, match="non-finite linear weight"):
        check(good, weights=np.array([math.inf], dtype=np.float32))
    with pytest.raises(refused, match="non-finite linear bias"):
        check(good, bias=np.array([math.nan], dtype=np.float32))
    for key in ("loss", "threshold", "temperature"):
        with pytest.raises(refused, match="non-finite"):
            check(good, **{key: math.inf})


def test_a_non_finite_score_never_reaches_the_temperature():
    with pytest.raises(script()["Refused"], match="non-finite score"):
        script()["fit_temperature"](np.array([[0.1, math.nan]]), [0])


def test_dart_min_keeps_nan():
    dart_min = script()["dart_min"]
    assert math.isnan(dart_min(1.0, math.nan)) and math.isnan(dart_min(math.nan, 1.0))
    assert dart_min(1.0, 0.5) == 0.5 and dart_min(0.5, 1.0) == 0.5


def test_measure_counts_from_the_artifact():
    case = golden("small-linear-labelled")
    probes = case["probes"]
    ids = [o["id"] for o in case["options"]]
    # The first eight probes are near an option (generator `smallSet`); the rest are not.
    labels: list[int | None] = [
        ids.index(p["text"].split()[1]) if not p["text"].startswith("probe elsewhere") else None
        for p in probes
    ]
    counts = script()["measure"](
        unit_rows(case, [p["text"] for p in probes]), labels, case["artifact"]
    )
    inside = [p for p, label in zip(probes, labels, strict=True) if label is not None]
    outside = [p for p, label in zip(probes, labels, strict=True) if label is None]
    answered = [p for p in inside if p["choice"] is not None]
    assert counts == {
        "in_scope": len(inside),
        "argmax_right": sum(
            1
            for p, label in zip(probes, labels, strict=True)
            if label is not None and p["ranked"][0]["id"] == ids[label]
        ),
        "answered": len(answered),
        "answered_wrong": sum(1 for p in answered if p["choice"] != p["text"].split()[1]),
        "out_of_scope": len(outside),
        "rejected": sum(1 for p in outside if p["choice"] is None),
    }
    assert 0 < counts["answered"] < counts["in_scope"] or counts["rejected"] > 0


# -- the embedding-space id ------------------------------------------------------------


@pytest.mark.parametrize(
    "case", json.loads((FIXTURES / "space-ids.json").read_text(encoding="utf-8"))["cases"]
)
def test_the_space_id_is_deciderSpaceIds(case):
    if "embedderId" in case:
        model = identity(install_name=None, size=None, embedder_id=case["embedderId"])
    else:
        (installed,) = case["installedFiles"]
        model = identity(install_name=installed["name"], size=installed["bytes"], embedder_id=None)
    assert space_id(model, case["dimension"]) == case["spaceId"]


def test_every_golden_artifact_carries_the_installed_files_space_id():
    model = identity(install_name=GOLDEN_NAME, size=GOLDEN_BYTES, embedder_id=None)
    for name in FITTED:
        case = golden(name)
        assert case["artifact"]["spaceId"] == space_id(model, case["dimension"])


def test_escape_takes_exactly_the_characters_dart_does():
    escaped = {chr(c) for c in range(0x80) if escape(chr(c)) != chr(c)}
    assert escaped == {chr(c) for c in range(0x20)} | set("%+;@")
    assert escape("\x7f é😀{x}") == "\x7f é😀{x}"
    assert escape("a;b") == "a%3Bb" and escape("\n") == "%0A"
    with pytest.raises(ValueError):
        identity(install_name=None, size=None, embedder_id=None)


# -- the loader's checks ---------------------------------------------------------------


def valid(name: str = "small-linear-labelled") -> tuple[dict[str, Any], str, int]:
    case = golden(name)
    return copy.deepcopy(case["artifact"]), case["artifact"]["spaceId"], case["dimension"]


@pytest.mark.parametrize("name", FITTED)
def test_every_dart_artifact_passes_the_loader_port(name):
    artifact, sid, dimension = valid(name)
    check_artifact(artifact, expected_space_id=sid, dimension=dimension)


def _set(path: tuple[Any, ...], value: Any) -> Any:
    def mutate(artifact: dict[str, Any]) -> None:
        target: Any = artifact
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return mutate


_DELETE = object()
UNIT_NOT = base64.b64encode(np.full(16, 0.5, dtype="<f4").tobytes()).decode()
NAN_VECTOR = base64.b64encode(np.full(16, np.nan, dtype="<f4").tobytes()).decode()
SHORT = base64.b64encode(np.ones(15, dtype="<f4").tobytes()).decode()

LOADER_REFUSALS = [
    (_set(("format",), "other"), "Not a decider artifact: format is other"),
    (_set(("version",), 2), "Decider artifact version 2; this release reads version 1 only."),
    (_set(("version",), True), "Decider artifact version True"),
    (_set(("version",), "1"), "Decider artifact version 1;"),
    (_set(("kind",), "inference"), 'A "inference" decider artifact cannot be loaded'),
    (_set(("taskType",), _DELETE), 'field "taskType" is Null, expected String'),
    (_set(("taskType",), "clustering"), 'taskType "clustering": this release embeds'),
    (_set(("spaceId",), 7), 'field "spaceId" is int, expected String'),
    (_set(("imageInputs",), "yes"), "imageInputs is not a bool"),
    (_set(("spaceId",), "model=x;dim=16"), "fitted in another embedding space"),
    (_set(("dimension",), 16.0), 'field "dimension" is double, expected int'),
    (_set(("dimension",), True), 'field "dimension" is bool, expected int'),
    (_set(("method",), "knn"), 'Unknown decider method "knn"'),
    (_set(("calibration",), []), 'field "calibration" is List<dynamic>'),
    (_set(("calibration", "kind"), "magic"), 'Unknown calibration kind "magic"'),
    (_set(("calibration", "threshold"), "0.5"), 'field "threshold" is not a number'),
    (_set(("calibration", "threshold"), True), 'field "threshold" is not a number'),
    (_set(("calibration", "temperature"), math.inf), 'field "temperature" is not a number'),
    (_set(("calibration", "threshold"), _DELETE), "A labelled calibration needs a threshold"),
    (_set(("calibration", "temperature"), _DELETE), "needs its temperature"),
    (_set(("calibration", "temperature"), 0.0), "temperature must be positive"),
    (_set(("calibration", "temperature"), -1.0), "temperature must be positive"),
    (_set(("options",), []), "A decider artifact needs options."),
    (_set(("options",), {}), 'field "options" is _Map'),
    (_set(("options", 0), "a"), "Option 0 of the decider artifact is not a map."),
    (_set(("options", 1, "id"), 3), 'Option 1: "id" is int, not text.'),
    (_set(("options", 1, "description"), 3), 'Option 1: "description" is int, not text.'),
    (_set(("options", 1, "examples"), "x"), 'Option 1: "examples" is String, not a list.'),
    (_set(("options", 1, "examples"), ["x", None]), '"examples" holds a Null, not text.'),
    (_set(("options", 1, "metadata"), [1]), 'Option 1: "metadata" is not a map.'),
    (_set(("options", 1, "id"), ""), "Option 1 of the decider artifact: must not be empty"),
    (_set(("options", 0, "description"), ""), "must not be empty; leave it out instead"),
    (_set(("options", 1, "mediaExamples"), -1), "Option 1: mediaExamples is not a count."),
    (_set(("options", 1, "mediaExamples"), 1.0), "Option 1: mediaExamples is not a count."),
    (_set(("options", 1, "centroid"), None), "Option 1 has no encoded vector."),
    (_set(("options", 1, "centroid"), "@@@"), "Option 1: a vector is not valid base64."),
    (_set(("options", 1, "centroid"), SHORT), "a vector has 15 values, expected 16."),
    (_set(("options", 1, "centroid"), NAN_VECTOR), "Option 1: a vector value is not finite."),
    (_set(("options", 1, "centroid"), UNIT_NOT), "Option 1: its centroid is not unit length"),
    (_set(("options", 1, "weights"), SHORT), "a vector has 15 values, expected 16."),
    (_set(("options", 1, "bias"), None), "Option 1 has no finite bias."),
    (_set(("options", 1, "bias"), math.nan), "Option 1 has no finite bias."),
    (_set(("options", 1, "id"), "alpha"), "Two options of the artifact share an id."),
]


@pytest.mark.parametrize(("mutate", "message"), LOADER_REFUSALS)
def test_the_loader_port_refuses_what_loadDecider_refuses(mutate, message):
    artifact, sid, dimension = valid()
    mutate(artifact)
    with pytest.raises(ArtifactError) as raised:
        check_artifact(artifact, expected_space_id=sid, dimension=dimension)
    assert message in str(raised.value)


def test_the_loader_port_reads_what_loadDecider_reads():
    artifact, sid, dimension = valid()
    artifact["version"] = 1.0  # Dart's `1.0 != 1` is false
    artifact["imageInputs"] = False
    artifact["options"][1]["mediaExamples"] = 2
    artifact["options"][1]["metadata"] = {"k": [1, None]}
    check_artifact(artifact, expected_space_id=sid, dimension=dimension)
    with pytest.raises(ArtifactError, match="does not match its space id"):
        check_artifact(artifact, expected_space_id=sid, dimension=17)
    with pytest.raises(ArtifactError, match="not a JSON object"):
        check_artifact([artifact], expected_space_id=sid, dimension=dimension)


def test_only_the_calibrations_own_fields_pass():
    for name, mutate, message in (
        ("small-linear-none", _set(("calibration", "threshold"), 0.5), "never abstains"),
        ("small-centroid-crossfit", _set(("calibration", "temperature"), 1.0), "Only a labelled"),
        ("small-centroid-crossfit", _set(("calibration", "threshold"), _DELETE), "needs a"),
    ):
        artifact, sid, dimension = valid(name)
        mutate(artifact)
        with pytest.raises(ArtifactError, match=message):
            check_artifact(artifact, expected_space_id=sid, dimension=dimension)


def test_a_linear_artifact_needs_two_options():
    artifact, sid, dimension = valid()
    artifact["options"] = artifact["options"][:1]
    with pytest.raises(ArtifactError, match="needs at least two options"):
        check_artifact(artifact, expected_space_id=sid, dimension=dimension)
    artifact, sid, dimension = valid("small-centroid-crossfit")
    artifact["options"] = artifact["options"][:1]
    check_artifact(artifact, expected_space_id=sid, dimension=dimension)


# -- reading the inputs, and createDecider's refusals ----------------------------------


def jsonl(path: Path, rows: list[Any]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def request(tmp_path: Path, **kwargs: Any) -> DecideRequest:
    embedder = tmp_path / "embedder.litertlm"
    if not embedder.exists():
        embedder.write_bytes(b"\0" * 100)
    fields: dict[str, Any] = {
        "train": tmp_path / "train.jsonl",
        "embedder": embedder,
        "output_dir": tmp_path / "out",
    }
    fields.update(kwargs)
    return DecideRequest(**fields)


TRAIN = [
    {"text": "a one", "label": "a"},
    {"text": "b one", "label": "b"},
    {"text": "a two", "label": "a"},
    {"text": "b two", "label": "b"},
    {"text": "a three", "label": "a"},
]


def test_without_options_the_labels_are_the_options_in_first_seen_order(tmp_path):
    jsonl(tmp_path / "train.jsonl", TRAIN)
    planned = plan(request(tmp_path, examples_in_artifact=2))
    assert [o.id for o in planned.options] == ["a", "b"]
    assert planned.stored_examples == (("a one", "a two"), ("b one", "b two"))
    inputs = planned.inputs()
    assert inputs["texts"] == ["a one", "a two", "a three", "b one", "b two"]
    assert inputs["options"] == [0, 0, 0, 1, 1] and inputs["ranks"] == [0, 1, 2, 0, 1]
    assert inputs["names"][0] == "train.jsonl line 1"


def test_a_description_is_its_options_first_input(tmp_path):
    jsonl(tmp_path / "train.jsonl", TRAIN)
    (tmp_path / "options.json").write_text(
        json.dumps([{"id": "b", "description": "about b", "metadata": {"k": 1}}, {"id": "a"}])
    )
    planned = plan(request(tmp_path, options=tmp_path / "options.json"))
    inputs = planned.inputs()
    assert inputs["texts"][:3] == ["about b", "b one", "b two"]
    assert inputs["ranks"][:3] == [0, 1, 2] and inputs["names"][0] == 'option "b", its description'
    assert planned.options[0].metadata == {"k": 1}
    assert planned.stored_examples[0] == ("b one", "b two")


@pytest.mark.parametrize(
    ("rows", "options", "kwargs", "message"),
    [
        ([{"text": "x", "label": "zz"}, *TRAIN], [{"id": "a"}, {"id": "b"}], {}, "names no option"),
        (TRAIN, [{"id": "a"}, {"id": "b"}, {"id": "c"}], {}, 'option "c" has neither'),
        (TRAIN, [{"id": "a"}, {"id": "b"}, {"id": "c", "description": "c"}], {}, "has 1"),
        (TRAIN, [{"id": "a"}, {"id": "a"}], {}, 'two options share the id "a"'),
        (TRAIN, [{"id": ""}], {}, "must not be empty"),
        (TRAIN, [{"id": "a", "description": ""}, {"id": "b"}], {}, "left out instead"),
        (TRAIN, [{"id": "a", "examples": []}, {"id": "b"}], {}, "has ['examples']"),
        (TRAIN, [{"id": 1}], {}, "'id' must be a string"),
        (TRAIN, [{"id": "a", "description": 2}], {}, "'description' must be a string"),
        (TRAIN, [{"id": "a", "metadata": []}], {}, "'metadata' must be an object"),
        (TRAIN, {"id": "a"}, {}, "expected a JSON array"),
        (TRAIN, [], {}, "no options"),
        (TRAIN[:1], None, {"method": "linear", "calibration": "none"}, "at least two options"),
    ],
)
def test_what_createDecider_refuses_is_refused_by_name(tmp_path, rows, options, kwargs, message):
    jsonl(tmp_path / "train.jsonl", rows)
    if options is not None:
        (tmp_path / "options.json").write_text(json.dumps(options))
        kwargs = {**kwargs, "options": tmp_path / "options.json"}
    with pytest.raises(DecideError, match=None) as raised:
        plan(request(tmp_path, **kwargs))
    assert message in str(raised.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"keep": 0.0}, "--keep must be in (0, 1]"),
        ({"keep": 1.5}, "--keep must be in (0, 1]"),
        ({"calibration": "labelled"}, "give them as --calib"),
        ({"calibration": "labelled", "max_error": 1.0}, "--max-error must be in [0, 1)"),
        ({"calibration": "labelled", "max_error": -0.1}, "--max-error must be in [0, 1)"),
        ({"calibration": "minSimilarity"}, "needs --min-similarity"),
        ({"calibration": "minSimilarity", "min_similarity": 1.5}, "must be in [-1, 1]"),
        ({"calibration": "none", "keep": 0.9}, "--keep is not read by --calibration none"),
        ({"max_error": 0.1, "min_similarity": 0.2}, "--max-error, --min-similarity are not read"),
        ({"calibration": "magic"}, "--calibration must be one of"),
        ({"method": "knn"}, "--method must be one of"),
        ({"examples_in_artifact": -1}, "0 or more"),
    ],
)
def test_a_flag_its_calibration_does_not_read_or_cannot_take_is_refused(tmp_path, kwargs, message):
    jsonl(tmp_path / "train.jsonl", TRAIN)
    with pytest.raises(DecideError) as raised:
        plan(request(tmp_path, **kwargs))
    assert message in str(raised.value)


def test_the_flags_take_their_defaults_and_values(tmp_path):
    jsonl(tmp_path / "train.jsonl", TRAIN)
    jsonl(tmp_path / "calib.jsonl", [{"text": "new a", "label": "a"}])
    assert plan(request(tmp_path)).calibration == {"kind": "crossFit", "keep": 0.95}
    assert plan(request(tmp_path, keep=1.0)).calibration == {"kind": "crossFit", "keep": 1.0}
    labelled = request(tmp_path, calibration="labelled", calib=tmp_path / "calib.jsonl")
    assert plan(labelled).calibration == {"kind": "labelled", "maxError": 0.05}
    zero = request(tmp_path, calibration="labelled", calib=tmp_path / "calib.jsonl", max_error=0.0)
    assert plan(zero).calibration == {"kind": "labelled", "maxError": 0.0}
    fixed = request(tmp_path, calibration="minSimilarity", min_similarity=-1.0)
    assert plan(fixed).calibration == {"kind": "minSimilarity", "threshold": -1.0}
    assert plan(request(tmp_path, calibration="none")).calibration == {"kind": "none"}


def test_held_out_rows_must_be_new_and_name_options(tmp_path):
    jsonl(tmp_path / "train.jsonl", TRAIN)
    (tmp_path / "options.json").write_text(
        json.dumps([{"id": "a", "description": "say a"}, {"id": "b"}])
    )
    base = {
        "calibration": "labelled",
        "calib": tmp_path / "calib.jsonl",
        "options": tmp_path / "options.json",
    }
    for rows, message in (
        ([{"text": "a two", "label": "b"}], "repeats a --train row or a description"),
        ([{"text": "say a", "label": "a"}], "repeats a --train row or a description"),
        ([{"text": "new", "label": "c"}], "label 'c' names no option"),
    ):
        jsonl(tmp_path / "calib.jsonl", rows)
        with pytest.raises(DecideError, match=message):
            plan(request(tmp_path, **base))
    jsonl(tmp_path / "test.jsonl", [{"text": "t", "label": "c"}])
    with pytest.raises(DecideError, match="names no option"):
        plan(request(tmp_path, test=tmp_path / "test.jsonl"))


def test_rows_are_read_strictly(tmp_path):
    path = tmp_path / "rows.jsonl"
    for text, message in (
        ('{"text": "x", "label": "a"}\nnot json\n', "rows.jsonl:2: not valid JSON"),
        ('{"text": NaN, "label": "a"}\n', "NaN is not JSON"),
        ('["x"]\n', "expected an object"),
        ('{"text": " ", "label": "a"}\n', "'text' must be a non-empty string"),
        ('{"text": "x"}\n', "'label' must be an option id"),
        ('{"text": "x", "label": null}\n', "'label' must be an option id"),
        ("\n\n", "no rows"),
    ):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(DecideError, match=message):
            read_rows(path)
    path.write_text('{"text": "x", "label": null}\n{"text": "y", "label": "a"}\n')
    rows = read_rows(path, out_of_scope=True)
    assert [(r.text, r.label, r.line) for r in rows] == [("x", None, 1), ("y", "a", 2)]
    path.write_text('{"text": "x"}\n')
    with pytest.raises(DecideError, match="or null for an out-of-scope row"):
        read_rows(path, out_of_scope=True)
    # A line separator inside a string is not the end of a row.
    path.write_text(json.dumps({"text": "a b", "label": "a"}, ensure_ascii=False) + "\n")
    assert read_rows(path)[0].text == "a b"
    path.write_bytes(b"\xff\xfe" + '{"text": "x", "label": "a"}'.encode("utf-16-le"))
    assert read_rows(path)[0].text == "x"
    path.write_bytes(b"\xc3\x28")
    with pytest.raises(DecideError, match="is not text in UTF-8"):
        read_rows(path)
    with pytest.raises(DecideError, match="could not read"):
        read_rows(tmp_path / "missing.jsonl")
    (tmp_path / "o.json").write_text("[{")
    with pytest.raises(DecideError, match="is not valid JSON"):
        read_options(tmp_path / "o.json")


# -- the stage, with litert_lm faked ---------------------------------------------------

FAKE_LITERT_LM = r"""
import json
from pathlib import Path

SETTINGS = json.loads((Path(__file__).parent / "settings.json").read_text(encoding="utf-8"))
LOG = Path(SETTINGS["log"])


def _log(entry):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


class InputOverflowStrategy:
    CHUNK_AND_AVERAGE = "CHUNK_AND_AVERAGE"
    TRUNCATE = "TRUNCATE"
    ERROR = "ERROR"


class EmbeddingOptions:
    def __init__(self, *, normalize=None, insert_special_tokens=None,
                 input_overflow_strategy=None, output_size=None, vision_tokens_per_image=None):
        self.normalize = normalize
        self.insert_special_tokens = insert_special_tokens
        self.input_overflow_strategy = input_overflow_strategy


class CPU:
    def __init__(self, thread_count=None):
        self.thread_count = thread_count


class Backend:
    CPU = CPU


class _Type:
    def __init__(self, name):
        self.name = name


class _Embedding:
    dimension = SETTINGS["dimension"]
    signature_selection = SETTINGS["signatures"]


class ModelInfo:
    def __init__(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        self.model_type = _Type(SETTINGS["model_type"])

    @property
    def is_embedding_model(self):
        return self.model_type.name == "EMBEDDING"

    @property
    def embedding(self):
        return _Embedding() if self.is_embedding_model else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class EmbeddingResponse:
    def __init__(self, embedding):
        self.embedding = embedding


class EmbeddingEngine:
    def __init__(self, model_path, backend=None, cache_dir=None, max_input_length=None, **kw):
        _log({"engine": {"model": model_path, "backend": type(backend).__name__,
                         "cache_dir_exists": Path(cache_dir).is_dir(),
                         "max_input_length": max_input_length}})

    def compute_embedding(self, contents, options=None):
        _log({"text": contents, "normalize": options.normalize,
              "special": options.insert_special_tokens,
              "overflow": options.input_overflow_strategy})
        prefix = SETTINGS["prefix"]
        if not contents.startswith(prefix) or contents[len(prefix):] in SETTINGS["fail"]:
            raise RuntimeError("Failed to compute embedding.")
        return EmbeddingResponse(list(SETTINGS["table"][contents[len(prefix):]]))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None
"""


@dataclass
class FakeToolchain:
    """`StageEnv.run`: the decide script, run for real by this interpreter, against a fake
    `litert_lm` serving `table`."""

    table: dict[str, list[float]] = field(default_factory=dict)
    dimension: int = 16
    signatures: list[int] = field(default_factory=lambda: [256, 512, 2048])
    model_type: str = "EMBEDDING"
    fail: list[str] = field(default_factory=list)
    returncode: int | None = None
    times_out: bool = False
    log: list[dict[str, Any]] = field(default_factory=list)
    specs: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        assert args[0] == "python" and str(args[1]).endswith("decide.py"), args
        if self.times_out:
            raise subprocess.TimeoutExpired(args, timeout)
        if self.returncode is not None:
            return subprocess.CompletedProcess(args, self.returncode, "", "Traceback\nboom\n")
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        self.specs.append(json.loads(Path(spec).read_text(encoding="utf-8")))
        shim = Path(spec).parent / "shim"
        package = shim / "litert_lm"
        package.mkdir(parents=True, exist_ok=True)
        log = shim / "calls.jsonl"
        (package / "__init__.py").write_text(FAKE_LITERT_LM, encoding="utf-8")
        (package / "settings.json").write_text(
            json.dumps(
                {
                    "table": self.table,
                    "dimension": self.dimension,
                    "signatures": self.signatures,
                    "model_type": self.model_type,
                    "fail": self.fail,
                    "prefix": decide.TASK_PREFIX,
                    "log": str(log),
                }
            ),
            encoding="utf-8",
        )
        dist = shim / "litert_lm-0.18.0.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: litert-lm\nVersion: 0.18.0\n")
        path = os.pathsep.join(p for p in (str(shim), os.environ.get("PYTHONPATH")) if p)
        proc = subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": path},
        )
        if log.exists():
            self.log.extend(json.loads(line) for line in log.read_text().splitlines())
        return proc


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


def stage_inputs(tmp_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    """A golden case as `decide`'s files: options with their descriptions and metadata, the
    examples as --train rows, the labelled rows as --calib, the probes as --test."""
    options = []
    for o in case["options"]:
        entry = {"id": o["id"]}
        for key in ("description", "metadata"):
            if key in o:
                entry[key] = o[key]
        options.append(entry)
    (tmp_path / "options.json").write_text(json.dumps(options), encoding="utf-8")
    jsonl(
        tmp_path / "train.jsonl",
        [{"text": e, "label": o["id"]} for o in case["options"] for e in o["examples"]],
    )
    kwargs: dict[str, Any] = {
        "options": tmp_path / "options.json",
        "method": case["method"],
        "calibration": case["calibration"]["kind"],
    }
    if case["calibration"]["kind"] == "labelled":
        jsonl(
            tmp_path / "calib.jsonl",
            [{"text": r["text"], "label": r["optionId"]} for r in case["labelled"]],
        )
        kwargs.update(calib=tmp_path / "calib.jsonl", max_error=case["calibration"]["maxError"])
    if case["calibration"]["kind"] == "crossFit":
        kwargs["keep"] = case["calibration"]["keep"]
    if case["calibration"]["kind"] == "minSimilarity":
        kwargs["min_similarity"] = case["calibration"]["threshold"]
    if case.get("probes"):
        jsonl(
            tmp_path / "test.jsonl",
            [
                {
                    "text": p["text"],
                    "label": None if "elsewhere" in p["text"] else p["text"].split()[1],
                }
                for p in case["probes"]
            ],
        )
        kwargs["test"] = tmp_path / "test.jsonl"
    return kwargs


@pytest.mark.parametrize(
    "name", ["small-linear-labelled", "small-centroid-crossfit", "small-centroid-minsimilarity"]
)
def test_the_stage_writes_the_darts_artifact(toolchain, tmp_path, name):
    case = golden(name)
    toolchain.table = case["table"]
    result = run_decide(request(tmp_path, **stage_inputs(tmp_path, case)))

    written = json.loads(result.artifact.read_text(encoding="utf-8"))
    sid = space_id(identity(install_name="embedder.litertlm", size=100, embedder_id=None), 16)
    assert written["spaceId"] == sid
    same_artifact({**written, "spaceId": case["artifact"]["spaceId"]}, case["artifact"])
    assert json.loads(result.report_path.read_text(encoding="utf-8")) == result.report
    assert not [p for p in result.artifact.parent.iterdir() if p.name.startswith(".")]

    # Every input embedded once, prefixed, as createDecider orders them, with the app's options.
    texts = [entry["text"] for entry in toolchain.log if "text" in entry]
    inputs = [
        t
        for o in case["options"]
        for t in ([o.get("description")] if "description" in o else []) + o["examples"]
    ]
    held = [r["text"] for r in case["labelled"]] + [p["text"] for p in case["probes"]]
    assert texts == [decide.TASK_PREFIX + t for t in inputs + held]
    calls = [entry for entry in toolchain.log if "text" in entry]
    assert {(c["normalize"], c["special"], c["overflow"]) for c in calls} == {(True, True, "ERROR")}
    (engine,) = [entry["engine"] for entry in toolchain.log if "engine" in entry]
    assert engine == {
        "model": str((tmp_path / "embedder.litertlm").absolute()),
        "backend": "CPU",
        "cache_dir_exists": True,
        "max_input_length": 1024,
    }


def test_the_report_is_measured_from_the_artifact_as_the_app_answers(toolchain, tmp_path):
    case = golden("small-linear-labelled")
    toolchain.table = case["table"]
    report = run_decide(request(tmp_path, **stage_inputs(tmp_path, case))).report

    probes = case["probes"]
    inside = [p for p in probes if "elsewhere" not in p["text"]]
    outside = [p for p in probes if "elsewhere" in p["text"]]
    answered = [p for p in inside if p["choice"] is not None]
    measured = report["measurements"]
    assert measured["rows"] == {"in_scope": len(inside), "out_of_scope": len(outside)}
    right = sum(1 for p in inside if p["ranked"][0]["id"] == p["text"].split()[1])
    assert (measured["argmax_accuracy"]["value"], measured["argmax_accuracy"]["n"]) == (
        round(right / len(inside), 6),
        len(inside),
    )
    assert measured["coverage"]["n"] == len(inside)
    assert measured["coverage"]["value"] == round(len(answered) / len(inside), 6)
    wrong = sum(1 for p in answered if p["choice"] != p["text"].split()[1])
    assert measured["error_on_answered"]["value"] == round(wrong / len(answered), 6)
    rejected = sum(1 for p in outside if p["choice"] is None)
    assert measured["out_of_scope_rejected"]["value"] == round(rejected / len(outside), 6)
    assert report["calibration"] == {
        "kind": "labelled",
        "max_error": 0.05,
        "threshold": case["fit"]["threshold"],
        "temperature": pytest.approx(case["fit"]["temperature"], rel=1e-7),
    }
    assert report["linear"]["iterations"] == case["internals"]["linear"]["iterations"]
    assert report["inputs"]["train"]["per_option"] == {o["id"]: 6 for o in case["options"]}
    assert report["inputs"]["test"]["out_of_scope"] == len(outside)
    assert report["runtime"]["litert_lm"] == "0.18.0"
    assert report["embedder"]["dimension"] == 16 and report["embedder"]["max_input_length"] == 1024
    assert (
        "install the embedder under the name 'embedder.litertlm' (100 bytes"
        in report["install_rule"]
    )
    assert report["artifact"]["sha256"].startswith("sha256:")


def test_an_embedder_id_replaces_the_installed_name(toolchain, tmp_path):
    case = golden("small-centroid-crossfit")
    toolchain.table = case["table"]
    kwargs = stage_inputs(tmp_path, case)
    result = run_decide(request(tmp_path, embedder_id="eg2;cpu@v1", **kwargs))
    written = json.loads(result.artifact.read_text(encoding="utf-8"))
    assert written["spaceId"] == space_id("eg2%3Bcpu%40v1", 16)
    assert "pass embedderId 'eg2;cpu@v1'" in result.report["install_rule"]
    named = run_decide(
        request(tmp_path, output_dir=tmp_path / "o2", embedder_name="eg2.litertlm", **kwargs)
    )
    assert json.loads(named.artifact.read_text())["spaceId"] == space_id("eg2.litertlm@100", 16)


def test_the_largest_signature_under_1024_caps_the_input(toolchain, tmp_path):
    case = golden("small-centroid-crossfit")
    toolchain.table, toolchain.signatures = case["table"], [-1, 128, 512]
    run_decide(request(tmp_path, **stage_inputs(tmp_path, case)))
    (engine,) = [entry["engine"] for entry in toolchain.log if "engine" in entry]
    assert engine["max_input_length"] == 512


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ({"fail": ["alpha example 3"]}, "train.jsonl line 4: litert-lm could not embed it"),
        ({"dimension": 15}, "returned 16 values for"),
        ({"model_type": "LLM"}, "is not an embedding model (its model type is LLM)"),
        ({"signatures": [0, -1]}, "declares no text signature length"),
        ({"dimension": 0}, "declares no embedding dimension"),
    ],
)
def test_what_the_script_refuses_is_refused_and_nothing_is_written(
    toolchain, tmp_path, setup, message
):
    case = golden("small-centroid-crossfit")
    toolchain.table = case["table"]
    for key, value in setup.items():
        setattr(toolchain, key, value)
    with pytest.raises(DecideError, match="nothing was written") as raised:
        run_decide(request(tmp_path, **stage_inputs(tmp_path, case)))
    assert message in str(raised.value)
    assert list((tmp_path / "out").iterdir()) == []


def test_a_refused_fit_writes_nothing(toolchain, tmp_path):
    case = golden("refused-no-threshold")
    toolchain.table = case["table"]
    kwargs = stage_inputs(tmp_path, case)
    with pytest.raises(DecideError, match="no threshold keeps the error on the --calib rows"):
        run_decide(request(tmp_path, **kwargs))
    assert list((tmp_path / "out").iterdir()) == []


def test_a_failed_or_slow_script_is_not_a_decider(toolchain, tmp_path):
    case = golden("small-centroid-crossfit")
    kwargs = stage_inputs(tmp_path, case)
    toolchain.returncode = 1
    with pytest.raises(DecideError, match="the decide script exited 1: boom"):
        run_decide(request(tmp_path, **kwargs))
    toolchain.returncode, toolchain.times_out = None, True
    with pytest.raises(DecideError, match="did not finish within 7s"):
        run_decide(request(tmp_path, timeout_s=7, **kwargs))
    assert list((tmp_path / "out").iterdir()) == []


def test_an_artifact_the_loader_would_refuse_is_not_written(toolchain, tmp_path, monkeypatch):
    case = golden("small-centroid-crossfit")
    toolchain.table = case["table"]
    # A script that writes another task type: the loader reads none but classification.
    wrong = decide._DECIDE_SCRIPT.replace(
        '"taskType": "classification"', '"taskType": "clustering"'
    )
    assert wrong != decide._DECIDE_SCRIPT
    monkeypatch.setattr(decide, "_DECIDE_SCRIPT", wrong)
    with pytest.raises(ArtifactError, match="loadDecider would refuse the artifact written"):
        run_decide(request(tmp_path, **stage_inputs(tmp_path, case)))
    assert not (tmp_path / "out" / "decider.json").exists()


def test_existing_outputs_and_a_missing_environment_are_refused_first(
    toolchain, tmp_path, monkeypatch
):
    case = golden("small-centroid-crossfit")
    kwargs = stage_inputs(tmp_path, case)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "decide.json").write_text("{}")
    with pytest.raises(DecideError, match="decide.json already exists"):
        run_decide(request(tmp_path, **kwargs))
    with pytest.raises(DecideError, match="environment 'runtime' is not provisioned"):
        run_decide(request(tmp_path, output_dir=tmp_path / "o2", auto_provision=False, **kwargs))
    with pytest.raises(FileNotFoundError):
        run_decide(request(tmp_path, embedder=tmp_path / "none.litertlm", **kwargs))
    for extra, message in (
        (
            {"embedder_id": "x", "embedder_name": "y"},
            "give one of --embedder-id and --embedder-name",
        ),
        ({"embedder_id": ""}, "--embedder-id must not be empty"),
        ({"embedder_name": "dir/x.litertlm"}, "must be a file name"),
    ):
        with pytest.raises(DecideError, match=message):
            run_decide(request(tmp_path, output_dir=tmp_path / "o3", **extra, **kwargs))
    assert toolchain.specs == []


def test_the_command_writes_and_reports(toolchain, tmp_path, capsys):
    case = golden("small-linear-labelled")
    toolchain.table = case["table"]
    kwargs = stage_inputs(tmp_path, case)
    request(tmp_path)
    argv = [
        "decide", "--train", str(tmp_path / "train.jsonl"),
        "--embedder", str(tmp_path / "embedder.litertlm"),
        "--options", str(kwargs["options"]), "--calib", str(kwargs["calib"]),
        "--test", str(kwargs["test"]),
        "--method", "linear", "--calibration", "labelled", "--examples-in-artifact", "2",
    ]  # fmt: skip
    assert main([*argv, "--output-dir", str(tmp_path / "a"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "litetune.decide/1" and report["outcome"] == "written"
    stored = json.loads((tmp_path / "a" / "decider.json").read_text())["options"][0]["examples"]
    assert stored == case["options"][0]["examples"][:2]
    assert main([*argv, "--output-dir", str(tmp_path / "b")]) == 0
    text = capsys.readouterr().out
    assert "4 options, linear, labelled (threshold" in text
    assert (
        "argmax accuracy (in scope): " in text
        and "space id: model=embedder.litertlm@100;dim=16" in text
    )
    assert main([*argv, "--output-dir", str(tmp_path / "b")]) == 4
    assert "decider.json already exists" in capsys.readouterr().err
    assert (
        main(
            [
                *argv[:-2],
                "--examples-in-artifact",
                "2",
                "--keep",
                "0.9",
                "--output-dir",
                str(tmp_path / "c"),
            ]
        )
        == 4
    )
    assert "--keep is not read by --calibration labelled" in capsys.readouterr().err


def test_the_fixtures_are_the_files_their_readme_hashes():
    """A golden edited by hand stops being the Dart code's output."""
    listed = (FIXTURES / "README.md").read_text(encoding="utf-8").split("## sha256")[1]
    pairs = [line.split("  ") for line in listed.splitlines() if "  " in line]
    assert {name for _, name in pairs} >= {p.name for p in FIXTURES.glob("*.json")}
    for digest, name in pairs:
        # Hashed as checked out, without a newline translation.
        data = (FIXTURES / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
