"""`tune --model google/embeddinggemma-2` on an image split, with no torch and no model here.

`StageEnv.run` is faked by `FakeImageToolchain`, this file's own: it answers the
device probe and, for the image training script, reads the spec the stage wrote
and writes what a real run would -- a checkpoint directory with its one
`model.safetensors` and the metrics. The split is written as `prepare` writes
one: `images/<sha256>.png` beside the JSONL, each row carrying the digest. The
pictures are bytes, not images: the parent only hashes them. The script's
helpers import nothing beyond the standard library and are `exec`'d here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned, probe_answer

from litetune import envs
from litetune.checks import Outcome
from litetune.cli import main
from litetune.decisions import Decision, Option
from litetune.embedding_tune import (
    _EMBED_IMAGE_SCRIPT,
    _EMBED_TRAIN_SCRIPT,
    BASE_CHECKPOINT_FILE,
    DECIDER_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_TYPE,
    EMBEDDING_REVISION,
    IMAGE_CACHE_DIR,
    IMAGE_MAX_SHARD_SIZE,
    IMAGE_PROMPT_NAME,
    IMAGE_RECIPE,
    IMAGE_RECIPE_UNMEASURED,
    IMAGE_SCRIPT_NAME,
    IMAGE_TOKEN_TEXT,
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    EmbeddingTuneError,
    EmbeddingTuneRequest,
    read_embedding_record,
    resolve_swaps,
    run_embedding_tune,
    summarise,
    write_embedding_record,
    write_report,
)
from litetune.tune import TRAINING_CHECK

OPTIONS = ("left", "right", "straight")


# ---------------------------------------------------------------------------
# A fake EMBED_TRAIN environment
# ---------------------------------------------------------------------------


@dataclass
class FakeImageToolchain:
    """Stands in for `envs.StageEnv.run`: the device probe, then the image script."""

    probe_device: str | None = "cpu"
    probe_fields: dict[str, Any] = field(default_factory=dict)
    returncode: int = 0
    stderr: str = ""
    write_safetensors: bool = True
    metrics_overrides: dict[str, Any] = field(default_factory=dict)
    specs: list[dict[str, Any]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if len(args) >= 2 and args[1] == "-c":
            stdout = probe_answer(self.probe_device, **self.probe_fields) + "\n"
            return subprocess.CompletedProcess(args, 0, stdout, "")
        self.scripts.append(Path(args[1]).read_text(encoding="utf-8"))
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        model_dir = Path(spec["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": EMBEDDING_MODEL_TYPE}), encoding="utf-8"
        )
        if self.write_safetensors:
            (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        train = Path(spec["train_rows"]).read_text(encoding="utf-8").splitlines()
        heldout = Path(spec["heldout_rows"]).read_text(encoding="utf-8").splitlines()
        items = len(train) * (2 if spec["mirror"] else 1)
        steps = spec["recipe"]["epochs"] * -(-items // spec["recipe"]["batch_size"])
        metrics = {
            "base_model": spec["model"],
            "base_model_revision": spec["revision"],
            "prefix": spec["prefix"],
            "labels": spec["labels"],
            "rows": {"train": len(train), "heldout": len(heldout)},
            "training_items": items,
            "mirror": spec["mirror"],
            "mirror_swap": spec["mirror_swap"],
            "vision_tokens": spec["vision_tokens"],
            "steps_done": steps,
            "steps_planned": steps,
            "train_seconds": 3.5,
            "device": spec["device"] or "cpu",
            "dtype": spec["recipe"]["dtype"],
            "head_train_accuracy": 1.0,
            "head_heldout_accuracy": 0.5,
            "mps_environment": None,
        }
        metrics.update(self.metrics_overrides)
        Path(spec["metrics_out"]).write_text(json.dumps(metrics), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, "", self.stderr)


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeImageToolchain:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeImageToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


# ---------------------------------------------------------------------------
# An image split, as `prepare` writes one
# ---------------------------------------------------------------------------


def _decision(ids: tuple[str, ...] = OPTIONS) -> Decision:
    return Decision("choice", "Where does the road go?", tuple(Option(o) for o in ids))


@dataclass
class Split:
    dir: Path
    train: Path
    heldout: Path
    decision: Path


def _image(root: Path, n: int) -> tuple[str, str]:
    """A picture's bytes, under the name prepare gives it; (relative path, sha256)."""
    data = f"picture {n}".encode()
    sha = hashlib.sha256(data).hexdigest()
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "images" / f"{sha}.png").write_bytes(data)
    return f"images/{sha}.png", sha


def _rows(root: Path, n: int, start: int, labels: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        relative, sha = _image(root, start + i)
        label = labels[i % len(labels)]
        rows.append(
            {
                "image": relative,
                "image_sha256": sha,
                "completion": label,
                "target": label,
                "source_line": start + i + 1,
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


def _split(tmp_path: Path, ids: tuple[str, ...] = OPTIONS, train: int = 12) -> Split:
    root = tmp_path / "split"
    root.mkdir(exist_ok=True)
    decision = root / "decision.json"
    decision.write_text(_decision(ids).canonical_text(), encoding="utf-8")
    return Split(
        root,
        _write(root / "train.jsonl", _rows(root, train, 0, ids)),
        _write(root / "heldout.jsonl", _rows(root, 6, 100, ids)),
        decision,
    )


@pytest.fixture
def split(tmp_path) -> Split:
    return _split(tmp_path)


def _request(split: Split, tmp_path: Path, **kwargs: Any) -> EmbeddingTuneRequest:
    params: dict[str, Any] = {
        "data": split.train,
        "heldout": split.heldout,
        "decision": split.decision,
        "output_dir": tmp_path / "tuned",
        "vision_tokens": 70,
    }
    params.update(kwargs)
    return EmbeddingTuneRequest(**params)


def _check(result, name):
    return next(c for c in result.checks.checks if c.name == name)


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


def test_an_image_run_writes_an_image_record_beside_the_checkpoint(toolchain, split, tmp_path):
    from litetune.storage import hash_file

    result = run_embedding_tune(_request(split, tmp_path))

    assert result.outcome is Outcome.PASSED, result.checks.as_dict()
    assert result.modality == "image"
    record = read_embedding_record(tmp_path / "tuned" / "model")
    assert record is not None and record == result.record
    assert record.is_image
    assert (record.modality, record.vision_tokens) == ("image", 70)
    assert (record.prefix, record.prompt_name) == ("", IMAGE_PROMPT_NAME)
    assert (record.mirror, record.mirror_swap) == (False, ())
    assert record.labels == tuple(sorted(OPTIONS))
    assert record.rows == {"train": 12, "heldout": 6}
    assert record.recipe == IMAGE_RECIPE.as_dict()
    assert record.train_sha256 == hash_file(split.train)
    on_disk = json.loads((tmp_path / "tuned" / "model" / "litetune.json").read_text())
    assert (on_disk["mirror"], on_disk["mirror_swap"]) == (False, [])
    assert on_disk["modality"] == "image" and on_disk["vision_tokens"] == 70


def test_the_image_script_is_run_with_the_image_spec(toolchain, split, tmp_path):
    run_embedding_tune(_request(split, tmp_path, vision_tokens=140))

    assert Path(toolchain.calls[1][1]).name == IMAGE_SCRIPT_NAME
    assert toolchain.scripts == [_EMBED_IMAGE_SCRIPT]
    (spec,) = toolchain.specs
    assert spec["model"] == EMBEDDING_MODEL and spec["revision"] == EMBEDDING_REVISION
    assert spec["model_type"] == EMBEDDING_MODEL_TYPE
    assert (spec["prefix"], spec["image_text"]) == ("", IMAGE_TOKEN_TEXT)
    assert spec["vision_tokens"] == 140
    assert (spec["mirror"], spec["mirror_swap"]) == (False, [])
    assert spec["labels"] == sorted(OPTIONS)
    assert spec["cache_dir"] == str(tmp_path / "tuned" / IMAGE_CACHE_DIR)
    assert spec["max_shard_size"] == IMAGE_MAX_SHARD_SIZE
    assert spec["model_dir"] == str(tmp_path / "tuned" / "model")
    recipe = spec["recipe"]
    assert (recipe["learning_rate"], recipe["head_learning_rate"]) == (2e-5, 1e-3)
    assert (recipe["weight_decay"], recipe["batch_size"], recipe["epochs"]) == (0.01, 16, 2)
    assert (recipe["seed"], recipe["embedding_dim"], recipe["logit_scale"]) == (0, 768, 20.0)
    assert (recipe["grad_clip"], recipe["dtype"], recipe["cache_dtype"]) == (
        1.0,
        "float32",
        "float16",
    )
    assert recipe["frozen"] == ["vision_tower", "embed_vision", "language_model.embed_tokens"]


def test_the_script_is_given_each_rows_file_digest_and_label(toolchain, split, tmp_path):
    run_embedding_tune(_request(split, tmp_path))

    (spec,) = toolchain.specs
    given = _records(Path(spec["train_rows"]))
    written = _records(split.train)
    assert len(given) == 12
    for row, source in zip(given, written, strict=True):
        assert row == {
            "image": str((split.dir / source["image"]).resolve()),
            "sha256": source["image_sha256"],
            "label": source["completion"],
        }
        assert hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() == row["sha256"]
    assert len(_records(Path(spec["heldout_rows"]))) == 6


def test_mirror_and_its_swaps_reach_the_script_and_the_record(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path, mirror=True, mirror_swap=("left:right",)))

    assert result.ok, result.checks.as_dict()
    (spec,) = toolchain.specs
    assert (spec["mirror"], spec["mirror_swap"]) == (True, [["left", "right"]])
    assert result.record is not None
    assert (result.record.mirror, result.record.mirror_swap) == (True, (("left", "right"),))
    texts = [t for t in result.limitations if t.startswith("--mirror")]
    assert len(texts) == 1
    assert "left and right" in texts[0] and "mirrored scene means what the original" in texts[0]


def test_mirror_without_swaps_notes_the_symmetry_it_assumes(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path, mirror=True))
    (text,) = [t for t in result.limitations if t.startswith("--mirror")]
    assert "exchanged" not in text


def test_no_mirror_no_mirror_limitation(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path))
    assert not [t for t in result.limitations if "--mirror" in t]


def test_an_image_run_says_its_recipe_has_no_litetune_measurement(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path))
    assert IMAGE_RECIPE_UNMEASURED in result.limitations
    assert "game frames" in IMAGE_RECIPE_UNMEASURED
    # The text recipe's CPU measurement is not this recipe's.
    assert not [t for t in result.limitations if "measured in float32 on the CPU only" in t]
    assert any("this run trained on cpu" in t for t in result.limitations)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_a_run_on_a_device_the_image_runs_used_makes_no_device_note(
    toolchain, split, tmp_path, device
):
    toolchain.probe_device = device
    if device == "cuda":
        toolchain.probe_fields = {"cuda_build": "12.8", "device_count": 1}
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.ok, result.checks.as_dict()
    assert toolchain.specs[0]["device"] == device
    assert not [t for t in result.limitations if "this run trained on" in t]
    assert IMAGE_RECIPE_UNMEASURED in result.limitations


def test_the_report_and_the_summary_name_the_images_and_the_vision_tokens(
    toolchain, split, tmp_path
):
    result = run_embedding_tune(_request(split, tmp_path, mirror=True))
    report = json.loads(write_report(result).read_text())
    assert report["modality"] == "image" and report["vision_tokens"] == 70
    assert report["request"]["modality"] == "image"
    assert report["request"]["recipe"] == IMAGE_RECIPE.as_dict()
    assert report["record"]["modality"] == "image"
    line = summarise(result)[1]
    assert "on images at 70 vision tokens per image, mirrored" in line
    assert "LiteRtEmbeddingBackend(visionTokensPerImage: 70)" in line
    assert DECIDER_PREFIX not in line


def test_the_parent_imports_no_training_stack(toolchain, split, tmp_path):
    run_embedding_tune(_request(split, tmp_path))
    # Not PIL: the prepare tests import the host's Pillow in this process.
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


# ---------------------------------------------------------------------------
# Endings
# ---------------------------------------------------------------------------


def test_exit_zero_without_model_safetensors_is_a_failure(toolchain, split, tmp_path):
    toolchain.write_safetensors = False
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert BASE_CHECKPOINT_FILE in _check(result, TRAINING_CHECK).detail
    assert result.record is None
    assert not (tmp_path / "tuned" / "model" / "litetune.json").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"vision_tokens": 140},
        {"mirror": True},
        {"mirror": True, "mirror_swap": [["left", "right"]]},
        {"prefix": DECIDER_PREFIX},
    ],
)
def test_metrics_that_say_another_input_than_given_are_a_failure(
    toolchain, split, tmp_path, overrides
):
    toolchain.metrics_overrides = overrides
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert result.record is None
    assert not (tmp_path / "tuned" / "model" / "litetune.json").exists()


def test_metrics_without_the_vision_tokens_are_incomplete(toolchain, split, tmp_path):
    toolchain.metrics_overrides = {"vision_tokens": None}
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert "vision_tokens" in _check(result, TRAINING_CHECK).detail


# ---------------------------------------------------------------------------
# Refusals, before the environment
# ---------------------------------------------------------------------------


def _refused(toolchain, request: EmbeddingTuneRequest, pattern: str) -> None:
    with pytest.raises(EmbeddingTuneError, match=pattern):
        run_embedding_tune(request)
    assert toolchain.calls == []


def test_an_image_split_without_vision_tokens_is_refused_naming_the_app_setting(
    toolchain, split, tmp_path
):
    _refused(
        toolchain,
        _request(split, tmp_path, vision_tokens=None),
        r"needs --vision-tokens \(70 or 140\).*LiteRtEmbeddingBackend\(visionTokensPerImage: N\)",
    )


def test_a_missing_image_is_refused_naming_the_row(toolchain, split, tmp_path):
    third = _records(split.train)[2]
    (split.dir / third["image"]).unlink()
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:3: the image .* does not exist")


def test_a_changed_image_is_refused_naming_the_row(toolchain, split, tmp_path):
    fifth = _records(split.heldout)[4]
    (split.dir / fifth["image"]).write_bytes(b"another picture")
    _refused(
        toolchain,
        _request(split, tmp_path),
        rf"heldout\.jsonl:5: the image .* hashes to [0-9a-f]{{64}}, not the "
        rf"{fifth['image_sha256']}",
    )


@pytest.mark.parametrize("value", ["../outside.png", "/etc/passwd", "images/../../x.png"])
def test_an_image_outside_the_split_is_refused(toolchain, split, tmp_path, value):
    rows = _records(split.train)
    rows[0]["image"] = value
    (tmp_path / "outside.png").write_bytes(b"x")
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:1: .*not inside the split")


@pytest.mark.parametrize("value", [None, "", "SHA", "0" * 63, "A" * 64])
def test_a_row_without_a_digest_is_refused(toolchain, split, tmp_path, value):
    rows = _records(split.train)
    rows[1]["image_sha256"] = value
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:2: 'image_sha256'")


@pytest.mark.parametrize("value", [None, "", 7])
def test_a_row_whose_image_is_not_a_path_is_refused(toolchain, split, tmp_path, value):
    rows = _records(split.train)
    rows[1]["image"] = value
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:2: 'image' must be")


def test_a_completion_that_is_not_an_option_is_refused(toolchain, split, tmp_path):
    rows = _records(split.train)
    rows[3]["completion"] = "reverse"
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:4: the completion 'reverse'")


def test_an_image_row_with_a_prompt_is_refused(toolchain, split, tmp_path):
    rows = _records(split.train)
    rows[0]["prompt"] = "a road"
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:1: an image row has no 'prompt'")


def test_a_split_mixing_image_and_text_rows_is_refused(toolchain, split, tmp_path):
    rows: list[dict[str, Any]] = _records(split.train)
    rows.append({"prompt": "a road", "completion": "left", "source_line": 99})
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"all image rows .* or all text rows")


def test_a_text_heldout_beside_an_image_training_split_is_refused(toolchain, split, tmp_path):
    decision = _decision()
    _write(
        split.heldout,
        [{"prompt": decision.render("a road"), "completion": "left", "source_line": 1}],
    )
    _refused(toolchain, _request(split, tmp_path), r"held-out split .* is text rows")


def test_an_option_with_no_training_image_is_refused(toolchain, tmp_path):
    s = _split(tmp_path)
    rows = [r for r in _records(s.train) if r["completion"] != "straight"]
    _write(s.train, rows)
    _refused(toolchain, _request(s, tmp_path), r"'straight'")


def test_an_empty_image_heldout_split_is_refused(toolchain, split, tmp_path):
    split.heldout.unlink()
    _refused(toolchain, _request(split, tmp_path), r"held-out split .* does not exist")


@pytest.mark.parametrize(
    ("swaps", "pattern"),
    [
        (("left:reverse",), r"left:reverse: not two of the decision's option ids"),
        (("left:left",), r"left:left: an option swapped with itself"),
        (("left:right", "right:straight"), r"right:straight: 'right' is already in another pair"),
        (("left:right", "right:left"), r"already in another pair"),
    ],
)
def test_a_swap_that_is_not_a_pair_of_option_ids_is_refused(
    toolchain, split, tmp_path, swaps, pattern
):
    _refused(toolchain, _request(split, tmp_path, mirror=True, mirror_swap=swaps), pattern)


def test_a_swap_is_split_at_the_one_colon_with_ids_on_both_sides():
    ids = ("lane:left", "lane:right", "a")
    assert resolve_swaps(["lane:left:lane:right"], ids) == (("lane:left", "lane:right"),)
    assert resolve_swaps([], ids) == ()
    with pytest.raises(EmbeddingTuneError, match="reads as 2 different pairs"):
        resolve_swaps(["a:b:c"], ("a", "b:c", "a:b", "c"))


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_a_swap_without_mirror_is_refused(split, tmp_path):
    with pytest.raises(EmbeddingTuneError, match="--mirror-swap without --mirror"):
        _request(split, tmp_path, mirror_swap=("left:right",))


def test_a_swap_without_a_colon_is_refused(split, tmp_path):
    with pytest.raises(EmbeddingTuneError, match="write it as A:B"):
        _request(split, tmp_path, mirror=True, mirror_swap=("left",))


@pytest.mark.parametrize("value", [0, 69, 100, 280, True])
def test_vision_tokens_other_than_70_or_140_are_refused(split, tmp_path, value):
    with pytest.raises(EmbeddingTuneError, match="70, 140"):
        _request(split, tmp_path, vision_tokens=value)


# ---------------------------------------------------------------------------
# A text split with image flags, and a text split unchanged
# ---------------------------------------------------------------------------


def _text_split(tmp_path: Path) -> Split:
    decision = _decision()
    root = tmp_path / "text-split"
    root.mkdir()
    (root / "decision.json").write_text(decision.canonical_text(), encoding="utf-8")
    rows = [
        {"prompt": decision.render(f"state {i}"), "completion": OPTIONS[i % 3], "source_line": i}
        for i in range(12)
    ]
    return Split(
        root,
        _write(root / "train.jsonl", rows),
        _write(root / "heldout.jsonl", rows[:6]),
        root / "decision.json",
    )


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"vision_tokens": 70}, "--vision-tokens"),
        ({"vision_tokens": None, "mirror": True}, "--mirror"),
        (
            {"vision_tokens": 140, "mirror": True, "mirror_swap": ("left:right",)},
            "--vision-tokens, --mirror, --mirror-swap",
        ),
    ],
)
def test_image_flags_on_a_text_split_are_refused_by_name(toolchain, tmp_path, kwargs, named):
    s = _text_split(tmp_path)
    _refused(toolchain, _request(s, tmp_path, **kwargs), rf"^{named}: .* is a text split")


def test_a_text_record_is_written_without_the_image_fields_it_never_had(tmp_path):
    record = EmbeddingRecord(
        base_model=EMBEDDING_MODEL,
        base_model_revision=EMBEDDING_REVISION,
        prompt_name=PROMPT_NAME,
        prefix=DECIDER_PREFIX,
        labels=("a", "b"),
        rows={"train": 4, "heldout": 2},
        steps_done=1,
        steps_planned=1,
        seconds=1.0,
        device="cpu",
        dtype="float32",
        head_heldout_accuracy=0.5,
        recipe=RECIPE.as_dict(),
        decision={},
        decision_sha256="sha256:" + "0" * 64,
    )
    data = record.as_dict()
    assert "mirror" not in data and "mirror_swap" not in data
    write_embedding_record(tmp_path, record)
    assert read_embedding_record(tmp_path) == record


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _image_record(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "direction": "embedding",
        "base_model": EMBEDDING_MODEL,
        "base_model_revision": EMBEDDING_REVISION,
        "prompt_name": "",
        "prefix": "",
        "labels": ["left", "right", "straight"],
        "rows": {"train": 10, "heldout": 4},
        "steps_done": 2,
        "steps_planned": 2,
        "seconds": 1.0,
        "device": "cpu",
        "dtype": "float32",
        "head_heldout_accuracy": 0.5,
        "recipe": IMAGE_RECIPE.as_dict(),
        "decision": {},
        "decision_sha256": "sha256:" + "0" * 64,
        "modality": "image",
        "vision_tokens": 70,
        "mirror": True,
        "mirror_swap": [["left", "right"]],
    }
    data.update(overrides)
    return data


def test_an_image_record_round_trips(tmp_path):
    record = EmbeddingRecord.from_dict(_image_record(), "x")
    assert record.mirror_swap == (("left", "right"),)
    write_embedding_record(tmp_path, record)
    assert read_embedding_record(tmp_path) == record
    assert json.loads((tmp_path / "litetune.json").read_text())["mirror_swap"] == [
        ["left", "right"]
    ]


def test_an_image_record_without_the_mirror_fields_reads_as_no_mirror():
    data = _image_record()
    del data["mirror"], data["mirror_swap"]
    record = EmbeddingRecord.from_dict(data, "x")
    assert (record.mirror, record.mirror_swap) == (False, ())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mirror": "yes"}, "'mirror' is 'yes'"),
        ({"mirror": 1}, "'mirror' is 1"),
        ({"mirror": False}, "'mirror' is false"),
        ({"mirror_swap": [["left", "reverse"]]}, "'reverse'"),
        ({"mirror_swap": [["left", "right"], ["right", "straight"]]}, "names a label twice"),
        ({"mirror_swap": [["left", "left"]]}, "names a label twice"),
        ({"mirror_swap": "left:right"}, "not a list of pairs"),
        ({"mirror_swap": [["left"]]}, "not a list of pairs"),
        ({"mirror_swap": [[1, "left"]]}, "names 1"),
        (
            {"modality": "text", "vision_tokens": None, "mirror_swap": []},
            "a text record has no mirror",
        ),
        (
            {"modality": "text", "vision_tokens": None, "mirror": False},
            "a text record has no mirror",
        ),
    ],
)
def test_a_record_whose_mirror_contradicts_itself_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingRecord.from_dict(_image_record(**overrides), "x")


def test_the_checkpoint_file_is_the_one_convert_reads():
    from litetune.embedding_export import BASE_FILE

    assert BASE_CHECKPOINT_FILE == BASE_FILE


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _argv(split: Split, tmp_path: Path, *extra: str, model: str = EMBEDDING_MODEL) -> list[str]:
    return [
        "tune",
        "--model",
        model,
        "--data",
        str(split.train),
        "--output-dir",
        str(tmp_path / "run"),
        "--decision",
        str(split.decision),
        *extra,
    ]


def test_a_whole_image_run_through_the_command_line(toolchain, split, tmp_path, capsys):
    argv = _argv(
        split,
        tmp_path,
        "--vision-tokens",
        "70",
        "--mirror",
        "--mirror-swap",
        "left:right",
    )
    assert main(argv) == 0
    record = read_embedding_record(tmp_path / "run" / "model")
    assert record is not None and record.is_image
    assert (record.vision_tokens, record.mirror, record.mirror_swap) == (
        70,
        True,
        (("left", "right"),),
    )
    report = json.loads((tmp_path / "run" / "tune.json").read_text())
    assert (report["modality"], report["vision_tokens"]) == ("image", 70)
    assert "on images at 70 vision tokens per image" in capsys.readouterr().out


def test_the_swap_flag_repeats(toolchain, tmp_path):
    s = _split(tmp_path, ids=("left", "right", "up", "down"))
    argv = _argv(
        s,
        tmp_path,
        "--vision-tokens",
        "140",
        "--mirror",
        "--mirror-swap",
        "left:right",
        "--mirror-swap",
        "up:down",
    )
    assert main(argv) == 0
    assert toolchain.specs[0]["mirror_swap"] == [["left", "right"], ["up", "down"]]


def test_an_image_split_without_vision_tokens_exits_could_not_check(
    toolchain, split, tmp_path, capsys
):
    assert main(_argv(split, tmp_path)) == 4
    assert "visionTokensPerImage" in capsys.readouterr().err
    assert toolchain.calls == []


def test_a_swap_without_mirror_exits_could_not_check(toolchain, split, tmp_path, capsys):
    argv = _argv(split, tmp_path, "--vision-tokens", "70", "--mirror-swap", "left:right")
    assert main(argv) == 4
    assert "--mirror-swap without --mirror" in capsys.readouterr().err


def test_vision_tokens_outside_the_choices_are_refused_by_argparse(split, tmp_path, capsys):
    assert main(_argv(split, tmp_path, "--vision-tokens", "100")) == 4
    assert "--vision-tokens: invalid choice: '100'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [["--vision-tokens", "70"], ["--mirror"], ["--mirror", "--mirror-swap", "a:b"]],
)
def test_image_flags_are_refused_for_an_llm(toolchain, split, tmp_path, capsys, extra):
    assert main(_argv(split, tmp_path, *extra, model="google/gemma-3-270m-it")) == 4
    err = capsys.readouterr().err
    assert extra[0] in err and "image split" in err
    assert toolchain.calls == []


# ---------------------------------------------------------------------------
# The script's helpers, run here
# ---------------------------------------------------------------------------


@pytest.fixture
def script() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "embed_image_train_script"}
    exec(compile(_EMBED_IMAGE_SCRIPT, IMAGE_SCRIPT_NAME, "exec"), namespace)
    return namespace


def test_the_swap_map_is_symmetric_and_refuses_a_repeated_label(script):
    swap_map = script["swap_map"]
    assert swap_map([]) == {}
    assert swap_map([["left", "right"]]) == {"left": "right", "right": "left"}
    assert swap_map([("a", "b"), ("c", "d")]) == {"a": "b", "b": "a", "c": "d", "d": "c"}
    for pairs in ([["a", "a"]], [["a", "b"], ["b", "c"]], [["a", "b"], ["c", "a"]]):
        with pytest.raises(ValueError, match="repeats a label"):
            swap_map(pairs)


def test_mirroring_adds_a_flipped_copy_of_every_row_with_its_label_swapped(script):
    rows = [
        {"sha256": "s1", "label": "left"},
        {"sha256": "s2", "label": "straight"},
        {"sha256": "s3", "label": "right"},
    ]
    swaps = script["swap_map"]([["left", "right"]])
    items = script["training_items"](rows, True, swaps)
    assert items == [
        ("s1", False, "left"),
        ("s2", False, "straight"),
        ("s3", False, "right"),
        ("s1", True, "right"),
        ("s2", True, "straight"),
        ("s3", True, "left"),
    ]
    assert script["training_items"](rows, False, swaps) == items[:3]
    # Mirrored, no swap: the copy keeps its label.
    assert script["training_items"](rows[:1], True, {}) == [
        ("s1", False, "left"),
        ("s1", True, "left"),
    ]


def test_class_weights_are_n_over_k_times_count(script):
    weights = script["class_weights"]
    # 6 items, 3 classes, counts 3/2/1: 6/9, 6/6, 6/3.
    assert weights([0, 0, 0, 1, 1, 2], 3) == pytest.approx([2 / 3, 1.0, 2.0])
    assert weights([0, 1], 2) == [1.0, 1.0]
    # zn3 recorded jump 3.745 / run 0.577, which is what the formula gives a 1:6.49 split.
    w = weights([0] * 100 + [1] * 649, 2)
    assert w[0] == pytest.approx(3.745, abs=1e-3) and w[1] == pytest.approx(0.577, abs=1e-3)
    with pytest.raises(ValueError, match="class 1 has no training item"):
        weights([0, 0, 2], 3)
    assert script["class_counts"]([2, 0, 2], 3) == [1, 0, 2]


def test_the_plan_counts_a_short_last_batch(script):
    plan = script["plan"]
    assert plan(32, 16, 2) == 4
    assert plan(33, 16, 2) == 6
    assert plan(20, 16, 1) == 2


def test_an_epochs_batches_follow_the_permutation(script):
    batches = script["epoch_batches"](5, 2, lambda n: list(range(n))[::-1])
    assert batches == [[4, 3], [2, 1], [0]]
    with pytest.raises(ValueError, match="every item once"):
        script["epoch_batches"](3, 2, lambda n: [0, 0, 1])


def test_the_cache_is_keyed_by_what_the_inputs_depend_on(script):
    key = script["cache_key"]
    fields = {"revision": "r", "vision_tokens": 70, "versions": {"torch": "2.9.1"}}
    assert key(fields) == key(dict(reversed(list(fields.items()))))
    assert len(key(fields)) == 16
    assert key(fields) != key({**fields, "vision_tokens": 140})
    assert key(fields) != key({**fields, "versions": {"torch": "2.9.2"}})
    name = script["cache_name"]
    assert name("ab", False) == "ab-original.npy"
    assert name("ab", True) == "ab-mirrored.npy"


def test_the_script_refuses_bytes_that_are_not_the_rows_picture(script):
    refusal = script["bytes_refusal"]
    data = b"picture"
    assert refusal(data, hashlib.sha256(data).hexdigest(), "p.png") is None
    found = refusal(b"other", hashlib.sha256(data).hexdigest(), "p.png")
    assert found is not None and "p.png" in found and hashlib.sha256(b"other").hexdigest() in found


def test_the_image_script_shares_the_text_scripts_small_helpers(script):
    refusal = script["model_type_refusal"]
    assert refusal({"model_type": "embedding_gemma2"}, "embedding_gemma2") is None
    assert "gemma3_text" in script["model_type_refusal"]({"model_type": "gemma3_text"}, "x")
    assert script["accuracy"](["a", "b", "a"], ["a", "b", "b"]) == 0.6667

    class NoCuda:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    assert script["training_device"](NoCuda, "mps") == "mps"
    assert script["training_device"](NoCuda, None) == "cpu"


def test_the_image_script_trains_what_the_recipe_says():
    """What `main` does that a helper cannot show: read in its text."""
    s = _EMBED_IMAGE_SCRIPT
    assert "model_config.audio_config = None" in s
    assert "for module in (model.vision_tower, model.embed_vision, lm.embed_tokens):" in s
    assert "module.requires_grad_(False)" in s
    assert 'images_kwargs={"max_soft_tokens": spec["vision_tokens"]}' in s
    assert 'logits = head(vectors) * recipe["logit_scale"]' in s
    assert 'weight=weight_tensor, reduction="sum"' in s and ") / len(chosen)" in s
    assert 'torch.nn.utils.clip_grad_norm_(lm_params, recipe["grad_clip"])' in s
    assert 'weight_decay=recipe["weight_decay"]' in s
    assert 'model.save_pretrained(model_dir, max_shard_size=spec["max_shard_size"])' in s
    assert "processor.save_pretrained(model_dir)" in s
    assert "picture = ImageOps.mirror(picture)" in s
    assert "array = values.astype(np.float16)" in s
    # The text script is not the image one.
    assert _EMBED_TRAIN_SCRIPT != s
