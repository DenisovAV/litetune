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

import dataclasses
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
    PROCESSOR_FILES,
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


def _processor_text(name: str) -> str:
    """What the fake run saves as processor file `name`: different for each file."""
    return json.dumps({"file": name})


@dataclass
class FakeImageToolchain:
    """Stands in for `envs.StageEnv.run`: the device probe, then the image script."""

    probe_device: str | None = "cpu"
    probe_fields: dict[str, Any] = field(default_factory=dict)
    returncode: int = 0
    stderr: str = ""
    write_safetensors: bool = True
    # Processor files the fake run does not save.
    skip_files: tuple[str, ...] = ()
    # The soft tokens the fake processor makes of every picture.
    soft_tokens: int = 63
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
        for name in PROCESSOR_FILES:
            if name not in self.skip_files:
                (model_dir / name).write_text(_processor_text(name), encoding="utf-8")
        train = Path(spec["train_rows"]).read_text(encoding="utf-8").splitlines()
        heldout = Path(spec["heldout_rows"]).read_text(encoding="utf-8").splitlines()
        pictures = sorted({json.loads(line)["sha256"] for line in train + heldout})
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
            "soft_tokens_per_image": {sha: self.soft_tokens for sha in pictures},
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
    pictures = {r["image_sha256"] for r in _records(split.train) + _records(split.heldout)}
    assert on_disk["soft_tokens"] == dict.fromkeys(sorted(pictures), 63)
    assert record.soft_tokens == on_disk["soft_tokens"]


def test_the_record_holds_the_sha256_of_each_processor_file_saved(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.PASSED, result.checks
    on_disk = json.loads((tmp_path / "tuned" / "model" / "litetune.json").read_text())
    expected = {
        name: hashlib.sha256(_processor_text(name).encode()).hexdigest() for name in PROCESSOR_FILES
    }
    assert on_disk["processor_sha256"] == expected
    assert result.record is not None and result.record.processor_sha256 == expected


@pytest.mark.parametrize("missing", [BASE_CHECKPOINT_FILE, *PROCESSOR_FILES])
def test_a_checkpoint_without_its_weights_or_processor_is_a_failure(
    toolchain, split, tmp_path, missing
):
    if missing == BASE_CHECKPOINT_FILE:
        toolchain.write_safetensors = False
    else:
        toolchain.skip_files = (missing,)
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    detail = _check(result, TRAINING_CHECK).detail
    assert f"no {missing}" in detail
    assert result.record is None


@pytest.mark.parametrize(
    ("soft", "said"),
    [
        ({}, "'soft_tokens' is {}"),
        ({"a" * 64: 63}, "soft tokens for 1 pictures, which are not the 18 pictures"),
        (None, "'soft_tokens' is None"),
    ],
)
def test_soft_tokens_that_do_not_cover_the_pictures_are_a_failure(
    toolchain, split, tmp_path, soft, said
):
    toolchain.metrics_overrides = {"soft_tokens_per_image": soft}
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert said in _check(result, TRAINING_CHECK).detail
    assert result.record is None


@pytest.mark.parametrize("count", [0, 71, True])
def test_a_soft_token_count_outside_the_budget_is_a_failure(toolchain, split, tmp_path, count):
    toolchain.soft_tokens = count
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert "not a count of 1 to the budget of 70" in _check(result, TRAINING_CHECK).detail


def test_tune_trains_on_the_split_prepare_wrote(toolchain, tmp_path):
    from test_embedding_prepare_images import FakeImageInspector, _fake_image

    from litetune.prepare import PrepareRequest, prepare

    rows = []
    for option in OPTIONS:
        for i in range(12):
            _fake_image(tmp_path / "raw" / f"{option}-{i}.png", width=8 + i, height=4)
            rows.append({"image": f"raw/{option}-{i}.png", "label": option})
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    questions = tmp_path / "questions.json"
    questions.write_text(_decision().canonical_text(), encoding="utf-8")
    prepared = prepare(
        PrepareRequest(
            data=data,
            output_dir=tmp_path / "split",
            context_length=None,
            base_model=EMBEDDING_MODEL,
            questions=questions,
            images=FakeImageInspector(),
        )
    )
    assert prepared.train is not None and prepared.heldout is not None
    split = Split(
        tmp_path / "split",
        prepared.train.path,
        prepared.heldout.path,
        tmp_path / "split" / "decision.json",
    )
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.ok, result.checks.as_dict()
    (spec,) = toolchain.specs
    given = _records(Path(spec["train_rows"]))
    assert len(given) == prepared.train.n
    for row in given:
        assert Path(row["image"]).parent == (tmp_path / "split" / "images").resolve()
        assert hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() == row["sha256"]


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


def test_an_image_run_says_its_recipe_has_one_litetune_measurement(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path))
    assert IMAGE_RECIPE_UNMEASURED in result.limitations
    assert "game frames" in IMAGE_RECIPE_UNMEASURED
    # MEASUREMENTS.md, "EmbeddingGemma 2 on images: beans": centroid_k10 of the
    # reference and of Google's file in verify_int8.json.
    for said in ("one litetune measurement", "beans", "0.9287", "0.8233", "is not known"):
        assert said in IMAGE_RECIPE_UNMEASURED
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
    assert "on images at most 70 vision tokens per image, mirrored" in line
    assert "same budget (LiteRtEmbeddingBackend(visionTokensPerImage: 70); 140 when" in line
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


def test_metrics_that_drop_the_swap_of_a_mirrored_run_are_a_failure_naming_it(
    toolchain, split, tmp_path
):
    toolchain.metrics_overrides = {"mirror_swap": []}
    result = run_embedding_tune(_request(split, tmp_path, mirror=True, mirror_swap=("left:right",)))
    assert result.outcome is Outcome.FAILED
    check = _check(result, TRAINING_CHECK)
    assert check.outcome is Outcome.FAILED
    assert check.detail == (
        "the training script reports a budget of 70 vision tokens, mirrored items True, swaps "
        "[], not the 70, True, [['left', 'right']] of the spec it was given"
    )
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


def test_an_image_path_inside_the_split_given_absolute_is_refused(toolchain, split, tmp_path):
    rows = _records(split.train)
    rows[0]["image"] = str((split.dir / rows[0]["image"]).resolve())
    _write(split.train, rows)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:1: .*not inside the split")


@pytest.mark.posix
def test_an_images_directory_linked_out_of_the_split_is_refused(toolchain, split, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    (split.dir / "images").rename(elsewhere)
    (split.dir / "images").symlink_to(elsewhere, target_is_directory=True)
    _refused(toolchain, _request(split, tmp_path), r"train\.jsonl:1: .*not inside the split")


def test_a_row_with_a_prompt_and_an_image_is_refused_with_prepares_reason(
    toolchain, split, tmp_path
):
    rows = _records(split.train)
    rows[3]["prompt"] = "a road"
    _write(split.train, rows)
    _refused(
        toolchain,
        _request(split, tmp_path),
        r"train\.jsonl:4: this row gives both an 'image' and a 'prompt'\. A row for "
        r"google/embeddinggemma-2 is an image row",
    )


def test_a_text_row_in_an_image_split_is_refused(toolchain, split, tmp_path):
    rows = _records(split.train)
    rows[3] = {"prompt": "a road", "completion": rows[3]["completion"], "source_line": 4}
    _write(split.train, rows)
    _refused(
        toolchain,
        _request(split, tmp_path),
        r"train\.jsonl:4: this is a text row, and line 1 is an image row",
    )


def test_the_reader_refuses_a_text_row_itself(split):
    from litetune.embedding_tune import read_image_split

    rows = _records(split.heldout)
    rows[2]["prompt"] = "a road"
    _write(split.heldout, rows)
    with pytest.raises(EmbeddingTuneError, match=r"heldout\.jsonl:3: a text row in the held-out"):
        read_image_split(split.heldout, _decision(), "held-out")


def test_a_text_split_whose_rows_also_name_an_image_is_refused(tmp_path):
    """`prepare` refuses such a row for EmbeddingGemma 2, and so does the split reader:
    read as text, its picture would be dropped without a word."""
    from litetune.embedding_tune import split_modality

    decision = _decision()
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(
            json.dumps({"prompt": decision.render(f"s{i}"), "completion": o, "image": "x.png"})
            + "\n"
            for i, o in enumerate(OPTIONS * 4)
        ),
        encoding="utf-8",
    )
    with pytest.raises(EmbeddingTuneError, match=r"train\.jsonl:1: this row gives both"):
        split_modality(path, "training")


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
    assert "mirror" not in data and "mirror_swap" not in data and "soft_tokens" not in data
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
        "soft_tokens": {"a" * 64: 63, "b" * 64: 54},
        "processor_sha256": dict.fromkeys(PROCESSOR_FILES, "d" * 64),
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("processor", "message"),
    [
        (None, "'processor_sha256' is None; an image record gives the sha256 of each of"),
        ({}, "'processor_sha256' is {}"),
        (
            dict.fromkeys(PROCESSOR_FILES[1:], "d" * 64),
            "an image record gives the sha256 of each of chat_template.jinja",
        ),
        (
            dict.fromkeys((*PROCESSOR_FILES, "extra.json"), "d" * 64),
            "an image record gives the sha256 of each of",
        ),
        (
            dict.fromkeys(PROCESSOR_FILES, "d" * 64) | {"tokenizer.json": "D" * 64},
            "gives tokenizer.json 'DDDD",
        ),
        (
            dict.fromkeys(PROCESSOR_FILES, "d" * 64) | {"tokenizer.json": 7},
            "gives tokenizer.json 7, which is not a sha256",
        ),
        ([["tokenizer.json", "d" * 64]], "'processor_sha256' is \\[\\["),
    ],
)
def test_an_image_record_without_the_processor_digests_is_refused(processor, message):
    data = _image_record(processor_sha256=processor)
    if processor is None:
        del data["processor_sha256"]
    with pytest.raises(ValueError, match=message):
        EmbeddingRecord.from_dict(data, "x")
    valid = EmbeddingRecord.from_dict(_image_record(), "x")
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(valid, processor_sha256=processor)


def test_a_text_record_with_processor_digests_is_refused():
    data = _image_record(
        modality="text", vision_tokens=None, mirror=False, mirror_swap=[], soft_tokens=None
    )
    with pytest.raises(ValueError, match="a text record has no processor files"):
        EmbeddingRecord.from_dict(data, "x")
    del data["processor_sha256"]
    record = EmbeddingRecord.from_dict(data, "x")
    assert record.processor_sha256 is None
    assert "processor_sha256" not in record.as_dict()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"soft_tokens": None}, "'soft_tokens' is None; an image record gives them per picture"),
        ({"soft_tokens": {}}, "'soft_tokens' is {}"),
        ({"soft_tokens": {"x": 3}}, "names 'x', which is not a picture's sha256"),
        ({"soft_tokens": {"a" * 64: 71}}, "71 soft tokens, not a count of 1 to the budget of 70"),
        ({"soft_tokens": {"a" * 64: 0}}, "0 soft tokens"),
        ({"soft_tokens": {"a" * 64: "63"}}, "'63' soft tokens"),
    ],
)
def test_a_record_whose_soft_tokens_are_not_counts_within_the_budget_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingRecord.from_dict(_image_record(**overrides), "x")


def test_a_text_record_with_soft_tokens_is_refused():
    data = _image_record(modality="text", vision_tokens=None, mirror=False, mirror_swap=[])
    with pytest.raises(ValueError, match="a text record has no soft tokens"):
        EmbeddingRecord.from_dict(data, "x")


def test_at_140_a_count_up_to_140_is_within_the_budget():
    record = EmbeddingRecord.from_dict(
        _image_record(vision_tokens=140, soft_tokens={"a" * 64: 130, "b" * 64: 121}), "x"
    )
    assert record.soft_tokens == {"a" * 64: 130, "b" * 64: 121}


def test_an_image_record_round_trips(tmp_path):
    record = EmbeddingRecord.from_dict(_image_record(), "x")
    assert record.mirror_swap == (("left", "right"),)
    write_embedding_record(tmp_path, record)
    assert read_embedding_record(tmp_path) == record
    on_disk = json.loads((tmp_path / "litetune.json").read_text())
    assert on_disk["mirror_swap"] == [["left", "right"]]
    assert on_disk["soft_tokens"] == {"a" * 64: 63, "b" * 64: 54}
    assert on_disk["processor_sha256"] == dict.fromkeys(PROCESSOR_FILES, "d" * 64)


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
    assert "on images at most 70 vision tokens per image" in capsys.readouterr().out


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
    # An unpublished game-frame run recorded 3.745 / 0.577 for a 1:6.49 split.
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


def _spec_for_cache(**changed: Any) -> dict[str, Any]:
    spec = {
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "vision_tokens": 70,
        "image_text": IMAGE_TOKEN_TEXT,
        "recipe": IMAGE_RECIPE.as_dict(),
        "litetune_version": "0.7.3",
        "train_rows": "ignored",
    }
    spec.update(changed)
    return spec


def test_the_cache_is_keyed_by_what_the_inputs_depend_on(script):
    fields, key = script["cache_fields"], script["cache_key"]
    versions = {"torch": "2.9.1", "transformers": "5.19.0"}
    base = fields(_spec_for_cache(), "cpu", versions, "s" * 64)
    assert base == {
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "vision_tokens": 70,
        "image_text": IMAGE_TOKEN_TEXT,
        "cache_dtype": "float16",
        "device": "cpu",
        "versions": versions,
        "litetune": "0.7.3",
        "script_sha256": "s" * 64,
    }
    assert key(base) == key(dict(reversed(list(base.items()))))
    assert len(key(base)) == 16
    # A rerun at 140 does not reuse the inputs computed at 70; nor does any other change.
    others = [
        fields(_spec_for_cache(vision_tokens=140), "cpu", versions, "s" * 64),
        fields(_spec_for_cache(revision="other"), "cpu", versions, "s" * 64),
        fields(_spec_for_cache(litetune_version="0.7.4"), "cpu", versions, "s" * 64),
        fields(
            _spec_for_cache(recipe=IMAGE_RECIPE.as_dict() | {"cache_dtype": "float32"}),
            "cpu",
            versions,
            "s" * 64,
        ),
        fields(_spec_for_cache(), "mps", versions, "s" * 64),
        fields(_spec_for_cache(), "cpu", {**versions, "torch": "2.9.2"}, "s" * 64),
        fields(_spec_for_cache(), "cpu", versions, "t" * 64),
    ]
    assert len({key(base), *(key(f) for f in others)}) == 1 + len(others)
    # The rows are not what the inputs depend on.
    assert key(fields(_spec_for_cache(train_rows="x"), "cpu", versions, "s" * 64)) == key(base)
    name = script["cache_name"]
    assert name("ab", False) == "ab-original.npy"
    assert name("ab", True) == "ab-mirrored.npy"


def test_main_keys_its_cache_with_cache_fields():
    s = _EMBED_IMAGE_SCRIPT
    assert "key_fields = cache_fields(spec, device, versions, script_sha256)" in s
    assert 'cache = Path(spec["cache_dir"]) / cache_key(key_fields)' in s
    assert 'np, cache, run["inputs"], hidden, dtype, compute, confirm\n' in s
    assert "def confirm(sha):\n        confirm_picture(paths[sha], sha)\n" in s


def test_inputs_come_from_the_cache_when_usable_and_are_computed_and_stored_otherwise(
    script, tmp_path
):
    np = pytest.importorskip("numpy")
    asked: list[tuple[str, bool]] = []
    confirmed: list[str] = []

    def compute(sha, flipped):
        asked.append((sha, flipped))
        return np.full((4, 3), 2.0 if flipped else 1.0, np.float16), 3

    wanted = [("aa", False), ("aa", True), ("bb", False)]
    load = script["load_inputs"]
    tokens, soft, counts = load(np, tmp_path, wanted, 3, "float16", compute, confirmed.append)
    assert asked == wanted
    # A computed input hashes its own bytes (`compute`); nothing else is confirmed.
    assert confirmed == []
    assert counts == {"computed": 3, "reused": 0, "recomputed": []}
    assert soft == {("aa", False): 3, ("aa", True): 3, ("bb", False): 3}
    assert float(tokens[("aa", True)][0, 0]) == 2.0
    # A second run reads every input back, with its soft tokens, and computes none,
    # each picture's file confirmed once whatever number of inputs it has.
    asked.clear()
    again, soft_again, counts = load(np, tmp_path, wanted, 3, "float16", compute, confirmed.append)
    assert asked == [] and counts == {"computed": 0, "reused": 3, "recomputed": []}
    assert confirmed == ["aa", "bb"]
    assert soft_again == soft
    assert all((again[k] == tokens[k]).all() for k in wanted)
    # An entry that cannot be used is computed again, and named.
    (tmp_path / "bb-original.npy").write_bytes(b"broken")
    confirmed.clear()
    _, _, counts = load(np, tmp_path, wanted, 3, "float16", compute, confirmed.append)
    assert asked == [("bb", False)]
    assert confirmed == ["aa"]
    assert counts["reused"] == 2 and counts["computed"] == 1
    assert counts["recomputed"][0].startswith("bb-original.npy: unreadable")


def test_a_refusal_while_computing_reaches_main_as_a_refusal(script, tmp_path):
    def compute(sha, flipped):
        raise script["Refusal"]("the split changed")

    with pytest.raises(script["Refusal"], match="the split changed"):
        script["load_inputs"](None, tmp_path, [("aa", False)], 3, "float16", compute, None)
    assert "except Refusal as exc:\n        return refuse(out, str(exc))" in _EMBED_IMAGE_SCRIPT


def test_a_cached_input_whose_picture_changed_is_refused_not_used(script, tmp_path):
    np = pytest.importorskip("numpy")
    picture = tmp_path / "picture.png"
    picture.write_bytes(b"the picture")
    sha = hashlib.sha256(b"the picture").hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    script["store_input"](np, cache, sha, False, np.ones((4, 3), np.float16), 3)

    def compute(sha, flipped):
        raise AssertionError("a cached input is not computed")

    def confirm(digest):
        script["confirm_picture"](picture, digest)

    tokens, _, counts = script["load_inputs"](
        np, cache, [(sha, False)], 3, "float16", compute, confirm
    )
    assert counts["reused"] == 1 and (sha, False) in tokens
    picture.write_bytes(b"another picture")
    with pytest.raises(script["Refusal"]) as refused:
        script["load_inputs"](np, cache, [(sha, False)], 3, "float16", compute, confirm)
    said = str(refused.value)
    assert str(picture) in said and hashlib.sha256(b"another picture").hexdigest() in said
    assert f"not the {sha} its row names" in said


def test_a_cached_input_reads_back_with_its_soft_tokens(script, tmp_path):
    np = pytest.importorskip("numpy")
    array = np.arange(12, dtype=np.float16).reshape(4, 3)
    assert script["cached_input"](np, tmp_path, "ab", False, 3, "float16") is None
    script["store_input"](np, tmp_path, "ab", False, array, 2)
    got, soft = script["cached_input"](np, tmp_path, "ab", False, 3, "float16")
    assert soft == 2 and got.dtype == np.float16 and (got == array).all()
    # The mirrored copy is another entry.
    assert script["cached_input"](np, tmp_path, "ab", True, 3, "float16") is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ab-original.json", "ab-original.npy"]


@pytest.mark.parametrize(
    ("damage", "said"),
    [
        ("hidden", "shape (4, 3), not (4, 5)"),
        ("dtype", "dtype float16, not float32"),
        ("nan", "not finite"),
        ("meta", "gives no soft-token and token counts"),
        ("tokens", "shape (4, 3), not (9, 3)"),
        ("soft", "4 soft tokens of 4"),
        ("garbage", "unreadable"),
    ],
)
def test_a_cached_input_that_cannot_be_used_says_why(script, tmp_path, damage, said):
    np = pytest.importorskip("numpy")
    array = np.ones((4, 3), np.float16)
    if damage == "nan":
        array[1, 1] = np.nan
    script["store_input"](np, tmp_path, "ab", False, array, 4 if damage == "soft" else 2)
    meta = tmp_path / "ab-original.json"
    if damage == "meta":
        meta.write_text(json.dumps({"soft_tokens": "2", "tokens": 4}), encoding="utf-8")
    if damage == "tokens":
        meta.write_text(json.dumps({"soft_tokens": 2, "tokens": 9}), encoding="utf-8")
    if damage == "garbage":
        (tmp_path / "ab-original.npy").write_bytes(b"not numpy")
    hidden = 5 if damage == "hidden" else 3
    dtype = "float32" if damage == "dtype" else "float16"
    found = script["cached_input"](np, tmp_path, "ab", False, hidden, dtype)
    assert isinstance(found, str) and said in found


def test_an_input_without_its_record_is_not_cached(script, tmp_path):
    np = pytest.importorskip("numpy")
    script["store_input"](np, tmp_path, "ab", False, np.ones((4, 3), np.float16), 2)
    (tmp_path / "ab-original.json").unlink()
    assert script["cached_input"](np, tmp_path, "ab", False, 3, "float16") is None


def test_the_run_items_mirror_training_only_and_weigh_the_copies(script):
    rows = [
        {"sha256": "s1", "label": "left", "image": "/p/1.png"},
        {"sha256": "s2", "label": "left", "image": "/p/2.png"},
        {"sha256": "s3", "label": "straight", "image": "/p/3.png"},
    ]
    heldout = [{"sha256": "h1", "label": "right", "image": "/p/h1.png"}]
    labels = ["left", "right", "straight"]
    run = script["run_items"](rows, heldout, labels, True, [["left", "right"]])
    assert run["heldout_items"] == [("h1", False, "right")]
    assert run["originals"] == [
        ("s1", False, "left"),
        ("s2", False, "left"),
        ("s3", False, "straight"),
    ]
    assert run["items"][3:] == [
        ("s1", True, "right"),
        ("s2", True, "right"),
        ("s3", True, "straight"),
    ]
    # Six items, counts left 2 / right 2 / straight 2: every weight 1.
    assert run["targets"] == [0, 0, 2, 1, 1, 2]
    assert run["weights"] == [1.0, 1.0, 1.0]
    assert run["paths"] == {
        "s1": "/p/1.png",
        "s2": "/p/2.png",
        "s3": "/p/3.png",
        "h1": "/p/h1.png",
    }
    assert run["inputs"] == [
        ("h1", False),
        ("s1", False),
        ("s1", True),
        ("s2", False),
        ("s2", True),
        ("s3", False),
        ("s3", True),
    ]
    plain = script["run_items"](rows, heldout, ["left", "straight"], False, [])
    assert plain["items"] == plain["originals"] == run["originals"]
    assert plain["weights"] == [0.75, 1.5]


def test_without_mirroring_a_class_with_no_training_item_is_refused(script):
    rows = [{"sha256": "s1", "label": "left", "image": "/p/1.png"}]
    with pytest.raises(ValueError, match="class 1 has no training item"):
        script["run_items"](rows, [], ["left", "right"], False, [])
    # The swapped copy is a training item of the other class.
    run = script["run_items"](rows, [], ["left", "right"], True, [["left", "right"]])
    assert run["weights"] == [1.0, 1.0]


def test_the_pool_is_the_mean_over_the_real_tokens_of_each_padded_row(script):
    np = pytest.importorskip("numpy")
    short = np.array([[1.0, 0.0], [3.0, 0.0]], np.float16)
    long = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 6.0]], np.float16)
    embeds, mask = script["padded"](np, [short, long])
    assert embeds.dtype == np.float32 and embeds.shape == (2, 3, 2)
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    # A padded position carries a large value: only the mask keeps it out of the mean.
    embeds[0, 2] = 1000.0
    pooled = script["masked_mean"](embeds, mask)
    np.testing.assert_allclose(pooled, [[2.0, 0.0], [0.0, 3.0]])


def test_a_flipped_picture_is_the_mirror_image_and_the_original_is_not(script):
    image_module = pytest.importorskip("PIL.Image")
    ops = pytest.importorskip("PIL.ImageOps")
    import io

    picture = image_module.new("RGB", (3, 1))
    picture.putpixel((0, 0), (255, 0, 0))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    data = buffer.getvalue()
    original = script["picture_of"](image_module, ops, data, False)
    flipped = script["picture_of"](image_module, ops, data, True)
    assert original.getpixel((0, 0)) == (255, 0, 0) and original.getpixel((2, 0)) == (0, 0, 0)
    assert flipped.getpixel((2, 0)) == (255, 0, 0) and flipped.getpixel((0, 0)) == (0, 0, 0)
    assert original.mode == flipped.mode == "RGB"


def test_strict_pillow_makes_a_decompression_bomb_an_error(script):
    import warnings

    image_module = pytest.importorskip("PIL.Image")
    with warnings.catch_warnings():
        script["strict_pillow"](image_module)
        with pytest.raises(image_module.DecompressionBombWarning):
            warnings.warn("bomb", image_module.DecompressionBombWarning, stacklevel=1)


def test_both_scripts_embed_a_picture_with_the_one_shared_source():
    from litetune.embedding_tune import IMAGE_INPUT_SOURCE
    from litetune.embedding_verify import IMAGE_REFERENCE_SCRIPT

    assert IMAGE_INPUT_SOURCE in _EMBED_IMAGE_SCRIPT
    assert IMAGE_INPUT_SOURCE in IMAGE_REFERENCE_SCRIPT
    for source in (_EMBED_IMAGE_SCRIPT, IMAGE_REFERENCE_SCRIPT):
        assert source.count("def image_input(") == 1 and source.count("def masked_mean(") == 1


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
    assert 'images_kwargs={"max_soft_tokens": vision_tokens}' in s
    assert 'spec["image_text"], spec["vision_tokens"]' in s
    assert 'run = run_items(rows, heldout, labels, spec["mirror"], spec["mirror_swap"])' in s
    assert "masked_mean(state, mask)" in s
    assert 'logits = head(vectors) * recipe["logit_scale"]' in s
    assert 'weight=weight_tensor, reduction="sum"' in s and ") / len(chosen)" in s
    assert 'torch.nn.utils.clip_grad_norm_(lm_params, recipe["grad_clip"])' in s
    assert 'weight_decay=recipe["weight_decay"]' in s
    assert 'model.save_pretrained(model_dir, max_shard_size=spec["max_shard_size"])' in s
    assert "processor.save_pretrained(model_dir)" in s
    assert "picture = picture_of(Image, ImageOps, data, flipped)" in s
    assert "array = embeds[0].float().cpu().numpy().astype(dtype)" in s
    # The text script is not the image one.
    assert _EMBED_TRAIN_SCRIPT != s
