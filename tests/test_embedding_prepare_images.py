"""`prepare --base-model google/embeddinggemma-2` on image rows: checked, copied, never re-encoded.

Two checkers stand in for the real one. `FakeImageInspector` reads this file's
own fake images -- a line `FAKE <format> <width> <height> <frames>` -- and the
command-line test answers `IMAGE_CHECK_SCRIPT` from a fake `StageEnv.run` the
same way. The script itself is exec'd in this process against real images
Pillow writes, when Pillow is importable here (a dev dependency).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned

from litetune import envs
from litetune.checks import Outcome
from litetune.cli import main
from litetune.embedding_tune import EMBEDDING_MODEL
from litetune.images import (
    IMAGE_CHECK_SCRIPT,
    ImageCheckUnavailable,
    ImageFacts,
    PillowImageInspector,
    facts_from,
)
from litetune.prepare import LENGTH_CHECK, PrepareError, PrepareRequest, prepare

OPTIONS = ("cat", "dog", "bird")


def _fake_facts(data: bytes) -> ImageFacts:
    sha = hashlib.sha256(data).hexdigest()
    parts = data.split()
    if len(parts) < 5 or parts[0] != b"FAKE":
        return ImageFacts(sha, None, None, None, None, "UnidentifiedImageError: cannot identify")
    fmt, width, height, frames = parts[1].decode(), int(parts[2]), int(parts[3]), int(parts[4])
    return ImageFacts(sha, fmt, width, height, frames, None)


@dataclass
class FakeImageInspector:
    """Reads `FAKE <format> <w> <h> <frames>` files; no Pillow, no subprocess."""

    calls: list[list[Path]] = field(default_factory=list)
    # Answers a digest the bytes do not have for the file of this name.
    misread: str | None = None

    def describe(self) -> dict[str, Any]:
        return {"checker": "fake", "pillow": "0.0-fake"}

    def inspect(self, paths: Sequence[Path]) -> list[ImageFacts]:
        self.calls.append(list(paths))
        found = [_fake_facts(p.read_bytes()) for p in paths]
        return [
            ImageFacts("a" * 64, f.format, f.width, f.height, f.frames, f.error)
            if path.name == self.misread
            else f
            for path, f in zip(paths, found, strict=True)
        ]


def _fake_image(
    path: Path, fmt: str = "PNG", width: int = 4, height: int = 3, frames: int = 1, tag: str = ""
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"FAKE {fmt} {width} {height} {frames} {tag or path.name}\n".encode())
    return path


def _questions(tmp_path: Path, name: str = "questions.json", **extra: Any) -> Path:
    path = tmp_path / name
    body = {
        "type": "choice",
        "question": "What animal is in the picture?",
        "options": [{"id": o} for o in OPTIONS],
        **extra,
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _rows_file(tmp_path: Path, rows: list[dict[str, Any]], name: str = "rows.jsonl") -> Path:
    path = tmp_path / name
    # Bytes, not text: the split is seeded by them, and `\r\n` would move it.
    path.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))
    return path


def _image_rows(tmp_path: Path, per_option: int = 12) -> list[dict[str, Any]]:
    rows = []
    for option in OPTIONS:
        for i in range(per_option):
            _fake_image(tmp_path / "pics" / f"{option}-{i}.png", width=10 + i, height=20 + i)
            rows.append({"image": f"pics/{option}-{i}.png", "label": option})
    return rows


def _request(tmp_path: Path, **kwargs: Any) -> PrepareRequest:
    # The default rows only when none are given: writing them rewrites the images.
    if "data" not in kwargs:
        kwargs["data"] = _rows_file(tmp_path, _image_rows(tmp_path))
    # No window: image rows refuse one. A text split passes its own.
    params: dict[str, Any] = {
        "output_dir": tmp_path / "split",
        "context_length": None,
        "base_model": EMBEDDING_MODEL,
        "questions": _questions(tmp_path),
        "images": FakeImageInspector(),
    }
    params.update(kwargs)
    return PrepareRequest(**params)


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_image_rows_are_split_with_their_images_copied_byte_for_byte(tmp_path):
    result = prepare(_request(tmp_path))
    assert result.outcome is not Outcome.FAILED
    assert result.train is not None and result.heldout is not None
    assert result.calibration is not None
    split = tmp_path / "split"
    records = [
        *_records(split / "train.jsonl"),
        *_records(split / "calibration.jsonl"),
        *_records(split / "heldout.jsonl"),
    ]
    assert len(records) == 36
    raw = _image_rows(tmp_path)
    for record in records:
        assert set(record) == {"image", "image_sha256", "completion", "target", "source_line"}
        given = raw[record["source_line"] - 1]
        source = (tmp_path / given["image"]).read_bytes()
        assert record["image_sha256"] == hashlib.sha256(source).hexdigest()
        assert record["image"] == f"images/{record['image_sha256']}.png"
        assert (split / record["image"]).read_bytes() == source
        assert record["completion"] == record["target"] == given["label"]
    assert sorted(p.name for p in (split / "images").iterdir()) == sorted(
        f"{r['image_sha256']}.png" for r in records
    )
    # Nothing left beside the split from the build.
    assert sorted(p.name for p in split.iterdir()) == [
        "calibration.jsonl",
        "decision.json",
        "heldout.jsonl",
        "images",
        "prepare.json",
        "train.jsonl",
    ]


def test_the_report_says_image_and_what_the_images_are(tmp_path):
    rows = _image_rows(tmp_path)
    _fake_image(tmp_path / "pics" / "big.jpg", fmt="JPEG", width=640, height=480)
    rows.append({"image": "pics/big.jpg", "label": "cat"})
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "given.jsonl")))
    report = json.loads((tmp_path / "split" / "prepare.json").read_text(encoding="utf-8"))
    assert report["modality"] == "image"
    images = report["images"]
    assert images["rows"] == 37 and images["files"] == 37
    assert images["formats"] == {"JPEG": 1, "PNG": 36}
    assert images["width"] == {"min": 10, "max": 640}
    assert images["height"] == {"min": 20, "max": 480}
    assert images["duplicate_rows"] == 0
    assert images["checker"]["checker"] == "fake"
    assert images["path"] == str(tmp_path / "split" / "images")
    assert result.images is not None
    jpeg = hashlib.sha256((tmp_path / "pics" / "big.jpg").read_bytes()).hexdigest()
    assert (tmp_path / "split" / "images" / f"{jpeg}.jpg").is_file()


def test_a_text_split_reports_text(tmp_path):
    rows = [{"prompt": f"a {o} {i}", "label": o} for o in OPTIONS for i in range(12)]
    result = prepare(
        _request(tmp_path, data=_rows_file(tmp_path, rows, "text.jsonl"), context_length=1024)
    )
    assert result.as_dict()["modality"] == "text"
    assert result.as_dict()["request"]["context_length"] == 1024
    assert result.as_dict()["images"] is None
    assert not (tmp_path / "split" / "images").exists()


def test_decision_json_is_the_one_a_text_split_writes(tmp_path):
    prepare(_request(tmp_path))
    text_rows = [{"prompt": f"a {o} {i}", "label": o} for o in OPTIONS for i in range(12)]
    prepare(
        _request(
            tmp_path,
            data=_rows_file(tmp_path, text_rows, "text.jsonl"),
            output_dir=tmp_path / "text-split",
            context_length=1024,
        )
    )
    assert (tmp_path / "split" / "decision.json").read_bytes() == (
        tmp_path / "text-split" / "decision.json"
    ).read_bytes()


def test_one_picture_used_by_several_rows_is_one_file_and_a_noted_limitation(tmp_path):
    rows = _image_rows(tmp_path)
    rows.append({"image": "pics/cat-0.png", "label": "cat"})
    (tmp_path / "copy").mkdir()
    (tmp_path / "copy" / "same.png").write_bytes((tmp_path / "pics" / "cat-0.png").read_bytes())
    rows.append({"image": "copy/same.png", "label": "cat"})
    inspector = FakeImageInspector()
    result = prepare(
        _request(tmp_path, data=_rows_file(tmp_path, rows, "dup.jsonl"), images=inspector)
    )
    assert result.images is not None
    assert result.images.rows == 38 and result.images.files == 36
    assert result.images.duplicate_rows == 2
    assert len(list((tmp_path / "split" / "images").iterdir())) == 36
    assert any("2 image row(s) give the same bytes" in t for t in result.limitations)
    # Each file is opened once, however many rows name it.
    assert len(inspector.calls[0]) == len(set(inspector.calls[0])) == 37


def _picture_halves(split: Path) -> dict[str, set[str]]:
    return {
        name: {r["image_sha256"] for r in _records(split / f"{name}.jsonl")}
        for name in ("train", "calibration", "heldout")
    }


@pytest.mark.parametrize("seed", range(6))
def test_the_rows_of_one_picture_are_never_in_two_splits(tmp_path, seed):
    rows = _image_rows(tmp_path, per_option=6)
    # Every picture of the first nine rows given three times: groups of three rows.
    rows += [dict(r) for r in rows[:9]] + [dict(r) for r in rows[:9]]
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "dup.jsonl"), seed=seed))
    assert result.train is not None and result.heldout is not None
    halves = _picture_halves(tmp_path / "split")
    assert not halves["train"] & halves["heldout"]
    assert not halves["train"] & halves["calibration"]
    assert not halves["calibration"] & halves["heldout"]
    assert result.train.n + result.heldout.n + result.calibration.n == len(rows)


def test_explicit_groups_are_kept_together_and_reported(tmp_path):
    rows = _image_rows(tmp_path, per_option=6)
    for i, row in enumerate(rows):
        row["group"] = f"run-{i // 3}"
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "groups.jsonl")))
    assert (
        result.train is not None and result.calibration is not None and result.heldout is not None
    )
    split_of: dict[str, str] = {}
    for split in ("train", "calibration", "heldout"):
        for row in _records(tmp_path / "split" / f"{split}.jsonl"):
            assert row["group"] not in split_of or split_of[row["group"]] == split
            split_of[row["group"]] = split
    report = json.loads((tmp_path / "split" / "prepare.json").read_text())
    assert report["source"]["groups"] == {"used": True, "count": 6}


def test_picture_identity_joins_two_explicit_groups(tmp_path):
    rows = _image_rows(tmp_path, per_option=6)
    rows[0]["group"] = "first"
    duplicate = dict(rows[0], group="second")
    rows.append(duplicate)
    rows[1]["group"] = "second"
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "bridge.jsonl")))
    assert result.train is not None
    located: dict[str, str] = {}
    for split in ("train", "calibration", "heldout"):
        for row in _records(tmp_path / "split" / f"{split}.jsonl"):
            if row.get("group") in {"first", "second"}:
                located[row["group"]] = split
    assert located["first"] == located["second"]


@pytest.mark.parametrize("value", [None, 7, True, ["run"]])
def test_a_non_string_group_is_refused(tmp_path, value):
    rows = _image_rows(tmp_path)
    rows[0]["group"] = value
    with pytest.raises(PrepareError, match="'group' must be a string"):
        prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "bad-group.jsonl")))


def test_without_repeated_pictures_the_split_is_the_row_split(tmp_path):
    from litetune.prepare import Row, split_calibration, split_grouped, split_rows

    rows = [Row(lineno=i, prompt=f"r{i}", completion="a", target="a") for i in range(40)]
    digest = "sha256:" + "ab" * 32
    train, held = split_rows(rows, digest, 3, 8)
    train_part, aside = split_calibration(rows, digest, 3, 8, 4)
    grouped = split_grouped(rows, [f"g{i}" for i in range(40)], digest, 3, 8, 4)
    assert grouped == (train_part, aside, held)


def test_a_grouped_split_keeps_a_group_for_training_and_needs_two_groups():
    from litetune.prepare import Row, split_grouped

    rows = [Row(lineno=i, prompt=f"r{i}", completion="a", target="a") for i in range(6)]
    digest = "sha256:" + "cd" * 32
    assert split_grouped(rows, ["one"] * 6, digest, 0, 1, 1) is None
    train, aside, held = split_grouped(rows, ["x"] * 3 + ["y"] * 3, digest, 0, 5, 1)
    assert len(train) == 3 and len(held) == 3 and aside == []


def test_a_held_out_split_the_pictures_made_larger_is_said(tmp_path):
    rows = _image_rows(tmp_path, per_option=6)
    rows += [dict(r) for r in rows] * 3  # every picture four times
    result = prepare(
        _request(tmp_path, data=_rows_file(tmp_path, rows, "dup.jsonl"), heldout_size=3)
    )
    assert result.heldout is not None and result.heldout.n % 4 == 0
    (said,) = [t for t in result.limitations if "rows were held out" in t]
    assert f"{result.heldout.n} rows were held out" in said and "where 3 and" in said


def test_an_absolute_path_is_accepted(tmp_path):
    elsewhere = _fake_image(tmp_path / "far" / "away.png", tag="far")
    rows = _image_rows(tmp_path)
    rows[0] = {"image": str(elsewhere.resolve()), "label": "cat"}
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "abs.jsonl")))
    assert result.train is not None
    sha = hashlib.sha256(elsewhere.read_bytes()).hexdigest()
    assert (tmp_path / "split" / "images" / f"{sha}.png").read_bytes() == elsewhere.read_bytes()


def test_a_relative_path_is_read_from_the_rows_files_directory(tmp_path):
    nested = tmp_path / "data"
    rows = _image_rows(nested)
    data = _rows_file(nested, rows)
    result = prepare(_request(tmp_path, data=data))
    assert result.train is not None


def test_image_rows_measure_no_length_and_report_no_window(tmp_path):
    result = prepare(_request(tmp_path))
    assert LENGTH_CHECK not in [c.name for c in result.checks.checks]
    assert any(s.name == LENGTH_CHECK and "no text" in s.reason for s in result.skipped)
    report = result.as_dict()
    assert report["lengths"]["reason"].startswith("image rows have no text")
    assert "context_length" not in report["request"]
    assert result.train is not None


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        ({"context_length": 1024}, "--context-length: "),
        ({"tokens": "counter"}, "--tokenizer (and --tokenizer-revision): "),
        ({"context_length": 1024, "tokens": "counter"}, "--context-length, --tokenizer"),
        # Without --tokenizer, nothing reads the revision; for image rows it is refused.
        ({"tokenizer_revision": "main"}, "--tokenizer-revision: "),
        (
            {"tokens": "counter", "tokenizer_revision": "main"},
            "--tokenizer (and --tokenizer-revision): ",
        ),
    ],
)
def test_a_window_or_a_tokenizer_for_image_rows_is_refused_by_name(tmp_path, kwargs, named):
    from test_embedding_prepare import FakeEmbedTokenCounter

    counter = FakeEmbedTokenCounter()
    if "tokens" in kwargs:
        kwargs = kwargs | {"tokens": counter}
    with pytest.raises(PrepareError) as raised:
        prepare(_request(tmp_path, **kwargs))
    assert str(raised.value).startswith(named) and "these are image rows" in str(raised.value)
    assert counter.calls == []
    assert not (tmp_path / "split").exists()


def test_a_revision_without_a_tokenizer_is_ignored_for_text_rows_as_before(tmp_path):
    rows = _text_rows()
    data = _rows_file(tmp_path, rows, "text.jsonl")
    result = prepare(_request(tmp_path, data=data, context_length=1024, tokenizer_revision="x"))
    assert result.train is not None


def test_text_rows_without_a_window_are_refused(tmp_path):
    rows = _text_rows()
    with pytest.raises(PrepareError, match="--context-length is required for text rows"):
        prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "text.jsonl")))


def test_the_thin_option_note_applies_to_image_rows(tmp_path):
    from litetune.embedding_verify import GATED_K

    rows = [r for r in _image_rows(tmp_path, per_option=40) if r["label"] != "bird"]
    rows += [{"image": f"pics/bird-{i}.png", "label": "bird"} for i in range(GATED_K - 1)]
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "thin.jsonl")))
    note = next(t for t in result.limitations if "will refuse this split" in t)
    assert "'bird'" in note and "'cat'" not in note


def test_a_second_run_replaces_the_images_it_does_not_use(tmp_path):
    prepare(_request(tmp_path))
    stale = tmp_path / "split" / "images" / ("0" * 64 + ".png")
    stale.write_bytes(b"stale")
    prepare(_request(tmp_path))
    assert not stale.exists()
    assert len(list((tmp_path / "split" / "images").iterdir())) == 36
    assert not [p for p in (tmp_path / "split").iterdir() if p.name.startswith(".images")]


@pytest.mark.posix
def test_the_images_directory_has_the_mode_any_directory_gets(tmp_path):
    prepare(_request(tmp_path))
    (tmp_path / "plain").mkdir()
    mode = (tmp_path / "split" / "images").stat().st_mode & 0o777
    assert mode == (tmp_path / "plain").stat().st_mode & 0o777


# ---------------------------------------------------------------------------
# Refused, naming the row, and leaving nothing behind
# ---------------------------------------------------------------------------


def _refused(tmp_path: Path, rows: list[dict[str, Any]], match: str, **kwargs: Any) -> None:
    with pytest.raises(PrepareError, match=match):
        prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "given.jsonl"), **kwargs))
    assert not (tmp_path / "split").exists()


def test_text_rows_after_image_rows_are_refused(tmp_path):
    rows = _image_rows(tmp_path)
    rows.insert(5, {"prompt": "a cat", "label": "cat"})
    _refused(tmp_path, rows, r"given\.jsonl:6: this is a text row, and line 1 is an image row")


def test_image_rows_after_text_rows_are_refused(tmp_path):
    rows: list[dict[str, Any]] = [{"prompt": "a cat", "label": "cat"}, *_image_rows(tmp_path)]
    _refused(tmp_path, rows, r"given\.jsonl:2: this is an image row, and line 1 is a text row")


@pytest.mark.parametrize("key", ["completion", "target"])
def test_an_image_row_with_a_text_field_is_refused(tmp_path, key):
    rows = _image_rows(tmp_path)
    rows[3][key] = "cat"
    _refused(tmp_path, rows, rf"given\.jsonl:4: an image row is an 'image' and a 'label'.*{key}")


def test_a_row_with_an_image_and_a_prompt_is_refused_by_name(tmp_path):
    rows = _image_rows(tmp_path)
    rows[3]["prompt"] = "a cat"
    _refused(tmp_path, rows, r"given\.jsonl:4: this row gives both an 'image' and a 'prompt'")


def test_text_rows_with_an_image_column_are_refused_for_embeddinggemma(tmp_path):
    rows = [dict(r, image="pics/x.png") for r in _text_rows()]
    _refused(
        tmp_path,
        rows,
        r"given\.jsonl:1: this row gives both an 'image' and a 'prompt'",
        context_length=1024,
    )


@pytest.mark.parametrize("model", [None, "google/gemma-3-270m-it"])
def test_image_rows_for_another_model_are_refused_as_rows_without_a_prompt(tmp_path, model):
    _refused(
        tmp_path,
        _image_rows(tmp_path),
        r"given\.jsonl:1: expected an object with a 'prompt' field; a row with an 'image' and "
        rf"no 'prompt' is an image row, which prepare reads for --base-model {EMBEDDING_MODEL} "
        "only",
        base_model=model,
        context_length=1024,
    )


def test_a_missing_image_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    rows[7]["image"] = "pics/nowhere.png"
    _refused(tmp_path, rows, r"given\.jsonl:8: the image 'pics/nowhere.png' does not exist")


def test_a_directory_is_refused_as_not_a_regular_file(tmp_path):
    rows = _image_rows(tmp_path)
    rows[2]["image"] = "pics"
    _refused(tmp_path, rows, r"given\.jsonl:3: the image 'pics' is not a regular file")


@pytest.mark.parametrize("value", ["", 7, None])
def test_an_image_that_is_not_a_path_is_refused(tmp_path, value):
    rows = _image_rows(tmp_path)
    rows[0]["image"] = value
    _refused(tmp_path, rows, r"given\.jsonl:1: 'image' must be the path")


def test_an_undecodable_image_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    (tmp_path / "pics" / "dog-3.png").write_bytes(b"\x89PNG not really")
    _refused(tmp_path, rows, r"given\.jsonl:16: .*dog-3\.png is not an image Pillow 0\.0-fake")


def test_an_animated_image_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    _fake_image(tmp_path / "pics" / "bird-0.png", fmt="PNG", frames=2)
    _refused(tmp_path, rows, r"given\.jsonl:25: .*bird-0\.png holds 2 frames")


# "XVThumb" is a Pillow format name in mixed case: a row refused, not a checker failure.
@pytest.mark.parametrize("fmt", ["GIF", "WEBP", "BMP", "TIFF", "XVThumb"])
def test_a_format_other_than_png_or_jpeg_is_refused_naming_the_row(tmp_path, fmt):
    rows = _image_rows(tmp_path)
    _fake_image(tmp_path / "pics" / "dog-1.png", fmt=fmt)
    _refused(
        tmp_path,
        rows,
        rf"given\.jsonl:14: .*dog-1\.png is '{fmt}' to Pillow 0\.0-fake, not PNG or JPEG\. "
        "LiteRT-LM decodes the picture an application gives it with stb_image",
    )


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "MPO"])
def test_png_jpeg_and_mpo_are_accepted(tmp_path, fmt):
    rows = _image_rows(tmp_path)
    _fake_image(tmp_path / "pics" / "dog-1.png", fmt=fmt)
    assert prepare(_request(tmp_path, data=_rows_file(tmp_path, rows))).train is not None


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_only_mpo_may_hold_more_than_one_frame(tmp_path, fmt):
    rows = _image_rows(tmp_path)
    _fake_image(tmp_path / "pics" / "bird-0.png", fmt=fmt, frames=3)
    _refused(tmp_path, rows, r"given\.jsonl:25: .*bird-0\.png holds 3 frames")


def test_an_mpo_of_two_frames_is_accepted_and_copied_whole_as_jpg(tmp_path):
    rows = _image_rows(tmp_path)
    mpo = _fake_image(tmp_path / "pics" / "bird-0.png", fmt="MPO", frames=2)
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows)))
    assert result.images is not None and result.images.formats.get("MPO") == 1
    sha = hashlib.sha256(mpo.read_bytes()).hexdigest()
    copied = tmp_path / "split" / "images" / f"{sha}.jpg"
    assert copied.read_bytes() == mpo.read_bytes()


def test_an_mpo_extension_is_jpg():
    assert ImageFacts("a" * 64, "MPO", 4, 3, 2, None).extension == "jpg"


@pytest.mark.parametrize(("width", "height"), [(0, 3), (4, 0)])
def test_a_zero_sized_image_is_refused(tmp_path, width, height):
    rows = _image_rows(tmp_path)
    _fake_image(tmp_path / "pics" / "cat-1.png", width=width, height=height)
    _refused(tmp_path, rows, rf"given\.jsonl:2: .*is {width}x{height} pixels")


def test_one_picture_with_two_labels_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    (tmp_path / "pics" / "dog-0.png").write_bytes((tmp_path / "pics" / "cat-0.png").read_bytes())
    _refused(
        tmp_path,
        rows,
        r"given\.jsonl:13: .*dog-0\.png has the same bytes .* as line 1, which is labelled "
        r"'cat', and this row is labelled 'dog'",
    )


def test_a_null_label_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    rows[0]["label"] = None
    _refused(tmp_path, rows, r"given\.jsonl:1: the label is null")


def test_a_label_that_is_not_a_string_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    rows[0]["label"] = 3
    _refused(tmp_path, rows, r"given\.jsonl:1: the label must be an option id")


def test_an_unlabelled_image_row_is_refused(tmp_path):
    rows = _image_rows(tmp_path)
    del rows[4]["label"]
    _refused(tmp_path, rows, r"given\.jsonl:5: no 'label'")


def test_a_none_option_is_refused_for_image_rows(tmp_path):
    rows = _image_rows(tmp_path)
    _refused(
        tmp_path,
        rows,
        r"none_option \('other'\)",
        questions=_questions(tmp_path, "none.json", none_option="other"),
    )


def test_an_unavailable_checker_is_a_refusal_not_an_empty_answer(tmp_path):
    @dataclass
    class Broken:
        def describe(self) -> dict[str, Any]:
            return {}

        def inspect(self, paths: Sequence[Path]) -> list[ImageFacts]:
            raise ImageCheckUnavailable("environment 'embed-train' unavailable: no network")

    _refused(tmp_path, _image_rows(tmp_path), "could not be checked.*no network", images=Broken())


def test_a_checker_answering_for_fewer_images_is_refused(tmp_path):
    @dataclass
    class Short(FakeImageInspector):
        def inspect(self, paths: Sequence[Path]) -> list[ImageFacts]:
            return super().inspect(paths)[:-1]

    _refused(tmp_path, _image_rows(tmp_path), "answered for 35 of 36 images", images=Short())


def test_an_image_changed_after_its_check_leaves_no_images_behind(tmp_path):
    # The checker answers a digest the bytes do not have, as it would for a file
    # rewritten between the check and the copy.
    with pytest.raises(PrepareError, match="changed after it was checked"):
        prepare(_request(tmp_path, images=FakeImageInspector(misread="dog-4.png")))
    split = tmp_path / "split"
    assert not (split / "images").exists()
    assert not [p for p in split.iterdir() if p.name.startswith(".images")]
    assert not (split / "train.jsonl").exists()


def test_a_failed_copy_keeps_the_previous_images(tmp_path):
    prepare(_request(tmp_path))
    before = sorted(p.name for p in (tmp_path / "split" / "images").iterdir())
    with pytest.raises(PrepareError, match="changed after it was checked"):
        prepare(_request(tmp_path, images=FakeImageInspector(misread="bird-7.png")))
    assert sorted(p.name for p in (tmp_path / "split" / "images").iterdir()) == before


def test_a_split_whose_options_check_fails_writes_no_images(tmp_path):
    rows = [r for r in _image_rows(tmp_path) if r["label"] != "bird"]
    result = prepare(_request(tmp_path, data=_rows_file(tmp_path, rows, "two.jsonl")))
    assert result.outcome is Outcome.FAILED
    assert not (tmp_path / "split" / "images").exists()
    assert (tmp_path / "split" / "prepare.json").is_file()


# -- write_images on its own -----------------------------------------------------


def _sources(tmp_path: Path, n: int = 3, tag: str = "") -> dict[str, Path]:
    out = {}
    for i in range(n):
        source = tmp_path / "src" / f"{tag}{i}.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"picture {tag}{i}".encode())
        out[f"{hashlib.sha256(source.read_bytes()).hexdigest()}.png"] = source
    return out


@pytest.mark.parametrize("kind", ["dir", "symlink"])
def test_write_images_refuses_an_images_dir_holding_what_it_did_not_write(tmp_path, kind):
    from litetune.images import ImageCopyError, write_images

    out = tmp_path / "split"
    write_images(out, _sources(tmp_path))
    foreign = out / "images" / "mine"
    if kind == "dir":
        foreign.mkdir()
    else:
        foreign.symlink_to(tmp_path)
    with pytest.raises(ImageCopyError, match=r"holds mine, which prepare did not write"):
        write_images(out, _sources(tmp_path, tag="b"))
    assert foreign.exists() or foreign.is_symlink()


def test_a_final_rename_that_fails_puts_the_previous_images_back(tmp_path, monkeypatch):
    from litetune.images import ImageCopyError, write_images

    out = tmp_path / "split"
    first = _sources(tmp_path)
    write_images(out, first)
    real = Path.rename

    def rename(self, target):
        if self.name.startswith(".images-") and Path(target).name == "images":
            raise OSError(28, "No space left on device")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(ImageCopyError, match="previous images are back in place") as raised:
        write_images(out, _sources(tmp_path, tag="b"))
    assert isinstance(raised.value.__cause__, OSError)
    assert sorted(p.name for p in (out / "images").iterdir()) == sorted(first)
    assert [p.name for p in out.iterdir()] == ["images"]


def test_a_restore_that_fails_too_names_both_and_is_chained(tmp_path, monkeypatch):
    from litetune.images import ImageCopyError, write_images

    out = tmp_path / "split"
    write_images(out, _sources(tmp_path))
    real = Path.rename

    def rename(self, target):
        if Path(target).name == "images" and self.parent.name != "split":
            raise OSError(13, "Permission denied: restore")
        if self.name.startswith(".images-") and Path(target).name == "images":
            raise OSError(28, "No space left on device")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", rename)
    with pytest.raises(ImageCopyError, match="could not be moved back") as raised:
        write_images(out, _sources(tmp_path, tag="b"))
    assert "restore" in str(raised.value.__cause__)
    assert "No space left" in str(raised.value.__cause__.__context__)


def test_leftovers_of_a_stopped_run_are_removed_before_writing(tmp_path):
    from litetune.images import write_images

    out = tmp_path / "split"
    staging = out / (".images-" + "a" * 16)
    retired = out / (".images-old-" + "b" * 16) / "images"
    for directory in (staging, retired):
        directory.mkdir(parents=True)
        (directory / "x.png").write_bytes(b"left")
    unrelated = out / ".images-mine"
    unrelated.mkdir()
    write_images(out, _sources(tmp_path))
    assert sorted(p.name for p in out.iterdir()) == [".images-mine", "images"]


def test_a_leftover_holding_a_tree_is_refused_not_removed(tmp_path):
    from litetune.images import ImageCopyError, write_images

    out = tmp_path / "split"
    nested = out / (".images-" + "c" * 16) / "deeper"
    nested.mkdir(parents=True)
    with pytest.raises(ImageCopyError, match="earlier prepare left when it was stopped"):
        write_images(out, _sources(tmp_path))
    assert nested.is_dir()


def test_a_source_that_cannot_be_read_is_an_image_copy_error_naming_it(tmp_path):
    from litetune.images import ImageCopyError, write_images

    sources = _sources(tmp_path)
    name, victim = next(iter(sorted(sources.items())))
    victim.unlink()
    # A Windows path's backslashes are regex escapes: escape it.
    with pytest.raises(ImageCopyError, match=rf"{re.escape(str(victim))} could not be copied"):
        write_images(tmp_path / "split", sources)
    assert not (tmp_path / "split" / "images").exists()


# ---------------------------------------------------------------------------
# Text splits are what they were
# ---------------------------------------------------------------------------

# Text rows that also carry an "image" column, split by main (3cf0652): the
# column is ignored there, as it is now for every model but EmbeddingGemma 2.
EXTRA_IMAGE_COLUMN_SHA256 = {
    "decision": {
        "train.jsonl": "5c74cdb159d9a802896933b6d658b8e16389d9d407c631f8975e0dd808e402f0",
        "calibration.jsonl": "6b60f093bf9b5ba124513b985730e2b05afa4fbc01070b7af2096be3c17808a2",
        "heldout.jsonl": "fdd0152269e9068f9e5be9fcad9bf35e167290d7a60e44ba64b7752c39934460",
        "decision.json": "f71a3b29efd9b30d82d104d7a0b6649376d19177890044e0435a44e202855db1",
    },
    "sft": {
        "train.jsonl": "110709ca423e64adcdafe3ea5070ebd7dac9babf3d8676f4fbbbeff1eb704e4a",
        "heldout.jsonl": "8c53e8d84ca2910209aacd98e050ac905edf612c5d45e7fd87357a5ab62839d2",
    },
}


@pytest.mark.parametrize("case", ["decision", "sft"])
def test_text_rows_with_an_image_column_split_as_on_main_for_an_llm(tmp_path, case):
    if case == "decision":
        rows = [
            {"prompt": f"photo {i} of a {o}", "label": o, "image": f"pics/{o}-{i}.png"}
            for o in OPTIONS
            for i in range(12)
        ]
        model, questions = "google/gemma-3-270m-it", _questions(tmp_path)
    else:
        rows = [
            {"prompt": f"say {i}", "completion": f"ok {i}", "image": "pics/x.png"}
            for i in range(30)
        ]
        # A named model that takes no pictures: with none named, rows that carry
        # an `image` are refused (`prepare.IMAGES_WITHOUT_A_MODEL`).
        model, questions = "google/gemma-3-270m-it", None
    inspector = FakeImageInspector()
    result = prepare(
        PrepareRequest(
            data=_rows_file(tmp_path, rows, f"{case}.jsonl"),
            output_dir=tmp_path / "split",
            context_length=1024,
            base_model=model,
            questions=questions,
            images=inspector,
        )
    )
    assert inspector.calls == [] and result.images is None
    assert result.as_dict()["modality"] == "text"
    assert not (tmp_path / "split" / "images").exists()
    for name, digest in EXTRA_IMAGE_COLUMN_SHA256[case].items():
        written = (tmp_path / "split" / name).read_bytes()
        assert hashlib.sha256(written).hexdigest() == digest, name


# sha256 of train/calibration/heldout.jsonl for `_text_rows` before image rows
# existed (3cf0652), written from bytes so every platform splits them alike.
TEXT_SPLIT_SHA256 = {
    "train.jsonl": "c01c720c359b8c98a781247748800819ab4309ab011d2b7ec820c637963a26e7",
    "calibration.jsonl": "1b73c0751994ad5ba797a5244fb6f6dc7dbdf6acacd0876f67506f7ef0590aa6",
    "heldout.jsonl": "c925b9db671307e3a87ba7e90264b83456f1779006cb2a5e7a4d9ba9050574ae",
}


def _text_rows() -> list[dict[str, Any]]:
    return [{"prompt": f"photo {i} of a {o}", "label": o} for o in OPTIONS for i in range(12)]


def test_a_text_split_is_byte_identical_to_before(tmp_path):
    prepare(
        _request(
            tmp_path, data=_rows_file(tmp_path, _text_rows(), "text.jsonl"), context_length=1024
        )
    )
    for name, digest in TEXT_SPLIT_SHA256.items():
        written = (tmp_path / "split" / name).read_bytes()
        assert hashlib.sha256(written).hexdigest() == digest, name


# ---------------------------------------------------------------------------
# The command line: the check runs in EMBED_TRAIN, the tokenizer does not
# ---------------------------------------------------------------------------


@dataclass
class FakeImageEnv:
    """`StageEnv.run` answering `IMAGE_CHECK_SCRIPT` from fake images, and recording the run."""

    scripts: list[str] = field(default_factory=list)
    envs_run: list[str] = field(default_factory=list)

    def __call__(self, env, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.scripts.append(Path(args[1]).read_text(encoding="utf-8"))
        self.envs_run.append(env.name)
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        found = []
        for path in spec["paths"]:
            facts = _fake_facts(Path(path).read_bytes())
            found.append(
                {
                    "sha256": facts.sha256,
                    "format": facts.format,
                    "width": facts.width,
                    "height": facts.height,
                    "frames": facts.frames,
                    "error": facts.error,
                }
            )
        Path(spec["out"]).write_text(
            json.dumps({"pillow": "12.3.0", "images": found}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, "", "")


def test_the_command_line_checks_images_in_embed_train_and_counts_no_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    fake = FakeImageEnv()

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
            str(_rows_file(tmp_path, _image_rows(tmp_path))),
            "--output-dir",
            str(tmp_path / "split"),
            "--questions",
            str(_questions(tmp_path)),
            "--base-model",
            EMBEDDING_MODEL,
        ]
    )
    assert code == 0
    assert fake.envs_run == [envs.EMBED_TRAIN.name]
    assert fake.scripts == [IMAGE_CHECK_SCRIPT]
    report = json.loads((tmp_path / "split" / "prepare.json").read_text(encoding="utf-8"))
    assert report["images"]["checker"]["pillow"] == "12.3.0"
    assert report["images"]["checker"]["environment"] == envs.EMBED_TRAIN.name
    assert len(list((tmp_path / "split" / "images").iterdir())) == 36


def test_the_command_line_refuses_a_tokenizer_revision_alone_for_image_rows(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    fake = FakeImageEnv()
    monkeypatch.setattr(
        envs.StageEnv, "run", lambda self, args, timeout=3600, **kw: fake(self, args, timeout, **kw)
    )
    code = main(
        [
            "prepare",
            "--data",
            str(_rows_file(tmp_path, _image_rows(tmp_path))),
            "--output-dir",
            str(tmp_path / "split"),
            "--questions",
            str(_questions(tmp_path)),
            "--base-model",
            EMBEDDING_MODEL,
            "--tokenizer-revision",
            "main",
        ]
    )
    assert code != 0
    assert "--tokenizer-revision: these are image rows" in capsys.readouterr().err
    assert fake.envs_run == []
    assert not (tmp_path / "split").exists()


@pytest.mark.parametrize(
    "answer",
    [
        {"sha256": "x" * 64, "format": "PNG", "width": 1, "height": 1, "frames": 1, "error": None},
        {"sha256": "a" * 64, "format": "../x", "width": 1, "height": 1, "frames": 1, "error": None},
        {
            "sha256": "a" * 64,
            "format": "PNG",
            "width": "1",
            "height": 1,
            "frames": 1,
            "error": None,
        },
        {
            "sha256": "a" * 64,
            "format": "PNG",
            "width": 1,
            "height": 1,
            "frames": True,
            "error": None,
        },
        {"sha256": None, "format": "PNG", "width": 1, "height": 1, "frames": 1, "error": None},
        {"sha256": "a" * 64, "error": 3},
    ],
)
def test_an_answer_not_in_the_scripts_shape_is_refused(answer):
    with pytest.raises(ImageCheckUnavailable):
        facts_from(answer)


def test_a_failed_check_script_is_unavailable_not_an_answer(monkeypatch, tmp_path):
    def run(self, args, timeout: int = 3600, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "ModuleNotFoundError: No module named PIL")

    monkeypatch.setattr(envs.StageEnv, "run", run)
    image = _fake_image(tmp_path / "a.png")
    with pytest.raises(ImageCheckUnavailable, match="No module named PIL"):
        PillowImageInspector(auto_provision=False).inspect([image])


def test_a_script_that_exits_non_zero_after_answering_is_not_believed(monkeypatch, tmp_path):
    image = _fake_image(tmp_path / "a.png")

    def run(self, args, timeout: int = 3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        facts = _fake_facts(image.read_bytes())
        answer = {"sha256": facts.sha256, "format": "PNG", "width": 4, "height": 3, "frames": 1}
        Path(spec["out"]).write_text(json.dumps({"images": [answer | {"error": None}]}))
        return subprocess.CompletedProcess(args, 1, "", "killed while flushing")

    monkeypatch.setattr(envs.StageEnv, "run", run)
    with pytest.raises(ImageCheckUnavailable, match="killed while flushing"):
        PillowImageInspector(auto_provision=False).inspect([image])


# ---------------------------------------------------------------------------
# The script itself, on images Pillow writes
# ---------------------------------------------------------------------------


def _exec_script(monkeypatch, config: Path) -> None:
    """Exec `IMAGE_CHECK_SCRIPT` in this process, with the host's Pillow, on `config`."""
    namespace: dict[str, Any] = {"__name__": "check_images"}
    exec(compile(IMAGE_CHECK_SCRIPT, "check_images.py", "exec"), namespace)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["check_images.py", str(config)])
        assert namespace["main"]() == 0


def _run_script(monkeypatch, tmp_path: Path, paths: Sequence[Path]) -> dict[str, Any]:
    out = tmp_path / "answer.json"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"paths": [str(p) for p in paths], "out": str(out)}))
    _exec_script(monkeypatch, config)
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_script_reads_real_images(monkeypatch, tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    png = tmp_path / "a.png"
    image_module.new("RGB", (5, 7), "red").save(png)
    jpeg = tmp_path / "b.jpeg"
    image_module.new("L", (9, 2)).save(jpeg, format="JPEG")
    gif = tmp_path / "c.gif"
    frames = [image_module.new("RGB", (3, 3), c) for c in ("red", "green", "blue")]
    frames[0].save(gif, save_all=True, append_images=frames[1:])
    broken = tmp_path / "d.png"
    broken.write_bytes(png.read_bytes()[:40])
    text = tmp_path / "e.png"
    text.write_text("not an image")
    # A header that opens, and pixel data cut short: only decoding finds it.
    noise = image_module.effect_noise((64, 64), 50).convert("RGB")
    whole = tmp_path / "whole.png"
    noise.save(whole)
    truncated = tmp_path / "f.png"
    truncated.write_bytes(whole.read_bytes()[: whole.stat().st_size // 2])

    answer = _run_script(monkeypatch, tmp_path, [png, jpeg, gif, broken, text, truncated])
    assert answer["pillow"] == pytest.importorskip("PIL").__version__
    facts = [facts_from(a) for a in answer["images"]]
    assert (facts[0].format, facts[0].width, facts[0].height, facts[0].frames) == (
        "PNG",
        5,
        7,
        1,
    )
    assert facts[0].sha256 == hashlib.sha256(png.read_bytes()).hexdigest()
    assert facts[0].error is None and facts[0].extension == "png"
    assert (facts[1].format, facts[1].extension, facts[1].width) == ("JPEG", "jpg", 9)
    assert facts[2].format == "GIF" and facts[2].frames == 3
    assert facts[3].error is not None and facts[3].sha256 is not None
    assert facts[4].error is not None and "cannot identify" in facts[4].error
    assert facts[5].error is not None and "truncated" in facts[5].error


def test_the_script_refuses_a_decompression_bomb(monkeypatch, tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    big = tmp_path / "big.png"
    image_module.new("1", (100, 100)).save(big)
    # Below the size the script is given, so the warning becomes its error.
    monkeypatch.setattr(image_module, "MAX_IMAGE_PIXELS", 100 * 100 - 1)
    answer = _run_script(monkeypatch, tmp_path, [big])
    assert "DecompressionBomb" in answer["images"][0]["error"]


def test_the_script_answers_a_missing_file_per_file(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    answer = _run_script(monkeypatch, tmp_path, [tmp_path / "gone.png"])
    assert answer["images"][0]["sha256"] is None
    assert answer["images"][0]["error"].startswith("FileNotFoundError")


def test_prepare_end_to_end_with_the_real_script(monkeypatch, tmp_path):
    """The real inspector, its env's run replaced by an in-process exec of the script."""
    pil = pytest.importorskip("PIL")
    image_module = pytest.importorskip("PIL.Image")
    rows = []
    for o, colour in zip(OPTIONS, ("red", "green", "blue"), strict=True):
        for i in range(12):
            path = tmp_path / "pics" / f"{o}-{i}.png"
            path.parent.mkdir(exist_ok=True)
            image_module.new("RGB", (8 + i, 8), colour).save(path)
            rows.append({"image": f"pics/{o}-{i}.png", "label": o})
    animated = tmp_path / "pics" / "anim.webp"
    frames = [image_module.new("RGB", (4, 4), c) for c in ("red", "blue")]
    frames[0].save(animated, save_all=True, append_images=frames[1:])

    def run(self, args, timeout: int = 3600, **kwargs):
        _exec_script(monkeypatch, Path(args[2]))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", run)
    real = PillowImageInspector(auto_provision=False)
    data = _rows_file(tmp_path, rows, "real.jsonl")
    result = prepare(_request(tmp_path, data=data, images=real))
    assert result.images is not None and result.images.formats == {"PNG": 36}
    assert result.images.checker["pillow"] == pil.__version__
    copied = tmp_path / "split" / "images"
    assert {hashlib.sha256(p.read_bytes()).hexdigest() + ".png" for p in copied.iterdir()} == {
        p.name for p in copied.iterdir()
    }

    rows.append({"image": "pics/anim.webp", "label": "cat"})
    # Pillow's own name for the format, refused for the format before its frames.
    with pytest.raises(PrepareError, match=r"anim\.webp is 'WEBP' to Pillow .*not PNG or JPEG"):
        prepare(
            _request(
                tmp_path,
                data=_rows_file(tmp_path, rows, "anim.jsonl"),
                output_dir=tmp_path / "split-anim",
                images=PillowImageInspector(auto_provision=False),
            )
        )
    assert not (tmp_path / "split-anim").exists()


def test_a_real_mpo_is_accepted_as_its_first_frame_and_an_apng_refused(monkeypatch, tmp_path):
    """Pillow writes both: an MPO of two frames, as a phone camera does, and an animated PNG."""
    image_module = pytest.importorskip("PIL.Image")
    rows = []
    for o, colour in zip(OPTIONS, ("red", "green", "blue"), strict=True):
        for i in range(12):
            path = tmp_path / "pics" / f"{o}-{i}.png"
            path.parent.mkdir(exist_ok=True)
            image_module.new("RGB", (8 + i, 8), colour).save(path)
            rows.append({"image": f"pics/{o}-{i}.png", "label": o})
    mpo = tmp_path / "pics" / "camera.jpg"
    first, second = (image_module.new("RGB", (8, 6), c) for c in ("red", "blue"))
    first.save(mpo, format="MPO", save_all=True, append_images=[second])
    with image_module.open(mpo) as opened:
        assert (opened.format, opened.n_frames) == ("MPO", 2)
    rows.append({"image": "pics/camera.jpg", "label": "cat"})

    def run(self, args, timeout: int = 3600, **kwargs):
        _exec_script(monkeypatch, Path(args[2]))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", run)
    result = prepare(
        _request(
            tmp_path,
            data=_rows_file(tmp_path, rows, "mpo.jsonl"),
            images=PillowImageInspector(auto_provision=False),
        )
    )
    assert result.images is not None and result.images.formats == {"PNG": 36, "MPO": 1}
    sha = hashlib.sha256(mpo.read_bytes()).hexdigest()
    assert (tmp_path / "split" / "images" / f"{sha}.jpg").read_bytes() == mpo.read_bytes()
    split = tmp_path / "split"
    written = [
        *_records(split / "train.jsonl"),
        *_records(split / "heldout.jsonl"),
        *_records(split / "calibration.jsonl"),
    ]
    assert [r["image"] for r in written if r["image_sha256"] == sha] == [f"images/{sha}.jpg"]

    apng = tmp_path / "pics" / "anim.png"
    first.save(apng, save_all=True, append_images=[second])
    rows[-1] = {"image": "pics/anim.png", "label": "cat"}
    with pytest.raises(PrepareError, match=r"anim\.png holds 2 frames"):
        prepare(
            _request(
                tmp_path,
                data=_rows_file(tmp_path, rows, "apng.jsonl"),
                output_dir=tmp_path / "split-apng",
                images=PillowImageInspector(auto_provision=False),
            )
        )
