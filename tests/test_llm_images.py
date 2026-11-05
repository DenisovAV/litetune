"""`llm_images`: reading a split's pictures, and an image checkpoint's record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from litetune import llm_images
from litetune.llm_images import (
    PROCESSOR_FILES,
    RECORD_KEY,
    TEMPLATE_FILE,
    VISION_TOKENS,
    ImageRowError,
    checked_template,
    image_record,
    is_image_model,
    processor_mismatch,
    split_images,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split(tmp_path: Path, n: int = 3, override: dict[int, dict[str, Any]] | None = None) -> Path:
    images = tmp_path / "images"
    images.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        data = f"picture {i}".encode()
        name = f"{_sha(data)}.jpg"
        (images / name).write_bytes(data)
        row = {
            "prompt": "q",
            "completion": "a",
            "image": f"images/{name}",
            "image_sha256": _sha(data),
        }
        row.update((override or {}).get(i, {}))
        lines.append(json.dumps(row))
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_split_gives_each_rows_picture_in_order(tmp_path):
    found = split_images(_split(tmp_path))
    assert found is not None
    assert [ref.sha256 for ref in found] == [_sha(f"picture {i}".encode()) for i in range(3)]
    assert all(ref.file.is_file() for ref in found)


def test_a_text_split_has_no_pictures(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"prompt": "q", "completion": "a"}\n', encoding="utf-8")
    assert split_images(path) is None


def test_a_mixed_split_is_refused(tmp_path):
    path = _split(tmp_path, override={1: {"image": None}})
    with pytest.raises(ImageRowError, match=r":2: this row gives no picture, and line 1 does"):
        split_images(path)


def test_a_changed_picture_is_refused(tmp_path):
    path = _split(tmp_path)
    ref = split_images(path)[0]  # type: ignore[index]
    ref.file.write_bytes(b"another picture")
    with pytest.raises(ImageRowError, match=r":1: the image .* hashes to .*changed after"):
        split_images(path)


def test_a_missing_picture_is_refused(tmp_path):
    path = _split(tmp_path, override={2: {"image": "images/gone.jpg"}})
    with pytest.raises(ImageRowError, match=r":3: the image 'images/gone.jpg' does not exist"):
        split_images(path)


@pytest.mark.parametrize("value", ["../outside.jpg", "/etc/passwd", "C:\\x.jpg", "images\\a.jpg"])
def test_a_picture_outside_the_split_is_refused(tmp_path, value):
    (tmp_path.parent / "outside.jpg").write_bytes(b"x")
    path = _split(tmp_path, override={0: {"image": value}})
    with pytest.raises(ImageRowError, match=r":1: the image .* is not inside the split"):
        split_images(path)


def test_a_row_without_a_digest_is_refused(tmp_path):
    path = _split(tmp_path, override={0: {"image_sha256": "abc"}})
    with pytest.raises(ImageRowError, match=r":1: 'image_sha256' is 'abc', not a sha256"):
        split_images(path)


def test_an_image_that_is_not_a_string_is_refused(tmp_path):
    path = _split(tmp_path, override={0: {"image": 3}})
    with pytest.raises(ImageRowError, match=r":1: 'image' must be the picture's path"):
        split_images(path)


def test_only_gemma_4_e2b_is_the_image_model():
    assert is_image_model("google/gemma-4-E2B-it")
    assert not is_image_model("google/gemma-4-E4B-it")
    assert not is_image_model("Qwen/Qwen3-0.6B")
    assert not is_image_model(None)


# -- the record ----------------------------------------------------------------


def _checkpoint(tmp_path: Path, **record: Any) -> Path:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    for name in PROCESSOR_FILES:
        (model / name).write_text(name, encoding="utf-8")
    (model / TEMPLATE_FILE).write_text("{{ bos_token }}", encoding="utf-8")
    body = {
        "vision_tokens": VISION_TOKENS,
        "template_sha256": _sha(b"{{ bos_token }}"),
        "processor_sha256": {n: _sha(n.encode()) for n in PROCESSOR_FILES},
    }
    body.update(record)
    (model / "litetune.json").write_text(
        json.dumps({"base_model": "google/gemma-4-E2B-it", RECORD_KEY: body}), encoding="utf-8"
    )
    return model


def test_a_record_reads_back_and_its_files_match(tmp_path):
    model = _checkpoint(tmp_path)
    record = image_record(model)
    assert record is not None
    assert checked_template(model, record) == "{{ bos_token }}"
    assert processor_mismatch(model, record) is None


def test_a_checkpoint_without_the_record_is_none(tmp_path):
    model = tmp_path / "plain"
    model.mkdir()
    (model / "litetune.json").write_text('{"base_model": "x"}', encoding="utf-8")
    assert image_record(model) is None
    assert image_record("google/gemma-4-E2B-it") is None


@pytest.mark.parametrize(
    ("field", "value", "said"),
    [
        ("vision_tokens", 140, "vision_tokens"),
        ("template_sha256", "x", "template_sha256"),
        ("processor_sha256", {"processor_config.json": None}, "processor_sha256"),
    ],
)
def test_a_record_this_does_not_read_is_refused(tmp_path, field, value, said):
    model = _checkpoint(tmp_path, **{field: value})
    with pytest.raises(ImageRowError, match=said):
        image_record(model)


def test_a_processor_digest_that_is_not_one_is_refused(tmp_path):
    digests = {n: _sha(n.encode()) for n in PROCESSOR_FILES}
    digests["tokenizer.json"] = "nope"
    with pytest.raises(ImageRowError, match=r"gives tokenizer.json 'nope'"):
        image_record(_checkpoint(tmp_path, processor_sha256=digests))


def test_a_changed_processor_file_is_named(tmp_path):
    model = _checkpoint(tmp_path)
    record = image_record(model)
    assert record is not None
    (model / "processor_config.json").write_text('{"max_soft_tokens": 140}', encoding="utf-8")
    said = processor_mismatch(model, record)
    assert said is not None and "processor_config.json hashes to" in said


def test_a_changed_or_missing_template_is_refused(tmp_path):
    model = _checkpoint(tmp_path)
    record = image_record(model)
    assert record is not None
    (model / TEMPLATE_FILE).write_text("{{ other }}", encoding="utf-8")
    with pytest.raises(ImageRowError, match=r"the template changed after `tune`"):
        checked_template(model, record)
    (model / TEMPLATE_FILE).unlink()
    with pytest.raises(ImageRowError, match=r"is not there"):
        checked_template(model, record)


def test_export_flags_name_the_budget_the_metadata_checks():
    assert f"--gemma4_vision_max_soft_tokens={VISION_TOKENS}" in llm_images.EXPORT_FLAGS
    assert llm_images.MAX_NUM_PATCHES == 2520
