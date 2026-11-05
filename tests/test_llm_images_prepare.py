"""`prepare --base-model google/gemma-4-E2B-it` on rows that carry a picture.

The pictures are this suite's fake images (`test_embedding_prepare_images`):
a line `FAKE <format> <width> <height> <frames>`, read by `FakeImageInspector`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_embedding_prepare_images import FakeImageInspector, _fake_image, _records, _rows_file

from litetune import envs
from litetune.checks import Outcome
from litetune.images import PillowImageInspector
from litetune.llm_images import IMAGE_MODEL, split_images
from litetune.prepare import (
    IMAGES_IGNORED,
    LENGTH_CHECK,
    LLM_IMAGE_LENGTHS,
    LLM_IMAGES_READ,
    PrepareError,
    PrepareRequest,
    prepare,
)

LABELS = ("angular_leaf_spot", "bean_rust", "healthy")
QUESTION = "Which of these does this bean leaf show? Answer with the label only."


def _rows(tmp_path: Path, per_label: int = 12) -> list[dict[str, Any]]:
    rows = []
    for label in LABELS:
        for i in range(per_label):
            _fake_image(tmp_path / "pics" / f"{label}-{i}.jpg", fmt="JPEG", width=500, height=500)
            rows.append({"prompt": QUESTION, "completion": label, "image": f"pics/{label}-{i}.jpg"})
    return rows


def _request(tmp_path: Path, rows: list[dict[str, Any]] | None = None, **kwargs: Any):
    params: dict[str, Any] = {
        "data": _rows_file(tmp_path, rows if rows is not None else _rows(tmp_path)),
        "output_dir": tmp_path / "split",
        "context_length": 4096,
        "base_model": IMAGE_MODEL,
        "images": FakeImageInspector(),
    }
    params.update(kwargs)
    return PrepareRequest(**params)


def test_rows_with_pictures_keep_prompt_completion_and_the_copied_picture(tmp_path):
    result = prepare(_request(tmp_path))
    assert result.outcome is not Outcome.FAILED
    split = tmp_path / "split"
    records = [*_records(split / "train.jsonl"), *_records(split / "heldout.jsonl")]
    assert len(records) == 36
    given = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in (tmp_path / "pics").iterdir()}
    for record in records:
        assert record["prompt"] == QUESTION
        assert record["completion"] in LABELS
        assert record["image"] == f"images/{record['image_sha256']}.jpg"
        copied = (split / record["image"]).read_bytes()
        # Byte for byte the file the row named, found by its digest.
        assert copied == given[record["image_sha256"]].read_bytes()
        assert record["completion"] in given[record["image_sha256"]].name
    # The split's own reader finds every picture and its digest.
    found = split_images(split / "train.jsonl")
    assert found is not None and len(found) == result.train.n
    report = json.loads((split / "prepare.json").read_text())
    assert report["modality"] == "image"
    assert report["images"]["files"] == 36


def test_the_pictures_are_opened_in_train_vision_by_default(tmp_path, monkeypatch):
    seen: list[envs.StageEnv] = []

    def inspect(self, paths):
        seen.append(self.env)
        return FakeImageInspector().inspect(paths)

    monkeypatch.setattr(PillowImageInspector, "inspect", inspect)
    prepare(_request(tmp_path, images=None))
    assert seen == [envs.TRAIN_VISION]


def test_a_file_mixing_rows_with_and_without_pictures_is_refused(tmp_path):
    rows = _rows(tmp_path)
    rows[5] = {"prompt": QUESTION, "completion": "healthy"}
    with pytest.raises(PrepareError, match=r":6: this row gives no picture, and line 1 does"):
        prepare(_request(tmp_path, rows))


def test_a_null_image_is_a_text_row(tmp_path):
    rows = [{"prompt": f"q{i}", "completion": "a", "image": None} for i in range(40)]
    result = prepare(_request(tmp_path, rows))
    assert result.images is None
    assert not any("ignores" in text for text in result.limitations)


def test_several_pictures_in_one_row_are_refused(tmp_path):
    rows = _rows(tmp_path)
    rows[0]["image"] = [rows[0]["image"], rows[1]["image"]]
    with pytest.raises(PrepareError, match=r":1: 'image' gives 2 pictures"):
        prepare(_request(tmp_path, rows))


def test_an_image_that_is_not_a_path_is_refused(tmp_path):
    rows = _rows(tmp_path)
    rows[2]["image"] = 7
    with pytest.raises(PrepareError, match=r":3: 'image' must be the path of an image file"):
        prepare(_request(tmp_path, rows))


def test_a_missing_picture_is_refused(tmp_path):
    rows = _rows(tmp_path)
    rows[3]["image"] = "pics/nowhere.jpg"
    with pytest.raises(PrepareError, match=r":4: the image 'pics/nowhere.jpg' does not exist"):
        prepare(_request(tmp_path, rows))


def test_a_picture_pillow_cannot_decode_is_refused(tmp_path):
    rows = _rows(tmp_path)
    (tmp_path / "pics" / "broken.jpg").write_bytes(b"not an image")
    rows[4]["image"] = "pics/broken.jpg"
    with pytest.raises(PrepareError, match=r":5: .*broken.jpg is not an image Pillow"):
        prepare(_request(tmp_path, rows))


def test_a_gif_is_refused(tmp_path):
    rows = _rows(tmp_path)
    _fake_image(tmp_path / "pics" / "anim.gif", fmt="GIF")
    rows[0]["image"] = "pics/anim.gif"
    with pytest.raises(PrepareError, match=r"not PNG or JPEG"):
        prepare(_request(tmp_path, rows))


def test_a_tool_call_target_with_a_picture_is_refused(tmp_path):
    rows = _rows(tmp_path)
    # A completion given beside it: the call is not rendered, and still refused.
    rows[0]["target"] = {"name": "set_label", "args": {"label": "healthy"}}
    with pytest.raises(PrepareError, match=r":1: this row's target is a tool call"):
        prepare(_request(tmp_path, rows))


def test_declarations_with_pictures_are_refused(tmp_path):
    declarations = tmp_path / "tools.json"
    declarations.write_text("[]", encoding="utf-8")
    with pytest.raises(PrepareError, match=r"--declarations was given"):
        prepare(_request(tmp_path, declarations=declarations))


def test_labelled_rows_with_pictures_are_refused_as_decisions(tmp_path):
    rows = [{k: v for k, v in r.items() if k != "completion"} for r in _rows(tmp_path)]
    for row, label in zip(rows, LABELS * 12, strict=False):
        row["label"] = label
    with pytest.raises(PrepareError, match=r"are read as decisions"):
        prepare(_request(tmp_path, rows))


def test_prerendered_prompts_with_pictures_are_refused(tmp_path):
    rows = _rows(tmp_path)
    for row in rows:
        row["prompt"] = "<|turn>user\n" + row["prompt"] + "<turn|>\n<|turn>model\n"
    with pytest.raises(PrepareError, match=r"already carry turn markers"):
        prepare(_request(tmp_path, rows))


def test_another_family_ignores_an_image_and_says_so(tmp_path):
    rows = [{"prompt": f"q{i}", "completion": "a", "image": "pics/x.png"} for i in range(40)]
    result = prepare(_request(tmp_path, rows, base_model="Qwen/Qwen3-0.6B", images=None))
    assert result.images is None
    assert result.outcome is not Outcome.FAILED
    said = IMAGES_IGNORED.format(count=40, model="Qwen/Qwen3-0.6B")
    assert said in result.limitations
    # The split is the text it always was: no picture, no digest.
    assert all("image_sha256" not in r for r in _records(tmp_path / "split" / "train.jsonl"))


def test_no_model_with_pictures_is_refused(tmp_path):
    rows = [{"prompt": f"q{i}", "completion": "a", "image": "pics/x.png"} for i in range(40)]
    with pytest.raises(PrepareError, match=r"40 row\(s\) carry an 'image', and no --base-model"):
        prepare(_request(tmp_path, rows, base_model=None, images=None))


@pytest.mark.parametrize("model", ["google/gemma-4-E2B", "someone/gemma-4-e2b-it-mirror"])
def test_another_id_of_the_image_family_is_refused_naming_the_one(tmp_path, model):
    rows = _rows(tmp_path)
    with pytest.raises(PrepareError, match=r"pass --base-model google/gemma-4-E2B-it"):
        prepare(_request(tmp_path, rows, base_model=model, images=None))


def test_a_text_split_for_gemma_4_is_unchanged(tmp_path):
    rows = [{"prompt": f"q{i}", "completion": "a", "label": "x"} for i in range(40)]
    result = prepare(_request(tmp_path, rows, images=None))
    assert result.images is None
    assert LLM_IMAGE_LENGTHS not in result.limitations


def test_rows_of_one_picture_are_kept_on_one_side_of_the_split(tmp_path):
    rows = []
    for p in range(12):
        _fake_image(tmp_path / "pics" / f"p{p}.jpg", fmt="JPEG", tag=f"picture {p}")
        rows += [
            {"prompt": f"question {q}", "completion": "a", "image": f"pics/p{p}.jpg"}
            for q in range(4)
        ]
    result = prepare(_request(tmp_path, rows, heldout_size=8))
    train = {r["image_sha256"] for r in _records(tmp_path / "split" / "train.jsonl")}
    held = {r["image_sha256"] for r in _records(tmp_path / "split" / "heldout.jsonl")}
    assert train and held and not (train & held)
    assert result.heldout is not None and result.heldout.n % 4 == 0


def test_the_lengths_count_each_picture(tmp_path):
    class Counter:
        def describe(self):
            return {"counter": "fake"}

        def count(self, texts):
            return [1 for _ in texts]

    result = prepare(_request(tmp_path, tokens=Counter(), context_length=200))
    # One token of text and the picture's 282: every row is over a 200-token window.
    assert result.lengths.total["max"] > 200
    assert LLM_IMAGES_READ in result.limitations


def test_an_image_that_is_a_directory_is_refused(tmp_path):
    rows = _rows(tmp_path)
    (tmp_path / "pics" / "folder.jpg").mkdir()
    rows[1]["image"] = "pics/folder.jpg"
    with pytest.raises(
        PrepareError, match=r":2: the image 'pics/folder.jpg' is not a regular file"
    ):
        prepare(_request(tmp_path, rows))


class _Counter:
    """One token per text."""

    def __init__(self):
        self.calls = 0

    def describe(self):
        return {"counter": "fake"}

    def count(self, texts):
        self.calls += 1
        return [1 for _ in texts]


def test_an_over_length_split_never_opens_its_pictures(tmp_path):
    inspector, counter = FakeImageInspector(), _Counter()
    result = prepare(_request(tmp_path, tokens=counter, context_length=200, images=inspector))
    # Every row is one token and a 282-token picture: over 200, refused unopened.
    assert inspector.calls == []
    # Measured once: the length check reports the measurement taken before.
    assert counter.calls == 1
    assert result.train is None
    lengths = next(c for c in result.checks.checks if c.name == LENGTH_CHECK)
    assert lengths.outcome is Outcome.FAILED


def test_a_split_that_fits_opens_its_pictures_once(tmp_path):
    inspector, counter = FakeImageInspector(), _Counter()
    result = prepare(_request(tmp_path, tokens=counter, context_length=4096, images=inspector))
    assert len(inspector.calls) == 1 and result.train is not None
    assert LLM_IMAGE_LENGTHS in result.limitations
    # The lengths measured before the pictures are opened are the ones checked after.
    assert counter.calls == 1


def test_without_a_tokenizer_no_length_is_claimed_for_the_pictures(tmp_path):
    result = prepare(_request(tmp_path))
    assert LLM_IMAGE_LENGTHS not in result.limitations


def test_a_clamped_grouped_split_reports_its_sizes_once(tmp_path):
    rows = []
    for p in range(12):
        _fake_image(tmp_path / "pics" / f"p{p}.jpg", fmt="JPEG", tag=f"picture {p}")
        rows += [
            {"prompt": f"question {q}", "completion": "a", "image": f"pics/p{p}.jpg"}
            for q in range(4)
        ]
    result = prepare(_request(tmp_path, rows, heldout_size=100))
    sizes = [t for t in result.limitations if "held out" in t]
    assert len(sizes) == 1
    assert result.heldout is not None and result.train is not None
    held, train = result.heldout.n, result.train.n
    assert (held, train) == (44, 4)
    assert f"{held} rows were held out" in sizes[0] and f"{train} kept for training" in sizes[0]
    assert "where 100 and 0 were asked" in sizes[0]


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda rows: rows[3].pop("image"), r":4: this row gives no picture"),
        (lambda rows: rows[3].update(image=["a.jpg", "b.jpg"]), r":4: 'image' gives 2 pictures"),
        (lambda rows: rows[3].update(image=7), r":4: 'image' must be the path"),
        (lambda rows: rows[3].update(image="pics/gone.jpg"), r":4: the image 'pics/gone.jpg'"),
        (
            lambda rows: rows[3].update(target={"name": "set", "args": {}}),
            r":4: this row's target is a tool call",
        ),
    ],
)
def test_a_malformed_split_is_refused_even_when_a_row_is_over_length(tmp_path, change, said):
    rows = _rows(tmp_path)
    change(rows)
    inspector = FakeImageInspector()
    with pytest.raises(PrepareError, match=said):
        prepare(_request(tmp_path, rows, tokens=_Counter(), context_length=200, images=inspector))
    assert inspector.calls == []
