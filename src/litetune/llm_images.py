"""Gemma 4 E2B fine-tuned on rows that carry a picture: what every stage shares.

A row is `{"prompt": <text>, "completion": <text>, "image": <path>}`, one user
turn whose content is the picture and then the text -- the order
flutter_edge_ai gives Gemma 4 (`multimodal_order.dart:35-46` at flutter_edge_ai
`503bc612`, images before the text) and the message shape its LiteRT-LM client
builds (`litert_lm_client.dart:2219-2245`: `{"type": "image", "blob":
base64(bytes)}` then `{"type": "text", ...}`, the bytes the caller gave).

**`prepare`** copies each picture into the split byte for byte, named by its
sha256, as it does for EmbeddingGemma 2's image rows (`images`), and writes the
digest into the row.

**`tune`** trains in `envs.TRAIN_VISION`. The prompt is rendered with the
template the converted bundle carries, not the checkpoint's: the checkpoint's
`chat_template.jinja` renders an image part as `<|image|>` and Google's bundle
template as `\\n\\n<|image|>\\n\\n` (a spike outside litetune, 2026-11-05, rendered
both; the runtime renders the bundle's). So `tune` fetches the template the
family's export rule names (`models._gemma4`, `--jinja_chat_template_override`),
trains on it, keeps it beside the checkpoint as `TEMPLATE_FILE`, and `convert`
hands that file to the exporter. The image goes through the checkpoint's own
processor at `VISION_TOKENS` soft tokens. Only the answer is supervised.

**`convert`** exports with litert-torch's `image_text_to_text` task and the
vision encoder (`EXPORT_FLAGS`), in `envs.EXPORT_VISION`, and reads the bundle
back: it must carry `VISION_SECTIONS`, `MAX_NUM_PATCHES`, and the template.

**`verify`** gives the reference the picture through the processor and the
same template, and the runtime the picture's bytes in the app's message shape
with a CPU vision backend.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from litetune import models

# The family `models.RULES` calls google/gemma-4-E2B-it, and the one model this
# path trains: the family and size litetune trains on text, and the one the
# spike above ran.
IMAGE_FAMILY = "gemma-4-e2b"
IMAGE_MODEL = "google/gemma-4-E2B-it"

# Soft tokens per picture, at most. `processor_config.json` at
# google/gemma-4-E2B-it 3e22461f gives `max_soft_tokens` 280; Google's bundle
# reports `Max Vision Token Budget: 280` (`litert-lm describe`, MEASUREMENTS.md
# *Gemma 4 E2B without its towers* bundle); and flutter_edge_ai_litertlm sets
# `max_num_images` and no vision-token budget (`litert_lm_client.dart:896-900`),
# so the bundle's applies. litert-torch's own default is 140
# (`model_ext/gemma4/vision_exportable.py:78, 118`), which the export overrides.
VISION_TOKENS = 280
# Gemma 4 pools 3x3 patches into one soft token (`pooling_kernel_size` 3), and
# the bundle's metadata gives the budget as patches (`max_num_patches`).
PATCHES_PER_SOFT_TOKEN = 9
MAX_NUM_PATCHES = VISION_TOKENS * PATCHES_PER_SOFT_TOKEN

# The placeholder both templates render for an image part.
IMAGE_TOKEN = "<|image|>"

# The sections a bundle that takes pictures carries, as `litert-lm-builder
# unpack` names their `model_type` (`tf_lite_` dropped): what litert-torch's
# vision export wrote in the spike, and what Google's E2B bundle carries.
VISION_SECTIONS = ("vision_encoder", "vision_adapter", "end_of_vision")

# The processor files `tune` records the digests of and `verify` checks: what
# the reference's `AutoProcessor` reads from the checkpoint.
PROCESSOR_FILES = (
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)

# The tensors a LoRA run on the text tower must leave as the base's, by their
# `state_dict` names in transformers 5.16.1's `Gemma4ForConditionalGeneration`:
# the vision and audio towers and their embedders, which sit beside
# `language_model` (`modeling_gemma4.py:2176-2190`), and the token embeddings,
# which hold the `<|image>` and `<image|>` rows and which litetune adapts no
# projection of. `tune` hashes each before training and after the merge.
FROZEN_PREFIXES = (
    "model.vision_tower.",
    "model.embed_vision.",
    "model.audio_tower.",
    "model.embed_audio.",
    "model.language_model.embed_tokens.",
    "model.language_model.embed_tokens_per_layer.",
)

# The bundle template `tune` trained with, kept beside the checkpoint.
TEMPLATE_FILE = "litetune_bundle_template.jinja"

# `litetune.json`'s key for an image checkpoint.
RECORD_KEY = "images"

# What `convert` adds for an image checkpoint, beside the template override.
EXPORT_FLAGS = (
    "--task=image_text_to_text",
    "--export_vision_encoder=True",
    f"--gemma4_vision_max_soft_tokens={VISION_TOKENS}",
)

# The sampler such a bundle declares is not litetune's choice. With the
# `image_text_to_text` task litert-torch keeps the checkpoint's `do_sample` (it
# sets it to False for `text_generation` only, `core/export_lib.py:357-359`), and
# from a sampling generation config writes a sampler into the bundle
# (`core/litert_lm_builder.py:274-301`): TOP_P, k 64, p 0.95, temperature 1.0 on
# the spike's export, where a litetune text export carries none. The one way the
# exporter offers to make it greedy, a top_k of 1, writes the GREEDY type
# (`:300-301`), and litert-lm 0.18.0 then refused every message sent with no
# sampler of the caller's: `RuntimeError: UNIMPLEMENTED: Sampler type: 3 not
# implemented yet.` on 300 of 300 held-out rows (litetune's own beans run,
# 2026-11-05). So the bundle keeps the sampler the exporter writes, `convert`
# records it, and `verify` sends the candidate this sampler of its own -- the one
# the spike measured with, top_k 1 -- in place of the bundle's.
CANDIDATE_SAMPLER = {"top_k": 1, "top_p": 1.0, "temperature": 1.0, "seed": 0}

SAMPLER_NOTE = (
    "the bundle declares the sampler litert-torch writes from the checkpoint's generation "
    "config ({sampler}); an application that passes no sampler of its own samples with it. "
    "litetune's measurements of it send top_k 1 instead"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


class ImageRowError(ValueError):
    """A split's image rows cannot be read as the split `prepare` wrote."""


def is_image_model(model: str | None) -> bool:
    """Whether `model` (an id, or a checkpoint that records one) is the image family."""
    if not model:
        return False
    rules = models.identify(model)
    return rules is not None and rules.family == IMAGE_FAMILY


def has_image(obj: Mapping[str, Any]) -> bool:
    """Whether a row gives a picture: an `image` that is not null."""
    return obj.get("image") is not None


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImageRef:
    """A split row's picture: as the row names it, its digest, and the file it resolves to."""

    relative: str
    sha256: str
    file: Path

    def as_dict(self) -> dict[str, Any]:
        return {"image": self.relative, "image_sha256": self.sha256}


def split_images(path: Path) -> tuple[ImageRef, ...] | None:
    """Each row's picture, in row order, checked against its digest; `None` for text rows.

    The one reader of an image split's pictures, for `tune` and for `verify`.
    Refused, naming the row: a split mixing rows with and without a picture, an
    `image` that is not a relative path inside the split's directory once every
    link is resolved, a file that is not there, and bytes whose sha256 is not
    the row's `image_sha256` -- the picture changed after `prepare` wrote it.
    """
    path = Path(path)
    root = path.parent.resolve()
    objects: list[tuple[int, dict[str, Any]]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ImageRowError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if isinstance(obj, dict):
            objects.append((lineno, obj))
    with_image = [lineno for lineno, obj in objects if has_image(obj)]
    if not with_image:
        return None
    if len(with_image) != len(objects):
        without = next(lineno for lineno, obj in objects if not has_image(obj))
        raise ImageRowError(
            f"{path}:{without}: this row gives no picture, and line {with_image[0]} does. A split "
            "is all image rows or all text rows; `prepare` writes one of the two"
        )
    digests: dict[Path, str] = {}
    out: list[ImageRef] = []
    for lineno, obj in objects:
        where = f"{path}:{lineno}"
        value, expected = obj.get("image"), obj.get("image_sha256")
        if not isinstance(value, str) or not value:
            raise ImageRowError(f"{where}: 'image' must be the picture's path in the split")
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ImageRowError(
                f"{where}: 'image_sha256' is {expected!r}, not a sha256. Use the split `prepare` "
                "wrote, whose rows carry the digest of their picture"
            )
        relative = PurePosixPath(value)
        # `prepare` writes forward-slash paths relative to the split. A drive, an
        # anchor or a backslash is refused on every host, as `embedding_tune` does.
        absolute = relative.is_absolute() or bool(PureWindowsPath(value).anchor) or "\\" in value
        file = (root / relative).resolve()
        if absolute or not file.is_relative_to(root):
            raise ImageRowError(
                f"{where}: the image {value!r} is not inside the split's directory {root}; "
                "`prepare` copies every picture into it"
            )
        if not file.is_file():
            raise ImageRowError(f"{where}: the image {value!r} does not exist ({file})")
        if file not in digests:
            digests[file] = sha256_hex(file)
        if digests[file] != expected:
            raise ImageRowError(
                f"{where}: the image {value!r} hashes to {digests[file]}, not the {expected} its "
                "row names; the picture changed after `prepare` wrote the split. Run prepare again"
            )
        out.append(ImageRef(value, expected, file))
    return tuple(out)


def processor_digests(model_dir: Path) -> dict[str, str | None]:
    """The sha256 of each `PROCESSOR_FILES` file in `model_dir`; `None` for one not there."""
    out: dict[str, str | None] = {}
    for name in PROCESSOR_FILES:
        file = Path(model_dir) / name
        out[name] = sha256_hex(file) if file.is_file() else None
    return out


def image_record(reference: str | Path) -> dict[str, Any] | None:
    """The `images` record of a checkpoint `tune` trained on pictures; `None` for any other.

    Raises `ImageRowError` for a record that names one and is not one this reads,
    and lets `models.read_training_record`'s own errors through: an unreadable
    `litetune.json` is not a checkpoint without pictures.
    """
    read = models.read_training_record(str(reference))
    if read is None:
        return None
    record = read[0].get(RECORD_KEY)
    if record is None:
        return None
    problem = record_problem(record)
    if problem is not None:
        raise ImageRowError(f"{read[1]}: {problem}")
    return dict(record)


def record_problem(record: Any) -> str | None:
    """Why `record` is not an `images` record `tune` writes, or `None`."""
    if not isinstance(record, Mapping):
        return f"'{RECORD_KEY}' is {record!r}, not an object"
    if record.get("vision_tokens") != VISION_TOKENS:
        return (
            f"'{RECORD_KEY}.vision_tokens' is {record.get('vision_tokens')!r}; this litetune "
            f"trains and exports pictures at {VISION_TOKENS}"
        )
    template = record.get("template_sha256")
    if not isinstance(template, str) or not _SHA256.fullmatch(template):
        return f"'{RECORD_KEY}.template_sha256' is {template!r}, not a sha256"
    digests = record.get("processor_sha256")
    if not isinstance(digests, Mapping) or set(digests) != set(PROCESSOR_FILES):
        return (
            f"'{RECORD_KEY}.processor_sha256' is {digests!r}; it gives the sha256 of each of "
            f"{', '.join(PROCESSOR_FILES)}"
        )
    for name, digest in digests.items():
        if not (digest is None or (isinstance(digest, str) and _SHA256.fullmatch(digest))):
            return f"'{RECORD_KEY}.processor_sha256' gives {name} {digest!r}, not a sha256"
    return None


def template_file(model_dir: str | Path) -> Path:
    """Where `tune` kept the bundle template it trained with."""
    return Path(model_dir) / TEMPLATE_FILE


def checked_template(model_dir: str | Path, record: Mapping[str, Any]) -> str:
    """The template `tune` trained with, refused unless its bytes are the recorded ones."""
    file = template_file(model_dir)
    if not file.is_file():
        raise ImageRowError(
            f"{file} is not there: `tune` keeps the bundle template it trained the pictures on "
            "beside the checkpoint, and without it nothing says which one the bundle must carry"
        )
    found = sha256_hex(file)
    if found != record["template_sha256"]:
        raise ImageRowError(
            f"{file} hashes to {found}, and the checkpoint records "
            f"{record['template_sha256']}: the template changed after `tune` trained with it"
        )
    return file.read_text(encoding="utf-8")


def processor_mismatch(model_dir: str | Path, record: Mapping[str, Any]) -> str | None:
    """Which processor file differs from the one `tune` recorded, said as a refusal; or `None`."""
    found = processor_digests(Path(model_dir))
    recorded = record["processor_sha256"]
    for name in PROCESSOR_FILES:
        if found[name] != recorded[name]:
            return (
                f"{Path(model_dir) / name} hashes to {found[name]}, and the checkpoint records "
                f"{recorded[name]}: the processor the pictures were trained through is not the "
                "one here"
            )
    return None
