"""`tune` for google/embeddinggemma-2: a decision's rows, as an embedding model.

`litetune tune --model google/embeddinggemma-2 --decision <decision.json>` lands
here instead of `tune.run_tune` (`cli._tune`). The model is fine-tuned so that
flutter_edge_ai's `EmbeddingModel.createDecider`, which embeds every input with
`DECIDER_PREFIX` (`flutter_edge_ai/lib/src/decision/embedding_decider.dart:41`,
`deciderTaskType`), separates the decision's options better. Nothing but the
SentenceTransformer is kept: `convert` grafts its text tower into Google's
`.litertlm` (`embedding_convert`).

**The recipe is the measured one, and takes no overrides** (`RECIPE`). It is
the one measured (MEASUREMENTS.md, "EmbeddingGemma 2: many labels"), which is
the graft measurement's (MEASUREMENTS.md, "EmbeddingGemma 2: the graft"): the whole model trained
with a Linear(768,256)-GELU-Linear(256,K) head on the sentence embedding,
cross-entropy + Brier, AdamW at 2e-5 (head 1e-3) with no weight decay, batch 8,
2 epochs, 10% linear warmup then linear decay, seed 0, at most 128 tokens,
float32, every text prefixed with the sentence-transformers prompt
`SentenceSimilarity`. The head is discarded. Nothing here is a parameter: this
is the recipe that was measured, and s2's soft-margin triplet loss
collapsed (the design, "Training recipe"). float32 on
every device: float16 is documented to give NaN embeddings (Google's model card
and Unsloth, as the design cites them), and bfloat16 was not measured.

**Data.** `prepare`'s decision split: each row's prompt is the state rendered
with the decision (`Decision.render`), and its completion is the option id. The
state is cut back out and is the text embedded; the id is the label. Option
descriptions are not trained on.

**What is recorded.** `litetune.json` beside the checkpoint, with
`"direction": "embedding"` (`EmbeddingRecord`), which `convert` and `verify`
dispatch on. The head's accuracy on the held-out split is in it, and is not a
measurement of the decider the application builds: that fits centroids on the
embeddings, without the head.

**Images.** A split whose rows name images (`prepare` writes them with each
picture copied into the split and its sha256 in the row) takes the image
recipe instead (`IMAGE_RECIPE`, docs/specs/2026-11-04-eg2-image-tune.md):
the model with its vision tower and no audio tower, the vision tower,
`embed_vision` and the token embeddings frozen, each picture's input computed
once at `--vision-tokens` soft tokens and cached in the output directory, a
Linear(768, K) head on the mean pool. `--mirror` adds a flipped copy of every
training picture, `--mirror-swap` exchanges two labels on the copies. The full
HF checkpoint is saved, as one `model.safetensors` convert grafts from, and
the record says `modality: "image"` with the vision tokens and the mirror.

The training runs in `envs.EMBED_TRAIN` as a generated script; this process
imports neither torch nor sentence-transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from litetune import devices, envs, models
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.decisions import Decision, DecisionError, read_questions
from litetune.embedding_convert import BASE_MODEL, BASE_REVISION
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.prepare import PrepareError, Row, read_objects, read_rows, rows_are_images
from litetune.storage import hash_file
from litetune.tune import (
    _GPU_FAILURE_RE,
    DEFAULT_TIMEOUT_S,
    DEVICE_CHECK,
    ENV_CHECK,
    TRAINING_CHECK,
    _mps_advice,
    _tail,
)

logger = logging.getLogger(__name__)

EMBEDDING_TUNE_SCHEMA = "litetune.tune-embedding/1"
DIRECTION = "embedding"

# The one model this path trains, at the revision the recipe and `convert`'s
# mapping were established on.
EMBEDDING_MODEL = BASE_MODEL
EMBEDDING_REVISION = BASE_REVISION
# `model_type` in google/embeddinggemma-2's config.json at that revision.
EMBEDDING_MODEL_TYPE = "embedding_gemma2"
# The sentence-transformers prompt the recipe trains with, and its text, which
# is flutter_edge_ai's `deciderTaskType` (embedding_decider.dart:41). The
# script refuses a checkpoint whose prompt of that name is another text.
PROMPT_NAME = "SentenceSimilarity"
DECIDER_PREFIX = "task: sentence similarity | query: "

# What an embedding checkpoint was tuned on: `EmbeddingRecord.modality`. A record
# without the field was written before image tuning existed and is a text one.
MODALITIES: tuple[str, ...] = ("text", "image")
DEFAULT_MODALITY = "text"
# An image checkpoint trains on the processor's own image input, with no task
# prompt in front of it (docs/specs/2026-11-04-eg2-image-tune.md, "tune (image)").
IMAGE_PREFIX = ""
# `EmbeddingRecord.vision_tokens`: the processor's `max_soft_tokens` per image the
# checkpoint was tuned at, which the app gives the runtime as its vision tokens per
# image. The two values the image recipe admits (the spec's `--vision-tokens`).
VISION_TOKENS: tuple[int, ...] = (70, 140)

# The one device the recipe was measured on.
MEASURED_DEVICE = "cpu"
# The devices the scratch runs the image recipe comes from trained on (zn3 and
# dg2 on mps, dg3 on cuda); there is no litetune measurement of it on any.
IMAGE_RECIPE_DEVICES: tuple[str, ...] = ("mps", "cuda")

MODEL_CHECK = "embedding model"
DATA_CHECK = "decision rows"

# The script's exit code for a model it refused after reading its config.
REFUSED_EXIT = 3

NOT_VERIFIED = (
    "training completed and nothing about the decider has been measured. The head's held-out "
    "accuracy is the discarded head's, not the centroid decider's the application builds from "
    "the embeddings; `litetune verify` measures that"
)
# Refused, not trained as one more label: the recipe and the decider were
# measured on decisions without one, and flutter_edge_ai's embedding decider
# abstains by a threshold on the similarity, not by an option.
NONE_OPTION_REFUSED = (
    "the decision has a none_option ({none!r}). The embedding recipe was measured on decisions "
    "without one, and flutter_edge_ai's embedding decider abstains by a threshold, not by an "
    "option, so a none_option trained as one more label is a decider nobody has measured. Leave "
    "the none_option out of the decision, and its rows out of the split"
)

_STDOUT_TAIL = 4000


class EmbeddingTuneError(ValueError):
    """A request the embedding path will not run. Not a statement about any model."""


@dataclass(frozen=True)
class Recipe:
    """The measured recipe (`train_s5.py`), as the script is given it."""

    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 8
    epochs: int = 2
    warmup_fraction: float = 0.1
    seed: int = 0
    max_seq_length: int = 128
    embedding_dim: int = 768
    head_hidden: int = 256
    eval_batch_size: int = 64
    dtype: str = "float32"
    loss: str = "cross_entropy + brier"

    @property
    def schedule(self) -> str:
        """The learning-rate schedule in words, from the fraction the script is given."""
        return (
            f"linear warmup over the first {self.warmup_fraction:.0%} of steps, "
            "then linear decay to 0"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "head_learning_rate": self.head_learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_fraction": self.warmup_fraction,
            "seed": self.seed,
            "max_seq_length": self.max_seq_length,
            "embedding_dim": self.embedding_dim,
            "head_hidden": self.head_hidden,
            "eval_batch_size": self.eval_batch_size,
            "dtype": self.dtype,
            "loss": self.loss,
            "schedule": self.schedule,
        }


RECIPE = Recipe()


@dataclass(frozen=True)
class ImageRecipe:
    """The image recipe, as the script is given it (docs/specs/2026-11-04-eg2-image-tune.md,
    "tune (image)"): the scratch runs dg2's and dg3's (zn3's at 3 epochs), which
    trained several heads at once; here one decision is one head.

    A Linear(768, K) head on the L2-normalised mean pool of every token, its logits
    times `logit_scale`, class-weighted cross-entropy (`class_weighting`), AdamW
    with no schedule, the gradient of the language model clipped. The image
    inputs are computed once with the vision tower frozen and kept in float16.
    """

    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 2
    seed: int = 0
    embedding_dim: int = 768
    logit_scale: float = 20.0
    grad_clip: float = 1.0
    eval_batch_size: int = 32
    dtype: str = "float32"
    cache_dtype: str = "float16"
    loss: str = "class-weighted cross-entropy"
    class_weighting: str = "n / (K * count) over the training items, mirrored copies included"
    schedule: str = "constant"
    pool: str = "mean over every token, then L2"
    frozen: tuple[str, ...] = ("vision_tower", "embed_vision", "language_model.embed_tokens")

    def as_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "head_learning_rate": self.head_learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "seed": self.seed,
            "embedding_dim": self.embedding_dim,
            "logit_scale": self.logit_scale,
            "grad_clip": self.grad_clip,
            "eval_batch_size": self.eval_batch_size,
            "dtype": self.dtype,
            "cache_dtype": self.cache_dtype,
            "loss": self.loss,
            "class_weighting": self.class_weighting,
            "schedule": self.schedule,
            "pool": self.pool,
            "frozen": list(self.frozen),
        }


IMAGE_RECIPE = ImageRecipe()

# The text the processor is given for one image; it expands it into the image's
# soft tokens between its begin and end markers (the scratch runs' `common.inputs`).
IMAGE_TOKEN_TEXT = "<|image|>"
# Under the output directory: the image inputs, computed once per picture.
IMAGE_CACHE_DIR = "embed_image_inputs"
# The image script's name in the output directory, beside the text one's.
IMAGE_SCRIPT_NAME = "embed_image_train_script.py"
# `save_pretrained` splits a checkpoint larger than `max_shard_size` into
# shards, and convert reads one `model.safetensors` (`embedding_export`). The
# float32 checkpoint is 438.8M parameters, about 1.76 GB, far below this.
IMAGE_MAX_SHARD_SIZE = "1000GB"
# The one file of the checkpoint convert grafts from: `embedding_export.BASE_FILE`
# (not imported: that module imports this one).
BASE_CHECKPOINT_FILE = "model.safetensors"
# An image record's `prompt_name`: the image input has no sentence-transformers prompt.
IMAGE_PROMPT_NAME = ""
# Where an application sets the vision tokens per image (flutter_edge_ai).
VISION_TOKENS_SETTING = "LiteRtEmbeddingBackend(visionTokensPerImage: {n})"

IMAGE_RECIPE_UNMEASURED = (
    "the image recipe has no litetune measurement yet: it comes from runs that trained on one "
    "task family -- game frames -- on Apple's GPU (mps) and on CUDA, several heads at once "
    "where this trains one. Whether it helps on other pictures is not known; `litetune verify` "
    "measures this checkpoint's decider"
)
IMAGE_MIRROR_ASSUMES_SYMMETRY = (
    "--mirror trained on a horizontally flipped copy of every training image{swapped}. That "
    "assumes a mirrored scene means what the original means: a picture with text, a HUD, or "
    "light from one side is not its own mirror image, and the model was taught pictures the "
    "application never shows it"
)


def is_embedding_model(model: str) -> bool:
    """Whether `tune` takes the embedding path for `model`: the exact id, and only it."""
    return model == EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# The script that runs inside envs.EMBED_TRAIN
# ---------------------------------------------------------------------------
#
# Written beside the checkpoint and kept, as `tune` keeps its own: a checkpoint
# whose training script is gone cannot be reproduced. Its helpers above `main`
# import nothing, so the tests run them in this process.

_EMBED_TRAIN_SCRIPT = (
    r'''
"""Fine-tune google/embeddinggemma-2 for a decision with the measured recipe.

Reads a spec JSON from argv[1]; saves the SentenceTransformer to spec["model_dir"]
and writes metrics JSON to spec["metrics_out"]. A model it refuses is written as
{"refused": <why>} and exits REFUSED_EXIT.
"""
import json
import math
import random
import sys
import time
from pathlib import Path

REFUSED_EXIT = 3
'''
    + devices.MPS_SCRIPT_SOURCE
    + r'''

def model_type_refusal(config, expected):
    """Why a config.json is not the model this recipe was measured on, or None."""
    kind = config.get("model_type") if isinstance(config, dict) else None
    if kind != expected:
        return (
            f"config.json declares model_type {kind!r}, not {expected!r}; the recipe was "
            "measured on EmbeddingGemma 2 only"
        )
    return None


def prefix_refusal(prompts, name, expected):
    """Why the checkpoint's prompt `name` is not the decider's prefix, or None."""
    found = prompts.get(name) if isinstance(prompts, dict) else None
    if found != expected:
        return (
            f"the checkpoint's sentence-transformers prompt {name!r} is {found!r}, not "
            f"{expected!r}, the prefix the application's decider embeds with"
        )
    return None


def plan(n_rows, batch_size, epochs, warmup_fraction):
    """(optimiser steps, warmup steps), as train_s5.py counts them."""
    steps = epochs * math.ceil(n_rows / batch_size)
    return steps, max(1, int(warmup_fraction * steps))


def lr_factor(step, warm, steps):
    """The multiplier on each group's rate at `step`: linear warmup, then linear decay."""
    if step < warm:
        return (step + 1) / warm
    return max(0.0, (steps - step) / max(1, steps - warm))


def epoch_batches(n_rows, batch_size, shuffle):
    """One epoch's batches of row indices, in the order `shuffle` leaves them."""
    order = list(range(n_rows))
    shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n_rows, batch_size)]


def accuracy(predicted, gold):
    return round(sum(p == g for p, g in zip(predicted, gold)) / len(gold), 4)


def training_device(torch, given=None):
    """What the parent resolved; without an answer, CUDA if any, else the CPU (never mps)."""
    if given is not None:
        return given
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def refuse(out, reason):
    Path(out).write_text(json.dumps({"refused": reason}), encoding="utf-8")
    print(reason, file=sys.stderr)
    return REFUSED_EXIT


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = spec["metrics_out"]
    recipe = spec["recipe"]

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(spec["model"], "config.json", revision=spec["revision"])
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    refusal = model_type_refusal(config, spec["model_type"])
    if refusal is not None:
        return refuse(out, refusal)

    import importlib.metadata

    import torch
    from sentence_transformers import SentenceTransformer

    # Seeded before the load, as train_s5.py does, so the head is initialised
    # from the same generator state.
    torch.manual_seed(recipe["seed"])
    random.seed(recipe["seed"])
    if recipe["dtype"] != "float32":
        return refuse(out, f"the recipe trains in float32, the spec says {recipe['dtype']!r}")
    device = training_device(torch, spec.get("device"))
    st = SentenceTransformer(
        spec["model"],
        revision=spec["revision"],
        device=device,
        model_kwargs={"dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    st.max_seq_length = recipe["max_seq_length"]
    refusal = prefix_refusal(st.prompts, spec["prompt_name"], spec["prefix"])
    if refusal is not None:
        return refuse(out, refusal)
    prefix = spec["prefix"]

    rows = read_jsonl(spec["train_rows"])
    heldout = read_jsonl(spec["heldout_rows"])
    labels = list(spec["labels"])
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    head = torch.nn.Sequential(
        torch.nn.Linear(recipe["embedding_dim"], recipe["head_hidden"]),
        torch.nn.GELU(),
        torch.nn.Linear(recipe["head_hidden"], k),
    ).to(device)

    def embed(texts):
        features = st.tokenize([prefix + t for t in texts])
        features = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in features.items()
        }
        return st(features)["sentence_embedding"]

    optimiser = torch.optim.AdamW(
        [
            {"params": st.parameters(), "lr": recipe["learning_rate"]},
            {"params": head.parameters(), "lr": recipe["head_learning_rate"]},
        ],
        weight_decay=recipe["weight_decay"],
    )
    steps, warm = plan(len(rows), recipe["batch_size"], recipe["epochs"], recipe["warmup_fraction"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: lr_factor(s, warm, steps)
    )
    log, step = [], 0
    st.train()
    head.train()
    started = time.time()
    for epoch in range(recipe["epochs"]):
        for batch in epoch_batches(len(rows), recipe["batch_size"], random.shuffle):
            chosen = [rows[j] for j in batch]
            target = torch.tensor([index[r["label"]] for r in chosen], device=device)
            logits = head(embed([r["text"] for r in chosen]))
            prob = logits.softmax(-1)
            ce = torch.nn.functional.cross_entropy(logits, target)
            brier = ((prob - torch.nn.functional.one_hot(target, k).float()) ** 2).sum(-1).mean()
            loss = ce + brier
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                log.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "ce": round(ce.item(), 4),
                        "brier": round(brier.item(), 4),
                        "seconds": round(time.time() - started, 1),
                    }
                )
                print(json.dumps(log[-1]), flush=True)
    train_seconds = round(time.time() - started, 1)

    st.eval()
    head.eval()
    predicted = []
    with torch.no_grad():
        for i in range(0, len(heldout), recipe["eval_batch_size"]):
            chosen = heldout[i : i + recipe["eval_batch_size"]]
            scores = head(embed([r["text"] for r in chosen]))
            predicted += [labels[q] for q in scores.argmax(-1).tolist()]
    head_accuracy = accuracy(predicted, [r["label"] for r in heldout])

    st.save(spec["model_dir"])
    versions = {}
    for name in ("torch", "transformers", "sentence-transformers", "pillow"):
        versions[name] = importlib.metadata.version(name)
    Path(out).write_text(
        json.dumps(
            {
                "base_model": spec["model"],
                "base_model_revision": spec["revision"],
                "prefix": prefix,
                "labels": labels,
                "rows": {"train": len(rows), "heldout": len(heldout)},
                "steps_done": step,
                "steps_planned": steps,
                "warmup_steps": warm,
                "train_seconds": train_seconds,
                "device": device,
                "dtype": "float32",
                "torch_threads": torch.get_num_threads(),
                "head_heldout_accuracy": head_accuracy,
                "log": log,
                "versions": versions,
                "mps_environment": environment_seen(),
                "saved_files": sorted(p.name for p in Path(spec["model_dir"]).iterdir()),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
)


# The image recipe's script, kept beside the checkpoint as the text one is. Its
# helpers above `main` import nothing outside the standard library, so the
# tests run them in this process.
_EMBED_IMAGE_SCRIPT = (
    r'''
"""Fine-tune google/embeddinggemma-2 on images for a decision with the image recipe.

Reads a spec JSON from argv[1]; saves the full HF checkpoint (vision tower
included, no audio tower) and the processor to spec["model_dir"] and writes
metrics JSON to spec["metrics_out"]. A model it refuses is written as
{"refused": <why>} and exits REFUSED_EXIT.

Each picture's language-model input -- its token embeddings with the frozen
vision tower's soft tokens scattered in -- is computed once and cached under
spec["cache_dir"] as float16; training and the held-out pass read the cache.
"""
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

REFUSED_EXIT = 3
'''
    + devices.MPS_SCRIPT_SOURCE
    + r'''

def model_type_refusal(config, expected):
    """Why a config.json is not the model this recipe was measured on, or None."""
    kind = config.get("model_type") if isinstance(config, dict) else None
    if kind != expected:
        return (
            f"config.json declares model_type {kind!r}, not {expected!r}; the recipe was "
            "measured on EmbeddingGemma 2 only"
        )
    return None


def training_device(torch, given=None):
    """What the parent resolved; without an answer, CUDA if any, else the CPU (never mps)."""
    if given is not None:
        return given
    return "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(predicted, gold):
    return round(sum(p == g for p, g in zip(predicted, gold)) / len(gold), 4)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def refuse(out, reason):
    Path(out).write_text(json.dumps({"refused": reason}), encoding="utf-8")
    print(reason, file=sys.stderr)
    return REFUSED_EXIT


def swap_map(pairs):
    """{label: its mirrored copy's label} for the swap pairs; a label in no pair keeps its own."""
    out = {}
    for a, b in pairs:
        if a == b or a in out or b in out:
            raise ValueError(f"the mirror swap {a}:{b} repeats a label")
        out[a], out[b] = b, a
    return out


def training_items(rows, mirror, swaps):
    """(sha256, flipped, label) for every row, then every row's mirrored copy when mirroring."""
    items = [(r["sha256"], False, r["label"]) for r in rows]
    if mirror:
        items += [(r["sha256"], True, swaps.get(r["label"], r["label"])) for r in rows]
    return items


def class_counts(targets, k):
    counts = [0] * k
    for t in targets:
        counts[t] += 1
    return counts


def class_weights(targets, k):
    """n / (k * count) per class, n the number of training items: the measured runs' weights."""
    counts = class_counts(targets, k)
    if 0 in counts:
        raise ValueError(f"class {counts.index(0)} has no training item")
    return [len(targets) / (k * c) for c in counts]


def plan(n_items, batch_size, epochs):
    """Optimiser steps: every item once an epoch, the last batch possibly short."""
    return epochs * math.ceil(n_items / batch_size)


def epoch_batches(n_items, batch_size, permutation):
    """One epoch's batches of item indices, in the order `permutation(n_items)` gives."""
    order = [int(i) for i in permutation(n_items)]
    if sorted(order) != list(range(n_items)):
        raise ValueError("the permutation does not hold every item once")
    return [order[i : i + batch_size] for i in range(0, n_items, batch_size)]


def cache_key(fields):
    """The cache directory's name: what the inputs depend on, hashed."""
    text = json.dumps(fields, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_name(sha256, flipped):
    return f"{sha256}-{'mirrored' if flipped else 'original'}.npy"


def bytes_refusal(data, expected, path):
    """Why the bytes read are not the picture the split names, or None."""
    found = hashlib.sha256(data).hexdigest()
    if found != expected:
        return (
            f"{path} hashes to {found}, not the {expected} its row names; the split changed "
            "after tune checked it"
        )
    return None


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = spec["metrics_out"]
    recipe = spec["recipe"]

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(spec["model"], "config.json", revision=spec["revision"])
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    refusal = model_type_refusal(config, spec["model_type"])
    if refusal is not None:
        return refuse(out, refusal)
    if recipe["dtype"] != "float32":
        return refuse(out, f"the recipe trains in float32, the spec says {recipe['dtype']!r}")

    import importlib.metadata
    import warnings

    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoConfig, AutoModel, AutoProcessor

    # Past Pillow's pixel limit a picture is refused, not decoded with a warning.
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    device = training_device(torch, spec.get("device"))
    # The audio tower is not loaded: nothing here uses it, and it is not saved.
    model_config = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"])
    model_config.audio_config = None
    model = AutoModel.from_pretrained(
        spec["model"], revision=spec["revision"], config=model_config, dtype=torch.float32
    ).to(device)
    processor = AutoProcessor.from_pretrained(spec["model"], revision=spec["revision"])
    missing = [
        name
        for name in ("vision_tower", "embed_vision", "language_model")
        if not hasattr(model, name)
    ]
    if missing:
        return refuse(out, f"the loaded model has no {', '.join(missing)}")
    lm = model.language_model
    for module in (model.vision_tower, model.embed_vision, lm.embed_tokens):
        module.requires_grad_(False)
    image_token = model.config.image_token_id
    pad_token = model.config.text_config.pad_token_id

    versions = {}
    for name in ("torch", "transformers", "sentence-transformers", "pillow", "numpy"):
        versions[name] = importlib.metadata.version(name)

    def merged(picture):
        """The picture's language-model input: [tokens, hidden], soft tokens scattered in."""
        enc = processor(
            text=[spec["image_text"]],
            images=[[picture]],
            return_tensors="pt",
            images_kwargs={"max_soft_tokens": spec["vision_tokens"]},
        )
        ids = enc["input_ids"].to(device)
        mask = ids == image_token
        embeds = model.get_input_embeddings()(torch.where(mask, pad_token, ids))
        features = model.get_image_features(
            enc["pixel_values"].to(device),
            enc["image_position_ids"].to(device),
            return_dict=True,
        ).pooler_output
        features = torch.cat(features, 0).to(embeds.dtype)
        if features.shape[0] != int(mask.sum()):
            raise ValueError(
                f"the vision tower gave {features.shape[0]} soft tokens for "
                f"{int(mask.sum())} image positions"
            )
        embeds = embeds.masked_scatter(mask.unsqueeze(-1).expand_as(embeds), features)
        return embeds[0].float().cpu().numpy(), int(mask.sum())

    # -- the inputs, computed once -------------------------------------------
    rows = read_jsonl(spec["train_rows"])
    heldout = read_jsonl(spec["heldout_rows"])
    labels = list(spec["labels"])
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    swaps = swap_map(spec["mirror_swap"])
    items = training_items(rows, spec["mirror"], swaps)
    heldout_items = training_items(heldout, False, {})
    paths = {r["sha256"]: r["image"] for r in rows + heldout}

    key_fields = {
        "model": spec["model"],
        "revision": spec["revision"],
        "vision_tokens": spec["vision_tokens"],
        "image_text": spec["image_text"],
        "device": device,
        "versions": versions,
    }
    cache = Path(spec["cache_dir"]) / cache_key(key_fields)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "key.json").write_text(json.dumps(key_fields, indent=1), encoding="utf-8")
    tokens, soft, computed, reused = {}, [], 0, 0
    model.eval()
    started = time.time()
    with torch.no_grad():
        for sha, flipped in sorted({(s, f) for s, f, _ in items + heldout_items}):
            target = cache / cache_name(sha, flipped)
            if target.is_file():
                array = np.load(target)
                reused += 1
            else:
                data = Path(paths[sha]).read_bytes()
                refusal = bytes_refusal(data, sha, paths[sha])
                if refusal is not None:
                    return refuse(out, refusal)
                with Image.open(io.BytesIO(data)) as image:
                    picture = image.convert("RGB")
                if flipped:
                    picture = ImageOps.mirror(picture)
                values, n_soft = merged(picture)
                array = values.astype(np.float16)
                if not np.isfinite(array).all():
                    raise ValueError(f"{paths[sha]}: its input does not fit in float16")
                soft.append(n_soft)
                partial = target.with_name(target.name + ".partial")
                with open(partial, "wb") as handle:
                    np.save(handle, array)
                os.replace(partial, target)
                computed += 1
            tokens[(sha, flipped)] = array
    precompute_seconds = round(time.time() - started, 1)
    lengths = [a.shape[0] for a in tokens.values()]

    def batch(keys):
        arrays = [tokens[key] for key in keys]
        width = max(a.shape[0] for a in arrays)
        embeds = np.zeros((len(arrays), width, arrays[0].shape[1]), np.float32)
        mask = np.zeros((len(arrays), width), np.int64)
        for j, a in enumerate(arrays):
            embeds[j, : a.shape[0]] = a
            mask[j, : a.shape[0]] = 1
        return torch.from_numpy(embeds).to(device), torch.from_numpy(mask).to(device)

    def pooled(embeds, mask):
        hidden = lm(inputs_embeds=embeds, attention_mask=mask, return_dict=True).last_hidden_state
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return torch.nn.functional.normalize((hidden * m).sum(1) / m.sum(1), dim=-1)

    # -- training ------------------------------------------------------------
    targets = [index[label] for _, _, label in items]
    weights = class_weights(targets, k)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(recipe["seed"])
    head = torch.nn.Linear(recipe["embedding_dim"], k).to(device)
    lm_params = [p for p in lm.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        [
            {"params": lm_params, "lr": recipe["learning_rate"]},
            {"params": head.parameters(), "lr": recipe["head_learning_rate"]},
        ],
        weight_decay=recipe["weight_decay"],
    )
    rng = np.random.RandomState(recipe["seed"])
    steps = plan(len(items), recipe["batch_size"], recipe["epochs"])
    log, epoch_losses, step = [], [], 0
    lm.train()
    head.train()
    started = time.time()
    for epoch in range(recipe["epochs"]):
        total = 0.0
        for chosen in epoch_batches(len(items), recipe["batch_size"], rng.permutation):
            vectors = pooled(*batch([items[j][:2] for j in chosen]))
            if vectors.shape[-1] != recipe["embedding_dim"]:
                raise ValueError(
                    f"the pool is {vectors.shape[-1]}-dimensional, the recipe's head takes "
                    f"{recipe['embedding_dim']}"
                )
            gold = torch.tensor([targets[j] for j in chosen], device=device)
            logits = head(vectors) * recipe["logit_scale"]
            loss = torch.nn.functional.cross_entropy(
                logits, gold, weight=weight_tensor, reduction="sum"
            ) / len(chosen)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm_params, recipe["grad_clip"])
            optimiser.step()
            step += 1
            total += loss.item() * len(chosen)
            if step % 50 == 0:
                log.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "loss": round(loss.item(), 4),
                        "seconds": round(time.time() - started, 1),
                    }
                )
                print(json.dumps(log[-1]), flush=True)
        epoch_losses.append(round(total / len(items), 4))
        print(json.dumps({"epoch": epoch, "loss": epoch_losses[-1]}), flush=True)
    train_seconds = round(time.time() - started, 1)

    lm.eval()
    head.eval()

    def predict(chosen_items):
        predicted = []
        with torch.no_grad():
            for i in range(0, len(chosen_items), recipe["eval_batch_size"]):
                part = chosen_items[i : i + recipe["eval_batch_size"]]
                scores = head(pooled(*batch([item[:2] for item in part])))
                predicted += [labels[q] for q in scores.argmax(-1).tolist()]
        return predicted

    originals = items[: len(rows)]
    train_accuracy = accuracy(predict(originals), [label for _, _, label in originals])
    heldout_accuracy = accuracy(
        predict(heldout_items), [label for _, _, label in heldout_items]
    )

    model_dir = Path(spec["model_dir"])
    model.save_pretrained(model_dir, max_shard_size=spec["max_shard_size"])
    processor.save_pretrained(model_dir)
    if not (model_dir / "model.safetensors").is_file():
        raise RuntimeError(
            f"save_pretrained wrote no single model.safetensors into {model_dir}: "
            f"{sorted(p.name for p in model_dir.iterdir())}"
        )
    heldout_counts = class_counts([index[label] for _, _, label in heldout_items], k)
    Path(out).write_text(
        json.dumps(
            {
                "base_model": spec["model"],
                "base_model_revision": spec["revision"],
                "prefix": spec["prefix"],
                "labels": labels,
                "rows": {"train": len(rows), "heldout": len(heldout)},
                "training_items": len(items),
                "mirror": spec["mirror"],
                "mirror_swap": [list(pair) for pair in spec["mirror_swap"]],
                "vision_tokens": spec["vision_tokens"],
                "class_counts": {
                    "train": dict(zip(labels, class_counts(targets, k))),
                    "heldout": dict(zip(labels, heldout_counts)),
                },
                "class_weights": {label: round(w, 6) for label, w in zip(labels, weights)},
                "input_tokens": {"min": min(lengths), "max": max(lengths)},
                "soft_tokens": {"min": min(soft), "max": max(soft)} if soft else None,
                "cache": {"dir": str(cache), "computed": computed, "reused": reused},
                "precompute_seconds": precompute_seconds,
                "steps_done": step,
                "steps_planned": steps,
                "train_seconds": train_seconds,
                "epoch_losses": epoch_losses,
                "device": device,
                "dtype": "float32",
                "torch_threads": torch.get_num_threads(),
                "head_train_accuracy": train_accuracy,
                "head_heldout_accuracy": heldout_accuracy,
                "log": log,
                "versions": versions,
                "mps_environment": environment_seen(),
                "saved_files": sorted(p.name for p in model_dir.iterdir()),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
)


# ---------------------------------------------------------------------------
# The rows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelledText:
    """One row as the script embeds it: the state, and the option id it is labelled with."""

    text: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "label": self.label}


def states_of(rows: Sequence[Row], decision: Decision, where: Path) -> list[LabelledText]:
    """Each row's state, cut out of the prompt `Decision.render` wrote, and its answer.

    Refuses a row that is not that decision rendered around a non-empty state,
    or whose completion is not one of its ids: the text embedded would then not
    be what the application asks about.
    """
    suffix = decision.render("")
    out: list[LabelledText] = []
    for row in rows:
        if not row.prompt.endswith(suffix) or len(row.prompt) == len(suffix):
            raise EmbeddingTuneError(
                f"{where}:{row.lineno}: the prompt is not a state rendered with the decision: it "
                "does not end with that decision's question and options. Render the rows with "
                "`prepare --questions` and train on its split"
            )
        if row.completion not in decision.ids:
            raise EmbeddingTuneError(
                f"{where}:{row.lineno}: the completion {row.completion!r} is not one of the "
                "decision's options"
            )
        out.append(LabelledText(row.prompt[: -len(suffix)], row.completion))
    return out


@dataclass(frozen=True)
class LabelledImage:
    """One image row as the script reads it: the file, its digest, and its option id."""

    path: Path
    sha256: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"image": str(self.path), "sha256": self.sha256, "label": self.label}


class _Labelled(Protocol):
    @property
    def label(self) -> str: ...


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_CHUNK_BYTES = 1 << 20


def _sha256_hex(path: Path) -> str:
    """The bare hex digest a split row's `image_sha256` holds (`hash_file` prefixes its own)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def split_modality(path: Path, what: str) -> str:
    """The split's modality: `image` when its rows name images, `text` when they carry prompts.

    A split that mixes the two is refused, as `prepare` refuses such a file.
    """
    if not path.is_file():
        raise EmbeddingTuneError(f"the {what} split {path} does not exist")
    try:
        objects = read_objects(path, images=True)
        return "image" if rows_are_images(path, objects) else "text"
    except PrepareError as exc:
        raise EmbeddingTuneError(str(exc)) from exc


def images_of(path: Path, decision: Decision) -> list[LabelledImage]:
    """Each image row of a split `prepare` wrote, its file checked against its digest.

    Refused, naming the row: an `image` that is not a relative path inside the
    split's directory, a file that is not there, bytes whose sha256 is not the
    row's `image_sha256`, and a completion that is not one of the decision's ids.
    """
    try:
        objects = read_objects(path, images=True)
    except PrepareError as exc:
        raise EmbeddingTuneError(str(exc)) from exc
    root = path.parent.resolve()
    digests: dict[Path, str] = {}
    out: list[LabelledImage] = []
    for lineno, obj in objects:
        where = f"{path}:{lineno}"
        value, expected = obj.get("image"), obj.get("image_sha256")
        if "prompt" in obj:
            raise EmbeddingTuneError(f"{where}: an image row has no 'prompt'")
        if not isinstance(value, str) or not value:
            raise EmbeddingTuneError(f"{where}: 'image' must be the image's path in the split")
        if not isinstance(expected, str) or not _SHA256_HEX.fullmatch(expected):
            raise EmbeddingTuneError(
                f"{where}: 'image_sha256' is {expected!r}, not a sha256. Train on the split "
                "`prepare` wrote, whose rows carry the digest of their picture"
            )
        relative = PurePosixPath(value)
        file = (root / relative).resolve()
        if relative.is_absolute() or not file.is_relative_to(root):
            raise EmbeddingTuneError(
                f"{where}: the image {value!r} is not inside the split's directory {root}; "
                "`prepare` copies every picture into it"
            )
        if not file.is_file():
            raise EmbeddingTuneError(f"{where}: the image {value!r} does not exist ({file})")
        if file not in digests:
            digests[file] = _sha256_hex(file)
        if digests[file] != expected:
            raise EmbeddingTuneError(
                f"{where}: the image {value!r} hashes to {digests[file]}, not the "
                f"{expected} its row names; the picture changed after `prepare` wrote the "
                "split. Run prepare again"
            )
        completion = obj.get("completion")
        if completion not in decision.ids:
            raise EmbeddingTuneError(
                f"{where}: the completion {completion!r} is not one of the decision's options"
            )
        out.append(LabelledImage(file, expected, completion))
    return out


def resolve_swaps(given: Sequence[str], ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """`--mirror-swap A:B` as id pairs. An id may hold a colon, so the split is the one
    colon with an option id on both sides; none, or more than one, is refused, as is
    a pair of one id or an id that is in two pairs."""
    known = set(ids)
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text in given:
        cuts = [
            (text[:i], text[i + 1 :])
            for i, char in enumerate(text)
            if char == ":" and text[:i] in known and text[i + 1 :] in known
        ]
        if not cuts:
            raise EmbeddingTuneError(
                f"--mirror-swap {text}: not two of the decision's option ids around a colon "
                f"(its ids: {', '.join(sorted(known)[:12])})"
            )
        if len(cuts) > 1:
            raise EmbeddingTuneError(
                f"--mirror-swap {text}: reads as {len(cuts)} different pairs of option ids"
            )
        a, b = cuts[0]
        if a == b:
            raise EmbeddingTuneError(f"--mirror-swap {text}: an option swapped with itself")
        for label in (a, b):
            if label in seen:
                raise EmbeddingTuneError(
                    f"--mirror-swap {text}: {label!r} is already in another pair; each option "
                    "has one mirror"
                )
            seen.add(label)
        pairs.append((a, b))
    return tuple(pairs)


def labels_for(decision: Decision, train: Sequence[_Labelled]) -> tuple[str, ...]:
    """The decision's ids, sorted, once every one of them has a training row."""
    labels = tuple(sorted(decision.ids))
    if len(labels) < 2:
        raise EmbeddingTuneError(
            f"the decision has {len(labels)} option(s); a classifier head needs at least two"
        )
    counts = dict.fromkeys(labels, 0)
    for row in train:
        counts[row.label] += 1
    missing = [label for label, n in counts.items() if n == 0]
    if missing:
        raise EmbeddingTuneError(
            f"{len(missing)} option(s) have no training row: {', '.join(map(repr, missing[:12]))}"
            f"{' and more' if len(missing) > 12 else ''}. The recipe's head has one output per "
            "option, and an option with no row is never taught"
        )
    return labels


# ---------------------------------------------------------------------------
# The request, the record and the result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingTuneRequest:
    """One embedding fine-tune. Everything not here is `RECIPE`."""

    data: Path
    heldout: Path
    decision: Path
    output_dir: Path
    model: str = EMBEDDING_MODEL
    # `None` is the pinned revision; any other is refused.
    revision: str | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    env: envs.StageEnv = envs.EMBED_TRAIN
    auto_provision: bool = True
    # Image splits only (`--vision-tokens`, `--mirror`, `--mirror-swap`); a text
    # split given any of them is refused once its rows are read.
    vision_tokens: int | None = None
    mirror: bool = False
    # As given, `A:B`; resolved against the decision's ids (`resolve_swaps`).
    mirror_swap: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # `True == 1` is not in VISION_TOKENS either.
        if self.vision_tokens is not None and self.vision_tokens not in VISION_TOKENS:
            raise EmbeddingTuneError(
                f"--vision-tokens {self.vision_tokens}: the image recipe takes one of "
                f"{', '.join(map(str, VISION_TOKENS))}"
            )
        object.__setattr__(self, "mirror_swap", tuple(self.mirror_swap))
        if self.mirror_swap and not self.mirror:
            raise EmbeddingTuneError(
                "--mirror-swap without --mirror: the swap exchanges labels on the mirrored "
                "copies, and there are none without --mirror"
            )
        for text in self.mirror_swap:
            if ":" not in text:
                raise EmbeddingTuneError(
                    f"--mirror-swap {text}: write it as A:B, two option ids around a colon"
                )
        if not is_embedding_model(self.model):
            raise EmbeddingTuneError(
                f"the embedding recipe trains {EMBEDDING_MODEL} only, not {self.model!r}: it was "
                "measured on that model, and `convert` grafts into Google's file for it"
            )
        if self.revision is not None and self.revision != EMBEDDING_REVISION:
            raise EmbeddingTuneError(
                f"revision {self.revision!r} is not {EMBEDDING_REVISION}, the revision the recipe "
                "and convert's tensor mapping were established on"
            )
        if self.timeout_s < 1:
            raise EmbeddingTuneError(f"timeout_s must be at least 1, got {self.timeout_s}")
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "heldout", Path(self.heldout))
        object.__setattr__(self, "decision", Path(self.decision))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def pinned_revision(self) -> str:
        return EMBEDDING_REVISION

    @property
    def model_dir(self) -> Path:
        """The SentenceTransformer directory: what `convert` is pointed at."""
        return self.output_dir / "model"

    def spec(
        self,
        *,
        labels: Sequence[str],
        train_rows: Path,
        heldout_rows: Path,
        metrics_out: Path,
        device: str | None,
    ) -> dict[str, Any]:
        """What the script reads. `labels` are sorted; the head's output i is labels[i]."""
        return {
            "model": self.model,
            "revision": self.pinned_revision,
            "model_type": EMBEDDING_MODEL_TYPE,
            "prompt_name": PROMPT_NAME,
            "prefix": DECIDER_PREFIX,
            "labels": list(labels),
            "train_rows": str(train_rows),
            "heldout_rows": str(heldout_rows),
            "recipe": RECIPE.as_dict(),
            "device": device,
            "model_dir": str(self.model_dir),
            "metrics_out": str(metrics_out),
        }

    @property
    def image_cache_dir(self) -> Path:
        return self.output_dir / IMAGE_CACHE_DIR

    def image_spec(
        self,
        *,
        labels: Sequence[str],
        swaps: Sequence[tuple[str, str]],
        train_rows: Path,
        heldout_rows: Path,
        metrics_out: Path,
        device: str | None,
    ) -> dict[str, Any]:
        """What the image script reads. `labels` are sorted; the head's output i is labels[i]."""
        if self.vision_tokens is None:
            raise EmbeddingTuneError("an image run needs its vision tokens per image")
        return {
            "model": self.model,
            "revision": self.pinned_revision,
            "model_type": EMBEDDING_MODEL_TYPE,
            "prefix": IMAGE_PREFIX,
            "image_text": IMAGE_TOKEN_TEXT,
            "vision_tokens": self.vision_tokens,
            "mirror": self.mirror,
            "mirror_swap": [list(pair) for pair in swaps],
            "labels": list(labels),
            "train_rows": str(train_rows),
            "heldout_rows": str(heldout_rows),
            "recipe": IMAGE_RECIPE.as_dict(),
            "device": device,
            "model_dir": str(self.model_dir),
            "metrics_out": str(metrics_out),
            "cache_dir": str(self.image_cache_dir),
            "max_shard_size": IMAGE_MAX_SHARD_SIZE,
        }

    def as_dict(self, modality: str = DEFAULT_MODALITY) -> dict[str, Any]:
        out = {
            "direction": DIRECTION,
            "model": self.model,
            "revision": self.pinned_revision,
            "data": str(self.data),
            "heldout": str(self.heldout),
            "decision": str(self.decision),
            "output_dir": str(self.output_dir),
            "timeout_s": self.timeout_s,
            "recipe": (IMAGE_RECIPE if modality == "image" else RECIPE).as_dict(),
            "environment": {
                "name": self.env.name,
                "identity": self.env.identity,
                "requirements": envs.shown_requirements(self.env.requirements),
            },
        }
        if modality == "image":
            out |= {
                "modality": modality,
                "vision_tokens": self.vision_tokens,
                "mirror": self.mirror,
                "mirror_swap": list(self.mirror_swap),
            }
        return out


@dataclass(frozen=True)
class EmbeddingRecord:
    """`litetune.json` beside an embedding checkpoint: what `convert` and `verify` read."""

    base_model: str
    base_model_revision: str
    prompt_name: str
    prefix: str
    labels: tuple[str, ...]
    rows: Mapping[str, int]
    steps_done: int
    steps_planned: int
    seconds: float
    device: str
    dtype: str
    head_heldout_accuracy: float
    recipe: Mapping[str, Any]
    decision: Mapping[str, Any]
    decision_sha256: str
    # `storage.hash_file` of the training split tune read; `None` in a record
    # that does not carry it. `verify` compares it with the split it is given.
    train_sha256: str | None = None
    # `MODALITIES`: "text" for a record that does not carry the field.
    modality: str = DEFAULT_MODALITY
    # One of `VISION_TOKENS` for an image record; `None` for a text one.
    vision_tokens: int | None = None
    # Image records only: whether a flipped copy of every training image was
    # trained on, and the option pairs whose labels the copies exchanged.
    mirror: bool = False
    mirror_swap: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        problem = _modality_problem(self.modality, self.vision_tokens) or _mirror_problem(
            self.modality, self.mirror, self.mirror_swap, self.labels
        )
        if problem is not None:
            raise ValueError(problem)

    @property
    def is_image(self) -> bool:
        return self.modality == "image"

    def as_dict(self) -> dict[str, Any]:
        out = {
            "direction": DIRECTION,
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "prompt_name": self.prompt_name,
            "prefix": self.prefix,
            "labels": list(self.labels),
            "rows": dict(self.rows),
            "steps_done": self.steps_done,
            "steps_planned": self.steps_planned,
            "seconds": self.seconds,
            "device": self.device,
            "dtype": self.dtype,
            "head_heldout_accuracy": self.head_heldout_accuracy,
            "recipe": dict(self.recipe),
            "decision": dict(self.decision),
            "decision_sha256": self.decision_sha256,
            "train_sha256": self.train_sha256,
            "modality": self.modality,
            "vision_tokens": self.vision_tokens,
        }
        # A text record is written as it was before images: it has no mirror.
        if self.is_image:
            out["mirror"] = self.mirror
            out["mirror_swap"] = [list(pair) for pair in self.mirror_swap]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str) -> EmbeddingRecord:
        """The record in `data`. Raises `ValueError` naming the field that is missing or wrong."""
        if data.get("direction") != DIRECTION:
            found = data.get("direction")
            raise ValueError(f"{where}: 'direction' is {found!r}, not {DIRECTION!r}")
        kinds: dict[str, type | tuple[type, ...]] = {
            "base_model": str,
            "base_model_revision": str,
            "prompt_name": str,
            "prefix": str,
            "labels": list,
            "rows": dict,
            "steps_done": int,
            "steps_planned": int,
            "seconds": (int, float),
            "device": str,
            "dtype": str,
            "head_heldout_accuracy": (int, float),
            "recipe": dict,
            "decision": dict,
            "decision_sha256": str,
        }
        for key, kind in kinds.items():
            value = data.get(key)
            # `bool` is an `int`; a record saying `true` for a count is not one.
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ValueError(f"{where}: {key!r} is {value!r}, which is not a {kind}")
        labels = data["labels"]
        if not all(isinstance(label, str) for label in labels):
            raise ValueError(f"{where}: 'labels' holds a label that is not a string: {labels!r}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"{where}: 'labels' repeats a label: {labels!r}")
        for name, count in data["rows"].items():
            if not isinstance(count, int) or isinstance(count, bool):
                raise ValueError(f"{where}: 'rows' has {name!r}: {count!r}, which is not an int")
        if data["steps_done"] > data["steps_planned"]:
            raise ValueError(
                f"{where}: 'steps_done' {data['steps_done']} is more than 'steps_planned' "
                f"{data['steps_planned']}"
            )
        train_sha256 = data.get("train_sha256")
        if train_sha256 is not None and not isinstance(train_sha256, str):
            raise ValueError(f"{where}: 'train_sha256' is {train_sha256!r}, which is not a str")
        modality = data.get("modality", DEFAULT_MODALITY)
        vision_tokens = data.get("vision_tokens")
        problem = _modality_problem(modality, vision_tokens)
        if problem is not None:
            raise ValueError(f"{where}: {problem}")
        mirror = data.get("mirror", False)
        raw_swap = data.get("mirror_swap", [])
        if not isinstance(raw_swap, list) or not all(
            isinstance(pair, list) and len(pair) == 2 for pair in raw_swap
        ):
            raise ValueError(f"{where}: 'mirror_swap' is {raw_swap!r}, not a list of pairs")
        mirror_swap = tuple((pair[0], pair[1]) for pair in raw_swap)
        problem = _mirror_problem(modality, mirror, mirror_swap, tuple(labels))
        if problem is not None:
            raise ValueError(f"{where}: {problem}")
        return cls(
            base_model=data["base_model"],
            base_model_revision=data["base_model_revision"],
            prompt_name=data["prompt_name"],
            prefix=data["prefix"],
            labels=tuple(labels),
            rows=dict(data["rows"]),
            steps_done=data["steps_done"],
            steps_planned=data["steps_planned"],
            seconds=float(data["seconds"]),
            device=data["device"],
            dtype=data["dtype"],
            head_heldout_accuracy=float(data["head_heldout_accuracy"]),
            recipe=dict(data["recipe"]),
            decision=dict(data["decision"]),
            decision_sha256=data["decision_sha256"],
            train_sha256=train_sha256,
            modality=modality,
            vision_tokens=vision_tokens,
            mirror=mirror,
            mirror_swap=mirror_swap,
        )


def _mirror_problem(
    modality: object, mirror: object, mirror_swap: Sequence[tuple[object, object]], labels: object
) -> str | None:
    """Why `mirror` and `mirror_swap` are not a record's, or `None`."""
    if not isinstance(mirror, bool):
        return f"'mirror' is {mirror!r}, not true or false"
    if modality != "image" and (mirror or mirror_swap):
        return f"a {modality} record has no mirror, and this one says mirror {mirror!r}, swap " + (
            f"{[list(p) for p in mirror_swap]!r}"
        )
    if mirror_swap and not mirror:
        return "'mirror_swap' names pairs and 'mirror' is false: there were no copies to swap on"
    seen: list[object] = [label for pair in mirror_swap for label in pair]
    known = labels if isinstance(labels, tuple | list) else ()
    for label in seen:
        if not isinstance(label, str) or label not in known:
            return f"'mirror_swap' names {label!r}, which is not one of the record's labels"
    if len(set(seen)) != len(seen):
        return f"'mirror_swap' names a label twice: {[list(p) for p in mirror_swap]!r}"
    return None


def _modality_problem(modality: object, vision_tokens: object) -> str | None:
    """Why `modality` and `vision_tokens` are not a record's, or `None`."""
    if modality not in MODALITIES:
        return f"'modality' is {modality!r}, not one of {', '.join(MODALITIES)}"
    if modality == "image":
        # `70.0 == 70`, and is no token count. (`True` is an `int`, but neither 70 nor 140.)
        if not isinstance(vision_tokens, int) or vision_tokens not in VISION_TOKENS:
            return (
                f"'vision_tokens' is {vision_tokens!r}; an image record needs one of "
                f"{', '.join(map(str, VISION_TOKENS))}"
            )
    elif vision_tokens is not None:
        return f"'vision_tokens' is {vision_tokens!r} in a {modality} record, which has none"
    return None


def write_embedding_record(model_dir: Path, record: EmbeddingRecord) -> Path:
    path = Path(model_dir) / models.PROVENANCE_NAME
    path.write_text(json.dumps(record.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def read_embedding_record(model_dir: Path) -> EmbeddingRecord | None:
    """The embedding record beside `model_dir`; `None` when there is no record, or it is not one.

    "Not one" is a record without `"direction": "embedding"` -- every record an
    LLM `tune` writes. A record that says it is one and is malformed raises
    `ValueError`, and an unreadable one raises as `models.read_training_record`
    does.
    """
    read = models.read_training_record(str(model_dir))
    if read is None:
        return None
    data, sidecar = read
    if data.get("direction") != DIRECTION:
        return None
    return EmbeddingRecord.from_dict(data, str(sidecar))


def model_dir_hint(path: str | Path) -> str:
    """For a refusal naming `path`: the sentence that points at the model
    directory when `path` is `tune`'s output directory, whose `model/` holds the
    record; empty otherwise."""
    try:
        record = read_embedding_record(Path(path) / "model")
    except (OSError, ValueError):
        return ""
    if record is None:
        return ""
    return f"; {Path(path) / 'model'} is the model directory, where `tune` writes it"


@dataclass
class EmbeddingTuneResult:
    """What the run produced. `verified` is a property, and False."""

    request: EmbeddingTuneRequest
    checks: CheckSet
    # What the split's rows are: "image" runs the image recipe.
    modality: str = DEFAULT_MODALITY
    labels: tuple[str, ...] = ()
    rows: dict[str, int] = field(default_factory=dict)
    trained_decision: Decision | None = None
    record: EmbeddingRecord | None = None
    metrics: dict[str, Any] | None = None
    model_dir: Path | None = None
    device: str | None = None
    device_probe: envs.DeviceProbe | None = None
    host: dict[str, Any] | None = None
    mps_memory: devices.MpsMemory | None = None
    returncode: int | None = None
    seconds: float | None = None
    stderr: str = ""
    stdout_tail: str = ""
    limitations: list[str] = field(default_factory=list)

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.PASSED

    @property
    def verified(self) -> bool:
        return False

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "schema": EMBEDDING_TUNE_SCHEMA,
            "direction": DIRECTION,
            "verified": False,
            "unverified_reason": NOT_VERIFIED,
            "outcome": self.outcome.value,
            "decision": (
                self.trained_decision.as_dict() if self.trained_decision is not None else None
            ),
            "decision_sha256": (
                self.trained_decision.digest() if self.trained_decision is not None else None
            ),
            "labels": list(self.labels),
            "rows": dict(self.rows),
            "record": self.record.as_dict() if self.record is not None else None,
            "device": self.device,
            "device_probe": self.device_probe.as_dict() if self.device_probe else None,
            "host": self.host,
            "mps_memory": self.mps_memory.as_dict() if self.mps_memory else None,
            "request": self.request.as_dict(self.modality),
            "metrics": self.metrics,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3) if self.seconds is not None else None,
            "checks": self.checks.as_dict(),
            "stderr": self.stderr,
            "stdout_tail": self.stdout_tail,
            "limitations": list(self.limitations),
        }
        # A text run's report is the one written before images existed.
        if self.modality == "image":
            out["modality"] = self.modality
            out["vision_tokens"] = self.request.vision_tokens
        return out


def write_report(result: EmbeddingTuneResult, path: Path | None = None) -> Path:
    """`tune.json`, written for a run that did not finish too."""
    target = path or result.request.output_dir / "tune.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _read_split(path: Path, decision: Decision, what: str) -> list[LabelledText]:
    if not path.is_file():
        raise EmbeddingTuneError(f"the {what} split {path} does not exist")
    try:
        rows = read_rows(path)
    except PrepareError as exc:
        raise EmbeddingTuneError(str(exc)) from exc
    states = states_of(rows, decision, path)
    if not states:
        raise EmbeddingTuneError(f"the {what} split {path} has no rows")
    return states


@dataclass(frozen=True)
class _Checked:
    """A request whose decision and rows were read: what the run trains on."""

    decision: Decision
    modality: str
    train: Sequence[LabelledText | LabelledImage]
    heldout: Sequence[LabelledText | LabelledImage]
    labels: tuple[str, ...]
    swaps: tuple[tuple[str, str], ...] = ()


def _read_image_split(path: Path, decision: Decision, what: str) -> list[LabelledImage]:
    rows = images_of(path, decision)
    if not rows:
        raise EmbeddingTuneError(f"the {what} split {path} has no rows")
    return rows


def _refuse_image_flags_for_text(request: EmbeddingTuneRequest) -> None:
    given = [
        flag
        for flag, value in (
            ("--vision-tokens", request.vision_tokens is not None),
            ("--mirror", request.mirror),
            ("--mirror-swap", bool(request.mirror_swap)),
        )
        if value
    ]
    if given:
        raise EmbeddingTuneError(
            f"{', '.join(given)}: {request.data} is a text split, and these apply to an image "
            "split only (rows with an 'image')"
        )


def _check_request(request: EmbeddingTuneRequest) -> _Checked:
    """The decision, both splits' rows and the labels. Raises `EmbeddingTuneError`."""
    try:
        decision, _ = read_questions(request.decision)
    except DecisionError as exc:
        raise EmbeddingTuneError(str(exc)) from exc
    if decision.type != "choice":
        raise EmbeddingTuneError(
            f"the decision is a {decision.type} question; the embedding recipe trains a choice "
            "question only, the type it was measured on and the one flutter_edge_ai's embedding "
            "decider answers"
        )
    if decision.none_option is not None:
        raise EmbeddingTuneError(NONE_OPTION_REFUSED.format(none=decision.none_option))
    modality = split_modality(request.data, "training")
    if modality == "text":
        _refuse_image_flags_for_text(request)
        train = _read_split(request.data, decision, "training")
        heldout = _read_split(request.heldout, decision, "held-out")
        return _Checked(decision, modality, train, heldout, labels_for(decision, train))
    if request.vision_tokens is None:
        raise EmbeddingTuneError(
            f"{request.data} is an image split, and needs --vision-tokens "
            f"({' or '.join(map(str, VISION_TOKENS))}): the vision tokens per image the "
            f"application gives the runtime, flutter_edge_ai's "
            f"{VISION_TOKENS_SETTING.format(n='N')}. The "
            "model is tuned at that count and embeds differently at another"
        )
    heldout_modality = split_modality(request.heldout, "held-out")
    if heldout_modality != modality:
        raise EmbeddingTuneError(
            f"the training split {request.data} is image rows and the held-out split "
            f"{request.heldout} is {heldout_modality} rows; give the held-out split prepare "
            "wrote beside it"
        )
    swaps = resolve_swaps(request.mirror_swap, decision.ids)
    images = _read_image_split(request.data, decision, "training")
    held = _read_image_split(request.heldout, decision, "held-out")
    return _Checked(decision, modality, images, held, labels_for(decision, images), swaps)


def _write_rows(path: Path, rows: Sequence[LabelledText | LabelledImage]) -> None:
    path.write_text(
        "".join(json.dumps(row.as_dict(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
    )


def _note_unmeasured_device(result: EmbeddingTuneResult, device: str) -> None:
    """Record as a limitation that `device` is not the one the recipe was measured on."""
    if result.modality == "image":
        if device not in IMAGE_RECIPE_DEVICES:
            result.limitation(
                f"this run trained on {device}; the runs the image recipe comes from trained on "
                f"{' and '.join(IMAGE_RECIPE_DEVICES)} only"
            )
        return
    if device != MEASURED_DEVICE:
        result.limitation(
            "the recipe was measured in float32 on the CPU only (MEASUREMENTS.md, "
            '"EmbeddingGemma 2: many labels"); this run trained on '
            f"{device}, a device nothing measured the recipe on"
        )


def _finish(
    result: EmbeddingTuneResult, events: EventStream, check: Check, attempted: bool
) -> EmbeddingTuneResult:
    result.checks.add(check)
    events.check(check)
    events.stage_finished(result.outcome.value, attempted=attempted, direction=DIRECTION)
    return result


def run_embedding_tune(
    request: EmbeddingTuneRequest, events: EventStream | None = None
) -> EmbeddingTuneResult:
    """Fine-tune inside `request.env`. A refused request raises `EmbeddingTuneError`;
    everything that happens once the environment is involved is a check in the result."""
    envs.forget_reported_drops()
    events = events or EventStream(echo_json=False)

    # -- the request and its rows, before the environment is touched --------
    checked = _check_request(request)
    decision, train, heldout, labels = (
        checked.decision,
        checked.train,
        checked.heldout,
        checked.labels,
    )
    image = checked.modality == "image"
    train_sha256 = hash_file(request.data)

    events.stage_started("train", model=request.model, direction=DIRECTION)
    result = EmbeddingTuneResult(
        request=request,
        checks=CheckSet(name=f"train:{request.model}:{DIRECTION}"),
        modality=checked.modality,
    )
    result.trained_decision = decision
    result.labels = labels
    result.rows = {"train": len(train), "heldout": len(heldout)}
    result.limitation(NOT_VERIFIED)
    if image:
        result.limitation(IMAGE_RECIPE_UNMEASURED)
        if request.mirror:
            pairs = "; ".join(f"{a} and {b}" for a, b in checked.swaps)
            swapped = f" (the copies' labels exchanged: {pairs})"
            result.limitation(
                IMAGE_MIRROR_ASSUMES_SYMMETRY.format(swapped=swapped if checked.swaps else "")
            )
    rows_check = Check.passed(
        DATA_CHECK,
        f"{len(train)} training and {len(heldout)} held-out "
        f"{'image rows, every picture matching its digest,' if image else 'rows'} over "
        f"{len(labels)} options of {decision.question!r}, {decision.digest()[:23]}",
        observed={
            "rows": result.rows,
            "labels": len(labels),
            "sha256": decision.digest(),
        }
        | ({"modality": "image"} if image else {}),
    )
    result.checks.add(rows_check)
    events.check(rows_check)

    try:
        setting = devices.device_setting(os.environ)
    except devices.DeviceSettingError as exc:
        return _finish(
            result,
            events,
            Check.unchecked(DEVICE_CHECK, f"training was not attempted: {exc}"),
            attempted=False,
        )

    # -- the environment ---------------------------------------------------
    with guard(ENV_CHECK) as sink:
        if request.auto_provision:
            request.env.provision(events=events)
        if request.env.ready:
            sink.append(
                Check.passed(
                    ENV_CHECK,
                    f"{request.env.name} ({request.env.identity}) ready at {request.env.path}",
                    observed={
                        "name": request.env.name,
                        "identity": request.env.identity,
                        "requirements": envs.shown_requirements(request.env.requirements),
                    },
                )
            )
        else:
            sink.append(
                Check.unchecked(
                    ENV_CHECK,
                    f"environment {request.env.name!r} {request.env.unready_detail}",
                    observed={"name": request.env.name, "identity": request.env.identity},
                )
            )
    environment = result.checks.add(sink[0])
    events.check(environment)
    if not environment.conclusive:
        result.limitation(f"training was not attempted: {environment.detail}")
        events.stage_finished(result.outcome.value, attempted=False, direction=DIRECTION)
        return result

    # -- the device --------------------------------------------------------
    probe = devices.apply_device_setting(envs.resolve_device(request.env, events=events), setting)
    result.device = probe.device
    result.device_probe = probe
    if not probe.answered:
        result.limitation(
            f"{probe.detail}, so this run's device was not established before it started. The "
            "training script decided for itself and reports what it chose in its metrics"
        )
    elif probe.cuda_build_without_a_device or probe.cpu_on_macos:
        result.limitation(probe.detail)
    if probe.source == devices.DEVICE_VARIABLE:
        result.limitation(probe.detail)
        events.note(probe.detail, device=probe.device, source=probe.source)
    result.host = devices.host_record(probe)
    if probe.answered and probe.device is not None:
        _note_unmeasured_device(result, probe.device)

    child_env: dict[str, str] | None = None
    if probe.device == "mps":
        try:
            result.mps_memory = devices.prepare_mps(probe, os.environ)
        except devices.MpsMemoryRefused as exc:
            result.limitation(f"training was not attempted: {exc}")
            return _finish(
                result,
                events,
                Check.unchecked(
                    DEVICE_CHECK,
                    f"training was not attempted on mps: {exc}",
                    observed={"device": probe.device, "host": result.host},
                ),
                attempted=False,
            )
        child_env = result.mps_memory.child_env
        for text in result.mps_memory.limitations:
            result.limitation(text)
        events.note(
            result.mps_memory.summary(),
            **{k: v for k, v in result.mps_memory.variables.items() if v is not None},
        )

    # -- run it ------------------------------------------------------------
    workspace = request.output_dir
    workspace.mkdir(parents=True, exist_ok=True)
    if image:
        script = workspace / IMAGE_SCRIPT_NAME
        script.write_text(_EMBED_IMAGE_SCRIPT, encoding="utf-8")
    else:
        script = workspace / "embed_train_script.py"
        script.write_text(_EMBED_TRAIN_SCRIPT, encoding="utf-8")
    train_rows = workspace / "embed_train_rows.jsonl"
    heldout_rows = workspace / "embed_heldout_rows.jsonl"
    _write_rows(train_rows, train)
    _write_rows(heldout_rows, heldout)
    metrics_out = workspace / "embed_metrics.json"
    spec_path = workspace / "embed_train_spec.json"
    if image:
        spec = request.image_spec(
            labels=labels,
            swaps=checked.swaps,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            metrics_out=metrics_out,
            device=probe.device,
        )
    else:
        spec = request.spec(
            labels=labels,
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            metrics_out=metrics_out,
            device=probe.device,
        )
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    # A previous attempt's metrics would be read as this one's.
    metrics_out.unlink(missing_ok=True)
    # Likewise a previous run's record: if this run fails after the script
    # overwrote the weights, the old record would sit beside weights it does not describe.
    (request.model_dir / models.PROVENANCE_NAME).unlink(missing_ok=True)

    events.note(
        f"training {request.model} on {len(train)} "
        f"{f'image rows at {request.vision_tokens} vision tokens' if image else 'rows'}, "
        f"{len(labels)} options",
        device=probe.device,
        dtype=RECIPE.dtype,
    )
    started = time.perf_counter()
    try:
        proc = request.env.run(
            ["python", str(script), str(spec_path)], timeout=request.timeout_s, env=child_env
        )
    except subprocess.TimeoutExpired:
        result.seconds = time.perf_counter() - started
        logger.warning("embedding training timed out after %ss", request.timeout_s)
        return _finish(
            result,
            events,
            Check.unchecked(
                TRAINING_CHECK,
                f"no result after {request.timeout_s}s (timeout): the run did not finish, which "
                "says nothing about the recipe or the data",
                observed={"timeout_s": request.timeout_s},
            ),
            attempted=True,
        )
    except OSError as exc:
        logger.exception("could not start the embedding training script")
        result.seconds = time.perf_counter() - started
        return _finish(
            result,
            events,
            Check.unchecked(
                TRAINING_CHECK,
                f"the training script could not be started: {type(exc).__name__}: {exc}",
            ),
            attempted=False,
        )
    result.seconds = time.perf_counter() - started
    result.returncode = proc.returncode
    result.stderr = proc.stderr or ""
    result.stdout_tail = (proc.stdout or "")[-_STDOUT_TAIL:]

    metrics_error = ""
    if metrics_out.is_file():
        try:
            loaded = json.loads(metrics_out.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.exception("the embedding training wrote metrics that could not be read")
            metrics_error = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(loaded, dict):
                result.metrics = loaded
            else:
                metrics_error = f"{metrics_out} does not hold a JSON object"
    else:
        metrics_error = f"no metrics file at {metrics_out}"

    reading = read_returncode(proc.returncode)
    if not reading.conclusive:
        return _finish(
            result,
            events,
            Check.unchecked(
                TRAINING_CHECK,
                f"training was {reading.describe('the recipe or the data')}. "
                f"stderr: {_tail(result.stderr) or 'none'}",
                observed=reading.as_dict() | {"seconds": round(result.seconds, 3)},
            ),
            attempted=True,
        )
    if proc.returncode == REFUSED_EXIT and result.metrics and "refused" in result.metrics:
        return _finish(
            result,
            events,
            Check.failed(
                MODEL_CHECK,
                f"the training script refused the model: {result.metrics['refused']}",
                observed={"model": request.model, "revision": request.pinned_revision},
            ),
            attempted=True,
        )
    gpu_failure = _GPU_FAILURE_RE.search(result.stderr) if proc.returncode != 0 else None
    if gpu_failure is not None:
        return _finish(
            result,
            events,
            Check.unchecked(
                TRAINING_CHECK,
                f"training exited {proc.returncode} on a machine failure "
                f"({gpu_failure.group(0)}), which says nothing about the recipe or the data"
                f"{_mps_advice(result.stderr, result.mps_memory)}: "
                f"{_tail(result.stderr) or 'no stderr'}",
                observed={
                    "returncode": proc.returncode,
                    "matched": gpu_failure.group(0),
                    "device": probe.device,
                },
            ),
            attempted=True,
        )
    if proc.returncode != 0:
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"training exited {proc.returncode}: {_tail(result.stderr) or 'no stderr'}",
                observed={"returncode": proc.returncode, "stderr_tail": _tail(result.stderr, 2000)},
            ),
            attempted=True,
        )
    if not (request.model_dir.is_dir() and any(request.model_dir.iterdir())):
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"training exited zero but wrote no checkpoint into {request.model_dir}",
                observed={"model_dir": str(request.model_dir)},
            ),
            attempted=True,
        )
    if result.metrics is None:
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"training exited zero and wrote a checkpoint, but no readable metrics "
                f"({metrics_error}), so what it trained cannot be recorded beside it",
                observed={"model_dir": str(request.model_dir)},
            ),
            attempted=True,
        )
    if result.mps_memory is not None:
        mismatch = devices.environment_mismatch(
            result.mps_memory, result.metrics.get("mps_environment"), "training script"
        )
        if mismatch is not None:
            result.limitation(mismatch)
    device_reported = result.metrics.get("device")
    if isinstance(device_reported, str):
        _note_unmeasured_device(result, device_reported)

    if image and not (request.model_dir / BASE_CHECKPOINT_FILE).is_file():
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"training exited zero but wrote no {BASE_CHECKPOINT_FILE} into "
                f"{request.model_dir}, the one file convert grafts from",
                observed={"saved_files": sorted(p.name for p in request.model_dir.iterdir())},
            ),
            attempted=True,
        )
    image_fields: dict[str, Any] = {}
    if image:
        image_fields = {
            "modality": "image",
            "vision_tokens": result.metrics.get("vision_tokens"),
            "mirror": result.metrics.get("mirror"),
            "mirror_swap": result.metrics.get("mirror_swap"),
        }
    try:
        record = EmbeddingRecord.from_dict(
            {
                "direction": DIRECTION,
                "base_model": request.model,
                "base_model_revision": request.pinned_revision,
                "prompt_name": IMAGE_PROMPT_NAME if image else PROMPT_NAME,
                "prefix": result.metrics.get("prefix"),
                "labels": list(result.metrics.get("labels") or ()),
                "rows": result.metrics.get("rows"),
                "steps_done": result.metrics.get("steps_done"),
                "steps_planned": result.metrics.get("steps_planned"),
                "seconds": result.metrics.get("train_seconds"),
                "device": result.metrics.get("device"),
                "dtype": result.metrics.get("dtype"),
                "head_heldout_accuracy": result.metrics.get("head_heldout_accuracy"),
                "recipe": (IMAGE_RECIPE if image else RECIPE).as_dict(),
                "decision": decision.as_dict(),
                "decision_sha256": decision.digest(),
                "train_sha256": train_sha256,
            }
            | image_fields,
            str(metrics_out),
        )
    except ValueError as exc:
        logger.exception("the embedding training's metrics are incomplete")
        return _finish(
            result,
            events,
            Check.failed(TRAINING_CHECK, f"the training script's metrics are incomplete: {exc}"),
            attempted=True,
        )
    prefix = IMAGE_PREFIX if image else DECIDER_PREFIX
    if record.labels != labels or record.prefix != prefix:
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"the training script reports labels {list(record.labels)[:6]}… and prefix "
                f"{record.prefix!r}, not the {len(labels)} sorted options and "
                f"{prefix!r} it was given",
            ),
            attempted=True,
        )
    if image and (record.vision_tokens, record.mirror, record.mirror_swap) != (
        request.vision_tokens,
        request.mirror,
        checked.swaps,
    ):
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"the training script reports {record.vision_tokens} vision tokens, mirror "
                f"{record.mirror}, swaps {[list(p) for p in record.mirror_swap]}, not the "
                f"{request.vision_tokens}, {request.mirror}, "
                f"{[list(p) for p in checked.swaps]} it was given",
            ),
            attempted=True,
        )
    write_embedding_record(request.model_dir, record)
    result.record = record
    result.model_dir = request.model_dir
    events.artifact(str(request.model_dir), name="model", verified=False)
    return _finish(
        result,
        events,
        Check.passed(
            TRAINING_CHECK,
            f"{record.steps_done} of {record.steps_planned} steps on {record.device} in "
            f"{record.seconds:.1f}s; the discarded head scores {record.head_heldout_accuracy:.4f} "
            "on the held-out split — trained, not verified",
            observed={
                "model_dir": str(request.model_dir),
                "head_heldout_accuracy": record.head_heldout_accuracy,
                "verified": False,
            },
        ),
        attempted=True,
    )


def summarise(result: EmbeddingTuneResult) -> list[str]:
    if result.modality == "image":
        request = result.request
        described = f"on images at {request.vision_tokens} vision tokens per image" + (
            ", mirrored" if request.mirror else ""
        )
        lines = [
            f"outcome: {result.outcome.value}",
            f"  embedding fine-tune of {request.model} at {request.pinned_revision[:12]}, "
            f"{len(result.labels)} options, {described}; the app must give the runtime the "
            f"same count ({VISION_TOKENS_SETTING.format(n=request.vision_tokens)})",
        ]
    else:
        lines = [
            f"outcome: {result.outcome.value}",
            f"  embedding fine-tune of {result.request.model} at "
            f"{result.request.pinned_revision[:12]}, {len(result.labels)} options, prefix "
            f"{DECIDER_PREFIX!r}",
        ]
    if result.record is not None:
        lines.append(
            f"  {result.record.steps_done}/{result.record.steps_planned} steps on "
            f"{result.record.device}; discarded head on held-out: "
            f"{result.record.head_heldout_accuracy:.4f}"
        )
    if result.model_dir is not None:
        lines.append(f"  checkpoint {result.model_dir}")
    for check in result.checks.checks:
        if check.outcome is not Outcome.PASSED:
            lines.append(f"  {check.outcome.value}: {check.name} — {check.detail}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    return lines
