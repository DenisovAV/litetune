"""`tune` for google/embeddinggemma-2: a decision's rows, as an embedding model.

`litetune tune --model google/embeddinggemma-2 --decision <decision.json>` lands
here instead of `tune.run_tune` (`cli._tune`). The model is fine-tuned so that
flutter_edge_ai's `EmbeddingModel.createDecider`, which embeds every input with
`DECIDER_PREFIX` (`flutter_edge_ai/lib/src/decision/embedding_decider.dart:41`,
`deciderTaskType`), separates the decision's options better. Nothing but the
SentenceTransformer is kept: `convert` grafts its text tower into Google's
`.litertlm` (`embedding_convert`).

**The recipe is the measured one, and takes no overrides** (`RECIPE`). It is
`train_s5.py` of lora_project `measurements/2026-10-25-eg2-many-labels/`, which is
s3/s4's recipe of `measurements/2026-10-22-eg2-graft/`: the whole model trained
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

The training runs in `envs.EMBED_TRAIN` as a generated script; this process
imports neither torch nor sentence-transformers.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import devices, envs, models
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.decisions import Decision, DecisionError, read_questions
from litetune.embedding_convert import BASE_MODEL, BASE_REVISION
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.prepare import PrepareError, Row, read_rows
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

MODEL_CHECK = "embedding model"
DATA_CHECK = "decision rows"

# The script's exit code for a model it refused after reading its config.
REFUSED_EXIT = 3

NOT_VERIFIED = (
    "training completed and nothing about the decider has been measured. The head's held-out "
    "accuracy is the discarded head's, not the centroid decider's the application builds from "
    "the embeddings; `litetune verify` measures that"
)
NONE_OPTION_UNMEASURED = (
    "the decision's none_option {none!r} is trained as one more label. The recipe was measured on "
    "decisions without one, so how its rows embed is not measured"
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
    schedule: str = "linear warmup over the first 10% of steps, then linear decay to 0"

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


def labels_for(decision: Decision, train: Sequence[LabelledText]) -> tuple[str, ...]:
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

    def __post_init__(self) -> None:
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": DIRECTION,
            "model": self.model,
            "revision": self.pinned_revision,
            "data": str(self.data),
            "heldout": str(self.heldout),
            "decision": str(self.decision),
            "output_dir": str(self.output_dir),
            "timeout_s": self.timeout_s,
            "recipe": RECIPE.as_dict(),
            "environment": {
                "name": self.env.name,
                "identity": self.env.identity,
                "requirements": envs.shown_requirements(self.env.requirements),
            },
        }


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

    def as_dict(self) -> dict[str, Any]:
        return {
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
        }

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
        )


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


@dataclass
class EmbeddingTuneResult:
    """What the run produced. `verified` is a property, and False."""

    request: EmbeddingTuneRequest
    checks: CheckSet
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
        return {
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
            "request": self.request.as_dict(),
            "metrics": self.metrics,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3) if self.seconds is not None else None,
            "checks": self.checks.as_dict(),
            "stderr": self.stderr,
            "stdout_tail": self.stdout_tail,
            "limitations": list(self.limitations),
        }


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


def _check_request(
    request: EmbeddingTuneRequest,
) -> tuple[Decision, list[LabelledText], list[LabelledText], tuple[str, ...]]:
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
    train = _read_split(request.data, decision, "training")
    heldout = _read_split(request.heldout, decision, "held-out")
    return decision, train, heldout, labels_for(decision, train)


def _write_rows(path: Path, rows: Sequence[LabelledText]) -> None:
    path.write_text(
        "".join(json.dumps(row.as_dict(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="",
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
    decision, train, heldout, labels = _check_request(request)

    events.stage_started("train", model=request.model, direction=DIRECTION)
    result = EmbeddingTuneResult(
        request=request, checks=CheckSet(name=f"train:{request.model}:{DIRECTION}")
    )
    result.trained_decision = decision
    result.labels = labels
    result.rows = {"train": len(train), "heldout": len(heldout)}
    result.limitation(NOT_VERIFIED)
    if decision.none_option is not None:
        result.limitation(NONE_OPTION_UNMEASURED.format(none=decision.none_option))
    rows_check = Check.passed(
        DATA_CHECK,
        f"{len(train)} training and {len(heldout)} held-out rows over {len(labels)} options of "
        f"{decision.question!r}, {decision.digest()[:23]}",
        observed={"rows": result.rows, "labels": len(labels), "sha256": decision.digest()},
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
    script = workspace / "embed_train_script.py"
    script.write_text(_EMBED_TRAIN_SCRIPT, encoding="utf-8")
    train_rows = workspace / "embed_train_rows.jsonl"
    heldout_rows = workspace / "embed_heldout_rows.jsonl"
    _write_rows(train_rows, train)
    _write_rows(heldout_rows, heldout)
    metrics_out = workspace / "embed_metrics.json"
    spec_path = workspace / "embed_train_spec.json"
    spec_path.write_text(
        json.dumps(
            request.spec(
                labels=labels,
                train_rows=train_rows,
                heldout_rows=heldout_rows,
                metrics_out=metrics_out,
                device=probe.device,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    # A previous attempt's metrics would be read as this one's.
    metrics_out.unlink(missing_ok=True)

    events.note(
        f"training {request.model} on {len(train)} rows, {len(labels)} options",
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

    try:
        record = EmbeddingRecord.from_dict(
            {
                "direction": DIRECTION,
                "base_model": request.model,
                "base_model_revision": request.pinned_revision,
                "prompt_name": PROMPT_NAME,
                "prefix": result.metrics.get("prefix"),
                "labels": list(result.metrics.get("labels") or ()),
                "rows": result.metrics.get("rows"),
                "steps_done": result.metrics.get("steps_done"),
                "steps_planned": result.metrics.get("steps_planned"),
                "seconds": result.metrics.get("train_seconds"),
                "device": result.metrics.get("device"),
                "dtype": result.metrics.get("dtype"),
                "head_heldout_accuracy": result.metrics.get("head_heldout_accuracy"),
                "recipe": RECIPE.as_dict(),
                "decision": decision.as_dict(),
                "decision_sha256": decision.digest(),
            },
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
    if record.labels != labels or record.prefix != DECIDER_PREFIX:
        return _finish(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"the training script reports labels {list(record.labels)[:6]}… and prefix "
                f"{record.prefix!r}, not the {len(labels)} sorted options and "
                f"{DECIDER_PREFIX!r} it was given",
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
    lines = [
        f"outcome: {result.outcome.value}",
        f"  embedding fine-tune of {result.request.model} at {result.request.pinned_revision[:12]}"
        f", {len(result.labels)} options, prefix {DECIDER_PREFIX!r}",
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
