"""`tune`: supervised fine-tuning, with the loss on the completion only.

The training itself runs inside `envs.TRAIN` as a generated script. This process
never imports torch -- that is the entire reason the per-stage environments
exist, since `torch`/`transformers` and `litert-torch`/`numpy<2.1` cannot share
an interpreter. Everything here writes a config, runs a subprocess, and reads
what it wrote back.

**The loss is masked to the completion.** In the data shape this tool was built
for the tool declarations are roughly 330 of roughly 350 tokens, so without
masking about 94% of the gradient goes into memorising a header that is already
sitting on the model's input. This is not a style preference. A LoRA run without
masking reached a *lower* training loss than a masked one -- 0.50 against 1.45 --
and scored **0.0625 against a 0.5625 base**: nine times worse than doing nothing
at all. The loss curve reported success from the first step to the last, and the
label-free liveness tier passed it too. The only instrument that caught it was
held-out measurement.

**So the mask is reported as a number, not assumed.** `supervised_token_fraction`
is computed from the actual `labels` tensor and travels with the result. It
should be near 0.07 on this data shape; a value near 1.0 means the mask silently
did not apply and the run is the failure above. That is a `failed` check, and it
is the one thing in this module that stops a pipeline.

**Learning rates are per method.** 1e-5 for a full fine-tune, 2e-4 for LoRA --
roughly the twentyfold difference the two methods are documented and measured to
need. A single shared rate starves one of them, and a comparison run that way
measures the rate rather than the method.

**Nothing here is verification.** A completed training run means the process
exited zero, not that the model is better. `TuneResult.verified` is a property
that returns False and is not a field, so no code path can set it otherwise --
the same construction `export.RecipeExport` uses, for the same reason.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs, models
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.evaluate import PromptMode
from litetune.events import EventStream
from litetune.exits import read_returncode

logger = logging.getLogger(__name__)

TUNE_SCHEMA = "litetune.tune/1"

METHODS = ("full", "lora")

# Per method, because they are not interchangeable. Measured and documented at
# roughly a twentyfold difference: an adapter sees gradient through a handful of
# low-rank matrices and needs a rate a full fine-tune would diverge at.
LEARNING_RATES: dict[str, float] = {"full": 1e-5, "lora": 2e-4}

# What a correctly masked run looks like on this data shape: declarations of
# ~330 tokens against a completion of ~20-25, so loss is computed on ~7% of the
# sequence. Reported for comparison, never enforced -- another dataset with
# shorter declarations legitimately sits higher.
EXPECTED_SUPERVISED_FRACTION = 0.07

# ... but nothing legitimately sits *here*. A tool-calling corpus whose loss
# covers 95% of its tokens is not a corpus with short prompts, it is a run whose
# `labels` were never masked. This is the 0.0625-against-0.5625 signature.
MASKING_NOT_APPLIED_ABOVE = 0.95

# bfloat16 and eager attention, because the export and evaluation paths use
# them: a checkpoint trained under a different attention implementation or
# accumulation dtype than it is served with produces fluent garbage that every
# label-free check passes. Gemma-family models are documented as requiring eager.
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN_IMPLEMENTATION = "eager"

# The projection set LoRA is applied to on Gemma-family checkpoints. Named
# rather than "all linear layers" so that two runs recorded as `lora` are
# comparable; an adapter over a different module set is a different method.
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# A 270M checkpoint over a few thousand examples. High enough that a slow CPU
# run is not cut off mid-epoch, low enough that a wedged run is a non-result the
# same day rather than an occupied runner overnight.
DEFAULT_TIMEOUT_S = 6 * 3600

TRAINING_CHECK = "training run"
MASKING_CHECK = "loss is masked to the completion"
MERGE_CHECK = "merged checkpoint written"
ENV_CHECK = "training environment"

NOT_VERIFIED = (
    "training completed and nothing has been measured. A falling loss curve is not evidence: the "
    "run that scored 0.0625 against a 0.5625 base reached a lower training loss than the run that "
    "worked, and passed every label-free check afterwards. Quality is established by "
    "`litetune verify` on held-out data, and by nothing else."
)

_STDOUT_TAIL = 4000
_DETAIL_TAIL = 400


class TuneError(ValueError):
    """The training request is not runnable. Not a statement about any model."""


def _tail(text: str, limit: int = _DETAIL_TAIL) -> str:
    stripped = (text or "").strip()
    return stripped[-limit:] if stripped else ""


# ---------------------------------------------------------------------------
# The script that runs inside envs.TRAIN
# ---------------------------------------------------------------------------
#
# Written to a file rather than passed with `python -c` so that a traceback
# carries usable line numbers -- a training failure six hours in is diagnosed
# from this stderr and nothing else.
#
# It hand-rolls the loop instead of using `transformers.Trainer`, for two
# reasons. Trainer requires `accelerate`, which is deliberately not in
# `envs.TRAIN`'s pinned set, and every masking decision below has to be visible
# in one place: the entire finding this module exists for is that a training
# loop can look completely healthy while computing the loss on the wrong tokens.
#
# The schedule is a constant learning rate with no warmup and no accumulation.
# That is a deliberate floor rather than a tuned recipe: it applies identically
# to both methods, so a full-against-LoRA comparison is not confounded by it,
# and every parameter that does vary is in the config file beside this script.

_TRAIN_SCRIPT = r'''
"""Supervised fine-tuning with the loss masked to the completion.

Reads a config JSON, writes a metrics JSON. Prints nothing: the parent turns the
metrics file into events.
"""
import json
import random
import sys
from pathlib import Path

# torch's own ignore index. Positions set to this contribute no gradient, and
# they are the whole mechanism by which the prompt is excluded from the loss.
IGNORE_INDEX = -100


def render_prompt(tok, prompt, runtime_rendered):
    """The prompt as the *serving runtime* will present it.

    Two mutually exclusive conventions, and the model learns whichever one it
    was trained against. `evaluate.py`'s generation script makes the same choice
    on the same flag; if the two disagree, every measurement is taken on a
    prompt the model was never trained on.
    """
    if not runtime_rendered:
        return prompt, True
    templated = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    # The template already emits the model's BOS. Asking the tokenizer to add
    # another produces two, which shifts every position by one and is invisible
    # in the loss.
    return templated, False


def turn_terminator(tok, runtime_rendered):
    """The token ids the serving convention puts after the model's answer.

    Training used to append `tok.eos_token_id` unconditionally. That is the
    terminator for a raw completion, but not necessarily the one a chat
    template closes the assistant turn with -- Gemma's `<end_of_turn>` and
    `<eos>` are different tokens, and a model trained to emit one while the
    runtime waits for the other never stops on its own.

    The consequence is not a lower score. It is a model that keeps generating:
    in a 40-generation sample taken while this was wrong,
    34 emitted more than one call -- the same call repeated up to eight times,
    or two alternating. On a phone every one of them fires.

    So derive it from the template rather than assume it: render an assistant
    turn around a probe string and take whatever the template appends after it.
    Falls back to the tokenizer's EOS when there is no template to ask, and the
    choice is recorded either way -- a terminator picked silently is one nobody
    can check.
    """
    probe = "\u0000LITETUNE_ANSWER\u0000"
    if runtime_rendered:
        try:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": "x"},
                 {"role": "assistant", "content": probe}],
                tokenize=False, add_generation_prompt=False,
            )
            tail = rendered.split(probe)[-1]
            ids = tok(tail, add_special_tokens=False)["input_ids"]
            if ids:
                return list(ids), "chat_template"
        except Exception as exc:  # noqa: BLE001 - a template that will not
            # render an assistant turn is a fact about the model, not a failure
            # here. But the fallback guards the most expensive failure this
            # module documents, so the reason it was taken is recorded rather
            # than discarded.
            probe_error = f"{type(exc).__name__}: {exc}"[:200]
            return ([tok.eos_token_id] if tok.eos_token_id is not None else [],
                    f"tokenizer_eos (chat template probe failed: {probe_error})")
    if tok.eos_token_id is not None:
        return [tok.eos_token_id], "tokenizer_eos"
    return [], "none"


def build_examples(tok, rows, max_seq_length, runtime_rendered):
    """One (input_ids, labels) pair per row, with the prompt masked out."""
    examples = []
    supervised = 0
    total = 0
    terminator, terminator_source = turn_terminator(tok, runtime_rendered)
    for row in rows:
        prompt_text, add_special = render_prompt(tok, row["prompt"], runtime_rendered)
        prompt_ids = tok(prompt_text, add_special_tokens=add_special)["input_ids"]
        completion_ids = tok(row["completion"], add_special_tokens=False)["input_ids"]
        completion_ids = list(completion_ids) + list(terminator)
        input_ids = list(prompt_ids) + list(completion_ids)
        if len(input_ids) > max_seq_length:
            # Never truncate. Cutting the sequence removes the end of the
            # completion -- the answer -- and the row then costs a full training
            # step while teaching nothing, with no sign of it in the loss.
            raise ValueError(
                "row on source line %s is %d tokens, over max_seq_length %d. Truncating it would "
                "drop the supervised span; fix the row or raise the limit"
                % (row.get("source_line", "?"), len(input_ids), max_seq_length)
            )
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
        supervised += len(completion_ids)
        total += len(input_ids)
        examples.append((input_ids, labels))
    return examples, supervised, total, {
        "ids": list(terminator),
        "source": terminator_source,
        "text": tok.decode(terminator) if terminator else "",
    }


def epoch_schedule(epochs, n_examples, batch_size):
    """[(number, portion, steps)] for a possibly fractional number of epochs.

    A fractional `epochs` is a real request over a large corpus, and rounding it
    down would train for two thirds of what the spec says while the manifest
    records the spec's figure.
    """
    steps_per_epoch = -(-n_examples // batch_size)
    whole = int(epochs)
    schedule = [(index + 1, 1.0, steps_per_epoch) for index in range(whole)]
    tail_steps = int(round((epochs - whole) * steps_per_epoch))
    if tail_steps > 0:
        schedule.append((whole + 1, epochs - whole, tail_steps))
    if not schedule:
        raise ValueError(
            "epochs %s over %d examples in batches of %d schedules no training steps at all"
            % (epochs, n_examples, batch_size)
        )
    return schedule


def batches(examples, size, pad_id, torch):
    """Pad to the longest member of each batch. Padding is masked out of the loss."""
    for start in range(0, len(examples), size):
        chunk = examples[start : start + size]
        width = max(len(ids) for ids, _ in chunk)
        input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in chunk]
        labels = [lab + [IGNORE_INDEX] * (width - len(lab)) for _, lab in chunk]
        attention = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in chunk]
        yield (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def carry_back_sentencepiece(tok, model_dir, model_id, revision):
    """Put `tokenizer.model` back beside the checkpoint. Returns what happened, as a string.

    `transformers` 5.x `save_pretrained` no longer writes `tokenizer.model`, and
    the tokenizer classes no longer expose `vocab_file`. The exporter's
    SentencePiece branch tests for exactly those, so without this every bundle
    silently gets an HF tokenizer section instead of `SP_Tokenizer` -- and
    LiteRT-LM's FST-constrained decoding is SentencePiece-only, so the artifact
    runs, scores the same, passes every liveness check, and cannot do
    constrained tool-calling. Nothing observable fails.

    The measurement harness has carried the file back since 2026-09-02 and its
    bundles read `Data Type: SP_Tokenizer`; the package did not, so every number
    this project published came from an artifact shaped unlike the one a user
    would get.

    A model with no `tokenizer.model` at all -- Qwen's tokenizer is BPE -- is not
    a failure. It is reported as such and the bundle records an HF tokenizer
    section honestly.
    """
    import shutil

    destination = Path(model_dir) / "tokenizer.model"
    if destination.exists():
        return "already present"

    source = None
    local = Path(getattr(tok, "name_or_path", "") or "")
    if local.is_dir() and (local / "tokenizer.model").exists():
        source = local / "tokenizer.model"
    else:
        try:
            from huggingface_hub import hf_hub_download

            source = Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename="tokenizer.model",
                    **({"revision": revision} if revision else {}),
                )
            )
        except Exception as exc:  # noqa: BLE001 -- any failure here is "no file"
            return f"unavailable: {type(exc).__name__}"

    try:
        shutil.copyfile(source, destination)
    except OSError as exc:
        return f"could not copy: {exc}"

    # The exporter reads `vocab_file` from the config, and an absolute path is
    # what the harness declared; a bare filename is not resolved from there.
    config = Path(model_dir) / "tokenizer_config.json"
    try:
        data = json.loads(config.read_text(encoding="utf-8")) if config.exists() else {}
        data["vocab_file"] = str(destination.resolve())
        config.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        return f"copied, but tokenizer_config.json was not updated: {exc}"
    return "carried back"


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])

    rows = [
        json.loads(line)
        for line in Path(spec["data"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("the training split is empty")

    from_kwargs = {"revision": spec["revision"]} if spec.get("revision") else {}
    tok = AutoTokenizer.from_pretrained(spec["model"], **from_kwargs)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise ValueError("the tokenizer has neither a pad token nor an eos token to pad with")

    runtime_rendered = spec["prompt_mode"] == "runtime_rendered"
    examples, supervised, total, terminator = build_examples(
        tok, rows, spec["max_seq_length"], runtime_rendered
    )

    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        dtype=getattr(torch, spec["dtype"]),
        attn_implementation=spec["attn_implementation"],
        **from_kwargs
    )
    model.config.use_cache = False

    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if spec["method"] == "lora":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=spec["lora_rank"],
                lora_alpha=spec["lora_alpha"],
                lora_dropout=spec["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(spec["lora_targets"]),
            ),
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.train()
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=spec["learning_rate"]
    )

    schedule = epoch_schedule(spec["epochs"], len(examples), spec["batch_size"])

    epochs = []
    for number, portion, limit in schedule:
        random.shuffle(examples)
        running = 0.0
        steps = 0
        for input_ids, attention, labels in batches(
            examples, spec["batch_size"], pad_id, torch
        ):
            if steps >= limit:
                break
            # The mask is applied here and nowhere else: transformers computes a
            # shifted cross-entropy that skips every IGNORE_INDEX position, so
            # the prompt contributes no gradient.
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            out.loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            running += float(out.loss.detach())
            steps += 1
        epochs.append(
            {
                "epoch": number,
                "portion": portion,
                "loss": running / steps if steps else None,
                "steps": steps,
            }
        )

    model_dir = Path(spec["model_dir"])
    adapter_dir = Path(spec["adapter_dir"]) if spec.get("adapter_dir") else None
    if spec["method"] == "lora":
        # The adapter is saved before the merge, as an artifact of its own. A
        # merged checkpoint cannot be un-merged, and an adapter that lived only
        # in a temp directory is a rank-16 delta nobody can inspect, re-apply to
        # a different base, or ship on its own.
        model.save_pretrained(str(adapter_dir))
        merged = model.merge_and_unload()
        merged.save_pretrained(str(model_dir))
    else:
        model.save_pretrained(str(model_dir))
    tok.save_pretrained(str(model_dir))
    sentencepiece = carry_back_sentencepiece(tok, model_dir, spec["model"], spec.get("revision"))

    # Beside the checkpoint, because a directory names nothing. `convert` keys
    # the per-family export flags on the model's name, and `config.json` no
    # longer carries one: transformers 5.x deletes `_name_or_path` on save. A
    # checkpoint that cannot say what it came from is exported without the flags
    # its family requires -- and the export succeeds, so nothing says otherwise.
    (Path(model_dir) / "litetune.json").write_text(
        json.dumps(
            {
                "base_model": spec["model"],
                "base_model_revision": spec.get("revision"),
                "prompt_mode": spec["prompt_mode"],
                "turn_terminator": terminator,
                "sentencepiece": sentencepiece,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    Path(spec["metrics_out"]).write_text(
        json.dumps(
            {
                # What this was trained from. Recorded because the checkpoint
                # this run produces is a directory, and a directory carries no
                # identity: `convert` keys its per-family export flags on the
                # model's name, and without this it has nothing to key on --
                # `config.json` cannot serve, since FunctionGemma and plain
                # Gemma 3 declare the same `model_type` and need different
                # overrides.
                "base_model": spec["model"],
                "base_model_revision": spec.get("revision"),
                "method": spec["method"],
                "learning_rate": spec["learning_rate"],
                "dtype": spec["dtype"],
                "attn_implementation": spec["attn_implementation"],
                # Whether the SentencePiece model made it back beside the
                # checkpoint. Recorded, not assumed: the difference between
                # `SP_Tokenizer` and an HF section is invisible in every
                # measurement and decides whether constrained decoding works.
                "sentencepiece": sentencepiece,
                # What the model now expects at serving time. The bundle's
                # contract has to state this, and this is where it is observed
                # rather than asserted.
                "prompt_mode": spec["prompt_mode"],
                "n_examples": len(examples),
                "supervised_tokens": supervised,
                "total_tokens": total,
                "supervised_token_fraction": (supervised / total) if total else None,
                "masked_tokens": total - supervised,
                # Which terminator the model was trained to emit, and where it
                # came from. A model trained on one terminator while the runtime
                # waits for another does not stop -- and that shows up as extra
                # tool calls firing on the device, not as a lower score.
                "turn_terminator": terminator,
                "trainable_parameters": trainable,
                "base_parameters": trainable_before,
                "epochs": epochs,
                "model_dir": str(model_dir),
                "adapter_dir": str(adapter_dir) if adapter_dir else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneRequest:
    """One training run.

    `dtype` and `attn_implementation` default to the pair the export and
    evaluation paths use. They are recorded rather than assumed because a
    checkpoint trained under one attention implementation and served under
    another produces output that is fluent, wrong, and passes every check that
    does not involve held-out labels.
    """

    model: str
    data: Path
    output_dir: Path
    method: str = "full"
    revision: str | None = None
    # None means "the rate for this method" -- see `LEARNING_RATES`. A shared
    # default here is how a full-against-LoRA comparison ends up measuring the
    # learning rate instead of the method.
    learning_rate: float | None = None
    epochs: float = 1.0
    batch_size: int = 8
    max_seq_length: int = 1024
    seed: int = 0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS
    dtype: str = DEFAULT_DTYPE
    attn_implementation: str = DEFAULT_ATTN_IMPLEMENTATION
    # Which of the two prompt conventions this run trains the model into, and
    # the *only* place the answer is decided. It is a training parameter, not a
    # serving one: a model trained on prompts that already contain the
    # declarations has learned a different input distribution from one trained
    # under the runtime's own chat template. From here it travels into
    # `bundle.Contract.prompt_mode` and back out through
    # `evaluate.resolve_prompt_mode`, so nothing downstream has to guess.
    #
    # `--no-template` on the serving side is right for a hand-rendered wire
    # format and wrong for anything else, and it was carried everywhere by
    # habit once.
    # Required and keyword-only. Which convention the prompt is built under is
    # not something this code can guess, and guessing wrong trains the model on
    # a prompt the runtime never sends -- silently, with a loss curve that looks
    # fine. The CLI always required it; the library API said so in a comment and
    # then supplied `PRERENDERED` anyway. Stating it as a field with no default
    # makes the type non-optional, so nothing downstream carries a None case.
    prompt_mode: PromptMode = field(kw_only=True)
    timeout_s: int = DEFAULT_TIMEOUT_S
    env: envs.StageEnv = envs.TRAIN
    auto_provision: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise TuneError(f"method must be one of {list(METHODS)}, got {self.method!r}")
        if not isinstance(self.prompt_mode, PromptMode):
            raise TuneError(
                f"prompt_mode must be a PromptMode, got {self.prompt_mode!r}. The two conventions "
                "are mutually exclusive and a model trained under one cannot be served under the "
                f"other; known modes: {[m.value for m in PromptMode]}"
            )
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise TuneError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise TuneError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise TuneError(f"batch_size must be at least 1, got {self.batch_size}")
        if self.max_seq_length < 1:
            raise TuneError(f"max_seq_length must be at least 1, got {self.max_seq_length}")
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "lora_targets", tuple(self.lora_targets))

    @property
    def rate(self) -> float:
        """The rate this run uses: the declared one, or the method's default."""
        return self.learning_rate if self.learning_rate is not None else LEARNING_RATES[self.method]

    @property
    def rate_is_default(self) -> bool:
        return self.learning_rate is None

    @property
    def model_dir(self) -> Path:
        """The merged, exportable checkpoint. What `convert` is pointed at."""
        return self.output_dir / "model"

    @property
    def adapter_dir(self) -> Path | None:
        """Where the adapter is kept. An artifact, not scratch -- see the script."""
        return self.output_dir / "adapter" if self.method == "lora" else None

    def config(self, metrics_out: Path) -> dict[str, Any]:
        """Everything the generated script needs. Also what the report records."""
        return {
            "model": self.model,
            "revision": self.revision,
            "data": str(self.data),
            "method": self.method,
            "learning_rate": self.rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_targets": list(self.lora_targets),
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "prompt_mode": self.prompt_mode.value,
            "model_dir": str(self.model_dir),
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "metrics_out": str(metrics_out),
        }

    def as_dict(self) -> dict[str, Any]:
        record = self.config(self.output_dir / "metrics.json")
        record["learning_rate_source"] = (
            f"default for method {self.method!r}" if self.rate_is_default else "declared"
        )
        record["environment"] = {
            "name": self.env.name,
            "identity": self.env.identity,
            "requirements": list(self.env.requirements),
        }
        return record


# ---------------------------------------------------------------------------
# What the run reported about itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochMetrics:
    """One epoch, or the fraction of one a non-integer `epochs` asked for."""

    epoch: int
    loss: float | None
    steps: int
    portion: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "steps": self.steps,
            "portion": self.portion,
        }


@dataclass(frozen=True)
class TrainingMetrics:
    """What the training script measured about its own run.

    `supervised_token_fraction` is the field this class exists for. Everything
    else here is provenance; that one is the observable that separates a working
    run from the one that scored nine times worse than its own base while
    reporting a better loss.
    """

    n_examples: int
    supervised_tokens: int
    total_tokens: int
    masked_tokens: int
    supervised_token_fraction: float | None
    epochs: tuple[EpochMetrics, ...]
    trainable_parameters: int | None = None
    base_parameters: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def final_loss(self) -> float | None:
        return self.epochs[-1].loss if self.epochs else None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainingMetrics:
        """Parse the script's metrics file. Raises `KeyError`/`TypeError` if it is not one."""
        epochs = tuple(
            EpochMetrics(
                epoch=int(e["epoch"]),
                loss=None if e.get("loss") is None else float(e["loss"]),
                steps=int(e.get("steps", 0)),
                portion=float(e.get("portion", 1.0)),
            )
            for e in data["epochs"]
        )
        fraction = data["supervised_token_fraction"]
        return cls(
            n_examples=int(data["n_examples"]),
            supervised_tokens=int(data["supervised_tokens"]),
            total_tokens=int(data["total_tokens"]),
            masked_tokens=int(data["masked_tokens"]),
            supervised_token_fraction=None if fraction is None else float(fraction),
            epochs=epochs,
            trainable_parameters=data.get("trainable_parameters"),
            base_parameters=data.get("base_parameters"),
            raw=dict(data),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "supervised_tokens": self.supervised_tokens,
            "masked_tokens": self.masked_tokens,
            "total_tokens": self.total_tokens,
            "supervised_token_fraction": self.supervised_token_fraction,
            "expected_supervised_token_fraction": EXPECTED_SUPERVISED_FRACTION,
            "trainable_parameters": self.trainable_parameters,
            "base_parameters": self.base_parameters,
            "epochs": [e.as_dict() for e in self.epochs],
            "final_loss": self.final_loss,
        }


def read_metrics(path: Path) -> TrainingMetrics:
    """Read the script's metrics file. Raises if it is absent or not what it claims."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TrainingMetrics.from_dict(payload)


def masking_check(metrics: TrainingMetrics | None, detail_if_missing: str) -> Check:
    """The one check in this module that can stop a pipeline.

    A fraction near 1.0 means `labels` was never masked, and the run is the
    measured failure in the module docstring: lower training loss, 0.0625 exact
    match against a 0.5625 base, every label-free check green.
    """
    if metrics is None or metrics.supervised_token_fraction is None:
        # No number means the masking was not observed. That is not evidence
        # that it worked, and it is not evidence that it did not.
        return Check.unchecked(MASKING_CHECK, detail_if_missing)

    fraction = metrics.supervised_token_fraction
    observed = {
        "supervised_token_fraction": round(fraction, 6),
        "expected": EXPECTED_SUPERVISED_FRACTION,
        "supervised_tokens": metrics.supervised_tokens,
        "masked_tokens": metrics.masked_tokens,
        "total_tokens": metrics.total_tokens,
    }
    if metrics.masked_tokens <= 0 or fraction >= MASKING_NOT_APPLIED_ABOVE:
        return Check.failed(
            MASKING_CHECK,
            f"loss was computed on {fraction:.4f} of tokens ({metrics.supervised_tokens} of "
            f"{metrics.total_tokens}); at or above {MASKING_NOT_APPLIED_ABOVE} the prompt was not "
            "masked. On this data shape the declarations are ~330 of ~350 tokens, so an unmasked "
            "run spends ~94% of its gradient memorising a header the model already has on its "
            "input. The measured outcome was a *lower* training loss (0.50 against 1.45) and "
            "0.0625 exact match against a 0.5625 base -- nine times worse than not training",
            observed=observed,
        )
    return Check.passed(
        MASKING_CHECK,
        f"loss was computed on {fraction:.4f} of tokens ({metrics.supervised_tokens} of "
        f"{metrics.total_tokens}); the prompt contributed no gradient "
        f"(expected ~{EXPECTED_SUPERVISED_FRACTION} on this data shape)",
        observed=observed,
    )


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class TuneResult:
    """What the run produced, and what none of it has been shown to be."""

    request: TuneRequest
    checks: CheckSet
    metrics: TrainingMetrics | None = None
    model_dir: Path | None = None
    adapter_dir: Path | None = None
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
        """Always False. A property, not a field, so nothing can set it True.

        Training can establish that a process finished and wrote a checkpoint.
        Establishing that the checkpoint is better than what it started from
        takes held-out data, and this module has none.
        """
        return False

    @property
    def prompt_mode(self) -> PromptMode:
        """The convention this checkpoint now expects at serving time.

        Surfaced here because `bundle.Contract` requires it and refuses to
        default it: this is the stage that decided it, so this is where a bundle
        should read it from rather than a human retyping it.
        """
        return self.request.prompt_mode

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TUNE_SCHEMA,
            "verified": False,
            "unverified_reason": NOT_VERIFIED,
            "outcome": self.outcome.value,
            # Top-level as well as inside `request`: this is the field a bundle's
            # contract is built from, and it must not be something a reader has
            # to go looking for.
            "prompt_mode": self.prompt_mode.value,
            "request": self.request.as_dict(),
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "model_dir": str(self.model_dir) if self.model_dir else None,
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "returncode": self.returncode,
            "seconds": round(self.seconds, 3) if self.seconds is not None else None,
            "checks": self.checks.as_dict(),
            "stderr": self.stderr,
            "stdout_tail": self.stdout_tail,
            "limitations": list(self.limitations),
        }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _emit_epochs(events: EventStream, metrics: TrainingMetrics) -> None:
    """Per-epoch numbers into the event stream.

    Emitted after the fact rather than streamed, because `StageEnv.run` captures
    a subprocess's output and returns it whole. The alternative -- a live pipe --
    would put a second way for a stage to report progress into the codebase, and
    the reason events exist at all is that there was previously more than one.
    """
    for epoch in metrics.epochs:
        if epoch.loss is None:
            events.note(f"epoch {epoch.epoch}: no steps ran", epoch=epoch.epoch)
            continue
        events.metric("train.loss", epoch.loss, epoch=epoch.epoch, steps=epoch.steps)
    if metrics.supervised_token_fraction is not None:
        events.metric(
            "supervised_token_fraction",
            round(metrics.supervised_token_fraction, 6),
            expected=EXPECTED_SUPERVISED_FRACTION,
        )


def _directory_is_populated(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def run_tune(request: TuneRequest, events: EventStream | None = None) -> TuneResult:
    """Fine-tune inside `envs.TRAIN`. A non-zero exit is recorded, never raised."""
    events = events or EventStream(echo_json=False)
    events.stage_started(
        "train",
        model=request.model,
        method=request.method,
        learning_rate=request.rate,
    )
    result = TuneResult(request=request, checks=CheckSet(name=f"train:{request.model}"))
    result.limitation(NOT_VERIFIED)
    if request.dtype == DEFAULT_DTYPE:
        # Named because it is a real cost of the bfloat16 default: the
        # parameters are bfloat16, so AdamW's moments are bfloat16 too. A
        # mixed-precision setup would keep an fp32 master copy and update more
        # precisely. bfloat16 is the training default, but it is not matched
        # downstream: export.py sets no dtype at all, and evaluate.py's float
        # reference loads with `torch_dtype=torch.float32` regardless of what
        # trained the checkpoint under test. So a run at the default dtype
        # still carries a dtype difference against the reference it is scored
        # on -- whatever originally decided the default, matching export and
        # evaluation was not it.
        result.limitation(
            "training runs with bfloat16 parameters, so the optimiser's moments are bfloat16 as "
            "well; updates are coarser than a mixed-precision run with an fp32 master copy. "
            "bfloat16 is the training default, but the export and evaluation paths do not load "
            "in it -- evaluation's float reference loads at float32 regardless of the training "
            "dtype -- so a run at the default dtype still carries a dtype difference against the "
            "reference it is scored on. What originally decided the bfloat16 default is not "
            "established here"
        )
    if (request.dtype, request.attn_implementation) != (
        DEFAULT_DTYPE,
        DEFAULT_ATTN_IMPLEMENTATION,
    ):
        result.limitation(
            f"this run trains with dtype {request.dtype!r} and attention "
            f"{request.attn_implementation!r}, not the {DEFAULT_DTYPE}/"
            f"{DEFAULT_ATTN_IMPLEMENTATION} pair the export and evaluation paths use. A "
            "checkpoint served under a different attention implementation than it was trained "
            "with produces output that is fluent, wrong, and passes every check that does not "
            "involve held-out labels"
        )
    if request.rate_is_default:
        events.note(
            f"learning rate {request.rate} (default for method {request.method!r})",
            learning_rate=request.rate,
            method=request.method,
        )

    # -- can this run at all? ---------------------------------------------
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
                        "requirements": list(request.env.requirements),
                    },
                )
            )
        else:
            sink.append(
                Check.unchecked(
                    ENV_CHECK,
                    f"environment {request.env.name!r} is not provisioned at {request.env.path}",
                    observed={"name": request.env.name, "identity": request.env.identity},
                )
            )
    environment = result.checks.add(sink[0])
    events.check(environment)
    if not environment.conclusive:
        # Nothing was attempted, so nothing failed. Recording a training failure
        # here would be a verdict about the model drawn from a fact about this
        # machine.
        result.limitation(f"training was not attempted: {environment.detail}")
        events.stage_finished(result.outcome.value, attempted=False)
        return result

    # -- does this environment's transformers support this model? -----------
    # Cheap, and it is checked here because the alternative is an
    # `AttributeError: 'list' object has no attribute 'keys'` from inside a
    # tokenizer load -- which arrives after the environment has been built and
    # the checkpoint downloaded, and reads as a litetune bug rather than a
    # version requirement.
    rules = models.identify(request.model)
    if rules is not None and rules.min_transformers:
        version_check = models.transformers_check(
            request.model,
            rules,
            models.declared_version(request.env.requirements),
            f"the {request.env.name} environment",
            unknown_reason="its requirements pin no transformers version",
        )
        events.check(version_check)
        if version_check.conclusive:
            result.checks.add(version_check)
        else:
            result.limitation(version_check.detail)
        if version_check.outcome is Outcome.FAILED:
            result.limitation(f"training was not attempted: {version_check.detail}")
            events.stage_finished(result.outcome.value, attempted=False)
            return result

    if not request.data.is_file():
        # A check, not an exception: "the split is not there" is an observation
        # about this run, and the report has to carry it.
        missing = Check.failed(
            TRAINING_CHECK,
            f"the training split {request.data} does not exist, so nothing was trained",
            observed={"data": str(request.data)},
        )
        result.checks.add(missing)
        events.check(missing)
        events.stage_finished(result.outcome.value, attempted=False)
        return result

    # -- run it ------------------------------------------------------------
    # The generated script and its config are written beside the checkpoint and
    # kept. They are the record of exactly what ran: a checkpoint whose training
    # script is gone is a checkpoint nobody can reproduce or diff against the
    # next one.
    workspace = request.output_dir
    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / "train_script.py"
    script.write_text(_TRAIN_SCRIPT, encoding="utf-8")
    metrics_out = workspace / "metrics.json"
    config_path = workspace / "train_config.json"
    config_path.write_text(json.dumps(request.config(metrics_out), indent=2), encoding="utf-8")
    # A previous attempt's metrics in place would be read as this attempt's, the
    # same way `export` refuses to inherit a stale artifact by mtime.
    metrics_out.unlink(missing_ok=True)

    events.note(
        f"training {request.method} on {request.data.name}",
        method=request.method,
        learning_rate=request.rate,
        epochs=request.epochs,
    )
    started = time.perf_counter()
    try:
        proc = request.env.run(["python", str(script), str(config_path)], timeout=request.timeout_s)
    except subprocess.TimeoutExpired:
        seconds = time.perf_counter() - started
        logger.warning("training timed out after %ss", request.timeout_s)
        # From out here a hang is indistinguishable from a stalled machine, so
        # it is recorded as not performed rather than as a verdict.
        timed_out = Check.unchecked(
            TRAINING_CHECK,
            f"no result after {request.timeout_s}s (timeout): the run did not finish, which says "
            "nothing about the method or the data",
            observed={"timeout_s": request.timeout_s},
        )
        result.seconds = seconds
        result.checks.add(timed_out)
        events.check(timed_out)
        result.checks.add(masking_check(None, "the run did not finish, so no mask was observed"))
        events.stage_finished(result.outcome.value, attempted=True)
        return result
    except OSError as exc:
        logger.exception("could not start the training script")
        blocked = Check.unchecked(
            TRAINING_CHECK,
            f"the training script could not be started: {type(exc).__name__}: {exc}",
        )
        result.seconds = time.perf_counter() - started
        result.checks.add(blocked)
        events.check(blocked)
        result.checks.add(masking_check(None, "the run never started, so no mask was observed"))
        events.stage_finished(result.outcome.value, attempted=False)
        return result

    result.seconds = time.perf_counter() - started
    result.returncode = proc.returncode
    result.stderr = proc.stderr or ""
    result.stdout_tail = (proc.stdout or "")[-_STDOUT_TAIL:]

    # -- what did it report about itself? ----------------------------------
    metrics_error = ""
    if metrics_out.is_file():
        try:
            result.metrics = read_metrics(metrics_out)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.exception("training wrote a metrics file that could not be read")
            metrics_error = f"{type(exc).__name__}: {exc}"
    else:
        metrics_error = f"no metrics file at {metrics_out}"
    if result.metrics is not None:
        _emit_epochs(events, result.metrics)

    reading = read_returncode(proc.returncode)
    if not reading.conclusive:
        # Killed, not failed. A training run is the longest-lived and hungriest
        # process litetune starts, so it is the likeliest to meet the
        # out-of-memory killer; reading `-9` as a training failure would blame
        # the method or the data for a fact about the machine. See
        # `litetune.exits`.
        killed = Check.unchecked(
            TRAINING_CHECK,
            f"training was {reading.describe('the method or the data')}. "
            f"stderr: {_tail(result.stderr) or 'none'}",
            observed=reading.as_dict() | {"seconds": round(result.seconds, 3)},
        )
        result.checks.add(killed)
        events.check(killed)
        result.checks.add(
            masking_check(result.metrics, "the run was killed, so no mask was observed")
        )
        events.stage_finished(result.outcome.value, attempted=True)
        return result

    if proc.returncode != 0:
        failed = Check.failed(
            TRAINING_CHECK,
            f"training exited {proc.returncode}: {_tail(result.stderr) or 'no stderr'}",
            observed={
                "returncode": proc.returncode,
                "seconds": round(result.seconds, 3),
                "stderr_tail": _tail(result.stderr, 2000),
            },
        )
    elif not _directory_is_populated(request.model_dir):
        # Exit zero and no checkpoint is the documented shape of this toolchain's
        # failures, and it is the reason `export` measures the same thing.
        failed = Check.failed(
            TRAINING_CHECK,
            f"training exited zero but wrote no checkpoint into {request.model_dir}",
            observed={"returncode": 0, "model_dir": str(request.model_dir)},
        )
    else:
        result.model_dir = request.model_dir
        failed = Check.passed(
            TRAINING_CHECK,
            f"{request.method} fine-tune finished in {result.seconds:.1f}s at learning rate "
            f"{request.rate:g} — trained, not verified",
            observed={
                "returncode": 0,
                "seconds": round(result.seconds, 3),
                "model_dir": str(request.model_dir),
                "learning_rate": request.rate,
                "final_loss": result.metrics.final_loss if result.metrics else None,
                "verified": False,
            },
        )
    result.checks.add(failed)
    events.check(failed)
    if result.model_dir is not None:
        events.artifact(str(result.model_dir), name="model", verified=False)

    # -- was the loss actually masked? -------------------------------------
    mask = masking_check(
        result.metrics,
        f"the run reported no supervised-token fraction ({metrics_error or 'field absent'}), so "
        "whether the prompt was masked out of the loss is unknown. An unmasked run is the one "
        "that scored 0.0625 against a 0.5625 base",
    )
    result.checks.add(mask)
    events.check(mask)
    if mask.outcome is Outcome.FAILED:
        result.limitation(
            "the loss was not masked to the completion; this checkpoint is expected to be worse "
            "than the model it started from, and its training loss will not show it"
        )

    # -- the adapter, and the merge ----------------------------------------
    result.checks.add(_merge_check(request, result, events))

    if metrics_error and result.metrics is None:
        result.limitation(
            f"training reported no metrics ({metrics_error}); the supervised-token fraction, the "
            "per-epoch losses and the trainable-parameter count are all unavailable for this run"
        )

    events.stage_finished(
        result.outcome.value,
        method=request.method,
        returncode=result.returncode,
        verified=False,
    )
    return result


def _merge_check(request: TuneRequest, result: TuneResult, events: EventStream) -> Check:
    """For LoRA: the adapter is kept and the merged checkpoint is a separate thing.

    `convert` cannot take an adapter, so the merge has to happen before export.
    Keeping the adapter afterwards is not tidiness: a merged checkpoint cannot be
    un-merged, and without the adapter there is no way to re-apply the delta to a
    different base revision, inspect what was learned, or ship the 30 MB instead
    of the 500 MB.
    """
    if request.method != "lora":
        return Check.passed(
            MERGE_CHECK,
            "a full fine-tune produces the checkpoint directly; there is no adapter to merge",
            observed={"method": request.method},
        )

    adapter = request.adapter_dir
    if adapter is None or not _directory_is_populated(adapter):
        return Check.failed(
            MERGE_CHECK,
            f"no adapter was retained at {adapter}: the merged checkpoint cannot be un-merged, so "
            "the learned delta is unrecoverable and cannot be re-applied to another base revision",
            observed={"adapter_dir": str(adapter) if adapter else None},
        )
    result.adapter_dir = adapter
    events.artifact(str(adapter), name="adapter", verified=False)
    if not _directory_is_populated(request.model_dir):
        return Check.failed(
            MERGE_CHECK,
            f"the adapter was written to {adapter} but no merged checkpoint is at "
            f"{request.model_dir}; conversion cannot read an adapter",
            observed={"adapter_dir": str(adapter), "model_dir": str(request.model_dir)},
        )
    return Check.passed(
        MERGE_CHECK,
        f"adapter retained at {adapter} and merged into {request.model_dir}",
        observed={"adapter_dir": str(adapter), "model_dir": str(request.model_dir)},
    )


def write_report(result: TuneResult, path: Path | None = None) -> Path:
    """Persist the training report. Written for a failed run too."""
    target = path or result.request.output_dir / "tune.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    return target
