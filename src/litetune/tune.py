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
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs, models
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.declarations import (
    DeclarationsError,
    entry_count,
    read_declarations,
    recorded_digest,
)
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.metrics import START_CALL, ToolCall, runtime_calls, text_after_the_calls
from litetune.models import (
    PROVENANCE_NAME,
    identify,
    renders_declarations_for,
    wire_format_for,
)
from litetune.prepare import (
    CONTROL_TEXT,
    PrepareError,
    Row,
    control_text_held,
    read_rows,
    refuse_undeclared_tools,
    render_call,
)
from litetune.prompt_mode import RENDERING_SOURCE, PromptMode, PromptModeDecision, prompt_evidence

logger = logging.getLogger(__name__)

TUNE_SCHEMA = "litetune.tune/1"

# The check the prompt-mode decision is recorded under, and the flag that keeps a
# declared mode the training prompts contradict. Library messages name the flag
# too: a refusal is read by whoever typed the command.
PROMPT_MODE_CHECK = "prompt mode"
DECLARATIONS_CHECK = "tool declarations"
CALLS_CHECK = "calls are written as the runtime reads them"
COMPLETIONS_CHECK = "every row carries its completion"

# A name no declaration could plausibly use, rendered only to ask the template how a call ends.
CALL_PROBE_NAME = "litetune_probe"


def _refuse_calls_without_declarations(
    request: TuneRequest, mode: PromptMode, rows: Sequence[Row]
) -> Check | None:
    """A structured target for a declaration-rendering family, with none supplied.

    `prepare` refuses the same thing, and this is not a duplicate: a split
    written by hand reaches `tune` without passing through that stage, and the
    defect it prevents -- training an answer to a prompt 730 characters shorter
    than the one the runtime sends -- is invisible in a loss curve.

    Only in `runtime_rendered`, because only there does the runtime render the
    declarations. In `prerendered` the application has already put them into
    the prompt -- flutter_gemma does that for FunctionGemma -- so the prompt is
    the declaration and there is nothing missing. The first version of this
    asked only the family, and refused the README's own walkthrough.
    """
    if request.declarations is not None or mode is not PromptMode.RUNTIME_RENDERED:
        return None
    renders, reason = renders_declarations_for(request.model)
    if not renders:
        return None
    if not any(isinstance(row.target, ToolCall) for row in rows):
        return None
    return Check.failed(
        DECLARATIONS_CHECK,
        f"this split trains tool calls for {request.model}, whose runtime renders the tool "
        "declarations into the prompt, and no declarations were given. Pass --declarations with "
        "the same JSON bundle takes; they cannot be derived from the targets, because the "
        f"descriptions, types and required lists are the application's contract. {reason}",
        observed={"model": request.model, "declarations": None},
    )


def _refuse_declarations_for_an_unrecorded_tool_channel(
    request: TuneRequest, mode: PromptMode
) -> Check | None:
    """Declarations for a `runtime_rendered` split of a family litetune records no tool channel for.

    Training renders the declarations into the prompt the way the family's
    runtime is recorded to; for a family with no such record -- one litetune
    has not measured, or one it cannot tell -- there is no measured prompt to
    train, and `verify` would measure the candidate on whatever the runtime
    sends instead. Refused rather than trained, as the prompt-mode disagreement
    is; a split whose application renders the tools itself is `prerendered`.
    """
    if request.declarations is None or mode is not PromptMode.RUNTIME_RENDERED:
        return None
    renders, _ = renders_declarations_for(request.model)
    if renders:
        return None
    rules = identify(request.model)
    # What litetune does not know, said as that: a runtime that does render
    # tools (Gemma 4's does) may be one litetune has simply not measured.
    unknown = (
        f"litetune cannot tell which model family {request.model} is"
        if rules is None or rules.config_only
        else f"litetune records no tool channel for {rules.family}"
    )
    return Check.failed(
        DECLARATIONS_CHECK,
        f"--declarations was given for a runtime_rendered split of {request.model}, and "
        f"{unknown}: it has not measured how that runtime renders tool declarations, so it "
        "cannot train the prompt the runtime sends. If your application renders the tools into "
        "the prompt itself, the split is prerendered",
        observed={"model": request.model, "declarations": str(request.declarations)},
    )


def _config_ambiguous(model: str) -> bool:
    """Whether `model` is a checkpoint whose config names two families.

    `gemma3_text` is FunctionGemma and Gemma 3 alike, and nothing in the config
    tells them apart; `litetune.json` beside it does. A model litetune has no
    entry for at all is another statement: its calls are the caller's text.
    """
    rules = identify(model)
    return rules is not None and rules.config_only


def _marks_calls(rows: Sequence[Row]) -> bool:
    """Whether a call row's completion carries FunctionGemma's call markers."""
    return any(isinstance(row.target, ToolCall) and START_CALL in row.completion for row in rows)


def _refuse_rows_without_a_completion(request: TuneRequest, rows: Sequence[Row]) -> Check | None:
    """A row that gives a target and no completion.

    The training script trains the completion a row carries -- `build_examples`
    reads `row["completion"]` from the file -- so such a row failed there, as a
    `KeyError`, after the training environment was provisioned. `prepare`
    writes the completion from the target.
    """
    bare = [row for row in rows if row.rendered]
    if not bare:
        return None
    return Check.failed(
        COMPLETIONS_CHECK,
        f"{request.data}:{bare[0].lineno}: the row gives a target and no completion, and `tune` "
        f"trains the completion a row carries ({len(bare)} row(s) like it). Run prepare on the "
        "file, which writes each completion from its target, and train the split it writes",
        observed={"data": str(request.data), "line": bare[0].lineno, "rows": len(bare)},
    )


def _refuse_calls_of_a_family_litetune_cannot_tell(
    request: TuneRequest, mode: PromptMode, rows: Sequence[Row]
) -> Check | None:
    """Marked calls, `runtime_rendered`, for a checkpoint whose config names two families.

    The calls say FunctionGemma and the config says FunctionGemma or Gemma 3;
    whether the runtime renders the declarations into the prompt, and how a
    call turn ends, are recorded per family. Trained with `--declarations` the
    prompt may carry a turn the runtime never sends, and trained without, it
    may lack one the runtime always sends. Refused either way, naming the
    record that settles it -- which does, for such a checkpoint.
    """
    if mode is not PromptMode.RUNTIME_RENDERED or not _config_ambiguous(request.model):
        return None
    if not _marks_calls(rows):
        return None
    return Check.failed(
        DECLARATIONS_CHECK,
        f"litetune cannot tell which model family {request.model} is, and this split's calls "
        "carry FunctionGemma's call markers. Whether a runtime renders the tool declarations "
        "into the prompt is recorded per family, so the prompt this run would train cannot be "
        "told from the one the runtime sends. Record which model the checkpoint is in "
        f"{PROVENANCE_NAME} beside its config.json, as a checkpoint litetune trained does "
        '(`{"base_model": "<model id>"}`)',
        observed={"model": request.model, "declarations": str(request.declarations)},
    )


def _refuse_calls_the_runtime_would_not_read(
    request: TuneRequest, mode: PromptMode, rows: Sequence[Row]
) -> Check | None:
    """A call row whose completion the runtime would not read as its target.

    The training script ends a call row with what the chat template puts after
    a call, and that ending is only right after a call. The runtime reads a
    reply's calls only between `<start_function_call>` and
    `<end_function_call>`, each block one whole call (`parser_utils.cc`); a
    model trained on calls without the markers, closed by `<end_of_turn>`,
    returned no call on 5 of 5 prompts, and splits prepared by litetune 0.1.6
    or earlier carry none. So a completion has to read, the way
    `metrics.runtime_calls` reads a reply, as exactly its target's call, with
    the target's JSON types: an escaped `7` reaches an application as the
    string `"7"`. Spelling and argument order are free, as the runtime's parser
    leaves them free; only with constrained decoding on is the order held,
    which `prepare` settles for rows it renders.

    Only where the call format is FunctionGemma's, in `runtime_rendered`. Every
    call row, including one whose target litetune could not render: its own
    completion is still what the runtime would read.

    Text *before* the call trains: the runtime hands it over as the reply's
    text beside the call, an application still gets the call, and the ending
    the template writes still comes right after the call. Two reviews have
    proposed refusing it; refusing would refuse a dataset whose answers say
    something before calling, which is a choice its author is entitled to.
    """
    if mode is not PromptMode.RUNTIME_RENDERED or not wire_format_for(request.model).known:
        return None
    for row in rows:
        calls = runtime_calls(row.completion)
        if not isinstance(row.target, ToolCall):
            if calls == []:
                continue
            return Check.failed(
                CALLS_CHECK,
                f"{request.data}:{row.lineno}: the row's target is not a call, and the runtime "
                f"would read its completion {row.completion[:160]!r} as "
                + ("no reply" if calls is None else f"{calls!r}")
                + ". A call trained on a row nothing checks is a call to anything",
                observed={"data": str(request.data), "line": row.lineno},
            )
        after = text_after_the_calls(row.completion)
        held = control_text_held(row.target.raw)
        if calls == [row.target] and not after and held is None:
            continue
        if calls == [row.target] and held is not None:
            why = f"a string in it holds {held!r}: {CONTROL_TEXT[held]}"
        elif calls == [row.target]:
            why = (
                f"text follows the call ({after[:80]!r}), which the runtime hands over as the "
                "reply's text, and the call ending the template writes is right only right "
                "after a call"
            )
        elif calls is None:
            why = "the runtime would give no reply: a block between its markers is not one call"
        elif not calls:
            why = (
                "the runtime reads no call in it: it reads one only between the call markers -- "
                "splits prepared by litetune 0.1.6 or earlier carry none"
            )
        else:
            why = f"the runtime reads it as {calls!r}"
        try:
            advice = f"or write it as prepare does: {render_call(row.target)[:160]!r}"
        except ValueError as exc:
            advice = f"though prepare cannot write this target either: {exc}"
        return Check.failed(
            CALLS_CHECK,
            f"{request.data}:{row.lineno}: the completion {row.completion[:160]!r} does not train "
            f"its target's call, {row.target!r}: {why}. Drop the row's completion and run "
            f"prepare, which writes it from the target, {advice}",
            observed={"data": str(request.data), "line": row.lineno},
        )
    return None


def _refused(result: TuneResult, events: EventStream, check: Check) -> TuneResult:
    """End the stage on a refusal made before anything was attempted."""
    result.checks.add(check)
    events.check(check)
    events.stage_finished(result.outcome.value, attempted=False)
    return result


FORCE_PROMPT_MODE_FLAG = "--force-prompt-mode"

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

# Eager attention, because the export and evaluation paths use it too:
# evaluate.py's `_HF_GENERATE_SCRIPT` passes the training attention
# implementation through unconditionally, and a checkpoint served under a
# different implementation than it was trained with produces fluent garbage
# that every label-free check passes. Gemma-family models are documented as
# requiring eager.
#
# bfloat16 is the training default for a different reason -- see the two
# limitations below, at the point a run actually trains in it or departs from
# it. Unlike attention, there is no pair here for it to match: evaluate.py's
# `_HF_GENERATE_SCRIPT` loads the float reference at an unconditional float32
# regardless of the training dtype, and export.py passes no dtype to the
# exporter at all.
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
# same day rather than an occupied runner overnight. This ceiling is exactly
# the run BFLOAT16_CPU_HINT describes: on the one CPU measured, bfloat16 was
# an order of magnitude slower than float32, and a run that crawls that way is
# a realistic way to spend most of the six hours below.
DEFAULT_TIMEOUT_S = 6 * 3600

# The default dtype stands, but a CPU run is told what it is paying and which
# flag buys it back -- before the wait when the device is known early enough,
# in the run's own detail text when it is not, and as a limitation for the
# record either way. See `envs.resolve_device`.
#
# What the flag costs is deliberately not asserted here. It is not the dtype
# mismatch with export and evaluation an earlier draft named: evaluate.py's
# float reference loads at an unconditional float32 and export.py passes no
# dtype at all, so float32 has nothing to mismatch. Nor is it comparability
# with the published numbers -- MEASUREMENTS.md records exactly one run's
# dtype, and that run trained in float32 for this reason. So the hint states
# the departure and leaves the cost to the limitation `run_tune` already
# records, which says the same thing at more length.
BFLOAT16_CPU_HINT = (
    "bfloat16 matmuls were single-threaded and roughly an order of magnitude slower than "
    "float32 on the one CPU measured; --dtype float32 trains on every core instead of one, and "
    "the report records the departure from the default dtype either way"
)

# Training stderr that names the accelerator rather than the method or the
# data. A `torch.OutOfMemoryError` exits 1 like any other uncaught exception,
# and `Check.failed` on that exit is a verdict about the recipe drawn from a
# fact about the machine -- the reading `litetune.exits` exists to forbid,
# reachable here only since the run stopped being pinned to the CPU. It is
# routed to `unchecked` the same way `evaluate.py` routes `_HOST_FAILURE_RE`.
#
# An enumeration, deliberately, and not the `CUDA error:` prefix. That prefix
# would be shorter and is wrong: `CUDA error: device-side assert triggered` is
# the GPU face of an out-of-range index -- a token id past the embedding table,
# a label past the vocabulary -- and on the CPU the identical defect raises
# `IndexError: index out of range in self` and is recorded, correctly, as a
# failed training run. Matching the prefix would make one bug in the data
# produce a verdict on a laptop and a shrug on a GPU box, with the words "says
# nothing about the method or the data" attached to the one case where it says
# exactly that. `CUDA error: an illegal memory access was encountered` is the
# same story. So each entry below is a fact about the machine on its own,
# whatever wrapper text arrives around it.
#
# `OutOfMemoryError` is the class name `torch.cuda.OutOfMemoryError` binds, so
# it matches whichever alias a traceback prints. The rest cover an allocation
# that could not be served, a device that was gone or never there when the run
# reached for it, a binary with no kernel for this architecture, a driver too
# old for the runtime, and -- since CPU training is a first-class destination
# here, not an edge case -- the host running out of RAM to give torch's
# CPU allocator. Consulted only on a non-zero exit, and only against stderr:
# nothing here can change what a clean run reports.
#
# `DefaultCPUAllocator: not enough memory`, `CUBLAS_STATUS_ALLOC_FAILED` and
# `CUDNN_STATUS_ALLOC_FAILED` are the exact wording each raises, not a broader
# prefix: the same reasoning that keeps `CUDA error: device-side assert
# triggered` off this list applies here too, and a bare `not enough memory`
# risks matching a message this list was never meant to speak for.
_GPU_FAILURE_RE = re.compile(
    r"(OutOfMemoryError"
    r"|CUDA out of memory"
    r"|HIP out of memory"
    r"|CUDA error: out of memory"
    r"|CUDA driver version is insufficient"
    r"|no kernel image is available for execution"
    r"|no CUDA-capable device is detected"
    r"|invalid device ordinal"
    r"|DefaultCPUAllocator: not enough memory"
    r"|CUBLAS_STATUS_ALLOC_FAILED"
    r"|CUDNN_STATUS_ALLOC_FAILED)",
    re.IGNORECASE,
)

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

_TRAIN_SCRIPT = (
    r'''
"""Supervised fine-tuning with the loss masked to the completion.

Reads a config JSON, writes a metrics JSON. Prints nothing: the parent turns the
metrics file into events.
"""
import json
import random
import re
import sys
from pathlib import Path

# torch's own ignore index. Positions set to this contribute no gradient, and
# they are the whole mechanism by which the prompt is excluded from the loss.
IGNORE_INDEX = -100
'''
    + RENDERING_SOURCE
    + r'''


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


def training_device(torch, given=None):
    """Where the run happens: what the parent already resolved, or CUDA-if-any.

    `given` is `envs.resolve_device`'s answer, asked once by the parent process
    before this script was even started, and taken here rather than decided
    again -- so the parent knows the device while it can still act on it,
    instead of reading it back out of the metrics file this run writes at the
    end. Falls back to asking `torch` directly only when the parent could not
    answer or never asked (`given is None`): a probe that could not run must
    not stop a training run that otherwise would. Recorded in the metrics
    either way, so a GPU box that trained on the CPU says so rather than
    looking exactly like a laptop.
    """
    if given is not None:
        return given
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_call_row(row):
    """A row whose target is a tool call, as `prepare` writes one."""
    target = row.get("target")
    return isinstance(target, dict) and "name" in target


def call_terminator(tok, runtime_rendered, probe):
    """What the chat template puts after a tool call, or `None` where it is not asked.

    A text answer and a call end differently. FunctionGemma's templates -- the
    published one and the one bundled with LiteRT-LM -- close a text turn with
    `<end_of_turn>` and a call turn with `<end_function_call>` followed by
    `<start_function_response>`, where the model stops and the application
    runs the tool. Measured 2026-09-17: a model trained with every completion
    closed by the text ending, and no call markers, returned no call from the
    runtime on 5 of 5 prompts and had nothing refused -- the runtime read plain
    text.

    So this asks the template, the way `turn_terminator` does for a text turn:
    render an assistant turn carrying a probe call and take what follows it.
    `probe` is the call `prepare` renders for the same probe, passed in by the
    parent because this script cannot import litetune. If the template does not
    render that exact text, the completions this run trains are not calls the
    runtime would read, and training stops rather than proceeds.

    Only in `runtime_rendered`: in `prerendered` the application builds the
    prompt and reads the reply, and flutter_gemma delimits a reply by
    `<end_of_turn>`, so the text ending stands.
    """
    if not runtime_rendered or not probe:
        return None
    try:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "x"},
             {"role": "assistant", "tool_calls": [
                 {"type": "function", "function": {"name": probe["name"], "arguments": {}}}]}],
            tokenize=False, add_generation_prompt=False,
        )
    except Exception as exc:  # noqa: BLE001 - reported, and training stops
        raise ValueError(
            "the chat template could not render a tool call, so there is no ending to train a "
            f"call with: {type(exc).__name__}: {exc}"[:400]
        ) from exc
    if probe["text"] not in rendered:
        raise ValueError(
            "the chat template renders a tool call differently from the completions this run "
            f"trains: prepare wrote {probe['text']!r}, and the template rendered "
            f"{rendered[-200:]!r}. Training on the first teaches a call the runtime would not read"
        )
    tail = rendered.split(probe["text"])[-1]
    ids = tok(tail, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(
            "the chat template puts nothing after a tool call, so a trained call would never stop"
        )
    return {"ids": list(ids), "source": "chat_template_call", "text": tok.decode(ids)}


def build_examples(tok, rows, max_seq_length, runtime_rendered, tools=None, call_probe=None):
    """One (input_ids, labels) pair per row, with the prompt masked out."""
    examples = []
    supervised = 0
    total = 0
    terminator, terminator_source = turn_terminator(tok, runtime_rendered)
    # Asked only when some row is a call: a family with no tool channel has a
    # template that cannot render one, and a text split has no reason to try.
    call = (
        call_terminator(tok, runtime_rendered, call_probe)
        if any(is_call_row(row) for row in rows)
        else None
    )
    for row in rows:
        prompt_text, add_special = render_prompt(tok, row["prompt"], runtime_rendered, tools)
        prompt_ids = tok(prompt_text, add_special_tokens=add_special)["input_ids"]
        completion_ids = tok(row["completion"], add_special_tokens=False)["input_ids"]
        ending = call["ids"] if call is not None and is_call_row(row) else terminator
        completion_ids = list(completion_ids) + list(ending)
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
        # How a call row ended, when one was trained and the template was asked.
        "call": call,
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

    mode = spec["prompt_mode"]
    if mode not in ("prerendered", "runtime_rendered"):
        raise ValueError(
            f"prompt_mode {mode!r} is not a mode this script can train. The parent decides it "
            "and writes it here; a missing one is not a default, because prerendered and "
            "runtime_rendered train different prompts."
        )
    runtime_rendered = mode == "runtime_rendered"
    examples, supervised, total, terminator = build_examples(
        tok, rows, spec["max_seq_length"], runtime_rendered, spec.get("tools"),
        spec.get("call_probe"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        dtype=getattr(torch, spec["dtype"]),
        attn_implementation=spec["attn_implementation"],
        **from_kwargs
    )
    model.config.use_cache = False
    device = training_device(torch, spec.get("device"))
    model.to(device)

    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target_modules = list(spec["lora_targets"])
    container = spec.get("lora_container")
    if container:
        # peft matches a list of names by suffix, and on a multimodal
        # checkpoint the same projection names occur in the vision and audio
        # towers. A regex is the only shape `target_modules` has that says
        # "these projections, and only under this container". The container has
        # to be a whole path segment: `.*language_model\.` also matches a
        # sibling named `xlanguage_model`, because `.*` absorbs the prefix, so
        # what precedes it is either nothing or something ending in a dot.
        target_modules = r"(?:.*\.)?%s\..*\.(%s)" % (
            re.escape(container),
            "|".join(re.escape(name) for name in target_modules),
        )
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
                target_modules=target_modules,
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
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                labels=labels.to(device),
            )
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
                "prompt_mode_decision": spec.get("prompt_mode_decision"),
                "declarations_sha256": spec.get("declarations_sha256"),
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
                # Where it ran. A CUDA box and a laptop produce the same
                # checkpoint; they do not take the same time, and a report
                # that cannot say which it was cannot explain the difference.
                "device": device,
                # Whether the SentencePiece model made it back beside the
                # checkpoint. Recorded, not assumed: the difference between
                # `SP_Tokenizer` and an HF section is invisible in every
                # measurement and decides whether constrained decoding works.
                "sentencepiece": sentencepiece,
                # What the model now expects at serving time. The bundle's
                # contract has to state this, and this is where it is observed
                # rather than asserted.
                "prompt_mode": spec["prompt_mode"],
                # How that mode was decided, and on what evidence: declared,
                # inferred from these prompts, or declared against them on purpose.
                "prompt_mode_decision": spec.get("prompt_mode_decision"),
                # Which tool declarations these calls were trained against.
                "declarations_sha256": spec.get("declarations_sha256"),
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
                # What peft was actually given: the projection list, or the
                # regex that restricts it to one container. Recorded because
                # two runs called `lora` are only the same method if this is
                # the same, and on a multimodal checkpoint it is not derivable
                # from the model id.
                "lora_target_modules": target_modules if spec["method"] == "lora" else None,
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
)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def decide_prompt_mode(
    prompts: Sequence[str], declared: PromptMode | None, force: bool = False
) -> PromptModeDecision:
    """Which convention this run trains under. Raises `TuneError` rather than guess.

    `verify` measures a split whose prompts disagree with each other verbatim,
    because something has to run and a measurement costs minutes. Here a wrong
    guess costs a training run and leaves a checkpoint that expects a prompt no
    caller sends, so an undeclared mixed split is refused, and so is a declared
    mode the prompts contradict unless `force` says the contradiction is
    deliberate. The check reads control tokens in the prompts; it cannot see how
    an application will call the model, which is what `force` is for.
    """
    evidence = prompt_evidence(prompts)
    found = ", ".join(evidence.markers) if evidence.markers else "none"
    observed = (
        f"{evidence.share:.0%} of {evidence.count} training prompts contain control tokens "
        f"({found})"
    )

    def decided(mode: PromptMode, source: str, reason: str) -> PromptModeDecision:
        return PromptModeDecision(
            mode=mode,
            source=source,
            evidence=f"{observed}; {reason}",
            marker_share=evidence.share,
            markers=evidence.markers,
        )

    if evidence.count == 0:
        if declared is None:
            raise TuneError(
                "the training split has no prompts to infer a prompt mode from; pass --prompt-mode"
            )
        return decided(declared, "declared", "there are no prompts to check it against")
    if declared is None:
        if evidence.mode is None:
            raise TuneError(
                f"{observed}: the split mixes rendered prompts and bare text, so no single prompt "
                "mode fits it and litetune will not guess one for a training run. Render every "
                "prompt the same way, or pass --prompt-mode to decide"
            )
        if evidence.mode is PromptMode.PRERENDERED:
            reason = "they were rendered before training, so the runtime must not template them"
        else:
            reason = (
                "they are bare text, so training renders the model's chat template around them "
                "the way the runtime will"
            )
        return decided(evidence.mode, "inferred", f"{reason}. No mode was declared")
    if evidence.mode is None:
        return decided(
            declared, "declared", "the split mixes both conventions, so the declared mode decides"
        )
    if evidence.mode is declared:
        return decided(declared, "declared", "the prompts agree with the declared mode")
    if declared is PromptMode.PRERENDERED:
        conflict = (
            "these prompts are bare text, and a runtime that applies its own chat template would "
            "present them inside a turn this training never showed the model"
        )
    else:
        conflict = (
            "these prompts were already rendered, and training under the chat template would "
            "wrap them a second time"
        )
    if not force:
        raise TuneError(
            f"{declared.value} was declared, but {observed}: {conflict}. Leave out --prompt-mode "
            f"to train in the mode the prompts show, or pass {FORCE_PROMPT_MODE_FLAG} if the "
            "application really calls the model this way"
        )
    return decided(
        declared, "overridden", f"{conflict}; {FORCE_PROMPT_MODE_FLAG} kept the declared mode"
    )


@dataclass(frozen=True)
class TuneRequest:
    """One training run.

    `attn_implementation` defaults to the implementation the export and
    evaluation paths use too: `evaluate.py` passes the training attention
    implementation through unconditionally, so a checkpoint trained under one
    and served under another produces output that is fluent, wrong, and
    passes every check that does not involve held-out labels. `dtype` has no
    such pair to default to -- `evaluate.py`'s float reference loads at an
    unconditional `float32` regardless of the training dtype, and
    `export.py` passes no dtype to the exporter at all -- so the `bfloat16`
    default matches nothing downstream and is not derived from anything here.
    It is stated, and `run_tune` records it as a limitation whether a run
    takes it or departs from it, because this repo establishes no winner
    between `bfloat16` and `float32`. Both fields are recorded on every run
    for the same underlying reason: a mismatch here produces output that
    looks fine and is not.
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
    # `prompt_mode.resolve_prompt_mode`, so nothing downstream has to guess.
    #
    # `--no-template` on the serving side is right for a hand-rendered wire
    # format and wrong for anything else, and it was carried everywhere by
    # habit once.
    # Keyword-only, and `None` means "not declared", never a default: `run_tune`
    # then reads the training prompts and decides from their control tokens,
    # refusing a split that mixes the two conventions (`decide_prompt_mode`). A
    # declared mode the prompts contradict is refused too, unless
    # `force_prompt_mode` says the contradiction is deliberate. Guessing wrong
    # trains the model on a prompt the runtime never sends -- silently, with a
    # loss curve that looks fine. The decided mode is `TuneResult.prompt_mode`;
    # this field is only what the caller said.
    prompt_mode: PromptMode | None = field(default=None, kw_only=True)
    force_prompt_mode: bool = field(default=False, kw_only=True)
    # The tool declarations the run trains against, in the shape `bundle` takes.
    # Their digest is recorded beside the checkpoint, where `verify` reads it
    # back rather than being told it. Absent, no declaration turn is rendered.
    declarations: Path | None = field(default=None, kw_only=True)
    timeout_s: int = DEFAULT_TIMEOUT_S
    env: envs.StageEnv = envs.TRAIN
    auto_provision: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise TuneError(f"method must be one of {list(METHODS)}, got {self.method!r}")
        if self.prompt_mode is not None and not isinstance(self.prompt_mode, PromptMode):
            raise TuneError(
                f"prompt_mode must be a PromptMode, got {self.prompt_mode!r}. The two conventions "
                "are mutually exclusive and a model trained under one cannot be served under the "
                f"other; known modes: {[m.value for m in PromptMode]}"
            )
        if self.force_prompt_mode and self.prompt_mode is None:
            raise TuneError(
                f"{FORCE_PROMPT_MODE_FLAG} keeps a declared prompt mode the training prompts "
                "contradict, and no mode was declared; pass --prompt-mode with it"
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

    def config(
        self,
        metrics_out: Path,
        device: str | None = None,
        decision: PromptModeDecision | None = None,
        declarations_sha256: str | None = None,
        declarations: list | None = None,
        lora_container: str | None = None,
    ) -> dict[str, Any]:
        """Everything the generated script needs. Also what the report records.

        `device` is what `envs.resolve_device` found before this config was
        written, asked once in the parent rather than left for the script to
        decide on its own -- see `tune.training_device`. `None` covers both
        "the probe was asked and could not answer" and "no probe was run", and
        the script falls back to asking itself in either case. Which of the two
        it was is in `envs.DeviceProbe.detail`, and reaches the report as a
        limitation rather than as a device.

        `decision` is the mode `run_tune` settled on. Without one the declared
        mode is written, which is `None` when nothing was declared.

        `declarations_sha256` is computed in this process, not in the script:
        the training environment has no litetune to compute it with, and the
        digest has to be the one `bundle` compares its contract against.
        """
        mode = decision.mode if decision is not None else self.prompt_mode
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
            # Which container those projections are restricted to, from the
            # family's rules -- decided in the parent, like the device and the
            # prompt mode, so a run says what it is about to adapt while there
            # is still something to do about it.
            "lora_container": lora_container,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
            "prompt_mode": mode.value if mode is not None else None,
            # How the mode was decided and on what evidence. The script writes it
            # beside the checkpoint, so the decision outlives this process.
            "prompt_mode_decision": decision.as_dict() if decision is not None else None,
            # What the calls in this split were trained against. The script
            # writes it beside the checkpoint, where `verify` reads it back
            # instead of being told which declarations a checkpoint knows.
            "declarations_sha256": declarations_sha256,
            # `tools`, not `declarations`: the file is the declarations, and
            # this is what the chat template's `tools=` receives. One key per
            # meaning, because the request record already carries the path
            # under the other name and a reader must not have to work out
            # which of two `declarations` a report means.
            "tools": declarations,
            # The call `prepare` renders for a probe, so the script can ask the
            # chat template what ends a call turn and check that the template
            # writes a call the way `render_call` does. The script cannot import
            # litetune, so the text travels in the spec. Only where the family
            # records FunctionGemma's call format and the runtime renders the
            # prompt: any other family's call rows are the caller's own text,
            # and in `prerendered` every row ends the way the application reads
            # a reply, so the script would not ask.
            "call_probe": (
                {"name": CALL_PROBE_NAME, "text": render_call(ToolCall(CALL_PROBE_NAME, {}))}
                if wire_format_for(self.model).known
                and decision is not None
                and decision.mode is PromptMode.RUNTIME_RENDERED
                else None
            ),
            "model_dir": str(self.model_dir),
            "adapter_dir": str(self.adapter_dir) if self.adapter_dir else None,
            "metrics_out": str(metrics_out),
            "device": device,
        }

    def as_dict(self, device: str | None = None) -> dict[str, Any]:
        record = self.config(self.output_dir / "metrics.json", device=device)
        # The request records what the caller said; the decision is the result's
        # and is reported at the top level of `TuneResult.as_dict`.
        record.pop("prompt_mode_decision")
        # Same division: the request says which file the caller named, and the
        # digest of what was in it belongs to the result. Leaving it here would
        # write `None` into `request` on every report, beside the real digest at
        # the top level -- two fields in one record disagreeing about one fact.
        record.pop("declarations_sha256")
        # And the probe, which the run decides from the split as well as the
        # family: recomputed here from the family alone, it disagreed with the
        # one the script was given (`train_config.json`).
        record.pop("call_probe")
        record["declarations"] = str(self.declarations) if self.declarations else None
        record["force_prompt_mode"] = self.force_prompt_mode
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
    # `None` when the script predates the field. Absent is absent, not "cpu".
    device: str | None = None
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
            device=data.get("device"),
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
            "device": self.device,
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
    # What `envs.resolve_device` answered, asked once before the run started --
    # not what `metrics.device` reports after one. The two agree whenever the
    # run used what it was told; `None` means no device was established before
    # the run, whether because the probe could not answer or because none ran,
    # and is never "cpu" (see `TrainingMetrics.device`, "absent is absent").
    # Which of the two it was is recorded as a limitation, not here.
    device: str | None = None
    # How the mode was settled, before the environment was touched. `None` only
    # when the run was refused before a mode could be decided.
    prompt_mode_decision: PromptModeDecision | None = None
    # The digest of the declarations this run trained against, read before the
    # environment was touched. `None` when none were supplied, which is every
    # run that trains plain text.
    declarations_sha256: str | None = None

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
    def prompt_mode(self) -> PromptMode | None:
        """The convention this checkpoint now expects at serving time.

        Surfaced here because `bundle.Contract` requires it and refuses to
        default it: this is the stage that decided it, so this is where a bundle
        should read it from rather than a human retyping it. `None` only for a
        run refused before a mode was decided.
        """
        if self.prompt_mode_decision is not None:
            return self.prompt_mode_decision.mode
        return self.request.prompt_mode

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def as_dict(self) -> dict[str, Any]:
        mode = self.prompt_mode
        decision = self.prompt_mode_decision
        return {
            "schema": TUNE_SCHEMA,
            "verified": False,
            "unverified_reason": NOT_VERIFIED,
            "outcome": self.outcome.value,
            # Top-level as well as inside `request`: this is the field a bundle's
            # contract is built from, and it must not be something a reader has
            # to go looking for.
            "prompt_mode": mode.value if mode is not None else None,
            "prompt_mode_decision": decision.as_dict() if decision is not None else None,
            # Top-level for the same reason as the mode: a bundle's contract is
            # built from this, and a reader must not have to go looking for it.
            "declarations_sha256": self.declarations_sha256,
            "request": self.request.as_dict(device=self.device),
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
    # This run's own record of which host variables it dropped, so a second
    # run in one process -- a notebook, a service -- says what the first
    # one did instead of inheriting its silence.
    envs.forget_reported_drops()
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
        # precisely. bfloat16 over float16 *is* decided, in this repo:
        # spec.py's `DTYPES` excludes float16 because a fine-tune in it
        # overflows where bfloat16 does not, and cli.py's `--dtype` argument
        # (`choices=sorted(DTYPES)`) pins the same choice to a 270M model
        # whose loss goes to NaN in float16 while bfloat16 holds.
        # bfloat16 versus float32, the other member of `DTYPES`, is not
        # decided the same way -- what is established is that neither
        # downstream path is known to match this default: evaluate.py's
        # float reference loads with `torch_dtype=torch.float32` regardless
        # of what trained the checkpoint under test, and export.py passes no
        # dtype to the exporter at all, so what dtype the exported
        # checkpoint actually loads in is not established here either.
        result.limitation(
            "training runs with bfloat16 parameters, so the optimiser's moments are bfloat16 as "
            "well; updates are coarser than a mixed-precision run with an fp32 master copy. "
            "bfloat16 over float16 is decided, not a guess: a fine-tune in float16 overflows "
            "where bfloat16 does not, and a 270M model's loss goes to NaN in float16 while "
            "bfloat16 holds. bfloat16 versus float32, the other dtype this tool supports, is not "
            "established the same way: evaluation's float reference loads at an unconditional "
            "float32 regardless of the training dtype, so a run at the default dtype still "
            "carries a dtype difference against the reference it is scored on; export passes no "
            "dtype to the exporter at all, so what dtype the exported checkpoint actually loads "
            "in is not established here either"
        )
    if request.attn_implementation != DEFAULT_ATTN_IMPLEMENTATION:
        result.limitation(
            f"this run trains with attention {request.attn_implementation!r}, not "
            f"{DEFAULT_ATTN_IMPLEMENTATION!r}: evaluate.py passes the training attention "
            "implementation through unconditionally (evaluate.py's `_HF_GENERATE_SCRIPT`), so a "
            "checkpoint served under a different implementation than it was trained with "
            "produces output that is fluent, wrong, and passes every check that does not "
            "involve held-out labels"
        )
    if request.dtype != DEFAULT_DTYPE:
        # Unlike attention, there is no pair here for a non-default dtype to
        # match or miss: evaluate.py's float reference always loads at
        # float32, and export.py passes no dtype at all. So this limitation
        # states the departure and stops there. What it must not claim is
        # which dtype the published numbers were taken at: MEASUREMENTS.md
        # gives a dtype for exactly one of them -- the second-family
        # banking77 run, trained in float32 "because bfloat16 on this CPU
        # runs on a single core" -- and says nothing about the dtype of the
        # headline table.
        result.limitation(
            f"this run trains with dtype {request.dtype!r}, not the {DEFAULT_DTYPE!r} default. "
            "evaluate.py's float reference loads at an unconditional float32 regardless of the "
            "training dtype, and export.py passes no dtype to the exporter at all, so there is "
            "no single dtype the export and evaluation paths are known to use for this run to "
            "match or miss. Which dtype a published number was taken at is recorded per run in "
            "MEASUREMENTS.md and is not a property of this default"
        )
    if request.rate_is_default:
        events.note(
            f"learning rate {request.rate} (default for method {request.method!r})",
            learning_rate=request.rate,
            method=request.method,
        )

    # -- is there a split, and which convention does it train? -------------
    # Before the environment, so a refusal costs nothing: provisioning installs
    # torch and transformers, and the device probe starts an interpreter in it.
    if not request.data.is_file():
        # A check, not an exception: "the split is not there" is an observation
        # about this run, and the report has to carry it.
        return _refused(
            result,
            events,
            Check.failed(
                TRAINING_CHECK,
                f"the training split {request.data} does not exist, so nothing was trained",
                observed={"data": str(request.data)},
            ),
        )

    try:
        rows = read_rows(request.data)
        prompts = [row.prompt for row in rows]
        decision = decide_prompt_mode(prompts, request.prompt_mode, force=request.force_prompt_mode)
    except (TuneError, PrepareError, OSError) as exc:
        return _refused(
            result,
            events,
            Check.failed(
                PROMPT_MODE_CHECK,
                str(exc),
                observed={
                    "declared": (
                        request.prompt_mode.value if request.prompt_mode is not None else None
                    ),
                    "force_prompt_mode": request.force_prompt_mode,
                },
            ),
        )
    result.prompt_mode_decision = decision
    decided = Check.passed(
        PROMPT_MODE_CHECK,
        f"{decision.mode.value} ({decision.source}): {decision.evidence}",
        observed=decision.as_dict(),
    )
    result.checks.add(decided)
    events.check(decided)

    # Before the environment, like the mode above: each of these is a fact about
    # the request, and finding it out after provisioning costs minutes and a
    # download to say so. A family whose runtime renders declarations trains a
    # prompt that carries them -- refused here as well as in `prepare`, because
    # a hand-written split reaches this stage without passing through that one,
    # and by the time the loss curve is available it looks fine.
    for refusal in (
        _refuse_rows_without_a_completion(request, rows),
        _refuse_calls_of_a_family_litetune_cannot_tell(request, decision.mode, rows),
        _refuse_calls_without_declarations(request, decision.mode, rows),
        _refuse_calls_the_runtime_would_not_read(request, decision.mode, rows),
        _refuse_declarations_for_an_unrecorded_tool_channel(request, decision.mode),
    ):
        if refusal is not None:
            return _refused(result, events, refusal)
    # Bound before the branch: the script's spec is built further down on every
    # path, including the one where no declarations were supplied.
    declarations: list | None = None
    if request.declarations is not None:
        try:
            declarations, digest = read_declarations(request.declarations)
        except DeclarationsError as exc:
            return _refused(
                result,
                events,
                Check.failed(
                    DECLARATIONS_CHECK,
                    str(exc),
                    observed={"declarations": str(request.declarations)},
                ),
            )
        # `prepare` refuses these too; a split written by hand reaches this
        # stage without it, and would train calls the prompt never offers.
        try:
            refuse_undeclared_tools(rows, request.data, request.declarations)
        except PrepareError as exc:
            return _refused(
                result,
                events,
                Check.failed(
                    DECLARATIONS_CHECK,
                    str(exc),
                    observed={"declarations": str(request.declarations)},
                ),
            )
        # The digest recorded for this run, which `verify` and `bundle` compare
        # with: the tool list's where the runtime renders them, the file's
        # bytes where the application does -- see `recorded_digest`.
        digest = recorded_digest(
            digest, request.declarations, decision.mode is PromptMode.PRERENDERED
        )
        result.declarations_sha256 = digest
        count = entry_count(declarations)
        read = Check.passed(
            DECLARATIONS_CHECK,
            f"{count if count is not None else 'the'} declaration(s) from "
            f"{request.declarations.name}, {digest[:23]}",
            observed={"declarations": str(request.declarations), "sha256": digest},
        )
        result.checks.add(read)
        events.check(read)

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
    lora_container = rules.lora_container if rules is not None else None
    if rules is not None and rules.lora_container and request.method == "lora":
        # Said out loud rather than applied quietly: this is the difference
        # between adapting a text tower and adapting a whole multimodal
        # checkpoint, and the artifact it produces looks identical either way.
        result.limitation(
            f"LoRA was restricted to modules under `{lora_container}`: "
            f"{rules.lora_container_reason}. The projection set is unchanged, so this run is "
            f"comparable with the other families here; an unscoped run on this checkpoint would "
            f"not be"
        )
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

    # -- where will this run? -----------------------------------------------
    # Asked once, here, in the parent, before the training script starts,
    # rather than left to the fallback inside it: that is what lets this run
    # say where it is about to train while there is still something to do
    # about it, instead of reading the device back out of a metrics file six
    # hours later.
    #
    # Not gated on `auto_provision`. The probe provisions nothing, so
    # readiness is its real precondition -- and the environment check above
    # has already returned when the environment is not ready. Gating it on
    # provisioning made `litetune tune --no-provision` (cli.py's flag) over an
    # environment that was already there report no device at all for a run
    # that had one.
    probe = envs.resolve_device(request.env, events=events)
    device = probe.device
    result.device = device
    if not probe.answered:
        # `logging` alone reaches nothing the report carries, and a `None`
        # device with nothing beside it is indistinguishable from a device
        # nobody asked about.
        result.limitation(
            f"{probe.detail}, so this run's device was not established before it started. The "
            "training script decided for itself and reports what it chose in its metrics"
        )
    elif probe.cuda_build_without_a_device:
        result.limitation(probe.detail)
    if request.dtype == DEFAULT_DTYPE and device != "cuda":
        # Before the wait, not after it: DEFAULT_TIMEOUT_S is sized for
        # exactly the run this combination produces, and the operator who set
        # it running should not have to wait to hear why.
        #
        # `!= "cuda"` rather than `== "cpu"`: an unanswered probe is precisely
        # the run that has nothing else to explain its six hours with, and the
        # script's own fallback resolves to cpu on every machine without a
        # reachable GPU. The wording says which of the two this is, because
        # "will" and "may" are different claims.
        where = (
            "training will run bfloat16 on the CPU"
            if device == "cpu"
            else "training may run bfloat16 on the CPU: its device is not established yet"
        )
        events.note(
            f"{where}: {BFLOAT16_CPU_HINT}",
            device=device,
            dtype=request.dtype,
        )

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
    config_path.write_text(
        json.dumps(
            request.config(
                metrics_out,
                device=device,
                decision=result.prompt_mode_decision,
                declarations_sha256=result.declarations_sha256,
                declarations=declarations,
                lora_container=lora_container,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
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
        timeout_detail = (
            f"no result after {request.timeout_s}s (timeout): the run did not finish, which says "
            "nothing about the method or the data"
        )
        if request.dtype == DEFAULT_DTYPE and device != "cuda":
            # DEFAULT_TIMEOUT_S is sized for exactly this: the run whose
            # ending the slowdown caused should be the one told it exists.
            # Nothing here is read from the metrics: this ending returns
            # before any metrics file is opened, so the pre-run probe is the
            # only thing that knows anything about the device -- and an
            # unanswered probe must not swallow the one hint that explains a
            # six-hour non-result. It also must not turn into an assertion
            # about a CPU nobody observed, so the lead-in says which it is.
            lead = (
                "it was running bfloat16 on the CPU"
                if device == "cpu"
                else "its device was never established, and bfloat16 on a CPU is one way to "
                "spend this timeout"
            )
            timeout_detail = f"{timeout_detail}. {lead}: {BFLOAT16_CPU_HINT}"
        timed_out = Check.unchecked(
            TRAINING_CHECK,
            timeout_detail,
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

    # -- where did it actually run? ------------------------------------------
    # From `result.metrics`, not the pre-run probe above: this is what the
    # script itself reported after using -- or falling back from -- what it
    # was given, and it needs only `result.metrics`, which is parsed by now.
    #
    # Placed ahead of the killed early return below rather than at the bottom
    # of the function. Two of the five endings this moves, not one: the killed
    # return right below, and the accelerator-failure return further down,
    # which was added in this same change and sits after this block for the
    # same reason. The timeout returns further up without ever opening a
    # metrics file, and the other two -- the blocked-start return and the
    # bottom of the function -- were already downstream of this block. The
    # window it buys the killed and accelerator-failure endings is narrow,
    # too -- `metrics.json` is the last thing the training script
    # writes, after the checkpoint and the tokenizer, so a process killed
    # mid-run has most likely written none. Narrow is not nothing: a run
    # killed just after that write is exactly the run whose device is worth
    # knowing, and there is no cost to reading a file that is already parsed.
    metrics_device = result.metrics.device if result.metrics is not None else None
    if metrics_device is not None:
        events.note(f"trained on {metrics_device}", device=metrics_device)
    if metrics_device == "cpu" and request.dtype == DEFAULT_DTYPE:
        result.limitation(f"this run trained bfloat16 on the CPU: {BFLOAT16_CPU_HINT}")

    reading = read_returncode(proc.returncode)
    if not reading.conclusive:
        # Killed, not failed. A training run is the longest-lived and hungriest
        # process litetune starts, so it is the likeliest to meet the
        # out-of-memory killer; reading `-9` as a training failure would blame
        # the method or the data for a fact about the machine. See
        # `litetune.exits`.
        killed_detail = (
            f"training was {reading.describe('the method or the data')}. "
            f"stderr: {_tail(result.stderr) or 'none'}"
        )
        observed_device = metrics_device or device
        # No longer withheld on a SIGKILL. It used to be, on the argument that
        # `reading.describe` had already named the near-certain cause -- the
        # out-of-memory killer -- so a speed hint beside it read as a second,
        # contradictory story. That text no longer says "near-certain": it
        # offers a cancelled job as an equal possibility and tells the reader
        # to check for one first. So a CI runner cancelling a bf16-on-CPU
        # training run would be told to find more memory while the one fact
        # this run did establish stayed unsaid.
        if request.dtype == DEFAULT_DTYPE and observed_device != "cuda":
            # What the script reported wins over what the probe predicted, and
            # an unanswered probe does not silence the hint: see the note
            # before the run. As there, the wording distinguishes a device
            # that was *observed* -- `metrics_device`, written by the script
            # itself -- from one that was only ever *predicted* by the
            # pre-run probe: a killed run that never wrote metrics has
            # nothing confirming it ran anywhere, and "it ran" would claim a
            # past tense nothing established.
            if metrics_device == "cpu":
                lead = "it ran bfloat16 on the CPU"
            elif observed_device == "cpu":
                lead = (
                    "the pre-run probe predicted the CPU, but the run was killed before its "
                    "own device was confirmed"
                )
            else:
                lead = (
                    "its device was never established, and bfloat16 on a CPU is one way to "
                    "reach this ending"
                )
            killed_detail = f"{killed_detail} {lead}: {BFLOAT16_CPU_HINT}."
        killed = Check.unchecked(
            TRAINING_CHECK,
            killed_detail,
            observed=reading.as_dict() | {"seconds": round(result.seconds, 3)},
        )
        result.checks.add(killed)
        events.check(killed)
        result.checks.add(
            masking_check(result.metrics, "the run was killed, so no mask was observed")
        )
        events.stage_finished(result.outcome.value, attempted=True)
        return result

    gpu_failure = _GPU_FAILURE_RE.search(result.stderr or "") if proc.returncode != 0 else None
    if gpu_failure is not None:
        # The machine, not the run. An out-of-memory allocation on the GPU or
        # the host, a device that disappeared between the probe and the run, a
        # driver too old for the binary: each of them exits 1, and each of
        # them says nothing at all about whether this method on this data
        # would have worked. Same judgement as the killed branch above,
        # arriving through stderr rather than through a signal.
        accelerator = Check.unchecked(
            TRAINING_CHECK,
            f"training exited {proc.returncode} on a machine failure "
            f"({gpu_failure.group(0)}), which says nothing about the method or the data: "
            f"{_tail(result.stderr) or 'no stderr'}",
            observed={
                "returncode": proc.returncode,
                "seconds": round(result.seconds, 3),
                "matched": gpu_failure.group(0),
                "device": metrics_device or device,
                "stderr_tail": _tail(result.stderr, 2000),
            },
        )
        result.checks.add(accelerator)
        events.check(accelerator)
        # Returns here, exactly as the killed branch above does, and for the
        # same reason: nothing downstream of this point has an observation to
        # make. Falling through would run `_merge_check`, which on a LoRA run
        # would record a *failed* merge -- "no adapter was retained, the
        # learned delta is unrecoverable" -- for a run that never got far
        # enough to have one, while a full fine-tune's merge check returns
        # `passed` unconditionally (there is no adapter to merge for that
        # method, so it has nothing to fail on). A false *passed* is worse
        # than the silence an early return leaves: it reports a merge that did
        # not happen as though it had. One machine event, two wrong reports,
        # chosen by `--method`.
        result.checks.add(
            masking_check(result.metrics, "the accelerator failed, so no mask was observed")
        )
        events.stage_finished(result.outcome.value, attempted=True)
        return result

    if proc.returncode != 0:
        training = Check.failed(
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
        training = Check.failed(
            TRAINING_CHECK,
            f"training exited zero but wrote no checkpoint into {request.model_dir}",
            observed={"returncode": 0, "model_dir": str(request.model_dir)},
        )
    else:
        result.model_dir = request.model_dir
        training = Check.passed(
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
    result.checks.add(training)
    events.check(training)
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
