"""One evaluator, used for every measurement point.

The points this tool reports -- reference float, tuned float, tuned converted --
are only meaningful when subtracted from one another, and subtraction is only
valid if both sides saw the same split, the same prompts, the same prompt
construction and the same decoding. Three separate evaluators drift within
weeks, at which point the difference between two points stops measuring what it
claims to. So there is one evaluator here, parameterized by (model reference,
backend, split), and the parameters it was given travel with the result.

`PromptMode` travels with every measurement for a specific reason: the
reference runtime's `--no-template` forces the tool list to null, so a model
whose declarations are rendered by the runtime and one whose prompt was built by
the application cannot be compared at all -- the difference measures the mode,
not the model. `harness_mismatch` exists to refuse that comparison rather than
report it.

Nothing in this module raises on a failed generation. A non-zero exit is an
observation; a process that never started is a *different* observation, and
`Generation` keeps them apart so that liveness can report `failed` for the first
and `could not check` for the second.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from litetune import envs
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.metrics import ToolCall, read_target

logger = logging.getLogger(__name__)

# How often a long generation run reports progress. Emitted as events; nothing
# here prints.
PROGRESS_EVERY = 25

DEFAULT_MAX_TOKENS = 256


class PromptMode(str, Enum):
    """How the text that reached the model was constructed."""

    # The prompt is used verbatim: the application rendered any declarations
    # into it and the runtime's own template is disabled (`--no-template`).
    PRERENDERED = "prerendered"
    # The runtime applies its own chat template and renders declarations.
    RUNTIME_RENDERED = "runtime_rendered"


# ---------------------------------------------------------------------------
# Which mode a measurement is taken in
# ---------------------------------------------------------------------------
#
# `--no-template` is narrow, and it was carried around as though it were
# general. Its own help says "the input should include all control tokens for
# the model expected", and what it actually does is route the runtime to
# `create_session()` instead of `create_conversation()`, bypassing the chat
# template, the `<|turn>model` anchor, tool handling and channel extraction. It
# is correct when the caller built the whole prompt including control tokens --
# the FunctionGemma case, where training used a hand-rendered wire format -- and
# wrong by default. A six-model probe run without it gave live structured output
# on Gemma 3 270M, Qwen3 0.6B and Qwen2.5 0.5B, all of which had previously been
# measured *with* it.
#
# The mode is not a property of the model. The same FunctionGemma trained
# through `apply_chat_template` would need the opposite flag, so it cannot be
# looked up by family (see `litetune.models`). It is decided by how the prompt
# was built at training time:
#
#     hand-rendered wire format with control tokens -> --no-template
#     apply_chat_template / native runtime tools     -> no flag
#
# `tune` records that decision, `bundle.Contract.prompt_mode` carries it, and
# `verify` reads it back. Only when there is no contract -- a foreign artifact,
# which is this tool's primary entry point -- is it inferred, and then the
# inference and its evidence are reported so a user can contradict them.

# What a backend falls back to when nobody declared a mode. It is what this tool
# has always done and not a considered answer, so every backend records whether
# the mode reaching it was declared or defaulted, and `verify` never leaves it
# to this.
UNDECLARED_PROMPT_MODE = PromptMode.PRERENDERED

# What `describe()["backend"]` says when a measurement's device was never
# established. The word matters: `verify.py` prints
# `unknown` when the key is missing or null, so this
# is the same word for what is, to a reader of the manifest, the same state.
UNKNOWN_BACKEND = "unknown"

# Control tokens that only appear in a prompt somebody already rendered. Bare
# user text does not contain them.
TURN_MARKERS = (
    "<start_of_turn>",
    "<|turn>",
    "<|im_start|>",
    "<|start_header_id|>",
    "<start_of_image>",
)

# Above this share of prompts carrying a marker the split is pre-rendered;
# below one minus it, it is bare. In between the split is inconsistent, which is
# reported rather than smoothed over.
_MARKER_CONFIDENT_SHARE = 0.9


@dataclass(frozen=True)
class PromptModeDecision:
    """Which mode a measurement runs in, where that came from, and on what evidence.

    `source` is `declared` (the caller said so), `contract` (the bundle that
    shipped the model said so) or `inferred` (neither existed and the prompts
    were inspected). Only the last one is a guess, and it says so in every
    report it reaches.
    """

    mode: PromptMode
    source: str
    evidence: str
    marker_share: float | None = None
    ambiguous: bool = False

    @property
    def inferred(self) -> bool:
        return self.source == "inferred"

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_mode": self.mode.value,
            "source": self.source,
            "evidence": self.evidence,
            "marker_share": self.marker_share,
            "ambiguous": self.ambiguous,
        }


def marker_share(prompts: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    """Share of prompts already carrying a control token, and which ones were seen."""
    if not prompts:
        return 0.0, ()
    seen: list[str] = []
    hits = 0
    for prompt in prompts:
        found = [marker for marker in TURN_MARKERS if marker in prompt]
        if found:
            hits += 1
            seen.extend(m for m in found if m not in seen)
    return hits / len(prompts), tuple(seen)


def resolve_prompt_mode(
    prompts: Sequence[str],
    declared: PromptMode | None = None,
    contract: PromptMode | None = None,
) -> PromptModeDecision:
    """Decide the mode this measurement runs in. An explicit choice always wins.

    Precedence is declared, then the contract the artifact shipped with, then
    the prompts themselves. The last is a heuristic and is labelled as one: a
    prompt that already contains `<start_of_turn>` was rendered by whoever wrote
    the split, and templating it again double-wraps it.
    """
    if declared is not None:
        return PromptModeDecision(
            mode=declared,
            source="declared",
            evidence="the caller declared this mode; no inference was made",
        )
    if contract is not None:
        return PromptModeDecision(
            mode=contract,
            source="contract",
            evidence=(
                "read from the bundle contract that shipped with this model, which records the "
                "convention the checkpoint was trained for"
            ),
        )

    share, seen = marker_share(prompts)
    found = ", ".join(seen) if seen else "none"
    if share >= _MARKER_CONFIDENT_SHARE:
        return PromptModeDecision(
            mode=PromptMode.PRERENDERED,
            source="inferred",
            evidence=(
                f"{share:.0%} of the held-out prompts already contain control tokens ({found}), so "
                "they were rendered before they reached this tool; applying a chat template to "
                "them would double-wrap them"
            ),
            marker_share=share,
        )
    if share <= 1.0 - _MARKER_CONFIDENT_SHARE:
        return PromptModeDecision(
            mode=PromptMode.RUNTIME_RENDERED,
            source="inferred",
            evidence=(
                f"{share:.0%} of the held-out prompts contain control tokens, so they are bare "
                "text and the runtime has to render its own template around them"
            ),
            marker_share=share,
        )
    # Neither shape. Something has to run, so the prompts are used as they are --
    # the option that transforms nothing and leaves the evidence in the record --
    # and the inconsistency is stated rather than hidden.
    return PromptModeDecision(
        mode=PromptMode.PRERENDERED,
        source="inferred",
        evidence=(
            f"{share:.0%} of the held-out prompts contain control tokens ({found}) and the rest do "
            "not: the split mixes the two conventions, so no single mode is right for all of it. "
            "The prompts were used verbatim; declare the mode explicitly to remove the guess"
        ),
        marker_share=share,
        ambiguous=True,
    )


@dataclass(frozen=True)
class DecodeConfig:
    """Decoding parameters. Part of what makes two points comparable."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0  # greedy
    top_k: int = 1

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    def as_dict(self) -> dict:
        return {"max_tokens": self.max_tokens, "temperature": self.temperature, "top_k": self.top_k}

    @property
    def fingerprint(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


GREEDY = DecodeConfig()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class DataError(ValueError):
    """The held-out file could not be read as held-out data."""


@dataclass(frozen=True)
class Example:
    index: int
    prompt: str
    # A call for `tool-call` scoring, a string for `exact-text`, `None` for an
    # unlabelled row -- which contributes to liveness only.
    target: ToolCall | str | None


@dataclass(frozen=True)
class Split:
    """A held-out split, identified by its content rather than its location.

    Replacing the file at a path must change the identity of every measurement
    taken on it, so the id is a hash of the examples actually used -- including
    the effect of `--limit`, because a 64-example slice is a different sample
    from the 640 it was cut out of, and the README's headline mistake was
    treating one as evidence about the other.
    """

    id: str
    source: str
    examples: tuple[Example, ...]
    limit: int | None = None

    @property
    def n(self) -> int:
        return len(self.examples)

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(e.prompt for e in self.examples)

    @property
    def labelled(self) -> tuple[Example, ...]:
        return tuple(e for e in self.examples if e.target is not None)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "n": self.n,
            "n_labelled": len(self.labelled),
            "limit": self.limit,
        }


def load_split(path: Path, limit: int | None = None) -> Split:
    """Read held-out JSONL. Raises `DataError` naming the offending line."""
    text = path.read_text(encoding="utf-8")
    examples: list[Example] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(obj, dict) or "prompt" not in obj:
            raise DataError(f"{path}:{lineno}: expected an object with a 'prompt' field")
        try:
            target = read_target(obj.get("target"))
        except ValueError as exc:
            raise DataError(f"{path}:{lineno}: {exc}") from exc
        examples.append(Example(index=len(examples), prompt=str(obj["prompt"]), target=target))
        if limit is not None and len(examples) >= limit:
            break
    if not examples:
        raise DataError(f"{path}: no examples")

    payload = json.dumps(
        [
            {
                "prompt": e.prompt,
                "target": (e.target.as_dict() if isinstance(e.target, ToolCall) else e.target),
            }
            for e in examples
        ],
        sort_keys=True,
    )
    return Split(
        id=hashlib.sha256(payload.encode()).hexdigest()[:16],
        source=str(path),
        examples=tuple(examples),
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Generations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """What came back for one prompt.

    `returncode is None` together with `harness_error` means the generation was
    never performed -- a missing shared library, an environment that would not
    build, a process that never started. That is not a model failure and must
    not be scored as one; it is the single most repeated mistake in the work
    that produced this tool.
    """

    index: int
    prompt: str
    text: str = ""
    returncode: int | None = None
    stderr: str = ""
    harness_error: str | None = None
    # The exit status of the process that produced the *batch*, when it differs
    # from this generation's own. One `transformers` process answers a whole
    # split, so a script that wrote every result and then failed on the way out
    # is not a failure of any single generation -- but recording only
    # `returncode=0` erased the fact entirely, and putting the real code here
    # would make `ok` false for output that exists and is scoreable.
    batch_returncode: int | None = None

    @property
    def ran(self) -> bool:
        return self.harness_error is None and self.returncode is not None

    @property
    def ok(self) -> bool:
        return self.ran and self.returncode == 0


# Output from a generation host that means the host, not the model, failed.
# `litert-lm` dlopen()s a vulkan-linked library even for the CPU backend; when
# it is absent every invocation, `--help` included, dies in under a second and
# looks exactly like a model that cannot generate.
_HOST_FAILURE_RE = re.compile(
    r"(libvulkan|error while loading shared libraries|cannot open shared object file"
    r"|ModuleNotFoundError|ImportError|command not found|No such file or directory)",
    re.IGNORECASE,
)

# glog-style banner lines and the runtime's own timing block, neither of which
# is model output.
_LOG_LINE_RE = re.compile(r"^[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d+\s")
_STATS_LINE_RE = re.compile(r"^\s*(Prefill|Decode)\s+(speed|latency)\b", re.IGNORECASE)


def strip_runtime_noise(stdout: str) -> str:
    """Drop the runtime's own chatter from captured stdout.

    A heuristic, and deliberately a narrow one: if it removes too much, the
    non-empty liveness check fails loudly rather than the score quietly
    dropping.
    """
    kept = [
        line
        for line in stdout.splitlines()
        if not _LOG_LINE_RE.match(line) and not _STATS_LINE_RE.match(line)
    ]
    return "\n".join(kept).strip()


class GenerationBackend(Protocol):
    """A model that can be asked for completions.

    Implementations report their own identity and prompt-construction mode so
    that the measurement can record what produced it, and so that
    `harness_mismatch` can refuse to compare two points that were not produced
    the same way.
    """

    @property
    def name(self) -> str: ...

    @property
    def model_ref(self) -> str: ...

    @property
    def prompt_mode(self) -> PromptMode: ...

    @property
    def decode(self) -> DecodeConfig: ...

    @property
    def decode_enforced(self) -> bool:
        """Whether `decode` governed this run or merely describes it.

        On the Protocol rather than inside `describe()`, because a backend that
        forgets it must fail to type-check rather than be silently recorded as
        enforcing parameters it never received. The first version of this was
        `describe().get("decode_passed_to_cli", True)` in one place and
        `.get("decode_passed_to_cli")` -- defaulting to None -- in another, so
        two backends that both omitted the key compared equal and the
        limitation they exist to raise disappeared from the manifest.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Engine identity: which backend and which pinned versions produced this."""

    def generate(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> list[Generation]:
        """One `Generation` per prompt, in order. Never raises for a failed run."""


@dataclass
class LiteRtLmBackend:
    """Generation through `litert-lm run`, one process per prompt.

    One process per prompt is what the measurement runs did, and roughly 90% of
    their wall time was process startup and model reload rather than decoding.
    The fix is `litert-lm serve` plus a persistent client -- worth about a
    thirtyfold reduction in evaluation time -- and it is deliberately not
    implemented yet, because the serve path changes the prompt-construction
    surface and that has to be measured before it is trusted.

    Measurement here runs on CPU while users run on a phone. Measured
    2026-09-05 on one Snapdragon Galaxy S24, one recipe: the device's CPU
    scored 0.8703 ±0.026 on 640 rows against 0.8906, 0.9016 and 0.8969 for
    the three reference runs of that recipe -- within about 0.03, with two of
    the three just outside the device interval. Its GPU is another matter --
    20/20 tool names on 20 rows when the bundle carries
    `prefer_activation_type = fp32`, `<pad>` floods and 3/20 when it does not
    (see `export.GPU_ACTIVATION`) -- and `describe()` records which backend
    and engine produced each figure.
    """

    model: Path
    backend_flag: str = "cpu"
    decode: DecodeConfig = GREEDY
    timeout_s: int = 300
    env: envs.StageEnv = envs.RUNTIME
    auto_provision: bool = True
    extra_args: tuple[str, ...] = ()
    # The mode this measurement is taken in, when the caller knows it. `None`
    # means nobody said, and the fallback below is what this backend has always
    # done rather than a considered answer for the model in hand -- so it is
    # recorded as undeclared in `describe()`. `verify` always sets it, from the
    # contract or from `resolve_prompt_mode`.
    declared_prompt_mode: PromptMode | None = None

    name = "litert-lm"
    # litetune passes no decoding flags to this CLI, so `decode` is a
    # declaration here and not an instruction. Stated as a value rather than
    # left to a default, because the asymmetry is what `verify` reports as a
    # limitation -- and it is litetune's gap, not the toolchain's: 0.16.1 does
    # accept --top-k, --top-p, --temperature and --seed.
    decode_enforced = False

    @property
    def prompt_mode(self) -> PromptMode:
        return self.declared_prompt_mode or UNDECLARED_PROMPT_MODE

    @property
    def uses_template(self) -> bool:
        return self.prompt_mode is PromptMode.RUNTIME_RENDERED

    @property
    def model_ref(self) -> str:
        return str(self.model)

    def argv(self, prompt: str) -> list[str]:
        # `--no-template` routes the CLI to create_session() instead of
        # create_conversation(), bypassing the chat template, the `<|turn>model`
        # anchor, tool handling and channel extraction. It belongs on a prompt
        # that already carries its own control tokens and nowhere else, which is
        # why it is tied to the mode rather than always passed.
        template_args = [] if self.uses_template else ["--no-template"]
        return [
            "litert-lm",
            "run",
            str(self.model),
            f"--backend={self.backend_flag}",
            *template_args,
            f"--prompt={prompt}",
            *self.extra_args,
        ]

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "litert-lm",
            "backend": self.backend_flag,
            # The flag as passed, not a device torch chose: see
            # `HuggingFaceBackend.describe`, where the same key carries the
            # other vocabulary.
            "backend_vocabulary": "litert-lm --backend flag",
            "requirements": list(self.env.requirements),
            "system_requirements": list(self.env.system_requirements),
            "argv_template": self.argv("<prompt>"),
            "prompt_mode": self.prompt_mode.value,
            "prompt_mode_declared": self.declared_prompt_mode is not None,
            "template_flag": None if self.uses_template else "--no-template",
            # Nothing here is passed to the CLI, so `decode` is the *declared*
            # configuration: greedy, to the runtime's own token limit. It is
            # recorded because comparability depends on it, and any deviation
            # must be passed through `extra_args` where it is visible in
            # `argv_template` above -- which is also how the CLI's own
            # --top-k/--top-p/--temperature/--seed would reach it today.
            "decode_declared": self.decode.as_dict(),
            "decode_passed_to_cli": self.decode_enforced,
        }

    def generate(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> list[Generation]:
        blocked = self._ensure_env(events)
        if blocked is not None:
            return [Generation(i, p, harness_error=blocked) for i, p in enumerate(prompts)]
        out: list[Generation] = []
        for i, prompt in enumerate(prompts):
            out.append(self._one(i, prompt))
            if events and (i + 1) % PROGRESS_EVERY == 0:
                events.note(
                    f"{self.name}: {i + 1}/{len(prompts)} prompts",
                    backend=self.name,
                    done=i + 1,
                    total=len(prompts),
                )
        return out

    def _ensure_env(self, events: EventStream | None) -> str | None:
        """Provision the runtime environment. Returns why it could not be, or None."""
        if not self.auto_provision:
            return None
        try:
            self.env.provision(events=events)
        except (RuntimeError, OSError) as exc:
            logger.exception("could not provision environment %r", self.env.name)
            return f"environment {self.env.name!r} unavailable: {exc}"
        return None

    def _one(self, index: int, prompt: str) -> Generation:
        argv = self.argv(prompt)
        try:
            proc = self.env.run(argv, timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("litert-lm timed out after %ss on prompt %d", self.timeout_s, index)
            # A hang is indistinguishable from a stalled environment from out
            # here, so it is recorded as not performed rather than as a verdict
            # about the model.
            return Generation(
                index, prompt, harness_error=f"no result after {self.timeout_s}s (timeout)"
            )
        except OSError as exc:
            logger.exception("could not start litert-lm for prompt %d", index)
            return Generation(index, prompt, harness_error=f"{type(exc).__name__}: {exc}")

        reading = read_returncode(proc.returncode)
        if not reading.conclusive:
            # Killed, not failed: the runtime never chose an exit status, so
            # this prompt was not measured. See `litetune.exits` -- a -9 read as
            # a failure is how a memory ceiling became a verdict about a model.
            logger.error("litert-lm was killed on prompt %d: %s", index, reading.describe())
            return Generation(
                index,
                prompt,
                returncode=proc.returncode,
                stderr=(proc.stderr or "")[-2000:],
                harness_error=reading.describe("the model"),
            )

        if proc.returncode != 0 and _HOST_FAILURE_RE.search(proc.stderr or ""):
            logger.error("litert-lm host failure on prompt %d: %s", index, proc.stderr[-400:])
            return Generation(
                index,
                prompt,
                returncode=proc.returncode,
                stderr=proc.stderr[-2000:],
                harness_error=f"generation host failed to start: {proc.stderr.strip()[-300:]}",
            )
        return Generation(
            index,
            prompt,
            text=strip_runtime_noise(proc.stdout or ""),
            returncode=proc.returncode,
            stderr=(proc.stderr or "")[-2000:],
        )


# Runs inside envs.TRAIN, which is the only environment with torch and
# transformers in it. Written to a temp file rather than passed with `-c` so
# that a traceback carries usable line numbers.
_HF_GENERATE_SCRIPT = r'''
"""Greedy generation for one split. Writes JSONL: {"index": int, "text": str}."""
import json
import sys
from pathlib import Path


def generation_device(torch, given=None):
    """Where generation happens: what the parent already resolved, or CUDA-if-any.

    `given` is `envs.resolve_device`'s answer, asked once by `HuggingFaceBackend`
    before this script started and taken here rather than decided again, so
    the parent knows the device of the run it is about to start. `tune.py`'s
    training script carries the same two lines for the same reason; they are
    duplicated rather than shared because neither script may import litetune --
    each runs inside a stage environment that has only torch in it.

    The fallback is not dead code. It is what runs whenever the parent has no
    answer to give: a probe that could not run, and an environment that was
    never probed. Without it the float reference would sit on the CPU of a GPU
    box because a sub-second probe failed.
    """
    if given is not None:
        return given
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["model"])
    # attn_implementation must match what training used. Gemma's default is
    # sdpa; training loads eager, and the reference notebooks warn three
    # separate times that a mismatch here produces garbage output. Leaving it
    # unset makes the float reference a different computation from the model
    # under test, which is a harness difference wearing a conversion cost's
    # clothes.
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        torch_dtype=torch.float32,
        attn_implementation=spec["attn_implementation"],
    )
    model.eval()
    device = generation_device(torch, spec.get("device"))
    model.to(device)
    # Structured, not only printed. The stderr line below is for a human
    # watching a run; `_assemble` discards stderr on a clean exit, so on the
    # path that matters it reaches nobody. This file is what the parent reads
    # back, and it is written before generation starts so that a run killed
    # part-way still says where it was running.
    Path(spec["run_report"]).write_text(json.dumps({"device": device}), encoding="utf-8")
    print(f"reference generation on {device}", file=sys.stderr, flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    with Path(spec["out"]).open("w", encoding="utf-8") as sink:
        for i, prompt in enumerate(spec["prompts"]):
            text = prompt
            if spec["runtime_rendered"]:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            enc = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                ids = model.generate(
                    **enc,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=spec["max_tokens"],
                    pad_token_id=pad_id,
                )
            # skip_special_tokens=False on purpose: the liveness tier checks for
            # padding-token leakage, and decoding it away would erase the
            # evidence it looks for.
            completion = tok.decode(ids[0][enc["input_ids"].shape[-1] :], skip_special_tokens=False)
            sink.write(json.dumps({"index": i, "text": completion}) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass
class HuggingFaceBackend:
    """Float reference generation through `transformers`, inside `envs.TRAIN`.

    The whole split runs in one process: unlike the runtime CLI there is no
    per-prompt startup to pay, and reloading a checkpoint 640 times would
    dominate the measurement.
    """

    model: str
    decode: DecodeConfig = GREEDY
    timeout_s: int = 3600
    env: envs.StageEnv = envs.TRAIN
    auto_provision: bool = True
    runtime_rendered: bool = False
    # Must match training. `spec.BaseModel.attn_implementation` carries the
    # same default and exists to be threaded here.
    attn_implementation: str = "eager"
    # Set by `verify` from the contract or from `resolve_prompt_mode`. It wins
    # over `runtime_rendered`, which is the low-level switch this backend has
    # always had; both are reported so a measurement never hides which one
    # decided it.
    declared_prompt_mode: PromptMode | None = None
    # Where the reference actually generated. Set twice: to what
    # `envs.resolve_device` found, before the run, and then to what the script
    # itself reported after it -- see `generate()`. `None` until `generate()`
    # has run, and after one whose probe could not answer and whose script
    # wrote no report; never silently "cpu", the same rule
    # `tune.TrainingMetrics.device` applies on the training side. `describe()`
    # reports this rather than a hardcoded constant, because a laptop's
    # manifest and a GPU box's must not be able to read identically in the one
    # field that says which produced the numbers.
    device: str | None = None
    # What the probe predicted, kept beside what the run reported rather than
    # replaced by it. They disagree whenever the script took its own fallback,
    # and a disagreement is a fact about the measurement -- `tune` keeps both
    # for the same reason (`TuneResult.device` against `metrics.device`). A
    # logger warning does not travel with a manifest.
    probed_device: str | None = None
    # This call's own probe, in full, overwritten every time `_ensure_env`
    # runs -- unlike `device` and `probed_device` above, which a failed probe
    # deliberately leaves alone so a reused backend keeps reporting its last
    # known answer. That do-not-erase rule protects the *report*; it must not
    # also protect a directive handed to the *child* process, which is what
    # reading `self.device` for `spec["device"]` used to do: a stale "cuda"
    # from a previous call, forced onto a run whose own probe could not vouch
    # for it, made `model.to("cuda")` fail on a box where CUDA had since gone
    # away. `generate()` reads this, not `self.device`, when it builds the
    # spec. `None` before `_ensure_env` has run, and after a call whose probe
    # was never asked or could not answer -- see `verify.py`, which also reads
    # `detail` and `cuda_build_without_a_device` off it to record the same
    # limitation `tune.py` records for its own probe.
    last_probe: envs.DeviceProbe | None = None

    name = "transformers"
    # `generate()` receives max_new_tokens and the stop condition, so here the
    # declared configuration is the applied one.
    decode_enforced = True

    @property
    def model_ref(self) -> str:
        return self.model

    @property
    def prompt_mode(self) -> PromptMode:
        # A declared mode wins. Otherwise the `runtime_rendered` switch decides,
        # and `harness_mismatch` refuses any comparison across the two modes
        # rather than attributing the mode's effect to conversion.
        if self.declared_prompt_mode is not None:
            return self.declared_prompt_mode
        return PromptMode.RUNTIME_RENDERED if self.runtime_rendered else PromptMode.PRERENDERED

    @property
    def uses_template(self) -> bool:
        return self.prompt_mode is PromptMode.RUNTIME_RENDERED

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "transformers",
            # Resolved by `generate()`, not hardcoded: this used to say "cpu"
            # unconditionally, which made a laptop's manifest and a GPU box's
            # byte-identical in the one field meant to distinguish them.
            # "unknown" before `generate()` has run, and after one whose device
            # was never established -- not "cpu", for the same reason
            # `TrainingMetrics.device` defaults to `None` rather than guessing.
            # "unknown" and not "unresolved": `verify.py` already prints
            # `unknown` into its limitation text for a missing or null value, so a
            # second word for the same state would have put two names for one
            # thing in one manifest.
            "backend": self.device if self.device is not None else UNKNOWN_BACKEND,
            # What was predicted before the run, so a reader can see the two
            # disagree rather than only the winner. `None` when nothing was
            # asked; equal to `backend` on every run that used what it was
            # told, which is the ordinary case.
            "backend_probed": self.probed_device,
            # `backend` is a torch device here and a `litert-lm --backend` flag
            # on `LiteRtLmBackend`: two vocabularies under one key, overlapping
            # only at the string "cpu", which means "torch placed the model on
            # the CPU" on this side and "the runtime was asked for its CPU
            # backend" on the other. Named so a reader comparing a candidate's
            # `backend` with a reference's is not comparing two different
            # questions.
            "backend_vocabulary": "torch device",
            "requirements": list(self.env.requirements),
            "decode_declared": self.decode.as_dict(),
            "decode_passed_to_cli": self.decode_enforced,
            "prompt_mode": self.prompt_mode.value,
            "prompt_mode_declared": self.declared_prompt_mode is not None,
            "applies_chat_template": self.uses_template,
        }

    def generate(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> list[Generation]:
        blocked = self._ensure_env(events)
        if blocked is not None:
            return [Generation(i, p, harness_error=blocked) for i, p in enumerate(prompts)]

        # This call's own probe only, not `self.device`: the do-not-erase rule
        # in `_ensure_env` deliberately lets `self.device` keep a stale answer
        # across calls for *reporting*, and that answer must not also become a
        # directive forced onto this child's device placement -- a probe that
        # could not vouch for it this run has no business deciding it. `None`
        # is exactly what `generation_device`'s own fallback inside the script
        # is for.
        device_for_run = self.last_probe.device if self.last_probe is not None else None

        with tempfile.TemporaryDirectory(prefix="litetune-hf-") as tmp:
            work = Path(tmp)
            script = work / "generate.py"
            script.write_text(_HF_GENERATE_SCRIPT, encoding="utf-8")
            results = work / "generations.jsonl"
            report = work / "run.json"
            spec = work / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "prompts": list(prompts),
                        "max_tokens": self.decode.max_tokens,
                        "runtime_rendered": self.uses_template,
                        "attn_implementation": self.attn_implementation,
                        "device": device_for_run,
                        "out": str(results),
                        "run_report": str(report),
                    }
                ),
                encoding="utf-8",
            )
            if events:
                events.note(
                    f"{self.name}: generating {len(prompts)} completions",
                    backend=self.name,
                    total=len(prompts),
                )
            try:
                proc = self.env.run(["python", str(script), str(spec)], timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning("transformers generation timed out after %ss", self.timeout_s)
                reason = f"no result after {self.timeout_s}s (timeout)"
                # Nothing ran, so nothing has a confirmed device: a prediction
                # left standing here would be reported by `describe()` as the
                # backend for a run that produced zero generations.
                self.device = None
                return [Generation(i, p, harness_error=reason) for i, p in enumerate(prompts)]
            except OSError as exc:
                logger.exception("could not start the generation script")
                reason = f"{type(exc).__name__}: {exc}"
                self.device = None
                return [Generation(i, p, harness_error=reason) for i, p in enumerate(prompts)]

            texts = self._read_results(results)
            # The script's own answer wins over the probe's prediction. The
            # temp directory goes away at the end of this block, so it is read
            # here rather than anywhere later.
            observed = self._read_run_report(report)

        if observed is not None:
            if self.device is not None and observed != self.device:
                logger.warning(
                    "the probe resolved %r but the generation script used %r; recording %r",
                    self.device,
                    observed,
                    observed,
                )
            self.device = observed
        return self._assemble(prompts, texts, proc)

    def _ensure_env(self, events: EventStream | None) -> str | None:
        """Provision the environment when asked to, then ask it its device.

        The probe is sub-second next to the generation run it precedes, and it
        is asked here -- once, in the parent -- rather than left to
        `generation_device`'s own fallback inside the subprocess.

        Gated on `env.ready`, not on `auto_provision`. The probe provisions
        nothing; a ready environment is its only precondition, and it is about
        to be used for the generation run regardless. Under the old gate a
        caller that constructs this backend with `auto_provision=False` over an
        environment that is already there -- a library caller managing the
        lifecycle itself, and every test in this suite that does the same --
        got `describe()["backend"] == "unknown"` for a run whose device was
        perfectly knowable. (No CLI path reaches that state: `verify` has no
        `--no-provision` flag and never sets this field.)

        A probe that cannot answer leaves `device` alone rather than
        overwriting it: on a reused backend the previous run's answer is a
        better record than `None`, and `None` here would claim the device was
        never established when it was. A blocked environment does clear it,
        because then no run happened at all.
        """
        self.last_probe = None
        if self.auto_provision:
            try:
                self.env.provision(events=events)
            except (RuntimeError, OSError) as exc:
                logger.exception("could not provision environment %r", self.env.name)
                self.device = None
                self.probed_device = None
                return f"environment {self.env.name!r} unavailable: {exc}"
        if not self.env.ready:
            # A third state, and it used to be reported as the first: "nobody
            # asked", "it was asked and could not say", and "there was nothing
            # to ask". Leaving `last_probe` at `None` made the manifest
            # byte-identical to a run where no probe was wanted.
            #
            # Recorded, not blocking. Whether an unprovisioned environment can
            # still generate is the caller's to decide -- a library caller that
            # manages the lifecycle itself, or a test that supplies its own
            # `run`, legitimately gets here and proceeds. What must not happen
            # is the run finishing with no trace of why its device is unknown.
            self.last_probe = envs._unanswered(
                self.env, f"it is not provisioned at {self.env.path}", events
            )
            return None
        probe = envs.resolve_device(self.env, events=events)
        self.last_probe = probe
        if probe.answered:
            self.probed_device = probe.device
            self.device = probe.device
        return None

    def _read_run_report(self, report: Path) -> str | None:
        """The device the generation script itself says it used, or `None`.

        The probe before the run is a prediction; this is the observation.
        They differ whenever the script took its own fallback -- an unanswered
        probe, an environment nobody probed -- and whenever the device changed
        underneath the two. `tune.py`'s training script already writes a
        metrics file for its own reasons, so recovering this observation from
        it costs nothing there; the reference side had no such file lying
        around and needed this purpose-built report to get the same fact.
        """
        try:
            device = json.loads(report.read_text(encoding="utf-8"))["device"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            # A script that died before writing this, an older one that never
            # wrote it, or one whose write was cut short mid-byte -- the same
            # decode failure `_read_results` below already accounts for. Not
            # an error: the prediction stands and says so.
            logger.warning("generation script wrote no usable run report: %s", exc)
            return None
        return device if isinstance(device, str) else None

    def _read_results(self, results: Path) -> dict[int, str]:
        if not results.exists():
            return {}
        texts: dict[int, str] = {}
        try:
            body = results.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            # Not decodable at all: report it as one fault rather than letting
            # it escape from a loop that exists to report per-line ones.
            logger.warning("results file %s is not valid UTF-8: %s", results.name, exc)
            return {}
        for lineno, line in enumerate(body.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # Which line, and what it looked like: the eventual error is
                # "no result for prompt N", which says nothing about why.
                logger.warning(
                    "malformed result on line %d of %s (%s): %.120s",
                    lineno,
                    results.name,
                    exc,
                    line,
                )
                continue
            try:
                texts[int(row["index"])] = str(row["text"])
            except (KeyError, TypeError, ValueError) as exc:
                # Valid JSON, wrong shape. Same fault class as a malformed line
                # and it must not escape the loop that reports them.
                logger.warning(
                    "result on line %d of %s lacks a usable index/text (%s)",
                    lineno,
                    results.name,
                    exc,
                )
        return texts

    def _assemble(
        self, prompts: Sequence[str], texts: dict[int, str], proc: subprocess.CompletedProcess
    ) -> list[Generation]:
        stderr = (proc.stderr or "")[-2000:]
        reading = read_returncode(proc.returncode)
        # A killed script is not a script that failed: `-9` is the out-of-memory
        # killer, and generation for a whole split is exactly the kind of job
        # that meets a memory ceiling. See `litetune.exits`.
        died = (
            reading.describe("the model")
            if not reading.conclusive
            else f"generation script exited {proc.returncode}"
        )
        # A script that wrote every result and *then* exited non-zero used to be
        # recorded as a clean run: each result carried `returncode=0, stderr=""`
        # and the process failure was erased. The generation did happen, so it
        # keeps its text and stays scoreable -- but the evidence travels with it.
        failed_after_writing = texts and proc.returncode != 0
        if failed_after_writing:
            logger.warning(
                "generation script wrote %d results and exited %s", len(texts), proc.returncode
            )
        out: list[Generation] = []
        for i, prompt in enumerate(prompts):
            if i in texts:
                out.append(
                    Generation(
                        i,
                        prompt,
                        text=texts[i].strip(),
                        returncode=0,
                        stderr=stderr if failed_after_writing else "",
                        batch_returncode=proc.returncode if failed_after_writing else None,
                    )
                )
                continue
            # A missing result cannot be distinguished from here between "the
            # model failed on this prompt" and "the environment died before
            # reaching it", so it is recorded as not performed. That is the
            # conservative direction: `could not check` rather than a verdict
            # the run did not earn.
            out.append(
                Generation(
                    i,
                    prompt,
                    returncode=proc.returncode,
                    stderr=stderr,
                    harness_error=(
                        f"{died} without a result for this prompt: "
                        f"{stderr.strip()[-300:] or 'no stderr'}"
                    ),
                )
            )
        return out


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasurementPoint:
    """One (model, backend, split) measurement and everything needed to trust it."""

    label: str
    model_ref: str
    backend: str
    prompt_mode: PromptMode
    decode: DecodeConfig
    split_id: str
    engine: dict[str, Any]
    generations: tuple[Generation, ...] = ()
    # Whether `decode` governed this measurement or merely describes it.
    # litetune hands the device side nothing, so a device point declares the
    # same config the reference point enforces -- and comparing the two
    # fingerprints finds them equal, which is agreement about a declaration
    # rather than about what ran.
    # Required and keyword-only, with no default. `True` here is the optimistic
    # answer, and a MeasurementPoint built anywhere other than `evaluate()` -- a
    # test, a replay, a manifest reader -- would take it, compare equal to its
    # counterpart, and drop the limitation. That is the `.get(key, True)`
    # failure this field replaced, moved one layer down.
    decode_enforced: bool = field(kw_only=True)

    @property
    def n(self) -> int:
        return len(self.generations)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(g.text for g in self.generations)

    @property
    def n_unavailable(self) -> int:
        return sum(1 for g in self.generations if g.harness_error is not None)

    @property
    def batch_failures(self) -> int:
        """Generations whose producing process exited non-zero after writing.

        Each one is usable output, so it is scored -- but the process that made
        them did not end cleanly, and a typed field nothing reads is the same
        erasure as the `returncode=0` it replaced.
        """
        return sum(1 for g in self.generations if g.batch_returncode not in (None, 0))

    @property
    def n_failed(self) -> int:
        return sum(1 for g in self.generations if g.ran and not g.ok)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "model": self.model_ref,
            "backend": self.backend,
            "prompt_mode": self.prompt_mode.value,
            "decode": self.decode.as_dict(),
            "split_id": self.split_id,
            "engine": self.engine,
            "generations": {
                "n": self.n,
                "ok": self.n - self.n_failed - self.n_unavailable,
                "failed": self.n_failed,
                "not_performed": self.n_unavailable,
                # Scored, but produced by a process that did not exit cleanly.
                "from_a_failed_batch": self.batch_failures,
            },
            "decode_enforced": self.decode_enforced,
        }


def evaluate(
    backend: GenerationBackend,
    split: Split,
    label: str,
    events: EventStream | None = None,
) -> MeasurementPoint:
    """Run one measurement point. The only generation entry point in the tool."""
    if events:
        events.note(
            f"{label}: {split.n} prompts through {backend.name} ({backend.prompt_mode.value})",
            label=label,
            backend=backend.name,
            model=backend.model_ref,
            n=split.n,
        )
    generations = backend.generate(split.prompts, events=events)
    if len(generations) != split.n:
        # The backend contract is one generation per prompt, in order. Anything
        # else makes indexing into targets wrong, which would silently score
        # the wrong pairs.
        raise ValueError(
            f"{backend.name} returned {len(generations)} generations for {split.n} prompts"
        )
    return MeasurementPoint(
        label=label,
        model_ref=backend.model_ref,
        backend=backend.name,
        prompt_mode=backend.prompt_mode,
        decode=backend.decode,
        decode_enforced=backend.decode_enforced,
        split_id=split.id,
        engine=backend.describe(),
        generations=tuple(generations),
    )


def device_mismatch(a: MeasurementPoint, b: MeasurementPoint) -> str | None:
    """The two points ran on different hardware, said in full, or None.

    A limitation, not a refusal, and the difference from `harness_mismatch` is
    the decision rather than an oversight. `build_backends` gives the candidate
    `LiteRtLmBackend`'s default `backend_flag="cpu"` and never overrides it, so
    on a machine where the reference now resolves to cuda the two sides differ
    in hardware as well as in conversion, and the "cost of conversion" carries
    both. Refusing that comparison would leave a GPU box unable to verify at
    all, which is worse than a number that says what else is in it.

    Read from `engine["backend"]`, which each backend fills with the device or
    the flag it actually used. The two vocabularies do not overlap except at
    the string "cpu" -- see `backend_vocabulary` in either `describe()` -- so
    this reports both values and names neither as the right one. A value that
    is missing or `UNKNOWN_BACKEND` is not a difference: nothing was
    established to differ from, and `describe()` already says so.
    """
    left = a.engine.get("backend")
    right = b.engine.get("backend")
    if not isinstance(left, str) or not isinstance(right, str):
        return None
    if UNKNOWN_BACKEND in (left, right) or left == right:
        return None
    return (
        f"{a.label} was measured on {left} ({a.engine.get('engine', 'unknown')}) and {b.label} on "
        f"{right} ({b.engine.get('engine', 'unknown')}), so the difference between them carries a "
        "hardware difference as well as a conversion one. The comparison is reported rather than "
        "refused: pinning both sides to one device is not something litetune can do for the "
        "runtime side, and a refusal would leave such a machine unable to verify at all"
    )


def harness_mismatch(a: MeasurementPoint, b: MeasurementPoint) -> str | None:
    """Why `a` and `b` cannot be compared, or None if they can.

    Prompt mode is checked first and is the reason this function exists. Under
    `--no-template` the runtime's tool list is null, so a runtime-rendered
    measurement and a pre-rendered one differ by the whole declaration block;
    subtracting them reports the effect of the rendering mode as though it were
    the effect of conversion. That comparison is refused, not annotated.
    """
    if a.prompt_mode is not b.prompt_mode:
        return (
            f"{a.label} was measured {a.prompt_mode.value} and {b.label} "
            f"{b.prompt_mode.value}; the two prompt-construction modes are mutually "
            "exclusive, so their difference measures the mode rather than the model"
        )
    if a.split_id != b.split_id:
        return f"different held-out splits: {a.split_id} against {b.split_id}"
    if a.decode.fingerprint != b.decode.fingerprint:
        return f"different decoding: {a.decode.fingerprint} against {b.decode.fingerprint}"
    # Unequal *enforcement* of equal decoding is deliberately not refused here.
    # The runtime may strip the terminator before we ever see the text, so
    # "ends without a terminator" cannot distinguish a cut generation from a
    # clean one on the device side -- and a check built on it would refuse every
    # three-point comparison forever. It is reported as a measured limitation
    # instead, with `liveness.unterminated_count` as its magnitude.
    if a.n != b.n:
        return f"{a.n} generations against {b.n}"
    return None
