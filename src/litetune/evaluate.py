"""One evaluator, used for every measurement point.

The points this tool reports -- reference float, tuned float, tuned converted --
are only meaningful when subtracted from one another, and subtraction is only
valid if both sides saw the same split, the same prompts, the same prompt
construction and the same decoding. Three separate evaluators drift within
weeks, at which point the difference between two points stops measuring what it
claims to. So there is one evaluator here, parameterized by (model reference,
backend, split), and the parameters it was given travel with the result.

`PromptMode` travels with every measurement for a specific reason: asking the
runtime for a pre-rendered prompt forces its tool list to null, so a model
whose declarations are rendered by the runtime and one whose prompt was built by
the application cannot be compared at all -- the difference measures the mode,
not the model. `harness_mismatch` exists to refuse that comparison rather than
report it.

Nothing in this module raises on a failed generation. A non-zero exit is an
observation; a process that never started is a *different* observation, and
`Generation` keeps them apart rather than scoring either as a wrong answer.

Where each lands: a driver script that wrote its results and then exited
non-zero keeps its text, and the exit travels beside it in `batch_returncode`;
a prompt with no result of its own carries a `harness_error`, which `ran` reads
as "not performed". No backend here builds a generation that ran and is not ok,
so the `failed` count in `MeasurementPoint.as_dict` is currently always zero --
the shape of a verdict, not one being reported.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from litetune import envs
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.export import GPU_ACTIVATION
from litetune.metrics import ToolCall, read_target
from litetune.prompt_mode import RENDERING_SOURCE, PromptMode

logger = logging.getLogger(__name__)


DEFAULT_MAX_TOKENS = 256


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

# The one backend a request establishes on its own: nothing falls back past it.
CPU_BACKEND = "cpu"

# Whether `describe()["backend"]` is something the run established or something
# it asked for. One key has carried both: `HuggingFaceBackend` reads its device
# back out of the script's own run report and writes `UNKNOWN_BACKEND` when
# nothing answered, while the litert-lm backends write the flag they passed.
#
# That difference is invisible while the flag can only be "cpu". It stops being
# invisible the moment a GPU flag exists, because `verify` silences its
# "the GPU backend is a different executor" caveat for a run whose backend
# *is* the GPU.
#
# Nothing in this tree reads an accelerator back off a litert-lm engine: the
# three construction sites (`_LITERTLM_GENERATE_SCRIPT`, `toolpath.py`,
# `rendering.py`) pass a `Backend` in and read no device out, and the only
# field any of them reads afterwards is a token count off `BenchmarkInfo`
# (`rendering.py`). Whether the runtime could answer at all is a question
# about litert-lm 0.16.1, which is installed in `envs.RUNTIME` and not here --
# so it is a question a reader has to take to that package, and one this
# comment does not answer for them.
#
# What a request is not: on Linux with litert-lm 0.16.1, 2026-09-25, an engine
# built with `Backend.GPU()` on a machine with no usable GPU was created
# without error and then produced no token for a single prompt in fifty-five
# minutes. That is what "asked for" has to be allowed to mean.
#
# So the caveat is silenced only where the value was observed. A backend that
# cannot observe says so, and says it here rather than by convention.
BACKEND_OBSERVED = "backend_observed"


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
                # Flattened, as every split id before targets kept their types:
                # the same file must keep the same id.
                "target": (
                    {"name": e.target.name, "args": dict(e.target.args)}
                    if isinstance(e.target, ToolCall)
                    else e.target
                ),
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
# What `errors="surrogateescape"` leaves behind: one lone surrogate per byte
# that was not UTF-8, in a range no correct text can contain. A generation
# carrying one is a decode that failed, not a model that answered.
#
# Deliberately not U+FFFD. `errors="replace"` would write that instead, and
# U+FFFD is a character a model may generate -- it is in the vocabulary of
# every byte-level tokenizer, and a dataset that has been through a lossy
# decode once is full of it. Refusing on U+FFFD would call a real answer a
# harness failure; a surrogate in this range cannot be anything but a byte
# that did not decode.
UNDECODED_BYTE = re.compile("[\udc80-\udcff]")

# `batch_returncode` for a run that was killed rather than one that exited.
# Not a signal number: `TimeoutExpired` carries no status, the stage kills with
# SIGTERM and only then SIGKILL (`envs.py`), and Windows has no SIGKILL
# (`envs.py` reads it with `getattr` for that reason; SIGTERM it has) -- so any
# number here would assert something nobody observed. What it has to be is
# non-zero and not `None`, which is all `MeasurementPoint.batch_failures`
# reads; the reason travels in `stderr` and in the limitation `verify` emits.
BATCH_KILLED = -1

# Shared by the two driver-script backends below. Neither touched `self`, and
# a second copy of this error handling is the duplication this file has been
# bitten by before: the same fault has to be reported the same way whichever
# environment the script ran in.


def read_jsonl_results(results: Path) -> tuple[dict[int, str], dict[int, str]]:
    """The texts a driver script wrote, and the rows that cannot be scored.

    Two channels, because "no result for this prompt" and "this prompt's
    result is not the model's answer" are different facts and the second one
    used to be reported as the first. A row carrying a byte that did not
    decode is the case: the runtime wrote it, `json.dumps` escaped it as a
    lone surrogate, and the file stayed valid UTF-8 all the way here -- so
    nothing raises and the row would score as an ordinary wrong answer, with
    every liveness check passing and the loss reported as a conversion cost.

    The shapes it accepts are {"index": int, "text": str} and, for a prompt the
    runtime refused, {"index": int, "error": str}. Everything else is a fault
    against whichever prompt it names, or, where it names none, a warning and
    a dropped line. The two maps it returns never both claim a prompt.
    """
    if not results.exists():
        return {}, {}
    texts: dict[int, str] = {}
    faults: dict[int, str] = {}

    def fault(index: int, message: str) -> None:
        # The two maps must not both claim a prompt: every caller checks
        # `faults` first, so a text left behind here is invisible until
        # someone reads `texts` alone -- and the duplicate branch below was
        # the only path that remembered to drop it. A file can disagree with
        # itself in more ways than that: a row with the text and a second row
        # with an error for the same prompt reached here as a fault with the
        # text still standing.
        faults[index] = message
        texts.pop(index, None)

    try:
        body = results.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # Not decodable at all: report it as one fault rather than letting
        # it escape from a loop that exists to report per-line ones.
        logger.warning("results file %s is not valid UTF-8: %s", results.name, exc)
        return {}, {}
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
            index = row["index"]
        except KeyError as exc:
            # Valid JSON, wrong shape. Same fault class as a malformed line
            # and it must not escape the loop that reports them.
            logger.warning(
                "result on line %d of %s lacks a usable index/text (%s)",
                lineno,
                results.name,
                exc,
            )
            continue
        # `int(index)` was how this read the field, and `int` is a coercion
        # rather than a check: `int(1.9)` is 1 and `int(True)` is 1, so a row
        # carrying either took prompt 1's slot and was scored as its answer.
        # What binds a row to a prompt is this integer and nothing else, so it
        # has to be one already. `bool` is an `int` subclass and is excluded
        # for the same reason it was reachable.
        if not isinstance(index, int) or isinstance(index, bool):
            logger.warning(
                "result on line %d of %s carries a %s where its index should be",
                lineno,
                results.name,
                type(index).__name__,
            )
            continue
        if isinstance(row.get("error"), str):
            # The driver reached this prompt, the runtime refused it, and the
            # split carried on. Not "no result": the run learned something
            # about this prompt and what it learned was a failure.
            logger.error("prompt %d was refused by the runtime: %s", index, row["error"])
            fault(
                index,
                f"the runtime raised on this prompt and the split continued without it: "
                f"{row['error'][:200]}",
            )
            continue
        text = row.get("text")
        if not isinstance(text, str):
            # `str(text)` accepted all of these, so the guard above only ever
            # caught a bad index: a JSON `null` came back as the four
            # characters "None" and scored as the model's answer, with every
            # liveness check green. Reported against its own prompt rather
            # than dropped, because the row did arrive.
            logger.error(
                "prompt %d came back with a %s where its text should be",
                index,
                type(text).__name__,
            )
            fault(
                index,
                f"the runtime's output for this prompt was a {type(text).__name__}, not a "
                f"string, so what it generated is not recoverable from this run",
            )
            continue
        undecoded = UNDECODED_BYTE.findall(text)
        if undecoded:
            # The runtime wrote a byte that is not UTF-8. `json.dumps` escaped
            # it as a lone surrogate, so the file is valid UTF-8 and nothing
            # raised on the way here -- scored, the row is wrong, every
            # liveness check passes, and the loss is reported as a conversion
            # cost. What the model generated is not in this string.
            logger.error("prompt %d came back with bytes that are not UTF-8", index)
            fault(
                index,
                f"the runtime's output for this prompt was not UTF-8: {len(undecoded)} "
                f"byte(s) did not decode, so what it generated is not recoverable from "
                f"this run",
            )
            continue
        if index in texts or index in faults:
            # Silently the last row won. Both are suspect: nothing here can
            # say which of them the runtime produced for that prompt.
            logger.error("two rows in %s claim index %d", results.name, index)
            fault(
                index,
                f"two rows in the results file claim prompt {index}, so which of them the "
                f"runtime produced for it is not recoverable from this run",
            )
            continue
        texts[index] = text
    return texts, faults


def report_stray(
    prompts: Sequence[str], texts: dict[int, str], faults: dict[int, str]
) -> list[int]:
    """Rows whose index no prompt claims, said out loud.

    Dropped without a word before -- including a fault the reader had already
    logged at ERROR, which then vanished for being unclaimable. What binds row
    N to prompt N is an integer in a file and nothing else, so a row nobody
    claims is the visible end of that.
    """
    stray = sorted((texts.keys() | faults.keys()) - set(range(len(prompts))))
    if stray:
        logger.error("%d result row(s) carry an index no prompt has: %s", len(stray), stray[:10])
    return stray


def salvage_after_kill(
    prompts: Sequence[str],
    texts: dict[int, str],
    faults: dict[int, str],
    stderr: str,
    reason: str,
) -> list[Generation]:
    """What a killed split still knows, with the kill travelling beside it.

    A salvaged row is a real answer and is scored -- and it has to say the run
    did not end. With `returncode=0` and nothing else, a split killed after
    writing every row was byte-identical to a clean one, and liveness reported
    "n/n generations exited zero" about a group that was SIGKILLed. That is the
    erasure `batch_returncode` exists to prevent.

    Shared rather than written twice: both driver scripts flush after every
    row, so both have rows to save, and the same fault has to be reported the
    same way whichever environment the script ran in.
    """
    report_stray(prompts, texts, faults)
    out: list[Generation] = []
    for i, prompt in enumerate(prompts):
        if i in faults:
            out.append(Generation(i, prompt, stderr=stderr, harness_error=faults[i]))
        elif i in texts:
            out.append(
                Generation(
                    i,
                    prompt,
                    text=texts[i].strip(),
                    returncode=0,
                    stderr=stderr,
                    batch_returncode=BATCH_KILLED,
                )
            )
        else:
            out.append(Generation(i, prompt, stderr=stderr, harness_error=reason))
    return out


def assemble_generations(
    prompts: Sequence[str],
    texts: dict[int, str],
    proc: subprocess.CompletedProcess,
    faults: dict[int, str] | None = None,
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
    faults = faults or {}
    report_stray(prompts, texts, faults)
    out: list[Generation] = []
    for i, prompt in enumerate(prompts):
        if i in faults:
            # Said in its own words rather than as the absence below: the row
            # arrived, and what is wrong with it is not that it is missing.
            out.append(
                Generation(
                    i, prompt, returncode=proc.returncode, stderr=stderr, harness_error=faults[i]
                )
            )
            continue
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

    @property
    def backend_observed(self) -> bool:
        """Whether `describe()["backend"]` is a device this run read back.

        Here for the same reason as `decode_enforced`, and against the same
        defect: the first version of this was a bare `describe()` key with an
        implicit default, which is exactly the shape that comment describes
        going wrong. A backend that can read its device back must say so, and
        one that forgets must fail to type-check rather than have its request
        quietly reported as a measurement.

        False is not "the backend is wrong". It is "nothing here established
        it", which is the truth for every backend that passes a flag to a
        runtime and is told nothing in return.
        """
        ...

    @property
    def scores_structurally(self) -> bool:
        """Whether this backend's answers are calls rather than text.

        Here for the same reason as `decode_enforced`: a run whose model
        answered in prose on every prompt is indistinguishable from a text run
        by looking at the rows, so the backend declares it and a backend that
        forgets fails to type-check rather than being scored the wrong way.
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
    """Generation through litert-lm's Python API, one process per split.

    It used to be `litert-lm run` once per prompt: a 600-row verify started
    the process and reloaded the bundle six hundred times. Beside the bundle
    litert-lm writes an XNNPACK cache, and MEASUREMENTS.md records both for
    `weight_only_wi8_afp32` -- 2,385,932,008 bytes of cache against that
    bundle's 771,404,928 -- and a cache of 601,946,856 for `dynamic_wi8_afp32`,
    whose bundle size it does not give. What a reload reads, and what share of
    a run's wall time six hundred of them were, nothing here measured: the file
    closes that paragraph with "Nothing measured whether the cache is the
    reason."

    `litert-lm serve` plus a persistent client was the fix this docstring used
    to name, and it was rejected for the right reason: the serve path changes
    the prompt-construction surface, and that would have to be measured before
    it could be trusted. The API does not. The CLI is a thin wrapper over the
    same `Engine` and the same `create_session` / `create_conversation`, so
    this is that code reached directly rather than a second way of asking --
    which is why `runner_call` names the call instead of a flag.

    Measured rather than argued, because a transport that changed the text
    would make every number taken before it incomparable with every number
    after. On Linux CPU with `litert-lm==0.16.1`, both prompt modes, against
    the CLI path's own stdout cleaning: 36 of 36 comparisons byte-identical --
    9 prompts in both prompt modes on `Qwen3-0.6B.litertlm` from
    `litert-community/Qwen3-0.6B`, which holds four bundles, and the same 9 in
    both on `mobile-actions_q8_ekv1024.litertlm` from
    `litert-community/functiongemma-mobile-actions_q8_ekv1024.litertlm`, a
    different family, template and tokenizer. The script compared was this one:
    the run extracted it from this file rather than restating it, and the
    extraction hashes equal.

    An earlier revision of the script was run beside it on the same prompts and
    agreed on all 36 as well, which retires the two rounds taken against that
    revision and answers the one thing they could not: that revision never
    closed a channel, so had any chunk carried one the two would have differed
    by a ` [/name]` and a newline. None did. Nothing in this run composed a
    channel.

    Which is what those 36 do not cover, and it is most of the risk. Eighteen
    are session-path comparisons, and `text_from_session` reads `chunk.texts`
    and nothing else, so there is no channel branch for them to take; the other
    eighteen took it and found nothing to compose, on two families whose
    templates differ.
    Channel composition -- the one part of the driver script that is not a
    direct call -- is held by the synthetic streams in
    `test_the_driver_composes_what_the_cli_printed` instead, because a sampled
    model can only show a branch was not taken. Nothing here covers the GPU
    backend.

    Two things the change costs, both named where they happen: `timeout_s`
    still means the budget one prompt gets and the split gets the sum, so a
    hung prompt now takes the budget the rest would have had; and progress is
    one note at the start rather than a count every twenty-five prompts,
    because the parent is blocked on one child instead of watching six hundred
    finish.

    Measurement here runs on this machine -- its CPU by default, its GPU with
    `backend_flag="gpu"` -- while users run on a phone. Measured
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
    # The mode this measurement is taken in, when the caller knows it. `None`
    # means nobody said, and the fallback below is what this backend has always
    # done rather than a considered answer for the model in hand -- so it is
    # recorded as undeclared in `describe()`. `verify` always sets it, from the
    # contract or from `resolve_prompt_mode`.
    declared_prompt_mode: PromptMode | None = None

    name = "litert-lm"
    # litetune passes no decoding parameters to the runtime, so `decode` is a
    # declaration here and not an instruction. Stated as a value rather than
    # left to a default, because the asymmetry is what `verify` reports as a
    # limitation -- and it is litetune's gap, not the toolchain's:
    # `create_session` and `create_conversation` both take a `sampler_config`
    # and the driver script passes `None`, so the engine uses its own.
    decode_enforced = False
    # What the driver script saw of the process after building its engine; see
    # `device_report` in `RUNTIME_ENGINE_SOURCE`. Cleared at the start of every
    # run, before any return, so one run's reading is never reported as the
    # next one's.
    device_report: dict[str, Any] | None = field(default=None, init=False)
    # Text, not structured calls; a call is whatever `parse_call` makes of it.
    scores_structurally = False

    @property
    def prompt_mode(self) -> PromptMode:
        return self.declared_prompt_mode or UNDECLARED_PROMPT_MODE

    @property
    def uses_template(self) -> bool:
        return self.prompt_mode is PromptMode.RUNTIME_RENDERED

    @property
    def model_ref(self) -> str:
        return str(self.model)

    @property
    def backend_observed(self) -> bool:
        return gpu_observed(self.backend_flag, self.device_report)

    @property
    def runner_call(self) -> str:
        """Which of litert-lm's two entry points this mode uses.

        The distinction `--no-template` used to carry on a command line, said
        as the call it selects. `create_session(apply_prompt_template=False)`
        bypasses the chat template, the `<|turn>model` anchor, tool handling
        and channel extraction; it belongs on a prompt that already carries
        its own control tokens and nowhere else, which is why it is tied to
        the mode rather than chosen.
        """
        if self.uses_template:
            return "Engine.create_conversation()"
        return "Engine.create_session(apply_prompt_template=False)"

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "litert-lm",
            "backend": self.backend_flag,
            BACKEND_OBSERVED: self.backend_observed,
            # Stated rather than inherited from the bundle; see
            # `runtime_engine_spec`. `None` on the CPU backend, where nothing
            # is passed.
            "activation_data_type": runtime_engine_spec(self.backend_flag)["activation_data_type"],
            # What the script saw of its own process after building the engine,
            # verbatim, so a reader can tell "looked and found no GPU client"
            # from "did not look". `None` before a run and after one that wrote
            # nothing.
            "device_report": self.device_report,
            # Read off that report: the kernel was asked and showed no GPU work
            # from this process although the GPU was asked for, and what the
            # bundle itself declares for GPU activations.
            "gpu_unused": gpu_unused(self.backend_flag, self.device_report),
            "bundle_activation": bundle_activation_of(self.device_report),
            "bundle_activation_error": bundle_activation_error_of(self.device_report),
            # The flag as passed, not a device torch chose: see
            # `HuggingFaceBackend.describe`, where the same key carries the
            # other vocabulary.
            "backend_vocabulary": "litert-lm Python API Backend",
            "requirements": list(self.env.requirements),
            "system_requirements": list(self.env.system_requirements),
            # What the script calls, in place of the command line this
            # backend used to build. A manifest that recorded `argv` for a run
            # that never spawned one would be describing a transport by name
            # only.
            "transport": "litert_lm python api, one process per split",
            "runner_call": self.runner_call,
            "prompt_mode": self.prompt_mode.value,
            "prompt_mode_declared": self.declared_prompt_mode is not None,
            "template_flag": None if self.uses_template else "apply_prompt_template=False",
            # Nothing here is passed to the runtime, so `decode` is the *declared*
            # configuration: greedy, to the runtime's own token limit. It is
            # recorded because comparability depends on it. There is no way to
            # pass a deviation today -- the driver script builds no sampler and
            # `create_*(sampler_config=None)` takes the engine's own -- which is
            # what `decode_enforced = False` says, and why `verify` reports the
            # asymmetry as a limitation rather than hiding it.
            "decode_declared": self.decode.as_dict(),
            "decode_passed_to_runtime": self.decode_enforced,
        }

    def generate(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> list[Generation]:
        # First, before any return: this run's reading or none.
        self.device_report = None
        blocked = self._ensure_env(events)
        if blocked is not None:
            return [Generation(i, p, harness_error=blocked) for i, p in enumerate(prompts)]
        if not prompts:
            return []
        if self.backend_flag == "gpu" and len(prompts) > 1:
            silent = self._gpu_preflight(prompts[0], events)
            if silent is not None:
                return [Generation(i, p, harness_error=silent) for i, p in enumerate(prompts)]

        with tempfile.TemporaryDirectory(prefix="litetune-litertlm-") as tmp:
            work = Path(tmp)
            script = work / "generate.py"
            script.write_text(_LITERTLM_GENERATE_SCRIPT, encoding="utf-8")
            results = work / "generations.jsonl"
            report = work / "device.json"
            spec = work / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "model": str(self.model),
                        "prompts": list(prompts),
                        "runtime_rendered": self.uses_template,
                        **runtime_engine_spec(self.backend_flag),
                        "out": str(results),
                        "report": str(report),
                    }
                ),
                encoding="utf-8",
            )
            if events:
                events.note(
                    f"{self.name}: generating {len(prompts)} completions on the "
                    f"{self.backend_flag} backend",
                    backend=self.name,
                    total=len(prompts),
                )
            try:
                # `timeout_s` still means what it always did -- the budget one
                # prompt gets -- so the whole split gets the sum of them, and a
                # run that used to be allowed 300 s per prompt is allowed no
                # less now. What is gone is the *granularity*: one prompt that
                # hangs takes the budget the rest would have had, where before
                # it was cut at 300 s and the run carried on. That is a real
                # loss and not a theoretical one: MEASUREMENTS.md records a
                # 600-row verify where one prompt exceeded the 300 s limit --
                # though `verify` refused that run rather than scoring the 599
                # that finished, so the granularity bought nothing there.
                proc = self.env.run(
                    ["python", str(script), str(spec)],
                    timeout=self.timeout_s * len(prompts),
                )
            except subprocess.TimeoutExpired as expired:
                budget = self.timeout_s * len(prompts)
                logger.warning("litert-lm generation timed out after %ss", budget)
                # Read what the script had already flushed, rather than
                # discarding it. The docstring promises a run killed at prompt
                # 400 keeps the first 399, and the first version of this made
                # that true of a kill and false of a timeout -- it returned
                # before reading the file, and the temp directory took the rows
                # with it. Every row here is a complete generation: the script
                # flushes after each one, and a half-written last line fails to
                # parse and is dropped by the reader.
                texts, faults = read_jsonl_results(results)
                self.device_report = _read_device_report(report)
                # `_run_guarded` kills the group and drains it precisely to put
                # the child's last words on this exception; the first version
                # of this discarded them, leaving the budget as the whole
                # diagnostic.
                killed = expired.stderr or ""
                if isinstance(killed, bytes):
                    killed = killed.decode("utf-8", "surrogateescape")
                killed = killed[-2000:]
                reason = f"no result after {budget}s on the {self.backend_flag} backend (timeout)"
                return salvage_after_kill(prompts, texts, faults, killed, reason)
            except OSError as exc:
                logger.exception("could not start the litert-lm generation script")
                reason = f"{type(exc).__name__}: {exc}"
                return [Generation(i, p, harness_error=reason) for i, p in enumerate(prompts)]

            texts, faults = read_jsonl_results(results)
            self.device_report = _read_device_report(report)
        return assemble_generations(prompts, texts, proc, faults)

    def _gpu_preflight(self, prompt: str, events: EventStream | None) -> str | None:
        """One prompt on the GPU backend, with one prompt's budget, before the split.

        An engine built for a GPU that is not usable raises nothing and answers
        nothing: measured on Linux with litert-lm 0.16.1, created without
        error, then no token for a single prompt in fifty-five minutes. The
        split's budget is `timeout_s` per prompt, so a 600-row run would wait
        fifty hours to report what one prompt can show in `timeout_s`. Only
        the silence is judged here -- anything else the run itself reports,
        with the split's own evidence. Costs one extra engine load.
        """
        if events:
            events.note(
                f"{self.name}: one prompt on the gpu backend first, within {self.timeout_s}s",
                backend=self.name,
            )
        with tempfile.TemporaryDirectory(prefix="litetune-litertlm-preflight-") as tmp:
            work = Path(tmp)
            script = work / "generate.py"
            script.write_text(_LITERTLM_GENERATE_SCRIPT, encoding="utf-8")
            spec = work / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "model": str(self.model),
                        "prompts": [prompt],
                        "runtime_rendered": self.uses_template,
                        **runtime_engine_spec(self.backend_flag),
                        "out": str(work / "generations.jsonl"),
                    }
                ),
                encoding="utf-8",
            )
            try:
                self.env.run(["python", str(script), str(spec)], timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                return (
                    f"the gpu backend gave no answer to one prompt in {self.timeout_s}s, so "
                    "the split was not started. On a host without a usable GPU litert-lm has "
                    "been measured to build a GPU engine without error and then produce nothing"
                )
            except OSError:
                return None
        return None

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


# Runs inside envs.RUNTIME, the only environment with litert-lm in it.
#
# One process for a whole split, where this module used to start one per
# prompt: six hundred process starts and six hundred bundle reloads for a
# 600-row verify. What that cost in wall time is not measured here; the class
# docstring carries the sizes that are.
#
# What it must not be is a different measurement. The CLI is a thin wrapper
# over the same three calls this makes -- `Engine`, then `create_session` or
# `create_conversation`, then the stream -- so this is that code reached
# directly rather than a second way of asking. Where the CLI's behaviour is
# not obvious the script copies it deliberately and says so.
# The backends a litert-lm candidate can be asked for, in the word both the
# CLI's `--backend` and `verify --backend` use.
CANDIDATE_BACKENDS = ("cpu", "gpu")


def runtime_engine_spec(backend: str) -> dict[str, Any]:
    """What a driver script needs to construct its engine, as spec keys.

    The activation type is stated on every GPU run rather than left to the
    bundle. The bundle's `prefer_activation_type` is only a default: the
    runtime's `--activation-data-type` overrides it (`litert_lm_cli/common.py:216`
    at v0.16.1, where the option is hidden and called experimental and "may
    not always work"; that it overrides in both directions is from
    LiteRT-LM#2992). A bundle that says nothing leaves the GPU text executor
    in F16 while the engine reports success: `<pad>` on 40 of 40 rows on an
    M4 Pro's Metal, and floods on 14 of 20 on a Galaxy S24. Passing
    `export.GPU_ACTIVATION` here keeps our runs out of that state whatever the
    bundle carries -- measured on Metal, where an unkeyed bundle answered as a
    keyed one did -- and `verify` records what the bundle alone declares,
    because an app that passes no override gets that instead.

    On the CPU backend nothing is passed: the option exists to force FP32 on a
    GPU, and the CPU path has been measured with nothing passed since the
    first number in MEASUREMENTS.md.
    """
    if backend not in CANDIDATE_BACKENDS:
        raise ValueError(
            f"unknown litert-lm backend {backend!r}: expected one of {CANDIDATE_BACKENDS}"
        )
    return {
        "backend": backend,
        "activation_data_type": GPU_ACTIVATION if backend == "gpu" else None,
    }


def _read_device_report(path: Path) -> dict[str, Any] | None:
    """The device report a driver script wrote, or `None` if it wrote none.

    Missing is ordinary -- a script that died before building its engine, or a
    run that never started -- and it means "not established", which is what
    `gpu_observed` makes of `None`. Unreadable is logged, because it is not.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("device report %s is unreadable: %s", path.name, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _gpu_work(report: Any) -> tuple[bool | None, Any]:
    """What a run report says about GPU work: (did it happen, the reading).

    True when the kernel was asked after generating and one of the process's
    GPU clients showed more GPU time than it had once the engine was built --
    the runtime opened the GPU *and* worked there. False when the kernel was
    asked at both readings and showed no client for this pid either time, or
    the same clients with time that did not grow: measured on the host where
    this was built, a GPU engine always owns one from construction and its
    time grows while it decodes, so those readings say the GPU was not used.
    None when nothing was established either way -- the kernel was not asked,
    the run died before the second reading, a client seen at the engine was
    gone by the end, or the report is missing.
    """
    if not isinstance(report, dict):
        return None, None
    end = report.get("after_generation")
    start = report.get("at_engine")
    if not isinstance(end, dict) or end.get("looked") is not True:
        return None, None
    if isinstance(end.get("gpu_client"), str):
        return _clients_worked(start, end), end
    if (
        isinstance(start, dict)
        and start.get("looked") is True
        and not isinstance(start.get("gpu_client"), str)
    ):
        return False, end
    return None, end


def _clients_worked(start: Any, end: dict) -> bool | None:
    """Whether GPU time grew, compared client by client across the readings.

    A client's total is compared only with that same client's: one that did
    work and closed would otherwise leave a sum that did not grow, and a
    replacement would stand in for it. The key is the registry id `ioreg`
    prints (`IORegistryEntryGetRegistryEntryID`, IOKitTools ioreg.c), which
    xnu hands out from a counter it only increments (IORegistryEntry.cpp), so
    an id is not reused before a reboot. A client absent from the engine
    reading was opened after it, so all its time came later and counts from
    zero. Without a successful engine reading, or with a client whose time
    was not read there, growth is not established. No growth is a "no" only
    when every client seen at the engine is still there and every time on
    both sides was read; otherwise it is None.
    """
    after = end.get("gpu_clients")
    if not isinstance(start, dict) or start.get("looked") is not True:
        return None
    before = start.get("gpu_clients")
    if not isinstance(after, dict) or not isinstance(before, dict):
        return None
    unread = False
    for key, time in after.items():
        base = before.get(key, 0)
        if not isinstance(base, int):
            unread = True
            continue
        if isinstance(time, int) and time > base:
            return True
    if unread or not set(before) <= set(after):
        return None
    if all(isinstance(t, int) for t in after.values()):
        return False
    return None


def gpu_observed(backend: str, report: Any) -> bool:
    """Whether a run asked for the GPU and the kernel shows it worked there.

    The only way a litert-lm backend here comes to say it observed its device.
    A CPU run is never "observed" this way -- no GPU work does not prove the
    CPU served the run -- and `backend_established` does not need it to be: a
    CPU request is stated on its own.
    """
    return backend == "gpu" and _gpu_work(report)[0] is True


def gpu_unused(backend: str, report: Any) -> bool:
    """Whether a run asked for the GPU and the kernel shows it was not used."""
    return backend == "gpu" and _gpu_work(report)[0] is False


def bundle_activation_of(report: Any) -> str | None:
    """The bundle's own activation key as the driver script read it, or None."""
    if isinstance(report, dict) and isinstance(report.get("bundle_activation"), str):
        return report["bundle_activation"]
    return None


def bundle_activation_error_of(report: Any) -> str | None:
    """Why the driver script could not read the bundle's key, if it could not.

    Kept apart from `bundle_activation_of`, whose None means both "the bundle
    declares none" and "nobody could tell": the first is a finding about the
    bundle, the second is not.
    """
    if isinstance(report, dict) and isinstance(report.get("bundle_activation_error"), str):
        return report["bundle_activation_error"]
    return None


# Pasted into each driver script that constructs a litert-lm engine -- the
# text path's below and the tool path's in `toolpath.py` -- so the two cannot
# disagree about how a backend is built or whether the activation type is
# passed. Scripts are written to a file and run in `envs.RUNTIME`; they cannot
# import from this package, so the one source is text.
RUNTIME_ENGINE_SOURCE = r'''

def backend_for(name):
    """The engine's backend, from the same word the CLI's --backend takes.

    litert-lm is imported inside the functions that need it rather than at the
    top, so the parent can exec a script and test what it composes without
    the runtime installed.
    """
    from litert_lm.interfaces import CPU, GPU

    if name == "gpu":
        return GPU()
    if name == "cpu":
        return CPU()
    raise SystemExit("unsupported backend %r: this script knows cpu and gpu" % (name,))


def engine_kwargs(spec):
    """Everything passed to `litert_lm.Engine` beside the model and cache.

    `ActivationDataType.from_str("fp32")` is `FLOAT32`, whose value is 0, so
    nothing here may test the converted value for truth. `from_str` returns
    None for a name it does not know, and an engine given None builds as if
    nothing were asked -- which on a GPU is the F16 default this key exists to
    prevent -- so an unknown name ends the run instead.
    """
    kwargs = {"backend": backend_for(spec.get("backend", "cpu"))}
    requested = spec.get("activation_data_type")
    if requested is not None:
        from litert_lm import ActivationDataType

        kind = ActivationDataType.from_str(requested)
        if kind is None:
            raise SystemExit("unknown activation data type %r" % (requested,))
        kwargs["activation_data_type"] = kind
    return kwargs


def device_report():
    """What the kernel says this process has done on the GPU, as of now.

    The engine cannot say where it ran: it echoes the backend it was given.
    The kernel can. A process that opens the GPU gets a user client in the
    IORegistry (`IOGPUDeviceUserClient` and its subclasses), which records
    the pid that created it and, under `AppUsage`, an `accumulatedGPUTime`
    for that client. Measured on an M4 Pro with litert-lm 0.16.1: a GPU
    engine owns such a client from the moment it is built and its GPU time
    grew by about four thousand times across one generated reply; a CPU
    engine owned none. The pid is the evidence, not a name -- every other
    process's clients are in the same listing and are ignored.

    Taken twice by each driver script -- once the engine exists, and again
    after generating -- because a client proves the runtime opened the GPU,
    and only the growth proves work was done there.

    A diagnostic, so it must never end a run: any failure is recorded in
    `error` and the report says it did not look. Other platforms are not
    looked at, because which signal marks litert-lm's GPU path there has not
    been observed.
    """
    import os
    import sys

    report = {
        "platform": sys.platform,
        "looked": False,
        "gpu_client": None,
        "gpu_time": None,
        "gpu_clients": None,
    }
    if sys.platform != "darwin":
        return report
    try:
        import subprocess

        done = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "IOGPUDeviceUserClient", "-l"],
            capture_output=True,
            timeout=30,
        )
        if done.returncode != 0:
            report["error"] = "ioreg exited %d: %s" % (
                done.returncode,
                done.stderr.decode("utf-8", "replace").strip()[-200:],
            )
            return report
        listing = done.stdout.decode("utf-8", "replace")
        if '"IOUserClientCreator"' not in listing:
            # Not "no client of ours": no GPU client of anyone's -- ioreg lists
            # them under the driver's subclass, AGXDeviceUserClient on an M4
            # Pro -- so there was nothing to find this process among.
            report["error"] = "ioreg listed no GPU client of any process"
            return report
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must not end a run
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        return report
    report["looked"] = True
    client, clients = gpu_client_of(listing, os.getpid())
    times = [t for t in clients.values() if t is not None]
    report["gpu_client"] = client
    report["gpu_time"] = sum(times) if times else None
    report["gpu_clients"] = clients
    return report


def gpu_client_of(listing, pid):
    """The GPU clients `pid` created: the first one's name, and each one's time.

    Separate from `device_report` so the rule can be tested without a GPU:
    only a client whose creator is this pid counts. Each is keyed by its
    registry id, with the sum of `accumulatedGPUTime` across its `AppUsage`
    entries, or None when it has none -- per client, because a process may
    hold more than one and the two readings are compared client by client.
    `ioreg -l` writes one object per `+-o` line, with its properties below
    it until the next one.
    """
    import re

    marker = '"IOUserClientCreator" = "pid %d,' % pid
    client, clients = None, {}
    for n, block in enumerate(listing.split("+-o ")):
        if marker not in block:
            continue
        if client is None:
            line = next(l for l in block.splitlines() if marker in l)
            client = line.split('" = ', 1)[1].strip().strip('"')
        found = re.search(r"\bid (0x[0-9a-fA-F]+)", block.splitlines()[0])
        times = [int(t) for t in re.findall(r'"accumulatedGPUTime"=(\d+)', block)]
        clients[found.group(1) if found else "#%d" % n] = sum(times) if times else None
    return client, clients


def bundle_activation(path):
    """The bundle's own `prefer_activation_type`, read from its header, or None.

    What an application gets on the GPU when it loads this bundle without an
    override. `verify` always passes fp32 itself, so its GPU number describes
    the bundle *plus* that override; this is what lets the manifest say so
    when the bundle alone would not get it. Read through
    `litert_lm_builder`, which `litert-lm` depends on, from the header only --
    no section is unpacked. None when the key is absent. Raises -- recorded
    by the caller as unreadable -- when the builder cannot be imported, the
    header cannot be read, a value is not a string, or the values disagree --
    across prefill-decode sections or within one: no single entry is the
    bundle's answer then.
    """
    import io

    from litert_lm_builder import litertlm_header_schema_py_generated as schema
    from litert_lm_builder import litertlm_peek as peek

    metadata = peek.read_litertlm_header(str(path), io.StringIO())
    sections = metadata.SectionMetadata()
    declared = []
    for i in range(sections.ObjectsLength() if sections else 0):
        section = sections.Objects(i)
        if peek.get_model_type(section) != "tf_lite_prefill_decode":
            continue
        found = []
        for j in range(section.ItemsLength()):
            item = section.Items(j)
            key = item.Key().decode("utf-8") if item is not None and item.Key() else None
            if key != "prefer_activation_type":
                continue
            if item.ValueType() != schema.VData.StringValue:
                raise ValueError("prefer_activation_type is not a string")
            value = schema.StringValue()
            value.Init(item.Value().Bytes, item.Value().Pos)
            raw = value.Value()
            found.append(raw.decode("utf-8") if raw else None)
        # The builder lets `additional_metadata` repeat the key in one section.
        declared.extend(found or [None])
    if len(set(declared)) > 1:
        raise ValueError("prefill-decode sections declare %s" % sorted(map(str, declared)))
    return declared[0] if declared else None


def run_report(spec, at_engine, after_generation=None):
    """The report a driver script writes about its run, beside its answers.

    `bundle_activation` is read here, in the script, because `litert_lm_builder`
    is installed where the script runs and not where litetune runs.
    """
    report = {"at_engine": at_engine, "after_generation": after_generation}
    try:
        report["bundle_activation"] = bundle_activation(spec["model"])
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must not end a run
        report["bundle_activation"] = None
        report["bundle_activation_error"] = "%s: %s" % (type(exc).__name__, exc)
    return report
'''


_LITERTLM_GENERATE_SCRIPT = (
    r'''
"""Generation for one split through litert-lm's Python API.

Writes JSONL, one line per prompt, flushed as it goes -- a run killed at
prompt 400 keeps the first 399. A line is either {"index": int, "text": str}
or, where the runtime refused that prompt, {"index": int, "error": str}: the
reader treats the second as a fault against that prompt rather than as a
missing row, so the split carries on and the refusal is not scored as an
answer.
"""
import json
import sys
from pathlib import Path
'''
    + RUNTIME_ENGINE_SOURCE
    + r'''

def text_from_session(session, prompt):
    """The pre-rendered path, which `--no-template` selects in the CLI.

    `create_session(apply_prompt_template=False)` is what that flag chooses,
    and prefill-then-decode is what the CLI does with it
    (`litert_lm_cli/commands/run.py::_execute_raw_prompt`).
    """
    session.run_prefill([prompt])
    parts = []
    for chunk in session.run_decode_async():
        if chunk.texts:
            parts.append(chunk.texts[0])
    return "".join(parts)


def text_from_conversation(conversation, prompt):
    """The runtime-rendered path, composed exactly the way the CLI composes stdout.

    Channel content reaches the score. `litert-lm run` prints it between
    `[name] ` and ` [/name]` ahead of the answer, the old stdout scraper kept
    both markers, and `metrics.REASONING_BLOCKS` is a fixed two-entry tuple
    holding `[thought]`/`[/thought]` and `<think>`/`</think>` -- so a thought
    channel comes off before scoring and any other channel name does not.
    A composition that opens a channel and never closes it does not merely
    look different: `_split_reasoning` cuts at
    the *last closing* marker, so with none the reasoning stays in the answer,
    the row scores against thought-plus-answer, and nothing fails. Where it
    lands afterwards depends on the shape: `_split_reasoning` calls it
    `unclosed` only when the text *starts* with an opening marker, so a stream
    that emitted an answer before opening the channel is counted as carrying no
    reasoning at all.

    So this mirrors `litert_lm_cli/commands/run.py` as a state machine rather
    than approximating it. Read at the v0.16.1 tag, the version `envs.RUNTIME`
    pins: `close_channel` (run.py:50-53) writes `" [/name]"` and a newline,
    and run.py:108-125 calls it before every text item, on a switch between
    channels, and at the end of the stream. The one branch not mirrored is the
    bare `click.echo()` the CLI emits instead when no channel was open at the
    end, which is a trailing newline the scorer strips.
    """
    parts = []
    active = [None]

    def close():
        if active[0] is not None:
            parts.append(" [/%s]\n" % active[0])
            active[0] = None

    for chunk in conversation.send_message_async(prompt):
        for item in chunk.get("content", []) or []:
            if item.get("type") == "text":
                close()
                parts.append(item.get("text", ""))
        for name, content in (chunk.get("channels", {}) or {}).items():
            if active[0] != name:
                close()
                parts.append("[%s] " % name)
                active[0] = name
            parts.append(content)
    close()
    return "".join(parts)


def main(spec_path):
    import litert_lm

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    runtime_rendered = bool(spec["runtime_rendered"])
    engine_args = engine_kwargs(spec)
    with Path(spec["out"]).open("w", encoding="utf-8") as out:
        # `cache_dir=""` is what the CLI passes on its default `--cache`:
        # `cache_dir_value_from_cache_mode` (common.py) maps both `None` and
        # "disk" to the empty string, and `Engine.__init__` reaches
        # `litert_lm_engine_settings_set_cache_dir` only when `cache_dir is not
        # None` (engine.py:149-151 at v0.16.1). Leaving it out would not be the
        # CLI's behaviour but a fourth state beside disk, memory and none.
        # Which of the three that fourth state resolves to is decided in the
        # C++ the binding calls, not in anything readable from here.
        with litert_lm.Engine(spec["model"], cache_dir="", **engine_args) as engine:
            # Before the first prompt, so a run killed part-way still says
            # where it was running -- the same reason the transformers script
            # writes its report before generating.
            at_engine = device_report()
            if spec.get("report"):
                Path(spec["report"]).write_text(
                    json.dumps(run_report(spec, at_engine)), encoding="utf-8"
                )
            for index, prompt in enumerate(spec["prompts"]):
                # A runner per prompt, not one per split. A conversation keeps
                # its history, so the second prompt would be answered with the
                # first still in context -- the CLI started a process per
                # prompt and could not get this wrong; this script can, and a
                # split scored with drifting context looks like nothing.
                try:
                    if runtime_rendered:
                        with engine.create_conversation(sampler_config=None) as runner:
                            text = text_from_conversation(runner, prompt)
                    else:
                        with engine.create_session(
                            apply_prompt_template=False, sampler_config=None
                        ) as runner:
                            text = text_from_session(runner, prompt)
                except MemoryError:
                    # Not one prompt's problem. The engine that could not
                    # allocate for this prompt will not allocate for the next
                    # one either, so containing it here turns one fatal
                    # condition into `len(prompts)` attempts at it, each slower
                    # than the last on a box that is already thrashing, under a
                    # parent budget of `timeout_s * len(prompts)`. Let it end
                    # the process: the rows already flushed are salvaged by the
                    # non-zero-exit path, which is what that path is for.
                    raise
                except Exception as exc:  # noqa: BLE001
                    # One prompt, not the rest of the split. The runtime
                    # raises `RuntimeError` when a prefill or a decode call
                    # fails (`litert_lm/session.py:70` and `:99` at v0.16.1)
                    # and when a send fails (`conversation.py:326`), and
                    # without this the first of those ends the process: every
                    # later prompt comes back as "the script exited 1".
                    #
                    # What this does not buy is the measurement. Any generation
                    # carrying a `harness_error` makes `liveness.exit_status_check`
                    # return UNCHECKED, and `verify` turns that into
                    # `failed_harness` -- so the run is still refused, exactly
                    # as MEASUREMENTS.md records for the split that generated
                    # 599 of 600. What it buys is that the manifest names the
                    # prompt and the reason, and carries the 599 rows, instead
                    # of saying the script exited 1 about all six hundred.
                    #
                    # The CLI contained it differently and worse -- it caught
                    # everything at the top of the command, printed "An error
                    # occurred" to stdout and exited 0, so the old transport
                    # scored that sentence as the model's answer. A row that
                    # says it failed is neither that nor a lost split.
                    row = {"index": index, "error": "%s: %s" % (type(exc).__name__, exc)}
                else:
                    row = {"index": index, "text": text}
                out.write(json.dumps(row) + "\n")
                out.flush()
            # Again after generating, with the engine still alive: a client
            # shows the runtime opened the GPU, and only its GPU time growing
            # across the split shows work was done there.
            if spec.get("report"):
                Path(spec["report"]).write_text(
                    json.dumps(run_report(spec, at_engine, device_report())), encoding="utf-8"
                )

if __name__ == "__main__":
    main(sys.argv[1])
'''
)


@functools.cache
def _litertlm_script() -> dict[str, Any]:
    """The driver script's own definitions, so the parent can test them.

    Read from the one source rather than copied. The composition in
    `text_from_conversation` is the only part of that script not a direct call
    into litert-lm, so it is the only part that can be wrong on its own -- and
    a copy here would be a second thing to keep right. Same device as
    `toolpath._script`.
    """
    namespace: dict[str, Any] = {"__name__": "litetune_litertlm_script"}
    exec(compile(_LITERTLM_GENERATE_SCRIPT, "litetune litertlm script", "exec"), namespace)  # noqa: S102
    return namespace


# Runs inside envs.TRAIN, which is the only environment with torch and
# transformers in it. Written to a temp file rather than passed with `-c` so
# that a traceback carries usable line numbers.
_HF_GENERATE_SCRIPT = (
    r'''
"""Greedy generation for one split. Writes JSONL: {"index": int, "text": str}."""
import json
import sys
from pathlib import Path
'''
    + RENDERING_SOURCE
    + r'''


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
            text, add_special = render_prompt(
                tok, prompt, spec["runtime_rendered"], spec.get("tools")
            )
            enc = tok(text, return_tensors="pt", add_special_tokens=add_special).to(device)
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
)


@dataclass
class HuggingFaceBackend:
    """Float reference generation through `transformers`, inside `envs.TRAIN`.

    The whole split runs in one process, so a checkpoint is loaded once rather
    than once per prompt. What that saves is not measured anywhere here, and
    the litert-lm backend no longer differs on it -- it stopped starting a
    process per prompt when the transport moved to the Python API.
    """

    model: str
    decode: DecodeConfig = GREEDY
    timeout_s: int = 3600
    env: envs.StageEnv = envs.TRAIN
    auto_provision: bool = True
    runtime_rendered: bool = False
    # The tool declarations the chat template renders into a developer turn, as
    # parsed JSON rather than a path: this backend's script runs in another
    # environment, which cannot read the caller's file. `None` renders the bare
    # user turn this rendered before declarations were an input.
    declarations: list | None = None
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
    # spec. `None` before `_ensure_env` has run, and after a call in which
    # provisioning itself raised -- that path returns its own blocking reason
    # and never gets as far as a probe. Where the environment merely could not
    # answer, or where there was nothing to ask, a `DeviceProbe` is left
    # carrying which of the two it was -- see `verify.py`, which also reads
    # `detail` and `cuda_build_without_a_device` off it to record the same
    # limitation `tune.py` records for its own probe.
    last_probe: envs.DeviceProbe | None = None

    name = "transformers"
    # `generate()` receives max_new_tokens and the stop condition, so here the
    # declared configuration is the applied one.
    decode_enforced = True
    # Set only where the run report answers, and cleared wherever `device` is.
    # `device is not None` is the wrong question: the probe writes a prediction
    # into it before the run, and `_read_run_report` deliberately leaves that
    # prediction standing when no report arrives -- so a script that died at
    # `model.to(device)` would otherwise have its probe reported as a reading.
    device_observed: bool = False
    # Text, not structured calls; a call is whatever `parse_call` makes of it.
    scores_structurally = False

    @property
    def backend_observed(self) -> bool:
        return self.device_observed

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
            BACKEND_OBSERVED: self.backend_observed,
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
            "decode_passed_to_runtime": self.decode_enforced,
            "prompt_mode": self.prompt_mode.value,
            "prompt_mode_declared": self.declared_prompt_mode is not None,
            "applies_chat_template": self.uses_template,
        }

    def generate(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> list[Generation]:
        # A run reports what it read back, not what an earlier one did:
        # `device` survives a run that established nothing, by the do-not-erase
        # rule below, and that survival must not be reported as this run's
        # reading. First, before any return: a blocked environment clears
        # `device` in `_ensure_env`, and a `True` left from the previous run
        # would then sit beside `UNKNOWN_BACKEND` claiming it was read back.
        self.device_observed = False
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
                        "tools": self.declarations,
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
            except subprocess.TimeoutExpired as expired:
                logger.warning("transformers generation timed out after %ss", self.timeout_s)
                reason = f"no result after {self.timeout_s}s (timeout)"
                # The run report is written after the last prompt, so a killed
                # run has none and the device stays unconfirmed. A prediction
                # left standing here would be reported by `describe()` as the
                # backend a run used when nothing confirmed it.
                self.device = None
                # This script flushes after every row too, and until now every
                # one of them was thrown away here: a reference run killed at
                # prompt 600 of 640 lost all 599 it had already written, which
                # is the same erasure the litert-lm path was fixed for.
                killed = expired.stderr or ""
                if isinstance(killed, bytes):
                    killed = killed.decode("utf-8", "surrogateescape")
                texts, faults = read_jsonl_results(results)
                return salvage_after_kill(prompts, texts, faults, killed[-2000:], reason)
            except OSError as exc:
                logger.exception("could not start the generation script")
                reason = f"{type(exc).__name__}: {exc}"
                self.device = None
                return [Generation(i, p, harness_error=reason) for i, p in enumerate(prompts)]

            texts, faults = read_jsonl_results(results)
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
            self.device_observed = True
        return assemble_generations(prompts, texts, proc, faults)

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
            detail = (
                f"no device probe was attempted: environment {self.env.name!r} is not "
                f"provisioned at {self.env.path}"
            )
            # Its own sentence, not `envs._unanswered`'s. That one says a probe
            # "could not answer", which is the second of the three states the
            # comment above names, and this is the third. It is also the only
            # cross-module reach into an `envs` private in the package, so the
            # moment a probe-specific side effect is added there this call site
            # would start lying.
            logger.warning("%s", detail)
            if events is not None:
                events.note(detail, environment=self.env.name)
            self.last_probe = envs.DeviceProbe(device=None, detail=detail, attempted=False)
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


def backend_established(engine: Mapping[str, Any]) -> bool:
    """Whether `backend` may be stated as where the work happened.

    The litert-lm backends fill it from the flag they passed. On a GPU run on
    macOS the kernel is asked as well (`BACKEND_OBSERVED`); otherwise the
    sentence is only as good as the ask. It is good enough for `cpu`: there
    is nothing below it to fall back to, and every number in MEASUREMENTS.md
    rests on asking litert-lm for its CPU backend and getting it. It is not
    good enough for an accelerator, which is the case measured to fail
    quietly -- an engine built for a GPU that was not usable was created
    without error and then produced no token for one prompt in fifty-five
    minutes (Linux, litert-lm 0.16.1).

    `UNKNOWN_BACKEND` passes because it claims nothing: "measured on the
    unknown backend" is already a statement of ignorance, and rewording it
    would put a second name on the state that constant exists to name.
    """
    if engine.get(BACKEND_OBSERVED) is True:
        return True
    return str(engine.get("backend") or UNKNOWN_BACKEND).lower() in (CPU_BACKEND, UNKNOWN_BACKEND)


def device_mismatch(a: MeasurementPoint, b: MeasurementPoint) -> str | None:
    """The two points ran on different hardware, said in full, or None.

    A limitation, not a refusal, and the difference from `harness_mismatch` is
    the decision rather than an oversight. The candidate runs on the litert-lm
    backend `verify --backend` names, `cpu` by default, and the reference
    resolves its own device, so on a machine where the reference resolves to
    cuda the two sides differ
    in hardware as well as in conversion, and the "cost of conversion" carries
    both. Refusing that comparison would leave a GPU box unable to verify at
    all, which is worse than a number that says what else is in it.

    Read from `engine["backend"]`, which each backend fills with the device it
    read back or the flag it passed -- `BACKEND_OBSERVED` says which, and
    `backend_established` turns that into the verb this sentence uses.

    The two vocabularies do not overlap except at
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

    # Each side gets the verb its evidence supports. Saying both "was measured
    # on" would attribute part of a score gap to hardware that, on a side
    # reporting a flag nobody read back, was never established to have served
    # the run.
    def ran(point: MeasurementPoint, value: str) -> str:
        verb = "was measured on" if backend_established(point.engine) else "asked for"
        return f"{point.label} {verb} {value} ({point.engine.get('engine') or UNKNOWN_BACKEND})"

    established = backend_established(a.engine) and backend_established(b.engine)
    carries = (
        "so the difference between them carries a hardware difference as well as a conversion one"
        if established
        else "so the difference between them may carry a hardware difference as well as a "
        "conversion one -- may, because a backend above that was asked for rather than read back"
    )
    return (
        f"{ran(a, left)} and {ran(b, right)}, {carries}. The comparison is reported rather than "
        "refused: pinning both sides to one device is not something litetune can do for the "
        "runtime side, and a refusal would leave such a machine unable to verify at all"
    )


def harness_mismatch(a: MeasurementPoint, b: MeasurementPoint) -> str | None:
    """Why `a` and `b` cannot be compared, or None if they can.

    Prompt mode is checked first and is the reason this function exists. On the
    pre-rendered path the runtime's tool list is null, so a runtime-rendered
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
