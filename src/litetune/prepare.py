"""`prepare`: a raw dataset into a split, plus the reasons it may not be measurable.

This stage exists because the two most expensive mistakes in the work behind
litetune were both made *before* a GPU was ever booked, and both were visible in
the data.

**The split is keyed by content, not by location.** Which rows land in the
held-out set is derived from the sha256 of the file's bytes and the seed,
nothing else. Two copies of the same bytes at different paths produce the same
split; one changed byte reshuffles everything, which is what makes a stale
cached checkpoint impossible to reuse under a green manifest. See
`storage.py` for the same rule one layer down.

**A small held-out set is a warning that travels with the result.** At n=64 a
recipe comparison read as 0.172; at n=640 the same comparison was 0.024, and
three separate conclusions drawn at the smaller size were overturned. So a split
below `MIN_HELDOUT_EXAMPLES` is recorded as a limitation on everything measured
from it rather than silently accepted.

**A row that does not fit is named, never dropped.** Truncating an over-length
example removes the supervised span -- the answer -- and trains the model on a
prompt with no target while the loss curve keeps going down. So the length check
fails and lists the offending rows by line number, and no split is written.

**An argument nobody could score is reported before the GPU is paid for.** In
one dataset a tool had 92 unique `message` values across 95 examples, invented
rather than quoted from the prompt, and exact match on it scored 0.00 for every
model tested including the untuned base -- there was no gradient of quality to
measure at all. Google's own reference dataset has 2,270 unique values in 2,276
examples for one field and scores fine, because the value is always a literal
span of the prompt. High cardinality alone is therefore not the signal:
`extractiveness` is, and the two are reported together per argument.

Nothing here prints. Everything is an event, a check or a limitation on the
result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from litetune import envs
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.liveness import SkippedCheck
from litetune.metrics import Proportion, ToolCall, Unavailable
from litetune.spec import DEFAULT_MIN_HELDOUT_EXAMPLES
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

PREPARE_SCHEMA = "litetune.prepare/1"

# Below this the Wald half-width swamps the effects this tool measures: at n=200
# it is ±0.0398 on a 0.91 proportion, against a measured recipe effect of 0.024.
# A warning, never a refusal -- a small split is still worth running, as long as
# nobody reads a difference off it. Shared with `spec.Gates.min_heldout_examples`
# so that the spec and this stage cannot drift apart.
MIN_HELDOUT_EXAMPLES = DEFAULT_MIN_HELDOUT_EXAMPLES

DEFAULT_HELDOUT_FRACTION = 0.2

# An argument whose values are this close to all-distinct carries no repeated
# label for a model to learn. Anchored on the two measured datasets: 92/95 =
# 0.968 (unscoreable) and 2270/2276 = 0.997 (fine, because extractive).
HIGH_CARDINALITY_SHARE = 0.9

# ... and this is what separates them. Below this share of values appearing
# verbatim in their own prompt, a high-cardinality argument is being *invented*
# rather than *quoted*, and exact match on it is unscoreable by construction.
MIN_EXTRACTIVE_SHARE = 0.5

# A base model already at or above this on a slice leaves no room to improve, so
# a fine-tune measured on it reports a difference of roughly zero whatever it
# did. Applied only when a headroom probe is supplied -- see `HeadroomProbe`.
NO_HEADROOM_ABOVE = 0.98

# Tokenizers add a BOS (and the training sequence an EOS) that a per-text count
# with `add_special_tokens=False` does not include. Counted here so that the
# length report is against the sequence that will actually be trained on rather
# than two token counts that happen to be adjacent.
SEQUENCE_OVERHEAD_TOKENS = 2

# Reading a tokenizer is metadata work, not model work; a tokenizer that has not
# answered in ten minutes is an environment problem the training run will
# surface again in a more expensive way.
TOKEN_COUNT_TIMEOUT_S = 600

LENGTH_CHECK = "every example fits the context window"
SPLIT_CHECK = "held-out split is usable"
SCOREABILITY_CHECK = "arguments are scoreable"

NO_HEADROOM_MEASUREMENT = (
    "no base-model measurement was supplied, so slices where the base model already scores at "
    "ceiling -- and where a fine-tune therefore cannot show a gain -- were not identified. Pass a "
    "HeadroomProbe built from a measured base-model run to fill this in."
)


class PrepareError(ValueError):
    """The dataset could not be read as a dataset. The message names the row."""


class TokenCountUnavailable(RuntimeError):
    """Token lengths could not be measured. Not a statement about the data."""


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def render_call(call: ToolCall) -> str:
    """Render a target back into FunctionGemma's wire format.

    The inverse of `metrics.parse_call`, and it has to stay one: the completion
    a model is trained to emit must be byte-identical to the string the scorer
    will parse, or training and measurement are working from different targets.
    """
    body = ",".join(f"{key}:<escape>{value}<escape>" for key, value in call.args.items())
    return f"call:{call.name}{{{body}}}"


@dataclass(frozen=True)
class Row:
    """One example, carrying the line it came from so it can be named later."""

    lineno: int
    prompt: str
    completion: str
    target: ToolCall | None

    @property
    def tool(self) -> str:
        return self.target.name if self.target is not None else "<unlabelled>"

    def as_record(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "target": self.target.as_dict() if self.target is not None else None,
            # Kept so that a row flagged downstream can be found in the file it
            # came from. A row identified only by its position in a shuffled
            # split is a row nobody can go and look at.
            "source_line": self.lineno,
        }


def read_rows(path: Path) -> list[Row]:
    """Read training JSONL. Raises `PrepareError` naming `path:lineno`.

    A row is `{"prompt": str}` plus either an explicit `"completion"` or a
    `"target"` this renders into one. A row with neither is an error rather than
    a skip: a training file that silently loses a tenth of its rows produces a
    model nobody can explain and a loss curve that looks normal.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("could not read dataset %s", path)
        raise

    rows: list[Row] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrepareError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
        if not isinstance(obj, dict) or "prompt" not in obj:
            raise PrepareError(f"{path}:{lineno}: expected an object with a 'prompt' field")
        try:
            target = ToolCall.from_target(obj.get("target"))
        except ValueError as exc:
            raise PrepareError(f"{path}:{lineno}: {exc}") from exc

        completion = obj.get("completion")
        if completion is None and target is not None:
            completion = render_call(target)
        if not isinstance(completion, str) or not completion.strip():
            raise PrepareError(
                f"{path}:{lineno}: no supervised span. A row needs a 'completion' string or a "
                "'target' to render one from; training on a row with no answer contributes "
                "gradient toward nothing while the loss curve keeps falling"
            )
        rows.append(
            Row(
                lineno=lineno,
                prompt=str(obj["prompt"]),
                completion=completion,
                target=target,
            )
        )
    if not rows:
        raise PrepareError(f"{path}: no examples")
    return rows


# ---------------------------------------------------------------------------
# The split, keyed by content
# ---------------------------------------------------------------------------


def _bare_digest(digest: str) -> str:
    """`sha256:<hex>` to `<hex>`, which is the form a spec field takes."""
    return digest.split(":", 1)[-1]


def split_seed(content_sha256: str, seed: int) -> int:
    """The permutation's seed: the data's content and the declared seed, nothing else.

    Derived explicitly rather than by handing a string to `random.seed`, so that
    the value in the report is reproducible from the two inputs by anyone
    holding this file, in any Python.
    """
    payload = f"{_bare_digest(content_sha256)}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _permutation(n: int, seed: int) -> list[int]:
    """A Fisher-Yates shuffle over an explicit LCG.

    `random.Random` would do, and its string seeding is stable; this is spelled
    out because the split assignment is part of the result and a reader has to
    be able to reproduce it without trusting the standard library's internals to
    hold still across versions.
    """
    # glibc's constants. Any full-period LCG works here; the requirement is that
    # the recipe is written down, not that it is a good generator.
    modulus = 2**48
    state = seed % modulus
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        state = (25214903917 * state + 11) % modulus
        j = (state >> 16) % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order


def heldout_size_for(n: int, fraction: float, requested: int | None = None) -> int:
    """How many rows are held out. At least one, and never all of them."""
    size = requested if requested is not None else round(n * fraction)
    return max(1, min(int(size), n - 1)) if n >= 2 else 0


def split_rows(
    rows: Sequence[Row], content_sha256: str, seed: int, heldout: int
) -> tuple[list[Row], list[Row]]:
    """Split deterministically. Same content and seed, same two sets, any machine."""
    order = _permutation(len(rows), split_seed(content_sha256, seed))
    held = {order[i] for i in range(heldout)}
    return (
        [row for i, row in enumerate(rows) if i not in held],
        [row for i, row in enumerate(rows) if i in held],
    )


# ---------------------------------------------------------------------------
# Token lengths
# ---------------------------------------------------------------------------


class TokenCounter(Protocol):
    """Counts tokens the way the training run will count them.

    A Protocol because the real one runs a tokenizer inside `envs.TRAIN` -- the
    parent process must not import transformers -- and the tests supply a plain
    object with the same shape instead.
    """

    @property
    def name(self) -> str:
        """What produced these counts. Travels with the report."""

    def describe(self) -> dict[str, Any]:
        """Tokenizer identity, so two length reports can be told apart."""

    def count(self, texts: Sequence[str]) -> list[int]:
        """One count per text, in order. Raises `TokenCountUnavailable` if it could not run."""


# Runs inside envs.TRAIN. transformers only -- no torch import -- because the
# whole point of a separate environment is that the parent never loads either.
_TOKEN_COUNT_SCRIPT = r'''
"""Count tokens for a list of texts. Writes JSON: {"counts": [int, ...]}."""
import json
import sys
from pathlib import Path


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())

    from transformers import AutoTokenizer

    kwargs = {"revision": spec["revision"]} if spec.get("revision") else {}
    tok = AutoTokenizer.from_pretrained(spec["model"], **kwargs)
    counts = [len(tok(text, add_special_tokens=False)["input_ids"]) for text in spec["texts"]]
    Path(spec["out"]).write_text(
        json.dumps({"counts": counts, "tokenizer": type(tok).__name__}), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass
class HuggingFaceTokenCounter:
    """The real counter: a tokenizer inside `envs.TRAIN`, one subprocess per call.

    Raises rather than returning an estimate. A guessed token count that turns
    out to be short is an over-length row reaching training, where it is
    truncated and its answer disappears -- the failure this whole check exists
    to prevent.
    """

    model: str
    revision: str | None = None
    env: envs.StageEnv = envs.TRAIN
    timeout_s: int = TOKEN_COUNT_TIMEOUT_S
    auto_provision: bool = True

    name = "transformers"

    def describe(self) -> dict[str, Any]:
        return {
            "tokenizer": self.model,
            "revision": self.revision,
            "environment": self.env.name,
            "requirements": list(self.env.requirements),
        }

    def count(self, texts: Sequence[str]) -> list[int]:
        if self.auto_provision:
            try:
                self.env.provision()
            except (RuntimeError, OSError) as exc:
                logger.exception("could not provision environment %r", self.env.name)
                raise TokenCountUnavailable(
                    f"environment {self.env.name!r} unavailable: {exc}"
                ) from exc

        with tempfile.TemporaryDirectory(prefix="litetune-tokens-") as tmp:
            work = Path(tmp)
            script = work / "count_tokens.py"
            script.write_text(_TOKEN_COUNT_SCRIPT, encoding="utf-8")
            out = work / "counts.json"
            config = work / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "model": self.model,
                        "revision": self.revision,
                        "texts": list(texts),
                        "out": str(out),
                    }
                ),
                encoding="utf-8",
            )
            try:
                proc = self.env.run(["python", str(script), str(config)], timeout=self.timeout_s)
            except subprocess.TimeoutExpired as exc:
                logger.warning("tokenizer did not answer within %ss", self.timeout_s)
                raise TokenCountUnavailable(
                    f"the tokenizer did not answer within {self.timeout_s}s"
                ) from exc
            except OSError as exc:
                logger.exception("could not start the tokenizer script")
                raise TokenCountUnavailable(f"{type(exc).__name__}: {exc}") from exc

            # A non-zero exit is data about this machine, and it is handed to
            # the caller as an exception only because a *missing count* cannot
            # be represented as a number without inventing one. A *negative*
            # code is not an exit status at all -- see `litetune.exits` -- so it
            # is described as the kill it was.
            if proc.returncode != 0 or not out.is_file():
                reading = read_returncode(proc.returncode)
                raise TokenCountUnavailable(
                    f"the tokenizer was {reading.describe('the data')} without counts: "
                    f"{(proc.stderr or '').strip()[-300:] or 'no stderr'}"
                )
            payload = json.loads(out.read_text(encoding="utf-8"))

        counts = [int(c) for c in payload["counts"]]
        if len(counts) != len(texts):
            raise TokenCountUnavailable(
                f"the tokenizer returned {len(counts)} counts for {len(texts)} texts"
            )
        return counts


@dataclass(frozen=True)
class OverLength:
    """One row that cannot be trained on without losing part of itself."""

    lineno: int
    tokens: int
    prompt_tokens: int
    completion_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_line": self.lineno,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def _summary(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p95": p95,
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class LengthStats:
    """Token lengths against the context window the artifact will be built with."""

    n: int
    context_length: int
    counter: dict[str, Any]
    prompt: dict[str, int]
    completion: dict[str, int]
    total: dict[str, int]
    supervised_tokens: int
    total_tokens: int
    over_length: tuple[OverLength, ...] = ()

    @property
    def available(self) -> bool:
        return True

    @property
    def expected_supervised_fraction(self) -> float:
        """The share of tokens a completion-masked run should compute loss on.

        Reported here so that `tune`'s observed fraction has something measured
        to be compared against: on the data shape behind this tool the
        declarations are ~330 of ~350 tokens, and the expected value is ~0.07. A
        run reporting ~1.0 did not mask anything.
        """
        return self.supervised_tokens / self.total_tokens if self.total_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": True,
            "n": self.n,
            "context_length": self.context_length,
            "counter": self.counter,
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "total_tokens": self.total,
            "expected_supervised_token_fraction": round(self.expected_supervised_fraction, 6),
            "over_length": [o.as_dict() for o in self.over_length],
            "sequence_overhead_tokens": SEQUENCE_OVERHEAD_TOKENS,
        }


def measure_lengths(rows: Sequence[Row], counter: TokenCounter, context_length: int) -> LengthStats:
    """Token statistics for the split, and every row that will not fit."""
    prompts = [row.prompt for row in rows]
    completions = [row.completion for row in rows]
    # One call, not two: the real counter pays a process launch and a tokenizer
    # load per call.
    counts = counter.count([*prompts, *completions])
    # `counter` is a Protocol, so a short answer is a thing that can happen --
    # a truncated subprocess, a third-party implementation, a tokenizer that
    # dropped a row. Slicing and zipping a short list silently drops rows from
    # the over-length check, and catching over-length rows before a GPU is
    # rented is the whole reason this stage exists.
    if len(counts) != 2 * len(rows):
        raise PrepareError(
            f"the token counter returned {len(counts)} counts for {2 * len(rows)} texts; "
            "the over-length check cannot be performed on a partial answer"
        )
    prompt_counts = counts[: len(rows)]
    completion_counts = counts[len(rows) :]
    totals = [
        p + c + SEQUENCE_OVERHEAD_TOKENS
        for p, c in zip(prompt_counts, completion_counts, strict=True)
    ]

    over = tuple(
        OverLength(lineno=row.lineno, tokens=total, prompt_tokens=p, completion_tokens=c)
        for row, total, p, c in zip(rows, totals, prompt_counts, completion_counts, strict=True)
        if total > context_length
    )
    return LengthStats(
        n=len(rows),
        context_length=context_length,
        counter=counter.describe(),
        prompt=_summary(prompt_counts),
        completion=_summary(completion_counts),
        total=_summary(totals),
        supervised_tokens=sum(completion_counts),
        total_tokens=sum(totals),
        over_length=over,
    )


# ---------------------------------------------------------------------------
# Scoreability: can exact match say anything about this argument at all?
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def is_extractive(value: str, prompt: str) -> bool:
    """Whether the label is a literal span of its own prompt.

    Whitespace-collapsed and case-folded, because a value quoted from a prompt
    across a line break is still quoted. An empty value is not extractive: `""
    in prompt` is true for every prompt, and counting it would report an
    unscoreable argument as perfectly quoted.
    """
    cleaned = _normalise(value)
    return bool(cleaned) and cleaned in _normalise(prompt)


@dataclass(frozen=True)
class ArgumentProfile:
    """One tool argument, and whether exact match on it can measure anything.

    Cardinality alone decides nothing. Google's `mobile-actions` has 2,270
    distinct values in 2,276 examples for one field and scores fine, because
    every value is a span of its own prompt -- the model copies rather than
    invents. The field that scored 0.00 for every model including the base had
    92 distinct values in 95 examples *and* almost none of them appeared in the
    prompt. It is the conjunction that is unscoreable.
    """

    tool: str
    argument: str
    n: int
    unique_values: int
    extractive: Proportion
    examples: tuple[str, ...] = ()

    @property
    def unique_share(self) -> float:
        return self.unique_values / self.n if self.n else 0.0

    @property
    def high_cardinality(self) -> bool:
        return self.unique_share >= HIGH_CARDINALITY_SHARE

    @property
    def scoreable(self) -> bool:
        return not (self.high_cardinality and self.extractive.value < MIN_EXTRACTIVE_SHARE)

    def limitation(self) -> str:
        return (
            f"{self.tool}.{self.argument} has {self.unique_values} distinct values in {self.n} "
            f"examples and only {self.extractive.value:.2f} of them appear verbatim in their own "
            "prompt: the label is invented rather than quoted, so exact match on this argument is "
            "unscoreable by construction and will read 0.00 for every model including the untuned "
            "base. One measured dataset failed exactly this way (92 distinct values in 95 "
            "examples); a comparable field with 2,270 distinct values in 2,276 examples scored "
            "normally because it was always a literal span of the prompt"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "argument": self.argument,
            "n": self.n,
            "unique_values": self.unique_values,
            "unique_share": round(self.unique_share, 6),
            "extractive": self.extractive.as_dict(),
            "high_cardinality": self.high_cardinality,
            "scoreable": self.scoreable,
            "example_values": list(self.examples),
        }


def profile_arguments(rows: Iterable[Row]) -> tuple[ArgumentProfile, ...]:
    """Cardinality and extractiveness for every (tool, argument) pair."""
    seen: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for row in rows:
        if row.target is None:
            continue
        for argument, value in row.target.args.items():
            seen.setdefault((row.target.name, argument), []).append((value, row.prompt))

    profiles: list[ArgumentProfile] = []
    for (tool, argument), pairs in sorted(seen.items()):
        values = [value for value, _ in pairs]
        quoted = sum(1 for value, prompt in pairs if is_extractive(value, prompt))
        profiles.append(
            ArgumentProfile(
                tool=tool,
                argument=argument,
                n=len(pairs),
                unique_values=len(set(values)),
                extractive=Proportion.of(quoted, len(pairs)),
                # Three is enough to recognise the shape of the field in a
                # report without reproducing the dataset in it.
                examples=tuple(values[:3]),
            )
        )
    return tuple(profiles)


# ---------------------------------------------------------------------------
# Headroom: the hook, deliberately not the measurement
# ---------------------------------------------------------------------------


class HeadroomProbe(Protocol):
    """Base-model accuracy per slice, measured elsewhere.

    Deliberately an injected Protocol and not an implementation. Measuring
    headroom means running the untuned base over the split, which is `evaluate`'s
    job and costs a generation pass; doing it here would make `prepare` -- the
    cheap step that runs before anyone books a GPU -- as expensive as the thing
    it is meant to prevent. When no probe is supplied the result records the
    headroom of every slice as `Unavailable` with the reason, which is the
    honest answer rather than an empty list that reads as "no problems found".
    """

    def base_accuracy(self, slice_name: str, rows: Sequence[Row]) -> Proportion | Unavailable:
        """The untuned base model's exact match on this slice."""


@dataclass(frozen=True)
class SliceProfile:
    """One slice of the data -- currently one tool -- and its base-model headroom."""

    name: str
    n: int
    base_accuracy: Proportion | Unavailable

    @property
    def has_headroom(self) -> bool | None:
        """True, False, or None for "not measured". Three values, as everywhere else."""
        if isinstance(self.base_accuracy, Unavailable):
            return None
        return self.base_accuracy.value < NO_HEADROOM_ABOVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "slice": self.name,
            "n": self.n,
            "base_accuracy": self.base_accuracy.as_dict(),
            "has_headroom": self.has_headroom,
        }


def profile_slices(rows: Sequence[Row], probe: HeadroomProbe | None) -> tuple[SliceProfile, ...]:
    """One profile per tool. Without a probe every headroom is explicitly unmeasured."""
    by_tool: dict[str, list[Row]] = {}
    for row in rows:
        by_tool.setdefault(row.tool, []).append(row)

    profiles: list[SliceProfile] = []
    for name, slice_rows in sorted(by_tool.items()):
        if probe is None:
            accuracy: Proportion | Unavailable = Unavailable(NO_HEADROOM_MEASUREMENT)
        else:
            accuracy = probe.base_accuracy(name, slice_rows)
        profiles.append(SliceProfile(name=name, n=len(slice_rows), base_accuracy=accuracy))
    return tuple(profiles)


# ---------------------------------------------------------------------------
# The request and its result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepareRequest:
    """One ingestion. `context_length` is required and is the *binding* limit.

    A sequence has to fit two windows, and they are declared in different
    sections of the spec: `export.context_length`, baked into the artifact and
    unchangeable afterwards, and `train.max_seq_length`. Pass the smaller of the
    two. `tune` re-checks against its own limit and refuses to truncate, so a
    caller who passes the wrong one gets a named row rather than a quietly
    shortened example -- but they pay for it after the model has loaded instead
    of before.
    """

    data: Path
    output_dir: Path
    context_length: int
    seed: int = 0
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION
    heldout_size: int | None = None
    min_heldout_examples: int = MIN_HELDOUT_EXAMPLES
    tokens: TokenCounter | None = None
    headroom: HeadroomProbe | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", Path(self.data))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.context_length < 1:
            raise ValueError(f"context_length must be at least 1, got {self.context_length}")
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError(
                f"heldout_fraction must be between 0 and 1 exclusive, got {self.heldout_fraction}"
            )
        if self.heldout_size is not None and self.heldout_size < 1:
            raise ValueError(f"heldout_size must be at least 1, got {self.heldout_size}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": str(self.data),
            "output_dir": str(self.output_dir),
            "context_length": self.context_length,
            "seed": self.seed,
            "heldout_fraction": self.heldout_fraction,
            "heldout_size": self.heldout_size,
            "min_heldout_examples": self.min_heldout_examples,
            "tokens": self.tokens.describe() if self.tokens is not None else None,
            "headroom_probe": type(self.headroom).__name__ if self.headroom else None,
        }


@dataclass(frozen=True)
class SplitFile:
    """One written split, identified by the content of what was written."""

    name: str
    path: Path
    n: int
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "n": self.n,
            "content_sha256": self.content_sha256,
        }


@dataclass
class PrepareResult:
    """What the dataset is, and every reason a measurement on it might not mean anything."""

    request: PrepareRequest
    checks: CheckSet
    content_sha256: str
    n_rows: int
    train: SplitFile | None = None
    heldout: SplitFile | None = None
    lengths: LengthStats | Unavailable = field(
        default_factory=lambda: Unavailable("token lengths were not measured")
    )
    arguments: tuple[ArgumentProfile, ...] = ()
    slices: tuple[SliceProfile, ...] = ()
    # Checks that were never in scope for this run, with the reason. Not a
    # fourth outcome: a reader must be able to tell that two checks ran rather
    # than assuming three did. Same construction as `liveness.LivenessResult`.
    skipped: list[SkippedCheck] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    report_path: Path | None = None

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.PASSED

    @property
    def unscoreable(self) -> tuple[ArgumentProfile, ...]:
        return tuple(p for p in self.arguments if not p.scoreable)

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def spec_fragment(self) -> dict[str, Any]:
        """The `dataset` and `eval` sections this split would be declared with.

        `spec.Dataset.content_sha256` and `spec.EvalSpec.heldout_content_sha256`
        are required fields with no default, and they identify files that do not
        exist until this stage has run. So the hashes are emitted here, for a
        spec to be written against -- otherwise a job spec can only ever
        describe a split someone produced by hand.
        """
        return {
            "dataset": {
                "uri": str(self.train.path) if self.train else None,
                "content_sha256": _bare_digest(self.train.content_sha256) if self.train else None,
                "format": "jsonl",
            },
            "eval": {
                "heldout_uri": str(self.heldout.path) if self.heldout else None,
                "heldout_content_sha256": (
                    _bare_digest(self.heldout.content_sha256) if self.heldout else None
                ),
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PREPARE_SCHEMA,
            "outcome": self.outcome.value,
            "request": self.request.as_dict(),
            "source": {
                "path": str(self.request.data),
                "content_sha256": self.content_sha256,
                "rows": self.n_rows,
                # The split is a function of these two and nothing else, which
                # is what makes it reproducible from the report alone.
                "split_seed": split_seed(self.content_sha256, self.request.seed),
            },
            "train": self.train.as_dict() if self.train else None,
            "heldout": self.heldout.as_dict() if self.heldout else None,
            "lengths": self.lengths.as_dict(),
            "arguments": [a.as_dict() for a in self.arguments],
            "unscoreable_arguments": [f"{a.tool}.{a.argument}" for a in self.unscoreable],
            "slices": [s.as_dict() for s in self.slices],
            "checks": self.checks.as_dict() | {"skipped": [s.as_dict() for s in self.skipped]},
            "limitations": list(self.limitations),
            "spec_fragment": self.spec_fragment(),
        }


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


def _write_split(path: Path, rows: Sequence[Row]) -> SplitFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row.as_record(), sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return SplitFile(name=path.stem, path=path, n=len(rows), content_sha256=hash_file(path))


def prepare(request: PrepareRequest, events: EventStream | None = None) -> PrepareResult:
    """Read, measure and split a dataset. Raises only if the file is not a dataset.

    Every other problem -- a row too long, a held-out set too small, an argument
    nothing can score -- is a recorded result, because those are facts about the
    data that a report has to carry rather than exceptions that lose them.
    """
    events = events or EventStream(echo_json=False)
    events.stage_started("prepare", data=str(request.data), seed=request.seed)

    rows = read_rows(request.data)
    content_sha256 = hash_file(request.data)
    result = PrepareResult(
        request=request,
        checks=CheckSet(name=f"prepare:{request.data.name}"),
        content_sha256=content_sha256,
        n_rows=len(rows),
    )
    events.metric("rows", len(rows), source=str(request.data))

    # -- lengths -----------------------------------------------------------
    measured: list[LengthStats] = []
    with guard(LENGTH_CHECK) as sink:
        if request.tokens is None:
            sink.append(
                Check.unchecked(
                    LENGTH_CHECK,
                    "no tokenizer was supplied, so no example was measured against the "
                    f"{request.context_length}-token context window. An over-length row reaches "
                    "training and is truncated there, which removes the answer it was supposed "
                    "to teach",
                    observed={"context_length": request.context_length},
                )
            )
        else:
            stats = measure_lengths(rows, request.tokens, request.context_length)
            measured.append(stats)
            if stats.over_length:
                named = ", ".join(
                    f"line {o.lineno} ({o.tokens} tokens)" for o in stats.over_length[:10]
                )
                extra = len(stats.over_length) - 10
                more = f" and {extra} more" if extra > 0 else ""
                sink.append(
                    Check.failed(
                        LENGTH_CHECK,
                        f"{len(stats.over_length)}/{len(rows)} rows exceed the "
                        f"{request.context_length}-token context window: {named}{more}. They are "
                        "named rather than dropped: truncating an example removes its supervised "
                        "span, so the row still costs a training step and teaches nothing",
                        observed={
                            "over_length": [o.as_dict() for o in stats.over_length],
                            "context_length": request.context_length,
                        },
                    )
                )
            else:
                sink.append(
                    Check.passed(
                        LENGTH_CHECK,
                        f"longest example is {stats.total['max']} tokens against a "
                        f"{request.context_length}-token window",
                        observed=stats.as_dict(),
                    )
                )
    length_check = result.checks.add(sink[0])
    events.check(length_check)
    result.lengths = measured[0] if measured else Unavailable(length_check.detail)
    if isinstance(result.lengths, LengthStats):
        events.metric(
            "expected_supervised_token_fraction",
            round(result.lengths.expected_supervised_fraction, 6),
        )
    if not length_check.conclusive:
        result.limitation(
            f"token lengths were not measured ({length_check.detail}); whether every example fits "
            "the context window is unknown, and a row that does not fit is truncated during "
            "training without a warning"
        )

    # -- scoreability, before anyone pays for a GPU -------------------------
    result.arguments = profile_arguments(rows)
    unscoreable = result.unscoreable
    if not result.arguments:
        scoreability = Check.unchecked(
            SCOREABILITY_CHECK,
            "no row carries a 'target', so there are no arguments to profile and nothing here "
            "can say whether exact match would measure anything",
            observed={"rows": len(rows)},
        )
    elif len(unscoreable) == len(result.arguments):
        # Every argument invented rather than quoted: exact match has nothing
        # left to measure on this dataset, and it will read 0.00 for the tuned
        # model, the base model and every quantization recipe alike. That is not
        # a gradient of quality with a caveat on it, it is no instrument at all,
        # and it is the state the measured 92-distinct-values-in-95-examples
        # dataset was in. Reported before a GPU is booked, which is the whole
        # point of this stage running first.
        scoreability = Check.failed(
            SCOREABILITY_CHECK,
            f"all {len(result.arguments)} profiled argument(s) are high-cardinality and "
            f"non-extractive ({', '.join(f'{a.tool}.{a.argument}' for a in unscoreable)}): exact "
            "match cannot measure anything on this dataset and will read 0.00 for every model",
            observed={"arguments": [a.as_dict() for a in unscoreable]},
        )
        for profile in unscoreable:
            result.limitation(profile.limitation())
    elif unscoreable:
        # Some, not all. The dataset may be exactly what the product needs and
        # the remaining arguments still carry signal, so this is a limitation on
        # what the numbers mean rather than a refusal to produce them --
        # `metrics.py` already reports operation name and argument accuracy
        # separately for the same reason.
        scoreability = Check.passed(
            SCOREABILITY_CHECK,
            f"{len(result.arguments) - len(unscoreable)} of {len(result.arguments)} arguments are "
            f"scoreable; {', '.join(f'{a.tool}.{a.argument}' for a in unscoreable)} "
            "cannot be scored by exact match and will drag every model's argument accuracy down "
            "by the same amount",
            observed={
                "scoreable": [f"{a.tool}.{a.argument}" for a in result.arguments if a.scoreable],
                "unscoreable": [a.as_dict() for a in unscoreable],
            },
        )
        for profile in unscoreable:
            result.limitation(profile.limitation())
    else:
        scoreability = Check.passed(
            SCOREABILITY_CHECK,
            f"{len(result.arguments)} argument(s) profiled; none is both high-cardinality and "
            "non-extractive",
            observed={"arguments": [a.as_dict() for a in result.arguments]},
        )
    result.checks.add(scoreability)
    events.check(scoreability)

    # -- headroom: the hook, reported as unmeasured until a probe exists ----
    result.slices = profile_slices(rows, request.headroom)
    if request.headroom is None:
        result.limitation(NO_HEADROOM_MEASUREMENT)
    else:
        for slice_profile in result.slices:
            base = slice_profile.base_accuracy
            # `has_headroom is False` already implies a measured proportion --
            # it returns None when the accuracy is Unavailable. The isinstance
            # states that invariant where it is relied on, so a later change to
            # the property fails the type check instead of raising here.
            if slice_profile.has_headroom is False and isinstance(base, Proportion):
                result.limitation(
                    f"slice {slice_profile.name!r} ({slice_profile.n} examples): the untuned "
                    f"base already scores {base.value:.4f}, at or above "
                    f"{NO_HEADROOM_ABOVE}. "
                    "A fine-tune measured on this slice reports roughly zero whatever it did, so "
                    "the slice cannot show a gain and should not be read as evidence of one"
                )

    # -- the split ---------------------------------------------------------
    heldout_n = heldout_size_for(len(rows), request.heldout_fraction, request.heldout_size)
    if request.heldout_size is not None and heldout_n != request.heldout_size:
        # Silently handing back 99 when 500 was asked for is the shape of every
        # failure in this codebase's history: the caller reads their own number
        # back out of the spec and never learns it was not honoured.
        result.limitation(
            f"a held-out split of {request.heldout_size} was requested but only {len(rows)} rows "
            f"are available, so {heldout_n} were held out and the training set has "
            f"{len(rows) - heldout_n}. Every interval on this split is computed at n={heldout_n}, "
            "not at the size that was asked for"
        )
    if heldout_n < 1:
        split_check = Check.failed(
            SPLIT_CHECK,
            f"{len(rows)} row(s) cannot be split into a training set and a held-out set; with no "
            "held-out data nothing downstream can be measured at all",
            observed={"rows": len(rows)},
        )
        result.checks.add(split_check)
        events.check(split_check)
    elif length_check.outcome is Outcome.FAILED:
        # Deliberate: no split is written when rows do not fit. Writing one
        # anyway would put the over-length rows into a file that the next stage
        # reads without ever seeing this check, which is precisely how a
        # truncated example gets trained on.
        #
        # Recorded as *skipped* rather than as an UNCHECKED check, following
        # `liveness.liveness_tier`. An unchecked item makes the whole CheckSet
        # `could not check`, which would bury a measured failure -- rows that do
        # not fit -- under "we could not tell". The check was never in scope;
        # that is a different statement from "it could not be performed".
        result.skipped.append(
            SkippedCheck(
                SPLIT_CHECK,
                f"not reached: {LENGTH_CHECK} did not pass, so no split was written. Fix or "
                "remove the rows it named and run prepare again",
            )
        )
    else:
        train_rows, heldout_rows = split_rows(rows, content_sha256, request.seed, heldout_n)
        result.train = _write_split(request.output_dir / "train.jsonl", train_rows)
        result.heldout = _write_split(request.output_dir / "heldout.jsonl", heldout_rows)
        for split in (result.train, result.heldout):
            events.artifact(
                str(split.path), name=split.name, n=split.n, content_hash=split.content_sha256
            )
        split_check = Check.passed(
            SPLIT_CHECK,
            f"{result.train.n} training and {result.heldout.n} held-out examples, split by "
            f"content {_bare_digest(content_sha256)[:16]} and seed {request.seed}",
            observed={
                "train": result.train.as_dict(),
                "heldout": result.heldout.as_dict(),
                "split_seed": split_seed(content_sha256, request.seed),
            },
        )
        result.checks.add(split_check)
        events.check(split_check)
        events.metric("heldout_examples", result.heldout.n)

        if result.heldout.n < request.min_heldout_examples:
            # A warning that travels with every number measured on this split,
            # not a refusal. See the module docstring: 0.172 at n=64 against
            # 0.024 at n=640, with three conclusions overturned in between.
            result.limitation(
                f"the held-out split has {result.heldout.n} examples, below "
                f"{request.min_heldout_examples}: at n=64 a recipe comparison read as 0.172 and "
                "at n=640 the same comparison was 0.024, and three conclusions drawn at the "
                "smaller size were overturned. Every difference measured on this split should be "
                "treated as unresolved unless its interval says otherwise"
            )
            events.note(
                f"held-out split is {result.heldout.n} examples, below "
                f"{request.min_heldout_examples}",
                n=result.heldout.n,
                minimum=request.min_heldout_examples,
            )

    # -- the report --------------------------------------------------------
    # Written whatever the outcome: a prepare that failed is exactly the run
    # whose reasons someone needs to read.
    request.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = request.output_dir / "prepare.json"
    report_path.write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    result.report_path = report_path
    events.artifact(str(report_path), name="prepare.json")

    events.stage_finished(
        result.outcome.value,
        rows=len(rows),
        heldout=result.heldout.n if result.heldout else 0,
        limitations=len(result.limitations),
    )
    return result
