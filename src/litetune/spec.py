"""The job spec: what to run, hashed so that only real changes re-run it.

The spec is the whole job, declared. That is what lets the same code run on a
laptop and behind a hosted service, and it is what makes a run reproducible
months later from the manifest alone. Four rules here were paid for.

**A base model must be pinned.** `main` is not a version. The same unchanged
definition produced a working export on 2026-08-26 and `AttributeError:
pad_token` on 2026-08-30; a mutable ref moves the same way and the manifest
would still say the same thing. `base_model.revision` is required and a mutable
ref is refused by name.

**Data is identified by content, never by location.** `dataset.uri` says where
to fetch; `dataset.content_sha256` says what was fetched, and only the second
one reaches a cache key. A file replaced at the same URI must invalidate every
stage downstream of it -- keying on the URI trains the next run on stale data
under a green manifest.

**Thresholds are not measurements.** `gates` is a separate section and
`slice_for` refuses to hand it to a measurement stage. Tightening a gate has to
re-judge from recorded metrics, not re-run hours of generation, and the only
way to guarantee that is for the measurement stages to be structurally unable
to see the threshold.

**The environment enters the key by its resolved identity, not by its
declaration.** `toolchain` is therefore in no stage's slice: it is resolved into
a `StageEnv` whose `identity` is passed to `hash_for` separately. A declaration
that resolves differently over time is precisely the failure being guarded
against, so hashing the declaration alone would not catch it.

Validation names the offending field and never substitutes a default for a
missing required one. A spec that is wrong in a way nobody notices is worse
than one that will not load.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from litetune import envs
from litetune.envs import StageEnv, UnpinnedRequirement

logger = logging.getLogger(__name__)

SPEC_SCHEMA = "litetune.spec/1"

# Length of a cache key. 64 bits of sha256 over a canonical payload; short
# enough to read in a manifest, far past collision risk for one project's runs.
_KEY_CHARS = 16

SECTION_NAMES = (
    "base_model",
    "dataset",
    "train",
    "export",
    "eval",
    "gates",
    "toolchain",
)

# `gates` is absent from this list on purpose: a spec that declares no
# thresholds measures without judging, which is a legitimate job and is recorded
# as a limitation rather than defaulted into a verdict.
REQUIRED_SECTIONS = tuple(name for name in SECTION_NAMES if name != "gates")

# Refs that move. A tag can be moved too, which is why a revision that is not a
# full commit sha is recorded as a limitation rather than accepted silently.
MUTABLE_REFS = frozenset(
    {"main", "master", "head", "trunk", "dev", "develop", "latest", "default", "tip"}
)

_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# Recipe names become directory names and command-line arguments downstream.
_RECIPE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

# No `float16`. The reference notebooks warn against it three times, and the
# design document that decided this project's shape records the warning as a
# directive rather than advice: a fine-tune in float16 overflows where bfloat16
# -- same width, more exponent -- does not, and the failure is silent loss
# rather than an error. Admitting it here and defaulting elsewhere to bfloat16
# would let a spec file quietly pick the one dtype the design forbids.
DTYPES = ("float32", "bfloat16")
# Gemma-family models are documented as needing eager attention for correct
# results; the alternatives are accepted but the default is the measured one.
ATTN_IMPLEMENTATIONS = ("eager", "sdpa", "flash_attention_2")
DATASET_FORMATS = ("jsonl",)
TRAIN_MODES = ("full", "lora", "skip")
SAMPLERS = ("greedy", "top_k", "top_p")
PROMPT_MODES = ("prerendered", "runtime_rendered")  # mirrors evaluate.PromptMode

# Which spec sections each stage's result depends on. This is the table that
# keeps a gate threshold out of a measurement key.
STAGE_SECTIONS: dict[str, tuple[str, ...]] = {
    "train": ("base_model", "dataset", "train"),
    "merge": ("base_model", "train"),
    # Not `dataset`: an export depends on the checkpoint it is given, and the
    # checkpoint reaches the key as an input content hash. Smearing the dataset
    # across every downstream slice would invalidate exports of checkpoints the
    # data never touched, and would still be no safer -- the chain of content
    # hashes already carries the change.
    "export": ("base_model", "export"),
    "measure": ("base_model", "eval"),
    "judge": ("gates",),
    "bundle": ("base_model", "export", "eval", "gates"),
}

# Only these may see `gates`. Everything else measures, and a measurement that
# can see the threshold it will be judged against is one refactor away from
# being re-run when the threshold moves.
JUDGEMENT_STAGES = frozenset({"judge", "bundle"})

# Which toolchain environments each stage runs in. `measure` needs two: the
# litert-lm runtime for the converted artifact and the training environment for
# the float reference, and the pair is what identifies the measurement.
STAGE_ENVS: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "merge": ("train",),
    "export": ("export",),
    "measure": ("runtime", "train"),
    "judge": (),
    "bundle": (),
}

TOOLCHAIN_ENVS = ("train", "export", "runtime")

# What `toolchain.<name>: default` resolves to. The requirements are recorded in
# the manifest either way, so a spec that says `default` still reports which
# pins it actually ran on.
DEFAULT_ENVS: dict[str, StageEnv] = {
    "train": envs.TRAIN,
    "export": envs.EXPORT,
    "runtime": envs.RUNTIME,
}

# The identity of "no environment": litetune's own interpreter. The version is
# in it because in-process work is version-sensitive too, and a key that cannot
# tell 3.11 from 3.14 is the same silent-resolution problem one level down.
IN_PROCESS = f"in-process:{sys.version.split()[0]}"

# Below this the interval swamps the effects this tool measures; see the README
# on the 0.172-at-64 against 0.024-at-640 comparison.
DEFAULT_MIN_HELDOUT_EXAMPLES = 200


class SpecError(ValueError):
    """The spec is not valid. The message names the offending field."""


# ---------------------------------------------------------------------------
# Reading fields with an error that names them
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    return type(value).__name__


class _Reader:
    """Reads one section, refusing unknown keys and naming every problem.

    An unknown key is an error rather than a warning because the shape of the
    mistake is a typo -- `content_sha_256` -- and a spec that silently ignores
    it keys the cache on a dataset identity nobody supplied.
    """

    def __init__(self, data: Any, path: str) -> None:
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise SpecError(f"{path}: expected a mapping, got {_type_name(data)}")
        self._data = dict(data)
        self._path = path
        self._seen: set[str] = set()

    def _field(self, name: str) -> str:
        return f"{self._path}.{name}" if self._path else name

    def require(self, name: str, kind, **kwargs) -> Any:
        self._seen.add(name)
        if name not in self._data or self._data[name] is None:
            raise SpecError(f"{self._field(name)} is required and was not set")
        return kind(self._data[name], self._field(name), **kwargs)

    def optional(self, name: str, kind, default: Any, **kwargs) -> Any:
        self._seen.add(name)
        if name not in self._data or self._data[name] is None:
            return default
        return kind(self._data[name], self._field(name), **kwargs)

    def present(self, name: str) -> bool:
        return self._data.get(name) is not None

    def done(self) -> None:
        unknown = sorted(set(self._data) - self._seen)
        if unknown:
            raise SpecError(
                f"{self._path or 'spec'}: unknown field(s) {unknown}. A field litetune does not "
                "read is a field that is not doing what it looks like it is doing"
            )


def _as_str(value: Any, field_name: str, choices: Sequence[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{field_name}: expected a non-empty string, got {value!r}")
    text = value.strip()
    if choices is not None and text not in choices:
        raise SpecError(f"{field_name}: {text!r} is not one of {list(choices)}")
    return text


def _as_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SpecError(f"{field_name}: expected true or false, got {value!r}")
    return value


def _as_int(value: Any, field_name: str, minimum: int | None = None) -> int:
    # `bool` is an `int` in Python; `context_length: true` must not read as 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{field_name}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise SpecError(f"{field_name}: expected at least {minimum}, got {value}")
    return value


def _as_float(value: Any, field_name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecError(f"{field_name}: expected a number, got {value!r}")
    number = float(value)
    if minimum is not None and number < minimum:
        raise SpecError(f"{field_name}: expected at least {minimum}, got {number}")
    return number


def _as_str_list(value: Any, field_name: str, pattern: re.Pattern[str] | None = None) -> tuple:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpecError(
            f"{field_name}: expected a list, got {_type_name(value)}. One entry is still a list "
            "of one"
        )
    items = [_as_str(item, f"{field_name}[{i}]") for i, item in enumerate(value)]
    if not items:
        raise SpecError(f"{field_name}: expected at least one entry")
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise SpecError(f"{field_name}: duplicate entries {duplicates}")
    if pattern is not None:
        bad = [item for item in items if not pattern.fullmatch(item)]
        if bad:
            raise SpecError(
                f"{field_name}: {bad} are not usable names; entries reach the filesystem and the "
                f"command line, so they must match {pattern.pattern}"
            )
    return tuple(items)


def _as_is(value: Any, field_name: str) -> Any:
    """No conversion. For a field whose shape depends on what it turned out to be."""
    return value


def _as_sha256(value: Any, field_name: str) -> str:
    text = _as_str(value, field_name).lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:") :]
    if not _SHA256_RE.fullmatch(text):
        raise SpecError(
            f"{field_name}: expected a 64-character sha256 hex digest, got {value!r}. This field "
            "is what identifies the data; a placeholder here means the cache is keyed on nothing"
        )
    return text


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def mutable_ref_refusal(revision: str, field: str) -> str | None:
    """Why this revision must be refused, or `None`. Shared: the CLI records one too.

    `bundle --base-model-revision main` was accepted while the same value in a
    spec file was refused, so the stricter of the two paths was the one fewer
    people use.
    """
    if revision.lower() in MUTABLE_REFS or revision.lower().startswith("refs/heads/"):
        return (
            f"{field}: {revision!r} is a mutable ref, not a revision. It resolves to different "
            "weights on different days while the spec and the manifest read identically -- the "
            "shape of the 2026-08-26/2026-08-30 pad_token failure. Pin a commit sha"
        )
    return None


def weak_revision_limitations(revision: str, field: str) -> list[str]:
    """A pin that is not a commit sha is recorded as weaker than it looks."""
    if _COMMIT_SHA_RE.fullmatch(revision.lower()):
        return []
    return [
        f"{field} {revision!r} is not a 40-character commit sha; tags can be moved, so this "
        "pin is weaker than it looks and two runs recording the same revision may not have "
        "used the same weights"
    ]


@dataclass(frozen=True)
class BaseModel:
    """The checkpoint everything starts from, pinned to one revision."""

    id: str
    revision: str
    # `bfloat16`, matching `tune.DEFAULT_DTYPE` and the CLI. Three defaults
    # disagreeing about the one parameter the design is emphatic on is how a
    # spec file ends up training in a precision nobody chose.
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "dtype": self.dtype,
            "attn_implementation": self.attn_implementation,
        }

    @classmethod
    def read(cls, data: Any, limitations: list[str]) -> BaseModel:
        reader = _Reader(data, "base_model")
        model_id = reader.require("id", _as_str)
        revision = reader.require("revision", _as_str)
        weak = mutable_ref_refusal(revision, "base_model.revision")
        if weak:
            raise SpecError(weak)
        limitations.extend(weak_revision_limitations(revision, "base_model.revision"))
        # Declared defaults, not silent ones: both are hashed, so changing
        # either re-runs training. `bfloat16` and `eager` are what every number
        # in the README was produced with -- the comment said `float32` for a
        # while after the line below stopped saying it, which is the same drift
        # this block exists to prevent.
        dtype = reader.optional("dtype", _as_str, "bfloat16", choices=DTYPES)
        attn = reader.optional(
            "attn_implementation", _as_str, "eager", choices=ATTN_IMPLEMENTATIONS
        )
        reader.done()
        return cls(id=model_id, revision=revision, dtype=dtype, attn_implementation=attn)


@dataclass(frozen=True)
class Dataset:
    """Training data. `uri` says where; `content_sha256` says what."""

    uri: str
    content_sha256: str
    format: str = "jsonl"
    split: str = "train"

    def as_dict(self) -> dict[str, Any]:
        # `uri` is deliberately absent from `identity()`, not from here: the
        # manifest must record where the data came from, and the cache must not
        # be keyed on it.
        return {
            "uri": self.uri,
            "content_sha256": self.content_sha256,
            "format": self.format,
            "split": self.split,
        }

    def identity(self) -> dict[str, Any]:
        """What reaches a cache key. Location is not part of it."""
        return {
            "content_sha256": self.content_sha256,
            "format": self.format,
            "split": self.split,
        }

    @classmethod
    def read(cls, data: Any, limitations: list[str]) -> Dataset:
        reader = _Reader(data, "dataset")
        uri = reader.require("uri", _as_str)
        content_sha256 = reader.require("content_sha256", _as_sha256)
        fmt = reader.optional("format", _as_str, "jsonl", choices=DATASET_FORMATS)
        split = reader.optional("split", _as_str, "train")
        reader.done()
        return cls(uri=uri, content_sha256=content_sha256, format=fmt, split=split)


@dataclass(frozen=True)
class TrainSpec:
    """Supervised fine-tuning parameters. `mode: skip` brings your own checkpoint."""

    mode: str
    epochs: float = 1.0
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_seq_length: int = 1024
    seed: int = 0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        common = {
            "mode": self.mode,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
        }
        if self.mode != "lora":
            return common
        return common | {
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    @classmethod
    def read(cls, data: Any, limitations: list[str]) -> TrainSpec:
        reader = _Reader(data, "train")
        mode = reader.require("mode", _as_str, choices=TRAIN_MODES)
        lora_set = [
            name for name in ("lora_rank", "lora_alpha", "lora_dropout") if reader.present(name)
        ]
        if mode != "lora" and lora_set:
            # Reading a field and then ignoring it is how a run that was
            # believed to be an adapter run turns out to have been a full one.
            raise SpecError(
                f"train.{lora_set[0]} is set but train.mode is {mode!r}; adapter settings only "
                "apply to mode 'lora'"
            )
        spec = cls(
            mode=mode,
            epochs=reader.optional("epochs", _as_float, 1.0, minimum=0.0),
            learning_rate=reader.optional("learning_rate", _as_float, 2e-5, minimum=0.0),
            batch_size=reader.optional("batch_size", _as_int, 8, minimum=1),
            max_seq_length=reader.optional("max_seq_length", _as_int, 1024, minimum=1),
            seed=reader.optional("seed", _as_int, 0),
            lora_rank=reader.optional("lora_rank", _as_int, 16, minimum=1),
            lora_alpha=reader.optional("lora_alpha", _as_int, 32, minimum=1),
            lora_dropout=reader.optional("lora_dropout", _as_float, 0.05, minimum=0.0),
        )
        reader.done()
        if mode == "lora":
            limitations.append(
                "train.mode is 'lora': an adapter run once scored 0.0625 against a 0.5625 base "
                "and passed every label-free check, so this job's result is only established by "
                "the held-out measurement"
            )
        return spec


@dataclass(frozen=True)
class ExportSpec:
    """Conversion to `.litertlm`, one artifact per candidate recipe."""

    recipes: tuple[str, ...]
    context_length: int
    # `None` means unspecified: export.py decides. A default here was the
    # second of three, and the spec file and the CLI disagreed.
    externalize_embedder: bool | None = None
    sampler: str = "greedy"

    def as_dict(self) -> dict[str, Any]:
        return {
            # Sorted, because a sweep is a set of candidates: reordering the
            # list does not change which artifacts come out, and re-exporting
            # for a reordering would cost minutes per recipe for nothing.
            "recipes": sorted(self.recipes),
            "context_length": self.context_length,
            "externalize_embedder": self.externalize_embedder,
            "sampler": self.sampler,
        }

    @classmethod
    def read(cls, data: Any, limitations: list[str]) -> ExportSpec:
        reader = _Reader(data, "export")
        recipes = reader.require("recipes", _as_str_list, pattern=_RECIPE_RE)
        # Required rather than defaulted: the context window is baked into the
        # artifact and cannot be changed afterwards, and there is no value that
        # is right for every model.
        context_length = reader.require("context_length", _as_int, minimum=1)
        externalize = reader.optional("externalize_embedder", _as_bool, None)
        sampler = reader.optional("sampler", _as_str, "greedy", choices=SAMPLERS)
        reader.done()
        if len(recipes) == 1:
            limitations.append(
                f"export.recipes contains only {recipes[0]!r}: a one-recipe sweep produces no "
                "accuracy-versus-size frontier, and the two recipes measured for the README "
                "differed by 0.024 exact match at the same bit width and 0.04% file size"
            )
        return cls(
            recipes=recipes,
            context_length=context_length,
            externalize_embedder=externalize,
            sampler=sampler,
        )


@dataclass(frozen=True)
class EvalSpec:
    """How measurement is performed. Judgement lives in `gates`, not here."""

    heldout_uri: str
    heldout_content_sha256: str
    limit: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 1
    backend: str = "cpu"
    prompt_mode: str = "prerendered"

    def as_dict(self) -> dict[str, Any]:
        return {
            "heldout_uri": self.heldout_uri,
            "heldout_content_sha256": self.heldout_content_sha256,
            "limit": self.limit,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "backend": self.backend,
            "prompt_mode": self.prompt_mode,
        }

    def identity(self) -> dict[str, Any]:
        """What reaches a cache key: the content of the split, not its location."""
        return {key: value for key, value in self.as_dict().items() if key != "heldout_uri"}

    @classmethod
    def read(cls, data: Any, limitations: list[str]) -> EvalSpec:
        reader = _Reader(data, "eval")
        uri = reader.require("heldout_uri", _as_str)
        content = reader.require("heldout_content_sha256", _as_sha256)
        # `limit` is part of the identity: a 64-example slice is a different
        # sample from the 640 it was cut out of, and treating one as evidence
        # about the other is the README's headline mistake.
        limit = reader.optional("limit", _as_int, None, minimum=1)
        spec = cls(
            heldout_uri=uri,
            heldout_content_sha256=content,
            limit=limit,
            max_tokens=reader.optional("max_tokens", _as_int, 256, minimum=1),
            temperature=reader.optional("temperature", _as_float, 0.0, minimum=0.0),
            top_k=reader.optional("top_k", _as_int, 1, minimum=1),
            backend=reader.optional("backend", _as_str, "cpu"),
            prompt_mode=reader.optional(
                "prompt_mode", _as_str, "prerendered", choices=PROMPT_MODES
            ),
        )
        reader.done()
        if spec.temperature > 0.0:
            limitations.append(
                f"eval.temperature is {spec.temperature}: decoding is not greedy, so two runs of "
                "this spec produce different generations and a difference between two "
                "measurements is not attributable to the models"
            )
        if spec.limit is not None and spec.limit < DEFAULT_MIN_HELDOUT_EXAMPLES:
            limitations.append(
                f"eval.limit is {spec.limit}, below {DEFAULT_MIN_HELDOUT_EXAMPLES}: at n=64 a "
                "recipe comparison read as 0.172 and at n=640 the same comparison was 0.024"
            )
        return spec


@dataclass(frozen=True)
class Gates:
    """Judgement parameters. Never in a measurement stage's slice.

    Every field is optional. A spec with no gates measures and does not judge,
    which is reported as such rather than as a pass.
    """

    min_exact_match: float | None = None
    max_conversion_cost: float | None = None
    max_artifact_bytes: int | None = None
    min_heldout_examples: int = DEFAULT_MIN_HELDOUT_EXAMPLES

    @property
    def empty(self) -> bool:
        return (
            self.min_exact_match is None
            and self.max_conversion_cost is None
            and self.max_artifact_bytes is None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_exact_match": self.min_exact_match,
            "max_conversion_cost": self.max_conversion_cost,
            "max_artifact_bytes": self.max_artifact_bytes,
            "min_heldout_examples": self.min_heldout_examples,
        }

    @classmethod
    def read(cls, data: Any, limitations: list[str]) -> Gates:
        reader = _Reader(data, "gates")
        gates = cls(
            min_exact_match=reader.optional("min_exact_match", _as_float, None, minimum=0.0),
            max_conversion_cost=reader.optional(
                "max_conversion_cost", _as_float, None, minimum=0.0
            ),
            max_artifact_bytes=reader.optional("max_artifact_bytes", _as_int, None, minimum=1),
            min_heldout_examples=reader.optional(
                "min_heldout_examples", _as_int, DEFAULT_MIN_HELDOUT_EXAMPLES, minimum=1
            ),
        )
        reader.done()
        if gates.empty:
            limitations.append(
                "no gates are declared: this job measures and does not judge, so a completed run "
                "means the numbers were produced, not that they met a bar"
            )
        return gates


@dataclass(frozen=True)
class ToolchainSpec:
    """The pinned environments the stages run in.

    Named `ToolchainSpec` rather than `Toolchain` because `export.Toolchain` is
    the *resolved* one -- what `pip show` reported at run time. This is the
    declaration; the difference between the two is the thing being guarded
    against.
    """

    environments: tuple[StageEnv, ...]

    def env(self, name: str) -> StageEnv:
        for env in self.environments:
            if env.name == name:
                return env
        raise SpecError(
            f"toolchain.{name} is not declared; known: {[e.name for e in self.environments]}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {env.name: list(env.requirements) for env in self.environments}

    def identities(self) -> dict[str, str]:
        return {env.name: env.identity for env in self.environments}

    def identity_for(self, env_names: Sequence[str]) -> str:
        """The combined identity of the environments a stage runs in.

        A stage that runs in litetune's own interpreter still has an identity:
        the interpreter. `hash_for` takes this rather than the declaration
        because the declaration is exactly what can resolve differently over
        time without changing.
        """
        if not env_names:
            return IN_PROCESS
        parts = [f"{name}:{self.env(name).identity}" for name in sorted(set(env_names))]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:_KEY_CHARS]

    @classmethod
    def read(
        cls, data: Any, limitations: list[str], defaults: Mapping[str, StageEnv]
    ) -> ToolchainSpec:
        reader = _Reader(data, "toolchain")
        environments: list[StageEnv] = []
        for name in TOOLCHAIN_ENVS:
            value = reader.require(name, _as_is)
            if isinstance(value, str):
                if value != "default":
                    raise SpecError(
                        f"toolchain.{name}: expected a list of pinned requirements or the word "
                        f"'default', got {value!r}"
                    )
                # Explicit in the spec, resolved from litetune's own pins, and
                # recorded in the manifest by identity. Not a silent default:
                # the word is in the file and the resolved requirements travel
                # with the run.
                environments.append(defaults[name])
                continue
            requirements = _as_str_list(value, f"toolchain.{name}")
            try:
                environments.append(StageEnv(name=name, requirements=requirements))
            except UnpinnedRequirement as exc:
                raise SpecError(f"toolchain.{name}: {exc}") from exc
        reader.done()
        return cls(environments=tuple(environments))


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """A whole job, validated. Hash it per stage; never re-run it for nothing."""

    base_model: BaseModel
    dataset: Dataset
    train: TrainSpec
    export: ExportSpec
    eval: EvalSpec
    gates: Gates
    toolchain: ToolchainSpec
    source: str = "<memory>"
    limitations: tuple[str, ...] = ()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_mapping(cls, data: Any, source: str = "<memory>") -> Spec:
        if not isinstance(data, Mapping):
            raise SpecError(
                f"{source}: expected a mapping at the top level, got {_type_name(data)}"
            )
        missing = [name for name in REQUIRED_SECTIONS if data.get(name) is None]
        if missing:
            raise SpecError(
                f"{source}: missing required section(s) {missing}. The spec is the whole job; a "
                "section litetune has to guess at is a run nobody can reproduce"
            )
        unknown = sorted(set(data) - set(SECTION_NAMES))
        if unknown:
            raise SpecError(f"{source}: unknown top-level section(s) {unknown}")

        limitations: list[str] = []
        spec = cls(
            base_model=BaseModel.read(data["base_model"], limitations),
            dataset=Dataset.read(data["dataset"], limitations),
            train=TrainSpec.read(data["train"], limitations),
            export=ExportSpec.read(data["export"], limitations),
            eval=EvalSpec.read(data["eval"], limitations),
            gates=Gates.read(data.get("gates"), limitations),
            toolchain=ToolchainSpec.read(data["toolchain"], limitations, DEFAULT_ENVS),
            source=source,
            limitations=(),
        )
        return replace(spec, limitations=tuple(limitations))

    @classmethod
    def from_yaml(cls, text: str, source: str = "<memory>") -> Spec:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            logger.exception("could not parse spec %s", source)
            raise SpecError(f"{source}: not valid YAML: {exc}") from exc
        return cls.from_mapping(data, source=source)

    # -- serialisation -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA,
            "base_model": self.base_model.as_dict(),
            "dataset": self.dataset.as_dict(),
            "train": self.train.as_dict(),
            "export": self.export.as_dict(),
            "eval": self.eval.as_dict(),
            "gates": self.gates.as_dict(),
            "toolchain": self.toolchain.as_dict(),
        }

    @property
    def hash(self) -> str:
        """Identity of the whole spec, for the manifest. Not a cache key."""
        return _digest(self.as_dict())

    # -- slicing and hashing ----------------------------------------------

    def section_identity(self, name: str) -> Any:
        """The part of one section that a cache key may see."""
        if name == "dataset":
            return self.dataset.identity()
        if name == "eval":
            return self.eval.identity()
        if name == "toolchain":
            raise SpecError(
                "the toolchain section is in no stage's slice: an environment enters a cache key "
                "through its resolved identity (StageEnv.identity, passed to hash_for), because a "
                "declaration that resolves differently over time is the failure being guarded "
                "against and hashing the declaration would not catch it"
            )
        if name not in SECTION_NAMES:
            raise SpecError(f"unknown spec section {name!r}; known: {list(SECTION_NAMES)}")
        return getattr(self, name).as_dict()

    def slice_for(self, stage: str, sections: Sequence[str] | None = None) -> dict[str, Any]:
        """Only the fields `stage` depends on.

        `sections` lets a stage declare its own dependencies (see
        `runner.Stage.spec_sections`); without it the table above decides.
        Either way a measurement stage asking for `gates` is refused, because a
        threshold in a measurement key means tightening the threshold re-runs
        hours of generation to produce numbers that were already recorded.
        """
        names = tuple(sections) if sections is not None else STAGE_SECTIONS.get(stage)
        if names is None:
            raise SpecError(
                f"stage {stage!r} declares no spec sections and is not in the known table "
                f"{sorted(STAGE_SECTIONS)}. An empty slice would make every spec change a cache "
                "hit, which is the dangerous direction"
            )
        if "gates" in names and stage not in JUDGEMENT_STAGES:
            raise SpecError(
                f"stage {stage!r} asked for the 'gates' section. Thresholds are judgement "
                "parameters, not measurement parameters: a gate in a measurement key means "
                "tightening it re-runs the measurement instead of re-judging what was recorded. "
                f"Only {sorted(JUDGEMENT_STAGES)} may see gates"
            )
        return {name: self.section_identity(name) for name in names}

    def hash_for(
        self,
        stage: str,
        inputs: Mapping[str, str],
        env_identity: str,
        sections: Sequence[str] | None = None,
    ) -> str:
        """The cache key: stage, spec slice, input content hashes, environment.

        `inputs` maps an input name to a *content* hash. A `None` there is
        refused rather than encoded, because a key over an input nobody could
        identify would match again next time and read as a hit.
        """
        if not isinstance(env_identity, str) or not env_identity:
            raise SpecError(
                f"stage {stage!r}: env_identity must be a non-empty string (use "
                f"spec.env_identity_for(...) or {IN_PROCESS!r} for in-process work)"
            )
        resolved: dict[str, str] = {}
        for name, content_hash in inputs.items():
            if not isinstance(content_hash, str) or not content_hash:
                raise SpecError(
                    f"stage {stage!r}: input {name!r} has no content hash ({content_hash!r}). An "
                    "input that cannot be identified by content must not be cached on"
                )
            resolved[str(name)] = content_hash
        return _digest(
            {
                "schema": SPEC_SCHEMA,
                "stage": stage,
                "spec": self.slice_for(stage, sections),
                "inputs": resolved,
                "env": env_identity,
            }
        )

    # -- environments ------------------------------------------------------

    def env_names_for(self, stage: str, declared: Sequence[str] | None = None) -> tuple[str, ...]:
        if declared is not None:
            return tuple(declared)
        names = STAGE_ENVS.get(stage)
        if names is None:
            raise SpecError(
                f"stage {stage!r} declares no environments and is not in the known table "
                f"{sorted(STAGE_ENVS)}"
            )
        return names

    def env_identity_for(self, stage: str, declared: Sequence[str] | None = None) -> str:
        return self.toolchain.identity_for(self.env_names_for(stage, declared))

    def environments(self) -> dict[str, str]:
        """Every declared environment's resolved identity, for the manifest."""
        return self.toolchain.identities()


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:_KEY_CHARS]


def load_spec(path: Path) -> Spec:
    """Read and validate a spec file. Raises `SpecError` naming the field."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("could not read spec %s", path)
        raise
    return Spec.from_yaml(text, source=str(path))
