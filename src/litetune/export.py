"""`convert`: a checkpoint to `.litertlm`, one artifact per candidate recipe.

Two rules shape this module, and both come from measurement rather than taste.

**The quantization recipe is an output.** On a fully fine-tuned
`functiongemma-270m-it` over 640 held-out examples, float scored 0.9094,
`dynamic_wi8_afp32` 0.8906 and `weight_only_wi8_afp32` 0.9141 — the same bit
width, a file 0.04% larger, and 0.024 more exact match. `dynamic_wi8_afp32` is
the toolchain's own default, and its own docstring says "the model quality may
suffer due to the on-the-fly quantization. If quality is a concern, consider
using weight-only quantization." So this stage sweeps candidates and reports
what each one cost in bytes; nothing here picks a winner, because the axis that
separates those two artifacts is accuracy and that is `verify`'s to measure.

**A produced file is not a successful export.** Exit code zero plus an output
file is the failure mode of this whole toolchain: the export succeeds, the
artifact loads, and the model is wrong on first use. Every result this module
returns therefore carries `verified: False` and says what has not been
established, so a caller physically cannot report success on export alone.
Liveness lives in `litetune.liveness`, quality in `litetune.verify`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from litetune import envs, models
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.recipes import (
    DEFINED_RECIPES,
    RecipeDefinitionError,
    definition_of,
    source_of,
    toolchain_argument,
)

logger = logging.getLogger(__name__)

EXPORT_SCHEMA = "litetune.export/1"

# The toolchain's default, and the one that costs accuracy. Named so that a
# sweep containing only this recipe can be flagged as such rather than reading
# as a deliberate choice.
EXTERNALIZE_FLAG = "--externalize_embedder"
"""The flag litetune passes unless the caller says otherwise.

On by default because every measurement in the README was produced with it,
and because two exports that differ in it cannot be compared on size at all:
the same 270M model is 286 MB without and 457 MB with.
"""

TOOLCHAIN_DEFAULT_RECIPE = "dynamic_wi8_afp32"

# The flag as a plan carries it when the template is to be composed at export.
COMPOSED_TEMPLATE_FLAG = f"{models.TEMPLATE_FLAG}={models.COMPOSED_TEMPLATE}"
# Beside the recipes' directories, named for its content: two converts into one
# directory each get their own file, and one composed the same way is the same
# file. The leading underscore is one `_RECIPE_NAME_RE` cannot produce, so no
# recipe's directory can take the name.
COMPOSED_TEMPLATE_PREFIX = "_litetune-chat-template-"
CHAT_TEMPLATE_CHECK = "chat template"
# The user's text in the probe `_COMPOSE_SCRIPT` renders: one token, so that a
# template that trims or splits text has nothing to change in it.
PROBE_TEXT = "litetune-probe-7d1c"

KNOWN_RECIPES = (
    "dynamic_wi8_afp32",
    "weight_only_wi8_afp32",
    "dynamic_wi4_afp32",
    "weight_only_wi4_afp32",
    # Block-wise 4-bit: both measured on two tuned checkpoints over 600 rows
    # (MEASUREMENTS.md, "What four bits cost"), so neither is a name litetune
    # merely passes through. Leaving them out told a caller their recipe was
    # "not among the ones litetune has measured" while the README tabulated
    # what it cost.
    "dynamic_wi4b32_afp32",
    "dynamic_wi4b32_emb8_afp32",
)

# The two the README's table was produced with: same bit width, opposite ends
# of the accuracy result. A caller that wants a defensible minimum sweep wants
# these; it is offered, never applied by default -- see `ExportRequest.recipes`.
MEASURED_RECIPES = ("dynamic_wi8_afp32", "weight_only_wi8_afp32")

# What the GPU text executor computes activations in, unless the bundle says
# otherwise. Measured 2026-09-05 on one Snapdragon Galaxy S24 (SC-51E) with the
# tuned FunctionGemma bundle, native tool path, 20 rows, greedy:
#
#     activation            <pad> floods   tool name   exact   ms/prompt
#     (unset -> F16), GPU       14/20         3/20      2/20     16763
#     fp32, GPU                  0/20        20/20     15/20      1375
#     fp32_fp16, GPU             0/20        20/20     15/20      1406
#     fp32, CPU (repacked file)  0/20        20/20     14/20      2477
#
# The CPU row is the repacked file; the key is read only by the GPU executor.
# CPU on the unrepacked file scored the same 14/20 in every earlier run.
#
# The engine reports success either way: it loads, every call returns, no
# exception. Only the content betrays it. Google's own Mobile Actions bundle
# does the same (20/20 `<pad>` on GPU) and the Gallery pins it to
# `"accelerators": "cpu"`. LiteRT-LM's `engine_settings.cc` names the default
# -- "Text executor defaults to F16" -- and the override is a string in the
# prefill/decode section's metadata. Google's bundle does not carry it, and
# `export_hf` can only write `fp32_fp16` (via --experimental_use_mixed_precision).
#
# `fp32` rather than `fp32_fp16`: the two gave the same exact-match count
# (15/20) and latency within 2%, and `fp32` changes one metadata string while
# `--experimental_use_mixed_precision` also runs a graph pass. The model
# sections are not re-exported by the repack; the builder regenerates the
# bundle uuid and timestamp, and the size held because section padding
# absorbed the key.
GPU_ACTIVATION = "fp32"
GPU_ACTIVATION_KEY = "prefer_activation_type"


class GpuActivationState(str, Enum):
    """What a bundle's `prefer_activation_type` means for the GPU backend.

    Three states, one place. The value itself stays a free string because it
    is read back out of the bundle; what is closed is how litetune reads it.
    """

    UNSET = "unset"  # no declaration: the runtime's F16 default on the GPU
    SET = "set"  # carries GPU_ACTIVATION, written by litetune or found upstream
    DECLARED_UPSTREAM = "declared_upstream"  # carries something else, left as found

    @classmethod
    def of(cls, value: str | None) -> GpuActivationState:
        if value is None:
            return cls.UNSET
        if value == GPU_ACTIVATION:
            return cls.SET
        return cls.DECLARED_UPSTREAM


def describe_gpu_activation(value: str | None) -> str:
    """The one sentence both the check detail and the console line print."""
    state = GpuActivationState.of(value)
    if state is GpuActivationState.UNSET:
        return "GPU activations not set: an app that passes none gets F16 on the GPU"
    if state is GpuActivationState.SET:
        return f"GPU activations {value}"
    return f"GPU activations {value} (declared upstream, not {GPU_ACTIVATION})"


# For the whole repack (unpack, patch, pack, unpack again): a ceiling for a
# stalled tool, not a budget. A real repack of a 455 MB bundle finishes in
# seconds. Charged against the recipe's export timeout at the call site so a
# `--timeout-s` bound holds for export and repack together.
REPACK_TIMEOUT_S = 300

# Measured: a 285,577,392-byte artifact in 122 s on CPU for a 270M checkpoint.
# The ceiling is an order of magnitude above that so a larger checkpoint is not
# cut off mid-write, and low enough that a hung export is recorded as a
# non-result the same day rather than occupying a runner overnight.
DEFAULT_TIMEOUT_S = 1800

# `pip show` is metadata collection, not model work; if it has not answered in
# a minute the environment is broken in a way the export itself will surface.
TOOLCHAIN_TIMEOUT_S = 60

# Export must not require an accelerator: it is a pure format conversion, so
# it has to run on a runner that has no GPU. Unlike training and the float
# reference, which run wherever `envs.resolve_device` finds a device, export's
# own device does not vary with the machine -- every export in this module
# reports `"backend": "cpu"` because every export is given this environment.
# Hiding the devices makes that true by construction rather than by trusting
# the exporter's default device selection -- which is also why the 122 s
# above is a CPU number: `export_recipe` has exactly one call site for the
# export subprocess, this dict is passed there unconditionally, and
# `--backend=gpu` never appears in `argv`, so nothing this module has ever
# exported ran anywhere else.
CPU_ONLY_ENV = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}

NO_ARTIFACT = "no artifact was produced, so there is nothing to verify"

NOT_VERIFIED = (
    "export produced this artifact and nothing has been run against it: liveness is not "
    "established and quality is unmeasured. A file of the right size that loads without "
    "error is the documented shape of a bad conversion, not evidence against one."
)

# Filesystem timestamp granularity is a whole second on some filesystems, so a
# file written immediately after the clock reading can carry an mtime just
# behind it.
_MTIME_SLACK_S = 2.0

# Recipe names reach the filesystem (one output directory each) and the command
# line, so they are constrained to a shape that cannot escape either.
_RECIPE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

_STDOUT_TAIL = 4000
_DETAIL_TAIL = 400


class NoRecipesRequested(ValueError):
    """Raised when a conversion would fall back to the toolchain's default recipe."""


def _tail(text: str, limit: int = _DETAIL_TAIL) -> str:
    stripped = (text or "").strip()
    return stripped[-limit:] if stripped else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_only_environ() -> dict[str, str]:
    """The overrides, not a whole environment.

    `StageEnv.run` builds the base -- the host's minus the variables that let
    it reach past the pins. Returning `dict(os.environ) | ...` from here, as
    this once did, handed `PYTHONPATH` straight back.
    """
    return dict(CPU_ONLY_ENV)


def _produced_files(directory: Path, since: float) -> list[Path]:
    """Files this run wrote, by mtime rather than by existence.

    "The file is there" is not the same claim as "this export wrote it". A
    directory reused between runs still holds the previous artifact, and an
    export that exits zero having written nothing would otherwise inherit it
    and read as a success. mtime also survives the overwrite case, which a
    before/after set difference does not.
    """
    if not directory.is_dir():
        return []
    cutoff = since - _MTIME_SLACK_S
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.stat().st_mtime >= cutoff)


# ---------------------------------------------------------------------------
# Toolchain provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Toolchain:
    """What the environment's pins resolved to, package by package.

    `declared` is the definition and `resolved` is what is installed. They are
    different facts: the definition pins direct requirements only, so the
    transitive set -- which is where a nightly's behaviour actually changed
    between 2026-08-26 and 2026-08-30 -- is recorded here or nowhere.
    """

    declared: tuple[str, ...]
    resolved: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    unresolved_reason: str | None = None

    @classmethod
    def unknown(cls, reason: str, declared: Sequence[str] = ()) -> Toolchain:
        return cls(declared=tuple(declared), unresolved_reason=reason)

    @property
    def available(self) -> bool:
        return self.unresolved_reason is None and bool(self.resolved)

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "available": self.available,
            "declared": envs.shown_requirements(self.declared),
            "resolved": dict(self.resolved),
        }
        if self.missing:
            record["missing"] = list(self.missing)
        if self.unresolved_reason:
            record["reason"] = self.unresolved_reason
        return record


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(requirements: Sequence[str]) -> tuple[str, ...]:
    """Distribution names from pinned requirement strings."""
    names: list[str] = []
    for requirement in requirements:
        if requirement.startswith(("http://", "https://", "file://")):
            # A direct URL carries no reliable distribution name; `pip show`
            # would report nothing for it, so it is left out rather than
            # producing a phantom "missing" entry.
            continue
        name = re.split(r"[=<>!~\[;]", requirement, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return tuple(names)


def parse_pip_show(text: str) -> dict[str, str]:
    """`Name:`/`Version:` pairs out of `pip show`'s record blocks."""
    versions: dict[str, str] = {}
    name: str | None = None
    for line in text.splitlines():
        if line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:") and name:
            versions[name] = line.split(":", 1)[1].strip()
            name = None
    return versions


def parse_pip_freeze(text: str) -> dict[str, str]:
    """`name==version` lines out of `pip freeze`.

    Non-version lines -- editable installs, direct URLs, `-e` entries -- are
    kept verbatim under their name so the record stays complete rather than
    silently dropping the packages hardest to reproduce.
    """
    versions: dict[str, str] = {}
    for raw in text.splitlines():
        # A direct URL is kept, and one can carry the credentials of the index
        # it was fetched from; the record is a manifest. Its user information
        # and query go; its fragment stays, since there pip writes the
        # archive's `#sha256=`, which is the provenance this record is for.
        line = envs.redact_text(raw.strip(), keep_fragment=True)
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, _, version = line.partition("==")
            versions[name.strip()] = version.strip()
        elif line.startswith("-e ") or " @ " in line:
            name = line.removeprefix("-e ").split("@", 1)[0].strip().rstrip("/").split("/")[-1]
            versions[name or line] = line
    return versions


def resolve_toolchain(env: envs.StageEnv, timeout_s: int = TOOLCHAIN_TIMEOUT_S) -> Toolchain:
    """`pip freeze` inside the stage environment.

    Freeze, not `pip show`: show reports only the packages the environment
    *declares*, and those are already pinned in the definition. The change that
    broke packaging with `AttributeError: pad_token` between 2026-08-26 and
    2026-08-30 happened in the transitive closure, which show never reports. A
    record that cannot contain the thing it exists to catch is decoration.

    Provenance, not a check: this is deliberately never added to the result's
    `CheckSet`. A pip that will not answer says nothing about the artifact, and
    an UNCHECKED item there would turn a perfectly good export into "could not
    check" -- exactly the collapse `litetune.checks` exists to prevent, run in
    reverse.
    """
    declared = tuple(env.requirements)
    names = requirement_names(declared)
    if not names:
        return Toolchain.unknown("the environment declares no named requirements", declared)
    try:
        proc = env.run(["pip", "freeze", "--all"], timeout=timeout_s)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.exception("could not read toolchain versions from environment %r", env.name)
        return Toolchain.unknown(f"{type(exc).__name__}: {exc}", declared)

    resolved = parse_pip_freeze(proc.stdout or "")
    if not resolved:
        return Toolchain.unknown(
            f"pip freeze exited {proc.returncode} and reported no packages: "
            f"{_tail(envs.redact_text(proc.stderr or '')) or 'no stderr'}",
            declared,
        )
    seen = {_canonical(n) for n in resolved}
    missing = tuple(n for n in names if _canonical(n) not in seen)
    return Toolchain(declared=declared, resolved=resolved, missing=missing)


def installed_version(toolchain: Toolchain, distribution: str) -> str | None:
    """What is actually installed, by canonical distribution name."""
    wanted = _canonical(distribution)
    for name, version in toolchain.resolved.items():
        if _canonical(name) == wanted:
            return version
    return None


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportRequest:
    """One conversion: one checkpoint, several candidate recipes.

    `recipes` has no default and an empty list is refused. Accepting a default
    here would reintroduce the exact configuration this stage exists to
    measure: `dynamic_wi8_afp32` is what the toolchain picks when nobody
    chooses, and it is the recipe that cost 0.024 exact match.
    """

    model: str
    output_dir: Path
    # Normalised to a de-duplicated tuple in __post_init__ so a caller may pass
    # whatever argparse handed it.
    recipes: Sequence[str]
    # Tri-state on purpose. `True`/`False` are the caller's answer; `None` means
    # they did not say and litetune's default applies. Collapsing that to a bool
    # in argparse made a defaulted flag indistinguishable from a typed one, so
    # `report.json` claimed the caller supplied a flag they had never heard of,
    # and the sentence explaining why Gemma 4 cannot export without it was never
    # printed. The default itself lives here and nowhere else: `cli` and `spec`
    # both used to carry their own, and the same job written two ways produced
    # artifacts differing by 171 MB that could not be compared on size at all.
    # What the per-family export rules key on, when `model` cannot serve.
    # After training, `model` is a directory, and a directory carries no
    # identity: `plan_export` matches on the model's *name*, so a local path
    # matches nothing and the family's required flags are silently not applied.
    # That is how a FunctionGemma checkpoint becomes a bundle typed
    # `generic_model`, with no tool-call channel and a template the runtime
    # cannot execute -- while the export succeeds and every check stays green.
    #
    # `config.json` cannot stand in for this: FunctionGemma and plain Gemma 3
    # both declare `model_type: gemma3_text` and need different overrides.
    base_model: str | None = None
    externalize_embedder: bool | None = None
    # Additional exporter flags, verbatim. Whatever `litetune.models` says this
    # family requires is merged in here by `__post_init__`, so no code path --
    # CLI, library or a future composed run -- can build an export that is
    # missing a flag the family cannot export without, or carrying one that is
    # refused.
    extra_flags: Sequence[str] = ()
    timeout_s: int = DEFAULT_TIMEOUT_S
    env: envs.StageEnv = envs.EXPORT
    auto_provision: bool = True
    hash_artifacts: bool = True

    def __post_init__(self) -> None:
        cleaned: list[str] = []
        for recipe in self.recipes:
            name = recipe.strip()
            if not _RECIPE_NAME_RE.fullmatch(name):
                raise ValueError(
                    f"{name!r} is not a usable recipe name: it names an output directory and a "
                    "command-line value, so it must be alphanumeric with '_', '-' or '.'"
                )
            if name not in cleaned:
                cleaned.append(name)
        if not cleaned:
            raise NoRecipesRequested(
                "no quantization recipes were requested. This stage does not fall back to a "
                f"default: {TOOLCHAIN_DEFAULT_RECIPE} is the toolchain's own choice and cost "
                f"0.024 exact match against {MEASURED_RECIPES[1]} on 640 held-out examples. "
                f"Name the candidates to sweep, e.g. {list(MEASURED_RECIPES)}."
            )
        object.__setattr__(self, "recipes", tuple(cleaned))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        requested = list(self.extra_flags)
        # `convert`'s own option is the caller's setting, so it enters the table
        # the plan reads like any `--flag=`: an opt-out a family cannot export
        # without is then refused by the plan, not overridden by its rule.
        if self.externalize_embedder is True:
            requested.append(EXTERNALIZE_FLAG)
        elif self.externalize_embedder is False:
            requested.append(f"--no{EXTERNALIZE_FLAG.lstrip('-')}")
        # Raises `models.FlagRefused` for a flag that must never be passed for
        # this family, and adds the ones it cannot export without. A plan whose
        # required flags could not be resolved is *kept*, not raised on: it is
        # `could not check`, and `run_export` refuses to attempt the sweep.
        # A checkpoint that records what it came from is consulted even when
        # the caller names something: `--base-model` beating the sidecar in
        # silence is how a stale copy-pasted command line exports the wrong
        # family with a passed check vouching for it.
        # `hint_for` reads the sidecar itself when handed a directory, so the
        # identity is just "the flag, or the path". The recorded value is read
        # here only to notice that the two disagree.
        recorded = models.recorded_identity(self.model)
        identity = self.base_model or self.model
        conflict = (
            None
            if not (self.base_model and recorded) or models.same_family(self.base_model, recorded)
            else (
                f"--base-model says {self.base_model!r} but the checkpoint records "
                f"{recorded!r}, and they are different families. The recorded value came "
                "from the run that produced these weights; the flag came from this command "
                "line. litetune will not pick one"
            )
        )
        object.__setattr__(self, "_identity_conflict", conflict)
        plan = models.plan_export(identity, requested, self.recipes)
        flags = list(plan.flags)
        # Read off the plan's table, as the exporter will read it: a caller's
        # `--noexternalize_embedder` or `--externalize-embedder=False` is theirs,
        # and the default is added only where nothing set it.
        setting = plan.setting(EXTERNALIZE_FLAG)
        if setting is not None and setting.token in plan.added:
            externalized, source = True, "family rule"
        elif setting is not None:
            externalized = bool(setting.value)
            source = "caller" if externalized else "caller opted out"
        else:
            flags.append(EXTERNALIZE_FLAG)
            externalized, source = True, "litetune default"
        object.__setattr__(self, "extra_flags", tuple(flags))
        object.__setattr__(self, "_externalize_source", source)
        object.__setattr__(self, "_externalized", externalized)
        object.__setattr__(self, "_plan", plan)

    @property
    def identity_conflict(self) -> str | None:
        """Why `--base-model` and the checkpoint's own record cannot both be right."""
        return self._identity_conflict  # type: ignore[attr-defined]

    @property
    def externalize_source(self) -> str:
        """Who decided `--externalize_embedder`: the caller, the family's rule, or this default."""
        return self._externalize_source  # type: ignore[attr-defined]

    @property
    def plan(self) -> models.ExportPlan:
        """What `litetune.models` said about this model, including added flags."""
        return self._plan  # type: ignore[attr-defined]

    def dir_for(self, recipe: str) -> Path:
        """One output directory per recipe: the sweep's artifacts must not collide."""
        return self.output_dir / recipe

    @property
    def composes_chat_template(self) -> bool:
        """Whether the plan asks for `models.COMPOSED_TEMPLATE`, which `run_export` writes."""
        return COMPOSED_TEMPLATE_FLAG in self.extra_flags

    def composed_template_path(self, composed_sha256: str) -> Path:
        """Where a composed template with this digest is kept.

        Absolute, so that the exporter's argv and the report name the file
        however they are read later. Not for the exporter's sake: `StageEnv.run`
        passes it no working directory (`envs.py` sets no `cwd`) and nothing in
        litert-torch's `export_hf` or the builder calls `chdir`, so a relative
        path would resolve the same way today.
        """
        name = f"{COMPOSED_TEMPLATE_PREFIX}{composed_sha256[:12]}.jinja"
        return (self.output_dir / name).absolute()

    def argv(self, recipe: str, chat_template: Path | None = None) -> list[str]:
        """The exporter's command line, with the composed template in place of the marker.

        A plan carrying `models.COMPOSED_TEMPLATE` needs `chat_template`, the
        file `compose_chat_template` wrote, and without it this raises: the
        marker is a description, and handed to the exporter it would be looked
        up as a repository. So does a marker in any other shape than the one
        the plan writes, which this would otherwise pass through.
        """
        flags = list(self.extra_flags)
        if COMPOSED_TEMPLATE_FLAG in flags:
            if chat_template is None:
                raise ValueError(
                    f"{self.model}: the plan asks for the checkpoint's own chat template behind "
                    "litetune's text-parts block, and no composed template was given"
                )
            flags = [
                f"{models.TEMPLATE_FLAG}={chat_template}" if f == COMPOSED_TEMPLATE_FLAG else f
                for f in flags
            ]
        leftover = [f for f in flags if models.COMPOSED_TEMPLATE in f]
        if leftover:
            raise ValueError(f"{self.model}: {leftover} would hand the exporter the marker")
        # One argument per keyword, none of them litetune's own three: the plan
        # is built so, and this is where a change that broke it would show.
        if len(models.parse_exporter_flags(self.model, flags)) != len(flags):
            raise ValueError(f"{self.model}: {flags} set a keyword more than once")
        return [
            "litert-torch",
            "export_hf",
            f"--model={self.model}",
            f"--output_dir={self.dir_for(recipe)}",
            f"--quantization_recipe={toolchain_argument(recipe)}",
            *flags,
        ]

    @property
    def unknown_recipes(self) -> tuple[str, ...]:
        """Requested recipes litetune neither defines nor has measured. Not an error."""
        return tuple(r for r in self.recipes if r not in KNOWN_RECIPES and r not in DEFINED_RECIPES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_model": self.base_model,
            "output_dir": str(self.output_dir),
            "recipes": list(self.recipes),
            "externalize_embedder": self._externalized,  # type: ignore[attr-defined]
            "externalize_embedder_source": self.externalize_source,
            "flags": list(self.extra_flags),
            "model_rules": self.plan.as_dict(),
            "timeout_s": self.timeout_s,
            "environment": {
                "name": self.env.name,
                "identity": self.env.identity,
                "path": str(self.env.path),
                "requirements": envs.shown_requirements(self.env.requirements),
                # What this platform needs: Debian names are recorded on Linux only.
                "system_requirements": list(self.env.system_requirements_on()),
            },
        }


# ---------------------------------------------------------------------------
# One recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeExport:
    """What one recipe produced, and what that does and does not establish."""

    recipe: str
    check: Check
    argv: tuple[str, ...] = ()
    # The rules of a recipe litetune defines, exactly as the toolchain received
    # them; None for a toolchain preset, whose rules live in the toolchain.
    recipe_definition: list[dict[str, Any]] | None = None
    artifact: Path | None = None
    # Everything the recipe produced besides the `.litertlm` itself. Measured,
    # this is normally empty: `--externalize_embedder` writes the embedding as
    # its own section *inside* the bundle (285,577,392 bytes without the flag,
    # 455,759,152 with it, one file either way), which is the opposite of what
    # this field was originally documented as existing for.
    #
    # It stays because the unit that ships is the directory: a toolchain that
    # begins emitting a sidecar would otherwise make every recorded size quietly
    # wrong, and here it becomes a named entry instead.
    companions: tuple[str, ...] = ()
    artifact_bytes: int | None = None
    shipped_bytes: int | None = None
    sha256: str | None = None
    seconds: float | None = None
    returncode: int | None = None
    # What the bundle tells the GPU text executor to compute activations in --
    # the bundle's own value, whether litetune wrote it or found it -- or None
    # when it carries no declaration and the toolchain's F16 default stands.
    # `gpu_activation_note` says why that is, when there is something to say.
    # See GPU_ACTIVATION and GpuActivationState.
    gpu_activation: str | None = None
    gpu_activation_note: str | None = None
    # Full, not a tail. A nightly-specific failure has to be diagnosable months
    # later from the record alone, and the interesting line is as often the
    # first traceback frame as the last.
    stderr: str = ""
    stdout_tail: str = ""

    @property
    def ok(self) -> bool:
        return self.check.outcome is Outcome.PASSED

    @property
    def attempted(self) -> bool:
        return self.returncode is not None

    @property
    def gpu_state(self) -> GpuActivationState:
        return GpuActivationState.of(self.gpu_activation)

    @property
    def recipe_source(self) -> str:
        """`litetune` for a recipe litetune defines, `toolchain` for a name passed as given."""
        return source_of(self.recipe)

    @property
    def verified(self) -> bool:
        """Always False. Not a field, so no code path can set it True.

        Export can establish that a file was written. Establishing that the
        model in it is right takes held-out data, and this module does not have
        any.
        """
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe,
            "recipe_source": self.recipe_source,
            "recipe_definition": self.recipe_definition,
            "verified": False,
            "unverified_reason": NOT_VERIFIED,
            "outcome": self.check.outcome.value,
            "check": self.check.as_dict(),
            "argv": list(self.argv),
            "artifact": str(self.artifact) if self.artifact else None,
            "companions": list(self.companions),
            "artifact_bytes": self.artifact_bytes,
            "shipped_bytes": self.shipped_bytes,
            "sha256": self.sha256,
            "gpu_activation": self.gpu_activation,
            "gpu_activation_state": self.gpu_state.value,
            "gpu_activation_note": self.gpu_activation_note,
            "seconds": round(self.seconds, 3) if self.seconds is not None else None,
            "returncode": self.returncode,
            "stderr": self.stderr,
            "stdout_tail": self.stdout_tail,
        }


def check_name(recipe: str) -> str:
    return f"export {recipe}"


def repair_vocab_file(model_dir: Path) -> tuple[bool, str | None]:
    """Point `vocab_file` at where the tokenizer actually is. Returns (ok, what happened).

    `ok` is False only when a repair was needed and could not be made -- which
    matters, because the export that follows will then die with the very error
    this exists to prevent. Returning one string for both "here is what I did"
    and "here is what went wrong" left the caller unable to tell them apart.

    `export_lib.export_tokenizer` reads `tokenizer.vocab_file` and opens it
    verbatim -- no resolution against the model directory. That field comes from
    `tokenizer_config.json`, and `tune` writes an absolute path into it so the
    exporter's SentencePiece branch fires at all: transformers 5.x stopped
    writing `tokenizer.model` and the tokenizer classes stopped exposing
    `vocab_file`, so without it every bundle silently gets an HF tokenizer
    section and loses FST-constrained decoding.

    An absolute path is correct exactly where it was written and wrong
    everywhere else. Train on one machine and convert on another -- or in a
    container with different mounts, or simply move the directory -- and the
    export dies with `FileNotFoundError: /tmp/merged/tokenizer.model`, naming a
    path that never existed here. Reproduced by exporting a checkpoint built on
    a Linux worker from a laptop.

    So the path is repaired at use, not trusted from write: where the checkpoint
    *is* is knowable here and was not knowable there.
    """
    config = model_dir / "tokenizer_config.json"
    beside = model_dir / "tokenizer.model"
    if not config.exists():
        return True, None
    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"could not read {config.name}: {exc}"
    if not isinstance(data, dict):
        return True, None

    declared = data.get("vocab_file")
    if declared and Path(declared).exists():
        return True, None
    if not beside.exists():
        # Nothing to point at. Not an error: a BPE tokenizer has no such file,
        # and a stale field with no file is left for the exporter to skip.
        return True, None

    data["vocab_file"] = str(beside.resolve())
    try:
        config.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        return False, f"could not rewrite {config.name}: {exc}"
    return True, (
        f"tokenizer_config.json in the checkpoint named a vocab_file that is not on this "
        f"machine ({declared!r}); litetune rewrote it to point at {beside}. This modifies "
        "the checkpoint directory, and the exporter reads that field verbatim"
    )


# Runs inside envs.EXPORT, beside the exporter's own transformers, and reads the
# template the way the exporter reads it when it is given no override:
# litert-torch's `export_lib.py:389` loads
# `transformers.AutoTokenizer.from_pretrained(model_path)`, `:390-402` falls
# back to a `chat_template.json` when that tokenizer has no template, and with
# its default `use_jinja_template`, `litert_lm_builder.py:432-433` embeds the
# tokenizer's `chat_template` (litert-torch-nightly 0.10.0.dev20260926). It
# renders the probe through the tokenizer's own `apply_chat_template`, which
# hands the template the messages as given (transformers 5.16.1,
# `chat_template_utils.py` `render_jinja_template`). It imports nothing from
# litetune, which the export environment does not install; the block arrives in
# the spec.
_COMPOSE_SCRIPT = r'''
"""Write litetune's text-parts block in front of a checkpoint's own chat template.

Reads JSON {model, block, out, probe} from argv[1]. Writes JSON to argv[2]:
  {"transformers": <str>, "found": <str>, "source": <str>, "fallback": <dict|null>,
   "names": <list|null>, "type": <str|null>, "sha256": <str|null>,
   "characters": <int|null>, "renders": <dict|null>}
`found` is "text" when `out` now holds the block followed by the template, and
otherwise says why it does not: "empty", "blank" (only whitespace),
"carriage_return", "unnamed" (named templates, none of them "default"),
"not_text", or "unreachable" (the chat_template.json lookup itself failed).
`source` is where the template came from, "tokenizer" or "chat_template.json";
`fallback` says what the chat_template.json lookup found, when the tokenizer had
no template. `type` is the template's Python type for
"empty" and "not_text". `sha256` and `characters` describe the template alone,
as UTF-8. For "text", `renders` holds one user message whose text is `probe`,
rendered three ways -- "string": the template, the content a string;
"composed": block and template, the content a list of one text part;
"parts_alone": the template, the content that list -- each {"text": ...} or
{"error": ...}, with "environment": true for an error of the environment's.
"""
import errno
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import transformers


def model_path_type(path_str):
    """`get_model_path_type`, as litert-torch writes it (export_hf/core/utils.py)."""
    if os.path.exists(path_str):
        return "local"
    if path_str.startswith((".", "/", "~")) or "\\" in path_str:
        return "local_not_found"
    if re.match(r"^(?:[\w\-\.]+\/)?[\w\-\.]+$", path_str):
        return "repo_id"
    return "unknown"


def lookup(outcome, where, exc=None):
    error = None if exc is None else f"{type(exc).__name__}: {exc}"
    return {"path": where, "outcome": outcome, "error": error}


def chat_template_json(model):
    """The exporter's fallback (export_lib.py:390-402): the Hub file for a repo
    id, `<model>/chat_template.json` for anything else, its "chat_template" key.

    The exporter goes on with no template whatever fails there (:403-404);
    this tells the failures apart. "absent": no such file -- a Hub 404
    (`RemoteEntryNotFoundError`) or a missing local file. "unreachable": the
    lookup itself failed -- `hf_hub_download` raises `LocalEntryNotFoundError`
    when it cannot reach the Hub, and any other Hub or file-system error is the
    environment's too. "unusable": a file the exporter could not use. "found"
    or "no_key": what it holds."""
    if model_path_type(model) == "repo_id":
        import huggingface_hub
        from huggingface_hub.errors import RemoteEntryNotFoundError

        try:
            where = huggingface_hub.hf_hub_download(model, filename="chat_template.json")
        except RemoteEntryNotFoundError as exc:
            return None, lookup("absent", None, exc)
        except Exception as exc:  # the Hub's or the network's, not the checkpoint's
            return None, lookup("unreachable", None, exc)
    else:
        where = os.path.join(model, "chat_template.json")
    if os.path.isdir(where):
        # Asked, not inferred from the error: on Windows opening a directory
        # raises an OSError other than IsADirectoryError -- the first
        # windows-latest run read it as the machine's, "unreachable" -- and a
        # directory there is the checkpoint's, whatever the platform calls it.
        exc = IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), where)
        return None, lookup("unusable", where, exc)
    try:
        with open(where, "rt") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        return None, lookup("absent", where, exc)
    except (IsADirectoryError, NotADirectoryError, UnicodeDecodeError) as exc:
        # What the checkpoint holds at that path, not what the machine allows.
        return None, lookup("unusable", where, exc)
    except OSError as exc:
        return None, lookup("unreachable", where, exc)
    try:
        data = json.loads(text)
        found = "chat_template" in data
        template = data["chat_template"] if found else None
    except Exception as exc:  # what the exporter catches and goes on without
        return None, lookup("unusable", where, exc)
    return template, lookup("found" if found else "no_key", where)


def render(tokenizer, template, content):
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
    except ImportError as exc:
        # transformers raises this for a jinja2 that is missing or older than
        # 3.1.0 (chat_template_utils.py:426-428, :473-476): the environment's.
        return {"error": f"{type(exc).__name__}: {exc}", "environment": True}
    except Exception as exc:  # the template's own failure is a finding, not the script's
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"text": text}


def report(found, **fields):
    """This script's whole result. The exit status says only whether the script
    itself ran; every outcome the parent tells apart is in here."""
    result = {
        "transformers": transformers.__version__,
        "found": found,
        "source": "tokenizer",
        "fallback": None,
        "names": None,
        "type": None,
        "sha256": None,
        "characters": None,
        "renders": None,
    }
    result.update(fields)
    return result


def main(spec):
    model = spec["model"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    template = getattr(tokenizer, "chat_template", None)
    found = {"source": "tokenizer"}
    if not template:
        from_json, fallback = chat_template_json(model)
        found["fallback"] = fallback
        if fallback["outcome"] == "unreachable":
            return report("unreachable", **found)
        if fallback["outcome"] == "found":
            template, found["source"] = from_json, "chat_template.json"
    if isinstance(template, dict) and template:
        # transformers keeps several named templates as a dict and, given no
        # tools, applies the one named "default" (`get_chat_template`).
        found["names"] = sorted(str(name) for name in template)
        if "default" not in template:
            return report("unnamed", **found)
        template = template["default"]
    if not template:
        return report("empty", type=type(template).__name__, **found)
    if not isinstance(template, str):
        return report("not_text", type=type(template).__name__, **found)
    if not template.strip():
        return report("blank", characters=len(template), **found)
    if "\r" in template:
        return report("carriage_return", **found)
    composed = spec["block"] + template
    # Bytes, not text: a text-mode write turns each newline into the platform's.
    Path(spec["out"]).write_bytes(composed.encode("utf-8"))
    parts = [{"type": "text", "text": spec["probe"]}]
    return report(
        "text",
        sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
        characters=len(template),
        renders={
            "string": render(tokenizer, template, spec["probe"]),
            "composed": render(tokenizer, composed, parts),
            "parts_alone": render(tokenizer, template, parts),
        },
        **found,
    )


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)), encoding="utf-8")
'''

# What `_COMPOSE_SCRIPT` may report as `found`; anything else is a script this
# litetune does not know how to read.
_TEMPLATE_FINDINGS = (
    "text",
    "empty",
    "blank",
    "carriage_return",
    "unnamed",
    "not_text",
    "unreachable",
)

# Without a template the exporter writes none: transformers leaves a tokenizer's
# template None (`tokenization_utils_base.py:1078`), `package_model` passes that
# on (`litert_lm_builder.py:433`) and `build_llm_metadata` writes a template only
# when it has one (`:314-318`). LiteRT-LM v0.18.0 then makes its own default for
# the model type (`engine_settings.cc:480-484`), which for qwen3, qwen2p5 and
# generic_model prints each message's text between its `prompt_templates`
# prefix and suffix (`model_type_utils.cc:359-384`) -- empty here, since
# `prompt_templates` is filled only from a template too (`:319-330`).
_NO_TEMPLATE_LEFT_ALONE = (
    "Left alone, the exporter would embed no template, and LiteRT-LM 0.18.0 would put its own "
    "default for the model type in its place, which prints the message text without the "
    "checkpoint's role markers"
)


def _fallback_said(fallback: Any) -> str:
    """What the `chat_template.json` lookup found, as a clause."""
    if not isinstance(fallback, dict):
        return ""
    where, outcome, error = fallback.get("path"), fallback.get("outcome"), fallback.get("error")
    if outcome == "absent" and where is None:
        return ", and its repository has no chat_template.json"
    if outcome == "absent":
        return f", and there is no {where}"
    if outcome == "no_key":
        return f", and {where} has no chat_template key"
    if outcome == "unusable":
        return f", and {where} could not be used ({error})"
    if outcome == "found":
        return f", and the chat_template in {where} is null"
    return ""


def _whose(report: dict[str, Any]) -> str:
    """Where the template the script judged came from, as the subject of a sentence."""
    if report.get("source") == "chat_template.json":
        fallback = report.get("fallback")
        where = fallback.get("path") if isinstance(fallback, dict) else None
        return f"the chat_template in {where or 'chat_template.json'}"
    return "the tokenizer's chat template"


def _template_refusal(model: str, report: dict[str, Any]) -> str | None:
    """Why the checkpoint gives the block no template to go in front of.

    None for "text", "unreachable", and any finding not named here: those are
    the environment's, or a script this litetune cannot read, which is `could
    not check`, never a fact about the checkpoint.
    """
    found, names, whose = report.get("found"), report.get("names"), _whose(report)
    from_tokenizer = report.get("source") != "chat_template.json"
    fallback = _fallback_said(report.get("fallback")) if from_tokenizer else ""
    if found == "unnamed":
        return (
            f"{model}: {whose} holds {len(names or ())} named templates "
            f"({', '.join(map(str, names or ()))}) and none named 'default', the one "
            "transformers applies when it is given no tools"
        )
    if found == "empty" and names:
        return f"{model}: {whose} named 'default' is empty"
    if found == "empty" and report.get("type") == "NoneType":
        fallback = _fallback_said(report.get("fallback"))
        return (
            f"{model}: the tokenizer carries no chat template{fallback}. {_NO_TEMPLATE_LEFT_ALONE}"
        )
    if found == "empty":
        return f"{model}: {whose} is empty (a {report.get('type')}){fallback}"
    if found == "blank":
        return (
            f"{model}: {whose} is {report.get('characters')} characters of whitespace, which would "
            "render as itself for every conversation"
        )
    if found == "carriage_return":
        return (
            f"{model}: {whose} contains a carriage return. The exporter reads the composed file in "
            "text mode, which turns each one into a newline, so the bundle would not carry the "
            "checkpoint's template unchanged"
        )
    if found == "not_text":
        return f"{model}: {whose} is a {report.get('type')}, not text"
    return None


def _template_advice() -> str:
    """What a refusal tells the caller to do instead, as `convert` accepts it."""
    return (
        "litetune puts its text-parts block in front of the chat template the checkpoint's own "
        "tokenizer carries, and exports nothing without the block. To choose the template "
        f"yourself, pass --flag={models.TEMPLATE_FLAG}=<file>; it is used as given, so on "
        "LiteRT-LM 0.18.0 it has to read a list of parts itself, which is what the block put in "
        f"front of a template does: {models.packaged_template(models.TEXT_PARTS_BLOCK)}"
    )


def _quoted(text: str, limit: int = 160) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _probe_refusal(model: str, renders: Any) -> tuple[str | None, bool | None]:
    """Why the block is not for this template, if it is not; and whether the
    template alone already renders the parts as the string (None: unknown).

    The rendering is transformers' Jinja2, not the runtime's MiniJinja: this
    shows what the template means, and verify's rendering check is what shows
    what LiteRT-LM makes of it.
    """
    if not isinstance(renders, dict) or not all(
        isinstance(renders.get(k), dict) for k in ("string", "composed", "parts_alone")
    ):
        raise ValueError(f"the script reported renders this litetune cannot read: {renders!r}")
    string, composed, alone = renders["string"], renders["composed"], renders["parts_alone"]
    alone_same = alone.get("text") == string.get("text") if "text" in alone else False
    if "text" not in string:
        return (
            f"{model}: the template raises on one user message given as a string "
            f"({string.get('error')}) -- an error of the template's own, not of the environment",
            None,
        )
    if PROBE_TEXT not in string["text"]:
        return (
            f"{model}: the template, given one user message as a string, does not print its "
            f"text: it rendered {_quoted(string['text'])}",
            alone_same,
        )
    if composed.get("text") != string["text"]:
        shown = (
            _quoted(composed["text"])
            if "text" in composed
            else f"an error ({composed.get('error')})"
        )
        return (
            f"{model}: the text-parts block turns a list of text parts into the string a template "
            "written for a string expects, and this template does not render that way. Rendered by "
            "transformers' Jinja2 (not LiteRT-LM's MiniJinja), one user message given as a string "
            f"printed {_quoted(string['text'])}, and the same message given as parts, behind the "
            f"block, printed {shown}",
            alone_same,
        )
    return None, alone_same


@dataclass(frozen=True)
class ComposedTemplate:
    """The file every recipe's exporter is handed, and what it holds."""

    path: Path
    # The block and the checkpoint's template, as written; what each bundle's
    # `jinja_prompt_template` is read back against.
    text: str
    template_sha256: str
    composed_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "template_sha256": self.template_sha256,
            "composed_sha256": self.composed_sha256,
        }


def compose_chat_template(request: ExportRequest) -> tuple[Check, ComposedTemplate | None]:
    """Write `models.TEXT_PARTS_BLOCK` in front of the checkpoint's own chat template.

    Returns the check and, when it passed, the file every recipe's exporter is
    handed in place of `models.COMPOSED_TEMPLATE`. The template is read in the
    export environment by `_COMPOSE_SCRIPT`, as the exporter would read it, and
    written after the block unchanged to a name of this run's own; that file is
    read back here and must be exactly the block followed by the template the
    script reported, and the script's probe must show the block is for this
    template. Only then is it renamed to its content-addressed name. A
    checkpoint whose template cannot be composed is `failed`, a script that
    could not say is `could not check`, and either way nothing is returned and
    this run's file is gone. A file another run left, under any name, is never
    touched.
    """
    output_dir = request.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=COMPOSED_TEMPLATE_PREFIX, suffix=".partial", dir=output_dir
    )
    os.close(handle)
    partial = Path(name)
    try:
        return _compose_chat_template(request, partial)
    finally:
        # Renamed away when it was accepted; this run's own name either way.
        partial.unlink(missing_ok=True)


def _compose_chat_template(
    request: ExportRequest, partial: Path
) -> tuple[Check, ComposedTemplate | None]:
    block_path = Path(models.packaged_template(models.TEXT_PARTS_BLOCK))
    block = block_path.read_bytes()
    observed: dict[str, Any] = {
        "model": request.model,
        "block": str(block_path),
        "block_sha256": hashlib.sha256(block).hexdigest(),
    }
    with tempfile.TemporaryDirectory(prefix="litetune-template-") as tmp:
        work = Path(tmp)
        script = work / "compose_template.py"
        script.write_text(_COMPOSE_SCRIPT, encoding="utf-8")
        spec = work / "spec.json"
        result = work / "result.json"
        spec.write_text(
            json.dumps(
                {
                    "model": request.model,
                    "block": block.decode("utf-8"),
                    "out": str(partial),
                    "probe": PROBE_TEXT,
                }
            ),
            encoding="utf-8",
        )
        try:
            proc = request.env.run(
                ["python", str(script), str(spec), str(result)], timeout=request.timeout_s
            )
        except subprocess.TimeoutExpired:
            return Check.unchecked(
                CHAT_TEMPLATE_CHECK,
                f"the script reading {request.model}'s chat template gave no result after "
                f"{request.timeout_s}s (timeout)",
                observed=observed,
            ), None
        stderr = proc.stderr or ""
        reading = read_returncode(proc.returncode)
        if proc.returncode != 0 or not result.is_file():
            # The script does not catch its own errors, so a failed tokenizer
            # load arrives as a traceback whose last line is the reason.
            last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
            if proc.returncode == 0:
                what = "exited 0 and wrote no result"
            elif reading.conclusive:
                what = reading.describe()
            else:
                what = f"was {reading.describe(f'{request.model} chat template')}"
            logger.warning("chat template script for %s failed:\n%s", request.model, _tail(stderr))
            return Check.unchecked(
                CHAT_TEMPLATE_CHECK,
                f"the script reading {request.model}'s chat template {what}: {last}",
                observed=observed | {"exit": reading.as_dict(), "stderr_tail": _tail(stderr, 2000)},
            ), None
        try:
            report = json.loads(result.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            return Check.unchecked(
                CHAT_TEMPLATE_CHECK,
                f"the script reading {request.model}'s chat template wrote a result that is not "
                f"JSON: {exc}",
                observed=observed,
            ), None
    if not isinstance(report, dict):
        return Check.unchecked(
            CHAT_TEMPLATE_CHECK,
            f"the script reading {request.model}'s chat template wrote {type(report).__name__}, "
            "not a report",
            observed=observed,
        ), None

    observed |= {
        "transformers": report.get("transformers"),
        "template_source": report.get("source"),
        "chat_template_json": report.get("fallback"),
        "template_names": report.get("names"),
        "template_sha256": report.get("sha256"),
        "template_characters": report.get("characters"),
    }
    found = report.get("found")
    if found not in _TEMPLATE_FINDINGS:
        return Check.unchecked(
            CHAT_TEMPLATE_CHECK,
            f"the script reading {request.model}'s chat template reported {found!r}, which is "
            f"not one of {list(_TEMPLATE_FINDINGS)}",
            observed=observed,
        ), None
    if found == "unreachable":
        fallback = report.get("fallback")
        error = fallback.get("error") if isinstance(fallback, dict) else None
        return Check.unchecked(
            CHAT_TEMPLATE_CHECK,
            f"{request.model}: the tokenizer carries no chat template, and the chat_template.json "
            f"the exporter would fall back to could not be looked up ({error}). That is the "
            "Hub's, the network's or the file system's, not a fact about the checkpoint",
            observed=observed,
        ), None
    if found != "text":
        refusal = _template_refusal(request.model, report)
        if refusal is None:
            return Check.unchecked(
                CHAT_TEMPLATE_CHECK,
                f"the script reading {request.model}'s chat template reported {found!r}, which "
                "this litetune does not know how to state",
                observed=observed,
            ), None
        return Check.failed(
            CHAT_TEMPLATE_CHECK, f"{refusal}. {_template_advice()}", observed=observed
        ), None

    written = partial.read_bytes()
    if not written.startswith(block) or (
        hashlib.sha256(written[len(block) :]).hexdigest() != report.get("sha256")
    ):
        return Check.unchecked(
            CHAT_TEMPLATE_CHECK,
            f"{partial} does not hold the text-parts block followed by the template whose "
            f"sha256 the script reported ({report.get('sha256')}); nothing was exported with it",
            observed=observed,
        ), None

    renders = report.get("renders")
    faults = [
        r["error"]
        for r in (renders.values() if isinstance(renders, dict) else ())
        if isinstance(r, dict) and r.get("environment")
    ]
    if faults:
        return Check.unchecked(
            CHAT_TEMPLATE_CHECK,
            f"{request.model}: the probe could not render in the {request.env.name} environment "
            f"({faults[0]}). That is the environment's, not a fact about the template",
            observed=observed,
        ), None
    try:
        refusal, alone_same = _probe_refusal(request.model, renders)
    except ValueError as exc:
        return Check.unchecked(CHAT_TEMPLATE_CHECK, str(exc), observed=observed), None
    observed["renders_parts_without_the_block"] = alone_same
    if refusal is not None:
        return Check.failed(
            CHAT_TEMPLATE_CHECK, f"{refusal}. {_template_advice()}", observed=observed
        ), None

    composed_sha256 = hashlib.sha256(written).hexdigest()
    final = request.composed_template_path(composed_sha256)
    os.replace(partial, final)
    composed = ComposedTemplate(
        path=final,
        text=written.decode("utf-8"),
        template_sha256=str(report.get("sha256")),
        composed_sha256=composed_sha256,
    )
    observed |= {"composed": str(final), "composed_sha256": composed_sha256}
    return Check.passed(
        CHAT_TEMPLATE_CHECK,
        f"{final}: litetune's text-parts block, then {request.model}'s own chat template "
        f"(from its {report.get('source')}, sha256 {composed.template_sha256[:12]}, "
        f"{report.get('characters')} characters) as transformers {report.get('transformers')} "
        f"reads it in the {request.env.name} environment. Rendered by transformers' Jinja2, one "
        "user message given as parts behind the block prints as the template prints it given as "
        "a string; verify's rendering check is what checks the runtime. Every recipe exports "
        f"with {models.TEMPLATE_FLAG}={final}, and each bundle's template is read back",
        observed=observed,
    ), composed


# Runs inside envs.EXPORT, where `litert_lm_builder` and `tomllib` live. The
# bundle format is the builder's to define, so every fact about it -- how a
# section is laid out, which keys the rebuild regenerates, what a declaration
# looks like -- is read through the builder's own `unpack`/`pack` and parsed
# with a real TOML parser. A regex over the text would read `[[section]]`
# inside a string value as a section; this cannot.
_REPACK_SCRIPT = r'''
"""Set one metadata key on a bundle's prefill/decode section; verify by round-trip.

Reads JSON {artifact, work, key, value, template_out} from argv[1]. Writes JSON
to argv[2]:
  {"declared": <str|null>, "written": <bool>, "reason": <str|null>,
   "broken": <str|null>}
`declared` is the value the section carries after this script (or before it,
if it was left alone); `written` says whether the file was replaced. `broken`
says why the bundle itself is unsound, found before anything was unpacked.

With `template_out` set, the bundle's LlmMetadata `jinja_prompt_template` is
written there as JSON as soon as the first unpack has it --
  {"sections": <int>, "present": <bool>, "template": <str|null>}
-- and a rebuild that would change it is refused, so it describes the file
that is left at `artifact` whatever happens after.
"""
import copy
import io
import json
import os
import struct
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: the builder itself depends on tomli
    import tomli as tomllib

from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_peek

VOLATILE = {"uuid", "creation_timestamp"}


def load(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def prefill_sections(doc):
    return [
        s
        for s in doc.get("section", [])
        if s.get("section_type") == "TFLiteModel"
        and str(s.get("model_type", "")).lower() == "prefill_decode"
    ]


def declared_in(section, key):
    for entry in section.get("additional_metadata", []) or []:
        if entry.get("key") == key:
            return str(entry.get("value", ""))
    if key in section:
        return str(section[key])
    return None


def comparable(doc):
    """The document with the two regenerated system keys removed and paths by name."""
    sm = doc.get("system_metadata", {})
    entries = [e for e in sm.get("entries", []) if e.get("key") not in VOLATILE]
    sections = copy.deepcopy(doc.get("section", []))
    for s in sections:
        if "data_path" in s:
            s["data_path"] = os.path.basename(s["data_path"])
    return {"system_metadata": {"entries": entries}, "section": sections}


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return toml_string(str(v))


def toml_string(text):
    """A TOML basic string. Not json.dumps: that writes a character outside the
    BMP as a UTF-16 surrogate pair (`\\ud83d\\ude80`), which TOML forbids and
    tomllib rejects, so one emoji in any metadata value would sink the pack."""
    out = []
    for ch in text:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append(f"\\u{o:04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def entry_line(e):
    """One element of an `entries` / `additional_metadata` array. Both arrays
    carry the same {key, value_type, value} shape, so both are written here."""
    return (
        f"  {{ key = {toml_value(e['key'])}, "
        f"value_type = {toml_value(e['value_type'])}, "
        f"value = {toml_value(e['value'])} }},"
    )


def write_toml(doc, path):
    """Emit TOML for the builder to read back. Only the shapes `unpack` writes."""
    lines = ["[system_metadata]", "entries = ["]
    lines += [entry_line(e) for e in doc.get("system_metadata", {}).get("entries", [])]
    lines += ["]", ""]
    for sec in doc.get("section", []):
        lines.append("[[section]]")
        for k, v in sec.items():
            if k == "additional_metadata":
                continue
            lines.append(f"{k} = {toml_value(v)}")
        meta = sec.get("additional_metadata") or []
        if meta:
            lines.append("additional_metadata = [")
            lines += [entry_line(e) for e in meta]
            lines.append("]")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def chat_template_in(directory, doc):
    """The LlmMetadata section's `jinja_prompt_template`, as `unpack` wrote it:
    text, through the builder's own proto (`litertlm_peek` `_dump_llm_metadata_proto`)."""
    from google.protobuf import text_format
    from litert_lm_builder.runtime.proto import llm_metadata_pb2

    sections = [s for s in doc.get("section", []) if s.get("section_type") == "LlmMetadata"]
    if len(sections) != 1:
        return {"sections": len(sections), "present": False, "template": None}
    text = (Path(directory) / os.path.basename(sections[0]["data_path"])).read_text(
        encoding="utf-8"
    )
    meta = text_format.Parse(text, llm_metadata_pb2.LlmMetadata())
    present = meta.HasField("jinja_prompt_template")
    return {
        "sections": 1,
        "present": present,
        "template": meta.jinja_prompt_template if present else None,
    }


def report(declared=None, written=False, reason=None, broken=None):
    """This script's whole result. The exit status says only whether the script
    itself ran; every outcome the parent tells apart is in here."""
    return {"declared": declared, "written": written, "reason": reason, "broken": broken}


def past_the_end(path):
    """Why the bundle is not whole, or None.

    `unpack` reads each section with one `read` of its length, which returns
    short at the end of a file and raises nothing (`litertlm_peek.py`), so a cut
    bundle would unpack, repack into a well-formed shorter one, and read back
    whole. So the header is checked first, with the builder's own reader and
    its own layout (`litertlm_core.py`): a file too short to name where its
    header ends, a header that ends past the file, one the reader refuses (a
    wrong magic number, a cut header), and then each section's end."""
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        handle.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
        raw = handle.read(8)
    if len(raw) < 8:
        return f"the {size:,}-byte file is cut short inside its header"
    header_end = struct.unpack("<Q", raw)[0]
    if header_end > size:
        return f"its header ends at byte {header_end:,}, past the end of the {size:,}-byte file"
    try:
        header = litertlm_peek.read_litertlm_header(path, io.StringIO())
    except OSError:
        raise
    except Exception as exc:  # the builder's own reader refuses it
        return f"its header cannot be read ({type(exc).__name__}: {exc})"
    sections = header.SectionMetadata()
    count = sections.ObjectsLength() if sections else 0
    for index in range(count):
        end = sections.Objects(index).EndOffset()
        if end > size:
            return (
                f"section {index} of {count} ends at byte {end:,}, past the end of the "
                f"{size:,}-byte file"
            )
    return None


def main(spec):
    artifact, work = spec["artifact"], Path(spec["work"])
    key, value = spec["key"], spec["value"]
    entry = {"key": key, "value_type": "String", "value": value}

    cut = past_the_end(artifact)
    if cut is not None:
        return report(reason=cut, broken=cut)

    before_dir = work / "before"
    lb.unpack(artifact, str(before_dir))
    before = load(before_dir / "model.toml")
    template_out = spec.get("template_out")
    if template_out:
        template = chat_template_in(before_dir, before)
        Path(template_out).write_text(json.dumps(template), encoding="utf-8")
    targets = prefill_sections(before)
    if len(targets) != 1:
        return report(
            reason=f"found {len(targets)} prefill_decode sections where exactly one was expected"
        )
    existing = declared_in(targets[0], key)
    if existing is not None:
        shown = existing or "(empty)"
        reason = f"the bundle already declares {key} = {shown}; left as written"
        return report(declared=existing, reason=reason)

    # Patch the structure, not the text.
    patched = copy.deepcopy(before)
    sec = prefill_sections(patched)[0]
    sec["additional_metadata"] = list(sec.get("additional_metadata") or []) + [entry]
    for s in patched["section"]:
        s["data_path"] = str(before_dir / os.path.basename(s["data_path"]))
    patched_toml = work / "patched.toml"
    write_toml(patched, patched_toml)
    rebuilt = work / Path(artifact).name
    lb.pack(str(patched_toml), str(rebuilt))

    # Verify by round-tripping through the same unpack: the parsed document
    # must be the original plus exactly the one entry -- which is what
    # `patched` already is, so it is the expectation.
    after_dir = work / "after"
    lb.unpack(str(rebuilt), str(after_dir))
    after = load(after_dir / "model.toml")
    if comparable(after) != comparable(patched):
        return report(reason="the rebuild changed the bundle beyond the one key")
    read_back = declared_in(prefill_sections(after)[0], key)
    if read_back != value:
        return report(reason=f"the rebuild wrote {key} = {read_back!r}; {value!r} was asked")
    if template_out and chat_template_in(after_dir, after) != template:
        return report(reason="the rebuild changed the bundle's chat template")
    os.replace(rebuilt, artifact)
    return report(declared=read_back, written=True)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)))
'''


def set_gpu_activation(
    artifact: Path, env: envs.StageEnv, *, timeout: int = REPACK_TIMEOUT_S
) -> tuple[str | None, str | None]:
    """Write `prefer_activation_type` into the bundle's prefill/decode section.

    Returns (what the bundle now declares, what happened). The first is None
    when the artifact is left as the toolchain wrote it with no declaration --
    a working CPU bundle and a broken GPU one -- so the caller records the
    reason rather than failing the export. It is the bundle's *own* value as
    the builder reads it back, never the requested one: a bundle that already
    declared something is left alone and reported as carrying that, because
    `--flag=--experimental_use_mixed_precision` produces `fp32_fp16` and an
    explicit `fp16` reproduces the fault, and labelling either as `fp32` would
    be the false pass this key exists to end.

    The work is done by `_REPACK_SCRIPT` inside the export environment: the
    builder's own `unpack`, a real TOML parse, one entry appended to the
    section's `additional_metadata`, the builder's own `pack`, and a second
    `unpack` whose parsed document must equal the first plus that one entry
    (the system `uuid` and `creation_timestamp` the builder regenerates are
    excluded). Section bytes are not compared; on the one bundle measured the
    size was unchanged (455,759,152 bytes) because section offsets are
    16 KiB-aligned. The original is replaced only after that comparison.

    The exporter offers no flag for this value, and the one flag it does offer
    (`--experimental_use_mixed_precision`) changes the graph as well.
    """
    repack = repack_bundle(artifact, env, timeout=timeout)
    return repack.declared, repack.note


@dataclass(frozen=True)
class BundleTemplate:
    """A bundle's LlmMetadata `jinja_prompt_template`, as the builder's `unpack` reads it.

    `text` is None when the bundle carries none, and `sections` is how many
    LlmMetadata sections it has; `empty` when the file has no bytes at all.
    `unread` is set when no reading was made, which says nothing about the bundle.
    """

    text: str | None = None
    sections: int = 1
    unread: str | None = None
    empty: bool = False


@dataclass(frozen=True)
class Repack:
    """What `repack_bundle` did: the GPU key, and what it found on the way."""

    declared: str | None
    note: str | None
    # The bundle's chat template, when it was asked for.
    template: BundleTemplate | None = None
    # Why the bundle is unsound, found before anything was unpacked: it is not
    # handed on, whatever its family.
    broken: str | None = None


def repack_bundle(
    artifact: Path,
    env: envs.StageEnv,
    *,
    timeout: int = REPACK_TIMEOUT_S,
    read_template: bool = False,
) -> Repack:
    """`set_gpu_activation`, and with `read_template` the bundle's chat template too.

    One unpack serves both. `_REPACK_SCRIPT` writes the template out as soon as
    its first unpack has it, before anything that can fail after, and refuses a
    rebuild that would change it -- so the reading describes the file left at
    `artifact` whether or not the key was written, and survives a repack that
    failed. `template` is None when it was not asked for. A bundle whose
    header places a section past the end of the file is `broken`, and is
    neither unpacked nor read.
    """
    # Absolute before anything is derived from it. Below Python 3.12
    # `tempfile.mkdtemp(dir=...)` returns a relative path for a relative `dir`,
    # the script then wrote relative `data_path`s, and the builder resolved each
    # against the TOML's own directory -- a doubled path, and a repack that
    # failed on every `convert --output-dir artifacts/...` run on 3.10 or 3.11.
    # `absolute`, not `resolve`: a symlink stays the file that is replaced.
    artifact = Path(artifact).absolute()
    if artifact.stat().st_size == 0:
        # Nothing to repack, and the sweep's size comparison downstream is
        # what reports a zero-byte export; this must not turn it into 8 bytes.
        empty = BundleTemplate(sections=0, empty=True) if read_template else None
        return Repack(None, "the artifact is empty; nothing to repack", empty)
    work: Path | None = None
    template_out: Path | None = None
    try:
        try:
            # A fresh private directory, never a deterministic name: a stale or
            # planted `.foo-repack` symlink would otherwise have its *target*
            # emptied by cleanup, and two runs on one output dir would delete
            # each other's unpack mid-flight. Same filesystem as the artifact so
            # the script's final `os.replace` is one rename.
            work = Path(tempfile.mkdtemp(prefix=f".{artifact.stem}-repack-", dir=artifact.parent))
            template_out = work / "template.json" if read_template else None
            declared, note, broken = _run_repack(artifact, env, work, timeout, template_out)
        except subprocess.TimeoutExpired:
            declared, note, broken = (
                None,
                f"the repack did not finish within {timeout}s; the original was kept",
                None,
            )
        except Exception as exc:  # noqa: BLE001 - a broken repack must not unmake a good export
            # Whatever went wrong here, the toolchain's artifact is intact and is a
            # working CPU bundle. Escaping would let `guard` record the whole
            # recipe as "could not check", which is a smaller truth than the one
            # available: exported, without the GPU key, and here is why. The frame is kept in
            # the log; the note is what the user reads.
            logger.exception("GPU activation repack of %s failed", artifact)
            declared, note, broken = (
                None,
                f"{type(exc).__name__}: {exc}; the original was kept",
                None,
            )
        template = _bundle_template(template_out, note) if read_template else None
        return Repack(declared, note, template, broken)
    finally:
        if work is not None:
            # A scratch dir the size of two bundles; a leak has to be said,
            # not hidden behind ignore_errors.
            leaked: list[str] = []
            shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
            if leaked:
                logger.warning(
                    "%d scratch file(s) left under %s after the repack; delete by hand",
                    len(leaked),
                    work,
                )


def _run_repack(
    artifact: Path, env: envs.StageEnv, work: Path, timeout: int, template_out: Path | None
) -> tuple[str | None, str | None, str | None]:
    """Run `_REPACK_SCRIPT` on `artifact` in `work`. Returns (declared, note, broken)."""
    script = work / "repack.py"
    script.write_text(_REPACK_SCRIPT, encoding="utf-8")
    spec = work / "spec.json"
    result = work / "result.json"
    spec.write_text(
        json.dumps(
            {
                "artifact": str(artifact),
                "work": str(work),
                "key": GPU_ACTIVATION_KEY,
                "value": GPU_ACTIVATION,
                "template_out": str(template_out) if template_out is not None else None,
            }
        ),
        encoding="utf-8",
    )
    proc = env.run(["python", str(script), str(spec), str(result)], timeout=timeout)
    if proc.returncode != 0 or not result.is_file():
        # The script does not catch its own errors, so a builder failure
        # arrives as a traceback. The last line is the reason; the rest
        # goes to the log where a frame is worth having.
        stderr = proc.stderr or ""
        last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
        logger.warning("GPU activation repack of %s failed:\n%s", artifact, _tail(stderr, 2000))
        note = f"the repack script exited {proc.returncode}: {last}; the original was kept"
        return None, note, None
    report = json.loads(result.read_text(encoding="utf-8"))
    declared = report.get("declared")
    reason = report.get("reason")
    broken = str(report["broken"]) if report.get("broken") else None
    if declared is None:
        return None, f"{reason}; the original was kept", broken
    if report.get("written") and reason is not None:
        # The script's contract: a replaced file has no reason to give.
        return None, f"the repack reported both a rebuild and a reason ({reason}); refused", broken
    return str(declared), reason, broken


def _bundle_template(path: Path | None, note: str | None) -> BundleTemplate:
    """What `_REPACK_SCRIPT` wrote to `template_out`, or why there is nothing."""
    if path is None or not path.is_file():
        return BundleTemplate(unread=f"the repack script took no reading: {note or 'no reason'}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return BundleTemplate(unread=f"the repack script's reading could not be read: {exc}")
    sections = data.get("sections") if isinstance(data, dict) else None
    text = data.get("template") if isinstance(data, dict) else None
    if not isinstance(sections, int) or not (text is None or isinstance(text, str)):
        return BundleTemplate(unread=f"the repack script's reading is not one this reads: {data!r}")
    return BundleTemplate(text=text if data.get("present") else None, sections=sections)


def _template_carried(
    name: str, artifact: Path, composed: ComposedTemplate, carried: BundleTemplate | None
) -> Check | None:
    """Refuse a bundle whose template is not exactly the composed file's text.

    None when it is. Handing the exporter the file does not settle what the
    bundle carries: it reads the file in text mode (`litert_lm_builder.py:430`),
    `--experimental_transpile_chat_template_for_minijinja` rewrites the template
    (`:316-317`), and a metadata override takes the place of the metadata it
    builds (`:438-456`). Nothing before this reads what it embedded.
    """
    observed: dict[str, Any] = {
        "artifact": str(artifact),
        "composed": str(composed.path),
        "composed_sha256": composed.composed_sha256,
    }
    if carried is None or carried.unread is not None:
        why = carried.unread if carried is not None else "it was not read"
        return Check.unchecked(
            name,
            f"{artifact.name}: the chat template it carries could not be read back ({why}), so "
            f"nothing shows it is {composed.path.name}",
            observed=observed,
        )
    if carried.empty:
        return Check.failed(
            name,
            f"{artifact.name} is empty (0 bytes), so it carries no template; it was handed "
            f"{composed.path.name}",
            observed=observed | {"artifact_bytes": 0},
        )
    if carried.sections != 1:
        return Check.failed(
            name,
            f"{artifact.name} has {carried.sections} LlmMetadata sections where one was expected, "
            f"so it does not carry {composed.path.name} as its template",
            observed=observed | {"llm_metadata_sections": carried.sections},
        )
    if carried.text is None:
        return Check.failed(
            name,
            f"{artifact.name} carries no jinja_prompt_template, where it was handed "
            f"{composed.path.name} (sha256 {composed.composed_sha256[:12]})",
            observed=observed | {"bundle_template_sha256": None},
        )
    carried_sha256 = hashlib.sha256(carried.text.encode("utf-8")).hexdigest()
    if carried.text != composed.text:
        return Check.failed(
            name,
            f"{artifact.name} carries a jinja_prompt_template with sha256 {carried_sha256[:12]}, "
            f"not {composed.path.name}'s {composed.composed_sha256[:12]} that it was handed",
            observed=observed | {"bundle_template_sha256": carried_sha256},
        )
    return None


REFUSED_SUFFIX = ".refused"


def _set_aside(check: Check, artifact: Path) -> Check:
    """`check`, with its bundle renamed out of the place a usable one goes.

    A refused bundle left at `<recipe>/model.litertlm` is picked up by anything
    that globs for bundles, or by a person who sees the file and not the exit
    code. Renamed, not deleted: it is what diagnoses the refusal.
    """
    aside = artifact.with_name(artifact.name + REFUSED_SUFFIX)
    try:
        os.replace(artifact, aside)
    except OSError as exc:
        logger.exception("could not move the refused bundle %s aside", artifact)
        said = (
            f"; it could not be moved aside ({type(exc).__name__}: {exc}) and is still at "
            f"{artifact}"
        )
        moved = None
    else:
        said = f"; the bundle was moved to {aside}"
        moved = str(aside)
    observed = dict(check.observed or {}) | {"refused_bundle": moved}
    return Check(check.name, check.outcome, check.detail + said, observed)


def export_recipe(
    request: ExportRequest, recipe: str, composed: ComposedTemplate | None = None
) -> RecipeExport:
    """Run one export. A non-zero exit is recorded, not raised.

    `composed` is what `compose_chat_template` wrote, for a plan that asks for
    one. Such a plan given none is recorded as not attempted, since there is no
    command line to give the exporter; given one, the bundle's own template is
    read back and must be its text.
    """
    name = check_name(recipe)
    if request.composes_chat_template and composed is None:
        return RecipeExport(
            recipe=recipe,
            check=Check.unchecked(
                name,
                "the plan asks for the checkpoint's own chat template behind litetune's "
                f"text-parts block, and none was composed; nothing was exported for {recipe}",
                observed={"recipe": recipe},
            ),
        )
    argv = tuple(request.argv(recipe, composed.path if composed is not None else None))
    try:
        definition = definition_of(recipe)
    except RecipeDefinitionError as exc:
        # A fact about this installation, not about the recipe: nothing ran,
        # and no other recipe is exported in its place.
        return RecipeExport(
            recipe=recipe,
            argv=argv,
            check=Check.unchecked(
                name,
                f"{exc}; nothing was exported for {recipe}",
                observed={"recipe": recipe},
            ),
        )
    out_dir = request.dir_for(recipe)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_wall = time.time()
    started = time.perf_counter()
    try:
        proc = request.env.run(list(argv), timeout=request.timeout_s, env=_cpu_only_environ())
    except subprocess.TimeoutExpired:
        seconds = time.perf_counter() - started
        logger.warning("litert-torch timed out after %ss on recipe %s", request.timeout_s, recipe)
        # A hang is a statement about this machine, not about the recipe. From
        # out here it is indistinguishable from a stalled environment, so it is
        # recorded as not performed.
        return RecipeExport(
            recipe=recipe,
            argv=argv,
            recipe_definition=definition,
            seconds=seconds,
            check=Check.unchecked(
                name,
                f"no result after {request.timeout_s}s (timeout): the export did not finish, "
                "which says nothing about the recipe",
                observed={"recipe": recipe, "timeout_s": request.timeout_s},
            ),
        )
    except OSError as exc:
        logger.exception("could not start litert-torch for recipe %s", recipe)
        return RecipeExport(
            recipe=recipe,
            argv=argv,
            recipe_definition=definition,
            seconds=time.perf_counter() - started,
            check=Check.unchecked(
                name,
                f"litert-torch could not be started: {type(exc).__name__}: {exc}",
                observed={"recipe": recipe},
            ),
        )

    seconds = time.perf_counter() - started
    stderr = proc.stderr or ""
    stdout = proc.stdout or ""
    produced = _produced_files(out_dir, started_wall)
    artifacts = [p for p in produced if p.suffix == ".litertlm"]
    observed: dict[str, Any] = {
        "recipe": recipe,
        "returncode": proc.returncode,
        "seconds": round(seconds, 3),
        "files_written": [p.name for p in produced],
        "backend": "cpu",
    }
    base: dict[str, Any] = {
        "recipe": recipe,
        "argv": argv,
        "recipe_definition": definition,
        "seconds": seconds,
        "returncode": proc.returncode,
        "stderr": stderr,
        "stdout_tail": stdout[-_STDOUT_TAIL:],
    }

    reading = read_returncode(proc.returncode)
    observed["exit"] = reading.as_dict()
    if not reading.conclusive:
        # The export was killed, not failed. This exact case -- a `-9` read as
        # "ran and failed" -- struck Gemma 4 from the catalogue for a reason
        # that had nothing to do with the model: the process had hit a 32 GiB
        # ceiling, and on a larger machine the same command produced a specific,
        # actionable error. See `litetune.exits`.
        logger.warning("litert-torch on recipe %s was %s", recipe, reading.describe())
        return RecipeExport(
            **base,
            check=Check.unchecked(
                name,
                f"litert-torch was {reading.describe('the recipe')}. "
                f"stderr: {_tail(stderr) or 'none'}",
                observed=observed | {"stderr_tail": _tail(stderr, 2000)},
            ),
        )

    if proc.returncode != 0:
        return RecipeExport(
            **base,
            check=Check.failed(
                name,
                f"litert-torch exited {proc.returncode}: {_tail(stderr) or 'no stderr'}",
                observed=observed | {"stderr_tail": _tail(stderr, 2000)},
            ),
        )

    if not artifacts:
        # The whole reason this branch exists: exit zero is not the result. An
        # export that writes nothing, or writes only companion files, has
        # failed however cleanly it returned.
        return RecipeExport(
            **base,
            check=Check.failed(
                name,
                f"litert-torch exited zero but wrote no .litertlm into {out_dir} "
                f"({len(produced)} file(s) written this run)",
                observed=observed,
            ),
        )

    if len(artifacts) > 1:
        return RecipeExport(
            **base,
            check=Check.failed(
                name,
                f"litert-torch wrote {len(artifacts)} .litertlm files into {out_dir} "
                f"({', '.join(p.name for p in artifacts)}): the artifact is ambiguous and "
                "picking one would be a guess",
                observed=observed,
            ),
        )

    artifact = artifacts[0]
    # Repacked before it is hashed or sized: the file that ships is the one
    # that is recorded. A repack that cannot be made is a limitation on a
    # passed check, not a failed one -- the CPU artifact is intact.
    remaining = max(1, int(request.timeout_s - seconds))
    repack = repack_bundle(
        artifact,
        request.env,
        timeout=min(REPACK_TIMEOUT_S, remaining),
        read_template=composed is not None,
    )
    gpu_activation, gpu_note = repack.declared, repack.note
    if gpu_activation is None:
        logger.warning("recipe %s: GPU activation not set: %s", recipe, gpu_note)
    refused = None
    if repack.broken is not None:
        # The same verdict as an export that exited zero having written no
        # bundle: the exporter's output is not a usable one.
        refused = Check.failed(
            name,
            f"litert-torch exited zero but {artifact.name} is not whole: {repack.broken}",
            observed=observed | {"artifact": str(artifact)},
        )
    elif composed is not None:
        refused = _template_carried(name, artifact, composed, repack.template)
    if refused is not None:
        # Not handed on as an artifact, and not left where one would be.
        return RecipeExport(**base, check=_set_aside(refused, artifact))
    if composed is not None:
        observed["chat_template_sha256"] = composed.composed_sha256
    artifact_bytes = artifact.stat().st_size
    companions = tuple(p.name for p in produced if p != artifact)
    shipped_bytes = sum(p.stat().st_size for p in produced)
    digest = _sha256(artifact) if request.hash_artifacts else None
    gpu_text = describe_gpu_activation(gpu_activation)
    if gpu_activation is None and gpu_note:
        gpu_text = f"{gpu_text}: {gpu_note}"
    return RecipeExport(
        **base,
        artifact=artifact,
        companions=companions,
        artifact_bytes=artifact_bytes,
        shipped_bytes=shipped_bytes,
        sha256=digest,
        gpu_activation=gpu_activation,
        gpu_activation_note=gpu_note,
        check=Check.passed(
            name,
            f"{artifact.name}: {artifact_bytes:,} bytes in {seconds:.1f}s exported on cpu; "
            f"{gpu_text} — produced, not verified",
            observed=observed
            | {
                "artifact": str(artifact),
                "artifact_bytes": artifact_bytes,
                "shipped_bytes": shipped_bytes,
                "sha256": digest,
                "gpu_activation": gpu_activation,
                "gpu_activation_note": gpu_note,
                "verified": False,
            },
        ),
    )


# ---------------------------------------------------------------------------
# Comparing the sweep
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncompared:
    """No comparison was made, and why.

    A single size is not a frontier. Reporting one number from one recipe as
    though it were the outcome of a sweep is how a default gets mistaken for a
    decision.
    """

    reason: str

    @property
    def compared(self) -> bool:
        return False

    def as_dict(self) -> dict[str, Any]:
        return {"compared": False, "reason": self.reason}


@dataclass(frozen=True)
class SizeComparison:
    """The sweep's artifacts, on the one axis this stage can measure.

    Bytes only, and bytes cannot rank recipes: the two wi8 recipes differ by
    0.04% in size and by 0.024 in exact match, with the *larger* file being the
    better model. This exists to show that the artifacts are near-identical on
    every axis available at export time, which is the argument for measuring
    them on held-out data instead.
    """

    sizes: dict[str, int]
    seconds: dict[str, float]

    @property
    def compared(self) -> bool:
        return True

    @property
    def smallest(self) -> str:
        return min(self.sizes, key=lambda r: self.sizes[r])

    @property
    def largest(self) -> str:
        return max(self.sizes, key=lambda r: self.sizes[r])

    @property
    def spread_share(self) -> float:
        low, high = self.sizes[self.smallest], self.sizes[self.largest]
        # `low` cannot be zero: `compare_sizes` returns `Uncompared` for a
        # zero-byte artifact rather than constructing this.
        return (high - low) / low

    def as_dict(self) -> dict[str, Any]:
        return {
            "compared": True,
            "axis": "bytes",
            "sizes": dict(self.sizes),
            "seconds": {k: round(v, 3) for k, v in self.seconds.items()},
            "smallest": self.smallest,
            "largest": self.largest,
            "spread_share": round(self.spread_share, 6),
            "accuracy": {
                "available": False,
                "reason": (
                    "accuracy is not observable at export time; two recipes 0.04% apart in "
                    "bytes were 0.024 apart in exact match. Run `litetune verify` on each "
                    "artifact against held-out data to rank them."
                ),
            },
        }


def compare_sizes(
    requested: Sequence[str], exports: Sequence[RecipeExport]
) -> SizeComparison | Uncompared:
    """Compare the sweep, or state precisely why there is nothing to compare."""
    succeeded = [e for e in exports if e.ok and e.shipped_bytes is not None]
    if len(requested) < 2:
        only = requested[0] if requested else "none"
        return Uncompared(
            f"only one recipe was requested ({only}): no alternative was measured, so this "
            "size is not a comparison and the recipe was not chosen on evidence. Sweeping "
            f"{list(MEASURED_RECIPES)} is what makes the number mean something."
        )
    if len(succeeded) < 2:
        return Uncompared(
            f"{len(succeeded)} of {len(requested)} requested recipes produced an artifact: "
            "there is nothing to compare against"
        )
    empty = [e.recipe for e in succeeded if not e.shipped_bytes]
    if empty:
        # A zero-byte artifact has no size to be a share of. Reporting "0.0%
        # spread" for it would be a measured-looking number for a comparison
        # that cannot be made -- everything else in this module distinguishes
        # those two, and `Uncompared` is what says so.
        return Uncompared(
            f"{', '.join(empty)} produced a zero-byte artifact: a size spread against nothing "
            "is not a comparison"
        )
    return SizeComparison(
        sizes={e.recipe: int(e.shipped_bytes or 0) for e in succeeded},
        seconds={e.recipe: float(e.seconds or 0.0) for e in succeeded},
    )


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """Every artifact the sweep produced, and what none of them have been shown to be."""

    request: ExportRequest
    checks: CheckSet
    toolchain: Toolchain
    exports: list[RecipeExport] = field(default_factory=list)
    # Recipes that never ran. Distinct from a recipe that failed: one is a
    # statement about the machine, the other about the export.
    not_attempted: tuple[str, ...] = ()
    comparison: SizeComparison | Uncompared = field(
        default_factory=lambda: Uncompared("the sweep did not run")
    )
    limitations: list[str] = field(default_factory=list)
    # Model-family guidance that is not a limitation and not a verdict: a recipe
    # somebody with more information recommends. Surfaced, never substituted --
    # the recipe that was asked for is the recipe that was exported.
    recommendations: list[str] = field(default_factory=list)
    # The file every recipe's exporter was handed in place of
    # `models.COMPOSED_TEMPLATE`, when the plan asked for one and it was made.
    chat_template: ComposedTemplate | None = None

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def succeeded(self) -> list[RecipeExport]:
        return [e for e in self.exports if e.ok]

    @property
    def failed(self) -> list[RecipeExport]:
        return [e for e in self.exports if e.check.outcome is Outcome.FAILED]

    @property
    def artifacts(self) -> list[Path]:
        return [e.artifact for e in self.succeeded if e.artifact is not None]

    @property
    def verified(self) -> bool:
        """Always False, structurally. See `RecipeExport.verified`."""
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPORT_SCHEMA,
            "verified": False,
            "unverified_reason": NOT_VERIFIED if self.artifacts else NO_ARTIFACT,
            "outcome": self.outcome.value,
            "request": self.request.as_dict(),
            "toolchain": self.toolchain.as_dict(),
            "exports": [e.as_dict() for e in self.exports],
            "not_attempted": list(self.not_attempted),
            "comparison": self.comparison.as_dict(),
            "checks": self.checks.as_dict(),
            "limitations": list(self.limitations),
            "recommendations": list(self.recommendations),
            "chat_template": self.chat_template.as_dict() if self.chat_template else None,
            "model_rules": self.request.plan.as_dict(),
        }


def _nothing_attempted(
    result: ExportResult, request: ExportRequest, detail: str, events: EventStream
) -> ExportResult:
    """End a run whose export environment was not usable.

    Nothing was attempted, so nothing failed. Recording these recipes as
    failures would be a verdict about the toolchain drawn from a fact about
    this machine.
    """
    result.not_attempted = tuple(request.recipes)
    result.toolchain = Toolchain.unknown(
        f"the export environment was not usable, so its versions were never read: {detail}",
        request.env.requirements,
    )
    result.comparison = Uncompared(f"no recipe was exported: {detail}")
    result.limitations.append(f"none of {list(request.recipes)} was attempted: {detail}")
    events.stage_finished(result.outcome.value, attempted=0)
    return result


def run_export(request: ExportRequest, events: EventStream | None = None) -> ExportResult:
    """Sweep the requested recipes. Returns a result; failures are recorded, not raised."""
    # This run's own record of which host variables it dropped, so a second
    # run in one process -- a notebook, a service -- says what the first
    # one did instead of inheriting its silence.
    envs.forget_reported_drops()
    events = events or EventStream(echo_json=False)
    events.stage_started("export", model=request.model, recipes=list(request.recipes))
    result = ExportResult(
        request=request,
        checks=CheckSet(name=f"export:{request.model}"),
        toolchain=Toolchain.unknown("not read yet", request.env.requirements),
    )
    # -- can this machine build the toolchain at all? -----------------------
    # First: a run that cannot convert here -- Windows, where `litert-converter`
    # publishes no wheel -- is refused before anything is built or installed.
    # `provision` asks the same question again for a caller that reaches it
    # another way.
    if not request.env.ready:
        refusal = request.env.refusal()
        if refusal:
            environment = result.checks.add(
                Check.unchecked(
                    "export environment",
                    refusal,
                    observed={"name": request.env.name, "identity": request.env.identity},
                )
            )
            events.check(environment)
            return _nothing_attempted(result, request, environment.detail, events)

    # Two answers to "what is this", and no way to choose. Refused before any
    # work: exporting under either one produces a bundle whose family flags may
    # be wrong, and the check would say `passed` about it.
    if request.identity_conflict:
        result.checks.add(
            Check.unchecked(
                "model identity",
                request.identity_conflict,
                observed={
                    "base_model_flag": request.base_model,
                    "recorded": models.recorded_identity(request.model),
                },
            )
        )
        result.not_attempted = tuple(request.recipes)
        events.stage_finished(result.outcome.value, attempted=0)
        return result

    # A sidecar that exists and cannot be read is a fault, not an absence: the
    # family was then guessed from the path, which is what the sidecar exists to
    # stop. Said where it will be read, not only recorded in the hint.
    broken_record = models.provenance_error(request.model)
    if broken_record:
        result.limitations.append(
            f"{broken_record}. The family was determined without it, so the export flags "
            "below may be the wrong ones; rewrite the file or pass --base-model"
        )
        events.note(broken_record, model=request.model)

    if request.unknown_recipes:
        result.limitations.append(
            f"recipes {list(request.unknown_recipes)} are not among the ones litetune has "
            f"measured ({list(KNOWN_RECIPES)}); they were passed through to the toolchain as given"
        )
    if tuple(request.recipes) == (TOOLCHAIN_DEFAULT_RECIPE,):
        result.limitations.append(
            f"{TOOLCHAIN_DEFAULT_RECIPE} is the toolchain's own default, and its docstring warns "
            "that quality 'may suffer due to the on-the-fly quantization'. It measured 0.024 "
            f"below {MEASURED_RECIPES[1]} on 640 held-out examples"
        )

    # -- what does this model family require? ------------------------------
    plan = request.plan
    for text in plan.limitations:
        result.limitations.append(text)
    for note in plan.notes:
        events.note(note, model=request.model, family=plan.rules.family if plan.rules else None)
    for text in plan.recommendations:
        result.recommendations.append(text)
        events.note(text, model=request.model, recommendation=True)
    for check in plan.checks:
        result.checks.add(check)
        events.check(check)
    if not plan.usable:
        # A required flag whose value litetune could not determine. Exporting
        # anyway would produce an artifact that fails silently at serving time,
        # which is worse than not exporting: nothing here is a verdict about the
        # recipes, so none of them is recorded as failed.
        result.not_attempted = tuple(request.recipes)
        unresolved = next(c for c in plan.checks if c.outcome is Outcome.UNCHECKED)
        detail = unresolved.detail
        result.toolchain = Toolchain.unknown(
            f"no export was attempted, so no versions were read: {detail}",
            request.env.requirements,
        )
        result.comparison = Uncompared(f"no recipe was exported: {detail}")
        result.limitations.append(f"none of {list(request.recipes)} was attempted: {detail}")
        events.stage_finished(result.outcome.value, attempted=0)
        return result

    # -- can this run at all? ---------------------------------------------
    with guard("export environment") as sink:
        if request.auto_provision:
            request.env.provision(events=events)
        if request.env.ready:
            sink.append(
                Check.passed(
                    "export environment",
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
                    "export environment",
                    f"environment {request.env.name!r} {request.env.unready_detail}",
                    observed={"name": request.env.name, "identity": request.env.identity},
                )
            )
    environment = result.checks.add(sink[0])
    events.check(environment)
    if not environment.conclusive:
        return _nothing_attempted(result, request, environment.detail, events)

    # -- the checkpoint's own `vocab_file` ------------------------------------
    # Once for the sweep, before the toolchain is asked anything: a stale
    # `vocab_file` fails the export outright, after the model has loaded,
    # naming a path that never existed here.
    #
    # Only now, with an environment that works. This writes to the *input*
    # directory, and a run that ends before any export is attempted -- a
    # refusal, a provision that failed, `LITETUNE_ALLOW_MISSING_WHEELS=1` whose
    # pip then found nothing -- must leave the checkpoint as it was handed.
    # Reported and not merely logged: a tool that edits what you handed it must
    # say so where you will see it.
    model_path = Path(request.model)
    if model_path.is_dir():
        ok, repaired = repair_vocab_file(model_path)
        if repaired:
            events.note(repaired, model=request.model)
            result.limitations.append(repaired)
        if not ok:
            result.limitations.append(
                "the export below is likely to fail at `export_tokenizer` for the reason above"
            )

    # -- provenance --------------------------------------------------------
    with guard("toolchain versions") as sink:
        result.toolchain = resolve_toolchain(request.env)
    if sink:
        result.toolchain = Toolchain.unknown(sink[0].detail, request.env.requirements)
    if not result.toolchain.available:
        result.limitations.append(
            f"toolchain versions were not resolved ({result.toolchain.unresolved_reason}): the "
            "artifact's provenance is incomplete and a version-specific failure will be harder "
            "to attribute later"
        )

    # -- does the toolchain support this model at all? ----------------------
    if plan.rules is not None and plan.rules.min_transformers:
        version_check = models.transformers_check(
            request.model,
            plan.rules,
            installed_version(result.toolchain, "transformers"),
            f"the {request.env.name} environment",
            unknown_reason=result.toolchain.unresolved_reason or "",
        )
        events.check(version_check)
        if version_check.conclusive:
            result.checks.add(version_check)
        else:
            # Deliberately a limitation and not an UNCHECKED item in the set, for
            # the same reason `resolve_toolchain` is provenance rather than a
            # check: a pip that would not answer says nothing about the artifact,
            # and folding it in would turn a perfectly good export into "could
            # not check". If the version really is too old, the export itself
            # fails loudly at tokenizer load and is recorded there.
            result.limitations.append(version_check.detail)
        if version_check.outcome is Outcome.FAILED:
            # Known too old: the export would die inside a tokenizer load. That
            # is a fact about the environment, so the recipes are recorded as
            # not attempted rather than as failures.
            result.not_attempted = tuple(request.recipes)
            result.comparison = Uncompared(f"no recipe was exported: {version_check.detail}")
            result.limitations.append(
                f"none of {list(request.recipes)} was attempted: {version_check.detail}"
            )
            events.stage_finished(result.outcome.value, attempted=0)
            return result

    # -- the chat template the bundle carries -------------------------------
    # Only when the plan asks for it: a caller's own override satisfies the
    # rule (`RequiredFlag.satisfied_by`) and is passed through as given.
    composed: ComposedTemplate | None = None
    if request.composes_chat_template:
        with guard(CHAT_TEMPLATE_CHECK) as sink:
            check, composed = compose_chat_template(request)
            sink.append(check)
        if composed is None and sink[0].outcome is Outcome.PASSED:
            # A pass with nothing to hand the exporter is not a pass: no recipe
            # can run on it, and a stage with none run must not read as one.
            sink[0] = Check.unchecked(
                CHAT_TEMPLATE_CHECK,
                f"the composition passed and named no file ({sink[0].detail}); nothing was "
                "exported",
            )
        composition = result.checks.add(sink[0])
        events.check(composition)
        # One condition for both: the sweep runs on a passed check that came
        # with a file, which is also what every exporter argv needs.
        if composed is None or composition.outcome is not Outcome.PASSED:
            # Never an export with the marker, which the exporter would look up
            # as a repository, nor one without the block, which is the bundle
            # the rule exists to prevent. Not a verdict about any recipe, so
            # none is recorded as failed.
            result.not_attempted = tuple(request.recipes)
            result.comparison = Uncompared(f"no recipe was exported: {composition.detail}")
            result.limitations.append(
                f"none of {list(request.recipes)} was attempted: {composition.detail}"
            )
            events.stage_finished(result.outcome.value, attempted=0)
            return result
        result.chat_template = composed

    # -- the sweep ---------------------------------------------------------
    for recipe in request.recipes:
        events.note(f"exporting {recipe} on cpu", recipe=recipe, model=request.model)
        with guard(check_name(recipe)) as sink:
            export = export_recipe(request, recipe, composed)
        if sink:
            path = composed.path if composed is not None else None
            export = RecipeExport(
                recipe=recipe, argv=tuple(request.argv(recipe, path)), check=sink[0]
            )
        result.exports.append(export)
        result.checks.add(export.check)
        events.check(export.check)
        if export.artifact is not None:
            events.artifact(
                str(export.artifact),
                recipe=recipe,
                bytes=export.artifact_bytes,
                sha256=export.sha256,
                gpu_activation=export.gpu_activation,
                verified=False,
            )
            events.metric(f"{recipe} bytes", export.artifact_bytes, recipe=recipe)

    # A recipe whose definition could not be read never ran, and `not_attempted`
    # is where a reader looks for that. Without this it held only the recipes an
    # early return skipped, so a sweep could carry a recipe that was not
    # attempted and not listed as such.
    result.not_attempted = tuple(e.recipe for e in result.exports if not e.attempted)
    produced = [e.recipe for e in result.exports if e.ok]
    if len(produced) < len(request.recipes):
        missing = [r for r in request.recipes if r not in produced]
        result.limitations.append(
            f"{', '.join(missing)} produced no artifact, so any comparison below is over "
            f"{len(produced)} of the {len(request.recipes)} recipes requested"
        )
    result.comparison = compare_sizes(list(request.recipes), result.exports)
    if isinstance(result.comparison, Uncompared):
        result.limitations.append(result.comparison.reason)

    # Said at the top level, not only in the per-recipe record: a bundle left
    # at F16 runs on CPU and floods `<pad>` on GPU, and the two look the same
    # from every check this stage has.
    upstream = GpuActivationState.DECLARED_UPSTREAM
    other = [e for e in result.succeeded if e.gpu_state is upstream]
    if other:
        named = ", ".join(f"{e.recipe} = {e.gpu_activation}" for e in other)
        result.limitations.append(
            f"{GPU_ACTIVATION_KEY} was already declared and left as found: {named}. "
            "`fp16` reproduces the `<pad>` fault on GPU; `fp32_fp16` is what "
            "--experimental_use_mixed_precision writes and it also changes the graph, so the "
            "bundle is no longer the one the CPU figures describe"
        )
    unset = [e for e in result.succeeded if e.gpu_state is GpuActivationState.UNSET]
    if unset:
        named = ", ".join(f"{e.recipe} ({e.gpu_activation_note})" for e in unset)
        result.limitations.append(
            f"{GPU_ACTIVATION_KEY} could not be written into {named}. On the GPU "
            "backend the runtime will compute activations in F16, which measured on a "
            "Snapdragon Galaxy S24 as `<pad>` floods and wrong tool names on litert-lm 0.16.1 "
            "(3/20 vs 20/20 on CPU, n=20) and as 8/20 replies with no text and no call on "
            "0.18.0. An app that loads these bundles without an activation override gets that "
            "default until they are repacked; "
            "`verify --backend gpu` "
            "passes fp32 itself and can still measure them on a GPU"
        )

    if result.artifacts:
        # Said only where there is an artifact for it to be about: a run that
        # refused before exporting, or exported nothing usable, has none.
        result.limitations.insert(0, NOT_VERIFIED)
    events.stage_finished(
        result.outcome.value,
        exported=len(result.succeeded),
        requested=len(request.recipes),
        verified=False,
    )
    return result
