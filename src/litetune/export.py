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
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs, models
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.events import EventStream
from litetune.exits import read_returncode

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

KNOWN_RECIPES = (
    "dynamic_wi8_afp32",
    "weight_only_wi8_afp32",
    "dynamic_wi4_afp32",
    "weight_only_wi4_afp32",
)

# The two the README's table was produced with: same bit width, opposite ends
# of the accuracy result. A caller that wants a defensible minimum sweep wants
# these; it is offered, never applied by default -- see `ExportRequest.recipes`.
MEASURED_RECIPES = ("dynamic_wi8_afp32", "weight_only_wi8_afp32")

# What the GPU text executor computes activations in, unless the bundle says
# otherwise. Measured 2026-09-05 on a Galaxy S24 (Adreno) with the tuned
# FunctionGemma bundle, native tool path, 20 rows:
#
#     activation      <pad> floods   tool name   exact   ms/prompt
#     (unset -> F16)      14/20         3/20      2/20     16763
#     fp32                 0/20        20/20     15/20      1375
#     fp32_fp16            0/20        20/20     15/20      1406
#     CPU, same file       0/20        20/20     14/20      2477
#
# The engine reports success either way: it loads, every call returns, no
# exception. Only the content betrays it. Google's own Mobile Actions bundle
# does the same (20/20 `<pad>` on GPU) and the Gallery pins it to
# `"accelerators": "cpu"`. LiteRT-LM's `engine_settings.cc` names the default
# -- "Text executor defaults to F16" -- and the override is a string in the
# prefill/decode section's metadata that neither `export_hf` nor Google sets.
#
# `fp32` rather than `fp32_fp16`: the two measured the same, and `fp32` changes
# one metadata string while `--experimental_use_mixed_precision` also runs a
# graph pass. The artifact that leaves here is then byte-for-byte the one the
# CPU numbers were taken on, plus one key.
GPU_ACTIVATION = "fp32"
GPU_ACTIVATION_KEY = "prefer_activation_type"

# Measured: a 285,577,392-byte artifact in 122 s on CPU for a 270M checkpoint.
# The ceiling is an order of magnitude above that so a larger checkpoint is not
# cut off mid-write, and low enough that a hung export is recorded as a
# non-result the same day rather than occupying a runner overnight.
DEFAULT_TIMEOUT_S = 1800

# `pip show` is metadata collection, not model work; if it has not answered in
# a minute the environment is broken in a way the export itself will surface.
TOOLCHAIN_TIMEOUT_S = 60

# Export must not require an accelerator: the measurement above is CPU, and the
# artifact has to be reproducible on a runner that has no GPU. Hiding the
# devices makes that true by construction rather than by trusting the
# exporter's default device selection.
CPU_ONLY_ENV = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}

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
    return dict(os.environ) | CPU_ONLY_ENV


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
            "declared": list(self.declared),
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
        line = raw.strip()
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
            f"{_tail(proc.stderr) or 'no stderr'}",
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
        if self.externalize_embedder is True:
            requested.append("--externalize_embedder")
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
        source = "caller"
        if self.externalize_embedder is None:
            source = "litetune default"
            if not any(
                f == EXTERNALIZE_FLAG or f.startswith(f"{EXTERNALIZE_FLAG}=") for f in flags
            ):
                flags.append(EXTERNALIZE_FLAG)
        elif self.externalize_embedder is False:
            source = "caller opted out"
        object.__setattr__(self, "extra_flags", tuple(flags))
        object.__setattr__(self, "_externalize_source", source)
        object.__setattr__(self, "_plan", plan)

    @property
    def identity_conflict(self) -> str | None:
        """Why `--base-model` and the checkpoint's own record cannot both be right."""
        return self._identity_conflict  # type: ignore[attr-defined]

    @property
    def externalize_source(self) -> str:
        """Who decided `--externalize_embedder`: the caller, or this default."""
        return self._externalize_source  # type: ignore[attr-defined]

    @property
    def plan(self) -> models.ExportPlan:
        """What `litetune.models` said about this model, including added flags."""
        return self._plan  # type: ignore[attr-defined]

    def dir_for(self, recipe: str) -> Path:
        """One output directory per recipe: the sweep's artifacts must not collide."""
        return self.output_dir / recipe

    def argv(self, recipe: str) -> list[str]:
        return [
            "litert-torch",
            "export_hf",
            f"--model={self.model}",
            f"--output_dir={self.dir_for(recipe)}",
            f"--quantization_recipe={recipe}",
            *self.extra_flags,
        ]

    @property
    def unknown_recipes(self) -> tuple[str, ...]:
        """Requested recipes litetune has no measurement for. Not an error."""
        return tuple(r for r in self.recipes if r not in KNOWN_RECIPES)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_model": self.base_model,
            "output_dir": str(self.output_dir),
            "recipes": list(self.recipes),
            "externalize_embedder": EXTERNALIZE_FLAG in self.extra_flags,
            "externalize_embedder_source": self.externalize_source,
            "flags": list(self.extra_flags),
            "model_rules": self.plan.as_dict(),
            "timeout_s": self.timeout_s,
            "environment": {
                "name": self.env.name,
                "identity": self.env.identity,
                "path": str(self.env.path),
                "requirements": list(self.env.requirements),
                "system_requirements": list(self.env.system_requirements),
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
    # What the bundle tells the GPU text executor to compute activations in, or
    # None when the repack could not be made and the toolchain's F16 default
    # stands. See GPU_ACTIVATION.
    gpu_activation: str | None = None
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


# One `[[section]]` table, from its header to the next header or end of file.
_TOML_TABLE_RE = re.compile(r"\[\[section\]\][^\[]*(?:\[(?!\[section\])[^\[]*)*", re.S)
_PREFILL_RE = re.compile(r'model_type\s*=\s*"prefill_decode"')
# Both shapes the builder round-trips the key in. Written by hand it is a bare
# key; read back by `unpack` it is one entry of the section's `additional_metadata`
# array, which is also what the file actually stores.
_ACTIVATION_VALUE_RE = re.compile(
    r'(?:^\s*prefer_activation_type\s*=\s*"([^"]*)"'
    r'|key\s*=\s*"prefer_activation_type"[^}]*?value\s*=\s*"([^"]*)")',
    re.M,
)


def set_gpu_activation(
    artifact: Path, env: envs.StageEnv, *, activation: str = GPU_ACTIVATION, timeout: int = 600
) -> tuple[bool, str | None]:
    """Write `prefer_activation_type` into the bundle's prefill/decode section.

    Returns (ok, what happened). `ok` is False when the artifact is left as the
    toolchain wrote it, which is a working CPU bundle and a broken GPU one -- so
    the caller records the reason rather than failing the export.

    Done by unpacking with `litert-lm-builder`, adding one key to `model.toml`,
    and rebuilding: the weights, tokenizer, template and model type are carried
    through untouched, which is checked by reading the result back with
    `litert-lm-peek`. The original is replaced only after that check passes.
    The exporter offers no flag for this value, and the one flag it does offer
    (`--experimental_use_mixed_precision`) changes the graph as well.
    """
    work = artifact.parent / f".{artifact.stem}-repack"
    if work.exists():
        _remove_tree(work)
    work.mkdir(parents=True)
    try:
        proc = env.run(
            ["litert-lm-builder", "unpack", "--input", str(artifact), "--output", str(work)],
            timeout=timeout,
        )
        if proc.returncode != 0:
            return False, f"litert-lm-builder unpack exited {proc.returncode}: {_tail(proc.stderr)}"
        toml_path = work / "model.toml"
        if not toml_path.exists():
            return False, f"litert-lm-builder unpack wrote no model.toml into {work}"
        text = toml_path.read_text(encoding="utf-8")
        tables = [m for m in _TOML_TABLE_RE.finditer(text) if _PREFILL_RE.search(m.group(0))]
        if len(tables) != 1:
            return False, (
                f"found {len(tables)} prefill_decode sections in model.toml where exactly one "
                "was expected, so the GPU activation type was not set"
            )
        table = tables[0]
        present = _ACTIVATION_VALUE_RE.search(table.group(0))
        if present:
            value = present.group(1) or present.group(2)
            return True, (
                f"the bundle already declares {GPU_ACTIVATION_KEY} = {value!r}; left as written"
            )
        # Key order inside a TOML table is free, so the key goes straight after
        # the header rather than after any particular line.
        header_end = table.start() + len("[[section]]")
        patched = text[:header_end] + f'\n{GPU_ACTIVATION_KEY} = "{activation}"' + text[header_end:]
        toml_path.write_text(patched, encoding="utf-8")
        rebuilt = work / artifact.name
        proc = env.run(
            [
                "litert-lm-builder",
                "toml",
                "--path",
                str(toml_path),
                "output",
                "--path",
                str(rebuilt),
            ],
            timeout=timeout,
        )
        if proc.returncode != 0 or not rebuilt.exists():
            return False, (
                f"litert-lm-builder rebuild exited {proc.returncode}: {_tail(proc.stderr)}"
            )
        proc = env.run(["litert-lm-peek", "--litertlm_file", str(rebuilt)], timeout=timeout)
        peek = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 or GPU_ACTIVATION_KEY not in peek or activation not in peek:
            return False, (
                f"the rebuilt bundle does not read back with {GPU_ACTIVATION_KEY} = "
                f"{activation!r}; the original was kept"
            )
        os.replace(rebuilt, artifact)
        return True, None
    except subprocess.TimeoutExpired:
        return False, f"litert-lm-builder did not finish within {timeout}s; the original was kept"
    except Exception as exc:  # noqa: BLE001 - a broken repack must not unmake a good export
        # Whatever went wrong here, the toolchain's artifact is intact and is a
        # working CPU bundle. Escaping would let `guard` record the whole
        # recipe as "could not check", which is a smaller truth than the one
        # available: exported, CPU-only, and here is why.
        return False, f"{type(exc).__name__}: {exc}; the original was kept"
    finally:
        _remove_tree(work)


def _remove_tree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            child.rmdir() if child.is_dir() else child.unlink()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def export_recipe(request: ExportRequest, recipe: str) -> RecipeExport:
    """Run one export. A non-zero exit is recorded, not raised."""
    name = check_name(recipe)
    argv = tuple(request.argv(recipe))
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
    gpu_ok, gpu_note = set_gpu_activation(artifact, request.env)
    gpu_activation = GPU_ACTIVATION if gpu_ok else None
    artifact_bytes = artifact.stat().st_size
    companions = tuple(p.name for p in produced if p != artifact)
    shipped_bytes = sum(p.stat().st_size for p in produced)
    digest = _sha256(artifact) if request.hash_artifacts else None
    gpu_text = (
        f"GPU activations {gpu_activation}"
        if gpu_ok
        else "GPU activations left at the toolchain's F16 default"
    )
    return RecipeExport(
        **base,
        artifact=artifact,
        companions=companions,
        artifact_bytes=artifact_bytes,
        shipped_bytes=shipped_bytes,
        sha256=digest,
        gpu_activation=gpu_activation,
        check=Check.passed(
            name,
            f"{artifact.name}: {artifact_bytes:,} bytes in {seconds:.1f}s on cpu, {gpu_text} — "
            "produced, not verified",
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
            "unverified_reason": NOT_VERIFIED,
            "outcome": self.outcome.value,
            "request": self.request.as_dict(),
            "toolchain": self.toolchain.as_dict(),
            "exports": [e.as_dict() for e in self.exports],
            "not_attempted": list(self.not_attempted),
            "comparison": self.comparison.as_dict(),
            "checks": self.checks.as_dict(),
            "limitations": list(self.limitations),
            "recommendations": list(self.recommendations),
            "model_rules": self.request.plan.as_dict(),
        }


def run_export(request: ExportRequest, events: EventStream | None = None) -> ExportResult:
    """Sweep the requested recipes. Returns a result; failures are recorded, not raised."""
    events = events or EventStream(echo_json=False)
    events.stage_started("export", model=request.model, recipes=list(request.recipes))
    result = ExportResult(
        request=request,
        checks=CheckSet(name=f"export:{request.model}"),
        toolchain=Toolchain.unknown("not read yet", request.env.requirements),
    )
    result.limitations.append(NOT_VERIFIED)

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

    # Before the toolchain is asked anything, and once for the sweep rather than
    # once per recipe. A stale `vocab_file` fails the export outright, after the
    # model has loaded, naming a path that never existed here.
    #
    # Reported and not merely logged: this writes to the *input* directory, and
    # a tool that edits what you handed it must say so where you will see it.
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

    model_path = Path(request.model)
    if model_path.is_dir():
        ok, note = repair_vocab_file(model_path)
        if note:
            events.note(note, model=request.model)
            result.limitations.append(note)
        if not ok:
            result.limitations.append(
                "the export below is likely to fail at `export_tokenizer` for the reason above"
            )
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
                        "requirements": list(request.env.requirements),
                    },
                )
            )
        else:
            sink.append(
                Check.unchecked(
                    "export environment",
                    f"environment {request.env.name!r} is not provisioned at {request.env.path}",
                    observed={"name": request.env.name, "identity": request.env.identity},
                )
            )
    environment = result.checks.add(sink[0])
    events.check(environment)
    if not environment.conclusive:
        # Nothing was attempted, so nothing failed. Recording these recipes as
        # failures would be a verdict about the toolchain drawn from a fact
        # about this machine.
        result.not_attempted = tuple(request.recipes)
        result.toolchain = Toolchain.unknown(
            f"the export environment was not usable, so its versions were never read: "
            f"{environment.detail}",
            request.env.requirements,
        )
        result.comparison = Uncompared(f"no recipe was exported: {environment.detail}")
        result.limitations.append(
            f"none of {list(request.recipes)} was attempted: {environment.detail}"
        )
        events.stage_finished(result.outcome.value, attempted=0)
        return result

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

    # -- the sweep ---------------------------------------------------------
    for recipe in request.recipes:
        events.note(f"exporting {recipe} on cpu", recipe=recipe, model=request.model)
        with guard(check_name(recipe)) as sink:
            export = export_recipe(request, recipe)
        if sink:
            export = RecipeExport(recipe=recipe, argv=tuple(request.argv(recipe)), check=sink[0])
        result.exports.append(export)
        result.checks.add(export.check)
        events.check(export.check)
        if export.artifact is not None:
            events.artifact(
                str(export.artifact),
                recipe=recipe,
                bytes=export.artifact_bytes,
                sha256=export.sha256,
                verified=False,
            )
            events.metric(f"{recipe} bytes", export.artifact_bytes, recipe=recipe)

    result.comparison = compare_sizes(list(request.recipes), result.exports)
    if isinstance(result.comparison, Uncompared):
        result.limitations.append(result.comparison.reason)

    # Said at the top level, not only in the per-recipe record: a bundle left
    # at F16 runs on CPU and floods `<pad>` on GPU, and the two look the same
    # from every check this stage has.
    unset = [e.recipe for e in result.succeeded if e.gpu_activation is None]
    if unset:
        result.limitations.append(
            f"{GPU_ACTIVATION_KEY} could not be written into {unset}: on the GPU backend "
            f"the runtime will compute activations in F16, which measured as `<pad>` floods "
            f"and wrong tool names on Adreno (3/20 vs 20/20 on CPU). These bundles are "
            f"CPU-only until repacked; see the per-recipe gpu_activation_note"
        )

    events.stage_finished(
        result.outcome.value,
        exported=len(result.succeeded),
        requested=len(request.recipes),
        verified=False,
    )
    return result
