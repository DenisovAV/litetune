"""`bundle`: the deliverable. A model file on its own is not one.

A bundle is **model + declarations + contract + report**, and the four travel
together because three of them are unrecoverable from the first.

**The contract records the prompt-rendering mode.** This is the field the module
exists for. A tool-calling model is trained against either runtime-rendered
declarations -- the serving runtime applies its own chat template and injects the
tool list -- or application-rendered ones, where the application builds the whole
prompt and the runtime is told not to touch it (`--no-template`, which forces the
runtime's tool list to null). Both modes are in the field simultaneously, the two
prompts differ by the entire declaration block, and **a runtime cannot infer
which one a model expects**: it will render the wrong prompt, the model will
answer fluently, and the answer will be wrong. `evaluate.harness_mismatch`
already refuses to *compare* two measurements taken in different modes; this
records which mode the artifact was built for so the question does not arise at
serving time. An unrecorded mode is `MissingRenderingMode`, never a default --
defaulting it is choosing one of two incompatible calling conventions on the
user's behalf and not telling them.

**A mode is established against versions, not in the abstract.** Which template a
runtime applies is a property of that runtime's release, so the contract carries
the pinned versions the mode was established against. An empty set is refused for
the same reason `envs.StageEnv` refuses an unpinned requirement: a claim that
resolves differently over time is not a record.

**A non-passing run still bundles.** The manifest and the report are written for
a failed, inconclusive or unmeasured run exactly as for a passing one, and they
name which of the three measurement points were not taken. A run that produced no
bundle produces no explanation either, and the explanation is the thing someone
needs at the point they find out something is wrong.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.checks import Check, CheckSet, Outcome
from litetune.evaluate import PromptMode
from litetune.events import EventStream
from litetune.manifest import RunManifest, RunStatus
from litetune.metrics import Unavailable
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA = "litetune.bundle/1"
CONTRACT_SCHEMA = "litetune.contract/1"

# The three points the README reports, and what each one is for. A bundle that
# is missing one says which, because "the fine-tune gained X" and "the conversion
# cost Y" are separate numbers and a single net figure cannot tell you which
# stage to fix.
MEASUREMENT_POINTS: dict[str, str] = {
    "base_float": (
        "the untuned base model in float, which is what a training gain is measured against"
    ),
    "tuned_float": (
        "the fine-tuned checkpoint in float, which is the float twin of the shipped artifact"
    ),
    "tuned_converted": ("the converted artifact itself, which is what actually runs on the device"),
}

CONTRACT_NAME = "contract.json"
REPORT_NAME = "report.json"
MANIFEST_NAME = "manifest.json"
DECLARATIONS_NAME = "declarations.json"
MODEL_DIR_NAME = "model"
ADAPTER_DIR_NAME = "adapter"

NOT_VERIFIED = (
    "this bundle was assembled from what the run recorded; assembling it establishes nothing "
    "about the model. Read `report.json` for which measurements were taken and which were not."
)

NO_MANIFEST = (
    "no run manifest was supplied to the bundle stage, so this file records only what the bundle "
    "stage itself observed: which files were packaged, not what produced them or what was measured"
)


class BundleError(ValueError):
    """The bundle could not be described. Not a statement about any model."""


class ContractError(BundleError):
    """The serving contract is incomplete. The message names the field."""


class MissingRenderingMode(ContractError):
    """No prompt-rendering mode was recorded.

    Separate from `ContractError` so that a caller can catch exactly this: it is
    the one field with no defensible default, because guessing it means guessing
    which of two mutually exclusive calling conventions the model was trained
    for, and the wrong guess is a fluent wrong answer rather than an error.
    """


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def versions_from(*stage_envs: envs.StageEnv) -> dict[str, str]:
    """`{name: version}` from pinned requirements, for `established_against`.

    Reads the *declaration*, which is what is available before anything runs.
    `export.resolve_toolchain` reads the resolved closure from a live
    environment and is strictly better; pass that when a run produced one.
    """
    versions: dict[str, str] = {}
    for env in stage_envs:
        for requirement in env.requirements:
            name, sep, version = requirement.partition("==")
            if sep:
                versions[name.strip()] = version.strip()
    return versions


class WireConvention(str, Enum):
    """The order a tool declaration's properties are rendered in.

    Two conventions are in the field for the same model. `flutter_gemma` renders
    them in declaration order; the jinja template inside the `.litertlm` renders
    them with `dictsort`. They disagree for 100% of `google/mobile-actions` rows,
    so one bundle presents two different prompts depending on which path a
    consumer takes.

    It is not a stylistic choice. Measured on the base checkpoint twice, on
    disjoint samples with different batching: declaration order scores 0.7266
    against dictsort's 0.7625 (n=640, paired -0.0359 +/-0.0219) and 0.7602
    against 0.7789 (n=1280, paired -0.0187 +/-0.0099). Both resolve, both favour
    dictsort, and the failure is not a reordered call -- argument dicts compare
    without regard to key order -- but a *wrong argument*: given a declaration in
    the order it did not learn, the model returns `email` where the target
    wanted `phone_number`.

    `declarations_sha256` cannot carry this. The same declarations rendered two
    ways hash identically, so without this field a bundle has no way to say
    which one it was trained under, and a consumer no way to match it.
    """

    DECLARATION_ORDER = "declaration_order"
    TEMPLATE_DICTSORT = "template_dictsort"


@dataclass(frozen=True)
class Contract:
    """How this model must be called. Not optional, and not inferable at runtime.

    `prompt_mode` is `PromptMode` rather than a new enum on purpose: the same
    value has to mean the same thing in a measurement (`evaluate.MeasurementPoint`)
    and in a shipped artifact, or a bundle can promise a mode nothing was ever
    measured in.
    """

    prompt_mode: PromptMode
    established_against: Mapping[str, str]
    base_model: str
    base_model_revision: str
    declarations_sha256: str | None = None
    context_length: int | None = None
    stop_tokens: Sequence[str] = ()
    # `None` means unrecorded, not "the usual one". A default here would be a
    # guess costing a measured, resolved 0.019-0.036 exact match when wrong.
    wire_convention: WireConvention | None = None
    notes: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.prompt_mode is None:
            raise MissingRenderingMode(
                "contract.prompt_mode is required. A runtime cannot infer whether this model "
                "expects the tool declarations to be rendered by the runtime "
                f"({PromptMode.RUNTIME_RENDERED.value}) or to arrive already in the prompt "
                f"({PromptMode.PRERENDERED.value}); both modes are in the field, the two prompts "
                "differ by the whole declaration block, and calling a model in the wrong one "
                "produces a fluent wrong answer rather than an error"
            )
        if not isinstance(self.prompt_mode, PromptMode):
            raise MissingRenderingMode(
                f"contract.prompt_mode must be a PromptMode, got {self.prompt_mode!r}. "
                f"Known modes: {[m.value for m in PromptMode]}"
            )
        if not self.established_against:
            raise ContractError(
                "contract.established_against is empty. Which prompt a runtime renders is a "
                "property of that runtime's release, so a mode recorded against no versions is a "
                "claim that resolves differently over time -- the same failure envs.StageEnv "
                "refuses an unpinned requirement for. Use bundle.versions_from(envs.RUNTIME) or "
                "the resolved closure from the run"
            )
        if not self.base_model or not self.base_model_revision:
            raise ContractError(
                "contract.base_model and contract.base_model_revision are required: a bundle "
                "whose starting weights are unrecorded cannot be reproduced or diffed against a "
                "later one"
            )
        object.__setattr__(self, "established_against", dict(self.established_against))
        object.__setattr__(self, "stop_tokens", tuple(self.stop_tokens))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def runtime_renders_declarations(self) -> bool:
        return self.prompt_mode is PromptMode.RUNTIME_RENDERED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_SCHEMA,
            "prompt_mode": self.prompt_mode.value,
            "prompt_mode_meaning": (
                "the serving runtime applies its own chat template and renders the tool "
                "declarations into the prompt"
                if self.runtime_renders_declarations
                else "the application renders the declarations into the prompt and the runtime "
                "must not apply a template of its own (litert-lm: --no-template)"
            ),
            "established_against": dict(self.established_against),
            "base_model": self.base_model,
            "base_model_revision": self.base_model_revision,
            "declarations_sha256": self.declarations_sha256,
            "context_length": self.context_length,
            "stop_tokens": list(self.stop_tokens),
            "wire_convention": (
                self.wire_convention.value if self.wire_convention is not None else None
            ),
            "wire_convention_meaning": (
                {
                    WireConvention.DECLARATION_ORDER: "declaration properties are rendered in the "
                    "order they were declared, as flutter_gemma renders them",
                    WireConvention.TEMPLATE_DICTSORT: "declaration properties are rendered "
                    "case-insensitively sorted by name, as the model's jinja template renders "
                    "them",
                }[self.wire_convention]
                if self.wire_convention is not None
                else "unrecorded: this bundle does not say which property order it was built "
                "under, and the two in the field disagree for every declaration with more than "
                "one property"
            ),
            "notes": list(self.notes),
        }

    @classmethod
    def read(cls, data: Mapping[str, Any]) -> Contract:
        """Build from a mapping -- a config file, a manifest, a CLI namespace.

        A missing or unrecognised `prompt_mode` raises rather than defaulting.
        """
        if not isinstance(data, Mapping):
            raise ContractError(f"contract: expected a mapping, got {type(data).__name__}")
        raw = data.get("prompt_mode")
        if raw is None:
            raise MissingRenderingMode(
                "contract.prompt_mode was not recorded. It has no default: a runtime cannot infer "
                "whether declarations are rendered by it or arrive already in the prompt, and the "
                f"wrong choice is silent. Set it to one of {[m.value for m in PromptMode]}"
            )
        try:
            mode = raw if isinstance(raw, PromptMode) else PromptMode(str(raw))
        except ValueError as exc:
            raise MissingRenderingMode(
                f"contract.prompt_mode {raw!r} is not a known mode; expected one of "
                f"{[m.value for m in PromptMode]}"
            ) from exc
        return cls(
            prompt_mode=mode,
            established_against=data.get("established_against") or {},
            base_model=str(data.get("base_model") or ""),
            base_model_revision=str(data.get("base_model_revision") or ""),
            declarations_sha256=data.get("declarations_sha256"),
            context_length=data.get("context_length"),
            stop_tokens=data.get("stop_tokens") or (),
            wire_convention=(
                WireConvention(data["wire_convention"]) if data.get("wire_convention") else None
            ),
            notes=data.get("notes") or (),
        )


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleRequest:
    """One deliverable.

    `status` is the run's status, carried in rather than derived: bundling is
    the last stage and it cannot re-measure anything, so a bundle that decided
    its own verdict would be deciding it from the fact that files were copied.
    """

    output_dir: Path
    model: Path
    declarations: Path
    contract: Contract
    # The LoRA weights, when the run produced them. Optional because a full
    # fine-tune has none -- not because they are optional to keep.
    #
    # `merge_and_unload()` is one-way: once the adapter is folded into the base
    # weights there is no way back to it, and the merged checkpoint is what
    # `convert` is pointed at. An adapter that stays in the training output
    # directory is an artifact of a directory someone will delete, so the
    # deliverable carries it or it is gone.
    adapter: Path | None = None
    status: RunStatus = RunStatus.INCONCLUSIVE
    # Point name -> whatever the measurement stage recorded. Absent points are
    # reported by name in the report; see `MEASUREMENT_POINTS`.
    measurements: Mapping[str, Any] = field(default_factory=dict)
    attribution: Mapping[str, Any] = field(default_factory=dict)
    manifest: RunManifest | None = None
    limitations: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "model", Path(self.model))
        object.__setattr__(self, "declarations", Path(self.declarations))
        if self.adapter is not None:
            # The one path field that was left out, four lines from its siblings.
            # A `str` reached `build_bundle` and raised `AttributeError` from
            # `.exists()`, breaking the "never raises for a missing input"
            # contract this module states.
            object.__setattr__(self, "adapter", Path(self.adapter))
        object.__setattr__(self, "measurements", dict(self.measurements))
        object.__setattr__(self, "attribution", dict(self.attribution))
        object.__setattr__(self, "limitations", tuple(self.limitations))
        unknown = sorted(set(self.measurements) - set(MEASUREMENT_POINTS))
        if unknown:
            raise BundleError(
                f"unknown measurement point(s) {unknown}; known points are "
                f"{sorted(MEASUREMENT_POINTS)}. A point under a name nothing else uses is a "
                "measurement no report will find"
            )


@dataclass(frozen=True)
class Member:
    """One file in the bundle, identified by its content."""

    name: str
    path: Path
    bytes: int
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "bytes": self.bytes,
            "content_sha256": self.content_sha256,
        }


@dataclass
class BundleResult:
    """What was packaged, and what the package does not establish."""

    request: BundleRequest
    # Packaging only: is every required member in the bundle. Deliberately does
    # not include measurement coverage -- "the run measured two of three points"
    # is a fact about the run, and folding it in here would make an incomplete
    # measurement read as a bundle litetune could not assemble, overwriting a
    # perfectly good `failed_gate` with `failed_harness`.
    checks: CheckSet
    # ... which is recorded here instead, three-valued and reported in full.
    coverage: Check | None = None
    members: list[Member] = field(default_factory=list)
    missing_measurements: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    report_path: Path | None = None
    manifest_path: Path | None = None

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def complete(self) -> bool:
        """Whether every required member is present. Not a claim about quality."""
        return self.outcome is Outcome.PASSED

    @property
    def verified(self) -> bool:
        """Always False. A property, not a field: packaging measures nothing."""
        return False

    @property
    def status(self) -> RunStatus:
        """The run's status, downgraded if the bundle itself is incomplete.

        An incomplete bundle is `failed_harness`, not `failed`: litetune could
        not assemble the deliverable, which says nothing about the model in it.
        """
        if self.outcome is Outcome.PASSED:
            return self.request.status
        return RunStatus.FAILED_HARNESS

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SCHEMA,
            "verified": False,
            "unverified_reason": NOT_VERIFIED,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "complete": self.complete,
            "contract": self.request.contract.as_dict(),
            "members": [m.as_dict() for m in self.members],
            "measurements": dict(self.request.measurements),
            "measurements_not_made": [
                {"point": name, "reason": MEASUREMENT_POINTS[name]}
                for name in self.missing_measurements
            ],
            "measurement_coverage": self.coverage.as_dict() if self.coverage else None,
            "attribution": dict(self.request.attribution),
            "checks": self.checks.as_dict(),
            "limitations": list(self.limitations),
        }


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------


def _clear(path: Path) -> None:
    """Remove a scratch path, and say so if anything is left behind.

    `ignore_errors=True` alone was the one place in this package where a failure
    was swallowed with no record -- inside the function that had just been
    rewritten from an invariant, which undercuts the argument the rewrite makes.
    A leftover here is not data loss (the invariant covers that); it is a
    surprise for the *next* run, which is exactly the kind of thing that should
    be in the log when that run fails.
    """
    if path.is_symlink():
        # Before `exists()`, which is False for a dangling one, and before
        # `is_dir()`, which is True for a symlink *to* a directory -- where
        # `rmtree` refuses and leaves the link in place. Removing the link
        # never touches its target.
        path.unlink(missing_ok=True)
    elif not path.exists():
        return
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        # A previous bundle whose model was a single file named `model`.
        path.unlink(missing_ok=True)
    if path.exists():
        logger.warning("could not remove %s; a later bundle in this directory may fail on it", path)


def _replace_directory(source: Path, destination: Path, what: str) -> Path:
    """Put `source` at `destination`, replacing whatever is there. Never loses either.

    Replace, never merge: `dirs_exist_ok=True` over a previous bundle left files
    the new tree no longer has, and `_directory_members` then hashed them and
    reported them as members of this one -- a manifest accurate about content
    that never existed.

    The obvious way to write that replace is to remove the destination and then
    copy, and it deletes the source whenever the two overlap. That has to be
    refused in *both* directions: a first attempt rejected
    `--model ./b/model --output-dir ./b` and still accepted
    `--model ./b --output-dir ./b`, which destroyed `./b/model` just the same.
    Rebuilding a bundle in place is a plausible command; losing the checkpoint
    to it is data loss, not a bad error message.

    The invariant, stated before the code because three attempts at this were
    written to fit a reproduction instead and each one lost data:

      at every moment the tree exists in at least one of `destination`,
      `previous` or `source`, and nothing is removed until a copy is known to
      be somewhere else.

    From that, everything follows: `staging` may be discarded whenever (it is a
    copy of `source`, which still exists); `previous` may be discarded only
    after the swap has succeeded; and the overlap check exists because `source`
    and `destination` being the same directory makes the first clause
    unsatisfiable.

    This is a function and not a paragraph of advice because it was a paragraph
    of advice: `_copy_adapter` was written twelve lines below the comment above
    and did `_clear` then `copytree`, so bundling a LoRA run's own output
    directory deleted the adapter and then reported that it "could not be
    copied". A safe replace has to be the thing you call, not the thing you
    read.
    """
    resolved_source = source.resolve()
    resolved_destination = destination.resolve()
    if (
        resolved_source == resolved_destination
        or resolved_destination in resolved_source.parents
        or resolved_source in resolved_destination.parents
    ):
        raise BundleError(
            f"the {what} at {source} overlaps the bundle directory it would be copied into "
            f"({destination}); name a source outside --output-dir"
        )

    # The pid in the names is not decoration: a fixed staging path is shared
    # state, and a second writer starting mid-copy would remove the first one's
    # tree.
    parent = destination.parent
    staging = parent / f".{destination.name}.incoming.{os.getpid()}"
    previous = parent / f".{destination.name}.previous.{os.getpid()}"
    try:
        # Both scratch paths, not just one: a `previous` left by an earlier run
        # -- same process, or a reused pid -- makes the rename below fail with
        # "Directory not empty" two runs after the event that caused it, naming
        # an internal path and nothing actionable.
        _clear(staging)
        _clear(previous)
        shutil.copytree(source, staging)
        moved_aside = False
        if destination.exists():
            destination.rename(previous)
            moved_aside = True
        try:
            staging.rename(destination)
        except OSError:
            if moved_aside:
                try:
                    previous.rename(destination)
                except OSError as restore_failed:
                    # Both renames failed. The tree is still on disk and the
                    # only thing that must not happen now is the operator not
                    # being told where -- so this path returns without touching
                    # `previous`.
                    raise BundleError(
                        f"could not install the new {what} at {destination}, and could not "
                        f"restore the previous one; it is intact at {previous} and must be "
                        "moved back by hand"
                    ) from restore_failed
            raise
        # Only now is the old copy redundant. A `finally` that removed it
        # unconditionally deleted the directory the message above had just told
        # the operator to recover by hand.
        if moved_aside:
            _clear(previous)
    finally:
        _clear(staging)
    return destination


def _copy_model(source: Path, output_dir: Path) -> Path:
    """Copy the artifact in, whether it is one `.litertlm` file or a checkpoint.

    The bundle owns its copy. A bundle that points at a path outside itself is
    an artifact whose model can be replaced without changing the bundle, which
    is the location-versus-content confusion `storage.py` exists to prevent.
    """
    if source.is_dir():
        destination = _replace_directory(source, output_dir / MODEL_DIR_NAME, "model")
        _clear_previous_model(output_dir, keep=destination)
        return destination
    # A single-file artifact keeps its own name, so rebuilding a bundle with a
    # differently-named model left the previous one beside the new one -- in the
    # directory, absent from `report.json`'s member list. The directory branch
    # above replaces; this one has to as well, and it can only know what to
    # replace by reading what the last build recorded.
    destination = output_dir / source.name
    # Copy first, then remove: the same invariant the directory branch above
    # spells out. Removing the recorded model before the copy meant a failure
    # partway -- a full disk, a source that vanished -- left the bundle without
    # the old artifact and without the new one.
    shutil.copyfile(source, destination)
    _clear_previous_model(output_dir, keep=destination)
    return destination


def _clear_previous_model(output_dir: Path, keep: Path) -> None:
    """Remove whatever the last build recorded as its model, except `keep`.

    Both shapes, not one. "Replace, never merge" held for file->file only: a
    directory rebuilt as a file left the old `model/` behind, and a file
    rebuilt as a directory left the old `.litertlm` beside it -- present in the
    bundle, absent from `report.json`'s member list, in the module whose thesis
    is that its members travel together and are identified by content.

    Top-level names, because a directory member is recorded as `model/<file>`
    and clearing those one at a time would leave an empty `model/`.
    """
    root = output_dir.resolve()
    kept = keep.resolve()
    for recorded in _previously_recorded_model(output_dir):
        top = recorded
        while top.parent != root:
            top = top.parent
        if top != kept:
            _clear(top)


def _previously_recorded_model(output_dir: Path) -> list[Path]:
    """The model member the last build in this directory wrote, if any.

    Read from the bundle's own report rather than guessed from the extension: a
    bundle names its members, and a rebuild that removed files by pattern would
    be deleting things it had not established were its own.

    But a report on disk is the *least* trustworthy input in this module -- this
    function did not write it, and `report.json` is not an exotic filename to
    find in a directory someone points `--output-dir` at. The first version
    joined member names to `output_dir` and deleted the result, so `../x` left
    the bundle entirely and took a file litetune had never written. Every path
    is therefore resolved and required to be inside the directory, exactly as
    `_copy_model` requires of `--model`.
    """
    report = output_dir / REPORT_NAME
    if not report.is_file():
        return []
    try:
        recorded = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Not "no previous model": that is the stale-artifact bug this exists to
        # fix, returning silently. Say what could not be read.
        logger.warning(
            "could not read %s to find the previous model (%s); a stale artifact may remain",
            report,
            exc,
        )
        return []
    members = recorded.get("members")
    if not isinstance(members, list):
        return []

    root = output_dir.resolve()
    own = {CONTRACT_NAME, MANIFEST_NAME, REPORT_NAME, DECLARATIONS_NAME}
    inside: list[Path] = []
    for member in members:
        if not isinstance(member, dict):
            continue
        name = member.get("name")
        if not isinstance(name, str) or name in own:
            continue
        candidate = (output_dir / name).resolve()
        if root not in candidate.parents:
            logger.warning("%s names a member outside the bundle (%s); ignoring it", report, name)
            continue
        inside.append(candidate)
    return inside


def _directory_members(name: str, directory: Path) -> list[Member]:
    return [
        Member(
            name=f"{name}/{path.relative_to(directory).as_posix()}",
            path=path,
            bytes=path.stat().st_size,
            content_sha256=hash_file(path),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _member(name: str, path: Path) -> Member:
    return Member(
        name=name,
        path=path,
        bytes=path.stat().st_size,
        content_sha256=hash_file(path),
    )


def _bundle_manifest(request: BundleRequest, result: BundleResult) -> RunManifest:
    """The manifest that ships. Synthesised when the run did not supply one."""
    if request.manifest is not None:
        return request.manifest
    result.limitation(NO_MANIFEST)
    return RunManifest(
        run_id=f"bundle-{request.output_dir.name}",
        spec_hash=Unavailable(NO_MANIFEST),
        scope="stage",
        spec=Unavailable(NO_MANIFEST),
        limitations=[NO_MANIFEST],
    )


def build_bundle(request: BundleRequest, events: EventStream | None = None) -> BundleResult:
    """Assemble the deliverable. Writes a report and a manifest whatever happened.

    Never raises for a missing input: a model file that is not there is recorded
    as a failed check and the bundle is still written, because the run that went
    wrong is exactly the one whose explanation someone needs.

    It does raise `BundleError` for a request it will not carry out at all -- a
    model path overlapping the output directory, or a swap it could neither
    finish nor undo. Those are refusals rather than findings: there is no
    bundle to describe, and in the second case the message names where the
    checkpoint survived.
    """
    events = events or EventStream(echo_json=False)
    events.stage_started(
        "bundle",
        output_dir=str(request.output_dir),
        prompt_mode=request.contract.prompt_mode.value,
    )
    result = BundleResult(
        request=request, checks=CheckSet(name=f"bundle:{request.output_dir.name}")
    )
    result.limitation(NOT_VERIFIED)
    for text in request.limitations:
        result.limitation(text)

    request.output_dir.mkdir(parents=True, exist_ok=True)

    # -- the model ---------------------------------------------------------
    if not request.model.exists():
        model_check = Check.failed(
            "model included",
            f"{request.model} does not exist, so the bundle has no model in it",
            observed={"model": str(request.model)},
        )
    else:
        try:
            copied = _copy_model(request.model, request.output_dir)
        except (OSError, shutil.Error) as exc:
            logger.exception("could not copy the model into %s", request.output_dir)
            model_check = Check.failed(
                "model included",
                f"the model at {request.model} could not be copied into the bundle: "
                f"{type(exc).__name__}: {exc}",
                observed={"model": str(request.model)},
            )
        else:
            members = (
                _directory_members(MODEL_DIR_NAME, copied)
                if copied.is_dir()
                else [_member(copied.name, copied)]
            )
            result.members.extend(members)
            total = sum(m.bytes for m in members)
            model_check = Check.passed(
                "model included",
                f"{len(members)} file(s), {total:,} bytes at {copied}",
                observed={"files": len(members), "bytes": total},
            )
    result.checks.add(model_check)
    events.check(model_check)

    # -- the adapter, when the run produced one ----------------------------
    # `merge_and_unload()` is one-way. The merged checkpoint is what `convert`
    # is pointed at, and nothing recovers the adapter from it, so a deliverable
    # that leaves the LoRA weights in a training directory is a deliverable that
    # loses them the first time someone cleans up.
    #
    # **After the model copy, not before.** `_copy_model` clears every member the
    # last build recorded except the one it is writing, and the adapter is such a
    # member -- so a rebuild removes the old adapter and this re-copies it. Run
    # the other way round, this would copy an adapter and then delete it.
    # `test_a_rebuild_without_an_adapter_removes_the_previous_one` fails if the
    # two are ever swapped.
    adapter_check = _copy_adapter(request, result)
    if adapter_check is not None:
        result.checks.add(adapter_check)
        events.check(adapter_check)

    # -- the declarations --------------------------------------------------
    # Shipped with the model because they are half of the calling convention:
    # the same weights against a different tool list are a different model from
    # the caller's point of view, and the contract's prompt mode is only
    # meaningful with respect to a specific set of declarations.
    declarations_check = _copy_declarations(request, result)
    result.checks.add(declarations_check)
    events.check(declarations_check)

    # -- the contract ------------------------------------------------------
    contract_path = request.output_dir / CONTRACT_NAME
    contract_path.write_text(
        json.dumps(request.contract.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    result.members.append(_member(CONTRACT_NAME, contract_path))
    contract_check = Check.passed(
        "contract recorded",
        f"prompt mode {request.contract.prompt_mode.value}, established against "
        + ", ".join(f"{k}=={v}" for k, v in sorted(request.contract.established_against.items())),
        observed=request.contract.as_dict(),
    )
    result.checks.add(contract_check)
    events.check(contract_check)

    # -- which measurements were not made ----------------------------------
    result.missing_measurements = [
        name for name in MEASUREMENT_POINTS if name not in request.measurements
    ]
    if result.missing_measurements:
        for name in result.missing_measurements:
            result.limitation(
                f"measurement point {name!r} was not taken: {MEASUREMENT_POINTS[name]}. Without "
                "it the corresponding difference cannot be attributed, and a single net figure "
                "cannot tell you which stage to fix"
            )
        measured = Check.unchecked(
            "measurements present",
            f"{len(request.measurements)} of {len(MEASUREMENT_POINTS)} measurement points are in "
            f"this bundle; missing: {', '.join(result.missing_measurements)}",
            observed={
                "present": sorted(request.measurements),
                "missing": list(result.missing_measurements),
            },
        )
    else:
        measured = Check.passed(
            "measurements present",
            "all three measurement points are recorded, so the training gain and the conversion "
            "cost are separable",
            observed={"present": sorted(request.measurements)},
        )
    # Recorded and emitted, but kept out of `result.checks`: see `BundleResult`.
    result.coverage = measured
    events.check(measured)

    # -- the manifest and the report, written whatever the outcome ---------
    manifest = _bundle_manifest(request, result)
    manifest_path = request.output_dir / MANIFEST_NAME
    manifest_path.write_text(manifest.as_json(), encoding="utf-8")
    result.manifest_path = manifest_path
    result.members.append(_member(MANIFEST_NAME, manifest_path))
    events.artifact(str(manifest_path), name=MANIFEST_NAME)

    report_path = request.output_dir / REPORT_NAME
    report_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    result.report_path = report_path
    # Added after writing: a file cannot contain its own hash. Recorded in the
    # result so the caller can still identify it.
    result.members.append(_member(REPORT_NAME, report_path))
    events.artifact(str(report_path), name=REPORT_NAME)

    events.stage_finished(
        result.status.value,
        complete=result.complete,
        members=len(result.members),
        missing_measurements=len(result.missing_measurements),
        verified=False,
    )
    return result


def _copy_adapter(request: BundleRequest, result: BundleResult) -> Check | None:
    """Copy the adapter in, or say why there is none. `None` when none was asked for.

    A full fine-tune has no adapter and this is silent for it. An adapter that
    was named and cannot be copied is a failed check rather than a refusal: the
    bundle is still worth writing, and the report is where someone reads what is
    missing from it.
    """
    if request.adapter is None:
        return None
    if not request.adapter.exists():
        return Check.failed(
            "adapter included",
            f"no adapter at {request.adapter}; LoRA weights are unrecoverable from the "
            "merged checkpoint, so a bundle without them cannot be re-merged or re-based",
            observed={"adapter": str(request.adapter)},
        )
    destination = request.output_dir / ADAPTER_DIR_NAME
    try:
        if request.adapter.is_dir():
            # The same guarded replace the checkpoint gets. Bundling a LoRA
            # run's own output directory -- `--adapter out/adapter
            # --output-dir out`, which is exactly where `tune` writes it -- made
            # source and destination the same path, and clearing the
            # destination first deleted the weights this function exists to
            # preserve.
            _replace_directory(request.adapter, destination, "adapter")
            members = _directory_members(ADAPTER_DIR_NAME, destination)
        else:
            destination = request.output_dir / request.adapter.name
            if request.adapter.resolve() == destination.resolve():
                raise BundleError(
                    f"the adapter at {request.adapter} is already inside --output-dir "
                    f"({request.output_dir}); name a source outside it"
                )
            shutil.copyfile(request.adapter, destination)
            members = [_member(destination.name, destination)]
    except BundleError as exc:
        return Check.failed(
            "adapter included",
            str(exc),
            observed={"adapter": str(request.adapter)},
        )
    except (OSError, shutil.Error) as exc:
        logger.exception("could not copy the adapter into %s", request.output_dir)
        return Check.failed(
            "adapter included",
            f"the adapter at {request.adapter} could not be copied into the bundle: "
            f"{type(exc).__name__}: {exc}",
            observed={"adapter": str(request.adapter)},
        )
    result.members.extend(members)
    total = sum(m.bytes for m in members)
    return Check.passed(
        "adapter included",
        f"{len(members)} file(s), {total:,} bytes at {destination}",
        observed={"files": len(members), "bytes": total},
    )


def _copy_declarations(request: BundleRequest, result: BundleResult) -> Check:
    """Copy the tool declarations in, and check they are readable as declarations."""
    name = "declarations included"
    source = request.declarations
    if not source.is_file():
        return Check.failed(
            name,
            f"{source} is not a file: the bundle would ship weights with no record of the tools "
            "they were trained to call, and the contract's prompt mode has nothing to apply to",
            observed={"declarations": str(source)},
        )
    destination = request.output_dir / DECLARATIONS_NAME
    try:
        shutil.copyfile(source, destination)
        parsed = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, shutil.Error) as exc:
        logger.exception("could not copy declarations from %s", source)
        return Check.failed(
            name,
            f"the declarations at {source} could not be copied: {type(exc).__name__}: {exc}",
            observed={"declarations": str(source)},
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.exception("declarations at %s are not JSON", source)
        # Copied and then rejected on purpose: the file is in the bundle so it
        # can be looked at, and the check says it is not usable.
        result.members.append(_member(DECLARATIONS_NAME, destination))
        return Check.failed(
            name,
            f"the declarations at {source} are not valid JSON ({exc}); a runtime cannot render a "
            "tool list it cannot parse",
            observed={"declarations": str(source)},
        )

    result.members.append(_member(DECLARATIONS_NAME, destination))
    count = len(parsed) if isinstance(parsed, list | dict) else None
    if request.contract.declarations_sha256 is not None:
        actual = hash_file(destination).split(":", 1)[-1]
        expected = request.contract.declarations_sha256.split(":", 1)[-1]
        if actual != expected:
            return Check.failed(
                name,
                f"the declarations in this bundle hash {actual[:16]} but the contract was written "
                f"against {expected[:16]}: the model's calling convention was established against "
                "a different tool list than the one being shipped",
                observed={"actual": actual, "contract": expected},
            )
    return Check.passed(
        name,
        f"{count if count is not None else 'the'} tool declaration(s) packaged from {source.name}",
        observed={"declarations": str(destination), "entries": count},
    )
