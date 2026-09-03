"""The run manifest: what ran, what it produced, and what it did not establish.

The manifest is the artifact the release decision is reconstructed from months
later, so its job is to be honest about its own gaps. Three rules follow.

**A field that could not be known is marked, not omitted.** A directly invoked
stage has no spec, so `spec_hash` is recorded as unavailable *with the reason*
rather than left out. An absent key reads as an oversight; an explicit
"unavailable: this stage was invoked without a job spec" reads as a fact.

**A cache outcome is not two-valued.** `hit`, `miss`, `not_consulted` and
`unusable` are different things: a stage invoked directly consulted nothing, and
a stage whose inputs could not be hashed could not have trusted a hit if it had
found one. Collapsing those into "miss" would be harmless; collapsing them into
"hit" would silently reuse an artifact nobody could identify.

**An aggregate that could not be established is not a pass.** Statuses that mean
"we could not tell" -- `error`, `failed_harness`, `inconclusive`, `unmeasured` --
dominate statuses that mean "we told you no", which dominate `passed`, exactly
as `checks.CheckSet.outcome` treats UNCHECKED. A manifest with no stages in it
has established nothing and says so.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from litetune.metrics import Unavailable
from litetune.storage import Storage

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA = "litetune.run/1"

# Captured output is kept for every stage, not only failing ones, because "what
# did the toolchain say the day it worked" is what a later failure is diagnosed
# against. Bounded so a chatty exporter cannot make the manifest unreadable.
MAX_CAPTURED_OUTPUT = 8000


class RunStatus(str, Enum):
    """The status of a stage, and of a run.

    `unmeasured` is here and is not in the composition brief's list: `verify`
    returns it for a run that completed without establishing the quantity that
    was asked for -- no labelled data at all, or labelled data whose difference
    cannot be attributed to conversion rather than to training. Folding either
    into `passed` or `failed` is the exact collapse this tool exists to
    prevent. This set must stay a superset of `verify.Status`.
    """

    PASSED = "passed"
    FAILED_SMOKE = "failed_smoke"
    FAILED_GATE = "failed_gate"
    INCONCLUSIVE = "inconclusive"
    UNMEASURED = "unmeasured"
    FAILED_HARNESS = "failed_harness"
    ERROR = "error"

    @property
    def conclusive(self) -> bool:
        """Whether this status is a verdict about the model at all."""
        return self in (RunStatus.PASSED, RunStatus.FAILED_SMOKE, RunStatus.FAILED_GATE)


# Worst first. The "could not tell" family leads, mirroring checks.CheckSet:
# a caller promised an answer and denied one has a worse result than a caller
# given a clear no.
_STATUS_PRECEDENCE: tuple[RunStatus, ...] = (
    RunStatus.ERROR,
    RunStatus.FAILED_HARNESS,
    RunStatus.INCONCLUSIVE,
    RunStatus.UNMEASURED,
    RunStatus.FAILED_SMOKE,
    RunStatus.FAILED_GATE,
    RunStatus.PASSED,
)


def worst_status(statuses: Iterable[RunStatus]) -> RunStatus:
    """The status of a set of stages. An empty set has established nothing."""
    present = set(statuses)
    if not present:
        # Same rule as an empty CheckSet: nothing was observed, so nothing is
        # concluded. Reporting PASSED here would be a green run that never ran.
        return RunStatus.INCONCLUSIVE
    for status in _STATUS_PRECEDENCE:
        if status in present:
            return status
    raise ValueError(f"unrecognised statuses: {sorted(s for s in present)}")


class CacheOutcome(str, Enum):
    """Why a stage did or did not reuse a previous result."""

    HIT = "hit"
    MISS = "miss"
    # No cache was given. A directly invoked stage is always this: it executes
    # rather than consulting something it was not handed.
    NOT_CONSULTED = "not_consulted"
    # A cache existed and could not be trusted -- an input with no content
    # hash, or an index that would not parse. The stage executes.
    UNUSABLE = "unusable"


def _tool_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    from litetune._version import __version__

    try:
        installed = version("litetune")
    except PackageNotFoundError:
        # Running from a checkout, a vendored copy or a zipapp. The module knows
        # its own version; only the distribution metadata is missing. Recording
        # "unknown" here put an unusable value in the field the whole manifest
        # exists to make trustworthy.
        return __version__
    if installed != __version__:
        logger.warning(
            "installed litetune distribution reports %s, package says %s; recording the package",
            installed,
            __version__,
        )
    return __version__


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_CAPTURED_OUTPUT:
        return text, False
    return text[-MAX_CAPTURED_OUTPUT:], True


@dataclass(frozen=True)
class ArtifactRecord:
    """One file a stage produced, identified by its content.

    `key` says where it is; `content_hash` says what it is. Only the second one
    reaches a downstream cache key.
    """

    name: str
    key: str
    bytes: int
    content_hash: str
    stage: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "bytes": self.bytes,
            "content_hash": self.content_hash,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRecord:
        return cls(
            name=str(data["name"]),
            key=str(data["key"]),
            bytes=int(data["bytes"]),
            content_hash=str(data["content_hash"]),
            stage=str(data.get("stage", "")),
        )


@dataclass(frozen=True)
class InputRecord:
    """One thing a stage was given, and whether it could be identified."""

    name: str
    locator: str
    content_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "locator": self.locator,
            "content_hash": self.content_hash
            or Unavailable(
                "this input could not be hashed, so nothing downstream of it may be cached"
            ).as_dict(),
        }


@dataclass
class StageRecord:
    """One stage's result, including everything it could not establish."""

    name: str
    status: RunStatus
    cache: CacheOutcome
    cache_key: str | Unavailable
    env_identity: str | Unavailable
    detail: str = ""
    inputs: tuple[InputRecord, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    checks: tuple[Mapping[str, Any], ...] = ()
    output: str = ""
    # Where the stage was allowed to write. Recorded because a failed stage's
    # partial output is retained rather than deleted, and a path in the
    # manifest is the only way anyone finds it again.
    workspace: str | Unavailable = field(
        default_factory=lambda: Unavailable("no workspace was recorded for this stage")
    )
    started_at: float = 0.0
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        output, truncated = _truncate(self.output)
        return {
            "name": self.name,
            "status": self.status.value,
            "cache": self.cache.value,
            "cache_key": _maybe(self.cache_key),
            "env_identity": _maybe(self.env_identity),
            "detail": self.detail,
            "inputs": [i.as_dict() for i in self.inputs],
            "artifacts": [a.as_dict() for a in self.artifacts],
            "metrics": dict(self.metrics),
            "checks": [dict(c) for c in self.checks],
            "workspace": _maybe(self.workspace),
            "started_at": round(self.started_at, 3),
            "duration_s": round(self.duration_s, 3),
            "output": output,
            "output_truncated": truncated,
        }


def _maybe(value: Any) -> Any:
    return value.as_dict() if isinstance(value, Unavailable) else value


@dataclass
class RunManifest:
    """What a run -- or one directly invoked stage -- established.

    `scope` is `"run"` or `"stage"`. A single-stage manifest carries the same
    schema with the fields it cannot know marked unavailable, so the two are
    read by the same code and the gaps are visible rather than absent.
    """

    run_id: str
    spec_hash: str | Unavailable
    scope: str = "run"
    stages: list[StageRecord] = field(default_factory=list)
    environments: Mapping[str, str] = field(default_factory=dict)
    # Stages that were declared and never reached, with the reason. Omitting
    # them would make a run that stopped at stage two look like a two-stage job.
    not_reached: list[dict[str, str]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    spec: Mapping[str, Any] | Unavailable = field(
        default_factory=lambda: Unavailable("no job spec was supplied")
    )
    created_at: float = field(default_factory=time.time)

    @classmethod
    def for_stage(
        cls,
        record: StageRecord,
        run_id: str,
        spec_hash: str | None = None,
        environments: Mapping[str, str] | None = None,
        spec: Mapping[str, Any] | None = None,
    ) -> RunManifest:
        """A manifest scaled to one stage, invoked on its own.

        The unknowable fields are marked rather than dropped: a reader must be
        able to tell "this ran outside a job" from "this manifest is missing
        half its content".
        """
        no_spec = Unavailable(
            "this stage was invoked directly, without a job spec: there is no spec hash and no "
            "record of which pipeline it belongs to"
        )
        return cls(
            run_id=run_id,
            spec_hash=spec_hash if spec_hash is not None else no_spec,
            scope="stage",
            stages=[record],
            environments=dict(environments or {}),
            spec=dict(spec) if spec is not None else no_spec,
        )

    @property
    def status(self) -> RunStatus:
        status = worst_status(stage.status for stage in self.stages)
        if self.not_reached and status is RunStatus.PASSED:
            # Defensive: a run that stopped early cannot be a pass. Reaching
            # this means a stage failed without a failing status, which is the
            # bug this line refuses to let out of the building.
            logger.error(
                "run %s left %d stage(s) unreached with no failing status",
                self.run_id,
                len(self.not_reached),
            )
            return RunStatus.ERROR
        return status

    def add(self, record: StageRecord) -> StageRecord:
        self.stages.append(record)
        return record

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def stage(self, name: str) -> StageRecord:
        for record in self.stages:
            if record.name == name:
                return record
        raise KeyError(f"no stage named {name!r} in manifest {self.run_id}")

    @property
    def artifacts(self) -> list[ArtifactRecord]:
        return [artifact for stage in self.stages for artifact in stage.artifacts]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "tool_version": _tool_version(),
            "run_id": self.run_id,
            "scope": self.scope,
            "created_at": round(self.created_at, 3),
            "status": self.status.value,
            "spec_hash": _maybe(self.spec_hash),
            "spec": _maybe(self.spec),
            "environments": dict(self.environments),
            "stages": [stage.as_dict() for stage in self.stages],
            "not_reached": list(self.not_reached),
            "limitations": list(self.limitations),
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str)

    def write(self, storage: Storage, key: str | None = None) -> str:
        """Persist the manifest. Returns the key it was written under."""
        target = key or f"runs/{self.run_id}/manifest.json"
        storage.write_text(target, self.as_json())
        return target


def summarise(manifest: RunManifest) -> list[str]:
    """One line per stage, for a renderer to print. Nothing here prints."""
    lines = [f"run {manifest.run_id}: {manifest.status.value}"]
    for stage in manifest.stages:
        note = f" ({stage.cache.value})" if stage.cache is not CacheOutcome.MISS else ""
        lines.append(f"  {stage.name}: {stage.status.value}{note} — {stage.detail}".rstrip(" —"))
    for skipped in manifest.not_reached:
        lines.append(f"  {skipped.get('name')}: not reached — {skipped.get('reason')}")
    for limitation in manifest.limitations:
        lines.append(f"  note: {limitation}")
    return lines
