"""Stage sequencing: caching and resume, added on top of stages that don't need it.

The correction this module is built around: **a stage is invocable directly.**
`run_stage(stage, inputs)` executes with explicit inputs, no run directory, no
spec and no prior stage, and produces a manifest. The product's entry point is
verifying an artifact litetune did not produce, so composition cannot be a
precondition for running one step -- it is a layer that adds a cache key, an
ordering and a shared manifest on top of something that already worked without
them. A directly invoked stage therefore *executes*: it was handed inputs, not a
cache, and consulting one it was never given would make the standalone path
depend on state it knows nothing about.

The cache key is (stage name, spec slice, input content hashes, environment
identity). Each of the four is there because leaving it out was observed to be
wrong:

- the *spec slice*, not the spec, so that tightening a gate re-judges recorded
  metrics instead of re-running hours of generation
- input *content* hashes, not locations, so that a dataset replaced at the same
  URI invalidates everything downstream of it rather than training on stale data
  under a green manifest
- the *environment identity*, because an unchanged definition that resolves
  differently over time changes behaviour silently -- a working export on
  2026-08-26 and `AttributeError: pad_token` on 2026-08-30
- the *stage name*, so two stages over the same inputs cannot share an entry

Nothing is ever deleted. A failing stage keeps its workspace and every artifact
already produced, and the manifest records where they are and what the stage
printed on its way down.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from litetune.checks import Check
from litetune.events import EventStream
from litetune.manifest import (
    ArtifactRecord,
    CacheOutcome,
    InputRecord,
    RunManifest,
    RunStatus,
    StageRecord,
)
from litetune.metrics import Unavailable
from litetune.spec import Spec, SpecError
from litetune.storage import Storage, StorageError, hash_file, put_file, validate_key

logger = logging.getLogger(__name__)

CACHE_SCHEMA = "litetune.cache/1"
DEFAULT_CACHE_KEY = "cache/index.json"


class StageError(Exception):
    """A stage could not be composed -- not a statement about any model."""


class MissingInput(StageError):
    """A stage declared an input that was not supplied."""


class CacheUnusable(Exception):
    """The cache index could not be read. Every stage executes instead."""


# ---------------------------------------------------------------------------
# What a stage is given and what it returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageInput:
    """One thing a stage consumes.

    `locator` says where it is -- a storage key, a local path, a model id.
    `content_hash` says what it is, and is the only one of the two that reaches
    a cache key. `None` means the content could not be identified, which makes
    the stage uncacheable rather than cacheable on its location.
    """

    name: str
    locator: str
    content_hash: str | None = None

    @property
    def identified(self) -> bool:
        return bool(self.content_hash)

    def as_record(self) -> InputRecord:
        return InputRecord(name=self.name, locator=self.locator, content_hash=self.content_hash)

    @classmethod
    def from_file(cls, name: str, path: Path) -> StageInput:
        """Identify a local file by its content. Raises if it is not there."""
        if not path.is_file():
            raise MissingInput(f"input {name!r}: {path} is not a file")
        return cls(name=name, locator=str(path), content_hash=hash_file(path))

    @classmethod
    def from_artifact(cls, artifact: ArtifactRecord, name: str | None = None) -> StageInput:
        return cls(
            name=name or artifact.name,
            locator=artifact.key,
            content_hash=artifact.content_hash,
        )


@dataclass(frozen=True)
class Artifact:
    """A file a stage produced, as the stage names it, before it is stored."""

    name: str
    path: Path


@dataclass(frozen=True)
class StageContext:
    """Everything a stage is given. Every composition field is optional.

    `spec` and `storage` are `None` for a directly invoked stage. A stage that
    cannot run without them must say so in its result rather than assume them --
    the standalone path is the product's entry point, not a special case.
    """

    stage: str
    workspace: Path
    events: EventStream
    inputs: tuple[StageInput, ...] = ()
    spec: Spec | None = None
    storage: Storage | None = None

    def input(self, name: str) -> StageInput:
        for candidate in self.inputs:
            if candidate.name == name:
                return candidate
        raise MissingInput(
            f"stage {self.stage!r} asked for input {name!r}; it was given "
            f"{[i.name for i in self.inputs]}"
        )

    def spec_slice(self, sections: Sequence[str] | None = None) -> dict[str, Any] | None:
        """The stage's slice of the spec, or None when it was invoked without one."""
        if self.spec is None:
            return None
        return self.spec.slice_for(self.stage, sections)


@dataclass(frozen=True)
class StageResult:
    """What a stage established, and what it produced while establishing it."""

    status: RunStatus
    detail: str = ""
    artifacts: tuple[Artifact, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    checks: tuple[Check, ...] = ()
    # Whatever the stage's subprocesses said. Retained in the manifest whether
    # the stage passed or failed: "what did the toolchain print the day it
    # worked" is what the next failure gets diagnosed against.
    output: str = ""


class Stage(Protocol):
    """One step of the pipeline.

    A stage declares what it depends on so the runner can build a cache key
    without knowing what the stage does: which spec sections (a measurement
    stage may not name `gates`), which inputs, and which toolchain environments.
    """

    @property
    def name(self) -> str:
        """Stable identifier. Part of the cache key."""

    @property
    def spec_sections(self) -> tuple[str, ...]:
        """Spec sections this stage's result depends on. See `spec.STAGE_SECTIONS`."""

    @property
    def input_names(self) -> tuple[str, ...]:
        """Inputs this stage requires, by name."""

    @property
    def env_names(self) -> tuple[str, ...]:
        """Toolchain environments it runs in; empty means litetune's interpreter."""

    def run(self, ctx: StageContext) -> StageResult:
        """Do the work. May raise: the runner records that as `error`, not a verdict."""


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedStage:
    """A previous stage result, keyed by everything that could change it."""

    stage: str
    cache_key: str
    status: RunStatus
    env_identity: str
    run_id: str
    artifacts: tuple[ArtifactRecord, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "cache_key": self.cache_key,
            "status": self.status.value,
            "env_identity": self.env_identity,
            "run_id": self.run_id,
            "artifacts": [a.as_dict() for a in self.artifacts],
            "metrics": dict(self.metrics),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CachedStage:
        return cls(
            stage=str(data["stage"]),
            cache_key=str(data["cache_key"]),
            status=RunStatus(data["status"]),
            env_identity=str(data["env_identity"]),
            run_id=str(data.get("run_id", "")),
            artifacts=tuple(ArtifactRecord.from_dict(a) for a in data.get("artifacts", ())),
            metrics=dict(data.get("metrics", {})),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class CacheLookup:
    """A hit, or a miss with the reason. There is no third answer worth guessing."""

    cached: CachedStage | None
    detail: str

    @property
    def hit(self) -> bool:
        return self.cached is not None


@dataclass
class CacheIndex:
    """Previous stage results, and whether their artifacts are still what they were.

    A recorded entry is only a hit if every artifact it names still exists *and
    still hashes to what was recorded*. An artifact edited or truncated in place
    under a green cache entry is the same failure as a dataset replaced at its
    URI, one directory down.
    """

    storage: Storage
    key: str = DEFAULT_CACHE_KEY

    def _load(self) -> dict[str, Any]:
        if not self.storage.exists(self.key):
            return {}
        raw = self.storage.read_text(self.key)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.exception("cache index %r is not readable", self.key)
            raise CacheUnusable(f"cache index {self.key!r} is not valid JSON: {exc}") from exc
        entries = data.get("entries")
        if not isinstance(entries, dict):
            raise CacheUnusable(f"cache index {self.key!r} has no 'entries' mapping")
        return entries

    def preload(self) -> None:
        """Read the index now, so an unusable one is reported before any work."""
        self._load()

    def lookup(self, cache_key: str) -> CacheLookup:
        entry = self._load().get(cache_key)
        if entry is None:
            return CacheLookup(None, f"no entry for key {cache_key}")
        cached = CachedStage.from_dict(entry)
        for artifact in cached.artifacts:
            if not self.storage.exists(artifact.key):
                return CacheLookup(
                    None, f"artifact {artifact.name!r} recorded at {artifact.key!r} is gone"
                )
            actual = self.storage.content_hash(artifact.key)
            if actual != artifact.content_hash:
                return CacheLookup(
                    None,
                    f"artifact {artifact.name!r} at {artifact.key!r} now hashes {actual}, "
                    f"recorded as {artifact.content_hash}",
                )
        return CacheLookup(cached, f"every artifact still matches (from run {cached.run_id})")

    def record(self, cached: CachedStage) -> None:
        entries = self._load()
        entries[cached.cache_key] = cached.as_dict()
        self.storage.write_text(
            self.key,
            json.dumps({"schema": CACHE_SCHEMA, "entries": entries}, indent=2, sort_keys=True),
        )


# ---------------------------------------------------------------------------
# Running one stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageOutcome:
    """The result of one stage, plus the manifest it produced."""

    stage: str
    result: StageResult
    record: StageRecord
    manifest: RunManifest

    @property
    def status(self) -> RunStatus:
        return self.record.status


def new_workspace(stage: str, root: Path | None = None) -> Path:
    """A directory the stage may write into.

    Never removed, on success or on failure. A stage that fails half way keeps
    what it produced, and when no storage was given the manifest's artifact
    records point straight into here.
    """
    if root is not None:
        path = root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix=f"litetune-{stage}-"))


def run_stage(
    stage: Stage,
    inputs: Sequence[StageInput] = (),
    workspace: Path | None = None,
    spec: Spec | None = None,
    storage: Storage | None = None,
    events: EventStream | None = None,
    run_id: str | None = None,
    key_prefix: str | None = None,
) -> StageOutcome:
    """Run one stage directly. No run directory, no spec and no cache required.

    This is the entry point the product is built on: `verify` on an artifact
    litetune did not produce runs exactly this way. The stage executes -- no
    cache is consulted, because none was given -- and the manifest it returns is
    scaled to one stage, with the fields a single stage cannot know marked
    unavailable rather than left out.
    """
    events = events or EventStream(echo_json=False)
    run_id = run_id or f"{stage.name}-{int(time.time())}"
    workspace = workspace or new_workspace(stage.name)
    supplied = tuple(inputs)

    missing = [name for name in stage.input_names if name not in {i.name for i in supplied}]
    if missing:
        raise MissingInput(
            f"stage {stage.name!r} requires input(s) {missing}; it was given "
            f"{[i.name for i in supplied]}"
        )

    if spec is not None:
        env_identity: str | Unavailable = spec.env_identity_for(stage.name, stage.env_names)
    else:
        env_identity = Unavailable(
            "no job spec was supplied, so the toolchain that resolves this stage's environment "
            "is not declared and its identity cannot be recorded"
        )

    result, record = _execute(
        stage,
        StageContext(
            stage=stage.name,
            workspace=workspace,
            events=events,
            inputs=supplied,
            spec=spec,
            storage=storage,
        ),
        cache_outcome=CacheOutcome.NOT_CONSULTED,
        cache_key=Unavailable("this stage was invoked directly and was given no cache to consult"),
        env_identity=env_identity,
        key_prefix=key_prefix or f"runs/{run_id}/{stage.name}",
    )
    manifest = RunManifest.for_stage(
        record,
        run_id=run_id,
        spec_hash=spec.hash if spec is not None else None,
        environments=spec.environments() if spec is not None else {},
        spec=spec.as_dict() if spec is not None else None,
    )
    if spec is not None:
        for text in spec.limitations:
            manifest.limitation(text)
    return StageOutcome(stage=stage.name, result=result, record=record, manifest=manifest)


def _execute(
    stage: Stage,
    ctx: StageContext,
    cache_outcome: CacheOutcome,
    cache_key: str | Unavailable,
    env_identity: str | Unavailable,
    key_prefix: str,
) -> tuple[StageResult, StageRecord]:
    """Run the stage body, ingest what it produced, and record all of it."""
    events = ctx.events
    started = time.time()
    events.stage_started(
        stage.name,
        inputs=[i.name for i in ctx.inputs],
        cache=cache_outcome.value,
        cache_key=cache_key if isinstance(cache_key, str) else None,
        workspace=str(ctx.workspace),
    )
    try:
        result = stage.run(ctx)
    except Exception as exc:  # noqa: BLE001 - stage boundary: a crash is not a verdict
        # A stage that raised established nothing about the model. `error` says
        # that; `failed` would say the model is bad, which is exactly the
        # confident-negative-without-a-measurement mistake this tool exists to
        # prevent.
        logger.exception("stage %r raised", stage.name)
        result = StageResult(
            status=RunStatus.ERROR,
            detail=f"stage raised {type(exc).__name__}: {exc}",
            output=traceback.format_exc(),
        )

    artifacts, problems = _ingest(stage.name, result.artifacts, ctx, key_prefix)
    status, detail = result.status, result.detail
    if problems:
        # The stage said it produced something and it is not there. Reporting
        # the stage's own status now would be reporting a result over an
        # artifact nobody can find.
        joined = "; ".join(problems)
        if status is RunStatus.PASSED:
            status = RunStatus.FAILED_HARNESS
        detail = f"{detail} [artifacts not recorded: {joined}]".strip()

    record = StageRecord(
        name=stage.name,
        status=status,
        cache=cache_outcome,
        cache_key=cache_key,
        env_identity=env_identity,
        detail=detail,
        inputs=tuple(i.as_record() for i in ctx.inputs),
        artifacts=tuple(artifacts),
        metrics=dict(result.metrics),
        checks=tuple(check.as_dict() for check in result.checks),
        output=result.output,
        workspace=str(ctx.workspace),
        started_at=started,
        duration_s=time.time() - started,
    )
    # `stage=` is passed explicitly because a stage that emits its own
    # stage_finished (verify does) clears the stream's current stage, and the
    # record must name the stage it belongs to either way.
    events.stage_finished(
        status.value,
        stage=stage.name,
        cache=cache_outcome.value,
        artifacts=len(artifacts),
        detail=detail,
    )
    return result, record


def _ingest(
    stage_name: str,
    artifacts: Sequence[Artifact],
    ctx: StageContext,
    key_prefix: str,
) -> tuple[list[ArtifactRecord], list[str]]:
    """Record every declared artifact by content, storing it if there is a store.

    A declared artifact that is not on disk is reported as a problem rather than
    skipped: a stage claiming an output it did not write is the shape of the
    "exit code zero and a file of the right size" failure this toolchain is
    known for.
    """
    records: list[ArtifactRecord] = []
    problems: list[str] = []
    for artifact in artifacts:
        path = artifact.path if artifact.path.is_absolute() else ctx.workspace / artifact.path
        if not path.is_file():
            problems.append(f"{artifact.name!r} declared at {path} does not exist")
            continue
        try:
            if ctx.storage is None:
                # No store: the artifact stays where the stage wrote it, and the
                # workspace is never cleaned up, so the manifest's path is live.
                key = str(path)
                content_hash = hash_file(path)
                size = path.stat().st_size
            else:
                key = f"{key_prefix}/{validate_key(artifact.name)}"
                put_file(ctx.storage, key, path)
                content_hash = ctx.storage.content_hash(key)
                size = ctx.storage.size(key)
        except (OSError, StorageError) as exc:
            # Includes an artifact whose *name* is not a usable key. A stage
            # naming its output badly must degrade to a recorded problem, not
            # take the run down with no manifest at all.
            logger.exception("could not record artifact %r from %s", artifact.name, path)
            problems.append(f"{artifact.name!r}: {type(exc).__name__}: {exc}")
            continue
        record = ArtifactRecord(
            name=artifact.name,
            key=key,
            bytes=size,
            content_hash=content_hash,
            stage=stage_name,
        )
        records.append(record)
        ctx.events.artifact(key, name=artifact.name, bytes=size, content_hash=content_hash)
    return records, problems


# ---------------------------------------------------------------------------
# Running a sequence
# ---------------------------------------------------------------------------


def new_run_id(spec: Spec) -> str:
    """A run id that sorts by time and names the spec it came from."""
    return f"{time.strftime('%Y%m%dT%H%M%S')}-{spec.hash}"


@dataclass
class Runner:
    """Stages in order, over one run directory, with a cache in front of each.

    Sequencing is the only thing added here. Every stage in `stages` still runs
    standalone through `run_stage`; this class supplies inputs from the previous
    stage's artifacts, computes a cache key, and writes one manifest.
    """

    stages: Sequence[Stage]
    spec: Spec
    storage: Storage
    # Artifacts are written under `runs/<run_id>/`, so a run id identifies one
    # attempt: a fresh one (see `new_run_id`) is what makes "nothing is ever
    # deleted" true across retries rather than only within one.
    run_id: str
    events: EventStream | None = None
    cache: CacheIndex | None = None
    # Scratch space. Deliberately outside `storage`, so a stage's intermediate
    # files never appear in `storage.list()` as though they were artifacts.
    workspace_root: Path | None = None

    def run(self, inputs: Sequence[StageInput] = (), resume: bool = True) -> RunManifest:
        """Run every stage in order. Returns the manifest, whatever happened."""
        events = self.events or EventStream(echo_json=False)
        manifest = RunManifest(
            run_id=self.run_id,
            spec_hash=self.spec.hash,
            scope="run",
            environments=self.spec.environments(),
            spec=self.spec.as_dict(),
        )
        for text in self.spec.limitations:
            manifest.limitation(text)

        cache = self.cache if resume else None
        if cache is not None:
            try:
                cache.preload()
            except CacheUnusable as exc:
                # A cache is the one genuinely optional resource here: losing it
                # costs time and cannot cost correctness. It is dropped loudly --
                # in the log and in the manifest -- never silently.
                logger.warning("cache disabled for run %s: %s", self.run_id, exc)
                manifest.limitation(
                    f"the cache index could not be read ({exc}); every stage in this run "
                    "executed rather than resuming"
                )
                cache = None
        elif self.cache is not None:
            manifest.limitation("resume was disabled; every stage executed")

        available: dict[str, StageInput] = {i.name: i for i in inputs}
        stopped: str | None = None

        for stage in self.stages:
            if stopped is not None:
                manifest.not_reached.append({"name": stage.name, "reason": stopped})
                continue
            record = self._run_one(stage, available, cache, events, manifest)
            manifest.add(record)
            self._persist(manifest)
            for artifact in record.artifacts:
                available[artifact.name] = StageInput.from_artifact(artifact)
            if record.status is not RunStatus.PASSED:
                stopped = (
                    f"{stage.name} reported {record.status.value}; every artifact produced so far "
                    "is retained"
                )

        self._persist(manifest)
        return manifest

    # -- one stage ---------------------------------------------------------

    def _run_one(
        self,
        stage: Stage,
        available: Mapping[str, StageInput],
        cache: CacheIndex | None,
        events: EventStream,
        manifest: RunManifest,
    ) -> StageRecord:
        missing = [name for name in stage.input_names if name not in available]
        if missing:
            return self._wiring_error(
                stage,
                f"requires input(s) {missing}, which no earlier stage produced and the run was "
                f"not given; available: {sorted(available)}",
            )

        stage_inputs = tuple(available[name] for name in stage.input_names)
        try:
            env_identity = self.spec.env_identity_for(stage.name, stage.env_names)
        except SpecError as exc:
            logger.exception("stage %r has no resolvable environment", stage.name)
            return self._wiring_error(stage, str(exc))

        cache_key: str | Unavailable
        unidentified = [i.name for i in stage_inputs if not i.identified]
        if unidentified:
            # An input nobody can identify by content cannot be keyed on. The
            # stage executes: a hit found under such a key would be reuse of an
            # artifact produced from unknown material.
            reason = (
                f"stage {stage.name!r} has input(s) {unidentified} with no content hash, so its "
                "cache key would not describe what it ran on; it executed rather than resuming"
            )
            manifest.limitation(reason)
            cache_key = Unavailable(reason)
            cache_outcome = CacheOutcome.UNUSABLE
        else:
            try:
                cache_key = self.spec.hash_for(
                    stage.name,
                    {i.name: str(i.content_hash) for i in stage_inputs},
                    env_identity,
                    sections=stage.spec_sections,
                )
            except SpecError as exc:
                logger.exception("stage %r cannot be keyed", stage.name)
                return self._wiring_error(stage, str(exc))
            if cache is None:
                cache_outcome = CacheOutcome.NOT_CONSULTED
            else:
                lookup = cache.lookup(cache_key)
                if lookup.cached is not None:
                    return self._replay(stage, lookup.cached, stage_inputs, cache_key, events)
                cache_outcome = CacheOutcome.MISS

        _, record = _execute(
            stage,
            StageContext(
                stage=stage.name,
                workspace=new_workspace(stage.name, self._workspace_root()),
                events=events,
                inputs=stage_inputs,
                spec=self.spec,
                storage=self.storage,
            ),
            cache_outcome=cache_outcome,
            cache_key=cache_key,
            env_identity=env_identity,
            key_prefix=f"runs/{self.run_id}/{stage.name}",
        )

        if cache is not None and isinstance(cache_key, str) and record.status is RunStatus.PASSED:
            # Only a pass is cached. A failure is at least as likely to be about
            # this machine as about the artifact, and replaying it as a hit
            # would make a transient failure permanent.
            cache.record(
                CachedStage(
                    stage=stage.name,
                    cache_key=cache_key,
                    status=record.status,
                    env_identity=env_identity,
                    run_id=self.run_id,
                    artifacts=record.artifacts,
                    metrics=record.metrics,
                    detail=record.detail,
                )
            )
        return record

    def _replay(
        self,
        stage: Stage,
        cached: CachedStage,
        stage_inputs: Sequence[StageInput],
        cache_key: str,
        events: EventStream,
    ) -> StageRecord:
        """Reuse a previous result. The stage body is not run."""
        events.stage_started(
            stage.name, cache=CacheOutcome.HIT.value, cache_key=cache_key, reused_from=cached.run_id
        )
        for artifact in cached.artifacts:
            events.artifact(
                artifact.key,
                name=artifact.name,
                bytes=artifact.bytes,
                content_hash=artifact.content_hash,
                reused=True,
            )
        detail = f"reused from run {cached.run_id}; the stage body did not run"
        events.stage_finished(
            cached.status.value,
            stage=stage.name,
            cache=CacheOutcome.HIT.value,
            artifacts=len(cached.artifacts),
            detail=detail,
        )
        return StageRecord(
            name=stage.name,
            status=cached.status,
            cache=CacheOutcome.HIT,
            cache_key=cache_key,
            env_identity=cached.env_identity,
            detail=detail,
            inputs=tuple(i.as_record() for i in stage_inputs),
            artifacts=cached.artifacts,
            metrics=dict(cached.metrics),
            workspace=Unavailable("no workspace: this stage was not run in this run"),
            started_at=time.time(),
        )

    def _wiring_error(self, stage: Stage, detail: str) -> StageRecord:
        """litetune could not run this stage. Not a statement about any model."""
        return StageRecord(
            name=stage.name,
            status=RunStatus.ERROR,
            cache=CacheOutcome.NOT_CONSULTED,
            cache_key=Unavailable("the stage could not be keyed"),
            env_identity=Unavailable("the stage was not run"),
            detail=f"stage {stage.name!r} {detail}",
        )

    def _workspace_root(self) -> Path | None:
        if self.workspace_root is None:
            return None
        root = self.workspace_root / self.run_id
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _persist(self, manifest: RunManifest) -> None:
        try:
            manifest.write(self.storage)
        except OSError:
            logger.exception("could not write the manifest for run %s", self.run_id)
            raise
