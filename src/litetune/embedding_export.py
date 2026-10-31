"""`convert` for an embedding checkpoint: Google's EmbeddingGemma 2 file, with the fine-tune in it.

`litetune convert --model <dir>` lands here instead of `export.run_export` when
the checkpoint's `litetune.json` is an embedding record (`embedding_tune`,
`"direction": "embedding"`). The work is `embedding_convert.graft`; this stage
finds its three inputs and writes its report:

- **Google's file**, `embedding_convert.GOOGLE_FILE` of
  `embedding_convert.GOOGLE_REPO` at `GOOGLE_REVISION`, downloaded through
  `huggingface_hub` inside `envs.EXPORT`, or `--base-litertlm PATH`. Either way
  its sha256 must be `embedding_convert.GOOGLE_SHA256`;
- **the base checkpoint**, `BASE_FILE` of `embedding_convert.BASE_MODEL` at
  `embedding_convert.BASE_REVISION`, downloaded the same way; its sha256 must be
  `embedding_convert.BASE_SHA256`. The graft reads the tensor mapping and the
  fold factors against it;
- **the fine-tune**, `BASE_FILE` in the checkpoint directory `tune` wrote; its
  sha256 is recorded in the report as `checkpoint.sha256`.

The file is written to `<output-dir>/<weights>/model.litertlm`, one directory
per weight mode as `convert` writes one per recipe (`export.ExportRequest.dir_for`),
with `convert.json` beside it. `verify` reads the weight mode from that report.

What the file embeds is not checked here: `litetune verify` measures it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from litetune import envs
from litetune.checks import Check, CheckSet, Outcome, guard
from litetune.embedding_convert import (
    BASE_MODEL,
    BASE_REVISION,
    BASE_SHA256,
    DEFAULT_WEIGHTS,
    GOOGLE_FILE,
    GOOGLE_REPO,
    GOOGLE_SHA256,
    GRAFT_TIMEOUT_S,
    EmbeddingConvertError,
    GraftResult,
    WeightMode,
    graft,
)
from litetune.embedding_npu import (
    NPU_SIGNATURES,
    NPU_TIMEOUT_S,
    EmbeddingNpuError,
    NpuResult,
    compile_for_npu,
    dir_for,
    host_problem,
    request_problem,
)
from litetune.embedding_tune import DECIDER_PREFIX, EmbeddingRecord
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

EMBEDDING_CONVERT_SCHEMA = "litetune.convert-embedding/1"

# The revision of `GOOGLE_REPO` the graft was built and measured on
# (MEASUREMENTS.md, "EmbeddingGemma 2: the graft"). The file fetched at it is checked against
# `GOOGLE_SHA256` all the same.
GOOGLE_REVISION = "9be6e8b90982095dc05c2bd162e4b954ee4dbac7"
# The checkpoint file the graft reads, in the base repository and in the
# directory `tune` saved the fine-tune to.
BASE_FILE = "model.safetensors"

WEIGHT_MODES: tuple[str, ...] = get_args(WeightMode)
# `DEFAULT_WEIGHTS` is `embedding_convert`'s, imported above, so that `graft`
# defaults to it too; this module's name for it stays.

ARTIFACT_NAME = "model.litertlm"
REPORT_NAME = "convert.json"

# An allowance for two files of hundreds of MB each, not a measurement.
DOWNLOAD_TIMEOUT_S = 3600

ENV_CHECK = "export environment"
BASE_FILE_CHECK = "Google's file"
BASE_CHECKPOINT_CHECK = "base checkpoint"
GRAFT_CHECK = "graft"


NPU_ENV_CHECK = "npu environment"
# Causes that say nothing about the model or the compiler: the run itself, and an error
# in litetune's own script.
NPU_RUN_CAUSES = ("run", "script")
NPU_PUBLISH_CHECK = "npu files"


def npu_check(soc: str) -> str:
    return f"npu {soc}"


def npu_artifact_name(soc: str) -> str:
    """The artifact event name of an NPU file, which `ARTIFACT_NAME` would not tell apart."""
    return f"{ARTIFACT_NAME} (npu {soc})"


NPU_NOT_VERIFIED = (
    "the NPU file was compiled, checked (one DISPATCH_OP per signature over one QNN context "
    "no larger than twice the encoder's weights, stamped for its SoC, signatures unchanged) "
    "and read back section by section; "
    "what it embeds has not been measured, and `litetune verify` cannot measure it: "
    "LiteRT-LM's CPU and GPU do not run a DISPATCH_OP. It needs a phone with that SoC and a "
    "LiteRT-LM build with the Qualcomm dispatch and a QAIRT 2.50 runtime"
)


NOT_VERIFIED = (
    "the file was written and read back section by section; what it embeds has not been "
    "measured. Run `litetune verify --model <file> --reference <checkpoint> --data <heldout>`"
)

_DOWNLOAD_SCRIPT = r'''
"""Download files from the Hugging Face Hub at pinned revisions.

Reads JSON {files: [{repo, filename, revision}], out} from argv[1]; writes
{"paths": [...], "huggingface_hub": <version>} to spec["out"].
"""
import importlib.metadata
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    paths = [
        hf_hub_download(f["repo"], f["filename"], revision=f["revision"]) for f in spec["files"]
    ]
    Path(spec["out"]).write_text(
        json.dumps(
            {"paths": paths, "huggingface_hub": importlib.metadata.version("huggingface_hub")}
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class EmbeddingExportError(ValueError):
    """A `convert` request the embedding path will not run. Not a statement about any model."""


class DownloadFailed(Exception):
    """The Hub files could not be fetched; says nothing about any model."""


@dataclass(frozen=True)
class HubFile:
    """One file at a pinned revision."""

    repo: str
    filename: str
    revision: str

    def as_dict(self) -> dict[str, str]:
        return {"repo": self.repo, "filename": self.filename, "revision": self.revision}


GOOGLE_HUB_FILE = HubFile(GOOGLE_REPO, GOOGLE_FILE, GOOGLE_REVISION)
BASE_HUB_FILE = HubFile(BASE_MODEL, BASE_FILE, BASE_REVISION)


def download(
    env: envs.StageEnv,
    files: list[HubFile],
    timeout: int = DOWNLOAD_TIMEOUT_S,
) -> list[Path]:
    """Each file's path in the Hub cache, fetched inside `env`. Raises `DownloadFailed`."""
    with tempfile.TemporaryDirectory(prefix="litetune-hub-") as tmp:
        work = Path(tmp)
        script, spec, out = work / "download.py", work / "spec.json", work / "paths.json"
        script.write_text(_DOWNLOAD_SCRIPT, encoding="utf-8")
        spec.write_text(
            json.dumps({"files": [f.as_dict() for f in files], "out": str(out)}),
            encoding="utf-8",
        )
        try:
            proc = env.run(["python", str(script), str(spec)], timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise DownloadFailed(f"the download did not finish within {timeout}s") from exc
        except OSError as exc:
            raise DownloadFailed(
                f"the download could not be started: {type(exc).__name__}: {exc}"
            ) from exc
        if proc.returncode != 0 or not out.is_file():
            reading = read_returncode(proc.returncode)
            said = (proc.stderr or "").strip()[-300:] or "no stderr"
            raise DownloadFailed(f"the download {reading.describe('the files')}: {said}")
        try:
            paths = json.loads(out.read_text(encoding="utf-8"))["paths"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DownloadFailed(f"the download script's answer could not be read: {exc}") from exc
    if not isinstance(paths, list) or len(paths) != len(files):
        raise DownloadFailed(f"the download script named {paths!r} for {len(files)} files")
    return [Path(p) for p in paths]


def check_google_file(path: Path) -> str | None:
    """Why `path` is not Google's file, or `None` when its sha256 is `GOOGLE_SHA256`."""
    return _google_file_problem(path, hash_file(path))


def _google_file_problem(path: Path, digest: str) -> str | None:
    if digest != f"sha256:{GOOGLE_SHA256}":
        return (
            f"{path} has {digest}, not sha256:{GOOGLE_SHA256}, the {GOOGLE_FILE} of {GOOGLE_REPO} "
            f"at {GOOGLE_REVISION} that the graft's mapping was established on"
        )
    return None


def _base_checkpoint_problem(path: Path, digest: str) -> str | None:
    if digest != f"sha256:{BASE_SHA256}":
        return (
            f"{path} has {digest}, not sha256:{BASE_SHA256}, the {BASE_FILE} of {BASE_MODEL} "
            f"at {BASE_REVISION} that the graft's mapping and fold factors were established on"
        )
    return None


@dataclass(frozen=True)
class EmbeddingExportRequest:
    """One graft of an embedding checkpoint into Google's file."""

    model: Path
    output_dir: Path
    weights: str = DEFAULT_WEIGHTS
    # A local copy of Google's file; `None` downloads it.
    base_litertlm: Path | None = None
    timeout_s: int = GRAFT_TIMEOUT_S
    env: envs.StageEnv = envs.EXPORT
    auto_provision: bool = True
    # `--npu`: Qualcomm SoCs to compile the int8 graft's text encoder for (embedding_npu).
    npu: tuple[str, ...] = ()
    npu_env: envs.StageEnv = envs.NPU_QUALCOMM
    # Seconds each NPU compile may run; the NPU environment's provision has its own
    # budget, `envs.PROVISION_TIMEOUT_S`.
    npu_timeout_s: int = NPU_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.weights not in WEIGHT_MODES:
            raise EmbeddingExportError(
                f"--weights {self.weights}: one of {', '.join(WEIGHT_MODES)}"
            )
        if self.timeout_s < 1:
            raise EmbeddingExportError(f"timeout_s must be at least 1, got {self.timeout_s}")
        if self.npu_timeout_s < 1:
            raise EmbeddingExportError(
                f"npu_timeout_s must be at least 1, got {self.npu_timeout_s}"
            )
        object.__setattr__(self, "npu", tuple(self.npu))
        problem = request_problem(self.npu, self.weights)
        if problem is not None:
            raise EmbeddingExportError(problem)
        object.__setattr__(self, "model", Path(self.model))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.base_litertlm is not None:
            object.__setattr__(self, "base_litertlm", Path(self.base_litertlm))

    @property
    def tuned_safetensors(self) -> Path:
        return self.model / BASE_FILE

    @property
    def output(self) -> Path:
        return self.output_dir / self.weights / ARTIFACT_NAME

    @property
    def report_path(self) -> Path:
        return self.output_dir / self.weights / REPORT_NAME

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": str(self.model),
            "output_dir": str(self.output_dir),
            "weights": self.weights,
            "base_litertlm": str(self.base_litertlm) if self.base_litertlm else None,
            "timeout_s": self.timeout_s,
            "environment": {"name": self.env.name, "identity": self.env.identity},
            "npu": list(self.npu),
            "npu_timeout_s": self.npu_timeout_s if self.npu else None,
            "npu_environment": (
                {"name": self.npu_env.name, "identity": self.npu_env.identity} if self.npu else None
            ),
        }


@dataclass
class EmbeddingExportResult:
    """What the graft wrote, or why it did not."""

    request: EmbeddingExportRequest
    record: EmbeddingRecord
    checks: CheckSet
    google: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    graft: GraftResult | None = None
    limitations: list[str] = field(default_factory=list)
    # `hash_file` of the fine-tune's safetensors, taken before the graft reads it.
    checkpoint_sha256: str | None = None
    # `--npu`: one entry per SoC compiled, at its published path once all are.
    npu: list[NpuResult] = field(default_factory=list)
    npu_published: bool = False
    # What publishing got to: each SoC whose directory this run created, with its result,
    # and why publishing stopped. `_npus_on_disk` reads `npu` and `npu_published` from these
    # and the disk when the report is written.
    npu_created: dict[str, NpuResult] = field(default_factory=dict)
    npu_publish_error: str | None = None

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.PASSED

    @property
    def artifact(self) -> Path | None:
        return self.graft.output if self.graft is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EMBEDDING_CONVERT_SCHEMA,
            "direction": "embedding",
            "verified": False,
            "outcome": self.outcome.value,
            "weights": self.request.weights,
            "prefix": self.record.prefix,
            "artifact": str(self.artifact) if self.artifact else None,
            "graft": self.graft.as_dict() if self.graft else None,
            "npu": [r.as_dict() for r in self.npu],
            "npu_published": self.npu_published if self.request.npu else None,
            "google_file": self.google,
            "base_checkpoint": self.base,
            "checkpoint": {
                "path": str(self.request.model),
                "sha256": self.checkpoint_sha256,
                "base_model": self.record.base_model,
                "base_model_revision": self.record.base_model_revision,
                "labels": len(self.record.labels),
                "decision_sha256": self.record.decision_sha256,
            },
            "request": self.request.as_dict(),
            "checks": self.checks.as_dict(),
            "limitations": list(self.limitations),
        }


def _refuse_request(request: EmbeddingExportRequest, record: EmbeddingRecord) -> None:
    if (record.base_model, record.base_model_revision) != (BASE_MODEL, BASE_REVISION):
        raise EmbeddingExportError(
            f"{request.model} records {record.base_model} at {record.base_model_revision}; the "
            f"graft's mapping was established for {BASE_MODEL} at {BASE_REVISION}"
        )
    if record.prefix != DECIDER_PREFIX:
        raise EmbeddingExportError(
            f"{request.model} was trained with the prefix {record.prefix!r}, not the decider's "
            f"{DECIDER_PREFIX!r}"
        )
    if not request.tuned_safetensors.is_file():
        found = sorted(p.name for p in request.model.iterdir()) if request.model.is_dir() else []
        raise EmbeddingExportError(
            f"no {BASE_FILE} in {request.model} (it holds {found[:12]}); the graft reads the "
            "fine-tune from that one file"
        )
    if request.base_litertlm is not None and not request.base_litertlm.is_file():
        raise EmbeddingExportError(f"--base-litertlm {request.base_litertlm}: no such file")
    if request.npu:
        problem = host_problem() or request.npu_env.refusal()
        if problem is not None:
            raise EmbeddingExportError(problem)
    for target in [request.output] + [dir_for(request.output_dir, soc) for soc in request.npu]:
        if target.exists() or target.is_symlink():
            raise EmbeddingExportError(
                f"{target} already exists; name another --output-dir, nothing is written "
                "over a file"
            )


def run_embedding_export(
    request: EmbeddingExportRequest,
    record: EmbeddingRecord,
    events: EventStream | None = None,
) -> EmbeddingExportResult:
    """Graft the checkpoint into Google's file. A refused request raises
    `EmbeddingExportError`; everything after the environment is a check."""
    envs.forget_reported_drops()
    events = events or EventStream(echo_json=False)
    _refuse_request(request, record)
    events.stage_started("convert", model=str(request.model), direction="embedding")
    result = EmbeddingExportResult(
        request=request, record=record, checks=CheckSet(name=f"convert:{request.weights}")
    )
    try:
        # The graft reads a copy of the fine-tune, and the digest convert.json
        # records -- the one `verify` binds the file to -- is the copy's: the
        # bytes the graft read, whatever happens to the checkpoint meanwhile.
        request.report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".litetune-checkpoint-", dir=request.report_path.parent
        ) as snapshot:
            _run(request, result, events, Path(snapshot))
    except BaseException as exc:
        # Whatever `_run` did not turn into a check: the report must not say
        # passed for a file the graft did not finish.
        logger.exception("convert stopped on an unexpected %s", type(exc).__name__)
        _add(result, events, Check.unchecked(GRAFT_CHECK, f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        if request.npu:
            _npus_on_disk(request, result, events)
        request.report_path.parent.mkdir(parents=True, exist_ok=True)
        request.report_path.write_text(
            json.dumps(result.as_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
        events.artifact(str(request.report_path), name=REPORT_NAME)
        events.stage_finished(result.outcome.value, direction="embedding")
    return result


def _add(result: EmbeddingExportResult, events: EventStream, check: Check) -> Check:
    result.checks.add(check)
    events.check(check)
    return check


def _run(
    request: EmbeddingExportRequest,
    result: EmbeddingExportResult,
    events: EventStream,
    snapshot: Path,
) -> None:
    env = request.env
    tuned = snapshot / request.tuned_safetensors.name
    shutil.copyfile(request.tuned_safetensors, tuned)
    result.checkpoint_sha256 = hash_file(tuned)
    with guard(ENV_CHECK) as sink:
        if request.auto_provision:
            env.provision(events=events)
        sink.append(
            Check.passed(ENV_CHECK, f"{env.name} ({env.identity}) ready at {env.path}")
            if env.ready
            else Check.unchecked(ENV_CHECK, f"environment {env.name!r} {env.unready_detail}")
        )
    if not _add(result, events, sink[0]).conclusive:
        result.limitations.append(f"the graft was not attempted: {sink[0].detail}")
        return

    # -- the inputs -------------------------------------------------------------
    wanted = [BASE_HUB_FILE] if request.base_litertlm else [GOOGLE_HUB_FILE, BASE_HUB_FILE]
    try:
        fetched = download(env, wanted, timeout=request.timeout_s)
    except DownloadFailed as exc:
        logger.error("could not download %s: %s", [f.filename for f in wanted], exc)
        _add(result, events, Check.unchecked(BASE_CHECKPOINT_CHECK, f"not fetched: {exc}"))
        result.limitations.append(f"the graft was not attempted: {exc}")
        return
    google = request.base_litertlm or fetched[0]
    base = fetched[-1]
    google_digest = hash_file(google)
    result.google = {
        "path": str(google),
        "source": "--base-litertlm" if request.base_litertlm else "downloaded",
        **GOOGLE_HUB_FILE.as_dict(),
        "sha256": google_digest,
    }
    result.base = {"path": str(base), **BASE_HUB_FILE.as_dict()}
    problem = _google_file_problem(google, google_digest)
    if problem is not None:
        _add(result, events, Check.failed(BASE_FILE_CHECK, problem, observed=result.google))
        return
    _add(
        result,
        events,
        Check.passed(BASE_FILE_CHECK, f"{google.name}, sha256 {GOOGLE_SHA256[:16]}…"),
    )
    result.base["sha256"] = hash_file(base)
    problem = _base_checkpoint_problem(base, result.base["sha256"])
    if problem is not None:
        _add(result, events, Check.failed(BASE_CHECKPOINT_CHECK, problem, observed=result.base))
        return
    _add(
        result,
        events,
        Check.passed(
            BASE_CHECKPOINT_CHECK, f"{BASE_MODEL} at {BASE_REVISION}, sha256 {BASE_SHA256[:16]}…"
        ),
    )

    # -- the graft ---------------------------------------------------------------
    try:
        grafted = graft(
            env,
            google,
            base,
            tuned,
            request.output,
            request.weights,  # type: ignore[arg-type]
            events=events,
            auto_provision=False,
            timeout=request.timeout_s,
        )
    except EmbeddingConvertError as exc:
        logger.error("the graft refused (%s): %s", exc.cause, exc)
        make = Check.unchecked if exc.cause == "run" else Check.failed
        _add(
            result, events, make(GRAFT_CHECK, f"{exc.cause}: {exc}", observed={"cause": exc.cause})
        )
        return
    except FileNotFoundError as exc:
        logger.exception("a graft input went missing")
        _add(result, events, Check.unchecked(GRAFT_CHECK, f"an input went missing: {exc}"))
        return
    result.graft = grafted
    _add(
        result,
        events,
        Check.passed(
            GRAFT_CHECK,
            f"{grafted.weights}: {grafted.bytes:,} bytes, read back; "
            f"{grafted.norms_written} norms written, folds {grafted.folds}",
            observed={"sha256": grafted.sha256, "bytes": grafted.bytes},
        ),
    )
    result.limitations.append(NOT_VERIFIED)
    events.artifact(str(grafted.output), name=ARTIFACT_NAME, verified=False)
    if request.npu:
        _compile_npus(request, result, events, grafted)


def _compile_npus(
    request: EmbeddingExportRequest,
    result: EmbeddingExportResult,
    events: EventStream,
    grafted: GraftResult,
) -> None:
    """Every `--npu` SoC compiled into a staging directory beside the outputs; when all of
    them compiled, each published into a directory of its own with its report
    (`_publish_npus`). Nothing published is ever removed or moved again."""
    with guard(NPU_ENV_CHECK) as sink:
        if request.auto_provision:
            request.npu_env.provision(events=events)
        sink.append(
            Check.passed(
                NPU_ENV_CHECK,
                f"{request.npu_env.name} ({request.npu_env.identity}) ready at "
                f"{request.npu_env.path}",
            )
            if request.npu_env.ready
            else Check.unchecked(
                NPU_ENV_CHECK,
                f"environment {request.npu_env.name!r} {request.npu_env.unready_detail}",
            )
        )
    if not _add(result, events, sink[0]).conclusive:
        result.limitations.append(f"no NPU file was compiled: {sink[0].detail}")
        return
    request.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".litetune-npu-stage-", dir=request.output_dir) as tmp:
        staged = [
            r
            for soc in request.npu
            if (r := _compile_npu(request, result, events, grafted.output, soc, Path(tmp)))
        ]
        if len(staged) != len(request.npu):
            failed = [soc for soc in request.npu if soc not in {r.soc for r in staged}]
            result.limitations.append(
                f"no NPU file was published: {', '.join(failed)} did not compile. Run again "
                "with another --output-dir"
            )
            return
        _publish_npus(request, result, staged)


def _mkdir(target: Path) -> None:
    """The SoC's directory, created by this run or not at all: an existing one -- made
    since the check before the compile -- is a FileExistsError, and nothing is replaced."""
    os.mkdir(target)


def _rename(source: Path, target: Path) -> None:
    """One staged file moved into the SoC's new directory; a name so a test can fail it."""
    source.rename(target)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def _publish_npus(
    request: EmbeddingExportRequest, result: EmbeddingExportResult, staged: list[NpuResult]
) -> None:
    """Each staged SoC published in turn: its report written in staging, its directory
    created exclusively, the model and then the report moved into it. The first failure
    stops the rest; an OSError is recorded, anything else (an interrupt) goes on once it
    is. Nothing is rolled back: what reached the disk is what the report says, read from
    the disk (`_npus_on_disk`)."""
    try:
        for compiled in staged:
            target = dir_for(request.output_dir, compiled.soc)
            published = compiled.moved_to(target / ARTIFACT_NAME)
            staged_report = compiled.output.parent / REPORT_NAME
            _write_report(staged_report, npu_report(result, published))
            _mkdir(target)
            result.npu_created[compiled.soc] = published
            _rename(compiled.output, target / ARTIFACT_NAME)
            _rename(staged_report, target / REPORT_NAME)
    except BaseException as exc:
        logger.exception("publishing the NPU files stopped")
        result.npu_publish_error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, OSError):
            raise


def _npus_on_disk(
    request: EmbeddingExportRequest, result: EmbeddingExportResult, events: EventStream
) -> None:
    """`npu` and `npu_published` from the disk: an SoC counts as published when the
    directory this run created holds a report whose `npu.sha256` is the model's there.
    A directory it created that does not is named as partial; none is removed."""
    published, partial = [], []
    for npu in result.npu_created.values():
        if _published_whole(npu.output):
            published.append(npu)
        else:
            partial.append(str(npu.output.parent))
    result.npu = published
    result.npu_published = bool(request.npu) and len(published) == len(request.npu)
    for npu in published:
        events.artifact(str(npu.output), name=npu_artifact_name(npu.soc), verified=False)
        events.artifact(str(npu.output.parent / REPORT_NAME), name=f"{REPORT_NAME} (npu {npu.soc})")
    if published:
        result.limitations.append(NPU_NOT_VERIFIED)
    if result.npu_publish_error is None and not partial:
        if published:
            _add(result, events, Check.passed(NPU_PUBLISH_CHECK, f"{len(published)} published"))
        return
    detail = f"published {len(published)} of {len(request.npu)}"
    if result.npu_publish_error:
        detail += f"; publishing stopped on {result.npu_publish_error}"
    if partial:
        detail += f"; written in part, left as they are: {', '.join(partial)}"
    _add(result, events, Check.unchecked(NPU_PUBLISH_CHECK, f"run: {detail}"))
    result.limitations.append(f"{detail}. Run the rest with another --output-dir")


def _published_whole(model: Path) -> bool:
    """Whether `model`'s directory holds it and a report naming its sha256."""
    try:
        report = json.loads((model.parent / REPORT_NAME).read_text(encoding="utf-8"))
        return model.is_file() and f"sha256:{report['npu']['sha256']}" == hash_file(model)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def npu_report(result: EmbeddingExportResult, npu: NpuResult) -> dict[str, Any]:
    """The report beside an NPU file: what it was compiled from, what it is, and that it
    is not verified. `kind` "npu" and `npu.sha256` are what `verify` reads to refuse it."""
    whole = result.as_dict()
    return {
        "schema": EMBEDDING_CONVERT_SCHEMA,
        "direction": "embedding",
        "kind": "npu",
        "verified": False,
        "outcome": Outcome.PASSED.value,
        "weights": result.request.weights,
        "prefix": result.record.prefix,
        "artifact": str(npu.output),
        "npu": npu.as_dict(),
        "signatures": list(NPU_SIGNATURES),
        "source_graft": whole["graft"],
        "google_file": result.google,
        "checkpoint": whole["checkpoint"],
        "request": result.request.as_dict(),
        "limitations": [NPU_NOT_VERIFIED],
    }


def _compile_npu(
    request: EmbeddingExportRequest,
    result: EmbeddingExportResult,
    events: EventStream,
    grafted: Path,
    soc: str,
    stage: Path,
) -> NpuResult | None:
    """The int8 graft's text encoder compiled for `soc` into `stage`, as one check; its
    result, or None. Anything that escapes the compile is this SoC's, as `run`."""
    name = npu_check(soc)
    try:
        compiled = compile_for_npu(
            request.npu_env,
            grafted,
            soc,
            stage / soc / ARTIFACT_NAME,
            events=events,
            auto_provision=False,
            timeout=request.npu_timeout_s,
        )
    except EmbeddingNpuError as exc:
        logger.error("the %s compile stopped (%s): %s", soc, exc.cause, exc)
        make = Check.unchecked if exc.cause in NPU_RUN_CAUSES else Check.failed
        _add(result, events, make(name, f"{exc.cause}: {exc}", observed={"cause": exc.cause}))
        return None
    except Exception as exc:  # noqa: BLE001 -- the SoC's own check, the others go on
        logger.exception("the %s compile raised", soc)
        _add(
            result,
            events,
            Check.unchecked(name, f"run: {type(exc).__name__}: {exc}", observed={"cause": "run"}),
        )
        return None
    context = compiled.report["qnn_context"]
    _add(
        result,
        events,
        Check.passed(
            name,
            f"{compiled.bytes:,} bytes; the text encoder ({compiled.encoder_bytes:,} bytes) one "
            f"DISPATCH_OP per signature over one {context['bytecode_size']:,}-byte QNN context, "
            f"{context['to_weights']}x the encoder's weight bytes (gate: {context['gate']}), "
            f"stamped for {'/'.join(compiled.report['stamp'])}, QAIRT {compiled.report['qairt']}; "
            f"compiled in {compiled.compile_s:.0f} s, read back",
            observed={"sha256": compiled.sha256, "bytes": compiled.bytes},
        ),
    )
    return compiled


def summarise(result: EmbeddingExportResult) -> list[str]:
    lines = [
        f"outcome: {result.outcome.value}",
        f"  embedding checkpoint {result.request.model}: {result.request.weights} graft into "
        f"{GOOGLE_FILE}",
    ]
    if result.graft is not None:
        lines.append(f"  {result.graft.bytes:,} bytes — {result.graft.output}")
    for npu in result.npu:
        lines.append(f"  {npu.soc} NPU: {npu.bytes:,} bytes — {npu.output}")
        if npu.as_dict()["device_run"]:
            lines.append(f"    on a device: {npu.as_dict()['device_run']}")
    for check in result.checks.checks:
        if check.outcome is not Outcome.PASSED:
            lines.append(f"  {check.outcome.value}: {check.name} — {check.detail}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    if result.graft is not None:
        lines.append("  produced, not verified: run `litetune verify` on the artifact")
    return lines
