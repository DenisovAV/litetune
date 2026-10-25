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
  `embedding_convert.BASE_REVISION`, downloaded the same way. The graft reads
  the tensor mapping and the fold factors against it;
- **the fine-tune**, `BASE_FILE` in the checkpoint directory `tune` wrote.

The file is written to `<output-dir>/<weights>/model.litertlm`, one directory
per weight mode as `convert` writes one per recipe (`export.ExportRequest.dir_for`),
with `convert.json` beside it. `verify` reads the weight mode from that report.

What the file embeds is not checked here: `litetune verify` measures it.
"""

from __future__ import annotations

import json
import logging
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
    GOOGLE_FILE,
    GOOGLE_REPO,
    GOOGLE_SHA256,
    GRAFT_TIMEOUT_S,
    EmbeddingConvertError,
    GraftResult,
    WeightMode,
    graft,
)
from litetune.embedding_tune import DECIDER_PREFIX, EmbeddingRecord
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

EMBEDDING_CONVERT_SCHEMA = "litetune.convert-embedding/1"

# The revision of `GOOGLE_REPO` the design pins (lora_project
# docs/specs/2026-10-25-tune-embeddinggemma2-design.md). The file fetched at it is
# checked against `GOOGLE_SHA256` all the same.
GOOGLE_REVISION = "9be6e8b90982095dc05c2bd162e4b954ee4dbac7"
# The checkpoint file the graft reads, in the base repository and in the
# directory `tune` saved the fine-tune to.
BASE_FILE = "model.safetensors"

WEIGHT_MODES: tuple[str, ...] = get_args(WeightMode)
# int4: on banking77 (77 labels) and CLINC150 (150) it cost 0.0027 and 0.0017 of
# the decider's accuracy (centroid of 10 examples per label) against the float
# fine-tune, int8 0.0002 on both, at 165 MB against 300 MB (measurement s5,
# 2026-10-25). `verify` measures the file written, and int8 is there for a task
# where int4 does not pass.
DEFAULT_WEIGHTS: WeightMode = "int4"

ARTIFACT_NAME = "model.litertlm"
REPORT_NAME = "convert.json"

# An allowance for two files of hundreds of MB each, not a measurement.
DOWNLOAD_TIMEOUT_S = 3600

ENV_CHECK = "export environment"
BASE_FILE_CHECK = "Google's file"
BASE_CHECKPOINT_CHECK = "base checkpoint"
GRAFT_CHECK = "graft"

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
    digest = hash_file(path)
    if digest != f"sha256:{GOOGLE_SHA256}":
        return (
            f"{path} has {digest}, not sha256:{GOOGLE_SHA256}, the {GOOGLE_FILE} of {GOOGLE_REPO} "
            f"at {GOOGLE_REVISION} that the graft's mapping was established on"
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

    def __post_init__(self) -> None:
        if self.weights not in WEIGHT_MODES:
            raise EmbeddingExportError(
                f"--weights {self.weights}: one of {', '.join(WEIGHT_MODES)}"
            )
        if self.timeout_s < 1:
            raise EmbeddingExportError(f"timeout_s must be at least 1, got {self.timeout_s}")
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
            "google_file": self.google,
            "base_checkpoint": self.base,
            "checkpoint": {
                "path": str(self.request.model),
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
    if request.output.exists() or request.output.is_symlink():
        raise EmbeddingExportError(
            f"{request.output} already exists; name another --output-dir, nothing is written "
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
        _run(request, result, events)
    finally:
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
    request: EmbeddingExportRequest, result: EmbeddingExportResult, events: EventStream
) -> None:
    env = request.env
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
    result.google = {
        "path": str(google),
        "source": "--base-litertlm" if request.base_litertlm else "downloaded",
        **GOOGLE_HUB_FILE.as_dict(),
        "sha256": GOOGLE_SHA256,
    }
    result.base = {"path": str(base), **BASE_HUB_FILE.as_dict()}
    problem = check_google_file(google)
    if problem is not None:
        _add(result, events, Check.failed(BASE_FILE_CHECK, problem, observed=result.google))
        return
    _add(
        result,
        events,
        Check.passed(BASE_FILE_CHECK, f"{google.name}, sha256 {GOOGLE_SHA256[:16]}…"),
    )
    _add(result, events, Check.passed(BASE_CHECKPOINT_CHECK, f"{BASE_MODEL} at {BASE_REVISION}"))

    # -- the graft ---------------------------------------------------------------
    try:
        grafted = graft(
            env,
            google,
            base,
            request.tuned_safetensors,
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


def summarise(result: EmbeddingExportResult) -> list[str]:
    lines = [
        f"outcome: {result.outcome.value}",
        f"  embedding checkpoint {result.request.model}: {result.request.weights} graft into "
        f"{GOOGLE_FILE}",
    ]
    if result.graft is not None:
        lines.append(f"  {result.graft.bytes:,} bytes — {result.graft.output}")
    for check in result.checks.checks:
        if check.outcome is not Outcome.PASSED:
            lines.append(f"  {check.outcome.value}: {check.name} — {check.detail}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    if result.graft is not None:
        lines.append("  produced, not verified: run `litetune verify` on the artifact")
    return lines
