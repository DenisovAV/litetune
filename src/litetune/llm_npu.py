"""`convert --npu <soc>` for an LLM checkpoint: a generation bundle compiled ahead of time for
one Qualcomm SoC, through litert-torch's `npu_export` stages.

What runs (MEASUREMENTS.md, "LLMs on a Qualcomm NPU", for every figure below):

  1. `npu_export` stage 1: the checkpoint exported with the split KV cache the NPU executor
     needs, `prefill_lengths` [128], `cache_length` from `--cache-length`, the flags this
     family's plain `convert` passes (chat template, model type, external embedder);
  2. stage 2: static-range calibration on prompts built from the training data and rendered
     with the template the bundle carries (`calibration_plan`): half of them the data's own
     prompts, half long prompts packed from its examples up to `cache_length - REPLY_TOKENS`,
     each followed by `REPLY_TOKENS` decode steps;
  3. stage 3: A16W8 static quantization, then `retype_dead_kv` before the bundle is packed;
  4. stage 4: the Qualcomm compile, once per SoC, and `check_compiled` on what it wrote.

Two defects of the public toolchain are worked around here, both measured on SM8750:

- **The last layer's prefill KV inputs.** The prefill graph ends at the last layer's KV
  slice, so that layer's attention over the cache feeds nothing and the exporter drops it;
  its `kv_cache_k_<last>` / `kv_cache_v_<last>` inputs then have no consumer, the quantizer
  (which works through consumers) leaves them FLOAT32 while every other graph has them
  INT16, and LiteRT-LM v0.18.0's NPU executor, which re-creates host-memory KV buffers from
  the decode signature, fails with "Custom allocation is too small" -- Qwen3-0.6B,
  Qwen2.5-0.5B and MiniCPM5-1B did not load. Retyping the two unused inputs to their decode
  namesakes changes no computation.
- **Calibration length.** A bundle calibrated on prompts up to ~230 tokens answered
  two-chunk prompts and turned five-chunk prompts into degenerate repetition (Qwen3-0.6B,
  Qwen2.5-0.5B, Gemma 3 270M); calibrated on prompts reaching the test prompts' length, the
  same models answered them, and a prompt past the longest calibration prompt failed again.
  So the calibration set must reach within one prefill chunk of the prompt limit, and the
  report says that longer prompts are outside what the bundle was calibrated for.

**Generation only.** A decision checkpoint is refused: option scoring rewinds the session
to the prompt once per option, and the NPU executor rolls back one token only
(`DECISION_REFUSAL`).

**Linux x86-64 only**, as `embedding_npu`: the compiler plugin ships in ai-edge-litert's
Linux x86-64 wheels only, and the QAIRT SDK it loads is downloaded when
ai-edge-litert-sdk-qualcomm is built (`envs.NPU_LLM`).

**What a device does with the file is not checked here**, and `litetune verify` refuses it:
LiteRT-LM's CPU and GPU cannot run its `DISPATCH_OP`s. `DEVICE_RUNS` says what ran where.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs, models
from litetune.checks import Check, CheckSet, Outcome
from litetune.embedding_npu import NPU_SOCS, host_problem
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

LLM_NPU_SCHEMA = "litetune.convert.llm-npu/1"
REPORT_NAME = "convert.json"
# The run's own report, beside the SoC directories: what was asked, what reached the disk.
RUN_REPORT_NAME = "npu-convert.json"
ARTIFACT_NAME = "model.litertlm"
# The weight recipe stage 1 exports with: the one litert-torch's packaged npu_export configs
# name (`gemma3_270m_qualcomm_sm8850.json`, `qwen3_0.6b_qualcomm_sm8750.json`); stage 3 then
# quantizes statically, A16W8.
STAGE1_RECIPE = "dynamic_wi8_afp32"
# The NPU runtime's prefill signature: the packaged configs and every bundle measured here
# use 128, and the npu_export notebook restricts it to 128.
PREFILL_LENGTH = 128
# What every measured bundle used. Not a property of any SoC.
DEFAULT_CACHE_LENGTH = 1024
MAX_CACHE_LENGTH = 4096
# Tokens of the cache kept free for the reply: the longest prompt the bundle is calibrated
# for is `cache_length - REPLY_TOKENS`, and calibration decodes up to this many steps after
# each prompt. A cache reservation, not a reply length that works: what held on a device is
# `REPLY_MEASURED`, per family.
REPLY_TOKENS = 128
CALIBRATION_DECODE_STEPS = REPLY_TOKENS
# The smallest cache that leaves one prefill chunk for a prompt besides the reply.
MIN_CACHE_LENGTH = PREFILL_LENGTH + REPLY_TOKENS
# 64 calibration prompts, as the packaged configs' calibration runs measured here used: half
# the data's own prompts, half long ones packed from its examples.
CALIBRATION_SHORT = 32
CALIBRATION_LONG = 32
# Fewer training rows than this cannot make a calibration set worth the name.
MIN_CALIBRATION_ROWS = 8
# Python flatbuffers cannot build a buffer of 2 GiB or more; the dead-KV retype repacks the
# stage-3 graph through it.
MAX_REPACK_BYTES = (2 << 30) - (64 << 20)
# Measured stage times on a 32-vCPU machine: stages 1-3 up to ~19 min with long
# calibration prompts, ~2 min per SoC compile. An allowance, not a measurement;
# `convert --timeout-s` replaces it.
NPU_LLM_TIMEOUT_S = 7200

# The families whose bundles have run on SM8750 through these stages (`DEVICE_RUNS` says how),
# and the one checkpoint size each ran at. A larger size of the same family has not compiled
# here and its stage-3 graph can pass what Python flatbuffers can repack, so the checkpoint
# itself is gated by its parameter count (`MEASURED_PARAMETERS`), and by `MEASURED_SIZES` on
# its name only when it has no safetensors to count.
FAMILIES: dict[str, str] = {
    "gemma-3-text": "gemma-3-270m-it",
    "qwen-3": "Qwen3-0.6B",
    "qwen-2.5": "Qwen2.5-0.5B-Instruct",
    "minicpm5": "MiniCPM5-1B",
}
MEASURED_SIZES: dict[str, str] = {
    "gemma-3-text": r"(?<![\d.])270m",
    "qwen-3": r"(?<![\d.])0[.-]6b",
    "qwen-2.5": r"(?<![\d.])0[.-]5b",
    "minicpm5": r"(?<![\d.])1b",
}
# Parameters in each measured checkpoint's safetensors without `lm_head.*` (the Hugging Face
# files of FAMILIES' models, counted from their headers). The output layer is left out because
# whether a file stores it is a choice of the saver, not of the model: Qwen3-0.6B's file holds
# a 151,936 x 1,024 `lm_head.weight` (155,582,464) although its config ties it, MiniCPM5-1B's
# an untied 130,560 x 1,536 (200,540,160), gemma-3-270m-it's and Qwen2.5-0.5B-Instruct's none.
MEASURED_PARAMETERS: dict[str, int] = {
    "gemma-3-text": 268_098_176,
    "qwen-3": 596_049_920,
    "qwen-2.5": 494_032_768,
    "minicpm5": 880_092_672,
}
# A fine-tune of the measured model has its count exactly; the band leaves room for a
# vocabulary grown by added tokens, 1% being 4,189 rows of gemma-3-270m's 640-wide embedding,
# 5,820 of Qwen3-0.6B's 1,024, 5,513 of Qwen2.5-0.5B's 896 and 5,729 of MiniCPM5-1B's 1,536.
# Any other size of these families is far outside it.
PARAMETER_BAND = 0.01
# Families that are known not to work here, or not measured on this toolchain, and why.
BLOCKED: dict[str, str] = {
    "gemma-4-e2b": (
        "the public exporter fuses Gemma 4's per-layer embedder into one lookup and the NPU "
        "runtime's fast path assumes Google's 35 (a SIGSEGV in HWPerLayerEmbeddingLookup)"
    ),
    "gemma-4-e4b": "as gemma-4-e2b",
    "qwen-3.5": (
        "litert-torch-nightly 0.10.0.dev20260926's split-cache export raises "
        "AttributeError: 'Qwen3_5StaticModelHFWrapper' object has no attribute 'model'"
    ),
    "functiongemma": (
        "not measured on this toolchain: its five-chunk tool prompts were measured only on an "
        "earlier one (0/20 parsed at cache_length 1024, 13/20 at 896), and the calibration "
        "here does not render tool declarations"
    ),
}
# What has run on a device, per SoC: the only statement this module makes about one.
DEVICE_RUNS: dict[str, str] = {
    "SM8750": (
        "one Galaxy S25 Ultra's NPU ran, on 2026-11-02, bundles `convert --npu` built "
        "(cache_length 1024), against float32 references: gemma-3-270m-it, Qwen3-0.6B and "
        "MiniCPM5-1B on needle prompts of two prefill chunks and up to the longest calibration "
        "prompt and on replies of up to 128 tokens; a banking77 fine-tune of "
        "Qwen2.5-0.5B-Instruct on 100 intent labels only. What this family's replies held to is "
        "among the limitations; MEASUREMENTS.md has the numbers"
    ),
}
# What a reply held to on that device, per family: put in every report, so that the 128 tokens
# the cache keeps free (`REPLY_TOKENS`) is not read as a reply length that works.
REPLY_MEASURED: dict[str, str] = {
    "gemma-3-text": (
        "gemma-3-text on SM8750: classification-length answers only. Answers of about five "
        "tokens matched float32 (11 of 12 after two-chunk prompts). Of 24 replies asked to count "
        "or copy for 128 tokens, one matched float32 throughout; after an 860-token prompt 8 of "
        '12 turned into repetition, after a 190-token prompt five counts stopped at "1" and '
        "six copies left the text within 19 tokens"
    ),
    "qwen-3": (
        "qwen-3 on SM8750: none of 48 replies of up to 128 tokens repeated itself; 30 of 36 "
        "short answers matched float32 exactly, six counts matched its 33 to 34 tokens, and six "
        "copies of the prompt left float32's text after 7 to 82 tokens. Replies past 128 tokens "
        "were not measured"
    ),
    "qwen-2.5": (
        "qwen-2.5 on SM8750: classification-length answers only were measured (a banking77 "
        "fine-tune's intent labels: 94 of 100 identical to float32); longer replies were not"
    ),
    "minicpm5": (
        "minicpm5 on SM8750: replies of 128 tokens, thinking on, found the code in 36 of 36 "
        "needle prompts as float32 did, but each left float32's wording within 2 to 127 tokens "
        "and one, after an 860-token prompt, looped on one phrase. Replies past 128 tokens were "
        "not measured"
    ),
}
NOT_RUN_SOC = (
    "{soc}: compiled, never run on a device; the calibration and prompt-length rules were "
    "measured on SM8750 (Hexagon V79) only"
)
DECISION_REFUSAL = (
    "a decision checkpoint is scored option by option: the decider prefills the prompt, "
    "saves a checkpoint and rewinds to it after each option, and LiteRT-LM's NPU executor "
    "rolls back one token only (v0.18.0 runtime/executor/"
    "llm_litert_npu_compiled_model_executor.cc:2139, SetCurrentStep: \"NPU executor's "
    'SetCurrentStep only supports rolling back one token at the end of decode."). '
    "Convert it without --npu"
)
# The keys a successful answer must carry.
ANSWER_KEYS: tuple[str, ...] = ("calibration", "socs", "versions", "stage_s", "dead_kv")
# The keys each SoC's entry must carry (`check_compiled`'s facts and the staged file).
SOC_KEYS: tuple[str, ...] = ("packed", "kv_types", "ops", "stamp", "qairt", "metadata")

NPU_ENV_CHECK = "npu environment"
NPU_REQUEST_CHECK = "npu request"
NPU_DATA_CHECK = "npu calibration data"
NPU_COMPILE_CHECK = "npu compile"
NPU_PUBLISH_CHECK = "npu publish"
NOT_VERIFIED = (
    "the NPU bundle is not verified: litetune verify runs LiteRT-LM's CPU or GPU, which "
    "cannot run its DISPATCH_OPs. Measure it on a phone with that SoC"
)
# A cause in an answer that says the run itself failed, not the model or the data.
RUN_CAUSES = ("run", "script")


class LlmNpuError(Exception):
    """The NPU bundle was not produced. `cause` is `run` for a failure of the run itself and
    `script` for an error in this module's own script; any other cause names what the
    request, the data or the compiled file failed."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


def dir_for(output_dir: Path, soc: str) -> Path:
    return output_dir / f"npu-{soc}"


def output_for(output_dir: Path, soc: str) -> Path:
    return dir_for(output_dir, soc) / ARTIFACT_NAME


def prompt_limit_tokens(cache_length: int) -> int:
    """The longest prompt a bundle of this cache length leaves room for, and calibrates to."""
    return cache_length - REPLY_TOKENS


def _normalised(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower())


def checkpoint_parameters(model: Path) -> int | None:
    """The parameter count in a checkpoint directory's safetensors without `lm_head.*`, read
    from their headers; `None` when it has none. Raises `ValueError` on a header that cannot be
    read or is not a safetensors header."""
    if not model.is_dir():
        return None
    files = sorted(model.glob("*.safetensors"))
    if not files:
        return None
    total = 0
    for path in files:
        try:
            with path.open("rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                if size > 100 << 20:
                    raise ValueError(f"a {size:,}-byte header")
                header = json.loads(f.read(size))
            if not isinstance(header, dict):
                raise ValueError(f"the header is a {type(header).__name__}, not an object")
            for name, entry in header.items():
                if name == "__metadata__" or name.startswith("lm_head."):
                    continue
                shape = entry.get("shape") if isinstance(entry, dict) else None
                if not (
                    isinstance(shape, list)
                    and all(
                        isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape
                    )
                ):
                    raise ValueError(f"tensor {name!r} has no shape of non-negative integers")
                total += math.prod(shape)
        except (OSError, struct.error, ValueError) as exc:
            raise ValueError(f"{path.name}: the safetensors header cannot be read: {exc}") from exc
    return total


def parameter_band(family: str) -> tuple[int, int]:
    """The parameter counts (without `lm_head.*`) accepted as `family`'s measured size."""
    measured = MEASURED_PARAMETERS[family]
    return math.floor(measured * (1 - PARAMETER_BAND)), math.ceil(measured * (1 + PARAMETER_BAND))


def family_problem(family: str | None, identity: str, parameters: int | None = None) -> str | None:
    """Why this family, at this checkpoint size, is refused on `--npu`, or `None`. The size is
    `parameters` when the checkpoint's own count is known, otherwise read from `identity`."""
    measured = ", ".join(f"{f} ({FAMILIES[f]})" for f in sorted(FAMILIES))
    if family in BLOCKED:
        return f"--npu: {identity} is {family}, which is refused here: {BLOCKED[family]}"
    if family not in FAMILIES:
        named = f"family {family}" if family else "no family litetune knows"
        return (
            f"--npu: {identity} is {named}; this path is open to {measured} only, the "
            "families and sizes measured on an NPU through its stages, and an unmeasured "
            "family is refused rather than compiled on trust"
        )
    if parameters is not None:
        low, high = parameter_band(family)
        if not low <= parameters <= high:
            return (
                f"--npu: {identity} is {family} with {parameters:,} parameters besides its "
                f"output layer; measured here at {FAMILIES[family]} "
                f"({MEASURED_PARAMETERS[family]:,}) only, and a checkpoint outside "
                f"{low:,}-{high:,} is another size that has not compiled on this path, so it "
                "is refused rather than compiled on trust"
            )
        return None
    if not re.search(MEASURED_SIZES[family], _normalised(identity)):
        return (
            f"--npu: {identity} is {family}, measured here at {FAMILIES[family]} only; another "
            "size has not compiled on this path (and a stage-3 graph over 2 GiB cannot be "
            "repacked), so it is refused rather than compiled on trust"
        )
    return None


def request_problem(
    socs: tuple[str, ...],
    cache_length: int,
    *,
    family: str | None,
    identity: str,
    decision: bool,
    parameters: int | None = None,
) -> str | None:
    """Why `convert --npu` of an LLM checkpoint is refused, or `None`."""
    unknown = [s for s in socs if s not in NPU_SOCS]
    if unknown:
        return f"--npu {', '.join(unknown)}: one of {', '.join(NPU_SOCS)}"
    if not socs:
        return "--npu names no SoC"
    if len(set(socs)) != len(socs):
        return f"--npu names a SoC twice: {', '.join(socs)}"
    if decision:
        return f"--npu: {identity} records a decision; {DECISION_REFUSAL}"
    if not MIN_CACHE_LENGTH <= cache_length <= MAX_CACHE_LENGTH:
        return (
            f"--cache-length {cache_length}: from {MIN_CACHE_LENGTH} (one {PREFILL_LENGTH}-token "
            f"prefill chunk for the prompt and {REPLY_TOKENS} tokens for the reply) to "
            f"{MAX_CACHE_LENGTH}"
        )
    if cache_length % PREFILL_LENGTH:
        return (
            f"--cache-length {cache_length}: a multiple of the prefill length {PREFILL_LENGTH}, "
            "as every measured bundle's was"
        )
    return family_problem(family, identity, parameters)


def read_calibration_rows(path: Path) -> list[dict[str, str]]:
    """The prepared data's rows, `prompt` and `completion` each a string. Raises `LlmNpuError`:
    cause `input` when the file cannot be read or holds too few rows (could-not-check), `data`
    when what it holds is not the training JSONL (failed)."""
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except OSError as exc:
        raise LlmNpuError("input", f"--calibration-data {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise LlmNpuError("data", f"--calibration-data {path}: not UTF-8: {exc}") from exc
    rows: list[dict[str, str]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise LlmNpuError("data", f"--calibration-data {path}:{number}: {exc}") from exc
        if not (
            isinstance(row, dict)
            and isinstance(row.get("prompt"), str)
            and isinstance(row.get("completion"), str)
        ):
            raise LlmNpuError(
                "data",
                f"--calibration-data {path}:{number}: a row needs a string prompt and completion "
                "(the training JSONL `prepare` writes)",
            )
        rows.append({"prompt": row["prompt"], "completion": row["completion"]})
    if len(rows) < MIN_CALIBRATION_ROWS:
        raise LlmNpuError(
            "input",
            f"--calibration-data {path}: {len(rows)} rows; at least {MIN_CALIBRATION_ROWS} are "
            "needed to calibrate",
        )
    return rows


@dataclass(frozen=True)
class LlmNpuRequest:
    model: str
    output_dir: Path
    socs: tuple[str, ...]
    calibration_data: Path
    cache_length: int = DEFAULT_CACHE_LENGTH
    base_model: str | None = None
    timeout_s: int = NPU_LLM_TIMEOUT_S
    auto_provision: bool = True
    env: envs.StageEnv = envs.NPU_LLM

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "output_dir": str(self.output_dir),
            "socs": list(self.socs),
            "calibration_data": str(self.calibration_data),
            "cache_length": self.cache_length,
            "prefill_length": PREFILL_LENGTH,
            "reply_tokens": REPLY_TOKENS,
            "calibration_decode_steps": CALIBRATION_DECODE_STEPS,
            "base_model": self.base_model,
            "timeout_s": self.timeout_s,
            "environment": {
                "name": self.env.name,
                "identity": self.env.identity,
                "system_requirements": list(self.env.system_requirements_on("linux")),
            },
        }


@dataclass(frozen=True)
class LlmNpuBundle:
    soc: str
    output: Path
    bytes: int
    sha256: str
    compiled: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "soc": self.soc,
            "output": str(self.output),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "kv_input_types": self.compiled.get("kv_types"),
            "ops_per_signature": self.compiled.get("ops"),
            "aux_ops_per_signature": (self.compiled.get("aux") or {}).get("ops"),
            "stamp": self.compiled.get("stamp"),
            "qairt": self.compiled.get("qairt"),
            "metadata": self.compiled.get("metadata"),
            "device_run": DEVICE_RUNS.get(self.soc),
        }


@dataclass
class LlmNpuResult:
    request: LlmNpuRequest
    checks: CheckSet = field(default_factory=lambda: CheckSet(name="convert:llm-npu"))
    family: str | None = None
    identity: str | None = None
    parameters: int | None = None
    prompt_mode: str | None = None
    exporter_flags: list[str] = field(default_factory=list)
    chat_template: dict[str, Any] | None = None
    answer: dict[str, Any] = field(default_factory=dict)
    bundles: list[LlmNpuBundle] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    publish_error: str | None = None
    limitations: list[str] = field(default_factory=list)

    @property
    def outcome(self) -> Outcome:
        return self.checks.outcome

    @property
    def calibration(self) -> dict[str, Any] | None:
        return self.answer.get("calibration")

    def as_dict(self) -> dict[str, Any]:
        calibration = self.calibration
        return {
            "schema": LLM_NPU_SCHEMA,
            "direction": "llm",
            "kind": "npu",
            "generation_only": True,
            "decisions": DECISION_REFUSAL,
            "verified": False,
            "outcome": self.outcome.value,
            "family": self.family,
            "identity": self.identity,
            "family_measured": FAMILIES.get(self.family or ""),
            "parameters": self.parameters,
            "prompt_mode": self.prompt_mode,
            "cache_length": self.request.cache_length,
            "prefill_length": PREFILL_LENGTH,
            "longest_calibrated_tokens": calibration.get("longest_tokens") if calibration else None,
            "calibration": calibration,
            "exporter_flags": list(self.exporter_flags),
            "chat_template": self.chat_template,
            "dead_kv": self.answer.get("dead_kv"),
            "stage_s": self.answer.get("stage_s"),
            "versions": self.answer.get("versions"),
            "npu": [b.as_dict() for b in self.bundles],
            "partial": list(self.partial),
            "request": self.request.as_dict(),
            "checks": self.checks.as_dict(),
            "limitations": list(self.limitations),
        }


def _add(result: LlmNpuResult, events: EventStream, check: Check) -> Check:
    result.checks.add(check)
    events.check(check)
    return check


def _cannot(result: LlmNpuResult, events: EventStream, name: str, detail: str) -> LlmNpuResult:
    """A refusal that says nothing about the model: could-not-check (exit 4)."""
    _add(result, events, Check.unchecked(name, detail))
    events.stage_finished(result.outcome.value, attempted=0)
    return result


def _failed(result: LlmNpuResult, events: EventStream, name: str, detail: str) -> LlmNpuResult:
    """The user's data is wrong: failed (exit 1)."""
    _add(result, events, Check.failed(name, detail))
    events.stage_finished(result.outcome.value, attempted=0)
    return result


def _decision_problem(record: dict[str, Any] | None) -> tuple[bool, str | None]:
    """(records a decision, why the record cannot be read as one)."""
    if record is None or "decision" not in record or record["decision"] is None:
        return False, None
    if not isinstance(record["decision"], dict):
        return True, (
            f"litetune.json's `decision` is a {type(record['decision']).__name__}, not an "
            "object; it cannot be read, and a decision checkpoint is refused on --npu anyway"
        )
    return True, None


def run_llm_npu_export(
    request: LlmNpuRequest, events: EventStream | None = None, *, host: str | None = None
) -> LlmNpuResult:
    """`convert --npu` for an LLM checkpoint. Every failure is a check in the result: a
    refusal of the request or a failure of the run is could-not-check, and only the data or
    the compiled file being wrong is failed."""
    # Imported here: export imports nothing of this module, and the CLI imports both.
    from litetune.export import (
        ExportRequest,
        compose_chat_template,
        installed_version,
        repair_vocab_file,
        resolve_toolchain,
    )

    envs.forget_reported_drops()
    events = events or EventStream(echo_json=False)
    events.stage_started("convert-npu", model=request.model, socs=list(request.socs))
    result = LlmNpuResult(request=request)
    problem = (host if host is not None else host_problem()) or request.env.refusal()
    if problem:
        return _cannot(result, events, NPU_REQUEST_CHECK, problem)
    try:
        export_request = ExportRequest(
            model=request.model,
            output_dir=request.output_dir,
            recipes=(STAGE1_RECIPE,),
            base_model=request.base_model,
            env=request.env,
            auto_provision=request.auto_provision,
        )
    except ValueError as exc:
        return _cannot(result, events, NPU_REQUEST_CHECK, str(exc))
    if export_request.identity_conflict:
        return _cannot(result, events, NPU_REQUEST_CHECK, export_request.identity_conflict)
    try:
        read = models.read_training_record(request.model)
    except (OSError, ValueError) as exc:
        return _cannot(result, events, NPU_REQUEST_CHECK, f"{request.model}: {exc}")
    record = read[0] if read else None
    plan = export_request.plan
    result.family = plan.rules.family if plan.rules else None
    result.identity = request.base_model or (record or {}).get("base_model") or request.model
    decision, unreadable = _decision_problem(record)
    if unreadable:
        return _cannot(result, events, NPU_REQUEST_CHECK, unreadable)
    try:
        parameters = checkpoint_parameters(Path(request.model))
    except ValueError as exc:
        return _cannot(result, events, NPU_REQUEST_CHECK, f"{request.model}: {exc}")
    result.parameters = parameters
    problem = request_problem(
        request.socs,
        request.cache_length,
        family=result.family,
        identity=result.identity,
        decision=decision,
        parameters=parameters,
    )
    if problem:
        return _cannot(result, events, NPU_REQUEST_CHECK, problem)
    if parameters is None:
        result.limitations.append(
            f"{request.model} has no safetensors to count its parameters in; its size was "
            f"taken from its name, {result.identity}"
        )
    mode = (record or {}).get("prompt_mode")
    if mode not in ("runtime_rendered", "prerendered"):
        return _cannot(
            result,
            events,
            NPU_REQUEST_CHECK,
            f"--npu: {request.model} records no prompt mode (its litetune.json, which `tune` "
            "writes); the calibration prompts are rendered by it",
        )
    if (record or {}).get("declarations_sha256"):
        return _cannot(
            result,
            events,
            NPU_REQUEST_CHECK,
            f"--npu: {request.model} records tool declarations; a tool-calling bundle has not "
            "been measured on this path, and its calibration would not render them",
        )
    result.prompt_mode = mode
    try:
        rows = read_calibration_rows(request.calibration_data)
    except LlmNpuError as exc:
        grade = _failed if exc.cause == "data" else _cannot
        return grade(result, events, NPU_DATA_CHECK, str(exc))
    taken = [dir_for(request.output_dir, soc) for soc in request.socs]
    taken.append(request.output_dir / RUN_REPORT_NAME)
    for path in taken:
        if path.exists() or path.is_symlink():
            return _cannot(
                result,
                events,
                NPU_REQUEST_CHECK,
                f"{path} already exists; nothing is written over it. Use another --output-dir",
            )
    _add(
        result,
        events,
        Check.passed(NPU_REQUEST_CHECK, f"{result.family} ({result.identity}), {len(rows)} rows"),
    )
    broken_record = models.provenance_error(request.model)
    if broken_record:
        result.limitations.append(broken_record)
        events.note(broken_record, model=request.model)
    for text in plan.limitations:
        result.limitations.append(text)
    for note in plan.notes:
        events.note(note, model=request.model, family=result.family)
    for check in plan.checks:
        _add(result, events, check)
    if not plan.usable:
        unresolved = next(c for c in plan.checks if c.outcome is Outcome.UNCHECKED)
        result.limitations.append(f"no NPU bundle was compiled: {unresolved.detail}")
        events.stage_finished(result.outcome.value, attempted=0)
        return result

    env = request.env
    try:
        if request.auto_provision:
            env.provision(events=events)
    except Exception as exc:  # noqa: BLE001 -- a provisioning failure is a check, not a crash
        logger.exception("provisioning %s failed", env.name)
        return _cannot(result, events, NPU_ENV_CHECK, f"{env.name}: {exc}")
    if not env.ready:
        return _cannot(
            result, events, NPU_ENV_CHECK, f"environment {env.name!r} {env.unready_detail}"
        )
    _add(result, events, Check.passed(NPU_ENV_CHECK, f"{env.name} ({env.identity}) ready"))

    # As run_export does before its sweep: a stale `vocab_file` kills stage 1 after the
    # model has loaded, and the toolchain's transformers must be one this family supports.
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
    if plan.rules is not None and plan.rules.min_transformers:
        toolchain = resolve_toolchain(env)
        version_check = models.transformers_check(
            request.model,
            plan.rules,
            installed_version(toolchain, "transformers"),
            f"the {env.name} environment",
            unknown_reason=toolchain.unresolved_reason or "",
        )
        if version_check.outcome is Outcome.FAILED:
            _add(result, events, version_check)
            events.stage_finished(result.outcome.value, attempted=0)
            return result
        if not version_check.conclusive:
            result.limitations.append(version_check.detail)

    template: Path | None = None
    if export_request.composes_chat_template:
        check, composed = compose_chat_template(export_request)
        _add(result, events, check)
        if composed is None:
            result.limitations.append("no NPU bundle was compiled: no chat template")
            events.stage_finished(result.outcome.value, attempted=0)
            return result
        template = composed.path
        result.chat_template = composed.as_dict()
    argv = export_request.argv(STAGE1_RECIPE, template)
    # argv: litert-torch export_hf --model --output_dir --quantization_recipe, then the flags.
    result.exporter_flags = list(argv[5:])

    request.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=".litetune-npu-llm-", dir=request.output_dir
        ) as tmp:
            work = Path(tmp)
            try:
                result.answer = _compile(request, result, rows, mode, template, work)
            except LlmNpuError as exc:
                check = (Check.unchecked if exc.cause in RUN_CAUSES else Check.failed)(
                    NPU_COMPILE_CHECK, f"{exc.cause}: {exc}"
                )
                _add(result, events, check)
                result.limitations.append("no NPU bundle was compiled")
                events.stage_finished(result.outcome.value, attempted=1)
                return result
            calibration = result.answer["calibration"]
            _add(
                result,
                events,
                Check.passed(
                    NPU_COMPILE_CHECK,
                    f"{len(request.socs)} SoC(s); calibrated on {calibration['count']} prompts "
                    f"of {calibration['shortest_tokens']}-{calibration['longest_tokens']} tokens",
                    observed={"dead_kv": result.answer["dead_kv"]},
                ),
            )
            _limitations(request, result)
            _publish(request, result, events, work)
    finally:
        _write_run_report(request, result, events)
    events.stage_finished(result.outcome.value, attempted=1)
    return result


def _limitations(request: LlmNpuRequest, result: LlmNpuResult) -> None:
    """What every SoC's report must carry, appended before any of them is written."""
    longest = result.answer["calibration"]["longest_tokens"]
    result.limitations.append(NOT_VERIFIED)
    result.limitations.append(
        f"prompts longer than {longest} tokens (rendered) are outside what this bundle was "
        "calibrated for: measured on SM8750, prompts past the longest calibration prompt "
        "turned into degenerate repetition. Do not serve them"
    )
    if result.family in REPLY_MEASURED:
        result.limitations.append(REPLY_MEASURED[result.family])
    decode = result.answer["calibration"].get("decode")
    if not decode:
        result.limitations.append(
            "how many decode steps the calibration replies ran was not recorded; a reply's "
            "positions past its prompt may be outside what was calibrated"
        )
    elif decode["max_steps"] < decode["cap"] or decode["full"] < decode["count"]:
        result.limitations.append(
            f"calibration replies ran {decode['median_steps']} decode steps at the median and "
            f"{decode['max_steps']} at most ({decode['full']} of {decode['count']} reached the "
            f"{decode['cap']}-step cap; each stops at a stop token): a reply longer than that "
            "reaches positions the calibration saw less often or not at all"
        )
    if request.cache_length != DEFAULT_CACHE_LENGTH:
        result.limitations.append(
            f"cache_length {request.cache_length} has not been measured; every measured bundle "
            f"used {DEFAULT_CACHE_LENGTH}"
        )
    for soc in request.socs:
        if soc not in DEVICE_RUNS:
            result.limitations.append(NOT_RUN_SOC.format(soc=soc))


def _compile(
    request: LlmNpuRequest,
    result: LlmNpuResult,
    rows: list[dict[str, str]],
    mode: str,
    template: Path | None,
    work: Path,
) -> dict[str, Any]:
    script, spec, answer = work / "npu_llm.py", work / "spec.json", work / "answer.json"
    data = work / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    script.write_text(_NPU_LLM_SCRIPT, encoding="utf-8")
    spec.write_text(
        json.dumps(
            {
                "model": request.model,
                "flags": result.exporter_flags,
                "chat_template": str(template) if template else None,
                "socs": list(request.socs),
                "cache_length": request.cache_length,
                "prefill_length": PREFILL_LENGTH,
                "prompt_limit_tokens": prompt_limit_tokens(request.cache_length),
                "short": CALIBRATION_SHORT,
                "long": CALIBRATION_LONG,
                "decode_steps": CALIBRATION_DECODE_STEPS,
                "max_repack_bytes": MAX_REPACK_BYTES,
                "prompt_mode": mode,
                "rows": str(data),
                "work": str(work),
                "answer": str(answer),
            }
        ),
        encoding="utf-8",
    )
    try:
        proc = request.env.run(["python", str(script), str(spec)], timeout=request.timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise LlmNpuError("run", f"the export did not finish within {request.timeout_s}s") from exc
    except OSError as exc:
        raise LlmNpuError(
            "run", f"the export could not be started: {type(exc).__name__}: {exc}"
        ) from exc
    said = (proc.stderr or "").strip()[-400:] or "no stderr"
    if not answer.is_file():
        reading = read_returncode(proc.returncode)
        raise LlmNpuError("run", f"the export {reading.describe('the script')}: {said}")
    try:
        got = json.loads(answer.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LlmNpuError("run", f"the export's answer could not be read: {exc}") from exc
    if not isinstance(got, dict):
        raise LlmNpuError("run", f"the export's answer is not an object: {got!r:.200}")
    if not got.get("ok"):
        raise LlmNpuError(str(got.get("cause", "run")), str(got.get("reason", got)))
    if proc.returncode != 0:
        reading = read_returncode(proc.returncode)
        raise LlmNpuError("run", f"the export answered ok but {reading.describe('the script')}")
    missing = [k for k in ANSWER_KEYS if k not in got]
    if missing:
        raise LlmNpuError("run", f"the export's answer has no {', '.join(missing)}")
    socs = got["socs"]
    if not isinstance(socs, dict) or sorted(socs) != sorted(request.socs):
        raise LlmNpuError("run", f"the export's answer names SoCs {sorted(socs or {})}")
    for soc, entry in socs.items():
        lacking = [k for k in SOC_KEYS if k not in (entry or {})]
        if lacking:
            raise LlmNpuError("run", f"the export's answer for {soc} has no {', '.join(lacking)}")
        packed = Path(str(entry["packed"]))
        if not packed.is_file():
            raise LlmNpuError("run", f"the export's answer names {packed} for {soc}, not there")
        problem = kv_types_problem(entry["kv_types"])
        if problem:
            raise LlmNpuError("compiled", f"{soc}: {problem}")
        if not entry["qairt"]:
            raise LlmNpuError("compiled", f"{soc}: the compiled bundle names no QAIRT build")
    return got


def kv_types_problem(kv_types: Any) -> str | None:
    """Why the compiled graphs' KV-cache inputs would not load, or `None`: every signature's
    KV inputs must be INT16 (`retype_dead_kv`'s purpose)."""
    if not isinstance(kv_types, dict) or not kv_types:
        return "the compiled bundle's KV-cache input types were not read"
    for signature, counts in kv_types.items():
        if not isinstance(counts, dict) or not counts:
            return f"signature {signature} has no KV-cache inputs"
        other = {t: n for t, n in counts.items() if t != "INT16"}
        if other:
            return (
                f"signature {signature} has KV-cache inputs {other}, not INT16 only: the NPU "
                "executor shares one buffer per input across signatures and would not load it"
            )
    return None


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def _published_whole(model: Path) -> bool:
    """Whether `model`'s directory holds it and a report naming its sha256."""
    try:
        report = json.loads((model.parent / REPORT_NAME).read_text(encoding="utf-8"))
        return model.is_file() and f"sha256:{report['npu']['sha256']}" == hash_file(model)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _publish(request: LlmNpuRequest, result: LlmNpuResult, events: EventStream, work: Path) -> None:
    """Each SoC's bundle published in turn, as `embedding_export._publish_npus` does: its
    report written beside the staged file, the SoC's directory created exclusively, the
    bundle and then the report hard-linked in. The first failure stops the rest; nothing is
    rolled back. What counts as published is read back from the disk afterwards
    (`_published_whole`), a directory this run created that is not whole is named as partial,
    and an interrupt is re-raised once that is recorded."""
    created: dict[str, LlmNpuBundle] = {}
    try:
        for soc in request.socs:
            entry = result.answer["socs"][soc]
            staged = Path(str(entry["packed"]))
            target = dir_for(request.output_dir, soc)
            bundle = LlmNpuBundle(
                soc=soc,
                output=target / ARTIFACT_NAME,
                bytes=staged.stat().st_size,
                sha256=hash_file(staged).removeprefix("sha256:"),
                compiled=entry,
            )
            staged_report = work / f"{soc}.{REPORT_NAME}"
            report = result.as_dict()
            report["npu"] = bundle.as_dict()
            _write_report(staged_report, report)
            # Recorded before the mkdir, so an interrupt right after it still names the
            # directory; taken back if it was not ours to make.
            created[soc] = bundle
            try:
                os.mkdir(target)
            except FileExistsError:
                del created[soc]
                raise
            os.link(staged, target / ARTIFACT_NAME)
            staged.unlink()
            os.link(staged_report, target / REPORT_NAME)
            staged_report.unlink()
    except BaseException as exc:
        logger.exception("publishing the NPU bundles stopped")
        result.publish_error = f"{type(exc).__name__}: {exc}"
        _on_disk(request, result, events, created)
        if not isinstance(exc, OSError):
            raise
        return
    _on_disk(request, result, events, created)


def _on_disk(
    request: LlmNpuRequest,
    result: LlmNpuResult,
    events: EventStream,
    created: dict[str, LlmNpuBundle],
) -> None:
    published, partial = [], []
    for bundle in created.values():
        if _published_whole(bundle.output):
            published.append(bundle)
            events.artifact(
                str(bundle.output), name=f"{ARTIFACT_NAME} (npu {bundle.soc})", verified=False
            )
            events.artifact(
                str(bundle.output.parent / REPORT_NAME), name=f"{REPORT_NAME} (npu {bundle.soc})"
            )
        else:
            partial.append(str(bundle.output.parent))
    result.bundles, result.partial = published, partial
    if result.publish_error is None and not partial:
        _add(result, events, Check.passed(NPU_PUBLISH_CHECK, f"{len(published)} published"))
        return
    detail = f"published {len(published)} of {len(request.socs)}"
    if result.publish_error:
        detail += f"; publishing stopped on {result.publish_error}"
    if partial:
        detail += f"; written in part, left as they are: {', '.join(partial)}"
    _add(result, events, Check.unchecked(NPU_PUBLISH_CHECK, f"run: {detail}"))
    result.limitations.append(f"{detail}. Run the rest with another --output-dir")


def _write_run_report(request: LlmNpuRequest, result: LlmNpuResult, events: EventStream) -> None:
    """The run's report beside the SoC directories, created new: what was asked and what
    reached the disk, whatever stopped the run after the output directory existed."""
    if not request.output_dir.is_dir():
        return
    path = request.output_dir / RUN_REPORT_NAME
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(json.dumps(result.as_dict(), indent=2, default=str) + "\n")
        events.artifact(str(path), name=RUN_REPORT_NAME)
    except OSError as exc:
        logger.exception("the run report was not written")
        result.limitations.append(f"the run report {path} was not written: {exc}")


def npu_target_of(model: Path) -> str | None:
    """The SoC `model` was compiled for, when the report beside it is this module's and names
    this file's sha256; else None."""
    report_path = Path(model).parent / REPORT_NAME
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(report, dict) or report.get("schema") != LLM_NPU_SCHEMA:
        return None
    npu = report.get("npu")
    if report.get("kind") != "npu" or not isinstance(npu, dict):
        return None
    try:
        if f"sha256:{npu.get('sha256')}" != hash_file(Path(model)):
            return None
    except OSError:
        return None
    soc = npu.get("soc")
    return soc if isinstance(soc, str) else None


def summarise(result: LlmNpuResult) -> list[str]:
    lines = [
        f"outcome: {result.outcome.value}",
        f"  {result.request.model} ({result.family or 'unknown family'}): generation bundle for "
        f"{', '.join(result.request.socs)}, cache_length {result.request.cache_length}",
    ]
    calibration = result.calibration
    if calibration:
        lines.append(
            f"  calibrated on {calibration['count']} prompts of {calibration['shortest_tokens']}-"
            f"{calibration['longest_tokens']} tokens; longer prompts are outside it"
        )
    for bundle in result.bundles:
        lines.append(f"  {bundle.soc} NPU: {bundle.bytes:,} bytes — {bundle.output}")
        if DEVICE_RUNS.get(bundle.soc):
            lines.append(f"    on a device: {DEVICE_RUNS[bundle.soc]}")
    for check in result.checks.checks:
        if check.outcome is not Outcome.PASSED:
            lines.append(f"  {check.outcome.value}: {check.name} — {check.detail}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    return lines


# The script that runs inside envs.NPU_LLM. Its top level imports the standard library only,
# so tests/test_llm_npu.py runs `retype_dead_kv`, `kv_quantization_problem`, `kv_type_counts`,
# `make_render`, `render_problem`, `long_targets`, `calibration_plan`, `exporter_kwargs`,
# `check_compiled(_aux)`, `decode_summary`,
# `recording_decode` and `compare_metadata` for real without the toolchain.
_NPU_LLM_SCRIPT = r'''
"""npu_export stages 1-4 for one LLM checkpoint, with calibration from its training data and
the dead last-layer KV inputs retyped. Reads JSON spec from argv[1]; writes spec["answer"]:
  {"ok": true, "calibration", "socs": {soc: {packed, kv_types, ops, stamp, qairt, metadata}},
   "versions", "stage_s", "dead_kv"}
  or {"ok": false, "cause", "reason"}; exits 0 for the first and 3 for the second.
"""

import hashlib
import importlib.metadata
import json
import os
import random
import re
import statistics
import sys
import time
import traceback

KV_PREFIX = "kv_cache_"
# The calibration prompts are rendered already: npu_calibrate's own chat formatting would wrap
# them a second time (`calibrate`'s enable_formatting defaults to True).
CALIBRATE_KWARGS = {"enable_formatting": False}
STAMP_BYTES = 250
QAIRT = re.compile(rb"v(2\.\d+\.\d+\.\d+)")
VERSIONED = (
    "litert-torch-nightly",
    "ai-edge-litert-nightly",
    "ai-edge-quantizer-nightly",
    "ai-edge-litert-sdk-qualcomm",
    "litert-lm-builder",
    "transformers",
    "torch",
    "flatbuffers",
    "numpy",
)


class Refused(Exception):
    def __init__(self, cause, reason):
        super().__init__(reason)
        self.cause = cause


def _name(x):
    return x.decode() if isinstance(x, bytes) else (x or "")


def kv_type_counts(model, type_names):
    """{signature: {type name: count}} over each signature's KV-cache inputs."""
    out = {}
    for sd in model.signatureDefs or []:
        sg = model.subgraphs[sd.subgraphIndex]
        counts = {}
        for i in sd.inputs or []:
            if _name(i.name).startswith(KV_PREFIX):
                t = type_names[sg.tensors[i.tensorIndex].type]
                counts[t] = counts.get(t, 0) + 1
        out[_name(sd.signatureKey)] = counts
    return out


def _read_tensors(sg, sd):
    """Tensor indices this signature's subgraph reads or hands out: every op's inputs, the
    subgraph's outputs and the signature's outputs."""
    read = set()
    for op in sg.operators or []:
        read.update(int(i) for i in (op.inputs if op.inputs is not None else []) if int(i) >= 0)
    read.update(int(i) for i in (sg.outputs if sg.outputs is not None else []))
    read.update(int(o.tensorIndex) for o in sd.outputs or [])
    return read


def retype_dead_kv(model, type_names, copy_quantization):
    """Every prefill KV-cache input that its subgraph neither reads nor hands out takes the
    type and quantization of the same-named input of the decode signature. Changes `model`;
    returns what changed. The inputs are unused, so no computation changes."""
    sigs = {_name(sd.signatureKey): sd for sd in model.signatureDefs or []}
    decode = sigs.get("decode")
    if decode is None:
        raise Refused("input", f"the stage-3 graph has no decode signature (it has {sorted(sigs)})")
    dsg = model.subgraphs[decode.subgraphIndex]
    decode_tensors = {_name(i.name): dsg.tensors[i.tensorIndex] for i in decode.inputs or []}
    changed = []
    for key, sd in sigs.items():
        if not key.startswith("prefill"):
            continue
        sg = model.subgraphs[sd.subgraphIndex]
        read = _read_tensors(sg, sd)
        for i in sd.inputs or []:
            name = _name(i.name)
            if not name.startswith(KV_PREFIX) or int(i.tensorIndex) in read:
                continue
            tensor, twin = sg.tensors[i.tensorIndex], decode_tensors.get(name)
            if twin is None or tensor.type == twin.type:
                continue
            if list(tensor.shape) != list(twin.shape):
                raise Refused(
                    "input",
                    f"{key}/{name}: shape {list(tensor.shape)} differs from decode's "
                    f"{list(twin.shape)}",
                )
            changed.append(
                {
                    "signature": key,
                    "input": name,
                    "from": type_names[tensor.type],
                    "to": type_names[twin.type],
                }
            )
            tensor.type = twin.type
            tensor.quantization = copy_quantization(twin.quantization)
    return changed


def _quant_key(tensor, type_names):
    q = tensor.quantization
    scale = tuple(float(x) for x in (getattr(q, "scale", None) if q is not None else None) or [])
    zero = tuple(int(x) for x in (getattr(q, "zeroPoint", None) if q is not None else None) or [])
    return (type_names[tensor.type], scale, zero)


def kv_quantization_problem(models, type_names):
    """Why the KV-cache inputs of these graphs (the prefill/decode graph and the aux
    cache-update graph) would not share one buffer each, or None: every input of one name
    must have one type, scale and zero point across every signature that takes it."""
    seen = {}
    for model in models:
        for sd in model.signatureDefs or []:
            sg = model.subgraphs[sd.subgraphIndex]
            for i in sd.inputs or []:
                name = _name(i.name)
                if name.startswith(KV_PREFIX):
                    key = _quant_key(sg.tensors[i.tensorIndex], type_names)
                    seen.setdefault(name, {}).setdefault(key, []).append(_name(sd.signatureKey))
    for name, keys in sorted(seen.items()):
        if len(keys) > 1:
            return f"{name} has {len(keys)} different type/quantizations across signatures: {keys}"
        ((kind, _, _),) = keys
        if kind != "INT16":
            return f"{name} is {kind}, not INT16"
    return None


def make_render(apply_chat_template, prompt_mode, parts):
    """How a training prompt reaches the model: rendered as one user turn for
    `runtime_rendered`, as it is for `prerendered`. With the bundle's own template (`parts`), the
    content is a list of text parts, as LiteRT-LM from 0.18.0 hands it to the template; the
    canonical Qwen 3 and MiniCPM5 templates iterate it, and given a string they render an empty
    user turn."""
    if prompt_mode != "runtime_rendered":
        return lambda text: text

    def render(text):
        content = [{"type": "text", "text": text}] if parts else text
        return apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )

    return render


# A user turn with nothing a template trims or escapes: if a render lacks it, the template
# dropped the content (a template may trim the user's own text, so that is not the test).
RENDER_SENTINEL = "litetune calibration probe 7d1f"


def render_problem(render):
    """Why the calibration prompts would not carry the training text, or None."""
    rendered = render(RENDER_SENTINEL)
    if RENDER_SENTINEL not in rendered:
        return (
            f"the chat template renders the user turn {RENDER_SENTINEL!r} as {rendered[:120]!r}, "
            "without its text; calibrating on that would measure an empty turn"
        )
    return None


def long_targets(limit, count):
    """`count` target lengths spread evenly from half of `limit` up to `limit` itself."""
    if count <= 0:
        return []
    low = max(1, limit // 2)
    if count == 1:
        return [limit]
    return [low + (limit - low) * k // (count - 1) for k in range(count)]


def calibration_plan(rows, render, count_tokens, limit, short, long, reach, seed=0):
    """The calibration prompts: `short` of the data's own prompts, rendered, and `long`
    prompts packed from its examples (prompt and completion, one after another) followed by
    one more prompt, each grown towards its target from `long_targets` with whatever example
    still fits. Refuses when a data prompt is longer than `limit`, and when the longest prompt
    falls short of `limit - reach`: the prompts a bundle serves must not outrun its
    calibration. Returns (prompts, summary)."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rendered = []
    for i in order:
        text = render(rows[i]["prompt"])
        n = count_tokens(text)
        if n > limit:
            raise Refused(
                "data",
                f"a training prompt renders to {n} tokens, more than the {limit} this cache "
                "length leaves for a prompt; raise --cache-length",
            )
        rendered.append((text, n))
    shorts = rendered[:short]
    targets = long_targets(limit, long)
    longs, cursor = [], 0
    for target in targets:
        pieces, best, misses = [], None, 0
        while misses < len(rows):
            row = rows[order[cursor % len(order)]]
            tail = rows[order[(cursor + 1) % len(order)]]["prompt"]
            cursor += 1
            candidate = pieces + [row["prompt"] + " " + row["completion"]]
            text = render("\n\n".join(candidate + [tail]))
            n = count_tokens(text)
            # Too long, or no longer than before (a template that drops the text): a miss, so the
            # loop ends whatever the template does.
            if n > limit or (best is not None and n <= best[1]):
                misses += 1
                continue
            pieces, best, misses = candidate, (text, n), 0
            if n >= target:
                break
        if best is not None:
            longs.append(best)
    prompts = [t for t, _ in shorts + longs]
    lengths = [n for _, n in shorts + longs]
    longest = max(lengths)
    if longest < limit - reach:
        raise Refused(
            "data",
            f"the longest calibration prompt is {longest} tokens; it must reach within {reach} "
            f"of the {limit} the cache leaves for a prompt, and the training examples could not "
            "be packed that far",
        )
    summary = {
        "count": len(prompts),
        "short": {
            "count": len(shorts),
            "min_tokens": min((n for _, n in shorts), default=0),
            "max_tokens": max((n for _, n in shorts), default=0),
        },
        "long": {
            "count": len(longs),
            "min_tokens": min((n for _, n in longs), default=0),
            "max_tokens": max((n for _, n in longs), default=0),
            "targets": [targets[0], targets[-1]] if targets else [],
            "reached": sum(1 for (_, n), t in zip(longs, targets) if n >= t),
        },
        "shortest_tokens": min(lengths),
        "longest_tokens": longest,
        "prompt_limit_tokens": limit,
        "sha256": hashlib.sha256("".join(p + "\n" for p in prompts).encode("utf-8")).hexdigest(),
    }
    return prompts, summary


def exporter_kwargs(flags):
    """`--name=value` / `--name` / `--noname` exporter flags as keyword arguments."""
    out = {}
    for flag in flags:
        if not flag.startswith("--"):
            raise Refused("input", f"exporter flag {flag!r} is not --name[=value]")
        name, eq, value = flag[2:].partition("=")
        if eq:
            out[name] = value
        elif name.startswith("no") and name[2:] in ("externalize_embedder",):
            out[name[2:]] = False
        else:
            out[name] = True
    return out


def compiled_signatures(prefill_length):
    """The signatures the Qualcomm compile turns into one DISPATCH_OP each: the prefill/decode
    graph's two, and the aux graph's masks (the compiler's own selection, `prefill_mask|
    decode_mask`); the aux graph's cache-update signatures stay on the CPU. A phone's LiteRT
    log for a bundle of this path delegated exactly these, 1 of 1 node each."""
    return (
        [f"prefill_{prefill_length}", "decode"],
        [f"prefill_mask_{prefill_length}", "decode_mask"],
    )


def check_compiled(described, soc, signatures, *, only=True, stamp_bytes=STAMP_BYTES):
    """A compiled graph against what it must be: each of `signatures` exactly one DISPATCH_OP
    (nothing left on the CPU), no other signature when `only`, and a LiteRtStamp naming
    Qualcomm and `soc`. Raises Refused; returns the stamp."""
    ops = described["ops"]
    missing = [k for k in signatures if k not in ops]
    extra = sorted(set(ops) - set(signatures)) if only else []
    if missing or extra:
        raise Refused(
            "partitioned",
            f"the compiled graph's signatures are {sorted(ops)}, not {sorted(signatures)}",
        )
    for key in signatures:
        if ops[key] != {"DISPATCH_OP": 1}:
            raise Refused(
                "partitioned",
                f"signature {key} compiled to {ops[key]}, not one DISPATCH_OP: part of it would "
                "run on the CPU, or none of it on the NPU",
            )
    raw = described["stamp"]
    if raw is None or len(raw) != stamp_bytes:
        raise Refused(
            "stamp",
            f"the LiteRtStamp is {None if raw is None else len(raw)} bytes, not {stamp_bytes}",
        )
    half = stamp_bytes // 2
    stamp = (raw[:half].rstrip(b"\0").decode("ascii", "replace"),
             raw[half:].rstrip(b"\0").decode("ascii", "replace"))
    if stamp[0] != "Qualcomm" or stamp[1].upper() != soc.upper():
        raise Refused("stamp", f"the compiled graph is stamped {stamp}, not ('Qualcomm', {soc!r})")
    return list(stamp)


def describe(s, path):
    """Per signature the op counts of a .tflite, its LiteRtStamp bytes and its QAIRT build."""
    data = open(path, "rb").read()
    m = s.Model.GetRootAs(bytearray(data), 0)
    names = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        names.append(
            oc.CustomCode().decode() if oc.CustomCode()
            else str(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        )
    ops = {}
    for i in range(m.SignatureDefsLength()):
        d = m.SignatureDefs(i)
        sg = m.Subgraphs(d.SubgraphIndex())
        counts = {}
        for j in range(sg.OperatorsLength()):
            n = names[sg.Operators(j).OpcodeIndex()]
            counts[n] = counts.get(n, 0) + 1
        ops[d.SignatureKey().decode()] = counts
    stamp = None
    for i in range(m.MetadataLength()):
        md = m.Metadata(i)
        if md.Name().decode() == "LiteRtStamp":
            buf = m.Buffers(md.Buffer())
            stamp = buf.DataAsNumpy().tobytes() if buf.DataLength() else b""
    found = QAIRT.search(data)
    return {"ops": ops, "stamp": stamp, "qairt": found.group(1).decode() if found else None}


def check_compiled_aux(described, soc, prefill_length, before):
    """The compiled aux graph against stage 3's (`before`, its op counts per signature): the
    same signatures, the two masks one DISPATCH_OP each, every other signature (the cache
    updates) without one, left on the CPU as the compiler chose, and the SoC's stamp. Raises
    Refused; returns the stamp."""
    masks = compiled_signatures(prefill_length)[1]
    if sorted(described["ops"]) != sorted(before):
        raise Refused(
            "partitioned",
            f"the compiled aux graph's signatures are {sorted(described['ops'])}, stage 3's "
            f"{sorted(before)}",
        )
    stamp = check_compiled(described, soc, masks, only=False)
    for key, ops in described["ops"].items():
        if key not in masks and "DISPATCH_OP" in ops:
            raise Refused(
                "partitioned",
                f"aux signature {key} compiled to {ops}; only {masks} are compiled for the NPU",
            )
    return stamp


def decode_summary(record, cap):
    """What the calibration replies covered: per prompt, the decode steps it ran (each stops at
    a stop token or at `cap`) and the last position it reached."""
    steps = sorted(r["steps"] for r in record)
    if not steps:
        return None
    return {
        "count": len(steps),
        "cap": cap,
        "min_steps": steps[0],
        "median_steps": statistics.median(steps),
        "max_steps": steps[-1],
        "full": sum(1 for n in steps if n >= cap),
        "longest_position": max((r["position"] for r in record), default=0),
    }


def recording_decode(executor_cls, record):
    """Wrap the calibration executor so each `sample_text` call records its decode steps."""
    sample_text, decode_step = executor_cls.sample_text, executor_cls.decode_step

    def sample(self, *args, **kwargs):
        record.append({"steps": 0, "position": 0})
        return sample_text(self, *args, **kwargs)

    def step(self, *args, **kwargs):
        state = decode_step(self, *args, **kwargs)
        if record:
            record[-1]["steps"] += 1
            record[-1]["position"] = int(state.time_step)
        return state

    executor_cls.sample_text, executor_cls.decode_step = sample, step


METADATA_FILES = ("LlmMetadataProto", "LlmMetadata", "SP_Tokenizer", "HF_Tokenizer")
METADATA_VALUES = ("chat_template", "stop_token_ids")


def compare_metadata(unpacked, reference, sha):
    """The compiled bundle's metadata and tokenizer against stage 3's, by sha256 of each
    section file and equality of the template and stop tokens read from the metadata. Stage 3
    must carry a metadata section and a tokenizer. Raises Refused; returns what was compared."""
    if not any(reference.get(k) for k in METADATA_FILES[:2]):
        raise Refused("metadata", "the stage-3 bundle has no LlmMetadata section to compare")
    if not any(reference.get(k) for k in METADATA_FILES[2:]):
        raise Refused("metadata", "the stage-3 bundle has no tokenizer section to compare")
    compared = {}
    for key in METADATA_FILES:
        want = reference.get(key)
        if not want:
            continue
        got = unpacked.get(key)
        if not got or sha(got) != sha(want):
            raise Refused("metadata", f"the compiled bundle's {key} is not stage 3's")
        compared[key] = sha(got)
    for key in METADATA_VALUES:
        if unpacked.get(key) != reference.get(key):
            raise Refused("metadata", f"the compiled bundle's {key} is not stage 3's")
        value = reference.get(key)
        # The unpacker reads the stop tokens into a set; the answer is JSON.
        compared[key] = sorted(value) if isinstance(value, (set, frozenset)) else value
    return compared


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(spec):
    import copy

    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as s
    from transformers import AutoTokenizer
    from litert_torch.generative.export_hf.experimental.calib import quantize as qmod
    from litert_torch.generative.export_hf.experimental.calib import sampling_executor as sx
    from litert_torch.generative.export_hf.experimental.litertlm_bundle import litertlm_bundle as lb
    from litert_torch.generative.export_hf.experimental.npu_export import stages
    from litert_torch.generative.export_hf.experimental.npu_export.config_manager import (
        build_pipeline_config,
    )

    type_names = {v: k for k, v in vars(s.TensorType).items() if not k.startswith("_")}
    work = spec["work"]
    tok = AutoTokenizer.from_pretrained(spec["model"])
    if spec.get("chat_template"):
        # The template the bundle carries, so the calibration prompts are what it renders.
        tok.chat_template = open(spec["chat_template"], encoding="utf-8").read()
    render = make_render(
        tok.apply_chat_template, spec["prompt_mode"], parts=bool(spec.get("chat_template"))
    )

    def count_tokens(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rows = [json.loads(l) for l in open(spec["rows"], encoding="utf-8") if l.strip()]
    problem = render_problem(render)
    if problem:
        raise Refused("template", problem)
    prompts, calibration = calibration_plan(
        rows, render, count_tokens, spec["prompt_limit_tokens"], spec["short"], spec["long"],
        spec["prefill_length"],
    )
    calib_dir = os.path.join(work, "calib")
    os.makedirs(calib_dir)
    with open(os.path.join(calib_dir, "ALL.jsonl"), "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")

    def config(soc):
        return build_pipeline_config(
            model_id=spec["model"], target_vendor="qualcomm", target_soc=soc,
            prefill_lengths=[spec["prefill_length"]], cache_length=spec["cache_length"],
            max_calibration_decode_steps=spec["decode_steps"],
            weight_quantization_recipe="dynamic_wi8_afp32", calibration_dataset_format="jsonl",
            calibration_eval_task_names="ALL", use_float_input_output_normalizer=False,
            skip_mlir_passes=True, use_profiler_based_calibration=True,
            enable_min_max_calibration_update=True, calibration_range_scale=1.0,
        )

    def load(path):
        raw = open(path, "rb").read()
        if len(raw) > spec["max_repack_bytes"]:
            raise Refused(
                "size",
                f"the stage-3 graph is {len(raw):,} bytes; Python flatbuffers cannot repack "
                "2 GiB, and a checkpoint this size has not been measured here",
            )
        m = s.ModelT.InitFromPackedBuf(bytearray(raw), 0)
        for b in m.buffers:
            if (b.offset or 0) > 1:
                b.data = np.frombuffer(
                    raw, dtype=np.uint8, count=b.size, offset=b.offset
                ).copy()
                b.offset, b.size = 0, 0
        return raw, m

    dead = {}

    def packed_with_fix(original):
        def pack(**kw):
            path = kw["prefilldecode_model_path"]
            raw, m = load(path)
            dead["before"] = kv_type_counts(m, type_names)
            dead["changed"] = retype_dead_kv(m, type_names, copy.deepcopy)
            graphs = [m]
            if kw.get("auxiliary_model_path"):
                graphs.append(load(kw["auxiliary_model_path"])[1])
                dead["aux"] = kv_type_counts(graphs[1], type_names)
            problem = kv_quantization_problem(graphs, type_names)
            if problem:
                raise Refused("quantization", problem)
            if dead["changed"]:
                builder = flatbuffers.Builder(len(raw) + (1 << 20))
                builder.Finish(m.Pack(builder), file_identifier=b"TFL3")
                open(path, "wb").write(builder.Output())
            return original(**kw)
        return pack

    qmod.litertlm_utils.pack_litertlm = packed_with_fix(qmod.litertlm_utils.pack_litertlm)
    kwargs = exporter_kwargs(spec["flags"])
    cfg = config(spec["socs"][0])
    stage_s = {}

    def timed(name, fn):
        start = time.time()
        out = fn()
        stage_s[name] = round(time.time() - start, 1)
        return out

    decoded = []
    recording_decode(sx.Executor, decoded)
    stage1 = timed("stage1", lambda: stages.npu_export(cfg, os.path.join(work, "stage1"), **kwargs))
    calib_result = os.path.join(work, "calib_result")
    timed("stage2", lambda: stages.npu_calibrate(
        cfg, stage1, calib_dir, calib_result, **CALIBRATE_KWARGS))
    if len(decoded) != len(prompts):
        raise Refused(
            "script",
            f"calibration ran {len(decoded)} prompts of the {len(prompts)} written; its decode "
            "steps cannot be accounted for",
        )
    calibration["decode"] = decode_summary(decoded, spec["decode_steps"])
    stage3 = os.path.join(work, "stage3.litertlm")
    timed("stage3", lambda: stages.npu_quantize(cfg, stage1, stage3, calib_result))
    reference = lb.unpack_litertlm(stage3, os.path.join(work, "unpacked-stage3"))
    main_signatures, _ = compiled_signatures(spec["prefill_length"])
    aux_before = (
        describe(s, reference["tf_lite_aux"])["ops"] if reference.get("tf_lite_aux") else None
    )

    def read_compiled(soc, out):
        unpacked = lb.unpack_litertlm(out, os.path.join(work, "unpacked-" + soc))
        main_graph = unpacked["tf_lite_prefill_decode"]
        graph = s.ModelT.InitFromPackedBuf(bytearray(open(main_graph, "rb").read()), 0)
        described = describe(s, main_graph)
        stamp = check_compiled(described, soc, main_signatures)
        if not described["qairt"]:
            raise Refused("compiled", "the compiled graph names no QAIRT build")
        entry = {
            "packed": out,
            "kv_types": kv_type_counts(graph, type_names),
            "ops": described["ops"],
            "stamp": stamp,
            "qairt": described["qairt"],
            "metadata": compare_metadata(unpacked, reference, _sha),
            "aux": None,
        }
        if aux_before is not None:
            if not unpacked.get("tf_lite_aux"):
                raise Refused("compiled", "the compiled bundle has no aux graph; stage 3 had one")
            aux = describe(s, unpacked["tf_lite_aux"])
            check_compiled_aux(aux, soc, spec["prefill_length"], aux_before)
            entry["aux"] = {"ops": aux["ops"]}
        return entry

    socs = {}
    for soc in spec["socs"]:
        out = os.path.join(work, f"{soc}.litertlm")
        timed(f"compile {soc}", lambda: stages.npu_compile(
            config(soc), stage3, out, keep_temporary_files=False,
            intermediate_dir=os.path.join(work, "tmp-" + soc)))
        try:
            socs[soc] = read_compiled(soc, out)
        except Refused:
            raise
        except Exception as exc:
            raise Refused(
                "compiled",
                f"{soc}: the compiled bundle cannot be read: {type(exc).__name__}: {exc}",
            ) from exc
    versions = {}
    for name in VERSIONED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["qairt"] = sorted({entry["qairt"] for entry in socs.values()})
    return {"ok": True, "calibration": calibration, "socs": socs, "versions": versions,
            "stage_s": stage_s, "dead_kv": dead}


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    try:
        answer, code = main(spec), 0
    except Refused as exc:
        answer, code = {"ok": False, "cause": exc.cause, "reason": str(exc)}, 3
    except Exception as exc:
        answer, code = {"ok": False, "cause": "script",
                        "reason": f"{type(exc).__name__}: {exc}\n"
                                  f"{traceback.format_exc()[-1500:]}"}, 3
    with open(spec["answer"], "w", encoding="utf-8") as f:
        json.dump(answer, f)
    sys.exit(code)
'''
