"""`convert --npu <soc>` for an LLM checkpoint: a generation bundle compiled ahead of time for
one Qualcomm SoC, through litert-torch's `npu_export` stages.

What runs (MEASUREMENTS.md, "LLMs on a Qualcomm NPU", for every figure below):

  1. `npu_export` stage 1: the checkpoint exported with the split KV cache the NPU executor
     needs, `prefill_lengths` [128], `cache_length` from `--cache-length`, the flags this
     family's plain `convert` would pass (chat template, model type);
  2. stage 2: static-range calibration on prompts built from the training data
     (`calibration_plan`): half of them the data's own prompts, half long prompts packed from
     its examples up to the longest prompt the bundle is to serve;
  3. stage 3: A16W8 static quantization, then `retype_dead_kv` before the bundle is packed;
  4. stage 4: the Qualcomm compile, once per SoC; every KV-cache input of the compiled graphs
     must be INT16 or nothing is written.

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
  same models answered them. The boundary sat at the longest calibration prompt. So the
  calibration set reaches `max_prompt_tokens`, and the report says that longer prompts are
  outside what this bundle was calibrated for.

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

import hashlib
import json
import logging
import os
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
# Tokens of the cache kept free for the reply: the longest prompt the bundle is calibrated
# for is `cache_length - REPLY_TOKENS`.
REPLY_TOKENS = 128
# The smallest cache that leaves one prefill chunk for a prompt besides the reply.
MIN_CACHE_LENGTH = PREFILL_LENGTH + REPLY_TOKENS
# 64 calibration prompts, as the packaged configs' calibration runs measured here used: half
# the data's own prompts, half long ones packed from its examples.
CALIBRATION_SHORT = 32
CALIBRATION_LONG = 32
# Fewer training rows than this cannot make a calibration set worth the name.
MIN_CALIBRATION_ROWS = 8
CALIBRATION_DECODE_STEPS = 32
# Measured stage times on a 32-vCPU machine: stages 1-3 up to ~19 min with long
# calibration prompts, ~2 min per SoC compile. An allowance, not a measurement;
# `convert --timeout-s` replaces it.
NPU_LLM_TIMEOUT_S = 7200

# The families this path has run on SM8750, and what ran. A family not here is refused.
FAMILIES: dict[str, str] = {
    "gemma-3-text": "gemma-3-270m-it (base weights): two-, five- and seven-chunk needle prompts",
    "functiongemma": (
        "the Gemma 3 270M export path with FunctionGemma's model type and template; "
        "functiongemma-270m-it base weights ran on SM8750 on an earlier toolchain, not this one"
    ),
    "qwen-3": "Qwen3-0.6B (base weights): two-, five- and seven-chunk needle prompts",
    "qwen-2.5": (
        "Qwen2.5-0.5B-Instruct: base weights on two-chunk needle prompts, and a banking77 "
        "LoRA fine-tune on held-out rows"
    ),
    "minicpm5": "MiniCPM5-1B (base weights, thinking off): two-chunk needle prompts",
}
# Families that are known not to work here, and why.
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
}
# What has run on a device, per SoC: the only statement this module makes about one.
DEVICE_RUNS: dict[str, str] = {
    "SM8750": (
        "bundles built this way ran on one Galaxy S25 Ultra's NPU on 2026-11-01 (Gemma 3 "
        "270M, Qwen3-0.6B, Qwen2.5-0.5B, MiniCPM5-1B and a Qwen2.5-0.5B fine-tune)"
    ),
}
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

NPU_ENV_CHECK = "npu environment"
NPU_REQUEST_CHECK = "npu request"
NPU_COMPILE_CHECK = "npu compile"
NPU_PUBLISH_CHECK = "npu publish"
NOT_VERIFIED = (
    "the NPU bundle is not verified: litetune verify runs LiteRT-LM's CPU or GPU, which "
    "cannot run its DISPATCH_OPs. Measure it on a phone with that SoC"
)


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


def max_prompt_tokens(cache_length: int) -> int:
    """The longest prompt a bundle of this cache length is calibrated for."""
    return cache_length - REPLY_TOKENS


def family_problem(family: str | None, model: str) -> str | None:
    """Why this family is refused on `--npu`, or `None`."""
    if family in FAMILIES:
        return None
    measured = ", ".join(sorted(FAMILIES))
    if family in BLOCKED:
        return f"--npu: {model} is {family}, which does not run this way: {BLOCKED[family]}"
    named = f"family {family}" if family else "no family litetune knows"
    return (
        f"--npu: {model} is {named}; this path has run on an NPU for {measured} only, and an "
        "unmeasured family is refused rather than compiled on trust"
    )


def request_problem(
    socs: tuple[str, ...],
    cache_length: int,
    *,
    family: str | None,
    model: str,
    decision: bool,
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
        return f"--npu: {model} records a decision; {DECISION_REFUSAL}"
    if cache_length < MIN_CACHE_LENGTH:
        return (
            f"--cache-length {cache_length}: at least {MIN_CACHE_LENGTH}, one "
            f"{PREFILL_LENGTH}-token "
            f"prefill chunk for the prompt and {REPLY_TOKENS} tokens for the reply"
        )
    if cache_length % PREFILL_LENGTH:
        return (
            f"--cache-length {cache_length}: a multiple of the prefill length {PREFILL_LENGTH}, "
            "as every measured bundle's was"
        )
    return family_problem(family, model)


def read_calibration_rows(path: Path) -> list[dict[str, str]]:
    """The prepared data's rows, `prompt` and `completion` each a string. Raises `LlmNpuError`."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise LlmNpuError("input", f"--calibration-data {path}: {exc}") from exc
    rows: list[dict[str, str]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise LlmNpuError("input", f"--calibration-data {path}:{number}: {exc}") from exc
        if not (
            isinstance(row, dict)
            and isinstance(row.get("prompt"), str)
            and isinstance(row.get("completion"), str)
        ):
            raise LlmNpuError(
                "input",
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
    kv_types: dict[str, dict[str, int]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "soc": self.soc,
            "output": str(self.output),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "kv_input_types": self.kv_types,
            "device_run": DEVICE_RUNS.get(self.soc),
        }


@dataclass
class LlmNpuResult:
    request: LlmNpuRequest
    checks: CheckSet = field(default_factory=lambda: CheckSet(name="convert:llm-npu"))
    family: str | None = None
    prompt_mode: str | None = None
    exporter_flags: list[str] = field(default_factory=list)
    chat_template: dict[str, Any] | None = None
    answer: dict[str, Any] = field(default_factory=dict)
    bundles: list[LlmNpuBundle] = field(default_factory=list)
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
            "family_measured": FAMILIES.get(self.family or ""),
            "prompt_mode": self.prompt_mode,
            "cache_length": self.request.cache_length,
            "prefill_length": PREFILL_LENGTH,
            "max_prompt_tokens": calibration.get("longest_tokens") if calibration else None,
            "calibration": calibration,
            "exporter_flags": list(self.exporter_flags),
            "chat_template": self.chat_template,
            "dead_kv": self.answer.get("dead_kv"),
            "stage_s": self.answer.get("stage_s"),
            "versions": self.answer.get("versions"),
            "npu": [b.as_dict() for b in self.bundles],
            "request": self.request.as_dict(),
            "checks": self.checks.as_dict(),
            "limitations": list(self.limitations),
        }


def _add(result: LlmNpuResult, events: EventStream, check: Check) -> Check:
    result.checks.add(check)
    events.check(check)
    return check


def _refuse(result: LlmNpuResult, events: EventStream, name: str, detail: str) -> LlmNpuResult:
    _add(result, events, Check.failed(name, detail))
    return result


def run_llm_npu_export(
    request: LlmNpuRequest, events: EventStream | None = None, *, host: str | None = None
) -> LlmNpuResult:
    """`convert --npu` for an LLM checkpoint. Every failure is a check in the result."""
    # Imported here: export imports nothing of this module, and the CLI imports both.
    from litetune.export import ExportRequest, compose_chat_template

    events = events or EventStream(echo_json=False)
    result = LlmNpuResult(request=request)
    problem = (host if host is not None else host_problem()) or request.env.refusal()
    if problem:
        return _refuse(result, events, NPU_REQUEST_CHECK, problem)
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
        return _refuse(result, events, NPU_REQUEST_CHECK, str(exc))
    if export_request.identity_conflict:
        return _refuse(result, events, NPU_REQUEST_CHECK, export_request.identity_conflict)
    rules = export_request.plan.rules
    result.family = rules.family if rules else None
    problem = request_problem(
        request.socs,
        request.cache_length,
        family=result.family,
        model=request.model,
        decision=export_request.decision_record() is not None,
    )
    if problem:
        return _refuse(result, events, NPU_REQUEST_CHECK, problem)
    try:
        record = models.read_training_record(request.model)
    except (OSError, ValueError) as exc:
        return _refuse(result, events, NPU_REQUEST_CHECK, f"{request.model}: {exc}")
    mode = record[0].get("prompt_mode") if record else None
    if mode not in ("runtime_rendered", "prerendered"):
        return _refuse(
            result,
            events,
            NPU_REQUEST_CHECK,
            f"--npu: {request.model} records no prompt mode (its litetune.json, which `tune` "
            "writes); the calibration prompts are rendered by it",
        )
    result.prompt_mode = mode
    try:
        rows = read_calibration_rows(request.calibration_data)
    except LlmNpuError as exc:
        return _refuse(result, events, NPU_REQUEST_CHECK, str(exc))
    for soc in request.socs:
        if dir_for(request.output_dir, soc).exists():
            return _refuse(
                result,
                events,
                NPU_REQUEST_CHECK,
                f"{dir_for(request.output_dir, soc)} already exists; nothing is written over it",
            )
    _add(result, events, Check.passed(NPU_REQUEST_CHECK, f"{result.family}, {len(rows)} rows"))

    env = request.env
    try:
        if request.auto_provision:
            env.provision(events=events)
    except Exception as exc:  # noqa: BLE001 -- a provisioning failure is a check, not a crash
        logger.exception("provisioning %s failed", env.name)
        _add(result, events, Check.unchecked(NPU_ENV_CHECK, f"{env.name}: {exc}"))
        result.limitations.append("no NPU bundle was compiled: the environment is not ready")
        return result
    if not env.ready:
        _add(
            result,
            events,
            Check.unchecked(NPU_ENV_CHECK, f"environment {env.name!r} {env.unready_detail}"),
        )
        return result
    _add(result, events, Check.passed(NPU_ENV_CHECK, f"{env.name} ({env.identity}) ready"))

    template: Path | None = None
    if export_request.composes_chat_template:
        check, composed = compose_chat_template(export_request)
        _add(result, events, check)
        if composed is None:
            result.limitations.append("no NPU bundle was compiled: no chat template")
            return result
        template = composed.path
        result.chat_template = composed.as_dict()
    argv = export_request.argv(STAGE1_RECIPE, template)
    # argv: litert-torch export_hf --model --output_dir --quantization_recipe, then the flags.
    result.exporter_flags = list(argv[5:])

    request.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".litetune-npu-llm-", dir=request.output_dir) as tmp:
        work = Path(tmp)
        try:
            answer = _compile(request, result, rows, mode, work)
        except LlmNpuError as exc:
            _add(result, events, Check.failed(NPU_COMPILE_CHECK, f"{exc.cause}: {exc}"))
            result.limitations.append("no NPU bundle was compiled")
            return result
        result.answer = answer
        calibration = answer["calibration"]
        _add(
            result,
            events,
            Check.passed(
                NPU_COMPILE_CHECK,
                f"{len(request.socs)} SoC(s); calibrated on {calibration['count']} prompts of "
                f"{calibration['shortest_tokens']}-{calibration['longest_tokens']} tokens",
                observed={"dead_kv": answer["dead_kv"]},
            ),
        )
        _publish(request, result, events, work)
    result.limitations.append(NOT_VERIFIED)
    result.limitations.append(
        f"prompts longer than {result.answer['calibration']['longest_tokens']} tokens (rendered) "
        "are outside what this bundle was calibrated for: measured on SM8750, prompts past the "
        "longest calibration prompt turned into degenerate repetition. Do not serve them"
    )
    return result


def _compile(
    request: LlmNpuRequest,
    result: LlmNpuResult,
    rows: list[dict[str, str]],
    mode: str,
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
                "socs": list(request.socs),
                "cache_length": request.cache_length,
                "prefill_length": PREFILL_LENGTH,
                "max_prompt_tokens": max_prompt_tokens(request.cache_length),
                "short": CALIBRATION_SHORT,
                "long": CALIBRATION_LONG,
                "decode_steps": CALIBRATION_DECODE_STEPS,
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
        packed = Path(str(entry.get("packed")))
        if not packed.is_file():
            raise LlmNpuError("run", f"the export's answer names {packed} for {soc}, not there")
        problem = kv_types_problem(entry.get("kv_types"))
        if problem:
            raise LlmNpuError("compiled", f"{soc}: {problem}")
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


def _publish(request: LlmNpuRequest, result: LlmNpuResult, events: EventStream, work: Path) -> None:
    """Each SoC's bundle published in turn, as `embedding_export._publish_npus` does: its
    report written beside the staged file, the SoC's directory created exclusively, the
    bundle and then the report hard-linked in. The first failure stops the rest; nothing is
    rolled back, and the report says what reached the disk."""
    published: list[LlmNpuBundle] = []
    try:
        for soc in request.socs:
            entry = result.answer["socs"][soc]
            staged = Path(str(entry["packed"]))
            target = dir_for(request.output_dir, soc)
            sha = hash_file(staged).removeprefix("sha256:")
            bundle = LlmNpuBundle(
                soc=soc,
                output=target / ARTIFACT_NAME,
                bytes=staged.stat().st_size,
                sha256=sha,
                kv_types=entry["kv_types"],
            )
            staged_report = work / f"{soc}.{REPORT_NAME}"
            report = result.as_dict()
            report["npu"] = bundle.as_dict()
            _write_report(staged_report, report)
            os.mkdir(target)
            os.link(staged, target / ARTIFACT_NAME)
            staged.unlink()
            os.link(staged_report, target / REPORT_NAME)
            staged_report.unlink()
            published.append(bundle)
            events.artifact(str(bundle.output), name=f"model.litertlm (npu {soc})", verified=False)
    except BaseException as exc:
        logger.exception("publishing the NPU bundles stopped")
        result.publish_error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, OSError):
            raise
    finally:
        result.bundles = published
    if result.publish_error is None:
        _add(result, events, Check.passed(NPU_PUBLISH_CHECK, f"{len(published)} published"))
        return
    detail = f"published {len(published)} of {len(request.socs)}; stopped on {result.publish_error}"
    _add(result, events, Check.unchecked(NPU_PUBLISH_CHECK, f"run: {detail}"))
    result.limitations.append(f"{detail}. Run the rest with another --output-dir")


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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# The script that runs inside envs.NPU_LLM. Its top level imports the standard library only,
# so tests/test_llm_npu.py runs `retype_dead_kv`, `kv_type_counts`, `long_targets`,
# `calibration_plan` and `exporter_kwargs` for real without the toolchain.
_NPU_LLM_SCRIPT = r'''
"""npu_export stages 1-4 for one LLM checkpoint, with calibration from its training data and
the dead last-layer KV inputs retyped. Reads JSON spec from argv[1]; writes spec["answer"]:
  {"ok": true, "calibration", "socs": {soc: {packed, kv_types}}, "versions", "stage_s", "dead_kv"}
  or {"ok": false, "cause", "reason"}; exits 0 for the first and 3 for the second.
"""

import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
import traceback

KV_PREFIX = "kv_cache_"
VERSIONED = (
    "litert-torch-nightly",
    "ai-edge-litert-nightly",
    "ai-edge-quantizer-nightly",
    "ai-edge-litert-sdk-qualcomm",
    "litert-lm-builder",
    "transformers",
    "torch",
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


def retype_dead_kv(model, type_names, copy_quantization):
    """Every prefill KV-cache input that no op of its subgraph reads takes the type and
    quantization of the same-named input of the decode signature. Changes `model`; returns
    what changed. The inputs are unused, so no computation changes."""
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
        read = {int(i) for op in sg.operators or [] for i in op.inputs if int(i) >= 0}
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


def long_targets(max_prompt, count):
    """`count` target lengths spread evenly from half of `max_prompt` up to `max_prompt`
    itself, so the longest calibration prompt reaches the longest prompt served."""
    if count <= 0:
        return []
    low = max(1, max_prompt // 2)
    if count == 1:
        return [max_prompt]
    return [low + (max_prompt - low) * k // (count - 1) for k in range(count)]


def calibration_plan(rows, render, count_tokens, max_prompt, short, long, seed=0):
    """The calibration prompts: `short` of the data's own prompts, rendered, and `long`
    prompts packed from its examples (prompt and completion, one after another) followed by
    one more prompt, each grown to its target from `long_targets`. Refuses when a data prompt
    is longer than `max_prompt`. Returns (prompts, summary)."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    rendered = []
    for i in order:
        text = render(rows[i]["prompt"])
        n = count_tokens(text)
        if n > max_prompt:
            raise Refused(
                "data",
                f"a training prompt renders to {n} tokens, more than the {max_prompt} this "
                "cache length leaves for a prompt; raise --cache-length",
            )
        rendered.append((text, n))
    shorts = rendered[:short]
    longs = []
    cursor = 0
    for target in long_targets(max_prompt, long):
        pieces, best = [], None
        for _ in range(len(rows) * 4):
            row = rows[order[cursor % len(order)]]
            cursor += 1
            candidate = pieces + [row["prompt"] + " " + row["completion"]]
            tail = rows[order[cursor % len(order)]]["prompt"]
            text = render("\n\n".join(candidate + [tail]))
            n = count_tokens(text)
            if n > max_prompt:
                break
            pieces, best = candidate, (text, n)
            if n >= target:
                break
        if best is None:
            break
        longs.append(best)
    prompts = [t for t, _ in shorts + longs]
    lengths = [n for _, n in shorts + longs]
    summary = {
        "count": len(prompts),
        "short": {"count": len(shorts), "min_tokens": min((n for _, n in shorts), default=0),
                  "max_tokens": max((n for _, n in shorts), default=0)},
        "long": {"count": len(longs), "min_tokens": min((n for _, n in longs), default=0),
                 "max_tokens": max((n for _, n in longs), default=0),
                 "targets": [long_targets(max_prompt, long)[0], max_prompt] if long else []},
        "shortest_tokens": min(lengths),
        "longest_tokens": max(lengths),
        "max_prompt_tokens": max_prompt,
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


def main(spec):
    import copy

    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as s
    from transformers import AutoTokenizer
    from litert_torch.generative.export_hf.experimental.calib import quantize as qmod
    from litert_torch.generative.export_hf.experimental.litertlm_bundle import litertlm_bundle as lb
    from litert_torch.generative.export_hf.experimental.npu_export import stages
    from litert_torch.generative.export_hf.experimental.npu_export.config_manager import (
        build_pipeline_config,
    )

    type_names = {v: k for k, v in vars(s.TensorType).items() if not k.startswith("_")}
    work = spec["work"]
    tok = AutoTokenizer.from_pretrained(spec["model"])
    if spec["prompt_mode"] == "runtime_rendered":
        def render(text):
            return tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
            )
    else:
        def render(text):
            return text

    def count_tokens(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rows = [json.loads(l) for l in open(spec["rows"], encoding="utf-8") if l.strip()]
    prompts, calibration = calibration_plan(
        rows, render, count_tokens, spec["max_prompt_tokens"], spec["short"], spec["long"]
    )
    if calibration["longest_tokens"] < spec["max_prompt_tokens"] // 2:
        raise Refused(
            "data",
            f"the longest calibration prompt is {calibration['longest_tokens']} tokens, under half "
            f"of the {spec['max_prompt_tokens']} the cache leaves for a prompt",
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

    dead = {}

    def packed_with_fix(original):
        def pack(**kw):
            path = kw["prefilldecode_model_path"]
            raw = open(path, "rb").read()
            m = s.ModelT.InitFromPackedBuf(bytearray(raw), 0)
            for b in m.buffers:
                if (b.offset or 0) > 1:
                    b.data = np.frombuffer(
                        raw, dtype=np.uint8, count=b.size, offset=b.offset
                    ).copy()
                    b.offset, b.size = 0, 0
            dead["before"] = kv_type_counts(m, type_names)
            dead["changed"] = retype_dead_kv(m, type_names, copy.deepcopy)
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

    stage1 = timed("stage1", lambda: stages.npu_export(cfg, os.path.join(work, "stage1"), **kwargs))
    calib_result = os.path.join(work, "calib_result")
    timed("stage2", lambda: stages.npu_calibrate(cfg, stage1, calib_dir, calib_result))
    stage3 = os.path.join(work, "stage3.litertlm")
    timed("stage3", lambda: stages.npu_quantize(cfg, stage1, stage3, calib_result))
    socs = {}
    for soc in spec["socs"]:
        out = os.path.join(work, f"{soc}.litertlm")
        timed(f"compile {soc}", lambda: stages.npu_compile(
            config(soc), stage3, out, keep_temporary_files=False,
            intermediate_dir=os.path.join(work, "tmp-" + soc)))
        unpacked = lb.unpack_litertlm(out, os.path.join(work, "unpacked-" + soc))
        raw = open(unpacked["tf_lite_prefill_decode"], "rb").read()
        graph = s.ModelT.InitFromPackedBuf(bytearray(raw), 0)
        socs[soc] = {"packed": out, "kv_types": kv_type_counts(graph, type_names)}
    versions = {}
    for name in VERSIONED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
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
