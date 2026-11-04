"""`convert --npu <soc>`: the int8 graft, its text encoder compiled ahead of time for one
Qualcomm SoC.

Google's own NPU files of EmbeddingGemma 2 (`<name>_Qualcomm_<SoC>.litertlm` beside
`GOOGLE_FILE`) hold the text encoder as one `DISPATCH_OP` per signature over one QNN
context, its weights inside; the embedder stays a graph plus weights. The graft has
nothing to rewrite in such a section, so this compiles the grafted encoder instead and
keeps everything else of the graft (MEASUREMENTS.md, "EmbeddingGemma 2 on a Qualcomm NPU",
for every figure below):

  1. the graft's `tf_lite_text_encoder` graph with its external weights copied inline,
     and only the signatures Google's NPU files carry (`NPU_SIGNATURES`) with the
     decomposition subgraphs they reach;
  2. LiteRT's `apply_plugin_main` with the Qualcomm compiler plugin, the binary
     `ai_edge_litert.aot.aot_compile` drives, with weight sharing on -- without it each
     signature carries its own copy of the weights (587 MB instead of 170 MB for int8);
  3. what the compiled encoder must be, or nothing is written: every signature one
     `DISPATCH_OP`, all of them naming one bytecode range inside the section; a QNN
     context no larger than twice the encoder's weight bytes -- a size gate, evidence of
     weight sharing and not a proof of it: the int8 graft's context was 1.24x its weights
     compiled with `--qualcomm_enable_weight_sharing=true` and 4.29x without, and a single
     range does not tell the two apart; a 250-byte `LiteRtStamp` naming Qualcomm and the
     SoC, the length flutter_edge_ai's reader takes; every signature's inputs and outputs
     as the cut encoder's;
  4. a `.litertlm` of the graft's metadata, tokenizer and embedder, byte for byte, and the
     compiled encoder, in the section order of Google's Qualcomm files; read back.

**int8 only.** The int4 graft's 218 int4 FULLY_CONNECTED ops are not taken by the plugin
with float activations: each signature split into 194 partitions with the FCs on the CPU,
and that file's QNN context failed to load on an SM8750 phone (`Failed to create QNN
context: 1002`). The int8 encoder compiles whole.

**Linux x86-64 only.** The compiler plugin and `apply_plugin_main` ship in ai-edge-litert's
Linux x86-64 wheels only, and the QAIRT SDK they load is downloaded when
ai-edge-litert-sdk-qualcomm is built (`envs.NPU_QUALCOMM`).

**What a device does with the file is not checked here**, and `litetune verify` cannot
check it: LiteRT-LM's CPU and GPU cannot run a `DISPATCH_OP`. `DEVICE_RUNS` says what ran
where.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.events import EventStream
from litetune.exits import read_returncode

logger = logging.getLogger(__name__)

# The Qualcomm phone SoCs Google publishes EmbeddingGemma 2 NPU files for
# (embeddinggemma-2-text-270m_Qualcomm_<SoC>.litertlm, the graft's revision), each of
# them in ai-edge-litert 2.3.0's Qualcomm target list. Their Hexagon versions are the four
# flutter_edge_ai ships Stub/Skel libraries for, V73, V75, V79 and V81
# (flutter_edge_ai_litertlm npu_stacks.dart `qualcommNpuLibs`). The plugin's older phone
# targets (SM8350, SM8450, SM8475) and SM8845 have no Google file to compare against, and
# an app built on flutter_edge_ai would carry no Skel for the first three.
NPU_SOCS: tuple[str, ...] = ("SM8550", "SM8650", "SM8750", "SM8850")
# What has run on a device, per SoC: the only statement this module makes about one.
DEVICE_RUNS: dict[str, str] = {
    "SM8750": (
        "a file compiled this way for SM8750 ran on one Galaxy S25 Ultra's NPU on "
        "2026-10-31; its vectors were byte-identical to those of another compile of the "
        "same graft"
    ),
}
# The signatures of Google's Qualcomm files (embeddinggemma-2-text-270m_Qualcomm_SM8750 at
# the graft's revision): Google's CPU file has 2048 and 8192 too.
NPU_SIGNATURES: tuple[str, ...] = (
    "encoder_1x128",
    "encoder_1x256",
    "encoder_1x512",
    "encoder_1x1024",
)
NPU_WEIGHTS = "int8"
# 215-321 s per SoC were measured on a 32-vCPU Linux machine; this is an allowance
# for one compile, not a measurement. `convert --timeout-s` replaces it.
NPU_TIMEOUT_S = 3600
# Two NUL-padded fields of 125 bytes, SoC manufacturer and model (LiteRT
# litert/core/build_stamp.h; flutter_edge_ai's reader refuses another length).
STAMP_BYTES = 250
# The keys a successful answer must carry; `compile_for_npu` reads all of them before it
# moves anything.
ANSWER_KEYS: tuple[str, ...] = (
    "packed",
    "bytes",
    "sha256",
    "compile_s",
    "encoder_bytes",
    "encoder_sha256",
    "partitions",
    "ops_per_signature",
    "stamp",
    "qnn_context",
    "signatures",
    "qairt",
    "compiler",
)


class EmbeddingNpuError(Exception):
    """The NPU compile did not produce a file. `cause` is `run` for a failure of the run
    itself -- the environment, the compiler process, the answer -- and `script` for an
    error in this module's own script; neither says anything about the model. Any other
    cause names what the compiled file lacked or what the compiler refused."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


def host_problem(system: str | None = None, machine: str | None = None) -> str | None:
    """Why this host cannot compile for a Qualcomm NPU, or `None`."""
    system = system if system is not None else sys.platform
    machine = (machine if machine is not None else platform.machine()).lower()
    if system.startswith("linux") and machine in ("x86_64", "amd64"):
        return None
    return (
        f"--npu compiles on Linux x86-64 only (this is {system} {machine}): the Qualcomm "
        "compiler plugin and apply_plugin_main ship in ai-edge-litert's Linux x86-64 wheels only"
    )


def request_problem(socs: tuple[str, ...], weights: str) -> str | None:
    """Why `--npu` with these SoCs and weights is refused, or `None`."""
    unknown = [s for s in socs if s not in NPU_SOCS]
    if unknown:
        return f"--npu {', '.join(unknown)}: one of {', '.join(NPU_SOCS)}"
    if len(set(socs)) != len(socs):
        return f"--npu names a SoC twice: {', '.join(socs)}"
    if socs and weights != NPU_WEIGHTS:
        return (
            f"--npu needs --weights {NPU_WEIGHTS}, not {weights}: the int4 encoder's int4 "
            "FULLY_CONNECTED ops stay on the CPU (194 partitions per signature), and that file's "
            "QNN context did not load on an SM8750 phone"
        )
    return None


def dir_for(output_dir: Path, soc: str) -> Path:
    return output_dir / f"{NPU_WEIGHTS}-npu-{soc}"


def output_for(output_dir: Path, soc: str) -> Path:
    return dir_for(output_dir, soc) / "model.litertlm"


@dataclass(frozen=True)
class NpuResult:
    soc: str
    output: Path
    bytes: int
    sha256: str
    compile_s: float
    encoder_bytes: int
    encoder_sha256: str
    report: dict[str, Any] = field(default_factory=dict)
    # An image checkpoint's: the compiled encoder spliced into Google's text-vision file
    # for this SoC (`embedding_export`), which `output`, `bytes` and `sha256` then
    # describe. Holds `google_vision_npu_file`, `splice` and `compiled_file`, the
    # compile's own file, which is not kept.
    image: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.image is not None:
            # DEVICE_RUNS is about the text file; no text-vision file has run.
            return {
                **self._as_dict(),
                "device_run": None,
                "text_file_device_run": DEVICE_RUNS.get(self.soc),
                **self.image,
            }
        return self._as_dict()

    def _as_dict(self) -> dict[str, Any]:
        return {
            "soc": self.soc,
            "output": str(self.output),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "compile_s": self.compile_s,
            "encoder_bytes": self.encoder_bytes,
            "encoder_sha256": self.encoder_sha256,
            "signatures": self.report["signatures"],
            "ops_per_signature": self.report["ops_per_signature"],
            "partitions": self.report["partitions"],
            "stamp": self.report["stamp"],
            "qnn_context": self.report["qnn_context"],
            "qairt": self.report["qairt"],
            "compiler": self.report["compiler"],
            "device_run": DEVICE_RUNS.get(self.soc),
        }

    def moved_to(self, output: Path) -> NpuResult:
        return replace(self, output=output)


def _result(soc: str, output: Path, got: dict[str, Any]) -> NpuResult:
    """The answer read whole, or `EmbeddingNpuError("run")` naming what it lacked."""
    missing = [k for k in ANSWER_KEYS if k not in got]
    if missing:
        raise EmbeddingNpuError("run", f"the compile's answer has no {', '.join(missing)}")
    try:
        return NpuResult(
            soc=soc,
            output=output,
            bytes=int(got["bytes"]),
            sha256=str(got["sha256"]),
            compile_s=float(got["compile_s"]),
            encoder_bytes=int(got["encoder_bytes"]),
            encoder_sha256=str(got["encoder_sha256"]),
            report=got,
        )
    except (TypeError, ValueError) as exc:
        raise EmbeddingNpuError("run", f"the compile's answer could not be read: {exc}") from exc


def compile_for_npu(
    env: envs.StageEnv,
    grafted: Path,
    soc: str,
    output: Path,
    *,
    events: EventStream | None = None,
    auto_provision: bool = True,
    timeout: int = NPU_TIMEOUT_S,
) -> NpuResult:
    """Compile `grafted`'s text encoder for `soc` and write the NPU file to `output`.

    Its working directory is a temporary one beside `output`. Raises `EmbeddingNpuError`.
    """
    if soc not in NPU_SOCS:
        raise EmbeddingNpuError("input", f"{soc}: one of {', '.join(NPU_SOCS)}")
    if output.exists() or output.is_symlink():
        raise EmbeddingNpuError("input", f"{output} already exists; nothing is written over a file")
    if auto_provision:
        env.provision(events=events)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".litetune-npu-", dir=output.parent) as tmp:
        work = Path(tmp)
        script, spec, answer = work / "npu.py", work / "spec.json", work / "answer.json"
        script.write_text(_NPU_SCRIPT, encoding="utf-8")
        spec.write_text(
            json.dumps(
                {
                    "grafted": str(grafted),
                    "soc": soc,
                    "signatures": list(NPU_SIGNATURES),
                    "stamp_bytes": STAMP_BYTES,
                    "system_requirements": list(env.system_requirements_on("linux")),
                    "work": str(work),
                    "answer": str(answer),
                }
            ),
            encoding="utf-8",
        )
        try:
            proc = env.run(["python", str(script), str(spec)], timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise EmbeddingNpuError("run", f"the compile did not finish within {timeout}s") from exc
        except OSError as exc:
            raise EmbeddingNpuError(
                "run", f"the compile could not be started: {type(exc).__name__}: {exc}"
            ) from exc
        said = (proc.stderr or "").strip()[-400:] or "no stderr"
        if not answer.is_file():
            reading = read_returncode(proc.returncode)
            raise EmbeddingNpuError("run", f"the compile {reading.describe('the script')}: {said}")
        try:
            got = json.loads(answer.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EmbeddingNpuError(
                "run", f"the compile's answer could not be read: {exc}"
            ) from exc
        if not isinstance(got, dict):
            raise EmbeddingNpuError("run", f"the compile's answer is not an object: {got!r:.200}")
        if not got.get("ok"):
            raise EmbeddingNpuError(str(got.get("cause", "run")), str(got.get("reason", got)))
        if proc.returncode != 0:
            # An answer the script wrote and a process that then died: killed after the
            # write, or a crash in teardown. Not trusted.
            reading = read_returncode(proc.returncode)
            raise EmbeddingNpuError(
                "run", f"the compile answered ok but {reading.describe('the script')}: {said}"
            )
        result = _result(soc, output, got)
        packed = Path(str(got["packed"]))
        if not packed.is_file():
            raise EmbeddingNpuError(
                "run", f"the compile's answer names {packed}, which is not there"
            )
        packed.replace(output)
    return result


# The script that runs inside envs.NPU_QUALCOMM. tests/test_embedding_npu.py runs it with
# stand-ins for the modules it imports and calls its decision functions -- `cut_model`,
# `sdk_problem`, `apply_plugin`, `check_compiled`, `answer_for` -- for real.
_NPU_SCRIPT = r'''
"""Compile an int8-grafted EmbeddingGemma 2 file's text encoder for one Qualcomm SoC.

Reads JSON {grafted, soc, signatures, stamp_bytes, system_requirements, work, answer}
from argv[1]. Writes to spec["answer"]:
  {"ok": true, "packed": <path>, "bytes", "sha256", ...}  (embedding_npu.ANSWER_KEYS)
  or {"ok": false, "cause": <str>, "reason": <str>}
and exits 0 for the first, 3 for the second.
"""

import collections
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

import flatbuffers
import numpy as np
from flatbuffers import flexbuffers

import ai_edge_litert_sdk_qualcomm
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.aot.core import common
from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder import litertlm_header_schema_py_generated as hs

STATS = re.compile(
    r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops. "
    r"resulted in (\d+) partitions."
)
QAIRT = re.compile(rb"v(2\.\d+\.\d+\.\d+)")
QNN_LIBS = ("libQnnHtp.so", "libQnnSystem.so")
SHARED_CONTEXT_LIMIT = 2


class Refused(Exception):
    def __init__(self, cause, reason):
        super().__init__(reason)
        self.cause = cause


def sections(data):
    end = int.from_bytes(data[24:32], "little")
    objs = hs.LiteRTLMMetaData.GetRootAs(data[32:end], 0).SectionMetadata()
    out = []
    for i in range(objs.ObjectsLength()):
        o = objs.Objects(i)
        items = {}
        for j in range(o.ItemsLength()):
            kv = o.Items(j)
            if kv.ValueType() == hs.VData.StringValue:
                v = hs.StringValue()
                v.Init(kv.Value().Bytes, kv.Value().Pos)
                items[kv.Key().decode()] = v.Value().decode()
            else:
                items[kv.Key().decode()] = None
        out.append(
            {"type": o.DataType(), "items": items, "begin": o.BeginOffset(), "end": o.EndOffset()}
        )
    return out


def one(secs, dtype, model_type=None):
    found = [
        x
        for x in secs
        if x["type"] == dtype and (model_type is None or x["items"].get("model_type") == model_type)
    ]
    if len(found) != 1:
        raise Refused(
            "input",
            f"the graft has {len(found)} sections of type {dtype} {model_type or ''}, not one",
        )
    return found[0]


def composites(sg):
    for op in sg.operators or []:
        if op.builtinOptions2Type == s.BuiltinOptions2.StableHLOCompositeOptions:
            yield op.builtinOptions2


# Ops whose options name subgraphs by index; `cut_model` renumbers composites only.
CONTROL_FLOW = ("WhileOptions", "IfOptions", "CallOnceOptions")


def cut_model(m, weights, keep):
    """`m` (a ModelT) with its external weights inline and only `keep`'s signatures, plus
    the composite decomposition subgraphs they reach, renumbered. Changes `m`."""
    sigs = {
        (d.signatureKey.decode() if isinstance(d.signatureKey, bytes) else d.signatureKey): d
        for d in m.signatureDefs
    }
    missing = [k for k in keep if k not in sigs]
    if missing:
        raise Refused("input", f"the encoder has no signature {missing} (it has {sorted(sigs)})")
    control = {getattr(s.BuiltinOptions, name) for name in CONTROL_FLOW}
    order = [sigs[k].subgraphIndex for k in keep]
    seen = set(order)
    i = 0
    while i < len(order):
        sg = m.subgraphs[order[i]]
        for op in sg.operators or []:
            if op.builtinOptionsType in control:
                raise Refused(
                    "input", f"subgraph {order[i]} has a control-flow op this cut cannot renumber"
                )
        for opt in composites(sg):
            if opt.decompositionSubgraphIndex not in seen:
                seen.add(opt.decompositionSubgraphIndex)
                order.append(opt.decompositionSubgraphIndex)
        i += 1
    index = {old: new for new, old in enumerate(order)}
    ext = {e.id: e for e in (m.externalBuffers or [])}
    groups = [
        (g.name.decode() if isinstance(g.name, bytes) else (g.name or ""))
        for g in (m.externalBufferGroups or [])
    ]
    inline = {}
    for old in order:
        for t in m.subgraphs[old].tensors or []:
            if not t.externalBuffer:
                continue
            e = ext.get(t.externalBuffer)
            if e is None or groups[e.group] != "tflite_weights":
                raise Refused(
                    "input", f"tensor {t.name!r} names an external buffer outside tflite_weights"
                )
            if e.offset + e.length > len(weights):
                raise Refused(
                    "input", f"tensor {t.name!r}'s external range ends past the weights section"
                )
            key = (e.offset, e.length)
            if key not in inline:
                b = s.BufferT()
                b.data = np.frombuffer(
                    weights, dtype=np.uint8, count=e.length, offset=e.offset
                ).copy()
                m.buffers.append(b)
                inline[key] = len(m.buffers) - 1
            t.buffer = inline[key]
            t.externalBuffer = 0
    m.externalBuffers = None
    m.externalBufferGroups = None
    m.subgraphs = [m.subgraphs[old] for old in order]
    for sg in m.subgraphs:
        for opt in composites(sg):
            opt.decompositionSubgraphIndex = index[opt.decompositionSubgraphIndex]
    m.signatureDefs = [sigs[k] for k in keep]
    for d in m.signatureDefs:
        d.subgraphIndex = index[d.subgraphIndex]
    return m


def cut_encoder(model_bytes, weights, keep):
    m = cut_model(s.ModelT.InitFromPackedBuf(bytearray(model_bytes), 0), weights, keep)
    b = flatbuffers.Builder(1024 * 1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def describe(path):
    """What `check_compiled` reads from a .tflite: per signature its ops and its inputs and
    outputs (name, shape, dtype); every custom op's options; the LiteRtStamp bytes."""
    data = open(path, "rb").read()
    m = s.Model.GetRootAs(bytearray(data), 0)
    names = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        names.append(
            oc.CustomCode().decode()
            if oc.CustomCode()
            else str(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        )
    ops, io, dispatch = {}, {}, []
    for i in range(m.SignatureDefsLength()):
        d = m.SignatureDefs(i)
        key = d.SignatureKey().decode()
        sg = m.Subgraphs(d.SubgraphIndex())
        ops[key] = dict(
            collections.Counter(
                names[sg.Operators(j).OpcodeIndex()] for j in range(sg.OperatorsLength())
            )
        )

        def spec(t):
            tensor = sg.Tensors(t.TensorIndex())
            shape = tensor.ShapeAsNumpy().tolist() if tensor.ShapeLength() else []
            return [t.Name().decode(), shape, int(tensor.Type())]

        io[key] = {
            "inputs": sorted(spec(d.Inputs(j)) for j in range(d.InputsLength())),
            "outputs": sorted(spec(d.Outputs(j)) for j in range(d.OutputsLength())),
        }
        for j in range(sg.OperatorsLength()):
            op = sg.Operators(j)
            if names[op.OpcodeIndex()] == "DISPATCH_OP":
                raw = op.CustomOptionsAsNumpy().tobytes() if op.CustomOptionsLength() else b""
                opts = flexbuffers.Loads(raw) if raw else {}
                dispatch.append(
                    {
                        "signature": key,
                        **{k: opts.get(k) for k in ("name", "bytecode_offset", "bytecode_size")},
                    }
                )
    stamp = None
    for i in range(m.MetadataLength()):
        md = m.Metadata(i)
        if md.Name().decode() == "LiteRtStamp":
            buf = m.Buffers(md.Buffer())
            stamp = buf.DataAsNumpy().tobytes() if buf.DataLength() else b""
    return {
        "ops": ops,
        "io": io,
        "dispatch": dispatch,
        "stamp": stamp,
        "bytes": len(data),
        "qairt": (QAIRT.search(data).group(1).decode() if QAIRT.search(data) else None),
    }


def read_compiled(path):
    """`describe` of the compiler's output; bytes it cannot read are the compiler's
    failure (`compiled_file`), not this script's."""
    try:
        return describe(path)
    except Exception as exc:  # noqa: BLE001 -- the compiler's output, not this script
        raise Refused(
            "compiled_file", f"the compiled encoder could not be read: {type(exc).__name__}: {exc}"
        ) from None


def check_compiled(
    compiled, cut_io, signatures, soc, stamp_bytes, stats, system_requirements, weight_bytes
):
    """The compiled encoder against what it must be; raises `Refused` naming the first
    thing it is not. Returns the facts the report records."""
    if len(stats) < len(signatures):
        needs = " and ".join(system_requirements) or "its system libraries"
        raise Refused(
            "run",
            f"apply_plugin_main reported partitioning for {len(stats)} of {len(signatures)} "
            f"signatures: the Qualcomm plugin did not run. It needs {needs}, and says only "
            "that it loaded no plugin when one is missing",
        )
    whole = {k for k, v in compiled["ops"].items() if v == {"DISPATCH_OP": 1}}
    if whole != set(signatures) or set(compiled["ops"]) != set(signatures):
        raise Refused(
            "partitioned", f"not every signature compiled to one DISPATCH_OP: {compiled['ops']}"
        )
    ranges = {(d["bytecode_offset"], d["bytecode_size"]) for d in compiled["dispatch"]}
    functions = [d["name"] for d in compiled["dispatch"]]
    if len(ranges) != 1 or None in next(iter(ranges), (None,)):
        raise Refused(
            "dispatch",
            f"the DISPATCH_OPs name {len(ranges)} bytecode ranges, not one: {compiled['dispatch']}",
        )
    ((offset, size),) = ranges
    if len(set(functions)) != len(functions) or None in functions:
        raise Refused(
            "dispatch", f"the DISPATCH_OPs do not each name their own function: {functions}"
        )
    if not (0 < offset and offset + size <= compiled["bytes"]):
        raise Refused(
            "dispatch",
            f"the bytecode range {offset}+{size} is not inside the "
            f"{compiled['bytes']}-byte section",
        )
    # A size gate, not a proof of sharing. One range is not enough: a compile without
    # weight sharing also leaves one context for the four functions, holding four copies
    # of the weights. Measured on the int8 graft (136,708,096 weight bytes): 169,586,688
    # (1.24x) compiled with --qualcomm_enable_weight_sharing=true, 586,875,872 (4.29x)
    # without.
    if size > SHARED_CONTEXT_LIMIT * weight_bytes:
        raise Refused(
            "context_size",
            f"the QNN context is {size:,} bytes for {weight_bytes:,} bytes of encoder weights, "
            f"more than {SHARED_CONTEXT_LIMIT}x (1.24x was measured with weight sharing, 4.29x "
            "without): the weights look unshared",
        )
    raw = compiled["stamp"]
    if raw is None or len(raw) != stamp_bytes:
        raise Refused(
            "stamp",
            f"the LiteRtStamp is {None if raw is None else len(raw)} bytes, not {stamp_bytes}",
        )
    half = stamp_bytes // 2
    try:
        stamp = (raw[:half].rstrip(b"\0").decode("ascii"), raw[half:].rstrip(b"\0").decode("ascii"))
    except UnicodeDecodeError as exc:
        raise Refused("stamp", f"the LiteRtStamp is not ASCII: {exc}") from None
    if stamp != ("Qualcomm", soc):
        raise Refused(
            "stamp", f"the compiled encoder is stamped for {stamp}, not ('Qualcomm', {soc!r})"
        )
    for key in signatures:
        if compiled["io"].get(key) != cut_io.get(key):
            raise Refused(
                "signature",
                f"{key}'s inputs and outputs changed in the compile: "
                f"{cut_io.get(key)} -> {compiled['io'].get(key)}",
            )
    return {
        "stamp": list(stamp),
        "ops_per_signature": compiled["ops"],
        "qnn_context": {
            "bytecode_offset": offset,
            "bytecode_size": size,
            "functions": functions,
            "encoder_weight_bytes": weight_bytes,
            "to_weights": round(size / weight_bytes, 3),
            "gate": f"at most {SHARED_CONTEXT_LIMIT}x the encoder's weight bytes",
            "measured": "1.24x with --qualcomm_enable_weight_sharing=true, 4.29x without",
        },
        "signatures": {k: compiled["io"][k] for k in signatures},
        "qairt": compiled["qairt"],
    }


def sdk_problem(libs):
    """Why the QAIRT libraries the compiler loads are not usable at `libs`, or None."""
    if libs is None:
        return "ai-edge-litert-sdk-qualcomm has no SDK directory: its QAIRT download did not happen"
    missing = [n for n in QNN_LIBS if not (pathlib.Path(libs) / n).is_file()]
    if missing:
        return f"{', '.join(missing)} not in {libs}: the SDK's QAIRT download did not complete"
    return None


def apply_plugin(model, out, soc, log, libs):
    binary = str(common.get_resource(pathlib.Path("tools/apply_plugin_main")))
    plugin = os.path.dirname(
        str(
            common.get_resource(
                pathlib.Path("vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so")
            )
        )
    )
    cmd = [
        binary,
        "--cmd=apply",
        f"--model={model}",
        f"--o={out}",
        "--soc_manufacturer=Qualcomm",
        f"--soc_model={soc}",
        "--err=--",
        f"--libs={plugin}",
        "--qualcomm_enable_weight_sharing=true",
    ]
    env = os.environ.copy()
    lib = common.construct_ld_library_path()
    env["LD_LIBRARY_PATH"] = str(libs) + (os.pathsep + lib if lib else "")
    t = time.time()
    r = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pathlib.Path(log).write_text(r.stdout or "")
    tail = (r.stdout or "").strip()[-400:]
    if r.returncode < 0:
        raise Refused("run", f"apply_plugin_main was ended by signal {-r.returncode}: {tail}")
    if r.returncode != 0 or not os.path.exists(out):
        raise Refused("compile", f"apply_plugin_main exited {r.returncode}: {tail}")
    return time.time() - t, [list(map(int, x)) for x in STATS.findall(r.stdout or "")]


def answer_for(spec, run):
    """`run(spec)`'s answer, or the refusal; an exception escaping it is cause `script`,
    with its type and the last lines of its traceback."""
    try:
        return run(spec)
    except Refused as exc:
        return {"ok": False, "cause": exc.cause, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001 -- recorded in the answer, not swallowed
        tail = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
        return {"ok": False, "cause": "script", "reason": f"{type(exc).__name__}: {exc}\n{tail}"}


def main(spec):
    work = pathlib.Path(spec["work"])
    try:
        libs = ai_edge_litert_sdk_qualcomm.path_to_sdk_libs()
    except Exception as exc:  # noqa: BLE001 -- the SDK raises on another host
        raise Refused("run", f"ai-edge-litert-sdk-qualcomm: {type(exc).__name__}: {exc}") from None
    problem = sdk_problem(libs)
    if problem:
        raise Refused("run", problem)
    data = open(spec["grafted"], "rb").read()
    T = hs.AnySectionDataType
    secs = sections(data)
    meta, tok = one(secs, T.EmbeddingMetadataProto), one(secs, T.SP_Tokenizer)
    emb = one(secs, T.TFLiteModel, "tf_lite_embedder")
    embw = one(secs, T.TFLiteWeights, "tf_lite_embedder")
    enc = one(secs, T.TFLiteModel, "tf_lite_text_encoder")
    encw = one(secs, T.TFLiteWeights, "tf_lite_text_encoder")

    def blob(x):
        return data[x["begin"] : x["end"]]

    cut = work / "encoder.tflite"
    cut.write_bytes(cut_encoder(blob(enc), blob(encw), spec["signatures"]))
    compiled = work / "encoder_npu.tflite"
    seconds, stats = apply_plugin(
        str(cut), str(compiled), spec["soc"], str(work / "compile.log"), libs
    )
    facts = check_compiled(
        read_compiled(str(compiled)),
        describe(str(cut))["io"],
        spec["signatures"],
        spec["soc"],
        spec["stamp_bytes"],
        stats,
        spec["system_requirements"],
        len(blob(encw)),
    )
    parts = {}
    for name, x in (
        ("meta.pb", meta),
        ("tok.spm", tok),
        ("emb.tflite", emb),
        ("emb.weights", embw),
    ):
        parts[name] = work / name
        parts[name].write_bytes(blob(x))
    b = lb.LitertLmFileBuilder()
    b.add_embedding_metadata(str(parts["meta.pb"]))
    b.add_sentencepiece_tokenizer(str(parts["tok.spm"]))
    b.add_tflite_model(str(parts["emb.tflite"]), lb.TfLiteModelType.EMBEDDER)
    b.add_tflite_model(str(compiled), lb.TfLiteModelType.TEXT_ENCODER)
    b.add_tflite_weights(str(parts["emb.weights"]), lb.TfLiteModelType.EMBEDDER)
    packed = work / "packed.litertlm"
    with open(packed, "wb") as f:
        b.build(f)
    back = open(packed, "rb").read()
    encoder = compiled.read_bytes()
    want = [blob(meta), blob(tok), blob(emb), encoder, blob(embw)]
    got = [back[x["begin"] : x["end"]] for x in sections(back)]
    if got != want:
        raise Refused("readback", "the packed file does not read back section by section")
    return {
        "ok": True,
        "packed": str(packed),
        "bytes": len(back),
        "sha256": hashlib.sha256(back).hexdigest(),
        "compile_s": round(seconds, 1),
        "encoder_bytes": len(encoder),
        "encoder_sha256": hashlib.sha256(encoder).hexdigest(),
        "partitions": stats,
        **facts,
        "compiler": {
            d: importlib.metadata.version(d)
            for d in ("ai-edge-litert", "ai-edge-litert-sdk-qualcomm", "litert-lm-builder")
        },
    }


if __name__ == "__main__":
    spec = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    answer = answer_for(spec, main)
    pathlib.Path(spec["answer"]).write_text(json.dumps(answer), encoding="utf-8")
    sys.exit(0 if answer["ok"] else 3)
'''
