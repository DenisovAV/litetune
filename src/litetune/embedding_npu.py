"""`convert --npu <soc>`: the int8 graft, its text encoder compiled ahead of time for one
Qualcomm SoC.

Google's own NPU files of EmbeddingGemma 2 (`<name>_Qualcomm_<SoC>.litertlm` beside
`GOOGLE_FILE`) hold the text encoder as one `DISPATCH_OP` per signature over one QNN
context, its weights inside; the embedder stays a graph plus weights. The graft has
nothing to rewrite in such a section, so this compiles the grafted encoder instead and
keeps everything else of the graft. Every figure below is from one measurement,
2026-10-31, not yet written into MEASUREMENTS.md (banking77 fine-tune, SM8750 and SM8650):

  1. the graft's `tf_lite_text_encoder` graph with its external weights copied inline,
     and only the signatures Google's NPU files carry (`NPU_SIGNATURES`) with the
     decomposition subgraphs they reach;
  2. LiteRT's `apply_plugin_main` with the Qualcomm compiler plugin, the binary
     `ai_edge_litert.aot.aot_compile` drives, with weight sharing on -- without it each
     signature carries its own copy of the weights (587 MB instead of 170 MB for int8);
  3. every signature must have become exactly one `DISPATCH_OP`, stamped for the SoC;
  4. a `.litertlm` of the graft's metadata, tokenizer and embedder, byte for byte, and the
     compiled encoder, in the section order of Google's Qualcomm files; read back.

**int8 only.** The int4 graft's 218 int4 FULLY_CONNECTED ops are not taken by the plugin
with float activations: each signature split into 194 partitions with the FCs on the CPU,
and that file's QNN context failed to load on an SM8750 phone (`Failed to create QNN
context: 1002`). The int8 encoder compiles whole.

**Linux x86-64 only.** ai-edge-litert-sdk-qualcomm 2.3.0 is a source distribution that
downloads QAIRT 2.50.0.260828 when it is built, and the compiler plugin and
`apply_plugin_main` ship in ai-edge-litert's Linux x86-64 wheels only.

What the file embeds on a device is not checked here: an SM8750 phone was measured once
(`MEASURED_ON_DEVICE`); other SoCs compile but have not run.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.events import EventStream
from litetune.exits import read_returncode

logger = logging.getLogger(__name__)

# The phone SoCs ai-edge-litert 2.3.0's Qualcomm target lists
# (ai_edge_litert/aot/vendors/qualcomm/target.py `SocModel`), automotive parts left out.
NPU_SOCS: tuple[str, ...] = ("SM8450", "SM8475", "SM8550", "SM8650", "SM8750", "SM8845", "SM8850")
# The SoCs a compiled file of this path has run on (a Galaxy S25 Ultra, 2026-10-31).
MEASURED_ON_DEVICE: tuple[str, ...] = ("SM8750",)
# The signatures of Google's Qualcomm files (embeddinggemma-2-text-270m_Qualcomm_SM8750 at
# the graft's revision): Google's CPU file has 2048 and 8192 too.
NPU_SIGNATURES: tuple[str, ...] = (
    "encoder_1x128",
    "encoder_1x256",
    "encoder_1x512",
    "encoder_1x1024",
)
NPU_WEIGHTS = "int8"
# 214-321 s per SoC were measured on a 32-vCPU Cloud Build machine; this is an allowance.
NPU_TIMEOUT_S = 3600


class EmbeddingNpuError(Exception):
    """The NPU compile did not produce a file. `cause` is `run` for a failure of the run
    itself (it says nothing about the model) or what the compiled file lacked."""

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


def output_for(output_dir: Path, soc: str) -> Path:
    return output_dir / f"{NPU_WEIGHTS}-npu-{soc}" / "model.litertlm"


@dataclass(frozen=True)
class NpuResult:
    soc: str
    output: Path
    bytes: int
    sha256: str
    compile_s: float
    encoder_bytes: int
    report: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "soc": self.soc,
            "output": str(self.output),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "compile_s": self.compile_s,
            "encoder_bytes": self.encoder_bytes,
            "measured_on_device": self.soc in MEASURED_ON_DEVICE,
            "compiler": self.report.get("compiler", {}),
        }


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

    Raises `EmbeddingNpuError`.
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
        if not answer.is_file():
            reading = read_returncode(proc.returncode)
            said = (proc.stderr or "").strip()[-400:] or "no stderr"
            raise EmbeddingNpuError("run", f"the compile {reading.describe('the script')}: {said}")
        try:
            got = json.loads(answer.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise EmbeddingNpuError(
                "run", f"the compile's answer could not be read: {exc}"
            ) from exc
        if not got.get("ok"):
            raise EmbeddingNpuError(str(got.get("cause", "run")), str(got.get("reason", got)))
        Path(got["packed"]).replace(output)
    return NpuResult(
        soc=soc,
        output=output,
        bytes=int(got["bytes"]),
        sha256=str(got["sha256"]),
        compile_s=float(got["compile_s"]),
        encoder_bytes=int(got["encoder_bytes"]),
        report=got,
    )


# The script that runs inside envs.NPU_QUALCOMM. Its helpers take plain arguments and
# are exercised in the tests only where they need no ai_edge_litert.
_NPU_SCRIPT = r'''
"""Compile an int8-grafted EmbeddingGemma 2 file's text encoder for one Qualcomm SoC.

Reads JSON {grafted, soc, signatures, work, answer} from argv[1]. Writes to spec["answer"]:
  {"ok": true, "packed": <path>, "bytes", "sha256", "compile_s", "encoder_bytes", "compiler"}
  or {"ok": false, "cause": <str>, "reason": <str>}
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

import flatbuffers
import numpy as np
import ai_edge_litert_sdk_qualcomm
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.aot.core import common
from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder import litertlm_header_schema_py_generated as hs

STATS = re.compile(
    r"Partitioned subgraph<(\d+)>, selected (\d+) ops, from a total of (\d+) ops. "
    r"resulted in (\d+) partitions."
)


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
            f"the graft has {len(found)} sections of type {dtype} {model_type or ''}, " "not one",
        )
    return found[0]


def composites(sg):
    for op in sg.operators or []:
        if op.builtinOptions2Type == s.BuiltinOptions2.StableHLOCompositeOptions:
            yield op.builtinOptions2


def cut_encoder(model_bytes, weights, keep):
    """The encoder graph with its external weights inline and only `keep`'s signatures."""
    m = s.ModelT.InitFromPackedBuf(bytearray(model_bytes), 0)
    sigs = {
        (d.signatureKey.decode() if isinstance(d.signatureKey, bytes) else d.signatureKey): d
        for d in m.signatureDefs
    }
    missing = [k for k in keep if k not in sigs]
    if missing:
        raise Refused("input", f"the encoder has no signature {missing} (it has {sorted(sigs)})")
    order = [sigs[k].subgraphIndex for k in keep]
    seen = set(order)
    i = 0
    while i < len(order):
        for opt in composites(m.subgraphs[order[i]]):
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
                    "input", f"tensor {t.name!r} names an external buffer " "outside tflite_weights"
                )
            if e.offset + e.length > len(weights):
                raise Refused(
                    "input", f"tensor {t.name!r}'s external range ends past " "the weights section"
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
    b = flatbuffers.Builder(1024 * 1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def compiled_shape(path):
    """Per signature, its op names; and the LiteRtStamp (manufacturer, model)."""
    m = s.Model.GetRootAs(bytearray(open(path, "rb").read()), 0)
    names = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        names.append(
            oc.CustomCode().decode()
            if oc.CustomCode()
            else str(max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode()))
        )
    ops = {}
    for i in range(m.SignatureDefsLength()):
        d = m.SignatureDefs(i)
        sg = m.Subgraphs(d.SubgraphIndex())
        ops[d.SignatureKey().decode()] = dict(
            collections.Counter(
                names[sg.Operators(j).OpcodeIndex()] for j in range(sg.OperatorsLength())
            )
        )
    stamp = None
    for i in range(m.MetadataLength()):
        md = m.Metadata(i)
        if md.Name().decode() == "LiteRtStamp":
            raw = m.Buffers(md.Buffer()).DataAsNumpy().tobytes()
            stamp = (raw[:125].rstrip(b"\0").decode(), raw[125:250].rstrip(b"\0").decode())
    return ops, stamp


def apply_plugin(model, out, soc, log):
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
    env["LD_LIBRARY_PATH"] = str(ai_edge_litert_sdk_qualcomm.path_to_sdk_libs()) + (
        os.pathsep + lib if lib else ""
    )
    t = time.time()
    r = subprocess.run(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pathlib.Path(log).write_text(r.stdout)
    if r.returncode != 0 or not os.path.exists(out):
        tail = r.stdout.strip()[-400:]
        raise Refused("compile", f"apply_plugin_main exited {r.returncode}: {tail}")
    return time.time() - t, [list(map(int, x)) for x in STATS.findall(r.stdout)]


def main(spec):
    work = pathlib.Path(spec["work"])
    data = open(spec["grafted"], "rb").read()
    T = hs.AnySectionDataType
    secs = sections(data)
    meta, tok = one(secs, T.EmbeddingMetadataProto), one(secs, T.SP_Tokenizer)
    emb, embw = (
        one(secs, T.TFLiteModel, "tf_lite_embedder"),
        one(secs, T.TFLiteWeights, "tf_lite_embedder"),
    )
    enc, encw = (
        one(secs, T.TFLiteModel, "tf_lite_text_encoder"),
        one(secs, T.TFLiteWeights, "tf_lite_text_encoder"),
    )
    blob = lambda x: data[x["begin"] : x["end"]]
    cut = work / "encoder.tflite"
    cut.write_bytes(cut_encoder(blob(enc), blob(encw), spec["signatures"]))
    compiled = work / "encoder_npu.tflite"
    seconds, stats = apply_plugin(str(cut), str(compiled), spec["soc"], str(work / "compile.log"))
    ops, stamp = compiled_shape(str(compiled))
    whole = {k: v for k, v in ops.items() if v == {"DISPATCH_OP": 1}}
    if set(whole) != set(spec["signatures"]):
        raise Refused("partitioned", f"not every signature compiled to one DISPATCH_OP: {ops}")
    if stamp != ("Qualcomm", spec["soc"]):
        raise Refused(
            "stamp",
            f"the compiled encoder is stamped {stamp}, " f"not ('Qualcomm', {spec['soc']!r})",
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
    want = [blob(meta), blob(tok), blob(emb), compiled.read_bytes(), blob(embw)]
    got = [back[x["begin"] : x["end"]] for x in sections(back)]
    if got != want:
        raise Refused("readback", "the packed file does not read back section by section")
    return {
        "ok": True,
        "packed": str(packed),
        "bytes": len(back),
        "sha256": hashlib.sha256(back).hexdigest(),
        "compile_s": round(seconds, 1),
        "encoder_bytes": compiled.stat().st_size,
        "partitions": stats,
        "compiler": {
            d: importlib.metadata.version(d)
            for d in ("ai-edge-litert", "ai-edge-litert-sdk-qualcomm", "litert-lm-builder")
        },
    }


if __name__ == "__main__":
    spec = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        answer = main(spec)
    except Refused as exc:
        answer = {"ok": False, "cause": exc.cause, "reason": str(exc)}
    pathlib.Path(spec["answer"]).write_text(json.dumps(answer), encoding="utf-8")
    sys.exit(0 if answer["ok"] else 3)
'''
