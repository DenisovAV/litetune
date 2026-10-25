"""`embedding_convert`: a fine-tuned EmbeddingGemma 2 text tower grafted into
Google's `.litertlm`.

The script under test is the real one, run by this interpreter, against
test_towers.py's fake litert-lm-builder (its bundle format, `unpack` and `pack`),
because the script packs through the towers script's `rebuild`. The model is
made up and tiny -- two layers, 6x16 weights (96 bytes at int8, so an offset
left unaligned shows), norms of 4 and 8 values -- but laid out the way the graft
spike's README describes Google's file: int4 weights in a weights section,
per-row float32 scales in the graph, one scale vector per subgraph (two here),
norm constants read by STABLEHLO_COMPOSITE ops after the FULLY_CONNECTED of
their sub-block, and a 1/512 folded into `per_layer_model_projection`.

The graph flatbuffer is written here by hand, and every position the script
patches is recorded as it is written. A graft patches the graph in place, so the
packed graph is read at those positions -- not through the script's own reader.

numpy is needed to run the script and to check its numbers; the dev environment
does not carry it, so without it these tests are skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned
from test_towers import FAKE_BUILDER, FAKE_FORMAT, FAKE_PEEK

from litetune import embedding_convert, envs
from litetune.embedding_convert import (
    _GRAFT_SCRIPT,
    EmbeddingConvertError,
    GraftResult,
    graft,
)
from litetune.towers import BUILDER_TYPES

np: Any = pytest.importorskip("numpy")

FORMAT: dict[str, Any] = {}
exec(FAKE_FORMAT, FORMAT)

# The script's functions, for the tests that call one directly.
SCRIPT: dict[str, Any] = {"__name__": "embedding_graft"}
exec(compile(_GRAFT_SCRIPT, "embedding_graft.py", "exec"), SCRIPT)

INT4, INT8, FLOAT32 = 17, 9, 0
P = "language_model."
LAYERS, HIDDEN, HEAD, ROWS, COLS, VOCAB = 2, 8, 4, 6, 16, 16
FOLD = 2.0**-9
# Graph sub-block, HF parameter.
LINEAR = [
    ("attn/q_einsum", "self_attn.q_proj"),
    ("attn/k_einsum", "self_attn.k_proj"),
    ("attn/v_einsum", "self_attn.v_proj"),
    ("attn/attn_vec_einsum", "self_attn.o_proj"),
    ("mlp/gating_einsum1", "mlp.gate_proj"),
    ("mlp/gating_einsum2", "mlp.up_proj"),
    ("mlp/linear", "mlp.down_proj"),
    ("ple/per_layer_embedding_gate", "ple_block.per_layer_input_gate"),
    ("ple/per_layer_embedding_projection", "ple_block.per_layer_projection"),
]
PMP = P + "ple.per_layer_model_projection.weight"
HEADS = P + "embedding_projection.weight"
EMBED = P + "embed_tokens.weight"

BUNDLE_TOML = """[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "The ODML Authors" },
  { key = "uuid", value_type = "String", value = "00000000-0000-4000-8000-000000000000" },
  { key = "creation_timestamp", value_type = "String", value = "2026-01-01T00:00:00+00:00" },
]

[[section]]
section_type = "LlmMetadata"
data_path = "LlmMetadataProto.pbtext"

[[section]]
section_type = "SP_Tokenizer"
data_path = "Section1_SP_Tokenizer.spiece"

[[section]]
model_type = "embedder"
section_type = "TFLiteModel"
data_path = "Section2_TFLiteModel_tf_lite_embedder.tflite"

[[section]]
model_type = "embedder"
section_type = "TFLiteWeights"
data_path = "Section3_TFLiteWeights_tf_lite_embedder.weight"

[[section]]
model_type = "text_encoder"
section_type = "TFLiteModel"
data_path = "Section4_TFLiteModel_tf_lite_text_encoder.tflite"

[[section]]
model_type = "text_encoder"
section_type = "TFLiteWeights"
data_path = "Section5_TFLiteWeights_tf_lite_text_encoder.weight"
"""


# -- a flatbuffer, written by hand, every patched position recorded ---------------

SCALARS = {"i8": "<b", "u32": "<I", "i32": "<i", "u64": "<Q"}
VECTORS = {"ints": "<i4", "floats": "<f4", "bytes": "u1"}


def flatbuffer(root: tuple) -> tuple[bytes, dict[str, int]]:
    """A flatbuffer and the positions of what is labelled in it. A table is
    ("table", [field or None, ...]) in schema order, each field in an 8-byte slot;
    a field is a scalar (kind, value[, label]) of SCALARS, labelled by its
    position, or a node. Nodes: ("tables", [...]), ("string", text), and
    (kind, values[, label]) of VECTORS, labelled by where their data starts."""
    out = bytearray(4) + b"TFL3"
    marks: dict[str, int] = {}

    def align() -> None:
        out.extend(bytes(-len(out) % 8))

    def emit(node: tuple) -> int:
        kind, value, *label = node
        align()
        start = len(out)
        if kind == "table":
            out.extend(struct.pack("<HH", 4 + 2 * len(value), 4 + 8 * len(value)))
            for i, item in enumerate(value):
                out.extend(struct.pack("<H", 0 if item is None else 4 + 8 * i))
            align()
            table = len(out)
            out.extend(struct.pack("<i", table - start))
            later = []
            for item in value:
                slot = len(out)
                out.extend(bytes(8))
                if item is None:
                    continue
                if item[0] in SCALARS:
                    struct.pack_into(SCALARS[item[0]], out, slot, item[1])
                    if len(item) > 2:
                        marks[item[2]] = slot
                else:
                    later.append((slot, item))
            for slot, item in later:
                struct.pack_into("<I", out, slot, emit(item) - slot)
            return table
        if kind == "tables":
            out.extend(struct.pack("<I", len(value)) + bytes(4 * len(value)))
            for i, child in enumerate(value):
                slot = start + 4 + 4 * i
                struct.pack_into("<I", out, slot, emit(child) - slot)
            return start
        if kind == "string":
            data = value.encode()
            out.extend(struct.pack("<I", len(data)) + data + b"\0")
            return start
        data = np.asarray(value, VECTORS[kind]).tobytes()
        count = len(data) // np.dtype(VECTORS[kind]).itemsize
        out.extend(struct.pack("<I", count))
        if label:
            marks[label[0]] = len(out)
        out.extend(data)
        return start

    struct.pack_into("<I", out, 0, emit(root))
    return bytes(out), marks


def tensor(name: str, shape: list[int], *, type_: int | None = None, buffer: int = 0,
           scales: Any = None, ext: int = 0, label: str = "") -> tuple:  # fmt: skip
    """A Tensor table: shape, type, buffer, name, quantization, then
    external_buffer as the eleventh field."""
    quant = (
        None if scales is None else ("table", [None, None, ("floats", scales, label + ":scale")])
    )
    fields: list[Any] = [
        ("ints", shape),
        None if type_ is None else ("i8", type_, label + ":type"),
        ("u32", buffer) if buffer else None,
        ("string", name),
        quant,
    ]
    fields += [None] * 5 + [("u32", ext) if ext else None]
    return ("table", fields)


def model(subgraphs: list[tuple], buffers: list[tuple], ext: list[tuple],
          opcodes: Sequence[tuple] = ()) -> tuple:  # fmt: skip
    """A Model table: operator_codes second, subgraphs third, buffers fifth,
    external_buffers tenth."""
    fields: list[Any] = [None] * 10
    fields[1] = ("tables", list(opcodes)) if opcodes else None
    fields[2] = ("tables", subgraphs)
    fields[4] = ("tables", buffers) if buffers else None
    fields[9] = ("tables", ext)
    return ("table", fields)


def external(ext_id: int, offset: int, length: int, label: str, *, with_length: bool = True):
    return (
        "table",
        [
            ("u32", ext_id),
            ("u32", 0),
            ("u64", offset, label + ":offset"),
            ("u64", length, label + ":length") if with_length else None,
        ],
    )


def subgraph(tensors: list[tuple], ops: Sequence[tuple] = ()) -> tuple:  # fmt: skip
    return ("table", [("tables", tensors), None, None, ("tables", list(ops)) if ops else None])


def operator(opcode: int, inputs: list[int], outputs: list[int]) -> tuple:
    return ("table", [("u32", opcode), ("ints", inputs), ("ints", outputs)])


FC, COMPOSITE = 0, 1
OPCODES = [
    ("table", [("i8", 9), None, None, ("i32", 9)]),  # FULLY_CONNECTED
    ("table", [("i8", 127), None, None, ("i32", 206)]),  # STABLEHLO_COMPOSITE
]


# -- the made-up model ------------------------------------------------------------


def bf16(x: Any) -> Any:
    """x, cut to what bfloat16 holds, as float32."""
    return (np.asarray(x, np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)


def google_int4(x: Any) -> tuple[Any, Any]:
    """Per-row absmax int4 codes and scales, standing in for Google's."""
    scales = (np.abs(x).max(axis=1) / 7).astype(np.float32)
    return np.clip(np.round(x / scales[:, None]), -8, 7).astype(np.int8), scales


def int4_bytes(codes: Any) -> bytes:
    u = (codes.astype(np.int16).reshape(-1) & 15).astype(np.uint8)
    return (u[0::2] | (u[1::2] << 4)).tobytes()


def int4_codes(raw: bytes, shape: list[int]) -> Any:
    a = np.frombuffer(raw, np.uint8)
    q = np.stack([a & 15, a >> 4], 1).reshape(-1).astype(np.int8)
    return np.where(q > 7, q - 16, q).reshape(shape)


def norm_names(layer: int) -> dict[str, str]:
    return {
        "q": f"{P}layers.{layer}.self_attn.q_norm.weight",
        "k": f"{P}layers.{layer}.self_attn.k_norm.weight",
        "post_attn": f"{P}layers.{layer}.post_attention_layernorm.weight",
        "pre_ff": f"{P}layers.{layer}.pre_feedforward_layernorm.weight",
        "post_ff": f"{P}layers.{layer}.post_feedforward_layernorm.weight",
        "ple": f"{P}layers.{layer}.ple_block.post_per_layer_input_norm.weight",
        "input": f"{P}layers.{layer}.input_layernorm.weight",
    }


def base_checkpoint(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    t: dict[str, Any] = {EMBED: rng.normal(size=(VOCAB, HIDDEN))}
    for layer in range(LAYERS):
        for _, hf in LINEAR:
            t[f"{P}layers.{layer}.{hf}.weight"] = rng.normal(size=(ROWS, COLS))
        for kind, name in norm_names(layer).items():
            t[name] = rng.normal(1.0, 0.5, size=HEAD if kind in ("q", "k") else HIDDEN)
        t[f"{P}layers.{layer}.layer_scalar"] = rng.normal(size=1)
    t[PMP] = rng.normal(size=(ROWS, COLS))
    t[HEADS] = rng.normal(size=(ROWS, COLS))
    t[P + "norm.weight"] = rng.normal(1.0, 0.5, size=HIDDEN)
    t[P + "ple.per_layer_projection_norm.weight"] = rng.normal(1.0, 0.5, size=HEAD)
    return {k: bf16(v) for k, v in t.items()}


def fine_tune(base: dict[str, Any], seed: int = 1) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        k: v if k.endswith("layer_scalar") else (v + rng.normal(0, 0.3, v.shape)).astype(np.float32)
        for k, v in base.items()
    }


def safetensors(path: Path, tensors: dict[str, Any], dtype: str = "F32") -> Path:
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for name, value in tensors.items():
        v = np.asarray(value, np.float32)
        raw = (v.view(np.uint32) >> 16).astype("<u2").tobytes() if dtype == "BF16" else v.tobytes()
        header[name] = {"dtype": dtype, "shape": list(v.shape),
                        "data_offsets": [len(data), len(data) + len(raw)]}  # fmt: skip
        data += raw
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + bytes(data))
    return path


@dataclass
class Google:
    """Google's file, made up: its sections and where everything patched is."""

    graphs: dict[str, bytes]
    weights: dict[str, bytes]
    marks: dict[str, dict[str, int]]
    # Graph tensor name -> (HF name, shape, external offset, length).
    tensors: dict[str, dict[str, tuple[str, list[int], int, int]]]
    # Norm constant name -> its value in Google's graph.
    constants: dict[str, Any]


def google_file(base: dict[str, Any], *, rename: dict[str, str] | None = None,
                scale_pmp: float = FOLD, drop_composite: str | None = None,
                without_length: str | None = None) -> Google:  # fmt: skip
    """Both graphs and their weights. `rename` gives a graph tensor another
    name; `scale_pmp` is the constant folded into per_layer_model_projection;
    `drop_composite` leaves the norm of that name unread by any composite;
    `without_length` leaves that tensor's external buffer without a length."""
    rng = np.random.default_rng(7)
    graphs, weights, marks, tensors = {}, {}, {}, {}

    def weights_of(entries: list[tuple[str, str, Any]]) -> tuple[bytes, list]:
        """The weights section, each tensor at a 64-byte boundary after a gap."""
        blob, placed = bytearray(), []
        for name, hf, value in entries:
            codes, scales = google_int4(value)
            data = int4_bytes(codes)
            offset = -(-(len(blob) + 32) // 64) * 64
            blob += b"\xaa" * (offset - len(blob)) + data
            placed.append((name, hf, value.shape, offset, len(data), scales))
        return bytes(blob), placed

    # The embedder: one table, one subgraph.
    blob, placed = weights_of([("embedder/table", EMBED, base[EMBED])])
    name, hf, shape, offset, length, scales = placed[0]
    graphs["embedder"], marks["embedder"] = flatbuffer(
        model(
            [
                subgraph(
                    [tensor(name, list(shape), type_=INT4, scales=scales, ext=1, label=f"0:{name}")]
                )
            ],  # fmt: skip
            [("table", [])],
            [external(1, offset, length, name)],
        )
    )
    weights["embedder"] = blob
    tensors["embedder"] = {name: (hf, list(shape), offset, length)}

    # The encoder's weights, in graph order.
    entries = [("text_encoder/per_layer_model_projection/w", PMP, base[PMP] * scale_pmp)]
    for layer in range(LAYERS):
        for block, hf in LINEAR:
            name = f"text_encoder/layer_{layer}/{block}/w;fused"
            entries.append((name, f"{P}layers.{layer}.{hf}.weight",
                            base[f"{P}layers.{layer}.{hf}.weight"]))  # fmt: skip
    entries.append(("text_encoder/heads_layer/default_projection/w", HEADS, base[HEADS]))
    for old, new in (rename or {}).items():
        entries = [(new if n == old else n, h, v) for n, h, v in entries]
    blob, placed = weights_of(entries)
    weights["text_encoder"] = blob
    tensors["text_encoder"] = {n: (h, list(s), o, ln) for n, h, s, o, ln, _ in placed}

    # Subgraph 0: the weights, the norm constants, activations, and the ops.
    sub0: list[tuple] = []
    buffers: list[tuple] = [("table", [])]
    constants: dict[str, Any] = {}
    index: dict[str, int] = {}
    for i, (name, _, shape, _, _, scales) in enumerate(placed):
        index[name] = len(sub0)
        sub0.append(tensor(name, list(shape), type_=INT4, scales=scales, ext=i + 1,
                           label=f"0:{name}"))  # fmt: skip

    v_norms = [f"v_norm.{layer}" for layer in range(LAYERS)]
    near = dict(base, **{v: np.full(HEAD, 0.5, np.float32) for v in v_norms})

    def constant(norm: str) -> int:
        """A norm constant near the base's value, in a buffer of its own."""
        cname = f"arith.constant{len(constants)}"
        value = (near[norm] + rng.normal(0, 0.01, near[norm].shape)).astype(np.float32)
        constants[cname] = value
        buffers.append(("table", [("bytes", value.view(np.uint8), cname)]))
        sub0.append(tensor(cname, [value.size], buffer=len(buffers) - 1))
        return len(sub0) - 1

    plan: list[tuple[str, str]] = [
        ("norm", P + "layers.0.input_layernorm.weight"),
        ("fc", "text_encoder/per_layer_model_projection/w"),
        ("norm", P + "ple.per_layer_projection_norm.weight"),
    ]
    for layer in range(LAYERS):
        n = norm_names(layer)
        w = {block: f"text_encoder/layer_{layer}/{block}/w;fused" for block, _ in LINEAR}
        w = {k: (rename or {}).get(v, v) for k, v in w.items()}
        plan += [("fc", w["attn/q_einsum"]), ("norm", n["q"]),
                 ("fc", w["attn/k_einsum"]), ("norm", n["k"]),
                 ("fc", w["attn/v_einsum"]), ("norm", v_norms[layer]),
                 ("fc", w["attn/attn_vec_einsum"])]  # fmt: skip
        # Layer 1 lists its two norms after attention the other way round, so
        # only their values can say which is which.
        after_attn = [n["post_attn"], n["pre_ff"]] if layer == 0 else [n["pre_ff"], n["post_attn"]]
        plan += [("norm", after_attn[0]), ("norm", after_attn[1]),
                 ("fc", w["mlp/gating_einsum1"]), ("fc", w["mlp/gating_einsum2"]),
                 ("fc", w["mlp/linear"]), ("norm", n["post_ff"]),
                 ("fc", w["ple/per_layer_embedding_gate"]),
                 ("fc", w["ple/per_layer_embedding_projection"]), ("norm", n["ple"])]  # fmt: skip
        nxt = norm_names(layer + 1)["input"] if layer + 1 < LAYERS else P + "norm.weight"
        plan.append(("norm", nxt))

    sub0.append(tensor("input", [1, HIDDEN]))
    current = len(sub0) - 1
    ops: list[tuple] = []
    which: dict[str, str] = {}  # constant name -> the norm it stands for
    for step, ref in plan:
        if step == "fc":
            inputs = [current, index[ref]]
            code = FC
        else:
            c = constant(ref)
            which[f"arith.constant{len(constants) - 1}"] = ref
            if ref == drop_composite:
                continue  # the constant is there, but no composite reads it
            inputs, code = [current, c], COMPOSITE
        sub0.append(tensor(f"act{len(ops)}", [1, HIDDEN]))
        ops.append(operator(code, inputs, [len(sub0) - 1]))
        current = len(sub0) - 1

    # Subgraph 1: the same weights, scale vectors of its own.
    sub1 = [
        tensor(name, list(shape), type_=INT4, scales=scales, ext=i + 1, label=f"1:{name}")
        for i, (name, _, shape, _, _, scales) in enumerate(placed)
    ]
    ext = [
        external(i + 1, offset, length, name, with_length=name != without_length)
        for i, (name, _, _, offset, length, _) in enumerate(placed)
    ]
    graphs["text_encoder"], marks["text_encoder"] = flatbuffer(
        model([subgraph(sub0, ops), subgraph(sub1)], buffers, ext, OPCODES)
    )
    return Google(graphs, weights, marks, tensors, {which[k]: v for k, v in constants.items()})


def bundle(path: Path, google: Google) -> Path:
    contents = [
        b"embedding metadata",
        b"tokenizer",
        google.graphs["embedder"],
        google.weights["embedder"],
        google.graphs["text_encoder"],
        google.weights["text_encoder"],
    ]
    FORMAT["write"](path, BUNDLE_TOML, contents)
    return path


def packed(path: Path, section_type: str, model_type: str) -> bytes:
    header, raw = FORMAT["read"](path)
    names = re.findall(r'data_path\s*=\s*"([^"]+)"', header["toml"])
    hits = [
        raw[b:e]
        for n, (b, e) in zip(names, header["sections"], strict=True)
        if section_type in n and n.endswith(f"_tf_lite_{model_type}" + n[n.rfind(".") :])
    ]
    assert len(hits) == 1, (section_type, model_type, names)
    return hits[0]


# -- running it -------------------------------------------------------------------


@dataclass
class FakeToolchain:
    """`StageEnv.run`: the graft script, run for real against the fake builder."""

    pack_alters_encoder: bool = False
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        assert args[0] == "python" and str(args[1]).endswith("embedding_graft.py"), args
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim"
        package = shim / "litert_lm_builder"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
        builder = FAKE_BUILDER
        if self.pack_alters_encoder:
            anchor = 's.get("model_type") == "prefill_decode"'
            assert builder.count(anchor) == 1
            builder = builder.replace(anchor, 's.get("model_type") == "text_encoder"')
        settings = (
            f"KNOWN_TYPES = {list(BUILDER_TYPES)!r}\nPACK_FAILS = False\n"
            f"PACK_ALTERS_VALUE = False\nPACK_ALTERS_BYTES = {self.pack_alters_encoder}\n"
            "SHRINK_INPUT = None\n"
        )
        (package / "litertlm_builder.py").write_text(settings + builder, encoding="utf-8")
        (package / "litertlm_peek.py").write_text(FAKE_PEEK, encoding="utf-8")
        (package / "litertlm_core.py").write_text(FAKE_FORMAT, encoding="utf-8")
        dist = shim / "litert_lm_builder-0.18.0.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: litert-lm-builder\nVersion: 0.18.0\n"
        )
        return subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim)},
        )


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


@dataclass
class Case:
    google: Google
    path: Path
    base: dict[str, Any]
    tuned: dict[str, Any]
    base_file: Path
    tuned_file: Path


@pytest.fixture
def case(tmp_path, monkeypatch) -> Case:
    return make_case(tmp_path, monkeypatch)


def make_case(tmp_path: Path, monkeypatch, tuned: dict[str, Any] | None = None,
              tuned_prefix: str = "", **google_kwargs) -> Case:  # fmt: skip
    base = base_checkpoint()
    tuned = fine_tune(base) if tuned is None else tuned
    google = google_file(base, **google_kwargs)
    path = bundle(tmp_path / "embeddinggemma-2-text-270m.litertlm", google)
    monkeypatch.setattr(
        embedding_convert, "GOOGLE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(embedding_convert, "NORM_SIZES", (HEAD, HIDDEN))
    base_file = safetensors(tmp_path / "base.safetensors", base, "BF16")
    tuned_file = safetensors(
        tmp_path / "tuned.safetensors", {tuned_prefix + k: v for k, v in tuned.items()}
    )
    return Case(google, path, base, tuned, base_file, tuned_file)


def run(case: Case, out: Path, weights: str = "int8") -> GraftResult:
    return graft(envs.EXPORT, case.path, case.base_file, case.tuned_file, out, weights)  # type: ignore[arg-type]


def refused(case: Case, out: Path, cause: str, match: str, weights: str = "int8") -> None:
    with pytest.raises(EmbeddingConvertError, match=match) as info:
        run(case, out, weights)
    assert info.value.cause == cause
    assert not out.exists()
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".")]


def f32(buf: bytes, pos: int, n: int) -> Any:
    return np.frombuffer(buf, "<f4", n, pos)


def cos(a: Any, b: Any) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def written(case: Case, out: Path, part: str) -> dict[str, tuple[int, Any, Any]]:
    """Each external tensor as the packed file holds it: (type, codes, scales of
    subgraph 0), read at the positions recorded when Google's graph was written."""
    graph = packed(out, "TFLiteModel", part)
    blob = packed(out, "TFLiteWeights", part)
    marks = case.google.marks[part]
    out_tensors = {}
    for name, (_, shape, _, _) in case.google.tensors[part].items():
        (offset,) = struct.unpack_from("<Q", graph, marks[f"{name}:offset"])
        (length,) = struct.unpack_from("<Q", graph, marks[f"{name}:length"])
        kind = graph[marks[f"0:{name}:type"]]
        raw = blob[offset : offset + length]
        codes = int4_codes(raw, shape) if kind == INT4 else np.frombuffer(raw, np.int8)
        out_tensors[name] = (
            kind,
            codes.reshape(shape),
            f32(graph, marks[f"0:{name}:scale"], shape[0]),
        )
    return out_tensors


# -- what is written --------------------------------------------------------------


def test_int8_writes_the_fine_tune_into_both_graphs(toolchain, case, tmp_path):
    out = tmp_path / "out" / "tuned.litertlm"
    result = run(case, out, "int8")
    assert out.is_file() and result.output == out
    assert result.sha256 == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest()
    assert result.bytes == out.stat().st_size
    for part in ("embedder", "text_encoder"):
        graph = packed(out, "TFLiteModel", part)
        marks = case.google.marks[part]
        for name, (hf, shape, _, _) in case.google.tensors[part].items():
            kind, codes, scales = written(case, out, part)[name]
            assert kind == INT8
            (offset,) = struct.unpack_from("<Q", graph, marks[f"{name}:offset"])
            (length,) = struct.unpack_from("<Q", graph, marks[f"{name}:length"])
            assert offset % 64 == 0 and length == shape[0] * shape[1]
            target = case.tuned[hf] * (FOLD if hf == PMP else 1.0)
            assert cos(codes * scales[:, None], target) > 0.9999
    assert result.parts["embedder"]["types"] == [INT8]
    assert result.parts["text_encoder"]["types"] == [INT8]
    assert result.parts["text_encoder"]["tensors"] == 2 + 9 * LAYERS
    assert result.builder == "0.18.0"
    # Every other section is Google's, byte for byte.
    header, raw = FORMAT["read"](out)
    kept = [raw[b:e] for b, e in header["sections"]]
    assert b"embedding metadata" in kept and b"tokenizer" in kept


def test_every_subgraphs_scale_vector_and_type_are_written(toolchain, case, tmp_path):
    out = tmp_path / "tuned.litertlm"
    run(case, out, "int8")
    graph = packed(out, "TFLiteModel", "text_encoder")
    marks = case.google.marks["text_encoder"]
    for name, (_, shape, _, _) in case.google.tensors["text_encoder"].items():
        first = f32(graph, marks[f"0:{name}:scale"], shape[0])
        second = f32(graph, marks[f"1:{name}:scale"], shape[0])
        original = f32(case.google.graphs["text_encoder"], marks[f"1:{name}:scale"], shape[0])
        assert np.array_equal(first, second)
        assert not np.array_equal(second, original)
        assert graph[marks[f"0:{name}:type"]] == graph[marks[f"1:{name}:type"]] == INT8


def test_int4_rewrites_the_codes_in_place_at_googles_size(toolchain, tmp_path, monkeypatch):
    # The fine-tune saved under a module prefix, as sentence-transformers may.
    case = make_case(tmp_path, monkeypatch, tuned_prefix="0.auto_model.")
    out = tmp_path / "tuned.litertlm"
    result = run(case, out, "int4")
    for part in ("embedder", "text_encoder"):
        graph = packed(out, "TFLiteModel", part)
        blob = packed(out, "TFLiteWeights", part)
        assert len(graph) == len(case.google.graphs[part])
        marks = case.google.marks[part]
        last = 0
        for name, (hf, shape, offset, length) in case.google.tensors[part].items():
            assert struct.unpack_from("<Q", graph, marks[f"{name}:offset"])[0] == offset
            assert struct.unpack_from("<Q", graph, marks[f"{name}:length"])[0] == length
            kind, codes, scales = written(case, out, part)[name]
            assert kind == INT4
            assert codes.min() >= -8 and codes.max() <= 7
            target = case.tuned[hf] * (FOLD if hf == PMP else 1.0)
            assert cos(codes * scales[:, None], target) > 0.97
            # The codes are the fine-tune's, not Google's left in place.
            google = int4_codes(case.google.weights[part][offset : offset + length], shape)
            assert not np.array_equal(codes, google)
            last = max(last, offset + length)
        assert len(blob) == last
    assert result.parts["text_encoder"]["types"] == [INT4]
    assert out.stat().st_size == case.path.stat().st_size


def test_int4_e8_writes_the_embedder_int8_and_the_encoder_int4(toolchain, case, tmp_path):
    out = tmp_path / "tuned.litertlm"
    result = run(case, out, "int4-e8")
    assert {k for k, _, _ in written(case, out, "embedder").values()} == {INT8}
    assert {k for k, _, _ in written(case, out, "text_encoder").values()} == {INT4}
    assert result.parts["embedder"]["bits"] == 8
    assert result.parts["text_encoder"]["bits"] == 4


def test_the_fold_found_against_the_base_is_applied_to_the_fine_tune(toolchain, case, tmp_path):
    out = tmp_path / "tuned.litertlm"
    result = run(case, out, "int8")
    assert result.folds == {PMP: FOLD}
    name = "text_encoder/per_layer_model_projection/w"
    _, codes, scales = written(case, out, "text_encoder")[name]
    values = codes * scales[:, None]
    tuned = case.tuned[PMP].astype(np.float64)
    alpha = float(values.ravel() @ tuned.ravel() / (tuned**2).sum())
    assert alpha == pytest.approx(FOLD, rel=1e-3)


def test_norms_are_placed_by_topology_and_value_and_v_norm_is_left(toolchain, case, tmp_path):
    out = tmp_path / "tuned.litertlm"
    result = run(case, out, "int8")
    graph = packed(out, "TFLiteModel", "text_encoder")
    marks = case.google.marks["text_encoder"]
    by_norm = {}
    for i, (norm, before) in enumerate(case.google.constants.items()):
        now = f32(graph, marks[f"arith.constant{i}"], before.size)
        by_norm[norm] = now
        if norm.startswith("v_norm"):
            assert np.array_equal(now, before)
        else:
            assert np.array_equal(now, case.tuned[norm].astype(np.float32)), norm
    # Layer 1 lists pre_feedforward before post_attention; each still got its own.
    for layer in (0, 1):
        n = norm_names(layer)
        assert np.array_equal(by_norm[n["pre_ff"]], case.tuned[n["pre_ff"]])
        assert np.array_equal(by_norm[n["post_attn"]], case.tuned[n["post_attn"]])
    norms = [k for k in case.base if k.endswith("norm.weight")]
    assert result.norms_written == len(norms) == 7 * LAYERS + 2
    assert len(result.norms_skipped) == LAYERS
    assert result.layer_scalars == LAYERS


# -- what is refused --------------------------------------------------------------


def test_a_file_that_is_not_googles_is_refused(toolchain, case, tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_convert, "GOOGLE_SHA256", "0" * 64)
    refused(case, tmp_path / "o.litertlm", "base_sha256", "not Google's 0000")


def test_a_tensor_that_maps_to_no_parameter_is_refused(toolchain, tmp_path, monkeypatch):
    new = "text_encoder/layer_1/mlp/gating_einsum3/w"
    case = make_case(
        tmp_path, monkeypatch, rename={"text_encoder/layer_1/mlp/gating_einsum1/w;fused": new}
    )
    refused(case, tmp_path / "o.litertlm", "unmapped", f"{new!r} maps to no parameter")


def test_a_parameter_the_fine_tune_lacks_is_refused(toolchain, tmp_path, monkeypatch):
    tuned = fine_tune(base_checkpoint())
    del tuned[f"{P}layers.1.mlp.up_proj.weight"]
    case = make_case(tmp_path, monkeypatch, tuned=tuned)
    refused(case, tmp_path / "o.litertlm", "unmapped", r"layers\.1\.mlp\.up_proj")


def test_a_fold_that_is_not_a_power_of_two_is_refused(toolchain, tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch, scale_pmp=0.3)
    refused(case, tmp_path / "o.litertlm", "unmapped", "not a power of two")


def test_a_norm_no_composite_reads_is_refused(toolchain, tmp_path, monkeypatch):
    post_attn = norm_names(0)["post_attn"]
    case = make_case(tmp_path, monkeypatch, drop_composite=post_attn)
    refused(case, tmp_path / "o.litertlm", "norm", r"1 norm constant\(s\).*needs 2")


def test_a_changed_layer_scalar_is_refused(toolchain, tmp_path, monkeypatch):
    tuned = fine_tune(base_checkpoint())
    tuned[f"{P}layers.1.layer_scalar"] = tuned[f"{P}layers.1.layer_scalar"] + 1e-3
    case = make_case(tmp_path, monkeypatch, tuned=tuned)
    refused(case, tmp_path / "o.litertlm", "layer_scalar", r"layers\.1\.layer_scalar")


def test_an_external_buffer_without_a_length_is_refused_for_int8(toolchain, tmp_path, monkeypatch):
    name = "text_encoder/layer_0/mlp/linear/w;fused"
    case = make_case(tmp_path, monkeypatch, without_length=name)
    refused(case, tmp_path / "o.litertlm", "field", re.escape(f"{name!r}'s external buffer"))


def test_a_pack_that_does_not_read_back_is_refused(toolchain, case, tmp_path):
    toolchain.pack_alters_encoder = True
    refused(case, tmp_path / "o.litertlm", "readback", "changed in the rebuild")


def test_an_existing_output_is_never_written_over(toolchain, case, tmp_path):
    out = tmp_path / "o.litertlm"
    out.write_bytes(b"mine")
    with pytest.raises(EmbeddingConvertError, match="already exists"):
        run(case, out)
    assert out.read_bytes() == b"mine"
    assert toolchain.calls == []


def test_an_unknown_weight_mode_is_refused_before_anything_runs(toolchain, case, tmp_path):
    with pytest.raises(EmbeddingConvertError, match="weights must be one of"):
        run(case, tmp_path / "o.litertlm", "int2")
    assert toolchain.calls == []


# -- the script's parts -----------------------------------------------------------


def test_int4_packs_the_first_code_into_the_low_nibble():
    codes = np.array([1, -2, 3, -8], np.int8)
    assert SCRIPT["pack_int4"](codes) == bytes([0xE1, 0x83])
    assert np.array_equal(SCRIPT["unpack_int4"](bytes([0xE1, 0x83]), [4]), codes)


@pytest.mark.parametrize("bits", [4, 8])
def test_the_scale_search_clips_an_outlier_row_and_never_loses_to_absmax(bits):
    rng = np.random.default_rng(3)
    w = rng.normal(size=(6, 64)).astype(np.float32)
    # One outlier: at four bits, clipping it is worth the rest of the row.
    w[0, 0] = 40.0
    codes, scales = SCRIPT["quant_rows"](w, bits)
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(w).max(axis=1) / qmax
    assert codes.dtype == np.int8 and codes.min() >= -qmax - 1 and codes.max() <= qmax

    def err(s: Any) -> Any:
        q = np.clip(np.round(w / s[:, None]), -qmax - 1, qmax)
        return ((q * s[:, None] - w) ** 2).sum(axis=1)

    assert np.all(err(scales) <= err(amax))
    assert scales[0] < amax[0] if bits == 4 else scales[0] == amax[0]
    assert np.array_equal(codes, np.clip(np.round(w / scales[:, None]), -qmax - 1, qmax))
    # Every scale is absmax times one of the search ratios.
    ratios = SCRIPT["search_ratios"](bits)
    for row in range(6):
        assert np.isclose(scales[row] / amax[row], ratios, rtol=1e-6).any()


def test_rows_are_quantized_alike_in_one_block_or_many(monkeypatch):
    rng = np.random.default_rng(4)
    w = rng.normal(size=(10, 32)).astype(np.float32)
    whole = SCRIPT["quant_rows"](w, 4)
    monkeypatch.setitem(SCRIPT, "CHUNK_ROWS", 3)
    blocks = SCRIPT["quant_rows"](w, 4)
    assert np.array_equal(whole[0], blocks[0]) and np.array_equal(whole[1], blocks[1])


def test_bfloat16_reads_as_its_float32(tmp_path):
    values = bf16(np.array([[1.5, -2.25], [3.0e-3, 7.0e5]], np.float32))
    path = safetensors(tmp_path / "w.safetensors", {"a": values}, "BF16")
    read = SCRIPT["Safetensors"](str(path), "w").get("a")
    assert read.dtype == np.float32 and np.array_equal(read, values)


def test_a_safetensors_tensor_past_the_end_is_refused(tmp_path):
    path = safetensors(tmp_path / "w.safetensors", {"a": np.ones((2, 2), np.float32)})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(SCRIPT["Refused"], match="'a'.*inside the file"):
        SCRIPT["Safetensors"](str(path), "w")


def test_the_read_back_refuses_a_scale_vector_that_differs(tmp_path):
    base = base_checkpoint()
    google = google_file(base)
    weights_in = tmp_path / "in.weight"
    weights_in.write_bytes(google.weights["text_encoder"])
    store = SCRIPT["Safetensors"](str(safetensors(tmp_path / "b.st", base)), "b")
    graph = bytearray(google.graphs["text_encoder"])
    names = {k: k for k in store.keys()}
    out = tmp_path / "w.bin"
    expected, _, _ = SCRIPT["write_part"](
        "text_encoder", 8, graph, str(weights_in), str(out), store, store, names, "g"
    )
    assert SCRIPT["read_back"]("text_encoder", bytes(graph), str(out), expected, {}, "g") == [INT8]
    name = "text_encoder/layer_1/attn/q_einsum/w;fused"
    graph[google.marks["text_encoder"][f"1:{name}:scale"]] ^= 1
    with pytest.raises(
        SCRIPT["Refused"], match=re.escape(f"{name!r} reads back with other scales")
    ):
        SCRIPT["read_back"]("text_encoder", bytes(graph), str(out), expected, {}, "g")
