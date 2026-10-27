"""`embedding_convert`: a fine-tuned EmbeddingGemma 2 text tower, written into
Google's own `.litertlm`.

No one publishes an export path for EmbeddingGemma 2; Google ships the converted
file (`litert-community/embeddinggemma-2-text-270m-litert-lm`,
`embeddinggemma-2-text-270m.litertlm`, 164,626,432 bytes, sha256 `GOOGLE_SHA256`).
So this takes that file, rewrites the weights of its two TFLite graphs --
`embedder` and `text_encoder` -- with the fine-tune's, and repacks it with
litert-lm-builder. Everything here is the graft spike's `graft_s2.py`
(MEASUREMENTS.md, "EmbeddingGemma 2: the graft"), for every figure below),
productized; where it differs, the script's docstring says so.

**What Google's file holds.** Every two-dimensional text-tower weight is an int4
tensor in the graph's external weights section, quantized per output row,
symmetric, low nibble first, its float32 scales inside the graph flatbuffer --
one scale vector per signature subgraph, six on Google's file. All 219 map one
to one onto the HF checkpoint's parameters by tensor name;
`per_layer_model_projection` carries a constant 1/512, which is found by fitting
Google's dequantized tensor against the BASE checkpoint's and folded into the
fine-tune's values. The RMSNorm weights are inline float32 constants, one
buffer each, shared by all subgraphs: each is an input of a `STABLEHLO_COMPOSITE`,
and tracing that op back to the nearest `FULLY_CONNECTED` names the layer and the
sub-block. That fixes q_norm and k_norm, and leaves at most two candidates for any
other norm, told apart by cosine to the base values. The two v_norm constants
have no HF weight and are left alone. A norm-sized constant that a composite in
any other subgraph reads must be one of those placed or left. HF holds
`layer_scalar` as a buffer, and nothing writes one into the graph, so the
fine-tune must leave it as the base's.

**Two weight modes**, the spike's variants: `int8` (`i8n_e8`: both graphs int8,
300,089,344 bytes in the spike) and `int4` (`i4n`: both int4, Google's size). The
spike's third, an int4 encoder with an int8 embedder, scored within 0.0007 of
int4 on 77 and 150 labels (10-shot centroid, mean of the draws: banking77
0.9320 against 0.9318, CLINC150 0.9652 against 0.9659; no confidence interval
was computed; MEASUREMENTS.md, "EmbeddingGemma 2: many labels"), and is not
offered. int8 changes each tensor's
type and moves its data to a 64-byte-aligned offset in a new weights section;
int4 rewrites the codes in place. Scales come from a per-row search over 141
clip ratios for the smallest squared error; int8 codes span [-127, 127].

**Refusals name their cause** (`EmbeddingConvertError.cause`): a base file whose
sha256 is not Google's; a tensor that maps to no HF parameter, whose fold factor
is not a power of two, or whose quantization is not per-row scales with zero
point 0; a norm that cannot be placed one to one; a `layer_scalar` the
fine-tune changed, or another parameter it changed that the file holds nowhere;
a non-finite value to be written; a field to patch that the flatbuffer does not
carry; and a pack that does not read back -- section by section, byte for byte
(the towers script's `rebuild`), then each graph against Google's outside the
bytes the graft wrote, then every written tensor, scale vector and norm re-read
from the packed file.

The work runs in `envs.EXPORT`, which has litert-lm-builder 0.18.0 and numpy
2.0.2 pinned and no torch requirement of this script: the safetensors files are
read by hand (an 8-byte header length, a JSON header, raw little-endian data).

What the file embeds is not checked here.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from litetune import envs
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.storage import hash_file
from litetune.towers import _TOWERS_SCRIPT

logger = logging.getLogger(__name__)

GOOGLE_REPO = "litert-community/embeddinggemma-2-text-270m-litert-lm"
GOOGLE_FILE = "embeddinggemma-2-text-270m.litertlm"
# The sha256 the graft measurement recorded for this file
# (MEASUREMENTS.md, "EmbeddingGemma 2: the graft").
GOOGLE_SHA256 = "2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb"
# The base checkpoint the mapping and fold factors were established against (the
# spike's README). Not checked here: this module is given the file, and
# `embedding_export` checks it against `BASE_SHA256` before the graft.
BASE_MODEL = "google/embeddinggemma-2"
BASE_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
# model.safetensors of BASE_MODEL at BASE_REVISION, 1,488,915,288 bytes: its LFS
# oid in https://huggingface.co/api/models/google/embeddinggemma-2/tree/<BASE_REVISION>,
# read 2026-10-26.
BASE_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"

WeightMode = Literal["int8", "int4"]
# int4: on banking77 (77 labels) and CLINC150 (150) it cost 0.0027 and 0.0017 of
# the decider's accuracy (centroid of 10 examples per label) against the float
# fine-tune, int8 0.0002 on both, at 165 MB against 300 MB
# (MEASUREMENTS.md, "EmbeddingGemma 2: many labels"). `verify` measures the file
# written, and int8 is there for a task where int4 does not pass.
DEFAULT_WEIGHTS: WeightMode = "int4"
WEIGHT_BITS: dict[str, dict[str, int]] = {
    "int8": {"embedder": 8, "text_encoder": 8},
    "int4": {"embedder": 4, "text_encoder": 4},
}
# The lengths of the norm constants the topology mapping looks at: EmbeddingGemma
# 2's head dimension and hidden size, as `graft_s2.py` has them.
NORM_SIZES = (256, 512)
# Not measured for this script; the spike's quantizer ran in torch.
GRAFT_TIMEOUT_S = 3600

_GRAFT_SCRIPT = r'''
"""Graft a fine-tuned EmbeddingGemma 2 text tower into Google's .litertlm; leave
the result, packed and read back, in the work directory for the parent to publish.

Reads JSON {google, google_sha256, base, tuned, work, bits: {embedder: 4|8,
text_encoder: 4|8}, norm_sizes: [n, ...]} from argv[1]. Writes JSON to argv[2]:
  {"ok": true, "rebuilt": <path>, "bytes": ..., "sha256": ..., "parts": {...}, ...}
  or {"ok": false, "cause": <str>, "reason": <str>}

As graft_s2.py, with these differences:
  - numpy, not torch. The 141 clip ratios are float32 and every product and sum is
    float32, as there, but numpy may sum a row in another order than torch, so a
    ratio that ties another to the last bit may be chosen differently. Whether the
    ratios equal torch.linspace's bit for bit was not checked.
  - rows are quantized in blocks of CHUNK_ROWS, which bounds memory; each row's
    search is independent of the others, so the result is the same.
  - the number of layers is read from the base checkpoint, not fixed at 24.
  - a fine-tune key matched by suffix is the shortest, then the first by name.
  - what graft_s2.py asserted, or recorded and went on past, is refused here.
"""
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

# TFLite's schema (tflite/converter/schema/schema.fbs), vtable offsets of the
# fields read or patched.
MODEL_OPCODES, MODEL_SUBGRAPHS, MODEL_BUFFERS, MODEL_EXTERNAL_BUFFERS = 6, 8, 12, 22
OPCODE_DEPRECATED, OPCODE_BUILTIN = 4, 10
SUBGRAPH_TENSORS, SUBGRAPH_OPERATORS = 4, 10
OP_OPCODE, OP_INPUTS, OP_OUTPUTS = 4, 6, 8
TENSOR_SHAPE, TENSOR_TYPE, TENSOR_BUFFER, TENSOR_NAME = 4, 6, 8, 10
TENSOR_QUANTIZATION, TENSOR_EXTERNAL = 12, 24
# QuantizationParameters: min, max, scale, zero_point ([long]), details (a union:
# its uint8 type, then its table), quantized_dimension (int) -- the offsets the
# generated tflite package's QuantizationParameters.py reads.
QUANT_SCALE, QUANT_ZERO_POINT, QUANT_DETAILS_TYPE, QUANT_DETAILS, QUANT_DIMENSION = (
    8, 10, 12, 14, 16)
BUFFER_DATA = 4
EXT_ID, EXT_OFFSET, EXT_LENGTH = 4, 8, 10
FLOAT32, INT8, INT4 = 0, 9, 17
FULLY_CONNECTED, STABLEHLO_COMPOSITE = 9, 206

PARTS = ("embedder", "text_encoder")
PREFIX = "language_model."
SEARCH_STEPS = 141
CHUNK_ROWS = 4096
INT8_ALIGN = 64
BACK_DEPTH = 16
# A fitted factor this close to 1 is no fold; past it, it must be a power of two.
FOLD_TOLERANCE = 0.05
FOLD_EXPONENT_TOLERANCE = 0.1
# The safetensors format caps its header at 100 MB.
MAX_SAFETENSORS_HEADER = 100 << 20
DTYPES = {"F32": 4, "BF16": 2, "F16": 2}

# graft_s2.py's LAYER_MAP: the graph's sub-block names and the HF parameters.
LAYER_MAP = [("q_einsum", "self_attn.q_proj"), ("k_einsum", "self_attn.k_proj"),
             ("v_einsum", "self_attn.v_proj"), ("attn_vec_einsum", "self_attn.o_proj"),
             ("gating_einsum1", "mlp.gate_proj"), ("gating_einsum2", "mlp.up_proj"),
             ("mlp/linear", "mlp.down_proj"),
             ("per_layer_embedding_gate", "ple_block.per_layer_input_gate"),
             ("per_layer_embedding_projection", "ple_block.per_layer_projection")]
# The FULLY_CONNECTED sub-blocks a norm can follow, in graft_s2.py's order.
KINDS = ["q_einsum", "k_einsum", "v_einsum", "attn_vec_einsum", "mlp/linear",
         "per_layer_embedding_projection", "per_layer_model_projection"]


class Refused(Exception):
    """A reason this script will not write the file asked for."""

    def __init__(self, cause, reason):
        super().__init__(reason)
        self.cause = cause


# -- TFLite flatbuffers, read by hand: no TFLite schema is installed -------------

class Flat:
    """Reads of one flatbuffer, each checked against its length, and the byte
    positions of the fields this script patches."""

    def __init__(self, buf, where):
        self.buf = buf
        self.where = where

    def at(self, fmt, pos):
        if not 0 <= pos <= len(self.buf) - struct.calcsize(fmt):
            raise Refused("input", f"{self.where}: an offset points outside its "
                          f"{len(self.buf)} bytes")
        return struct.unpack_from(fmt, self.buf, pos)[0]

    def field(self, table, slot):
        """Where `table`'s field at vtable offset `slot` is, or None when unset."""
        vtable = table - self.at("<i", table)
        if slot >= self.at("<H", vtable):
            return None
        offset = self.at("<H", vtable + slot)
        return table + offset if offset else None

    def ref(self, pos):
        return pos + self.at("<I", pos)

    def scalar(self, table, slot, fmt):
        # An unset scalar is its schema default, 0 for every one read here.
        pos = self.field(table, slot)
        return 0 if pos is None else self.at(fmt, pos)

    def vector(self, table, slot, size):
        """(position of the first element, count) of a vector field, None when unset."""
        pos = self.field(table, slot)
        if pos is None:
            return None
        start = self.ref(pos)
        count = self.at("<I", start)
        if count > (len(self.buf) - start - 4) // size:
            raise Refused("input", f"{self.where}: a vector of {count} elements runs past "
                          "its end")
        return start + 4, count

    def tables(self, table, slot):
        found = self.vector(table, slot, 4)
        return [] if found is None else [self.ref(found[0] + 4 * i) for i in range(found[1])]

    def ints(self, table, slot):
        found = self.vector(table, slot, 4)
        if found is None:
            return []
        return [int(x) for x in np.frombuffer(self.buf, "<i4", found[1], found[0])]

    def string(self, table, slot):
        found = self.vector(table, slot, 1)
        if found is None:
            return ""
        return bytes(self.buf[found[0]:found[0] + found[1]]).decode("utf-8", "replace")


def scan(buf, where):
    """The graph's externalized weights, by their offset in the weights section:
    name, shape, length, and for every tensor that refers to one -- in every
    subgraph -- where its type byte and its scale vector are, and where every
    ExternalBuffer table naming that offset is."""
    fb = Flat(buf, where)
    model = fb.ref(0)
    by_id, at_offset = {}, {}
    for e in fb.tables(model, MODEL_EXTERNAL_BUFFERS):
        offset, length = fb.scalar(e, EXT_OFFSET, "<Q"), fb.scalar(e, EXT_LENGTH, "<Q")
        # An unset length reads as 0, and there would be no field to rewrite.
        by_id[fb.scalar(e, EXT_ID, "<I")] = (offset, length, fb.field(e, EXT_LENGTH) is not None)
        at_offset.setdefault(offset, []).append(e)
    weights = {}
    for si, sub in enumerate(fb.tables(model, MODEL_SUBGRAPHS)):
        for j, t in enumerate(fb.tables(sub, SUBGRAPH_TENSORS)):
            ext = fb.scalar(t, TENSOR_EXTERNAL, "<I")
            if not ext:
                continue
            name = fb.string(t, TENSOR_NAME)
            if ext not in by_id:
                raise Refused("input", f"{where}: tensor {name!r} names external buffer {ext}, "
                              "which the graph does not list")
            offset, length, has_length = by_id[ext]
            if not has_length:
                raise Refused("field", f"{where}: tensor {name!r}'s external buffer carries no "
                              "length field")
            shape = fb.ints(t, TENSOR_SHAPE)
            w = weights.setdefault(offset, {"name": name, "shape": shape, "length": length,
                                            "places": [], "types": [], "type_pos": [],
                                            "scales": [], "ext_pos": at_offset[offset]})
            if shape != w["shape"] or length != w["length"]:
                raise Refused("input", f"{where}: tensors {w['name']!r} and {name!r} share "
                              "external data but not its shape or length")
            w["places"].append([si, j])
            w["types"].append(fb.scalar(t, TENSOR_TYPE, "<b"))
            w["type_pos"].append(fb.field(t, TENSOR_TYPE))
            quant = fb.field(t, TENSOR_QUANTIZATION)
            q = None if quant is None else fb.ref(quant)
            scale = None if q is None else fb.vector(q, QUANT_SCALE, 4)
            if scale is None:
                raise Refused("field", f"{where}: tensor {name!r} in subgraph {si} carries no "
                              "scale vector to write its scales into")
            # What is written is one scale per output row, zero point 0, and no
            # other scheme: a file that says otherwise would read the codes another way.
            zero = fb.vector(q, QUANT_ZERO_POINT, 8)
            if fb.scalar(q, QUANT_DIMENSION, "<i") != 0:
                raise Refused("input", f"{where}: tensor {name!r} in subgraph {si} is quantized "
                              f"along dimension {fb.scalar(q, QUANT_DIMENSION, '<i')}, not "
                              "per output row")
            if zero is not None and np.frombuffer(buf, "<i8", zero[1], zero[0]).any():
                raise Refused("input", f"{where}: tensor {name!r} in subgraph {si} has a "
                              "non-zero zero point")
            if (fb.scalar(q, QUANT_DETAILS_TYPE, "<B")
                    or fb.field(q, QUANT_DETAILS) is not None):
                raise Refused("input", f"{where}: tensor {name!r} in subgraph {si} carries "
                              "quantization details, a scheme other than per-row scales")
            w["scales"].append(scale)
    referred = set(weights)
    stray = sorted(set(at_offset) - referred)
    if stray:
        raise Refused("input", f"{where}: external data at offset {stray[0]} belongs to no "
                      "tensor, and would not be carried into the new weights")
    return fb, model, weights


def buffer_data(fb, model, index):
    """(position, byte count) of buffer `index`'s inline data, None when it has none."""
    buffers = fb.tables(model, MODEL_BUFFERS)
    if index >= len(buffers):
        raise Refused("input", f"{fb.where}: a tensor names buffer {index} of {len(buffers)}")
    return fb.vector(buffers[index], BUFFER_DATA, 1)


# -- safetensors, read by hand: no safetensors package is required ---------------

class Safetensors:
    """A safetensors file's tensors as float32 arrays, checked against its length."""

    def __init__(self, path, where):
        self.path, self.where = path, where
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise Refused("input", f"{where} is {size} bytes, too short for safetensors")
            (n,) = struct.unpack("<Q", raw)
            if n > MAX_SAFETENSORS_HEADER or 8 + n > size:
                raise Refused("input", f"{where}: a {n}-byte header does not fit its "
                              f"{size} bytes")
            try:
                header = json.loads(f.read(n))
            except ValueError as exc:
                raise Refused("input", f"{where}: its header is not JSON: {exc}") from None
        if not isinstance(header, dict):
            raise Refused("input", f"{where}: its header is not a JSON object")
        header.pop("__metadata__", None)
        self.data_at = 8 + n
        self.tensors = {}
        for name, t in header.items():
            try:
                dtype, shape, (begin, end) = t["dtype"], [int(d) for d in t["shape"]], \
                    [int(o) for o in t["data_offsets"]]
            except (KeyError, TypeError, ValueError):
                raise Refused("input", f"{where}: tensor {name!r} has no dtype, shape and "
                              "data_offsets") from None
            if dtype not in DTYPES:
                raise Refused("input", f"{where}: tensor {name!r} is {dtype}, not one of "
                              f"{sorted(DTYPES)}")
            count = math.prod(shape)
            if (min(shape, default=0) < 0 or not 0 <= begin <= end
                    or end - begin != count * DTYPES[dtype] or self.data_at + end > size):
                raise Refused("input", f"{where}: tensor {name!r}'s data_offsets do not hold "
                              f"its {dtype} {shape} inside the file")
            self.tensors[name] = (dtype, shape, begin, end)

    def keys(self):
        return list(self.tensors)

    def shape(self, name):
        return self.tensors[name][1]

    def get(self, name):
        dtype, shape, begin, end = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.data_at + begin)
            raw = f.read(end - begin)
        if len(raw) != end - begin:
            raise Refused("input", f"{self.where} ended inside tensor {name!r}")
        if dtype == "F32":
            out = np.frombuffer(raw, "<f4").astype(np.float32)
        elif dtype == "F16":
            out = np.frombuffer(raw, "<f2").astype(np.float32)
        else:  # BF16: the upper half of a float32
            out = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        return out.reshape(shape)


def tuned_names(base_keys, tuned_keys):
    """Each base key's name in the fine-tune: the same name, else the shortest that
    ends with `language_model.<the rest>` -- a fine-tune saved by
    sentence-transformers may sit under a module prefix."""
    present = set(tuned_keys)
    out = {}
    for k in base_keys:
        if k in present:
            out[k] = k
            continue
        tail = PREFIX + k[len(PREFIX):]
        hits = sorted((x for x in tuned_keys if x.endswith(tail)), key=lambda x: (len(x), x))
        if hits:
            out[k] = hits[0]
    return out


# -- quantization -----------------------------------------------------------------

def search_ratios(bits):
    """graft_s2.py's torch.linspace(lo, 1.0, 141), in float32."""
    lo = 0.3 if bits == 4 else 0.6
    return np.linspace(lo, 1.0, SEARCH_STEPS).astype(np.float32)


def code_range(bits):
    """(qmin, qmax) of a symmetric per-row code. int8: [-127, 127], the range
    TFLite's quantization spec gives int8 per-axis weights with zero point 0
    (graft_s2.py used -128). int4: [-8, 7] as graft_s2.py has it; that range
    was not checked against Google's file."""
    qmax = 2 ** (bits - 1) - 1
    return (-qmax if bits == 8 else -(2 ** (bits - 1))), qmax


def quant_rows(w, bits):
    """Per-row symmetric codes and float32 scales: for each row, the scale
    absmax * c / qmax over the search ratios c with the smallest squared error,
    the first such c on a tie. Refused for a non-finite value; with every value
    finite, absmax is finite and at least 1e-12 and c is in (0, 1], so no scale
    is NaN, infinite or zero."""
    qmin, qmax = code_range(bits)
    fmax = np.float32(qmax)
    ratios = search_ratios(bits)
    w = np.asarray(w, np.float32)
    if not np.isfinite(w).all():
        raise Refused("nonfinite", "a tensor to quantize holds non-finite values")
    codes = np.empty(w.shape, np.int8)
    scales = np.empty(w.shape[0], np.float32)
    for start in range(0, w.shape[0], CHUNK_ROWS):
        part = w[start:start + CHUNK_ROWS]
        amax = np.maximum(np.abs(part).max(axis=1), np.float32(1e-12))
        best_err = np.full_like(amax, np.inf)
        best = amax / fmax
        for c in ratios:
            s = amax * c / fmax
            err = ((np.clip(np.round(part / s[:, None]), qmin, qmax) * s[:, None] - part) ** 2
                   ).sum(axis=1)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best = np.where(better, s, best)
        codes[start:start + len(part)] = np.clip(np.round(part / best[:, None]), qmin, qmax)
        scales[start:start + len(part)] = best
    return codes, scales


def pack_int4(codes):
    """Two codes a byte, the first in the low nibble."""
    u = (codes.astype(np.int16).reshape(-1) & 15).astype(np.uint8)
    if u.size % 2:
        raise Refused("input", "an int4 tensor of an odd number of values cannot be packed")
    return (u[0::2] | (u[1::2] << 4)).tobytes()


def unpack_int4(raw, shape):
    a8 = np.frombuffer(raw, np.uint8)
    q = np.stack([(a8 & 15).astype(np.int8), (a8 >> 4).astype(np.int8)], 1).reshape(-1)
    return np.where(q > 7, q - 16, q).astype(np.int8).reshape(shape)


def cosine(x, ref):
    x = np.asarray(x, np.float64).reshape(-1)
    ref = np.asarray(ref, np.float64).reshape(-1)
    return float(x @ ref / np.linalg.norm(x) / np.linalg.norm(ref))


# -- mapping ----------------------------------------------------------------------

def map_name(part, tensor):
    """The HF parameter a graph tensor holds, or None."""
    if part == "embedder":
        return PREFIX + "embed_tokens.weight"
    first = tensor.split(";")[0]
    if "per_layer_model_projection" in first:
        return PREFIX + "ple.per_layer_model_projection.weight"
    if "heads_layer/default_projection" in first:
        return PREFIX + "embedding_projection.weight"
    m = re.search(r"/layer_(\d+)/", first)
    hits = [h for g, h in LAYER_MAP if g in first]
    if not m or len(hits) != 1:
        return None
    return f"{PREFIX}layers.{m.group(1)}.{hits[0]}.weight"


def fold_factor(google, base, name):
    """The constant the graph folded into `name`: Google's dequantized tensor
    fitted against the base checkpoint's, by least squares."""
    b = base.reshape(-1).astype(np.float64)
    alpha = float(google.reshape(-1).astype(np.float64) @ b / (b ** 2).sum())
    if abs(alpha - 1) <= FOLD_TOLERANCE:
        return 1.0
    # `round` of a NaN raises, so a factor at or below zero is refused before it.
    e = math.log2(alpha) if alpha > 0 else math.nan
    if not alpha > 0 or not abs(e - round(e)) < FOLD_EXPONENT_TOLERANCE:
        raise Refused("unmapped", f"{name}: Google's tensor is {alpha:.6g} times the base's, "
                      "not a power of two, so it does not hold this parameter as mapped")
    return float(2.0 ** round(e))


def opcodes(fb, model):
    """Each operator code's builtin code."""
    return [max(fb.scalar(c, OPCODE_DEPRECATED, "<b"), fb.scalar(c, OPCODE_BUILTIN, "<i"))
            for c in fb.tables(model, MODEL_OPCODES)]


def op_code(fb, op, codes):
    index = fb.scalar(op, OP_OPCODE, "<I")
    if index >= len(codes):
        raise Refused("input", f"{fb.where}: an operator names opcode {index} of {len(codes)}")
    return codes[index]


def topology(fb, model):
    """Subgraph 0's operators as (builtin code, inputs, outputs), which op makes
    each tensor, which ops read it, and its tensors' table positions."""
    codes = opcodes(fb, model)
    subgraphs = fb.tables(model, MODEL_SUBGRAPHS)
    if not subgraphs:
        raise Refused("input", f"{fb.where} has no subgraph")
    ops, made_by, read_by = [], {}, {}
    for i, op in enumerate(fb.tables(subgraphs[0], SUBGRAPH_OPERATORS)):
        ins, outs = fb.ints(op, OP_INPUTS), fb.ints(op, OP_OUTPUTS)
        ops.append((op_code(fb, op, codes), ins, outs))
        for o in outs:
            made_by[o] = i
        for t in ins:
            read_by.setdefault(t, []).append(i)
    return ops, made_by, read_by, fb.tables(subgraphs[0], SUBGRAPH_TENSORS)


def inline_float32(fb, model, t, sizes):
    """(buffer index, value count) of tensor table `t` when it is an inline
    float32 constant of one of `sizes` values, else None."""
    buffer = fb.scalar(t, TENSOR_BUFFER, "<I")
    if fb.scalar(t, TENSOR_EXTERNAL, "<I") or not buffer:
        return None
    if fb.scalar(t, TENSOR_TYPE, "<b") != FLOAT32:
        return None
    data = buffer_data(fb, model, buffer)
    n = 0 if data is None else data[1] // 4
    return (buffer, n) if n in sizes else None


def composite_constants(fb, model, sizes):
    """Every subgraph's inline float32 constants of a norm's length that a
    STABLEHLO_COMPOSITE reads: buffer index -> (subgraph, tensor name), the
    first subgraph that reads it."""
    codes = opcodes(fb, model)
    found = {}
    for si, sub in enumerate(fb.tables(model, MODEL_SUBGRAPHS)):
        tensors = fb.tables(sub, SUBGRAPH_TENSORS)
        read = set()
        for op in fb.tables(sub, SUBGRAPH_OPERATORS):
            if op_code(fb, op, codes) == STABLEHLO_COMPOSITE:
                read.update(fb.ints(op, OP_INPUTS))
        for j in sorted(read):
            if j < 0:  # an optional input left out
                continue
            if j >= len(tensors):
                raise Refused("input", f"{fb.where}: an operator in subgraph {si} reads tensor "
                              f"{j} of {len(tensors)}")
            hit = inline_float32(fb, model, tensors[j], sizes)
            if hit is not None and hit[0] not in found:
                found[hit[0]] = (si, fb.string(tensors[j], TENSOR_NAME))
    return found


def back_fc(fb, ops, made_by, tensors, start):
    """The weight name of the nearest FULLY_CONNECTED before op `start`, searched
    breadth first through the ops that make its inputs, BACK_DEPTH steps deep."""
    seen, frontier = set(), [start]
    for _ in range(BACK_DEPTH):
        nxt = []
        for o in frontier:
            code, ins, _ = ops[o]
            if code == FULLY_CONNECTED and o != start:
                if len(ins) < 2 or not 0 <= ins[1] < len(tensors):
                    raise Refused("input", f"{fb.where}: a FULLY_CONNECTED has no weight input")
                return fb.string(tensors[ins[1]], TENSOR_NAME).split(";")[0]
            for t in ins:
                if t in made_by and made_by[t] not in seen:
                    seen.add(made_by[t])
                    nxt.append(made_by[t])
        frontier = nxt
    return None


def norm_constants(fb, model, sizes):
    """Subgraph 0's inline float32 constants of a norm's length that a
    STABLEHLO_COMPOSITE reads, by buffer index: name, length, value, and the layer
    and sub-block of the FULLY_CONNECTED before it."""
    ops, made_by, read_by, tensors = topology(fb, model)
    found = {}
    for j, t in enumerate(tensors):
        hit = inline_float32(fb, model, t, sizes)
        if hit is None or j not in read_by:
            continue
        buffer, n = hit
        data = buffer_data(fb, model, buffer)
        for o in read_by[j]:
            if ops[o][0] != STABLEHLO_COMPOSITE:
                continue
            fc = back_fc(fb, ops, made_by, tensors, o)
            layer = re.search(r"/layer_(\d+)/", fc) if fc else None
            kind = next((k for k in KINDS if fc and k in fc), None)
            found[buffer] = {"name": fb.string(t, TENSOR_NAME), "n": n,
                             "value": np.frombuffer(fb.buf, "<f4", n, data[0]).copy(),
                             "layer": int(layer.group(1)) if layer else None, "kind": kind}
    return found


def candidates(g, layers):
    """The HF norms a constant can be, from what precedes it (graft_s2.py)."""
    l, k = g["layer"], g["kind"]
    if k == "q_einsum":
        return [f"{PREFIX}layers.{l}.self_attn.q_norm.weight"]
    if k == "k_einsum":
        return [f"{PREFIX}layers.{l}.self_attn.k_norm.weight"]
    if k == "v_einsum":
        return []  # HF's v_norm has no weight
    if k == "attn_vec_einsum":
        return [f"{PREFIX}layers.{l}.post_attention_layernorm.weight",
                f"{PREFIX}layers.{l}.pre_feedforward_layernorm.weight"]
    if k == "mlp/linear":
        return [f"{PREFIX}layers.{l}.post_feedforward_layernorm.weight"]
    if k == "per_layer_embedding_projection":
        return [f"{PREFIX}layers.{l}.ple_block.post_per_layer_input_norm.weight",
                f"{PREFIX}layers.{l + 1}.input_layernorm.weight" if l + 1 < layers
                else f"{PREFIX}norm.weight"]
    if k == "per_layer_model_projection":
        return [f"{PREFIX}ple.per_layer_projection_norm.weight"]
    return [f"{PREFIX}layers.0.input_layernorm.weight"]


def place_norms(found, base, layers):
    """Each norm constant's HF norm, by buffer index: inside each set of
    candidates, the one-to-one assignment with the largest summed cosine to the
    base values. Refused unless every HF norm gets exactly one constant."""
    by_set, skipped = {}, []
    for buffer, g in found.items():
        c = candidates(g, layers)
        if not c:
            skipped.append(g["name"])
            continue
        by_set.setdefault(tuple(c), []).append(buffer)
    placed = {}
    for cset, buffers in by_set.items():
        if len(buffers) != len(cset):
            raise Refused("norm", f"{len(buffers)} norm constant(s) in the encoder graph lead "
                          f"to {', '.join(cset)}; placing them one to one needs {len(cset)}")
        for k in cset:
            if k not in base.tensors:
                raise Refused("norm", f"the graph has a norm constant for {k}, which the base "
                              "checkpoint does not have")
        best = None
        for perm in itertools.permutations(cset):
            score = 0.0
            for buffer, k in zip(buffers, perm):
                v, h = found[buffer]["value"], base.get(k)
                score += cosine(v, h) if v.size == h.size else -9
            if best is None or score > best[0]:
                best = (score, perm)
        for buffer, k in zip(buffers, best[1]):
            if math.prod(base.shape(k)) != found[buffer]["n"]:
                raise Refused("norm", f"{k} has {math.prod(base.shape(k))} values and the "
                              f"constant it was placed on ({found[buffer]['name']!r}) "
                              f"{found[buffer]['n']}")
            placed[buffer] = k
    norms = [k for k in base.keys() if k.startswith(PREFIX) and k.endswith("norm.weight")]
    missing = sorted(set(norms) - set(placed.values()))
    if missing:
        raise Refused("norm", f"no norm constant in the encoder graph was placed for "
                      f"{missing[0]}" + (f" and {len(missing) - 1} more" if len(missing) > 1
                                         else ""))
    return placed, skipped


# -- writing ----------------------------------------------------------------------

def find_parts(doc, where):
    """Each part's graph and weights section, from the TOML `unpack` wrote."""
    out = {}
    for part in PARTS:
        for kind in ("TFLiteModel", "TFLiteWeights"):
            hits = [i for i, s in enumerate(doc["section"])
                    if s.get("section_type") == kind
                    and str(s.get("model_type", "")).lower().removeprefix("tf_lite_") == part]
            if len(hits) != 1:
                raise Refused("input", f"{where} has {len(hits)} {kind} sections for {part}, "
                              "not one")
            out[(part, kind)] = hits[0]
    return out


def write_part(part, bits, graph, weights_in, weights_out, base, tuned, names, where):
    """Quantize the fine-tune's tensors into `graph` (a bytearray, patched in place)
    and write the new weights section to `weights_out`. Returns what the read-back
    expects, this part's statistics, the folds, the HF parameters written, and
    every (position, byte count) of `graph` patched."""
    fb, _, found = scan(graph, where)
    expected, cos_src, cos_google, folds, mapped, changed = {}, [], [], {}, {}, 0
    patched = []
    written = 0
    with open(weights_in, "rb") as src, open(weights_out, "wb") as dst:
        for offset in sorted(found):
            w = found[offset]
            name, shape, length = w["name"], w["shape"], w["length"]
            if set(w["types"]) != {INT4}:
                raise Refused("input", f"{where}: tensor {name!r} is of type {w['types']}, "
                              "not Google's int4")
            if len(shape) != 2 or length * 2 != shape[0] * shape[1]:
                raise Refused("input", f"{where}: tensor {name!r} of shape {shape} does not "
                              f"fill its {length} bytes at four bits a value")
            rows = shape[0]
            for _, count in w["scales"]:
                if count != rows:
                    raise Refused("field", f"{where}: tensor {name!r} has a scale vector of "
                                  f"{count} values for its {rows} rows")
            hf = map_name(part, name)
            if hf is None or hf not in base.tensors:
                raise Refused("unmapped", f"{where}: tensor {name!r} maps to no parameter of "
                              "the base checkpoint")
            if hf in mapped:
                raise Refused("unmapped", f"{where}: tensors {mapped[hf]!r} and {name!r} both "
                              f"map to {hf}")
            mapped[hf] = name
            if hf not in names:
                raise Refused("unmapped", f"the fine-tune has no tensor for {hf}")
            if base.shape(hf) != shape or tuned.shape(names[hf]) != shape:
                raise Refused("unmapped", f"{where}: tensor {name!r} is {shape}, the base's "
                              f"{hf} {base.shape(hf)} and the fine-tune's "
                              f"{tuned.shape(names[hf])}")
            src.seek(offset)
            raw = src.read(length)
            if len(raw) != length:
                raise Refused("input", f"{where}: the weights end inside tensor {name!r}")
            pos, _ = w["scales"][0]
            google = unpack_int4(raw, shape).astype(np.float32) * np.frombuffer(
                graph, "<f4", rows, pos)[:, None]
            b = base.get(hf)
            fold = fold_factor(google, b, hf)
            if fold != 1.0:
                folds[hf] = fold
            t = tuned.get(names[hf])
            changed += int(not np.array_equal(t, b))
            del b
            t = t * fold
            if not np.isfinite(t).all():
                raise Refused("nonfinite", f"the fine-tune's {names[hf]} holds non-finite "
                              "values" + (f" once multiplied by its fold {fold}" if fold != 1.0
                                          else ""))
            codes, scales = quant_rows(t, bits)
            approx = codes.astype(np.float32) * scales[:, None]
            cos_src.append(cosine(approx, t))
            cos_google.append(cosine(approx, google))
            del t, google, approx
            scale_bytes = scales.astype("<f4").tobytes()
            for pos, _ in w["scales"]:
                graph[pos:pos + 4 * rows] = scale_bytes
                patched.append((pos, 4 * rows))
            if bits == 4:
                data = pack_int4(codes)
                if offset < written:
                    raise Refused("input", f"{where}: tensor {name!r} overlaps the one before")
                new_offset = offset
            else:
                data = codes.astype(np.int8).tobytes()
                new_offset = -(-written // INT8_ALIGN) * INT8_ALIGN
                for pos in w["type_pos"]:
                    graph[pos] = INT8
                    patched.append((pos, 1))
                for e in w["ext_pos"]:
                    at_offset, at_length = fb.field(e, EXT_OFFSET), fb.field(e, EXT_LENGTH)
                    # An unset offset reads as 0: left so only where 0 is the new one.
                    if at_length is None or (at_offset is None and new_offset):
                        raise Refused("field", f"{where}: tensor {name!r}'s external buffer "
                                      "carries no field to write its new offset or length to")
                    if at_offset is not None:
                        struct.pack_into("<Q", graph, at_offset, new_offset)
                        patched.append((at_offset, 8))
                    struct.pack_into("<Q", graph, at_length, len(data))
                    patched.append((at_length, 8))
            dst.write(bytes(new_offset - written))
            dst.write(data)
            written = new_offset + len(data)
            expected[new_offset] = {"name": name, "places": w["places"],
                                    "type": INT4 if bits == 4 else INT8, "length": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest(),
                                    "scales": scale_bytes}
    stats = {"bits": bits, "tensors": len(found), "changed_by_fine_tune": changed,
             "weights_bytes": written,
             "cos_to_source_mean": float(np.mean(cos_src)), "cos_to_source_min": min(cos_src),
             "cos_to_google_mean": float(np.mean(cos_google)),
             "cos_to_google_min": min(cos_google)}
    return expected, stats, folds, set(mapped), patched


def unpatched_difference(graph, original, patched):
    """The first byte position where `graph` differs from `original` outside
    every (position, byte count) in `patched`, or None; -1 for another length."""
    if len(graph) != len(original):
        return -1
    differs = np.frombuffer(graph, np.uint8) != np.frombuffer(original, np.uint8)
    for pos, count in patched:
        differs[pos:pos + count] = False
    hits = np.flatnonzero(differs)
    return int(hits[0]) if hits.size else None


def read_back(part, graph, weights_path, expected, norms, where, original, patched):
    """Refused unless the packed graph, scanned afresh, holds what was written:
    Google's graph `original`, byte for byte, outside the ranges the graft patched
    (`patched`: type bytes, scale vectors, external offsets and lengths, norm
    buffers); every external tensor in every place, of its type, at its offset and
    length, with its scales and its bytes; and every norm constant written."""
    at = unpatched_difference(graph, original, patched)
    if at == -1:
        raise Refused("readback", f"{where} is {len(graph)} bytes where Google's is "
                      f"{len(original)}; the graft patches it in place")
    if at is not None:
        raise Refused("readback", f"{where} differs from Google's at byte {at}, which the "
                      "graft did not write")
    fb, model, found = scan(graph, where)
    if sorted(found) != sorted(expected):
        raise Refused("readback", f"{where}: external data sits at {len(found)} offsets "
                      f"where {len(expected)} were written")
    size = os.path.getsize(weights_path)
    end = max((o + w["length"] for o, w in found.items()), default=0)
    if end != size:
        raise Refused("readback", f"{where}: the external data ends at byte {end} of a "
                      f"{size}-byte weights section")
    types = set()
    with open(weights_path, "rb") as f:
        for offset, w in found.items():
            want = expected[offset]
            name = want["name"]
            if w["places"] != want["places"] or w["length"] != want["length"]:
                raise Refused("readback", f"{where}: tensor {name!r} reads back elsewhere or "
                              "at another length")
            if set(w["types"]) != {want["type"]}:
                raise Refused("readback", f"{where}: tensor {name!r} reads back as type "
                              f"{w['types']}")
            types |= set(w["types"])
            for pos, count in w["scales"]:
                if bytes(graph[pos:pos + 4 * count]) != want["scales"]:
                    raise Refused("readback", f"{where}: tensor {name!r} reads back with "
                                  "other scales")
            f.seek(offset)
            if hashlib.sha256(f.read(w["length"])).hexdigest() != want["sha256"]:
                raise Refused("readback", f"{where}: tensor {name!r} reads back with other "
                              "bytes")
    for buffer, value in norms.items():
        data = buffer_data(fb, model, buffer)
        if data is None or bytes(graph[data[0]:data[0] + data[1]]) != value:
            raise Refused("readback", f"{where}: norm buffer {buffer} reads back with other "
                          "bytes")
    return sorted(types)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def graft(spec, T):
    work = Path(spec["work"])
    google = spec["google"]
    if sha256(google) != spec["google_sha256"]:
        raise Refused("base_sha256", f"{Path(google).name} has sha256 {sha256(google)}, not "
                      f"Google's {spec['google_sha256']}")
    try:
        places = T.spans(google, "Google's file")
        before_dir = work / "before"
        doc, sources = T.unpacked(google, before_dir, "Google's file", places)
    except T.Refused as exc:
        raise Refused("input", str(exc)) from None
    sections = find_parts(doc, "Google's file")

    base = Safetensors(spec["base"], "the base checkpoint")
    tuned = Safetensors(spec["tuned"], "the fine-tune")
    base_keys = [k for k in base.keys() if k.startswith(PREFIX)]
    names = tuned_names(base_keys, tuned.keys())
    layer_ids = {int(m.group(1)) for k in base_keys
                 if (m := re.match(re.escape(PREFIX) + r"layers\.(\d+)\.", k))}
    if not layer_ids:
        raise Refused("input", "the base checkpoint has no language_model.layers")
    layers = max(layer_ids) + 1

    scalars = [k for k in base_keys if k.endswith("layer_scalar")]
    worst = 0.0
    for k in scalars:
        if k not in names or tuned.shape(names[k]) != base.shape(k):
            raise Refused("layer_scalar", f"the fine-tune has no {k} of the base's shape")
        diff = float(np.abs(tuned.get(names[k]) - base.get(k)).max(initial=0.0))
        if diff:
            # Nothing writes a layer_scalar into the graph, so a changed one
            # would not reach the file.
            raise Refused("layer_scalar", f"the fine-tune changed {k} (by up to {diff:.6g}); "
                          "HF holds it as a buffer, which the fine-tune must leave as the "
                          "base's")
        worst = max(worst, diff)

    out_dir = work / "graft"
    out_dir.mkdir()
    composed = json.loads(json.dumps(doc))
    sources = list(sources)
    expected, parts, folds, norms_bytes = {}, {}, {}, {}
    placed, skipped, written_hf, originals, patches = {}, [], set(), {}, {}
    sizes = set(spec["norm_sizes"])
    for part in PARTS:
        gi, wi = sections[(part, "TFLiteModel")], sections[(part, "TFLiteWeights")]
        where = f"Google's {part} graph"
        graph = bytearray(Path(doc["section"][gi]["data_path"]).read_bytes())
        original = bytes(graph)
        if part == "text_encoder":
            fb = Flat(graph, where)
            found = norm_constants(fb, fb.ref(0), sizes)
            placed, skipped = place_norms(found, base, layers)
            left = {b for b, g in found.items() if not candidates(g, layers)}
            # The norms are found in subgraph 0; one that only another subgraph
            # reads would keep Google's value.
            for buffer, (si, name) in sorted(composite_constants(fb, fb.ref(0), sizes).items()):
                if buffer not in placed and buffer not in left:
                    raise Refused("norm", f"{where}: subgraph {si}'s {name!r} (buffer "
                                  f"{buffer}) is a norm-sized constant a STABLEHLO_COMPOSITE "
                                  "reads, and no norm was placed on it")
        new_weights = out_dir / os.path.basename(doc["section"][wi]["data_path"])
        expected[part], parts[part], more, mapped, patched = write_part(
            part, spec["bits"][part], graph, doc["section"][wi]["data_path"], new_weights,
            base, tuned, names, where)
        folds.update(more)
        written_hf |= mapped
        if part == "text_encoder":
            fb = Flat(graph, where)
            model = fb.ref(0)
            for buffer, k in placed.items():
                if k not in names:
                    raise Refused("unmapped", f"the fine-tune has no tensor for {k}")
                values = tuned.get(names[k])
                if not np.isfinite(values).all():
                    raise Refused("nonfinite", f"the fine-tune's {names[k]} holds non-finite "
                                  "values")
                value = values.astype("<f4").tobytes()
                pos, count = buffer_data(fb, model, buffer)
                if count != len(value):
                    raise Refused("norm", f"{k} is {len(value)} bytes and its constant {count}")
                graph[pos:pos + count] = value
                norms_bytes[buffer] = value
                patched.append((pos, count))
        originals[part], patches[part] = original, patched
        new_graph = out_dir / os.path.basename(doc["section"][gi]["data_path"])
        new_graph.write_bytes(graph)
        for i, path in ((gi, new_graph), (wi, new_weights)):
            composed["section"][i]["data_path"] = str(path)
            sources[i] = (str(path), (0, path.stat().st_size))

    # A parameter the fine-tune changed that the file holds nowhere would be lost
    # without a word.
    kept = written_hf | set(placed.values()) | set(scalars)
    for k in base_keys:
        if k in kept or k not in names:
            continue
        if tuned.shape(names[k]) != base.shape(k) or not np.array_equal(tuned.get(names[k]),
                                                                        base.get(k)):
            raise Refused("unmapped", f"the fine-tune changed {k}, which the graft writes "
                          "nowhere in Google's file")

    try:
        report = T.rebuild(work, composed, sources, [before_dir])
        after_dir = work / "after"
        after = T.load(after_dir / "model.toml")
    except T.Refused as exc:
        raise Refused("readback", str(exc)) from None
    after_parts = find_parts(after, "the packed file")
    for part in PARTS:
        where = f"the packed {part} graph"
        gi, wi = after_parts[(part, "TFLiteModel")], after_parts[(part, "TFLiteWeights")]
        graph = (after_dir / os.path.basename(after["section"][gi]["data_path"])).read_bytes()
        weights = after_dir / os.path.basename(after["section"][wi]["data_path"])
        parts[part]["types"] = read_back(part, graph, weights, expected[part],
                                         norms_bytes if part == "text_encoder" else {}, where,
                                         originals[part], patches[part])
    rebuilt = report["rebuilt"]
    return {
        "ok": True, "rebuilt": rebuilt, "bytes": os.path.getsize(rebuilt),
        "sha256": sha256(rebuilt), "google_sha256": spec["google_sha256"],
        "parts": parts, "folds": folds, "layers": layers,
        "norms": {"written": len(norms_bytes), "skipped": skipped,
                  "placed": {k: found[b]["name"] for b, k in placed.items()},
                  "cos_to_base_min": min((cosine(found[b]["value"], base.get(k))
                                          for b, k in placed.items()), default=None)},
        "layer_scalar": {"n": len(scalars), "max_abs_diff": worst},
        "tuned_keys": {"matched": len(names), "base": len(base_keys)},
        "sections": report["sections"],
    }


def main(spec):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import towers_script as T  # litetune's towers script, written beside this one

    builder = importlib.metadata.version("litert-lm-builder")
    try:
        result = graft(spec, T)
    except Refused as exc:
        result = {"ok": False, "cause": exc.cause, "reason": str(exc)}
    return dict(result, builder=builder)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)), encoding="utf-8")
'''


class EmbeddingConvertError(Exception):
    """A graft this module will not write, and its cause: `base_sha256`,
    `unmapped`, `norm`, `layer_scalar`, `nonfinite`, `field`, `readback`, `input`
    or `run`."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True)
class GraftResult:
    """The file written and what the script recorded writing it."""

    output: Path
    weights: WeightMode
    bytes: int
    sha256: str
    # The litert-lm-builder version that packed it.
    builder: str
    # Per part (`embedder`, `text_encoder`): bits, tensors, bytes, types read
    # back, and the cosine of each dequantized tensor to the values it was made
    # from and to Google's.
    parts: dict[str, dict[str, Any]]
    # HF parameter -> the factor folded into it (1/512 for per_layer_model_projection).
    folds: dict[str, float]
    norms_written: int
    norms_skipped: tuple[str, ...]
    layer_scalars: int
    report: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "weights": self.weights,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "builder": self.builder,
            "parts": self.parts,
            "folds": self.folds,
            "norms_written": self.norms_written,
            "norms_skipped": list(self.norms_skipped),
            "layer_scalars": self.layer_scalars,
        }


def graft(
    env: envs.StageEnv,
    google_file: Path,
    base_safetensors: Path,
    tuned_safetensors: Path,
    out_path: Path,
    weights: WeightMode = DEFAULT_WEIGHTS,
    *,
    events: EventStream | None = None,
    auto_provision: bool = True,
    timeout: int = GRAFT_TIMEOUT_S,
) -> GraftResult:
    """Write `out_path`: Google's file with the fine-tune's text tower in it.

    `base_safetensors` is the checkpoint the fine-tune started from; the mapping
    and fold factors are read against it, the values written are the fine-tune's.
    Raises `EmbeddingConvertError` for a graft it refuses, `FileNotFoundError`
    for a missing input.
    """
    if weights not in WEIGHT_BITS:
        raise EmbeddingConvertError(
            "input", f"weights must be one of {sorted(WEIGHT_BITS)}, not {weights!r}"
        )
    inputs = [Path(p).absolute() for p in (google_file, base_safetensors, tuned_safetensors)]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"no file at {path}")
    output = Path(out_path).absolute()
    # `lexists`: a symlink at `output`, dangling or not, is a name already taken.
    if os.path.lexists(output):
        raise EmbeddingConvertError(
            "input", f"{output} already exists; name a new file -- nothing is written over one"
        )
    if auto_provision:
        env.provision(events=events)
    if not env.ready:
        raise EmbeddingConvertError("run", f"environment {env.name!r} {env.unready_detail}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Beside the output: the result is hard-linked into place, and `os.link`
    # needs both names on one filesystem. It holds the unpacked file, the new
    # sections, the packed file and its unpacked read-back.
    work = Path(tempfile.mkdtemp(prefix=f".{output.stem}-graft-", dir=output.parent))
    try:
        (work / "towers_script.py").write_text(_TOWERS_SCRIPT, encoding="utf-8")
        script = work / "embedding_graft.py"
        script.write_text(_GRAFT_SCRIPT, encoding="utf-8")
        spec_path, result_path = work / "spec.json", work / "result.json"
        google, base, tuned = inputs
        spec_path.write_text(
            json.dumps(
                {
                    "google": str(google),
                    "google_sha256": GOOGLE_SHA256,
                    "base": str(base),
                    "tuned": str(tuned),
                    "work": str(work),
                    "bits": WEIGHT_BITS[weights],
                    "norm_sizes": list(NORM_SIZES),
                }
            ),
            encoding="utf-8",
        )
        if events:
            events.note(f"embedding graft: {weights} ({tuned.name} into {google.name})")
        try:
            proc = env.run(
                ["python", str(script), str(spec_path), str(result_path)], timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise EmbeddingConvertError(
                "run", f"the graft did not finish within {timeout}s"
            ) from None
        if proc.returncode != 0:
            stderr = proc.stderr or ""
            last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
            logger.error("embedding graft into %s failed:\n%s", output, stderr[-2000:])
            reading = read_returncode(proc.returncode)
            raise EmbeddingConvertError(
                "run", f"the graft script {reading.describe('the graft')}: {last}"
            )
        if not result_path.is_file():
            raise EmbeddingConvertError("run", "the graft script exited 0 without its report")
        report: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
        if not report.get("ok"):
            raise EmbeddingConvertError(
                str(report.get("cause") or "run"),
                f"{report.get('reason') or 'the graft wrote nothing'}; nothing was written",
            )
        digest = _publish(Path(report["rebuilt"]), output, report["sha256"])
    finally:
        leaked: list[str] = []
        shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
        if leaked:
            logger.warning("%d scratch file(s) left under %s", len(leaked), work)

    norms = report["norms"]
    return GraftResult(
        output=output,
        weights=weights,
        bytes=output.stat().st_size,
        sha256=digest,
        builder=report["builder"],
        parts=report["parts"],
        folds=report["folds"],
        norms_written=norms["written"],
        norms_skipped=tuple(norms["skipped"]),
        layer_scalars=report["layer_scalar"]["n"],
        report=report,
    )


def _publish(rebuilt: Path, output: Path, packed_sha256: str) -> str:
    """Put the packed file at `output` and return its digest, once it hashes to
    the `packed_sha256` the script read back -- checked before the name exists,
    so a mismatch leaves nothing at `output`. Never over a file already there: a
    hard link fails if the name exists, and the file appears whole (as `towers`)."""
    digest = hash_file(rebuilt)
    if digest != f"sha256:{packed_sha256}":
        raise EmbeddingConvertError(
            "readback", f"{rebuilt.name} hashes to {digest}, not the {packed_sha256} packed"
        )
    try:
        os.link(rebuilt, output)
    except FileExistsError:
        raise EmbeddingConvertError(
            "input", f"{output} appeared while the graft ran; nothing was written over it"
        ) from None
    except OSError as exc:
        raise EmbeddingConvertError(
            "run", f"could not link the packed file to {output} ({exc})"
        ) from None
    return digest
