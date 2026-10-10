"""`towers`: a bundle's vision and audio sections dropped, or taken from another.

The script under test is the real one, run by this interpreter. Only
`litert_lm_builder` is replaced, by a fake that behaves like 0.18.0's in each
place the script could go wrong unnoticed:

  - a bundle is laid out as `_build_sections` lays it out: the header's end at
    byte 24, the header from byte 32, each section at the next 16 KiB boundary
    -- here the header is JSON rather than a flatbuffer -- and
    `litertlm_peek.read_litertlm_header` and `unpack` read it the way the real
    ones do, trusting every offset, so the script's own checks of them are what
    a test sees;
  - `unpack` names each file `SectionN_...` by its index, writes LlmMetadata as
    text and drops from it every field the proto does not declare (a line
    holding `unknown_field` here, `gemma4` field 13 on Google's bundle), and
    leaves out of `model.toml` the sections it has no type for;
  - `pack` maps every `model_type` through `TfLiteModelType`, raising for one it
    does not have, opens every `data_path` as the real `_resolve_path` resolves
    it, and drops and appends again the system `uuid` and `creation_timestamp`;
  - the package is installed as `litert-lm-builder` 0.18.0, as far as
    `importlib.metadata` can tell.

The TOMLs are the ones `unpack` wrote for Google's Gemma 4 E2B CPU/GPU bundle
(`litert-community/gemma-4-E2B-it-litert-lm` at b3ca0d2f) and for a Gemma 4 E2B
build for SM8850, sections only and with its uuid and timestamp replaced. The
graphs, tokenizers and metadata are made up, small, and only as real as the
script's reading of them needs.
"""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import logging
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned

from litetune import envs
from litetune.cli import main
from litetune.towers import (
    BUILDER_TYPES,
    TOWER_SECTIONS,
    TowersError,
    add_towers,
    drop_towers,
)

# tomllib on 3.11+, tomli below -- the fallback the towers script itself uses.
try:
    tomllib: Any = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = importlib.import_module("tomli")

GEMMA4_E2B_TOML = """[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "The ODML Authors" },
  { key = "uuid", value_type = "String", value = "2fa073f5-2d5e-44ff-8bb9-64d926dc40e2" },
  { key = "creation_timestamp", value_type = "String", value = "2026-04-28T22:06:55.560103+00:00" },
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
model_type = "per_layer_embedder"
section_type = "TFLiteModel"
data_path = "Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite"

[[section]]
model_type = "audio_encoder_hw"
backend_constraint = "cpu"
section_type = "TFLiteModel"
data_path = "Section4_TFLiteModel_tf_lite_audio_encoder_hw.tflite"

[[section]]
model_type = "audio_adapter"
backend_constraint = "cpu"
section_type = "TFLiteModel"
data_path = "Section5_TFLiteModel_tf_lite_audio_adapter.tflite"

[[section]]
model_type = "end_of_audio"
section_type = "TFLiteModel"
data_path = "Section6_TFLiteModel_tf_lite_end_of_audio.tflite"

[[section]]
additional_metadata = [
  { key = "prefer_activation_type", value_type = "String", value = "fp16" },
]
model_type = "vision_encoder"
section_type = "TFLiteModel"
data_path = "Section7_TFLiteModel_tf_lite_vision_encoder.tflite"

[[section]]
model_type = "vision_adapter"
backend_constraint = "cpu"
section_type = "TFLiteModel"
data_path = "Section8_TFLiteModel_tf_lite_vision_adapter.tflite"

[[section]]
model_type = "end_of_vision"
section_type = "TFLiteModel"
data_path = "Section9_TFLiteModel_tf_lite_end_of_vision.tflite"

[[section]]
additional_metadata = [
  { key = "prefer_activation_type", value_type = "String", value = "fp16" },
]
model_type = "prefill_decode"
section_type = "TFLiteModel"
data_path = "Section10_TFLiteModel_tf_lite_prefill_decode.tflite"

[[section]]
model_type = "mtp_drafter"
section_type = "TFLiteModel"
data_path = "Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
"""

SM8850_TOML = """[system_metadata]
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
model_type = "aux"
section_type = "TFLiteModel"
data_path = "Section2_TFLiteModel_tf_lite_aux.tflite"

[[section]]
model_type = "embedder"
section_type = "TFLiteModel"
data_path = "Section3_TFLiteModel_tf_lite_embedder.tflite"

[[section]]
model_type = "per_layer_embedder"
section_type = "TFLiteModel"
data_path = "Section4_TFLiteModel_tf_lite_per_layer_embedder.tflite"

[[section]]
model_type = "prefill_decode"
backend_constraint = "npu"
section_type = "TFLiteModel"
data_path = "Section5_TFLiteModel_tf_lite_prefill_decode.tflite"
"""

# The fake's bundle layout. The fake builder, the fake peek, the fake
# `litertlm_core` and the helpers below all read and write it through this text.
FAKE_FORMAT = r'''
import json
import struct
from pathlib import Path

# `litertlm_core.py:28-31` in 0.18.0.
HEADER_MAGIC_BYTES = b"LITERTLM"
BLOCK_SIZE = 16 * 1024
HEADER_BEGIN_BYTE_OFFSET = 32
HEADER_END_LOCATION_BYTE_OFFSET = 24


def read(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == HEADER_MAGIC_BYTES, "not a bundle"
    (end,) = struct.unpack_from("<Q", raw, HEADER_END_LOCATION_BYTE_OFFSET)
    return json.loads(raw[HEADER_BEGIN_BYTE_OFFSET:end]), raw


def _up(offset):
    return -(-offset // BLOCK_SIZE) * BLOCK_SIZE


def write(path, toml, contents, listed=None):
    """As `_build_sections` lays a file out (litertlm_builder.py:1323-1333): the
    header, then each section at the next block boundary. `listed` replaces the
    offsets the header gives, for a test whose header lies."""
    spans = []
    for _ in range(4):  # the header's length depends on the offsets it holds
        header = json.dumps({"toml": toml, "sections": listed or spans}).encode()
        at, placed = _up(HEADER_BEGIN_BYTE_OFFSET + len(header)), []
        for data in contents:
            placed.append([at, at + len(data)])
            at = _up(at + len(data))
        if placed == spans:
            break
        spans = placed
    else:
        raise AssertionError("the header's length did not settle")
    out = bytearray(HEADER_MAGIC_BYTES + struct.pack("<III", 1, 0, 0))
    out += bytes(HEADER_END_LOCATION_BYTE_OFFSET - len(out))
    out += struct.pack("<Q", HEADER_BEGIN_BYTE_OFFSET + len(header))
    out += bytes(HEADER_BEGIN_BYTE_OFFSET - len(out)) + header
    for (begin, _), data in zip(spans, contents):
        out += bytes(begin - len(out)) + data
    Path(path).write_bytes(bytes(out))
'''

# The fake builder. Its settings are prepended as plain assignments, so this
# text carries no format placeholders.
FAKE_BUILDER = (
    FAKE_FORMAT
    + r"""
import enum
import os
import re
import uuid as _uuid
from datetime import datetime, timezone

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

TfLiteModelType = enum.Enum(
    "TfLiteModelType", {t.upper(): "tf_lite_" + t for t in KNOWN_TYPES}
)


def _from_free_value(cls, value):
    name = value.lower()
    return cls(name if name.startswith("tf_lite_") else "tf_lite_" + name)


TfLiteModelType.get_enum_from_tf_free_value = classmethod(_from_free_value)


def unpack(litertlm_path, output_dir, jinja_prompt_template_path=None):
    header, raw = read(litertlm_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = tomllib.loads(header["toml"])
    names = []
    for i, (section, (b, e)) in enumerate(zip(doc.get("section", []), header["sections"])):
        content = raw[b:e]
        if section.get("section_type") == "LlmMetadata":
            # Through the builder's proto: a field it does not declare is lost.
            lines = content.decode("utf-8").split("\n")
            (out / "LlmMetadataProto.pbtext").write_text(
                "\n".join(l for l in lines if "unknown_field" not in l), encoding="utf-8")
            names.append("LlmMetadataProto.pbtext")
        else:
            name = "Section%d_%s" % (i, re.sub(r"^Section\d+_", "",
                                               os.path.basename(section["data_path"])))
            (out / name).write_bytes(content)
            names.append(name)
    it = iter(names)
    toml = re.sub(r'(data_path\s*=\s*")([^"]+)(")',
                  lambda m: m.group(1) + next(it) + m.group(3), header["toml"])
    (out / "model.toml").write_text(toml, encoding="utf-8")
    if SHRINK_INPUT is not None and "rebuilt" not in str(litertlm_path):
        # The input changes after it was read: cut to its first SHRINK_INPUT bytes.
        Path(litertlm_path).write_bytes(raw[:SHRINK_INPUT])
    return str(out / "model.toml")


def _value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(v)


def _entry(e):
    return "  { " + ", ".join(f"{k} = {_value(v)}" for k, v in e.items()) + " },"


def _toml(doc):
    lines = ["[system_metadata]", "entries = ["]
    lines += [_entry(e) for e in doc["system_metadata"]["entries"]]
    lines.append("]")
    for s in doc["section"]:
        lines += ["", "[[section]]"]
        for k, v in s.items():
            if k == "additional_metadata":
                lines += ["additional_metadata = ["] + [_entry(e) for e in v] + ["]"]
            else:
                lines.append(f"{k} = {_value(v)}")
    return "\n".join(lines) + "\n"


def pack(toml_path, output_path, jinja_prompt_template_path=None):
    if PACK_FAILS:
        raise RuntimeError("pack: boom")
    toml_path = Path(toml_path)
    doc = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    contents = []
    for s in doc.get("section", []):
        if s.get("section_type") in ("TFLiteModel", "TFLiteWeights"):
            TfLiteModelType.get_enum_from_tf_free_value(s["model_type"])
        path = Path(s["data_path"])
        data = (path if path.is_absolute() else toml_path.parent / path).read_bytes()
        if PACK_ALTERS_BYTES and s.get("model_type") == "prefill_decode":
            data += b"!"
        contents.append(data)
        s["data_path"] = os.path.basename(s["data_path"])
        if PACK_ADDS_KEY and s.get("model_type") == "embedder":
            s["extra"] = "x"
    # `populate_system_metadata`: both regenerated, appended at the end.
    entries = [e for e in doc["system_metadata"]["entries"]
               if e["key"] not in ("uuid", "creation_timestamp")]
    entries += [{"key": "uuid", "value_type": "String", "value": str(_uuid.uuid4())},
                {"key": "creation_timestamp", "value_type": "String",
                 "value": datetime.now(timezone.utc).isoformat()}]
    doc["system_metadata"]["entries"] = entries
    write(output_path, _toml(doc), contents)
    return str(output_path)
"""
)

FAKE_PEEK = (
    FAKE_FORMAT
    + r"""
class _Object:
    def __init__(self, span):
        self.span = span

    def BeginOffset(self):
        return self.span[0]

    def EndOffset(self):
        return self.span[1]


class _Listed:
    def __init__(self, spans):
        self.spans = spans

    def ObjectsLength(self):
        return len(self.spans)

    def Objects(self, i):
        return _Object(self.spans[i])


class _Header:
    def __init__(self, spans):
        self.spans = spans

    def SectionMetadata(self):
        return _Listed(self.spans)


def read_litertlm_header(file_path, output_stream):
    header, _ = read(file_path)
    return _Header(header["sections"])
"""
)

# The same layout, for the tests to build and read bundles with.
FORMAT: dict[str, Any] = {}
exec(FAKE_FORMAT, FORMAT)
BLOCK_SIZE = FORMAT["BLOCK_SIZE"]


def _section_names(toml: str) -> list[str]:
    return [
        re.sub(r"^Section\d+_", "", os.path.basename(p))
        for p in re.findall(r'data_path\s*=\s*"([^"]+)"', toml)
    ]


def make_bundle(
    path: Path, toml: str, files: dict[str, bytes] | None = None, hidden: int = 0
) -> Path:
    """A bundle in the fake's format: each section's bytes are its own name unless
    `files` gives others (by name, without the `SectionN_` index). `hidden`
    sections are in the header and not in the TOML, as TTS or ASR metadata is."""
    contents = [(files or {}).get(n, f"bytes of {n}".encode()) for n in _section_names(toml)]
    FORMAT["write"](path, toml, contents + [b"hidden metadata"] * hidden)
    return path


def read_bundle(path: Path) -> tuple[dict, list[bytes]]:
    """The bundle's TOML, parsed, and each listed section's raw bytes."""
    header, raw = FORMAT["read"](path)
    return tomllib.loads(header["toml"]), [raw[b:e] for b, e in header["sections"]]


def offsets(path: Path) -> list[list[int]]:
    header, _ = FORMAT["read"](path)
    return [list(span) for span in header["sections"]]


def rewrite_header(path: Path, listed: list[list[int]]) -> None:
    """`path` again, its sections where they were laid out and its header naming
    `listed` as their offsets."""
    header, raw = FORMAT["read"](path)
    contents = [raw[b:e] for b, e in header["sections"]]
    FORMAT["write"](path, header["toml"], contents, listed)


def section(path: Path, section_type: str, model_type: str | None = None) -> bytes:
    doc, contents = read_bundle(path)
    found = [
        c
        for s, c in zip(doc["section"], contents, strict=False)
        if s["section_type"] == section_type
        and (model_type is None or s.get("model_type") == model_type)
    ]
    assert len(found) == 1, (section_type, model_type, len(found))
    return found[0]


def types(path: Path) -> list[str]:
    doc, _ = read_bundle(path)
    return [s.get("model_type") for s in doc["section"] if "model_type" in s]


@dataclass
class FakeToolchain:
    """`StageEnv.run`: the towers script, run for real against the fake builder."""

    pack_fails: bool = False
    pack_adds_key: bool = False
    pack_alters_bytes: bool = False
    known_types: list[str] = field(default_factory=lambda: list(BUILDER_TYPES))
    builder_version: str = "0.18.0"
    # `unpack` cuts its input to this many bytes once it has read it.
    shrink_input: int | None = None
    # Something writes this path while the rebuild runs.
    output_appears: Path | None = None
    times_out: bool = False
    # The child's return code, in place of running it.
    returncode: int | None = None
    calls: list[list[str]] = field(default_factory=list)
    timeouts: list[int] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        self.timeouts.append(timeout)
        assert args[0] == "python" and str(args[1]).endswith("towers.py"), args
        if self.times_out:
            raise subprocess.TimeoutExpired(args, timeout)
        if self.returncode is not None:
            return subprocess.CompletedProcess(args, self.returncode, "", "")
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim"
        package = shim / "litert_lm_builder"
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("")
        settings = (
            f"KNOWN_TYPES = {self.known_types!r}\n"
            f"PACK_FAILS = {self.pack_fails}\n"
            f"PACK_ADDS_KEY = {self.pack_adds_key}\n"
            f"PACK_ALTERS_BYTES = {self.pack_alters_bytes}\n"
            f"SHRINK_INPUT = {self.shrink_input!r}\n"
        )
        (package / "litertlm_builder.py").write_text(settings + FAKE_BUILDER, encoding="utf-8")
        (package / "litertlm_peek.py").write_text(FAKE_PEEK, encoding="utf-8")
        (package / "litertlm_core.py").write_text(FAKE_FORMAT, encoding="utf-8")
        dist = shim / f"litert_lm_builder-{self.builder_version}.dist-info"
        dist.mkdir(exist_ok=True)
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: litert-lm-builder\nVersion: {self.builder_version}\n"
        )
        proc = subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim)},
        )
        if self.output_appears is not None:
            self.output_appears.write_bytes(b"someone else's")
        return proc


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


def no_scratch(directory: Path) -> bool:
    return not [p for p in directory.iterdir() if p.name.startswith(".")]


def builder_with(monkeypatch, anchor: str, extra: str) -> None:
    """The fake builder with `extra` after the one line `anchor`."""
    assert FAKE_BUILDER.count(anchor) == 1, anchor
    monkeypatch.setitem(globals(), "FAKE_BUILDER", FAKE_BUILDER.replace(anchor, anchor + extra))


# -- graphs, tokenizers and metadata ---------------------------------------------


def flatbuffer(root: tuple[str, Any]) -> bytes:
    """A flatbuffer, written front to back: the root offset, the file identifier,
    then each table's vtable, the table, and what its fields point at. A node is
    ("table", [field or None, ...]) with fields in schema order, ("tables", [node,
    ...]), ("ints", [...]), ("string", text) or ("uint", n)."""
    out = bytearray(4) + b"TFL3"

    def align() -> None:
        out.extend(bytes(-len(out) % 4))

    def emit(node: tuple[str, Any]) -> int:
        kind, value = node
        align()
        start = len(out)
        if kind == "table":
            out.extend(struct.pack("<HH", 4 + 2 * len(value), 4 + 4 * len(value)))
            for i, item in enumerate(value):
                out.extend(struct.pack("<H", 0 if item is None else 4 + 4 * i))
            align()
            table = len(out)
            out.extend(struct.pack("<i", table - start))
            later = []
            for item in value:
                if item is not None and item[0] == "uint":
                    out.extend(struct.pack("<I", item[1]))
                    continue
                if item is not None:
                    later.append((len(out), item))
                out.extend(bytes(4))
            for pos, item in later:
                struct.pack_into("<I", out, pos, emit(item) - pos)
            return table
        if kind == "string":
            data = value.encode()
            out.extend(struct.pack("<I", len(data)) + data + b"\0")
        elif kind == "ints":
            out.extend(struct.pack(f"<I{len(value)}i", len(value), *value))
        else:
            out.extend(struct.pack("<I", len(value)) + bytes(4 * len(value)))
            for i, child in enumerate(value):
                slot = start + 4 + 4 * i
                struct.pack_into("<I", out, slot, emit(child) - slot)
        return start

    struct.pack_into("<I", out, 0, emit(root))
    return bytes(out)


def tflite(*signatures: tuple[str, list[list[int]]]) -> bytes:
    """A TFLite model holding only what the width check reads: per signature, a
    subgraph whose tensors are its outputs, and the signature mapping each output
    to its tensor. Fields in TFLite's schema order: Model.subgraphs is the third
    and signature_defs the eighth; SignatureDef.outputs the second, signature_key
    the third, subgraph_index the fifth; TensorMap.tensor_index the second."""
    subgraphs, defs = [], []
    for index, (key, shapes) in enumerate(signatures):
        tensors = [("table", [("ints", shape)]) for shape in shapes]
        subgraphs.append(("table", [("tables", tensors)]))
        outputs = [("table", [("string", f"output_{i}"), ("uint", i)]) for i in range(len(shapes))]
        defs.append(("table", [None, ("tables", outputs), ("string", key), None, ("uint", index)]))
    model = [None, None, ("tables", subgraphs), None, None, None, None, ("tables", defs)]
    return flatbuffer(("table", model))


# The values per token every made-up graph below writes. The signature names
# are made up too, but for `decode_embedder`, which is the one LiteRT-LM's NPU
# embedder runs.
WIDTH = 16


def embedder(width: int = WIDTH, decode: int | None = None) -> bytes:
    return tflite(("prefill", [[1, 8, width]]), ("decode_embedder", [[1, 1, decode or width]]))


def vision_adapter(width: int = WIDTH, second: int | None = None) -> bytes:
    return tflite(("patches_a", [[1, 280, width]]), ("patches_b", [[1, 70, second or width]]))


def audio_adapter(width: int = WIDTH) -> bytes:
    return tflite(("audio", [[1, 750, width]]))


def graph_file(model_type: str) -> str:
    return f"TFLiteModel_tf_lite_{model_type}.tflite"


PIECES = ["<pad>", "<eos>", "<bos>", "a", "b", "<|image>", "<|audio>", "<|video|>"]

# SentencePiece's piece types (`ModelProto.SentencePiece.Type`).
NORMAL, CONTROL, USER_DEFINED = 1, 3, 4

# Gemma 4 metadata in the text form `unpack` writes, cut to some of the fields
# that name tokens and to two media settings. The last line stands for a field
# the builder's proto does not declare: `unpack` loses it, the raw copy must not.
GEMMA4_META = """start_token {
  token_ids {
    ids: 2
  }
}
stop_tokens {
  token_ids {
    ids: 1
  }
}
llm_model_type {
  gemma4 {
    start_of_image_token {
      token_str: "<|image>"
    }
    start_of_audio_token {
      token_str: "<|audio>"
    }
    patch_width: 16
    max_num_patches: 2520
  }
}
unknown_field: 1
"""

# A text-only build's: ids, an old-style prompt template, no model type.
BARE_META = """start_token {
  token_ids {
    ids: 2
  }
}
stop_tokens {
  token_ids {
    ids: 1
  }
}
prompt_templates {
  user {
    prefix: "<|turn>user\\n"
  }
}
"""


def sentencepiece(
    pieces: list[str], kinds: dict[str, int] | None = None, typed: bool = True
) -> bytes:
    """A SentencePiece ModelProto: field 1 per piece (piece, score and, when
    `typed`, its type -- NORMAL unless `kinds` says otherwise), then an empty
    trainer spec, as the real file carries one. Untyped, a piece ends on its
    score, as most of Gemma's do."""

    def varint(n: int) -> bytes:
        out = b""
        while True:
            byte = n & 0x7F
            n >>= 7
            out += bytes([byte | (0x80 if n else 0)])
            if not n:
                return out

    def field_(number: int, wire: int, payload: bytes) -> bytes:
        key = varint(number << 3 | wire)
        return key + (varint(len(payload)) + payload if wire == 2 else payload)

    body = b""
    for text in pieces:
        piece = field_(1, 2, text.encode()) + field_(2, 5, b"\0\0\0\0")
        if typed:
            piece += field_(3, 0, varint((kinds or {}).get(text, NORMAL)))
        body += field_(1, 2, piece)
    return body + field_(2, 2, b"")


def bundle(
    path: Path,
    toml: str,
    meta: str = GEMMA4_META,
    pieces: list[str] | None = None,
    hidden: int = 0,
    tokenizer: bytes | None = None,
    graphs: dict[str, bytes] | None = None,
) -> Path:
    files = {
        "LlmMetadataProto.pbtext": meta.encode(),
        "SP_Tokenizer.spiece": tokenizer or sentencepiece(PIECES if pieces is None else pieces),
        graph_file("embedder"): embedder(),
        graph_file("vision_adapter"): vision_adapter(),
        graph_file("audio_adapter"): audio_adapter(),
    }
    files.update({graph_file(t): data for t, data in (graphs or {}).items()})
    return make_bundle(path, toml, files, hidden=hidden)


def gemma4(tmp_path: Path, toml: str = GEMMA4_E2B_TOML, **kwargs) -> Path:
    return bundle(tmp_path / "gemma4.litertlm", toml, **kwargs)


def sm8850(
    tmp_path: Path, meta: str = BARE_META, pieces: list[str] | None = None, **kwargs
) -> Path:
    # The SM8850 tokenizer lacks `<|video|>`.
    return bundle(tmp_path / "sm8850.litertlm", SM8850_TOML, meta, pieces or PIECES[:-1], **kwargs)


def donor(tmp_path: Path, toml: str = GEMMA4_E2B_TOML, **kwargs) -> Path:
    return bundle(tmp_path / "donor.litertlm", toml, **kwargs)


def without(toml: str, *model_types: str) -> str:
    """`toml` without the sections of `model_types`."""
    blocks = toml.split("\n\n[[section]]")
    return "\n\n[[section]]".join(
        b for b in blocks if not any(f'model_type = "{t}"' in b for t in model_types)
    )


# -- dropping ------------------------------------------------------------------------


def test_both_towers_go_and_every_other_section_is_the_inputs_bytes(toolchain, tmp_path):
    model = gemma4(tmp_path)
    before = model.read_bytes()
    out = tmp_path / "text.litertlm"

    result = drop_towers(model, out, ["vision", "audio"])

    assert types(out) == ["embedder", "per_layer_embedder", "prefill_decode", "mtp_drafter"]
    kept = [("LlmMetadata", None), ("SP_Tokenizer", None), ("TFLiteModel", "embedder")]
    kept += [("TFLiteModel", "prefill_decode"), ("TFLiteModel", "mtp_drafter")]
    for s_type, m_type in kept:
        assert section(out, s_type, m_type) == section(model, s_type, m_type)
    assert model.read_bytes() == before, "the input is never touched"
    assert all(b % BLOCK_SIZE == 0 for b, _ in offsets(out)), "laid out like the real builder"
    assert [d["model_type"] for d in result.dropped] == [
        "audio_encoder_hw", "audio_adapter", "end_of_audio",
        "vision_encoder", "vision_adapter", "end_of_vision",
    ]  # fmt: skip
    assert [s["model_type"] for s in result.sections if s["model_type"]] == types(out)
    assert result.bytes_before == model.stat().st_size
    assert result.bytes_after == out.stat().st_size
    assert result.sha256 == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest()
    assert result.builder == "0.18.0"
    assert no_scratch(tmp_path)


def test_the_bytes_dropped_are_the_sections_lengths(toolchain, tmp_path):
    model = gemma4(tmp_path)
    result = drop_towers(model, tmp_path / "o", ["vision"])
    want = sum(len(section(model, "TFLiteModel", t))
               for t in ("vision_encoder", "vision_adapter", "end_of_vision"))  # fmt: skip
    assert sum(d["bytes"] for d in result.dropped) == want


def test_the_metadata_keeps_a_field_the_builders_proto_does_not_know(toolchain, tmp_path):
    """`unpack` writes LlmMetadata as text and loses such a field (Google's E2B
    bundle sets `gemma4` field 13, reserved in 0.18.0's proto); a repack from that
    text would write 2 bytes fewer and still read back the same."""
    out = tmp_path / "text.litertlm"
    drop_towers(gemma4(tmp_path), out, ["vision"])

    assert b"unknown_field: 1" in section(out, "LlmMetadata")


def test_one_tower_goes_and_the_other_stays(toolchain, tmp_path):
    out = tmp_path / "no-vision.litertlm"
    drop_towers(gemma4(tmp_path), out, ["vision"])

    assert not {"vision_encoder", "vision_adapter", "end_of_vision"} & set(types(out))
    assert {"audio_encoder_hw", "audio_adapter", "end_of_audio"} <= set(types(out))


@pytest.mark.parametrize("stored", ["tf_lite_end_of_vision", "TF_LITE_END_OF_VISION"])
def test_a_type_stored_with_the_prefix_or_in_capitals_is_its_tower(toolchain, tmp_path, stored):
    """0.18.0's `unpack` strips `tf_lite_` from the stored value; a TOML that keeps
    it, or names it in capitals, still names the same type."""
    toml = GEMMA4_E2B_TOML.replace('model_type = "end_of_vision"', f'model_type = "{stored}"')
    out = tmp_path / "out.litertlm"
    result = drop_towers(gemma4(tmp_path, toml), out, ["vision"])

    assert stored not in types(out)
    assert len(result.dropped) == 3


def test_a_type_the_builder_cannot_write_back_is_refused_not_guessed(toolchain, tmp_path):
    toml = GEMMA4_E2B_TOML.replace('model_type = "mtp_drafter"', 'model_type = "vision_new"')
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="'vision_new', which litert-lm-builder cannot write"):
        drop_towers(gemma4(tmp_path, toml), out, ["vision"])
    assert not out.exists()


def test_a_section_unpack_leaves_out_is_refused_not_lost(toolchain, tmp_path):
    out = tmp_path / "out.litertlm"
    with pytest.raises(
        TowersError,
        match=r"has 13 sections and litert-lm-builder unpacks 12 .* would be lost "
        r"\(sections `unpack` leaves out: TTS or ASR metadata, NONE, Deprecated\)",
    ):
        drop_towers(gemma4(tmp_path, hidden=1), out, ["vision"])
    assert not out.exists()


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        ("stray = 1\n", "stray"),
        ("", "system_metadata.other"),
    ],
)
def test_a_toml_key_the_script_would_not_write_back_is_refused(toolchain, tmp_path, extra, key):
    toml = extra + GEMMA4_E2B_TOML
    if key == "system_metadata.other":
        toml = toml.replace("entries = [", "other = 1\nentries = [", 1)
    with pytest.raises(TowersError, match=f"TOML for the bundle has {re.escape(key)}, which"):
        drop_towers(gemma4(tmp_path, toml), tmp_path / "o", ["vision"])


WEIGHTS = """
[[section]]
model_type = "vision_encoder"
section_type = "TFLiteWeights"
data_path = "Section12_TFLiteWeights_tf_lite_vision_encoder.bin"
"""


def test_a_towers_externalized_weights_go_with_it(toolchain, tmp_path):
    out = tmp_path / "out.litertlm"
    result = drop_towers(gemma4(tmp_path, GEMMA4_E2B_TOML + WEIGHTS), out, ["vision"])

    doc, _ = read_bundle(out)
    assert "TFLiteWeights" not in {s["section_type"] for s in doc["section"]}
    assert "TFLiteWeights" in {d["section_type"] for d in result.dropped}


def test_a_section_is_chosen_by_its_type_and_section_type_together(toolchain, tmp_path):
    toml = GEMMA4_E2B_TOML.replace(
        'section_type = "SP_Tokenizer"',
        'model_type = "vision_encoder"\nsection_type = "SP_Tokenizer"',
    )
    out = tmp_path / "out.litertlm"
    drop_towers(gemma4(tmp_path, toml), out, ["vision"])

    assert section(out, "SP_Tokenizer") == sentencepiece(PIECES)


@pytest.mark.parametrize("towers", [["vision"], ["vision", "audio"], ["audio", "vision"]])
def test_a_tower_the_bundle_does_not_carry_is_refused_and_nothing_written(
    toolchain, tmp_path, towers
):
    model = gemma4(tmp_path)
    no_vision = tmp_path / "no-vision.litertlm"
    drop_towers(model, no_vision, ["vision"])
    out = tmp_path / "out.litertlm"

    with pytest.raises(TowersError, match="carries no vision section"):
        drop_towers(no_vision, out, towers)
    assert not out.exists()


def test_an_existing_output_is_refused_before_anything_runs(toolchain, tmp_path):
    model = gemma4(tmp_path)
    with pytest.raises(TowersError, match="already exists"):
        drop_towers(model, model, ["vision"])
    assert toolchain.calls == []


def test_a_dangling_symlink_at_the_output_is_refused_before_anything_runs(toolchain, tmp_path):
    out = tmp_path / "out.litertlm"
    out.symlink_to(tmp_path / "nowhere")
    with pytest.raises(TowersError, match="already exists"):
        drop_towers(gemma4(tmp_path), out, ["vision"])
    assert toolchain.calls == []


def test_a_missing_output_directory_is_made(toolchain, tmp_path):
    out = tmp_path / "new" / "dir" / "o.litertlm"
    drop_towers(gemma4(tmp_path), out, ["vision"])
    assert out.is_file()


def test_a_missing_model_is_refused_before_anything_runs(toolchain, tmp_path):
    with pytest.raises(FileNotFoundError, match="no bundle at"):
        drop_towers(tmp_path / "missing.litertlm", tmp_path / "o", ["vision"])
    assert toolchain.calls == []


def test_the_work_directory_is_beside_the_output(toolchain, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    drop_towers(gemma4(tmp_path), sub / "o.litertlm", ["vision"])
    assert Path(toolchain.calls[0][2]).parent.parent == sub


def test_an_output_written_during_the_rebuild_is_not_overwritten(toolchain, tmp_path):
    out = tmp_path / "out.litertlm"
    toolchain.output_appears = out
    with pytest.raises(TowersError, match="appeared while the bundle was rebuilt"):
        drop_towers(gemma4(tmp_path), out, ["vision"])
    assert out.read_bytes() == b"someone else's"
    assert no_scratch(tmp_path)


def test_a_filesystem_without_hard_links_is_refused_before_the_rebuild(
    toolchain, tmp_path, monkeypatch
):
    def no_link(src, dst):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr("litetune.towers.os.link", no_link)
    out = tmp_path / "out.litertlm"
    with pytest.raises(
        TowersError, match="nothing was rebuilt .*Operation not permitted.*hard links"
    ):
        drop_towers(gemma4(tmp_path), out, ["vision"])
    assert toolchain.calls == []
    assert not out.exists()
    assert no_scratch(tmp_path)


def test_a_link_that_fails_for_another_reason_gives_no_filesystem_advice(
    toolchain, tmp_path, monkeypatch
):
    real = os.link

    def link(src, dst):
        if Path(dst).name == "out.litertlm":
            raise OSError(errno.ENOSPC, "No space left on device")
        real(src, dst)

    monkeypatch.setattr("litetune.towers.os.link", link)
    with pytest.raises(TowersError, match="No space left on device") as raised:
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["vision"])
    assert "hard links" not in str(raised.value)
    assert no_scratch(tmp_path)


@pytest.mark.parametrize(
    ("knob", "reason"),
    [
        ("pack_adds_key", r"does not read back as asked: section 2 \(TFLiteModel embedder\)"),
        ("pack_alters_bytes", r"the bytes of section 4 \(TFLiteModel prefill_decode\) changed"),
    ],
)
def test_a_rebuild_that_does_not_read_back_is_not_kept(toolchain, tmp_path, knob, reason):
    setattr(toolchain, knob, True)
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match=reason):
        drop_towers(gemma4(tmp_path), out, ["vision", "audio"])
    assert not out.exists()
    assert no_scratch(tmp_path)


def test_sections_swapped_by_pack_are_refused(toolchain, tmp_path, monkeypatch):
    builder_with(monkeypatch, '    doc["system_metadata"]["entries"] = entries\n',
                 "    contents[2], contents[3] = contents[3], contents[2]\n")  # fmt: skip
    out = tmp_path / "o"
    with pytest.raises(TowersError, match=r"the bytes of section 2 \(TFLiteModel embedder\)"):
        drop_towers(gemma4(tmp_path), out, ["vision"])
    assert not out.exists()


def test_an_additional_metadata_value_changed_by_pack_is_refused(toolchain, tmp_path, monkeypatch):
    builder_with(monkeypatch, '        s["data_path"] = os.path.basename(s["data_path"])\n',
                 '        for e in s.get("additional_metadata", []):\n'
                 '            e["value"] = "fp32"\n')  # fmt: skip
    with pytest.raises(TowersError, match="does not read back as asked: section 7"):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])


def test_a_system_metadata_value_changed_by_pack_is_refused(toolchain, tmp_path, monkeypatch):
    builder_with(monkeypatch, '    doc["system_metadata"]["entries"] = entries\n',
                 '    entries[0]["value"] = "someone else"\n')  # fmt: skip
    with pytest.raises(TowersError, match="does not read back as asked: its system metadata"):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])


def test_a_toml_key_pack_adds_is_refused_on_the_read_back(toolchain, tmp_path, monkeypatch):
    monkeypatch.setitem(
        globals(),
        "FAKE_BUILDER",
        FAKE_BUILDER.replace(
            "    write(output_path, _toml(doc), contents)\n",
            '    write(output_path, "stray = 1\\n" + _toml(doc), contents)\n',
        ),
    )
    with pytest.raises(TowersError, match="TOML for the rebuilt bundle has stray"):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ('    doc["section"].pop()\n    contents.pop()\n',
         "does not read back as asked: 8 sections where 9 were packed"),
        ('    contents.append(b"hidden")\n',
         "the rebuilt bundle has 10 sections where 9 were packed"),
    ],
)  # fmt: skip
def test_a_rebuild_with_another_number_of_sections_is_refused(
    toolchain, tmp_path, monkeypatch, extra, reason
):
    builder_with(monkeypatch, '    doc["system_metadata"]["entries"] = entries\n', extra)
    with pytest.raises(TowersError, match=reason):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])


def test_the_unpacked_inputs_are_gone_before_the_read_back(toolchain, tmp_path, monkeypatch):
    """The scratch directory then holds the result twice, never the inputs too."""
    check = (
        '    if "rebuilt" in str(litertlm_path):\n'
        "        for left in ('before', 'donor'):\n"
        "            assert not (Path(output_dir).parent / left).exists(), left\n"
    )
    builder_with(monkeypatch, "    header, raw = read(litertlm_path)\n", check)
    drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])
    add_towers(sm8850(tmp_path), tmp_path / "o2", ["vision"], donor(tmp_path),
               metadata_from_donor=True)  # fmt: skip


def test_a_failed_builder_is_reported_with_its_last_line(toolchain, tmp_path):
    toolchain.pack_fails = True
    with pytest.raises(TowersError, match="the rebuild script exited 1: RuntimeError: pack: boom"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["audio"])


def test_a_killed_rebuild_is_not_read_as_an_exit(toolchain, tmp_path):
    toolchain.returncode = -9
    with pytest.raises(TowersError, match=r"killed by SIGKILL \(return code -9\)"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["audio"])
    assert no_scratch(tmp_path)


def test_a_rebuild_that_exits_0_without_its_report_is_refused(toolchain, tmp_path):
    toolchain.returncode = 0
    with pytest.raises(TowersError, match="exited 0 without writing its report"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["audio"])


def test_a_rebuild_that_runs_out_of_time_is_refused(toolchain, tmp_path):
    toolchain.times_out = True
    with pytest.raises(TowersError, match="did not finish within 7s"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["audio"], timeout=7)
    assert toolchain.timeouts == [7]
    assert no_scratch(tmp_path)


def test_an_environment_not_provisioned_is_refused_without_provisioning(
    toolchain, tmp_path, monkeypatch
):
    provisioned = []
    monkeypatch.setattr(envs.StageEnv, "provision", lambda self, **kw: provisioned.append(1))
    with pytest.raises(TowersError, match="is not provisioned"):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["audio"], auto_provision=False)
    assert provisioned == [] and toolchain.calls == []


def test_no_tower_or_an_unknown_one_is_refused(toolchain, tmp_path):
    model = gemma4(tmp_path)
    for towers in ([], ["video"]):
        with pytest.raises(TowersError, match="name the towers to drop"):
            drop_towers(model, tmp_path / "out.litertlm", towers)


def test_a_builder_without_a_tower_type_is_refused(toolchain, tmp_path):
    """The tower lists were read from 0.18.0's `TfLiteModelType`; a builder that
    renamed one would leave half a tower behind."""
    toolchain.known_types = [t for t in BUILDER_TYPES if t != "end_of_vision"]
    with pytest.raises(TowersError, match=r"\(0.18.0\) has no model type end_of_vision"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["vision"])


def test_a_builder_with_a_type_0_18_0_lacks_is_refused(toolchain, tmp_path):
    """One the lists do not name could be part of a tower."""
    toolchain.known_types = [*BUILDER_TYPES, "vision_projector"]
    toolchain.builder_version = "0.19.0"
    with pytest.raises(TowersError, match=r"\(0.19.0\) has model types 0.18.0 does not: vision_p"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["vision"])


def test_the_tower_lists_are_builder_0_18_0s_types():
    """`BUILDER_TYPES` pins the 0.18.0 list (`TfLiteModelType`,
    `litertlm_builder.py:211-232`); re-read it when the builder pin moves. The
    script refuses a builder whose list differs."""
    named = {t for types_ in TOWER_SECTIONS.values() for t in types_}
    assert named <= set(BUILDER_TYPES)
    assert {t for t in BUILDER_TYPES if t.startswith(("vision", "audio", "end_of"))} == named


ODD_SYSTEM = """[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "The \\"ODML\\" Authors \\\\ x" },
  { key = "uuid", value_type = "String", value = "u" },
  { key = "creation_timestamp", value_type = "String", value = "t" },
]
"""


def test_every_value_type_unpack_writes_comes_back_as_it_was(toolchain, tmp_path):
    """`unpack` writes numbers and booleans bare and strings escaped; the TOML this
    writes for `pack` must keep each one's type and value."""
    toml = ODD_SYSTEM + GEMMA4_E2B_TOML[GEMMA4_E2B_TOML.index("\n[[section]]") :].replace(
        '  { key = "prefer_activation_type", value_type = "String", value = "fp16" },\n]\n'
        'model_type = "prefill_decode"',
        '  { key = "prefer_activation_type", value_type = "String", value = "fp16" },\n'
        '  { key = "n", value_type = "Int32", value = 5 },\n'
        '  { key = "on", value_type = "Bool", value = true },\n'
        '  { key = "f", value_type = "Float32", value = 0.5 },\n'
        '  { key = "s", value_type = "String", value = "a\\nb\\rc\\td\\u0001" },\n'
        ']\nmodel_type = "prefill_decode"',
    )
    model = gemma4(tmp_path, toml)
    out = tmp_path / "out.litertlm"
    drop_towers(model, out, ["vision"])

    before, _ = read_bundle(model)
    after, _ = read_bundle(out)
    assert after["system_metadata"]["entries"][0] == before["system_metadata"]["entries"][0]
    prefill = [s for s in after["section"] if s.get("model_type") == "prefill_decode"][0]
    values = {e["key"]: e["value"] for e in prefill["additional_metadata"]}
    assert values == {"prefer_activation_type": "fp16", "n": 5, "on": True, "f": 0.5,
                      "s": "a\nb\rc\td\x01"}  # fmt: skip


# -- the header, before anything reads by it -------------------------------------------


def test_an_input_cut_inside_its_last_section_is_refused(toolchain, tmp_path):
    model = gemma4(tmp_path)
    last = offsets(model)[-1]
    model.write_bytes(model.read_bytes()[: last[1] - 5])
    out = tmp_path / "o.litertlm"
    with pytest.raises(
        TowersError,
        match=rf"the bundle \(gemma4.litertlm\): section 11 spans bytes {last[0]} to {last[1]}, ",
    ):
        drop_towers(model, out, ["vision"])
    assert not out.exists()
    assert toolchain.calls  # the script refused, not the host


def test_an_input_cut_where_a_section_starts_is_refused(toolchain, tmp_path):
    """Cut at prefill_decode's first byte: it and the section after it are gone."""
    model = gemma4(tmp_path)
    prefill = offsets(model)[10]
    model.write_bytes(model.read_bytes()[: prefill[0]])
    with pytest.raises(
        TowersError, match=r"the bundle \(gemma4.litertlm\): section 10 spans bytes"
    ):
        drop_towers(model, tmp_path / "o.litertlm", ["vision"])


def test_a_donor_cut_short_is_refused_before_either_is_unpacked(toolchain, tmp_path, monkeypatch):
    unpacked = tmp_path / "unpacked.txt"
    log = f"    open({str(unpacked)!r}, 'a').write(str(litertlm_path) + '\\n')\n"
    builder_with(monkeypatch, "    header, raw = read(litertlm_path)\n", log)
    giver = donor(tmp_path)
    giver.write_bytes(giver.read_bytes()[:-1])
    with pytest.raises(TowersError, match=r"the donor \(donor.litertlm\): section 11 spans bytes"):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision"], giver, metadata_from_donor=True)
    assert not (tmp_path / "o").exists()
    assert not unpacked.exists(), unpacked.read_text()


def test_a_section_that_begins_after_it_ends_is_refused(toolchain, tmp_path):
    model = gemma4(tmp_path)
    listed = offsets(model)
    begin, end = listed[2]
    listed[2] = [end, begin]
    rewrite_header(model, listed)
    with pytest.raises(
        TowersError,
        match=f"the bundle .*: section 2 begins at byte {end}, after it ends at {begin}",
    ):
        drop_towers(model, tmp_path / "o", ["vision"])


def test_overlapping_sections_are_refused(toolchain, tmp_path):
    model = gemma4(tmp_path)
    listed = offsets(model)
    listed[3] = [listed[2][0], listed[3][1]]
    rewrite_header(model, listed)
    with pytest.raises(TowersError, match="the bundle .*: sections 2 and 3 overlap"):
        drop_towers(model, tmp_path / "o", ["vision"])


def test_a_section_inside_the_header_is_refused(toolchain, tmp_path):
    model = gemma4(tmp_path)
    listed = offsets(model)
    listed[0] = [0, listed[0][1] - listed[0][0]]
    rewrite_header(model, listed)
    with pytest.raises(TowersError, match="the bundle .*: section 0 spans bytes 0 to"):
        drop_towers(model, tmp_path / "o", ["vision"])


@pytest.mark.parametrize("past", [True, False])
def test_a_header_that_ends_outside_the_file_is_refused(toolchain, tmp_path, past):
    """Past its end, or before the header's own start at byte 32."""
    model = gemma4(tmp_path)
    raw = bytearray(model.read_bytes())
    end = len(raw) + 1 if past else 31
    struct.pack_into("<Q", raw, 24, end)
    model.write_bytes(bytes(raw))
    with pytest.raises(
        TowersError,
        match=f"gemma4.litertlm\\): its header ends at byte {end}, outside its {len(raw)} bytes",
    ):
        drop_towers(model, tmp_path / "o", ["vision"])


def test_a_header_that_cannot_be_read_is_refused(toolchain, tmp_path):
    model = gemma4(tmp_path)
    raw = bytearray(model.read_bytes())
    raw[32:34] = b"]["
    model.write_bytes(bytes(raw))
    with pytest.raises(
        TowersError, match=r"the bundle \(gemma4.litertlm\): its header could not be read: "
    ):
        drop_towers(model, tmp_path / "o", ["vision"])


def test_a_file_too_short_for_a_header_is_refused(toolchain, tmp_path):
    model = tmp_path / "short.litertlm"
    model.write_bytes(b"LITERTLM" + bytes(12))
    with pytest.raises(
        TowersError,
        match=r"the bundle \(short.litertlm\) is 20 bytes, too short for a LiteRT-LM header",
    ):
        drop_towers(model, tmp_path / "o", ["vision"])


@pytest.mark.parametrize("where", ["metadata", "last"])
def test_an_input_that_shrinks_after_its_header_was_read_is_refused(
    toolchain, tmp_path, monkeypatch, where
):
    """The raw metadata copy, made before the pack, and the read-back's hashes,
    after it, both read the input again."""
    packed = tmp_path / "packed.txt"
    pack = "def pack(toml_path, output_path, jinja_prompt_template_path=None):\n"
    builder_with(monkeypatch, pack, f"    open({str(packed)!r}, 'a').close()\n")
    model = gemma4(tmp_path)
    listed = offsets(model)
    toolchain.shrink_input = listed[0][0] + 1 if where == "metadata" else listed[-1][1] - 1
    end = listed[0][1] if where == "metadata" else listed[-1][1]
    with pytest.raises(TowersError, match=f"ended before byte {end}, which its header names"):
        drop_towers(model, tmp_path / "o", ["vision"])
    assert not (tmp_path / "o").exists()
    assert packed.exists() is (where == "last")


# -- adding --------------------------------------------------------------------------


def test_towers_are_added_with_the_donors_metadata_and_nothing_else_moves(toolchain, tmp_path):
    model, giver = sm8850(tmp_path), donor(tmp_path)
    before = model.read_bytes()
    out = tmp_path / "out.litertlm"

    result = add_towers(model, out, ["vision", "audio"], giver, metadata_from_donor=True)

    assert types(out) == [
        "aux", "embedder", "per_layer_embedder", "prefill_decode",
        "audio_encoder_hw", "audio_adapter", "end_of_audio",
        "vision_encoder", "vision_adapter", "end_of_vision",
    ]  # fmt: skip
    for s_type, m_type in [("SP_Tokenizer", None), ("TFLiteModel", "aux"),
                           ("TFLiteModel", "prefill_decode")]:  # fmt: skip
        assert section(out, s_type, m_type) == section(model, s_type, m_type)
    assert section(out, "LlmMetadata") == section(giver, "LlmMetadata"), "whole, raw"
    assert section(out, "TFLiteModel", "vision_encoder") == section(
        giver, "TFLiteModel", "vision_encoder"
    )
    doc, _ = read_bundle(out)
    assert "npu" in [s.get("backend_constraint") for s in doc["section"]]
    assert result.metadata_source == "donor"
    # Named from the text `unpack` writes, so a field the proto does not declare is
    # carried (the raw comparison above) but cannot be named.
    assert result.metadata_changes == (
        "llm_model_type.gemma4.max_num_patches: unset → 2520",
        "llm_model_type.gemma4.patch_width: unset → 16",
        'llm_model_type.gemma4.start_of_audio_token.token_str: unset → "<|audio>"',
        'llm_model_type.gemma4.start_of_image_token.token_str: unset → "<|image>"',
        'prompt_templates.user.prefix: "<|turn>user\\n" → unset',
    )
    assert result.metadata_raw_differs is True
    assert any(
        n.startswith("LlmMetadata was taken from the donor") and "prompt_templates.user" in n
        for n in result.notes
    )
    assert result.tokens_checked == 4
    assert result.embedding_width == WIDTH
    assert result.builder == "0.18.0"
    assert result.donor == giver
    assert {d["tower"] for d in result.added} == {"vision", "audio"}
    assert model.read_bytes() == before
    assert no_scratch(tmp_path)


def test_a_repeated_field_that_differs_anywhere_is_reported(toolchain, tmp_path):
    """Both end on the same `stop_tokens`; the first ones differ."""

    def stops(first: int) -> str:
        extra = f"stop_tokens {{\n  token_ids {{\n    ids: {first}\n  }}\n}}\n"
        return GEMMA4_META.replace("stop_tokens {\n", extra + "stop_tokens {\n", 1)

    result = add_towers(sm8850(tmp_path, meta=stops(2)), tmp_path / "o", ["vision"],
                        donor(tmp_path, meta=stops(0)), metadata_from_donor=True)  # fmt: skip
    assert result.metadata_changes == ("stop_tokens.token_ids.ids: [2, 1] → [0, 1]",)
    assert any(
        "as the builder's text form shows them: stop_tokens.token_ids.ids: [2, 1] → [0, 1]." in n
        for n in result.notes
    )


@pytest.mark.parametrize(
    ("donor_meta", "said"),
    [
        (GEMMA4_META.replace("unknown_field: 1", "unknown_field: 2"), "but the raw bytes do"),
        (GEMMA4_META, "It is byte for byte the bundle's own"),
    ],
)
def test_metadata_that_differs_only_in_its_raw_bytes_is_said_to(
    toolchain, tmp_path, donor_meta, said
):
    result = add_towers(sm8850(tmp_path, meta=GEMMA4_META), tmp_path / "o", ["vision"],
                        donor(tmp_path, meta=donor_meta), metadata_from_donor=True)  # fmt: skip
    assert result.metadata_changes == ()
    assert result.metadata_raw_differs is (said == "but the raw bytes do")
    assert any(said in n for n in result.notes)


def test_a_long_value_is_reported_by_its_length(toolchain, tmp_path):
    long = "x" * 61
    own = GEMMA4_META + f'jinja_prompt_template: "{long}"\n'
    result = add_towers(sm8850(tmp_path, meta=own), tmp_path / "o", ["vision"],
                        donor(tmp_path), metadata_from_donor=True)  # fmt: skip
    assert result.metadata_changes == ("jinja_prompt_template: <61 characters> → unset",)


def test_the_bundles_own_metadata_is_what_is_kept_without_the_flag(toolchain, tmp_path):
    own = GEMMA4_META + "max_num_tokens: 4096\n"
    model = sm8850(tmp_path, meta=own)
    out = tmp_path / "out.litertlm"

    result = add_towers(model, out, ["vision"], donor(tmp_path))

    assert result.metadata_source == "bundle"
    assert result.metadata_changes == ()
    assert result.metadata_raw_differs is None
    assert section(out, "LlmMetadata") == own.encode()
    assert "audio_encoder_hw" not in types(out)
    assert not any("taken from the donor" in n for n in result.notes)


def test_a_bundle_and_a_donor_laid_out_on_blocks_keep_their_bytes(toolchain, tmp_path):
    own = GEMMA4_META + "max_num_tokens: 4096\n"
    model, giver = sm8850(tmp_path, meta=own), donor(tmp_path)
    assert bytes(64) in model.read_bytes(), "the input has padding between sections"
    out = tmp_path / "o.litertlm"
    add_towers(model, out, ["vision"], giver)
    assert section(out, "LlmMetadata") == own.encode()
    assert section(out, "TFLiteModel", "vision_adapter") == vision_adapter()


@pytest.mark.parametrize(
    ("meta", "towers", "reason"),
    [
        (BARE_META, ["vision"],
         "the bundle's LlmMetadata has llm_model_type unset, not gemma4.*--metadata-from-donor"),
        (GEMMA4_META.replace("gemma4 {", "generic_model {"), ["vision"],
         "has llm_model_type generic_model, not gemma4"),
        # Naming one media token is not naming the other.
        (GEMMA4_META.replace('start_of_audio_token {\n      token_str: "<|audio>"\n    }\n', ""),
         ["vision", "audio"], "the bundle's LlmMetadata names no start_of_audio_token, so the "
         "runtime would put its built-in one before the input.*--metadata-from-donor"),
        # An empty field names nothing.
        (GEMMA4_META.replace('token_str: "<|image>"', ""), ["vision"], "names no start_of_image"),
        # The runtime reads a media token only as a string.
        (GEMMA4_META.replace('token_str: "<|image>"', "token_ids {\n        ids: 5\n      }"),
         ["vision"], "gives start_of_image_token without a token_str, the only form"),
        (GEMMA4_META.replace("    patch_width", "    end_of_audio_token {\n      token_ids {\n"
                             "        ids: 6\n      }\n    }\n    patch_width"),
         ["vision"], "gives end_of_audio_token without a token_str"),
    ],
)  # fmt: skip
def test_metadata_the_tower_cannot_be_added_under_is_refused(
    toolchain, tmp_path, meta, towers, reason
):
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path, meta=meta), out, towers, donor(tmp_path, meta=meta))
    assert not out.exists()


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        (GEMMA4_META.replace('token_str: "<|image>"', ""),
         "the donor's LlmMetadata names no start_of_image_token"),
        (GEMMA4_META.replace("gemma4 {", "gemma3n {"),
         "the donor's LlmMetadata has llm_model_type gemma3n, not gemma4"),
    ],
)  # fmt: skip
def test_the_donors_metadata_is_held_to_the_same(toolchain, tmp_path, meta, reason):
    giver = donor(tmp_path, meta=meta)
    with pytest.raises(TowersError, match=reason) as raised:
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision"], giver, metadata_from_donor=True)
    assert "--metadata-from-donor, which" not in str(raised.value)


@pytest.mark.parametrize(
    ("donor_pieces", "kinds", "reason"),
    [
        # A media token at another id: the tower's input would start elsewhere.
        (["<pad>", "<eos>", "<bos>", "a", "b", "<|audio>", "<|image>"], None,
         r"'<\|audio>' is id 6 \(NORMAL\) in the bundle's tokenizer and id 5 \(NORMAL\)"),
        # The same piece at the same id, encoded otherwise.
        (PIECES, {"<|image>": USER_DEFINED},
         r"'<\|image>' is id 5 \(NORMAL\) in the bundle's tokenizer and id 5 \(USER_DEFINED\)"),
        # A stop token's id naming another piece.
        (["<pad>", "<end>", "<bos>", "a", "b", "<|image>", "<|audio>"], None,
         r"id 1 is '<eos>' \(NORMAL\) in the bundle's tokenizer and '<end>' \(NORMAL\)"),
        # The start token's id, the same piece of another type.
        (PIECES, {"<bos>": CONTROL},
         r"id 2 is '<bos>' \(NORMAL\) in the bundle's tokenizer and '<bos>' \(CONTROL\)"),
    ],
)  # fmt: skip
def test_tokenizers_that_disagree_on_a_named_token_are_refused(
    toolchain, tmp_path, donor_pieces, kinds, reason
):
    out = tmp_path / "out.litertlm"
    giver = donor(tmp_path, tokenizer=sentencepiece(donor_pieces, kinds))
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path), out, ["vision", "audio"], giver, metadata_from_donor=True)
    assert not out.exists()


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        (GEMMA4_META.replace("<|image>", "<|frame>"),
         r"neither tokenizer has '<\|frame>' as one piece, so this check"),
        (GEMMA4_META.replace("ids: 1\n", "ids: 99\n"),
         "id 99 is no piece in the bundle's tokenizer and no piece"),
        # `suppress_tokens` is a TokenIds of its own (llm_metadata.proto:130).
        (GEMMA4_META + "suppress_tokens {\n  ids: 99\n}\n", "id 99 is no piece"),
    ],
)  # fmt: skip
def test_a_token_neither_tokenizer_has_is_refused(toolchain, tmp_path, meta, reason):
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision", "audio"],
                   donor(tmp_path, meta=meta), metadata_from_donor=True)  # fmt: skip


def test_suppressed_ids_are_counted(toolchain, tmp_path):
    meta = GEMMA4_META + "suppress_tokens {\n  ids: 0\n  ids: 3\n}\n"
    result = add_towers(sm8850(tmp_path), tmp_path / "o", ["vision", "audio"],
                        donor(tmp_path, meta=meta), metadata_from_donor=True)  # fmt: skip
    assert result.tokens_checked == 6


@pytest.mark.parametrize("length", [70, 90])
def test_a_piece_longer_than_one_varint_byte_keeps_the_ids_after_it(toolchain, tmp_path, length):
    """A 210-byte piece makes its length, and its message's, two varint bytes
    ending in 0x01; a 270-byte one's second byte is 0x02, where reading the
    continuation bit as part of the value gives another length."""
    long = ["<pad>", "<eos>", "<bos>", "▁" * length, "b", "<|image>", "<|audio>"]
    result = add_towers(
        sm8850(tmp_path, pieces=long), tmp_path / "out.litertlm", ["vision", "audio"],
        donor(tmp_path, pieces=long + ["<|video|>"]), metadata_from_donor=True,
    )  # fmt: skip
    assert result.tokens_checked == 4
    swapped = long[:5] + ["<|audio>", "<|image>"]
    with pytest.raises(TowersError, match=r"'<\|audio>' is id 6 .* tokenizer and id 5"):
        add_towers(sm8850(tmp_path, pieces=long), tmp_path / "o2", ["vision", "audio"],
                   donor(tmp_path, pieces=swapped), metadata_from_donor=True)  # fmt: skip


def test_a_tokenizer_whose_pieces_end_on_the_score_is_read(toolchain, tmp_path):
    model = sm8850(tmp_path, tokenizer=sentencepiece(PIECES[:-1], typed=False))
    giver = donor(tmp_path, tokenizer=sentencepiece(PIECES, typed=False))
    result = add_towers(model, tmp_path / "o", ["vision", "audio"], giver,
                        metadata_from_donor=True)  # fmt: skip
    assert result.tokens_checked == 4


def test_a_piece_without_a_type_is_normal(toolchain, tmp_path):
    """SentencePiece's `type` is NORMAL when absent: an untyped and a typed NORMAL
    piece agree, an untyped and a CONTROL one do not."""
    model = sm8850(tmp_path, tokenizer=sentencepiece(PIECES[:-1], typed=False))
    add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)
    control = donor(tmp_path, tokenizer=sentencepiece(PIECES, {"<|image>": CONTROL}))
    with pytest.raises(TowersError, match=r"id 5 \(NORMAL\) .* and id 5 \(CONTROL\)"):
        add_towers(model, tmp_path / "o2", ["vision"], control, metadata_from_donor=True)


def test_a_tokenizer_that_is_not_one_is_refused(toolchain, tmp_path):
    tokenizer = bytearray(sentencepiece(PIECES[:-1]))
    tokenizer[-3:] = b"\xff\xff\xff"
    model = sm8850(tmp_path, tokenizer=bytes(tokenizer))
    with pytest.raises(TowersError, match="is not a SentencePiece model"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


@pytest.mark.parametrize(
    "cut",
    [
        b"\x15\0\0",  # field 2, a 32-bit score, with 2 of its 4 bytes
        b"\x11\0\0\0",  # field 2 as a 64-bit value, with 3 of its 8 bytes
    ],
)
def test_a_tokenizer_cut_inside_a_fixed_width_field_is_refused(toolchain, tmp_path, cut):
    tokenizer = sentencepiece(PIECES[:-1]) + b"\x0a" + bytes([len(cut) + 5]) + b"\x0a\x03<x>" + cut
    model = sm8850(tmp_path, tokenizer=tokenizer)
    with pytest.raises(TowersError, match="is not a SentencePiece model: a (32|64)-bit field"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


def test_a_tokenizer_without_pieces_is_refused(toolchain, tmp_path):
    model = sm8850(tmp_path, tokenizer=sentencepiece([]))
    with pytest.raises(TowersError, match="holds no SentencePiece pieces"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        (GEMMA4_META + "max_num_tokens:\n", "the donor's LlmMetadata, as `unpack` writes it, "
         "could not be read: it ends inside a field"),
        (GEMMA4_META + "stop_tokens {\n", "could not be read: unbalanced braces"),
        (GEMMA4_META.replace('"<|image>"', '"<\\q>"'), r"could not be read: unknown escape \\q"),
        (GEMMA4_META.replace("ids: 2", "ids: two"), "the LlmMetadata names a token id 'two'"),
    ],
)  # fmt: skip
def test_metadata_text_that_cannot_be_read_is_refused(toolchain, tmp_path, meta, reason):
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision"], donor(tmp_path, meta=meta),
                   metadata_from_donor=True)  # fmt: skip


def test_a_piece_one_byte_short_is_refused(toolchain, tmp_path):
    tokenizer = sentencepiece(PIECES[:-1]) + b"\x0a\x05" + b"\x0a\x04<x>"
    model = sm8850(tmp_path, tokenizer=tokenizer)
    with pytest.raises(TowersError, match="is not a SentencePiece model"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


def test_an_escaped_token_string_is_read_as_the_token(toolchain, tmp_path):
    pieces = PIECES[:-1] + ["<it's>"]
    # `MessageToString` writes an apostrophe as `\'`.
    meta = GEMMA4_META.replace('token_str: "<|image>"', 'token_str: "<it\\\'s>"')
    result = add_towers(
        sm8850(tmp_path, meta=meta, pieces=pieces), tmp_path / "o", ["vision"],
        donor(tmp_path, meta=meta, pieces=pieces),
    )  # fmt: skip
    assert result.metadata_source == "bundle"


def test_an_escaped_double_quote_in_the_metadata_is_read(toolchain, tmp_path):
    """Gemma 4's metadata carries `open_quote: "<|\\"|>"`."""
    meta = GEMMA4_META.replace(
        "    patch_width: 16\n", '    open_quote: "<|\\"|>"\n    patch_width: 16\n'
    )
    result = add_towers(sm8850(tmp_path), tmp_path / "o", ["vision"],
                        donor(tmp_path, meta=meta), metadata_from_donor=True)  # fmt: skip
    assert 'llm_model_type.gemma4.open_quote: unset → "<|\\"|>"' in result.metadata_changes


def test_an_octal_escaped_token_is_the_utf8_piece(toolchain, tmp_path):
    """`MessageToString` writes non-ASCII as octal escapes: `▁` is `\\342\\226\\201`."""
    octal = GEMMA4_META.replace('token_str: "<|image>"', 'token_str: "\\342\\226\\201"')
    pieces = PIECES[:3] + ["▁"] + PIECES[4:-1]
    result = add_towers(
        sm8850(tmp_path, meta=octal, pieces=pieces), tmp_path / "o", ["vision"],
        donor(tmp_path, meta=octal, pieces=pieces + ["<|video|>"]),
    )  # fmt: skip
    assert result.metadata_source == "bundle"
    swapped = PIECES[:3] + ["b", "▁"] + PIECES[5:-1]
    with pytest.raises(TowersError, match="'▁' is id 3 .* in the bundle's tokenizer and id 4"):
        add_towers(sm8850(tmp_path, meta=octal, pieces=pieces), tmp_path / "o2", ["vision"],
                   donor(tmp_path, meta=octal, pieces=swapped))  # fmt: skip


# Each media setting of `Gemma4` in llm_model_type.proto:249-305, as a test's own
# list rather than the script's, and a value other than any the metadata above sets.
VISION_SETTINGS = [
    "patch_width: 14", "patch_height: 14", "max_num_patches: 1024",
    "pooling_kernel_size: 2", "merge_patches: true",
]  # fmt: skip
AUDIO_SETTINGS = ["skip_mel_spectrogram_extraction: true"]


@pytest.mark.parametrize(
    ("setting", "tower"),
    [(s, "vision") for s in VISION_SETTINGS] + [(s, "audio") for s in AUDIO_SETTINGS],
)
def test_each_media_setting_the_towers_were_not_built_for_is_refused(
    toolchain, tmp_path, setting, tower
):
    name = setting.split(":")[0]
    own = re.sub(rf"    {name}: \S+\n", "", GEMMA4_META).replace(
        "  }\n}\nunknown_field", f"    {setting}\n  }}\n}}\nunknown_field"
    )
    assert setting in own
    with pytest.raises(TowersError, match=f"the bundle's LlmMetadata sets gemma4 with .*{name}"):
        add_towers(sm8850(tmp_path, meta=own), tmp_path / "o", [tower], donor(tmp_path))


@pytest.mark.parametrize(
    "setting",
    ["patch_width: 16", "patch_height: 16", "max_num_patches: 2520", "pooling_kernel_size: 3"],
)
def test_a_setting_spelled_out_at_the_runtimes_own_value_is_not_a_difference(
    toolchain, tmp_path, setting
):
    """The runtime keeps its built-in value where a field is left at the proto's
    default (model_data_processor_factory.cc:246-262), so one side setting
    `patch_width: 16` and the other leaving it unset hand the towers the same
    images (gemma4_data_processor_config.h:39-62)."""
    name = setting.split(":")[0]
    unset = re.sub(rf"    {name}: \S+\n", "", GEMMA4_META)
    spelled = unset.replace("  }\n}\nunknown_field", f"    {setting}\n  }}\n}}\nunknown_field")
    assert name not in unset and setting in spelled
    result = add_towers(sm8850(tmp_path, meta=spelled), tmp_path / "o", ["vision"],
                        donor(tmp_path, meta=unset))  # fmt: skip
    assert result.metadata_source == "bundle"


def test_only_the_added_towers_settings_are_compared(toolchain, tmp_path):
    own = GEMMA4_META.replace("max_num_patches: 2520", "max_num_patches: 1024")
    result = add_towers(sm8850(tmp_path, meta=own), tmp_path / "o", ["audio"], donor(tmp_path))
    assert result.metadata_source == "bundle"


def test_the_same_settings_under_another_model_type_are_refused(toolchain, tmp_path):
    """The settings belong to the model type the processor is chosen by."""
    giver = donor(tmp_path, meta=GEMMA4_META.replace("gemma4 {", "generic_model {"))
    with pytest.raises(
        TowersError, match="sets gemma4 with .* where the donor's, .*, sets generic_model with"
    ):
        add_towers(sm8850(tmp_path, meta=GEMMA4_META), tmp_path / "o", ["vision"], giver)


@pytest.mark.parametrize(
    ("own", "theirs", "reason"),
    [
        ("max_num_tokens: 4224\n", "", "would change max_num_tokens from 4224 to unset"),
        ("", "max_num_tokens: 32768\n", "would change max_num_tokens from unset to 32768"),
        ("kv_cache_init_value: -32768\n", "",
         "would change kv_cache_init_value from -32768 to unset, and LiteRT-LM's NPU executor"),
    ],
)  # fmt: skip
def test_donor_metadata_that_changes_what_the_npu_reads_is_refused(
    toolchain, tmp_path, own, theirs, reason
):
    model, giver = (
        sm8850(tmp_path, meta=BARE_META + own),
        donor(tmp_path, meta=GEMMA4_META + theirs),
    )
    with pytest.raises(TowersError, match=reason):
        add_towers(model, tmp_path / "o", ["vision"], giver, metadata_from_donor=True)


def test_an_unset_npu_field_and_its_default_are_the_same(toolchain, tmp_path):
    """Both read as 0 (npu/llm_litert_npu_kv_cache.cc:88-96)."""
    result = add_towers(sm8850(tmp_path, meta=BARE_META + "kv_cache_init_value: 0\n"),
                        tmp_path / "o", ["vision"], donor(tmp_path),
                        metadata_from_donor=True)  # fmt: skip
    assert "kv_cache_init_value: 0 → unset" in result.metadata_changes


def test_donor_metadata_over_a_tower_the_bundle_keeps_must_match_it(toolchain, tmp_path):
    keeps_audio = GEMMA4_META.replace(
        "    max_num_patches", "    skip_mel_spectrogram_extraction: true\n    max_num_patches"
    )
    audio_only = without(GEMMA4_E2B_TOML, *TOWER_SECTIONS["vision"])
    model = bundle(tmp_path / "audio-only.litertlm", audio_only, keeps_audio)
    with pytest.raises(
        TowersError,
        match="the bundle keeps its audio tower, built for gemma4 with "
        "skip_mel_spectrogram_extraction: true, and --metadata-from-donor would set gemma4 "
        "with skip_mel_spectrogram_extraction: false",
    ):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)
    result = add_towers(model, tmp_path / "o2", ["vision"], donor(tmp_path, meta=keeps_audio),
                        metadata_from_donor=True)  # fmt: skip
    assert "audio_encoder_hw" in types(tmp_path / "o2") and result.metadata_source == "donor"


def test_a_tower_already_there_is_refused(toolchain, tmp_path):
    with pytest.raises(TowersError, match="already carries vision"):
        add_towers(gemma4(tmp_path), tmp_path / "a", ["vision"], donor(tmp_path))


@pytest.mark.parametrize(
    "missing",
    ["vision_encoder", "vision_adapter", "end_of_vision",
     "audio_encoder_hw", "audio_adapter", "end_of_audio"],
)  # fmt: skip
def test_a_donor_tower_without_one_of_its_graphs_is_refused(toolchain, tmp_path, missing):
    tower = missing.split("_")[0] if not missing.startswith("end_of") else missing[7:]
    with pytest.raises(TowersError, match=f"the donor's {tower} tower lacks {missing}"):
        add_towers(sm8850(tmp_path), tmp_path / "b", [tower],
                   donor(tmp_path, without(GEMMA4_E2B_TOML, missing)),
                   metadata_from_donor=True)  # fmt: skip


def test_a_donor_tower_needs_each_graph_once_not_weights_alone(toolchain, tmp_path):
    adapter = (
        'model_type = "vision_adapter"\nbackend_constraint = "cpu"\nsection_type = "TFLiteModel"'
    )
    assert GEMMA4_E2B_TOML.count(adapter) == 1
    weights_only = GEMMA4_E2B_TOML.replace(adapter, adapter.replace("TFLiteModel", "TFLiteWeights"))
    with pytest.raises(TowersError, match="donor's vision tower lacks vision_adapter"):
        add_towers(sm8850(tmp_path), tmp_path / "a", ["vision"], donor(tmp_path, weights_only),
                   metadata_from_donor=True)  # fmt: skip
    twice = GEMMA4_E2B_TOML + (
        '\n[[section]]\nmodel_type = "end_of_vision"\nsection_type = "TFLiteModel"\n'
        'data_path = "Section12_TFLiteModel_tf_lite_end_of_vision_2.tflite"\n'
    )
    with pytest.raises(TowersError, match="more than one end_of_vision graph"):
        add_towers(sm8850(tmp_path), tmp_path / "b", ["vision"], donor(tmp_path, twice),
                   metadata_from_donor=True)  # fmt: skip


def test_the_donors_tower_weights_come_with_its_graphs(toolchain, tmp_path):
    out = tmp_path / "out.litertlm"
    giver = donor(tmp_path, GEMMA4_E2B_TOML + WEIGHTS)
    result = add_towers(sm8850(tmp_path), out, ["vision"], giver, metadata_from_donor=True)
    doc, _ = read_bundle(out)
    assert "TFLiteWeights" in {s["section_type"] for s in doc["section"]}
    assert "TFLiteWeights" in {d["section_type"] for d in result.added}


def test_a_donor_section_that_is_not_a_graph_is_not_taken(toolchain, tmp_path):
    toml = GEMMA4_E2B_TOML.replace(
        'section_type = "SP_Tokenizer"',
        'model_type = "vision_encoder"\nsection_type = "SP_Tokenizer"',
    )
    out = tmp_path / "out.litertlm"
    add_towers(sm8850(tmp_path), out, ["vision"], donor(tmp_path, toml), metadata_from_donor=True)

    doc, _ = read_bundle(out)
    assert [s["section_type"] for s in doc["section"]].count("SP_Tokenizer") == 1


def test_a_donor_with_its_metadata_last_gives_it_whole(toolchain, tmp_path):
    meta_block = (
        '[[section]]\nsection_type = "LlmMetadata"\ndata_path = "LlmMetadataProto.pbtext"\n\n'
    )
    assert GEMMA4_E2B_TOML.count(meta_block) == 1
    giver = donor(tmp_path, GEMMA4_E2B_TOML.replace(meta_block, "") + "\n" + meta_block)
    out = tmp_path / "o"
    add_towers(sm8850(tmp_path), out, ["vision"], giver, metadata_from_donor=True)
    assert section(out, "LlmMetadata") == section(giver, "LlmMetadata")


def test_a_bundle_without_metadata_is_refused(toolchain, tmp_path):
    no_meta = SM8850_TOML.replace(
        '[[section]]\nsection_type = "LlmMetadata"\ndata_path = "LlmMetadataProto.pbtext"\n\n', ""
    )
    model = bundle(tmp_path / "m.litertlm", no_meta, pieces=PIECES[:-1])
    with pytest.raises(TowersError, match="has 0 LlmMetadata sections"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


def test_a_bundle_with_two_tokenizers_is_refused(toolchain, tmp_path):
    two = SM8850_TOML + ('\n[[section]]\nsection_type = "SP_Tokenizer"\n'
                         'data_path = "Section6_SP_Tokenizer.spiece"\n')  # fmt: skip
    model = bundle(tmp_path / "m.litertlm", two, BARE_META, PIECES[:-1])
    with pytest.raises(TowersError, match="has 2 SP_Tokenizer sections, not one"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


# -- the width an adapter writes ------------------------------------------------------


@pytest.mark.parametrize(
    ("graphs", "reason"),
    [
        ({"vision_adapter": vision_adapter(8)},
         "the donor's vision_adapter writes 8 values per token and the bundle's embedder 16; "
         "LiteRT-LM would copy"),
        ({"audio_adapter": audio_adapter(32)},
         "the donor's audio_adapter writes 32 values per token"),
        # Every signature the runtime may run the vision adapter with.
        ({"vision_adapter": vision_adapter(WIDTH, second=8)},
         r"the width the donor's vision_adapter writes could not be established: the "
         r"adapter's outputs give \[8, 16\]"),
        ({"vision_adapter": b""}, "vision_adapter writes could not be established: the graph "
         "is empty"),
        ({"audio_adapter": tflite()}, "audio_adapter writes could not be established: the "
         "adapter has no signature"),
        ({"audio_adapter": tflite(("audio", []))}, "the adapter's signature 'audio' has no output"),
    ],
)  # fmt: skip
def test_an_adapter_that_does_not_write_the_embedders_width_is_refused(
    toolchain, tmp_path, graphs, reason
):
    out = tmp_path / "o"
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path), out, ["vision", "audio"], donor(tmp_path, graphs=graphs),
                   metadata_from_donor=True)  # fmt: skip
    assert not out.exists()


@pytest.mark.parametrize(
    ("graph", "reason"),
    [
        # The NPU runs `decode_embedder`, the CPU and GPU the first signature.
        (embedder(WIDTH, decode=8), r"the embedder's outputs give \[8, 16\]"),
        (tflite(("prefill", [[1, WIDTH]])), r"an output has the shape \[1, 16\], fewer than 3"),
        (b"bytes of an NPU graph", "an offset points outside its 21 bytes"),
    ],
)
def test_an_embedder_whose_width_cannot_be_read_is_refused(toolchain, tmp_path, graph, reason):
    with pytest.raises(
        TowersError,
        match="the width the bundle's text model embeds at could not be established from its "
        f"embedder graph: {reason}",
    ):
        add_towers(sm8850(tmp_path, graphs={"embedder": graph}), tmp_path / "o", ["vision"],
                   donor(tmp_path), metadata_from_donor=True)  # fmt: skip


def test_an_embedder_signature_the_runtime_does_not_run_is_not_read(toolchain, tmp_path):
    graph = tflite(("prefill", [[1, 8, WIDTH]]), ("other", [[1, 8, 4]]))
    result = add_towers(sm8850(tmp_path, graphs={"embedder": graph}), tmp_path / "o",
                        ["vision"], donor(tmp_path), metadata_from_donor=True)  # fmt: skip
    assert result.embedding_width == WIDTH


def test_only_the_output_the_runtime_reads_gives_the_embedders_width(toolchain, tmp_path):
    """The runtime reads output 0 (`output_buffers_[0]`, embedding_lookup_text.cc:378-400);
    a second output of another width is not the embedding."""
    graph = tflite(("prefill", [[1, 8, WIDTH], [1, 8, 4]]))
    result = add_towers(sm8850(tmp_path, graphs={"embedder": graph}), tmp_path / "o",
                        ["vision"], donor(tmp_path), metadata_from_donor=True)  # fmt: skip
    assert result.embedding_width == WIDTH


def test_the_embedders_width_is_the_product_of_its_dims_after_the_second(toolchain, tmp_path):
    graph = tflite(("prefill", [[1, 8, 4, 4]]))
    result = add_towers(sm8850(tmp_path, graphs={"embedder": graph}), tmp_path / "o",
                        ["vision"], donor(tmp_path), metadata_from_donor=True)  # fmt: skip
    assert result.embedding_width == 16


def test_a_bundle_without_an_embedder_is_refused(toolchain, tmp_path):
    model = bundle(tmp_path / "m.litertlm", without(SM8850_TOML, "embedder"), BARE_META,
                   PIECES[:-1])  # fmt: skip
    with pytest.raises(TowersError, match="the bundle has 0 embedder graphs, not one, so the"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


def test_the_test_graphs_read_as_tflite_lays_them_out():
    """The hand-built flatbuffer, read with the vtable offsets TFLite's schema
    gives (Model.subgraphs 8, signature_defs 18; SubGraph.tensors 4;
    SignatureDef.outputs 6, signature_key 8, subgraph_index 12;
    TensorMap.tensor_index 6; Tensor.shape 4), by a reader independent of the
    script's."""
    buf = tflite(("a", [[1, 2, 3]]), ("decode_embedder", [[4, 5], [6]]))

    def u32(pos: int) -> int:
        return struct.unpack_from("<I", buf, pos)[0]

    def fieldpos(table: int, slot: int) -> int:
        vtable = table - struct.unpack_from("<i", buf, table)[0]
        return table + struct.unpack_from("<H", buf, vtable + slot)[0]

    def deref(pos: int) -> int:
        return pos + u32(pos)

    def vec(table: int, slot: int) -> list[int]:
        start = deref(fieldpos(table, slot))
        return [start + 4 + 4 * i for i in range(u32(start))]

    model = u32(0)
    assert buf[4:8] == b"TFL3"
    subgraphs = [deref(p) for p in vec(model, 8)]
    found = []
    for sig in (deref(p) for p in vec(model, 18)):
        key = deref(fieldpos(sig, 8))
        tensors = [deref(p) for p in vec(subgraphs[u32(fieldpos(sig, 12))], 4)]
        shapes = []
        for output in (deref(p) for p in vec(sig, 6)):
            tensor = tensors[u32(fieldpos(output, 6))]
            shapes.append([struct.unpack_from("<i", buf, p)[0] for p in vec(tensor, 4)])
        found.append((buf[key + 4 : key + 4 + u32(key)].decode(), shapes))
    assert found == [("a", [[1, 2, 3]]), ("decode_embedder", [[4, 5], [6]])]


# -- the command ---------------------------------------------------------------------


def test_drop_writes_the_report_and_exits_0(toolchain, tmp_path, capsys):
    out = tmp_path / "out.litertlm"
    argv = ["towers", "--model", str(gemma4(tmp_path)), "--drop", "vision", "--drop", "audio"]
    code = main([*argv, "--output", str(out), "--json"])

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "litetune.towers/1" and report["operation"] == "drop"
    assert report["towers"] == ["vision", "audio"] and len(report["dropped"]) == 6
    assert report["builder"] == "0.18.0"


def test_a_refusal_exits_4_and_says_why(toolchain, tmp_path, capsys):
    argv = ["towers", "--model", str(sm8850(tmp_path)), "--drop", "audio"]
    assert main([*argv, "--output", str(tmp_path / "o")]) == 4
    assert "carries no audio section" in capsys.readouterr().err


def test_add_wiring_and_its_refusals(toolchain, tmp_path, capsys):
    model, giver = sm8850(tmp_path), donor(tmp_path)
    out = tmp_path / "out.litertlm"
    base = ["towers", "--model", str(model), "--add", "vision", "--add", "audio"]

    assert main([*base, "--output", str(out)]) == 4
    assert "--from another bundle" in capsys.readouterr().err
    assert main([*base, "--from", str(giver), "--metadata-from-donor", "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "added vision: 3 sections (vision_encoder, vision_adapter, end_of_vision)" in text
    assert "added audio: 3 sections" in text
    assert "each added adapter writes 16 values per token, as the bundle's embedder does" in text
    assert "tokenizers agree on the 4 token strings and ids the metadata names as tokens" in text
    assert "written by litert-lm-builder 0.18.0" in text
    drop = ["towers", "--model", str(giver), "--drop", "vision", "--output", str(tmp_path / "x")]
    assert main([*drop, "--from", str(giver)]) == 4
    assert "go with --add, not --drop" in capsys.readouterr().err
    assert main([*drop, "--metadata-from-donor"]) == 4


def test_add_without_the_flag_keeps_the_bundles_metadata(toolchain, tmp_path):
    own = GEMMA4_META + "max_num_tokens: 4096\n"
    model, giver, out = sm8850(tmp_path, meta=own), donor(tmp_path), tmp_path / "o.litertlm"
    argv = ["towers", "--model", str(model), "--add", "vision", "--from", str(giver)]
    assert main([*argv, "--output", str(out)]) == 0
    assert section(out, "LlmMetadata") == own.encode()


def test_a_missing_donor_is_refused_before_anything_runs(toolchain, tmp_path):
    with pytest.raises(FileNotFoundError, match="no donor bundle at"):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision"], tmp_path / "missing.litertlm")
    assert toolchain.calls == []


@pytest.mark.parametrize("mode", ["drop", "add"])
def test_no_provision_reaches_the_library(toolchain, tmp_path, capsys, mode):
    """The fixture's provision makes the environment ready; with the flag it is
    never called, so the fresh environment is refused."""
    if mode == "drop":
        argv = ["towers", "--model", str(gemma4(tmp_path)), "--drop", "audio"]
    else:
        argv = ["towers", "--model", str(sm8850(tmp_path)), "--add", "vision",
                "--from", str(donor(tmp_path)), "--metadata-from-donor"]  # fmt: skip
    assert main([*argv, "--no-provision", "--output", str(tmp_path / "a")]) == 4
    assert "is not provisioned" in capsys.readouterr().err
    assert main([*argv, "--output", str(tmp_path / "b")]) == 0


@pytest.mark.parametrize("mode", ["drop", "add"])
def test_a_written_bundle_whose_report_is_not_delivered_exits_4_and_says_so(
    toolchain, tmp_path, monkeypatch, caplog, mode
):
    monkeypatch.setattr("litetune.cli._report", lambda *a, **k: False)
    out = tmp_path / "o.litertlm"
    if mode == "drop":
        argv = ["towers", "--model", str(gemma4(tmp_path)), "--drop", "vision"]
    else:
        argv = ["towers", "--model", str(sm8850(tmp_path)), "--add", "vision",
                "--from", str(donor(tmp_path)), "--metadata-from-donor"]  # fmt: skip
    with caplog.at_level(logging.ERROR, logger="litetune.cli"):
        assert main([*argv, "--output", str(out), "--json"]) == 4
    assert out.is_file()
    assert f"{out} was written, but its --json report could not be delivered" in caplog.text
