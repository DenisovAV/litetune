"""`towers`: a bundle's vision and audio sections dropped, or taken from another.

The script under test is the real one, run by this interpreter. Only
`litert_lm_builder` is replaced, by a fake that behaves like 0.18.0's in each
place the script could go wrong unnoticed:

  - a bundle is a header of section offsets and the sections' raw bytes, read
    back by `litertlm_peek.read_litertlm_header`, so the script's own raw
    comparison reads real offsets;
  - `unpack` names each file `SectionN_...` by its index, writes LlmMetadata as
    text and drops from it every field the proto does not declare (a line
    holding `unknown_field` here, `gemma4` field 13 on Google's bundle), and
    leaves out of `model.toml` the sections it has no type for;
  - `pack` maps every `model_type` through `TfLiteModelType`, raising for one it
    does not have, opens every `data_path` as the real `_resolve_path` resolves
    it, and drops and appends again the system `uuid` and `creation_timestamp`.

The TOMLs are the ones `unpack` wrote for Google's Gemma 4 E2B CPU/GPU bundle
(`litert-community/gemma-4-E2B-it-litert-lm` at b3ca0d2f) and for a Gemma 4 E2B
build for SM8850, sections only and with its uuid and timestamp replaced.
"""

from __future__ import annotations

import importlib
import json
import os
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
from litetune.towers import TOWER_SECTIONS, TowersError, add_towers, drop_towers

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


# litert-lm-builder 0.18.0's `TfLiteModelType`, as values without the prefix.
BUILDER_TYPES = [
    "prefill_decode", "embedder", "per_layer_embedder", "aux",
    "audio_frontend", "audio_encoder_hw", "audio_adapter", "end_of_audio",
    "vision_encoder", "vision_adapter", "end_of_vision",
    "artisan_text_decoder", "mtp_drafter", "mtp_aux", "text_encoder",
]  # fmt: skip

# The fake builder. Its settings are prepended as plain assignments, so this
# text carries no format placeholders.
FAKE_BUILDER = r"""
import enum
import json
import os
import re
import struct
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

MAGIC = b"LITERTLM"

TfLiteModelType = enum.Enum(
    "TfLiteModelType", {t.upper(): "tf_lite_" + t for t in KNOWN_TYPES}
)


def _from_free_value(cls, value):
    name = value.lower()
    return cls(name if name.startswith("tf_lite_") else "tf_lite_" + name)


TfLiteModelType.get_enum_from_tf_free_value = classmethod(_from_free_value)


def read(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == MAGIC, "not a bundle"
    (n,) = struct.unpack("<Q", raw[8:16])
    return json.loads(raw[16:16 + n]), raw


def write(path, toml, contents):
    begin, spans = 0, []
    for data in contents:
        spans.append([begin, begin + len(data)])
        begin += len(data)
    for _ in range(2):  # offsets depend on the header's own length
        header = json.dumps({"toml": toml, "sections": spans}).encode()
        base = 16 + len(header)
        first = spans[0][0] if spans else 0
        spans = [[b - first + base, e - first + base] for b, e in spans]
    header = json.dumps({"toml": toml, "sections": spans}).encode()
    Path(path).write_bytes(MAGIC + struct.pack("<Q", len(header)) + header + b"".join(contents))


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

FAKE_PEEK = r"""
import json
import struct
from pathlib import Path


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
    raw = Path(file_path).read_bytes()
    (n,) = struct.unpack("<Q", raw[8:16])
    return _Header(json.loads(raw[16:16 + n])["sections"])
"""


def _section_names(toml: str) -> list[str]:
    import re

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
    contents += [b"hidden metadata"] * hidden
    begin, spans = 0, []
    for data in contents:
        spans.append([begin, begin + len(data)])
        begin += len(data)
    header = json.dumps({"toml": toml, "sections": spans}).encode()
    for _ in range(2):
        base = 16 + len(header)
        shifted = [[b + base, e + base] for b, e in spans]
        header = json.dumps({"toml": toml, "sections": shifted}).encode()
    path.write_bytes(b"LITERTLM" + struct.pack("<Q", len(header)) + header + b"".join(contents))
    return path


def read_bundle(path: Path) -> tuple[dict, list[bytes]]:
    """The bundle's TOML, parsed, and each listed section's raw bytes."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[8:16])
    header = json.loads(raw[16 : 16 + n])
    doc = tomllib.loads(header["toml"])
    return doc, [raw[b:e] for b, e in header["sections"]]


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
    # Something writes the output path while the rebuild runs.
    output_appears: bool = False
    times_out: bool = False
    calls: list[list[str]] = field(default_factory=list)
    timeouts: list[int] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        self.timeouts.append(timeout)
        assert args[0] == "python" and str(args[1]).endswith("towers.py"), args
        if self.times_out:
            raise subprocess.TimeoutExpired(args, timeout)
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim" / "litert_lm_builder"
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "__init__.py").write_text("")
        settings = (
            f"KNOWN_TYPES = {self.known_types!r}\n"
            f"PACK_FAILS = {self.pack_fails}\n"
            f"PACK_ADDS_KEY = {self.pack_adds_key}\n"
            f"PACK_ALTERS_BYTES = {self.pack_alters_bytes}\n"
        )
        (shim / "litertlm_builder.py").write_text(settings + FAKE_BUILDER, encoding="utf-8")
        (shim / "litertlm_peek.py").write_text(FAKE_PEEK, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim.parent)},
        )
        if self.output_appears:
            Path(json.loads(Path(spec).read_text())["output"]).write_bytes(b"someone else's")
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


# -- tokenizers and metadata ------------------------------------------------------

PIECES = ["<pad>", "<eos>", "<bos>", "a", "b", "<|image>", "<|audio>", "<|video|>"]

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

# A text-only build's: ids, an old-style prompt template, no media token.
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


def sentencepiece(pieces: list[str]) -> bytes:
    """A SentencePiece ModelProto: field 1 per piece (piece, score, type), then an
    empty trainer spec, as the real file carries one."""

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
        piece = field_(1, 2, text.encode()) + field_(2, 5, b"\0\0\0\0") + field_(3, 0, varint(1))
        body += field_(1, 2, piece)
    return body + field_(2, 2, b"")


def bundle(
    path: Path,
    toml: str,
    meta: str = GEMMA4_META,
    pieces: list[str] | None = None,
    hidden: int = 0,
) -> Path:
    files = {
        "LlmMetadataProto.pbtext": meta.encode(),
        "SP_Tokenizer.spiece": sentencepiece(PIECES if pieces is None else pieces),
    }
    return make_bundle(path, toml, files, hidden=hidden)


def gemma4(tmp_path: Path, toml: str = GEMMA4_E2B_TOML, **kwargs) -> Path:
    return bundle(tmp_path / "gemma4.litertlm", toml, **kwargs)


def sm8850(tmp_path: Path, meta: str = BARE_META, pieces: list[str] | None = None) -> Path:
    # The SM8850 tokenizer lacks `<|video|>`.
    return bundle(tmp_path / "sm8850.litertlm", SM8850_TOML, meta, pieces or PIECES[:-1])


def donor(tmp_path: Path, toml: str = GEMMA4_E2B_TOML, **kwargs) -> Path:
    return bundle(tmp_path / "donor.litertlm", toml, **kwargs)


# -- dropping ------------------------------------------------------------------------


def test_both_towers_go_and_every_other_section_is_the_inputs_bytes(toolchain, tmp_path):
    model = gemma4(tmp_path)
    before = model.read_bytes()
    out = tmp_path / "text.litertlm"

    result = drop_towers(model, out, ["vision", "audio"])

    assert types(out) == ["embedder", "per_layer_embedder", "prefill_decode", "mtp_drafter"]
    kept = [("LlmMetadata", None), ("SP_Tokenizer", None)]
    kept += [("TFLiteModel", "prefill_decode"), ("TFLiteModel", "mtp_drafter")]
    for s_type, m_type in kept:
        assert section(out, s_type, m_type) == section(model, s_type, m_type)
    assert model.read_bytes() == before, "the input is never touched"
    assert [d["model_type"] for d in result.dropped] == [
        "audio_encoder_hw", "audio_adapter", "end_of_audio",
        "vision_encoder", "vision_adapter", "end_of_vision",
    ]  # fmt: skip
    assert [s["model_type"] for s in result.sections if s["model_type"]] == types(out)
    assert result.bytes_before == model.stat().st_size
    assert result.bytes_after == out.stat().st_size
    import hashlib

    assert result.sha256 == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest()
    assert no_scratch(tmp_path)


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


def test_a_toml_that_still_carries_the_prefix_is_read_as_its_tower(toolchain, tmp_path):
    """0.18.0's `unpack` strips `tf_lite_` from the stored value; a TOML that keeps
    it still names the same type."""
    toml = GEMMA4_E2B_TOML.replace(
        'model_type = "end_of_vision"', 'model_type = "tf_lite_end_of_vision"'
    )
    out = tmp_path / "out.litertlm"
    drop_towers(gemma4(tmp_path, toml), out, ["vision"])

    assert "tf_lite_end_of_vision" not in types(out)


def test_a_type_the_builder_cannot_write_back_is_refused_not_guessed(toolchain, tmp_path):
    toml = GEMMA4_E2B_TOML.replace('model_type = "mtp_drafter"', 'model_type = "vision_new"')
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="'vision_new', which litert-lm-builder cannot write"):
        drop_towers(gemma4(tmp_path, toml), out, ["vision"])
    assert not out.exists()


def test_a_section_unpack_leaves_out_is_refused_not_lost(toolchain, tmp_path):
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="has 13 sections and litert-lm-builder unpacks 12"):
        drop_towers(gemma4(tmp_path, hidden=1), out, ["vision"])
    assert not out.exists()


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


def test_an_output_written_during_the_rebuild_is_not_overwritten(toolchain, tmp_path):
    toolchain.output_appears = True
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="appeared while the bundle was rebuilt"):
        drop_towers(gemma4(tmp_path), out, ["vision"])
    assert out.read_bytes() == b"someone else's"
    assert no_scratch(tmp_path)


def test_a_filesystem_without_hard_links_is_refused_and_nothing_left(
    toolchain, tmp_path, monkeypatch
):
    def no_link(src, dst):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("litetune.towers.os.link", no_link)
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="Operation not permitted.*supports hard links"):
        drop_towers(gemma4(tmp_path), out, ["vision"])
    assert not out.exists()
    assert no_scratch(tmp_path)


@pytest.mark.parametrize(
    ("knob", "reason"),
    [
        ("pack_adds_key", "does not read back as asked"),
        ("pack_alters_bytes", "a section's bytes changed"),
    ],
)
def test_a_rebuild_that_does_not_read_back_is_not_kept(toolchain, tmp_path, knob, reason):
    setattr(toolchain, knob, True)
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match=reason):
        drop_towers(gemma4(tmp_path), out, ["vision", "audio"])
    assert not out.exists()
    assert no_scratch(tmp_path)


def test_a_failed_builder_is_reported_with_its_last_line(toolchain, tmp_path):
    toolchain.pack_fails = True
    with pytest.raises(TowersError, match="pack: boom"):
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
    with pytest.raises(TowersError, match="has no model type end_of_vision"):
        drop_towers(gemma4(tmp_path), tmp_path / "out.litertlm", ["vision"])


def test_the_tower_lists_are_builder_0_18_0s_types():
    """Pins the 0.18.0 list (`TfLiteModelType`, `litertlm_builder.py`); re-read it
    when the builder pin moves. The script refuses a builder that lacks one."""
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
    assert result.metadata_changes == ("llm_model_type", "prompt_templates")
    assert any("taken from the donor" in n and "prompt_templates" in n for n in result.notes)
    assert result.tokens_checked == 4
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
    assert result.metadata_changes == ("stop_tokens",)
    assert any("as the builder's text form shows them: stop_tokens." in n for n in result.notes)


def test_the_bundles_own_metadata_is_what_is_kept_without_the_flag(toolchain, tmp_path):
    own = GEMMA4_META + "max_num_tokens: 4096\n"
    model = sm8850(tmp_path, meta=own)
    out = tmp_path / "out.litertlm"

    result = add_towers(model, out, ["vision"], donor(tmp_path))

    assert result.metadata_source == "bundle"
    assert section(out, "LlmMetadata") == own.encode()
    assert "audio_encoder_hw" not in types(out)


@pytest.mark.parametrize(
    ("meta", "towers", "missing"),
    [
        (BARE_META, ["vision"], "start_of_image_token"),
        # Naming one media token is not naming the other.
        (GEMMA4_META.replace('start_of_audio_token {\n      token_str: "<|audio>"\n    }\n', ""),
         ["vision", "audio"], "start_of_audio_token"),
        # An empty field names nothing.
        (GEMMA4_META.replace('token_str: "<|image>"', ""), ["vision"], "start_of_image_token"),
    ],
)  # fmt: skip
def test_metadata_that_names_no_input_token_needs_the_donors(
    toolchain, tmp_path, meta, towers, missing
):
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match=f"names no {missing}"):
        add_towers(sm8850(tmp_path, meta=meta), out, towers, donor(tmp_path))
    assert not out.exists()


def test_the_donors_metadata_must_name_the_token_too(toolchain, tmp_path):
    giver = donor(tmp_path, meta=GEMMA4_META.replace('token_str: "<|image>"', ""))
    with pytest.raises(TowersError, match="donor's LlmMetadata names no start_of_image_token"):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision"], giver, metadata_from_donor=True)


@pytest.mark.parametrize(
    ("donor_pieces", "reason"),
    [
        # A media token at another id: the tower's input would start elsewhere.
        (["<pad>", "<eos>", "<bos>", "a", "b", "<|audio>", "<|image>"], "'<|image>' is id 5"),
        # A stop token's id naming another piece.
        (["<pad>", "<end>", "<bos>", "a", "b", "<|image>", "<|audio>"], "id 1 is b'<eos>'"),
    ],
)
def test_tokenizers_that_disagree_on_a_named_token_are_refused(
    toolchain, tmp_path, donor_pieces, reason
):
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path), out, ["vision", "audio"],
                   donor(tmp_path, pieces=donor_pieces), metadata_from_donor=True)  # fmt: skip
    assert not out.exists()


@pytest.mark.parametrize(
    ("meta", "reason"),
    [
        (GEMMA4_META.replace("<|image>", "<|frame>"), "'<|frame>' is id None"),
        (GEMMA4_META.replace("ids: 1\n", "ids: 99\n"), "id 99 is None"),
    ],
)
def test_a_token_neither_tokenizer_has_is_refused(toolchain, tmp_path, meta, reason):
    with pytest.raises(TowersError, match=reason):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["vision", "audio"],
                   donor(tmp_path, meta=meta), metadata_from_donor=True)  # fmt: skip


def test_a_piece_longer_than_one_varint_byte_keeps_the_ids_after_it(toolchain, tmp_path):
    """A 210-byte piece makes its length, and its message's, two varint bytes."""
    long = ["<pad>", "<eos>", "<bos>", "▁" * 70, "b", "<|image>", "<|audio>"]
    result = add_towers(
        sm8850(tmp_path, pieces=long), tmp_path / "out.litertlm", ["vision", "audio"],
        donor(tmp_path, pieces=long + ["<|video|>"]), metadata_from_donor=True,
    )  # fmt: skip
    assert result.tokens_checked == 4
    swapped = long[:5] + ["<|audio>", "<|image>"]
    with pytest.raises(TowersError, match="'<|image>' is id 5 in the bundle's tokenizer and 6"):
        add_towers(sm8850(tmp_path, pieces=long), tmp_path / "o2", ["vision", "audio"],
                   donor(tmp_path, pieces=swapped), metadata_from_donor=True)  # fmt: skip


def test_a_tokenizer_that_is_not_one_is_refused(toolchain, tmp_path):
    model = sm8850(tmp_path)
    doc_bytes = bytearray(model.read_bytes())
    tokenizer = sentencepiece(PIECES[:-1])
    at = doc_bytes.find(tokenizer)
    doc_bytes[at + len(tokenizer) - 3 : at + len(tokenizer)] = b"\xff\xff\xff"
    model.write_bytes(bytes(doc_bytes))
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
    files = {"LlmMetadataProto.pbtext": BARE_META.encode(), "SP_Tokenizer.spiece": tokenizer}
    model = make_bundle(tmp_path / "m.litertlm", SM8850_TOML, files)
    with pytest.raises(TowersError, match="is not a SentencePiece model: a (32|64)-bit field"):
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


def test_media_settings_the_towers_were_not_built_for_are_refused(toolchain, tmp_path):
    own = GEMMA4_META.replace("max_num_patches: 2520", "max_num_patches: 1024")
    with pytest.raises(TowersError, match="max_num_patches"):
        add_towers(sm8850(tmp_path, meta=own), tmp_path / "o", ["vision"], donor(tmp_path))


def test_a_tower_already_there_or_incomplete_in_the_donor_is_refused(toolchain, tmp_path):
    with pytest.raises(TowersError, match="already carries vision"):
        add_towers(gemma4(tmp_path), tmp_path / "a", ["vision"], donor(tmp_path))
    blocks = GEMMA4_E2B_TOML.split("\n\n[[section]]")
    partial = "\n\n[[section]]".join(
        b for b in blocks if '"audio_adapter"' not in b and '"end_of_audio"' not in b
    )
    with pytest.raises(TowersError, match="donor's audio tower lacks audio_adapter, end_of_audio"):
        add_towers(sm8850(tmp_path), tmp_path / "b", ["audio"], donor(tmp_path, partial),
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


def test_a_bundle_without_metadata_is_refused(toolchain, tmp_path):
    no_meta = SM8850_TOML.replace(
        '[[section]]\nsection_type = "LlmMetadata"\ndata_path = "LlmMetadataProto.pbtext"\n\n', ""
    )
    model = bundle(tmp_path / "m.litertlm", no_meta, pieces=PIECES[:-1])
    with pytest.raises(TowersError, match="has 0 LlmMetadata sections"):
        add_towers(model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True)


# -- the command ---------------------------------------------------------------------


def test_drop_writes_the_report_and_exits_0(toolchain, tmp_path, capsys):
    out = tmp_path / "out.litertlm"
    argv = ["towers", "--model", str(gemma4(tmp_path)), "--drop", "vision", "--drop", "audio"]
    code = main([*argv, "--output", str(out), "--json"])

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "litetune.towers/1" and report["operation"] == "drop"
    assert report["towers"] == ["vision", "audio"] and len(report["dropped"]) == 6


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
    assert "tokenizers agree on all 4 tokens" in text
    drop = ["towers", "--model", str(giver), "--drop", "vision", "--output", str(tmp_path / "x")]
    assert main([*drop, "--from", str(giver)]) == 4
    assert "go with --add, not --drop" in capsys.readouterr().err
    assert main([*drop, "--metadata-from-donor"]) == 4


def test_no_provision_reaches_the_library(toolchain, tmp_path, capsys):
    """The fixture's provision makes the environment ready; with the flag it is
    never called, so the fresh environment is refused."""
    argv = ["towers", "--model", str(gemma4(tmp_path)), "--drop", "audio"]
    assert main([*argv, "--no-provision", "--output", str(tmp_path / "a")]) == 4
    assert "is not provisioned" in capsys.readouterr().err
    assert main([*argv, "--output", str(tmp_path / "b")]) == 0
