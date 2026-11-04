"""`embedding_vision`: the graft's text tower spliced into Google's text-vision file,
and the image fields of the embedding record.

The splice script under test is the real one, run by this interpreter, against
test_towers.py's fake litert-lm-builder (its bundle format, `unpack` and `pack`).
The bundles are made up: the splice reads no section's contents, only where each
one is and what the TOML says of it, so each section holds its own name. The
text-vision file's TOML is laid out as Google's 440m file's is (litert-lm-builder
0.18.0's `unpack` of it, 2026-11-04): the text sections, then the vision tower's,
with the vision encoder's activation-type metadata and the adapter's CPU
constraint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned
from test_towers import FAKE_BUILDER, FAKE_FORMAT, FAKE_PEEK

from litetune import embedding_vision, envs
from litetune.embedding_convert import EmbeddingConvertError
from litetune.embedding_tune import (
    DECIDER_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    IMAGE_PREFIX,
    PROCESSOR_FILES,
    PROMPT_NAME,
    RECIPE,
    EmbeddingRecord,
    read_embedding_record,
    write_embedding_record,
)
from litetune.embedding_vision import _SPLICE_SCRIPT, SpliceResult, splice
from litetune.towers import _TOWERS_SCRIPT, BUILDER_TYPES

# tomli below Python 3.11, as the builder itself; by name, so that mypy, checking
# for 3.10, needs stubs for neither.
tomllib: Any = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")

FORMAT: dict[str, Any] = {}
exec(FAKE_FORMAT, FORMAT)

SCRIPT: dict[str, Any] = {"__name__": "embedding_splice"}
exec(compile(_SPLICE_SCRIPT, "embedding_splice.py", "exec"), SCRIPT)

PARTS, KINDS = list(embedding_vision.TEXT_PARTS), list(embedding_vision.TEXT_KINDS)

SYSTEM = """[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "The ODML Authors" },
  { key = "uuid", value_type = "String", value = "UUID" },
  { key = "creation_timestamp", value_type = "String", value = "STAMP" },
]
"""

# (TOML of the section, its name) in Google's 270m order; the graft keeps it.
TEXT_TOML = [
    ('section_type = "EmbeddingMetadata"\ndata_path = "EmbeddingMetadataProto.pbtext"', "meta"),
    ('section_type = "SP_Tokenizer"\ndata_path = "Section1_SP_Tokenizer.spiece"', "tokenizer"),
    (
        'model_type = "embedder"\nsection_type = "TFLiteModel"\n'
        'data_path = "Section2_TFLiteModel_tf_lite_embedder.tflite"',
        "embedder graph",
    ),
    (
        'model_type = "text_encoder"\nsection_type = "TFLiteModel"\n'
        'data_path = "Section3_TFLiteModel_tf_lite_text_encoder.tflite"',
        "encoder graph",
    ),
    (
        'model_type = "embedder"\nsection_type = "TFLiteWeights"\n'
        'data_path = "Section4_TFLiteWeights_tf_lite_embedder.weight"',
        "embedder weights",
    ),
    (
        'model_type = "text_encoder"\nsection_type = "TFLiteWeights"\n'
        'data_path = "Section5_TFLiteWeights_tf_lite_text_encoder.weight"',
        "encoder weights",
    ),
]

# The 440m order: the text graphs, the vision graphs, then every weights section.
VISION_TOML = [
    TEXT_TOML[0],
    TEXT_TOML[1],
    TEXT_TOML[2],
    TEXT_TOML[3],
    (
        'additional_metadata = [\n  { key = "prefer_activation_type", value_type = "String", '
        'value = "fp16" },\n]\nmodel_type = "vision_encoder"\nsection_type = "TFLiteModel"\n'
        'data_path = "Section4_TFLiteModel_tf_lite_vision_encoder.tflite"',
        "vision graph",
    ),
    (
        'model_type = "vision_adapter"\nbackend_constraint = "cpu"\nsection_type = "TFLiteModel"\n'
        'data_path = "Section5_TFLiteModel_tf_lite_vision_adapter.tflite"',
        "adapter graph",
    ),
    (
        'model_type = "end_of_vision"\nsection_type = "TFLiteModel"\n'
        'data_path = "Section6_TFLiteModel_tf_lite_end_of_vision.tflite"',
        "end of vision",
    ),
    (TEXT_TOML[4][0].replace("Section4", "Section7"), "embedder weights"),
    (TEXT_TOML[5][0].replace("Section5", "Section8"), "encoder weights"),
    (
        'model_type = "vision_encoder"\nsection_type = "TFLiteWeights"\n'
        'data_path = "Section9_TFLiteWeights_tf_lite_vision_encoder.weight"',
        "vision weights",
    ),
    (
        'model_type = "vision_adapter"\nsection_type = "TFLiteWeights"\n'
        'data_path = "Section10_TFLiteWeights_tf_lite_vision_adapter.weight"',
        "adapter weights",
    ),
]

TEXT_NAMES = {"embedder graph", "encoder graph", "embedder weights", "encoder weights"}


def toml_of(sections: list[tuple[str, str]], system: str = SYSTEM) -> str:
    return system + "".join(f"\n[[section]]\n{body}\n" for body, _ in sections)


def write(
    path: Path,
    sections: list[tuple[str, str]],
    contents: dict[str, bytes] | None = None,
    system: str = SYSTEM,
) -> Path:
    """A bundle in the fake's format; each section holds `contents[name]`, else
    "Google's <name>"."""
    data = [(contents or {}).get(name, f"Google's {name}".encode()) for _, name in sections]
    FORMAT["write"](path, toml_of(sections, system), data)
    return path


GRAFTED = {name: f"grafted {name}".encode() for name in TEXT_NAMES}


def read(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    header, raw = FORMAT["read"](path)
    return tomllib.loads(header["toml"]), [raw[b:e] for b, e in header["sections"]]


def without_volatile(doc: dict[str, Any]) -> dict[str, Any]:
    entries = [
        e
        for e in doc["system_metadata"]["entries"]
        if e["key"] not in ("uuid", "creation_timestamp")
    ]
    sections = [{k: v for k, v in s.items() if k != "data_path"} for s in doc["section"]]
    return {"entries": entries, "sections": sections}


def install_shim(
    shim: Path, *, alters_bytes_of: str | None = None, alters_value: bool = False
) -> None:
    """The fake litert-lm-builder 0.18.0, importable from `shim`. `alters_bytes_of`:
    pack appends a byte to that model_type's sections; `alters_value`: pack gives
    the embedder's sections a backend constraint."""
    package = shim / "litert_lm_builder"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    builder = FAKE_BUILDER
    if alters_bytes_of is not None:
        anchor = 's.get("model_type") == "prefill_decode"'
        assert builder.count(anchor) == 1
        builder = builder.replace(anchor, f's.get("model_type") == "{alters_bytes_of}"')
    settings = (
        f"KNOWN_TYPES = {list(BUILDER_TYPES)!r}\nPACK_FAILS = False\n"
        f"PACK_ALTERS_VALUE = {alters_value}\nPACK_ALTERS_BYTES = {alters_bytes_of is not None}\n"
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


@dataclass
class FakeToolchain:
    """`StageEnv.run`: the splice script, run for real against the fake builder."""

    alters_bytes_of: str | None = None
    alters_value: bool = False
    corrupt_rebuilt: bool = False
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        assert args[0] == "python" and str(args[1]).endswith("embedding_splice.py"), args
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim"
        install_shim(shim, alters_bytes_of=self.alters_bytes_of, alters_value=self.alters_value)
        proc = subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim)},
        )
        if self.corrupt_rebuilt:
            rebuilt = Path(json.loads(Path(result).read_text(encoding="utf-8"))["rebuilt"])
            data = bytearray(rebuilt.read_bytes())
            data[-1] ^= 1
            rebuilt.write_bytes(data)
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


@dataclass
class Files:
    vision: Path
    grafted: Path


def files(
    tmp_path: Path,
    monkeypatch,
    *,
    graft_sections: list[tuple[str, str]] | None = None,
    vision_sections: list[tuple[str, str]] | None = None,
) -> Files:
    vision = write(tmp_path / "vision.litertlm", vision_sections or VISION_TOML)
    grafted = write(tmp_path / "grafted.litertlm", graft_sections or TEXT_TOML, GRAFTED)
    monkeypatch.setattr(
        embedding_vision, "VISION_SHA256", hashlib.sha256(vision.read_bytes()).hexdigest()
    )
    return Files(vision, grafted)


def run_splice(f: Files, out: Path) -> SpliceResult:
    return splice(envs.EXPORT, f.grafted, f.vision, out)


def refused(f: Files, out: Path, cause: str, match: str) -> None:
    with pytest.raises(EmbeddingConvertError, match=match) as info:
        run_splice(f, out)
    assert info.value.cause == cause
    assert not out.exists()
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".")]


# -- the splice, end to end ---------------------------------------------------------


def test_the_text_sections_are_the_grafts_and_every_other_googles(toolchain, tmp_path, monkeypatch):
    f = files(tmp_path, monkeypatch)
    out = tmp_path / "out" / "model.litertlm"
    result = run_splice(f, out)
    doc, contents = read(out)
    names = [name for _, name in VISION_TOML]
    assert len(contents) == len(names)
    for name, data in zip(names, contents, strict=True):
        want = GRAFTED[name] if name in TEXT_NAMES else f"Google's {name}".encode()
        assert data == want, name
    # The TOML is the text-vision file's, apart from what pack stamps anew.
    google, _ = read(f.vision)
    assert without_volatile(doc) == without_volatile(google)
    stamped = {e["key"]: e["value"] for e in doc["system_metadata"]["entries"]}
    assert stamped["uuid"] != "UUID"
    assert result.sha256 == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest()
    assert result.bytes == out.stat().st_size
    assert result.builder == "0.18.0"
    assert sorted(result.replaced) == sorted(f"{kind} {part}" for part in PARTS for kind in KINDS)
    sources = [s["source"] for s in result.sections]
    assert sources == ["graft" if n in TEXT_NAMES else "google_vision" for n in names]
    assert [s["changed"] for s in result.sections] == [n in TEXT_NAMES for n in names]
    assert [s["sha256"] for s in result.sections] == [
        hashlib.sha256(c).hexdigest() for c in contents
    ]
    # Nothing but the output is left beside it.
    assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_a_text_vision_file_that_is_not_googles_is_refused(toolchain, tmp_path, monkeypatch):
    f = files(tmp_path, monkeypatch)
    monkeypatch.setattr(embedding_vision, "VISION_SHA256", "0" * 64)
    refused(f, tmp_path / "out" / "m.litertlm", "base_sha256", "not Google's 0000")


def test_a_graft_without_a_text_section_is_refused(toolchain, tmp_path, monkeypatch):
    f = files(tmp_path, monkeypatch, graft_sections=TEXT_TOML[:5])
    refused(f, tmp_path / "out" / "m.litertlm", "input", "0 TFLiteWeights sections for text_enc")


def test_a_text_vision_file_with_two_encoders_is_refused(toolchain, tmp_path, monkeypatch):
    doubled = VISION_TOML + [(VISION_TOML[3][0].replace("Section3", "Section11"), "again")]
    f = files(tmp_path, monkeypatch, vision_sections=doubled)
    refused(f, tmp_path / "out" / "m.litertlm", "input", "2 TFLiteModel sections for text_enc")


def test_a_graft_whose_toml_says_otherwise_of_a_text_section_is_refused(
    toolchain, tmp_path, monkeypatch
):
    constrained = list(TEXT_TOML)
    constrained[3] = ('backend_constraint = "gpu"\n' + TEXT_TOML[3][0], TEXT_TOML[3][1])
    f = files(tmp_path, monkeypatch, graft_sections=constrained)
    refused(f, tmp_path / "out" / "m.litertlm", "input", "TFLiteModel text_encoder section is")


@pytest.mark.parametrize("model_type", ["vision_encoder", "text_encoder"])
def test_a_pack_that_changes_a_sections_bytes_is_refused(
    toolchain, tmp_path, monkeypatch, model_type
):
    toolchain.alters_bytes_of = model_type
    f = files(tmp_path, monkeypatch)
    refused(f, tmp_path / "out" / "m.litertlm", "readback", "changed in the rebuild")


def test_a_pack_that_changes_the_toml_is_refused(toolchain, tmp_path, monkeypatch):
    toolchain.alters_value = True
    f = files(tmp_path, monkeypatch)
    refused(f, tmp_path / "out" / "m.litertlm", "readback", "does not read back as asked")


def test_a_packed_file_that_hashes_otherwise_is_not_left_at_the_output(
    toolchain, tmp_path, monkeypatch
):
    toolchain.corrupt_rebuilt = True
    f = files(tmp_path, monkeypatch)
    refused(f, tmp_path / "out" / "m.litertlm", "readback", "hashes to sha256:")


def test_an_existing_output_is_never_written_over(toolchain, tmp_path, monkeypatch):
    f = files(tmp_path, monkeypatch)
    out = tmp_path / "m.litertlm"
    out.write_bytes(b"mine")
    with pytest.raises(EmbeddingConvertError, match="already exists"):
        run_splice(f, out)
    assert out.read_bytes() == b"mine" and toolchain.calls == []


def test_a_missing_input_is_file_not_found(toolchain, tmp_path, monkeypatch):
    f = files(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        splice(envs.EXPORT, tmp_path / "nowhere", f.vision, tmp_path / "m.litertlm")
    assert toolchain.calls == []


def test_a_script_that_fails_is_a_run_refusal(toolchain, tmp_path, monkeypatch):
    f = files(tmp_path, monkeypatch)

    def fails(self, args, timeout: int = 0, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "Traceback\nMemoryError")

    monkeypatch.setattr(envs.StageEnv, "run", fails)
    refused(f, tmp_path / "out" / "m.litertlm", "run", "MemoryError")


@pytest.mark.parametrize(
    ("written", "match"),
    [("[1, 2]", "reported list, not an object"), ("{not json", "could not be read")],
)
def test_a_report_that_is_not_an_object_is_a_run_refusal(
    toolchain, tmp_path, monkeypatch, written, match
):
    f = files(tmp_path, monkeypatch)

    def answers(self, args, timeout: int = 0, **kwargs):
        Path(args[3]).write_text(written, encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", answers)
    refused(f, tmp_path / "out" / "m.litertlm", "run", match)


# -- the frozen tensors, held to the base's ---------------------------------------


def _st(tensors: dict[str, tuple[str, list[int], list[float]]]) -> bytes:
    import struct

    header: dict[str, Any] = {}
    data = b""
    for name, (dtype, shape, values) in tensors.items():
        packed = struct.pack(f"<{len(values)}f", *values)
        if dtype == "BF16":
            packed = b"".join(packed[i + 2 : i + 4] for i in range(0, len(packed), 4))
        if dtype == "F16":
            packed = struct.pack(f"<{len(values)}e", *values)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(packed)],
        }
        data += packed
    text = json.dumps(header).encode()
    return struct.pack("<Q", len(text)) + text + data


BASE_TENSORS = {
    "vision_tower.a": ("BF16", [3], [1.0, -2.0, 0.5]),
    "embed_vision.b": ("BF16", [1], [4.0]),
    "language_model.embed_tokens.weight": ("BF16", [2, 2], [3.0, -0.25, 8.0, 1.5]),
    "language_model.layers.0.w": ("BF16", [1], [1.0]),
    "audio_tower.c": ("BF16", [1], [7.0]),
}


def _frozen(tmp_path: Path, **tuned: Any) -> list[str]:
    base = tmp_path / "base.safetensors"
    base.write_bytes(_st(BASE_TENSORS))
    as_f32 = {name: ("F32", s, v) for name, (_, s, v) in BASE_TENSORS.items()}
    del as_f32["audio_tower.c"]  # tune saves no audio tower
    as_f32["language_model.layers.0.w"] = ("F32", [1], [1.125])  # trained
    for name, value in tuned.items():
        if value is None:
            del as_f32[name]
        else:
            as_f32[name] = value
    path = tmp_path / "tuned.safetensors"
    path.write_bytes(_st(as_f32))
    return embedding_vision.frozen_changes(base, path)


def test_frozen_tensors_widened_from_bf16_are_the_base_and_trained_ones_are_not_read(tmp_path):
    assert _frozen(tmp_path) == []


def test_a_frozen_tensor_that_moved_by_less_than_a_bf16_step_is_a_change(tmp_path):
    # 1.0 + 2**-20 is a float32 a BF16 cannot hold: a change below BF16's resolution.
    moved = ("F32", [3], [1.0 + 2**-20, -2.0, 0.5])
    assert _frozen(tmp_path, **{"vision_tower.a": moved}) == ["vision_tower.a (other values)"]


@pytest.mark.parametrize(
    ("tuned", "said"),
    [
        ({"embed_vision.b": None}, "embed_vision.b (not in the fine-tune)"),
        ({"embed_vision.extra": ("F32", [1], [0.0])}, "embed_vision.extra (not in the base)"),
        (
            {"language_model.embed_tokens.weight": ("F32", [4], [3.0, -0.25, 8.0, 1.5])},
            "shape [4], the base's [2, 2]",
        ),
        ({"embed_vision.b": ("F16", [1], [4.0])}, "F16 against the base's BF16, not compared"),
    ],
)
def test_a_frozen_tensor_missing_reshaped_or_retyped_is_a_change(tmp_path, tuned, said):
    (change,) = _frozen(tmp_path, **tuned)
    assert said in change


def test_the_comparison_does_not_depend_on_the_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_vision, "_CHUNK_VALUES", 1)
    assert _frozen(tmp_path) == []
    late = ("F32", [2, 2], [3.0, -0.25, 8.0, 1.75])
    assert _frozen(tmp_path, **{"language_model.embed_tokens.weight": late}) == [
        "language_model.embed_tokens.weight (other values)"
    ]


def test_files_that_are_not_safetensors_or_hold_no_frozen_tensor_are_refused(tmp_path):
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"\x05\x00")
    tuned = tmp_path / "tuned.safetensors"
    tuned.write_bytes(_st({"language_model.layers.0.w": ("F32", [1], [1.0])}))
    with pytest.raises(EmbeddingConvertError, match="too short for safetensors"):
        embedding_vision.frozen_changes(base, tuned)
    base.write_bytes(_st({"language_model.layers.0.w": ("BF16", [1], [1.0])}))
    with pytest.raises(EmbeddingConvertError, match="holds a tensor under"):
        embedding_vision.frozen_changes(base, tuned)


# -- the read-back, against hand-made packed files ----------------------------------


@pytest.fixture
def towers_script(tmp_path, monkeypatch) -> types.ModuleType:
    """The towers script, imported in this interpreter against the fake builder."""
    shim = tmp_path / "shim"
    install_shim(shim)
    monkeypatch.syspath_prepend(str(shim))
    for name in [m for m in sys.modules if m.startswith("litert_lm_builder")]:
        monkeypatch.delitem(sys.modules, name)
    module = types.ModuleType("towers_script")
    exec(compile(_TOWERS_SCRIPT, "towers_script.py", "exec"), module.__dict__)
    return module


def read_back(T, vision: Path, grafted: Path, rebuilt: Path) -> list[dict[str, Any]]:
    vision_doc, _ = read(vision)
    graft_doc, _ = read(grafted)
    after, _ = read(rebuilt)
    graft_text = SCRIPT["text_sections"](graft_doc, "the graft", PARTS, KINDS, T)
    return SCRIPT["read_back"](
        str(rebuilt),
        after,
        str(vision),
        vision_doc,
        T.spans(str(vision), "v"),
        str(grafted),
        T.spans(str(grafted), "g"),
        graft_text,
        PARTS,
        KINDS,
        T,
    )


def read_back_refused(T, tmp_path: Path, rebuilt: Path, match: str) -> None:
    vision = write(tmp_path / "vision.litertlm", VISION_TOML)
    grafted = write(tmp_path / "grafted.litertlm", TEXT_TOML, GRAFTED)
    with pytest.raises(SCRIPT["Refused"], match=match) as info:
        read_back(T, vision, grafted, rebuilt)
    assert info.value.cause == "readback"


def spliced(**changed: bytes) -> dict[str, bytes]:
    return {**GRAFTED, **changed}


def test_the_read_back_takes_a_right_splice(towers_script, tmp_path):
    vision = write(tmp_path / "vision.litertlm", VISION_TOML)
    grafted = write(tmp_path / "grafted.litertlm", TEXT_TOML, GRAFTED)
    stamped = SYSTEM.replace('"UUID"', '"another"').replace('"STAMP"', '"later"')
    rebuilt = write(tmp_path / "rebuilt.litertlm", VISION_TOML, spliced(), stamped)
    sections = read_back(towers_script, vision, grafted, rebuilt)
    assert [s["source"] == "graft" for s in sections] == [n in TEXT_NAMES for _, n in VISION_TOML]


@pytest.mark.parametrize("name", ["vision weights", "adapter graph", "end of vision", "meta"])
def test_the_read_back_refuses_a_section_that_is_not_googles(towers_script, tmp_path, name):
    rebuilt = write(tmp_path / "r.litertlm", VISION_TOML, spliced(**{name: b"tampered"}))
    read_back_refused(towers_script, tmp_path, rebuilt, "bytes of Google's text-vision file")


@pytest.mark.parametrize("name", sorted(TEXT_NAMES))
def test_the_read_back_refuses_a_text_section_that_is_not_the_grafts(towers_script, tmp_path, name):
    # Google's own bytes left where the graft's belong.
    rebuilt = write(
        tmp_path / "r.litertlm", VISION_TOML, spliced(**{name: f"Google's {name}".encode()})
    )
    read_back_refused(towers_script, tmp_path, rebuilt, "bytes of the graft")


def test_the_read_back_refuses_another_toml(towers_script, tmp_path):
    other = list(VISION_TOML)
    other[5] = (VISION_TOML[5][0].replace('"cpu"', '"gpu"'), VISION_TOML[5][1])
    rebuilt = write(tmp_path / "r.litertlm", other, spliced())
    read_back_refused(towers_script, tmp_path, rebuilt, "TOML is not")


def test_the_read_back_refuses_another_author(towers_script, tmp_path):
    rebuilt = write(tmp_path / "r.litertlm", VISION_TOML, spliced(), SYSTEM.replace("ODML", "X"))
    read_back_refused(towers_script, tmp_path, rebuilt, "TOML is not")


def test_the_read_back_refuses_a_header_listing_another_number_of_sections(towers_script, tmp_path):
    rebuilt = write(tmp_path / "r.litertlm", VISION_TOML, spliced())
    header, raw = FORMAT["read"](rebuilt)
    contents = [raw[b:e] for b, e in header["sections"]]
    FORMAT["write"](rebuilt, header["toml"], contents + [b"one more"])
    # The TOML lists eleven sections and the header twelve.
    read_back_refused(towers_script, tmp_path, rebuilt, "has 12 sections where")


def test_the_read_back_names_a_text_section_moved_elsewhere(towers_script, tmp_path):
    # The graft's encoder weights in the vision weights' place, and the reverse.
    swapped = spliced(**{"encoder weights": b"Google's vision weights"})
    swapped["vision weights"] = GRAFTED["encoder weights"]
    rebuilt = write(tmp_path / "r.litertlm", VISION_TOML, swapped)
    read_back_refused(towers_script, tmp_path, rebuilt, r"section 8 \(TFLiteWeights text_enc")


# -- the record ---------------------------------------------------------------------


def _record(**overrides: Any) -> EmbeddingRecord:
    fields: dict[str, Any] = {
        "base_model": EMBEDDING_MODEL,
        "base_model_revision": EMBEDDING_REVISION,
        "prompt_name": PROMPT_NAME,
        "prefix": IMAGE_PREFIX,
        "labels": ("a", "b"),
        "rows": {"train": 10, "heldout": 4},
        "steps_done": 4,
        "steps_planned": 4,
        "seconds": 1.5,
        "device": "cpu",
        "dtype": "float32",
        "head_heldout_accuracy": 0.5,
        "recipe": RECIPE.as_dict(),
        "decision": {"kind": "choice"},
        "decision_sha256": "sha256:" + "0" * 64,
        "modality": "image",
        "vision_tokens": 70,
    }
    fields.update(overrides)
    if fields["modality"] == "image":
        fields.setdefault("soft_tokens", {"c" * 64: 54})
        fields.setdefault("processor_sha256", dict.fromkeys(PROCESSOR_FILES, "d" * 64))
    return EmbeddingRecord(**fields)


@pytest.mark.parametrize("tokens", [70, 140])
def test_an_image_record_round_trips(tmp_path, tokens):
    record = _record(vision_tokens=tokens)
    write_embedding_record(tmp_path, record)
    assert read_embedding_record(tmp_path) == record
    data = json.loads((tmp_path / "litetune.json").read_text())
    assert (data["modality"], data["vision_tokens"]) == ("image", tokens)
    assert record.is_image


def test_a_record_without_the_fields_is_a_text_one(tmp_path):
    data = _record(modality="text", vision_tokens=None, prefix=DECIDER_PREFIX).as_dict()
    del data["modality"], data["vision_tokens"]
    (tmp_path / "litetune.json").write_text(json.dumps(data))
    record = read_embedding_record(tmp_path)
    assert record is not None
    assert (record.modality, record.vision_tokens, record.is_image) == ("text", None, False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"modality": "audio"}, "'modality' is 'audio'"),
        ({"modality": None}, "'modality' is None"),
        ({"vision_tokens": None}, "'vision_tokens' is None; an image record needs one of 70, 140"),
        ({"vision_tokens": 100}, "'vision_tokens' is 100"),
        ({"vision_tokens": True}, "'vision_tokens' is True"),
        ({"vision_tokens": 70.0}, "'vision_tokens' is 70.0"),
        ({"vision_tokens": "70"}, "'vision_tokens' is '70'"),
        ({"modality": "text", "vision_tokens": 70}, "in a text record, which has none"),
    ],
)
def test_a_record_with_wrong_image_fields_is_refused(tmp_path, changes, message):
    data = _record().as_dict() | changes
    (tmp_path / "litetune.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        read_embedding_record(tmp_path)
    with pytest.raises(ValueError, match="vision_tokens|modality"):
        _record(**changes)
