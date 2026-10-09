"""`towers`: dropping a bundle's vision and audio sections, against a fake builder.

The script under test is the real one, run by this interpreter. Only
`litert_lm_builder` is replaced, by a module faithful where the script could go
wrong unnoticed:

  - the bundle is self-describing: its TOML and every section's bytes are inside
    the file, so a section the script dropped or kept is visible in the output;
  - `unpack` names each file `SectionN_...` by its index, as the real one does, so
    after a drop the kept files are renumbered and the read-back must allow it;
  - `pack` resolves each `data_path` the way the real `_resolve_path` does and
    opens it, and regenerates the system `uuid` and `creation_timestamp`.

The TOMLs are the real ones `unpack` wrote for Google's Gemma 4 E2B CPU/GPU bundle
(`litert-community/gemma-4-E2B-it-litert-lm` at b3ca0d2f) and for the SM8850 NPU
bundle, sections only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import mark_provisioned

from litetune import envs
from litetune.cli import main
from litetune.towers import TOWER_SECTIONS, TowersError, drop_towers

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
  { key = "uuid", value_type = "String", value = "9da432c2-306d-471e-8792-88be760e5d8c" },
  { key = "creation_timestamp", value_type = "String", value = "2026-06-17T09:49:03.176785+00:00" },
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

# The builder: a bundle is MAGIC + JSON {"toml": ..., "files": {name: base64 bytes}}.
# Knobs are formatted in as Python literals; braces that belong to the module
# are doubled for `str.format`.
FAKE_BUILDER = """
import base64
import json
import os
import re
import uuid as _uuid
from pathlib import Path

MAGIC = b"LITERTLM"
PACK_FAILS = {pack_fails}
PACK_ADDS_KEY = {pack_adds_key}
PACK_ALTERS_BYTES = {pack_alters_bytes}


def _read(path):
    raw = Path(path).read_bytes()
    assert raw.startswith(MAGIC), "not a bundle"
    return json.loads(raw[len(MAGIC):].decode("utf-8"))


def _section_files(toml):
    return re.findall(r'data_path\\s*=\\s*"([^"]+)"', toml)


def unpack(litertlm_path, output_dir, jinja_prompt_template_path=None):
    doc = _read(litertlm_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    toml = doc["toml"]
    # Name each file by its index, as the real unpack does.
    index = iter(range(1000))
    def rename(m):
        name = re.sub(r"^Section\\d+_", "", os.path.basename(m.group(2)))
        i = next(index)
        if not name.endswith(".pbtext"):
            name = "Section%d_%s" % (i, name)
        content = doc["files"][re.sub(r"^Section\\d+_", "", os.path.basename(m.group(2)))]
        (out / name).write_bytes(base64.b64decode(content))
        return m.group(1) + name + m.group(3)
    toml = re.sub(r'(data_path\\s*=\\s*")([^"]+)(")', rename, toml)
    (out / "model.toml").write_text(toml, encoding="utf-8")
    return str(out / "model.toml")


def pack(toml_path, output_path, jinja_prompt_template_path=None):
    if PACK_FAILS:
        raise RuntimeError("pack: boom")
    toml_path = Path(toml_path)
    toml = toml_path.read_text(encoding="utf-8")
    files = {{}}
    for raw in _section_files(toml):
        path = Path(raw) if os.path.isabs(raw) else toml_path.parent / raw
        content = path.read_bytes()
        if PACK_ALTERS_BYTES and "prefill_decode" in raw:
            content += b"!"
        name = re.sub(r"^Section\\d+_", "", os.path.basename(raw))
        files[name] = base64.b64encode(content).decode()
    toml = re.sub(r'(data_path\\s*=\\s*")([^"]+)(")',
                  lambda m: m.group(1) + os.path.basename(m.group(2)) + m.group(3), toml)
    toml = re.sub(r'value = "[0-9a-f-]{{36}}"', 'value = "%s"' % _uuid.uuid4(), toml)
    if PACK_ADDS_KEY:
        toml = toml.replace('model_type = "embedder"', 'model_type = "embedder"\\nextra = "x"', 1)
    Path(output_path).write_bytes(MAGIC + json.dumps({{"toml": toml, "files": files}}).encode())
    return str(output_path)
"""


def bundle_bytes(toml: str, files: dict[str, bytes] | None = None) -> bytes:
    """A bundle in the fake's format: each section's file holds its own name,
    unless `files` gives it other bytes (by name, without the `SectionN_` index)."""
    import base64
    import re

    names = [
        re.sub(r"^Section\d+_", "", os.path.basename(p))
        for p in re.findall(r'data_path\s*=\s*"([^"]+)"', toml)
    ]
    content = {n: (files or {}).get(n, f"bytes of {n}".encode()) for n in names}
    encoded = {n: base64.b64encode(b).decode() for n, b in content.items()}
    return b"LITERTLM" + json.dumps({"toml": toml, "files": encoded}).encode()


def read_bundle(path: Path) -> dict:
    """The fake bundle with its files decoded back to text where they are text."""
    import base64

    doc = json.loads(path.read_bytes()[len(b"LITERTLM") :].decode("utf-8"))
    doc["files"] = {n: base64.b64decode(v).decode("latin-1") for n, v in doc["files"].items()}
    return doc


@dataclass
class FakeToolchain:
    """`StageEnv.run`: the towers script, run for real against the fake builder."""

    pack_fails: bool = False
    pack_adds_key: bool = False
    pack_alters_bytes: bool = False
    script_returncode: int = 0
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        assert args[0] == "python" and str(args[1]).endswith("towers.py"), args
        if self.script_returncode:
            return subprocess.CompletedProcess(args, self.script_returncode, "", "Traceback: boom")
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim" / "litert_lm_builder"
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "__init__.py").write_text("")
        (shim / "litertlm_builder.py").write_text(
            FAKE_BUILDER.format(
                pack_fails=self.pack_fails,
                pack_adds_key=self.pack_adds_key,
                pack_alters_bytes=self.pack_alters_bytes,
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim.parent)},
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


def _gemma4(tmp_path: Path, toml: str = GEMMA4_E2B_TOML) -> Path:
    path = tmp_path / "gemma4.litertlm"
    path.write_bytes(bundle_bytes(toml))
    return path


def _types(doc: dict) -> list[str]:
    import re

    return re.findall(r'model_type = "([^"]+)"', doc["toml"])


def test_both_towers_go_and_everything_else_stays_byte_for_byte(toolchain, tmp_path):
    model = _gemma4(tmp_path)
    before = model.read_bytes()
    out = tmp_path / "text.litertlm"

    result = drop_towers(model, out, ["vision", "audio"])

    doc = read_bundle(out)
    assert _types(doc) == ["embedder", "per_layer_embedder", "prefill_decode", "mtp_drafter"]
    # The metadata, the tokenizer and every kept section, as they were.
    old = read_bundle(model)["files"]
    assert doc["files"] == {k: v for k, v in old.items() if "vision" not in k and "audio" not in k}
    assert "LlmMetadataProto.pbtext" in doc["files"] and "SP_Tokenizer.spiece" in doc["files"]
    assert 'value = "fp16"' in doc["toml"], "the prefill section keeps its metadata"
    assert model.read_bytes() == before, "the input is never touched"
    assert [d["model_type"] for d in result.dropped] == [
        "audio_encoder_hw",
        "audio_adapter",
        "end_of_audio",
        "vision_encoder",
        "vision_adapter",
        "end_of_vision",
    ]
    assert {d["tower"] for d in result.dropped} == {"vision", "audio"}
    assert result.bytes_after == out.stat().st_size
    assert result.sha256.startswith("sha256:")
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")], "no scratch left"


def test_one_tower_goes_and_the_other_stays(toolchain, tmp_path):
    out = tmp_path / "no-vision.litertlm"
    drop_towers(_gemma4(tmp_path), out, ["vision"])

    types = _types(read_bundle(out))
    assert not {"vision_encoder", "vision_adapter", "end_of_vision"} & set(types)
    assert {"audio_encoder_hw", "audio_adapter", "end_of_audio"} <= set(types)


def test_a_type_it_does_not_know_is_kept_and_a_prefixed_one_is_still_read(toolchain, tmp_path):
    """A later builder may add a type; a guess would drop it. Older bundles
    spell types with the `tf_lite_` prefix the builder's enum carries."""
    toml = GEMMA4_E2B_TOML.replace(
        'model_type = "end_of_vision"', 'model_type = "tf_lite_end_of_vision"'
    ).replace('model_type = "mtp_drafter"', 'model_type = "vision_something_new"')
    out = tmp_path / "out.litertlm"
    drop_towers(_gemma4(tmp_path, toml), out, ["vision"])

    types = _types(read_bundle(out))
    assert "vision_something_new" in types
    assert "tf_lite_end_of_vision" not in types


def test_a_section_is_chosen_by_its_type_and_section_type_together(toolchain, tmp_path):
    """`section_type` alone matches the text model; a type alone could match a
    section that is not a TFLite graph."""
    toml = GEMMA4_E2B_TOML.replace(
        'section_type = "SP_Tokenizer"',
        'model_type = "vision_encoder"\nsection_type = "SP_Tokenizer"',
    )
    out = tmp_path / "out.litertlm"
    drop_towers(_gemma4(tmp_path, toml), out, ["vision"])

    assert "SP_Tokenizer.spiece" in read_bundle(out)["files"]


def test_a_tower_the_bundle_does_not_carry_is_refused_and_nothing_is_written(toolchain, tmp_path):
    model = tmp_path / "sm8850.litertlm"
    model.write_bytes(bundle_bytes(SM8850_TOML))
    out = tmp_path / "out.litertlm"

    with pytest.raises(TowersError, match="carries no vision section"):
        drop_towers(model, out, ["vision"])
    assert not out.exists()


def test_an_existing_output_is_refused_before_anything_runs(toolchain, tmp_path):
    model = _gemma4(tmp_path)
    with pytest.raises(TowersError, match="already exists"):
        drop_towers(model, model, ["vision"])
    assert toolchain.calls == []


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
        drop_towers(_gemma4(tmp_path), out, ["vision", "audio"])
    assert not out.exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_failed_builder_is_reported_with_its_last_line(toolchain, tmp_path):
    toolchain.pack_fails = True
    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="pack: boom"):
        drop_towers(_gemma4(tmp_path), out, ["audio"])
    assert not out.exists()


def test_no_tower_or_an_unknown_one_is_refused(toolchain, tmp_path):
    model = _gemma4(tmp_path)
    for towers in ([], ["video"]):
        with pytest.raises(TowersError, match="name the towers to drop"):
            drop_towers(model, tmp_path / "out.litertlm", towers)


def test_the_tower_types_are_the_builders_own():
    """`TfLiteModelType` in litert-lm-builder 0.18.0 (`litertlm_builder.py`),
    without its `tf_lite_` prefix. A later builder that renames one must fail
    here before it drops nothing on a real bundle."""
    builder_0_18 = {
        "prefill_decode", "embedder", "per_layer_embedder", "aux",
        "audio_frontend", "audio_encoder_hw", "audio_adapter", "end_of_audio",
        "vision_encoder", "vision_adapter", "end_of_vision",
        "artisan_text_decoder", "mtp_drafter", "mtp_aux", "text_encoder",
    }  # fmt: skip
    named = {t for types in TOWER_SECTIONS.values() for t in types}
    assert named <= builder_0_18
    assert {t for t in builder_0_18 if t.startswith(("vision", "audio", "end_of"))} == named


def test_the_command_writes_the_report_and_exits_0(toolchain, tmp_path, capsys):
    model = _gemma4(tmp_path)
    out = tmp_path / "out.litertlm"
    code = main(
        ["towers", "--model", str(model), "--drop", "vision", "--drop", "audio",
         "--output", str(out), "--json"]
    )  # fmt: skip

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "litetune.towers/1"
    assert report["towers"] == ["vision", "audio"]
    assert len(report["dropped"]) == 6


def test_the_command_refuses_with_exit_4(toolchain, tmp_path, capsys):
    model = tmp_path / "sm8850.litertlm"
    model.write_bytes(bundle_bytes(SM8850_TOML))
    code = main(
        ["towers", "--model", str(model), "--drop", "audio", "--output", str(tmp_path / "o")]
    )

    assert code == 4
    assert "carries no audio section" in capsys.readouterr().err


# -- adding towers from another bundle ------------------------------------------

PIECES = ["<pad>", "<eos>", "<bos>", "a", "b", "<|image>", "<|audio>", "<|video|>"]

# Gemma 4 metadata as the CPU/GPU bundle's pbtext spells it, cut to the fields
# that name tokens: start and stop tokens by id, media tokens by string.
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
  }
}
"""

# The SM8850 bundle's: ids and the old user/model prefixes, no media token.
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

    def field(number: int, wire: int, payload: bytes) -> bytes:
        key = varint(number << 3 | wire)
        return key + (varint(len(payload)) + payload if wire == 2 else payload)

    body = b""
    for text in pieces:
        piece = field(1, 2, text.encode()) + field(2, 5, b"\0\0\0\0") + field(3, 0, varint(1))
        body += field(1, 2, piece)
    return body + field(2, 2, b"")


def _bundle(path: Path, toml: str, meta: str, pieces: list[str]) -> Path:
    path.write_bytes(
        bundle_bytes(
            toml,
            {
                "LlmMetadataProto.pbtext": meta.encode(),
                "SP_Tokenizer.spiece": sentencepiece(pieces),
            },
        )
    )
    return path


def _sm8850(tmp_path: Path, meta: str = BARE_META, pieces: list[str] | None = None) -> Path:
    # The SM8850 tokenizer lacks `<|video|>`.
    return _bundle(tmp_path / "sm8850.litertlm", SM8850_TOML, meta, pieces or PIECES[:-1])


def _donor(tmp_path: Path, toml: str = GEMMA4_E2B_TOML, pieces: list[str] | None = None) -> Path:
    return _bundle(tmp_path / "donor.litertlm", toml, GEMMA4_META, pieces or PIECES)


def test_towers_are_added_with_the_donors_metadata_and_nothing_else_moves(toolchain, tmp_path):
    from litetune.towers import add_towers

    model, donor = _sm8850(tmp_path), _donor(tmp_path)
    before = model.read_bytes()
    out = tmp_path / "out.litertlm"

    result = add_towers(model, out, ["vision", "audio"], donor, metadata_from_donor=True)

    doc = read_bundle(out)
    assert _types(doc) == [
        "aux", "embedder", "per_layer_embedder", "prefill_decode",
        "audio_encoder_hw", "audio_adapter", "end_of_audio",
        "vision_encoder", "vision_adapter", "end_of_vision",
    ]  # fmt: skip
    old = read_bundle(model)["files"]
    for name in ("aux.tflite", "prefill_decode.tflite", "SP_Tokenizer.spiece"):
        key = next(k for k in old if k.endswith(name))
        assert doc["files"][key] == old[key], f"{name} is the bundle's own, byte for byte"
    assert doc["files"]["LlmMetadataProto.pbtext"] == GEMMA4_META
    assert 'backend_constraint = "npu"' in doc["toml"], "the NPU section keeps its keys"
    assert 'backend_constraint = "cpu"' in doc["toml"], "a donor section keeps its keys"
    assert result.metadata_source == "donor"
    assert result.tokens_checked == 4
    assert {d["tower"] for d in result.added} == {"vision", "audio"}
    assert model.read_bytes() == before
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_a_bundle_whose_metadata_names_no_media_token_needs_the_donors(toolchain, tmp_path):
    from litetune.towers import add_towers

    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match="names no start_of_image_token"):
        add_towers(_sm8850(tmp_path), out, ["vision"], _donor(tmp_path))
    assert not out.exists()


def test_a_bundle_that_names_the_token_keeps_its_own_metadata(toolchain, tmp_path):
    from litetune.towers import add_towers

    out = tmp_path / "out.litertlm"
    result = add_towers(_sm8850(tmp_path, meta=GEMMA4_META), out, ["vision"], _donor(tmp_path))

    assert result.metadata_source == "bundle"
    assert "audio_encoder_hw" not in _types(read_bundle(out))


@pytest.mark.parametrize(
    ("donor_pieces", "reason"),
    [
        # A media token at another id: the tower's input would start elsewhere.
        (["<pad>", "<eos>", "<bos>", "a", "b", "<|audio>", "<|image>"], "'<|image>' is id 5"),
        # A stop token's id naming another piece.
        (["<pad>", "<end>", "<bos>", "a", "b", "<|image>", "<|audio>"], "id 1 is '<eos>'"),
    ],
)
def test_tokenizers_that_disagree_on_a_named_token_are_refused(
    toolchain, tmp_path, donor_pieces, reason
):
    from litetune.towers import add_towers

    out = tmp_path / "out.litertlm"
    with pytest.raises(TowersError, match=reason):
        add_towers(
            _sm8850(tmp_path),
            out,
            ["vision", "audio"],
            _donor(tmp_path, pieces=donor_pieces),
            metadata_from_donor=True,
        )
    assert not out.exists()


def test_a_tower_already_there_or_missing_from_the_donor_is_refused(toolchain, tmp_path):
    from litetune.towers import add_towers

    with pytest.raises(TowersError, match="already carries vision"):
        add_towers(
            _bundle(tmp_path / "full.litertlm", GEMMA4_E2B_TOML, GEMMA4_META, PIECES),
            tmp_path / "a.litertlm",
            ["vision"],
            _donor(tmp_path),
        )
    no_audio = GEMMA4_E2B_TOML.replace('model_type = "audio_', 'model_type = "other_').replace(
        'model_type = "end_of_audio"', 'model_type = "other_end"'
    )
    with pytest.raises(TowersError, match="donor carries no audio"):
        add_towers(
            _sm8850(tmp_path),
            tmp_path / "b.litertlm",
            ["audio"],
            _donor(tmp_path, toml=no_audio),
            metadata_from_donor=True,
        )


def test_the_command_adds_with_from_and_refuses_without_it(toolchain, tmp_path, capsys):
    model, donor = _sm8850(tmp_path), _donor(tmp_path)
    out = tmp_path / "out.litertlm"
    base = ["towers", "--model", str(model), "--add", "vision", "--output", str(out)]

    assert main(base) == 4
    assert "--from another bundle" in capsys.readouterr().err
    assert main([*base, "--from", str(donor), "--metadata-from-donor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["operation"] == "add" and report["metadata_source"] == "donor"
    # A bundle that does carry vision, so only the stray --from can refuse it.
    drop_with_donor = ["towers", "--model", str(donor), "--drop", "vision", "--from", str(donor)]
    assert main([*drop_with_donor, "--output", str(tmp_path / "x")]) == 4
    assert "go with --add, not --drop" in capsys.readouterr().err
