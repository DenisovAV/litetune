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

# The builder: a bundle is MAGIC + JSON {"toml": ..., "files": {name: text}}.
# Knobs are formatted in as Python literals; braces that belong to the module
# are doubled for `str.format`.
FAKE_BUILDER = """
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
        (out / name).write_text(content, encoding="utf-8")
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
        content = path.read_text(encoding="utf-8")
        if PACK_ALTERS_BYTES and "prefill_decode" in raw:
            content += "!"
        files[re.sub(r"^Section\\d+_", "", os.path.basename(raw))] = content
    toml = re.sub(r'(data_path\\s*=\\s*")([^"]+)(")',
                  lambda m: m.group(1) + os.path.basename(m.group(2)) + m.group(3), toml)
    toml = re.sub(r'value = "[0-9a-f-]{{36}}"', 'value = "%s"' % _uuid.uuid4(), toml)
    if PACK_ADDS_KEY:
        toml = toml.replace('model_type = "embedder"', 'model_type = "embedder"\\nextra = "x"', 1)
    Path(output_path).write_bytes(MAGIC + json.dumps({{"toml": toml, "files": files}}).encode())
    return str(output_path)
"""


def bundle_bytes(toml: str) -> bytes:
    """A bundle in the fake's format: each section's file holds its own name."""
    import re

    names = [
        re.sub(r"^Section\d+_", "", os.path.basename(p))
        for p in re.findall(r'data_path\s*=\s*"([^"]+)"', toml)
    ]
    return (
        b"LITERTLM"
        + json.dumps({"toml": toml, "files": {n: f"bytes of {n}" for n in names}}).encode()
    )


def read_bundle(path: Path) -> dict:
    return json.loads(path.read_bytes()[len(b"LITERTLM") :].decode("utf-8"))


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
        ("pack_adds_key", "does not read back as the input minus those sections"),
        ("pack_alters_bytes", "a kept section's bytes changed"),
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
