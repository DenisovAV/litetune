"""`convert` of a Gemma 4 E2B checkpoint `tune` trained on pictures.

Driven by `test_export`'s `FakeToolchain`: the real repack script reads back a
fake bundle whose TOML (`repack_toml`) lists its sections and whose LlmMetadata
text (`metadata_extra`) carries Gemma 4's media fields and the sampler.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from conftest import buildable_here, mark_provisioned
from llm_image_fixtures import TEMPLATE, pin_template, write_checkpoint
from test_export import FAKE_TOML, FakeToolchain

from litetune import envs, models
from litetune.checks import Outcome
from litetune.export import REFUSED_SUFFIX, ExportRequest, run_export
from litetune.llm_images import (
    EXPORT_FLAGS,
    RECORD_KEY,
    TEMPLATE_FILE,
    VISION_TOKENS,
)

VISION_TOML = FAKE_TOML + "".join(
    f'\n[[section]]\nmodel_type = "{kind}"\nsection_type = "TFLiteModel"\n'
    f'data_path = "{kind}.tflite"\n'
    for kind in ("vision_encoder", "vision_adapter", "end_of_vision")
)


def _gemma4(patches: int = 2520, start: str | None = "<|image>", end: str | None = "<image|>"):
    fields: dict = {"max_num_patches": patches}
    if start is not None:
        fields["start_of_image_token"] = {"token_str": start}
    if end is not None:
        fields["end_of_image_token"] = {"token_str": end}
    return "llm_model_type: " + json.dumps({"gemma4": fields}) + "\n"


# SamplerParameters.Type numbers in litert-lm-builder 0.18.0: the proto gives the
# number, and the readback names it.
SAMPLER_TYPES = {"TOP_K": 1, "TOP_P": 2, "GREEDY": 3}


def _sampler(kind: str) -> str:
    fields = {"type": SAMPLER_TYPES[kind], "k": 1, "p": 0.95, "temperature": 1.0}
    return "sampler_params: " + json.dumps(fields) + "\n"


def _checkpoint(tmp_path: Path, template: str = TEMPLATE) -> Path:
    return write_checkpoint(tmp_path / "tuned" / "model", template=template)


def _request(tmp_path: Path, model: Path, **kwargs) -> ExportRequest:
    return ExportRequest(
        model=str(model), output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",), **kwargs
    )


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    pin_template(monkeypatch)


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    """`test_export`'s fake export environment."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    buildable_here(monkeypatch)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


@pytest.fixture
def vision_bundle(toolchain):
    toolchain.repack_toml = VISION_TOML
    toolchain.metadata_extra = _gemma4() + _sampler("GREEDY")
    return toolchain


def test_a_picture_checkpoint_exports_its_vision_encoder_from_its_template(tmp_path, vision_bundle):
    model = _checkpoint(tmp_path)
    request = _request(tmp_path, model)
    assert request.env is envs.EXPORT_VISION
    result = run_export(request)
    (export,) = result.exports
    assert export.ok, export.check.detail
    (call,) = vision_bundle.exports
    for flag in EXPORT_FLAGS:
        assert flag in call.argv
    # The file the pictures were trained on, not the repository the rule names.
    template = f"{models.TEMPLATE_FLAG}={(model / TEMPLATE_FILE).absolute()}"
    assert template in call.argv
    assert sum(a.startswith(models.TEMPLATE_FLAG) for a in call.argv) == 1
    assert export.vision == {
        "vision_encoder": True,
        "section_types": sorted(
            ["embedder", "end_of_vision", "prefill_decode", "vision_adapter", "vision_encoder"]
        ),
        "vision_tokens": VISION_TOKENS,
        "max_num_patches": 2520,
        "start_of_image_token": "<|image>",
        "end_of_image_token": "<image|>",
        "sampler": {"type": "GREEDY", "k": 1, "p": 0.95, "temperature": 1.0},
        "backend_constraints": {"vision_encoder": "cpu", "vision_adapter": "cpu"},
        "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
    }
    assert "takes pictures at 280 soft tokens" in export.check.detail
    report = result.as_dict()
    assert report["exports"][0]["vision"]["vision_encoder"] is True
    assert report["request"]["images"]["vision_tokens"] == VISION_TOKENS


def test_a_bundle_with_no_sampler_is_greedy_enough(tmp_path, vision_bundle):
    vision_bundle.metadata_extra = _gemma4()
    (export,) = run_export(_request(tmp_path, _checkpoint(tmp_path))).exports
    assert export.ok, export.check.detail
    assert export.vision is not None and export.vision["sampler"] is None


def test_a_text_checkpoint_exports_as_it_did(tmp_path, toolchain):
    model = tmp_path / "text"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "litetune.json").write_text(
        json.dumps({"base_model": "google/gemma-4-E2B-it"}), encoding="utf-8"
    )
    request = _request(tmp_path, model)
    assert request.env is envs.EXPORT
    (export,) = run_export(request).exports
    (call,) = toolchain.exports
    assert not any(flag in call.argv for flag in EXPORT_FLAGS)
    assert f"{models.TEMPLATE_FLAG}=litert-community/gemma-4-E2B-it-litert-lm" in call.argv
    assert export.vision is None


@pytest.mark.parametrize(
    "flag",
    [
        "--task=text_generation",
        "--export_vision_encoder=False",
        "--gemma4_vision_max_soft_tokens=140",
        "--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm",
    ],
)
def test_a_callers_setting_of_a_flag_litetune_owns_for_pictures_is_refused(tmp_path, flag):
    with pytest.raises(models.FlagRefused, match="trained on pictures"):
        _request(tmp_path, _checkpoint(tmp_path), extra_flags=(flag,))


def test_a_changed_template_exports_nothing(tmp_path, vision_bundle):
    model = _checkpoint(tmp_path, template="{{ another }}")
    result = run_export(_request(tmp_path, model))
    assert vision_bundle.exports == []
    refused = next(c for c in result.checks.checks if c.name == "pictures")
    assert refused.outcome is Outcome.FAILED
    assert "the template changed after `tune`" in refused.detail


def test_an_unreadable_picture_record_exports_nothing(tmp_path, vision_bundle):
    model = _checkpoint(tmp_path)
    data = json.loads((model / "litetune.json").read_text())
    data[RECORD_KEY]["vision_tokens"] = 140
    (model / "litetune.json").write_text(json.dumps(data), encoding="utf-8")
    result = run_export(_request(tmp_path, model))
    assert vision_bundle.exports == []
    refused = next(c for c in result.checks.checks if c.name == "pictures")
    assert "vision_tokens" in refused.detail


def _refused(tmp_path) -> tuple[str, Path]:
    (export,) = run_export(_request(tmp_path, _checkpoint(tmp_path))).exports
    assert export.check.outcome is Outcome.FAILED
    artifact = Path(export.check.observed["artifact"])
    assert artifact.with_name(artifact.name + REFUSED_SUFFIX).exists()
    assert not artifact.exists()
    return export.check.detail, artifact


def test_a_bundle_without_the_vision_sections_is_refused_and_set_aside(tmp_path, vision_bundle):
    vision_bundle.repack_toml = FAKE_TOML
    detail, _ = _refused(tmp_path)
    assert "carries no vision_encoder, vision_adapter, end_of_vision section" in detail


def test_a_bundle_at_another_budget_is_refused(tmp_path, vision_bundle):
    vision_bundle.metadata_extra = _gemma4(patches=1260)
    detail, _ = _refused(tmp_path)
    assert "max_num_patches 1260" in detail


@pytest.mark.parametrize("missing", ["start", "end"])
def test_a_bundle_without_the_image_markers_is_refused(tmp_path, vision_bundle, missing):
    vision_bundle.metadata_extra = _gemma4(**{missing: None})
    detail, _ = _refused(tmp_path)
    assert "as its start and end of image, where the processor puts '<|image>'" in detail


def test_a_bundle_naming_other_image_markers_is_refused(tmp_path, vision_bundle):
    vision_bundle.metadata_extra = _gemma4(start="<start_of_image>")
    detail, _ = _refused(tmp_path)
    assert "names '<start_of_image>' and '<image|>'" in detail


def test_the_vision_sections_are_constrained_to_the_cpu(tmp_path, vision_bundle):
    (export,) = run_export(_request(tmp_path, _checkpoint(tmp_path))).exports
    assert export.ok, export.check.detail
    # Read back from the rebuilt bundle, which the fake builder packs from the TOML.
    assert export.vision["backend_constraints"] == {
        "vision_encoder": "cpu",
        "vision_adapter": "cpu",
    }
    assert "constrained to the CPU" in export.check.detail


def test_a_bundle_whose_vision_sections_were_left_unconstrained_is_refused(tmp_path, vision_bundle):
    vision_bundle.repack_pack_fails = True
    detail, _ = _refused(tmp_path)
    assert "constrains its vision sections to" in detail


def test_a_text_bundle_is_not_constrained(tmp_path, toolchain):
    model = tmp_path / "text"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "litetune.json").write_text(
        json.dumps({"base_model": "google/gemma-4-E2B-it"}), encoding="utf-8"
    )
    toolchain.repack_toml = VISION_TOML
    (export,) = run_export(_request(tmp_path, model)).exports
    assert export.ok and export.vision is None


def test_the_sampler_a_bundle_declares_is_recorded_not_refused(tmp_path, vision_bundle):
    vision_bundle.metadata_extra = _gemma4() + _sampler("TOP_P")
    (export,) = run_export(_request(tmp_path, _checkpoint(tmp_path))).exports
    assert export.ok, export.check.detail
    assert export.vision is not None and export.vision["sampler"]["type"] == "TOP_P"
    assert "an application that passes no sampler of its own samples with it" in (
        export.check.detail
    )


def test_a_bundle_carrying_another_template_is_refused(tmp_path, vision_bundle):
    vision_bundle.template_embedded = lambda template: "{{ not the trained one }}"
    detail, _ = _refused(tmp_path)
    assert "not the" in detail and "the pictures were trained with" in detail


def test_a_bundle_whose_pictures_could_not_be_read_back_is_not_handed_on(tmp_path, vision_bundle):
    vision_bundle.repack_script_returncode = 1
    (export,) = run_export(_request(tmp_path, _checkpoint(tmp_path))).exports
    assert export.check.outcome is Outcome.UNCHECKED
    assert "could not be read back" in export.check.detail


def test_an_unreadable_record_beside_a_kept_template_exports_nothing(tmp_path, vision_bundle):
    model = _checkpoint(tmp_path)
    (model / "litetune.json").write_text("{not json", encoding="utf-8")
    result = run_export(_request(tmp_path, model, base_model="google/gemma-4-E2B-it"))
    assert vision_bundle.exports == []
    refused = next(c for c in result.checks.checks if c.name == "pictures")
    assert "keeps litetune_bundle_template.jinja" in refused.detail


def test_a_processor_file_that_changed_exports_nothing(tmp_path, vision_bundle):
    model = _checkpoint(tmp_path)
    (model / "processor_config.json").write_text('{"max_soft_tokens": 140}', encoding="utf-8")
    result = run_export(_request(tmp_path, model))
    assert vision_bundle.exports == []
    refused = next(c for c in result.checks.checks if c.name == "pictures")
    assert "processor_config.json hashes to" in refused.detail


def test_the_template_override_is_reported_as_litetunes(tmp_path, vision_bundle):
    request = _request(tmp_path, _checkpoint(tmp_path))
    said = [c.detail for c in request.plan.checks if models.TEMPLATE_FLAG in c.detail]
    assert said and all("was set by litetune" in d for d in said)
    assert not any("supplied by the caller" in d for d in said)
