"""Per-model rules: the flags a family requires, and the toolchain it needs.

No network, no toolchain, no model load. `StageEnv.run` is faked exactly as in
`test_export.py`: it returns a `CompletedProcess` and never raises on a non-zero
exit, because the distinction between "failed" and "was never performed" is what
the code under test is built to keep.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import FakeBackend, correct_texts, labelled_rows

from litetune import envs, models
from litetune.checks import Outcome
from litetune.evaluate import PromptMode
from litetune.export import ExportRequest, run_export
from litetune.models import (
    EXPORT_FLAGS_CHECK,
    TRANSFORMERS_CHECK,
    FlagRefused,
    identify,
    plan_export,
    transformers_check,
    version_tuple,
)
from litetune.tune import TuneRequest, run_tune
from litetune.verify import BackendPair, Status, VerifyRequest, run_verify

GEMMA4_E2B = "google/gemma-4-E2B-it"
FUNCTIONGEMMA = "google/functiongemma-270m-it"

E2B_TEMPLATE = "--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm"


# ---------------------------------------------------------------------------
# Identifying a family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, family",
    [
        (GEMMA4_E2B, "gemma-4-e2b"),
        ("litert-community/gemma-4-E2B-it-litert-lm", "gemma-4-e2b"),
        ("google/gemma-4-E4B-it", "gemma-4-e4b"),
        ("Gemma4_E4B", "gemma-4-e4b"),
        # No variant in the name: the family is known, the template repository
        # is not, and that difference is the point.
        ("google/gemma-4-it", "gemma-4"),
        ("Qwen/Qwen3.5-4B-Instruct", "qwen-3.5"),
    ],
)
def test_families_are_recognised(model, family):
    rules = identify(model)
    assert rules is not None
    assert rules.family == family


@pytest.mark.parametrize(
    "model",
    [
        # FunctionGemma and Gemma 3 used to sit here. They have rules now: both
        # declare `model_type: gemma3_text`, which the exporter does not
        # recognise, so without an override both bundle as `generic_model`.
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        # A size suffix is not the generation number: matching these would
        # refuse an export for a family these rules say nothing about.
        "org/gemma-40m",
        "org/gemma-4b-it",
        "Qwen/Qwen3-5B",
    ],
)
def test_a_family_with_no_rules_is_reported_as_unknown_not_as_fine(model):
    assert identify(model) is None
    record = models.report(model)
    assert record["known"] is False
    assert record["family"] is None
    assert "no per-model rules" in record["reason"]


def test_a_local_checkpoint_is_identified_from_its_config(tmp_path):
    # A merged checkpoint in `runs/out/model` carries no family in its path.
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "_name_or_path": GEMMA4_E2B}), encoding="utf-8"
    )
    rules = identify(str(checkpoint))
    assert rules is not None
    assert rules.family == "gemma-4-e2b"


def test_a_config_that_cannot_be_read_says_so_rather_than_reporting_no_rules(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{not json", encoding="utf-8")
    hint = models.hint_for(str(checkpoint))
    assert hint.config_error is not None
    assert models.report(str(checkpoint))["hint"]["config_error"] is not None


# ---------------------------------------------------------------------------
# Required flags
# ---------------------------------------------------------------------------


def test_gemma4_gets_both_required_flags_with_the_reason_stated():
    plan = plan_export(GEMMA4_E2B, (), ("dynamic_wi8_afp32",))
    assert plan.usable
    assert "--externalize_embedder" in plan.flags
    assert E2B_TEMPLATE in plan.flags
    assert set(plan.added) == {"--externalize_embedder", E2B_TEMPLATE}

    reasons = " ".join(plan.notes)
    # Not just "required": the observation behind each rule travels with it.
    assert "External embedder is required for Gemma4" in reasons
    assert "minijinja" in reasons
    assert "fails silently" in reasons
    assert all(check.outcome is Outcome.PASSED for check in plan.checks)
    assert all(check.name == EXPORT_FLAGS_CHECK for check in plan.checks)


def test_a_gemma4_export_without_the_override_is_refused_when_the_variant_is_unknown():
    # The value is per-variant and this id does not say which. Guessing produces
    # a bundle whose template the runtime cannot render, and it fails silently.
    plan = plan_export("google/gemma-4-it", (), ("dynamic_wi8_afp32",))
    assert not plan.usable
    unchecked = [c for c in plan.checks if c.outcome is Outcome.UNCHECKED]
    assert len(unchecked) == 1
    assert "--jinja_chat_template_override" in unchecked[0].detail
    assert "will not guess" in unchecked[0].detail


def test_a_caller_supplied_override_satisfies_the_requirement():
    plan = plan_export("google/gemma-4-it", ("--jinja_chat_template_override=me/my-template",), ())
    assert plan.usable
    assert "--jinja_chat_template_override=me/my-template" in plan.flags
    # The one flag litetune can resolve on its own is still added.
    assert plan.added == ("--externalize_embedder",)


def test_a_caller_who_disagrees_with_litetune_keeps_their_value_and_it_is_recorded():
    plan = plan_export(GEMMA4_E2B, ("--jinja_chat_template_override=me/mine",), ())
    assert "--jinja_chat_template_override=me/mine" in plan.flags
    assert E2B_TEMPLATE not in plan.flags
    detail = " ".join(c.detail for c in plan.checks)
    assert "litetune would have used" in detail
    assert "kept" in detail


def test_the_model_type_override_is_refused_with_an_explanation():
    with pytest.raises(FlagRefused) as exc:
        plan_export(GEMMA4_E2B, ("--litert_lm_model_type_override=gemma4",), ())
    message = str(exc.value)
    assert "refuses" in message
    # The refusal has to say what the flag actually does, or it reads as
    # pedantry and gets removed.
    assert "generic_model" in message
    assert "metadata builder" in message


def test_an_export_request_cannot_be_built_with_a_refused_flag(tmp_path):
    # Structural: no code path -- CLI, library or a composed run -- can pass it.
    with pytest.raises(FlagRefused):
        ExportRequest(
            model=GEMMA4_E2B,
            output_dir=tmp_path,
            recipes=("dynamic_wi8_afp32",),
            extra_flags=("--litert_lm_model_type_override=gemma4",),
        )


def test_an_export_request_carries_the_required_flags_into_its_argv(tmp_path):
    request = ExportRequest(model=GEMMA4_E2B, output_dir=tmp_path, recipes=("dynamic_wi8_afp32",))
    argv = request.argv("dynamic_wi8_afp32")
    assert "--externalize_embedder" in argv
    assert E2B_TEMPLATE in argv


def test_a_family_with_no_rules_has_its_flags_left_alone(tmp_path):
    plan = plan_export("Qwen/Qwen3-0.6B", ("--some_flag=1",), ("dynamic_wi8_afp32",))
    assert plan.flags == ("--some_flag=1",)
    assert plan.added == ()
    assert plan.usable
    assert "no per-model rules" in " ".join(plan.notes)


def test_functiongemma_gets_the_model_type_its_runtime_needs():
    """Its config says `gemma3_text`, which the exporter does not recognise.

    The exporter matches `config.model_type` against a fixed list and drops
    anything else into `generic_model` with no warning. `gemma3_text` is not on
    that list, so the model whose *name* is FunctionGemma bundles as generic --
    and a generic bundle gets no tool-call channel and silently loses
    constrained decoding. Google's own published artifact for this model
    declares `function_gemma`; ours declared `generic_model` until this rule.
    """
    plan = plan_export(FUNCTIONGEMMA, (), ("dynamic_wi8_afp32",))

    assert "--litert_lm_model_type_override=function_gemma" in plan.flags
    assert plan.usable


def test_plain_gemma_3_gets_a_different_value_from_the_same_config():
    """Both families declare `gemma3_text`; only their identity separates them.

    Which is why this cannot be derived from the config and has to be a rule.
    """
    plan = plan_export("google/gemma-3-270m-it", (), ("dynamic_wi8_afp32",))

    assert "--litert_lm_model_type_override=gemma3" in plan.flags


# ---------------------------------------------------------------------------
# Recipes: a recommendation, never a substitution
# ---------------------------------------------------------------------------


def test_a_recipe_recommendation_does_not_change_the_recipe():
    plan = plan_export(GEMMA4_E2B, (), ("dynamic_wi8_afp32",))
    assert len(plan.recommendations) == 1
    text = plan.recommendations[0]
    assert "dynamic_wi4c_hr_afp32" in text
    assert "dynamic_wi4b32_afp32" in text
    assert "recommendation and not a substitution" in text


def test_no_recommendation_when_a_recommended_recipe_was_asked_for():
    assert plan_export(GEMMA4_E2B, (), ("dynamic_wi4b32_afp32",)).recommendations == ()


def test_the_ceiling_is_recorded_as_a_limitation():
    limitations = models.limitations_for(GEMMA4_E2B)
    assert len(limitations) == 1
    assert "NOT equivalent to Google's published" in limitations[0]
    assert "QAT" in limitations[0]
    assert models.limitations_for(FUNCTIONGEMMA) == []


# ---------------------------------------------------------------------------
# transformers versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5.0.0", (5, 0, 0)),
        ("4.57.3", (4, 57, 3)),
        ("v5.1", (5, 1)),
        ("5.0.0.dev0", (5, 0, 0)),
        ("not-a-version", None),
    ],
)
def test_version_parsing(text, expected):
    assert version_tuple(text) == expected


def test_a_too_old_transformers_is_a_named_failure_not_a_traceback():
    rules = identify(GEMMA4_E2B)
    check = transformers_check(GEMMA4_E2B, rules, "4.57.3", "the export environment")
    assert check.outcome is Outcome.FAILED
    assert check.name == TRANSFORMERS_CHECK
    # The model, the version installed and the version needed, all named.
    assert GEMMA4_E2B in check.detail
    assert "4.57.3" in check.detail
    assert "5.5.0" in check.detail
    # And the signature of the failure it replaces, so the message is
    # recognisable to anyone who has already hit it.
    assert "'list' object has no attribute 'keys'" in check.detail
    assert check.observed["installed"] == "4.57.3"


# 5.5.0 is the floor now: the catalogue measured two minimums -- 5.0.0 to load
# the tokenizer, 5.5.0 for AutoConfig to recognise the `gemma4` architecture --
# and the code encoded the lower one, so 5.2.0 passed a check for a model that
# dies on load.
@pytest.mark.parametrize("installed", ["5.5.0", "5.6.2", "6.0.0"])
def test_a_new_enough_transformers_passes(installed):
    check = transformers_check(GEMMA4_E2B, identify(GEMMA4_E2B), installed, "the environment")
    assert check.outcome is Outcome.PASSED


@pytest.mark.parametrize("installed", [None, "", "nightly"])
def test_a_version_that_could_not_be_read_is_could_not_check(installed):
    check = transformers_check(GEMMA4_E2B, identify(GEMMA4_E2B), installed, "the environment")
    assert check.outcome is Outcome.UNCHECKED
    assert "5.5.0" in check.detail


def test_the_declared_pin_is_read_from_a_requirement_list():
    assert models.declared_version(envs.TRAIN.requirements) == "5.16.1"
    assert models.declared_version(("Transformers==5.0.0",)) == "5.0.0"
    assert models.declared_version(("torch==2.5.1",)) is None


# ---------------------------------------------------------------------------
# The rules where the stages consult them
# ---------------------------------------------------------------------------


@dataclass
class FakeToolchain:
    """Stands in for `envs.StageEnv.run`, as in test_export.py."""

    pip_stdout: str = "transformers==5.5.0\nlitert-torch-nightly==0.10.0.dev20260826\n"
    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, 0, self.pip_stdout, "")
        flags = dict(a.removeprefix("--").split("=", 1) for a in args[2:] if "=" in a)
        out_dir = Path(flags["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.litertlm").write_bytes(b"\0" * 4096)
        return subprocess.CompletedProcess(args, 0, "", "")

    @property
    def exports(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "litert-torch"]


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".litetune-ready").write_text(self.identity)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


def test_convert_auto_corrects_a_gemma4_export_and_says_why(toolchain, tmp_path):
    result = run_export(
        ExportRequest(
            model=GEMMA4_E2B, output_dir=tmp_path / "out", recipes=("dynamic_wi4b32_afp32",)
        )
    )
    assert result.outcome is Outcome.PASSED
    argv = toolchain.exports[0]
    assert "--externalize_embedder" in argv
    assert E2B_TEMPLATE in argv
    # The correction is in the report, not only in the command line.
    added = [c for c in result.checks.checks if c.name == EXPORT_FLAGS_CHECK]
    assert len(added) == 2
    assert any("External embedder is required for Gemma4" in c.detail for c in added)
    assert any(_GOOGLES_ARTIFACT in text for text in result.limitations)


_GOOGLES_ARTIFACT = "NOT equivalent to Google's published"


def test_convert_refuses_to_export_when_the_template_override_is_undeterminable(
    toolchain, tmp_path
):
    result = run_export(
        ExportRequest(
            model="google/gemma-4-it", output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",)
        )
    )
    assert result.outcome is Outcome.UNCHECKED
    assert result.not_attempted == ("dynamic_wi8_afp32",)
    # Nothing ran, so nothing failed: the recipes are not recorded as failures.
    assert result.exports == []
    assert toolchain.exports == []


def test_convert_stops_on_a_too_old_transformers_rather_than_running_into_it(toolchain, tmp_path):
    toolchain.pip_stdout = "transformers==4.57.6\n"
    result = run_export(
        ExportRequest(
            model=GEMMA4_E2B, output_dir=tmp_path / "out", recipes=("dynamic_wi4b32_afp32",)
        )
    )
    assert result.outcome is Outcome.FAILED
    assert result.not_attempted == ("dynamic_wi4b32_afp32",)
    assert toolchain.exports == []
    version = next(c for c in result.checks.checks if c.name == TRANSFORMERS_CHECK)
    assert version.outcome is Outcome.FAILED
    assert "4.57.6" in version.detail


def test_convert_still_runs_when_the_version_could_not_be_read(toolchain, tmp_path):
    # A pip that will not answer says nothing about the artifact. Turning that
    # into "could not check" would bury a perfectly good export.
    toolchain.pip_stdout = ""
    result = run_export(
        ExportRequest(
            model=GEMMA4_E2B, output_dir=tmp_path / "out", recipes=("dynamic_wi4b32_afp32",)
        )
    )
    assert result.outcome is Outcome.PASSED
    assert len(toolchain.exports) == 1
    assert any("transformers>=5.5.0" in text for text in result.limitations)


def test_the_shipped_training_environment_can_tokenize_every_model_in_the_table():
    """The rule must not indict the environment that ships with it.

    envs.TRAIN pinned 4.57.3 when this rule was written, which is inside the
    broken 4.55.0-4.57.6 range -- so litetune could not have tuned the models
    its own table describes. The pin was raised to 5.16.1 after a six-model
    probe showed the three previously measured families unchanged on it.
    """
    declared = models.declared_version(envs.TRAIN.requirements)
    for model_id in (GEMMA4_E2B, "Qwen/Qwen3.5-0.8B"):
        check = transformers_check(
            model_id, identify(model_id), declared, "the training environment"
        )
        assert check.outcome is Outcome.PASSED, (model_id, check.detail)


def test_tune_refuses_a_model_the_training_environment_cannot_tokenize(
    toolchain, tmp_path, monkeypatch
):
    """A too-old pin must stop the run before any training happens.

    The version is forced rather than taken from envs.TRAIN: pinning the test to
    whatever ships would make it pass or fail for reasons unrelated to the rule.
    """
    monkeypatch.setattr(models, "declared_version", lambda _requirements: "4.57.3")
    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "a", "completion": "call:a{}"}\n', encoding="utf-8")
    result = run_tune(
        TuneRequest(
            model=GEMMA4_E2B,
            data=data,
            output_dir=tmp_path / "run",
            prompt_mode=PromptMode.PRERENDERED,
        )
    )
    assert result.outcome is Outcome.FAILED
    version = next(c for c in result.checks.checks if c.name == TRANSFORMERS_CHECK)
    assert version.outcome is Outcome.FAILED
    assert "4.57.3" in version.detail
    # Nothing was trained: no python was run in the environment.
    assert [c for c in toolchain.calls if c[0] == "python"] == []


# ---------------------------------------------------------------------------
# verify reports the rules
# ---------------------------------------------------------------------------


def test_verify_reports_the_family_and_its_recorded_ceiling(write_split):
    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(model=Path("model.litertlm"), reference=GEMMA4_E2B, data=write_split(rows)),
        backends=BackendPair(
            candidate=FakeBackend(texts=correct_texts(rows)),
            reference=FakeBackend(model=GEMMA4_E2B, texts=correct_texts(rows)),
        ),
    )
    record = result.manifest["model_rules"]
    assert record["family"] == "gemma-4-e2b"
    assert record["recommended_recipes"] == ["dynamic_wi4c_hr_afp32", "dynamic_wi4b32_afp32"]
    assert any(_GOOGLES_ARTIFACT in text for text in result.manifest["limitations"])


def test_verify_refuses_a_reference_whose_environment_cannot_load_it(write_split):
    class OldTransformers(FakeBackend):
        def describe(self) -> dict:
            return {"engine": "fake", "backend": "cpu", "requirements": ["transformers==4.57.3"]}

    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(model=Path("model.litertlm"), reference=GEMMA4_E2B, data=write_split(rows)),
        backends=BackendPair(
            candidate=FakeBackend(texts=correct_texts(rows)),
            reference=OldTransformers(model=GEMMA4_E2B, texts=correct_texts(rows)),
        ),
    )
    # A harness fact, not a verdict about the model.
    assert result.status is Status.FAILED_HARNESS
    assert result.exit_code == 4
    version = next(c for c in result.manifest["checks"] if c["name"] == TRANSFORMERS_CHECK)
    assert version["outcome"] == "failed"


def test_functiongemma_declares_the_terminator_training_cannot_reveal():
    """A FunctionGemma turn does not end at the call.

    After `<end_function_call>` the application executes the tool and sends the
    result back, so the model must stop and wait. Training cannot reveal that
    terminator: the completions end at `<end_of_turn>`, so a run records that
    one and nothing else. Google's published bundle for this model declares
    both -- read with `litertlm_peek` -- while ours declared seventeen
    auto-derived punctuation variants of `<end_of_turn>\\n` and not this one.

    The `.litertlm` does carry it -- as token id 50, out of
    `generation_config.eos_token_id`. This is about `contract.json`, which a
    consumer reads without a protobuf parser and which otherwise names only the
    terminator the run observed.

    Keyed on model identity because it is not in `config.json`: plain Gemma 3
    shares the architecture and has no function-response channel at all.
    """
    from litetune.models import stop_tokens_for

    tokens, reason = stop_tokens_for("google/functiongemma-270m-it")
    assert tokens == ("<start_function_response>",)
    assert "application has to execute the tool" in reason
    assert "token id 50" in reason, "the reason must not claim the bundle lacks it"

    assert stop_tokens_for("google/gemma-3-270m-it") == ((), "")
    assert stop_tokens_for("some/unknown-model") == ((), "")
