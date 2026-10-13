"""Per-model rules: the flags a family requires, and the toolchain it needs.

No network, no toolchain, no model load. `StageEnv.run` is faked exactly as in
`test_export.py`: it returns a `CompletedProcess` and never raises on a non-zero
exit, because the distinction between "failed" and "was never performed" is what
the code under test is built to keep.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import (
    FakeBackend,
    buildable_here,
    correct_texts,
    labelled_rows,
    mark_provisioned,
    run_compose_script,
)
from minijinja_runtime import minijinja_render

from litetune import envs, models
from litetune.checks import Outcome
from litetune.export import ExportRequest, run_export
from litetune.models import (
    EXPORT_FLAGS_CHECK,
    TRANSFORMERS_CHECK,
    FlagRefused,
    identify,
    plan_export,
    renders_declarations_for,
    transformers_check,
    version_tuple,
    wire_format_for,
)
from litetune.prompt_mode import PromptMode
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
        ("Qwen/Qwen3-0.6B", "qwen-3"),
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
        # Qwen3-0.6B sat here until its export and conversion cost were
        # measured, and Qwen2.5-0.5B-Instruct until 2026-09-20. Both rules now
        # compose the checkpoint's own template behind the text-parts block at
        # export; the rest of both families stays below.
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        # Claimed by size, and only the size that was measured. The TTS model is
        # here because its name ends in 0.6B too.
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-VL-2B-Instruct",
        "litert-community/Qwen3-TTS-12Hz-0.6B-Base",
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


def test_a_local_checkpoint_is_identified_from_what_tune_recorded(tmp_path):
    """A merged checkpoint in `runs/out/model` carries no family in its path.

    It used to carry one in `config.json`: `from_pretrained` sets
    `_name_or_path` to the id you loaded, and this test hand-wrote that key.
    transformers 5.x deletes it on save -- `to_diff_dict` drops it before
    serialising -- so no checkpoint `tune` produces has ever had it, and the
    test was verifying a contract upstream had already broken.

    The identity now travels in a sidecar `tune` writes. The `config.json` here
    is what transformers 5.x actually emits, so this fails if the sidecar stops
    being read.
    """
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": GEMMA4_E2B}), encoding="utf-8"
    )
    rules = identify(str(checkpoint))
    assert rules is not None
    assert rules.family == "gemma-4-e2b"


def test_a_qwen3_checkpoint_tune_wrote_is_claimed_through_its_sidecar(tmp_path):
    """What `convert` was handed in the 2026-09-14 measurement: `runs/tuned/model`.

    The config and sidecar below have the shape that run's `tune` wrote. Its
    export printed "no per-model rules", because there were none.
    """
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        encoding="utf-8",
    )
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": "Qwen/Qwen3-0.6B"}), encoding="utf-8"
    )
    rules = identify(str(checkpoint))
    assert rules is not None
    assert rules.family == "qwen-3"


def test_gemma_4_declares_the_stop_tokens_its_generation_config_names():
    """Gemma 4 closes a turn with `<turn|>`, not `<end_of_turn>`.

    Until this was declared, `metrics.TERMINATORS` did not carry it, and a
    600-row `verify` against `google/gemma-4-E2B-it` refused every comparison:
    the reference generations ended in a marker scoring did not recognise, 600
    of 600. `generation_config.json` names eos_token_id [1, 106, 50] and
    `tokenizer.json` names 106 `<turn|>` and 50 `<|tool_response>`, the stop
    reached once the model has called a tool.

    Asserted per variant rather than once: `_gemma4` builds three, and a tuple
    dropped from the factory would leave all three silent while every other
    test stayed green.
    """
    for family in ("gemma-4-e2b", "gemma-4-e4b", "gemma-4"):
        rules = next(r for r in models.RULES if r.family == family)
        assert rules.extra_stop_tokens == ("<turn|>", "<|tool_response>"), family
        assert rules.stop_token_reason, f"{family} declares stop tokens with no evidence"


def test_gemma_4_scopes_lora_to_its_text_tower():
    """A LoRA run on Gemma 4 must adapt the text tower and nothing else.

    peft matches `target_modules` by name suffix, and this checkpoint's vision
    and audio towers use the same projection names as its text layers, so a run
    scoped by name alone reaches all three, and peft then refuses the tower
    ones: transformers wraps them in `Gemma4ClippableLinear` and peft
    dispatches on a bare `nn.Linear`, so the run stops in `get_peft_model`.
    The scope is what lets the projection set apply at all.

    Per variant, because `_gemma4` builds three and one dropped argument would
    leave all three unscoped while every other test stayed green.
    """
    for family in ("gemma-4-e2b", "gemma-4-e4b", "gemma-4"):
        rules = next(r for r in models.RULES if r.family == family)
        assert rules.lora_container == "language_model", family
        assert rules.lora_container_reason, f"{family} scopes LoRA with no evidence"


def test_only_a_multimodal_family_scopes_lora_at_all():
    """The container is a claim about structure, so a text-only family makes none.

    The scoped set alone does not hold this. `lora_container` defaults to
    `None`, so a multimodal family added without filling it in leaves the
    scoped set unchanged and passes -- adapting every tower, reporting cleanly,
    with no limitation and a green suite, which is the pre-scope behaviour
    reached through the path of least resistance. Found by reviewing the commit
    that added the check.

    So the whole roster is asserted, not the scoped part of it. Adding a family
    fails here and the author has to decide the question before the list can be
    updated. That is the only moment anyone is looking at the checkpoint's
    module graph.
    """
    families = {r.family for r in models.RULES}
    assert families == {
        "functiongemma",
        "gemma-3-text",
        "gemma3-text-unidentified",
        "gemma-4",
        "gemma-4-e2b",
        "gemma-4-e4b",
        "qwen-2.5",
        "qwen-3",
        "qwen-3.5",
    }, sorted(families)
    scoped = {r.family for r in models.RULES if r.lora_container}
    assert scoped == {"gemma-4-e2b", "gemma-4-e4b", "gemma-4"}, sorted(scoped)
    # And every family says why, including the ones with no container. The
    # reason used to be asserted *empty* there, which made "examined, and it
    # has one tower" and "nobody decided" arrive as the same empty string --
    # in `ModelRules.as_dict`, which every convert and verify manifest
    # publishes. An entry cannot leave the question at its default now.
    for rules in models.RULES:
        assert rules.lora_container_reason, rules.family


def test_the_scope_reason_says_what_an_unscoped_run_actually_does():
    """It refuses; it does not train the wrong thing quietly.

    The first draft of this string said an unscoped run "trains, it saves, and
    every check passes". Executed against peft 0.20.0 with the module shapes
    transformers 5.16.1 gives this family, it raises in `get_peft_model`: the
    tower projections are `Gemma4ClippableLinear` and peft dispatches on a bare
    `nn.Linear`. The quiet version is the more frightening claim and the false
    one, so it is the one worth pinning against.
    """
    reason = models.identify("google/gemma-4-E2B-it").lora_container_reason

    assert "Gemma4ClippableLinear" in reason
    assert "get_peft_model" in reason
    assert "trains" not in reason


def test_the_family_report_names_the_lora_container():
    """The scope has to reach a record a user reads.

    `ModelRules.as_dict` is what `convert --json` and every verify manifest
    publish under `model_rules`, and README points a reader there. Dropping
    either key from it changed nothing any test could see.
    """
    record = models.report("google/gemma-4-E2B-it")

    assert record["lora_container"] == "language_model"
    assert "same projection names" in record["lora_container_reason"]


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
    plan = plan_export("Qwen/Qwen2.5-1.5B-Instruct", ("--some_flag=1",), ("dynamic_wi8_afp32",))
    assert plan.flags == ("--some_flag=1",)
    assert plan.added == ()
    assert plan.usable
    assert "no per-model rules" in " ".join(plan.notes)


QWEN_RULES = [("Qwen/Qwen3-0.6B", "qwen-3"), ("Qwen/Qwen2.5-0.5B-Instruct", "qwen-2.5")]
# The template override each Qwen rule adds: Qwen 3 may get LiteRT-LM's own.
QWEN_MARKERS = {
    "qwen-3": models.CANONICAL_OR_COMPOSED_TEMPLATE,
    "qwen-2.5": models.COMPOSED_TEMPLATE,
}


@pytest.mark.parametrize(("model", "family"), QWEN_RULES)
def test_a_qwen_export_asks_for_its_own_template_composed_and_no_type_override(model, family):
    """The exporter types both Qwen checkpoints from their own `config.json`
    (measured 2026-09-14 and 2026-09-20), so no type override. What LiteRT-LM
    0.18.0 needs is a template that reads content as a list of parts: on it a
    Qwen3 bundle carrying the checkpoint's template rendered an empty user turn
    with no error (2026-10-08). So the plan adds exactly that one flag, as the
    marker `run_export` composes from the checkpoint's own template, and says
    why."""
    plan = plan_export(model, ("--some_flag=1",), ("dynamic_wi8_afp32",))
    assert plan.rules is not None
    assert plan.rules.family == family
    assert plan.flags[0] == "--some_flag=1"
    assert len(plan.flags) == 2
    assert plan.flags[1] == f"--jinja_chat_template_override={QWEN_MARKERS[family]}"
    assert plan.added == (plan.flags[1],)
    assert not any("--litert_lm_model_type_override" in f for f in plan.flags)
    assert plan.usable
    assert models.UNKNOWN_FAMILY not in plan.notes
    assert any("0.18.0" in note and "list of" in note for note in plan.notes)


def test_no_checkpoints_template_ships_in_the_package():
    """A checkpoint's template is read at export, never packaged: the package
    carries litetune's block, FunctionGemma's template and LiteRT-LM's Qwen 3
    template, which the qwen-3 rule hands over only when it renders single
    turns as the checkpoint's own does."""
    templates = Path(models.packaged_template(models.TEXT_PARTS_BLOCK)).parent
    assert sorted(p.name for p in templates.iterdir()) == [
        "functiongemma.jinja",
        models.QWEN3_CANONICAL_TEMPLATE,
        models.TEXT_PARTS_BLOCK,
    ]


def test_the_text_parts_block_can_print_nothing():
    """Put in front of a template, the block must not add a character to the
    prompt: every line is a statement or a comment, each trimming the
    whitespace around it, and it ends on a comment that leaves the template's
    first character alone."""
    import pathlib
    import re

    block = pathlib.Path(models.packaged_template("text_parts.jinja")).read_text(encoding="utf-8")
    lines = block.split("\n")
    assert lines[0].startswith("{#- litetune:") and lines[0].endswith("-#}")
    assert lines[-1].startswith("{#- litetune: end") and lines[-1].endswith(" #}")
    for line in lines[1:-1]:
        # One statement per line: `.*` alone would also accept
        # `{%- if x -%}TEXT{%- endif -%}`, which prints TEXT.
        assert re.fullmatch(r"\{%-(?:(?!%\}).)*-%\}", line), line
    assert "{{" not in block


# MiniJinja as LiteRT-LM v0.18.0 renders with it (`minijinja_runtime.py`).
_runtime_render = minijinja_render


def _as_parts(messages: list[dict]) -> list[dict]:
    """What v0.18.0 hands the template: each content a list of text parts
    (`NormalizeMessageContent`, `data_utils.cc`)."""
    return [dict(m, content=[{"type": "text", "text": m["content"]}]) for m in messages]


_CONVERSATIONS = [
    [{"role": "user", "content": "What is my balance?"}],
    [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
    [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ],
    # An assistant turn that carries its reasoning: Qwen3's template splits and
    # strips it, which only the runtime's rewrite makes MiniJinja accept.
    [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nb"},
        {"role": "user", "content": "c"},
    ],
]

_FIXTURES = Path(__file__).parent / "fixtures"

# sha256 of each checkpoint's `chat_template` in `tokenizer_config.json` at the
# revision the fixture names, read from the Hugging Face cache for litetune
# 0.4.0, which packaged each one behind the block. The fixtures are those
# templates, cut from the 0.4.0 files after the block, byte for byte.
_CHECKPOINT_TEMPLATE_SHA256 = {
    "qwen3-0.6b-c1899de.chat_template.jinja": (
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    ),
    "qwen2.5-0.5b-instruct-7ae5576.chat_template.jinja": (
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
    ),
}


def _block() -> str:
    return Path(models.packaged_template(models.TEXT_PARTS_BLOCK)).read_bytes().decode("utf-8")


def _composed(fixture: str) -> tuple[str, str]:
    """What `export.compose_chat_template` writes for a checkpoint carrying
    `fixture` -- the block, then the template unchanged -- and the template."""
    body = (_FIXTURES / fixture).read_bytes().decode("utf-8")
    return _block() + body, body


def test_the_render_harness_writes_tojson_as_the_runtime_does():
    """serde_json writes UTF-8 and spaces after `,` and `:` (`minijinja_template.rs`)."""
    rendered = _runtime_render(
        "{{ messages[0].content | tojson }}",
        [{"role": "user", "content": {"city": "München", "n": [1, 2]}}],
    )
    assert rendered == '{"city": "München", "n": [1, 2]}'


@pytest.mark.parametrize("fixture", [*sorted(_CHECKPOINT_TEMPLATE_SHA256), None])
def test_a_template_rendered_here_uses_nothing_the_render_harness_gets_wrong(fixture):
    """`_runtime_render` cannot reproduce the runtime's `none` test, and its
    `strftime_now` is Python's: a template that used either would be checked
    under different semantics from the device's. `None` is the block alone."""
    import re

    text = _composed(fixture)[0] if fixture else _block()
    assert not re.search(r"\bis\s+(?:not\s+)?none\b", text)
    assert "strftime_now" not in text


@pytest.mark.parametrize("fixture", sorted(_CHECKPOINT_TEMPLATE_SHA256))
def test_each_fixture_is_the_checkpoints_template_byte_for_byte(fixture):
    import hashlib

    body = (_FIXTURES / fixture).read_bytes()
    assert hashlib.sha256(body).hexdigest() == _CHECKPOINT_TEMPLATE_SHA256[fixture]


@pytest.mark.parametrize("fixture", sorted(_CHECKPOINT_TEMPLATE_SHA256))
@pytest.mark.parametrize("conversation", _CONVERSATIONS)
def test_a_composed_template_renders_parts_as_the_checkpoints_renders_strings(
    fixture, conversation
):
    """The point of the block, rendered the way the runtime renders: given the
    list of parts 0.18.0 hands over, the block followed by the checkpoint's
    template prints what that template alone prints for the string
    `transformers` and 0.17.1 hand over -- and the string still renders the
    same through it."""
    full, body = _composed(fixture)
    trained_on = _runtime_render(body, conversation)
    assert _runtime_render(full, _as_parts(conversation)) == trained_on
    assert _runtime_render(full, conversation) == trained_on


@pytest.mark.parametrize("fixture", sorted(_CHECKPOINT_TEMPLATE_SHA256))
def test_without_the_block_the_parts_lose_the_users_text(fixture):
    """The check above can fail: the checkpoint's template alone, given parts,
    is the 0.18.0 behaviour measured on 2026-10-08 -- Qwen3 renders an empty
    user turn, Qwen2.5 raises -- and neither prints the user's text."""
    _, body = _composed(fixture)
    conversation = _CONVERSATIONS[0]
    try:
        rendered = _runtime_render(body, _as_parts(conversation))
    except Exception as error:  # noqa: BLE001 -- MiniJinja's own error type
        assert "unsupported types string and sequence" in str(error)
    else:
        assert conversation[0]["content"] not in rendered
        assert "<|im_start|>user\n<|im_end|>" in rendered


@pytest.mark.parametrize(
    ("content", "seen"),
    [
        ("a string", "a string"),
        ([{"type": "text", "text": "one"}], "one"),
        ([{"type": "text", "text": "one "}, {"type": "text", "text": "two"}], "one two"),
        ([], ""),
        # A part that is not text: the whole message is left as it came.
        (
            [{"type": "text", "text": "see"}, {"type": "image", "path": "x.png"}],
            [{"type": "text", "text": "see"}, {"type": "image", "path": "x.png"}],
        ),
        (None, None),
    ],
)
def test_the_text_parts_block_gives_text_parts_back_as_one_string(content, seen):
    probe = _block() + "{{ messages[0].content | tojson }}"
    rendered = _runtime_render(probe, [{"role": "user", "content": content}])
    assert json.loads(rendered) == seen


def _block_sees(content):
    rendered = _runtime_render(
        _block() + "{{ messages[0].content | tojson }}", [{"role": "user", "content": content}]
    )
    return json.loads(rendered)


def test_a_part_that_is_not_text_is_not_read_as_text_even_with_a_text_field():
    content = [
        {"type": "image", "path": "x.png", "text": "caption"},
        {"type": "text", "text": "see"},
    ]
    assert _block_sees(content) == content


def test_a_text_part_whose_text_is_not_a_string_is_left_as_it_came():
    content = [{"type": "text", "text": 7}]
    assert _block_sees(content) == content


_TOOL_CONVERSATION = [
    {"role": "user", "content": "Wake me at 7"},
    {
        "role": "assistant",
        "content": "Setting it.",
        "tool_calls": [
            {"type": "function", "function": {"name": "set_alarm", "arguments": {"hour": 7}}}
        ],
    },
    {"role": "tool", "content": "done"},
    {"role": "user", "content": "thanks"},
]


@pytest.mark.parametrize("fixture", sorted(_CHECKPOINT_TEMPLATE_SHA256))
def test_an_assistant_turn_in_parts_keeps_its_tool_calls(fixture):
    """What v0.18.0 keeps in history for a reply with a call: text parts and
    `tool_calls` (`ResponseTextToMessage`, `data_utils.cc`). The block hands
    every other key of the message on. (The tool's own reply is a string here;
    sent as a `tool_response` part it stays a list, which README says.)"""
    full, body = _composed(fixture)
    trained_on = _runtime_render(body, _TOOL_CONVERSATION)
    assert "set_alarm" in trained_on
    assert _runtime_render(full, _as_parts(_TOOL_CONVERSATION)) == trained_on


@pytest.mark.parametrize(("model", "family"), QWEN_RULES)
def test_the_reason_says_the_checkpoints_own_template_is_kept(model, family):
    """No revision is named: the template is whichever the checkpoint carries."""
    plan = plan_export(model, (), ("dynamic_wi8_afp32",))
    assert plan.rules is not None
    (flag,) = plan.rules.required_flags
    assert "reads the checkpoint's own chat template as the exporter reads it" in flag.reason
    assert "reads each bundle's template back" in flag.reason
    assert "with and without a system message, the text in one part and in two" in flag.reason
    assert "configured as LiteRT-LM's MiniJinja, which is not the runtime" in flag.reason
    assert "verify's rendering check is what checks the runtime, on single turns" in flag.reason
    assert "past its first message, in the contexts checked, is checked after export" in flag.reason
    assert "enable_thinking undefined, true and false, with no tools and with one" in flag.reason
    assert "a template using what that renderer does not vouch for is not checked" in flag.reason
    assert "revision" not in flag.reason
    if family == "qwen-2.5":
        assert "keeps its own template, or no bundle is handed over" in flag.reason
    else:
        assert "LiteRT-LM v0.18.0 publishes its own Qwen 3 template" in flag.reason
        assert "trained runtime_rendered without tool declarations" in flag.reason
        assert "no litetune.json and the checkpoint's template is Qwen/Qwen3-0.6B's own" in (
            flag.reason
        )
        assert "a prerendered run, a run with declarations" in flag.reason
        assert "no record and any other template" in flag.reason
        assert models.QWEN3_CANONICAL_TEMPLATE in flag.reason
        assert "google-ai-edge/LiteRT-LM#3443" in flag.reason


def test_an_unknown_family_is_told_about_the_list_of_parts():
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    notes = " ".join(plan_export(model, (), ("dynamic_wi8_afp32",)).notes)
    assert "no per-model rules" in notes
    assert "from LiteRT-LM 0.18.0" in notes and "list of parts" in notes and "run verify" in notes
    assert "list of parts" in models.report(model)["reason"]


_SNAPSHOT = "/home/u/.cache/huggingface/hub/models--Qwen--{}/snapshots/" + "c1899de2" * 5


# Repacks named for their format, which the export environment cannot load as
# a full-precision model or which hold no PyTorch weights (`models.py`, the
# qwen-3 rule).
_QWEN3_REPACKS = (
    "Qwen/Qwen3-0.6B-FP8",
    "Qwen/Qwen3-0.6B-MLX-4bit",
    "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
    "Qwen/Qwen3-0.6B-GGUF",
    "onnx-community/Qwen3-0.6B-ONNX",
    "someorg/Qwen3-0.6B-Qwen3-GGUF",
    "Qwen/Qwen3-0.6B-GPTQ-Int8",
    "mlx-community/Qwen3-0.6B-4bit",
)


def test_the_qwen3_rule_claims_a_name_that_still_contains_its_size():
    """The template is composed from the checkpoint's own, so the base model, a
    local snapshot and a copy or fine-tune under another organisation that kept
    the name each keep theirs: the rule claims them all, as it did before
    litetune 0.4.0 packed one revision's template and narrowed it. A name is all
    a Hub id is matched on, so a fine-tune renamed past the size is not
    claimed; nor is a repack named for its format."""
    for claimed in (
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B-Base",
        _SNAPSHOT.format("Qwen3-0.6B"),
        _SNAPSHOT.format("Qwen3-0.6B-Base"),
        "unsloth/Qwen3-0.6B",
        "someorg/Qwen3-0.6B",
        "someorg/Qwen3-0.6B-sql-assistant",
        # What a name cannot show: a format not spelled after the size.
        "mlx-community/Qwen3-0.6B-bf16",
    ):
        assert identify(claimed).family == "qwen-3", claimed
        plan = plan_export(claimed, (), ("dynamic_wi8_afp32",))
        marker = models.CANONICAL_OR_COMPOSED_TEMPLATE
        assert f"--jinja_chat_template_override={marker}" in plan.flags
    for unclaimed in (*_QWEN3_REPACKS, "acme/banking77-assistant", "Qwen/Qwen3-1.7B"):
        assert identify(unclaimed) is None, unclaimed


def test_the_qwen3_rule_still_matches_the_checkpoint_tune_wrote(tmp_path):
    """`hint_for` folds the sidecar's base model and the config's `model_type`
    and architecture into one text, and none of those is a repack's marker."""
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": "Qwen/Qwen3-0.6B"}), encoding="utf-8"
    )
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    hint = models.hint_for(str(checkpoint)).text
    assert hint == "qwen-qwen3-0-6b-qwen3-qwen3forcausallm"
    assert identify(str(checkpoint)).family == "qwen-3"


@pytest.mark.parametrize(
    "name",
    ["someorg/Qwen3-0.6B-MLX-bf16", "someorg/Qwen3-0.6B-AWQ", "someorg/Qwen3-0.6B-bnb-nf4"],
)
def test_each_repack_marker_on_its_own_is_unclaimed(name):
    """A name whose only marker is that one."""
    assert identify(name) is None, name


def test_the_qwen3_rule_reads_a_dash_after_qwen():
    assert identify("someorg/Qwen-3-0.6B").family == "qwen-3"


def _local_checkpoint(tmp_path, config: dict, base: str | None) -> str:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if base is not None:
        (checkpoint / "litetune.json").write_text(json.dumps({"base_model": base}))
    return str(checkpoint)


@pytest.mark.parametrize("base", ["Qwen/Qwen3-0.6B", None])
def test_a_local_full_precision_merge_of_a_repack_is_claimed(tmp_path, base):
    """For a directory the checkpoint says whether its weights are quantized, so
    a config whose `_name_or_path` names the bnb repack it was tuned from is
    claimed when it carries no `quantization_config` -- with the sidecar naming
    Qwen/Qwen3-0.6B or with none."""
    config = {
        "model_type": "qwen3",
        "_name_or_path": "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "architectures": ["Qwen3ForCausalLM"],
    }
    checkpoint = _local_checkpoint(tmp_path, config, base)
    assert "bnb-4bit" in models.hint_for(checkpoint).text
    assert identify(checkpoint).family == "qwen-3"


@pytest.mark.parametrize("where", ["top", "text_config"])
def test_a_local_checkpoint_with_a_quantization_config_is_not_claimed(tmp_path, where):
    """What makes transformers load the weights as pre-quantized
    (`quantizers/auto.py:334-338`): the config's, or its text config's."""
    quantization = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    config: dict = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    if where == "top":
        config["quantization_config"] = quantization
    else:
        config["text_config"] = {"quantization_config": quantization}
    checkpoint = _local_checkpoint(tmp_path, config, "Qwen/Qwen3-0.6B")
    assert models.hint_for(checkpoint).quantized
    assert identify(checkpoint) is None


@pytest.mark.parametrize(
    ("quantization", "quantized"),
    [
        ({"quant_method": "gptq", "bits": 4}, True),
        ({"load_in_8bit": True}, True),
        # transformers skips a method it does not know and loads the weights as
        # they are (`quantizers/auto.py:292-298`) ...
        ({"quant_method": "unknown"}, False),
        # ... and raises on a config with none, `{}` included: the exporter's
        # failure to report, not a repack (`:287-290`, `:334-340`).
        ({}, False),
        ({"bits": 4}, False),
        ("not a mapping", False),
        ({"quant_method": ["gptq"]}, False),
    ],
    ids=["gptq", "bnb-8bit", "unknown", "empty", "no-method", "not-a-mapping", "unhashable"],
)
def test_a_local_checkpoint_is_a_repack_only_as_transformers_would_load_it(
    tmp_path, quantization, quantized
):
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "quantization_config": quantization,
    }
    checkpoint = _local_checkpoint(tmp_path, config, "Qwen/Qwen3-0.6B")
    assert models.hint_for(checkpoint).quantized is quantized
    assert (identify(checkpoint) is None) is quantized


def test_a_qwen25_checkpoint_whose_config_names_its_source_is_claimed(tmp_path):
    """`hint_for` folds `_name_or_path` in between `model_type` and the
    architecture, so the tail must match at the `model_type`."""
    config = {
        "model_type": "qwen2",
        "_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "architectures": ["Qwen2ForCausalLM"],
    }
    checkpoint = _local_checkpoint(tmp_path, config, "Qwen/Qwen2.5-0.5B-Instruct")
    assert identify(checkpoint).family == "qwen-2.5"


def test_the_qwen25_rule_claims_the_checkpoint_that_was_run_and_not_its_repacks():
    """A size suffix is not a licence over the family, and neither is a prefix.

    `Qwen2-5B` is a Qwen 2 of five billion parameters, not a Qwen 2.5; the
    1.5B and the base 0.5B were never exported here; and the AWQ, GPTQ and
    bnb repacks are already-quantized checkpoints that no run in this
    repository touched. `identify` searches rather than matches, so an
    unanchored pattern claims every one of those and silences the
    unknown-family note on it -- which is the note that would otherwise be
    the only thing telling the user litetune has never seen this artifact.
    """
    for claimed in (
        "Qwen/Qwen2.5-0.5B-Instruct",
        _SNAPSHOT.format("Qwen2.5-0.5B-Instruct"),
        # Any organisation: the template composed is the checkpoint's own.
        "someorg/Qwen2.5-0.5B-Instruct",
        # What a name cannot show: an ONNX export under the bare name.
        "onnx-community/Qwen2.5-0.5B-Instruct",
    ):
        assert identify(claimed).family == "qwen-2.5", claimed
    for unclaimed in (
        "Qwen/Qwen2-5B",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "me/Qwen2.5-0.5B-Instruct-banking77",
        # The `model_type` tail is also where a repack's suffix can begin.
        "someorg/Qwen2.5-0.5B-Instruct-qwen2-GGUF",
    ):
        assert identify(unclaimed) is None, unclaimed


def test_the_qwen25_rule_still_matches_the_checkpoint_tune_wrote(tmp_path):
    """The anchor must not cost the tuned model its own rule.

    `hint_for` folds a local checkpoint's `model_type` and architecture into
    the text it matches against -- this run's merged model read
    `qwen-qwen2-5-0-5b-instruct-qwen2-qwen2forcausallm` -- so a pattern
    anchored at the end of the string alone would identify the Hub id and
    lose the checkpoint converted from it.
    """
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": "Qwen/Qwen2.5-0.5B-Instruct"}), encoding="utf-8"
    )
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    rules = identify(str(checkpoint))
    assert rules is not None, models.hint_for(str(checkpoint)).text
    assert rules.family == "qwen-2.5"


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


def test_a_declared_version_is_read_without_its_marker_and_only_where_it_applies(monkeypatch):
    """`sentencepiece==0.2.0; python_version < "3.13"` is the version 0.2.0 on
    3.12 and no answer on 3.13; split on `==`, the marker became the version."""
    from conftest import machine

    from litetune import wheels

    pins = (
        'sentencepiece==0.2.0; python_version < "3.13"',
        'sentencepiece==0.2.1; python_version >= "3.13"',
    )
    real = wheels.marker_environment
    for minor, expected in ((12, "0.2.0"), (13, "0.2.1")):
        asked = real(machine("linux_x86_64", (3, minor)))
        monkeypatch.setattr(wheels, "marker_environment", lambda host=None, a=asked: a)
        assert models.declared_version(pins, "sentencepiece") == expected


# ---------------------------------------------------------------------------
# The rules where the stages consult them
# ---------------------------------------------------------------------------


@dataclass
class FakeToolchain:
    """Stands in for `envs.StageEnv.run`, as in test_export.py."""

    pip_stdout: str = "transformers==5.5.0\nlitert-torch-nightly==0.10.0.dev20260826\n"
    calls: list[list[str]] = field(default_factory=list)
    # What the checkpoint's tokenizer carries, for the chat template script.
    chat_template: object = "{%- for m in messages %}{{ m.content }}{% endfor %}"
    embedded: dict[str, str | None] = field(default_factory=dict)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, 0, self.pip_stdout, "")
        if args[0] == "python" and str(args[1]).endswith("repack.py"):
            # This fake has no export environment to run the repack script in:
            # the repack reports that and the export stays a passed, CPU-only one.
            # The template reading the script takes first is answered from what
            # the exporter below embedded, as `test_export.py`'s fake builder does.
            spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
            if spec.get("template_out"):
                template = self.embedded.get(spec["artifact"])
                reading = {"sections": 1, "present": template is not None, "template": template}
                Path(spec["template_out"]).write_text(json.dumps(reading), encoding="utf-8")
            return subprocess.CompletedProcess(
                args, 1, "", "ModuleNotFoundError: No module named 'litert_lm_builder'"
            )
        if args[0] == "python" and str(args[1]).endswith(
            ("compose_template.py", "conversation_template.py")
        ):
            return run_compose_script(args, timeout, self.chat_template)[0]
        flags = dict(a.removeprefix("--").split("=", 1) for a in args[2:] if "=" in a)
        out_dir = Path(flags["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.litertlm").write_bytes(b"\0" * 4096)
        # The override's text, as `package_model` reads it (`litert_lm_builder.py:430`).
        override = Path(flags.get("jinja_chat_template_override") or "")
        known = override.name != "" and override.is_file()
        template = override.read_text(encoding="utf-8") if known else None
        self.embedded[str((out_dir / "model.litertlm").absolute())] = template
        return subprocess.CompletedProcess(args, 0, "", "")

    @property
    def exports(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "litert-torch"]


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    buildable_here(monkeypatch)
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


def test_convert_composes_the_template_a_qwen_rule_asks_for(toolchain, tmp_path):
    """The rule's marker reaches the stage, and the stage hands every exporter
    the composed file instead: the block, then the checkpoint's own template."""
    result = run_export(
        ExportRequest(
            model="Qwen/Qwen3-0.6B", output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",)
        )
    )
    assert result.outcome is Outcome.PASSED
    assert result.chat_template is not None
    composed = result.chat_template.path
    assert composed.parent == tmp_path / "out"
    block = Path(models.packaged_template(models.TEXT_PARTS_BLOCK)).read_bytes()
    assert composed.read_bytes() == block + str(toolchain.chat_template).encode("utf-8")
    (argv,) = toolchain.exports
    assert f"--jinja_chat_template_override={composed}" in argv
    assert not any(m in a for m in models.WRITTEN_AT_EXPORT for a in argv)


# -- how the exporter reads a flag's name, and what a value-bearing flag needs --


@pytest.mark.parametrize(
    "spelling",
    [
        "--jinja-chat-template-override=/my/own.jinja",
        "-jinja_chat_template_override=/my/own.jinja",
        "--jinja_chat_template_override=/my/own.jinja",
    ],
)
def test_a_callers_template_is_recognised_in_every_spelling_the_exporter_reads(spelling):
    """Fire strips the leading hyphens and reads `-` as `_` (`fire/core.py:868`,
    `:875`): every spelling sets the one keyword, so each is the caller's own
    template and nothing is composed beside it."""
    plan = plan_export("Qwen/Qwen3-0.6B", (spelling,), ("dynamic_wi8_afp32",))
    assert plan.flags == (spelling,)
    assert plan.added == ()


@pytest.mark.parametrize(
    "spelling",
    [
        "--litert-lm-model-type-override=gemma4",
        "-litert_lm_model_type_override=gemma4",
        "--litert_lm_model_type_override=gemma4",
    ],
)
def test_the_gemma4_type_override_is_refused_in_every_spelling(spelling):
    with pytest.raises(FlagRefused, match="does the opposite"):
        plan_export(GEMMA4_E2B, (spelling,), ())


@pytest.mark.parametrize("value", ["gemma4", "function_gemma", "generic", "1"])
def test_every_model_type_override_is_refused_for_gemma4(value):
    """Any override the exporter acts on skips the Gemma 4 metadata builder: it
    runs only when there is none (`litert_lm_builder.py:379-387`)."""
    with pytest.raises(FlagRefused, match="does the opposite"):
        plan_export(GEMMA4_E2B, (f"--litert-lm-model-type-override={value}",), ())


def test_a_falsy_model_type_override_is_none_to_the_exporter():
    """`if not litert_lm_model_type_override` (`litert_lm_builder.py:380`): no override."""
    plan = plan_export(GEMMA4_E2B, ("--litert_lm_model_type_override=''",), ())
    assert "--litert_lm_model_type_override=''" in plan.flags


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", GEMMA4_E2B, "google/gemma-4-it"])
@pytest.mark.parametrize(
    ("value", "read_as"),
    [("", "''"), ("None", "None"), ("0", "0"), ("False", "False"), ("''", "''"), ('""', "''")],
    ids=["empty", "None", "zero", "False", "quoted", "double-quoted"],
)
def test_a_value_fire_reads_as_falsy_does_not_satisfy_a_flag_that_needs_one(model, value, read_as):
    """Fire hands the exporter a parsed value (`fire/parser.py` `DefaultParseValue`),
    and the exporter reads a falsy override as none (`litert_lm_builder.py:414`)
    and embeds the tokenizer's template: the rule would hold in name only. The
    variant-unknown gemma-4 rule takes a value too."""
    with pytest.raises(FlagRefused, match=f"reads its value as {re.escape(read_as)}, which is not"):
        plan_export(model, (f"--jinja_chat_template_override={value}",), ())


@pytest.mark.parametrize("value", ["1", "[a]", "{a: b}"])
def test_a_value_fire_reads_as_no_string_is_refused(value):
    """Not a path or repository: Fire hands the exporter a number, a list or a dict."""
    with pytest.raises(FlagRefused, match="which is not a value"):
        plan_export("Qwen/Qwen3-0.6B", (f"--jinja_chat_template_override={value}",), ())


def test_a_value_in_the_next_argument_is_refused_with_the_form_to_use():
    """Fire takes the next argument as the value when it is not a flag
    (`fire/core.py:876-921`); litetune passes each flag on as one argument."""
    with pytest.raises(FlagRefused, match="pass --flag=--jinja_chat_template_override=<value>"):
        plan_export("Qwen/Qwen3-0.6B", ("--jinja_chat_template_override", "mine.jinja"), ())


def test_a_valueless_template_flag_is_true_to_the_exporter_and_refused():
    """With no next argument, or a flag next, Fire reads it as True (`:902-912`)."""
    with pytest.raises(FlagRefused, match="reads its value as True"):
        plan_export("Qwen/Qwen3-0.6B", ("--jinja_chat_template_override",), ())


@pytest.mark.parametrize(
    ("flags", "composes"),
    [
        # Fire keeps the last: the empty one wins and is refused ...
        (("--jinja_chat_template_override=/good.jinja", "--jinja-chat-template-override="), None),
        # ... and the other way round the caller's file wins, and nothing is composed.
        (("--jinja-chat-template-override=", "--jinja_chat_template_override=/good.jinja"), False),
    ],
    ids=["empty-last", "file-last"],
)
def test_a_repeated_flag_is_read_as_its_last_occurrence(flags, composes):
    """`fire/core.py:925` assigns each keyword as it meets it."""
    if composes is None:
        with pytest.raises(FlagRefused, match="reads its value as ''"):
            plan_export("Qwen/Qwen3-0.6B", flags, ())
        return
    plan = plan_export("Qwen/Qwen3-0.6B", flags, ())
    assert plan.added == ()
    (check,) = plan.checks
    assert check.observed["flag"] == "--jinja_chat_template_override=/good.jinja"
    assert any(
        "--jinja_chat_template_override=/good.jinja was passed by the caller" in text
        for text in plan.limitations
    )


def test_a_longer_flag_name_does_not_satisfy_the_rule():
    """`--jinja_chat_template_override_x` is another keyword to Fire."""
    plan = plan_export("Qwen/Qwen3-0.6B", ("--jinja_chat_template_override_x=1",), ())
    assert f"--jinja_chat_template_override={models.CANONICAL_OR_COMPOSED_TEMPLATE}" in plan.added


@pytest.mark.parametrize(
    "flag", ['--litert_lm_model_type_override="gemma4"', "--litert_lm_model_type_override='gemma4'"]
)
def test_a_quoted_forbidden_value_is_refused_as_the_bare_one(flag):
    with pytest.raises(FlagRefused, match="does the opposite"):
        plan_export(GEMMA4_E2B, (flag,), ())


def test_a_forbidden_flag_is_judged_by_its_last_occurrence():
    refused = ("--litert_lm_model_type_override=''", "--litert-lm-model-type-override=gemma4")
    with pytest.raises(FlagRefused, match="does the opposite"):
        plan_export(GEMMA4_E2B, refused, ())
    # The other way round the last is empty, which is no override to the
    # exporter, and it is the one argument passed on.
    kept = tuple(reversed(refused))
    flags = plan_export(GEMMA4_E2B, kept, ()).flags
    assert "--litert_lm_model_type_override=''" in flags
    assert "--litert-lm-model-type-override=gemma4" not in flags


def test_a_forbidden_flag_with_its_value_in_the_next_argument_is_refused():
    with pytest.raises(FlagRefused, match="pass --flag=--litert_lm_model_type_override=<value>"):
        plan_export(GEMMA4_E2B, ("--litert_lm_model_type_override", "gemma4"), ())


# -- one parse of the caller's exporter flags, as Fire reads them ------------


QWEN3 = "Qwen/Qwen3-0.6B"


@pytest.mark.parametrize(
    ("flags", "says"),
    [
        (("jinja_chat_template_override=/mine.jinja",), "as a positional one"),
        (("-",), "as a positional one"),
        (("-1",), "as a positional one"),
        (("--jinja_chat_template_override=/x.jinja", "--", "--y=1"), "after a bare '--'"),
        (("--=x",), "it names no keyword"),
        (("--model=Qwen/Qwen2.5-0.5B-Instruct",), "Use --model"),
        (("-output-dir=/tmp/elsewhere",), "Use --output-dir"),
        (("--quantization_recipe=dynamic_wi4_afp32",), "Use --recipe"),
        (("--jinja_chat_template_override", "mine.jinja"), "pass --flag="),
    ],
    ids=[
        "no-hyphen",
        "dash",
        "negative-number",
        "double-dash",
        "no-name",
        "model",
        "output-dir",
        "recipe",
        "two-token",
    ],
)
def test_a_flag_fire_would_read_otherwise_is_refused(flags, says):
    """Fire's `_IsFlag` (`fire/core.py:936-956`), its bare `--` (`fire/parser.py:39-56`)
    and its last-one-wins assignment (`:925`), which would let a caller's token
    replace litetune's own `--model`, `--output_dir` or `--quantization_recipe`."""
    for model in (QWEN3, "meta-llama/Llama-3.2-1B"):
        with pytest.raises(FlagRefused, match=re.escape(says)):
            plan_export(model, flags, ())


@pytest.mark.parametrize(
    ("flags", "effective"),
    [
        (
            (
                "--jinja_chat_template_override=/a.jinja",
                "--jinja-chat-template-override=/b.jinja",
                "--jinja_chat_template_override=/a.jinja",
            ),
            "--jinja_chat_template_override=/a.jinja",
        ),
        (
            (
                "--jinja_chat_template_override=/b.jinja",
                "--jinja_chat_template_override=/a.jinja",
                "--jinja-chat-template-override=/b.jinja",
            ),
            "--jinja-chat-template-override=/b.jinja",
        ),
    ],
    ids=["a-b-a", "b-a-b"],
)
def test_the_exporter_is_given_one_argument_per_keyword_the_last(flags, effective):
    """What litetune reasons about is what the exporter is handed: the token in
    effect, once."""
    plan = plan_export(QWEN3, flags, ())
    assert plan.flags == (effective,)


@pytest.mark.parametrize(
    ("flags", "outcome"),
    [
        (
            ("--jinja_chat_template_override=/good.jinja", "--nojinja_chat_template_override"),
            "False",
        ),
        (("--nojinja_chat_template_override", "--jinja_chat_template_override=/good.jinja"), None),
    ],
    ids=["no-last", "file-last"],
)
def test_a_valueless_no_prefix_sets_the_keyword_to_false(flags, outcome):
    """`--no<name>` alone is `<name>=False` to Fire, given `export`'s `**kwargs`
    (`fire/core.py:881-884`, `:902-912`); the exporter reads False as no override."""
    if outcome is not None:
        with pytest.raises(FlagRefused, match=f"reads its value as {outcome}"):
            plan_export(QWEN3, flags, ())
        return
    plan = plan_export(QWEN3, flags, ())
    assert plan.flags == ("--jinja_chat_template_override=/good.jinja",)
    assert plan.added == ()


def test_a_no_prefix_with_a_value_is_a_keyword_of_its_own():
    """Only a valueless `--no<name>` is read as a negation (`fire/core.py:902-912`)."""
    plan = plan_export(QWEN3, ("--nojinja_chat_template_override=/x.jinja",), ())
    assert plan.setting("--nojinja_chat_template_override").value == "/x.jinja"
    assert f"--jinja_chat_template_override={models.CANONICAL_OR_COMPOSED_TEMPLATE}" in plan.added


def test_a_hyphenated_no_prefix_negates_the_keyword_fire_makes_of_it():
    """Fire reads `-` as `_` before it looks for `no`, so `--no-x` sets `_x`."""
    plan = plan_export(QWEN3, ("--no-jinja-chat-template-override",), ())
    assert plan.setting("_jinja_chat_template_override").value is False
    assert f"--jinja_chat_template_override={models.CANONICAL_OR_COMPOSED_TEMPLATE}" in plan.added


def test_a_value_is_reported_as_the_exporter_will_get_it():
    """`#` starts a comment to Fire's literal parse: the exporter is given `mine`."""
    plan = plan_export(QWEN3, ("--jinja_chat_template_override=mine#v2.jinja",), ())
    assert any(
        "--jinja_chat_template_override=mine was passed" in text for text in plan.limitations
    )
    (check,) = plan.checks
    assert check.observed["flag"] == "--jinja_chat_template_override=mine"
    assert check.observed["token"] == "--jinja_chat_template_override=mine#v2.jinja"


def test_the_value_fire_hands_the_exporter():
    """`DefaultParseValue`: a literal, a bare word as a string, anything else as given."""
    assert models.exporter_value("None") is None
    assert models.exporter_value("0") == 0
    assert models.exporter_value('"gemma4"') == "gemma4"
    assert models.exporter_value("gemma4") == "gemma4"
    assert models.exporter_value("/x/y.jinja") == "/x/y.jinja"
    assert models.exporter_value("mine.jinja") == "mine.jinja"
    assert models.exporter_value("a-b.jinja") == "a-b.jinja"
    assert models.exporter_value("[a, 1]") == ["a", 1]


def test_a_callers_template_on_a_composing_rule_is_said_to_have_no_block():
    plan = plan_export("Qwen/Qwen3-0.6B", ("--jinja_chat_template_override=/my/own.jinja",), ())
    (note,) = [text for text in plan.limitations if "no text-parts block" in text]
    assert "--jinja_chat_template_override=/my/own.jinja" in note
    assert "LiteRT-LM 0.18.0" in note and "verify's rendering check" in note
    assert "loses the prompt or fails every send" in note


def test_a_callers_template_on_another_rule_carries_no_such_note():
    plan = plan_export(GEMMA4_E2B, ("--jinja_chat_template_override=me/mine",), ())
    assert not any("no text-parts block" in text for text in plan.limitations)


@pytest.mark.parametrize(
    "spelling",
    ["--litert_lm_llm_metadata_override=meta.pbtext", "--litert-lm-llm-metadata-override=x"],
)
def test_a_metadata_override_is_refused_where_the_template_is_composed(spelling):
    """The exporter puts the override in place of the metadata it builds
    (`litert_lm_builder.py:438-456`), the only way the composed template reaches
    the bundle (`:314-318`)."""
    with pytest.raises(FlagRefused, match="in place of the LlmMetadata it builds"):
        plan_export("Qwen/Qwen3-0.6B", (spelling,), ())


def test_a_metadata_override_beside_the_callers_own_template_is_refused_too():
    """The exporter then never uses the template flag (`litert_lm_builder.py:438-456`),
    so "used as given" would be untrue of the caller's template."""
    flags = ("--litert_lm_llm_metadata_override=meta.pbtext", "--jinja_chat_template_override=t")
    with pytest.raises(FlagRefused, match="and not the one passed"):
        plan_export("Qwen/Qwen3-0.6B", flags, ())


def test_a_metadata_override_the_exporter_reads_as_none_is_not_one():
    """The last occurrence is empty, which the exporter reads as no override."""
    flags = ("--litert_lm_llm_metadata_override=meta.pbtext", "--litert_lm_llm_metadata_override=")
    plan = plan_export("Qwen/Qwen3-0.6B", flags, ())
    assert f"--jinja_chat_template_override={models.CANONICAL_OR_COMPOSED_TEMPLATE}" in plan.added


def test_a_metadata_override_on_a_rule_that_does_not_compose_is_left_alone():
    flags = ("--litert_lm_llm_metadata_override=meta.pbtext",)
    assert plan_export(FUNCTIONGEMMA, flags, ()).flags[0] == flags[0]


def test_the_gemma4_advice_is_a_command_line_convert_accepts():
    plan = plan_export("google/gemma-4-it", (), ())
    (unchecked,) = [c for c in plan.checks if c.outcome is Outcome.UNCHECKED]
    assert "--flag=--jinja_chat_template_override=<repo>" in unchecked.detail


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
            # Bare text, which is what this mode trains; `tune` refuses
            # `prerendered` on it before anything else is checked.
            prompt_mode=PromptMode.RUNTIME_RENDERED,
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


def test_functiongemma_carries_a_template_the_runtime_can_execute():
    """The checkpoint's own template cannot run on-device.

    It uses `macro` and `dictsort`; LiteRT-LM renders with MiniJinja, which
    supports neither. A bundle built from it exports cleanly, is the right size,
    passes every liveness check and answers the text path `flutter_gemma` uses —
    and fails the native tool path with an opaque
    `litert_lm_conversation_send_message_stream failed`, because the wrapper
    keeps the return code and drops the message under it.

    Measured, on the same checkpoint: with the override the runtime answers
    `[tool_call] set_alarm{hour:7}`; without it, `INTERNAL: Failed to apply
    template`.
    """
    import pathlib

    from litetune.models import functiongemma_template, plan_export

    template = pathlib.Path(functiongemma_template())
    assert template.is_file(), "the template must ship with the package"
    text = template.read_text(encoding="utf-8")

    # The header names both constructs while explaining their absence, so the
    # body is what gets checked. A test that matched the explanation instead of
    # the code would pass on a file that had neither.
    body = text.split("-#}", 1)[1]
    assert "{% macro" not in body
    assert "dictsort" not in body
    # And the attribution, which is the whole reason this file may sit in an
    # Apache-2.0 package at all.
    assert "NOT WRITTEN HERE" in text
    assert "Gemma Terms of Use" in text

    plan = plan_export("google/functiongemma-270m-it", (), ("dynamic_wi8_afp32",))
    override = [f for f in plan.flags if f.startswith("--jinja_chat_template_override=")]
    assert len(override) == 1
    # By path parts: the path is the platform's own, `\`-separated on Windows.
    packaged = Path(override[0].partition("=")[2])
    assert packaged.parts[-2:] == ("templates", "functiongemma.jinja")
    assert any(
        "--jinja_chat_template_override" in f for f in plan.added
    ), "litetune added this, and the report has to say so"


def test_a_caller_who_names_their_own_template_keeps_it():
    """Their value wins: litetune knows the flag is needed, not which template
    a differently-trained checkpoint wants."""
    from litetune.models import plan_export

    plan = plan_export(
        "google/functiongemma-270m-it",
        ("--jinja_chat_template_override=/my/own.jinja",),
        ("dynamic_wi8_afp32",),
    )
    overrides = [f for f in plan.flags if f.startswith("--jinja_chat_template_override=")]
    assert overrides == ["--jinja_chat_template_override=/my/own.jinja"]


def test_an_unnamed_gemma3_text_checkpoint_refuses_rather_than_guessing(tmp_path):
    """`config.json` proves a flag is needed and cannot say what it should be.

    FunctionGemma and Gemma 3 270M/1B all declare `model_type: gemma3_text` and
    need different override values and different templates. Guessing wrong
    exports a bundle typed `generic_model` with no tool-call channel — which
    passes every check this package runs, because the text path never notices.

    So the plan is unusable and the export refuses. This is the same shape
    Gemma 4 already had for its template variant, not a new kind of verdict.
    """
    from litetune.models import plan_export

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text"}), encoding="utf-8"
    )

    plan = plan_export(str(checkpoint), (), ("dynamic_wi8_afp32",))

    assert plan.rules is not None and plan.rules.family == "gemma3-text-unidentified"
    assert not plan.usable, "an export that would be mis-typed must not proceed"
    unresolved = [c for c in plan.checks if c.outcome is not Outcome.PASSED]
    assert unresolved, "the refusal has to reach the report"
    detail = " ".join(c.detail or "" for c in plan.checks)
    assert (
        "--base-model" in detail and "--train-metrics" in detail
    ), "a refusal must name the way out of it"


def test_the_one_family_whose_tool_path_was_measured_carries_it():
    """FunctionGemma is the only entry with a wire format, and it says why.

    The reason has to name what was read rather than assert the format, because
    a family rule is a claim to have checked: the declaration turn was measured
    against a bundle and the call spelling comes from the runtime's own goldens.
    """
    fg = wire_format_for(FUNCTIONGEMMA)

    assert fg.family == "functiongemma"
    assert fg.name == "functiongemma"
    assert fg.known
    assert "<escape>" in fg.reason and "goldens" in fg.reason
    assert renders_declarations_for(FUNCTIONGEMMA) == (True, fg.reason)


@pytest.mark.parametrize("model", ["Qwen/Qwen3-0.6B", "google/gemma-3-270m-it"])
def test_a_family_with_an_entry_and_no_measured_format_says_so(model):
    """Why the answer has three states rather than two.

    Qwen-3 has an entry that deliberately records nothing about its calls. Under
    a single "does litetune know this model" question it would read as fine, and
    a structured target would be rendered in FunctionGemma's spelling for a
    runtime that does not read it.
    """
    answer = wire_format_for(model)

    assert answer.family is not None
    assert answer.name is None
    assert not answer.known
    assert answer.family in answer.reason
    assert "completion" in answer.reason
    assert renders_declarations_for(model)[0] is False


def test_a_model_with_no_entry_is_a_third_distinct_answer():
    """Not the same as a family that records no format, and not the same message.

    Knowing nothing about Llama is not knowing it is wrong, so this names the
    two ways forward rather than the family -- there is no family to name.
    """
    answer = wire_format_for("meta-llama/Llama-3.2-1B")

    assert answer.family is None
    assert answer.name is None
    assert "no entry for this model" in answer.reason
    assert answer.reason != wire_format_for("Qwen/Qwen3-0.6B").reason


def test_an_architecture_with_no_rules_is_a_note_not_a_refusal(tmp_path):
    """Llama, Phi, Mistral: litetune knows nothing, which is not knowing they
    are wrong. Same answer by path and by Hub id — the verdict is about the
    model, not about how `--model` was typed."""
    from litetune.models import plan_export

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

    by_path = plan_export(str(checkpoint), (), ("dynamic_wi8_afp32",))
    by_id = plan_export("meta-llama/Llama-3.2-1B", (), ("dynamic_wi8_afp32",))

    assert by_path.rules is None and by_id.rules is None
    assert by_path.usable and by_id.usable


def test_the_sidecar_beats_config_json(tmp_path):
    """What produced the checkpoint knows what it was; `config.json` knows only
    the architecture, and for this family that is not enough."""
    from litetune.models import hint_for, rules_for_hint

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text"}), encoding="utf-8"
    )
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": "google/functiongemma-270m-it"}), encoding="utf-8"
    )

    rules = rules_for_hint(hint_for(str(checkpoint)))
    assert rules is not None and rules.family == "functiongemma"


def test_a_folder_named_after_the_base_model_does_not_decide_the_family(tmp_path):
    """The path is what somebody called a directory, not what is inside it.

    A FunctionGemma checkpoint in `runs/gemma-3-270m-tools/model` was claimed by
    `gemma-3-text` on the path substring, exported with `=gemma3` and no template
    override, and reported passed — the exact artifact this refusal exists to
    stop, produced by the most natural naming convention there is.
    """
    from litetune.models import plan_export

    checkpoint = tmp_path / "runs" / "gemma-3-270m-tools" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text"}), encoding="utf-8"
    )

    plan = plan_export(str(checkpoint), (), ("dynamic_wi8_afp32",))
    assert plan.rules is not None and plan.rules.family == "gemma3-text-unidentified"
    assert not plan.usable

    # And a recorded identity settles it, so the ordinary rules apply again.
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": "google/functiongemma-270m-it"}), encoding="utf-8"
    )
    settled = plan_export(str(checkpoint), (), ("dynamic_wi8_afp32",))
    assert settled.rules is not None and settled.rules.family == "functiongemma"
    assert settled.usable


def test_a_name_that_merely_contains_gemma3_text_is_not_refused():
    """The refusal is about `config.json`, and its message says so.

    Matching the merged hint text meant `myorg/gemma3_text_lora` — a Hub id with
    no config anywhere — was refused with a message asserting what a file that
    was never opened declares.
    """
    from litetune.models import plan_export

    for name in ("myorg/gemma3_text_lora", "unsloth/gemma3-text-it"):
        plan = plan_export(name, (), ("dynamic_wi8_afp32",))
        assert plan.rules is None, name
        assert plan.usable, name


def test_an_unreadable_sidecar_is_a_fault_not_an_absence(tmp_path):
    """Returning None for both let a truncated file look like an older run.

    A `litetune.json` cut short by a full disk or a half-finished copy would
    then be guessed past, from the path — which is the fault the sidecar exists
    to remove.
    """
    from litetune.models import hint_for

    counter = iter(range(100))

    def hint_with(content: str):
        d = tmp_path / f"case{next(counter)}" / "model"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"model_type": "gemma3_text"}), encoding="utf-8")
        (d / "litetune.json").write_text(content, encoding="utf-8")
        return hint_for(str(d))

    assert "could not be read" in (hint_with('{"base_model": "goo').provenance_error or "")
    assert "does not contain a JSON object" in (hint_with("[1, 2]").provenance_error or "")
    assert "records no 'base_model'" in (hint_with('{"prompt_mode": "x"}').provenance_error or "")
    assert "not a name" in (hint_with('{"base_model": ["a"]}').provenance_error or "").replace(
        "non-string 'base_model'", "not a name"
    )

    # A checkpoint that never had one is not a fault.
    plain = tmp_path / "plain" / "model"
    plain.mkdir(parents=True)
    (plain / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
    assert hint_for(str(plain)).provenance_error is None
