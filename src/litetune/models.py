"""Per-model rules, with the reason for each one written next to it.

A rule recorded without its evidence is a rule that gets refactored away by the
next person who reads it as arbitrary. Every entry below therefore carries what
was observed, or who said it and where -- a run, an issue number, a line in a
vendor's source. None of them is a preference.

Three kinds of knowledge live here:

**Export flags a family requires.** Gemma 4 will not export without
`--externalize_embedder`, and will export *wrongly* -- silently -- without
`--jinja_chat_template_override`. litetune adds them and says so;
`--litert_lm_model_type_override=gemma4` is refused rather than dropped,
because a flag that quietly does the opposite of its name is worse than an
error.

**The minimum `transformers` a family needs.** A version too old does not
produce a subtly worse model, it raises `AttributeError: 'list' object has no
attribute 'keys'` from inside a tokenizer load. That is a diagnosable message
here and an unreadable traceback six hours into a training run.

**A ceiling that has to be stated.** No public recipe reproduces Google's
published Gemma 4 artifact, so an export made here is not equivalent to it. That
is recorded as a limitation on every result, because the alternative is a user
assuming parity that was never claimed.

**What is deliberately *not* here: the prompt-rendering mode.** It is tempting
to write `gemma-4 -> runtime_rendered` in the table below and be done with it,
and it would be wrong. The mode is a property of *how the checkpoint was
trained*, not of its family: FunctionGemma needed `--no-template` because the
tool declarations were hand-rendered into the prompt, and the same FunctionGemma
trained through `apply_chat_template` would need the opposite. Same weights, same
family, different answer. It is decided by `tune`, carried by
`bundle.Contract.prompt_mode`, and resolved for a foreign artifact by
`prompt_mode.resolve_prompt_mode` -- never inferred from a model id.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from litetune.checks import Check, Outcome

logger = logging.getLogger(__name__)

MODELS_SCHEMA = "litetune.models/1"

EXPORT_FLAGS_CHECK = "export flags this model family requires"
TRANSFORMERS_CHECK = "transformers supports this model"

# Files consulted when the model is a local checkpoint rather than a hub id. A
# merged checkpoint in `runs/out/model` carries no family in its path, and
# `config.json` is the only place the family is written down.
CONFIG_NAME = "config.json"
_CONFIG_KEYS = ("model_type", "_name_or_path", "architectures")


class FlagRefused(ValueError):
    """A requested export flag is refused, with the reason it is refused.

    Not a silent drop: a caller who asked for a flag and did not get it, with no
    error, would reasonably believe the artifact was built with it.
    """


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequiredFlag:
    """A flag this family's export must carry, and why.

    `value_unknown` is set when the flag is required but litetune cannot work
    out its value for this particular model -- the Gemma 4 template override is
    per-variant. That is `could not check`, not a guess: the wrong template is
    the failure the flag exists to prevent.
    """

    name: str
    value: str | None = None
    reason: str = ""
    value_unknown: str = ""

    @property
    def rendered(self) -> str | None:
        """The flag as it reaches the command line, or None if the value is unknown."""
        if self.value_unknown:
            return None
        return self.name if self.value is None else f"{self.name}={self.value}"

    def satisfied_by(self, flags: Sequence[str]) -> str | None:
        """The caller's own form of this flag, if they passed one.

        Matched on the flag *name*, so a caller who supplied their own value has
        satisfied the requirement with it. Their value wins: litetune knows this
        flag is needed, and does not know better than the caller which template
        their checkpoint was trained against.
        """
        for flag in flags:
            if flag == self.name or flag.startswith(f"{self.name}="):
                return flag
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "flag": self.rendered,
            "name": self.name,
            "reason": self.reason,
            "value_unknown": self.value_unknown or None,
        }


@dataclass(frozen=True)
class ForbiddenFlag:
    """A flag that must never be passed for this family, and what it really does."""

    name: str
    value: str | None
    reason: str

    def matches(self, flag: str) -> bool:
        if self.value is None:
            return flag == self.name or flag.startswith(f"{self.name}=")
        return flag == f"{self.name}={self.value}"

    @property
    def rendered(self) -> str:
        return self.name if self.value is None else f"{self.name}={self.value}"

    def as_dict(self) -> dict[str, Any]:
        return {"flag": self.rendered, "reason": self.reason}


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRules:
    """Everything litetune knows about one model family."""

    family: str
    # Regexes matched against a normalised model id (see `_normalise`). Ordered
    # most specific first in `RULES`, because a variant's rules are the family's
    # rules plus a value only the variant fixes.
    patterns: tuple[str, ...]
    required_flags: tuple[RequiredFlag, ...] = ()
    forbidden_flags: tuple[ForbiddenFlag, ...] = ()
    min_transformers: str | None = None
    min_transformers_reason: str = ""
    recommended_recipes: tuple[str, ...] = ()
    recipe_reason: str = ""
    limitations: tuple[str, ...] = ()
    # Terminators the serving convention requires that a training run cannot
    # reveal. `tune` records the terminator it supervised, which is the one the
    # completions end with -- and for a function-calling family that is only
    # half the answer: the model must also stop where the application has to
    # take over. Not derivable from `config.json` either, which is what this
    # module is for.
    extra_stop_tokens: tuple[str, ...] = ()
    stop_token_reason: str = ""

    # Which part of a multimodal checkpoint a LoRA run may adapt, named as the
    # container its modules sit under -- `tune` restricts its projection set to
    # modules whose path passes through it. `None` for a text-only family:
    # there is no second tower to exclude, so the projection list alone decides
    # which modules are adapted, and naming a container would be a claim about
    # a structure that does not exist.
    lora_container: str | None = None
    lora_container_reason: str = ""

    # The tool path, recorded together because one measurement establishes both:
    # whether this family's serving runtime renders tool declarations into the
    # prompt, and the spelling its calls use. Neither is in `config.json` and
    # neither can be asked of the user without letting them contradict the
    # runtime. A family that records no `wire_format` is not a family litetune
    # knows nothing about -- that is `identify` returning `None` -- it is one
    # whose calls this project has never measured, which is why the two are
    # different answers with different messages.
    renders_declarations: bool = False
    wire_format: str | None = None
    tool_path_reason: str = ""

    # Whether this rule's patterns may be satisfied by the *path*. `hint_for`
    # merges the model string and the config values into one text, so by default
    # a directory named after a base model matches as if the config had said so.
    # For a rule that exists to refuse an ambiguous config that is exactly wrong:
    # `runs/gemma-3-270m-tools/model` holding a FunctionGemma checkpoint would be
    # claimed by the wrong family and exported with the wrong flags, green.
    config_only: bool = False

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self.patterns)

    def matches_hint(self, hint: ModelHint) -> bool:
        if not self.config_only:
            return self.matches(hint.text)
        # The path is precisely the evidence this rule declares worthless.
        return any(self.matches(_normalise(value)) for value in hint.from_config)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "required_flags": [f.as_dict() for f in self.required_flags],
            "forbidden_flags": [f.as_dict() for f in self.forbidden_flags],
            "min_transformers": self.min_transformers,
            "min_transformers_reason": self.min_transformers_reason,
            "recommended_recipes": list(self.recommended_recipes),
            "recipe_reason": self.recipe_reason,
            "limitations": list(self.limitations),
            "extra_stop_tokens": list(self.extra_stop_tokens),
            "stop_token_reason": self.stop_token_reason,
            "lora_container": self.lora_container,
            "lora_container_reason": self.lora_container_reason,
            "renders_declarations": self.renders_declarations,
            "wire_format": self.wire_format,
            "tool_path_reason": self.tool_path_reason,
        }


# -- the evidence, once, so the entries below can share it -------------------

# Measured: without it the exporter raises
# `AssertionError: External embedder is required for Gemma4`.
_EXTERNALIZE_REASON = (
    "Gemma 4 does not export without it: the exporter raises 'AssertionError: External embedder "
    "is required for Gemma4'. It is harmless on other families, where it writes the tied embedding "
    "as its own section -- but it is not cosmetic. On Gemma-3-270M the embedding table is over 60% "
    "of the model's parameters, so this changes the artifact's structure and not just its "
    "packaging: the same model measures 286 MB one way and 457 MB the other, depending on nothing "
    "but this flag. Two exports that differ in it cannot be compared on .litertlm size (compare "
    "shipped bytes, which include the externalised embedder, or do not compare them at all)"
)

# Sourced: litert-torch#998, a Google engineer -- "the flag is indeed needed as
# the rendering engine litert-lm uses (minijinja) doesn't support many pythonic
# semantics from the original jinja template."
_TEMPLATE_REASON = (
    "Gemma 4's own chat_template.jinja calls .get()/.map() 23 times and LiteRT-LM renders with "
    "MiniJinja, which supports neither. A Google engineer, litert-torch#998: 'the flag is indeed "
    "needed as the rendering engine litert-lm uses (minijinja) doesn't support many pythonic "
    "semantics from the original jinja template.' LiteRT-LM/models/gemma4/README.md records that "
    "the forks differ first on tool-response handling -- exactly the structured-output path. "
    "Without the override the bundle carries a template the runtime cannot render, and it fails "
    "silently rather than erroring"
)

_TEMPLATE_VARIANT_UNKNOWN = (
    "the override names a per-variant repository (E2B and E4B have different ones) and this model "
    "id does not say which variant it is, so litetune will not guess: the wrong chat template is "
    "the exact failure this flag exists to prevent, and it fails silently. Name the variant in the "
    "model id, or pass --jinja_chat_template_override=<repo> yourself"
)

# Measured: the flag leaves `generic_model` set *and* skips the Gemma 4 metadata
# builder, so it produces a generic-model artifact with no Gemma 4 metadata.
_TYPE_OVERRIDE_REASON = (
    "--litert_lm_model_type_override=gemma4 does the opposite of what its name suggests: it leaves "
    "generic_model set *and* skips the Gemma 4 metadata builder, so the artifact is written as a "
    "generic model with none of the metadata the flag appears to request. litetune refuses it "
    "rather than dropping it silently, because a dropped flag reads as an applied one"
)

# Measured on both families: every 4.x release in [4.55.0, 4.57.6] dies at
# tokenizer load. Fixed in 5.0.0, never backported.
_TRANSFORMERS_5_REASON = (
    "every transformers 4.x from 4.55.0 to 4.57.6 raises \"AttributeError: 'list' object has no "
    "attribute 'keys'\" at tokenizer load for this family, because extra_special_tokens ships as a "
    "list where 4.x expects a mapping. Fixed only in 5.0.0 and never backported. Observed on both "
    "Gemma 4 and Qwen3.5"
)

_GEMMA4_TRANSFORMERS_REASON = (
    _TRANSFORMERS_5_REASON + ". Gemma 4 additionally needs 5.5.0, where AutoConfig first "
    "recognises the `gemma4` architecture; 5.0.0 loads the tokenizer but not the config"
)

# Sourced: a Google engineer, on remaining "the model quality" for Gemma 4.
_GEMMA4_RECIPE_REASON = (
    "for Gemma 4 a Google engineer recommends dynamic_wi4c_hr_afp32 or dynamic_wi4b32_afp32 'to "
    "remain the model quality', noting that the published artifact is half int2 while the public "
    "recipes reach int4. This is a recommendation and not a substitution: the recipe you asked for "
    "is the recipe that was exported. Of the two, litetune has measured dynamic_wi4b32_afp32 once, "
    "on base weights rather than a tuned checkpoint (MEASUREMENTS.md), and dynamic_wi4c_hr_afp32 "
    "not at all"
)

# Sourced: peft 0.20.0 ships this scope itself. Its
# `TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` maps `gemma4` to the
# regex `.*language_model\..*\.(q_proj|v_proj)` -- so upstream agrees both
# that the container is called `language_model` and that a name list is not
# enough to reach it. litetune cannot take that default: it names its
# projection set explicitly so that two runs recorded as `lora` are the same
# method, and passing `target_modules` is what switches the default off.
#
# The reason below says which modules repeat across the towers and not how
# many. An earlier draft carried three counts off the module graph; nothing in
# this repository derives, stores or re-derives them, no revision was attached
# to them, and the checkpoint they describe is one this project does not pin --
# so they were precision the tree cannot point at. What a reader needs is the
# structure, and that is checkable against any Gemma 4 config.
# Sourced: the same peft mapping, read the other way. It gives every entry
# below a plain projection-name list and reserves the container regex for
# `gemma4` alone -- which is upstream recording that a suffix match on these
# has no second tower to reach. litetune records the decision rather than the
# default, so that "examined, text-only" and "nobody looked" do not arrive as
# the same empty string.
_NO_CONTAINER_UPSTREAM = (
    "no container. peft 0.20.0 maps this architecture to a plain projection-name list and "
    "reserves its container regex for the multimodal `gemma4`, which is upstream recording "
    "that a suffix match here has no second tower to reach"
)

# And the one family that mapping does not cover at all.
_NO_CONTAINER_NOT_ESTABLISHED = (
    "no container is needed: `AutoModelForCausalLM` loads this architecture as a text-only "
    "model and its vision projections are named qkv/proj/linear_fc1/linear_fc2, which none of "
    "the seven names reaches. What a LoRA run here does NOT reach is the other half of its own "
    "text tower: the gated-delta-net layers project through in_proj_qkv/in_proj_z/in_proj_b/"
    "in_proj_a/out_proj, so `lora` adapts attention in the full-attention layers only, and the "
    "MLP everywhere. Read off transformers 5.16.1; peft 0.20.0 has no entry for this "
    "architecture to compare against"
)

# Sourced twice, because an earlier draft of this string was wrong in the more
# frightening direction. The counts are re-derivable from `config.json` at
# 3e22461f: 16 vision layers x 7 names, 12 audio layers x 3 (their MLPs are
# `ffw_layer_*`, which none of the seven names matches), and 35 text layers of
# which `num_kv_shared_layers: 20` carry no `k_proj`/`v_proj`, so 15 x 7 +
# 20 x 5. What an unscoped run then *does* was read out of transformers 5.16.1
# -- the tower projections are `Gemma4ClippableLinear`, an `nn.Module` holding
# an `nn.Linear`, while the text ones are bare -- and executed against peft
# 0.20.0, whose `dispatch_default` takes a bare `nn.Linear` and raises on
# anything else. The earlier draft said such a run trained quietly. It does
# not; it stops before the first step.
_GEMMA4_LORA_CONTAINER_REASON = (
    "the checkpoint is multimodal and its vision and audio towers use the same projection names "
    "as its text layers, so a name-suffix match reaches all three: 112 modules in the vision "
    "tower and 36 in the audio tower against 205 in the text one, on google/gemma-4-E2B-it at "
    "3e22461f. peft cannot adapt the tower ones -- transformers wraps them in "
    "Gemma4ClippableLinear and peft 0.20.0 dispatches on a bare nn.Linear -- so an unscoped run "
    "stops in get_peft_model with 'Target module ... is not supported' rather than training the "
    "wrong thing quietly. Scoping to `language_model` is what lets the projection set apply at "
    "all. The refusal is loud because the towers are unadaptable, not because they are unwanted: "
    "a peft that learned to wrap them would make the same run silent"
)

# Sourced: litert-torch#1044 -- "Right now litert-torch don't support QAT
# checkpoint conversion".

_GEMMA4_NOT_GOOGLES_ARTIFACT = (
    "a Gemma 4 export made here is NOT equivalent to Google's published .litertlm. Google's comes "
    "from a quantized-safetensors (QAT) path that litert-torch does not support -- 'Right now "
    "litert-torch don't support QAT checkpoint conversion', litert-torch#1044 -- and no public "
    "recipe reproduces its int2/int4/int8 mixture. Whatever recipe you choose, this artifact is a "
    "different quantization of the same weights, and Google's published numbers are not a baseline "
    "for it"
)


def _gemma4(family: str, patterns: tuple[str, ...], override_repo: str | None) -> ModelRules:
    """One Gemma 4 variant. The variants differ only in the template repository."""
    template = (
        RequiredFlag(
            name="--jinja_chat_template_override", value=override_repo, reason=_TEMPLATE_REASON
        )
        if override_repo is not None
        else RequiredFlag(
            name="--jinja_chat_template_override",
            reason=_TEMPLATE_REASON,
            value_unknown=_TEMPLATE_VARIANT_UNKNOWN,
        )
    )
    return ModelRules(
        family=family,
        patterns=patterns,
        required_flags=(
            RequiredFlag(name="--externalize_embedder", reason=_EXTERNALIZE_REASON),
            template,
        ),
        forbidden_flags=(
            ForbiddenFlag(
                name="--litert_lm_model_type_override",
                value="gemma4",
                reason=_TYPE_OVERRIDE_REASON,
            ),
        ),
        min_transformers="5.5.0",
        min_transformers_reason=_GEMMA4_TRANSFORMERS_REASON,
        recommended_recipes=("dynamic_wi4c_hr_afp32", "dynamic_wi4b32_afp32"),
        recipe_reason=_GEMMA4_RECIPE_REASON,
        limitations=(_GEMMA4_NOT_GOOGLES_ARTIFACT,),
        extra_stop_tokens=("<turn|>", "<|tool_response>"),
        stop_token_reason=_GEMMA4_STOP_REASON,
        lora_container="language_model",
        lora_container_reason=_GEMMA4_LORA_CONTAINER_REASON,
    )


# Most specific first: `gemma-4-e2b` must win over the generic `gemma-4` entry,
# whose template override is deliberately unknown.
# -- the model-type trap ------------------------------------------------------
#
# `litert_lm_builder.py` chooses the bundle's declared type by matching
# `config.json`'s `model_type` against a fixed list -- `qwen3`, `qwen2`,
# `gemma3`, `function_gemma`, `gemma3n` -- and everything else falls to a silent
# `case _` that writes `generic_model`.
#
# FunctionGemma's config says `model_type: "gemma3_text"`. So does plain
# Gemma 3. Neither string is in that list -- not even `gemma3_text` against
# `gemma3` -- so both export as `generic_model` with no warning, and the model's
# *name* is the only place the word "function" appears.
#
# What a generic type costs, from LiteRT-LM's own source: the runtime picks a
# data processor by type, and `GenericDataProcessor` returns empty code fences,
# so `internal_callback_util.cc` never creates the tool-call channel; and its
# `CreateConstraint` returns `Unimplemented`, which `conversation.cc` swallows,
# so constrained decoding turns itself off without a diagnostic. A consumer that
# parses the response text (flutter_gemma) is unaffected; one that passes tools
# natively receives no calls at all.
#
# Measured, not inferred: Google's published
# `mobile_actions_q8_ekv1024.litertlm` declares `llm_model_type {
# function_gemma {} }`, ours declared `generic_model`, and the override produces
# byte-identical metadata to theirs. Google's only published FunctionGemma
# recipe sets the type by hand through the older converter; no Google page
# mentions this flag at all.
#
# The value cannot be derived from `model_type`, because the two families share
# one. It has to come from the model's identity, which is what these rules are.
def functiongemma_template() -> str:
    """Path to the prompt template a FunctionGemma bundle must carry.

    Not the one the checkpoint ships with. That template uses `{% macro %}` and
    `dictsort`, which the MiniJinja engine inside LiteRT-LM does not support, so
    a bundle built from it fails the native tool path -- the caller sees
    `litert_lm_conversation_send_message_stream failed` and nothing else,
    because the wrapper drops the message under it.

    Shipped as a file rather than fetched from a repository the way Gemma 4's
    override is. That one points at `litert-community`, which Google publishes
    and maintains under Apache-2.0; pointing at anything else would make an
    export depend on a mutable ref and a network round trip, which is the same
    objection this package raises to `--base-model-revision main`.
    """
    return str(files("litetune") / "templates" / "functiongemma.jinja")


_FUNCTION_TEMPLATE_REASON = (
    "FunctionGemma's own chat template uses `macro` and `dictsort`, and LiteRT-LM renders "
    "with MiniJinja, which supports neither. A bundle carrying it exports cleanly, passes "
    "every liveness check, answers the text path that flutter_gemma uses -- and fails the "
    "native tool path with an opaque `send_message_stream failed`. Measured: with this "
    "override the same checkpoint answers `[tool_call] set_alarm{hour:7}`; without it, "
    "`INTERNAL: Failed to apply template`"
)

# Sourced from the runtime rather than from a description of it. The declaration
# side: `create_conversation(tools=...)` renders a developer turn carrying
# `<start_function_declaration>declaration:...<end_function_declaration>`,
# measured 2026-09-16 against a local bundle, and the declaration text is built
# by `fc_tool_format_utils.cc` before any template runs. The call side:
# LiteRT-LM's own goldens in `function_gemma_data_processor_test.cc` carry
# `call:get_weather{location:<escape>Paris<escape>}` beside `call:tool_name{x:1}`.
_TOOL_PATH_REASON = (
    "FunctionGemma's runtime renders tool declarations into a developer turn of the prompt, and "
    "its calls spell a string between `<escape>` markers and a number, a boolean or a null bare: "
    "`call:name{who:<escape>ann<escape>,n:3}`. Both were read from the runtime -- the declaration "
    "turn measured against a bundle through the Python API, the call spelling from LiteRT-LM's "
    "own goldens -- not from the model card. This is the only family here whose tool path has "
    "been measured, which is why every other entry records none rather than a guess"
)

_FUNCTION_RESPONSE_REASON = (
    "a FunctionGemma turn does not end at the call: after `<end_function_call>` the "
    "application has to execute the tool and send the result back, and the model must stop "
    "and wait for it. Training cannot reveal this terminator -- the completions end at "
    "`<end_of_turn>`, so a run records that one and nothing else. The .litertlm itself does "
    "carry it, as token id 50 out of generation_config.eos_token_id; this is about the "
    "contract, which a consumer reads without a protobuf parser and which otherwise names "
    "only the terminator the run observed"
)

# Sourced: generation_config.json on `google/gemma-4-E2B-it` declares eos_token_id
# [1, 106, 50]; tokenizer.json names those `<eos>`, `<turn|>` and
# `<|tool_response>`. Measured 2026-09-11 on an A100: all five reference
# generations stopped on their own after 6-17 tokens, each ending `<turn|>`.
# No generation stopping at `<|tool_response>` has been observed here, and the
# reason below says so rather than describing a role taken from its name.
_GEMMA4_STOP_REASON = (
    "Gemma 4 closes a turn with `<turn|>`, not `<end_of_turn>`: generation_config.json declares "
    "eos_token_id [1, 106, 50] and tokenizer.json names 106 `<turn|>`. Measured, all five "
    "reference generations stopped on their own and ended there. Neither marker was in "
    "`metrics.TERMINATORS`, so every Gemma 4 generation ended in one scoring did not recognise "
    "and a 600-row `verify` refused the comparison outright: 600 of 600 reference generations "
    "did not end in a terminator it knew. 50 is `<|tool_response>`, which the chat template uses "
    "to open a tool-response turn; it is declared because the model's own eos set names it, not "
    "because a generation stopping there has been observed here"
)

_GEMMA3_TEXT_AMBIGUOUS = (
    "config.json declares model_type 'gemma3_text', which is FunctionGemma and also Gemma 3 "
    "270M/1B -- they need different override values and different prompt templates, and "
    "nothing in the config distinguishes them. transformers 5.x deletes `_name_or_path` on "
    "save, so a trained checkpoint carries no name either. Say which it is: `--base-model "
    "<id>`, or `--train-metrics` from the run that produced it. litetune refuses rather than "
    "guessing, because guessing wrong exports a bundle with no tool-call channel that passes "
    "every check"
)

_MODEL_TYPE_REASON = (
    "config.json declares model_type 'gemma3_text', which the exporter does not "
    "recognise, so the bundle is typed generic_model and the runtime creates no "
    "tool-call channel and silently disables constrained decoding"
)


RULES: tuple[ModelRules, ...] = (
    _gemma4(
        "gemma-4-e2b",
        (r"gemma-?4-e2b",),
        "litert-community/gemma-4-E2B-it-litert-lm",
    ),
    _gemma4(
        "gemma-4-e4b",
        (r"gemma-?4-e4b",),
        "litert-community/gemma-4-E4B-it-litert-lm",
    ),
    # `(?![\db])` so that a size suffix is not read as the generation number:
    # `gemma-40m` and a `gemma-4b` are not Gemma 4, and matching them would
    # refuse a perfectly good export for a family these rules say nothing about.
    _gemma4("gemma-4", (r"gemma-?4(?![\db])",), None),
    ModelRules(
        family="functiongemma",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        patterns=(r"function-?gemma",),
        required_flags=(
            RequiredFlag(
                name="--litert_lm_model_type_override",
                value="function_gemma",
                reason=_MODEL_TYPE_REASON,
            ),
            RequiredFlag(
                name="--jinja_chat_template_override",
                value=functiongemma_template(),
                reason=_FUNCTION_TEMPLATE_REASON,
            ),
        ),
        extra_stop_tokens=("<start_function_response>",),
        stop_token_reason=_FUNCTION_RESPONSE_REASON,
        renders_declarations=True,
        wire_format="functiongemma",
        tool_path_reason=_TOOL_PATH_REASON,
    ),
    ModelRules(
        family="gemma-3-text",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # After functiongemma, which is also a gemma3_text config and needs a
        # different value. Order in this tuple is the disambiguation.
        #
        # The two text-only sizes by name, not `gemma-?3-\d`. That pattern also
        # claimed 4B, 12B and 27B, which are `Gemma3ForConditionalGeneration`
        # with a vision tower and a `model_type` of plain `gemma3` -- so the
        # override below is both unnecessary for them and asserts a reason
        # ("config.json says gemma3_text, which the exporter does not
        # recognise") that is untrue of them. Multimodal export is not
        # something this project has run, and a family rule is a claim to have
        # checked. They fall through to the unknown-family note instead.
        #
        # Both sizes have now been run end to end on banking77, each exported
        # with this override added by litetune and each with a conversion cost
        # in MEASUREMENTS.md; the 1B on 2026-09-19, which is the run that first
        # exercised the flag at that size. The rule had claimed the 1B on the
        # strength of the 270M until then. What the 1B run adds is that the
        # claim held, and that the two sizes answer differently at four bits --
        # 8.83 points against 34.83 -- which is a fact about the model rather
        # than about this rule.
        patterns=(r"gemma-?3-270m", r"gemma-?3-1b"),
        required_flags=(
            RequiredFlag(
                name="--litert_lm_model_type_override",
                value="gemma3",
                reason=_MODEL_TYPE_REASON,
            ),
        ),
    ),
    ModelRules(
        family="qwen-3.5",
        lora_container_reason=_NO_CONTAINER_NOT_ESTABLISHED,
        # Same guard: `Qwen3-5B` would be a Qwen 3, not a Qwen 3.5.
        patterns=(r"qwen-?3-5(?![\db])",),
        min_transformers="5.0.0",
        min_transformers_reason=_TRANSFORMERS_5_REASON,
    ),
    ModelRules(
        family="qwen-3",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # Nothing to add, and that is what this entry records. `qwen3` is on the
        # exporter's own type list (the model-type trap, above), so a config
        # that says `model_type: "qwen3"` is typed correctly with no override.
        # Measured 2026-09-14 on Qwen/Qwen3-0.6B: both int8 recipes exported
        # with no flag from litetune, and the conversion cost is in
        # MEASUREMENTS.md. What the artifact's own `llm_model_type` reads is not
        # recorded here: that observation appears in no manifest or log of any
        # run, and was struck from the documents for the same reason.
        #
        # By size, not `qwen-?3`: that also claims the other sizes and the
        # Qwen 3 models built on other architectures, none of which this
        # project has run. No `min_transformers` either -- none was measured.
        # Sizes are added here as they are.
        patterns=(r"qwen-?3-0-6b",),
    ),
    ModelRules(
        family="qwen-2.5",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # Nothing to add here either, and this entry says so with a run behind
        # it. `config.json` declares `model_type: "qwen2"`, which
        # `litert_lm_builder.py` matches as `case 'qwen2' | 'qwen2p5'`, so no
        # override: unlike the gemma3_text families there is no ambiguity for
        # one to resolve. Qwen 3 is typed from its own config in the same way,
        # one case up; what is particular here is only that the runtime's type
        # is named for the later generation than the config asks for.
        #
        # Measured 2026-09-20 on Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760: both
        # int8 recipes and all four 4-bit recipes exported with no flag from
        # litetune, six bundles, conversion costs in MEASUREMENTS.md. Before
        # that run `identify` returned None for this checkpoint and `export`
        # printed its unknown-family note on every convert, which was the
        # honest state and is what this rule replaces.
        #
        # By size *and* variant, for the reason the qwen-3 rule above gives:
        # `Qwen2.5-0.5B-Instruct` was run and `Qwen2.5-0.5B` was not. They are
        # two checkpoints, and this file's own history is that a rule written
        # on the strength of a neighbouring one goes unexercised for weeks --
        # the gemma-3-text rule claimed the 1B for three weeks before anything
        # exported it.
        #
        # The tail is what makes that true. `identify` searches rather than
        # matches, so a bare `qwen-?2-5-0-5b-instruct` also claims
        # any id that continues past it -- an `-AWQ`, a `-GPTQ-Int4`, a
        # bnb-4bit repack -- which name already-quantized checkpoints nobody
        # here exported, and for which "no flags needed" is least likely to
        # hold. The tests pin the regex against those spellings; whether each
        # repository exists is not something this checks. So the
        # pattern ends either at the end of the hint text or at the
        # `model_type` `hint_for` appends for a local checkpoint: this run's
        # merged model read `qwen-qwen2-5-0-5b-instruct-qwen2-qwen2forcausallm`
        # and must keep matching, a repack must not.
        #
        # No `min_transformers` -- the pinned 5.16.1 loaded it, and nothing
        # here establishes a floor, which is not the same as there being none.
        patterns=(r"qwen-?2-5-0-5b-instruct(?:$|-qwen2\b)",),
    ),
    ModelRules(
        family="gemma3-text-unidentified",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # Last, so a checkpoint that names its family is matched by name first.
        # This is the fallback for one that does not: `config.json` establishes
        # with certainty that an override is *required*, and cannot establish
        # what it should be. FunctionGemma and Gemma 3 270M/1B all declare
        # `gemma3_text` and need different values -- and different templates.
        #
        # So the plan is unusable and the export refuses, rather than producing
        # a bundle typed `generic_model` with no tool-call channel while every
        # check stays green. Any other architecture -- llama, phi3, mistral --
        # matches nothing here and is a note and exit 0, by path or by id alike:
        # about those litetune knows nothing, which is not the same as knowing
        # the artifact will be wrong.
        # `hint_for` folds `model_type` into the match text as `gemma3-text`.
        # Ordered last in RULES, so `functiongemma` and `gemma-3-270m` claim a
        # checkpoint that names itself and only an unnamed one reaches here.
        patterns=(r"^gemma3-text$",),
        config_only=True,
        required_flags=(
            RequiredFlag(
                name="--litert_lm_model_type_override",
                reason=_MODEL_TYPE_REASON,
                value_unknown=_GEMMA3_TEXT_AMBIGUOUS,
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Identifying a model
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lower-case, with every run of non-alphanumerics collapsed to a hyphen.

    So `google/gemma-4-E2B-it`, `Gemma4_E2B` and `/models/gemma-4-e2b/` all
    reduce to the same shape before matching.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class ModelHint:
    """What the family was matched against, and where it came from.

    A local checkpoint directory is the case this exists for: `runs/out/model`
    names no family, and the answer is in its `config.json`. When that file
    cannot be read the failure is recorded rather than swallowed -- an unmatched
    model means no rules were applied, and a reader has to be able to tell "this
    family has no rules" from "litetune could not tell what this is".
    """

    model: str
    text: str
    from_config: tuple[str, ...] = ()
    config_error: str | None = None
    # True when `tune` recorded what this checkpoint came from. Without it, a
    # match can only have come from the path or the architecture -- and a folder
    # name is what the user called a directory, not what is inside it.
    recorded: bool = False
    # Why the sidecar could not be used, when there was one. Distinct from
    # `recorded=False` with no error: one is a fault to fix, the other is just a
    # checkpoint from before this existed.
    provenance_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "matched_against": self.text,
            "from_config": list(self.from_config),
            "config_error": self.config_error,
            "identity_recorded": self.recorded,
            "provenance_error": self.provenance_error,
        }


PROVENANCE_NAME = "litetune.json"
"""What `tune` leaves beside a checkpoint so `convert` knows what it is.

`config.json` was meant to carry this: `ModelHint` was written around
`_name_or_path`, which `from_pretrained` sets to the id you loaded. transformers
5.x deletes that key on save -- `to_diff_dict` drops it before serialising -- so
a checkpoint this package produces names nothing, matches no family rule, and
exports without the flags its family requires. Silently: the file is the right
size and every check is green.

The same shape as `tokenizer.model`, which `carry_back_sentencepiece` restores
for the same reason. A sidecar rather than writing `_name_or_path` back, because
that is a vendor-format file and `from_pretrained` overwrites the field with the
load path on the next load anyway.
"""


def _provenance(path: Path) -> tuple[str | None, str | None]:
    """The base model `tune` recorded beside this checkpoint. Returns (id, error).

    An unreadable sidecar is not the same as an absent one, and returning `None`
    for both was the one place in this module that failed to say which. A file
    truncated by a full disk or half-copied by an interrupted `rsync` would then
    look like a checkpoint that never recorded anything -- and be guessed at from
    the path, which is the fault this sidecar exists to remove.
    """
    sidecar = path / PROVENANCE_NAME
    if not sidecar.is_file():
        return None, None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, RecursionError) as exc:
        return None, f"{sidecar} could not be read: {type(exc).__name__}: {exc}"
    if not isinstance(data, dict):
        return None, f"{sidecar} does not contain a JSON object"
    base = data.get("base_model")
    if base is None:
        return None, f"{sidecar} records no 'base_model'"
    if not isinstance(base, str) or not base.strip():
        # A list or a dict here would `str()` into something that matches a
        # family regex by accident, which is a confident answer from nonsense.
        return None, f"{sidecar} records a non-string 'base_model': {base!r}"
    return base, None


def recorded_identity(model: str) -> str | None:
    """What the checkpoint at `model` says it came from, if it says anything."""
    path = Path(model)
    return _provenance(path)[0] if path.is_dir() else None


def provenance_error(model: str) -> str | None:
    """Why the checkpoint's own record could not be used, if there was one.

    Read from the path rather than from a plan's hint: when `--base-model` names
    an id, the plan is built from that string and never opens the directory --
    which is exactly when a broken record most needs saying out loud.
    """
    path = Path(model)
    return _provenance(path)[1] if path.is_dir() else None


def same_family(a: str | None, b: str | None) -> bool:
    """Whether two identities resolve to the same rules. Unknown counts as same.

    Two names litetune has no rules for cannot disagree about which rules apply,
    so a conflict between them is not worth refusing over.
    """
    if not a or not b:
        return True
    left, right = identify(a), identify(b)
    if left is None and right is None:
        return True
    return left is not None and right is not None and left.family == right.family


def hint_for(model: str) -> ModelHint:
    """The text `identify` matches against: the model id, plus a local checkpoint."""
    path = Path(model)
    normalised = _normalise(model)
    config = path / CONFIG_NAME
    if not (path.is_dir() and config.is_file()):
        return ModelHint(model=model, text=normalised)

    # Before config.json: what produced the checkpoint knows what it was, and
    # config.json only knows what architecture it is -- which is not enough,
    # since FunctionGemma and Gemma 3 declare the same `model_type` and need
    # different overrides.
    recorded_base, provenance_error = _provenance(path)
    if recorded_base:
        normalised = _normalise(recorded_base)

    try:
        data = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        # A checkpoint whose config cannot be read is one whose family is
        # unknown, which is a different statement from "it has no rules".
        logger.warning("could not read %s to identify %s: %s", config, model, exc)
        return ModelHint(
            model=model,
            text=normalised,
            config_error=f"{type(exc).__name__}: {exc}",
            recorded=bool(recorded_base),
            provenance_error=provenance_error,
        )
    if not isinstance(data, dict):
        return ModelHint(
            model=model,
            text=normalised,
            config_error=f"{config} does not contain a JSON object",
            recorded=bool(recorded_base),
            provenance_error=provenance_error,
        )

    values: list[str] = []
    for key in _CONFIG_KEYS:
        value = data.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    extra = tuple(values)
    text = "-".join([normalised, *(_normalise(v) for v in extra)])
    return ModelHint(
        model=model,
        text=text,
        from_config=extra,
        recorded=bool(recorded_base),
        provenance_error=provenance_error,
    )


def identify(model: str) -> ModelRules | None:
    """The rules for this model, or None when litetune knows of none.

    None is not "no rules apply" -- it is "litetune has no entry for this". The
    difference is reported by every caller.
    """
    hint = hint_for(model)
    return rules_for_hint(hint)


def rules_for_hint(hint: ModelHint) -> ModelRules | None:
    # An ambiguous config outranks a match from the path. `config.json` saying
    # `gemma3_text` is a fact about the weights; `runs/gemma-3-270m-tools/model`
    # is what somebody called a folder, and a FunctionGemma checkpoint sitting in
    # one would otherwise be claimed by the wrong family and exported with the
    # wrong flags -- successfully, and with a passed check vouching for it.
    #
    # Only when the identity was not recorded: a sidecar settles the question,
    # and then the ordinary name rules apply as they should.
    if not hint.recorded:
        for rules in RULES:
            if rules.config_only and rules.matches_hint(hint):
                return rules
    for rules in RULES:
        if rules.matches_hint(hint):
            return rules
    return None


UNKNOWN_FAMILY = (
    "litetune has no per-model rules for this checkpoint. That is not a statement that none apply: "
    "the rules it does hold were paid for one model family at a time, and a family it has not met "
    "is a family whose required export flags and minimum toolchain versions are simply unknown here"
)


def report(model: str) -> dict[str, Any]:
    """The rules for one model, in a shape a manifest can carry."""
    hint = hint_for(model)
    rules = rules_for_hint(hint)
    record: dict[str, Any] = {
        "schema": MODELS_SCHEMA,
        "hint": hint.as_dict(),
        "known": rules is not None,
    }
    if rules is None:
        record["family"] = None
        record["reason"] = UNKNOWN_FAMILY
        return record
    return record | rules.as_dict()


def limitations_for(model: str) -> list[str]:
    """Recorded limitations for this model. Empty when litetune knows of none."""
    rules = identify(model)
    return list(rules.limitations) if rules is not None else []


def stop_tokens_for(model: str) -> tuple[tuple[str, ...], str]:
    """Terminators this family needs beyond the one training recorded, and why.

    Read with `litertlm_peek`, Google's bundle names `<end_of_turn>` and
    `<start_function_response>` as strings; ours names the same two as token ids
    106 and 50, taken from `generation_config.eos_token_id`, plus `<eos>` and a
    set of punctuation-prefixed string variants the exporter adds deliberately
    to catch SentencePiece merging `.` and the terminator into one token. The
    two bundles agree; a first reading of the peek grepped for `token_str` and
    reported a difference that was an encoding.

    What did differ is the contract. `bundle` named only the terminator the
    training run observed, so a consumer reading `contract.json` -- rather than
    parsing the bundle's protobuf -- was not told where the application has to
    take over.
    """
    rules = identify(model)
    if rules is None:
        return (), ""
    return rules.extra_stop_tokens, rules.stop_token_reason


@dataclass(frozen=True)
class WireFormat:
    """How a family spells a tool call, or why litetune cannot say.

    Three answers, not two, because they need three different messages. `family`
    is `None` when litetune has no entry for this model at all -- knowing nothing
    about Llama is not knowing it is wrong, and
    `test_an_architecture_with_no_rules_is_a_note_not_a_refusal` pins that. A
    named family with `name` `None` is the live case: Qwen-3 has an entry that
    deliberately records no format, and under a single "is it known" question it
    would read as fine.
    """

    family: str | None
    name: str | None
    reason: str

    @property
    def known(self) -> bool:
        return self.name is not None


_NO_ENTRY = (
    "litetune has no entry for this model, so it records neither a wire format for its calls "
    "nor the absence of one. Name a model it knows with `--base-model`, or supply each row's "
    "completion text, which is trained exactly as written"
)
_NO_FORMAT = (
    "litetune has an entry for {family} but records no wire format for its calls: nothing in "
    "this project has measured how that family's runtime spells one, and rendering "
    "FunctionGemma's spelling for it would train a format its runtime does not read. Supply "
    "each row's completion text instead, which is trained exactly as written"
)


def wire_format_for(model: str) -> WireFormat:
    """The spelling this family's calls use, with the reason it was recorded."""
    rules = identify(model)
    if rules is None:
        return WireFormat(family=None, name=None, reason=_NO_ENTRY)
    if rules.wire_format is None:
        return WireFormat(
            family=rules.family, name=None, reason=_NO_FORMAT.format(family=rules.family)
        )
    return WireFormat(family=rules.family, name=rules.wire_format, reason=rules.tool_path_reason)


def renders_declarations_for(model: str) -> tuple[bool, str]:
    """Whether this family's runtime puts tool declarations in the prompt, and why.

    A separate question from `wire_format_for` on purpose: a family could render
    declarations and spell its calls in a way nobody here has measured, and
    answering both from one flag would make the second a guess dressed as the
    first.
    """
    rules = identify(model)
    if rules is None:
        return False, ""
    return rules.renders_declarations, rules.tool_path_reason


# ---------------------------------------------------------------------------
# Planning an export
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportPlan:
    """The flags an export will actually carry, and how each one got there.

    `checks` is three-valued and is the point of the whole object: a required
    flag litetune could not resolve is `could not check`, and the caller must
    refuse to export rather than produce an artifact that fails silently.
    """

    model: str
    hint: ModelHint
    rules: ModelRules | None
    flags: tuple[str, ...] = ()
    added: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()
    notes: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def outcome(self) -> Outcome:
        if any(c.outcome is Outcome.UNCHECKED for c in self.checks):
            return Outcome.UNCHECKED
        if any(c.outcome is Outcome.FAILED for c in self.checks):
            return Outcome.FAILED
        return Outcome.PASSED

    @property
    def usable(self) -> bool:
        """Whether an export built on this plan may be attempted at all."""
        return self.outcome is not Outcome.UNCHECKED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MODELS_SCHEMA,
            "model": self.model,
            "family": self.rules.family if self.rules else None,
            "hint": self.hint.as_dict(),
            "flags": list(self.flags),
            "added_by_litetune": list(self.added),
            "checks": [c.as_dict() for c in self.checks],
            "notes": list(self.notes),
            "recommendations": list(self.recommendations),
            "limitations": list(self.limitations),
            "rules": self.rules.as_dict() if self.rules else None,
        }


def plan_export(
    model: str, requested_flags: Sequence[str] = (), recipes: Sequence[str] = ()
) -> ExportPlan:
    """Work out the flags this model's export must carry.

    Raises `FlagRefused` for a flag that must never be passed. Everything else
    is reported: a flag litetune added is named with its reason, and a required
    flag whose value it cannot determine makes the plan unusable rather than
    guessed at.
    """
    hint = hint_for(model)
    rules = rules_for_hint(hint)
    flags = list(dict.fromkeys(requested_flags))

    if rules is None:
        return ExportPlan(
            model=model,
            hint=hint,
            rules=None,
            flags=tuple(flags),
            notes=(UNKNOWN_FAMILY,),
        )

    for forbidden in rules.forbidden_flags:
        for flag in flags:
            if forbidden.matches(flag):
                raise FlagRefused(f"{model}: litetune refuses {flag!r}. {forbidden.reason}")

    added: list[str] = []
    notes: list[str] = []
    checks: list[Check] = []
    for required in rules.required_flags:
        supplied = required.satisfied_by(flags)
        if supplied is not None:
            wanted = required.rendered
            detail = f"{supplied} was supplied by the caller"
            if wanted is not None and supplied != wanted:
                # Their value, not litetune's: the caller may have trained
                # against a template litetune has never seen. Recorded, because
                # a difference here is invisible in the artifact.
                detail = (
                    f"{supplied} was supplied by the caller; litetune would have used {wanted}. "
                    "The caller's value was kept"
                )
            checks.append(
                Check.passed(
                    EXPORT_FLAGS_CHECK,
                    f"{rules.family}: {detail}",
                    observed={"flag": supplied, "family": rules.family},
                )
            )
            continue

        rendered = required.rendered
        if rendered is None:
            checks.append(
                Check.unchecked(
                    EXPORT_FLAGS_CHECK,
                    f"{rules.family} requires {required.name} and litetune cannot determine its "
                    f"value for {model!r}: {required.value_unknown}",
                    observed={"flag": required.name, "family": rules.family},
                )
            )
            continue

        flags.append(rendered)
        added.append(rendered)
        notes.append(f"added {rendered}: {required.reason}")
        checks.append(
            Check.passed(
                EXPORT_FLAGS_CHECK,
                f"{rules.family}: {rendered} was added by litetune — {required.reason}",
                observed={"flag": rendered, "family": rules.family, "added": True},
            )
        )

    recommendations: list[str] = []
    if rules.recommended_recipes and recipes:
        overlap = [r for r in recipes if r in rules.recommended_recipes]
        if not overlap:
            recommendations.append(
                f"{rules.family}: none of the requested recipes {list(recipes)} is one of the "
                f"recommended {list(rules.recommended_recipes)}. {rules.recipe_reason}"
            )

    return ExportPlan(
        model=model,
        hint=hint,
        rules=rules,
        flags=tuple(flags),
        added=tuple(added),
        checks=tuple(checks),
        notes=tuple(notes),
        recommendations=tuple(recommendations),
        limitations=tuple(rules.limitations),
    )


def refuse_forbidden_flags(model: str, flags: Sequence[str]) -> None:
    """Raise `FlagRefused` if any flag must never be passed for this model."""
    rules = identify(model)
    if rules is None:
        return
    for forbidden in rules.forbidden_flags:
        for flag in flags:
            if forbidden.matches(flag):
                raise FlagRefused(f"{model}: litetune refuses {flag!r}. {forbidden.reason}")


# ---------------------------------------------------------------------------
# Toolchain versions
# ---------------------------------------------------------------------------


_RELEASE_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)")


def version_tuple(text: str) -> tuple[int, ...] | None:
    """The numeric release part of a version, or None if there is not one.

    Pre-release ordering is deliberately not modelled: `5.0.0.dev0` reads as
    `5.0.0` here. The comparison this feeds is "does this build contain the fix",
    and a dev build of 5.0.0 does.
    """
    match = _RELEASE_RE.match(text or "")
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def transformers_check(
    model: str,
    rules: ModelRules,
    installed: str | None,
    source: str,
    unknown_reason: str = "",
) -> Check:
    """Three-valued: does the environment's transformers support this model.

    The failure this replaces is a traceback ending in `AttributeError: 'list'
    object has no attribute 'keys'` from inside a tokenizer load, six hours into
    a training run or twenty minutes into an export. The message here names the
    model, the version that is installed and the version that is needed.
    """
    minimum = rules.min_transformers
    if minimum is None:
        raise ValueError(f"{rules.family} declares no minimum transformers version")

    observed: dict[str, Any] = {
        "model": model,
        "family": rules.family,
        "minimum": minimum,
        "installed": installed,
        "source": source,
    }
    if not installed:
        return Check.unchecked(
            TRANSFORMERS_CHECK,
            f"{model} needs transformers>={minimum}, and the version installed in {source} could "
            f"not be read{f': {unknown_reason}' if unknown_reason else ''}. "
            f"{rules.min_transformers_reason}",
            observed=observed,
        )

    have, want = version_tuple(installed), version_tuple(minimum)
    if have is None or want is None:
        return Check.unchecked(
            TRANSFORMERS_CHECK,
            f"{model} needs transformers>={minimum} and {source} reports {installed!r}, which is "
            "not a version this can compare. The requirement stands and was not checked",
            observed=observed,
        )

    if have < want:
        return Check.failed(
            TRANSFORMERS_CHECK,
            f"{model} needs transformers>={minimum}; {source} has {installed}. "
            f"{rules.min_transformers_reason}. Pin transformers=={minimum} (or later) in that "
            "environment before running this stage",
            observed=observed,
        )
    return Check.passed(
        TRANSFORMERS_CHECK,
        f"{source} has transformers {installed}, at or above the {minimum} {rules.family} needs",
        observed=observed,
    )


def declared_version(requirements: Sequence[str], distribution: str = "transformers") -> str | None:
    """The pinned version of one distribution from a requirement list, if it is pinned.

    The *declaration*, which is what is available before an environment is
    built. `export.resolve_toolchain` reads what is actually installed and is
    strictly better when a run has one.
    """
    wanted = re.sub(r"[-_.]+", "-", distribution).lower()
    for requirement in requirements:
        name, sep, version = requirement.partition("==")
        if not sep:
            continue
        if re.sub(r"[-_.]+", "-", name.strip()).lower() == wanted:
            return version.strip()
    return None
