"""Per-model rules, with the reason for each one written next to it.

A rule recorded without its evidence is a rule that gets refactored away by the
next person who reads it as arbitrary. Every entry below therefore carries what
was observed, or who said it and where -- a run, an issue number, a line in a
vendor's source. None of them is a preference.

Three kinds of knowledge live here:

**Export flags a family requires.** Gemma 4 will not export without
`--externalize_embedder`, and will export *wrongly* -- silently -- without
`--jinja_chat_template_override`. litetune adds them and says so; a
`--litert_lm_model_type_override` is refused rather than dropped, because a flag
that quietly does the opposite of its name is worse than an error.

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

import ast
import hashlib
import json
import logging
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from litetune import wheels
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

TEMPLATE_FLAG = "--jinja_chat_template_override"
METADATA_OVERRIDE_FLAG = "--litert_lm_llm_metadata_override"


def flag_key(name: str) -> str:
    """The keyword a flag name sets on the exporter's command line.

    `litert-torch export_hf` hands its arguments to Fire (`litert_torch/cli.py`),
    which strips every leading hyphen and then reads `-` in the name as `_`
    (`fire/core.py:868` and `:875`, fire 0.7.1).
    """
    return name.split("=", 1)[0].lstrip("-").replace("-", "_")


@dataclass(frozen=True)
class ExporterFlag:
    """One keyword the exporter's command line sets, as Fire reads it."""

    name: str  # Fire's keyword: leading hyphens stripped, `-` read as `_`
    value: object  # what Fire hands the exporter for it
    token: str  # the argument that sets it, handed to the exporter unchanged

    @property
    def shown(self) -> str:
        """The flag as it takes effect: its keyword and the value the exporter gets."""
        value = self.value if isinstance(self.value, str) else repr(self.value)
        return f"--{self.name}={value}"


# The keywords litetune passes itself, from `convert`'s own options. Fire keeps
# the last of a repeated keyword (`fire/core.py:925`), so a caller's token for
# one would replace litetune's in the exporter's argv while the report named
# litetune's.
OWNED_FLAGS = {"model": "--model", "output_dir": "--output-dir", "quantization_recipe": "--recipe"}

# litert-torch's `export` (`export_hf/export.py:79-142`, litert-torch-nightly
# 0.10.0.dev20260926) takes `**kwargs`, so Fire accepts every keyword
# (`fire/core.py:881-884`). A valueless `--no<name>` then sets <name> to False
# unless `no<name>` is itself a parameter (`:902-912`); none of `export`'s
# parameters begins with "no".
_EXPORT_PARAMETERS_NAMED_NO: frozenset[str] = frozenset()


def _is_fire_flag(token: str) -> bool:
    """Fire's `_IsFlag` (`fire/core.py:936-956`): two hyphens, or one and a letter."""
    return token.startswith("--") or re.match(r"-[a-zA-Z]", token) is not None


def parse_exporter_flags(model: str, tokens: Sequence[str]) -> dict[str, ExporterFlag]:
    """The keywords `tokens` set on the exporter, by Fire's rules; raises `FlagRefused`.

    Each token is read as Fire's `_ParseKeywordArgs` reads it (`fire/core.py:
    816-934`): `--name=text` sets `name` to `exporter_value(text)`; a valueless
    `--name` sets it to True, or `--no<name>` sets `<name>` to False. The last
    token for a keyword is the one in effect, and the only one kept: the table
    is what litetune reasons about and, one token per keyword, what the
    exporter is given, so the two cannot disagree. Refused, each with what to
    pass instead, are the tokens Fire would read some other way -- a positional
    argument, a bare `--` and everything after it, a value in the next
    argument -- and the keywords litetune sets itself.
    """
    table: dict[str, ExporterFlag] = {}
    for index, token in enumerate(tokens):
        if token == "--":
            raise FlagRefused(_DOUBLE_DASH.format(model=model))
        if not _is_fire_flag(token):
            raise FlagRefused(_POSITIONAL.format(model=model, token=token))
        stripped = token.lstrip("-")
        if "=" in token:
            key, text = stripped.split("=", 1)
            key = key.replace("-", "_")
            value: object = exporter_value(text)
        else:
            key = stripped.replace("-", "_")
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is not None and not _is_fire_flag(following):
                raise FlagRefused(_split_value(model, token, f"--{key}"))
            if key not in _EXPORT_PARAMETERS_NAMED_NO and key.startswith("no"):
                key, value = key[2:], False
            else:
                value = True
        if not key:
            raise FlagRefused(f"{model}: litetune refuses {token!r}: it names no keyword")
        if key in OWNED_FLAGS:
            raise FlagRefused(
                f"{model}: litetune refuses {token!r}: litetune passes --{key} to the exporter "
                f"itself, from convert's {OWNED_FLAGS[key]}, and Fire would let this one replace "
                f"it (`fire/core.py:925`). Use {OWNED_FLAGS[key]}"
            )
        table.pop(key, None)
        table[key] = ExporterFlag(key, value, token)
    return table


_DOUBLE_DASH = (
    "{model}: litetune refuses '--': Fire hands every argument after a bare '--' to its own "
    "options and drops what they do not name (`fire/parser.py:39-56`, `fire/core.py:121-124`), "
    "so litetune's own flags after it would never reach the exporter"
)

_POSITIONAL = (
    "{model}: litetune refuses {token!r}: Fire reads an argument that begins with neither two "
    "hyphens nor a hyphen and a letter as a positional one (`fire/core.py:936-956`), and the "
    "exporter would take it for its first parameter not yet given. Pass an exporter flag as "
    "--flag=--name=value"
)


def exporter_value(text: str) -> object:
    """The value Fire hands the exporter for `--name=<text>`.

    Fire's `DefaultParseValue` (`fire/parser.py:59-76`): `_LiteralEval`
    (`:79-116`) reads the text as a Python literal after turning every bare
    name but `True`, `False` and `None` into a string (`_Replacement`,
    `:119-132`), refuses a binary operation, and on a syntax or value error the
    text stays the string it was. So `None` is None, `0` is 0, `''` and `""` are
    empty, `"gemma4"` is `gemma4`, and a path is itself. Written out here rather
    than imported: litetune does not depend on Fire.
    """
    try:
        root = ast.parse(text, mode="eval")
        if isinstance(root.body, ast.BinOp):
            raise ValueError(text)
        for node in ast.walk(root):
            for field, child in ast.iter_fields(node):
                if isinstance(child, list):
                    for index, item in enumerate(child):
                        if isinstance(item, ast.Name):
                            child[index] = _bare_word(item)
                elif isinstance(child, ast.Name):
                    setattr(node, field, _bare_word(child))
        return ast.literal_eval(root)
    except (SyntaxError, ValueError):
        return text


def _bare_word(node: ast.Name) -> ast.expr:
    if node.id in ("True", "False", "None"):
        return node
    return ast.Constant(node.id)


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

    @property
    def takes_value(self) -> bool:
        """Whether the flag means anything without a value after `=`."""
        return self.value is not None or bool(self.value_unknown)

    def satisfied_by(self, table: dict[str, ExporterFlag]) -> ExporterFlag | None:
        """The caller's own setting of this flag, if they passed one.

        Looked up by keyword in `parse_exporter_flags`' table, so a caller who
        supplied their own value, in any spelling Fire reads as this keyword,
        has satisfied the requirement with the value the exporter will get.
        Their value wins: litetune knows this flag is needed, and does not know
        better than the caller which template their checkpoint was trained
        against.
        """
        return table.get(flag_key(self.name))

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

    def matches(self, flag: ExporterFlag) -> bool:
        """Whether `flag` sets this one, to this value or, with none named, to any
        the exporter acts on (a falsy override is none to it)."""
        if flag.name != flag_key(self.name):
            return False
        return bool(flag.value) if self.value is None else flag.value == self.value

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

    # Whether a repack -- already-quantized or non-PyTorch weights -- is turned
    # away rather than claimed. The evidence differs by where the checkpoint
    # is: a directory's own `config.json` says whether its weights are
    # quantized, so the names in its hint (the path, the sidecar's base, the
    # config's `_name_or_path`, which may name the repack a full-precision merge
    # came from) are not consulted for it; a Hub id has only its name, which is
    # read for one of `_REPACK_MARKERS` after the match.
    repack_guard: bool = False

    # A chat template LiteRT-LM publishes for this family, packaged under
    # `templates/` and pinned in `CANONICAL_TEMPLATES`, which a rule whose
    # template override is `CANONICAL_OR_COMPOSED_TEMPLATE` packs in place of
    # the composed one when it renders single turns as the checkpoint's own.
    canonical_template: str | None = None

    # Why `verify` builds a LiteRT-LM Engine for every prompt instead of one per
    # split, for a family whose state a session on a shared Engine may not
    # reset; empty for every other family. `verify` refuses a decision for such
    # a family, whose scoring rewinds within one session.
    engine_per_prompt_reason: str = ""

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self.patterns)

    def matches_hint(self, hint: ModelHint) -> bool:
        if self.config_only:
            # The path is precisely the evidence this rule declares worthless.
            return any(self.matches(_normalise(value)) for value in hint.from_config)
        if not self.repack_guard:
            return self.matches(hint.text)
        if hint.local:
            return not hint.quantized and self.matches(hint.text)
        return any(re.search(pattern + _NOT_A_REPACK, hint.text) for pattern in self.patterns)

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
            "canonical_template": self.canonical_template,
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
    "model id, or pass --flag=--jinja_chat_template_override=<repo> to convert yourself"
)

# Measured when litetune 0.1.0a1 pinned an earlier litert-torch nightly: the
# value `gemma4` left `generic_model` set *and* skipped the Gemma 4 metadata
# builder. In litert-torch-nightly 0.10.0.dev20260926, pinned now, `gemma4` does
# select the gemma4 type (`litert_lm_builder.py:368-371`), but the
# model-specific metadata builder -- Gemma 4's, for a Gemma 4 config
# (`model_ext/metadata_builder.py:32-35`) -- runs only when there is no override
# at all (`litert_lm_builder.py:379-387`). So every value the exporter acts on is
# refused, not only `gemma4`.
_TYPE_OVERRIDE_REASON = (
    "--litert_lm_model_type_override does the opposite of what it suggests on Gemma 4: whatever "
    "its value, the exporter then skips the Gemma 4 metadata builder, which it runs only when "
    "there is no override, so the artifact has none of the Gemma 4 metadata. With gemma4 it was "
    "measured, on an earlier exporter, to leave generic_model set as well. litetune refuses it "
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

# Granite, which that mapping does not cover either, and which needs no container
# for a plainer reason than Qwen3.5.
_NO_CONTAINER_GRANITE = (
    "no container is needed: `GraniteForCausalLM` is a text-only decoder whose attention "
    "projects through q_proj/k_proj/v_proj/o_proj and whose MLP through gate_proj/up_proj/"
    "down_proj, all seven of the names, so `lora` reaches every layer. Read off transformers "
    "5.16.1's modeling_granite.py; peft 0.20.0 has no entry for this architecture to compare "
    "against"
)

# And the one family that mapping does not cover at all.
# Sourced: google-ai-edge/LiteRT-LM#3165 and its fix #3945, both open when read
# on 2026-10-20.
_ENGINE_PER_PROMPT_REASON = (
    "Qwen3.5's gated-delta-net layers carry recurrent state, and on a shared LiteRT-LM Engine "
    "a new conversation that shares a prefix with the previous one skips the shared tokens and "
    "rewinds only the step counter, not that state, so it can continue from the previous "
    "conversation's (google-ai-edge/LiteRT-LM#3165, open, its fix #3945 unmerged): held-out "
    "prompts share their instructions, so one answer could depend on the prompts scored before "
    "it. verify builds an Engine for every "
    "prompt and closes it before the next, which costs an engine load per prompt and is "
    "recorded in the manifest"
)

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
        RequiredFlag(name=TEMPLATE_FLAG, value=override_repo, reason=_TEMPLATE_REASON)
        if override_repo is not None
        else RequiredFlag(
            name=TEMPLATE_FLAG,
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
                value=None,
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
    return packaged_template("functiongemma.jinja")


def packaged_template(name: str) -> str:
    """Path to a prompt template that ships inside this package."""
    return str(files("litetune") / "templates" / name)


TEXT_PARTS_BLOCK = "text_parts.jinja"
"""The packaged block a Qwen rule puts in front of the checkpoint's own template."""

COMPOSED_TEMPLATE = "<the checkpoint's own chat template behind litetune's text-parts block>"
"""The `--jinja_chat_template_override` value that means "compose it at export".

Not a path and not a repository: the template it stands for is the converted
checkpoint's own, which only the export environment can read, so
`export.run_export` writes it there -- `TEXT_PARTS_BLOCK`, then the checkpoint's
template -- and hands the exporter that file in place of this value. A plan that
carries it and cannot be composed exports nothing.
"""

CANONICAL_OR_COMPOSED_TEMPLATE = (
    "<LiteRT-LM's canonical chat template if it renders single turns as the checkpoint's own "
    "does, else the checkpoint's own behind litetune's text-parts block>"
)
"""Like `COMPOSED_TEMPLATE`, for a rule that names a `canonical_template`.

`export.compose_chat_template` reads the checkpoint's template as it does for
`COMPOSED_TEMPLATE` and writes the rule's canonical template in place of this
value when that renders every single-turn case in `template_checks` as the
checkpoint's own does; otherwise the composed one, under the same refusals.
"""

WRITTEN_AT_EXPORT = (COMPOSED_TEMPLATE, CANONICAL_OR_COMPOSED_TEMPLATE)
"""The template override values `export.run_export` replaces with a file it writes."""

QWEN3_CANONICAL_TEMPLATE = "litert-lm-v0.18.0-qwen3.jinja"
"""LiteRT-LM v0.18.0's `models/qwen3/chat_template.jinja`, as packaged here."""

MINICPM5_CANONICAL_TEMPLATE = "litert-lm-v0.18.0-minicpm5-litetune.jinja"
"""LiteRT-LM v0.18.0's `models/minicpm5/chat_template.jinja` with `CANONICAL_EDITS`' edit."""

QWEN35_CANONICAL_TEMPLATE = "litert-lm-v0.18.0-qwen3_5-litetune.jinja"
"""LiteRT-LM v0.18.0's `models/qwen3_5/chat_template.jinja` with `CANONICAL_EDITS`' edits."""

# sha256 of each packaged canonical template's text after its provenance
# comment: the upstream file the comment names by path and git blob, byte for
# byte or with the edits `CANONICAL_EDITS` lists for it.
# `canonical_template_text` refuses a packaged file that is not.
CANONICAL_TEMPLATES = {
    QWEN3_CANONICAL_TEMPLATE: "4315c96833e8b2a21b878d70e8d2f6415416d5e23ed8ee0ae0a111a540d844f7",
    MINICPM5_CANONICAL_TEMPLATE: "e9d0abefd2c9261b2ad585fff7ba9323aa865e96817b5bbfac9de568baa9b7ba",
    QWEN35_CANONICAL_TEMPLATE: "a25cb17845462872f17aee8f7debdc26b40ee8d155f2c499f38a3a4721e8f6d7",
}

# The edits a packaged canonical template makes to the upstream file, as
# (sha256 of the upstream file, ((text, replacement), ...)): each text occurs
# once and is replaced in order. Each makes the template render a single turn
# as the family's own checkpoint template does, which upstream's does not --
# tests/test_template_checks.py renders both before and after, and checks that
# these edits applied to the upstream fixture give the packaged text.
CANONICAL_EDITS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    # LiteRT-LM's adds an empty think block when enable_thinking is undefined;
    # MiniCPM5-1B's own adds nothing then.
    MINICPM5_CANONICAL_TEMPLATE: (
        "a634466cb817628316761c75829c01b5315038c3ed23f626f43026fafcda9535",
        (
            (
                "    {%- if enable_thinking | default(false) -%}\n"
                "        {{- '<think>\\n' -}}\n"
                "    {%- else -%}\n"
                "        {{- '<think>\\n\\n</think>\\n\\n' -}}\n"
                "    {%- endif -%}",
                "    {%- if enable_thinking is defined -%}\n"
                "        {%- if enable_thinking is false -%}\n"
                "            {{- '<think>\\n\\n</think>\\n\\n' -}}\n"
                "        {%- elif enable_thinking is true -%}\n"
                "            {{- '<think>\\n' -}}\n"
                "        {%- endif -%}\n"
                "    {%- endif -%}",
            ),
        ),
    ),
    # Qwen3.5's own trims each message's text and the system message's, and
    # with enable_thinking undefined closes an empty think block; LiteRT-LM's
    # does neither and opens one.
    QWEN35_CANONICAL_TEMPLATE: (
        "ba2e9a48ec90026504236cf463d0d92b06409260382183a147e7377c706be13f",
        (
            (
                "{{- '\\n\\n' + format_content(messages[0]['content']) -}}",
                "{{- '\\n\\n' + format_content(messages[0]['content']) | trim -}}",
            ),
            (
                "{{- format_content(messages[0]['content']) -}}",
                "{{- format_content(messages[0]['content']) | trim -}}",
            ),
            (
                "{%- set content = format_content(message['content']) -%}",
                "{%- set content = format_content(message['content']) | trim -%}",
            ),
            (
                "{%- if enable_thinking | default(true) -%}",
                "{%- if enable_thinking is defined and enable_thinking is true -%}",
            ),
        ),
    ),
}

# Where a packaged canonical template's provenance comment ends; the upstream
# text follows it.
CANONICAL_HEADER_END = "-#}\n"

# The chat templates each packaged canonical template is LiteRT-LM's version
# of, by the sha256 of the template's text as a checkpoint ships it -- the
# `chat_template` of its tokenizer_config.json, which is what transformers'
# AutoTokenizer reads and `export._COMPOSE_SCRIPT` hashes -- and where those
# bytes were read. A checkpoint that ships one of these unchanged is in that
# model family's own chat format, and LiteRT-LM v0.18.0 publishes
# `models/qwen3/chat_template.jinja` as its template for Qwen 3's format
# (`models/qwen3/README.md`: "Canonical LiteRT-LM prompt templates and
# metadata configuration for the Qwen 3 model family"; the conversion it
# describes passes the metadata carrying it). `export._canonical_basis` offers
# the canonical template on that ground to a checkpoint with no training
# record, and it is chosen only if the single-turn renderings match. That is a
# compatibility heuristic, not training provenance: these bytes say which chat
# format a checkpoint's tokenizer is configured for, not what its weights were
# trained on.
#
# Read from the Hub on 2026-10-14: Qwen/Qwen3-1.7B (70d244c), -4B (1cfa9a7),
# -8B (b968826), -14B (40c0698), -32B (9216db5) and -30B-A3B (ad44e77) ship
# these same bytes, so the one digest is theirs too -- though the qwen-3 rule
# claims only names containing Qwen3-0.6B or Qwen3-1.7B, so of them only the
# 1.7B reaches this.
# Qwen/Qwen3-0.6B-Base at da87bfb ships other bytes (sha256 87a2728c...), and
# is left out: nothing here establishes which format a base model is in.
#
# MiniCPM5-1B and Qwen3.5 ship their templates as chat_template.jinja, which
# transformers' AutoTokenizer reads as it reads a tokenizer_config.json's: read
# from the Hub on 2026-10-20, openbmb/MiniCPM5-1B at 87179e5 has no
# `chat_template` in tokenizer_config.json, and Qwen/Qwen3.5-0.8B (2fc0636) and
# -2B (15852e8) ship the same bytes in both places, so the one digest is both
# sizes'.
CANONICAL_UPSTREAM: dict[str, dict[str, str]] = {
    QWEN3_CANONICAL_TEMPLATE: {
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8": (
            "Qwen/Qwen3-0.6B's own at c1899de289a04d12100db370d81485cdf75e47ca "
            "(tokenizer_config.json; tests/fixtures/qwen3-0.6b-c1899de.chat_template.jinja)"
        ),
    },
    MINICPM5_CANONICAL_TEMPLATE: {
        "7451a05cf1e28a79d97d7c0bc951028c0b1915119bf9046acd06a0e3d931f47c": (
            "openbmb/MiniCPM5-1B's own at 87179e5c1f455ef22e6223592d2d61351b525bfc "
            "(chat_template.jinja; tests/fixtures/minicpm5-1b-87179e5.chat_template.jinja)"
        ),
    },
    QWEN35_CANONICAL_TEMPLATE: {
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80": (
            "Qwen/Qwen3.5-0.8B's own at 2fc06364715b967f1860aea9cf38778875588b17 "
            "(chat_template.jinja; tests/fixtures/qwen3.5-0.8b-2fc0636.chat_template.jinja)"
        ),
    },
}


def canonical_template_text(name: str) -> str:
    """The packaged canonical template `name`, provenance comment included.

    Raises ValueError when the text after the comment is not the text its
    sha256 is pinned to -- the upstream file, or it with `CANONICAL_EDITS`' edits:
    a bundle must carry what the comment says it does.
    """
    text = Path(packaged_template(name)).read_bytes().decode("utf-8")
    header, found, body = text.partition(CANONICAL_HEADER_END)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if not found or not header.startswith("{#-") or digest != CANONICAL_TEMPLATES.get(name):
        raise ValueError(
            f"the packaged {name} is not the template its sha256 is pinned to "
            f"({CANONICAL_TEMPLATES.get(name)}); after its comment it hashes to {digest}"
        )
    return text


# Sourced from the runtime and measured on it. From LiteRT-LM v0.18.0
# `ModelDataProcessor::MessageToTemplateInput` (`model_data_processor.h`) hands
# the template every message's content as a list of parts, through
# `NormalizeMessageContent` (`data_utils.cc`, LiteRT-LM #3544), and the Qwen
# processor's own override, which handed a single text part over as the string
# in v0.17.1 (`qwen3_data_processor.cc`), is gone. Both Qwen checkpoints'
# templates read `content` as a string.
#
# "As the exporter reads it": given no override and its default
# `use_jinja_template`, litert-torch's exporter embeds the `chat_template` of
# `transformers.AutoTokenizer.from_pretrained` on the model it converts
# (`export_lib.py`, `litert_lm_builder.py` `package_model`).
# `export.compose_chat_template` loads the tokenizer the same way, in the same
# environment, and falls back to `chat_template.json` as `export_lib.py:390-402`
# does. Where the two part: a tokenizer carrying several named templates, of
# which litetune takes "default" where the exporter would unpack the dict into
# three prompt parts (`litert_lm_builder.py:320`), raising unless it holds
# exactly three; and
# what it refuses -- no template after that fallback, one that is only
# whitespace, one with a carriage return (the exporter reads the override file
# in text mode, `litert_lm_builder.py:430`, so it could not arrive unchanged),
# and one the block is not for.
#
# The cases and the renderer are `template_checks`': the checkpoint's side as
# training rendered it, the bundle's as the export environment's Jinja2
# configured as LiteRT-LM v0.18.0 configures MiniJinja.
_SINGLE_TURN_SAID = (
    "one that, given single-turn prompts -- with and without a system message, the text in one "
    "part and in two, enable_thinking undefined, true and false, with no tools and with one -- "
    "as parts behind the block, does not print what the checkpoint's template prints given them "
    "as strings through transformers, the way training rendered them"
)
_RENDERERS_SAID = (
    "The bundle's side is rendered by the export environment's Jinja2 configured as LiteRT-LM's "
    "MiniJinja, which is not the runtime: a template using what that renderer does not vouch "
    "for is not checked, and verify's rendering check is what checks the runtime, on single "
    "turns. Whether a bundle holds a conversation past its first message, in the contexts "
    "checked, is checked after export for every family, and a bundle that does not says so in "
    "the report"
)

_COMPOSED_SAID = (
    "At export litetune "
    "reads the checkpoint's own chat template as the exporter reads it, through transformers' "
    "AutoTokenizer on the model being converted, and hands the exporter that template, "
    "unchanged, behind a block that turns a list of text parts back into the string, then "
    "reads each bundle's template back -- so a checkpoint this rule claims keeps its own "
    "template, or no bundle is handed over: one that does not carry it fails its recipe, and "
    "litetune tries to rename it with a '.refused' suffix, naming the path it ends up at. It "
    f"refuses a template the block is not for: {_SINGLE_TURN_SAID}. {_RENDERERS_SAID}"
)

_TEXT_PARTS_REASON = (
    "LiteRT-LM from 0.18.0 hands a bundle's template each message's content as a list of "
    "parts, and the chat templates of Qwen3 0.6B and Qwen2.5 0.5B Instruct read it as a "
    "string. Measured 2026-10-08 on litert-lm 0.18.0: a Qwen3 0.6B bundle carrying its "
    "checkpoint's template rendered '<|im_start|>user\\n<|im_end|>' -- the user's text gone, "
    "no error -- where 0.17.1 rendered it; with Qwen2.5's template every send failed, because "
    "it adds the list to a string, which the runtime's MiniJinja refuses. " + _COMPOSED_SAID
)

# granite-4.1-3b's template at c065040 reads a list of parts itself, and joins the
# text parts with a newline, where training rendered the one string: the two-part
# cases in tests/test_template_checks.py.
_GRANITE_TEMPLATE_REASON = (
    "LiteRT-LM from 0.18.0 hands a bundle's template each message's content as a list of "
    "parts. granite-4.1-3b's chat template reads such a list itself, but joins its text "
    "parts with a newline, so a message the runtime hands over in more than one part renders "
    "otherwise than the one string training rendered. " + _COMPOSED_SAID
)


# Sourced. Qwen3 0.6B's template (`Qwen/Qwen3-0.6B` at c1899de, the fixture
# `tests/fixtures/qwen3-0.6b-c1899de.chat_template.jinja`) renders an assistant
# turn after `ns.last_query_index` -- the last user message -- with
# `<think>\n`, its reasoning and `\n</think>\n\n`, and an earlier one without:
# the reply a conversation has just received is rendered one way while it is
# last and the other once the next user message arrives, and
# `Conversation::GetSingleTurnTextFromFullHistory` (`conversation.cc`, LiteRT-LM
# v0.18.0) refuses the second rendering because it does not begin with the
# first (google-ai-edge/LiteRT-LM#3443). LiteRT-LM v0.18.0's own Qwen 3
# template (`models/qwen3/chat_template.jinja`) prints a reply's text as it is
# wherever the reply falls, and reads the list of parts itself. Which of the
# two a checkpoint renders, and how each renders single turns, is
# `template_checks`' to show, and tests/test_template_checks.py renders both.
def _canonical_said(packaged: str, renderer: str, own: str, format_said: str) -> str:
    """How litetune chooses between the rule's packaged canonical template and the composed
    one at export, for a family whose canonical template is named `packaged` and then
    `renderer`, offered without a training record to a checkpoint whose template is
    `own`'s, byte for byte."""
    return (
        f"At export litetune reads the checkpoint's own chat template as the exporter reads it, "
        "through transformers' AutoTokenizer on the model being converted, and hands the exporter "
        f"{packaged} only when {renderer} renders single-turn prompts (with and without "
        "a system message, the text in one part and in two, enable_thinking undefined, true and "
        "false, with no tools and with one) exactly as the checkpoint's own renders them, and "
        "either the litetune.json `tune` wrote beside the checkpoint records that it was trained "
        "runtime_rendered without tool declarations -- one user message rendered through that "
        f"template -- or there is no litetune.json and the checkpoint's template is {own}'s "
        f"own, byte for byte: {format_said}. "
        "Otherwise -- a prerendered run, a run with declarations, which the record keeps only as a "
        "digest, a record it cannot read, no record and any other template, or a template that "
        "renders otherwise -- it hands the exporter the checkpoint's "
        "template behind a block that turns a list of text parts back into the string, and refuses "
        "a template the block is not for: "
        f"{_SINGLE_TURN_SAID}. It reads each bundle's template back, and one that does not carry "
        "the file handed over fails its recipe. " + _RENDERERS_SAID
    )


_QWEN3_TEMPLATE_REASON = (
    "LiteRT-LM from 0.18.0 hands a bundle's template each message's content as a list of "
    "parts, which Qwen3 0.6B's chat template reads as a string (measured 2026-10-08 on "
    "litert-lm 0.18.0: a bundle carrying it rendered '<|im_start|>user\\n<|im_end|>'), and "
    "LiteRT-LM renders a conversation's whole history again on every message and refuses one "
    "that does not begin with the previous rendering, which that template's think block breaks: "
    "it renders the latest reply with `<think>...</think>` and an earlier one without, so the "
    "second message of every conversation fails with INTERNAL (google-ai-edge/LiteRT-LM#3443). "
    "LiteRT-LM v0.18.0 publishes its own Qwen 3 template (models/qwen3/chat_template.jinja), "
    f"which litetune packages as {QWEN3_CANONICAL_TEMPLATE} with its sha256 pinned. "
) + _canonical_said(
    "LiteRT-LM's template",
    "LiteRT-LM's",
    "Qwen/Qwen3-0.6B",
    "Qwen 3's own chat format, which LiteRT-LM " "publishes that template for",
)

# Sourced: the fixtures `minicpm5-1b-87179e5.chat_template.jinja` (lines 30-40
# keep content only `if message.content is string`; line 6 passes `ensure_ascii`
# to `tojson`) and `litert-lm-v0.18.0-minicpm5.chat_template.jinja`, rendered in
# tests/test_template_checks.py.
_MINICPM5_TEMPLATE_REASON = (
    "LiteRT-LM from 0.18.0 hands a bundle's template each message's content as a list of "
    "parts, which MiniCPM5-1B's chat template keeps only when it is a string, so the user's "
    "text renders empty; behind litetune's text-parts block its tools branch passes "
    "`ensure_ascii` to `tojson`, which LiteRT-LM's MiniJinja does not take "
    "(google-ai-edge/LiteRT-LM#3576), and a reply with a think block renders otherwise once "
    "the next message arrives, which LiteRT-LM refuses from a conversation's second message. "
    "LiteRT-LM v0.18.0 publishes its own MiniCPM5 template (models/minicpm5/chat_template.jinja), "
    "which reads parts and holds a conversation, but with enable_thinking undefined adds an "
    "empty think block the checkpoint's own does not add. litetune packages it as "
    f"{MINICPM5_CANONICAL_TEMPLATE} with that one edit, so that it renders as the checkpoint's "
    "own, its sha256 pinned. "
) + _canonical_said(
    "that template",
    "it",
    "openbmb/MiniCPM5-1B",
    "MiniCPM5's own chat format, which LiteRT-LM " "publishes its template for",
)

# Sourced: the fixtures `qwen3.5-0.8b-2fc0636.chat_template.jinja` and
# `litert-lm-v0.18.0-qwen3_5.chat_template.jinja`, rendered in
# tests/test_template_checks.py.
_QWEN35_TEMPLATE_REASON = (
    "Qwen3.5's chat template reads a list of parts itself, but LiteRT-LM renders a "
    "conversation's whole history again on every message and refuses one that does not "
    "begin with the previous rendering, and that template renders the latest reply with a "
    "think block and an earlier one without, so the second message of a conversation fails "
    "(google-ai-edge/LiteRT-LM#3166). LiteRT-LM v0.18.0 publishes its own Qwen 3.5 template "
    "(models/qwen3_5/chat_template.jinja), which holds a conversation, but does not trim a "
    "message's text as the checkpoint's own does, and with enable_thinking undefined opens a "
    "think block where the 0.8B's and 2B's close an empty one. litetune packages it as "
    f"{QWEN35_CANONICAL_TEMPLATE} with those four edits, so that it renders as the checkpoint's "
    "own, its sha256 pinned. "
) + _canonical_said(
    "that template",
    "it",
    "Qwen/Qwen3.5-0.8B",
    "Qwen 3.5's own chat format, which LiteRT-LM " "publishes its template for",
)


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


# Formats a Hub repack names after the model it repacks, as a whole token of
# the normalised name: FP8, MLX, GGUF, ONNX, AWQ, GPTQ, bitsandbytes (`bnb`)
# and an n-bit build (`4bit`). `_NOT_A_REPACK` refuses a match that continues,
# past any number of other tokens, into one of them.
_REPACK_MARKERS = r"(?:fp8|mlx|gguf|onnx|awq|gptq|bnb|\d+bit)"
_NOT_A_REPACK = rf"(?!(?:-[a-z0-9]+)*?-{_REPACK_MARKERS}(?:-|$))"


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
                name=TEMPLATE_FLAG,
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
        # The template, as for Qwen 3, but LiteRT-LM's with four edits
        # (`CANONICAL_EDITS`): see `_QWEN35_TEMPLATE_REASON`. Every size is
        # claimed, as before; the canonical template is offered without a
        # training record only to the 0.8B's and 2B's bytes (`CANONICAL_UPSTREAM`),
        # and chosen for any checkpoint only when its single turns render as
        # that checkpoint's own template renders them.
        required_flags=(
            RequiredFlag(
                name=TEMPLATE_FLAG,
                value=CANONICAL_OR_COMPOSED_TEMPLATE,
                reason=_QWEN35_TEMPLATE_REASON,
            ),
        ),
        canonical_template=QWEN35_CANONICAL_TEMPLATE,
        engine_per_prompt_reason=_ENGINE_PER_PROMPT_REASON,
    ),
    ModelRules(
        family="qwen-3",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # No type override. `qwen3` is on the exporter's own type list (the
        # model-type trap, above), so a config that says `model_type: "qwen3"`
        # is typed correctly with none. Measured 2026-09-14 on Qwen/Qwen3-0.6B:
        # both int8 recipes exported with no flag from litetune, and the
        # conversion cost is in MEASUREMENTS.md. The template override below is
        # LiteRT-LM 0.18.0's, not the exporter's: see `_QWEN3_TEMPLATE_REASON`,
        # and `_TEXT_PARTS_REASON` for the composed fallback. What
        # the artifact's own `llm_model_type` reads is not recorded here: that
        # observation appears in no manifest or log of any run, and was struck
        # from the documents for the same reason.
        #
        # By size, not `qwen-?3`: that also claims the other sizes and the
        # Qwen 3 models built on other architectures, none of which this
        # project has run. No `min_transformers` either -- none was measured.
        # Sizes are added here as they are.
        #
        # Qwen3-1.7B: Qwen/Qwen3-1.7B at 70d244c declares the same
        # `model_type` (`qwen3`) and architecture (`Qwen3ForCausalLM`) as the
        # 0.6B, and its tokenizer_config.json's chat template is the 0.6B's at
        # c1899de byte for byte (`CANONICAL_UPSTREAM`, read 2026-10-14 and
        # again 2026-10-19), so what this rule adds -- the template choice and
        # its refusals -- is decided on the same bytes. Measured 2026-10-19 on
        # banking77 (MEASUREMENTS.md, "The two Qwen families, one size up"): both
        # int8 recipes exported with this rule's template override and convert's
        # default --externalize_embedder, no other flag, packed the
        # canonical template and verified; the dynamic_wi8_afp32 bundle held a
        # LiteRT-LM conversation (the other was not run in one).
        #
        # Not by organisation, variant or revision, but by name, as every rule
        # here matches: a Hub id is matched on its spelling alone, and
        # `hint_for` reads `config.json` and `tune`'s sidecar only for a local
        # directory. So a name that still contains Qwen3-0.6B is claimed --
        # Qwen3-0.6B-Base, a copy or a fine-tune under another organisation
        # that kept the name, a local snapshot -- and a fine-tune renamed past
        # it is not. Base, which no run here measured, has been claimed since
        # this pattern was first written by size alone. Claiming a checkpoint
        # no longer hands it another's template, as litetune 0.4.0 did when it
        # packed Qwen/Qwen3-0.6B's at c1899de for every checkpoint it claimed,
        # and so narrowed this pattern to the checkpoints that one template had
        # been checked against: `CANONICAL_OR_COMPOSED_TEMPLATE` is chosen
        # against the converted checkpoint's own template, LiteRT-LM's only when
        # it renders single turns as that one does, and
        # `export.compose_chat_template` refuses one the block is not for.
        #
        # Not a repack the exporter cannot load (`repack_guard`). For a Hub id
        # that is read from the name: one that continues past the size into one
        # of `_REPACK_MARKERS` names already-quantized or non-PyTorch weights
        # (`Qwen/Qwen3-0.6B-FP8`, `Qwen/Qwen3-0.6B-MLX-4bit`,
        # `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`, `Qwen/Qwen3-0.6B-GGUF`,
        # `onnx-community/Qwen3-0.6B-ONNX`). The export environment's
        # transformers 5.16.1 loads neither FP8 nor bitsandbytes weights without
        # accelerate, which that environment does not install
        # (`quantizer_finegrained_fp8.py:31-32`, `quantizer_bnb_4bit.py:57-64`),
        # and refuses a quantization config with no `quant_method`, as MLX
        # writes it (`quantizers/auto.py:158-166`). Claiming those would trade
        # the unknown-family note for an exporter error. A name cannot show a
        # format it does not spell after the size: an MLX or ONNX build
        # published under the bare model name is still claimed, whatever its
        # repository holds, and if that is not weights transformers loads the
        # exporter fails at its model load. For a local directory it is read
        # from the checkpoint: a `quantization_config` in its `config.json`. A
        # full-precision merge whose config's `_name_or_path` names the bnb
        # repack it was tuned from is claimed, which a name read would refuse.
        patterns=(r"qwen-?3-(?:0-6b|1-7b)",),
        repack_guard=True,
        required_flags=(
            RequiredFlag(
                name=TEMPLATE_FLAG,
                value=CANONICAL_OR_COMPOSED_TEMPLATE,
                reason=_QWEN3_TEMPLATE_REASON,
            ),
        ),
        canonical_template=QWEN3_CANONICAL_TEMPLATE,
    ),
    ModelRules(
        family="qwen-2.5",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # No type override either, and this entry says so with a run behind
        # it. `config.json` declares `model_type: "qwen2"`, which
        # `litert_lm_builder.py` matches as `case 'qwen2' | 'qwen2p5'`, so no
        # override: unlike the gemma3_text families there is no ambiguity for
        # one to resolve. The template override is LiteRT-LM 0.18.0's, as for
        # Qwen 3. Qwen 3 is typed from its own config in the same way,
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
        # By size *and* variant: `Qwen2.5-0.5B-Instruct` was run and
        # `Qwen2.5-0.5B` was not. They are two checkpoints, and this file's own
        # history is that a rule written on the strength of a neighbouring one
        # goes unexercised for weeks -- the gemma-3-text rule claimed the 1B for
        # three weeks before anything exported it. (The qwen-3 rule claims its
        # base model, which it has done since its pattern was first written by
        # size alone; this rule has never claimed this one's.)
        #
        # The tail is what makes that true. `identify` searches rather than
        # matches, so a bare `qwen-?2-5-0-5b-instruct` also claims
        # any id that continues past it -- an `-AWQ`, a `-GPTQ-Int4`, a
        # bnb-4bit repack -- which name already-quantized checkpoints nobody
        # here exported, and for which what this rule records is least likely
        # to hold. The tests pin the regex against those spellings; whether each
        # repository exists is not something this checks. So the
        # pattern ends either at the end of the hint text or at the
        # `model_type` `hint_for` appends for a local checkpoint: this run's
        # merged model read `qwen-qwen2-5-0-5b-instruct-qwen2-qwen2forcausallm`
        # and must keep matching, a repack must not. A Hub id is also held to
        # the qwen-3 rule's `repack_guard` after the tail, because `-qwen2` can
        # begin a repack's own suffix (`someorg/Qwen2.5-0.5B-Instruct-qwen2-GGUF`);
        # a local directory to its own `config.json`, as there. A name cannot show a
        # format it does not spell after the model name: an ONNX export
        # published under the bare name, `onnx-community/Qwen2.5-0.5B-Instruct`,
        # is claimed whatever its repository holds, and if that is not weights
        # transformers loads the exporter fails at its model load.
        #
        # Qwen2.5-1.5B-Instruct, by the same size-and-variant rule:
        # Qwen/Qwen2.5-1.5B-Instruct at 989aa79 declares `model_type: "qwen2"`
        # and `Qwen2ForCausalLM`, as the 0.5B does, and its tokenizer_config.json's
        # chat template is the 0.5B's at 7ae5576 byte for byte (read
        # 2026-10-19), so the composition and the two reasons below for no
        # `canonical_template` rest on the same bytes. The base 1.5B is not
        # claimed, as the base 0.5B is not. Measured 2026-10-19 on banking77
        # (MEASUREMENTS.md, "The two Qwen families, one size up"): both int8
        # recipes exported with this rule's template override and convert's default
        # --externalize_embedder, no other flag, composed and verified; the
        # dynamic_wi8_afp32 bundle held a LiteRT-LM conversation (the other was
        # not run in one).
        #
        # No `min_transformers` -- the pinned 5.16.1 loaded it, and nothing
        # here establishes a floor, which is not the same as there being none.
        #
        # Under any organisation, as before litetune 0.4.0: the template is
        # composed at export from the checkpoint's own (`COMPOSED_TEMPLATE`), so
        # the same name under another organisation -- a mirror, or a fine-tune
        # that kept it -- keeps its own template, and 0.4.0's reason for
        # claiming only Qwen's repository -- it packed that repository's
        # template at 7ae5576 -- is gone. The optional snapshot
        # segment 0.4.0 added stays: a local snapshot of the Hub repository
        # (`.../models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<sha>`, which
        # `tune` records verbatim when it trained from one) normalises to the
        # name followed by `-snapshots-<sha>`, which the tail alone refuses.
        #
        # No `canonical_template`, unlike Qwen 3, for two reasons rendered in
        # tests/test_template_checks.py on Qwen2.5-0.5B-Instruct's template at
        # 7ae5576. Composed behind the block it already holds a conversation:
        # every case in `template_checks.CONVERSATIONS`, in every context in
        # `template_checks.CONTEXTS`, renders as a prefix of the next. And
        # LiteRT-LM v0.18.0's `models/qwen2_5/chat_template.jinja` does not
        # render single turns as it does: with no system message the
        # checkpoint's template prints its default one ("You are Qwen, created
        # by Alibaba Cloud. You are a helpful assistant.") and LiteRT-LM's
        # prints none, so 48 of the 96 single-turn renderings differ: exactly
        # those without a system message, in every context.
        patterns=(r"qwen-?2-5-(?:0-5b|1-5b)-instruct(?:-snapshots-[0-9a-f]{40})?(?:$|-qwen2\b)",),
        repack_guard=True,
        required_flags=(
            RequiredFlag(
                name=TEMPLATE_FLAG,
                value=COMPOSED_TEMPLATE,
                reason=_TEXT_PARTS_REASON,
            ),
        ),
    ),
    ModelRules(
        family="granite-4.1",
        lora_container_reason=_NO_CONTAINER_GRANITE,
        # No type override. `config.json` at c065040 declares `model_type: "granite"`
        # and `GraniteForCausalLM`, a dense decoder with no hybrid layers, which
        # the pinned exporter takes on its generic path and packs as
        # `generic_model`: there is no granite type in litert_lm_builder 0.18.0
        # to override to. The checkpoint's own chat template (chat_template.jinja
        # at c065040, `tests/fixtures/granite-4.1-3b-c065040.chat_template.jinja`)
        # reads a list of parts itself but joins the text parts with a newline,
        # so through the runtime's renderer it prints the two-part cases in
        # `template_checks.SINGLE_TURNS` otherwise than transformers prints the
        # string; behind the text-parts block it prints every case as
        # transformers does, and both hold every conversation in
        # `template_checks.CONVERSATIONS` (tests/test_template_checks.py). So the
        # composed template, as for Qwen2.5, and no `canonical_template`: LiteRT-LM
        # v0.18.0 publishes none for Granite. Its BOS is its EOS,
        # `<|end_of_text|>`, and the template prints neither at the start.
        #
        # By size and with the tail the qwen-2.5 rule uses, for its reasons: the
        # 3B is the one size this project has read, and a bare
        # `granite-?4-1-3b` would also claim `granite-4.1-3b-base`, whose
        # template nothing here has read, and any repack named after it. A
        # local checkpoint's hint ends in `-granite-graniteforcausallm`.
        patterns=(r"granite-?4-1-3b(?:-snapshots-[0-9a-f]{40})?(?:$|-granite\b)",),
        repack_guard=True,
        required_flags=(
            RequiredFlag(
                name=TEMPLATE_FLAG,
                value=COMPOSED_TEMPLATE,
                reason=_GRANITE_TEMPLATE_REASON,
            ),
        ),
    ),
    ModelRules(
        family="minicpm5",
        lora_container_reason=_NO_CONTAINER_UPSTREAM,
        # No type override. `config.json` at 87179e5 declares `model_type:
        # "llama"` and `LlamaForCausalLM`, which the pinned exporter takes on its
        # generic path and packs as `generic_model`. LiteRT-LM 0.18.0 has a
        # `minicpm5` type, reachable only by a metadata override, which litetune
        # refuses, or a repack; what it adds is a tool-call parser litetune's text
        # path does not use. The template is LiteRT-LM's with one edit: see
        # `_MINICPM5_TEMPLATE_REASON`.
        #
        # By size and with the qwen-2.5 rule's tail, for its reasons: openbmb
        # publishes -1B-Base, -1B-SFT, -1B-MLX and -1B-GGUF beside -1B (the Hub's
        # listing, 2026-10-20), whose templates nothing here has read. A local
        # checkpoint's hint ends in `-llama-llamaforcausallm`.
        patterns=(r"minicpm-?5-1b(?:-snapshots-[0-9a-f]{40})?(?:$|-llama\b)",),
        repack_guard=True,
        required_flags=(
            RequiredFlag(
                name=TEMPLATE_FLAG,
                value=CANONICAL_OR_COMPOSED_TEMPLATE,
                reason=_MINICPM5_TEMPLATE_REASON,
            ),
        ),
        canonical_template=MINICPM5_CANONICAL_TEMPLATE,
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
    # A directory on this machine, whose own files say what it holds; a Hub id
    # has only its name. `quantized` is what the directory's `config.json` says:
    # a `quantization_config`, which is what makes transformers load the weights
    # as pre-quantized (`quantizers/auto.py:334-338`, transformers 5.16.1).
    local: bool = False
    quantized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "matched_against": self.text,
            "from_config": list(self.from_config),
            "config_error": self.config_error,
            "identity_recorded": self.recorded,
            "provenance_error": self.provenance_error,
            "local": self.local,
            "quantized": self.quantized,
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


def read_training_record(reference: str) -> tuple[dict, Path] | None:
    """The checkpoint's own record and where it was read from, or `None` for no record.

    One reader for every question asked of `litetune.json`: the base model
    here (`_provenance`), the mode and declarations in `verify`, and in
    `export` whether a canonical chat template is offered on a record's
    absence. None of them may mistake an unreadable record for none, and the
    distinction below is subtle enough that a second copy would drift from it,
    and the two copies would then disagree about whether a checkpoint said
    anything. Here rather than in `verify`, which imports this module.

    `None` -- no record -- in two cases only: the reference is not a
    directory, links followed (a Hugging Face id, a file, a path that is not
    there); or it is one, and holds no `litetune.json` entry. Everything else
    raises: a link on the way to the reference that no longer resolves
    (`_refuse_a_broken_link_in`), and a `litetune.json` entry that cannot be
    read -- a link going nowhere, or through a file, one that is not a regular
    file, a FIFO or a device, without being opened, as far as
    `_read_regular_file` says it can, one that cannot be opened, or text that
    is not a JSON object.
    """
    reference_path = Path(reference)
    sidecar = reference_path / PROVENANCE_NAME
    if not _is_a_directory(reference_path):
        return None
    try:
        os.lstat(sidecar)
    except FileNotFoundError:
        return None
    # The entry is there, so every failure from here on propagates -- a
    # permission, a stale mount, a link that no longer reaches a file --
    # because falling through to the contract, to inference, to "no
    # declarations recorded" or to a family guessed from the path would
    # silently replace what the checkpoint wrote down.
    text = _read_regular_file(sidecar).decode("utf-8")
    try:
        data = json.loads(text)
    except RecursionError as exc:
        # Nested past the interpreter's recursion limit: as unreadable as text
        # that is not JSON, and said the same way, so that every reader's
        # `ValueError` covers it.
        raise ValueError(f"{sidecar} is nested too deeply to read") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar} does not contain a JSON object")
    return data, sidecar


def _is_a_directory(path: Path) -> bool:
    """Whether `path`, links followed, is a directory; an error for a broken link on the way."""
    try:
        return stat.S_ISDIR(os.stat(path).st_mode)
    except (FileNotFoundError, NotADirectoryError):
        _refuse_a_broken_link_in(path)
        return False


# Python defines `stat.IO_REPARSE_TAG_MOUNT_POINT` on Windows alone; the value
# is Windows' own for a directory junction (winnt.h), and nothing else reports
# an `st_reparse_tag`, so on another system nothing matches it.
_MOUNT_POINT_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def _is_link(entry: os.stat_result) -> bool:
    # A symbolic link, or on Windows a directory junction, which `os.lstat`
    # opens itself rather than its target (Python 3.8 and later) and reports by
    # its reparse tag: `os.path.isjunction` is 3.12 and later, and litetune
    # runs on 3.10.
    return stat.S_ISLNK(entry.st_mode) or getattr(entry, "st_reparse_tag", 0) == _MOUNT_POINT_TAG


def _refuse_a_broken_link_in(path: Path) -> None:
    """Raise if a link on the way to `path` -- `path` itself included -- no longer resolves.

    Called once `path` is known not to be there. Each of its prefixes is looked
    at in turn, without following it: the first that is not there, or that
    lies under a file, ends the path, which is then simply absent. A prefix
    that is a link (`_is_link`) and does not resolve is something that once
    led to a checkpoint and no longer does: an error, not an absence.
    """
    for prefix in (*reversed(path.parents), path):
        try:
            entry = os.lstat(prefix)
        except (FileNotFoundError, NotADirectoryError):
            return
        if _is_link(entry):
            try:
                os.stat(prefix)
            except (FileNotFoundError, NotADirectoryError) as exc:
                raise FileNotFoundError(
                    f"{prefix}, a link on the way to {path}, no longer resolves"
                ) from exc


_NOT_REGULAR = (
    (stat.S_ISDIR, "a directory"),
    (stat.S_ISFIFO, "a FIFO"),
    (stat.S_ISCHR, "a character device"),
    (stat.S_ISBLK, "a block device"),
    (stat.S_ISSOCK, "a socket"),
)


def _refuse_unless_regular(path: Path, mode: int) -> None:
    if stat.S_ISREG(mode):
        return
    kind = next((name for test, name in _NOT_REGULAR if test(mode)), "not a file")
    raise OSError(f"{path} is not a regular file: it is {kind}")


def _read_regular_file(path: Path) -> bytes:
    """The bytes of `path`, a regular file once its links are followed, or `OSError`.

    Looked at before it is opened, because opening a FIFO waits for a writer
    and reading a device need not end; and looked at again through the
    descriptor once open. On POSIX the open does not wait (`O_NONBLOCK`), so the
    second look closes the gap in which the entry could be replaced after the
    first. Windows has neither `O_NONBLOCK` nor `os.mkfifo` (both Unix-only in
    Python's `os` documentation), so there the guard is the first look alone:
    an entry replaced between the look and the open is not caught before the
    open.
    """
    _refuse_unless_regular(path, os.stat(path).st_mode)
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags), "rb") as handle:
        _refuse_unless_regular(path, os.fstat(handle.fileno()).st_mode)
        return handle.read()


def _provenance(path: Path) -> tuple[str | None, str | None]:
    """The base model `tune` recorded beside this checkpoint. Returns (id, error).

    An unreadable sidecar is not the same as an absent one, and returning `None`
    for both was the one place in this module that failed to say which. A file
    truncated by a full disk or half-copied by an interrupted `rsync` would then
    look like a checkpoint that never recorded anything -- and be guessed at from
    the path, which is the fault this sidecar exists to remove.

    Absent is what `read_training_record` says is absent, and nothing else: a
    `litetune.json` that is there and is a directory, a link going nowhere, a
    FIFO or a device, or that cannot be opened or parsed, or holds no JSON
    object, is an error here, as it is to `verify` and `export`.
    """
    sidecar = path / PROVENANCE_NAME
    try:
        read = read_training_record(str(path))
    except (OSError, ValueError) as exc:
        return None, f"{sidecar} could not be read: {type(exc).__name__}: {exc}"
    if read is None:
        return None, None
    data = read[0]
    base = data.get("base_model")
    if base is None:
        return None, f"{sidecar} records no 'base_model'"
    if not isinstance(base, str) or not base.strip():
        # A list or a dict here would `str()` into something that matches a
        # family regex by accident, which is a confident answer from nonsense.
        return None, f"{sidecar} records a non-string 'base_model': {base!r}"
    return base, None


def recorded_identity(model: str) -> str | None:
    """What the checkpoint at `model` says it came from, if it says anything.

    Asked of every reference: `read_training_record` tells a Hugging Face id
    and anything else that is not a directory from a checkpoint.
    """
    return _provenance(Path(model))[0]


def provenance_error(model: str) -> str | None:
    """Why the checkpoint's own record could not be used, if there was one.

    Read from the path rather than from a plan's hint: when `--base-model` names
    an id, the plan is built from that string and never opens the directory --
    which is exactly when a broken record most needs saying out loud.
    """
    return _provenance(Path(model))[1]


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


# transformers 5.16.1's `AUTO_QUANTIZATION_CONFIG_MAPPING` (`quantizers/auto.py:
# 107-134`): the methods it loads as pre-quantized.
_PRE_QUANTIZED_METHODS = frozenset(
    {
        "awq",
        "bitsandbytes_4bit",
        "bitsandbytes_8bit",
        "eetq",
        "gptq",
        "aqlm",
        "quanto",
        "quark",
        "fouroversix",
        "fp_quant",
        "hqq",
        "compressed-tensors",
        "fbgemm_fp8",
        "higgs",
        "torchao",
        "bitnet",
        "vptq",
        "spqr",
        "fp8",
        "nvfp4",
        "mxfp8",
        "auto-round",
        "mxfp4",
        "metal",
        "sinq",
        "gemma",
    }
)


def _pre_quantized(config: dict[str, Any]) -> bool:
    """Whether transformers 5.16.1 loads this `config.json`'s weights as pre-quantized.

    `get_hf_quantizer` (`quantizers/auto.py:334-340`) takes the config's
    `quantization_config`, or else its text config's -- the config itself when
    it has no sub-config -- and counts it only when `supports_quant_method`
    (`:282-299`) knows its method: `load_in_4bit`/`load_in_8bit` name
    bitsandbytes, an unknown method is skipped with a warning and the weights
    load as they are. A config with no method -- `{}` among them, since an empty
    one falls through to the text config, which for a plain config is itself,
    and is then not None -- makes transformers raise, which is the exporter's
    failure to report, not a repack to turn away.
    """
    text_config = config.get("text_config")
    params = config.get("quantization_config") or (
        text_config.get("quantization_config") if isinstance(text_config, dict) else None
    )
    if not isinstance(params, dict) or not params:
        return False
    method: object = params.get("quant_method")
    if params.get("load_in_8bit", False) or params.get("load_in_4bit", False):
        method = "bitsandbytes_4bit" if params.get("load_in_4bit", False) else "bitsandbytes_8bit"
    return isinstance(method, str) and method in _PRE_QUANTIZED_METHODS


def hint_for(model: str) -> ModelHint:
    """The text `identify` matches against: the model id, plus a local checkpoint."""
    path = Path(model)
    normalised = _normalise(model)
    config = path / CONFIG_NAME
    # Read first, whatever the checkpoint holds: a record that cannot be read
    # is said even where there is no config.json to identify by.
    recorded_base, provenance_error = _provenance(path)
    if not (path.is_dir() and config.is_file()):
        return ModelHint(
            model=model, text=normalised, local=path.is_dir(), provenance_error=provenance_error
        )

    # Before config.json: what produced the checkpoint knows what it was, and
    # config.json only knows what architecture it is -- which is not enough,
    # since FunctionGemma and Gemma 3 declare the same `model_type` and need
    # different overrides.
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
            local=True,
        )
    if not isinstance(data, dict):
        return ModelHint(
            model=model,
            text=normalised,
            config_error=f"{config} does not contain a JSON object",
            recorded=bool(recorded_base),
            provenance_error=provenance_error,
            local=True,
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
    quantized = _pre_quantized(data)
    return ModelHint(
        model=model,
        text=text,
        from_config=extra,
        recorded=bool(recorded_base),
        provenance_error=provenance_error,
        local=True,
        quantized=quantized,
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


# The second sentence is LiteRT-LM v0.18.0's, not a rule's: #3544 took the
# string unwrapping out of the generic processor as well as the Qwen one
# (`generic_data_processor.cc` in v0.17.1; `NormalizeMessageContent` in v0.18.0),
# so it reaches a family no rule names.
UNKNOWN_FAMILY = (
    "litetune has no per-model rules for this checkpoint. That is not a statement that none apply: "
    "the rules it does hold were paid for one model family at a time, and a family it has not met "
    "is a family whose required export flags and minimum toolchain versions are simply unknown "
    "here. One change applies to every family: from LiteRT-LM 0.18.0 the runtime hands the "
    "bundle's template each message's content as a list of parts, and a template written for a "
    "string may render the prompt wrongly or fail -- run verify, whose rendering check is what "
    "shows it"
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
    # What `flags` set, keyword by keyword, as `parse_exporter_flags` reads them.
    table: tuple[ExporterFlag, ...] = ()

    def setting(self, name: str) -> ExporterFlag | None:
        """What the exporter will get for this flag, if the plan sets it."""
        return next((f for f in self.table if f.name == flag_key(name)), None)

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

    Raises `FlagRefused` for a flag that must never be passed, or that the
    exporter would read otherwise than litetune can say (`parse_exporter_flags`).
    Everything else is reported: a flag litetune added is named with its reason,
    and a required flag whose value it cannot determine makes the plan unusable
    rather than guessed at. Every decision reads the parsed table, and `flags`
    is that table, one token per keyword: what the exporter is given.
    """
    hint = hint_for(model)
    rules = rules_for_hint(hint)
    table = parse_exporter_flags(model, requested_flags)

    if rules is None:
        return ExportPlan(
            model=model,
            hint=hint,
            rules=None,
            flags=tuple(f.token for f in table.values()),
            notes=(UNKNOWN_FAMILY,),
            table=tuple(table.values()),
        )

    for forbidden in rules.forbidden_flags:
        setting = table.get(flag_key(forbidden.name))
        if setting is not None and forbidden.matches(setting):
            raise FlagRefused(f"{model}: litetune refuses {setting.token!r}. {forbidden.reason}")

    added: list[str] = []
    notes: list[str] = []
    checks: list[Check] = []
    limitations = list(rules.limitations)
    composes = False
    for required in rules.required_flags:
        supplied = required.satisfied_by(table)
        if supplied is not None and required.takes_value:
            _refuse_unusable_value(model, supplied, required.name)
        if supplied is not None and not required.takes_value and not supplied.value:
            # A switch the family cannot export without, turned off by the
            # caller: their value would win below, and the export would fail or
            # be wrong for the reason the rule gives.
            raise FlagRefused(
                f"{model}: litetune refuses {supplied.token!r}: this family needs "
                f"{required.name}. {required.reason}"
            )
        if supplied is not None and required.value in WRITTEN_AT_EXPORT:
            limitations.append(_CALLER_TEMPLATE.format(flag=supplied.shown))
        if supplied is not None:
            wanted = required.rendered
            detail = f"{supplied.shown} was supplied by the caller"
            if wanted is not None and supplied.shown != wanted:
                # Their value, not litetune's: the caller may have trained
                # against a template litetune has never seen. Recorded, because
                # a difference here is invisible in the artifact.
                detail = (
                    f"{supplied.shown} was supplied by the caller; litetune would have used "
                    f"{wanted}. The caller's value was kept"
                )
            checks.append(
                Check.passed(
                    EXPORT_FLAGS_CHECK,
                    f"{rules.family}: {detail}",
                    observed={
                        "flag": supplied.shown,
                        "token": supplied.token,
                        "family": rules.family,
                    },
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

        composes = composes or required.value in WRITTEN_AT_EXPORT
        table.update(parse_exporter_flags(model, [rendered]))
        added.append(rendered)
        notes.append(f"added {rendered}: {required.reason}")
        checks.append(
            Check.passed(
                EXPORT_FLAGS_CHECK,
                f"{rules.family}: {rendered} was added by litetune — {required.reason}",
                observed={"flag": rendered, "family": rules.family, "added": True},
            )
        )

    # On a rule that composes, whether it did or the caller brought a template:
    # either way the template reaches the bundle only through the metadata the
    # exporter builds, and an override of its own takes that metadata's place.
    if any(required.value in WRITTEN_AT_EXPORT for required in rules.required_flags):
        override = table.get(flag_key(METADATA_OVERRIDE_FLAG))
        if override is not None and override.value:
            reason = _METADATA_OVERRIDE_REASON if composes else _METADATA_OVERRIDE_OWN_REASON
            raise FlagRefused(f"{model}: litetune refuses {override.token!r}. {reason}")

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
        flags=tuple(f.token for f in table.values()),
        added=tuple(added),
        checks=tuple(checks),
        notes=tuple(notes),
        recommendations=tuple(recommendations),
        limitations=tuple(limitations),
        table=tuple(table.values()),
    )


def _refuse_unusable_value(model: str, flag: ExporterFlag, name: str) -> None:
    """Raise `FlagRefused` for a value-bearing flag the exporter would get no value from."""
    if not (isinstance(flag.value, str) and flag.value):
        raise FlagRefused(
            f"{model}: litetune refuses {flag.token!r}. The exporter reads its value as "
            f"{flag.value!r}, {_UNUSABLE_VALUE.format(name=name)}"
        )


def _split_value(model: str, flag: str, name: str) -> str:
    return (
        f"{model}: litetune refuses {flag!r} without a value in the same argument. Fire, which "
        "reads the exporter's command line, takes the next argument as its value when that "
        "argument is not a flag (`fire/core.py:876-921`), and litetune passes each flag on as "
        f"one argument: pass --flag={name}=<value>"
    )


# The exporter tests both value-bearing overrides litetune requires for truth
# and proceeds as if they were absent when falsy: the template at
# `litert_lm_builder.py:414`, the model type at `:380`. A value that is not a
# string is no path, repository or model type either.
_UNUSABLE_VALUE = (
    "which is not a value {name} can take: the exporter would read a falsy one as no override "
    "at all, so the rule would be satisfied in name only. Give it a value, or leave it out for "
    "litetune to add"
)

_CALLER_TEMPLATE = (
    "{flag} was passed by the caller and used as given: litetune put no text-parts block in front "
    "of it. From LiteRT-LM 0.18.0 the runtime hands the template each message's content as a list "
    "of parts, and a template that reads it as a string loses the prompt or fails every send; "
    "verify's rendering check is what shows it"
)

# `package_model` takes a metadata override in place of the LlmMetadata it
# would build (`litert_lm_builder.py:438-456`, then `add_llm_metadata` at
# `:487`), and `build_llm_metadata` is the only place a chat template is
# written into it (`:314-318`).
_METADATA_OVERRIDE_REASON = (
    "The exporter puts a metadata override in the bundle in place of the LlmMetadata it builds, "
    "and the chat template litetune writes at export reaches the bundle only through the "
    "metadata it builds: the bundle would carry the override's template, or none, and not the "
    "one litetune checked. Leave the override out; a template of your own goes in "
    f"--flag={TEMPLATE_FLAG}=<file>, and on LiteRT-LM 0.18.0 it has to read a list of parts itself"
)

_METADATA_OVERRIDE_OWN_REASON = (
    "The exporter puts a metadata override in the bundle in place of the LlmMetadata it builds, "
    f"and the template given with {TEMPLATE_FLAG} reaches the bundle only through the metadata it "
    "builds: the bundle would carry the override's template, or none, and not the one passed. "
    "Pass one or the other; a template written into the override has to read LiteRT-LM 0.18.0's "
    "list of parts itself"
)


def refuse_forbidden_flags(model: str, flags: Sequence[str]) -> None:
    """Raise `FlagRefused` if any flag must never be passed for this model."""
    table = parse_exporter_flags(model, flags)
    rules = identify(model)
    if rules is None:
        return
    for forbidden in rules.forbidden_flags:
        setting = table.get(flag_key(forbidden.name))
        if setting is not None and forbidden.matches(setting):
            raise FlagRefused(f"{model}: litetune refuses {setting.token!r}. {forbidden.reason}")


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
    wanted = wheels.canonical(distribution)
    environment = wheels.marker_environment()
    for requirement in requirements:
        # Read as pip reads it, and only where it applies: a marker is not
        # part of the version, and a pin for another Python is not this one's.
        pinned = wheels.pin(requirement)
        if pinned and wheels.applies(requirement, environment):
            if wheels.canonical(pinned[0]) == wanted:
                return pinned[1]
    return None
