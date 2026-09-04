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
`evaluate.resolve_prompt_mode` -- never inferred from a model id.
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

    def matches(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self.patterns)

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
    "is the recipe that was exported, and litetune has measured neither of these two"
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

_FUNCTION_RESPONSE_REASON = (
    "a FunctionGemma turn does not end at the call: after `<end_function_call>` the "
    "application has to execute the tool and send the result back, and the model must stop "
    "and wait for it. Training cannot reveal this terminator -- the completions end at "
    "`<end_of_turn>`, so a run records that one and nothing else. The .litertlm itself does "
    "carry it, as token id 50 out of generation_config.eos_token_id; this is about the "
    "contract, which a consumer reads without a protobuf parser and which otherwise names "
    "only the terminator the run observed"
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
    ),
    ModelRules(
        family="gemma-3-text",
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
        # Same guard: `Qwen3-5B` would be a Qwen 3, not a Qwen 3.5.
        patterns=(r"qwen-?3-5(?![\db])",),
        min_transformers="5.0.0",
        min_transformers_reason=_TRANSFORMERS_5_REASON,
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "matched_against": self.text,
            "from_config": list(self.from_config),
            "config_error": self.config_error,
        }


def hint_for(model: str) -> ModelHint:
    """The text `identify` matches against: the model id, plus a local config.json."""
    path = Path(model)
    normalised = _normalise(model)
    config = path / CONFIG_NAME
    if not (path.is_dir() and config.is_file()):
        return ModelHint(model=model, text=normalised)

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
        )
    if not isinstance(data, dict):
        return ModelHint(
            model=model,
            text=normalised,
            config_error=f"{config} does not contain a JSON object",
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
    return ModelHint(model=model, text=text, from_config=extra)


def identify(model: str) -> ModelRules | None:
    """The rules for this model, or None when litetune knows of none.

    None is not "no rules apply" -- it is "litetune has no entry for this". The
    difference is reported by every caller.
    """
    hint = hint_for(model)
    return rules_for_hint(hint)


def rules_for_hint(hint: ModelHint) -> ModelRules | None:
    for rules in RULES:
        if rules.matches(hint.text):
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
