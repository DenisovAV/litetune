"""Two properties a bundle's chat template has to have, and how `convert` checks them.

**Single-turn identity.** For single-turn prompts -- with and without a system
message, the text in one part and in two -- the template the bundle carries
renders exactly what the checkpoint's own template renders. That is what
training rendered and what the float reference renders (`prompt_mode.
RENDERING_SOURCE`: one user message, its content a string, through the
tokenizer's `apply_chat_template`), so a template that renders anything else
puts different tokens in front of the model than the ones it learned.

**The conversation prefix invariant.** Rendering a history that ends in an
assistant reply, with no generation prompt, is a byte prefix of rendering that
history plus the next user message with the generation prompt. LiteRT-LM's
`Conversation` renders the whole history again on every message after the
first and requires the new rendering to begin with the previous one
(`Conversation::GetSingleTurnTextFromFullHistory`,
`runtime/conversation/conversation.cc`, v0.18.0); where it does not, the send
fails with INTERNAL, "The new rendered template string does not start with the
previous rendered template string" (google-ai-edge/LiteRT-LM#3443). The first
message of a conversation takes the other branch of that function and is not
compared -- unless the conversation was created with `prefill_preface_on_init`,
which this does not check -- so a template that breaks the invariant answers one
message and fails the second.

What the runtime hands the template, which the cases below reproduce: each
message's content as a list of text parts (`NormalizeMessageContent`,
`model_data_processor/data_utils.cc`), a reply as the text the model produced
(`ResponseTextToMessage`, same file), `bos_token` and `eos_token` empty
(`PromptTemplateInput`, `runtime/components/prompt_template.h`), and no `tools`
unless the conversation declares some (`minijinja_template.rs` leaves a null
out). A bundle whose metadata declares a `thought` channel has the text between
its delimiters taken out of the reply and put in `reasoning_content`
(`ExtractChannelContent` and `InsertChannelContentIntoMessage`,
`runtime/conversation/channel_util.cc`), and the exporter declares one, with
`<think>` and `</think>`, for a template that contains `<think>` when the
metadata declares no channel yet (`build_llm_metadata`, `litert_torch/
generative/export_hf/core/litert_lm_builder.py`, litert-torch-nightly
0.10.0.dev20260926). So a reply with a think block is checked in both shapes.

**The renderer.** LiteRT-LM renders with MiniJinja 2.14.0 (`Cargo.toml`), after
rewriting the Python string methods MiniJinja lacks into filters and tests
(`EditTemplateForMinijinja`, `runtime/components/prompt_template.cc`), in an
environment `create_env` configures (`runtime/components/rust/
minijinja_template.rs`). None of that is installed where `convert` runs: the
export environment has the Jinja2 transformers renders with. `RENDERER_SOURCE`
is that Jinja2 configured as `create_env` configures MiniJinja -- trailing
newlines kept, blocks trimmed, its `tojson`, `lstrip`, `rstrip`,
`raise_exception`, `strftime_now` and `none`, the same rewrite first, the
MiniJinja tests and filters the rewrite produces, no Python methods on a string,
list or mapping, no `{% break %}`, and values printed as MiniJinja prints them.
It is an emulation, and the tests hold it to MiniJinja 2.14.0 itself on every
template and case checked here, and to LiteRT-LM's own goldens for its Qwen 3
template. One difference is known and left: Jinja2's `~` joins a non-string as
Python prints it (`True`, where MiniJinja writes `true`). The reference side of
single-turn identity is what training rendered, transformers'
`apply_chat_template`, run in the export environment by
`export._COMPOSE_SCRIPT`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# The text of the first single-turn case, which every template must print: a
# rendering that does not contain it lost the user's message.
PROBE_TEXT = "litetune-probe-7d1c"

SYSTEM_TEXT = "You are a banking assistant."

# Short, multi-line, padded and non-ASCII: each is a way a template could
# treat the text other than as given.
SINGLE_TURN_TEXTS = (PROBE_TEXT, "line one\nline two", "  spaced out  ", "ünïcödé — ✓ 東京")

# How the runtime may hand one message's text: one part, which is what a string
# becomes (`NormalizeContent`, `data_utils.cc`), or several, which a caller
# sending a list gets. The reference is the checkpoint's template given the
# parts' text joined into one string.
FORMS = ("one_part", "two_parts")

LIMITATION_PREFIX = "this bundle answers one message"


# Python source, put at the top of the scripts `export` runs in the export
# environment, where litetune itself is not installed. Each setting is the
# runtime's: `create_env` and `MinijinjaTemplate::apply_impl` in
# `runtime/components/rust/minijinja_template.rs`, `EditTemplateForMinijinja` in
# `runtime/components/prompt_template.cc` (LiteRT-LM v0.18.0), and for the
# builtins it relies on, MiniJinja 2.14.0 as LiteRT-LM's Bazel build compiles it
# (`crate_features` without `loop_controls`, so no `{% break %}`).
RENDERER_SOURCE = r'''
import json as _json
import re as _re
import time as _time

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

# `EditTemplateForMinijinja`, in its order. RE2's non-greedy `.*?` and
# GlobalReplace read the same in Python's `re.sub`.
MINIJINJA_EDITS = (
    (r"\.startswith\((.*?)\)", r" is startingwith \1"),
    (r"\.endswith\((.*?)\)", r" is endingwith \1"),
    (r"\.replace\((.*?),(.*?)\)", r" | replace(\1,\2)"),
    (r"\.split\((.*?)\)\[0\]", r" | split(\1) | first"),
    (r"\.split\((.*?)\)\[-1\]", r" | split(\1) | last"),
    (r"\.split\((.*?)\)", r" | split(\1)"),
    (r"\.join\((.*?)\)", r" | join(\1)"),
    (r"\.lstrip\(\)", " | lstrip"),
    (r"\.lstrip\((.*?)\)", r" | lstrip(\1)"),
    (r"\.rstrip\(\)", " | rstrip"),
    (r"\.rstrip\((.*?)\)", r" | rstrip(\1)"),
    (r"\.strip\(\)", " | trim"),
    (r"\.strip\((.*?)\)", r" | trim(\1)"),
    (r"\.items\(\)", " | items"),
    (r"{% generation %}", ""),
    (r"{% endgeneration %}", ""),
)


def edit_for_minijinja(source):
    for pattern, replacement in MINIJINJA_EDITS:
        source = _re.sub(pattern, replacement, source)
    return source


# MiniJinja looks a name up on a map as its key and finds nothing on any other
# plain value (`Value::get_attr`), and a method it does not know is an error
# (`Value::call_method`, `UnknownMethod`): the runtime sets no callback for
# them, so a string has no `.upper()` there. Jinja2 would find Python's.
_PLAIN = (str, list, tuple, dict, int, float, bool, type(None))


class RuntimeEnvironment(ImmutableSandboxedEnvironment):
    def getattr(self, obj, attribute):
        if isinstance(obj, dict):
            return obj[attribute] if attribute in obj else self.undefined(obj=obj, name=attribute)
        if isinstance(obj, _PLAIN):
            return self.undefined(obj=obj, name=attribute)
        return super().getattr(obj, attribute)

    def getitem(self, obj, argument):
        if isinstance(obj, dict):
            return obj[argument] if argument in obj else self.undefined(obj=obj, name=argument)
        if isinstance(obj, _PLAIN) and not isinstance(argument, (int, slice)):
            return self.undefined(obj=obj, name=argument)
        return super().getitem(obj, argument)


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def minijinja_str(value):
    """A value as MiniJinja prints it (`impl Display for Value`) and as its string
    arguments take it (`ArgType for Cow<str>`): `true`, `none`, nothing for an
    undefined value, and a list or map in Rust's debug form."""
    if isinstance(value, jinja2.Undefined):
        return ""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_minijinja_debug(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = (f"{_minijinja_debug(k)}: {_minijinja_debug(v)}" for k, v in value.items())
        return "{" + ", ".join(pairs) + "}"
    return str(value)


def _minijinja_debug(value):
    if isinstance(value, str):
        return _json.dumps(value, ensure_ascii=False)
    return minijinja_str(value)


def _strings(test):
    """MiniJinja's `startingwith` and `endingwith`, whose arguments are strings
    and take any other value as it prints."""

    def check(value, other):
        return getattr(minijinja_str(value), test)(minijinja_str(other))

    return check


def runtime_environment():
    # No `loopcontrols` extension: LiteRT-LM builds MiniJinja without the
    # `loop_controls` feature (`Cargo.toml`; Bazel's `crate_features`), so a
    # `{% break %}` is a syntax error on the device.
    env = RuntimeEnvironment(
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        finalize=minijinja_str,
    )
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = lambda fmt: _time.strftime(fmt, _time.gmtime())
    # serde_json with a formatter that spaces after `,` and `:`, writing UTF-8.
    env.filters["tojson"] = lambda value: _json.dumps(
        value, ensure_ascii=False, separators=(", ", ": ")
    )
    env.filters["lstrip"] = lambda s, chars=None: minijinja_str(s).lstrip(chars)
    env.filters["rstrip"] = lambda s, chars=None: minijinja_str(s).rstrip(chars)
    # MiniJinja's own, where Jinja2 has none or reads them otherwise.
    env.filters["string"] = minijinja_str
    env.filters["split"] = lambda s, sep=None, maxsplits=None: (
        minijinja_str(s).split(sep)
        if maxsplits is None
        else minijinja_str(s).split(sep, maxsplits)
    )
    env.tests["startingwith"] = _strings("startswith")
    env.tests["endingwith"] = _strings("endswith")
    env.tests["sequence"] = lambda value: isinstance(value, (list, tuple))
    env.tests["number"] = lambda value: (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    # `create_env` replaces MiniJinja's `none` with "is undefined".
    env.tests["none"] = lambda value: isinstance(value, jinja2.Undefined)
    return env


def runtime_render(template, messages, add_generation_prompt, **context):
    """{"text": ...}, or {"error": ...} for a template that raised."""
    values = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "",
        "eos_token": "",
        "now": int(_time.time()),
    }
    values.update(context)
    try:
        source = edit_for_minijinja(template)
        text = runtime_environment().from_string(source).render(**values)
    except Exception as exc:  # the template's own failure is a finding, not the script's
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"text": text}
'''


def _parts(text: str, form: str = "one_part") -> list[dict[str, Any]]:
    if form == "two_parts" and len(text) > 1:
        middle = len(text) // 2
        return [{"type": "text", "text": text[:middle]}, {"type": "text", "text": text[middle:]}]
    if form not in FORMS:
        raise ValueError(f"{form!r} is not one of {FORMS}")
    return [{"type": "text", "text": text}]


@dataclass(frozen=True)
class SingleTurn:
    """One single-turn prompt: a user message, after a system message or not."""

    text: str
    system: str | None = None

    @property
    def name(self) -> str:
        where = "after a system message" if self.system else "alone"
        return f"the user message {self.text!r} {where}"

    def as_strings(self) -> list[dict[str, Any]]:
        """As training renders it: each content a string."""
        system = [{"role": "system", "content": self.system}] if self.system else []
        return [*system, {"role": "user", "content": self.text}]

    def as_parts(self, form: str) -> list[dict[str, Any]]:
        """As LiteRT-LM v0.18.0 hands it to the bundle's template."""
        system = [{"role": "system", "content": _parts(self.system, form)}] if self.system else []
        return [*system, {"role": "user", "content": _parts(self.text, form)}]


SINGLE_TURNS: tuple[SingleTurn, ...] = tuple(
    SingleTurn(text, system) for text in SINGLE_TURN_TEXTS for system in (None, SYSTEM_TEXT)
)


@dataclass(frozen=True)
class Reply:
    """An assistant reply as LiteRT-LM keeps it in a conversation's history."""

    name: str
    message: dict[str, Any]


_REASONING = "\nThe customer lost a card.\n"
_ANSWER = "card_lost"

REPLIES: tuple[Reply, ...] = (
    Reply("a plain reply", {"role": "assistant", "content": _parts(_ANSWER)}),
    Reply(
        "a reply carrying a think block",
        {"role": "assistant", "content": _parts(f"<think>{_REASONING}</think>\n\n{_ANSWER}")},
    ),
    # The same reply from a bundle with a `thought` channel: what
    # `ExtractChannelContent` leaves of the text, and what it took out.
    Reply(
        "a reply whose think block the runtime moved to its thought channel",
        {
            "role": "assistant",
            "content": _parts(f"\n\n{_ANSWER}"),
            "reasoning_content": _REASONING,
            "channels": {"thought": _REASONING},
        },
    ),
)

_USER_TEXTS = ("I lost my card", "And my PIN?", "Thanks.")


@dataclass(frozen=True)
class Conversation:
    """A history ending in an assistant reply, and the user message that follows it.

    `turns` counts user messages once the following one is added: two is one
    exchange and a second message, three is two exchanges and a third.
    """

    reply: Reply
    turns: int
    system: str | None = None

    @property
    def name(self) -> str:
        where = "after a system message" if self.system else "with no system message"
        return f"{self.reply.name}, {self.turns} turns, {where}"

    def history(self) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": _parts(self.system)}] if self.system else []
        for text in _USER_TEXTS[: self.turns - 1]:
            messages += [{"role": "user", "content": _parts(text)}, dict(self.reply.message)]
        return messages

    def following(self) -> dict[str, Any]:
        return {"role": "user", "content": _parts(_USER_TEXTS[self.turns - 1])}


CONVERSATIONS: tuple[Conversation, ...] = tuple(
    Conversation(reply, turns, system)
    for system in (None, SYSTEM_TEXT)
    for turns in (2, 3)
    for reply in REPLIES
)


def single_turn_spec() -> list[dict[str, Any]]:
    """The single-turn cases as a script in the export environment reads them."""
    return [
        {"strings": case.as_strings(), **{form: case.as_parts(form) for form in FORMS}}
        for case in SINGLE_TURNS
    ]


def conversation_spec() -> list[dict[str, Any]]:
    """The conversations as a script in the export environment reads them."""
    return [{"history": case.history(), "following": case.following()} for case in CONVERSATIONS]


# ---------------------------------------------------------------------------
# Reading what a script rendered
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rendering:
    """One rendering as a script reported it: the text, or the error it raised."""

    text: str | None = None
    error: str | None = None
    # An error of the environment's (a Jinja2 transformers will not use), not
    # of the template's.
    environment: bool = False

    @classmethod
    def read(cls, entry: object) -> Rendering:
        """Raises ValueError for anything but `{"text": str}` or `{"error": str, ...}`."""
        if isinstance(entry, dict) and isinstance(entry.get("text"), str):
            return cls(text=entry["text"])
        if isinstance(entry, dict) and isinstance(entry.get("error"), str):
            return cls(error=entry["error"], environment=entry.get("environment") is True)
        raise ValueError(f"a rendering this litetune cannot read: {entry!r}")

    def did(self, limit: int = 160) -> str:
        """What the template did, as a predicate: "printed '...'" or "raised ..."."""
        if self.text is None:
            return f"raised {self.error}"
        return f"printed {quoted(self.text, limit)}"

    def same_as(self, other: Rendering) -> bool:
        """Equal texts, or both raised: neither renders something the other does not."""
        if self.text is None or other.text is None:
            return self.text is None and other.text is None
        return self.text == other.text


def quoted(text: str, limit: int = 160) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")


def _readings(entries: object, count: int, what: str) -> list[object]:
    if not isinstance(entries, list) or len(entries) != count:
        raise ValueError(f"{what} is not {count} renderings: {entries!r}"[:400])
    return entries


# ---------------------------------------------------------------------------
# Single-turn identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Difference:
    """One single-turn case a candidate template renders otherwise than the checkpoint's."""

    case: SingleTurn
    form: str
    reference: Rendering
    candidate: Rendering

    def describe(self) -> str:
        how = "in one part" if self.form == "one_part" else "in two parts"
        return (
            f"{self.case.name}, the text {how}: the checkpoint's template "
            f"{self.reference.did()} and the bundle's {self.candidate.did()}"
        )


def references(entries: object) -> list[Rendering]:
    """The checkpoint's own renderings, one per `SINGLE_TURNS` case."""
    return [Rendering.read(e) for e in _readings(entries, len(SINGLE_TURNS), "the reference")]


def single_turn_differences(reference: object, candidate: object) -> list[Difference]:
    """Every case and form where `candidate` does not render what `reference` does.

    `reference` is the script's list of the checkpoint's renderings, one per
    `SINGLE_TURNS` case; `candidate` a list of `{form: rendering}` for the
    template under test. Raises ValueError for a shape this cannot read.
    """
    expected = references(reference)
    rendered = _readings(candidate, len(SINGLE_TURNS), "the candidate")
    differences = []
    for case, wanted, entry in zip(SINGLE_TURNS, expected, rendered, strict=True):
        if not isinstance(entry, dict):
            raise ValueError(f"the candidate's rendering of {case.name} is {entry!r}")
        for form in FORMS:
            got = Rendering.read(entry.get(form))
            if not got.same_as(wanted):
                differences.append(Difference(case, form, wanted, got))
    return differences


# ---------------------------------------------------------------------------
# The conversation prefix invariant
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Break:
    """One conversation whose rendering does not grow by appending."""

    case: Conversation
    previous: Rendering
    following: Rendering

    def describe(self) -> str:
        for side, rendering in (("ending in the reply", self.previous), ("", self.following)):
            if rendering.text is None:
                where = f"the history {side}" if side else "the history and the next message"
                return f"{self.case.name}: the template raises on {where} ({rendering.error})"
        previous, following = self.previous.text or "", self.following.text or ""
        at = next(
            (i for i, (a, b) in enumerate(zip(previous, following, strict=False)) if a != b),
            min(len(previous), len(following)),
        )
        start = max(0, at - 40)
        return (
            f"{self.case.name}: from character {at} the history ending in the reply renders "
            f"{quoted(previous[start : at + 40])} and with the next message "
            f"{quoted(following[start : at + 40])}"
        )


def conversation_breaks(entries: object) -> list[Break]:
    """Every `CONVERSATIONS` case whose two renderings break the prefix invariant.

    `entries` is the script's list of `{"previous": rendering, "next": rendering}`,
    one per case. A rendering that raised is a break: the runtime reports it as
    INTERNAL, "Failed to apply template" (`PromptTemplate::Apply`). Raises
    ValueError for a shape this cannot read.
    """
    breaks = []
    for case, entry in zip(
        CONVERSATIONS,
        _readings(entries, len(CONVERSATIONS), "the conversations"),
        strict=True,
    ):
        if not isinstance(entry, dict):
            raise ValueError(f"the rendering of {case.name} is {entry!r}")
        previous = Rendering.read(entry.get("previous"))
        following = Rendering.read(entry.get("next"))
        if previous.text is None or following.text is None:
            breaks.append(Break(case, previous, following))
        elif not following.text.startswith(previous.text):
            breaks.append(Break(case, previous, following))
    return breaks


@dataclass(frozen=True)
class ConversationCheck:
    """Whether one bundled template holds a LiteRT-LM conversation.

    `holds` is None when it was not checked, and `detail` says why.
    """

    holds: bool | None
    detail: str
    template_sha256: str | None = None
    breaks: tuple[str, ...] = ()

    @classmethod
    def of(cls, template: str, breaks: Sequence[Break], renderer: str) -> ConversationCheck:
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        cases = len(CONVERSATIONS)
        if breaks:
            return cls(
                holds=False,
                detail=(
                    f"{len(breaks)} of the {cases} conversations break it, rendered by {renderer}"
                ),
                template_sha256=digest,
                breaks=tuple(b.describe() for b in breaks),
            )
        return cls(
            holds=True,
            detail=(
                f"each of the {cases} conversations -- a plain reply, one carrying a think block "
                "and one whose think block a thought channel took, over two and three turns, "
                f"with and without a system message -- renders as a prefix of the next, by "
                f"{renderer}"
            ),
            template_sha256=digest,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "detail": self.detail,
            "template_sha256": self.template_sha256,
            "breaks": list(self.breaks),
        }


def conversation_limitation(recipes: Sequence[str], check: ConversationCheck) -> str:
    """What a report says about bundles whose template breaks the invariant."""
    named = ", ".join(recipes)
    first = check.breaks[0] if check.breaks else check.detail
    return (
        f"{named}: {LIMITATION_PREFIX}; a second message in a LiteRT-LM conversation fails, "
        "with INTERNAL 'The new rendered template string does not start with the previous "
        "rendered template string' or, where the template raises, 'Failed to apply template', "
        "because its chat template does not render a conversation as the one before it with "
        "the next message appended. LiteRT-LM renders the whole history again on every message "
        "after the first and requires the new rendering to begin with the previous one "
        "(Conversation::GetSingleTurnTextFromFullHistory, runtime/conversation/conversation.cc, "
        f"v0.18.0; google-ai-edge/LiteRT-LM#3443). {check.detail}; the first: {first}"
    )


def unchecked_limitation(recipes: Sequence[str], reason: str) -> str:
    """What a report says when whether the bundles hold a conversation is unknown."""
    return (
        f"{', '.join(recipes)}: whether this bundle holds a LiteRT-LM conversation past its first "
        f"message was not checked: {reason}"
    )
