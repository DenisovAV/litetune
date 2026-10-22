"""The decision a model is trained to make by option scoring, and the text it is asked in.

A decision model is an ordinary chat model, trained to answer a question about a
state with the id of one of the listed options. The application never lets it
generate: flutter_edge_ai's `InferenceModel.createDecider` (a proposal, not
shipped) is to render the same prompt and score each option's log-probability
through the engine. So the prompt is a contract between two renderers, this one
and flutter_edge_ai's proposed one, and the golden strings in
`tests/fixtures/decision-contract-v1.json` pin it for both.

**Contract version 1** is the text the option-scoring measurement of
2026-10-15 trained and scored, on four models on three tasks (6 to 151 options). It is the
content of one user message, which the model's own chat template renders:

    <state>

    Question: <question>
    Options:
    - <id>: <description>
    - <id>
    Answer with the id of one option.

An option with a description is `- <id>: <description>`, one without is
`- <id>`, in the order given, and the `none_option` comes last. The completion
is the id alone; `tune` closes the turn with the template's end of turn.
`render` reproduces all 74,489 prompts and answers of that measurement byte for
byte, rendered from the datasets' own text, and the fixture's measured cases
were taken from them. A change to this text is a new contract
version: models trained on the old text keep the old one.

**Three question types**, as Jev answers them. A Choice question picks one of
its options. A Noul question has the options `no` and `yes`, in that order, and
its answer is P(yes). A Score question lists its levels lowest first; a level's
id is its index as a string and the text given is its description, which Jev's
Score answer returns as its legend. Option scoring has measured Choice on
several tasks; how far it has measured Noul and Score is `TYPE_MEASURED`.

**The record.** `canonical_text` is what `prepare` writes to `decision.json`
and what `read_questions` reads back to the same decision, so a later stage
works from the decision the split was rendered with, including one inferred
from the labels.

A leaf module, like `declarations`: any stage may import it without a cycle.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litetune import jsonfile
from litetune.prompt_mode import TURN_MARKERS
from litetune.storage import hash_bytes

CONTRACT_VERSION = 1

QUESTION_TYPES = ("choice", "noul", "score")

# The prompt lists `no` then `yes`, as trained; another order is not measured.
# The answer is P(yes), read by id.
NOUL_IDS = ("no", "yes")

# How far option scoring has trained and measured each type but `choice`,
# which it has on several tasks: irony and SST-2 as `noul`, SST-5 as `score`
# (MEASUREMENTS.md, "Decisions"). `prepare` and `verify` carry it as a
# limitation.
TYPE_MEASURED = {
    "noul": "on two",
    "score": "on one only",
}

ANSWER_LINE = "Answer with the id of one option."

# The question a split is rendered with when no `--questions` names one. The
# measurement always asked a task-specific question, so this one is not
# measured, and `prepare` says so.
GENERIC_QUESTION = "Which option applies?"

# Text that a template or a tokenizer reads as structure rather than as words:
# the turn markers `prompt_mode` recognises, and each family's own control
# tokens, for every family with a rule in `models.RULES`. A state or an option
# carrying one would close the user turn early in one renderer and not in the
# other, or reach the model as a control token rather than as the text written.
# One list for every family: a split is rendered before the model is fixed
# (`prepare` without --base-model), and a decision moves between models.
#
# This is the fallback. Where a tokenizer is in hand, `prepare` asks it for its
# own special tokens (`prepare.refuse_tokenizer_specials`), which is
# authoritative: a tokenizer has specials no hand list holds. `tune`'s own check,
# `verify` and a `prepare` without --tokenizer have only this list.
#
# Gemma 3 and FunctionGemma:
# `<end_of_turn>`, `<bos>`, `<eos>`, `<pad>`.
# ChatML, which Qwen 2.5, Qwen 3, Qwen 3.5 and MiniCPM5 close turns with:
# `<|im_start|>` (in `TURN_MARKERS`) and `<|im_end|>`, in each of their
# templates under tests/fixtures; `<|endoftext|>` is the Qwen tokenizers'
# `pad_token` (`metrics.TERMINATORS` lists it too).
# Qwen 3, Qwen 3.5 and MiniCPM5 templates write reasoning between `<think>`
# and `</think>` (tests/fixtures/qwen3-0.6b-c1899de.chat_template.jinja,
# qwen3.5-0.8b-2fc0636, minicpm5-1b-87179e5).
# Granite 4.1: `<|start_of_role|>`, `<|end_of_role|>` and `<|end_of_text|>`,
# its BOS and EOS (tests/fixtures/granite-4.1-3b-c065040.chat_template.jinja;
# `models.RULES`, the granite-4.1 rule).
# MiniCPM5: `<s>` and `</s>`, its tokenizer's `bos_token` and `eos_token`
# (openbmb/MiniCPM5-1B's tokenizer_config.json at 87179e5, read 2026-10-20);
# its template prints the `bos_token` first
# (tests/fixtures/minicpm5-1b-87179e5.chat_template.jinja).
# Gemma 4: `<|turn>` (in `TURN_MARKERS`) and `<turn|>`, which closes its turn
# (`models._gemma4`'s stop tokens).
CONTROL_TEXT = tuple(
    dict.fromkeys(
        (
            *TURN_MARKERS,
            "<end_of_turn>",
            "<bos>",
            "<eos>",
            "<pad>",
            "<|im_end|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<|start_of_role|>",
            "<|end_of_role|>",
            "<|end_of_text|>",
            "<s>",
            "</s>",
            "<turn|>",
        )
    )
)

# The keys a decision has: in `--questions`, in `decision.json`, in the checkpoint's
# record, and first in the bundle's section, which adds how the options were scored.
CONTRACT_KEYS = frozenset({"contract_version", "type", "question", "options", "none_option"})
_KEYS = CONTRACT_KEYS


class DecisionError(ValueError):
    """A decision or a row that cannot be rendered. Names the file or row and the field."""


# The characters, besides \\n and \\r, on which `str.splitlines()` splits; a reader
# of the prompt that splits lines this way would see another option.
_LINE_BREAKS = frozenset("\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")


def _id_text(value: object, what: str, where: str) -> str:
    """An option id: one line of text, and no ': ' in it.

    The prompt lists an option as `- <id>: <description>`, so `- a: b` reads the
    same as option `a` described as `b`. Every id the prompt lists is one: the
    options, the `none_option`, and labels a split's question is inferred from.
    """
    option_id = _line_text(value, what, where)
    if ": " in option_id:
        raise DecisionError(
            f"{where}: {what} {option_id!r} contains ': '. The line '- {option_id}' would read "
            "the same as an option with the text before it as its id and the rest as its "
            "description"
        )
    return option_id


def control_text_in(text: str) -> str | None:
    """The first control-token string `text` contains, or `None`."""
    return next((marker for marker in CONTROL_TEXT if marker in text), None)


def _line_text(value: object, what: str, where: str) -> str:
    """A non-empty string on one line, without control text, as written."""
    if not isinstance(value, str):
        raise DecisionError(f"{where}: {what} must be a string, got {json.dumps(value)}")
    if not value.strip():
        raise DecisionError(f"{where}: {what} is empty")
    if "\n" in value or "\r" in value or _LINE_BREAKS.intersection(value):
        raise DecisionError(
            f"{where}: {what} {value!r} spans more than one line. The prompt lists one option "
            "per line, so a line break inside one is read as another option"
        )
    for char in value:
        if "\ud800" <= char <= "\udfff":
            raise DecisionError(
                f"{where}: {what} {value!r} contains the lone surrogate U+{ord(char):04X}, "
                "which is not text and cannot be encoded"
            )
        if ord(char) < 0x20 and char != "\t":
            raise DecisionError(
                f"{where}: {what} {value!r} contains the control character U+{ord(char):04X}. "
                "Only a tab is allowed besides printable text"
            )
    if value != value.strip():
        raise DecisionError(
            f"{where}: {what} {value!r} starts or ends with whitespace. It is written into the "
            "prompt as given, and an application listing it trimmed would send another prompt"
        )
    marker = control_text_in(value)
    if marker is not None:
        raise DecisionError(
            f"{where}: {what} {value!r} contains {marker!r}, which the chat template or the "
            "tokenizer reads as a control token rather than as text"
        )
    return value


@dataclass(frozen=True)
class Option:
    id: str
    description: str | None = None

    def line(self) -> str:
        if self.description is None:
            return f"- {self.id}"
        return f"- {self.id}: {self.description}"

    def as_dict(self) -> dict[str, str]:
        if self.description is None:
            return {"id": self.id}
        return {"id": self.id, "description": self.description}


@dataclass(frozen=True)
class Decision:
    """One question with its options, as a model is trained and asked it.

    `options` are the ones given, in order. `listed` adds the `none_option`
    last: the order the prompt shows. The application passes `options` only, in
    any order; the renderer lists them in this order and adds the `none_option`
    itself.
    """

    type: str
    question: str
    options: tuple[Option, ...]
    none_option: str | None = None

    @property
    def listed(self) -> tuple[Option, ...]:
        if self.none_option is None:
            return self.options
        return (*self.options, Option(self.none_option))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(option.id for option in self.listed)

    def render(self, state: str) -> str:
        """The user message for `state`: contract version 1, byte for byte."""
        lines = "\n".join(option.line() for option in self.listed)
        return f"{state}\n\nQuestion: {self.question}\nOptions:\n{lines}\n{ANSWER_LINE}"

    def answer_for(self, label: str | None, where: str) -> str:
        """The id a row with `label` is trained to answer.

        `None` is a row outside every option, trained as the `none_option`; a
        decision without one refuses it rather than inventing an answer.
        """
        if label is None:
            if self.none_option is None:
                raise DecisionError(
                    f"{where}: the label is null, which marks a row outside every option, and "
                    "the decision declares no none_option to train it as. Add one to "
                    "--questions, or drop the row"
                )
            return self.none_option
        if label not in self.ids:
            raise DecisionError(
                f"{where}: the label {label!r} is not one of the options "
                f"({_listing(self.ids)}). Training it would teach an answer the prompt never "
                "offers"
            )
        return label

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "type": self.type,
            "question": self.question,
            "options": [option.as_dict() for option in self.options],
            "none_option": self.none_option,
        }

    def canonical_text(self) -> str:
        """What `decision.json` holds. `read_questions` reads it back to this decision."""
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n"

    def digest(self) -> str:
        return hash_bytes(self.canonical_text().encode("utf-8"))


def _listing(ids: Iterable[str], limit: int = 12) -> str:
    shown = list(ids)
    head = ", ".join(repr(i) for i in shown[:limit])
    return head + (f" and {len(shown) - limit} more" if len(shown) > limit else "")


def _options(raw: object, kind: str, where: str) -> tuple[Option, ...]:
    if not isinstance(raw, list):
        raise DecisionError(f"{where}: 'options' must be a list, got {json.dumps(raw)}")
    options: list[Option] = []
    for i, entry in enumerate(raw):
        at = f"{where}: options[{i}]"
        if kind == "score" and isinstance(entry, str):
            # A level given as its text alone: the id is its place, lowest first.
            options.append(Option(str(i), _line_text(entry, "the level", at)))
            continue
        if not isinstance(entry, dict):
            raise DecisionError(f"{at} must be an object with an 'id', got {json.dumps(entry)}")
        unknown = sorted(set(entry) - {"id", "description"})
        if unknown:
            raise DecisionError(f"{at} has keys this contract does not read: {unknown}")
        if "id" not in entry:
            raise DecisionError(f"{at} has no 'id'")
        option_id = _id_text(entry["id"], "the id", at)
        description = entry.get("description")
        if description is not None:
            description = _line_text(description, "the description", at)
        options.append(Option(option_id, description))
    return tuple(options)


def decision_from(obj: object, where: str) -> Decision:
    """A decision from the parsed `--questions` object. Raises `DecisionError` naming the field."""
    if not isinstance(obj, dict):
        raise DecisionError(f"{where}: expected an object with a 'question', got {json.dumps(obj)}")
    unknown = sorted(set(obj) - _KEYS)
    if unknown:
        raise DecisionError(
            f"{where}: keys this contract does not read: {unknown}. It reads "
            f"{sorted(_KEYS)}; one question per model"
        )
    version = obj.get("contract_version", CONTRACT_VERSION)
    # `type is int` refuses true, which equals 1, and 1.0.
    if type(version) is not int or version != CONTRACT_VERSION:
        raise DecisionError(
            f"{where}: contract_version {json.dumps(version)} is not one this litetune renders; "
            f"it renders version {CONTRACT_VERSION}"
        )
    kind = obj.get("type", "choice")
    if kind not in QUESTION_TYPES:
        raise DecisionError(
            f"{where}: type {json.dumps(kind)} is not one of {list(QUESTION_TYPES)}"
        )
    if "question" not in obj:
        raise DecisionError(f"{where}: no 'question'")
    question = _line_text(obj["question"], "the question", where)
    none_option = obj.get("none_option")
    if none_option is not None:
        if kind != "choice":
            raise DecisionError(
                f"{where}: a {kind} question has no none_option. Only a Choice answer abstains"
            )
        none_option = _id_text(none_option, "the none_option", where)

    if kind == "noul":
        options = (
            _options(obj["options"], kind, where)
            if "options" in obj
            else tuple(Option(i) for i in NOUL_IDS)
        )
        if tuple(o.id for o in options) != NOUL_IDS:
            raise DecisionError(
                f"{where}: a noul question's options are {list(NOUL_IDS)}, in that order; got "
                f"{[o.id for o in options]}. The prompt lists them in this order, as trained "
                "(another order is not measured), and the answer is P(yes), read by id"
            )
    else:
        if "options" not in obj:
            raise DecisionError(f"{where}: a {kind} question needs 'options'")
        options = _options(obj["options"], kind, where)
        if kind == "score":
            levels = [o.id for o in options]
            if levels != [str(i) for i in range(len(options))]:
                raise DecisionError(
                    f"{where}: a score question's level ids are '0', '1', … lowest first; got "
                    f"{levels}"
                )
            missing = [o.id for o in options if o.description is None]
            if missing:
                raise DecisionError(
                    f"{where}: score levels {missing} have no text. A level's text is the "
                    "legend the answer returns"
                )

    decision = Decision(kind, question, options, none_option)
    ids = decision.ids
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise DecisionError(f"{where}: option ids {repeated} are listed more than once")
    if len(ids) < 2:
        raise DecisionError(f"{where}: a decision needs at least two options, got {list(ids)}")
    return decision


def read_questions(path: Path) -> tuple[Decision, str]:
    """The decision in `path`, and the digest of the file's bytes."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise DecisionError(
            f"the questions at {path} could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        text = jsonfile.decode(raw)
    except UnicodeDecodeError as exc:
        raise DecisionError(f"the questions at {path} {jsonfile.UNREADABLE} ({exc})") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DecisionError(f"the questions at {path} are not valid JSON ({exc})") from exc
    return decision_from(parsed, str(path)), hash_bytes(raw)


def inferred_decision(labels: Iterable[str]) -> Decision:
    """The Choice question a split without `--questions` is rendered with.

    The options are the labels' distinct ids, sorted, without descriptions; the
    question is `GENERIC_QUESTION`. Each id is checked as a `--questions` one
    would be, since it is written into the prompt the same way.
    """
    ids = sorted(set(labels))
    options = tuple(Option(_id_text(i, "the label", "the labels")) for i in ids)
    if len(options) < 2:
        raise DecisionError(
            f"the labels name {len(options)} distinct option(s) ({_listing(ids)}); a decision "
            "needs at least two. Pass --questions to list the options"
        )
    return Decision("choice", GENERIC_QUESTION, options)


def state_problem(state: object) -> str | None:
    """Why `state` cannot be rendered into a decision prompt, or `None`.

    A state may span lines -- a conversation is one line per turn -- but it has
    to say something, and it cannot carry text a template reads as structure.
    """
    if not isinstance(state, str):
        return f"the prompt must be a string, got {json.dumps(state)}"
    if not state.strip():
        return "the prompt is empty: there is no state to decide about"
    marker = control_text_in(state)
    if marker is not None:
        return (
            f"the prompt contains {marker!r}. A decision prompt is rendered by the model's chat "
            "template, as the application renders it, and that text would be read as a control "
            "token rather than as part of the state"
        )
    return None
