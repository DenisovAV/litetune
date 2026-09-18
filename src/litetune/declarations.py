"""The tool declarations a run is made against, read once and identified by digest.

Declarations decide what a tool-calling model is asked and what it may answer
with, so they belong to the stages that build and measure a model, not only to
the one that packages it. `bundle` has taken them since it existed; `prepare`,
`tune` and `verify` take the same file.

**One reader, because the digest has to match.** `tune` records a digest beside
the checkpoint, `verify` refuses a set that disagrees with it, and a bundle's
contract carries it. Where the runtime renders the declarations it identifies
the tool list, not the file: the digest of `canonical_text` over the
declarations as read here, so the file a user wrote, the one a
`runtime_rendered` bundle ships and the same list reformatted all hash alike. A
digest over the file's bytes made a bundle refuse the declarations file it had
itself shipped, because the bundle writes them in the order the model learned
and the user's file was in another. Where the application renders them the
file's bytes are the record, because there the order in the file is the prompt
(`recorded_digest`, `digest_matches`).

**The shape is the runtime's, and the rules here are its rules.** One entry is
the OpenAI function-tool object: `{"type": "function", "function": {"name",
"description", "parameters"}}`. That is not a preference. The runtime's Python
API raises `interfaces.Tool description must contain ['function']['name']`
before it forwards anything, and the Hub chat template a checkpoint carries
reads `tool_data['function']['name']` to render its declaration -- so both of
the renderers the measurement compares require it, and a file without it is
refused by the runtime rather than rendered differently.

**Two renderers see this file, and they disagree.** The runtime formats the
declaration in C++ (`fc_tool_format_utils.cc`) by walking an insertion-ordered
JSON, so it prints the key order it is handed; the chat template a FunctionGemma
checkpoint carries pipes every mapping through `dictsort`, so it prints them
in `dictsort`'s order. Measured on 2026-09-17: sorting every mapping here the way
`dictsort` does makes the two agree on ordering, and what survives the sort are
shapes the template drops or invents a field for. Those are refused below, each
against a measured difference rather than against a schema this project
preferred, and the refusal names the property so a caller learns it here instead
of as a token position in the rendering check. `_RENDERABLE` records the
template's own emission rules.

So this enforces the three rules the runtime enforces -- a list, each entry an
object, `function.name` a string -- and then the subset both renderers render
identically. `parameters` is no longer opaque, and the reason it stopped being
opaque is written down: not that a stricter schema is nicer, but that outside
this subset the model would be trained on a prompt the runtime does not send.

A leaf module on purpose: `prompt_mode.py` is the precedent, and the three
stages that need declarations share almost no imports with each other. Anything
here may be imported by any stage without a cycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from litetune.metrics import NAME_RULE, readable_name
from litetune.storage import HASH_ALGORITHM, hash_file


class DeclarationsError(Exception):
    """A declarations file that cannot be used. Names the file and what was wrong."""


def read_declarations(path: Path) -> tuple[list[Any], str]:
    """The parsed declarations and their digest, in the shape `bundle` accepts.

    Raises `DeclarationsError` when the file cannot be read, is not JSON, or
    does not carry what the runtime requires: see the module docstring.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DeclarationsError(
            f"the declarations at {path} could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        parsed = json.loads(text, object_pairs_hook=_refuse_repeated_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DeclarationsError(
            f"the declarations at {path} are not valid JSON ({exc}); a runtime cannot render a "
            "tool list it cannot parse"
        ) from exc
    except _RepeatedKey as exc:
        raise DeclarationsError(
            f"the declarations at {path} give the key {exc.args[0]!r} twice in one object. JSON "
            "leaves which one counts to the reader (RFC 8259, section 4: implementations keep "
            "the last, refuse the object, or keep both), so the application and the model could "
            "be reading different lists"
        ) from exc
    _check(parsed, Path(path))
    ordered = _ordered(parsed)
    _check_renderable(ordered, Path(path))
    try:
        return ordered, content_digest(ordered)
    except UnicodeEncodeError as exc:
        raise DeclarationsError(
            f"the declarations at {path} carry text that is not valid Unicode ({exc.reason} at "
            f"position {exc.start}), a lone surrogate from a \\u escape; no renderer can write it"
        ) from exc


class _RepeatedKey(Exception):
    pass


def _refuse_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _RepeatedKey(key)
        seen[key] = value
    return seen


def recorded_digest(digest: str, path: Path, prerendered: bool) -> str:
    """The digest a stage records for declarations read from `path`.

    The tool list's (`digest`) where the runtime renders them, because there
    litetune hands them over in its own order and the file's order is not the
    model's. The file's bytes where the application renders them, because there
    the order in the file is the convention it renders, and a file with the same
    tools in another order is another prompt.
    """
    return hash_file(Path(path)) if prerendered else digest


def canonical_text(entries: list[Any]) -> str:
    """The one way these declarations are written out: what a bundle ships.

    Two-space JSON, UTF-8 left as UTF-8, a trailing newline, in the order
    `read_declarations` returns them. `content_digest` is the digest of exactly
    this text, so a shipped file's bytes hash to its contract's digest.
    """
    return json.dumps(entries, ensure_ascii=False, indent=2) + "\n"


def content_digest(entries: list[Any]) -> str:
    """The digest of a tool list, in the format `storage.hash_file` writes."""
    body = hashlib.sha256(canonical_text(entries).encode("utf-8")).hexdigest()
    return f"{HASH_ALGORITHM}:{body}"


def digest_matches(recorded: str, digest: str | None, path: Path, prerendered: bool) -> bool:
    """Whether a recorded digest names the declarations read from `path`.

    What `recorded_digest` records, for the same mode. In `prerendered` only
    the file's bytes: the order in the file is the convention the application
    renders, so the same tools in another order are another prompt, and a list
    digest would accept them. Otherwise the tool list's (`digest`, which is
    `read_declarations`'s, or `None` where that refused the file) -- or the
    file's bytes, which is what a `Contract` built in code before digests
    identified the tool list could only have carried. A record may carry the
    algorithm prefix or not, in either case; one naming another algorithm is
    not this digest.
    """
    algorithm, _, wanted = recorded.rpartition(":")
    if algorithm and algorithm.lower() != HASH_ALGORITHM:
        return False
    known = [hash_file(Path(path))]
    if digest is not None and not prerendered:
        known.append(digest)
    return wanted.lower() in (d.split(":", 1)[-1] for d in known)


def declared_order(keys: Iterable[str]) -> list[str]:
    """`keys` in the order the reference template's `dictsort` prints them.

    One key for every place that has to agree with that template: the
    declarations handed to both renderers, and the argument order
    `prepare.render_call` trains, which the runtime's grammar holds to the
    declared order. `dictsort` is case-insensitive by default, so a plain
    `sorted` disagrees with it on `URL` beside `body`: rendered with jinja2
    3.1.6, `dictsort` puts `body` first, `sorted` puts `URL` first, and the
    runtime -- which prints the order it is handed -- would have been handed the
    other one. Names equal but for case are refused in `_check_properties`,
    because no sort orders them.
    """
    return sorted(keys, key=str.lower)


def _ordered(value: Any) -> Any:
    """The same declarations with every mapping in one key order.

    The runtime's formatter prints the order it is given and the reference
    template sorts; handing both a sorted file removes the difference. Done here
    rather than at each renderer because three stages and three subprocess
    scripts read this file, and a path that skipped the sort would surface as a
    differing token id rather than as a missing call.
    """
    if isinstance(value, dict):
        return {key: _ordered(value[key]) for key in declared_order(value)}
    if isinstance(value, list):
        return [_ordered(item) for item in value]
    return value


def _check(parsed: Any, path: Path) -> None:
    """The runtime's own three rules, refused here instead of at conversation time."""
    if not isinstance(parsed, list):
        raise DeclarationsError(
            f"the declarations at {path} are a {type(parsed).__name__}, and the runtime takes a "
            "list of tools"
        )
    for index, entry in enumerate(parsed):
        where = f"{path}: tool {index}"
        if not isinstance(entry, dict):
            raise DeclarationsError(f"{where} is a {type(entry).__name__}, not an object")
        function = entry.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise DeclarationsError(
                f"{where} has no ['function']['name'] string. A tool is the OpenAI function "
                'object: {"type": "function", "function": {"name": ..., "parameters": ...}}. '
                "The runtime refuses anything else before it renders a declaration, and the "
                "reference chat template reads the same two keys"
            )
        _refuse_unreadable_name(function["name"], f"{where} is named {function['name']!r}")
    names = [entry["function"]["name"] for entry in parsed]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise DeclarationsError(
            f"{path} declares {repeated} more than once, so a call to that name cannot say which "
            "declaration it answers. Where the runtime renders the list, one that keys tools by "
            "name keeps one of them -- LiteRT-LM's Kotlin API does -- and renders a different "
            "list from the one the model was trained on"
        )


# The seven names the runtime's formatter uppercases. It leaves anything else
# alone while the reference template uppercases whatever it is given. So a
# lowercase name agrees -- the formatter uppercases it -- and so does the same
# name already in capitals, which neither side changes; any other spelling,
# `String` among them, renders two ways. Measured 2026-09-17, the capitals on
# google/mobile-actions' own declarations, which write `OBJECT` and `STRING`. The
# first version of this refused capitals too, and with them every tool in that
# dataset that takes an argument.
_TYPES = ("array", "boolean", "integer", "null", "number", "object", "string")

# Said wherever a refusal tells the caller to remove something. This file is the
# declarations the application sends, and a key dropped here and left in the
# application reaches the runtime as it was.
_ALSO_IN_THE_APP = (
    "drop the key here and in the declarations your application sends, which should be this file"
)


def _type_name(kind: Any) -> str | None:
    """The lowercase type name, or `None` for a spelling the two render differently."""
    if not isinstance(kind, str):
        return None
    if kind in _TYPES or (kind.isupper() and kind.lower() in _TYPES):
        return kind.lower()
    return None


# The five words the reference template treats as structural. A property named
# one of them is skipped by its `standard_keys` guard and vanishes from the
# prompt, while the runtime prints it like any other.
_RESERVED = ("description", "nullable", "properties", "required", "type")

# What the reference template emits for a property, by that property's type. It
# emits `description`, then these, then `type`, and nothing else in any branch.
_RENDERABLE = {"string": ("enum",), "object": ("properties", "required"), "array": ("items",)}

_WHY_TYPE = (
    f"The runtime's formatter uppercases only the lowercase names {', '.join(_TYPES)} and "
    "leaves anything else as written, while the reference template uppercases whatever it is "
    "given -- so a lowercase name, or the same name in capitals, renders one way and any other "
    "spelling renders two"
)

_WHY_ORDER = (
    "the runtime's formatter prints every key it is given and the reference chat template "
    "prints only the keys it knows, so the two prompts would differ"
)


def _check_renderable(parsed: Any, path: Path) -> None:
    """Refuse what the runtime and the reference template render differently.

    Every rule is one measured difference, not a preference: see the module
    docstring. Runs after `_ordered`, so a mapping here is already in the order
    both renderers will be given.
    """
    for entry in parsed:
        function = entry["function"]
        where = f"{path}: tool {function['name']!r}"
        _refuse_extra_keys(set(function), {"name", "description", "parameters"}, where)
        if not isinstance(function.get("description"), str):
            raise DeclarationsError(
                f"{where} has no description string. The reference template renders a "
                "description for every tool whether or not there is one, so it would write an "
                "empty <escape><escape> where the runtime writes nothing"
            )
        if "parameters" in function:
            _check_parameters(function["parameters"], f"{where}: parameters")


def _check_parameters(parameters: Any, where: str) -> None:
    if not isinstance(parameters, dict) or not parameters:
        raise DeclarationsError(
            f"{where} is empty. The reference template omits an empty parameters object and the "
            f"runtime prints `parameters:{{}}`; {_ALSO_IN_THE_APP}"
        )
    _refuse_extra_keys(set(parameters), {"type", "properties", "required"}, where)
    for key in ("type", "properties", "required"):
        if key in parameters and not parameters[key]:
            raise DeclarationsError(
                f"{where} has an empty {key}. The reference template omits it and the runtime "
                f"prints it empty; {_ALSO_IN_THE_APP}"
            )
    if "type" in parameters and _type_name(parameters["type"]) is None:
        raise DeclarationsError(f"{where} has type {parameters['type']!r}. {_WHY_TYPE}")
    if "properties" in parameters:
        if not isinstance(parameters["properties"], dict):
            raise DeclarationsError(f"{where}: properties is not an object")
        # These names are the call's argument keys, so they have to be names
        # the runtime's call parser reads. Nested properties never reach a
        # call: `prepare` refuses an object argument.
        for name in parameters["properties"]:
            _refuse_unreadable_name(name, f"{where}.{name}")
        _check_properties(parameters["properties"], where)


def _refuse_unreadable_name(name: str, where: str) -> None:
    """A name a call must carry has to be one the runtime's call parser reads."""
    if not readable_name(name):
        raise DeclarationsError(
            f"{where}, which the runtime's call parser does not read as a name: its lexer "
            f"takes {NAME_RULE} (AntlrFcLexer.g4, LiteRT-LM v0.16.1), so no call can come back "
            "under that name. Rename it here and in the declarations your application sends"
        )


def _check_properties(properties: dict[str, Any], where: str) -> None:
    folded: dict[str, str] = {}
    for name in properties:
        if name.lower() in folded:
            raise DeclarationsError(
                f"{where} has properties {folded[name.lower()]!r} and {name!r}, equal but for "
                "case. The reference template's `dictsort` ignores case, so no sort puts them in "
                "the same order for both renderers, and the runtime's grammar holds a call to "
                "that order. Rename one here and in the declarations your application sends"
            )
        folded[name.lower()] = name
    for name, prop in properties.items():
        here = f"{where}.{name}"
        if name in _RESERVED:
            raise DeclarationsError(
                f"{here} is named after one of the words the reference template reserves "
                f"({', '.join(_RESERVED)}). That template skips such a property entirely, so the "
                "declaration it renders would not offer it while the runtime's does"
            )
        if not isinstance(prop, dict):
            raise DeclarationsError(f"{here} is a {type(prop).__name__}, not an object")
        if not isinstance(prop.get("description"), str):
            raise DeclarationsError(
                f"{here} has no description string. The reference template renders one for every "
                "property whether or not there is one, so it would write an empty "
                "<escape><escape> where the runtime writes nothing"
            )
        written = prop.get("type")
        kind = _type_name(written)
        if kind is None:
            raise DeclarationsError(f"{here} has type {written!r}. {_WHY_TYPE}")
        _refuse_extra_keys(
            set(prop), {"description", "type", *_RENDERABLE.get(kind, ())}, here, kind=written
        )
        if kind == "object":
            # Not optional the way the others are: the template renders
            # `properties:{...}` for every object property, and with none to
            # render it writes an empty one where the runtime writes nothing.
            if not isinstance(prop.get("properties"), dict) or not prop["properties"]:
                raise DeclarationsError(
                    f"{here} is an object with no properties. The reference template renders "
                    "`properties:{}` for it and the runtime renders nothing"
                )
            _check_properties(prop["properties"], here)
        for key in ("enum", "required", "items"):
            if key in prop and not prop[key]:
                raise DeclarationsError(
                    f"{here} has an empty {key}. The reference template omits it and the runtime "
                    f"prints it empty; {_ALSO_IN_THE_APP}"
                )
        if kind == "array" and "items" in prop:
            _check_items(prop["items"], f"{here}.items")


def _check_items(items: Any, where: str) -> None:
    if not isinstance(items, dict) or not items:
        raise DeclarationsError(
            f"{where} is not a non-empty object. The reference template renders an array's items "
            "only when they are one, and the runtime renders whatever is there"
        )
    _refuse_extra_keys(set(items), {"type", "description", "properties", "required"}, where)
    if "type" in items and _type_name(items["type"]) is None:
        raise DeclarationsError(
            f"{where} has type {items['type']!r}. Only a lowercase type name or the same name in "
            "capitals renders one way: the reference template uppercases whatever it is given, "
            "including a list of types, and the runtime's formatter uppercases only the lowercase "
            "names"
        )
    if "required" in items and not items["required"]:
        raise DeclarationsError(f"{where} has an empty required; {_ALSO_IN_THE_APP}")
    if "properties" in items:
        if not isinstance(items["properties"], dict) or not items["properties"]:
            raise DeclarationsError(f"{where}: properties is not a non-empty object")
        _check_properties(items["properties"], where)


def _refuse_extra_keys(
    present: set[str], allowed: set[str], where: str, kind: str | None = None
) -> None:
    extra = sorted(present - allowed)
    if not extra:
        return
    of_type = f" of type {kind!r}" if kind else ""
    raise DeclarationsError(
        f"{where}{of_type} carries {extra} beside {sorted(allowed)}. Here {_WHY_ORDER}"
    )


# Which Python values a JSON-typed argument may hold. A bool is an int in
# Python and never a number in JSON Schema, so it is excluded by name.
# An integral float is an integer here: the runtime returns every number as a
# double, and `prepare.render_call` writes an integral float as its integer.
_HOLDS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool))
        or (isinstance(v, float) and v.is_integer())
    ),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def argument_problem(parsed: list[Any], name: str, args: dict[str, Any]) -> str | None:
    """Why a call's arguments contradict the declaration of `name`, or `None`.

    The declaration is what the prompt tells the model it may send; a target
    that sends something else trains the model to break the contract it was
    just shown. Checked: every argument declared, every required one present,
    each value of its declared type, and within its enum. An array or object
    value's own contents are not checked: `prepare` refuses a list or an object
    when it renders a target, and a row with its own completion is its author's.
    """
    function = next(
        (entry["function"] for entry in parsed if entry["function"]["name"] == name), None
    )
    if function is None:
        return f"calls {name!r}, which is not declared"
    parameters = function.get("parameters") or {}
    properties = parameters.get("properties") or {}
    undeclared = sorted(set(args) - set(properties))
    if undeclared:
        return f"sends {undeclared}, which the declaration of {name!r} does not have"
    missing = [key for key in parameters.get("required") or [] if key not in args]
    if missing:
        return f"leaves out {missing}, which the declaration of {name!r} requires"
    for key, value in args.items():
        prop = properties[key]
        kind = _type_name(prop["type"])
        holds = _HOLDS.get(kind or "")
        if holds is not None and not holds(value):
            return (
                f"sends {key}={value!r}, a {type(value).__name__}, where the declaration of "
                f"{name!r} says {prop['type']}"
            )
        if "enum" in prop and value not in prop["enum"]:
            return f"sends {key}={value!r}, which is not in the declared enum {prop['enum']}"
    return None


def tool_names(parsed: Any) -> frozenset[str]:
    """Every tool the declarations offer, by the name a call must use.

    Read after `read_declarations` has accepted the file, so the shape is the
    one checked above; anything it would have refused never reaches here.
    """
    return frozenset(entry["function"]["name"] for entry in parsed)


def entry_count(parsed: Any) -> int | None:
    """How many declarations were read, or `None` when the shape has no answer.

    The same question `bundle` asks when it reports what it packaged, kept here
    so a stage reporting on declarations does not invent a second convention.
    """
    return len(parsed) if isinstance(parsed, list | dict) else None
