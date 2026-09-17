"""The tool declarations a run is made against, read once and identified by digest.

Declarations decide what a tool-calling model is asked and what it may answer
with, so they belong to the stages that build and measure a model, not only to
the one that packages it. `bundle` has taken them since it existed; `prepare`,
`tune` and `verify` take the same file.

**One reader, because the digest has to match.** `tune` records a digest beside
the checkpoint and `verify` refuses a set that disagrees with it, and both are
compared against the `declarations_sha256` a bundle's contract carries. A second
way of computing it -- hashing the parsed value, or the text rather than the
bytes -- would produce a different string for the same file and turn that
comparison into a false refusal. So the digest is `storage.hash_file`, the
function `bundle` already compares with, over the same bytes.

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
alphabetically. Measured on 2026-09-17: sorting every mapping here makes the two
agree on ordering, and what survives the sort is four shapes the template drops
or invents a field for. Those are refused below, each against a measured
difference rather than against a schema this project preferred, and the refusal
names the property so a caller learns it here instead of as a token position in
the rendering check. `_RENDERABLE` records the template's own emission rules.

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

import json
from pathlib import Path
from typing import Any

from litetune.storage import hash_file


class DeclarationsError(Exception):
    """A declarations file that cannot be used. Names the file and what was wrong."""


def read_declarations(path: Path) -> tuple[Any, str]:
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
        parsed = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DeclarationsError(
            f"the declarations at {path} are not valid JSON ({exc}); a runtime cannot render a "
            "tool list it cannot parse"
        ) from exc
    _check(parsed, Path(path))
    ordered = _ordered(parsed)
    _check_renderable(ordered, Path(path))
    # The digest is over the file's bytes, not over `text` or `parsed`: sorting
    # changes what is rendered, never what is recorded, and the digest is what
    # `bundle` compares its contract against.
    return ordered, hash_file(Path(path))


def _ordered(value: Any) -> Any:
    """The same declarations with every mapping in one key order.

    The runtime's formatter prints the order it is given and the reference
    template sorts; handing both a sorted file removes the difference. Done here
    rather than at each renderer because three stages and three subprocess
    scripts read this file, and a path that skipped the sort would surface as a
    differing token id rather than as a missing call.
    """
    if isinstance(value, dict):
        return {key: _ordered(value[key]) for key in sorted(value)}
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


# The seven names the runtime's formatter uppercases. It leaves anything else
# alone while the reference template uppercases whatever it is given, so a type
# outside this set -- or one already capitalised -- renders two different ways.
_TYPES = ("array", "boolean", "integer", "null", "number", "object", "string")

# The five words the reference template treats as structural. A property named
# one of them is skipped by its `standard_keys` guard and vanishes from the
# prompt, while the runtime prints it like any other.
_RESERVED = ("description", "nullable", "properties", "required", "type")

# What the reference template emits for a property, by that property's type. It
# emits `description`, then these, then `type`, and nothing else in any branch.
_RENDERABLE = {"string": ("enum",), "object": ("properties", "required"), "array": ("items",)}

_WHY_ORDER = (
    "the runtime's formatter prints every key it is given and the reference chat template "
    "prints only the keys it knows, so the two prompts would differ"
)


def _check_renderable(parsed: Any, path: Path) -> None:
    """Refuse what the runtime and the reference template render differently.

    Every rule is one measured difference, not a preference: see the module
    docstring and design decision D13. Runs after `_ordered`, so a mapping here
    is already in the order both renderers will be given.
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
            "runtime prints `parameters:{}`; drop the key instead"
        )
    _refuse_extra_keys(set(parameters), {"type", "properties", "required"}, where)
    for key in ("type", "properties", "required"):
        if key in parameters and not parameters[key]:
            raise DeclarationsError(
                f"{where} has an empty {key}. The reference template omits it and the runtime "
                "prints it empty; drop the key instead"
            )
    if "properties" in parameters:
        if not isinstance(parameters["properties"], dict):
            raise DeclarationsError(f"{where}: properties is not an object")
        _check_properties(parameters["properties"], where)


def _check_properties(properties: dict[str, Any], where: str) -> None:
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
        kind = prop.get("type")
        if kind not in _TYPES:
            raise DeclarationsError(
                f"{here} has type {kind!r}. The runtime's formatter uppercases only "
                f"{', '.join(_TYPES)} and leaves anything else as written, while the reference "
                "template uppercases whatever it is given"
            )
        _refuse_extra_keys(
            set(prop), {"description", "type", *_RENDERABLE.get(kind, ())}, here, kind=kind
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
                    "prints it empty; drop the key instead"
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
    if "type" in items and items["type"] not in _TYPES:
        raise DeclarationsError(
            f"{where} has type {items['type']!r}. The reference template uppercases a list of "
            "types and escapes it as one; the runtime's formatter leaves a non-string type alone"
        )
    if "required" in items and not items["required"]:
        raise DeclarationsError(f"{where} has an empty required; drop the key instead")
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
