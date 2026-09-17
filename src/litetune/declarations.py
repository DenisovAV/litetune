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

So this enforces exactly the three rules the runtime enforces -- a list, each
entry an object, `function.name` a string -- and nothing beyond them.
`parameters` stays opaque: the declaration text is produced by the runtime's own
C++ formatter, and nothing in this project establishes what that formatter
accepts beyond those three. A stricter schema here would refuse files that
render, which is the mistake the first version of this docstring made in the
opposite direction.

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
    # The digest is over the file's bytes, not over `text` or `parsed`: that is
    # what `bundle` compares its contract against.
    return parsed, hash_file(Path(path))


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
