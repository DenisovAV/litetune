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

**The shape is deliberately the loose one.** `bundle` accepts any JSON here and
counts the entries when it is a list or a mapping. What a declaration may
contain is the serving runtime's business, and a stricter schema in litetune
would refuse files the runtime renders happily. This reads the file, insists it
is JSON, and says how many entries it holds when that question has an answer.

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

    Raises `DeclarationsError` when the file cannot be read or is not JSON. It
    does not raise on an unexpected JSON shape: see the module docstring.
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
    # The digest is over the file's bytes, not over `text` or `parsed`: that is
    # what `bundle` compares its contract against.
    return parsed, hash_file(Path(path))


def entry_count(parsed: Any) -> int | None:
    """How many declarations were read, or `None` when the shape has no answer.

    The same question `bundle` asks when it reports what it packaged, kept here
    so a stage reporting on declarations does not invent a second convention.
    """
    return len(parsed) if isinstance(parsed, list | dict) else None
