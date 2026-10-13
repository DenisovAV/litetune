"""MiniJinja as LiteRT-LM v0.18.0 renders with it, for the tests.

MiniJinja 2.14.0 (`Cargo.toml`, `=2.14.0`, the version pinned in this
project's dev extras), with the settings, functions and filters `create_env`
gives it in `runtime/components/rust/minijinja_template.rs`, after the rewrite
`EditTemplateForMinijinja` (`runtime/components/prompt_template.cc`) makes.
Shared by `test_models.py`, which renders the text-parts block with it, and
`test_template_checks.py`, which holds `template_checks.RENDERER_SOURCE`, the
Jinja2 `convert` renders with, to it.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

# What LiteRT-LM v0.18.0 does to a template before rendering it, in order:
# `EditTemplateForMinijinja` rewrites Python method calls MiniJinja lacks into
# filters and tests, with RE2's non-greedy `.*?`, which Python's `re` reads the
# same way.
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


def minijinja_render(
    source: str, messages: list[dict], add_generation_prompt: bool = True, **context: Any
) -> str:
    """Render as LiteRT-LM v0.18.0 does; `context` is what `extra_context` adds.

    Not its `none` test, which the runtime makes true only for an undefined
    value: the Python binding hands a test `None` for both undefined and null,
    so the difference cannot be reproduced here. The templates rendered with
    this use neither `none` nor `strftime_now`, which `test_models.py` holds."""
    import minijinja

    for pattern, replacement in MINIJINJA_EDITS:
        source = re.sub(pattern, replacement, source)

    def raise_exception(message: str) -> str:
        raise RuntimeError(message)

    # `pycompat=False`: the Python binding answers `.startswith()` and friends
    # by default, the runtime's `Environment::new()` does not -- which is why
    # it rewrites them first, and why the rewrite above is under test here.
    env = minijinja.Environment(
        keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True, pycompat=False
    )
    env.add_function("raise_exception", raise_exception)
    env.add_function("strftime_now", lambda fmt: time.strftime(fmt))
    # serde_json with a formatter that spaces after `,` and `:`, writing UTF-8.
    env.add_filter(
        "tojson", lambda value: json.dumps(value, separators=(", ", ": "), ensure_ascii=False)
    )
    env.add_filter("lstrip", lambda s, chars=None: s.lstrip(chars))
    env.add_filter("rstrip", lambda s, chars=None: s.rstrip(chars))
    return env.render_str(
        source,
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token="",
        eos_token="",
        **context,
    )
