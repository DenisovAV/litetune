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

import functools
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
    so the difference cannot be reproduced here. Nor its build: the binding is
    compiled with `loop_controls`, which LiteRT-LM's is not. The templates
    rendered with this use neither, which `test_models.py` holds for the ones
    it renders."""
    return _environment(source).render_template(
        "template",
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        bos_token="",
        eos_token="",
        **context,
    )


@functools.lru_cache(maxsize=32)
def _environment(source: str) -> Any:
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
    # The binding adds `minijinja-contrib`'s `cycler` and `joiner`; LiteRT-LM
    # depends on `minijinja` alone (`Cargo.toml`) and builds its environment
    # with `Environment::new()`, whose globals are `range`, `dict`, `debug` and
    # `namespace` (`get_globals`, `src/defaults.rs`).
    for name in ("cycler", "joiner"):
        env.remove_global(name)
    env.add_function("raise_exception", raise_exception)
    env.add_function("strftime_now", lambda fmt: time.strftime(fmt))
    # serde_json with a formatter that spaces after `,` and `:`, writing UTF-8.
    env.add_filter(
        "tojson", lambda value: json.dumps(value, separators=(", ", ": "), ensure_ascii=False)
    )
    env.add_filter("lstrip", lambda s, chars=None: s.lstrip(chars))
    env.add_filter("rstrip", lambda s, chars=None: s.rstrip(chars))
    env.add_template("template", source)
    return env


def minijinja_render_native(
    source: str, messages: list[dict], add_generation_prompt: bool = True, **context: Any
) -> str:
    """`minijinja_render`, with every value handed over as MiniJinja's own.

    The runtime's values are MiniJinja maps and lists made from JSON
    (`MinijinjaTemplate::apply_impl`), and MiniJinja prints a map as Rust's
    debug form (`impl Display for Value`). The Python binding wraps a Python
    dict instead and prints it as Python does, so a template that prints a
    message or a tool is rendered otherwise here than on the device. Written as
    literals in front of the template, the values are MiniJinja's own.
    """
    values = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "",
        "eos_token": "",
        **context,
    }
    # One statement a line: `trim_blocks` takes each newline, and the template
    # then begins a line, as it does on the device, for `lstrip_blocks`.
    preamble = "".join(
        f"{{% set {name} = {_literal(value)} %}}\n" for name, value in values.items()
    )
    return minijinja_render(preamble + source, [], add_generation_prompt)


def _literal(value: Any) -> str:
    """`value` as a MiniJinja expression that evaluates to it."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = (f"{_literal(key)}: {_literal(item)}" for key, item in value.items())
        return "{" + ", ".join(pairs) + "}"
    raise TypeError(f"no MiniJinja literal for {type(value).__name__}")
