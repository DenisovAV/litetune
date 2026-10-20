"""The two properties a bundle's chat template must have, on the templates litetune handles.

Single-turn identity and the conversation prefix invariant (`template_checks`),
checked on Qwen3 0.6B's and Qwen2.5 0.5B Instruct's own templates, the composed
templates litetune builds from them, LiteRT-LM v0.18.0's Qwen 3 and Qwen 2.5
templates and the FunctionGemma template litetune packages; and the renderer
`convert` checks them with, held to MiniJinja 2.14.0 and to LiteRT-LM's own
goldens. Every template comes from a fixture or the package, with its
provenance in `fixtures/README.md` or its header: nothing is fetched.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    GRANITE_SPECIAL_TOKENS,
    MINICPM5_SPECIAL_TOKENS,
    run_compose_script,
)
from minijinja_runtime import MINIJINJA_EDITS, minijinja_render_native

from litetune import export, models, template_checks
from litetune.template_checks import (
    CONTEXTS,
    CONVERSATIONS,
    FORMS,
    LIMITATION_PREFIX,
    PROBE_TEXT,
    SINGLE_TURNS,
    TOOLS,
    Break,
    Context,
    ConversationCheck,
    Refused,
    Rendering,
    conversation_breaks,
    conversation_limitation,
    conversation_spec,
    single_turn_differences,
    unchecked_limitation,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_bytes().decode("utf-8")


def _packaged(name: str) -> str:
    return Path(models.packaged_template(name)).read_bytes().decode("utf-8")


QWEN3 = _fixture("qwen3-0.6b-c1899de.chat_template.jinja")
QWEN25 = _fixture("qwen2.5-0.5b-instruct-7ae5576.chat_template.jinja")
LITERT_QWEN25 = _fixture("litert-lm-v0.18.0-qwen2_5.chat_template.jinja")
BLOCK = _packaged(models.TEXT_PARTS_BLOCK)
LITERT_QWEN3 = models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE)
GRANITE = _fixture("granite-4.1-3b-c065040.chat_template.jinja")
MINICPM5 = _fixture("minicpm5-1b-87179e5.chat_template.jinja")
QWEN35 = _fixture("qwen3.5-0.8b-2fc0636.chat_template.jinja")
LITERT_MINICPM5 = _fixture("litert-lm-v0.18.0-minicpm5.chat_template.jinja")
LITERT_QWEN35 = _fixture("litert-lm-v0.18.0-qwen3_5.chat_template.jinja")
FUNCTIONGEMMA = _packaged("functiongemma.jinja")

# Every template a check here renders for a bundle, by what it is.
BUNDLED = {
    "qwen3 composed": BLOCK + QWEN3,
    "qwen2.5 composed": BLOCK + QWEN25,
    "qwen3 litert-lm": LITERT_QWEN3,
    "qwen2.5 litert-lm": LITERT_QWEN25,
    "functiongemma packaged": FUNCTIONGEMMA,
    # The checkpoints' own, which LiteRT-LM 0.18.0 cannot use alone.
    "qwen3 alone": QWEN3,
    "qwen2.5 alone": QWEN25,
}


@functools.cache
def _script(name: str) -> dict[str, Any]:
    """A script `export` writes for the export environment, loaded here."""
    source = {
        "renderer": template_checks.RENDERER_SOURCE,
        "conversation": export._CONVERSATION_SCRIPT,
    }[name]
    namespace: dict[str, Any] = {"__name__": name}
    exec(compile(source, f"{name}.py", "exec"), namespace)
    return namespace


def _render(template: str, messages: list, gen: bool = True, **context: Any) -> dict:
    return _script("renderer")["runtime_render"](template, messages, gen, **context)


def _unvouched(template: str) -> list[str]:
    return _script("renderer")["unvouched"](template)


def _conversation(template: str) -> dict[str, Any]:
    """What the conversation script reports for `template`."""
    return _script("conversation")["main"]({"template": template, **conversation_spec()})


def _breaks(template: str) -> tuple[list[Break], list[Refused]]:
    """What the conversation script renders for `template`, read as `convert` reads it."""
    return conversation_breaks(_conversation(template)["renders"])


def _minijinja(template: str, messages: list, gen: bool, **context: Any) -> dict:
    """MiniJinja 2.14.0's rendering, the values handed over as its own, as the runtime's are."""
    try:
        return {"text": minijinja_render_native(template, messages, gen, **context)}
    except Exception as error:  # noqa: BLE001 -- MiniJinja's own error type
        return {"error": str(error)}


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


def _every_rendering() -> list[tuple[list, bool, dict]]:
    """Each message list, generation-prompt flag and context the checks render a bundle with."""
    renderings = [
        (case.as_parts(form), True, case.context.values())
        for case in SINGLE_TURNS
        for form in FORMS
    ]
    for case in CONVERSATIONS:
        for context in CONTEXTS:
            values = context.values()
            renderings += [
                (case.first(), True, values),
                (case.history(), False, values),
                ([*case.history(), case.following()], True, values),
            ]
    return renderings


@pytest.mark.parametrize("name", sorted(BUNDLED))
def test_the_renderer_renders_every_case_as_minijinja_does(name):
    """`convert` renders with Jinja2 configured as LiteRT-LM configures MiniJinja;
    on every template and case it checks, in every context, the two agree byte
    for byte, or both raise."""
    template = BUNDLED[name]
    assert _unvouched(template) == []
    for messages, gen, context in _every_rendering():
        ours = _render(template, messages, gen, **context)
        theirs = _minijinja(template, messages, gen, **context)
        assert ("text" in ours) == ("text" in theirs), (messages, context, ours, theirs)
        assert ours.get("text") == theirs.get("text"), (messages, context)


def test_the_renderer_rewrites_as_the_runtime_does():
    """The same `EditTemplateForMinijinja` list as the MiniJinja harness, in order."""
    assert _script("renderer")["MINIJINJA_EDITS"] == MINIJINJA_EDITS


# Each name the renderer vouches for, with what it does to values that are not
# strings -- booleans, none, numbers, lists, maps, undefined -- where Jinja2,
# left to itself, coerces as Python prints and MiniJinja as Rust does. Every
# snippet must render the same as MiniJinja 2.14.0 does, or raise in both.
_ADVERSARIAL_FILTERS = {
    "default": ["[{{ none | default('d') }}|{{ nope | default('d') }}|{{ false | default('d') }}]"],
    "d": ["[{{ nope | d('x') }}|{{ '' | d('x', true) }}]"],
    "dictsort": [
        "{% for k, v in {'b': 1, 'A': 2, 'a': 3} | dictsort %}{{ k }}={{ v }};{% endfor %}",
        "{% for k, v in {'b': 1, 'a': 2} | dictsort(by='value') %}{{ k }};{% endfor %}",
    ],
    "first": ["{{ 'abc' | first }}|{{ [true, 1] | first }}|{{ [] | first }}"],
    "items": [
        "{% for k, v in {'a': true, 'b': none} | items %}{{ k }}={{ v }};{% endfor %}",
        "{% for k, v in [1, 2] | items %}x{% endfor %}",
        "{% for k, v in nope | items %}x{% endfor %}y",
    ],
    "join": [
        "{{ [true, none, 1, 'x', 2.0, [1, 'y'], {'k': false}] | join(',') }}",
        "{{ 'abc' | join('-') }}|{{ {'a': 1, 'b': 2} | join }}|[{{ none | join }}]",
        "[{{ nope | join(',') }}]",
        "{{ 3 | join(',') }}",
        "{{ [1, 2] | join(true) }}",
    ],
    "last": ["{{ 'abc' | last }}|{{ [1, false] | last }}"],
    "length": ["{{ {'a': 1} | length }}|{{ 'abc' | length }}|{{ [none] | length }}"],
    "list": ["{{ 'ab' | list }}|{{ {'x': 1} | list }}|{{ [true] | list }}"],
    "lower": ["{{ 'aB' | lower }}|{{ true | lower }}|{{ none | lower }}"],
    "lstrip": ["[{{ '  a ' | lstrip }}]|{{ 'xxa' | lstrip('x') }}"],
    "map": [
        "{{ [{'n': 'a'}, {'n': true}] | map(attribute='n') | join(',') }}",
        "{{ ['a', true] | map('upper') | join(',') }}",
        "{{ [1, none] | map('string') | join(',') }}",
    ],
    "replace": ["{{ 'aXb' | replace('X', true) }}|{{ true | replace('t', 'T') }}"],
    "rstrip": ["[{{ ' a  ' | rstrip }}]|{{ 'axx' | rstrip('x') }}"],
    "split": [
        "{{ 'a,b,,c' | split(',') | join('|') }}|{{ ' a  b ' | split | join('|') }}",
        "{{ 'a,b,c' | split(',', 1) | last }}|{{ true | split('u') | first }}",
    ],
    "string": ["{{ [1, 'a', true, none] | string }}|{{ {'a': none} | string }}|{{ 1.5 | string }}"],
    "tojson": ["{{ {'a': none, 'b': true, 'c': 1.5, 'd': ['é', 2]} | tojson }}"],
    "trim": ["[{{ '  a ' | trim }}]|{{ 'xxaxx' | trim('x') }}|{{ true | trim }}|{{ none | trim }}"],
    "upper": ["{{ 'aB' | upper }}|{{ true | upper }}|{{ [1] | upper }}"],
}
_ADVERSARIAL_TESTS = {
    "boolean": ["{{ true is boolean }}|{{ 1 is boolean }}|{{ none is boolean }}"],
    "defined": ["{{ nope is defined }}|{{ none is defined }}|{{ messages is defined }}"],
    "undefined": ["{{ nope is undefined }}|{{ none is undefined }}"],
    "endingwith": ["{{ 'abc' is endingwith 'c' }}|{{ true is endingwith 'e' }}"],
    "startingwith": ["{{ 'abc' is startingwith 'a' }}|{{ 12 is startingwith '1' }}"],
    "eq": ["{{ 1 is eq 1 }}|{{ true is eq 1 }}|{{ 'a' is eq 'a' }}"],
    "ne": ["{{ 1 is ne 2 }}|{{ none is ne false }}"],
    "false": ["{{ false is false }}|{{ 0 is false }}|{{ none is false }}"],
    "true": ["{{ true is true }}|{{ 1 is true }}"],
    "in": ["{{ 'a' is in 'abc' }}|{{ 1 is in [1] }}|{{ 'k' is in {'k': 1} }}"],
    "iterable": ["{{ 'a' is iterable }}|{{ 1 is iterable }}|{{ {} is iterable }}"],
    "mapping": ["{{ {} is mapping }}|{{ [] is mapping }}|{{ 'a' is mapping }}"],
    "number": ["{{ 1 is number }}|{{ 1.5 is number }}|{{ true is number }}"],
    "sequence": ["{{ 'a' is sequence }}|{{ [1] is sequence }}|{{ {} is sequence }}"],
    "string": ["{{ 'a' is string }}|{{ 1 is string }}|{{ none is string }}"],
}
_ADVERSARIAL_FUNCTIONS = {
    "dict": ["{{ dict(a=1, b=true) | tojson }}"],
    "namespace": ["{% set ns = namespace(a=1) %}{% set ns.a = ns.a + 1 %}{{ ns.a }}"],
    "range": ["{{ range(3) | join(',') }}|{{ range(1, 6, 2) | list }}"],
    "raise_exception": ["{{ raise_exception('no') }}"],
}
# `~`, printing, and the coercions every template meets, beside any name.
_ADVERSARIAL_SYNTAX = [
    "{{ 'a' ~ true ~ none ~ [1, 'x'] ~ {'k': false} ~ 1.0 ~ 2 ~ nope }}",
    "{% set marker = true ~ '' %}{{ marker }}|{{ marker == 'True' }}",
    "{{ (false ~ '') | upper }}",
    # Constant, so Jinja2's optimizer would fold the comparison as Python joins.
    "{{ (true ~ '') == 'true' }}|{{ (none ~ 'x') == 'nonex' }}",
    "{{ true }}|{{ none }}|{{ [1, 'a', none] }}|{{ {'a': true} }}|{{ 1e20 }}|{{ 1e-7 }}|{{ 0.5 }}",
    "{{ 'a' + 'b' }}|{{ 1 + 2 }}|{{ [1] + [2] }}|{{ 7 // 2 }}|{{ 7 / 2 }}|{{ 7 % 3 }}",
    "{{ 'a' + 1 }}",
    "{{ 'ab' * 2 }}|{{ -1 }}|{{ not none }}|{{ not nope }}",
    "{{ 'y' if 0 else 'n' }}|{{ 'y' if [] else 'n' }}|{{ 'y' if 'x' else 'n' }}",
    "{{ [1, 2, 3][1:] }}|{{ 'abc'[::-1] }}|{{ [1, 2][-1] }}",
    "{% for x in [1, 2, 3] %}{{ loop.index }}{{ loop.revindex }}{{ loop.first }}{% endfor %}",
    "{% macro f(x) %}<{{ x }}>{% endmacro %}{{ f(true) ~ 'z' }}",
    "{{ 'abc'.foo }}|{{ {'a': 1}.a }}|{{ {'a': 1}['b'] }}",
    "{{ nope.x }}",
    "{{ 'a'.upper() }}",
    # MiniJinja's arithmetic (`src/value/ops.rs`): Euclidean `//` and integer
    # `%`, `/` in f64, integers checked in i128, a negative integer exponent
    # refused, a repetition by a non-negative count, `-` for numbers alone.
    "{{ -7 // 2 }}|{{ 7 // -2 }}|{{ 7.5 // 2 }}|{{ -7.5 // 2 }}|{{ 7.5 // -2 }}|{{ -7 // 2.0 }}",
    "{{ -7 % 3 }}|{{ 7 % -3 }}|{{ 7.5 % 2 }}|{{ -7.5 % 2 }}|{{ 7.5 % -2 }}|{{ 1.0 % 0 }}",
    "{{ 1 / 0 }}|{{ -1 / 0 }}|{{ 0 / 0 }}|{{ 1.0 // 0 }}|{{ 10 / 4 }}|{{ 6 / 3 }}",
    "{{ 1 // 0 }}",
    "{{ 1 % 0 }}",
    "{{ 2 ** 3 }}|{{ 2 ** 0.5 }}|{{ 2.0 ** -1 }}|{{ 0 ** 0 }}|{{ -2 ** 2 }}|{{ 2 ** 3 ** 2 }}",
    "{{ 0.0 ** -1 }}|{{ (-8) ** (1 / 3) }}|{{ 1e308 * 10 }}|{{ true + true }}|{{ true * 3 }}",
    "{{ 2 ** -1 }}",
    "{{ 2 ** 127 }}",
    "{{ 1 ** 4294967296 }}",
    "{{ 2 ** 126 * 2 }}",
    "{{ 2 ** 64 }}|{{ 9223372036854775807 + 1 }}|{{ (-2) ** 127 }}|{{ -(-9223372036854775808) }}",
    "{{ 2 * 'ab' }}|{{ 'ab' * 2.0 }}|{{ 'ab' * true }}|{{ [1] * 2 }}|{{ 2 * [1] }}|{{ [1] * 0 }}",
    "{{ 'ab' * -1 }}",
    "{{ [1] * -1 }}",
    "{{ 'ab' * 2.5 }}",
    "{{ 'a' - 1 }}",
    "{{ none + 1 }}",
    "{{ {'a': 1} + {'b': 2} }}",
    "{{ - 1 + 2 }}|{{ -(1 + 2) }}|{{ -0.0 }}",
    "{{ -true }}",
    "{{ -'a' }}",
    "{{ -none }}",
    # Equality (`impl PartialEq for Value`): item by item, a tuple a list.
    "{{ (1, 2) == [1, 2] }}|{{ [(1, 2)] == [[1, 2]] }}|{{ 1 == 1.0 }}|{{ true == 1 }}",
    "{{ 'a' == 1 }}|{{ nope == none }}|{{ nope == nope }}|{{ none != nope }}",
    "{{ ({'b': 1, 'a': 2} | dictsort)[0] == ['a', 2] }}|{{ (1,) is eq [1] }}|{{ 1 is ne 'a' }}",
    # Ordering across kinds (`impl Ord for Value`).
    "{{ 1 < 'a' }}|{{ none < 1 }}|{{ [1] < 'a' }}|{{ true < 2 }}|{{ 'a' < none }}",
    "{{ nope < 1 }}|{{ [1] < [1, 0] }}|{{ {'a': 1} < {'b': 1} }}|{{ [1, 'a'] < [1, 2] }}",
    # Containment (`ops::contains`, and the `in` test).
    "{{ 1 in 'a1' }}|{{ 'a' in nope }}|{{ [1, 2] in [(1, 2)] }}|{{ [1] in {'a': 1} }}",
    "{{ 1 is in 1 }}|{{ 'a' is in none }}|{{ 1 is in [1.0] }}",
    "{{ 'a' in none }}",
    "{{ 1 in 1 }}",
    # Loops (`Value::try_iter`, `vm/loop_object.rs`).
    "{% for x in none %}x{% else %}e{% endfor %}|{% for x in nope %}x{% else %}e{% endfor %}",
    "{% for x in 1 %}x{% endfor %}",
    "{% for x in 'ab' %}{{ loop.length }}|{{ loop.revindex }}|{{ loop.last }};{% endfor %}",
    "{% for x in 'a,b' | split(',') %}{{ loop.length }}|{{ loop.last }};{% endfor %}",
    "{% for x in [1, 2, 3] %}{{ loop.changed(x % 2) }}{{ loop.cycle('a', 'b') }}"
    "[{{ loop.previtem }}|{{ loop.nextitem }}]{% endfor %}",
    "{% for x in [1] %}{{ loop }}|{{ loop.depth }}|[{{ loop.foo }}]{% endfor %}",
    # What MiniJinja holds as an iterable, of a known length or of none.
    "{{ 'a,b' | split(',') }}|{{ ('a,b' | split(','))[0] }}"
    "|{{ 'a,b' | split(',') | list | length }}",
    "{{ 'a,b' | split(',') | length }}",
    "{{ nope | length }}",
    "{% set v = [1, 2, 3][1:] %}{{ v is sequence }}|{{ v | length }}|{{ v < [9] }}|{{ v }}",
    "{{ ([1] + [2]) is sequence }}|{{ ({'a': 1} | items) is sequence }}|{{ range(2) is sequence }}",
    "{{ ([1] * 2) is sequence }}|{{ ([1, 2] | map('string')) is sequence }}|{{ range(3) }}",
    "{% set p = 'a,b,c' | split(',') %}{{ p[1:] }}|{{ p[:2] }}|{{ p[::-1] }}|{{ p[-1] }}"
    "|{{ p + [1] }}",
    "{% set p = 'a,b' | split(',') %}{{ p * 2 }}",
    "{{ range(100001) | length }}",
    # Slices (`ops::slice`).
    "{{ [1, 2, 3][0:0:-1] }}|{{ [1, 2, 3][:0:-1] }}|{{ 'abc'[:0:-1] }}|{{ [1, 2, 3][-10:-20:-1] }}",
    "{{ 'abcdef'[4:1:-2] }}|{{ none[1:] }}|{{ nope[1:] }}|{{ [1, 2, 3][1.0:] }}"
    "|{{ [1, 2, 3][1.0] }}",
    "{{ [1, 2, 3][::0] }}",
    "{{ {'a': 1}[1:] }}",
    "{{ [1, 2, 3][1.5:] }}",
    "{% for m in [1, 2, 3][1:] %}{{ loop.index }}{{ loop.last }}{{ loop.length }};{% endfor %}",
    # A namespace is a map with its keys in order (`value/namespace_object.rs`).
    "{{ namespace(b=[1], a='x') }}|{{ 'y' if namespace() else 'n' }}|{{ namespace(a=1) | tojson }}",
    "{% for k in namespace(b=1, a=2) %}{{ k }}{% endfor %}|{{ 'a' in namespace(a=1) }}"
    "|{{ namespace(a=1) == {'a': 1} }}|{{ namespace(a=1) is mapping }}",
    "{% set ns = namespace(b=0) %}{% set ns.a, ns.b = 1, 2 %}{{ ns }}",
    "{% set ns = namespace(x=[]) %}{% for i in [3, 1] %}{% set ns.x = ns.x + [i] %}{% endfor %}"
    "{{ ns.x }}|{{ ns.x is sequence }}|{{ ns.x[-1] }}",
    "{{ namespace({'a': 1}, b=2) }}",
    "{{ namespace(1) }}",
    # A macro (`vm/macro_object.rs`, `compile_macro_expression`).
    "{% macro f(a=1) %}[{{ a }}]{% endmacro %}{{ f }}|{{ f(nope) }}|{{ f(a=nope) }}|{{ f.name }}"
    "|{{ f.arguments }}|[{{ f.catch_kwargs }}]",
    "{% macro f() %}{{ caller(nope) }}{% endmacro %}{% call(x=2) f() %}[{{ x }}]{% endcall %}",
    "{% macro f() %}{{ caller(1, 2) }}{% endmacro %}{% call(a, b) f() %}{{ a + b }}{% endcall %}",
    "{% macro f(x) %}{{ x }}{% endmacro %}{{ f(1, x=2) }}",
    # Syntax the two read alike.
    "{{ 1_000 }}|{{ 0x10 }}|{{ 0o10 }}|{{ 0b11 }}|{{ 1e3 }}|{{ 'a' 'b' }}|{{ [1, 2,] }}"
    "|{{ (1, 2,) }}",
    "{% filter upper %}ab{% endfilter %}|{% set x | upper %}cd{% endset %}{{ x }}"
    "|{{ not 1 in [1] }}",
    "{% set x = 1 %}{% for i in [1] %}{% set x = 2 %}{% endfor %}{{ x }}|{% with a = 1 %}{{ a }}"
    "{% endwith %}[{{ a }}]",
    "{% for x in [[1, [2]], 3] recursive %}{% if x is sequence %}{{ loop(x) }}{% else %}{{ x }}"
    "{{ loop.depth }}{% endif %}{% endfor %}",
    # A loop's iterable is evaluated in the scope around it: here, the
    # recursive loop's.
    "{% for x in [[[]]] recursive %}{% for y in loop(x) %}{{ y }}{% endfor %}{% endfor %}",
    "{% for x in [['ab']] recursive %}{% if x is string %}{{ x }}{% else %}"
    "{% for y in loop(x) %}{{ y }};{% endfor %}{% endif %}{% endfor %}",
    # `dict`: a map or none, and keyword arguments (`functions::dict`).
    "{{ dict(a=1) | tojson }}|{{ dict({'b': 2}, a=1) | tojson }}|{{ dict({'b': 2}, b=1) | tojson }}"
    "|{{ dict(nope, a=1) | tojson }}|{{ dict(namespace(b=1, a=2)) | tojson }}",
    "{% set x = none %}{{ dict(x) | tojson }}",
    "{% set x = [['a', 1]] %}{{ dict(x) | tojson }}",
    "{{ ['a'] | map('replace', 'a', 'b') | list }}",
    # `range` takes a float that is an integer.
    "{{ range(2.0) | list }}|{{ range(true) | list }}|{{ range(5, 1, -2) | list }}",
    "{{ range(2.5) | list }}",
]


def test_every_name_the_renderer_vouches_for_is_held_to_minijinja():
    renderer = _script("renderer")
    # `none` alone: the Python binding cannot show the runtime's; its own test is below.
    assert set(_ADVERSARIAL_FILTERS) == renderer["VOUCHED_FILTERS"]
    assert set(_ADVERSARIAL_TESTS) == renderer["VOUCHED_TESTS"] - {"none"}
    assert set(_ADVERSARIAL_FUNCTIONS) == renderer["VOUCHED_FUNCTIONS"]


_EVERY_ADVERSARIAL = [
    snippet
    for table in (_ADVERSARIAL_FILTERS, _ADVERSARIAL_TESTS, _ADVERSARIAL_FUNCTIONS)
    for snippets in table.values()
    for snippet in snippets
] + _ADVERSARIAL_SYNTAX


@pytest.mark.parametrize("source", _EVERY_ADVERSARIAL)
def test_the_renderer_renders_what_jinja2_coerces_otherwise_as_minijinja_does(source):
    """Each case renders as MiniJinja renders it, and is vouched for: a case the
    renderer reproduced but refused would document nothing `convert` checks."""
    ours, theirs = _render(source, [], True), _minijinja(source, [], True)
    assert ("text" in ours) == ("text" in theirs), (ours, theirs)
    assert ours.get("text") == theirs.get("text")
    assert _unvouched(source) == []


@pytest.mark.parametrize(
    ("source", "printed"),
    [
        # What `create_env`'s own `lstrip` and `rstrip` take: `Cow<str>`, which
        # takes any value as it prints (`minijinja_template.rs`; `ArgType for
        # Cow<str>`, MiniJinja 2.14.0 `src/value/argtypes.rs`). The Python
        # binding's harness cannot show it: its filters are Python's.
        ("{{ true | lstrip('t') }}|{{ none | rstrip('e') }}", "rue|non"),
        ("{{ 'a' ~ true }}", "atrue"),
        ("{{ [true, none] | join(',') }}", "true,none"),
        ("{{ 1e20 }}|{{ 1e-7 }}", "100000000000000000000.0|0.0000001"),
        ("{{ 'kept' }}\n", "kept\n"),
    ],
)
def test_the_renderer_prints_as_minijinja_does(source, printed):
    assert _render(source, [], True) == {"text": printed}
    assert _unvouched(source) == []


# What Codex found: a template whose verdict a Python `True` reverses. With
# `~` joined as Python prints, `marker` is "True" and every message is rendered
# -- a prefix-safe template; as MiniJinja prints, it is "true" and only the last
# message is, which rewrites the history on every message.
_REVERSED_BY_COERCION = (
    "{%- set marker = true ~ '' -%}"
    "{%- if marker == 'True' -%}{%- for m in messages %}{{ m.role }};{% endfor -%}"
    "{%- else -%}{{ messages[-1].role }};{%- endif -%}"
    "{%- if add_generation_prompt %}assistant:{% endif %}"
)


def test_a_verdict_jinja2_would_reverse_is_the_runtimes():
    # Jinja2 as transformers configures it, from the renderer's own import.
    stock = _script("renderer")["ImmutableSandboxedEnvironment"](
        trim_blocks=True, lstrip_blocks=True
    )
    history = CONVERSATIONS[0].history()
    stock_old = stock.from_string(_REVERSED_BY_COERCION).render(
        messages=history, add_generation_prompt=False
    )
    stock_new = stock.from_string(_REVERSED_BY_COERCION).render(
        messages=[*history, CONVERSATIONS[0].following()], add_generation_prompt=True
    )
    assert stock_new.startswith(stock_old), "stock Jinja2 calls it prefix-safe"
    assert _minijinja(_REVERSED_BY_COERCION, history, False) == {"text": "assistant;"}
    breaks, refused = _breaks(_REVERSED_BY_COERCION)
    assert refused == []
    assert len(breaks) == len(CONVERSATIONS) * len(CONTEXTS)


def test_the_renderers_none_is_the_runtimes_is_undefined():
    """`create_env` replaces MiniJinja's `none` test with `value.is_undefined()`
    (`minijinja_template.rs`), so a null the runtime hands over is not none and a
    missing key is. The Python binding cannot show this: it hands a test `None`
    for both."""
    messages = [{"role": "user", "content": None}]
    source = "{{ messages[0].content is none }}|{{ messages[0].missing is none }}"
    assert _render(source, messages, True) == {"text": "false|true"}


def test_the_renderer_refuses_a_break_as_the_runtime_build_does():
    """LiteRT-LM builds MiniJinja 2.14.0 without `loop_controls` (`Cargo.toml`
    names `json` and `preserve_order` besides the defaults), so `{% break %}` is
    a syntax error on the device -- the Python binding, built with it, accepts it."""
    source = "{% for x in [1, 2] %}{{ x }}{% break %}{% endfor %}"
    assert "error" in _render(source, [], True)
    assert _minijinja(source, [], True) == {"text": "1"}
    assert any("cannot parse" in gap for gap in _unvouched(source))


@pytest.mark.parametrize(
    ("source", "named"),
    [
        ("{{ messages[0].content | title }}", "the filter `title`"),
        ("{{ messages | selectattr('role') | list }}", "the filter `selectattr`"),
        ("{{ 1 is divisibleby 1 }}", "the test `divisibleby`"),
        ("{{ strftime_now('%Y') }}", "the function `strftime_now`"),
        ("{{ cycler(1, 2).next() }}", "the function `cycler`"),
        ("{% include 'x' %}", "a `Include` tag"),
        ("{% import 'x' as y %}", "a `Import` tag"),
        ("{% block b %}{% endblock %}", "a `Block` tag"),
        ("{% autoescape true %}{% endautoescape %}", "a `ScopedEvalContextModifier` tag"),
        ("{{ 'x' if is_appending_to_prefill }}", "`is_appending_to_prefill`"),
        ("{% if %}", "Jinja2 cannot parse it"),
        # `loop(...)` re-enters only the `recursive` loop whose body it is in.
        ("{% for x in [1] %}{{ loop([2]) }}{% endfor %}", "the function `loop`"),
        (
            "{% for x in [1] recursive %}{% for y in [2] %}{{ loop([y]) }}{% endfor %}{% endfor %}",
            "the function `loop`",
        ),
        (
            "{% for x in [1] recursive %}{% if x %}{% for y in [2] %}{{ loop([y]) }}{% endfor %}"
            "{% endif %}{% endfor %}",
            "the function `loop`",
        ),
    ],
)
def test_what_the_renderer_does_not_vouch_for_is_named(source, named):
    assert any(named in gap for gap in _unvouched(source)), _unvouched(source)


# What the renderer renders otherwise than MiniJinja 2.14.0, and so does not
# vouch for: each one differs, and each one is named.
_RENDERED_OTHERWISE = [
    # Python's chaining, which MiniJinja has only from 2.20.0; 2.14 reads
    # `(1 == 2) == false`.
    ("{{ 1 == 2 == false }}", "a chained comparison, which MiniJinja 2.14 reads as nested ones"),
    ("{{ 1 < 3 < 2 }}", "a chained comparison"),
    ("{{ +3 }}", "a unary `+`, which MiniJinja 2.14 cannot parse"),
    ("{{ 0 and 1 }}", "an `and` of constants"),
    ("{{ 'a\\/b' }}", "whose escapes MiniJinja 2.14 reads otherwise"),
    ("{{ 'a\\sb' }}", "whose escapes MiniJinja 2.14 reads otherwise"),
    ("{{ 'a\\ab' }}", "whose escapes MiniJinja 2.14 reads otherwise"),
    ("{{ '\\ud83d\\ude80' }}", "whose escapes MiniJinja 2.14 reads otherwise"),
    ("{{ [1, 2].0 }}", "a dotted integer lookup"),
    ("{% set x = [{'a': 1}] %}{{ x.0.a }}", "a dotted integer lookup"),
    ("{% macro f(a) %}{{ a }}{{ varargs }}{% endmacro %}{{ f(1, 2) }}", "the name `varargs`"),
    ("{% macro f(a) %}{{ kwargs }}{% endmacro %}{{ f(1, k=3) }}", "the name `kwargs`"),
    ("{{ self }}", "the name `self`"),
    (
        "{{ -170141183460469231731687303715884105728 }}",
        "the integer literal 170141183460469231731687303715884105728",
    ),
    ("{% macro f(a, b=a) %}[{{ b }}]{% endmacro %}{{ f(1) }}", "a macro default that names"),
    ("{% macro f(a=b, b=1) %}[{{ a }}]{% endmacro %}{{ f() }}", "a macro default that names"),
    # A function printed: Python's repr here, a Rust type name there.
    ("[{{ range }}]", "the function `range` as a value"),
    ("[{{ debug }}]", "the function `debug` as a value"),
    ("[{{ __mj_iter__ }}]", "the name `__mj_iter__`, which this renderer reserves"),
    # `map` by name finds Jinja2's filters here; MiniJinja has no `wordcount`.
    ("{{ ['a b'] | map('wordcount') | list }}", "`map` of the filter `wordcount`"),
    # MiniJinja's `map` drops a keyword argument the named filter is given.
    (
        "{{ ['a'] | map('replace', 'a', 'b', count=1) | list }}",
        "`map` with keyword arguments other than one plain `attribute`",
    ),
]


# Jinja2's globals MiniJinja 2.14.0 does not have, used other than in a call:
# printed, aliased, aliased under a macro's name, assigned over. The renderer
# has none of them, so each renders as on the device; each is still not
# vouched for.
_JINJA2_ONLY_GLOBAL = "a Jinja2 global MiniJinja 2.14 does not have"
_RENDERED_ALIKE_NOT_VOUCHED_FOR = [
    ("[{{ cycler }}]", _JINJA2_ONLY_GLOBAL),
    ("{% set c = cycler %}[{{ c }}]", _JINJA2_ONLY_GLOBAL),
    ("{% macro m() %}{% endmacro %}{% set m = joiner %}{{ m(',') }}", _JINJA2_ONLY_GLOBAL),
    ("{% set cycler = 1 %}{{ cycler }}", _JINJA2_ONLY_GLOBAL),
    ("[{{ lipsum }}]", _JINJA2_ONLY_GLOBAL),
    # Pairs: Jinja2's `dict` is Python's and takes them; MiniJinja's refuses
    # them, and so does the renderer's.
    ("{{ dict([['a', 1]]) | tojson }}", "`dict` of a value that is not a map"),
    ("{{ dict({'a': 1}, {'b': 2}) }}", "`dict` of more than one value"),
    ("{% set kw = {'a': 1} %}{{ dict(**kw) }}", "`dict` with `*` or `**` arguments"),
    # `map` shapes not vouched for, whatever they render.
    ("{% set f = 'upper' %}{{ ['a'] | map(f) | list }}", "`map` of a filter named at run time"),
    (
        "{{ [{'a': {'b': 1}}] | map(attribute='a.b') | list }}",
        "`map` with keyword arguments other than one plain `attribute`",
    ),
    (
        "{{ [{'a': 1}] | map(attribute='b', default=2) | list }}",
        "`map` with keyword arguments other than one plain `attribute`",
    ),
]


@pytest.mark.parametrize(("source", "named"), _RENDERED_ALIKE_NOT_VOUCHED_FOR)
def test_what_the_renderer_renders_alike_but_does_not_vouch_for_is_named(source, named):
    """Rendered as MiniJinja renders it, and still not vouched for: a shape this
    renderer's tests do not hold to MiniJinja in general."""
    ours, theirs = _render(source, [], True), _minijinja(source, [], True)
    assert ("text" in ours) == ("text" in theirs), (ours, theirs)
    assert ours.get("text") == theirs.get("text")
    assert any(named in gap for gap in _unvouched(source)), _unvouched(source)


def test_jinja2_as_transformers_configures_it_renders_these_otherwise():
    stock = _script("renderer")["ImmutableSandboxedEnvironment"](
        trim_blocks=True, lstrip_blocks=True
    )
    assert stock.from_string("[{{ cycler }}]").render() == "[<class 'jinja2.utils.Cycler'>]"
    assert _render("[{{ cycler }}]", [], True) == {"text": "[]"}
    assert stock.from_string("{{ dict([['a', 1]]) | tojson }}").render() == '{"a": 1}'
    assert "error" in _render("{{ dict([['a', 1]]) | tojson }}", [], True)


@pytest.mark.parametrize(("source", "named"), _RENDERED_OTHERWISE)
def test_what_the_renderer_renders_otherwise_than_minijinja_is_named(source, named):
    ours, theirs = _render(source, [], True), _minijinja(source, [], True)
    same = ("text" in ours) == ("text" in theirs) and ours.get("text") == theirs.get("text")
    assert not same, (ours, theirs)
    assert any(named in gap for gap in _unvouched(source)), _unvouched(source)


def test_a_composed_template_with_a_chained_comparison_is_not_checked(tmp_path):
    """Every single-turn case renders as the checkpoint's template renders it --
    the renderer chains as Jinja2 does -- so only the construct keeps it from
    passing: MiniJinja 2.14 would render the comparison true, and print `!`."""
    template = (
        "{%- for m in messages %}{{ m.role }}:{{ m.content }};"
        "{% if 1 == 2 == false %}!{% endif %}{% endfor %}"
    )
    renders = _compose(tmp_path, template, None)
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []
    assert renders["unvouched"]["composed"] == [
        "a chained comparison, which MiniJinja 2.14 reads as nested ones"
    ]
    choice = export._choose_template("m", renders, None)
    assert choice.source is None and choice.refusal is None
    assert choice.unchecked is not None
    assert "a chained comparison" in choice.unchecked and "was not checked" in choice.unchecked


def test_a_conversation_with_a_chained_comparison_is_not_checked():
    """Rendered, it would pass: every case grows by appending."""
    template = (
        "{%- for m in messages %}{{ m.role }};{% if 1 == 2 == false %}!{% endif %}{% endfor %}"
    )
    report = _conversation(template)
    breaks, _ = conversation_breaks(report["renders"])
    assert breaks == []
    assert report["unvouched"] == [
        "a chained comparison, which MiniJinja 2.14 reads as nested ones"
    ]


def test_a_macro_of_the_templates_own_is_vouched_for():
    assert _unvouched("{% macro f() %}x{% endmacro %}{{ f() }}{{ namespace(a=1).a }}") == []


def test_a_template_the_renderer_does_not_vouch_for_is_not_checked():
    """Rendered, it would pass: every case grows by appending. Not vouched for, it
    is not checked -- never passed."""
    template = "{%- for m in messages %}{{ m.content[0].text | title }};{% endfor %}"
    report = _conversation(template)
    breaks, _ = conversation_breaks(report["renders"])
    assert breaks == []
    assert report["unvouched"] == ["the filter `title`"]


GOLDENS = FIXTURES / "litert-lm-v0.18.0-qwen3-testdata"
# `git ls-tree v0.18.0 models/qwen3/testdata` in google-ai-edge/LiteRT-LM.
GOLDEN_BLOBS = {
    "golden/system_instruction-thinking.txt": "975391d9ee4a633dce8e89a75b50ab00a362dac7",
    "golden/system_instruction.txt": "0b8d85d3778c63f21888b90fe8a95db35ac6322b",
    "golden/tool_response_ending-thinking.txt": "1deefe597a82cd9671df1ded0968a6e5c1f850c2",
    "golden/tool_response_ending.txt": "61b32a2d5dc9b61f47c8f89da3ba47c2f4dd88a9",
    "golden/tools-thinking.txt": "759fe508507fae8d76ab0438724e942ac111a887",
    "golden/tools.txt": "e6d9bb9e3092999e9791f7cc788c3ed5d76452b0",
    "input/system_instruction.json": "285a4f11ae9af1095b48e9636f1c3c769fa168a2",
    "input/tool_response_ending.json": "37e51dad2f47c6ed166df16387fe0e4b65e89781",
    "input/tools.json": "eee9d6b3f0fc2e608e62baa510bff055e7392b69",
}


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_each_litert_lm_fixture_is_the_file_at_v0_18_0():
    assert sorted(
        p.relative_to(GOLDENS).as_posix() for p in GOLDENS.rglob("*") if p.is_file()
    ) == sorted(GOLDEN_BLOBS)
    for name, blob in GOLDEN_BLOBS.items():
        assert _git_blob((GOLDENS / name).read_bytes()) == blob, name
    qwen25 = (FIXTURES / "litert-lm-v0.18.0-qwen2_5.chat_template.jinja").read_bytes()
    assert _git_blob(qwen25) == "154f74aaeac0427d24bb789270fd060f90028a0b"


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("name", ["system_instruction", "tool_response_ending", "tools"])
def test_the_renderer_reproduces_litert_lms_goldens_for_its_qwen3_template(name, thinking):
    """What LiteRT-LM's own `chat_template_test_runner.cc` compares MiniJinja's
    rendering of `models/qwen3/chat_template.jinja` with -- a multi-turn
    conversation, tool declarations, a tool response -- rendered here from the
    packaged template, its provenance comment included."""
    data = json.loads((GOLDENS / "input" / f"{name}.json").read_bytes())
    suffix = "-thinking" if thinking else ""
    golden = (GOLDENS / "golden" / f"{name}{suffix}.txt").read_bytes().decode("utf-8")
    context: dict[str, Any] = {"enable_thinking": thinking}
    if "tools" in data:
        context["tools"] = data["tools"]
    gen = data.get("add_generation_prompt", True)
    assert _render(LITERT_QWEN3, data["messages"], gen, **context) == {"text": golden}
    assert _minijinja(LITERT_QWEN3, data["messages"], gen, **context) == {"text": golden}


# ---------------------------------------------------------------------------
# The packaged canonical template
# ---------------------------------------------------------------------------


def test_the_packaged_qwen3_template_is_litert_lms_byte_for_byte():
    raw = Path(models.packaged_template(models.QWEN3_CANONICAL_TEMPLATE)).read_bytes()
    header, found, body = raw.partition(models.CANONICAL_HEADER_END.encode("utf-8"))
    assert found
    sha256 = "4315c96833e8b2a21b878d70e8d2f6415416d5e23ed8ee0ae0a111a540d844f7"
    assert hashlib.sha256(body).hexdigest() == sha256
    assert models.CANONICAL_TEMPLATES[models.QWEN3_CANONICAL_TEMPLATE] == sha256
    assert _git_blob(body) == "72501d10e25fd5382a2a60ecf03f8b25acd8de4e"
    said = header.decode("utf-8")
    for fact in (
        "https://github.com/google-ai-edge/LiteRT-LM",
        "tag v0.18.0",
        "b2f686e2ed4718fb84ec398a61dd59ca0f0aff27",
        "models/qwen3/chat_template.jinja",
        "72501d10e25fd5382a2a60ecf03f8b25acd8de4e",
        sha256,
        "Apache-2.0",
    ):
        assert fact in said, fact
    # The runtime rewrites nothing in it, and it prints nothing.
    assert _script("renderer")["edit_for_minijinja"](said) == said
    assert _render(said + "-#}", [], True) == {"text": ""}
    assert models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE) == raw.decode("utf-8")


@pytest.mark.parametrize(
    "damage",
    [
        lambda text: text + " ",
        lambda text: text.replace("{#- litetune:", "{# litetune:", 1),
        lambda text: text.replace(models.CANONICAL_HEADER_END, "#}\n", 1),
    ],
    ids=["body", "header", "header-end"],
)
def test_a_packaged_canonical_template_that_is_not_the_pinned_one_is_refused(
    monkeypatch, tmp_path, damage
):
    copy = tmp_path / models.QWEN3_CANONICAL_TEMPLATE
    copy.write_bytes(damage(LITERT_QWEN3).encode("utf-8"))
    monkeypatch.setattr(models, "packaged_template", lambda name: str(tmp_path / name))
    with pytest.raises(ValueError, match="is not the template its sha256 is pinned to"):
        models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE)


def test_qwens_own_template_is_pinned_as_the_fixture_holds_it():
    """The digest a checkpoint with no record is offered LiteRT-LM's template on is
    Qwen/Qwen3-0.6B's own at c1899de, the fixture byte for byte -- whose
    single turns LiteRT-LM's renders as it does (the test above)."""
    upstream = models.CANONICAL_UPSTREAM[models.QWEN3_CANONICAL_TEMPLATE]
    digest = hashlib.sha256(QWEN3.encode("utf-8")).hexdigest()
    assert digest == "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    assert list(upstream) == [digest]
    assert "c1899de289a04d12100db370d81485cdf75e47ca" in upstream[digest]
    assert set(models.CANONICAL_UPSTREAM) <= set(models.CANONICAL_TEMPLATES)


def test_every_rule_offering_a_canonical_template_names_a_pinned_one():
    for rules in models.RULES:
        markers = {r.value for r in rules.required_flags if r.name == models.TEMPLATE_FLAG}
        offers = models.CANONICAL_OR_COMPOSED_TEMPLATE in markers
        assert offers == (rules.canonical_template is not None), rules.family
        if rules.canonical_template is not None:
            assert rules.canonical_template in models.CANONICAL_TEMPLATES
            models.canonical_template_text(rules.canonical_template)


# ---------------------------------------------------------------------------
# Single-turn identity
# ---------------------------------------------------------------------------


def _compose(
    tmp_path: Path,
    checkpoint: str,
    canonical: str | None,
    special: dict | None = None,
    add_bos: bool = False,
) -> dict:
    """`export._COMPOSE_SCRIPT` on a checkpoint carrying `checkpoint`: the
    checkpoint's side through transformers -- the conftest fake renders as
    5.16.1 does -- and each candidate through the runtime's renderer."""
    work = tmp_path / hashlib.sha256((checkpoint + str(canonical)).encode()).hexdigest()[:8]
    work.mkdir()
    script, spec, result = work / "compose_template.py", work / "spec.json", work / "result.json"
    script.write_text(export._COMPOSE_SCRIPT, encoding="utf-8")
    spec.write_text(
        json.dumps(
            {
                "model": "m",
                "block": BLOCK,
                "out": str(work / "composed.jinja"),
                "canonical": canonical,
                "single_turns": template_checks.single_turn_spec(),
            }
        ),
        encoding="utf-8",
    )
    proc, _ = run_compose_script(
        ["python", str(script), str(spec), str(result)],
        60,
        checkpoint,
        special_tokens=special,
        add_bos=add_bos,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["found"] == "text"
    return report["renders"]


TOTAL = len(SINGLE_TURNS) * len(FORMS)


def test_litert_lms_qwen3_template_renders_single_turns_as_qwen3s_own(tmp_path):
    """The condition the qwen-3 rule packs it on, on the checkpoint it was run
    on: every case, with enable_thinking undefined, true and false, with no
    tools and with one."""
    renders = _compose(tmp_path, QWEN3, LITERT_QWEN3)
    assert single_turn_differences(renders["reference"], renders["candidates"]["canonical"]) == []
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []
    assert renders["unvouched"] == {"composed": [], "canonical": []}
    choice = export._choose_template("m", renders, models.QWEN3_CANONICAL_TEMPLATE)
    assert choice.source == "canonical"
    assert choice.refusal is None and choice.unchecked is None


def test_the_contexts_reach_both_sides(tmp_path):
    """enable_thinking false adds Qwen3's empty think block to the generation
    prompt, and a tool adds a system turn of declarations, on both sides."""
    renders = _compose(tmp_path, QWEN3, LITERT_QWEN3)
    for case, reference, canonical in zip(
        SINGLE_TURNS, renders["reference"], renders["candidates"]["canonical"], strict=True
    ):
        text = reference["text"]
        assert canonical["one_part"]["text"] == text
        assert text.endswith("<think>\n\n</think>\n\n") is (case.context.thinking is False)
        assert ("<tools>" in text) is case.context.tools


def test_litert_lms_qwen25_template_does_not_render_qwen25s_default_system_prompt(tmp_path):
    """Why the qwen-2.5 rule offers no canonical template (the comment beside it):
    with no system message the checkpoint's template prints its default one and
    LiteRT-LM's prints none -- half the renderings, exactly those."""
    renders = _compose(tmp_path, QWEN25, LITERT_QWEN25)
    differences = single_turn_differences(renders["reference"], renders["candidates"]["canonical"])
    assert len(differences) == TOTAL // 2 == 48
    assert {d.case.system for d in differences} == {None}
    default = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert all(default in (d.reference.text or "") for d in differences)
    assert all(default not in (d.candidate.text or "") for d in differences)
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []
    choice = export._choose_template("m", renders, "litert-qwen2.5")
    assert choice.source == "composed"
    assert len(choice.canonical_differences) == 48


def _renders(**unvouched: list[str]) -> dict:
    """Renderings in which both candidates render as the checkpoint's template does."""
    reference, candidate = _same(text=PROBE_TEXT)
    return {
        "reference": reference,
        "candidates": {"canonical": candidate, "composed": candidate},
        "unvouched": {"canonical": [], "composed": [], **unvouched},
        "parts_alone": {"text": f"{PROBE_TEXT}0"},
    }


def test_a_canonical_template_the_renderer_does_not_vouch_for_is_not_chosen():
    choice = export._choose_template("m", _renders(canonical=["the filter `title`"]), "c")
    assert choice.source == "composed"
    assert choice.canonical_differences == (
        "it uses the filter `title`, which the renderer does not vouch for",
    )


def test_a_composed_template_the_renderer_does_not_vouch_for_is_not_checked():
    choice = export._choose_template("m", _renders(composed=["the filter `title`"]), None)
    assert choice.source is None and choice.refusal is None
    assert choice.unchecked is not None
    assert "uses the filter `title`" in choice.unchecked and "was not checked" in choice.unchecked


@pytest.mark.parametrize("unvouched", [None, "x", [1], {"composed": []}])
def test_unvouched_constructs_this_cannot_read_are_refused(unvouched):
    renders = _renders()
    renders["unvouched"] = unvouched
    with pytest.raises(ValueError):
        export._choose_template("m", renders, "c")


def test_the_cases_cover_a_system_message_two_parts_thinking_and_tools():
    assert len(SINGLE_TURNS) == 48
    assert {case.system is None for case in SINGLE_TURNS} == {True, False}
    assert {case.context for case in SINGLE_TURNS} == set(CONTEXTS)
    assert {context.thinking for context in CONTEXTS} == {None, True, False}
    assert {context.tools for context in CONTEXTS} == {True, False}
    # The probe is what a run without declarations trained on: no context at all.
    assert SINGLE_TURNS[0] == template_checks.SingleTurn(PROBE_TEXT)
    assert SINGLE_TURNS[0].context.values() == {}
    two = SINGLE_TURNS[1].as_parts("two_parts")
    assert [len(m["content"]) for m in two] == [2, 2]
    assert "".join(p["text"] for p in two[1]["content"]) == PROBE_TEXT
    assert SINGLE_TURNS[1].as_strings()[0] == {"role": "system", "content": SINGLE_TURNS[1].system}
    assert Context(False, True).values() == {"enable_thinking": False, "tools": list(TOOLS)}


def _same(count: int = len(SINGLE_TURNS), text: str = "x") -> tuple[list, list]:
    reference = [{"text": f"{text}{i}"} for i in range(count)]
    candidate = [{form: {"text": f"{text}{i}"} for form in FORMS} for i in range(count)]
    return reference, candidate


def test_identical_renderings_are_no_difference():
    assert single_turn_differences(*_same()) == []


@pytest.mark.parametrize("form", FORMS)
def test_one_differing_rendering_is_one_difference(form):
    reference, candidate = _same()
    candidate[3][form] = {"text": "other"}
    (difference,) = single_turn_differences(reference, candidate)
    assert difference.case == SINGLE_TURNS[3] and difference.form == form
    assert "printed 'x3'" in difference.describe() and "printed 'other'" in difference.describe()


def test_a_candidate_that_raises_differs_and_two_that_raise_do_not():
    reference, candidate = _same()
    candidate[2]["one_part"] = {"error": "TemplateError: boom"}
    (difference,) = single_turn_differences(reference, candidate)
    assert "the bundle's raised TemplateError: boom" in difference.describe()
    reference[2] = {"error": "TemplateError: boom"}
    candidate[2]["two_parts"] = {"error": "TemplateError: also"}
    assert single_turn_differences(reference, candidate) == []
    candidate[2]["two_parts"] = {"text": "x2"}
    (difference,) = single_turn_differences(reference, candidate)
    assert difference.form == "two_parts"


@pytest.mark.parametrize(
    "spoil",
    [
        lambda r, c: (r[:-1], c),
        lambda r, c: (r, c[:-1]),
        lambda r, c: (r, [*c[:-1], "text"]),
        lambda r, c: (r, [*c[:-1], {"one_part": {"text": "x"}}]),
        lambda r, c: ([*r[:-1], {"txt": "x"}], c),
        lambda r, c: ("renders", c),
    ],
)
def test_renderings_this_cannot_read_are_refused(spoil):
    with pytest.raises(ValueError):
        single_turn_differences(*spoil(*_same()))


# ---------------------------------------------------------------------------
# The conversation prefix invariant
# ---------------------------------------------------------------------------

ALL_CASES = [(case, context) for case in CONVERSATIONS for context in CONTEXTS]


def test_qwen3s_own_template_breaks_every_conversation():
    """The defect: behind the block, which it needs on 0.18.0, Qwen3's template
    renders the latest reply with a think block and an earlier one without --
    for a plain reply, one carrying a think block and one a thought channel
    took, over two turns and three, with a system message or none, in every
    context."""
    breaks, refused = _breaks(BLOCK + QWEN3)
    assert refused == []
    assert [(b.case, b.context) for b in breaks] == ALL_CASES
    plain = breaks[0]
    assert plain.case.reply.name == "a plain reply" and plain.case.turns == 2
    assert plain.previous.text is not None and plain.following.text is not None
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\ncard_lost" in plain.previous.text
    assert "<|im_start|>assistant\ncard_lost<|im_end|>" in plain.following.text
    assert "from character" in plain.describe()
    assert "enable_thinking undefined, no tools" in plain.describe()


@pytest.mark.parametrize(
    "name",
    ["qwen3 litert-lm", "qwen2.5 composed", "qwen2.5 litert-lm", "functiongemma packaged"],
)
def test_these_templates_hold_every_conversation(name):
    assert _breaks(BUNDLED[name]) == ([], [])


# Templates that break only where one kind of case reaches them: what each case
# and each context is there to catch.
_LAST_REPLY_MARKED_FROM_TURN_THREE = (
    "{%- for m in messages %}{{ m.role }}:{{ m.content[0].text }}"
    "{% if m.role == 'assistant' and loop.index0 >= 3 and loop.last %}*{% endif %};"
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
)
_LAST_THINK_BLOCK_BRACKETED = (
    "{%- for m in messages %}{% set t = m.content[0].text %}"
    "{% if '<think>' in t and loop.last %}[{{ t }}]{% else %}{{ t }}{% endif %};{% endfor %}"
)
_LAST_REASONING_SHOWN = (
    "{%- for m in messages %}{{ m.content[0].text }}"
    "{% if m.reasoning_content and loop.last %}<{{ m.reasoning_content }}>{% endif %};"
    "{% endfor %}"
)
_LAST_REPLY_MARKED_WITHOUT_THINKING = (
    "{%- for m in messages %}{{ m.content[0].text }}"
    "{% if enable_thinking is false and m.role == 'assistant' and loop.last %}!{% endif %};"
    "{% endfor %}"
)
_LAST_REPLY_MARKED_WITH_THINKING = (
    "{%- for m in messages %}{{ m.content[0].text }}"
    "{% if enable_thinking is true and m.role == 'assistant' and loop.last %}!{% endif %};"
    "{% endfor %}"
)
_LAST_REPLY_MARKED_WITH_TOOLS = (
    "{%- for m in messages %}{{ m.content[0].text }}"
    "{% if tools and m.role == 'assistant' and loop.last %}!{% endif %};"
    "{% endfor %}"
)


@pytest.mark.parametrize(
    ("template", "reached"),
    [
        (_LAST_REPLY_MARKED_FROM_TURN_THREE, lambda case, context: case.turns == 3),
        (
            _LAST_THINK_BLOCK_BRACKETED,
            lambda case, context: case.reply.name.endswith("a think block"),
        ),
        (
            _LAST_REASONING_SHOWN,
            lambda case, context: "thought channel" in case.reply.name,
        ),
        (_LAST_REPLY_MARKED_WITHOUT_THINKING, lambda case, context: context.thinking is False),
        (_LAST_REPLY_MARKED_WITH_THINKING, lambda case, context: context.thinking is True),
        (_LAST_REPLY_MARKED_WITH_TOOLS, lambda case, context: context.tools),
    ],
    ids=[
        "three-turns",
        "think-block",
        "thought-channel",
        "thinking-false",
        "thinking-true",
        "tools",
    ],
)
def test_each_kind_of_case_catches_what_only_it_reaches(template, reached):
    breaks, refused = _breaks(template)
    assert refused == []
    assert breaks, "a case is missing"
    assert [(b.case, b.context) for b in breaks] == [
        (case, context) for case, context in ALL_CASES if reached(case, context)
    ]


def test_a_template_that_raises_on_a_reply_breaks_with_the_error():
    breaks, refused = _breaks(
        "{%- for m in messages %}{% if m.role == 'assistant' %}"
        "{{ raise_exception('no replies') }}{% endif %}{% endfor %}"
    )
    assert refused == []
    assert len(breaks) == len(ALL_CASES)
    said = breaks[0].describe()
    assert "the template raises on the history ending in the reply" in said
    assert "no replies" in said


def test_a_first_message_the_template_refuses_is_not_a_break():
    """A template that refuses a system message refuses it in the first message
    too: that says nothing about the second, and is not counted either way."""
    template = (
        "{%- for m in messages %}{% if m.role == 'system' %}{{ raise_exception('no system') }}"
        "{% endif %}{{ m.content[0].text }};{% endfor %}"
    )
    breaks, refused = _breaks(template)
    assert breaks == []
    assert [(r.case, r.context) for r in refused] == [
        (case, context) for case, context in ALL_CASES if case.system
    ]
    assert "the first message raised" in refused[0].describe()
    check = ConversationCheck.of(template, breaks, refused, "the renderer")
    assert check.holds is True
    assert "36 of them the template refuses the conversation's first message" in check.detail


def test_a_template_that_refuses_every_first_message_is_not_checked():
    template = "{{ raise_exception('never') }}"
    breaks, refused = _breaks(template)
    check = ConversationCheck.of(template, breaks, refused, "the renderer")
    assert check.holds is None
    assert len(check.refused) == len(ALL_CASES)


def test_a_template_that_grows_by_appending_holds():
    assert _breaks("{%- for m in messages %}{{ m.content[0].text }};{% endfor %}") == ([], [])


def test_the_conversations_cover_both_reply_shapes_and_three_turns():
    assert len(CONVERSATIONS) == 12
    assert len(CONTEXTS) == 6
    assert {case.turns for case in CONVERSATIONS} == {2, 3}
    assert {case.system is None for case in CONVERSATIONS} == {True, False}
    three = next(c for c in CONVERSATIONS if c.turns == 3 and c.system)
    roles = [m["role"] for m in three.history()]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert [m["role"] for m in three.first()] == ["system", "user"]
    assert three.following()["role"] == "user"
    think, channel = (r.message for r in template_checks.REPLIES[1:])
    assert "<think>" in think["content"][0]["text"]
    assert "<think>" not in channel["content"][0]["text"]
    assert channel["reasoning_content"] == channel["channels"]["thought"]


def _rendered(first: dict, previous: dict, following: dict) -> list:
    entry = {"first": first, "previous": previous, "next": following}
    return [[dict(entry) for _ in CONTEXTS] for _ in CONVERSATIONS]


@pytest.mark.parametrize(
    ("previous", "following", "breaks"),
    [
        ({"text": "ab"}, {"text": "abc"}, False),
        ({"text": "ab"}, {"text": "ab"}, False),
        ({"text": "ab"}, {"text": "xab"}, True),
        ({"text": "abc"}, {"text": "ab"}, True),
        ({"error": "E"}, {"text": "ab"}, True),
        ({"text": "ab"}, {"error": "E"}, True),
    ],
)
def test_the_invariant_is_a_byte_prefix(previous, following, breaks):
    found, refused = conversation_breaks(_rendered({"text": "a"}, previous, following))
    assert bool(found) is breaks
    assert refused == []


def test_a_refused_first_message_is_never_a_break():
    found, refused = conversation_breaks(_rendered({"error": "E"}, {"text": "ab"}, {"text": "xab"}))
    assert found == [] and len(refused) == len(ALL_CASES)


@pytest.mark.parametrize(
    "entries",
    [
        "renders",
        [],
        _rendered({"text": "a"}, {"text": "a"}, {"text": "a"})[:-1],
        [[{"previous": {"text": "a"}}] * len(CONTEXTS)] * len(CONVERSATIONS),
        [[{"first": {"text": "a"}, "previous": {"text": "a"}, "next": {"text": "a"}}]]
        * len(CONVERSATIONS),
    ],
)
def test_conversation_renderings_this_cannot_read_are_refused(entries):
    with pytest.raises(ValueError):
        conversation_breaks(entries)


def test_the_limitation_names_the_recipes_the_runtime_the_contexts_and_the_first_break():
    breaks, refused = _breaks(BLOCK + QWEN3)
    check = ConversationCheck.of(BLOCK + QWEN3, breaks, refused, "the renderer")
    assert check.holds is False and check.applicable
    assert check.template_sha256 == hashlib.sha256((BLOCK + QWEN3).encode()).hexdigest()
    text = conversation_limitation(["dynamic_wi8_afp32", "weight_only_wi8_afp32"], check)
    assert text.startswith(f"dynamic_wi8_afp32, weight_only_wi8_afp32: {LIMITATION_PREFIX}")
    for fact in (
        "The new rendered template string does not start with the previous rendered template",
        "Conversation::GetSingleTurnTextFromFullHistory",
        "google-ai-edge/LiteRT-LM#3443",
        "72 of the 72 cases break it",
        "enable_thinking undefined, true and false, each with no tools and with one tool",
        check.breaks[0],
    ):
        assert fact in text
    held = ConversationCheck.of(LITERT_QWEN3, [], [], "the renderer")
    assert held.holds is True and held.breaks == ()
    assert "Other contexts were not checked" in held.detail
    assert held.as_dict() == {
        "holds": True,
        "applicable": True,
        "detail": held.detail,
        "template_sha256": hashlib.sha256(LITERT_QWEN3.encode()).hexdigest(),
        "contexts": [context.name for context in CONTEXTS],
        "breaks": [],
        "refused": [],
    }
    assert "was not checked: why" in unchecked_limitation(["r"], "why")
    assert LIMITATION_PREFIX not in unchecked_limitation(["r"], "why")


def test_a_template_on_the_single_turn_protocol_is_not_applicable():
    """LiteRT-LM finds `is_appending_to_prefill` by reading the source and then
    never compares full-history renderings for that template; the check says
    the invariant does not apply, and never that the bundle answers one message."""
    template = "{%- if is_appending_to_prefill %}{% endif %}{{ messages[-1].content[0].text }}"
    assert template_checks.single_turn_protocol(template)
    assert not template_checks.single_turn_protocol(LITERT_QWEN3)
    check = ConversationCheck.not_applicable(template)
    assert check.holds is None and not check.applicable
    text = template_checks.not_applicable_limitation(["r"], check.detail)
    assert "does not apply" in text and LIMITATION_PREFIX not in text
    # Rendered as a full history it would look broken: that is what the check avoids.
    assert _breaks(template)[0]


@pytest.mark.parametrize(
    ("entry", "reads"),
    [
        ({"text": "a"}, Rendering(text="a")),
        ({"error": "E"}, Rendering(error="E")),
        ({"error": "E", "environment": True}, Rendering(error="E", environment=True)),
    ],
)
def test_a_rendering_reads_as_written(entry, reads):
    assert Rendering.read(entry) == reads


@pytest.mark.parametrize("entry", [None, "a", {"text": 1}, {"error": None}, {}])
def test_a_rendering_in_another_shape_is_refused(entry):
    with pytest.raises(ValueError):
        Rendering.read(entry)


def test_granites_own_template_joins_text_parts_with_a_newline(tmp_path):
    """Why the granite-4.1 rule composes: through the runtime's renderer the
    checkpoint's template alone prints the two-part cases otherwise than
    transformers prints the string -- exactly those -- and behind the block it
    prints every case as transformers does."""
    renders = _compose(tmp_path, GRANITE, GRANITE)
    alone = single_turn_differences(renders["reference"], renders["candidates"]["canonical"])
    assert alone and {d.form for d in alone} == {"two_parts"}
    assert len(alone) == len(SINGLE_TURNS)
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []
    assert renders["unvouched"] == {"composed": [], "canonical": []}
    choice = export._choose_template("m", renders, None)
    assert choice.source == "composed"
    assert choice.refusal is None and choice.unchecked is None


def test_granites_own_template_holds_every_conversation():
    assert _breaks(GRANITE) == ([], [])


def test_granites_composed_template_holds_every_conversation():
    assert _breaks(BLOCK + GRANITE) == ([], [])


# ---------------------------------------------------------------------------
# The edited canonical templates: MiniCPM5 and Qwen 3.5
# ---------------------------------------------------------------------------

EDITED = {
    "minicpm5": (
        models.MINICPM5_CANONICAL_TEMPLATE,
        LITERT_MINICPM5,
        "c4184e05132f0bc3fda1a39a15775ca41ca5c5f8",
        "models/minicpm5/chat_template.jinja",
        MINICPM5,
    ),
    "qwen3.5": (
        models.QWEN35_CANONICAL_TEMPLATE,
        LITERT_QWEN35,
        "cb0fb5fc820fc7cb70cfb1f7d507cd1882fed84e",
        "models/qwen3_5/chat_template.jinja",
        QWEN35,
    ),
}


@pytest.mark.parametrize("family", list(EDITED))
def test_an_edited_template_is_litert_lms_with_exactly_its_listed_edits(family):
    name, upstream, blob, path, _ = EDITED[family]
    assert _git_blob(upstream.encode("utf-8")) == blob
    upstream_sha, edits = models.CANONICAL_EDITS[name]
    assert hashlib.sha256(upstream.encode("utf-8")).hexdigest() == upstream_sha
    edited = upstream
    for text, replacement in edits:
        assert edited.count(text) == 1, text
        edited = edited.replace(text, replacement)
    raw = models.canonical_template_text(name)
    header, found, body = raw.partition(models.CANONICAL_HEADER_END)
    assert found and body == edited
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == models.CANONICAL_TEMPLATES[name]
    for fact in (
        "https://github.com/google-ai-edge/LiteRT-LM",
        "tag v0.18.0",
        "b2f686e2ed4718fb84ec398a61dd59ca0f0aff27",
        path,
        blob,
        upstream_sha,
        "Apache-2.0",
        "models.CANONICAL_EDITS",
    ):
        assert fact in header, fact
    assert _script("renderer")["edit_for_minijinja"](header) == header
    assert _render(header + "-#}", [], True) == {"text": ""}


@pytest.mark.parametrize("family", list(EDITED))
def test_a_checkpoints_own_template_is_pinned_as_the_fixture_holds_it(family):
    name, *_, own = EDITED[family]
    digest = hashlib.sha256(own.encode("utf-8")).hexdigest()
    assert list(models.CANONICAL_UPSTREAM[name]) == [digest]


def test_litert_lms_minicpm5_template_adds_a_think_block_minicpm5s_does_not(tmp_path):
    """Why the edit: with enable_thinking undefined LiteRT-LM's adds an empty think
    block to the generation prompt and the checkpoint's adds nothing -- in exactly
    those cases -- and the edited one renders every case as the checkpoint's."""
    renders = _compose(tmp_path, MINICPM5, LITERT_MINICPM5)
    upstream = single_turn_differences(renders["reference"], renders["candidates"]["canonical"])
    assert {d.case.context.thinking for d in upstream} == {None}
    assert len(upstream) == len([c for c in SINGLE_TURNS if c.context.thinking is None]) * 2
    assert all(d.candidate.text.endswith("<think>\n\n</think>\n\n") for d in upstream)
    edited = models.canonical_template_text(models.MINICPM5_CANONICAL_TEMPLATE)
    renders = _compose(tmp_path, MINICPM5, edited)
    assert single_turn_differences(renders["reference"], renders["candidates"]["canonical"]) == []
    assert renders["unvouched"]["canonical"] == []
    choice = export._choose_template("m", renders, models.MINICPM5_CANONICAL_TEMPLATE)
    assert choice.source == "canonical"


def test_minicpm5s_own_template_needs_more_than_the_block(tmp_path):
    """What the reason says of the checkpoint's template: alone it drops text
    handed as parts; behind the block the tools cases raise and a reply with a
    think block breaks the conversation."""
    alone = _render(MINICPM5, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert alone["text"] is not None and "hi" not in alone["text"]
    renders = _compose(tmp_path, MINICPM5, LITERT_MINICPM5)
    composed = single_turn_differences(renders["reference"], renders["candidates"]["composed"])
    assert composed and all(d.case.context.tools for d in composed)
    assert all(d.candidate.text is None for d in composed)
    breaks, _ = _breaks(BLOCK + MINICPM5)
    assert breaks and all("think" in b.case.reply.name for b in breaks)


def test_litert_lms_qwen35_template_neither_trims_nor_defaults_as_qwen35s_does(tmp_path):
    renders = _compose(tmp_path, QWEN35, LITERT_QWEN35)
    upstream = single_turn_differences(renders["reference"], renders["candidates"]["canonical"])
    undefined = [d for d in upstream if d.case.context.thinking is None]
    defined = [d for d in upstream if d.case.context.thinking is not None]
    assert len(undefined) == len([c for c in SINGLE_TURNS if c.context.thinking is None]) * 2
    assert defined and all(d.case.text != d.case.text.strip() for d in defined)
    edited = models.canonical_template_text(models.QWEN35_CANONICAL_TEMPLATE)
    renders = _compose(tmp_path, QWEN35, edited)
    assert single_turn_differences(renders["reference"], renders["candidates"]["canonical"]) == []
    assert renders["unvouched"]["canonical"] == []
    choice = export._choose_template("m", renders, models.QWEN35_CANONICAL_TEMPLATE)
    assert choice.source == "canonical"


def test_qwen35s_own_template_breaks_a_conversation():
    breaks, refused = _breaks(QWEN35)
    assert refused == [] and breaks
    assert any(b.case.reply.name == "a plain reply" and b.case.turns == 2 for b in breaks)


@pytest.mark.parametrize("family", list(EDITED))
def test_an_edited_template_holds_every_conversation(family):
    name = EDITED[family][0]
    assert _breaks(models.canonical_template_text(name)) == ([], [])


@pytest.mark.parametrize("reply", [" answer ", "answer\n", "\n\nanswer"])
def test_qwen35s_trim_keeps_the_prefix_for_a_reply_with_whitespace(reply):
    """The edit trims an assistant reply, as the checkpoint's own template does: the
    history ending in it and that history with the next message both print it
    trimmed, so the one is still a prefix of the other -- the invariant the
    runtime holds a conversation to from its second message."""
    template = models.canonical_template_text(models.QWEN35_CANONICAL_TEMPLATE)
    part = lambda text: [{"type": "text", "text": text}]  # noqa: E731
    history = [
        {"role": "user", "content": part("hi")},
        {"role": "assistant", "content": part(reply)},
    ]
    before = _render(template, history, False)["text"]
    after = _render(template, [*history, {"role": "user", "content": part("and?")}], True)["text"]
    assert after.startswith(before)
    assert "assistant\nanswer<|im_end|>" in before


def test_minicpm5s_bos_is_the_start_token_the_runtime_prepends(tmp_path):
    """The checkpoint's template prints `{{ bos_token }}`, `<s>`, which training
    tokenises as the BOS. The runtime hands the template an empty `bos_token` and
    prepends the bundle's start token instead, which the exporter writes only when
    the template it is handed prints the BOS (`litert_lm_builder._tokenizer_prepends_bos`).
    With MiniCPM5-1B's own tokens, the edited template renders every case as the
    checkpoint's does, the start token included, and is chosen."""
    edited = models.canonical_template_text(models.MINICPM5_CANONICAL_TEMPLATE)
    renders = _compose(tmp_path, MINICPM5, edited, MINICPM5_SPECIAL_TOKENS)
    assert all(r["text"].startswith("<s>") for r in renders["reference"])
    assert single_turn_differences(renders["reference"], renders["candidates"]["canonical"]) == []
    assert renders["start_tokens"]["canonical"] == "<s>"
    choice = export._choose_template("m", renders, models.MINICPM5_CANONICAL_TEMPLATE)
    assert choice.source == "canonical"


def test_litert_lms_minicpm5_template_would_lose_the_bos(tmp_path):
    """Unedited, LiteRT-LM's template prints no BOS, so the exporter writes no start
    token and the runtime's prompt lacks the `<s>` training saw: every case differs."""
    renders = _compose(tmp_path, MINICPM5, LITERT_MINICPM5, MINICPM5_SPECIAL_TOKENS)
    assert renders["start_tokens"]["canonical"] is None
    differences = single_turn_differences(renders["reference"], renders["candidates"]["canonical"])
    assert len(differences) == len(SINGLE_TURNS) * len(FORMS)


def test_granites_composed_template_gets_no_start_token(tmp_path):
    """granite-4.1-3b's BOS is its EOS, its tokenizer prepends none and its template
    prints none, so no start token is written and none is needed."""
    renders = _compose(tmp_path, GRANITE, GRANITE, GRANITE_SPECIAL_TOKENS)
    assert renders["start_tokens"] == {"composed": None, "canonical": None}
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []


def test_a_tokenizer_that_prepends_its_bos_gets_a_start_token_whatever_the_template(tmp_path):
    """The exporter's first probe: a tokenizer whose own tokenisation begins with the
    BOS (`add_bos_token: true`) gets it as the start token even from a template that
    prints none -- and then the runtime's prompt begins with a BOS training never saw
    through that template, so every case differs and the composition is refused."""
    renders = _compose(tmp_path, QWEN25, None, {"bos_token": "<s>", "eos_token": "</s>"}, True)
    assert renders["start_tokens"] == {"composed": "<s>"}
    differences = single_turn_differences(renders["reference"], renders["candidates"]["composed"])
    assert len(differences) == len(SINGLE_TURNS) * len(FORMS)
    assert export._choose_template("m", renders, None).refusal is not None
