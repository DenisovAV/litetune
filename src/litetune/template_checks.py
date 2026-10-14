"""Two properties a bundle's chat template has to have, and how `convert` checks them.

**Single-turn identity.** For single-turn prompts -- with and without a system
message, the text in one part and in two, `enable_thinking` undefined, true
and false, with no tools and with one -- the template the bundle carries
renders exactly what the checkpoint's own template renders through
transformers. The first case, no system message and no context, is what
training rendered for a `runtime_rendered` run without declarations and what
the float reference renders (`prompt_mode.RENDERING_SOURCE`: one user message,
its content a string, through the tokenizer's `apply_chat_template`); the
others are what the runtime may hand the same template. Whether the cases
stand for what a checkpoint was trained on is `export._canonical_basis`'s to
decide, from the record `tune` leaves beside it or, without one, from whether
its template is the one its family ships.

**The conversation prefix invariant.** Rendering a history that ends in an
assistant reply, with no generation prompt, is a byte prefix of rendering that
history plus the next user message with the generation prompt, in the same
context. LiteRT-LM's `Conversation` renders the whole history again on every
message after the first, with the resolved `enable_thinking`, the tools and
any extra context on both renderings, and requires the new rendering to begin
with the previous one (`Conversation::GetSingleTurnTextFromFullHistory`,
`runtime/conversation/conversation.cc` v0.18.0, L257-L324: the preface and its
tools L259-L261, `enable_thinking` L263-L269, the comparison L316-L321); where
it does not, the send fails with INTERNAL, "The new rendered template string
does not start with the previous rendered template string"
(google-ai-edge/LiteRT-LM#3443). The first message of a conversation takes the
other branch of that function and is not compared -- unless the conversation
was created with `prefill_preface_on_init`, which this does not check -- so a
template that breaks the invariant answers one message and fails the second. A
template LiteRT-LM renders through its single-turn protocol instead
(`single_turn_protocol`) is not held to it. What is checked is a set of
conversations in six contexts, `CONTEXTS`, and a result says so: other
contexts are not covered.

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
minijinja_template.rs`). Neither LiteRT-LM's renderer nor MiniJinja is where
`convert` runs: LiteRT-LM's Python API renders only through a `Conversation`,
which needs an `Engine` built from a model file (`c/conversation.h`,
`c/engine.h`, `python/litert_lm/engine.py` at v0.18.0), and the export
environment has the Jinja2 transformers renders with. `RENDERER_SOURCE` is
that Jinja2 configured as `create_env` configures MiniJinja -- trailing
newlines kept, blocks trimmed, its `tojson`, `lstrip`, `rstrip`,
`raise_exception` and `none`, the same rewrite first, the MiniJinja tests and
filters the rewrite produces, no Python methods on a string, list or mapping,
no `{% break %}`, values printed, joined with `~` and coerced by filters as
MiniJinja does, and its operators, comparisons, slices, loops, namespaces and
macro arguments MiniJinja's (`src/value/ops.rs`, `src/value/mod.rs`,
`src/vm/loop_object.rs`, `src/compiler/codegen.rs` of that crate). It is an
emulation, so it vouches only for what is held to MiniJinja 2.14.0 in the
tests -- the filters, tests and functions it names, on non-string values as
well as strings, and the syntax those tests render -- and to LiteRT-LM's own
goldens for its Qwen 3 template; a template that uses anything else, or syntax
the two read differently (a chained comparison, which MiniJinja reads as
nested ones until 2.20.0, among them), is not checked, never passed
(`unvouched`). The reference side of single-turn identity is what
training rendered, transformers' `apply_chat_template`, run in the export
environment by `export._COMPOSE_SCRIPT`.
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
# (`crate_features` without `loop_controls`, so no `{% break %}`), read in
# `src/filters.rs`, `src/tests.rs` and `src/value/` of that crate.
RENDERER_SOURCE = r"""
import decimal as _decimal
import functools as _functools
import json as _json
import math as _math
import re as _re
import struct as _struct
import time as _time

import jinja2
import jinja2.runtime
import jinja2.utils
from jinja2 import nodes as _nodes
from jinja2.compiler import CodeGenerator as _CodeGenerator
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
    # MiniJinja's arithmetic (`src/value/ops.rs`): integers checked in i128,
    # `//` and integer `%` Euclidean, `/` in f64 (so `1 / 0` is `inf`), a
    # negative integer exponent an error, a string or list repeated only a
    # non-negative number of times, and `-` only for numbers. Jinja2 hands
    # each of these operators to `call_binop`/`call_unop` when it is listed.
    intercepted_binops = frozenset({"+", "-", "*", "/", "//", "%", "**"})
    intercepted_unops = frozenset({"-"})

    def getattr(self, obj, attribute):
        if isinstance(obj, dict):
            return obj[attribute] if attribute in obj else self.undefined(obj=obj, name=attribute)
        if isinstance(obj, _PLAIN):
            return self.undefined(obj=obj, name=attribute)
        if isinstance(obj, jinja2.runtime.LoopContext) and isinstance(
            getattr(obj, "_iterable", None), _Unsized
        ):
            # MiniJinja knows no length for a string or an iterator it loops
            # over (`vm/loop_object.rs`): no `length` or `revindex`, and
            # `last` is false.
            if attribute in ("length", "revindex", "revindex0"):
                return self.undefined(obj=obj, name=attribute)
            if attribute == "last":
                return False
        if isinstance(obj, jinja2.runtime.Macro):
            # `name`, `arguments` and `caller`, and nothing else
            # (`vm/macro_object.rs`).
            if attribute == "arguments":
                return list(obj.arguments)
            if attribute in ("name", "caller"):
                return getattr(obj, attribute)
            return self.undefined(obj=obj, name=attribute)
        return super().getattr(obj, attribute)

    def getitem(self, obj, argument):
        if isinstance(argument, slice):
            return _slice(obj, argument)
        if isinstance(obj, dict):
            return obj[argument] if argument in obj else self.undefined(obj=obj, name=argument)
        # An index is `Value::as_i64` (`get_item_opt`, `src/value/mod.rs`): a
        # float that is an integer is one. An iterator of no known length has
        # no item counted from its end.
        if isinstance(argument, float) and argument.is_integer() and abs(argument) < 2.0**63:
            argument = int(argument)
        if isinstance(obj, _Unsized) and isinstance(argument, int) and argument < 0:
            return self.undefined(obj=obj, name=argument)
        if isinstance(obj, _PLAIN) and not isinstance(argument, (int, slice)):
            return self.undefined(obj=obj, name=argument)
        return super().getitem(obj, argument)

    def call(__self, __context, __obj, *args, **kwargs):  # noqa: N805 -- Jinja2's own signature
        # A macro takes a parameter's default for an argument passed undefined,
        # not only for one not passed (`IsUndefined` in
        # `compile_macro_expression`, `src/compiler/codegen.rs`).
        if isinstance(__obj, jinja2.runtime.Macro):
            missing = jinja2.utils.missing
            args = tuple(missing if isinstance(a, jinja2.Undefined) else a for a in args)
            kwargs = {
                k: missing if isinstance(v, jinja2.Undefined) else v for k, v in kwargs.items()
            }
        return super().call(__context, __obj, *args, **kwargs)

    def call_binop(self, context, operator, left, right):
        return _binop(operator, left, right)

    def call_unop(self, context, operator, arg):
        if isinstance(arg, bool) or not isinstance(arg, (int, float)):
            raise TypeError("invalid operation")
        return -arg if isinstance(arg, float) else _checked(-arg, "-", 0, arg)

    def _parse(self, source, name, filename):
        # Every loop's iterable through `__mj_iter__`: MiniJinja loops over
        # none or undefined as over nothing, and over a string without a length.
        tree = super()._parse(source, name, filename)
        for node in tree.find_all(_nodes.For):
            call = _nodes.Call(_nodes.Name("__mj_iter__", "load"), [node.iter], [], None, None)
            node.iter = call.set_lineno(node.lineno).set_environment(self)
        return tree


class _Iterable(list):
    # What MiniJinja holds as an iterable rather than a sequence
    # (`ObjectRepr::Iterable`): a slice, a sum or a repetition of sequences
    # (`ops::slice`, `ops::add`, `ops::mul`), and what `items` and `range`
    # return. Iterated, indexed and printed as a list; not a `sequence`, and
    # ordered after every map.
    pass


class _Unsized(_Iterable):
    # An iterable of no known length: `split`'s result, a string a loop goes
    # over, and what is sliced or summed from them. Printed `<iterator>`, true
    # even when empty, and `length` refuses it.
    def __bool__(self):
        return True


class _Namespace(dict, jinja2.utils.Namespace):
    # MiniJinja's namespace is a map with its keys in order (a `BTreeMap`,
    # `src/value/namespace_object.rs`): looked up, iterated, printed, compared
    # and false when empty as one. A `Namespace` still, for Jinja2's check on
    # `{% set ns.a = ... %}`, which assigns through `__setitem__`.
    __getattribute__ = object.__getattribute__
    __repr__ = dict.__repr__

    def __init__(self, *args, **kwargs):
        super().__init__(sorted(dict(*args, **kwargs).items()))

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        ordered = sorted(self.items())
        self.clear()
        dict.update(self, ordered)


def _namespace(*args, **kwargs):
    # MiniJinja's `namespace(defaults)`: one map or keyword arguments, not both,
    # and only its string keys.
    if len(args) + bool(kwargs) > 1:
        raise TypeError("too many arguments")
    defaults = args[0] if args else kwargs
    if not isinstance(defaults, dict):
        raise TypeError("expected object or keyword arguments")
    return _Namespace({key: value for key, value in defaults.items() if isinstance(key, str)})


_I128 = (-(2**127), 2**127 - 1)


def _checked(value, operator, left, right):
    if _I128[0] <= value <= _I128[1]:
        return value
    raise OverflowError(
        f"unable to calculate {minijinja_str(left)} {operator} {minijinja_str(right)}"
    )


def _is_int(value):
    return isinstance(value, int)


def _is_number(value):
    return isinstance(value, (int, float))


def _fdiv(a, b):
    # f64 division as Rust does it: by zero, an infinity or NaN.
    if b == 0:
        if a == 0 or _math.isnan(a):
            return _math.nan
        return _math.copysign(_math.inf, a) * _math.copysign(1.0, b)
    return a / b


def _fmod(a, b):
    # Rust's f64 `%`, C's fmod: NaN for a zero divisor or an infinite dividend.
    if b == 0 or _math.isinf(a) or _math.isnan(a) or _math.isnan(b):
        return _math.nan
    return _math.fmod(a, b)


def _times(sequence, count, left, right):
    # `Value::as_usize`: a bool, a non-negative integer, or a float that is
    # one; and a sequence repeated is an iterable, of a length it must know.
    if isinstance(count, float) and count.is_integer() and abs(count) < 2.0**63:
        count = int(count)
    if not (isinstance(count, int) and count >= 0):
        raise TypeError(f"cannot repeat by {minijinja_str(count)}")
    if isinstance(sequence, str):
        return sequence * count
    if isinstance(sequence, _Unsized):
        raise TypeError("cannot repeat unsized iterables")
    return _Iterable(list(sequence) * count)


def _bound(value):
    # A slice bound is `i64::try_from`: a bool, an integer, or a float that is one.
    if isinstance(value, float) and value.is_integer() and abs(value) < 2.0**63:
        return int(value)
    if isinstance(value, int):
        return int(value)
    raise TypeError(f"cannot convert {minijinja_str(value)} to a slice index")


def _backwards(start, stop, step, end):
    # `range_step_backwards`: Python's negative step, but for a stop of 0,
    # which reads as no stop, and for an empty value, whose one index panics.
    if start is None or start >= end:
        first = max(end - 1, 0)
    else:
        first = start if start >= 0 else max(end + start, 0)
    last = 0 if stop is None else (max(end + stop, 0) if stop < 0 else stop)
    if last > first:
        return []
    length = (first + step) // step if last == 0 else (first - last + step - 1) // step
    indices = list(range(first, last - 1, -step))[:length]
    if any(index >= end for index in indices):
        raise IndexError("index out of bounds")
    return indices


def _slice(value, bounds):
    # `ops::slice` (`src/value/ops.rs`): none and undefined slice to an empty
    # sequence; a sequence slices to an iterable, which with a positive step
    # counts a missing or negative bound from a length an unsized one does
    # not have, 0.
    start, stop, step = (
        None if bound is None else _bound(bound)
        for bound in (bounds.start, bounds.stop, bounds.step)
    )
    step = 1 if step is None else step
    if step == 0:
        raise ValueError("cannot slice by step size of 0")
    if value is None or isinstance(value, jinja2.Undefined):
        return []
    if not isinstance(value, (str, list, tuple)):
        raise TypeError(f"value of type {type(value).__name__} cannot be sliced")
    if step > 0:
        end = 0 if isinstance(value, _Unsized) else len(value)
        first = 0 if start is None else start
        if first < 0 or stop is None or stop < 0:
            last = end if stop is None else (max(end + stop, 0) if stop < 0 else stop)
            first = max(end + first, 0) if first < 0 else first
        else:
            last = stop
        items = value[first : max(first, last)][::step]
    else:
        items = [value[index] for index in _backwards(start, stop, -step, len(value))]
    if isinstance(value, str):
        return "".join(items)
    # An unsized slice knows its length only when it takes nothing: `take(0)`.
    unsized = isinstance(value, _Unsized) and step > 0 and max(first, last) > first
    return (_Unsized if unsized else _Iterable)(items)


def _binop(operator, left, right):
    ints = _is_int(left) and _is_int(right)
    numbers = _is_number(left) and _is_number(right)
    if operator == "+":
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            unsized = isinstance(left, _Unsized) or isinstance(right, _Unsized)
            return (_Unsized if unsized else _Iterable)(list(left) + list(right))
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if ints:
            return _checked(left + right, "+", left, right)
        if numbers:
            return float(left) + float(right)
    elif operator == "-":
        if ints:
            return _checked(left - right, "-", left, right)
        if numbers:
            return float(left) - float(right)
    elif operator == "*":
        if isinstance(left, str):
            return _times(left, right, left, right)
        if isinstance(right, str):
            return _times(right, left, left, right)
        if isinstance(left, (list, tuple)):
            return _times(left, right, left, right)
        if isinstance(right, (list, tuple)):
            return _times(right, left, left, right)
        if ints:
            return _checked(left * right, "*", left, right)
        if numbers:
            return float(left) * float(right)
    elif operator == "/":
        if numbers:
            return _fdiv(float(left), float(right))
    elif operator == "//":
        if ints:
            if right == 0:
                raise ZeroDivisionError(f"unable to calculate {left} // {right}")
            quotient = left // right
            if right < 0 and left % right != 0:
                quotient += 1
            return _checked(quotient, "//", left, right)
        if numbers:
            a, b = float(left), float(right)
            quotient = _fdiv(a, b)
            if _math.isfinite(quotient):
                quotient = float(_math.trunc(quotient))
            if _fmod(a, b) < 0:
                quotient = quotient - 1.0 if b > 0 else quotient + 1.0
            return quotient
    elif operator == "%":
        if ints:
            if right == 0:
                raise ZeroDivisionError(f"unable to calculate {left} % {right}")
            return left % abs(right)
        if numbers:
            return _fmod(float(left), float(right))
    elif operator == "**":
        if ints:
            if right < 0 or right > 2**32 - 1 or (abs(left) > 1 and right > 128):
                raise OverflowError(f"unable to calculate {left} ** {right}")
            return _checked(left**right, "**", left, right)
        if numbers:
            a, b = float(left), float(right)
            try:
                return _math.pow(a, b)
            except OverflowError:
                odd = b.is_integer() and int(b) % 2 == 1
                return _math.copysign(_math.inf, a) if odd else _math.inf
            except ValueError:
                if a == 0 and b < 0:
                    odd = b.is_integer() and int(b) % 2 == 1
                    return _math.copysign(_math.inf, a) if odd else _math.inf
                return _math.nan
    raise TypeError(
        f"tried to use {operator} operator on unsupported types {type(left).__name__} and "
        f"{type(right).__name__}"
    )


# The order MiniJinja sorts values of different kinds in (`ValueKind`, and
# `impl Ord for Value`, `src/value/mod.rs`): undefined, none, bool, number,
# string, bytes, sequence, map, iterable, plain object.
def _kind(value):
    if isinstance(value, jinja2.Undefined):
        return 0
    if value is None:
        return 1
    if isinstance(value, bool):
        return 2
    if isinstance(value, (int, float)):
        return 3
    if isinstance(value, str):
        return 4
    if isinstance(value, _Iterable):
        return 8
    if isinstance(value, (list, tuple)):
        return 6
    if isinstance(value, dict):
        return 7
    return 9


def _total_order(value):
    # `f64_total_cmp`: the order Rust's `f64::total_cmp` gives, -0.0 before 0.0.
    bits = _struct.unpack("<q", _struct.pack("<d", value))[0]
    return bits ^ (((bits >> 63) & 0xFFFFFFFFFFFFFFFF) >> 1)


def _order(left, right):
    a, b = _kind(left), _kind(right)
    if a != b:
        return (a > b) - (a < b)
    if a <= 1:
        return 0
    if a in (2, 3):
        if isinstance(left, float) or isinstance(right, float):
            x, y = _total_order(float(left)), _total_order(float(right))
        else:
            x, y = left, right
        return (x > y) - (x < y)
    if a == 4:
        return (left > right) - (left < right)
    if a in (6, 8):
        for x, y in zip(left, right):
            found = _order(x, y)
            if found:
                return found
        return (len(left) > len(right)) - (len(left) < len(right))
    if a == 7:
        return _order(
            [[k, v] for k, v in left.items()], [[k, v] for k, v in right.items()]
        )
    raise TypeError("cannot order these values")


def _as_f64(value):
    # `ops::as_f64` without loss: a float, a bool, an integer a float holds.
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value) if int(float(value)) == value else None
    return None


def _equal(left, right):
    # `impl PartialEq for Value` (`src/value/mod.rs`): none and undefined each
    # equal only themselves, a string only a string, numbers and bools by value
    # (through `ops::coerce`), sequences of every kind item by item -- a tuple
    # is a list there -- maps key by key, and other objects as they print.
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, jinja2.Undefined) or isinstance(right, jinja2.Undefined):
        return isinstance(left, jinja2.Undefined) and isinstance(right, jinja2.Undefined)
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    if _is_number(left) and _is_number(right):
        if isinstance(left, float) or isinstance(right, float):
            a, b = _as_f64(left), _as_f64(right)
            return a is not None and b is not None and a == b
        return int(left) == int(right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(map(_equal, left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return len(left) == len(right) and all(
            key in right and _equal(right[key], item) for key, item in left.items()
        )
    if isinstance(left, _PLAIN) or isinstance(right, _PLAIN):
        return False
    return left is right or minijinja_str(left) == minijinja_str(right)


def _contains(container, value):
    # `ops::contains`: false in undefined, a substring in a string -- of the
    # value as it prints, when it is not a string -- a key of a map, an item of
    # a sequence by `_equal`, nothing in a plain object, an error in anything
    # else.
    if isinstance(container, jinja2.Undefined):
        return False
    if isinstance(container, str):
        return (value if isinstance(value, str) else minijinja_str(value)) in container
    if isinstance(container, _Namespace):
        return isinstance(value, str) and value in container
    if isinstance(container, dict):
        try:
            return value in container
        except TypeError:
            return False
    if isinstance(container, (list, tuple)):
        return any(_equal(item, value) for item in container)
    if isinstance(container, (int, float)) or container is None:
        raise TypeError("cannot perform a containment check on this value")
    return False


def _is_in(value, container):
    # MiniJinja's `in` test: the operator's answer, and false where it errs.
    try:
        return _contains(container, value)
    except TypeError:
        return False


def minijinja_compare(operator, left, right):
    if operator == "in":
        return _contains(right, left)
    if operator == "notin":
        return not _contains(right, left)
    if operator == "eq":
        return _equal(left, right)
    if operator == "ne":
        return not _equal(left, right)
    found = _order(left, right)
    return {"lt": found < 0, "lteq": found <= 0, "gt": found > 0, "gteq": found >= 0}[operator]


class _MiniJinjaCodeGenerator(_CodeGenerator):
    # Nothing but a constant is folded at compile time: Jinja2 folds with
    # Python's operators, and every operator here is MiniJinja's. `~` joins
    # its operands as MiniJinja prints them -- Jinja2's `str_join` would write
    # `True` where MiniJinja writes `true` -- and a comparison of one operator
    # is MiniJinja's (`_equal`, `_order`, `_contains`), and so is a slice.
    def _output_child_to_const(self, node, frame, finalize):
        if not isinstance(node, (_nodes.Const, _nodes.TemplateData)):
            raise _nodes.Impossible()
        return super()._output_child_to_const(node, frame, finalize)

    def visit_Concat(self, node, frame):
        self.write("str_join((")
        for arg in node.nodes:
            self.write("environment.minijinja_str(")
            self.visit(arg, frame)
            self.write("), ")
        self.write("))")

    def visit_Getitem(self, node, frame):
        # Jinja2 writes a slice as Python's, around `getitem`; here it goes
        # through `getitem` too, which slices as MiniJinja does (`_slice`).
        if not isinstance(node.arg, _nodes.Slice):
            return super().visit_Getitem(node, frame)
        self.write("environment.getitem(")
        self.visit(node.node, frame)
        self.write(", slice(")
        for bound in (node.arg.start, node.arg.stop, node.arg.step):
            if bound is None:
                self.write("None")
            else:
                self.visit(bound, frame)
            self.write(", ")
        self.write("))")

    def visit_Compare(self, node, frame):
        operator = node.ops[0].op if len(node.ops) == 1 else None
        if operator not in ("eq", "ne", "lt", "lteq", "gt", "gteq", "in", "notin"):
            return super().visit_Compare(node, frame)
        self.write(f"environment.minijinja_compare({operator!r}, ")
        self.visit(node.expr, frame)
        self.write(", ")
        self.visit(node.ops[0].expr, frame)
        self.write(")")


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def minijinja_str(value):
    # A value as MiniJinja prints it (`impl Display for Value`) and as its
    # string arguments take it (`ArgType for Cow<str>`): `true`, `none`, nothing
    # for an undefined value, a float without an exponent, and a list or map in
    # Rust's debug form.
    if isinstance(value, jinja2.Undefined):
        return ""
    if value is None:
        return "none"
    if isinstance(value, _Unsized):
        return "<iterator>"
    if isinstance(value, jinja2.runtime.Macro):
        return f"<macro {value.name}>"
    if isinstance(value, jinja2.runtime.LoopContext):
        length = "?" if isinstance(value._iterable, _Unsized) else value.length
        return f"<loop {value.index0}/{length}>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if _math.isnan(value):
            return "NaN"
        if _math.isinf(value):
            return "-inf" if value < 0 else "inf"
        text = format(_decimal.Decimal(repr(value)), "f")
        return text if "." in text else text + ".0"
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
    # MiniJinja's `startingwith` and `endingwith`, whose arguments are strings
    # and take any other value as it prints.
    def check(value, other):
        return getattr(minijinja_str(value), test)(minijinja_str(other))

    return check


def _join(value, joiner=None):
    # MiniJinja's `join`: nothing for undefined or none, a string's characters,
    # a map's keys, and an item that is not a string as it prints.
    if isinstance(value, jinja2.Undefined) or value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        raise TypeError(f"cannot join value of type {type(value).__name__}")
    joiner = "" if joiner is None else minijinja_str(joiner)
    return joiner.join(item if isinstance(item, str) else minijinja_str(item) for item in value)


def _split(value, sep=None, maxsplits=None):
    # MiniJinja's `split`, whose value is an `Arc<str>`: a string and nothing
    # else; it returns an iterator of no known length.
    if not isinstance(value, str):
        raise TypeError("value is not a string")
    return _Unsized(value.split(sep) if maxsplits is None else value.split(sep, maxsplits))


def _length(value):
    # MiniJinja's `length`, which refuses undefined and an iterator of no
    # known length, where Jinja2 gives 0 and counts.
    if isinstance(value, (jinja2.Undefined, _Unsized)) or value is None:
        raise TypeError("cannot calculate length of value")
    return len(value)


def _range(*args):
    # MiniJinja's `range`: an iterable of no more than 100000 items.
    values = range(*args)
    if len(values) > 100000:
        raise OverflowError("range has too many elements")
    return _Iterable(values)


def _loop_iterable(value):
    # What a MiniJinja loop iterates (`Value::try_iter`): nothing for none or
    # undefined, a string's characters with no known length, a map's keys,
    # and an error for a number.
    if value is None or isinstance(value, jinja2.Undefined):
        return []
    if isinstance(value, str):
        return _Unsized(value)
    if isinstance(value, (int, float)):
        raise TypeError(f"{type(value).__name__} is not iterable")
    return value


@jinja2.pass_context
def _map(context, value, *args, **kwargs):
    # MiniJinja's `map` returns a sequence; Jinja2's a generator.
    return list(jinja2.filters.do_map(context, value, *args, **kwargs))


def _items(value):
    # MiniJinja's `items`, which takes a map and nothing else, and returns an
    # iterable.
    if not isinstance(value, dict):
        raise TypeError("cannot convert value into pairs")
    return _Iterable([key, item] for key, item in value.items())


def runtime_environment():
    # No `loopcontrols` extension: LiteRT-LM builds MiniJinja without the
    # `loop_controls` feature (`Cargo.toml`; Bazel's `crate_features`), so a
    # `{% break %}` is a syntax error on the device. Not `optimized`: Jinja2's
    # optimizer folds `'a' ~ true` to `aTrue` before `visit_Concat` sees it.
    env = RuntimeEnvironment(
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        finalize=minijinja_str,
        optimized=False,
    )
    env.code_generator_class = _MiniJinjaCodeGenerator
    # Only the globals MiniJinja 2.14.0 has as LiteRT-LM builds it, so that a
    # template naming another one -- Jinja2's `cycler`, `joiner`, `lipsum` --
    # finds it undefined, as on the device.
    for name in set(env.globals) - MINIJINJA_GLOBALS:
        del env.globals[name]
    env.minijinja_str = minijinja_str
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
    env.filters["split"] = _split
    env.filters["trim"] = lambda s, chars=None: minijinja_str(s).strip(chars)
    env.filters["join"] = _join
    env.filters["items"] = _items
    env.filters["length"] = _length
    env.filters["map"] = _map
    env.globals["range"] = _range
    env.globals["namespace"] = _namespace
    env.globals["__mj_iter__"] = _loop_iterable
    env.minijinja_compare = minijinja_compare
    env.filters["upper"] = lambda s: minijinja_str(s).upper()
    env.filters["lower"] = lambda s: minijinja_str(s).lower()
    env.filters["replace"] = lambda s, old, new: minijinja_str(s).replace(
        minijinja_str(old), minijinja_str(new)
    )
    env.tests["eq"] = _equal
    env.tests["ne"] = lambda value, other: not _equal(value, other)
    env.tests["in"] = _is_in
    env.tests["startingwith"] = _strings("startswith")
    env.tests["endingwith"] = _strings("endswith")
    env.tests["sequence"] = lambda value: (
        isinstance(value, (list, tuple)) and not isinstance(value, _Iterable)
    )
    env.tests["number"] = lambda value: (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    # `create_env` replaces MiniJinja's `none` with "is undefined".
    env.tests["none"] = lambda value: isinstance(value, jinja2.Undefined)
    return env


# What this renderer vouches for: the filters, tests and functions it renders
# as MiniJinja does, each held to MiniJinja 2.14.0 by the tests on strings and
# on the other values a template is handed here. A template that uses anything
# else, or a construct below, is not checked -- never passed on a guess.
VOUCHED_FILTERS = frozenset(
    {
        "d", "default", "dictsort", "first", "items", "join", "last", "length", "list",
        "lower", "lstrip", "map", "replace", "rstrip", "split", "string", "tojson", "trim",
        "upper",
    }
)
VOUCHED_TESTS = frozenset(
    {
        "boolean", "defined", "endingwith", "eq", "false", "in", "iterable", "mapping", "ne",
        "none", "number", "sequence", "startingwith", "string", "true", "undefined",
    }
)
# Not `strftime_now`: the runtime formats `now` in UTC and transformers the
# local time, so the two sides of a comparison could print different dates.
VOUCHED_FUNCTIONS = frozenset({"dict", "namespace", "raise_exception", "range"})
_UNVOUCHED_NODES = (
    _nodes.Extends, _nodes.Include, _nodes.Import, _nodes.FromImport, _nodes.Block,
    _nodes.EvalContextModifier,
)
# Names Jinja2 gives a meaning MiniJinja 2.14.0 does not: a macro's `varargs`
# and `kwargs` (MiniJinja refuses an extra argument, "too many arguments"), and
# `self`.
_UNVOUCHED_NAMES = frozenset({"varargs", "kwargs", "self"})
# The globals MiniJinja 2.14.0 has as LiteRT-LM builds it: the crate's own
# (`get_globals`, `src/defaults.rs`; LiteRT-LM uses no `minijinja-contrib`,
# `Cargo.toml`) and the two `create_env` adds (`minijinja_template.rs`).
# Every one of them is a function, which MiniJinja prints by its Rust type
# name (`minijinja::functions::builtins::range`, in the 2.14.0 binding) and
# this renderer as a Python object, so each is vouched for only as the name of
# a call.
MINIJINJA_GLOBALS = frozenset({"range", "dict", "debug", "namespace"})
_GLOBAL_FUNCTIONS = MINIJINJA_GLOBALS | {"raise_exception", "strftime_now"}
# Jinja2's globals MiniJinja does not have, whatever a template does with
# them: called, printed, aliased or assigned.
_JINJA2_ONLY_GLOBALS = frozenset(ImmutableSandboxedEnvironment().globals) - MINIJINJA_GLOBALS
# The name `__mj_iter__` is this renderer's own (`RuntimeEnvironment._parse`).
_RESERVED = "__mj_iter__"
# The escapes MiniJinja 2.14.0 reads in a string literal as Python does
# (`Unescaper::unescape`, `src/utils.rs`): it refuses `\s`, `\a` and other
# escapes Python keeps or reads, reads `\/` as `/`, and joins a surrogate pair.
_SAME_ESCAPES = _re.compile(
    r"\\(?:[nrtbf'\"\\]|x[0-9A-Fa-f]{2}|u(?![dD][89a-fA-F])[0-9A-Fa-f]{4})"
)


def _constant(node):
    # Whether MiniJinja folds `node` at compile time (`Expr::as_const`,
    # `src/compiler/ast.rs`).
    if isinstance(node, _nodes.Const):
        return True
    if isinstance(node, (_nodes.List, _nodes.Tuple)):
        return all(_constant(item) for item in node.items)
    if isinstance(node, _nodes.Dict):
        return all(_constant(pair.key) and _constant(pair.value) for pair in node.items)
    if isinstance(node, (_nodes.Neg, _nodes.Not)):
        return _constant(node.node)
    if isinstance(node, _nodes.BinExpr):
        return _constant(node.left) and _constant(node.right)
    if isinstance(node, _nodes.Compare):
        return _constant(node.expr) and all(_constant(op.expr) for op in node.ops)
    if isinstance(node, _nodes.Concat):
        return all(_constant(item) for item in node.nodes)
    return False


def _recursive_loop_calls(tree):
    # The `loop(...)` calls that re-enter the loop they are in: in the body of
    # a `recursive` loop, outside any loop, macro or call block within it.
    found = set()

    def walk(node):
        for child in node.iter_child_nodes():
            if isinstance(child, (_nodes.For, _nodes.Macro, _nodes.CallBlock)):
                continue
            if (
                isinstance(child, _nodes.Call)
                and isinstance(child.node, _nodes.Name)
                and child.node.name == "loop"
            ):
                found.add(id(child))
            walk(child)

    for loop in tree.find_all(_nodes.For):
        if loop.recursive:
            for statement in loop.body:
                if not isinstance(statement, (_nodes.For, _nodes.Macro, _nodes.CallBlock)):
                    walk(statement)
    return found


def _lexical_gaps(source):
    # What MiniJinja 2.14.0 reads otherwise than Jinja2 before any parsing: a
    # dotted integer lookup (`x.0`, added in MiniJinja 2.18), and a string
    # literal with an escape the two read differently.
    found = set()
    previous = None
    for _, token, value in runtime_environment().lexer.tokeniter(source, None):
        if token == "whitespace":
            continue
        if previous == "." and token in ("integer", "float"):
            found.add("a dotted integer lookup (`x.0`), which MiniJinja 2.14 does not read")
        if token == "string":
            body = value[1:-1]
            if _SAME_ESCAPES.sub("", body).count("\\"):
                found.add(
                    f"the string literal {value}, whose escapes MiniJinja 2.14 reads otherwise"
                )
        previous = value if token == "operator" else token
    return found


def unvouched(template):
    # What in `template` this renderer does not render as LiteRT-LM does;
    # empty when nothing.
    try:
        tree = runtime_environment().parse(edit_for_minijinja(template))
    except jinja2.exceptions.TemplateSyntaxError as exc:
        return [f"Jinja2 cannot parse it ({exc})"]
    found = set()
    macros = {node.name for node in tree.find_all(_nodes.Macro)}
    for node in tree.find_all(_nodes.Filter):
        if node.name not in VOUCHED_FILTERS:
            found.add(f"the filter `{node.name}`")
    for node in tree.find_all(_nodes.Test):
        if node.name not in VOUCHED_TESTS:
            found.add(f"the test `{node.name}`")
    recursive_calls = _recursive_loop_calls(tree)
    callees = set()
    for node in tree.find_all(_nodes.Call):
        if not isinstance(node.node, _nodes.Name):
            continue
        callees.add(id(node.node))
        if id(node) not in recursive_calls and node.node.name not in (
            VOUCHED_FUNCTIONS | macros | {"caller", _RESERVED}
        ):
            found.add(f"the function `{node.node.name}`")
    for node in tree.find_all(_nodes.Name):
        if node.name in _JINJA2_ONLY_GLOBALS:
            found.add(f"the name `{node.name}`, a Jinja2 global MiniJinja 2.14 does not have")
        elif (
            node.name in _GLOBAL_FUNCTIONS
            and node.ctx == "load"
            and id(node) not in callees
        ):
            found.add(f"the function `{node.name}` as a value, which the two print otherwise")
    if _RESERVED in template:
        found.add(f"the name `{_RESERVED}`, which this renderer reserves")
    for node in tree.find_all(_UNVOUCHED_NODES):
        found.add(f"a `{type(node).__name__}` tag")
    for node in tree.find_all(_nodes.Compare):
        # MiniJinja 2.14 reads `a < b < c` as `(a < b) < c`: `parse_compare`
        # (`src/compiler/parser.rs`) nests each comparison in the next. Python's
        # chaining arrived in 2.20.0 ("Added support for chained comparisons",
        # MiniJinja's CHANGELOG.md).
        if len(node.ops) > 1:
            found.add("a chained comparison, which MiniJinja 2.14 reads as nested ones")
    for node in tree.find_all((_nodes.Macro, _nodes.CallBlock)):
        # MiniJinja assigns parameters last to first, each default evaluated
        # as its parameter is (`compile_macro_expression`): a default sees the
        # parameters after it, where Jinja2's sees those before.
        parameters = {arg.name for arg in node.args}
        if any(
            name.name in parameters
            for default in node.defaults
            for name in [default, *default.find_all(_nodes.Name)]
            if isinstance(name, _nodes.Name)
        ):
            found.add(
                "a macro default that names a parameter, which MiniJinja 2.14 binds otherwise"
            )
    for node in tree.find_all(_nodes.Pos):
        found.add("a unary `+`, which MiniJinja 2.14 cannot parse")
    for node in tree.find_all(_nodes.Name):
        if node.name in _UNVOUCHED_NAMES:
            found.add(f"the name `{node.name}`, which MiniJinja 2.14 does not read as Jinja2")
    for node in tree.find_all(_nodes.And):
        # Folded at compile time to `false` when either side is falsy, where at
        # run time, and in Jinja2, `and` gives the falsy operand itself.
        if _constant(node):
            found.add(
                "an `and` of constants, which MiniJinja 2.14 folds to false when either is falsy"
            )
    for node in tree.find_all(_nodes.Const):
        # A literal that does not fit 64 bits is a `u128` there
        # (`Token::Int128`, `src/compiler/lexer.rs`), which `ops::neg` and
        # `ops::coerce` read otherwise, and one past 128 bits does not parse.
        if type(node.value) is int and node.value >= 2**64:
            found.add(
                f"the integer literal {node.value}, which MiniJinja 2.14 holds as a 128-bit one"
            )
    found |= _lexical_gaps(edit_for_minijinja(template))
    # `detect_capabilities` (`minijinja_template.rs`): such a template is
    # rendered through the single-turn protocol, with variables of its own.
    if "is_appending_to_prefill" in template:
        found.add("`is_appending_to_prefill`, which sends LiteRT-LM down its single-turn protocol")
    return sorted(found)


@_functools.lru_cache(maxsize=16)
def _compiled(template):
    return runtime_environment().from_string(edit_for_minijinja(template))


def runtime_render(template, messages, add_generation_prompt, **context):
    # {"text": ...}, or {"error": ...} for a template that raised or that
    # Jinja2 could not compile.
    values = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "",
        "eos_token": "",
        "now": int(_time.time()),
    }
    values.update(context)
    try:
        text = _compiled(template).render(**values)
    except Exception as exc:  # the template's own failure is a finding, not the script's
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"text": text}
"""


def _parts(text: str, form: str = "one_part") -> list[dict[str, Any]]:
    if form == "two_parts" and len(text) > 1:
        middle = len(text) // 2
        return [{"type": "text", "text": text[:middle]}, {"type": "text", "text": text[middle:]}]
    if form not in FORMS:
        raise ValueError(f"{form!r} is not one of {FORMS}")
    return [{"type": "text", "text": text}]


# One function declaration, as an application hands LiteRT-LM its tools and the
# runtime hands them to the template: `create_conversation(tools=...)` passes
# each tool's description as JSON (`python/litert_lm/engine.py`), and
# `FillPrefaceForPromptTemplateInput` (`runtime/conversation/prompt_utils.cc`)
# hands it on through the model's `FormatTools`, which returns it unchanged
# (`model_data_processor.h`) for every processor but FunctionGemma's and Gemma
# 3's, which rewrite it (`function_gemma_data_processor.cc`,
# `gemma3_data_processor.cc`) in a way this does not reproduce. transformers'
# `apply_chat_template` takes the same list as `tools=`.
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "block_card",
            "description": "Block a card that was lost or stolen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_digits": {
                        "type": "string",
                        "description": "The last four digits of the card.",
                    }
                },
                "required": ["last_digits"],
            },
        },
    },
)


@dataclass(frozen=True)
class Context:
    """What LiteRT-LM adds to a template's input besides the messages.

    `thinking` is `enable_thinking`, which `Conversation` sets from a thinking
    config only when one is given (`ResolveThinkingConfig`, then
    `GetSingleTurnTextFromFullHistory` and `GetSingleTurnTextFromSingleTurnTemplate`,
    `runtime/conversation/conversation.cc` v0.18.0, L100-L109, L241-L244 and
    L263-L269): None is the default, no key at all. `tools` is `TOOLS`, which
    reaches the template only when the conversation declares tools
    (`minijinja_template.rs` leaves a null out).
    """

    thinking: bool | None = None
    tools: bool = False

    @property
    def name(self) -> str:
        thinking = "undefined" if self.thinking is None else str(self.thinking).lower()
        return f"enable_thinking {thinking}, {'one tool' if self.tools else 'no tools'}"

    def values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if self.thinking is not None:
            values["enable_thinking"] = self.thinking
        if self.tools:
            values["tools"] = [dict(tool) for tool in TOOLS]
        return values


CONTEXTS: tuple[Context, ...] = tuple(
    Context(thinking, tools) for tools in (False, True) for thinking in (None, True, False)
)


@dataclass(frozen=True)
class SingleTurn:
    """One single-turn prompt: a user message, after a system message or not, in a context."""

    text: str
    system: str | None = None
    context: Context = Context()

    @property
    def name(self) -> str:
        where = "after a system message" if self.system else "alone"
        return f"the user message {self.text!r} {where}, {self.context.name}"

    def as_strings(self) -> list[dict[str, Any]]:
        """As training renders it: each content a string."""
        system = [{"role": "system", "content": self.system}] if self.system else []
        return [*system, {"role": "user", "content": self.text}]

    def as_parts(self, form: str) -> list[dict[str, Any]]:
        """As LiteRT-LM v0.18.0 hands it to the bundle's template."""
        system = [{"role": "system", "content": _parts(self.system, form)}] if self.system else []
        return [*system, {"role": "user", "content": _parts(self.text, form)}]


# The first is the probe: no system message and the default context, which is
# what `prompt_mode.RENDERING_SOURCE` renders for a run without declarations.
SINGLE_TURNS: tuple[SingleTurn, ...] = tuple(
    SingleTurn(text, system, context)
    for context in CONTEXTS
    for text in SINGLE_TURN_TEXTS
    for system in (None, SYSTEM_TEXT)
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

    def first(self) -> list[dict[str, Any]]:
        """The conversation's first message, which LiteRT-LM renders without comparing."""
        return self.history()[: 2 if self.system else 1]

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
        {
            "strings": case.as_strings(),
            "context": case.context.values(),
            **{form: case.as_parts(form) for form in FORMS},
        }
        for case in SINGLE_TURNS
    ]


def conversation_spec() -> dict[str, Any]:
    """The conversations and contexts as a script in the export environment reads them."""
    return {
        "conversations": [
            {"first": case.first(), "history": case.history(), "following": case.following()}
            for case in CONVERSATIONS
        ],
        "contexts": [context.values() for context in CONTEXTS],
    }


SINGLE_TURN_PROTOCOL_MARKER = "is_appending_to_prefill"


def single_turn_protocol(template: str) -> bool:
    """Whether LiteRT-LM renders `template` through its single-turn protocol.

    The runtime decides by looking for the variable's name in the source
    (`detect_capabilities`, `runtime/components/rust/minijinja_template.rs`
    v0.18.0, L69-L73, on the source `EditTemplateForMinijinja` returns, which
    leaves the name alone), and then renders every message with
    `GetSingleTurnTextFromSingleTurnTemplate` instead of comparing full-history
    renderings (`Conversation::GetSingleTurnText`, `conversation.cc` L326-L340).
    Neither property is checked for such a template: both model the other path.
    """
    return SINGLE_TURN_PROTOCOL_MARKER in template


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


def unvouched(entries: object) -> list[str]:
    """What a script says its renderer does not vouch for in one template.

    Raises ValueError for anything but a list of strings.
    """
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise ValueError(f"a list of unvouched constructs this litetune cannot read: {entries!r}")
    return list(entries)


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
    """One conversation, in one context, whose rendering does not grow by appending."""

    case: Conversation
    context: Context
    previous: Rendering
    following: Rendering

    def describe(self) -> str:
        where = f"{self.case.name}, {self.context.name}"
        for side, rendering in (("ending in the reply", self.previous), ("", self.following)):
            if rendering.text is None:
                what = f"the history {side}" if side else "the history and the next message"
                return f"{where}: the template raises on {what} ({rendering.error})"
        previous, following = self.previous.text or "", self.following.text or ""
        at = next(
            (i for i, (a, b) in enumerate(zip(previous, following, strict=False)) if a != b),
            min(len(previous), len(following)),
        )
        start = max(0, at - 40)
        return (
            f"{where}: from character {at} the history ending in the reply renders "
            f"{quoted(previous[start : at + 40])} and with the next message "
            f"{quoted(following[start : at + 40])}"
        )


@dataclass(frozen=True)
class Refused:
    """A conversation whose first message the template refuses in one context.

    Not a break: LiteRT-LM renders that first message too, and fails on it, so
    the template does not take that input at all -- in a conversation or out
    of one. Recorded, and the case is not counted either way.
    """

    case: Conversation
    context: Context
    first: Rendering

    def describe(self) -> str:
        return f"{self.case.name}, {self.context.name}: the first message {self.first.did()}"


def conversation_breaks(entries: object) -> tuple[list[Break], list[Refused]]:
    """The `CONVERSATIONS` x `CONTEXTS` cases that break the prefix invariant, and those
    whose first message the template refuses.

    `entries` is the script's list, one per conversation, of lists, one per
    context, of `{"first", "previous", "next"}` renderings. A history the
    template raises on, after a first message it rendered, is a break: the
    runtime reports it as INTERNAL, "Failed to apply template"
    (`PromptTemplate::Apply`). Raises ValueError for a shape this cannot read.
    """
    breaks, refused = [], []
    cases = _readings(entries, len(CONVERSATIONS), "the conversations")
    for case, per_context in zip(CONVERSATIONS, cases, strict=True):
        contexts = _readings(per_context, len(CONTEXTS), f"the contexts of {case.name}")
        for context, entry in zip(CONTEXTS, contexts, strict=True):
            if not isinstance(entry, dict):
                raise ValueError(f"the rendering of {case.name} is {entry!r}")
            first = Rendering.read(entry.get("first"))
            previous = Rendering.read(entry.get("previous"))
            following = Rendering.read(entry.get("next"))
            if first.text is None:
                refused.append(Refused(case, context, first))
            elif previous.text is None or following.text is None:
                breaks.append(Break(case, context, previous, following))
            elif not following.text.startswith(previous.text):
                breaks.append(Break(case, context, previous, following))
    return breaks, refused


_CHECKED = (
    f"{len(CONVERSATIONS)} conversations -- a plain reply, one carrying a think block and one "
    "whose think block a thought channel took, over two and three turns, with and without a "
    f"system message -- in {len(CONTEXTS)} contexts: enable_thinking undefined, true and false, "
    "each with no tools and with one tool"
)


@dataclass(frozen=True)
class ConversationCheck:
    """Whether one bundled template holds a LiteRT-LM conversation, in the contexts checked.

    `holds` is None when it was not checked, and `detail` says why;
    `applicable` is False for a template LiteRT-LM renders through its
    single-turn protocol, where the invariant checked here is not the one the
    runtime applies.
    """

    holds: bool | None
    detail: str
    template_sha256: str | None = None
    breaks: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    applicable: bool = True

    @classmethod
    def of(
        cls,
        template: str,
        breaks: Sequence[Break],
        refused: Sequence[Refused],
        renderer: str,
    ) -> ConversationCheck:
        digest = _sha256(template)
        total = len(CONVERSATIONS) * len(CONTEXTS)
        said_refused = (
            f"; in {len(refused)} of them the template refuses the conversation's first message, "
            "which says nothing about the second, and those were not counted"
            if refused
            else ""
        )
        if len(refused) == total:
            return cls(
                holds=None,
                detail=(
                    f"the template refuses the first message of every one of the {total} cases "
                    f"checked, so nothing was compared, rendered by {renderer}"
                ),
                template_sha256=digest,
                refused=tuple(r.describe() for r in refused),
            )
        if breaks:
            return cls(
                holds=False,
                detail=(
                    f"{len(breaks)} of the {total} cases break it -- {_CHECKED} -- rendered by "
                    f"{renderer}{said_refused}"
                ),
                template_sha256=digest,
                breaks=tuple(b.describe() for b in breaks),
                refused=tuple(r.describe() for r in refused),
            )
        return cls(
            holds=True,
            detail=(
                f"each of {total} cases renders as a prefix of the next -- {_CHECKED} -- by "
                f"{renderer}{said_refused}. Other contexts were not checked"
            ),
            template_sha256=digest,
            refused=tuple(r.describe() for r in refused),
        )

    @classmethod
    def not_checked(cls, reason: str, template: str | None = None) -> ConversationCheck:
        return cls(None, reason, _sha256(template) if template is not None else None)

    @classmethod
    def not_applicable(cls, template: str) -> ConversationCheck:
        return cls(
            None,
            (
                f"the template uses `{SINGLE_TURN_PROTOCOL_MARKER}`, so LiteRT-LM renders each "
                "message through its single-turn protocol "
                "(GetSingleTurnTextFromSingleTurnTemplate, conversation.cc) and never compares "
                "full-history renderings; the invariant "
                "checked here does not apply, and that protocol was not checked"
            ),
            _sha256(template),
            applicable=False,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "holds": self.holds,
            "applicable": self.applicable,
            "detail": self.detail,
            "template_sha256": self.template_sha256,
            "contexts": [context.name for context in CONTEXTS],
            "breaks": list(self.breaks),
            "refused": list(self.refused),
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def not_applicable_limitation(recipes: Sequence[str], reason: str) -> str:
    """What a report says about bundles LiteRT-LM renders through its single-turn protocol."""
    return f"{', '.join(recipes)}: the conversation check does not apply to this bundle: {reason}"
