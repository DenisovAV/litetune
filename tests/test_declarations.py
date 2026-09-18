"""The tool declarations a run is made against, and the rules the runtime sets.

litetune does not invent a schema here. The runtime's Python API refuses a tool
whose description has no `['function']['name']` before it renders anything, and
the chat template a checkpoint carries reads the same two keys to build its
declaration. So both renderers the measurement compares require the OpenAI
function object, and these tests hold this module to those rules.

Beyond them it holds the module to one more thing, measured rather than
preferred: the two renderers disagree about key order and about four shapes, so
the file is sorted and those shapes are refused. Each refusal below names the
difference it prevents, and every one of them was reproduced against a real
bundle and the published chat template on 2026-09-17.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litetune.declarations import (
    DeclarationsError,
    canonical_text,
    content_digest,
    entry_count,
    read_declarations,
    tool_names,
)
from litetune.storage import hash_file

WRAPPED = [
    {
        "type": "function",
        "function": {
            "name": "change_background_color",
            "description": "Changes the app background colour",
            "parameters": {
                "type": "object",
                "properties": {"color": {"type": "string", "description": "The colour name"}},
                "required": ["color"],
            },
        },
    },
    {"type": "function", "function": {"name": "open_app", "description": "Opens an app"}},
]


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "declarations.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_the_digest_names_the_tool_list_not_the_file(tmp_path):
    """`tune` records this digest, `verify` refuses a set that disagrees with it,
    and a bundle's contract carries it. Over the file's bytes it made a bundle
    refuse the declarations it had itself shipped: a `runtime_rendered` bundle
    writes them in the order the model learned, and the user's file was in
    another. So the digest is over `canonical_text`, and the text a bundle ships
    hashes to it byte for byte.
    """
    path = _write(tmp_path, WRAPPED)
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(WRAPPED, indent=4), encoding="utf-8")
    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(list(reversed(WRAPPED))), encoding="utf-8")

    parsed, digest = read_declarations(path)

    assert digest == content_digest(parsed)
    assert digest.startswith("sha256:")
    shipped = tmp_path / "shipped.json"
    shipped.write_text(canonical_text(parsed), encoding="utf-8")
    assert hash_file(shipped) == digest
    assert read_declarations(shipped)[1] == digest
    assert read_declarations(pretty)[1] == digest
    # Another tool order is another list: the order of the tools is the prompt's.
    assert read_declarations(reordered)[1] != digest
    assert entry_count(parsed) == 2


def test_the_names_are_the_ones_a_call_has_to_use(tmp_path):
    parsed, _ = read_declarations(_write(tmp_path, WRAPPED))

    assert tool_names(parsed) == frozenset({"change_background_color", "open_app"})


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"function": {"name": "open_app"}}, "the runtime takes a list of tools"),
        (["open_app"], "tool 0 is a str, not an object"),
        ([{"name": "open_app"}], "tool 0 has no ['function']['name'] string"),
        ([{"type": "function", "function": {}}], "tool 0 has no ['function']['name'] string"),
        ([{"type": "function", "function": {"name": 7}}], "tool 0 has no ['function']['name']"),
    ],
)
def test_a_file_the_runtime_would_refuse_is_refused_here(tmp_path, payload, expected):
    """Refused at the stage that reads the file rather than at conversation time.

    The runtime raises `interfaces.Tool description must contain
    ['function']['name']` when it is handed one of these, which would surface
    minutes into a run, after an environment was provisioned and a model loaded,
    for a fact the file carried all along.
    """
    with pytest.raises(DeclarationsError) as caught:
        read_declarations(_write(tmp_path, payload))

    assert expected in str(caught.value)


def test_the_index_of_the_offending_tool_is_named(tmp_path):
    """A list of twenty tools with one bad entry is the case this is for."""
    payload = [{"type": "function", "function": {"name": "ok"}}, {"name": "broken"}]

    with pytest.raises(DeclarationsError, match="tool 1"):
        read_declarations(_write(tmp_path, payload))


def test_a_file_that_is_not_json_says_so(tmp_path):
    with pytest.raises(DeclarationsError, match="not valid JSON"):
        read_declarations(_write(tmp_path, "{not json"))


def test_a_file_that_is_not_there_says_so(tmp_path):
    with pytest.raises(DeclarationsError, match="could not be read"):
        read_declarations(tmp_path / "absent.json")


def test_a_shape_with_no_entry_count_answers_none():
    """`entry_count` is `bundle`'s question, kept here so a stage reporting on
    declarations does not invent a second convention. It answers `None` rather
    than guessing for a shape that has no count -- which `read_declarations`
    refuses anyway, so this is about the helper, not about the file."""
    assert entry_count("a string") is None
    assert entry_count([]) == 0


def test_every_mapping_is_ordered_before_either_renderer_sees_it(tmp_path):
    """The runtime's formatter prints the key order it is given and the reference
    template sorts, so the file is sorted once here. Measured on 2026-09-17: this
    is the whole of the ordering difference -- seven shapes, agreement on all
    seven, including two properties declared in reverse alphabetical order.
    """
    payload = [
        {
            "function": {
                "parameters": {
                    "required": ["zebra"],
                    "properties": {
                        "zebra": {"type": "string", "description": "z"},
                        "alpha": {"type": "integer", "description": "a"},
                    },
                    "type": "object",
                },
                "name": "t",
                "description": "d",
            },
            "type": "function",
        }
    ]

    parsed, digest = read_declarations(_write(tmp_path, payload))

    function = parsed[0]["function"]
    assert list(parsed[0]) == ["function", "type"]
    assert list(function) == ["description", "name", "parameters"]
    assert list(function["parameters"]) == ["properties", "required", "type"]
    assert list(function["parameters"]["properties"]) == ["alpha", "zebra"]
    assert list(function["parameters"]["properties"]["alpha"]) == ["description", "type"]
    # The same list written in another key order is the same list.
    assert digest == content_digest(parsed)


@pytest.mark.parametrize(
    "parameters, expected",
    [
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string", "description": "d", "nullable": True}},
            },
            "carries ['nullable']",
        ),
        ({"type": "object", "properties": {"x": {"type": "string"}}}, "no description string"),
        (
            {"type": "object", "properties": {"type": {"type": "string", "description": "d"}}},
            "named after one of the words the reference template reserves",
        ),
        (
            {
                "type": "object",
                "description": "d",
                "properties": {"x": {"type": "string", "description": "d"}},
            },
            "carries ['description']",
        ),
        ({"type": "object", "properties": {}}, "empty properties"),
        (
            {
                "type": "object",
                "properties": {"x": {"type": "string", "description": "d"}},
                "required": [],
            },
            "empty required",
        ),
        ({}, "is empty"),
        (
            {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "d", "enum": [1, 2]}},
            },
            "carries ['enum']",
        ),
        (
            {"type": "object", "properties": {"o": {"type": "object", "description": "d"}}},
            "object with no properties",
        ),
        (
            {"type": "object", "properties": {"x": {"type": "String", "description": "d"}}},
            "has type 'String'",
        ),
        (
            {
                "type": "object",
                "properties": {"xs": {"type": "array", "description": "d", "items": "string"}},
            },
            "not a non-empty object",
        ),
        (
            {"type": "Object", "properties": {"x": {"type": "string", "description": "d"}}},
            "has type 'Object'",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "Name": {"type": "string", "description": "d"},
                    "name": {"type": "string", "description": "d"},
                },
            },
            "equal but for case",
        ),
        (
            {"type": "object", "properties": {"caf\u00e9": {"type": "string", "description": "d"}}},
            "not a name the runtime's call parser reads",
        ),
        (
            {"type": "object", "properties": {"two words": {"type": "string", "description": "d"}}},
            "not a name the runtime's call parser reads",
        ),
    ],
)
def test_a_shape_the_two_renderers_render_differently_is_refused(tmp_path, parameters, expected):
    """One measured difference per case; none of these is a schema preference.

    The reference template emits, per property, `description`, then `enum` for a
    string, `properties`/`required` for an object, `items` for an array, then
    `type` -- and nothing else in any branch, while the runtime's formatter
    prints every key it is handed. So each shape here renders two ways, and the
    run is refused at the file rather than at a token position minutes later.
    """
    payload = [
        {
            "type": "function",
            "function": {"name": "t", "description": "d", "parameters": parameters},
        }
    ]

    with pytest.raises(DeclarationsError) as caught:
        read_declarations(_write(tmp_path, payload))

    assert expected in str(caught.value)


def test_a_tool_named_outside_the_runtimes_grammar_is_refused(tmp_path):
    """The runtime's lexer reads an identifier as `[a-zA-Z_][a-zA-Z0-9_.-]*`
    (AntlrFcLexer.g4, v0.16.1). A tool named otherwise can be declared, but no
    call to it can ever be parsed back."""
    payload = [{"type": "function", "function": {"name": "caf\u00e9", "description": "d"}}]

    with pytest.raises(DeclarationsError, match="not a name the runtime's call parser reads"):
        read_declarations(_write(tmp_path, payload))


def test_the_order_is_the_templates_dictsort_which_ignores_case(tmp_path):
    """Found in review, checked with jinja2 3.1.6 against functiongemma-270m-it's
    template: `dictsort` puts `body` before `URL`, where a plain `sorted` puts
    `URL` first -- and the runtime prints whichever order it is handed."""
    payload = [
        {
            "type": "function",
            "function": {
                "name": "send",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "URL": {"type": "string", "description": "u"},
                        "body": {"type": "string", "description": "b"},
                    },
                },
            },
        }
    ]

    parsed, _ = read_declarations(_write(tmp_path, payload))

    assert list(parsed[0]["function"]["parameters"]["properties"]) == ["body", "URL"]


def test_a_tool_with_no_description_is_refused(tmp_path):
    """The template renders a tool's description unconditionally; the formatter
    omits the key. So a tool without one is two different declarations."""
    payload = [{"type": "function", "function": {"name": "t"}}]

    with pytest.raises(DeclarationsError, match="no description string"):
        read_declarations(_write(tmp_path, payload))


def test_a_key_the_reference_template_never_reads_is_refused(tmp_path):
    payload = [{"type": "function", "function": {"name": "t", "description": "d", "strict": True}}]

    with pytest.raises(DeclarationsError, match=r"carries \['strict'\]"):
        read_declarations(_write(tmp_path, payload))


def test_the_subset_is_not_tighter_than_the_two_renderers_agree_on(tmp_path):
    """The guard against reading the rules as "keep it simple": every shape the
    two renderers do render identically has to pass, including the nested ones.
    """
    payload = [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "description": "m", "enum": ["on", "off"]},
                        "who": {
                            "type": "object",
                            "description": "w",
                            "properties": {"first": {"type": "string", "description": "f"}},
                            "required": ["first"],
                        },
                        "rows": {
                            "type": "array",
                            "description": "r",
                            "items": {
                                "type": "object",
                                "properties": {"k": {"type": "string", "description": "k"}},
                            },
                        },
                        "count": {"type": "integer", "description": "c"},
                        "flag": {"type": "boolean", "description": "f"},
                    },
                    "required": ["mode"],
                },
            },
        }
    ]

    parsed, _ = read_declarations(_write(tmp_path, payload))

    assert tool_names(parsed) == frozenset({"t"})


# google/mobile-actions, the one real dataset this project trains FunctionGemma
# from, verbatim: its declarations write types in capitals and give a tool with
# no arguments an empty `properties`.
MOBILE_ACTIONS_CALENDAR = {
    "function": {
        "name": "create_calendar_event",
        "description": "Creates a new calendar event.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING", "description": "The title of the event."},
                "datetime": {
                    "type": "STRING",
                    "description": "The date and time of the event in the format "
                    "YYYY-MM-DDTHH:MM:SS.",
                },
            },
            "required": ["title", "datetime"],
        },
    }
}


def test_a_type_name_in_capitals_is_accepted(tmp_path):
    """Neither renderer changes a capitalised name, so it renders one way.

    Measured on this exact declaration against a bundle and the published
    template: identical. The first version of the type rule refused it, and with
    it every tool in mobile-actions that takes an argument.
    """
    payload = [
        MOBILE_ACTIONS_CALENDAR,
        {
            "function": {
                "name": "tag",
                "description": "d",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "xs": {"type": "ARRAY", "description": "x", "items": {"type": "STRING"}}
                    },
                },
            }
        },
    ]

    parsed, _ = read_declarations(_write(tmp_path, payload))

    assert tool_names(parsed) == frozenset({"create_calendar_event", "tag"})


@pytest.mark.parametrize("spelling", ["String", "sTRING", "text"])
def test_any_other_spelling_of_a_type_is_still_refused(tmp_path, spelling):
    """The runtime uppercases only the lowercase names; the template uppercases
    anything. So `String` becomes `STRING` on one side and stays `String` on the
    other -- measured."""
    payload = [
        {
            "function": {
                "name": "t",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": spelling, "description": "d"}},
                },
            }
        }
    ]

    with pytest.raises(DeclarationsError, match=f"has type '{spelling}'"):
        read_declarations(_write(tmp_path, payload))


def test_an_empty_properties_says_to_drop_it_in_the_application_too(tmp_path):
    """mobile-actions writes `"properties": {}` for a tool with no arguments.

    That one really does render two ways -- the runtime prints `properties:{}`
    and the template omits it, measured -- and dropping the key makes them agree,
    also measured. But the runtime at serving time renders whatever the
    application sends, so a key dropped only in this file comes back.
    """
    no_args = {
        "function": {
            "name": "turn_off_flashlight",
            "description": "Turns the flashlight off.",
            "parameters": {"type": "OBJECT", "properties": {}},
        }
    }

    with pytest.raises(DeclarationsError) as caught:
        read_declarations(_write(tmp_path, [no_args]))
    assert "declarations your application sends" in str(caught.value)

    del no_args["function"]["parameters"]["properties"]
    parsed, _ = read_declarations(_write(tmp_path, [no_args]))
    assert tool_names(parsed) == frozenset({"turn_off_flashlight"})
