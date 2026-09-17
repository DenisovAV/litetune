"""The tool declarations a run is made against, and the rules the runtime sets.

litetune does not invent a schema here. The runtime's Python API refuses a tool
whose description has no `['function']['name']` before it renders anything, and
the chat template a checkpoint carries reads the same two keys to build its
declaration. So both renderers the measurement compares require the OpenAI
function object, and these tests hold this module to exactly those rules --
no stricter, because the declaration text is produced by the runtime's own
formatter and nothing in this project establishes what else it accepts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litetune.declarations import (
    DeclarationsError,
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
                "properties": {"color": {"type": "string"}},
                "required": ["color"],
            },
        },
    },
    {"type": "function", "function": {"name": "open_app"}},
]


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "declarations.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_the_digest_is_the_one_a_bundle_contract_is_compared_against(tmp_path):
    """`tune` records this digest and `verify` refuses a set that disagrees with
    it, and `bundle` compares its contract against `hash_file` over the same
    bytes. A second way of computing it -- over the parsed value, or over the
    text rather than the bytes -- is a different string for the same file, and
    would turn that comparison into a refusal of a matching set.
    """
    path = _write(tmp_path, WRAPPED)

    parsed, digest = read_declarations(path)

    assert digest == hash_file(path)
    assert digest.startswith("sha256:")
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
