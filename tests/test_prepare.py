"""Dataset ingestion, with no tokenizer, no network and no subprocess.

`FakeTokenCounter` satisfies `prepare.TokenCounter` structurally and inherits
nothing -- which is why that interface is a Protocol. It counts whitespace
tokens, so an over-length row is written by making a prompt long rather than by
loading a real tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from stage_fakes import spec_mapping

from litetune.checks import Outcome
from litetune.declarations import read_declarations
from litetune.events import EventStream
from litetune.metrics import Proportion, ToolCall, Unavailable, parse_call, runtime_calls
from litetune.models import PROVENANCE_NAME
from litetune.prepare import (
    ASSUMED_WIRE_FORMAT,
    HIGH_CARDINALITY_SHARE,
    LENGTH_CHECK,
    MIN_HELDOUT_EXAMPLES,
    SCOREABILITY_CHECK,
    SPLIT_CHECK,
    LengthStats,
    PrepareError,
    PrepareRequest,
    Row,
    TokenCountUnavailable,
    is_extractive,
    prepare,
    profile_arguments,
    read_rows,
    refuse_undeclared_tools,
    render_call,
    split_rows,
    split_seed,
)
from litetune.spec import Spec

FUNCTIONGEMMA = "google/functiongemma-270m-it"

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeTokenCounter:
    """One token per whitespace-separated word. No tokenizer, no subprocess."""

    raises: BaseException | None = None
    calls: list[list[str]] = field(default_factory=list)

    name = "fake"

    def describe(self) -> dict[str, Any]:
        return {"tokenizer": "fake", "unit": "whitespace words"}

    def count(self, texts: Sequence[str]) -> list[int]:
        self.calls.append(list(texts))
        if self.raises is not None:
            raise self.raises
        return [len(text.split()) for text in texts]


@dataclass
class FakeHeadroomProbe:
    """Base-model accuracy per slice, canned."""

    scores: dict[str, float] = field(default_factory=dict)

    def base_accuracy(self, slice_name: str, rows: Sequence[Row]):
        if slice_name not in self.scores:
            return Unavailable(f"no base-model run covered slice {slice_name!r}")
        share = self.scores[slice_name]
        return Proportion.of(round(share * len(rows)), len(rows))


def rows(n: int, tool: str = "change_background_color", start: int = 0) -> list[dict]:
    """`n` examples whose argument is quoted verbatim from its own prompt."""
    return [
        {
            "prompt": f"set the background to colour swatch{i}",
            "target": {"name": tool, "args": {"color": f"swatch{i}"}},
        }
        for i in range(start, start + n)
    ]


@pytest.fixture
def write_jsonl(tmp_path: Path):
    def _write(records: Sequence[dict], name: str = "data.jsonl") -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def request_for(tmp_path):
    def _build(data: Path, **kwargs) -> PrepareRequest:
        params: dict[str, Any] = {
            "context_length": 128,
            "tokens": FakeTokenCounter(),
            "output_dir": tmp_path / "prepared",
        }
        params.update(kwargs)
        return PrepareRequest(data=data, **params)

    return _build


# The arguments this file's targets send, declared as optional strings so a
# declaration does not refuse a row for a reason the test is not about.
_ARGUMENTS = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "description": "c"},
        "colour": {"type": "string", "description": "c"},
    },
}


def _declarations(tmp_path: Path, *names: str) -> Path:
    """The OpenAI function objects the runtime requires, for `names`."""
    path = tmp_path / "declarations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {"name": name, "description": "d", "parameters": _ARGUMENTS},
                }
                for name in names
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_a_row_calling_an_undeclared_tool_is_refused_by_name(tmp_path, write_jsonl, request_for):
    """Training it would teach a call the prompt never offers.

    The model would learn to ask for a tool no runtime declares to it, and
    nothing downstream would say so: the loss curve of such a run looks exactly
    like one that worked, and the failure only appears when an application gets
    a call it has no handler for.
    """
    data = write_jsonl(rows(3, tool="open_app") + rows(1, tool="send_email", start=3))
    declarations = _declarations(tmp_path, "open_app")

    with pytest.raises(PrepareError) as caught:
        prepare(request_for(data, declarations=declarations))

    message = str(caught.value)
    # The fourth record is the fourth line, and the row carries its line number
    # precisely so it can be found in the file it came from.
    assert f"{data}:4" in message
    assert "'send_email'" in message
    assert "It offers open_app" in message
    assert not (tmp_path / "prepared" / "train.jsonl").exists()


def test_a_split_whose_calls_are_all_declared_is_prepared(tmp_path, write_jsonl, request_for):
    data = write_jsonl(rows(6, tool="open_app"))
    declarations = _declarations(tmp_path, "open_app", "set_timer")

    result = prepare(request_for(data, declarations=declarations))

    assert result.n_rows == 6
    assert result.train is not None


def test_without_declarations_nothing_about_tools_is_checked(tmp_path, write_jsonl, request_for):
    """Without declarations no row is checked against a tool list: a split that
    names tools prepares without a file saying which exist."""
    data = write_jsonl(rows(3, tool="open_app") + rows(1, tool="send_email", start=3))

    result = prepare(request_for(data))

    assert result.n_rows == 4


def test_a_row_that_brings_its_own_completion_is_not_second_guessed(
    tmp_path, write_jsonl, request_for
):
    """Only a structured target is checked. A caller who wrote the completion
    text said what to train, and litetune does not parse it back to work out
    which tool it names -- it would be guessing at a string it did not render.
    """
    data = write_jsonl(
        [{"prompt": "call it", "completion": "call:send_email{to:<escape>x<escape>}"}] * 4
    )
    declarations = _declarations(tmp_path, "open_app")

    result = prepare(request_for(data, declarations=declarations))

    assert result.n_rows == 4


def heldout_lines(result) -> set[int]:
    return {
        json.loads(line)["source_line"]
        for line in result.heldout.path.read_text(encoding="utf-8").splitlines()
    }


# ---------------------------------------------------------------------------
# The split is keyed by content, not by location
# ---------------------------------------------------------------------------


def test_the_same_bytes_at_two_paths_produce_the_same_split(write_jsonl, request_for, tmp_path):
    records = rows(60)
    here = write_jsonl(records, "a/data.jsonl")
    there = write_jsonl(records, "b/renamed.jsonl")

    first = prepare(request_for(here, output_dir=tmp_path / "one"))
    second = prepare(request_for(there, output_dir=tmp_path / "two"))

    assert first.content_sha256 == second.content_sha256
    # Location is not part of the identity, so moving the file must not move a
    # single example between the two sets.
    assert heldout_lines(first) == heldout_lines(second)
    assert first.heldout.content_sha256 == second.heldout.content_sha256


def test_changing_one_row_changes_the_split(write_jsonl, request_for, tmp_path):
    records = rows(60)
    original = prepare(request_for(write_jsonl(records, "x.jsonl"), output_dir=tmp_path / "one"))

    edited = [dict(record) for record in records]
    edited[0]["prompt"] = "set the background to colour swatch999"
    changed = prepare(request_for(write_jsonl(edited, "y.jsonl"), output_dir=tmp_path / "two"))

    assert original.content_sha256 != changed.content_sha256
    # A file replaced under the same name must invalidate everything downstream
    # of it, which starts with the split itself.
    assert heldout_lines(original) != heldout_lines(changed)


def test_the_seed_moves_the_split_and_nothing_else_does(write_jsonl, request_for, tmp_path):
    path = write_jsonl(rows(60))
    zero = prepare(request_for(path, seed=0, output_dir=tmp_path / "zero"))
    one = prepare(request_for(path, seed=1, output_dir=tmp_path / "one"))
    again = prepare(request_for(path, seed=0, output_dir=tmp_path / "zero-again"))

    assert heldout_lines(zero) != heldout_lines(one)
    assert heldout_lines(zero) == heldout_lines(again)


def test_split_seed_is_reproducible_from_the_report(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(40))))
    recorded = result.as_dict()["source"]["split_seed"]
    assert recorded == split_seed(result.content_sha256, result.request.seed)


def test_no_row_is_dropped_or_duplicated(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(50))))

    assert result.n_rows == 50
    assert result.train.n + result.heldout.n == 50
    train_lines = {
        json.loads(line)["source_line"]
        for line in result.train.path.read_text(encoding="utf-8").splitlines()
    }
    assert train_lines.isdisjoint(heldout_lines(result))
    assert train_lines | heldout_lines(result) == set(range(1, 51))


def test_split_rows_is_a_pure_function_of_content_and_seed():
    parsed = [Row(lineno=i + 1, prompt=f"p{i}", completion=f"c{i}", target=None) for i in range(20)]
    a_train, a_held = split_rows(parsed, "sha256:" + "a" * 64, 7, 5)
    b_train, b_held = split_rows(parsed, "sha256:" + "a" * 64, 7, 5)

    assert [r.lineno for r in a_held] == [r.lineno for r in b_held]
    assert len(a_held) == 5
    assert len(a_train) == 15


# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------


def test_a_small_heldout_split_is_a_recorded_limitation(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(40))))

    assert result.heldout.n == 8
    assert result.heldout.n < MIN_HELDOUT_EXAMPLES
    # A warning that travels with the result, not a refusal: the split is still
    # written and the checks still pass.
    assert result.outcome is Outcome.PASSED
    limitation = next(text for text in result.limitations if "below" in text and "held-out" in text)
    assert "0.172" in limitation and "0.024" in limitation
    assert any("below" in text for text in result.as_dict()["limitations"])


def test_a_large_enough_heldout_split_carries_no_size_limitation(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(1200))))

    assert result.heldout.n == 240
    assert not [text for text in result.limitations if "at n=64" in text]


def test_a_heldout_size_that_cannot_be_honoured_says_so(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(40)), heldout_size=500))

    assert result.heldout.n == 39
    limitation = next(text for text in result.limitations if "was requested" in text)
    assert "n=39" in limitation


def test_a_single_row_cannot_be_split(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(1))))

    check = next(c for c in result.checks.checks if c.name == SPLIT_CHECK)
    assert check.outcome is Outcome.FAILED
    assert result.train is None and result.heldout is None


# ---------------------------------------------------------------------------
# Length: named, never dropped
# ---------------------------------------------------------------------------


def test_an_over_length_example_fails_and_names_the_row(write_jsonl, request_for):
    records = rows(40)
    records[6]["prompt"] = " ".join(["word"] * 300)
    path = write_jsonl(records)

    result = prepare(request_for(path, context_length=64))

    check = next(c for c in result.checks.checks if c.name == LENGTH_CHECK)
    assert check.outcome is Outcome.FAILED
    # The row is identified by the line it is on, in the file the user has open.
    assert "line 7" in check.detail
    assert check.observed["over_length"][0]["source_line"] == 7
    assert result.outcome is Outcome.FAILED


def test_an_over_length_example_stops_the_split_being_written(write_jsonl, request_for):
    records = rows(40)
    records[6]["prompt"] = " ".join(["word"] * 300)

    result = prepare(request_for(write_jsonl(records), context_length=64))

    # Nothing is dropped and nothing is truncated: no split is produced at all,
    # so the next stage cannot read the file without seeing this check first.
    assert result.train is None and result.heldout is None
    assert not (result.request.output_dir / "train.jsonl").exists()
    # Skipped, not "could not check": the split check was never in scope, and
    # recording it as unchecked would make the whole set read `could not check`
    # and bury the measured failure underneath it.
    assert [s.name for s in result.skipped] == [SPLIT_CHECK]
    assert LENGTH_CHECK in result.skipped[0].reason
    assert SPLIT_CHECK not in {c.name for c in result.checks.checks}


def test_every_over_length_row_is_listed_not_just_the_first(write_jsonl, request_for):
    records = rows(40)
    for index in (2, 11, 30):
        records[index]["prompt"] = " ".join(["word"] * 300)

    result = prepare(request_for(write_jsonl(records), context_length=64))

    check = next(c for c in result.checks.checks if c.name == LENGTH_CHECK)
    assert [row["source_line"] for row in check.observed["over_length"]] == [3, 12, 31]


def test_length_statistics_report_the_expected_supervised_fraction(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(40))))

    assert isinstance(result.lengths, LengthStats)
    # Prompts here are 6 words and completions 1, so the supervised share is
    # small in the same direction the real data's is -- this is the number
    # `tune` has to reproduce.
    assert 0.0 < result.lengths.expected_supervised_fraction < 0.5
    assert result.lengths.as_dict()["available"] is True


def test_no_tokenizer_reports_could_not_check_and_still_splits(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(40)), tokens=None))

    check = next(c for c in result.checks.checks if c.name == LENGTH_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    # An unchecked item makes the set unchecked -- never a pass, never a fail.
    assert result.outcome is Outcome.UNCHECKED
    assert isinstance(result.lengths, Unavailable)
    assert result.train is not None and result.heldout is not None
    assert any("token lengths were not measured" in text for text in result.limitations)
    # Found in review: the check said an over-length row is truncated in
    # training while the limitation quoting it said `tune` refuses one.
    assert not any("truncated there" in text for text in [check.detail, *result.limitations])


def test_a_tokenizer_that_will_not_run_is_could_not_check(write_jsonl, request_for):
    counter = FakeTokenCounter(raises=TokenCountUnavailable("environment unavailable"))

    result = prepare(request_for(write_jsonl(rows(40)), tokens=counter))

    check = next(c for c in result.checks.checks if c.name == LENGTH_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    assert "TokenCountUnavailable" in check.detail
    assert result.outcome is Outcome.UNCHECKED


def test_the_counter_is_called_once_for_prompts_and_completions_together(write_jsonl, request_for):
    counter = FakeTokenCounter()
    prepare(request_for(write_jsonl(rows(40)), tokens=counter))

    assert len(counter.calls) == 1
    assert len(counter.calls[0]) == 80


# ---------------------------------------------------------------------------
# Scoreability: is exact match measuring anything?
# ---------------------------------------------------------------------------


def invented_message_rows(n: int, with_recipient: bool = False) -> list[dict]:
    """The measured shape: 92 distinct `message` values in 95 examples, invented."""
    records = []
    for i in range(n):
        args = {"message": f"Hi there, I have rescheduled our appointment number {i}."}
        if with_recipient:
            args["recipient"] = "colleague"
        records.append(
            {
                "prompt": f"tell my colleague that meeting {i} is moved",
                "target": {"name": "send_message", "args": args},
            }
        )
    return records


def test_a_high_cardinality_invented_argument_is_flagged(write_jsonl, request_for):
    # Exact match on this argument scored 0.00 for every model tested, the
    # untuned base included.
    result = prepare(request_for(write_jsonl(invented_message_rows(95, with_recipient=True))))

    flagged = result.unscoreable
    assert [(p.tool, p.argument) for p in flagged] == [("send_message", "message")]
    assert flagged[0].unique_share >= HIGH_CARDINALITY_SHARE
    assert flagged[0].extractive.value == 0.0
    limitation = next(text for text in result.limitations if "send_message.message" in text)
    assert "unscoreable by construction" in limitation
    assert "0.00 for every model" in limitation


def test_one_unscoreable_argument_among_several_is_a_limitation_not_a_refusal(
    write_jsonl, request_for
):
    # `recipient` is quoted from the prompt and still carries signal, so the
    # dataset is trainable and measurable -- with a caveat on one argument.
    # `metrics.py` reports operation name and argument accuracy separately for
    # exactly this reason.
    result = prepare(request_for(write_jsonl(invented_message_rows(95, with_recipient=True))))

    check = next(c for c in result.checks.checks if c.name == SCOREABILITY_CHECK)
    assert check.outcome is Outcome.PASSED
    assert result.outcome is Outcome.PASSED
    assert check.observed["scoreable"] == ["send_message.recipient"]
    assert result.train is not None


def test_a_dataset_where_nothing_is_scoreable_fails(write_jsonl, request_for):
    # No argument left with any signal: exact match is not a blunt instrument
    # here, it is no instrument at all, and it will read 0.00 for the tuned
    # model, the base and every quantization recipe alike.
    result = prepare(request_for(write_jsonl(invented_message_rows(95))))

    check = next(c for c in result.checks.checks if c.name == SCOREABILITY_CHECK)
    assert check.outcome is Outcome.FAILED
    assert "cannot measure anything" in check.detail
    assert result.outcome is Outcome.FAILED


def test_a_high_cardinality_quoted_argument_is_not_flagged(write_jsonl, request_for):
    # Google's reference dataset: 2,270 distinct values in 2,276 examples for
    # one field, and it scores fine -- because the value is always a literal
    # span of the prompt. Cardinality alone decides nothing.
    result = prepare(request_for(write_jsonl(rows(300))))

    profile = next(
        p for p in result.arguments if (p.tool, p.argument) == ("change_background_color", "color")
    )
    assert profile.unique_share == 1.0
    assert profile.high_cardinality is True
    assert profile.extractive.value == 1.0
    assert profile.scoreable is True
    assert result.unscoreable == ()
    check = next(c for c in result.checks.checks if c.name == SCOREABILITY_CHECK)
    assert check.outcome is Outcome.PASSED


def test_a_low_cardinality_invented_argument_is_not_flagged(write_jsonl, request_for):
    # Three repeated labels are learnable and scoreable even though no value is
    # quoted, so cardinality has to be part of the test and not extractiveness
    # alone.
    records = [
        {
            "prompt": f"turn the lights {'up' if i % 3 == 0 else 'down'} please, request {i}",
            "target": {"name": "set_mode", "args": {"mode": ["bright", "dim", "off"][i % 3]}},
        }
        for i in range(90)
    ]

    result = prepare(request_for(write_jsonl(records)))

    profile = next(p for p in result.arguments if p.argument == "mode")
    assert profile.unique_values == 3
    assert profile.high_cardinality is False
    assert profile.scoreable is True


def test_extractiveness_ignores_case_and_whitespace():
    assert is_extractive("Blue  Sky", "please make it a blue sky today")
    assert not is_extractive("cerulean", "please make it blue")
    # An empty value is in every string; counting it would report an unscoreable
    # argument as perfectly quoted.
    assert not is_extractive("", "anything at all")


def test_profiles_are_per_tool_not_per_argument_name():
    parsed = [
        Row(1, "book a table", "x", ToolCall(name="reserve", args={"name": "table"})),
        Row(2, "call mum", "y", ToolCall(name="dial", args={"name": "mum"})),
    ]
    profiles = profile_arguments(parsed)

    assert {(p.tool, p.argument) for p in profiles} == {("reserve", "name"), ("dial", "name")}


def test_unlabelled_rows_leave_scoreability_unchecked(write_jsonl, request_for):
    records = [{"prompt": f"do thing {i}", "completion": f"done {i}"} for i in range(40)]

    result = prepare(request_for(write_jsonl(records)))

    check = next(c for c in result.checks.checks if c.name == SCOREABILITY_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    assert result.arguments == ()


# ---------------------------------------------------------------------------
# Headroom: the hook, not the measurement
# ---------------------------------------------------------------------------


def test_headroom_is_reported_as_unmeasured_without_a_probe(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(40))))

    assert [s.name for s in result.slices] == ["change_background_color"]
    assert isinstance(result.slices[0].base_accuracy, Unavailable)
    # Three-valued, here too: not "has headroom", not "has none", but "nobody
    # looked".
    assert result.slices[0].has_headroom is None
    assert any("no base-model measurement was supplied" in t for t in result.limitations)


def test_a_slice_with_no_headroom_is_flagged_when_a_probe_supplies_one(write_jsonl, request_for):
    records = rows(40) + rows(40, tool="open_app", start=100)
    probe = FakeHeadroomProbe(scores={"change_background_color": 1.0, "open_app": 0.5})

    result = prepare(request_for(write_jsonl(records), headroom=probe))

    by_name = {s.name: s for s in result.slices}
    assert by_name["change_background_color"].has_headroom is False
    assert by_name["open_app"].has_headroom is True
    limitation = next(text for text in result.limitations if "change_background_color" in text)
    assert "cannot show a gain" in limitation


# ---------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------


def test_a_row_with_no_answer_names_itself(write_jsonl, request_for):
    records = rows(5)
    records[2] = {"prompt": "do something"}
    path = write_jsonl(records)

    with pytest.raises(PrepareError) as exc:
        prepare(request_for(path))

    assert ":3:" in str(exc.value)
    assert "no supervised span" in str(exc.value)


def test_malformed_json_names_the_line(tmp_path, request_for):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"prompt": "ok", "completion": "x"}\nnot json\n', encoding="utf-8")

    with pytest.raises(PrepareError) as exc:
        prepare(request_for(path))

    assert ":2:" in str(exc.value)


def test_an_explicit_completion_wins_over_a_rendered_target(write_jsonl, request_for):
    records = [
        {
            "prompt": "make it blue",
            "completion": "hand written answer",
            "target": {"name": "paint", "args": {"color": "blue"}},
        }
    ] * 40

    result = prepare(request_for(write_jsonl(records)))

    first = json.loads(result.train.path.read_text(encoding="utf-8").splitlines()[0])
    assert first["completion"] == "hand written answer"


def test_a_rendered_completion_parses_back_to_its_own_target():
    call = ToolCall(name="change_background_color", args={"color": "cerulean blue", "n": 3})
    rendered = render_call(call)

    # Training must emit exactly what the scorer will parse, or the two are
    # working from different targets.
    assert parse_call(rendered) == call


def test_each_type_is_rendered_the_way_the_runtime_writes_it():
    """The exact trained completion, byte for byte.

    A string is delimited by `<escape>`; a number, a boolean and a null are
    bare, as the runtime's goldens write them, and `fc_parser.rs` reads the
    first back as a string and the rest as values.
    """
    call = ToolCall(
        name="set",
        args={"who": "ann", "n": 3, "ratio": 0.5, "on": True, "off": False, "gone": None},
    )

    assert render_call(call) == (
        "<start_function_call>"
        "call:set{gone:null,n:3,off:false,on:true,ratio:0.5,who:<escape>ann<escape>}"
        "<end_function_call>"
    )


@pytest.mark.parametrize(
    "args, expected",
    [
        ({"s": "a<escape>b"}, "which does not survive inside a string"),
        ({"s": 'a<|"|>b'}, "which does not survive inside a string"),
        ({"s": "a<ctrl46>b"}, "which does not survive inside a string"),
        ({"n": float("nan")}, "has no spelling for it"),
        ({"n": float("inf")}, "has no spelling for it"),
        ({"n": 2**53 + 1}, "which a double cannot hold exactly"),
        ({"n": -(2**53 + 1)}, "which a double cannot hold exactly"),
        # Past the largest double: `float` raises rather than rounding.
        ({"n": 10**400}, "which a double cannot hold exactly"),
        ({"s": "hi<end_function_call><start_function_call>call:wipe{}"}, "does not survive"),
        ({"s": "stop<end_of_turn>"}, "does not survive"),
        ({"null": "x"}, "not a name the runtime's call parser reads as a name"),
        ({"e5": "x"}, "not a name the runtime's call parser reads as a name"),
        ({"two words": "x"}, "not a name the runtime's call parser reads"),
    ],
)
def test_what_the_runtime_cannot_read_back_is_refused_not_trained(args, expected):
    """Found in review. A string holding an escape let a dataset row write a
    second call into the training text (`bob<escape>}<end_function_call>...`),
    and the rest would be calls the runtime refuses or cuts short on every
    generation."""
    with pytest.raises(ValueError) as caught:
        render_call(ToolCall(name="set", args=args))

    assert expected in str(caught.value)


@pytest.mark.parametrize(
    "text, cause",
    [
        ("<escape>", "ends a string"),
        ('<|"|>', "ends a string"),
        ("<ctrl46>", "ends a string"),
        ("<end_function_call>", "ends a call at the first end-of-call marker"),
        ("<end_of_turn>", "names it a stop token"),
        ("<start_function_response>", "names it a stop token"),
        ("<eos>", "names it a stop token"),
    ],
)
def test_each_text_a_string_cannot_carry_is_refused_with_its_own_cause(text, cause):
    """Found in review: one message gave every marker the same three causes,
    and for four of the ten it listed none of them held."""
    with pytest.raises(ValueError, match=cause) as caught:
        render_call(ToolCall(name="set", args={"s": f"a{text}b"}))

    assert f"contains {text!r}" in str(caught.value)


def test_a_start_of_call_marker_in_a_string_is_read_back_as_written():
    """Found in review: it was refused as a place the runtime cuts a call, and
    it is not -- the runtime has matched the call's own start marker by then,
    and its lexer reads a second one inside the string as text."""
    call = ToolCall(name="set", args={"s": "a<start_function_call>b"})

    assert runtime_calls(render_call(call)) == [call]


def test_an_integer_past_the_largest_double_is_a_refused_row_not_a_crash(tmp_path):
    """Found in review: `float(value)` raises `OverflowError`, which is not a
    `ValueError`, so the row reader let it out as a traceback naming no row."""
    data = tmp_path / "rows.jsonl"
    data.write_text(
        json.dumps({"prompt": "p", "target": {"name": "set", "args": {"n": 10**400}}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PrepareError, match=r"rows.jsonl:1: .*a double cannot hold exactly"):
        read_rows(data)


def test_a_tool_name_the_runtime_cannot_read_is_refused():
    for name in ("caf\u00e9", "call", "true"):
        with pytest.raises(ValueError, match="not one the runtime's call parser reads"):
            render_call(ToolCall(name=name, args={}))


@pytest.mark.parametrize(
    "value, spelled",
    [
        (1.5e-07, "0.00000015"),
        (12345678901234567890.0, "12345678901234567168"),
        (7.0, "7"),
        (2**53, "9007199254740992"),
        # Past 2**53 but held exactly by a double: nothing is lost, so it is
        # written, where the first version refused every integer this large.
        (10**20, "100000000000000000000"),
    ],
)
def test_a_number_is_spelled_the_way_the_runtimes_lexer_reads_it(value, spelled):
    """`json.dumps` writes `1.5e-07` and `1.2345678901234567e+19`, and the lexer
    takes a fraction or an exponent, never both. The same double, in digits."""
    rendered = render_call(ToolCall(name="set", args={"n": value}))

    assert f"{{n:{spelled}}}" in rendered
    assert parse_call(rendered) == ToolCall(name="set", args={"n": value})


def test_the_arguments_follow_the_declared_order_whatever_the_case(tmp_path):
    """One sort for both, and it is the template's `dictsort`, which ignores
    case: the grammar holds a call to the order the declarations were handed
    over in, so the order trained here has to be that order."""
    path = tmp_path / "declarations.json"
    properties = {
        "URL": {"type": "string", "description": "u"},
        "body": {"type": "string", "description": "b"},
    }
    path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "send",
                        "description": "d",
                        "parameters": {"type": "object", "properties": properties},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    declared = list(read_declarations(path)[0][0]["function"]["parameters"]["properties"])

    parsed = parse_call(render_call(ToolCall(name="send", args={"URL": "u", "body": "b"})))

    assert parsed is not None
    assert list(parsed.args) == declared == ["body", "URL"]


def _one_tool(tmp_path: Path, properties: dict, required: list[str] | None = None) -> Path:
    parameters: dict = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    path = tmp_path / "declarations.json"
    path.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {"name": "set", "description": "d", "parameters": parameters},
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "args, expected",
    [
        ({"level": 3, "extra": "x"}, "sends ['extra'], which the declaration of 'set' does not"),
        ({"mode": "loud"}, "leaves out ['level'], which the declaration of 'set' requires"),
        ({"level": "3"}, "sends level='3', a str, where the declaration of 'set' says integer"),
        ({"level": True}, "sends level=True, a bool, where the declaration of 'set' says integer"),
        ({"level": 3, "mode": "shout"}, "which is not in the declared enum ['loud', 'soft']"),
        ({"level": 3, "ratio": True}, "sends ratio=True, a bool"),
        ({"level": 3, "loud": 1}, "sends loud=1, a int"),
        ({"level": 3, "mode": 3}, "sends mode=3, a int"),
        ({"level": 3, "tags": "a,b"}, "sends tags='a,b', a str"),
        ({"level": 3, "shout": 1}, "sends shout=1, a int"),
        ({"level": 7.5}, "sends level=7.5, a float, where the declaration of 'set' says integer"),
        ({"level": 3, "opts": "x"}, "sends opts='x', a str"),
    ],
)
def test_a_target_that_contradicts_its_declaration_is_refused(
    tmp_path, write_jsonl, args, expected
):
    """Found in review: only the tool's name was checked. A target that sends an
    undeclared argument, leaves out a required one, or sends another type or a
    value outside the enum trains a call contradicting the declaration the
    prompt shows the model."""
    declarations = _one_tool(
        tmp_path,
        {
            "level": {"type": "integer", "description": "l"},
            "mode": {"type": "string", "description": "m", "enum": ["loud", "soft"]},
            "ratio": {"type": "number", "description": "r"},
            "loud": {"type": "boolean", "description": "b"},
            "tags": {"type": "array", "description": "t", "items": {"type": "string"}},
            # Capitals, the way google/mobile-actions writes its types.
            "shout": {"type": "BOOLEAN", "description": "s"},
            "opts": {
                "type": "object",
                "description": "o",
                "properties": {"x": {"type": "string", "description": "x"}},
            },
        },
        required=["level"],
    )
    data = write_jsonl([{"prompt": "set it", "target": {"name": "set", "args": args}}])

    with pytest.raises(PrepareError) as caught:
        refuse_undeclared_tools(read_rows(data), data, declarations)

    assert expected in str(caught.value)
    assert f"{data}:1" in str(caught.value)


def test_a_target_that_keeps_to_its_declaration_passes(tmp_path, write_jsonl):
    declarations = _one_tool(
        tmp_path,
        {
            "level": {"type": "INTEGER", "description": "l"},
            "ratio": {"type": "number", "description": "r"},
            "mode": {"type": "string", "description": "m", "enum": ["loud", "soft"]},
        },
        required=["level"],
    )
    data = write_jsonl(
        [
            {"prompt": "set it", "target": {"name": "set", "args": {"level": 3, "ratio": 1}}},
            # An integral float is an integer: it is written `7` and returned 7.0.
            {"prompt": "set it", "target": {"name": "set", "args": {"level": 7.0}}},
        ]
    )

    refuse_undeclared_tools(read_rows(data), data, declarations)


CALL_ROWS = [
    {"prompt": "make it red", "target": {"name": "set_colour", "args": {"colour": "red"}}},
]


def test_a_family_whose_calls_were_measured_is_rendered_in_its_own_format(
    tmp_path, write_jsonl, request_for
):
    result = prepare(
        request_for(
            write_jsonl(CALL_ROWS * 40),
            base_model=FUNCTIONGEMMA,
            declarations=_declarations(tmp_path, "set_colour"),
        )
    )

    assert result.identity is not None
    assert result.identity["family"] == "functiongemma"
    assert result.identity["wire_format"] == "functiongemma"
    first = json.loads(result.train.path.read_text(encoding="utf-8").splitlines()[0])
    assert first["completion"] == (
        "<start_function_call>call:set_colour{colour:<escape>red<escape>}<end_function_call>"
    )
    assert ASSUMED_WIRE_FORMAT not in result.limitations


def test_a_family_with_no_measured_call_format_is_refused_by_row(write_jsonl, request_for):
    """Qwen-3 has an entry that deliberately records nothing about its calls.

    Rendering FunctionGemma's spelling for it trains a format its runtime does
    not read, and no check downstream can see that -- which is the defect this
    gate exists to close.
    """
    with pytest.raises(PrepareError) as caught:
        prepare(request_for(write_jsonl(CALL_ROWS * 40), base_model="Qwen/Qwen3-0.6B"))

    assert "qwen-3" in str(caught.value)
    assert "completion" in str(caught.value)


def test_a_model_litetune_has_no_entry_for_says_that_instead(write_jsonl, request_for):
    with pytest.raises(PrepareError) as caught:
        prepare(request_for(write_jsonl(CALL_ROWS * 40), base_model="meta-llama/Llama-3.2-1B"))

    assert "no entry for this model" in str(caught.value)


def test_only_a_structured_target_reaches_the_refusal(write_jsonl, request_for):
    """A row that brings its own completion is the caller saying what to train.

    litetune does not parse it back to work out which family it is in, so the
    gate never fires on it -- whatever the model is, and whether or not litetune
    has an entry for it.
    """
    rows = [{"prompt": "make it red", "completion": "call:set_colour{colour:red}"}] * 40

    for model in ("Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", FUNCTIONGEMMA):
        result = prepare(request_for(write_jsonl(rows), base_model=model))
        first = json.loads(result.train.path.read_text(encoding="utf-8").splitlines()[0])
        assert first["completion"] == "call:set_colour{colour:red}"


def test_naming_no_model_renders_functiongemmas_format_and_says_it_assumed_it(
    write_jsonl, request_for
):
    """Every caller that predates the flag. Refusing them would refuse splits
    that work; saying nothing would leave the assumption invisible."""
    result = prepare(request_for(write_jsonl(CALL_ROWS * 40)))

    assert result.identity is None
    assert ASSUMED_WIRE_FORMAT in result.limitations
    first = json.loads(result.train.path.read_text(encoding="utf-8").splitlines()[0])
    assert first["completion"] == (
        "<start_function_call>call:set_colour{colour:<escape>red<escape>}<end_function_call>"
    )


def test_a_plain_text_split_never_mentions_a_wire_format(write_jsonl, request_for):
    """Plain text at this level: nothing about declarations or formats applies."""
    rows = [{"prompt": f"q{i}", "completion": "an answer"} for i in range(40)]

    result = prepare(request_for(write_jsonl(rows)))

    assert ASSUMED_WIRE_FORMAT not in result.limitations
    assert result.identity is None


def test_the_family_resolves_from_a_checkpoint_the_way_convert_resolves_it(
    tmp_path, write_jsonl, request_for
):
    """Sidecar, then config, then the name -- the existing order, not a second one.

    The folder is named after the wrong family on purpose: that is the case the
    order exists for, and a path must not outrank what is inside.
    """
    checkpoint = tmp_path / "runs" / "qwen-ish" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "gemma3_text"}), encoding="utf-8"
    )
    (checkpoint / PROVENANCE_NAME).write_text(
        json.dumps({"base_model": FUNCTIONGEMMA}), encoding="utf-8"
    )

    result = prepare(
        request_for(
            write_jsonl(CALL_ROWS * 40),
            base_model=str(checkpoint),
            declarations=_declarations(tmp_path, "set_colour"),
        )
    )

    assert result.identity is not None
    assert result.identity["family"] == "functiongemma"
    assert result.identity["identity_recorded"] is True


def test_a_declaration_rendering_family_refuses_to_train_calls_without_them(
    write_jsonl, request_for
):
    """The runtime puts the declarations in the prompt before the model sees the
    question -- 809 characters where a run without them trains 79. A split that
    skips them teaches an answer to a prompt no application sends, and the loss
    curve of such a run is indistinguishable from one that worked."""
    with pytest.raises(PrepareError) as caught:
        prepare(request_for(write_jsonl(CALL_ROWS * 40), base_model=FUNCTIONGEMMA))

    message = str(caught.value)
    assert "--declarations" in message
    assert "cannot be derived from the targets" in message


def test_that_refusal_does_not_fire_on_plain_text_for_the_same_family(write_jsonl, request_for):
    rows_ = [{"prompt": f"q{i}", "completion": "an answer"} for i in range(40)]

    result = prepare(request_for(write_jsonl(rows_), base_model=FUNCTIONGEMMA))

    assert result.n_rows == 40


def test_a_value_with_no_measured_shape_is_refused_by_row(write_jsonl, request_for):
    """A list or an object has no established spelling in a call.

    Nothing in this project says what the runtime's parser accepts for one, and
    guessing would teach the model a format nobody has seen the runtime read.
    The row names the way through, which is the path `read_rows` already
    prefers: supply the completion text.
    """
    data = write_jsonl(
        [{"prompt": "tag it", "target": {"name": "tag", "args": {"labels": ["a", "b"]}}}]
    )

    with pytest.raises(PrepareError) as caught:
        read_rows(data)

    assert ":1:" in str(caught.value)
    assert "'labels' is a list" in str(caught.value)
    assert "'completion'" in str(caught.value)


# ---------------------------------------------------------------------------
# The report and what comes after it
# ---------------------------------------------------------------------------


def test_a_report_is_written_even_when_prepare_fails(write_jsonl, request_for):
    records = rows(40)
    records[0]["prompt"] = " ".join(["word"] * 300)

    result = prepare(request_for(write_jsonl(records), context_length=64))

    assert result.outcome is Outcome.FAILED
    assert result.report_path is not None and result.report_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["outcome"] == "failed"
    assert report["train"] is None
    assert report["lengths"]["over_length"][0]["source_line"] == 1


def test_the_report_carries_the_hashes_a_spec_needs(write_jsonl, request_for):
    result = prepare(request_for(write_jsonl(rows(400))))

    fragment = result.spec_fragment()
    # `spec.Dataset.content_sha256` and `spec.EvalSpec.heldout_content_sha256`
    # are required fields identifying files that did not exist until now, so
    # prepare has to emit them or no spec can ever describe its own split.
    spec = Spec.from_mapping(
        spec_mapping(dataset=fragment["dataset"], eval=fragment["eval"]), source="fragment"
    )
    assert spec.dataset.content_sha256 == fragment["dataset"]["content_sha256"]
    assert spec.eval.heldout_content_sha256 == fragment["eval"]["heldout_content_sha256"]


def test_nothing_is_printed_and_everything_is_an_event(write_jsonl, request_for, capsys):
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = prepare(request_for(write_jsonl(rows(40))), events=events)

    assert capsys.readouterr().out == ""
    kinds = [event.kind for event in seen]
    assert kinds[0] == "stage_started" and kinds[-1] == "stage_finished"
    assert "check" in kinds and "metric" in kinds and "artifact_written" in kinds
    names = {e.data.get("name") for e in seen if e.kind == "check"}
    assert {LENGTH_CHECK, SCOREABILITY_CHECK, SPLIT_CHECK} <= names
    assert result.outcome is Outcome.PASSED


def test_the_heldout_file_is_readable_by_the_evaluator(write_jsonl, request_for):
    from litetune.evaluate import load_split

    result = prepare(request_for(write_jsonl(rows(400))))

    split = load_split(result.heldout.path)
    assert split.n == result.heldout.n
    assert len(split.labelled) == split.n


def test_a_bad_request_is_refused_before_anything_is_read(tmp_path):
    with pytest.raises(ValueError):
        PrepareRequest(data=tmp_path / "x.jsonl", output_dir=tmp_path, context_length=0)
    with pytest.raises(ValueError):
        PrepareRequest(
            data=tmp_path / "x.jsonl",
            output_dir=tmp_path,
            context_length=64,
            heldout_fraction=1.0,
        )


def test_a_string_target_is_a_task_declaration_not_a_malformed_call(tmp_path):
    """The shape of the target says what the task is.

    An object with a `name` is a tool call, scored by `tool-call`; a bare string
    is the answer itself, scored by `exact-text`. Two shapes rather than a
    target plus a `--target-kind`, because those two could disagree and the
    shape cannot disagree with itself.
    """
    from litetune.metrics import ToolCall, read_target

    assert read_target(None) is None
    assert read_target("a cat sat") == "a cat sat"
    assert read_target({"name": "set_alarm", "args": {"hour": "7"}}) == ToolCall(
        name="set_alarm", args={"hour": "7"}
    )

    with pytest.raises(ValueError, match="a tool call.*or a string"):
        read_target(42)


def test_prepare_splits_a_text_task_and_says_the_profile_does_not_apply(tmp_path):
    """A string-target split is scoreable; it just has no arguments to profile.

    "No argument profile" has two causes and they are not the same news: string
    targets are a task with no arguments, absent targets are a split nobody can
    score at all. The check now distinguishes them, because the second is a
    reason to stop and the first is not.
    """
    data = tmp_path / "raw.jsonl"
    data.write_text(
        "\n".join(
            json.dumps({"prompt": f"classify: item {i}", "target": "red" if i % 2 else "blue"})
            for i in range(40)
        ),
        encoding="utf-8",
    )

    result = prepare(PrepareRequest(data=data, output_dir=tmp_path / "out", context_length=512))

    assert (tmp_path / "out" / "train.jsonl").exists()
    assert (tmp_path / "out" / "heldout.jsonl").exists()

    profile = [c for c in result.checks.checks if c.name == SCOREABILITY_CHECK][0]
    assert "every target is a string" in (profile.detail or "")
    assert profile.observed["labelled"] == 40

    # And the completion was rendered from the string, not from a wire format.
    first = json.loads((tmp_path / "out" / "train.jsonl").read_text().splitlines()[0])
    assert first["completion"] in {"red", "blue"}
    assert first["target"] == first["completion"]


def test_prerendered_calls_prepare_without_declarations_because_the_prompt_carries_them(
    write_jsonl, request_for
):
    """In `prerendered` the application rendered the declarations into the
    prompt already, so there is nothing for `--declarations` to add. The first
    version of this refusal asked only the family, which refused the README's
    own walkthrough."""
    decl = (
        "<start_of_turn>developer\n<start_function_declaration>declaration:set_colour"
        "{description:<escape>d<escape>}<end_function_declaration>\n<end_of_turn>\n"
    )
    rows_ = [
        {
            "prompt": f"{decl}<start_of_turn>user\nmake it red {i}<end_of_turn>\n"
            "<start_of_turn>model\n",
            "target": {"name": "set_colour", "args": {"colour": "red"}},
        }
        for i in range(40)
    ]

    result = prepare(request_for(write_jsonl(rows_), base_model=FUNCTIONGEMMA))

    assert result.n_rows == 40


def test_the_report_records_the_declarations_the_targets_were_checked_against(
    tmp_path, write_jsonl, request_for
):
    """Found in review: `tune` and `verify` record the digest and `prepare` did
    not, so a split could not be traced to the tool list it was checked with.
    Bare prompts, so the runtime adds a declaration turn this stage cannot
    count, and the report says the length check undercounts."""
    declarations = _declarations(tmp_path, "set_colour")
    rows_ = [
        {"prompt": f"make it red {i}", "target": {"name": "set_colour", "args": {"colour": "red"}}}
        for i in range(40)
    ]

    result = prepare(
        request_for(write_jsonl(rows_), base_model=FUNCTIONGEMMA, declarations=declarations)
    )

    record = result.as_dict()
    assert record["declarations_sha256"] == read_declarations(declarations)[1]
    assert record["request"]["declarations"] == str(declarations)
    assert any("without the declaration turn" in text for text in result.limitations)


@pytest.mark.parametrize(
    "extra",
    [
        {"tokens": None},  # no lengths were measured, so none undercount
        # A counter that raised measured nothing either.
        {"tokens": FakeTokenCounter(raises=TokenCountUnavailable("environment unavailable"))},
        {"base_model": "Qwen/Qwen3-0.6B"},  # no declaration turn is added for it
    ],
)
def test_the_length_note_is_said_only_where_it_is_true(tmp_path, write_jsonl, request_for, extra):
    """Found in review: it fired with no tokenizer and for a family whose
    runtime puts no declaration turn in front of the prompt."""
    declarations = _declarations(tmp_path, "set_colour")
    rows_ = [
        {"prompt": f"make it red {i}", "target": {"name": "set_colour", "args": {"colour": "red"}}}
        for i in range(40)
    ]
    params = {"base_model": FUNCTIONGEMMA, "declarations": declarations} | extra
    if params["base_model"] != FUNCTIONGEMMA:
        rows_ = [{"prompt": r["prompt"], "completion": "red"} for r in rows_]

    result = prepare(request_for(write_jsonl(rows_), **params))

    assert not any("without the declaration turn" in text for text in result.limitations)


def test_an_undeclared_tool_is_a_reason_not_an_exception(tmp_path):
    from litetune.declarations import argument_problem

    parsed = read_declarations(_declarations(tmp_path, "set_colour"))[0]

    assert argument_problem(parsed, "open_app", {}) == "calls 'open_app', which is not declared"


def test_the_arguments_are_written_in_the_order_the_declarations_are_sorted_into():
    """The runtime's grammar enforces the declared property order, and
    `declarations.py` sorts the declarations. Measured 2026-09-17: a model
    trained in the dataset's own argument order lost an argument on 112 of 640
    rows under the grammar -- `send_email` came back as `subject, to`, never
    with `body`, because `body` sorts first and the model wrote it last.
    """
    call = ToolCall(name="send_email", args={"to": "a@b.c", "subject": "hi", "body": "text"})

    assert render_call(call) == (
        "<start_function_call>call:send_email{"
        "body:<escape>text<escape>,subject:<escape>hi<escape>,to:<escape>a@b.c<escape>"
        "}<end_function_call>"
    )
