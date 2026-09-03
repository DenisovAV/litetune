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
from litetune.events import EventStream
from litetune.metrics import Proportion, ToolCall, Unavailable, parse_call
from litetune.prepare import (
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
    render_call,
    split_rows,
    split_seed,
)
from litetune.spec import Spec

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
