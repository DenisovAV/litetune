"""The decision prompt contract: the golden strings, the record, and every refusal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from litetune.decisions import (
    ANSWER_LINE,
    CONTRACT_VERSION,
    GENERIC_QUESTION,
    NOUL_IDS,
    Decision,
    DecisionError,
    Option,
    decision_from,
    inferred_decision,
    read_questions,
    state_problem,
)
from litetune.storage import hash_file

FIXTURE = Path(__file__).parent / "fixtures" / "decision-contract-v1.json"

# The fixture is the contract litetune's renderer tests against, and the one
# flutter_edge_ai's proposed renderer is to test against. A change to it is a
# change to what both sides render, so it is made on purpose: update
# this digest in the same commit, and the contract version with it.
FIXTURE_SHA256 = "sha256:bcac6b4fd2f3131b9bb713d9aa5d9270a4dc7707b411985861fa1c7ce552a051"

CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def test_the_fixture_is_the_pinned_contract():
    assert json.loads(FIXTURE.read_text(encoding="utf-8"))["contract_version"] == CONTRACT_VERSION
    assert hash_file(FIXTURE) == FIXTURE_SHA256


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_each_golden_prompt_is_rendered_byte_for_byte(case):
    decision = decision_from(case["questions"], case["name"])
    assert decision.render(case["state"]) == case["prompt"]
    assert decision.answer_for(case["label"], case["name"]) == case["completion"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_each_case_spells_out_the_decision_a_reader_of_the_section_gets(case):
    assert case["decision"] == decision_from(case["questions"], case["name"]).as_dict()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_record_reads_back_to_the_same_decision(case):
    decision = decision_from(case["questions"], case["name"])
    again = decision_from(json.loads(decision.canonical_text()), "decision.json")
    assert again == decision
    assert again.digest() == decision.digest()


def test_the_measured_cases_cover_descriptions_and_the_none_option():
    """What each measured case is in the fixture for, so a trimmed fixture fails here."""
    by_name = {case["name"]: case for case in CASES}
    abcd = by_name["abcd"]["prompt"]
    assert "\n- recover_username: account_access\n" in abcd
    out_of_scope = by_name["clinc151 out of scope"]
    assert out_of_scope["label"] is None
    assert out_of_scope["completion"] == "none_of_these"
    assert out_of_scope["prompt"].endswith(f"\n- none_of_these\n{ANSWER_LINE}")


def test_the_contract_text_is_written_out_once():
    decision = Decision("choice", "Which?", (Option("a", "first"), Option("b")), none_option="none")
    assert decision.render("the state\nover two lines") == (
        "the state\nover two lines\n\nQuestion: Which?\nOptions:\n- a: first\n- b\n- none\n"
        "Answer with the id of one option."
    )
    assert decision.ids == ("a", "b", "none")


def test_a_null_label_is_the_none_option_and_its_name_is_too():
    decision = decision_from(
        {"question": "Which?", "options": [{"id": "a"}, {"id": "b"}], "none_option": "none"}, "q"
    )
    assert decision.answer_for(None, "row") == "none"
    assert decision.answer_for("none", "row") == "none"


def test_a_null_label_without_a_none_option_is_refused():
    decision = decision_from({"question": "Which?", "options": [{"id": "a"}, {"id": "b"}]}, "q")
    with pytest.raises(DecisionError, match="row 3: the label is null.*no none_option"):
        decision.answer_for(None, "row 3")


def test_a_label_outside_the_options_is_refused_by_name():
    decision = decision_from({"question": "Which?", "options": [{"id": "a"}, {"id": "b"}]}, "q")
    with pytest.raises(DecisionError, match="row 3: the label 'c' is not one of the options"):
        decision.answer_for("c", "row 3")


def test_type_defaults_to_choice():
    decision = decision_from({"question": "Which?", "options": [{"id": "a"}, {"id": "b"}]}, "q")
    assert decision.type == "choice"


def test_a_noul_question_lists_no_then_yes():
    decision = decision_from({"type": "noul", "question": "Is it?"}, "q")
    assert decision.ids == NOUL_IDS == ("no", "yes")
    described = decision_from(
        {
            "type": "noul",
            "question": "Is it?",
            "options": [{"id": "no", "description": "it is not"}, {"id": "yes"}],
        },
        "q",
    )
    assert described.render("s").endswith("- no: it is not\n- yes\n" + ANSWER_LINE)


def test_a_noul_question_in_another_order_is_refused():
    with pytest.raises(DecisionError, match=r"\['no', 'yes'\], in that order"):
        decision_from(
            {"type": "noul", "question": "Is it?", "options": [{"id": "yes"}, {"id": "no"}]}, "q"
        )


def test_a_score_question_numbers_its_levels_lowest_first():
    decision = decision_from(
        {"type": "score", "question": "How much?", "options": ["none", "some", "a lot"]}, "q"
    )
    assert decision.options == (Option("0", "none"), Option("1", "some"), Option("2", "a lot"))
    # The record writes the levels as objects, and reads them back the same.
    assert decision_from(json.loads(decision.canonical_text()), "q") == decision


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ([{"id": "1", "description": "a"}, {"id": "0", "description": "b"}], "lowest first"),
        ([{"id": "0", "description": "a"}, {"id": "1"}], r"levels \['1'\] have no text"),
    ],
)
def test_a_score_question_with_misnumbered_or_untitled_levels_is_refused(options, message):
    with pytest.raises(DecisionError, match=message):
        decision_from({"type": "score", "question": "How much?", "options": options}, "q")


@pytest.mark.parametrize("kind", ["noul", "score"])
def test_only_a_choice_question_has_a_none_option(kind):
    obj = {"type": kind, "question": "Q?", "options": ["a", "b"], "none_option": "none"}
    if kind == "noul":
        del obj["options"]
    with pytest.raises(DecisionError, match="has no none_option"):
        decision_from(obj, "q")


@pytest.mark.parametrize(
    ("obj", "message"),
    [
        ([], "expected an object"),
        ({"question": "Q?", "options": [{"id": "a"}, {"id": "b"}], "questions": []}, "keys this"),
        (
            {"contract_version": 2, "question": "Q?", "options": [{"id": "a"}, {"id": "b"}]},
            "contract_version 2",
        ),
        ({"type": "rank", "question": "Q?", "options": [{"id": "a"}, {"id": "b"}]}, "type"),
        ({"options": [{"id": "a"}, {"id": "b"}]}, "no 'question'"),
        ({"question": "Q?"}, "needs 'options'"),
        ({"question": "Q?", "options": {"a": "b"}}, "must be a list"),
        ({"question": "Q?", "options": ["a", "b"]}, r"options\[0\] must be an object"),
        ({"question": "Q?", "options": [{"id": "a"}, {"name": "b"}]}, "does not read"),
        ({"question": "Q?", "options": [{"id": "a"}, {"description": "b"}]}, "has no 'id'"),
        ({"question": "Q?", "options": [{"id": "a"}, {"id": 2}]}, "must be a string"),
        ({"question": "Q?", "options": [{"id": "a"}, {"id": ""}]}, "is empty"),
        ({"question": "Q?", "options": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}, "more than"),
        (
            {"question": "Q?", "options": [{"id": "a"}, {"id": "b"}], "none_option": "a"},
            r"\['a'\] are listed more than once",
        ),
        ({"question": "Q?", "options": [{"id": "a"}]}, "at least two options"),
        ({"question": "two\nlines", "options": [{"id": "a"}, {"id": "b"}]}, "more than one"),
        ({"question": "Q?", "options": [{"id": "a\r"}, {"id": "b"}]}, "more than one line"),
        ({"question": "Q?", "options": [{"id": " a"}, {"id": "b"}]}, "whitespace"),
        (
            {"question": "Q?", "options": [{"id": "a", "description": "x "}, {"id": "b"}]},
            "whitespace",
        ),
        (
            {"question": "Q?", "options": [{"id": "a<end_of_turn>"}, {"id": "b"}]},
            "control token",
        ),
    ],
)
def test_a_decision_that_cannot_be_rendered_is_refused_naming_the_field(obj, message):
    with pytest.raises(DecisionError, match=message):
        decision_from(obj, "questions.json")


@pytest.mark.parametrize("version", [True, 1.0])
def test_the_contract_version_must_be_the_int_one(version):
    with pytest.raises(DecisionError, match="contract_version"):
        decision_from(
            {"contract_version": version, "question": "Q?", "options": [{"id": "a"}, {"id": "b"}]},
            "q",
        )


@pytest.mark.parametrize(
    "char", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_a_character_that_splitlines_splits_on_is_a_line_break(char):
    with pytest.raises(DecisionError, match="more than one line"):
        decision_from({"question": "Q?", "options": [{"id": f"a{char}b"}, {"id": "b"}]}, "q")


def test_a_lone_surrogate_is_refused_naming_it():
    with pytest.raises(DecisionError, match="lone surrogate U\\+D800"):
        decision_from({"question": "Q?", "options": [{"id": "a\ud800"}, {"id": "b"}]}, "q")


def test_a_control_character_is_refused_naming_it_but_a_tab_is_text():
    with pytest.raises(DecisionError, match="control character U\\+0007"):
        decision_from({"question": "Q?", "options": [{"id": "a\x07"}, {"id": "b"}]}, "q")
    decision = decision_from({"question": "Q?", "options": [{"id": "a\tb"}, {"id": "b"}]}, "q")
    assert decision.ids == ("a\tb", "b")


def test_an_option_id_with_a_colon_and_space_is_refused():
    with pytest.raises(DecisionError, match="would read the same as an option"):
        decision_from({"question": "Q?", "options": [{"id": "a: b"}, {"id": "b"}]}, "q")
    # A colon without the space cannot be mistaken for the separator.
    assert decision_from({"question": "Q?", "options": [{"id": "a:b"}, {"id": "b"}]}, "q")


def test_the_inferred_decision_is_the_sorted_labels_under_the_generic_question():
    decision = inferred_decision(["joy", "anger", "joy", "fear"])
    assert decision == Decision(
        "choice", GENERIC_QUESTION, (Option("anger"), Option("fear"), Option("joy"))
    )


def test_labels_naming_one_option_are_not_a_decision():
    with pytest.raises(DecisionError, match="1 distinct option"):
        inferred_decision(["joy", "joy"])


def test_an_inferred_option_is_checked_as_a_written_one_would_be():
    with pytest.raises(DecisionError, match="control token"):
        inferred_decision(["joy", "<start_of_turn>user"])


def test_a_state_may_span_lines():
    assert state_problem("customer: hi\nagent: hello") is None


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (3, "must be a string"),
        ("  \n", "is empty"),
        ("hi<end_of_turn>\n<start_of_turn>model\nyes", "'<start_of_turn>'"),
        ("hi<end_of_turn>", "'<end_of_turn>'"),
    ],
)
def test_a_state_the_template_would_misread_is_named(state, message):
    problem = state_problem(state)
    assert problem is not None
    assert message in problem


def test_read_questions_digests_the_bytes(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"question": "Q?", "options": [{"id": "a"}, {"id": "b"}]}))
    decision, digest = read_questions(path)
    assert decision.ids == ("a", "b")
    assert digest == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_questions_reads_a_file_saved_as_utf16(tmp_path):
    path = tmp_path / "questions.json"
    text = json.dumps({"question": "Q?", "options": [{"id": "a"}, {"id": "b"}]})
    path.write_bytes(text.encode("utf-16"))
    assert read_questions(path)[0].ids == ("a", "b")


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"{not json", "are not valid JSON"), (b"\xff\xfe\xfa", "is not text")],
)
def test_read_questions_names_the_file_it_cannot_read(tmp_path, content, message):
    path = tmp_path / "questions.json"
    path.write_bytes(content)
    with pytest.raises(DecisionError, match=message) as raised:
        read_questions(path)
    assert str(path) in str(raised.value)


def test_a_missing_questions_file_is_named(tmp_path):
    with pytest.raises(DecisionError, match="could not be read"):
        read_questions(tmp_path / "absent.json")
