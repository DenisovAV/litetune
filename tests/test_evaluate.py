"""The evaluator and its two backends.

Nothing here starts a process or loads a model: `StageEnv.run` is monkeypatched
at the boundary, which is the only place either backend touches the outside
world.

The backend tests are mostly about one distinction -- a process that ran and
failed against a process that never ran. Collapsing the two is what produced
eight confident negatives during the measurement work, so each has its own test.
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeBackend, call_text, fake_torch, labelled_rows, mark_provisioned

from litetune import envs, toolpath
from litetune.evaluate import (
    BACKEND_OBSERVED,
    UNKNOWN_BACKEND,
    DataError,
    DecodeConfig,
    HuggingFaceBackend,
    LiteRtLmBackend,
    _litertlm_script,
    assemble_generations,
    device_mismatch,
    evaluate,
    gpu_observed,
    gpu_unused,
    harness_mismatch,
    load_split,
    read_jsonl_results,
    runtime_engine_spec,
)
from litetune.metrics import score_exact_text, trim_terminator
from litetune.prompt_mode import PromptMode

# -- the split --------------------------------------------------------------


def test_load_split_reads_prompts_and_targets(write_split):
    path = write_split(labelled_rows(3))
    split = load_split(path)
    assert split.n == 3
    assert len(split.labelled) == 3
    assert split.examples[0].target is not None
    assert split.examples[0].target.name == "change_background_color"


def test_examples_without_a_target_are_kept_but_unlabelled(write_split):
    path = write_split([{"prompt": "do a thing"}])
    split = load_split(path)
    assert split.n == 1
    assert split.labelled == ()


def test_limit_takes_the_first_n(write_split):
    path = write_split(labelled_rows(10))
    assert load_split(path, limit=4).n == 4


def test_split_identity_follows_content_not_location(write_split):
    a = load_split(write_split(labelled_rows(3), name="a.jsonl"))
    b = load_split(write_split(labelled_rows(3), name="b.jsonl"))
    assert a.id == b.id
    assert a.source != b.source


def test_a_limited_split_is_a_different_sample(write_split):
    # A 4-example slice is not evidence about the 10 it came from, so it must
    # not share their identity.
    path = write_split(labelled_rows(10))
    assert load_split(path).id != load_split(path, limit=4).id


def test_malformed_line_names_itself(write_split, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prompt": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DataError) as exc:
        load_split(path)
    assert ":2:" in str(exc.value)


def test_row_without_a_prompt_is_refused(write_split):
    with pytest.raises(DataError):
        load_split(write_split([{"target": {"name": "x", "args": {}}}]))


# -- litert-lm --------------------------------------------------------------


def _litertlm(tmp_path: Path, **kwargs) -> LiteRtLmBackend:
    return LiteRtLmBackend(model=tmp_path / "model.litertlm", auto_provision=False, **kwargs)


def _writes_results(rows, returncode: int = 0, stderr: str = "", device=None):
    """A stand-in for the driver script: writes the JSONL it would write.

    The transport is a file now, not a pipe, so a double that hands back
    stdout is testing a channel nothing reads. `args[2]` is the spec path, the
    same shape the reference backend's doubles use. `device`, when given, is
    the report the script writes about its own process after building the
    engine.
    """

    def fake_run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        if device is not None:
            Path(spec["report"]).write_text(json.dumps(device), encoding="utf-8")
        return subprocess.CompletedProcess(args, returncode, stdout="", stderr=stderr)

    return fake_run


def test_a_generation_with_bytes_that_did_not_decode_is_not_the_model_s_answer(
    monkeypatch, tmp_path
):
    """An undecodable byte is the runtime's failure, and scoring it is the bug one step on.

    The byte reaches here as a lone surrogate: the runtime handed the script a
    `str` with one in it, `json.dumps` escaped it as `\\udcXX`, and the file
    stayed valid UTF-8 the whole way -- so nothing raises, and the row would
    score as an ordinary wrong answer while every liveness check passed and
    the loss was reported as a conversion cost.
    """
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "\udcbf\udce5"}])
    )

    generation = _litertlm(tmp_path).generate(["hello"])[0]

    assert not generation.ok
    assert generation.harness_error is not None
    assert "not UTF-8" in generation.harness_error
    assert generation.text == "", "nothing of it is the model's answer, so none of it is scored"


def test_a_generation_that_really_contains_the_replacement_character_is_scored(
    monkeypatch, tmp_path
):
    """U+FFFD is an ordinary character, and a model may write it.

    It is in the vocabulary of every byte-level tokenizer, and a dataset that
    has been through one lossy decode is full of it. Refusing on U+FFFD --
    the first shape of this check, which a review caught -- would call such an
    answer a harness failure, which is the same class of wrong as scoring
    garbage, pointing the other way.
    """
    answer = "the file is named \ufffd, literally"
    monkeypatch.setattr(envs.StageEnv, "run", _writes_results([{"index": 0, "text": answer}]))

    generation = _litertlm(tmp_path).generate(["hello"])[0]

    assert generation.ok
    assert generation.harness_error is None
    assert generation.text == answer


def test_a_non_ascii_generation_survives_the_file_end_to_end(monkeypatch, tmp_path):
    """The claim this path rests on, through a real child and a real file.

    Every other test of this backend replaces `StageEnv.run` with a double
    that writes the results itself, so the encode/decode boundary -- where the
    generation actually is -- is never crossed. This one runs a real child
    that writes the file, with the host asking for an encoding that cannot
    hold the answer.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    answer = "перевод — 変換"

    def a_real_child(self, args, timeout: int = 3600, env=None):
        out = json.loads(Path(args[2]).read_text(encoding="utf-8"))["out"]
        code = (
            "import json,pathlib;"
            f"pathlib.Path({out!r}).write_text("
            f"json.dumps({{'index':0,'text':{answer!r}}}), encoding='utf-8')"
        )
        return envs._run_guarded([sys.executable, "-c", code], timeout=timeout, env=env)

    monkeypatch.setattr(envs.StageEnv, "run", a_real_child)

    generation = _litertlm(tmp_path).generate(["hello"])[0]

    assert generation.ran
    assert generation.text == answer


def test_the_runner_call_pins_the_prompt_construction_mode(tmp_path):
    """The distinction `--no-template` used to carry, said as the call it picks.

    `create_session(apply_prompt_template=False)` forces the runtime's tool
    list to null, so the prompt must arrive already rendered. The mode has to
    say so.
    """
    backend = _litertlm(tmp_path)

    assert backend.runner_call == "Engine.create_session(apply_prompt_template=False)"
    assert backend.prompt_mode is PromptMode.PRERENDERED
    record = backend.describe()
    assert record["backend"] == "cpu"
    assert record["transport"] == "litert_lm python api, one process per split"
    assert record["template_flag"] == "apply_prompt_template=False"


class _ScriptedConversation:
    """A conversation that yields the chunks it was given, and nothing else."""

    def __init__(self, chunks):
        self._chunks = chunks

    def send_message_async(self, prompt):
        return iter(self._chunks)


# What `litert-lm run` prints for each of these streams, on a pipe, after
# the `.strip()` the scorer sees. Written out rather
# than produced, because the CLI is not installed in the environment this suite
# runs in -- so the pin is on `litert_lm_cli/commands/run.py` at the version
# `envs.RUNTIME` pins, read at lines 50-53 (`close_channel` writes
# `" [/name]"` and a newline) and 108-125 (it is called before every text item,
# on a switch between channels, and at end of stream).
#
# The composition in the driver script is the one part of it that is not a
# direct call into litert-lm, so it is the one part that can be wrong by
# itself -- and it was: the first version opened a channel and never closed it.
# `metrics.REASONING_BLOCKS` keys on the closing marker, and `_split_reasoning`
# cuts at the *last* one, so without it the reasoning stays in the answer, the
# row is scored against thought-plus-answer, and the generation is counted as
# `unclosed` instead of `closed`. Nothing fails; the number moves.
CHANNEL_CASES = [
    (
        "text only",
        [{"content": [{"type": "text", "text": "hello"}]}],
        "hello",
    ),
    (
        "a channel and nothing else",
        [{"channels": {"thought": "abc"}}],
        "[thought] abc [/thought]",
    ),
    (
        "a channel, then the answer",
        [
            {"channels": {"thought": "abc"}},
            {"content": [{"type": "text", "text": "answer"}]},
        ],
        "[thought] abc [/thought]\nanswer",
    ),
    (
        "the same channel reopened after the answer",
        [
            {"channels": {"thought": "a"}},
            {"content": [{"type": "text", "text": "X"}]},
            {"channels": {"thought": "b"}},
        ],
        "[thought] a [/thought]\nX[thought] b [/thought]",
    ),
    (
        "two channels in turn",
        [{"channels": {"thought": "a"}}, {"channels": {"plan": "b"}}],
        "[thought] a [/thought]\n[plan] b [/plan]",
    ),
    (
        "one channel split across chunks",
        [{"channels": {"thought": "ab"}}, {"channels": {"thought": "cd"}}],
        "[thought] abcd [/thought]",
    ),
    (
        "a channel that carried nothing",
        [{"channels": {"thought": ""}}],
        "[thought]  [/thought]",
    ),
]


@pytest.mark.parametrize(
    ("name", "chunks", "expected"),
    CHANNEL_CASES,
    ids=[case[0].replace(" ", "-") for case in CHANNEL_CASES],
)
def test_the_driver_composes_what_the_cli_printed(name, chunks, expected):
    compose = _litertlm_script()["text_from_conversation"]

    assert compose(_ScriptedConversation(chunks), "p").strip() == expected


def test_a_timeout_keeps_the_answers_the_script_had_already_written(monkeypatch, tmp_path):
    """The promise the docstring makes, which only held for a kill.

    The script flushes a row per prompt so a run that dies partway keeps what
    it finished. The first version of this returned before reading the file,
    and the temp directory took the rows with it -- so 599 finished
    generations were thrown away because the six-hundredth hung.
    """

    def times_out_after_writing(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": "label_3"}) + "\n", encoding="utf-8"
        )
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", times_out_after_writing)

    answered, missing = _litertlm(tmp_path).generate(["one", "two"])

    assert answered.ok and answered.text == "label_3"
    assert answered.harness_error is None
    assert not missing.ran
    assert missing.harness_error is not None and "timeout" in missing.harness_error


@pytest.mark.parametrize(
    ("value", "shape"),
    [(None, "NoneType"), (12345, "int"), ({"a": 1}, "dict")],
    ids=["null", "number", "object"],
)
def test_a_row_whose_text_is_not_a_string_is_not_the_model_s_answer(
    monkeypatch, tmp_path, value, shape
):
    """`str(row["text"])` accepted all three, so the wrong-shape guard was inert.

    A JSON `null` came back as the four characters "None", scored as an
    ordinary wrong answer with every liveness check green and the loss folded
    into the conversion cost.
    """
    monkeypatch.setattr(envs.StageEnv, "run", _writes_results([{"index": 0, "text": value}]))

    generation = _litertlm(tmp_path).generate(["hello"])[0]

    assert not generation.ok
    assert generation.text == ""
    assert generation.harness_error is not None
    assert shape in generation.harness_error
    assert "not a string" in generation.harness_error


def test_two_rows_claiming_one_prompt_are_both_refused(monkeypatch, tmp_path):
    """The last row silently won, and nothing said the file had disagreed."""
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _writes_results([{"index": 0, "text": "first"}, {"index": 0, "text": "second"}]),
    )

    generation = _litertlm(tmp_path).generate(["hello"])[0]

    assert not generation.ok
    assert generation.text == ""
    assert "two rows" in (generation.harness_error or "")


def test_an_index_that_is_not_an_integer_claims_no_prompt(monkeypatch, tmp_path, caplog):
    """`int()` is a coercion, not a check.

    `int(1.9)` is 1 and `int(True)` is 1, so a row carrying either took prompt
    1's slot and was scored as its answer -- in the one function whose whole
    job is that a row which is not the model's answer does not reach the score.
    """
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _writes_results(
            [
                {"index": 0, "text": "real"},
                {"index": 1.9, "text": "a float claiming prompt 1"},
                {"index": True, "text": "a bool claiming prompt 1"},
            ]
        ),
    )

    generations = _litertlm(tmp_path).generate(["zero", "one"])

    assert generations[0].text == "real"
    # Prompt 1 got no row of its own, and neither impostor became one.
    assert not generations[1].ran
    assert "a float" not in (generations[1].text or "")
    assert "a bool" not in (generations[1].text or "")


def test_the_reader_never_returns_a_prompt_in_both_maps(tmp_path):
    """One prompt is either an answer or a fault, never both.

    Every caller checks `faults` first, so a text left standing beside a fault
    is invisible until something reads `texts` alone -- and a file can disagree
    with itself in more ways than a plain duplicate: a row with the text and a
    second row with an error for the same prompt used to arrive as a fault with
    the text still there.
    """
    results = tmp_path / "out.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps({"index": 0, "text": "an answer"}),
                json.dumps({"index": 0, "error": "RuntimeError: and also a refusal"}),
                json.dumps({"index": 1, "text": "untroubled"}),
            ]
        ),
        encoding="utf-8",
    )

    texts, faults = read_jsonl_results(results)

    assert not (texts.keys() & faults.keys())
    assert 0 in faults
    assert texts == {1: "untroubled"}


def test_a_row_no_prompt_claims_is_reported(monkeypatch, tmp_path, caplog):
    """It used to vanish -- including a fault already logged at ERROR.

    What binds row N to prompt N is an integer in a file and nothing else, so
    a row nobody claims is the visible end of that binding going wrong.
    """
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _writes_results([{"index": 0, "text": "mine"}, {"index": 7, "text": "nobody's"}]),
    )

    with caplog.at_level("ERROR"):
        generations = _litertlm(tmp_path).generate(["hello"])

    assert len(generations) == 1
    assert generations[0].text == "mine"
    assert any("index no prompt has" in r.getMessage() for r in caplog.records)


def test_a_timed_out_split_does_not_look_like_a_clean_one(monkeypatch, tmp_path):
    """Salvaged rows are scored, and they have to say the run was killed.

    Given `returncode=0` and nothing else, a split killed after writing every
    row was byte-identical to a clean one and liveness reported "n/n
    generations exited zero" about a group that was SIGKILLed. `_run_guarded`
    puts the child's last words on the exception for exactly this, and the
    first version of the salvage threw them away.
    """

    def times_out_after_writing_everything(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(
            "".join(json.dumps({"index": i, "text": f"answer {i}"}) + "\n" for i in range(2)),
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(
            cmd=args, timeout=timeout, stderr="killed while decoding prompt 2"
        )

    monkeypatch.setattr(envs.StageEnv, "run", times_out_after_writing_everything)

    generations = _litertlm(tmp_path).generate(["one", "two"])

    assert [g.text for g in generations] == ["answer 0", "answer 1"]
    assert all(g.ok for g in generations), "a finished row is still a real answer"
    assert all(
        g.batch_returncode not in (None, 0) for g in generations
    ), "a killed run must not read as a clean one"
    assert all("killed while decoding" in g.stderr for g in generations)


def test_a_stray_row_is_reported_when_the_split_was_killed_too(monkeypatch, tmp_path, caplog):
    """The salvage used to walk the prompts and never look at the leftovers.

    On the clean path a row whose index no prompt claims is logged at ERROR.
    The timeout path iterated `enumerate(prompts)` only, so the same row --
    written by a child that was about to be killed, which is when the binding
    between rows and prompts is least trustworthy -- disappeared without a
    word.
    """

    def times_out_after_writing_a_stray(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": "mine"})
            + "\n"
            + json.dumps({"index": 9, "text": "nobody's"})
            + "\n",
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout, stderr="killed")

    monkeypatch.setattr(envs.StageEnv, "run", times_out_after_writing_a_stray)

    with caplog.at_level("ERROR"):
        generations = _litertlm(tmp_path).generate(["one"])

    assert [g.text for g in generations] == ["mine"]
    assert any("index no prompt has" in r.getMessage() for r in caplog.records)


def test_a_prompt_the_runtime_refuses_does_not_take_the_split_with_it(monkeypatch, tmp_path):
    """One row, not the rest of the run.

    The runtime raises for a prefill that will not fit and for any stream
    error it does not recognise. Without containment the first of those ends
    the split: prompt 1 raises and everything after it comes back as "the
    script exited 1".

    The CLI contained it differently and worse -- it caught everything at the
    top of the command, printed "An error occurred" to stdout and exited 0, so
    the old transport scored that sentence as the model's answer.
    """
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _writes_results(
            [
                {"index": 0, "text": "first"},
                {"index": 1, "error": "RuntimeError: prefill failed"},
                {"index": 2, "text": "third"},
            ]
        ),
    )

    first, refused, third = _litertlm(tmp_path).generate(["a", "b", "c"])

    assert first.ok and first.text == "first"
    assert third.ok and third.text == "third", "the split carried on past the failure"
    assert not refused.ok
    assert refused.text == "", "an error is not an answer"
    assert "prefill failed" in (refused.harness_error or "")


def test_one_process_per_split(monkeypatch, tmp_path):
    """The point of the transport: the whole split in one process.

    It used to be one per prompt: six hundred process starts and six hundred
    bundle reloads for a 600-row verify. What that cost is not measured here.
    """
    seen = []
    write = _writes_results([{"index": 0, "text": "a"}, {"index": 1, "text": "b"}])

    def fake_run(self, args, timeout=3600, **kwargs):
        seen.append(args)
        return write(self, args, timeout=timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gens = _litertlm(tmp_path).generate(["one", "two"])

    assert len(seen) == 1, "one process, not one per prompt"
    assert [g.text for g in gens] == ["a", "b"]
    assert all(g.ok for g in gens)


def test_a_missing_binary_is_not_a_model_failure(monkeypatch, tmp_path):
    def fake_run(self, args, timeout=3600, **kwargs):
        raise FileNotFoundError("litert-lm")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is not None
    assert not gen.ran
    assert gen.returncode is None


def test_a_timeout_is_recorded_as_unperformed(monkeypatch, tmp_path):
    def fake_run(self, args, timeout=3600, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is not None
    assert "timeout" in gen.harness_error


def test_a_missing_system_library_is_unperformed_not_failed(monkeypatch, tmp_path):
    # litert-lm links vulkan even for the CPU backend; without it every
    # invocation dies in under a second and looks exactly like a dead model.
    def fake_run(self, args, timeout=3600, **kwargs):
        return subprocess.CompletedProcess(
            args, 127, stdout="", stderr="error while loading shared libraries: libvulkan.so.1"
        )

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is not None
    assert not gen.ran


def test_a_script_that_answered_and_then_failed_keeps_its_answers(monkeypatch, tmp_path):
    """What a non-zero exit means moved with the transport, and it had to.

    One process per prompt let "this prompt produced nothing and the process
    exited 1" be an observation about the model. One process per split cannot:
    a script that exits non-zero having written nothing has broken, and
    reporting that as 600 models declining to answer would be the reverse of
    the error this file exists to prevent. So a row that arrived is kept and
    scored, with the failure travelling beside it, and a prompt with no row is
    a harness error.
    """
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _writes_results([{"index": 0, "text": "label_3"}], returncode=1, stderr="died at 2"),
    )

    answered, missing = _litertlm(tmp_path).generate(["one", "two"])

    assert answered.ran and answered.ok
    assert answered.text == "label_3"
    assert answered.batch_returncode == 1, "the failure travels with the answer"
    assert missing.harness_error is not None and "exited 1" in missing.harness_error


def test_backend_reports_which_engine_produced_the_numbers(tmp_path):
    described = _litertlm(tmp_path).describe()
    assert described["engine"] == "litert-lm"
    assert described["backend"] == "cpu"
    assert "litert-lm==0.16.1" in described["requirements"]


# -- transformers -----------------------------------------------------------


def test_hugging_face_backend_runs_the_whole_split_in_one_process(monkeypatch):
    calls = []

    def fake_run(self, args, timeout=3600, **kwargs):
        calls.append(args)
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            "\n".join(
                json.dumps({"index": i, "text": call_text("a", i=str(i))})
                for i in range(len(spec["prompts"]))
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gens = HuggingFaceBackend(model="org/model", auto_provision=False).generate(["a", "b", "c"])
    assert len(calls) == 1
    assert [g.ok for g in gens] == [True, True, True]
    assert gens[2].text == call_text("a", i="2")


def test_a_generation_script_that_dies_reports_unperformed(monkeypatch):
    def fake_run(self, args, timeout=3600, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="ModuleNotFoundError: torch")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gens = HuggingFaceBackend(model="org/model", auto_provision=False).generate(["a"])
    assert gens[0].harness_error is not None
    assert not gens[0].ran


def test_a_reference_generation_reaches_scoring_with_its_terminator_intact(tmp_path):
    """The HF script keeps the terminator on purpose; nothing between the
    results file and the scorer may strip it, or liveness goes blind to
    leakage. A transport mutation stripping `<eos>` in `read_jsonl_results`
    survived the whole suite before this test existed.
    """
    results = tmp_path / "r.jsonl"
    results.write_text('{"index": 0, "text": "label_3<end_of_turn>\\n<eos>"}\n', encoding="utf-8")
    texts, _ = read_jsonl_results(results)
    assert texts[0] == "label_3<end_of_turn>\n<eos>"
    assert trim_terminator(texts[0]) == "label_3"
    assert score_exact_text(["label_3"], [texts[0]]).exact_match.value == 1.0


def test_turning_on_the_chat_template_changes_the_measured_mode():
    plain = HuggingFaceBackend(model="m", auto_provision=False)
    templated = HuggingFaceBackend(model="m", auto_provision=False, runtime_rendered=True)
    assert plain.prompt_mode is PromptMode.PRERENDERED
    assert templated.prompt_mode is PromptMode.RUNTIME_RENDERED


def test_hugging_face_backend_reports_an_unknown_device_before_it_has_run():
    # Not "cpu": a manifest read before `generate()` ran must not claim a
    # device nothing measured yet. "unknown" and not a second word for it --
    # `verify.py` already prints the `unknown` fallback for the
    # same state when the key is missing entirely.
    described = HuggingFaceBackend(model="org/m", auto_provision=False).describe()
    assert described["backend"] == UNKNOWN_BACKEND == "unknown"


def test_the_two_backends_say_which_vocabulary_their_backend_field_is_in(tmp_path):
    """One key, two answers to two different questions.

    `backend` is a torch device on the reference side and the word litert-lm's
    `Backend` takes on the candidate side. They overlap at exactly one string,
    "cpu", where they mean different things -- so each `describe()` says which
    question it answered.

    It used to say "litert-lm --backend flag", which named a command line this
    backend has not built since the transport moved to the Python API, and
    `toolpath` -- the other backend that speaks this vocabulary -- said
    something else. One spelling, and it is the one that is true.
    """
    reference = HuggingFaceBackend(model="org/m", auto_provision=False).describe()
    candidate = _litertlm(tmp_path).describe()
    assert reference["backend_vocabulary"] == "torch device"
    assert candidate["backend_vocabulary"] == "litert-lm Python API Backend"
    assert reference["backend_vocabulary"] != candidate["backend_vocabulary"]
    # The other backend that reaches the same runtime the same way. Two
    # spellings of one vocabulary is how a reader concludes they are two.
    tool_path = (Path(toolpath.__file__).read_text(encoding="utf-8")).count(
        '"backend_vocabulary": "litert-lm Python API Backend"'
    )
    assert tool_path == 1, "toolpath.py no longer spells the vocabulary the same way"


def test_hugging_face_backend_reports_the_device_it_actually_used(monkeypatch):
    """Used to hardcode "cpu" unconditionally, which made a laptop's manifest
    and a GPU box's byte-identical in the one field meant to tell them apart.
    """

    def fake_provision(self, events=None, force=False):
        # Marker and interpreter both: `_ensure_env` gates the probe on
        # `env.ready`, and a directory with only the marker is not ready.
        mark_provisioned(self)
        return self.path

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(
        envs,
        "resolve_device",
        lambda env, timeout=30, events=None: envs.DeviceProbe(device="cuda", detail="fake"),
    )

    def fake_run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        Path(spec["run_report"]).write_text(json.dumps({"device": "cuda"}), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)

    backend = HuggingFaceBackend(model="org/model")
    backend.generate(["a"])

    assert backend.describe()["backend"] == "cuda"


def _ready_env() -> None:
    """Make `envs.TRAIN` look provisioned without provisioning anything.

    `_ensure_env` gates the device probe on `env.ready`, which is the marker
    file and the interpreter beside it.
    """
    mark_provisioned(envs.TRAIN)


def _generating_env(monkeypatch, *, probe: str | None, script_device: str | None) -> list:
    """A stage environment that answers the probe and then generates.

    Returns the argv of every call, so a test can assert that the probe
    happened -- or did not.
    """
    calls: list = []

    def fake_run(self, args, timeout=3600, **kwargs):
        calls.append(list(args))
        if args[1] == "-c":
            if probe is None:
                return subprocess.CompletedProcess(args, 1, "", "no torch")
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps({"device": probe, "cuda_build": None, "device_count": 0}),
                "",
            )
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        if script_device is not None:
            Path(spec["run_report"]).write_text(
                json.dumps({"device": script_device}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    return calls


def test_the_probe_runs_over_a_ready_environment_nobody_asked_to_provision(monkeypatch):
    """The probe provisions nothing, so readiness is its real precondition.

    Gated on `auto_provision` instead, a caller that manages the environment's
    lifecycle itself -- a library caller, and every backend in this file --
    reported `describe()["backend"] == "unknown"` for a run whose device was
    perfectly knowable. No CLI path reaches that state: `verify` has no
    `--no-provision` flag.
    """
    _ready_env()
    calls = _generating_env(monkeypatch, probe="cuda", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert [c[1] for c in calls][0] == "-c"
    assert backend.describe()["backend"] == "cuda"


def test_no_probe_against_an_environment_that_is_not_there(monkeypatch):
    """An environment with no marker is one nothing can be asked of."""
    calls = _generating_env(monkeypatch, probe="cuda", script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert all(c[1] != "-c" for c in calls)
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_the_script_reports_the_device_it_used_and_that_answer_wins(monkeypatch):
    """The probe is a prediction; the run report is the observation.

    They differ whenever the script took its own fallback. `tune.py` has had
    the observation all along through `metrics.device`; the reference side
    predicted "cpu" and published it as though it had watched.
    """
    _ready_env()
    _generating_env(monkeypatch, probe="cpu", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert backend.device == "cuda"
    assert backend.describe()["backend"] == "cuda"
    # Both, not only the winner. A manifest that records "cuda" and nothing
    # else cannot show that the prediction was wrong, and the logger line that
    # says so does not travel with the measurement.
    assert backend.probed_device == "cpu"
    assert backend.describe()["backend_probed"] == "cpu"


def test_a_run_that_used_what_it_was_told_records_the_two_as_equal(monkeypatch):
    _ready_env()
    _generating_env(monkeypatch, probe="cuda", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    described = backend.describe()
    assert described["backend"] == described["backend_probed"] == "cuda"


def test_a_script_that_wrote_no_report_leaves_the_prediction_standing(monkeypatch):
    """A run that died before writing it, or an older script that never did.
    The prediction is still the best thing known, and `None` would be worse."""
    _ready_env()
    _generating_env(monkeypatch, probe="cuda", script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert backend.device == "cuda"


def test_a_run_report_that_is_not_valid_utf8_leaves_the_prediction_standing(tmp_path):
    """Same family as the blocker: `read_jsonl_results` a few lines below already
    catches `UnicodeDecodeError` on a damaged results file; `_read_run_report`
    used to let it escape, which would have aborted verification after the
    generations that report was only ever supposed to annotate had already
    been produced."""
    report = tmp_path / "run.json"
    report.write_bytes(b"\xff\xfe{not json")
    device = HuggingFaceBackend(model="org/m", auto_provision=False)._read_run_report(report)
    assert device is None


def test_a_failed_probe_does_not_erase_a_device_already_known(monkeypatch):
    """Reuse. The previous run's answer is a better record than `None`, and
    `None` would claim the device was never established when it was."""
    _ready_env()
    _generating_env(monkeypatch, probe=None, script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False, device="cuda")
    backend.generate(["a"])

    assert backend.device == "cuda"


def test_a_timeout_clears_the_device_it_never_confirmed(monkeypatch):
    """The probe answered, but the generation subprocess it precedes never
    finished: nothing ran, so `describe()` must not report the probe's
    prediction as though the run had confirmed it."""
    _ready_env()

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"device": "cuda", "cuda_build": None, "device_count": 1}), ""
            )
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    gens = backend.generate(["a"])

    assert gens[0].harness_error is not None
    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_a_script_that_could_not_start_clears_the_device_it_never_confirmed(monkeypatch):
    _ready_env()

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"device": "cuda", "cuda_build": None, "device_count": 1}), ""
            )
        raise FileNotFoundError("python")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    gens = backend.generate(["a"])

    assert gens[0].harness_error is not None
    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_a_failed_probe_does_not_force_a_stale_device_onto_the_child(monkeypatch):
    """The do-not-erase rule above protects the *report*; it must not also
    hand the *child* a directive its own probe could not vouch for. Before
    this was fixed, `spec["device"]` read `self.device` -- the previous run's
    "cuda" -- even though this call's probe could not answer, which would make
    `model.to("cuda")` fail on a box where CUDA had since gone away."""
    _ready_env()
    specs: list[dict] = []

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            # No torch: the probe cannot answer this call.
            return subprocess.CompletedProcess(args, 1, "", "no torch")
        specs.append(json.loads(Path(args[2]).read_text()))
        Path(json.loads(Path(args[2]).read_text())["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False, device="cuda")
    backend.generate(["a"])

    assert backend.device == "cuda", "the report still remembers the previous answer"
    assert len(specs) == 1
    assert (
        specs[0]["device"] is None
    ), "a probe that could not answer must not be forced onto the child"


def test_a_blocked_environment_clears_the_device(monkeypatch):
    """Nothing ran, so nothing has a device. The opposite of the case above:
    there is no result here for a stale answer to describe."""

    def explode(self, events=None, force=False):
        raise RuntimeError("no interpreter")

    monkeypatch.setattr(envs.StageEnv, "provision", explode)

    backend = HuggingFaceBackend(model="org/model", device="cuda")
    gens = backend.generate(["a"])

    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND
    assert gens[0].harness_error is not None


def test_a_blocked_run_does_not_inherit_the_previous_runs_reading(monkeypatch):
    """Found by review: the reset sat after the blocked return.

    One run reads its device back; the next cannot provision. `_ensure_env`
    clears `device`, and a `True` left over from the first run would then sit
    beside `UNKNOWN_BACKEND` claiming this run read something back.
    """
    _ready_env()
    _generating_env(monkeypatch, probe="cpu", script_device="cuda")
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])
    assert backend.describe()[BACKEND_OBSERVED] is True

    def explode(self, events=None, force=False):
        raise RuntimeError("no interpreter")

    monkeypatch.setattr(envs.StageEnv, "provision", explode)
    backend.auto_provision = True
    gens = backend.generate(["b"])

    assert gens[0].harness_error is not None
    described = backend.describe()
    assert described["backend"] == UNKNOWN_BACKEND
    assert described[BACKEND_OBSERVED] is False


# -- comparing two points that ran on different hardware ---------------------


def _point(label: str, backend: str, engine: str = "transformers", observed: bool = False):
    from litetune.evaluate import GREEDY, MeasurementPoint

    return MeasurementPoint(
        label=label,
        model_ref="org/m",
        backend=engine,
        prompt_mode=PromptMode.PRERENDERED,
        decode=GREEDY,
        split_id="s",
        engine={"engine": engine, "backend": backend, BACKEND_OBSERVED: observed},
        decode_enforced=True,
    )


def test_a_side_that_read_nothing_back_is_said_to_have_asked(write_split=None):
    """One sentence, two verbs, because the two sides have different evidence.

    The reference reads its device out of its own run report; the runtime side
    passes a flag and is told nothing. Giving both "was measured on" attributes
    part of a score gap to hardware that, on the flag side, nobody established
    served the run -- and on an accelerator that is the failure this whole key
    exists for, since an engine built for an absent GPU neither raises nor
    answers.
    """
    note = device_mismatch(
        _point("candidate", "gpu", engine="litert-lm", observed=False),
        _point("reference", "cuda", observed=True),
    )
    assert note is not None
    assert "candidate asked for gpu" in note
    assert "reference was measured on cuda" in note
    # And the conclusion softens with them.
    assert "may carry a hardware difference" in note


def test_two_sides_that_both_read_back_are_said_to_have_measured():
    note = device_mismatch(
        _point("candidate", "gpu", engine="litert-lm", observed=True),
        _point("reference", "cuda", observed=True),
    )
    assert note is not None
    assert "candidate was measured on gpu" in note and "reference was measured on cuda" in note
    assert "carries a hardware difference" in note


def test_two_points_on_different_hardware_are_annotated_not_refused():
    """By default the candidate runs on litert-lm's CPU backend and the
    reference resolves its own device, so on a GPU box the two differ and
    the conversion cost carries a hardware difference. Refusing would leave
    such a machine unable to verify at all; the number is kept and told what
    is in it.
    """
    note = device_mismatch(
        _point("candidate", "cpu", engine="litert-lm"), _point("reference", "cuda")
    )
    assert note is not None
    assert "cpu" in note and "cuda" in note
    assert "candidate" in note and "reference" in note
    # Still comparable: this is a limitation, not a refusal.
    assert (
        harness_mismatch(
            _point("candidate", "cpu", engine="litert-lm"), _point("reference", "cuda")
        )
        is None
    )


def test_two_points_on_the_same_device_are_not_annotated():
    assert (
        device_mismatch(_point("candidate", "cpu", engine="litert-lm"), _point("reference", "cpu"))
        is None
    )


def test_an_unestablished_device_is_not_a_difference():
    """Nothing was established to differ from, and `describe()` already says
    so in the same field."""
    assert (
        device_mismatch(
            _point("candidate", "cpu", engine="litert-lm"),
            _point("reference", UNKNOWN_BACKEND),
        )
        is None
    )


# -- the evaluator ----------------------------------------------------------


def test_measurement_point_records_what_produced_it(write_split):
    split = load_split(write_split(labelled_rows(2)))
    point = evaluate(FakeBackend(texts=["call:a{}"]), split, label="candidate")
    assert point.label == "candidate"
    assert point.split_id == split.id
    assert point.prompt_mode is PromptMode.PRERENDERED
    assert point.n == 2


def test_a_backend_that_drops_a_prompt_is_a_contract_violation(write_split):
    class ShortBackend(FakeBackend):
        def generate(self, prompts, events=None):
            return super().generate(prompts[:-1])

    split = load_split(write_split(labelled_rows(3)))
    with pytest.raises(ValueError):
        evaluate(ShortBackend(texts=["call:a{}"]), split, label="candidate")


def test_comparison_across_prompt_modes_is_refused(write_split):
    # --no-template forces the tool list to null, so the two modes differ by the
    # whole declaration block: their difference measures the mode, not the model.
    split = load_split(write_split(labelled_rows(2)))
    a = evaluate(FakeBackend(texts=["call:a{}"]), split, label="a")
    b = evaluate(
        FakeBackend(texts=["call:a{}"], prompt_mode=PromptMode.RUNTIME_RENDERED), split, label="b"
    )
    reason = harness_mismatch(a, b)
    assert reason is not None
    assert "mode" in reason


def test_comparison_across_splits_is_refused(write_split):
    a = evaluate(FakeBackend(texts=["call:a{}"]), load_split(write_split(labelled_rows(2))), "a")
    b = evaluate(
        FakeBackend(texts=["call:a{}"]),
        load_split(write_split(labelled_rows(2)[:1] + [{"prompt": "other"}], name="o.jsonl")),
        "b",
    )
    assert harness_mismatch(a, b) is not None


def test_comparison_across_decoding_is_refused(write_split):
    split = load_split(write_split(labelled_rows(2)))
    a = evaluate(FakeBackend(texts=["call:a{}"]), split, "a")
    b = evaluate(FakeBackend(texts=["call:a{}"], decode=DecodeConfig(max_tokens=8)), split, "b")
    assert harness_mismatch(a, b) is not None


def test_equivalent_measurements_are_comparable(write_split):
    split = load_split(write_split(labelled_rows(2)))
    a = evaluate(FakeBackend(texts=["call:a{}"]), split, "a")
    b = evaluate(FakeBackend(model="other", texts=["call:b{}"]), split, "b")
    assert harness_mismatch(a, b) is None


def test_attention_implementation_is_threaded_into_the_generate_script():
    """A float reference on sdpa against a model trained on eager is a harness
    difference wearing a conversion cost's clothes. The reference notebooks warn
    about this mismatch three separate times."""
    from litetune.evaluate import _HF_GENERATE_SCRIPT, HuggingFaceBackend

    assert 'attn_implementation=spec["attn_implementation"]' in _HF_GENERATE_SCRIPT
    assert HuggingFaceBackend(model="m").attn_implementation == "eager"


def test_a_script_that_failed_after_writing_results_is_not_recorded_as_clean():
    """Results plus a non-zero exit used to produce `returncode=0, stderr=""`.

    The generation did happen, so it keeps its text and stays scoreable — but
    erasing the process failure meant a reference side could be reported healthy
    while its environment had died on the way out.
    """
    import subprocess

    proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="teardown blew up")

    out = assemble_generations(["p0", "p1"], {0: "a", 1: "b"}, proc)

    assert [g.text for g in out] == ["a", "b"]
    # Typed, not buried in a message: `returncode` stays 0 so the generation is
    # still scoreable, and the batch's exit is its own field so a consumer can
    # see it without parsing prose.
    assert all(g.batch_returncode == 1 for g in out)
    assert all(g.ok for g in out)


def test_a_failed_batch_is_counted_where_something_reads_it():
    """A typed field nothing reads is the erasure it replaced, with a type.

    `batch_returncode` carried the evidence and no production caller looked at
    it: liveness sees `.ok`, and the manifest did not serialise it. The count
    is what `verify` turns into a limitation.
    """
    import subprocess

    from litetune.evaluate import GREEDY, MeasurementPoint

    proc = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr="died")
    generations = assemble_generations(["p0", "p1"], {0: "a", 1: "b"}, proc)

    point = MeasurementPoint(
        label="candidate",
        model_ref="m",
        backend="fake",
        prompt_mode=PromptMode.PRERENDERED,
        decode=GREEDY,
        split_id="s",
        engine={},
        generations=tuple(generations),
        decode_enforced=True,
    )

    assert point.batch_failures == 2
    assert point.as_dict()["generations"]["from_a_failed_batch"] == 2


def test_the_reference_decoder_keeps_the_terminator_scoring_trims():
    """The premise the whole normalisation rests on, pinned in the module that
    makes it. `skip_special_tokens=False` is deliberate -- the liveness tier
    needs to see leakage -- and it is why every reference generation carries a
    marker scoring has to remove. Flip it and the leakage check goes blind
    while the trimming becomes dead code, with nothing else noticing: the whole
    suite stayed green through exactly that mutation.
    """
    from litetune.evaluate import _HF_GENERATE_SCRIPT

    # The call, not the word. An earlier version of this test searched for the
    # bare string and passed against a flipped call, because the comment above
    # it explaining the choice contains the same text.
    assert "skip_special_tokens=False)" in _HF_GENERATE_SCRIPT
    assert "skip_special_tokens=True" not in _HF_GENERATE_SCRIPT


class _GenerationStopped(RuntimeError):
    """Stands in for a reference run that dies once the model is placed.

    The OOM killer is the realistic cause -- this project has been killed at
    32 GiB more than once -- but any death after placement and before the last
    completion has the same shape, and the script's run report exists to
    survive it.
    """


def _run_hf_generate_script(
    tmp_path: Path,
    monkeypatch,
    *,
    dtype: str = "bfloat16",
    cuda: bool = False,
    given=None,
    stop_generation: bool = False,
    runtime_rendered: bool = False,
) -> dict:
    """Runs `_HF_GENERATE_SCRIPT`'s real `main()` against faked `torch` and
    `transformers`, and returns what the fakes captured.

    `torch` and `transformers` are faked at the import boundary -- the same
    pattern `test_tune.py`'s `script_namespace` fixture uses to test the
    training script's body without a real torch. `cuda` and `given` are the
    two things that decide where the script places the model: `cuda` is what
    the fake `torch.cuda.is_available()` answers, `given` is `spec["device"]`,
    what the parent already resolved. `FakeInputIds` and `FakeEncoding` track
    their own device and hand back a new object when `.to()` actually moves
    them, the way a real tensor does -- a double that returns `self`
    unconditionally cannot tell a dropped `.to()` call from one that ran.
    """
    import sys
    import types

    from litetune.evaluate import _HF_GENERATE_SCRIPT

    captured: dict = {}

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def to(self, device: str) -> "FakeModel":
            # The script places the model before generating. Recorded rather
            # than ignored so a reader can see this double is standing in for
            # a real move, not silently swallowing one.
            captured["model_device"] = device
            return self

        def generate(self, **kwargs: object) -> list[list[int]]:
            captured["generate_input_ids_device"] = getattr(kwargs["input_ids"], "device", None)
            if stop_generation:
                raise _GenerationStopped("the run stopped after the model was placed")
            return [[0, 1, 2, 3]]

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model: str, torch_dtype: object, attn_implementation: str) -> FakeModel:
            captured["torch_dtype"] = torch_dtype
            return FakeModel()

    class FakeInputIds(list):
        device: str = "cpu"

        @property
        def shape(self) -> tuple[int, int]:
            return (1, len(self[0]))

        def to(self, device: str) -> "FakeInputIds":
            if device == self.device:
                return self
            moved = FakeInputIds(self)
            moved.device = device
            return moved

    class FakeEncoding(dict):
        """What the tokenizer hands back, which the script places on a device.

        A plain dict has no `.to`, and the script calls it -- so a double that
        is a bare dict makes this test fail for a reason that has nothing to do
        with the dtype or device it exists to pin.
        """

        device: str = "cpu"

        def to(self, device: str) -> "FakeEncoding":
            if device == self.device:
                return self
            moved = FakeEncoding(
                {k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()}
            )
            moved.device = device
            return moved

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        bos_token_id = 2

        def __call__(
            self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True
        ) -> "FakeEncoding":
            # One id per word, `<bos>` read as the BOS id, and the tokenizer's
            # own BOS in front unless told not to -- Gemma 3's tokenizer does both.
            ids = [
                self.bos_token_id if word == "<bos>" else 10 + n
                for n, word in enumerate(text.split())
            ]
            if add_special_tokens:
                ids = [self.bos_token_id, *ids]
            captured.setdefault("tokenized", []).append(ids)
            return FakeEncoding({"input_ids": FakeInputIds([ids])})

        def apply_chat_template(
            self, messages: list, tokenize: bool = False, add_generation_prompt: bool = False
        ) -> str:
            return "<bos> user " + messages[0]["content"] + " model"

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            return "generated"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str) -> FakeTokenizer:
            return FakeTokenizer()

    # `Any`: a real module has none of these attributes declared, and this
    # double exists to be assigned onto dynamically -- same as the untyped
    # test bodies elsewhere in this suite that `check_untyped_defs = false`
    # leaves unchecked, made explicit here because this function is typed.
    fake_torch_module: Any = types.ModuleType("torch")
    fake_torch_module.float32 = "sentinel:float32"
    fake_torch_module.bfloat16 = "sentinel:bfloat16"
    fake_torch_module.no_grad = __import__("contextlib").nullcontext
    # `fake_torch` from `conftest.py`: the one shape a fake `torch.cuda` takes
    # in this suite, reused rather than a fourth hand-rolled `is_available`.
    fake_torch_module.cuda = fake_torch(cuda=cuda).cuda

    fake_transformers: Any = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeAutoTokenizer

    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "out.jsonl"
    report_path = tmp_path / "run.json"
    spec_path.write_text(
        json.dumps(
            {
                "model": "org/m",
                "prompts": ["hi"],
                "max_tokens": 8,
                "runtime_rendered": runtime_rendered,
                "attn_implementation": "eager",
                "dtype": dtype,
                "device": given,
                "out": str(out_path),
                "run_report": str(report_path),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["generate.py", str(spec_path)])

    # `exec` on this module's own `_HF_GENERATE_SCRIPT` constant, not on
    # external input.
    namespace: dict = {"__name__": "litetune_generate_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "generate_script.py", "exec"), namespace)
    if stop_generation:
        with pytest.raises(_GenerationStopped):
            namespace["main"]()
    else:
        namespace["main"]()
    # The run report is the script's structured answer about its own device,
    # and it is what the backend reads back -- so the fixture returns it
    # beside what the torch doubles captured. Whether it exists at all is
    # reported separately, because a run that stopped early is exactly the
    # case where the answer is worth having and the file might be missing.
    captured["run_report_written"] = report_path.exists()
    if report_path.exists():
        captured["run_report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return captured


def test_the_run_report_is_written_before_any_generation_runs(tmp_path, monkeypatch):
    """Pins the *ordering*, which the comment beside the write claims and no
    other test enforces.

    Moving that write to after the generation loop -- present, otherwise
    identical -- passes every other test in this suite, because they all read
    the report from a run that finished. Deleting the write is caught; writing
    it late is not. The difference only shows on a run that stops part-way,
    and that is the run whose device is hardest to recover afterwards: the
    parent falls back to its own prediction, or to `unknown`, for a reference
    that had demonstrably reached a device and started work.

    So the model is placed, the first `generate` raises, and the report must
    already be on disk with the right answer in it.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=True, stop_generation=True)

    assert captured["model_device"] == "cuda"
    assert captured["run_report_written"], (
        "the run report must exist once the model is placed, not only once "
        "generation has finished"
    )
    assert captured["run_report"]["device"] == "cuda"


def test_the_float_reference_loads_at_float32_regardless_of_spec_dtype(tmp_path, monkeypatch):
    """S7: pins the *fact*, not the string. `tune.py`'s comments and
    limitations quote the prose "evaluate.py's float reference loads at an
    unconditional float32 regardless of the training dtype" -- three tests
    assert that prose, and all three survive changing the hardcoded
    `torch_dtype=torch.float32` to `torch_dtype=getattr(torch,
    spec.get("dtype", "float32"))`, because nothing writes a "dtype" key into
    the spec `HuggingFaceBackend` builds, so the mutation is inert against
    every fixture that goes through the backend.

    This runs the script's actual `main()` against a spec that carries a
    "dtype" key anyway (the backend does not have to write one for this to
    matter -- only the script's own indifference to it does), and asserts the
    dtype the fake model loader actually received. No GPU and no `given`
    keeps this test about the dtype: the device choice is the tests below's
    to make.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, dtype="bfloat16")
    assert captured["torch_dtype"] == "sentinel:float32"


def test_the_reference_script_generates_on_cuda_when_there_is_one():
    """Falls back to asking torch only when the parent could not answer."""
    from litetune.evaluate import _HF_GENERATE_SCRIPT

    namespace: dict = {"__name__": "litetune_hf_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf_generate.py", "exec"), namespace)

    assert namespace["generation_device"](fake_torch(cuda=True)) == "cuda"
    assert namespace["generation_device"](fake_torch(cuda=False)) == "cpu"


def test_the_reference_script_moves_everything_to_the_device_it_resolves(tmp_path, monkeypatch):
    """Kills three mutants at once, all invisible against a cuda-less fixture:
    `device = generation_device(...)` hardcoded to `"cpu"`, `model.to(device)`
    dropped or aimed at `"meta"`, and `.to(device)` dropped off the encoding
    before `model.generate(**enc)`. Each leaves the model and its inputs on
    different devices, which only shows up once the fake `torch` can say
    there is a GPU.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=True)
    assert captured["model_device"] == "cuda"
    assert captured["generate_input_ids_device"] == "cuda"


def test_the_reference_script_writes_its_device_where_the_parent_can_read_it(tmp_path, monkeypatch):
    """Structured, not only printed to stderr -- `assemble_generations` discards stderr
    on a clean exit, which is the path that matters."""
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=True)
    assert captured["run_report"] == {"device": "cuda"}


def test_the_reference_script_prefers_what_the_parent_already_resolved(tmp_path, monkeypatch):
    """`given` (`spec["device"]`, what `HuggingFaceBackend` asked
    `envs.resolve_device` before this script started) wins over asking torch
    -- not a second, possibly-disagreeing guess made inside the subprocess.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=False, given="cuda")
    assert captured["model_device"] == "cuda"


def test_an_unprovisioned_environment_is_a_third_state(monkeypatch, tmp_path):
    """ "Nobody asked", "it was asked and could not say", and "there was nothing
    to ask" are three different facts, and the third used to be filed as the
    first: `last_probe` stayed `None`, so the manifest was byte-identical to a
    run where no probe was wanted.

    It must also not block. Whether an unprovisioned environment can still
    generate is the caller's decision -- a library caller managing the lifecycle
    itself, or a test supplying its own `run`, legitimately gets here.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)

    blocked = backend._ensure_env(events=None)

    assert blocked is None, "recording is not blocking"
    assert backend.last_probe is not None, "the third state must leave a record"
    assert "not provisioned" in backend.last_probe.detail
    assert (
        "could not answer" not in backend.last_probe.detail
    ), "no probe was attempted, so it cannot be reported as one that failed to answer"
    assert backend.last_probe.device is None


def test_the_unprovisioned_state_is_a_field_not_a_phrase(monkeypatch, tmp_path):
    """`verify` used to tell the two apart by looking for "not provisioned" in
    the sentence, which is a protocol made of prose: the first rewording takes
    the branch out silently, and a caller that supplies its own runnable `run`
    over an unready environment makes the sentence wrong anyway."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)

    backend._ensure_env(events=None)

    assert backend.last_probe is not None
    assert backend.last_probe.attempted is False, "no probe was run, and that is a fact not a word"


def test_a_rendered_prompt_reaches_the_reference_with_one_bos(tmp_path, monkeypatch):
    """The template already emits `<bos>`; the tokenizer must not add a second.

    transformers' chat-templating guide: pass `add_special_tokens=False` when
    tokenizing text `apply_chat_template(tokenize=False)` rendered. The cached
    `google/gemma-3-270m-it` tokenizer gave two leading BOS without it.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, runtime_rendered=True)
    assert captured["tokenized"] == [[2, 11, 12, 13]]


def test_a_prerendered_prompt_still_gets_the_tokenizers_bos(tmp_path, monkeypatch):
    captured = _run_hf_generate_script(tmp_path, monkeypatch)
    assert captured["tokenized"] == [[2, 10]]


def test_a_splits_id_does_not_depend_on_whether_a_target_kept_its_types(tmp_path):
    """Found in review: a split's target now keeps its argument types, and the id
    computed from it changed for the same file. The id is over the flattened
    target, as it was before, so a file prepared by main and by this version
    identifies the same split."""
    from litetune.evaluate import load_split

    typed = tmp_path / "typed.jsonl"
    typed.write_text('{"prompt": "p", "target": {"name": "f", "args": {"n": 3}}}\n')
    flat = tmp_path / "flat.jsonl"
    flat.write_text('{"prompt": "p", "target": {"name": "f", "args": {"n": "3"}}}\n')

    assert load_split(typed).id == load_split(flat).id


DECLS = [{"type": "function", "function": {"name": "open_app", "description": "d"}}]


def test_the_reference_backend_hands_the_declarations_to_its_script(monkeypatch):
    """Found in review: removing the declarations from the reference's spec left
    every test green, and the reference was then measured on a bare prompt while
    the candidate's runtime rendered the tool list."""
    specs: list[dict] = []

    def fake_run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text())
        specs.append(spec)
        Path(spec["out"]).write_text(json.dumps({"index": 0, "text": "x"}) + "\n")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    HuggingFaceBackend(
        model="org/m", auto_provision=False, runtime_rendered=True, declarations=DECLS
    ).generate(["a"])

    assert specs[0]["tools"] == DECLS


def test_the_reference_script_renders_the_prompt_with_the_declarations(tmp_path, monkeypatch):
    """The script side of the same wiring: the spec's tools reach the chat
    template the reference generates from."""
    import sys
    import types

    from litetune.evaluate import _HF_GENERATE_SCRIPT

    rendered: list[str] = []

    class Ids(list):
        shape = (1, 1)

        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False, tools=None
        ):
            text = (
                "".join(f"decl:{t['function']['name']} " for t in (tools or []))
                + messages[0]["content"]
            )
            rendered.append(text)
            return text

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            return _Encoded({"input_ids": Ids([[5]])})

        def decode(self, ids, skip_special_tokens=False):
            return "answer"

    class _Encoded(dict):
        def to(self, device):
            return self

    class Model:
        def eval(self):
            return self

        def to(self, device):
            return self

        def generate(self, **kwargs):
            return [[5, 6]]

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    torch = types.SimpleNamespace(
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: False),
        no_grad=NoGrad,
    )
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda model: Tokenizer()),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: Model()),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "model": "org/m",
                "prompts": ["hi"],
                "runtime_rendered": True,
                "tools": DECLS,
                "attn_implementation": "eager",
                "max_tokens": 4,
                "out": str(tmp_path / "out.jsonl"),
                "run_report": str(tmp_path / "report.json"),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["generate.py", str(spec)])
    namespace: dict = {"__name__": "hf_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf_script_under_test", "exec"), namespace)  # noqa: S102

    assert namespace["main"]() == 0
    assert rendered == ["decl:open_app hi"]


# -- the driver script's own main -------------------------------------------
#
# Until this block the script's `main` ran nowhere: the suite drove
# `LiteRtLmBackend` with a `StageEnv.run` double that fabricated the JSONL, so
# the reader and the writer agreed only by inspection. Writing `"err"` instead
# of `"error"` in the script, or dropping the per-row flush, passed everything.
# `litert_lm` is imported inside `backend_for` and `main` precisely so a fake
# can stand in for it, which is what the cloud equivalence harness already does.


class _FakeRunner:
    """One conversation or session. Records what it was asked."""

    def __init__(self, registry, raise_on):
        self.registry = registry
        self.raise_on = raise_on
        self.closed = False
        registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def _answer(self, prompt):
        if prompt in self.raise_on:
            raise self.raise_on[prompt]
        return "answer to " + prompt

    def run_prefill(self, prompts):
        self._prompt = prompts[0]

    def run_decode_async(self):
        chunk = type("Chunk", (), {})()
        chunk.texts = [self._answer(self._prompt)]
        return [chunk]

    def send_message_async(self, prompt):
        return [{"content": [{"type": "text", "text": self._answer(prompt)}], "channels": {}}]


class _FakeEngine:
    def __init__(self, path, backend=None, **kwargs):
        self.path = path
        self.backend = backend
        self.kwargs = kwargs
        self.runners: list[_FakeRunner] = []
        self.raise_on: dict[str, BaseException] = {}
        # The fake module, so a test can see whether an engine is open at the
        # moment something else runs.
        self.module = None

    def __enter__(self):
        if self.module is not None:
            self.module.engine_open = True
        return self

    def __exit__(self, *exc):
        if self.module is not None:
            self.module.engine_open = False
        return False

    def create_session(self, **kwargs):
        return _FakeRunner(self.runners, self.raise_on)

    def create_conversation(self, **kwargs):
        return _FakeRunner(self.runners, self.raise_on)


def _fake_litert_lm(monkeypatch, raise_on=None):
    """A `litert_lm` the script can import, and the engine it hands back."""
    engines: list[_FakeEngine] = []

    class _Backend:
        name = "?"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Cpu(_Backend):
        name = "cpu"

    class _Gpu(_Backend):
        name = "gpu"

    class _ActivationDataType:
        # 0 in the real enum, and so here: code that drops it for being falsy
        # has to fail a test.
        FLOAT32 = 0

        @classmethod
        def from_str(cls, value):
            return {"fp32": cls.FLOAT32}.get(value.lower())

    def make_engine(path, backend=None, **kwargs):
        engine = _FakeEngine(path, backend, **kwargs)
        engine.raise_on = raise_on or {}
        engine.module = module
        engines.append(engine)
        return engine

    module = types.ModuleType("litert_lm")
    module.engine_open = False
    module.Engine = make_engine
    module.ActivationDataType = _ActivationDataType
    interfaces = types.ModuleType("litert_lm.interfaces")
    interfaces.CPU = _Cpu
    interfaces.GPU = _Gpu
    monkeypatch.setitem(sys.modules, "litert_lm", module)
    monkeypatch.setitem(sys.modules, "litert_lm.interfaces", interfaces)
    return engines


def _run_script(
    tmp_path,
    prompts,
    runtime_rendered=True,
    raise_on=None,
    monkeypatch=None,
    engine=None,
    report=None,
):
    engines = _fake_litert_lm(monkeypatch, raise_on)
    spec = tmp_path / "spec.json"
    out = tmp_path / "out.jsonl"
    spec.write_text(
        json.dumps(
            {
                "model": str(tmp_path / "m.litertlm"),
                "prompts": prompts,
                "runtime_rendered": runtime_rendered,
                **(engine if engine is not None else runtime_engine_spec("cpu")),
                "out": str(out),
                **({"report": str(report)} if report is not None else {}),
            }
        ),
        encoding="utf-8",
    )
    _litertlm_script()["main"](str(spec))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    return rows, engines


@pytest.mark.parametrize("runtime_rendered", [True, False])
def test_the_script_writes_one_row_per_prompt_in_order(tmp_path, monkeypatch, runtime_rendered):
    rows, engines = _run_script(
        tmp_path, ["one", "two", "three"], runtime_rendered, monkeypatch=monkeypatch
    )
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert [r["text"] for r in rows] == ["answer to one", "answer to two", "answer to three"]
    # A runner per prompt, not one per split: a conversation keeps its history,
    # so a shared one would answer the second prompt with the first still in
    # context and the split would score as drifting nonsense.
    assert len(engines[0].runners) == 3
    assert all(runner.closed for runner in engines[0].runners)


def test_one_refused_prompt_is_one_row(tmp_path, monkeypatch):
    """The whole of `15b9ada`, which until now no test touched.

    The runtime raises mid-split -- `session.py:70` on a prefill that fails,
    `conversation.py:326` on a failed send, both at v0.16.1 -- and without the
    containment the first of those ends the process, so every later prompt
    comes back as "the script exited 1".
    """
    rows, _ = _run_script(
        tmp_path,
        ["one", "two", "three"],
        raise_on={"two": RuntimeError("litert_lm_session_run_prefill failed")},
        monkeypatch=monkeypatch,
    )
    assert [r["index"] for r in rows] == [0, 1, 2]
    assert rows[0]["text"] == "answer to one"
    assert "text" not in rows[1]
    assert rows[1]["error"] == "RuntimeError: litert_lm_session_run_prefill failed"
    assert rows[2]["text"] == "answer to three"
    # And the reader turns that row into a fault rather than an answer.
    texts, faults = read_jsonl_results(tmp_path / "out.jsonl")
    assert set(texts) == {0, 2}
    assert "litert_lm_session_run_prefill failed" in faults[1]


def test_memory_error_is_not_contained(tmp_path, monkeypatch):
    """Containing it would turn one fatal condition into `len(prompts)` of them.

    The engine that could not allocate for this prompt will not allocate for
    the next, so the loop would re-OOM every remaining prompt under a parent
    budget of `timeout_s * len(prompts)`. The rows already flushed are salvaged
    by the non-zero-exit path instead.
    """
    with pytest.raises(MemoryError):
        _run_script(
            tmp_path,
            ["one", "two"],
            raise_on={"one": MemoryError("cannot allocate the KV cache")},
            monkeypatch=monkeypatch,
        )


def test_the_script_asks_for_the_cache_the_cli_asks_for(tmp_path, monkeypatch):
    """`cache_dir=""` is the CLI's own default, and omitting it is a fourth state."""
    _, engines = _run_script(tmp_path, ["one"], monkeypatch=monkeypatch)
    assert engines[0].kwargs["cache_dir"] == ""


def test_the_script_refuses_a_backend_it_does_not_know(tmp_path, monkeypatch):
    _fake_litert_lm(monkeypatch)
    backend_for = _litertlm_script()["backend_for"]
    assert backend_for("cpu") is not None
    assert backend_for("gpu") is not None
    with pytest.raises(SystemExit):
        backend_for("npu")


# -- which backends can say they read a device back -------------------------
#
# Every one of these was a mutant that survived: flipping either litert-lm
# backend's answer to `True`, or deriving the transformers one from `device is
# not None`, left the whole suite green, because absent and `False` are
# indistinguishable at the only consumer. The consumer-side tests in
# `test_verify.py` pin the gate; these pin what the shipped backends feed it.


def test_the_litertlm_backend_never_claims_it_read_a_device_back(tmp_path):
    """It passes a flag and litert-lm's Python API tells it nothing in return."""
    assert _litertlm(tmp_path).backend_observed is False
    assert _litertlm(tmp_path).describe()[BACKEND_OBSERVED] is False


def test_a_transformers_backend_that_never_ran_claims_nothing(monkeypatch):
    _ready_env()
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    assert backend.describe()[BACKEND_OBSERVED] is False
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_a_transformers_run_whose_script_reported_its_device_says_so(monkeypatch):
    _ready_env()
    _generating_env(monkeypatch, probe="cpu", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    described = backend.describe()
    assert described["backend"] == "cuda"
    assert described[BACKEND_OBSERVED] is True


def test_a_prediction_left_standing_is_not_reported_as_a_reading(monkeypatch):
    """The whole reason this is not `device is not None`.

    The probe writes its answer into `device` before the run, and a script
    that died before writing its report leaves that prediction standing --
    deliberately, and `test_a_script_that_wrote_no_report_leaves_the_prediction_standing`
    pins it. The prediction is still the best thing known and stays in
    `backend`; what it is not is something this run read back.
    """
    _ready_env()
    _generating_env(monkeypatch, probe="cuda", script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    described = backend.describe()
    assert described["backend"] == "cuda"
    assert described[BACKEND_OBSERVED] is False


def test_a_device_kept_by_the_do_not_erase_rule_is_not_reported_as_read_back(monkeypatch):
    """`device` survives a run that established nothing; the claim must not.

    This is the reuse case `test_a_failed_probe_does_not_erase_a_device_already_known`
    pins: a later run whose probe cannot answer and whose script writes no
    report keeps the earlier `cuda`, because that is a better record than
    `None`. It is still not something *this* run read back.
    """
    _ready_env()
    _generating_env(monkeypatch, probe=None, script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False, device="cuda")
    backend.generate(["a"])

    described = backend.describe()
    assert described["backend"] == "cuda"
    assert described[BACKEND_OBSERVED] is False


# -- the GPU backend -----------------------------------------------------------
#
# What the parent asks for and what the script builds from it. The engine
# echoes back the backend it was given and names no device, so these are the
# only places the choice can be checked before a real GPU run.


def test_a_gpu_run_states_the_activation_type_and_a_cpu_run_does_not():
    """fp32 on every GPU run, whatever the bundle carries.

    The bundle's `prefer_activation_type` is only a default; a bundle without
    it leaves the GPU text executor in F16, which floods `<pad>` while the
    engine reports success. On the CPU backend nothing is passed, as it never
    has been.
    """
    assert runtime_engine_spec("gpu") == {"backend": "gpu", "activation_data_type": "fp32"}
    assert runtime_engine_spec("cpu") == {"backend": "cpu", "activation_data_type": None}


def test_a_backend_nobody_listed_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="npu"):
        runtime_engine_spec("npu")


def test_the_driver_builds_a_gpu_engine_with_fp32(tmp_path, monkeypatch):
    rows, engines = _run_script(
        tmp_path, ["one"], monkeypatch=monkeypatch, engine=runtime_engine_spec("gpu")
    )

    assert rows[0]["text"] == "answer to one"
    (engine,) = engines
    assert engine.backend.name == "gpu"
    # `is 0`, via the enum: FLOAT32's value is 0 and must still reach the engine.
    assert "activation_data_type" in engine.kwargs
    assert engine.kwargs["activation_data_type"] == 0


def test_the_driver_passes_no_activation_type_on_cpu(tmp_path, monkeypatch):
    _, engines = _run_script(tmp_path, ["one"], monkeypatch=monkeypatch)

    (engine,) = engines
    assert engine.backend.name == "cpu"
    assert "activation_data_type" not in engine.kwargs


def test_an_activation_type_the_runtime_does_not_know_ends_the_run(tmp_path, monkeypatch):
    """A typo must not quietly become the F16 default.

    `from_str` answers None for a name it does not know, and an engine given
    None builds as if nothing were asked -- on a GPU, exactly the state this
    key exists to prevent.
    """
    with pytest.raises(SystemExit, match="fp23"):
        _run_script(
            tmp_path,
            ["one"],
            monkeypatch=monkeypatch,
            engine={"backend": "gpu", "activation_data_type": "fp23"},
        )


def test_the_runtime_backend_describes_what_it_asked_for(tmp_path):
    gpu = _litertlm(tmp_path, backend_flag="gpu").describe()
    assert (gpu["backend"], gpu["activation_data_type"], gpu[BACKEND_OBSERVED]) == (
        "gpu",
        "fp32",
        False,
    )
    cpu = _litertlm(tmp_path).describe()
    assert (cpu["backend"], cpu["activation_data_type"]) == ("cpu", None)


# -- whether the GPU was used, asked of the kernel ----------------------------
#
# litert-lm echoes the backend it was given. The kernel says more: on macOS a
# GPU engine owns a GPU user client from the moment it is built, recorded
# against the pid that created it, and the client's GPU time grows while it
# generates. Measured 2026-09-25 on an M4 Pro with litert-lm 0.16.1: one client
# after a GPU engine was built, its GPU time grown about four thousand times
# after one reply; none for a CPU engine.


def _reading(
    looked=True, client="pid 4242, python3.12", gpu_time=None, platform="darwin", clients=None
):
    """One reading; `clients` defaults to a single client holding `gpu_time`."""
    if clients is None and looked:
        clients = {"0x2": gpu_time} if client is not None else {}
    return {
        "platform": platform,
        "looked": looked,
        "gpu_client": client,
        "gpu_time": gpu_time,
        "gpu_clients": clients,
    }


def _run_report(at_engine, after_generation, bundle_activation="fp32"):
    return {
        "at_engine": at_engine,
        "after_generation": after_generation,
        "bundle_activation": bundle_activation,
    }


_WORKED = _run_report(_reading(gpu_time=1_000), _reading(gpu_time=5_000_000))
_OPENED_IDLE = _run_report(_reading(gpu_time=1_000), _reading(gpu_time=1_000))
_NO_CLIENT = _run_report(_reading(client=None), _reading(client=None))
_NOT_LOOKED = _run_report(
    _reading(looked=False, client=None, platform="linux"),
    _reading(looked=False, client=None, platform="linux"),
    bundle_activation=None,
)
_KILLED_AFTER_ENGINE = _run_report(_reading(gpu_time=1_000), None)
# 0x1 closed after the engine reading and 0x2 did not grow: whatever 0x1 did
# before it closed was never read, so "no growth" is not established.
_SWAPPED = _run_report(
    _reading(gpu_time=200, clients={"0x1": 100, "0x2": 100}),
    _reading(gpu_time=100, clients={"0x2": 100}),
)
_NEW_CLIENT = _run_report(
    _reading(gpu_time=100, clients={"0x1": 100}),
    _reading(gpu_time=150, clients={"0x1": 100, "0x2": 50}),
)
_SECOND_WORKED = _run_report(
    _reading(gpu_time=200, clients={"0x1": 100, "0x2": 100}),
    _reading(gpu_time=300, clients={"0x1": 100, "0x2": 200}),
)

_LISTING = "\n".join(
    [
        "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x1>",
        '    |   "IOUserClientCreator" = "pid 11478, Google Chrome He"',
        '    |   "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=999999})',
        "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x2>",
        '    |   "IOUserClientCreator" = "pid 4242, python3.12"',
        '    |   "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=40},'
        '{"API"="Metal","accumulatedGPUTime"=2})',
    ]
)


def test_only_this_processs_client_and_its_gpu_time_count():
    """Every process's GPU clients are in one listing -- a browser, a chat app.
    The pid is the evidence, and a prefix of another pid is not this pid."""
    client_of = _litertlm_script()["gpu_client_of"]
    assert client_of(_LISTING, 4242) == ("pid 4242, python3.12", {"0x2": 42})
    assert client_of(_LISTING, 424) == (None, {})
    assert client_of(_LISTING, 11) == (None, {})


def test_every_client_of_this_process_is_read_by_its_own_id():
    """A process may hold more than one GPU client; each is kept apart so the
    two readings can be compared client by client."""
    second = (
        _LISTING
        + "\n"
        + "\n".join(
            [
                "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x3>",
                '    |   "IOUserClientCreator" = "pid 4242, python3.12"',
                '    |   "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=1000})',
            ]
        )
    )
    assert _litertlm_script()["gpu_client_of"](second, 4242) == (
        "pid 4242, python3.12",
        {"0x2": 42, "0x3": 1000},
    )


def test_a_platform_nobody_observed_is_not_looked_at(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    report = _litertlm_script()["device_report"]()
    assert report == _reading(looked=False, client=None, platform="linux")


def test_on_macos_the_kernel_is_asked_about_this_pid(monkeypatch):
    """The darwin branch, with values: which class is asked for, and that the
    answer is read for this process and not its parent."""
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        listing = _LISTING.replace("4242", str(os.getpid()))
        return subprocess.CompletedProcess(argv, 0, listing.encode(), b"")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", fake_run)
    report = _litertlm_script()["device_report"]()

    assert seen == [["/usr/sbin/ioreg", "-r", "-c", "IOGPUDeviceUserClient", "-l"]]
    assert report["looked"] is True
    assert report["gpu_client"] == f"pid {os.getpid()}, python3.12"
    assert report["gpu_time"] == 42
    assert report["gpu_clients"] == {"0x2": 42}


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("no ioreg"),
        subprocess.TimeoutExpired(cmd="ioreg", timeout=30),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
        MemoryError(),
    ],
)
def test_a_kernel_query_that_fails_says_so_and_never_ends_the_run(monkeypatch, failure):
    """A diagnostic that raised here would end the run before its first
    prompt, losing the measurement to find out where it ran."""

    def fake_run(argv, **kwargs):
        raise failure

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", fake_run)
    report = _litertlm_script()["device_report"]()

    assert report["looked"] is False
    assert type(failure).__name__ in report["error"]


def test_a_kernel_query_that_exits_non_zero_did_not_look(monkeypatch):
    """Recorded as looked-and-found-none it would read as "the GPU was not
    used", which the kernel never said."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 1, b"", b"denied")
    )
    report = _litertlm_script()["device_report"]()
    assert report["looked"] is False
    assert "denied" in report["error"]


@pytest.mark.parametrize("listing", [b"", b"+-o Root  <class IORegistryEntry>\n"])
def test_a_listing_with_no_gpu_client_of_anyone_did_not_look(monkeypatch, listing):
    """Exit 0 and no client at all is not "no client of ours": the class
    asked for is not there to be asked. Read as looked it would say the GPU
    was not used."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, listing, b"")
    )
    report = _litertlm_script()["device_report"]()
    assert report["looked"] is False
    assert "no GPU client of any process" in report["error"]


def _fake_builder(monkeypatch, sections):
    """A `litert_lm_builder` whose header holds `sections`, each a list of
    (key, value) items as the builder writes them -- `model_type` among them,
    and a key may repeat. A str value is a StringValue, anything else another
    type. `get_model_type` reads the items the way the real one does
    (`litertlm_peek.get_model_type`, 0.16.1), not a side attribute."""
    string_type = 1

    class _Value:
        def __init__(self, raw):
            self.Bytes, self.Pos = {1: raw}, 1

    class _Item:
        def __init__(self, key, value):
            self._key, self._value = key.encode(), value

        def Key(self):
            return self._key

        def ValueType(self):
            return string_type if isinstance(self._value, str) else string_type + 1

        def Value(self):
            return _Value(self._value.encode() if isinstance(self._value, str) else b"")

    class _Section:
        def __init__(self, items):
            self._items = [_Item(k, v) for k, v in items]

        def ItemsLength(self):
            return len(self._items)

        def Items(self, j):
            return self._items[j]

    class _Sections:
        def __init__(self, sections):
            self._sections = [_Section(items) for items in sections]

        def ObjectsLength(self):
            return len(self._sections)

        def Objects(self, i):
            return self._sections[i]

    class _StringValue:
        def Init(self, buf, pos):
            self._raw = buf[pos]

        def Value(self):
            return self._raw

    def get_model_type(section):
        for j in range(section.ItemsLength()):
            item = section.Items(j)
            if item.Key() == b"model_type" and item.ValueType() == string_type:
                value = _StringValue()
                value.Init(item.Value().Bytes, item.Value().Pos)
                return value.Value().decode("utf-8")
        return None

    header = types.SimpleNamespace(SectionMetadata=lambda: _Sections(sections))
    peek = types.SimpleNamespace(
        read_litertlm_header=lambda path, out: header, get_model_type=get_model_type
    )
    schema = types.SimpleNamespace(
        VData=types.SimpleNamespace(StringValue=string_type), StringValue=_StringValue
    )
    package = types.ModuleType("litert_lm_builder")
    package.litertlm_peek = peek
    package.litertlm_header_schema_py_generated = schema
    monkeypatch.setitem(sys.modules, "litert_lm_builder", package)
    monkeypatch.setitem(sys.modules, "litert_lm_builder.litertlm_peek", peek)
    monkeypatch.setitem(
        sys.modules, "litert_lm_builder.litertlm_header_schema_py_generated", schema
    )


def _pd(*items):
    """A prefill-decode section with these extra items."""
    return [("model_type", "tf_lite_prefill_decode"), *items]


_KEY = "prefer_activation_type"


@pytest.mark.parametrize(
    ("sections", "declared"),
    [
        ([_pd((_KEY, "fp32"))], "fp32"),
        ([_pd(("other", "x"))], None),
        ([[("model_type", "tf_lite_embedder"), (_KEY, "fp16")], _pd()], None),
        # The key without a model_type item is in no prefill-decode section.
        ([[(_KEY, "fp32")]], None),
        ([_pd((_KEY, "fp32"))] * 2, "fp32"),
    ],
)
def test_the_bundle_key_is_read_from_its_prefill_decode_sections(monkeypatch, sections, declared):
    _fake_builder(monkeypatch, sections)
    assert _litertlm_script()["bundle_activation"]("m.litertlm") == declared


@pytest.mark.parametrize(
    "sections",
    [
        # The first section is not the bundle's answer when a later one differs.
        [_pd((_KEY, "fp32")), _pd((_KEY, "fp32_fp16"))],
        [_pd((_KEY, "fp32")), _pd()],
        # Within one section: `additional_metadata` may repeat the key.
        [_pd((_KEY, "fp32"), (_KEY, "fp16"))],
        # Not a string: unreadable, not "declares none".
        [_pd((_KEY, 7)), _pd((_KEY, "fp32"))],
    ],
)
def test_a_bundle_key_that_is_not_one_answer_is_unreadable(monkeypatch, sections):
    _fake_builder(monkeypatch, sections)
    report = _litertlm_script()["run_report"]({"model": "m.litertlm"}, None)
    assert report["bundle_activation"] is None
    assert report["bundle_activation_error"].startswith("ValueError")


@pytest.mark.parametrize(
    ("backend", "report", "observed", "unused"),
    [
        ("gpu", _WORKED, True, False),
        ("gpu", _OPENED_IDLE, False, True),  # opened the GPU, did no work there
        ("gpu", _NO_CLIENT, False, True),  # asked, looked, no client for this pid
        ("gpu", _NOT_LOOKED, False, False),  # asked, not looked: not established
        ("gpu", _KILLED_AFTER_ENGINE, False, False),  # opened, work not established
        ("gpu", _SWAPPED, False, False),  # a client closed: its work is not established
        ("gpu", _NEW_CLIENT, True, False),  # a client opened while generating did work
        ("gpu", _SECOND_WORKED, True, False),  # an idle first client does not hide it
        ("gpu", None, False, False),  # no report at all
        ("cpu", _WORKED, False, False),  # a CPU run is never read as a GPU one
    ],
)
def test_the_gpu_is_observed_only_when_it_did_work(backend, report, observed, unused):
    assert gpu_observed(backend, report) is observed
    assert gpu_unused(backend, report) is unused


def test_the_runtime_backend_reports_what_its_process_did(monkeypatch, tmp_path):
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}], device=_WORKED)
    )
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one"])

    described = backend.describe()
    assert described[BACKEND_OBSERVED] is True
    assert described["gpu_unused"] is False
    assert described["bundle_activation"] == "fp32"
    assert described["device_report"] == _WORKED


def test_a_gpu_request_the_kernel_shows_unused_is_said_to_be_unused(monkeypatch, tmp_path):
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}], device=_NO_CLIENT)
    )
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one"])

    assert backend.describe()[BACKEND_OBSERVED] is False
    assert backend.describe()["gpu_unused"] is True


def test_a_reading_is_not_carried_into_a_run_that_wrote_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}], device=_WORKED)
    )
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one"])
    assert backend.backend_observed is True

    monkeypatch.setattr(envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}]))
    backend.generate(["two"])

    assert backend.backend_observed is False
    assert backend.describe()["device_report"] is None


def test_a_run_that_returns_early_does_not_keep_the_last_reading(monkeypatch, tmp_path):
    """The reset matters only on an early return; an empty prompt list is one."""
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}], device=_WORKED)
    )
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one"])
    assert backend.backend_observed is True

    assert backend.generate([]) == []
    assert backend.backend_observed is False


def test_a_run_killed_after_its_engine_opened_the_gpu_is_not_called_a_gpu_run(
    monkeypatch, tmp_path
):
    """The script writes its first reading before the first prompt, so a
    killed run still says where it was -- but only a second reading can show
    work, and a killed run never took one."""

    def times_out(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        if "report" not in spec:
            # The one-prompt preflight a GPU split starts with: it answers.
            Path(spec["out"]).write_text(json.dumps({"index": 0, "text": "a"}) + "\n")
            return subprocess.CompletedProcess(args, 0, "", "")
        Path(spec["report"]).write_text(json.dumps(_KILLED_AFTER_ENGINE), encoding="utf-8")
        Path(spec["out"]).write_text(json.dumps({"index": 0, "text": "a"}) + "\n")
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout, stderr="killed")

    monkeypatch.setattr(envs.StageEnv, "run", times_out)
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one", "two"])

    assert backend.describe()["device_report"] == _KILLED_AFTER_ENGINE
    assert backend.backend_observed is False
    assert backend.describe()["gpu_unused"] is False


@pytest.mark.parametrize("written", ["{", "[]", '"a string"'])
def test_a_report_that_cannot_be_read_is_no_report(monkeypatch, tmp_path, written):
    """Truncated by a kill mid-write, or not an object: either way nothing was
    established, and neither may escape `generate`."""

    def writes(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["report"]).write_text(written, encoding="utf-8")
        Path(spec["out"]).write_text(json.dumps({"index": 0, "text": "a"}) + "\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", writes)
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one"])

    assert backend.describe()["device_report"] is None
    assert backend.backend_observed is False


def test_the_driver_takes_a_reading_before_and_after_generating(tmp_path, monkeypatch):
    """Both readings, each taken while the engine exists: the first so a
    killed run still says where it was, the second because only growth shows
    work. Found by review: a reading moved above `Engine()` passed every
    test."""
    readings = []

    def fake_report():
        live = bool(sys.modules["litert_lm"].engine_open)
        readings.append(live)
        return _reading(gpu_time=len(readings))

    report = tmp_path / "device.json"
    ns_main = _litertlm_script()
    ns_main["device_report"] = fake_report
    ns_main["bundle_activation"] = lambda path: "fp32"
    engines = _fake_litert_lm(monkeypatch)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "model": str(tmp_path / "m.litertlm"),
                "prompts": ["one"],
                "runtime_rendered": True,
                **runtime_engine_spec("gpu"),
                "out": str(tmp_path / "out.jsonl"),
                "report": str(report),
            }
        ),
        encoding="utf-8",
    )
    ns_main["main"](str(spec))

    assert readings == [True, True]
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["at_engine"]["gpu_time"] == 1
    assert written["after_generation"]["gpu_time"] == 2
    assert written["bundle_activation"] == "fp32"
    assert engines


def test_a_gpu_that_never_answers_is_reported_in_one_prompts_time(monkeypatch, tmp_path):
    """Not in the split's: 300 s per prompt makes a 600-row run wait fifty hours
    for what one prompt shows, and the reason has to name the GPU, because
    that silence is the measured behaviour of a GPU engine with no GPU."""
    budgets = []

    def silent(self, args, timeout=3600, **kwargs):
        budgets.append(timeout)
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", silent)
    backend = _litertlm(tmp_path, backend_flag="gpu", timeout_s=7)
    generations = backend.generate([f"p{i}" for i in range(600)])

    assert budgets == [7], "one prompt's budget, and the split never started"
    assert all(not g.ran for g in generations)
    assert "gpu backend gave no answer to one prompt in 7s" in generations[0].harness_error


def test_a_cpu_split_starts_without_a_preflight(monkeypatch, tmp_path):
    calls = []

    def once(self, args, timeout=3600, **kwargs):
        calls.append(timeout)
        return _writes_results([{"index": 0, "text": "a"}, {"index": 1, "text": "b"}])(
            self, args, timeout
        )

    monkeypatch.setattr(envs.StageEnv, "run", once)
    _litertlm(tmp_path, timeout_s=7).generate(["one", "two"])

    assert calls == [14]


def test_a_timed_out_split_names_its_backend(monkeypatch, tmp_path):
    def silent(self, args, timeout=3600, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", silent)
    generations = _litertlm(tmp_path, timeout_s=7).generate(["one", "two"])

    assert "on the cpu backend" in generations[0].harness_error
