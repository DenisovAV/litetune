"""The evaluator and its two backends.

Nothing here starts a process or loads a model: `StageEnv.run` is monkeypatched
at the boundary, which is the only place either backend touches the outside
world.

The backend tests are mostly about one distinction -- a process that ran and
failed against a process that never ran. Collapsing the two is what produced
eight confident negatives during the measurement work, so each has its own test.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    FakeBackend,
    call_text,
    fake_torch,
    labelled_rows,
    mark_provisioned,
    probe_answer,
)

import litetune
from litetune import envs, toolpath
from litetune.embedding_verify import SHADOWABLE_MODULES
from litetune.evaluate import (
    BACKEND_OBSERVED,
    GENERATION_SETTINGS_NOT_APPLIED,
    UNKNOWN_BACKEND,
    DataError,
    DecodeConfig,
    HuggingFaceBackend,
    LiteRtLmBackend,
    _litertlm_script,
    _read_device_report,
    assemble_generations,
    bundle_activation_error_of,
    device_mismatch,
    engine_init_of,
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
# `" [/name]"` and a newline) and 108-123 (it prints `str(chunk)` first, and
# calls it before that text when the text is not empty, on a switch between
# channels, and at end of stream).
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
    (
        # v0.18.0's run.py closes a channel only before a chunk whose text is
        # not empty (`text = str(chunk)`); v0.17.1 closed it before every text
        # item, so this printed `[thought] a [/thought]` and reopened it.
        "an empty text item inside a channel",
        [
            {"channels": {"thought": "a"}},
            {"content": [{"type": "text", "text": ""}]},
            {"channels": {"thought": "b"}},
        ],
        "[thought] ab [/thought]",
    ),
    (
        "two text items in one chunk",
        [{"content": [{"type": "text", "text": "an"}, {"type": "text", "text": "swer"}]}],
        "answer",
    ),
    (
        # A v0.18.0 `Message` keeps a string `content` as the string.
        "content given as a string",
        [{"content": "answer"}],
        "answer",
    ),
    (
        # ...and reads a bare string inside a `content` list as a text part.
        "a bare string inside a content list",
        [{"content": ["an", "swer"]}],
        "answer",
    ),
    (
        # run.py prints any `str(chunk)` that is not empty; a newline is text.
        "a text chunk that is only a newline",
        [{"content": [{"type": "text", "text": t}]} for t in ("a", "\n", "b")],
        "a\nb",
    ),
    (
        "a newline between two pieces of one channel",
        [
            {"channels": {"thought": "a"}},
            {"content": [{"type": "text", "text": "\n"}]},
            {"channels": {"thought": "b"}},
        ],
        "[thought] a [/thought]\n\n[thought] b [/thought]",
    ),
    (
        # run.py prints a chunk's text before its channels.
        "text and a channel in one chunk",
        [
            {"channels": {"thought": "a"}},
            {"content": [{"type": "text", "text": "X"}], "channels": {"thought": "b"}},
        ],
        "[thought] a [/thought]\nX[thought] b [/thought]",
    ),
    (
        # `Contents.__str__` joins the text parts only.
        "a part that is not text is not printed",
        [{"content": [{"type": "image", "text": "no"}, {"type": "text", "text": "answer"}]}],
        "answer",
    ),
    (
        # `Message` reads a text part with no `text` as empty text.
        "a text part with no text prints nothing",
        [{"content": [{"type": "text"}, {"type": "text", "text": "answer"}]}],
        "answer",
    ),
    (
        "a null content prints nothing",
        [{"content": None}, {"content": [{"type": "text", "text": "answer"}]}],
        "answer",
    ),
    (
        # `Message` keeps only a string or a list as content.
        "a content that is an object prints nothing",
        [
            {"content": {"type": "text", "text": "no"}},
            {"content": [{"type": "text", "text": "answer"}]},
        ],
        "answer",
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
    assert "litert-lm==0.18.0" in described["requirements"]


@pytest.mark.parametrize(("platform", "debian"), [("linux", ["libvulkan1"]), ("win32", [])])
def test_a_manifest_names_a_debian_package_only_on_linux(tmp_path, monkeypatch, platform, debian):
    """A `--json` manifest from a Mac or a Windows machine listed `libvulkan1`,
    a Debian name nobody there can install."""
    # The `sys` envs reads, which is the one `sys` -- named so it stays that.
    monkeypatch.setattr(envs.sys, "platform", platform)
    assert _litertlm(tmp_path).describe()["system_requirements"] == debian


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
                probe_answer(probe),
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
            return subprocess.CompletedProcess(args, 0, probe_answer("cuda", device_count=1), "")
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    gens = backend.generate(["a"])

    assert gens[0].harness_error is not None
    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_a_timeout_keeps_the_mps_observation_written_before_generation(monkeypatch):
    """A timeout after placement must not erase the child's run report.

    The generation script writes this report before its first prompt. Reading
    it only after a clean return made a part-way MPS run look as though its
    backend had never been observed and discarded the allocator environment
    that had already reached the child.
    """
    _ready_env()

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(args, 0, probe_answer("mps"), "")
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["run_report"]).write_text(
            json.dumps({"device": "mps", "mps_environment": _SENT}), encoding="utf-8"
        )
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    (generation,) = backend.generate(["a"])

    assert generation.harness_error is not None
    assert backend.describe()["backend"] == "mps"
    assert backend.backend_observed
    assert backend.mps_environment_seen == _SENT
    assert backend.mps_environment_mismatch is None


def test_a_script_that_could_not_start_clears_the_device_it_never_confirmed(monkeypatch):
    _ready_env()

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(args, 0, probe_answer("cuda", device_count=1), "")
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


# What transformers 5.16.1 falls back to for a generation field nobody set
# (`GenerationConfig._get_default_generation_params`,
# generation/configuration_utils.py:609-645), verbatim.
_TRANSFORMERS_DEFAULTS: dict[str, Any] = {
    "max_length": 20,
    "min_length": 0,
    "do_sample": False,
    "use_cache": True,
    "early_stopping": False,
    "num_beams": 1,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": None,
    "num_return_sequences": 1,
    "output_scores": False,
    "return_dict_in_generate": False,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "remove_invalid_values": False,
    "exponential_decay_length_penalty": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "epsilon_cutoff": 0.0,
    "eta_cutoff": 0.0,
    "encoder_repetition_penalty": 1.0,
    "num_assistant_tokens": 20,
    "num_assistant_tokens_schedule": "constant",
    "assistant_confidence_threshold": 0.4,
    "assistant_lookbehind": 10,
    "target_lookbehind": 10,
    "num_beam_groups": 1,
    "diversity_penalty": 0.0,
}

# Every field a fresh 5.16.1 `GenerationConfig` serializes, all of them `None`
# but the version: what `__init__` sets (generation/configuration_utils.py:
# 392-497) as `to_dict` writes it out (:1172-1191).
_GENERATION_CONFIG_FIELDS = tuple(
    """
    max_length max_new_tokens min_length min_new_tokens early_stopping max_time
    stop_strings do_sample num_beams use_mtp use_cache cache_implementation
    cache_config max_cache_len temperature top_k top_p min_p top_h typical_p
    epsilon_cutoff eta_cutoff repetition_penalty encoder_repetition_penalty
    length_penalty no_repeat_ngram_size bad_words_ids renormalize_logits
    forced_bos_token_id forced_eos_token_id remove_invalid_values
    exponential_decay_length_penalty suppress_tokens begin_suppress_tokens
    sequence_bias token_healing guidance_scale watermarking_config
    num_return_sequences output_attentions output_hidden_states output_scores
    output_logits return_dict_in_generate pad_token_id bos_token_id eos_token_id
    encoder_no_repeat_ngram_size decoder_start_token_id is_assistant
    num_assistant_tokens num_assistant_tokens_schedule
    assistant_confidence_threshold prompt_lookup_num_tokens
    max_matching_ngram_size assistant_early_exit assistant_lookbehind
    target_lookbehind assistant_ensemble_weight speculation_type compile_config
    disable_compile continuous_batching_config low_memory penalty_alpha
    dola_layers diversity_penalty num_beam_groups constraints force_words_ids
    prefill_chunk_size _from_model_config transformers_version
    """.split()
)

# 5.16.1's `METADATA_FIELDS` (generation/configuration_utils.py:46).
_METADATA_FIELDS = (
    "_from_model_config",
    "_commit_hash",
    "_original_object_hash",
    "transformers_version",
)


class FakeGenerationConfig:
    """`transformers.GenerationConfig`, as far as the generation script and
    `generate`'s merge use it.

    A field nobody set is `None`, as every field of a fresh one is in 5.16.1
    (`__init__`, generation/configuration_utils.py:385-518), and a field it
    does not know is kept, as `__init__` keeps one read from a
    `generation_config.json` (:499-505).
    """

    def __init__(self, **settings: Any) -> None:
        for key in _GENERATION_CONFIG_FIELDS:
            setattr(self, key, None)
        for key, value in settings.items():
            setattr(self, key, value)

    def to_json_string(self, use_diff: bool = True, ignore_metadata: bool = False) -> str:
        # `to_dict` writes every field, and the running version over the one
        # read from the file (configuration_utils.py:1172-1191). `use_diff`
        # keeps what a fresh config does not have: a field set to anything but
        # `None`, a field a fresh one lacks, and the version (`to_diff_dict`,
        # :1149-1170).
        settings = {**vars(self), "transformers_version": "5.16.1"}
        if use_diff:
            settings = {
                key: value
                for key, value in settings.items()
                if value is not None
                or key not in _GENERATION_CONFIG_FIELDS
                or key == "transformers_version"
            }
        if ignore_metadata:
            settings = {k: v for k, v in settings.items() if k not in _METADATA_FIELDS}
        return json.dumps(settings, sort_keys=True)

    @staticmethod
    def _get_default_generation_params() -> dict[str, Any]:
        return dict(_TRANSFORMERS_DEFAULTS)


def merged_generation(
    passed: FakeGenerationConfig | None, model_config: FakeGenerationConfig, kwargs: dict
) -> dict[str, Any]:
    """The settings a 5.16.1 `generate` call runs under, by its own merge.

    `_prepare_generation_config` (generation/utils.py:1784-1818): the config
    passed to the call, or a fresh one; every field still unset filled from
    `model.generation_config`, then from the global defaults; then the call's
    keyword arguments over all of it; then a "hybrid" cache cleared.
    """
    merged = {k: v for k, v in vars(passed).items() if v is not None} if passed else {}
    for source in (vars(model_config), _TRANSFORMERS_DEFAULTS):
        for key, value in source.items():
            if merged.get(key) is None:
                merged[key] = value
    merged.update({k: v for k, v in kwargs.items() if k != "input_ids"})
    if merged.get("cache_implementation") == "hybrid":
        merged["cache_implementation"] = None
    return merged


# Qwen/Qwen2.5-0.5B-Instruct's `generation_config.json` at 7ae55760, verbatim.
_QWEN25_GENERATION_CONFIG: dict[str, Any] = {
    "bos_token_id": 151643,
    "pad_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "repetition_penalty": 1.1,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "transformers_version": "4.37.0",
}

# google/functiongemma-270m-it's `generation_config.json` at 39eccb09, verbatim.
_FUNCTIONGEMMA_GENERATION_CONFIG: dict[str, Any] = {
    "cache_implementation": "hybrid",
    "do_sample": True,
    "eos_token_id": [1, 50, 106],
    "top_k": 64,
    "top_p": 0.95,
    "transformers_version": "4.57.3",
}


def _run_hf_generate_script(
    tmp_path: Path,
    monkeypatch,
    *,
    dtype: str = "bfloat16",
    cuda: bool = False,
    given=None,
    stop_generation: bool = False,
    runtime_rendered: bool = False,
    prompts: tuple[str, ...] = ("hi",),
    generation_config: dict[str, Any] | None = None,
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

    `generation_config` is the checkpoint's `generation_config.json`, which
    the fake model loads as its own `generation_config`. Its `generate`
    merges that into the call the way transformers does, and records the
    result as `captured["generation"]`.
    """
    import sys
    import types

    from litetune.evaluate import _HF_GENERATE_SCRIPT

    captured: dict = {}

    class FakeModel:
        def __init__(self) -> None:
            self.generation_config = FakeGenerationConfig(**(generation_config or {}))

        def eval(self) -> "FakeModel":
            return self

        def to(self, device: str) -> "FakeModel":
            # The script places the model before generating. Recorded rather
            # than ignored so a reader can see this double is standing in for
            # a real move, not silently swallowing one.
            captured["model_device"] = device
            return self

        def generate(
            self, generation_config: "FakeGenerationConfig | None" = None, **kwargs: object
        ) -> list[list[int]]:
            captured["generate_input_ids_device"] = getattr(kwargs["input_ids"], "device", None)
            captured["generation"] = merged_generation(
                generation_config, self.generation_config, kwargs
            )
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

    class FakeMps:
        """torch.mps's two counters, answering a sequence that peaks first."""

        current = iter((700, 100, 300))
        driver = iter((900, 200, 400))

        def current_allocated_memory(self) -> int:
            captured["mps_samples"] = captured.get("mps_samples", 0) + 1
            return next(self.current)

        def driver_allocated_memory(self) -> int:
            return next(self.driver)

    fake_torch_module.mps = FakeMps()

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
                "prompts": list(prompts),
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
    assert captured["run_report"]["device"] == "cuda"
    # Off mps nothing is sampled, and the counters are never asked.
    assert captured["run_report"]["mps_memory_sampled_max"] is None
    assert "mps_samples" not in captured


def test_the_reference_script_samples_mps_memory_after_each_generate_call(tmp_path, monkeypatch):
    captured = _run_hf_generate_script(tmp_path, monkeypatch, given="mps", prompts=("a", "b", "c"))
    assert captured["mps_samples"] == 3
    assert captured["run_report"]["mps_memory_sampled_max"] == {
        "current_allocated_bytes": 700,
        "driver_allocated_bytes": 900,
        "samples": 3,
        "sampled_at": ["after each generate call"],
    }


def test_the_reference_script_reports_the_mps_variables_it_saw(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    captured = _run_hf_generate_script(tmp_path, monkeypatch, given="mps")
    assert captured["run_report"]["mps_environment"] == {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.5",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": None,
        "PYTORCH_ENABLE_MPS_FALLBACK": None,
    }


def test_a_reference_run_killed_part_way_still_reports_what_it_saw(tmp_path, monkeypatch):
    captured = _run_hf_generate_script(tmp_path, monkeypatch, given="mps", stop_generation=True)
    assert captured["run_report"]["device"] == "mps"
    assert "mps_environment" in captured["run_report"]


def test_the_reference_script_prefers_what_the_parent_already_resolved(tmp_path, monkeypatch):
    """`given` (`spec["device"]`, what `HuggingFaceBackend` asked
    `envs.resolve_device` before this script started) wins over asking torch
    -- not a second, possibly-disagreeing guess made inside the subprocess.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=False, given="cuda")
    assert captured["model_device"] == "cuda"


# -- the checkpoint's generation config --------------------------------------


def test_the_checkpoints_repetition_penalty_does_not_reach_generate(tmp_path, monkeypatch):
    """transformers fills every field a `generate` call leaves unset from the
    model's own generation config (generation/utils.py:1807 at 5.16.1), and
    builds the repetition-penalty processor whether or not it samples (:1174,
    ahead of the sampling-only block at :1296). Asking for greedy decoding and
    nothing more left Qwen2.5-0.5B-Instruct's `repetition_penalty: 1.1` on
    the float reference.
    """
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config=_QWEN25_GENERATION_CONFIG
    )
    generation = captured["generation"]
    assert generation["repetition_penalty"] == 1.0
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1


def test_the_checkpoints_stop_ids_still_reach_generate(tmp_path, monkeypatch):
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config=_QWEN25_GENERATION_CONFIG
    )
    generation = captured["generation"]
    assert generation["eos_token_id"] == [151645, 151643]
    assert generation["bos_token_id"] == 151643
    # The tokenizer's, as before: the call passes it, and a call's own
    # argument wins over the checkpoint's.
    assert generation["pad_token_id"] == 0


def test_the_reference_reports_the_settings_it_left_out(tmp_path, monkeypatch):
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config=_QWEN25_GENERATION_CONFIG
    )
    assert captured["run_report"]["generation_settings_not_applied"] == {"repetition_penalty": 1.1}
    # Read back under the key `describe()` publishes, from the report the
    # script really wrote: the two sides spell the key separately.
    backend = HuggingFaceBackend(model="org/m", auto_provision=False)
    backend._take_run_report(captured["run_report"])
    assert backend.describe()[GENERATION_SETTINGS_NOT_APPLIED] == {"repetition_penalty": 1.1}


@pytest.mark.parametrize(
    "report",
    [{"device": "cpu"}, {"device": "cpu", GENERATION_SETTINGS_NOT_APPLIED: ["top_k"]}],
    ids=["no record", "not a mapping"],
)
def test_a_report_without_a_record_is_not_published_as_one(report):
    """`None`, not `{}`: a report that says nothing must not read as one
    saying nothing was left out."""
    backend = HuggingFaceBackend(model="org/m", auto_provision=False)
    backend._take_run_report(report)
    assert backend.describe()[GENERATION_SETTINGS_NOT_APPLIED] is None


def test_the_settings_left_out_are_reported_before_generation(tmp_path, monkeypatch):
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config=_QWEN25_GENERATION_CONFIG, stop_generation=True
    )
    assert captured["run_report"]["generation_settings_not_applied"] == {"repetition_penalty": 1.1}


def test_sampling_settings_and_the_cache_are_not_reported_as_left_out(tmp_path, monkeypatch):
    """FunctionGemma's config asks for sampling, which greedy decoding does
    not use -- those processors are built only under `do_sample`
    (generation/utils.py:1296-1343) -- and for a "hybrid" cache, which the
    reference keeps and 5.16.1 clears after the merge (:1817-1818), as it did
    before."""
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config=_FUNCTIONGEMMA_GENERATION_CONFIG
    )
    assert captured["run_report"]["generation_settings_not_applied"] == {}
    assert captured["generation"]["cache_implementation"] is None
    assert captured["generation"]["eos_token_id"] == [1, 50, 106]


def test_every_cache_setting_is_kept(tmp_path, monkeypatch):
    """The four settings 5.16.1 groups as "Parameters that control the cache"
    (generation/configuration_utils.py:405-409) reach the call as before."""
    cache = {
        "use_cache": False,
        "cache_implementation": "static",
        "cache_config": {"device": "cpu"},
        "max_cache_len": 512,
    }
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config={"eos_token_id": 1, **cache}
    )
    assert {key: captured["generation"][key] for key in cache} == cache
    assert captured["run_report"]["generation_settings_not_applied"] == {}


def test_a_setting_at_a_value_that_does_nothing_is_not_reported(tmp_path, monkeypatch):
    """At transformers' own default, at a guidance scale of 1, which builds no
    processor (generation/utils.py:1144), or with `renormalize_logits` off,
    which appends none (:1354)."""
    captured = _run_hf_generate_script(
        tmp_path,
        monkeypatch,
        generation_config={
            "eos_token_id": 1,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "guidance_scale": 1.0,
            "renormalize_logits": False,
        },
    )
    assert captured["run_report"]["generation_settings_not_applied"] == {}


# Settings a 5.16.1 generation config can carry that change what a greedy call
# generates or returns, each at a value that does: a logits processor built
# whatever `do_sample` says (generation/utils.py:1144-1290, watermarking at
# :1346, the log-softmax at :1354-1355, which can move the pick on a logit that
# is not finite), a stopping criterion (:1373-1391), a rewritten prompt (:2595),
# a dict returned instead of a tensor (:2946-2973), or a decoding mode other than
# greedy (`get_generation_mode`, configuration_utils.py:548-598). `constraints`,
# which holds objects, takes `force_words_ids`' branch (:548).
_SETTINGS_THAT_CHANGE_THE_CALL: dict[str, Any] = {
    "guidance_scale": 1.5,
    "sequence_bias": [[[7], -1.0]],
    "encoder_repetition_penalty": 1.2,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "encoder_no_repeat_ngram_size": 3,
    "bad_words_ids": [[7]],
    "min_length": 4,
    "min_new_tokens": 4,
    "forced_bos_token_id": 2,
    "forced_eos_token_id": 1,
    "remove_invalid_values": True,
    "exponential_decay_length_penalty": [4, 1.5],
    "suppress_tokens": [5],
    "begin_suppress_tokens": [6],
    "watermarking_config": {"bias": 2.0},
    "force_words_ids": [[7]],
    "penalty_alpha": 0.6,
    "use_mtp": True,
    "prompt_lookup_num_tokens": 3,
    "assistant_early_exit": 2,
    "dola_layers": "low",
    "max_time": 1.0,
    "stop_strings": ["stop"],
    "is_assistant": True,
    "token_healing": True,
    "renormalize_logits": True,
    "return_dict_in_generate": True,
    # The scores `_sample` collects go to stopping (generation/utils.py:2836,
    # :2900-2902, :2936), and a criterion `is_assistant` adds reads them
    # (:1385-1392; stopping_criteria.py:611).
    "output_scores": True,
    # Not shown by the source to be without effect, so reported too: these two
    # reach the model's forward (generation/utils.py:1831-1834), and more than
    # one sequence stops a greedy call (configuration_utils.py:788-795).
    "output_attentions": True,
    "output_hidden_states": True,
    "num_return_sequences": 2,
}

# Set beside a case so that 5.16.1 accepts its config at all: it refuses more
# than one returned sequence without sampling (configuration_utils.py:788-795).
_COMPANIONS: dict[str, dict[str, Any]] = {"num_return_sequences": {"do_sample": True}}


@pytest.mark.parametrize(("field", "value"), sorted(_SETTINGS_THAT_CHANGE_THE_CALL.items()))
def test_no_setting_that_changes_the_call_reaches_it_unreported(
    tmp_path, monkeypatch, field, value
):
    checkpoint = {"eos_token_id": 1, **_COMPANIONS.get(field, {}), field: value}
    captured = _run_hf_generate_script(tmp_path, monkeypatch, generation_config=checkpoint)
    assert captured["generation"].get(field) == _TRANSFORMERS_DEFAULTS.get(field)
    assert captured["run_report"]["generation_settings_not_applied"] == {field: value}


# A value off transformers' default for every entry of the script's
# `NO_EFFECT_ON_THE_CALL`.
_SETTINGS_WITHOUT_EFFECT: dict[str, Any] = {
    "do_sample": True,
    "num_beams": 4,
    "max_new_tokens": 99,
    "max_length": 99,
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "min_p": 0.1,
    "typical_p": 0.9,
    "top_h": 0.5,
    "epsilon_cutoff": 0.001,
    "eta_cutoff": 0.001,
    "length_penalty": 1.2,
    "early_stopping": True,
    "num_beam_groups": 2,
    "diversity_penalty": 0.5,
    "output_logits": True,
}


@pytest.mark.parametrize(("field", "value"), sorted(_SETTINGS_WITHOUT_EFFECT.items()))
def test_a_setting_without_effect_is_left_out_unreported(tmp_path, monkeypatch, field, value):
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config={"eos_token_id": 1, field: value}
    )
    assert captured["generation"].get(field) != value
    assert captured["run_report"]["generation_settings_not_applied"] == {}


def test_an_assistant_config_is_reported_whole(tmp_path, monkeypatch):
    """Together these stop a 5.16.1 call on a token below the assistant's
    confidence threshold: `is_assistant` adds `ConfidenceCriteria`
    (generation/utils.py:1385-1392), which reads the scores `_sample` collects
    only when both output flags are on (:2836, :2900-2902, :2936). None of the
    three may reach the call, and the record names all three."""
    assistant = {"is_assistant": True, "return_dict_in_generate": True, "output_scores": True}
    captured = _run_hf_generate_script(
        tmp_path, monkeypatch, generation_config={"eos_token_id": 1, **assistant}
    )
    for field in assistant:
        assert captured["generation"].get(field) == _TRANSFORMERS_DEFAULTS.get(field)
    assert captured["run_report"]["generation_settings_not_applied"] == assistant


def test_every_setting_the_script_calls_without_effect_is_pinned():
    """A field added to the script's list without a case above is a claim
    about transformers nothing here checks."""
    from litetune.evaluate import _HF_GENERATE_SCRIPT

    namespace: dict = {"__name__": "litetune_hf_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf_generate.py", "exec"), namespace)  # noqa: S102

    assert set(namespace["NO_EFFECT_ON_THE_CALL"]) == set(_SETTINGS_WITHOUT_EFFECT)


def test_a_setting_litetune_has_no_name_for_is_left_out_and_reported(tmp_path, monkeypatch):
    """A field a later transformers adds, or one a checkpoint invents, does
    not reach the call by default: the reference keeps a named few and leaves
    out the rest. It is reported rather than refused, because leaving it out
    is the greedy decode whatever the field does."""
    captured = _run_hf_generate_script(
        tmp_path,
        monkeypatch,
        generation_config={"eos_token_id": 1, "no_repeat_ngram_size": 3, "logit_floor": -5.0},
    )
    assert captured["generation"].get("logit_floor") is None
    assert captured["generation"]["no_repeat_ngram_size"] == 0
    assert captured["generation"]["eos_token_id"] == 1
    assert captured["run_report"]["generation_settings_not_applied"] == {
        "logit_floor": -5.0,
        "no_repeat_ngram_size": 3,
    }


def test_a_run_with_no_report_does_not_keep_the_last_runs_settings(monkeypatch):
    """One run reports what it left out; the next writes no report at all.
    The first run's record must not be published as the second's."""
    _ready_env()
    reports: list[dict | None] = [
        {"device": "cpu", "generation_settings_not_applied": {"repetition_penalty": 1.1}},
        None,
    ]

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(args, 0, probe_answer("cpu"), "")
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        report = reports.pop(0)
        if report is not None:
            Path(spec["run_report"]).write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)

    backend.generate(["a"])
    assert backend.describe()[GENERATION_SETTINGS_NOT_APPLIED] == {"repetition_penalty": 1.1}
    backend.generate(["b"])
    assert backend.describe()[GENERATION_SETTINGS_NOT_APPLIED] is None


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
        def __init__(self):
            # Every model transformers can generate with carries one
            # (modeling_utils.py:1356-1361 at 5.16.1).
            self.generation_config = FakeGenerationConfig()

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

    def __init__(self, registry, raise_on, leaks=False):
        self.registry = registry
        self.raise_on = raise_on
        self.closed = False
        # A model whose state a new runner on the same engine inherits
        # (google-ai-edge/LiteRT-LM#3165): the answer says how many runners this
        # engine had served before it.
        self.inherited = len(registry) if leaks else None
        self.kwargs: dict = {}
        self.messages: list[dict] = []
        registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def _answer(self, prompt):
        if isinstance(prompt, dict):
            # A row with a picture: answered by its text part, the message kept.
            self.messages.append(prompt)
            prompt = prompt["content"][-1]["text"]
        if prompt in self.raise_on:
            raise self.raise_on[prompt]
        if self.inherited:
            return f"answer to {prompt} after {self.inherited}"
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
        self.leaks = False
        # The fake module, so a test can see whether an engine is open at the
        # moment something else runs.
        self.module = None

    def __enter__(self):
        if self.module is not None:
            # Two at once is what an Engine per prompt must never do.
            assert not self.module.engine_open, "an engine opened while another was open"
            self.module.engine_open = True
        return self

    def __exit__(self, *exc):
        if self.module is not None:
            self.module.engine_open = False
        return False

    def create_session(self, **kwargs):
        runner = _FakeRunner(self.runners, self.raise_on, self.leaks)
        runner.kwargs = kwargs
        return runner

    def create_conversation(self, **kwargs):
        runner = _FakeRunner(self.runners, self.raise_on, self.leaks)
        runner.kwargs = kwargs
        return runner


def _fake_litert_lm(monkeypatch, raise_on=None, leaks=False, engine_error=None):
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
        if engine_error is not None:
            raise engine_error
        engine = _FakeEngine(path, backend, **kwargs)
        engine.raise_on = raise_on or {}
        engine.leaks = leaks
        engine.module = module
        engines.append(engine)
        return engine

    module = types.ModuleType("litert_lm")
    module.engine_open = False
    module.Engine = make_engine
    module.SamplerConfig = lambda **settings: ("SamplerConfig", settings)
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
    engine_per_prompt=None,
    leaks=False,
    engine_error=None,
    bundle_sampler=None,
    extra=None,
):
    engines = _fake_litert_lm(monkeypatch, raise_on, leaks, engine_error)
    # What the bundle's LlmMetadata declares, in place of reading a real header.
    monkeypatch.setitem(_litertlm_script(), "bundle_sampler", lambda path: bundle_sampler)
    spec = tmp_path / "spec.json"
    out = tmp_path / "out.jsonl"
    spec.write_text(
        json.dumps(
            {
                "model": str(tmp_path / "m.litertlm"),
                "prompts": prompts,
                "runtime_rendered": runtime_rendered,
                **(engine if engine is not None else runtime_engine_spec("cpu")),
                "cache_dir": "",
                "out": str(out),
                **({"report": str(report)} if report is not None else {}),
                **(
                    {"engine_per_prompt": engine_per_prompt}
                    if engine_per_prompt is not None
                    else {}
                ),
                **(extra or {}),
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


def test_the_cpu_script_asks_for_the_cache_the_cli_asks_for(tmp_path, monkeypatch):
    """`cache_dir=""` is the CLI's own default, and omitting it is a fourth state."""
    _, engines = _run_script(tmp_path, ["one"], monkeypatch=monkeypatch)
    assert engines[0].kwargs["cache_dir"] == ""


def test_the_gpu_script_uses_the_cache_directory_the_parent_created(tmp_path, monkeypatch):
    cache = tmp_path / "cache-candidate"
    cache.mkdir()

    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        engine=runtime_engine_spec("gpu"),
        extra={"cache_dir": str(cache)},
    )

    assert engines[0].kwargs["cache_dir"] == str(cache)


def test_an_engine_per_prompt_on_the_gpu_uses_the_parents_cache_directory_every_time(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache-candidate"
    cache.mkdir()

    _, engines = _run_script(
        tmp_path,
        ["one", "two"],
        monkeypatch=monkeypatch,
        engine=runtime_engine_spec("gpu"),
        engine_per_prompt=True,
        extra={"cache_dir": str(cache)},
    )

    assert len(engines) == 2
    assert [e.kwargs["cache_dir"] for e in engines] == [str(cache), str(cache)]


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
# after a GPU engine was built, its GPU time grown from 6.5-6.9 million to
# 174-181 million after one reply, in three runs; none for a CPU engine.


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
# The engine reading failed: the client's 10 may all be engine construction.
_ENGINE_UNREAD = _run_report(
    _reading(looked=False, client=None, clients=None), _reading(gpu_time=10, clients={"0x2": 10})
)
_BASELINE_UNTIMED = _run_report(
    _reading(gpu_time=None, clients={"0x2": None}), _reading(gpu_time=10, clients={"0x2": 10})
)
_CLOSED_BY_END = _run_report(_reading(gpu_time=10), _reading(client=None))
_NO_CLIENT_THEN_KILLED = _run_report(_reading(client=None), None)
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


def test_an_empty_app_usage_is_zero_time_and_a_missing_one_is_unread():
    """Real `ioreg -l` prints `"AppUsage" = ()` for a client that has done no
    GPU work yet (65 of 88 clients on an M4 Pro): that is a reading of zero.
    Read as unread, a GPU engine whose client started idle could never be
    observed working."""
    listing = "\n".join(
        [
            "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100000f52, active>",
            '    |   "IOUserClientCreator" = "pid 4242, python3.12"',
            '    |   "AppUsage" = ()',
            "+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100000f53, active>",
            '    |   "IOUserClientCreator" = "pid 4242, python3.12"',
        ]
    )
    client_of = _litertlm_script()["gpu_client_of"]
    assert client_of(listing, 4242)[1] == {"0x100000f52": 0, "0x100000f53": None}


def test_a_client_with_no_registry_id_is_not_compared():
    """Its fallback key is its position in one listing, which moves between
    readings: counted from zero, any time on it would read as new work."""
    listing = "\n".join(
        [
            "+-o AGXDeviceUserClient  <class AGXDeviceUserClient>",
            '    |   "IOUserClientCreator" = "pid 4242, python3.12"',
            '    |   "AppUsage" = ({"API"="Metal","accumulatedGPUTime"=900})',
        ]
    )
    assert _litertlm_script()["gpu_client_of"](listing, 4242)[1] == {"#1": None}


def test_an_idle_client_that_starts_working_is_observed():
    report = _run_report(
        _reading(gpu_time=0, clients={"0x1": 0}), _reading(gpu_time=5000, clients={"0x1": 5000})
    )
    assert gpu_observed("gpu", report) is True
    idle = _run_report(
        _reading(gpu_time=0, clients={"0x1": 0}), _reading(gpu_time=0, clients={"0x1": 0})
    )
    assert gpu_unused("gpu", idle) is True


def test_a_bool_in_a_report_is_not_gpu_time():
    report = _run_report(
        _reading(gpu_time=0, clients={"0x1": 0}), _reading(gpu_time=True, clients={"0x1": True})
    )
    assert gpu_observed("gpu", report) is False


def test_a_platform_nobody_observed_is_not_looked_at(monkeypatch):
    monkeypatch.setattr(sys, "platform", "freebsd14")
    report = _litertlm_script()["device_report"]()
    assert report == _reading(looked=False, client=None, platform="freebsd14")


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

        def as_dict(self):
            return {"key": self._key.decode(), "value": self._value}

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
    peek = types.ModuleType("litert_lm_builder.litertlm_peek")
    peek.read_litertlm_header = lambda path, out: header
    peek.get_model_type = get_model_type
    peek.kvp_to_dict = lambda item: item.as_dict()
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
        # An empty value is a declaration of "", not an absent key.
        [_pd((_KEY, "fp32")), _pd((_KEY, ""))],
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


def test_bundle_vision_constraints_reads_the_production_header_reader(monkeypatch):
    cases = [
        (
            [
                [("model_type", "tf_lite_vision_encoder"), ("backend_constraint", "cpu")],
                [("model_type", "vision_adapter"), ("backend_constraint", "cpu,gpu")],
            ],
            {"vision_encoder": "cpu", "vision_adapter": "cpu,gpu"},
        ),
        (
            [[("model_type", "tf_lite_vision_adapter"), ("backend_constraint", "gpu")]],
            {"vision_adapter": "gpu"},
        ),
        (
            [
                [("model_type", "vision_encoder"), ("backend_constraint", "gpu")],
                [("model_type", "vision_encoder"), ("backend_constraint", "cpu,gpu")],
            ],
            {"vision_encoder": "gpu"},
        ),
        (
            [[("model_type", "vision_encoder"), ("backend_constraint", " gpu ")]],
            {"vision_encoder": "gpu"},
        ),
        (
            [[("model_type", "vision_encoder"), ("backend_constraint", "cpu, gpu")]],
            {"vision_encoder": "cpu,gpu"},
        ),
        ([_pd(("backend_constraint", "cpu"))], {}),
        ([], {}),
    ]
    for sections, expected in cases:
        _fake_builder(monkeypatch, sections)
        assert _litertlm_script()["bundle_vision_constraints"]("m.litertlm") == expected


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
        ("gpu", _ENGINE_UNREAD, False, False),  # no baseline: the time may predate decoding
        ("gpu", _BASELINE_UNTIMED, False, False),  # a client with no time at the engine
        ("gpu", _CLOSED_BY_END, False, False),  # a client that worked may have closed
        ("gpu", _NO_CLIENT_THEN_KILLED, False, False),  # one reading proves nothing
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


def test_the_manifest_names_the_runtime_that_ran_not_only_the_pin(monkeypatch, tmp_path):
    """Runtimes do not reproduce one another's answers, so a result has to say
    which one produced it; the pin says only what was asked for."""
    ran = {**_WORKED, "runtime_version": "0.17.1"}
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}], device=ran)
    )
    backend = _litertlm(tmp_path, backend_flag="cpu")
    backend.generate(["one"])
    assert backend.describe()["runtime_version"] == "0.17.1"

    silent = _litertlm(tmp_path, backend_flag="cpu")
    monkeypatch.setattr(envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}]))
    silent.generate(["one"])
    assert silent.describe()["runtime_version"] is None


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

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(envs.StageEnv, "run", times_out)
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one", "two"])

    assert backend.describe()["device_report"] == _KILLED_AFTER_ENGINE
    assert backend.backend_observed is False
    assert backend.describe()["gpu_unused"] is False


@pytest.mark.parametrize("written", ["{", "[]", '"a string"'])
def test_a_report_that_cannot_be_read_establishes_nothing(monkeypatch, tmp_path, written):
    """Truncated by a kill mid-write, or not an object: either way nothing was
    established, neither may escape `generate`, and the manifest says why."""

    def writes(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["report"]).write_text(written, encoding="utf-8")
        Path(spec["out"]).write_text(json.dumps({"index": 0, "text": "a"}) + "\n")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", writes)
    backend = _litertlm(tmp_path, backend_flag="gpu")
    backend.generate(["one"])

    described = backend.describe()
    assert set(described["device_report"]) == {"unreadable"}
    assert described[BACKEND_OBSERVED] is False
    assert described["gpu_unused"] is False
    assert "unreadable" in described["bundle_activation_error"]
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
    monkeypatch.setitem(ns_main, "device_report", fake_report)
    monkeypatch.setitem(ns_main, "bundle_activation", lambda path: "fp32")
    monkeypatch.setitem(ns_main, "runtime_version", lambda: "0.17.1")
    monkeypatch.setitem(ns_main, "bundle_sampler", lambda path: None)
    engines = _fake_litert_lm(monkeypatch)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "model": str(tmp_path / "m.litertlm"),
                "prompts": ["one"],
                "runtime_rendered": True,
                **runtime_engine_spec("gpu"),
                "cache_dir": str(tmp_path / "cache-candidate"),
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
    assert written["runtime_version"] == "0.17.1"
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


def test_a_silent_gpu_keeps_the_runtimes_last_words_and_claims_nothing_about_the_bundle(
    monkeypatch, tmp_path
):
    """The preflight's timeout keeps the child's stderr, as the split's does;
    and with no report written, the bundle's key was never read -- "declares
    none" would be a claim about a bundle nobody opened."""

    def silent(self, args, timeout=3600, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout, stderr=b"opencl: no device")

    monkeypatch.setattr(envs.StageEnv, "run", silent)
    backend = _litertlm(tmp_path, backend_flag="gpu", timeout_s=7)
    generations = backend.generate(["a", "b"])

    assert "opencl: no device" in generations[0].harness_error
    described = backend.describe()
    assert described["bundle_activation"] is None
    assert described["bundle_activation_error"] == "the run wrote no device report"


@pytest.mark.parametrize(
    ("content", "why"),
    [('{"at_engine": ', "JSONDecodeError"), ("[1, 2]", "a JSON list, not an object")],
)
def test_an_unreadable_device_report_says_so_in_the_manifest(tmp_path, content, why):
    """Not only in a log line: read as "no report" it would be indistinguishable
    from a run that never started."""
    path = tmp_path / "device.json"
    path.write_text(content, encoding="utf-8")
    report = _read_device_report(path)

    assert why in report["unreadable"]
    assert gpu_observed("gpu", report) is False
    assert gpu_unused("gpu", report) is False
    assert "unreadable" in bundle_activation_error_of(report)


def test_a_report_without_a_bundle_reading_is_not_a_bundle_that_declares_none():
    assert bundle_activation_error_of({"at_engine": None}) == (
        "the run's device report carries no reading of the bundle"
    )
    assert bundle_activation_error_of({"bundle_activation": None}) is None


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


# -- which CPU a number was computed on ---------------------------------------

_X86_CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 7B12
flags\t\t: fpu sse2 avx avx2 fma f16c bmi2
"""
_ARM_CPUINFO = """processor\t: 0
Features\t: fp asimd evtstrm aes crc32 atomics fphp asimdhp
CPU implementer\t: 0x41
"""


@pytest.mark.parametrize(
    ("cpuinfo", "model", "flags"),
    [
        (_X86_CPUINFO, "AMD EPYC 7B12", ["avx", "avx2", "f16c", "fma"]),
        (
            _ARM_CPUINFO,
            None,
            ["aes", "asimd", "asimdhp", "atomics", "crc32", "evtstrm", "fp", "fphp"],
        ),
    ],
)
def test_the_driver_names_the_cpu_it_computes_on(monkeypatch, cpuinfo, model, flags):
    """The arithmetic depends on the CPU model: one runtime and one bundle gave
    per-token scores equal to the bit on one Intel model and different on
    every prompt on an AMD one."""
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            return io.StringIO(cpuinfo)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)
    report = _litertlm_script()["cpu_report"]()

    assert report["model"] == model
    assert report["flags"] == flags
    assert "error" not in report


def test_a_cpu_that_cannot_be_read_does_not_end_the_run(monkeypatch):
    import builtins

    def refuse(path, *args, **kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", refuse)
    report = _litertlm_script()["cpu_report"]()

    assert report["model"] is None
    assert "PermissionError" in report["error"]


def test_the_manifest_names_the_cpu_beside_the_runtime(monkeypatch, tmp_path):
    cpu = {"machine": "x86_64", "model": "AMD EPYC 7B12", "flags": ["avx2"], "count": 32}
    ran = {**_WORKED, "runtime_version": "0.17.1", "cpu": cpu}
    monkeypatch.setattr(
        envs.StageEnv, "run", _writes_results([{"index": 0, "text": "a"}], device=ran)
    )
    backend = _litertlm(tmp_path, backend_flag="cpu")
    backend.generate(["one"])

    assert backend.describe()["cpu"] == cpu


def test_the_run_report_carries_the_cpu(monkeypatch):
    ns = _litertlm_script()
    monkeypatch.setitem(ns, "bundle_activation", lambda path: "fp32")
    monkeypatch.setitem(ns, "runtime_version", lambda: "0.17.1")
    monkeypatch.setitem(ns, "cpu_report", lambda: {"model": "Intel(R) Xeon(R) CPU @ 2.20GHz"})

    report = ns["run_report"]({"model": "m.litertlm"}, None)

    assert report["cpu"] == {"model": "Intel(R) Xeon(R) CPU @ 2.20GHz"}


_TWO_CORE_TYPES = """processor\t: 0
model name\t: Big Core
flags\t\t: avx2 avx512f

processor\t: 1
model name\t: Little Core
flags\t\t: avx2
"""


def _with_cpuinfo(monkeypatch, text):
    import builtins
    import io

    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            return io.StringIO(text)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(builtins, "open", fake_open)


def test_the_cpu_named_is_one_this_process_may_run_on(monkeypatch):
    """Found in review: a container pinned off processor 0 on a host with two
    core types was reported as the core it could not use."""
    _with_cpuinfo(monkeypatch, _TWO_CORE_TYPES)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {1}, raising=False)

    report = _litertlm_script()["cpu_report"]()

    assert report["model"] == "Little Core"
    assert report["flags"] == ["avx2"]


def test_an_arm_model_listed_outside_the_per_cpu_blocks_is_found(monkeypatch):
    _with_cpuinfo(monkeypatch, _ARM_CPUINFO + "\nHardware\t: Qualcomm Technologies, Inc SM8650\n")
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)

    report = _litertlm_script()["cpu_report"]()

    assert report["model"] == "Qualcomm Technologies, Inc SM8650"


def test_a_machine_that_will_not_name_itself_does_not_end_the_run(monkeypatch):
    """Found in review: `platform.machine()` ran outside the guard, and a
    sandbox that denies `uname` ended the driver before its first prompt."""
    import platform

    def denied():
        raise PermissionError("uname")

    monkeypatch.setattr(platform, "machine", denied)
    report = _litertlm_script()["cpu_report"]()

    assert report["machine"] is None
    assert "PermissionError" in report["error"]


def test_a_cpu_query_that_times_out_on_macos_does_not_end_the_run(monkeypatch):
    """Found by mutation: every guard test raised an OSError, so narrowing the
    guard to OSError kept them green while a sysctl timeout ended the driver."""

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=10)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", slow)
    report = _litertlm_script()["cpu_report"]()

    assert report["model"] is None
    assert "TimeoutExpired" in report["error"]


def test_an_arm_server_names_its_core_by_implementer_and_part(monkeypatch):
    """arm64 Linux servers print no model line; the implementer and part are
    what name the core, and dropping them made two ARM parts one."""
    graviton = (
        "processor\t: 0\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\n"
        "CPU architecture: 8\nCPU variant\t: 0x1\nCPU part\t: 0xd40\nCPU revision\t: 1\n"
    )
    _with_cpuinfo(monkeypatch, graviton)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)

    report = _litertlm_script()["cpu_report"]()

    assert report["model"] == "implementer 0x41 part 0xd40 variant 0x1 revision 1"


def test_cores_of_two_kinds_open_to_the_process_are_not_named_as_one(monkeypatch):
    """Found in review: with both core types allowed the first was named, though
    either may have computed the number."""
    _with_cpuinfo(monkeypatch, _TWO_CORE_TYPES)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)

    report = _litertlm_script()["cpu_report"]()

    assert report["model"] is None
    assert report["models"] == ["Big Core", "Little Core"]
    assert report["flags"] is None


# ---------------------------------------------------------------------------
# The reference on Apple's GPU, and LITETUNE_DEVICE
# ---------------------------------------------------------------------------


def _reference_env(
    monkeypatch, probe: str = "mps", report_extra: dict | None = None, **probe_fields
) -> list[dict]:
    """A ready `envs.TRAIN` whose probe answers `probe`, and a log of every call.

    Each entry is the argv, the `env` overrides `StageEnv.run` was handed, and
    the spec the generation script was given.
    """
    _ready_env()
    calls: list[dict] = []

    def fake_run(self, args, timeout=3600, **kwargs):
        call: dict = {"argv": list(args), "env": kwargs.get("env"), "spec": None}
        calls.append(call)
        if args[1] == "-c":
            return subprocess.CompletedProcess(args, 0, probe_answer(probe, **probe_fields), "")
        spec = json.loads(Path(args[2]).read_text())
        call["spec"] = spec
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        Path(spec["run_report"]).write_text(
            json.dumps({"device": spec["device"], **(report_extra or {})}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    return calls


def _generation_calls(calls: list[dict]) -> list[dict]:
    return [c for c in calls if c["argv"][1] != "-c"]


def test_an_mps_reference_starts_under_the_memory_policy_and_records_it(monkeypatch):
    calls = _reference_env(monkeypatch, "mps")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    (generation,) = backend.generate(["a"])

    assert generation.harness_error is None
    (run,) = _generation_calls(calls)
    assert run["spec"]["device"] == "mps"
    assert run["env"] == {
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.812500",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.650000",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    described = backend.describe()
    assert described["backend"] == "mps"
    assert described["device_source"] == "probe"
    assert described["device_probe"]["mps_available"] is True
    assert described["mps_memory"]["computed_budget_bytes"] == 13 * 1024**3
    assert described["mps_memory"]["available_bytes"] == 16 * 1024**3
    assert described["host"]["os"] == "Darwin"
    assert described["host"]["os_version"] == "15.0"
    assert described["host"]["machine"] == "arm64"


def test_a_reference_off_mps_gets_no_memory_policy(monkeypatch):
    calls = _reference_env(monkeypatch, "cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    (run,) = _generation_calls(calls)
    assert run["env"] is None
    assert backend.describe()["mps_memory"] is None


def test_an_mps_reference_without_the_memory_is_not_started(monkeypatch):
    from conftest import fake_sysctl, sysctl_available

    from litetune import devices

    calls = _reference_env(monkeypatch, "mps")
    monkeypatch.setattr(devices, "read_sysctl", fake_sysctl(sysctl_available(15 * 1024**3 // 10)))

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    (generation,) = backend.generate(["a"])

    assert _generation_calls(calls) == []
    assert generation.harness_error is not None
    assert "LITETUNE_DEVICE=cpu" in generation.harness_error
    assert backend.mps_refusal == generation.harness_error
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_litetune_device_cpu_puts_the_reference_on_the_cpu(monkeypatch):
    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    calls = _reference_env(monkeypatch, "mps")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    (run,) = _generation_calls(calls)
    assert run["spec"]["device"] == "cpu"
    assert run["env"] is None
    described = backend.describe()
    assert described["backend"] == "cpu"
    assert described["device_source"] == "LITETUNE_DEVICE"
    assert described["mps_memory"] is None


def test_an_unknown_litetune_device_stops_the_reference_before_anything_runs(monkeypatch):
    monkeypatch.setenv("LITETUNE_DEVICE", "gpu")
    calls = _reference_env(monkeypatch, "mps")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    (generation,) = backend.generate(["a"])

    assert calls == []
    assert generation.harness_error is not None
    assert "auto, cpu" in generation.harness_error


def test_the_reference_records_the_chip_on_macos(monkeypatch):
    _reference_env(monkeypatch, "cpu")
    import types

    from litetune import devices

    # The parent's platform, read when the record is made; this suite also
    # runs off macOS.
    monkeypatch.setattr(devices, "sys", types.SimpleNamespace(platform="darwin"))

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert backend.describe()["host"]["chip"] == "Apple M-test"


def test_neither_fallback_chooses_mps_where_torch_could():
    """The scripts' own fallbacks run only without the parent's answer, and so
    without the memory policy: they must not put the run on mps."""
    import types

    from litetune.evaluate import _HF_GENERATE_SCRIPT
    from litetune.tune import _TRAIN_SCRIPT

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
        ),
    )
    generate: dict = {"__name__": "hf_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf_generate.py", "exec"), generate)  # noqa: S102
    train: dict = {"__name__": "train_script_under_test"}
    exec(compile(_TRAIN_SCRIPT, "train_script.py", "exec"), train)  # noqa: S102

    assert generate["generation_device"](torch) == "cpu"
    assert train["training_device"](torch) == "cpu"


def test_litetune_device_cpu_reaches_a_reference_whose_environment_was_not_probed(
    monkeypatch, tmp_path
):
    """The library path: an environment that is not provisioned is not
    probed, and LITETUNE_DEVICE=cpu still decides where the child runs."""
    from litetune.events import EventStream

    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "unprovisioned"))
    specs: list = []

    def fake_run(self, args, timeout=3600, **kwargs):
        assert args[1] != "-c", "an unprovisioned environment was probed"
        spec = json.loads(Path(args[2]).read_text())
        specs.append(spec)
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"], events=events)

    assert specs[0]["device"] == "cpu"
    described = backend.describe()
    assert described["device_source"] == "LITETUNE_DEVICE"
    assert "not provisioned" in backend.last_probe.detail
    assert [
        e
        for e in seen
        if e.kind == "note" and e.data["message"].startswith("LITETUNE_DEVICE=cpu places this run")
    ]


def test_litetune_device_cpu_on_a_probed_reference_is_noted(monkeypatch):
    from litetune.events import EventStream

    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    _reference_env(monkeypatch, "mps")
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    HuggingFaceBackend(model="org/model", auto_provision=False).generate(["a"], events=events)

    notes = [e.data for e in seen if e.kind == "note"]
    assert [
        n for n in notes if n["message"].startswith("LITETUNE_DEVICE=cpu") and n["device"] == "cpu"
    ]


_SENT = {
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.812500",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.650000",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}


def test_an_mps_reference_records_its_samples_and_what_it_saw(monkeypatch):
    sampled = {"current_allocated_bytes": 5, "driver_allocated_bytes": 6, "samples": 1}
    _reference_env(
        monkeypatch,
        "mps",
        report_extra={"mps_environment": _SENT, "mps_memory_sampled_max": sampled},
    )

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    described = backend.describe()
    assert described["mps_memory_sampled_max"] == sampled
    assert described["mps_environment_seen"] == _SENT
    assert backend.mps_environment_mismatch is None


def test_an_mps_reference_that_saw_other_variables_is_a_mismatch(monkeypatch):
    seen = dict(_SENT, PYTORCH_MPS_HIGH_WATERMARK_RATIO=None)
    _reference_env(monkeypatch, "mps", report_extra={"mps_environment": seen})

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert backend.mps_environment_mismatch is not None
    assert "reference generation script" in backend.mps_environment_mismatch


def test_a_reused_backend_forgets_a_refusal_once_the_memory_is_back(monkeypatch):
    """M3: `mps_refusal` is this call's, not the backend's history."""
    from conftest import FAKE_SYSCTL, fake_sysctl, sysctl_available

    from litetune import devices

    calls = _reference_env(monkeypatch, "mps")
    monkeypatch.setattr(devices, "read_sysctl", fake_sysctl(sysctl_available(15 * 1024**3 // 10)))
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    (refused,) = backend.generate(["a"])
    assert refused.harness_error is not None
    assert backend.describe()["mps_refusal"] == refused.harness_error

    monkeypatch.setattr(devices, "read_sysctl", fake_sysctl(FAKE_SYSCTL))
    (ran,) = backend.generate(["a"])

    assert ran.harness_error is None
    assert backend.mps_refusal is None
    assert backend.describe()["mps_refusal"] is None
    assert backend.describe()["mps_memory"]["computed_budget_bytes"] == 13 * 1024**3
    assert len(_generation_calls(calls)) == 1


def test_a_reused_backend_forgets_its_mps_policy_once_litetune_device_is_cpu(monkeypatch):
    """M4: `mps_memory` and the samples are this call's too."""
    sampled = {"current_allocated_bytes": 5, "driver_allocated_bytes": 6, "samples": 1}
    calls = _reference_env(
        monkeypatch,
        "mps",
        report_extra={"mps_environment": _SENT, "mps_memory_sampled_max": sampled},
    )
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])
    assert backend.mps_memory is not None

    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    backend.generate(["a"])

    described = backend.describe()
    assert described["backend"] == "cpu"
    assert described["mps_memory"] is None
    assert described["mps_refusal"] is None
    assert _generation_calls(calls)[-1]["env"] is None


def test_an_mps_reference_leaves_the_parents_environment_as_it_was(monkeypatch):
    import os

    _reference_env(monkeypatch, "mps")
    before = dict(os.environ)

    HuggingFaceBackend(model="org/model", auto_provision=False).generate(["a"])

    assert dict(os.environ) == before


@pytest.mark.parametrize("runtime_rendered", [True, False])
@pytest.mark.parametrize("per_prompt", [False, True])
def test_an_engine_per_prompt_keeps_one_prompts_state_from_the_next(
    tmp_path, monkeypatch, runtime_rendered, per_prompt
):
    """The harness can tell the two apart: on an engine whose runners inherit
    state, one engine for the split answers the second and third prompts from
    what came before, and an engine per prompt -- each closed before the next
    opens, which the fake asserts -- answers every prompt fresh."""
    report = tmp_path / "device.json"
    rows, engines = _run_script(
        tmp_path,
        ["one", "two", "three"],
        runtime_rendered,
        monkeypatch=monkeypatch,
        report=report,
        engine_per_prompt=per_prompt,
        leaks=True,
    )
    texts = [r["text"] for r in rows]
    init = json.loads(report.read_text(encoding="utf-8"))["engine_init_seconds"]
    if per_prompt:
        assert texts == ["answer to one", "answer to two", "answer to three"]
        assert len(engines) == 3 and all(len(e.runners) == 1 for e in engines)
        assert len(init) == 3 and all(t >= 0 for t in init)
        assert engine_init_of({"engine_init_seconds": init})["n"] == 3
    else:
        assert texts == ["answer to one", "answer to two after 1", "answer to three after 2"]
        assert len(engines) == 1
        assert init is None
    report_after = json.loads(report.read_text(encoding="utf-8"))
    assert report_after["after_generation"] is not None


def test_the_backend_asks_for_an_engine_per_prompt_only_when_its_rule_does(tmp_path):
    plain = LiteRtLmBackend(model=tmp_path / "m.litertlm")
    assert plain.describe()["engine_per_prompt"] is False
    assert plain.describe()["transport"] == "litert_lm python api, one process per split"
    fresh = LiteRtLmBackend(model=tmp_path / "m.litertlm", engine_per_prompt_reason="because")
    described = fresh.describe()
    assert described["engine_per_prompt"] is True
    assert described["engine_per_prompt_reason"] == "because"
    assert described["transport"].endswith(", one engine per prompt")


@pytest.mark.parametrize(
    "report",
    [
        None,
        {},
        {"engine_init_seconds": None},
        {"engine_init_seconds": []},
        {"engine_init_seconds": [1, "x"]},
        {"engine_init_seconds": [True]},
    ],
)
def test_init_times_that_are_not_a_list_of_seconds_are_not_reported(report):
    assert engine_init_of(report) is None


def test_init_times_are_summarised():
    assert engine_init_of({"engine_init_seconds": [1.0, 3.0, 2.0]}) == {
        "n": 3,
        "median": 2.0,
        "max": 3.0,
        "total": 6.0,
    }


@pytest.mark.parametrize(("reason", "expected"), [("", False), ("state carries over", True)])
def test_the_script_is_told_to_build_an_engine_per_prompt(tmp_path, monkeypatch, reason, expected):
    specs: list[dict] = []
    write = _writes_results([{"index": 0, "text": "a"}, {"index": 1, "text": "b"}])

    def fake_run(self, args, timeout=3600, **kwargs):
        specs.append(json.loads(Path(args[2]).read_text(encoding="utf-8")))
        return write(self, args, timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    _litertlm(tmp_path, engine_per_prompt_reason=reason).generate(["one", "two"])
    assert [spec["engine_per_prompt"] for spec in specs] == [expected]


def test_a_split_that_dies_part_way_reports_every_engine_it_built(tmp_path, monkeypatch):
    report = tmp_path / "device.json"
    with pytest.raises(MemoryError):
        _run_script(
            tmp_path,
            ["one", "two", "three"],
            monkeypatch=monkeypatch,
            report=report,
            engine_per_prompt=True,
            raise_on={"two": MemoryError()},
        )
    written = json.loads(report.read_text(encoding="utf-8"))
    assert len(written["engine_init_seconds"]) == 2
    assert [
        json.loads(line)["index"] for line in (tmp_path / "out.jsonl").read_text().splitlines()
    ] == [0]


# -- the CPU engine's threads on a CPU with SME2 ------------------------------


def test_a_thread_count_reaches_the_cpu_spec_only_when_given():
    assert runtime_engine_spec("cpu", 1) == {
        "backend": "cpu",
        "activation_data_type": None,
        "thread_count": 1,
    }
    assert "thread_count" not in runtime_engine_spec("cpu", None)
    # A GPU engine takes no thread count, so none is written for one.
    assert runtime_engine_spec("gpu", 1) == {"backend": "gpu", "activation_data_type": "fp32"}


def test_the_driver_passes_a_thread_count_to_the_cpu_backend_only_when_given(tmp_path, monkeypatch):
    _, engines = _run_script(
        tmp_path, ["one"], monkeypatch=monkeypatch, engine=runtime_engine_spec("cpu", 1)
    )
    assert engines[0].backend.name == "cpu"
    assert engines[0].backend.kwargs == {"thread_count": 1}
    _, engines = _run_script(
        tmp_path, ["one"], monkeypatch=monkeypatch, engine=runtime_engine_spec("cpu")
    )
    # `CPU()`, the runtime's default count, as before.
    assert engines[0].backend.kwargs == {}


def test_backend_for_gives_the_gpu_no_thread_count(tmp_path, monkeypatch):
    _fake_litert_lm(monkeypatch)
    backend_for = _litertlm_script()["backend_for"]
    assert backend_for("gpu", 1).kwargs == {}
    assert backend_for("cpu", "2").kwargs == {"thread_count": 2}


def _spec_seen(monkeypatch) -> list[dict]:
    seen: list[dict] = []
    inner = _writes_results([{"index": 0, "text": "fine"}])

    def fake_run(self, args, timeout=3600, **kwargs):
        seen.append(json.loads(Path(args[2]).read_text(encoding="utf-8")))
        return inner(self, args, timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    return seen


def test_on_a_cpu_with_sme2_the_text_path_runs_one_thread_and_records_it(
    tmp_path, monkeypatch, sme2_host
):
    seen = _spec_seen(monkeypatch)
    backend = _litertlm(tmp_path)

    assert backend.generate(["hello"])[0].ok

    assert seen[0]["thread_count"] == 1
    recorded = backend.describe()["cpu_threads"]
    assert recorded["threads"] == 1 and recorded["requested"] is None
    assert "this CPU has SME2" in recorded["reason"]


def test_off_such_a_cpu_the_text_path_passes_no_thread_count(tmp_path, monkeypatch):
    seen = _spec_seen(monkeypatch)
    backend = _litertlm(tmp_path)

    backend.generate(["hello"])

    assert "thread_count" not in seen[0]
    assert backend.describe()["cpu_threads"] == {"threads": None, "requested": None, "reason": None}


def test_a_text_path_given_threads_keeps_them_off_such_a_cpu(tmp_path, monkeypatch):
    seen = _spec_seen(monkeypatch)
    _litertlm(tmp_path, threads=6).generate(["hello"])
    assert seen[0]["thread_count"] == 6


def test_the_gpu_text_path_records_no_cpu_threads(tmp_path, monkeypatch, sme2_host):
    seen = _spec_seen(monkeypatch)
    backend = _litertlm(tmp_path, backend_flag="gpu")

    backend.generate(["hello"])

    assert "thread_count" not in seen[-1]
    assert backend.describe()["cpu_threads"] is None


# -- the sampler and the pictures in the driver's main ------------------------

TOP_P = {"type": "TOP_P", "k": 64, "p": 0.95, "temperature": 1.0}
GREEDY_SENT = {"top_k": 1, "top_p": 1.0, "temperature": 1.0, "seed": 0}


@pytest.mark.parametrize("runtime_rendered", [True, False])
@pytest.mark.parametrize("engine_per_prompt", [False, True])
def test_a_bundle_that_samples_is_sent_top_k_1_on_text_rows_too(
    tmp_path, monkeypatch, runtime_rendered, engine_per_prompt
):
    report = tmp_path / "report.json"
    _, engines = _run_script(
        tmp_path,
        ["one", "two"],
        runtime_rendered,
        monkeypatch=monkeypatch,
        report=report,
        engine_per_prompt=engine_per_prompt,
        bundle_sampler=TOP_P,
        extra={"greedy_sampler": GREEDY_SENT},
    )
    runners = [r for e in engines for r in e.runners]
    assert len(runners) == 2
    assert all(r.kwargs["sampler_config"] == ("SamplerConfig", GREEDY_SENT) for r in runners)
    recorded = json.loads(report.read_text())
    assert recorded["bundle_sampler"] == TOP_P and recorded["sampler_sent"] == GREEDY_SENT


@pytest.mark.parametrize(
    "declared", [None, {"type": "TOP_P", "k": 1, "p": 1.0, "temperature": 1.0}]
)
def test_a_bundle_that_does_not_sample_keeps_its_own(tmp_path, monkeypatch, declared):
    report = tmp_path / "report.json"
    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        report=report,
        bundle_sampler=declared,
        extra={"greedy_sampler": GREEDY_SENT},
    )
    assert engines[0].runners[0].kwargs["sampler_config"] is None
    assert json.loads(report.read_text())["sampler_sent"] is None


def test_the_sampler_rule_in_the_script_is_llm_images_rule():
    from litetune.llm_images import sampler_replaced, sampler_samples

    for declared in (
        None,
        TOP_P,
        {"type": "TOP_K", "k": 1},
        {"type": "TOP_P", "k": 1},
        {"type": "GREEDY", "k": 1},
        {"type": 3, "k": 1},
    ):
        assert _litertlm_script()["sampler_replaced"](declared) is sampler_replaced(declared)
    # GREEDY is replaced, because 0.18.0 does not run it, and does not sample.
    assert sampler_replaced({"type": "GREEDY", "k": 1}) is True
    assert sampler_samples({"type": "GREEDY", "k": 1}) is False
    assert sampler_replaced({"type": "TOP_P", "k": 1}) is False
    assert sampler_samples(TOP_P) is True and sampler_samples(None) is False


@pytest.mark.parametrize("engine_per_prompt", [False, True])
def test_pictures_reach_the_engine_and_each_conversation(tmp_path, monkeypatch, engine_per_prompt):
    pictures = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.jpg"
        path.write_bytes(name.encode())
        pictures.append({"path": str(path), "sha256": hashlib.sha256(name.encode()).hexdigest()})
    monkeypatch.setitem(_litertlm_script(), "bundle_vision_constraints", lambda path: {})
    rows, engines = _run_script(
        tmp_path,
        ["one", "two"],
        monkeypatch=monkeypatch,
        engine_per_prompt=engine_per_prompt,
        extra={"images": pictures, "vision_backend": "cpu"},
    )
    assert [r["text"] for r in rows] == ["answer to one", "answer to two"]
    assert all(e.kwargs["vision_backend"].name == "cpu" for e in engines)
    messages = [m for e in engines for r in e.runners for m in r.messages]
    assert [m["content"][0]["type"] for m in messages] == ["image", "image"]
    assert [m["content"][1]["text"] for m in messages] == ["one", "two"]


def test_a_cpu_constrained_picture_bundle_requests_cpu_vision_and_records_the_sections(
    tmp_path, monkeypatch
):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    report = tmp_path / "report.json"
    monkeypatch.setitem(
        _litertlm_script(),
        "bundle_vision_constraints",
        lambda path: {"vision_encoder": "cpu", "vision_adapter": "cpu"},
    )
    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        report=report,
        extra={
            "backend": "gpu",
            "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
            "vision_backend": "gpu",
        },
    )
    assert engines[0].kwargs["vision_backend"].name == "cpu"
    recorded = json.loads(report.read_text())
    assert recorded["vision_backend"] == "cpu"
    assert recorded["vision_backend_constraints"] == {
        "vision_encoder": "cpu",
        "vision_adapter": "cpu",
    }
    assert "vision_encoder='cpu'" in recorded["vision_backend_reason"]


def test_every_vision_constraint_is_recorded_separately_from_the_gpu_decision(
    tmp_path, monkeypatch
):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    report = tmp_path / "report.json"
    monkeypatch.setitem(
        _litertlm_script(),
        "bundle_vision_constraints",
        lambda path: {"vision_encoder": "cpu,gpu", "vision_adapter": "gpu"},
    )
    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        report=report,
        extra={
            "backend": "gpu",
            "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
            "vision_backend": "cpu",
        },
    )
    assert engines[0].kwargs["vision_backend"].name == "gpu"
    recorded = json.loads(report.read_text())
    assert recorded["vision_backend"] == "gpu"
    assert recorded["vision_backend_constraints"] == {
        "vision_encoder": "cpu,gpu",
        "vision_adapter": "gpu",
    }
    assert "vision_backend_reason" not in recorded


def test_google_embeddinggemma_layout_keeps_gpu_vision_and_records_adapter_limitation(
    tmp_path, monkeypatch
):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    report = tmp_path / "report.json"
    monkeypatch.setitem(
        _litertlm_script(), "bundle_vision_constraints", lambda path: {"vision_adapter": "cpu"}
    )
    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        report=report,
        extra={
            "backend": "gpu",
            "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
            "vision_backend": "gpu",
        },
    )
    assert engines[0].kwargs["vision_backend"].name == "gpu"
    recorded = json.loads(report.read_text())
    assert recorded["vision_backend"] == "gpu"
    assert recorded["vision_backend_constraints"] == {"vision_adapter": "cpu"}
    assert "measured on one Mac with litert-lm 0.18.0" in recorded["vision_backend_reason"]
    assert "which device the adapter used was not observed" in recorded["vision_backend_reason"]


def test_the_driver_refuses_cpu_text_with_gpu_only_vision_as_a_backstop(tmp_path, monkeypatch):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    report = tmp_path / "report.json"
    monkeypatch.setitem(
        _litertlm_script(),
        "bundle_vision_constraints",
        lambda path: {"vision_encoder": "gpu", "vision_adapter": "cpu,gpu"},
    )
    with pytest.raises(ValueError) as raised:
        _run_script(
            tmp_path,
            ["one"],
            monkeypatch=monkeypatch,
            report=report,
            extra={
                "backend": "cpu",
                "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
                "vision_backend": "cpu",
            },
        )
    said = str(raised.value)
    assert "--backend cpu" in said and "vision_encoder='gpu'" in said
    assert "Pass --backend gpu" in said


def test_the_cpu_picture_preflight_returns_the_header_probes_refusal(tmp_path):
    class ProbeEnv:
        name = "runtime"

        def provision(self, events=None):
            return None

        def run(self, argv, timeout=None):
            spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            Path(spec["report"]).write_text(
                json.dumps(
                    {
                        "constraints": {
                            "vision_encoder": "gpu",
                            "vision_adapter": "cpu,gpu",
                        },
                        "error": "vision_encoder='gpu'; Pass --backend gpu",
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, "", "")

    backend = _litertlm(
        tmp_path,
        backend_flag="cpu",
        images=({"path": "a.jpg", "sha256": "0" * 64},),
        env=ProbeEnv(),
    )
    assert backend.picture_backend_problem() == "vision_encoder='gpu'; Pass --backend gpu"


def test_vision_constraints_are_written_before_engine_construction(tmp_path, monkeypatch):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    report = tmp_path / "report.json"
    monkeypatch.setitem(
        _litertlm_script(),
        "bundle_vision_constraints",
        lambda path: {"vision_encoder": "cpu", "vision_adapter": "cpu,gpu"},
    )
    with pytest.raises(RuntimeError, match="engine did not open"):
        _run_script(
            tmp_path,
            ["one"],
            monkeypatch=monkeypatch,
            report=report,
            engine_error=RuntimeError("engine did not open"),
            extra={
                "backend": "gpu",
                "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
                "vision_backend": "gpu",
            },
        )
    recorded = json.loads(report.read_text())
    assert recorded["vision_backend"] == "cpu"
    assert recorded["vision_backend_constraints"] == {
        "vision_encoder": "cpu",
        "vision_adapter": "cpu,gpu",
    }
    assert "--backend gpu" in recorded["vision_backend_reason"]


def test_an_encoder_allowing_neither_cpu_nor_gpu_ends_the_candidate_without_guessing(
    tmp_path, monkeypatch
):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    monkeypatch.setitem(
        _litertlm_script(),
        "bundle_vision_constraints",
        lambda path: {"vision_encoder": "npu", "vision_adapter": "gpu"},
    )
    with pytest.raises(ValueError) as exc:
        _run_script(
            tmp_path,
            ["one"],
            monkeypatch=monkeypatch,
            extra={
                "backend": "gpu",
                "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
                "vision_backend": "gpu",
            },
        )
    assert "vision_encoder='npu'" in str(exc.value)
    assert "leave no cpu or gpu backend" in str(exc.value)


def test_an_unconstrained_gpu_picture_bundle_requests_gpu_vision(tmp_path, monkeypatch):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"a")
    monkeypatch.setitem(_litertlm_script(), "bundle_vision_constraints", lambda path: {})
    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        extra={
            "backend": "gpu",
            "images": [{"path": str(picture), "sha256": hashlib.sha256(b"a").hexdigest()}],
            "vision_backend": "gpu",
        },
    )
    assert engines[0].kwargs["vision_backend"].name == "gpu"


def test_a_bundle_that_declares_greedy_is_sent_top_k_1(tmp_path, monkeypatch):
    """litert-lm 0.18.0 does not run GREEDY, so the driver replaces it like a sampling one."""
    report = tmp_path / "report.json"
    _, engines = _run_script(
        tmp_path,
        ["one"],
        monkeypatch=monkeypatch,
        report=report,
        bundle_sampler={"type": "GREEDY", "k": 1, "p": 0.95, "temperature": 1.0},
        extra={"greedy_sampler": GREEDY_SENT},
    )
    assert engines[0].runners[0].kwargs["sampler_config"] == ("SamplerConfig", GREEDY_SENT)
    assert json.loads(report.read_text())["sampler_sent"] == GREEDY_SENT


# -- the adapter the runtime selected, and the GPU on Windows and Linux --------
#
# The three lines litert-lm 0.18.0 printed on 2026-11-06: an Intel Arc B390 on
# Windows 11, an NVIDIA L4 on Ubuntu 22.04, and the same Ubuntu host without
# NVIDIA's Vulkan driver, where the runtime used Mesa's llvmpipe and reported
# success (measurements/2026-11-06-gpu-linux-windows).

_ARC = (
    "I0000 00:00:1793930400.978420   13428 environment.cc:532] Selected adapter: Intel(R) "
    "Arc(TM) B390 GPU, arch=xe-3lpg, vendor=intel, backend=Direct3D 12, "
    "adapterType=Integrated GPU"
)
_L4 = (
    "I0000 00:00:1793934514.713555    3383 environment.cc:532] Selected adapter: NVIDIA L4, "
    "arch=lovelace, vendor=nvidia, backend=Vulkan, adapterType=Discrete GPU"
)
_LLVMPIPE = (
    "I0000 00:00:1793934412.006476    2858 environment.cc:532] Selected adapter: llvmpipe "
    "(LLVM 15.0.7, 256 bits), arch=software, vendor=mesa, backend=Vulkan, "
    "adapterType=CPU / Software"
)


def _stderr(*lines: str) -> str:
    return "\n".join(["INFO: [environment.cc:36] Creating LiteRT environment", *lines, ""])


def test_the_adapter_line_is_read_field_by_field_with_commas_in_the_name():
    from litetune.evaluate import adapters_of

    assert adapters_of(_stderr(_ARC, _LLVMPIPE)) == [
        {
            "name": "Intel(R) Arc(TM) B390 GPU",
            "arch": "xe-3lpg",
            "vendor": "intel",
            "backend": "Direct3D 12",
            "adapter_type": "Integrated GPU",
        },
        {
            "name": "llvmpipe (LLVM 15.0.7, 256 bits)",
            "arch": "software",
            "vendor": "mesa",
            "backend": "Vulkan",
            "adapter_type": "CPU / Software",
        },
    ]
    # Bytes as a pipe may hand them back, a Windows line ending, and the same
    # adapter named twice (an image engine builds two environments): once.
    assert adapters_of((_L4 + "\r\n" + _L4 + "\n").encode()) == [
        {
            "name": "NVIDIA L4",
            "arch": "lovelace",
            "vendor": "nvidia",
            "backend": "Vulkan",
            "adapter_type": "Discrete GPU",
        }
    ]
    assert adapters_of(None) == [] and adapters_of("no adapter here") == []


@pytest.mark.parametrize(
    ("adapter", "software"),
    [
        (
            {
                "name": "llvmpipe (LLVM 15.0.7, 256 bits)",
                "arch": "software",
                "adapter_type": "CPU / Software",
            },
            True,
        ),
        # The type alone, the arch alone, and each named rasteriser alone.
        ({"name": "Some Adapter", "arch": "x", "adapter_type": "CPU / Software"}, True),
        ({"name": "Some Adapter", "arch": "software", "adapter_type": "Unknown"}, True),
        ({"name": "llvmpipe", "adapter_type": "Unknown"}, True),
        ({"name": "lavapipe", "adapter_type": "Unknown"}, True),
        ({"name": "SwiftShader Device (Subzero)", "adapter_type": "Unknown"}, True),
        ({"name": "Microsoft Basic Render Driver", "adapter_type": "Unknown"}, True),
        ({"name": "WARP", "adapter_type": "Unknown"}, True),
        ({"name": "NVIDIA L4", "arch": "lovelace", "adapter_type": "Discrete GPU"}, False),
        (
            {
                "name": "Intel(R) Arc(TM) B390 GPU",
                "arch": "xe-3lpg",
                "adapter_type": "Integrated GPU",
            },
            False,
        ),
    ],
)
def test_a_software_adapter_is_told_from_a_gpu(adapter, software):
    from litetune.evaluate import hardware_adapter, software_adapter

    assert (software_adapter([adapter]) is not None) is software
    assert (hardware_adapter([adapter]) is not None) is not software


def test_one_software_adapter_among_gpus_leaves_no_hardware_adapter():
    from litetune.evaluate import adapters_of, hardware_adapter, software_adapter

    both = adapters_of(_stderr(_L4, _LLVMPIPE))
    assert software_adapter(both)["name"].startswith("llvmpipe")
    assert hardware_adapter(both) is None
    assert hardware_adapter([]) is None and software_adapter("not a list") is None


@pytest.fixture
def nvidia_host(monkeypatch):
    """A host with an NVIDIA driver and no Vulkan loader variables set."""
    from litetune import evaluate

    monkeypatch.setattr(evaluate, "_nvidia_driver_present", lambda: True)
    for name in evaluate.VULKAN_DRIVER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_the_software_refusal_names_the_adapter_and_on_linux_the_packages(nvidia_host):
    from litetune.evaluate import adapters_of, software_adapter_refusal

    (adapter,) = adapters_of(_LLVMPIPE)
    linux = software_adapter_refusal(adapter, platform="linux")
    assert "llvmpipe (LLVM 15.0.7, 256 bits) (vendor mesa, Vulkan, CPU / Software)" in linux
    assert "libnvidia-gl-<driver version>" in linux and "libegl1" in linux
    windows = software_adapter_refusal(adapter, platform="win32")
    assert "llvmpipe" in windows and "libegl1" not in windows


@pytest.mark.parametrize(
    ("environ", "nvidia", "said", "not_said"),
    [
        # The forced run of MEASUREMENTS.md: the packages were there.
        (
            {"VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"},
            True,
            "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
            "libegl1",
        ),
        ({"VK_DRIVER_FILES": "x.json"}, False, "VK_DRIVER_FILES=x.json", "libegl1"),
        ({"VK_ADD_DRIVER_FILES": "y.json"}, False, "VK_ADD_DRIVER_FILES=y.json", "libegl1"),
        ({}, True, "libegl1", "VK_"),
        ({}, False, "No hardware Vulkan driver offered the runtime an adapter", "libegl1"),
    ],
)
def test_the_linux_advice_says_what_this_host_shows(environ, nvidia, said, not_said):
    from litetune.evaluate import linux_adapter_advice

    advice = linux_adapter_advice(environ, nvidia)
    assert said in advice and not_said not in advice


def test_an_nvidia_driver_is_seen_by_its_proc_file_or_its_tool(monkeypatch):
    from litetune import evaluate

    monkeypatch.setattr(evaluate.os.path, "exists", lambda path: False)
    monkeypatch.setattr(evaluate.shutil, "which", lambda name: None)
    assert evaluate._nvidia_driver_present() is False
    monkeypatch.setattr(
        evaluate.os.path, "exists", lambda path: path == "/proc/driver/nvidia/version"
    )
    assert evaluate._nvidia_driver_present() is True
    monkeypatch.setattr(evaluate.os.path, "exists", lambda path: False)
    monkeypatch.setattr(evaluate.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert evaluate._nvidia_driver_present() is True


def test_a_type_field_ends_at_the_next_comma_and_any_case_of_software_is_software():
    from litetune.evaluate import adapters_of, hardware_adapter, software_adapter

    (later,) = adapters_of(_L4 + ", driver=580")
    assert later["adapter_type"] == "Discrete GPU"
    (lower,) = adapters_of(_L4.replace("Discrete GPU", "cpu / software"))
    assert software_adapter([lower]) is not None
    (arch,) = adapters_of(_L4.replace("arch=lovelace", "arch=Software"))
    assert software_adapter([arch]) is not None and hardware_adapter([arch]) is None


@pytest.mark.parametrize(
    ("line", "software"),
    [
        ("Selected adapter: llvmpipe in a new shape", True),
        ("Selected adapter: Some GPU; adapterType=CPU / Software", True),
        ("Selected adapter: a GPU in a new shape", False),
    ],
)
def test_a_line_that_does_not_parse_is_kept_and_judged_by_its_words(line, software):
    from litetune.evaluate import adapters_of, hardware_adapter, software_adapter

    (adapter,) = adapters_of(line)
    assert adapter == {"raw": line}
    assert (software_adapter([adapter]) is not None) is software
    # Never a hardware adapter: what it is was not read.
    assert hardware_adapter([adapter]) is None


@pytest.mark.parametrize(
    ("platform", "read"),
    [("darwin", True), ("win32", True), ("linux", True), ("freebsd14", False), ("cygwin", False)],
)
def test_the_gpu_is_read_on_three_platforms(platform, read):
    from litetune.evaluate import gpu_read_here

    assert gpu_read_here(platform) is read


def _smi(listed=True, memory=3105, hidden=False, gpus=1):
    return {
        "listed": listed,
        "used_memory_mib": memory if listed else None,
        "pids_hidden": hidden,
        "gpus": gpus,
    }


def _linux(adapters=None, ended=True, smi=None):
    reading = {**_reading(looked=False, client=None, platform="linux"), "nvidia_smi": smi}
    report = _run_report(reading, reading if ended else None)
    if adapters is not None:
        report["adapters"] = adapters
    return report


def test_on_linux_a_hardware_adapter_and_nvidia_smi_listing_the_process_is_the_reading():
    from litetune.evaluate import adapter_selected, adapters_of, gpu_evidence

    l4 = _linux(adapters_of(_L4), smi=_smi())
    assert gpu_observed("gpu", l4) is True and gpu_unused("gpu", l4) is False
    assert adapter_selected("gpu", l4) is False
    assert gpu_evidence("gpu", l4) == (
        "the runtime selected a hardware adapter, NVIDIA L4 (vendor nvidia, Vulkan, Discrete "
        "GPU), and nvidia-smi listed the process holding 3105 MiB on the host's only NVIDIA GPU "
        "after the work: the process held a context there, which places the run on that GPU "
        "and is not a reading of each operation"
    )
    # A run killed before its second reading has not shown its work.
    assert gpu_observed("gpu", _linux(adapters_of(_L4), ended=False, smi=_smi())) is False
    assert gpu_observed("cpu", l4) is False


@pytest.mark.parametrize(
    "smi",
    [None, _smi(listed=False), _smi(listed=False, hidden=True), {"error": "nvidia-smi exited 9"}],
)
def test_on_linux_an_adapter_nothing_showed_working_is_selected_not_observed(smi):
    from litetune.evaluate import adapter_selected, adapter_selected_note, adapters_of

    report = _linux(adapters_of(_L4), smi=smi)
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is False
    assert adapter_selected("gpu", report) is True
    note = adapter_selected_note("gpu", report)
    assert note.startswith("the runtime selected NVIDIA L4 (vendor nvidia, Vulkan, Discrete GPU)")
    assert "said as selected there, not as measured on it" in note
    assert adapter_selected_note("cpu", report) is None


def test_the_selected_note_says_why_nvidia_smi_did_not_show_the_process():
    from litetune.evaluate import adapter_selected_note, adapters_of

    def note(smi):
        return adapter_selected_note("gpu", _linux(adapters_of(_L4), smi=smi))

    assert "no nvidia-smi was there" in note(None)
    assert "nvidia-smi did not list this process" in note(_smi(listed=False))
    assert "without pids" in note(_smi(listed=False, hidden=True))
    assert "could not be read (nvidia-smi exited 9)" in note({"error": "nvidia-smi exited 9"})


def test_a_line_on_another_platform_or_no_line_on_linux_is_not_selected():
    from litetune.evaluate import adapter_selected, adapters_of

    assert adapter_selected("gpu", _linux(smi=_smi())) is False
    darwin = {**_NOT_LOOKED, "adapters": adapters_of(_L4)}
    darwin["after_generation"] = {**darwin["after_generation"], "platform": "darwin"}
    assert gpu_observed("gpu", darwin) is False and adapter_selected("gpu", darwin) is False
    # nvidia-smi alone, with no adapter line: not observed.
    assert gpu_observed("gpu", _linux(smi=_smi())) is False


def test_a_finished_gpu_run_on_linux_with_no_hardware_adapter_line_is_refused(nvidia_host):
    from litetune.evaluate import adapter_refusal, adapters_of

    nothing = adapter_refusal([], finished=True, platform="linux")
    assert "named no adapter on stderr" in nothing and "software rasteriser" in nothing
    unread = adapter_refusal(
        adapters_of("Selected adapter: a GPU in a new shape"), finished=True, platform="linux"
    )
    assert "a line litetune cannot read ('Selected adapter: a GPU in a new shape')" in unread
    # A run that failed says why itself; other platforms read the kernel or counters.
    assert adapter_refusal([], finished=False, platform="linux") is None
    assert adapter_refusal([], finished=True, platform="win32") is None
    assert adapter_refusal(adapters_of(_L4), finished=True, platform="linux") is None
    # A software adapter whatever happened, on every platform.
    for platform, finished in (("win32", False), ("darwin", True), ("linux", False)):
        said = adapter_refusal(adapters_of(_LLVMPIPE), finished=finished, platform=platform)
        assert "selected a software adapter" in said


def test_a_software_adapter_is_unused_gpu_even_beside_kernel_work():
    from litetune.evaluate import DARWIN_EVIDENCE, adapters_of, gpu_evidence

    linux = _linux(adapters_of(_LLVMPIPE), smi=_smi())
    assert gpu_observed("gpu", linux) is False and gpu_unused("gpu", linux) is True
    darwin = {**_WORKED, "adapters": adapters_of(_stderr(_ARC, _LLVMPIPE))}
    assert gpu_observed("gpu", darwin) is False and gpu_unused("gpu", darwin) is True
    assert gpu_evidence("gpu", darwin) is None
    # macOS says it as it always has.
    assert (
        gpu_evidence("gpu", _WORKED)
        == DARWIN_EVIDENCE
        == (
            "the kernel showed the candidate's process doing work on this host's GPU while it "
            "generated"
        )
    )


# The two adapters of the Panther Lake machine's counters: the process held
# counters on both (windows-ptl/gpu_split.json), and all its time went to A62B.
_ARC_LUID = "0x00000000_0x0000A62B"
_OTHER_LUID = "0x00000000_0x0000AA0B"
_DXGI = [
    {"description": "Intel(R) Arc(TM) B390 GPU", "luid": _ARC_LUID, "software": False},
    {
        "description": "Microsoft Basic Render Driver",
        "luid": "0x00000000_0x0000AA36",
        "software": True,
    },
]


# Win32_PerfRawData_GPUPerformanceCounters_GPUEngine as the query hands it
# back: every process's instances, RunningTime as a string.
def _engines(pid, times, others=True, luid=_ARC_LUID):
    rows = [
        {"n": f"pid_{pid}_luid_{luid}_phys_0_eng_{i}_engtype_3D", "t": str(t)}
        for i, t in enumerate(times)
    ]
    if others:
        rows.insert(0, {"n": f"pid_4_luid_{_ARC_LUID}_phys_0_eng_0_engtype_3D", "t": "999999"})
        rows.append({"n": f"pid_{pid}0_luid_0x0_0x1_phys_0_eng_0_engtype_3D", "t": "77"})
    return rows


def _windows(monkeypatch, *answers, dxgi=_DXGI):
    """`subprocess.run` answering the counter query with each of `answers` in turn."""
    seen, queue = [], list(answers)

    def fake_run(argv, **kwargs):
        seen.append(argv)
        answer = queue.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, subprocess.CompletedProcess):
            return answer
        return subprocess.CompletedProcess(argv, 0, json.dumps(answer).encode(), b"")

    def listed():
        if isinstance(dxgi, BaseException):
            raise dxgi
        return list(dxgi)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", "C:/Windows")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setitem(_litertlm_script(), "dxgi_adapters", listed)
    return seen


def _arc(report):
    """`report` with the adapter line the runtime printed on the Panther Lake machine."""
    from litetune.evaluate import adapters_of

    return {**report, "adapters": adapters_of(_ARC)}


def test_on_windows_this_pids_gpu_engine_counters_and_dxgis_adapters_are_read(monkeypatch):
    pid = os.getpid()
    seen = _windows(monkeypatch, _engines(pid, [60343, 0]))
    script = _litertlm_script()
    report = script["device_report"]()

    (argv,) = seen
    assert argv[0] == os.path.join(
        "C:/Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )
    assert argv[1:4] == ["-NoProfile", "-NonInteractive", "-Command"]
    assert argv[4] == script["WINDOWS_GPU_QUERY"]
    assert "Win32_PerfRawData_GPUPerformanceCounters_GPUEngine" in argv[4]
    assert report["looked"] is True
    # No instance is singled out any more.
    assert report["gpu_client"] is None
    assert report["dxgi_adapters"] == _DXGI
    # Another pid, and a pid this one is a prefix of, are not this process's.
    assert report["gpu_clients"] == {
        f"luid_{_ARC_LUID}_phys_0_eng_0_engtype_3D": 60343,
        f"luid_{_ARC_LUID}_phys_0_eng_1_engtype_3D": 0,
    }
    assert report["gpu_time"] == 60343


def test_a_dxgi_listing_that_fails_is_recorded_and_the_counters_still_read(monkeypatch):
    pid = os.getpid()
    _windows(monkeypatch, _engines(pid, [5]), dxgi=OSError("CreateDXGIFactory1 returned 0x1"))
    report = _litertlm_script()["device_report"]()
    assert report["dxgi_adapters"] is None
    assert report["dxgi_error"] == "OSError: CreateDXGIFactory1 returned 0x1"
    assert report["looked"] is True and report["gpu_time"] == 5


def test_on_windows_growth_on_the_selected_adapter_is_work(monkeypatch):
    from litetune.evaluate import gpu_evidence, windows_luid_times

    pid = os.getpid()
    _windows(monkeypatch, _engines(pid, [60343, 7]), _engines(pid, [30493406, 7]))
    device = _litertlm_script()["device_report"]
    worked = _arc(_run_report(device(), device()))
    assert gpu_observed("gpu", worked) is True
    assert windows_luid_times(worked) == {_ARC_LUID.lower(): {"start": 60350, "end": 30493413}}
    said = gpu_evidence("gpu", worked)
    assert "Intel(R) Arc(TM) B390 GPU" in said and f"LUID {_ARC_LUID.lower()}" in said
    assert "grew from 60350 to 30493413 (100 ns units)" in said
    assert f"most on luid_{_ARC_LUID}_phys_0_eng_0_engtype_3D" in said


def test_on_windows_growth_on_another_adapter_is_not_the_selected_ones(monkeypatch):
    """The process held counters on two adapters; only the named one's count."""
    pid = os.getpid()
    _windows(
        monkeypatch,
        _engines(pid, [0], luid=_ARC_LUID) + _engines(pid, [10], others=False, luid=_OTHER_LUID),
        _engines(pid, [0], luid=_ARC_LUID) + _engines(pid, [900], others=False, luid=_OTHER_LUID),
    )
    device = _litertlm_script()["device_report"]
    report = _arc(_run_report(device(), device()))
    assert gpu_observed("gpu", report) is False
    assert gpu_unused("gpu", report) is True


def test_on_windows_no_line_no_mapping_or_no_counter_on_the_adapter_is_not_observed(monkeypatch):
    pid = os.getpid()
    device = _litertlm_script()["device_report"]
    # Growth, and no adapter line: not observed.
    _windows(monkeypatch, _engines(pid, [1]), _engines(pid, [99]))
    report = _run_report(device(), device())
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is False
    # Growth, a line whose name DXGI does not list: not observed.
    _windows(monkeypatch, _engines(pid, [1]), _engines(pid, [99]), dxgi=_DXGI[1:])
    report = _arc(_run_report(device(), device()))
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is False
    # DXGI could not be read: not observed.
    _windows(monkeypatch, _engines(pid, [1]), _engines(pid, [99]), dxgi=OSError("no"))
    assert gpu_observed("gpu", _arc(_run_report(device(), device()))) is False
    # Mapped, and no counter of this process on that adapter at either reading.
    _windows(monkeypatch, _engines(pid, []), _engines(pid, []))
    report = _arc(_run_report(device(), device()))
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is False


def test_on_windows_another_named_hardware_adapters_growth_is_not_the_selected_ones(monkeypatch):
    pid = os.getpid()
    dxgi = [*_DXGI, {"description": "Another GPU", "luid": _OTHER_LUID, "software": False}]
    _windows(
        monkeypatch,
        _engines(pid, [3], luid=_ARC_LUID) + _engines(pid, [10], others=False, luid=_OTHER_LUID),
        _engines(pid, [3], luid=_ARC_LUID) + _engines(pid, [900], others=False, luid=_OTHER_LUID),
        dxgi=dxgi,
    )
    device = _litertlm_script()["device_report"]
    report = _arc(_run_report(device(), device()))
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is True


def test_on_windows_a_software_dxgi_adapter_of_the_same_name_is_not_mapped(monkeypatch):
    pid = os.getpid()
    same = [{**_DXGI[0], "software": True}]
    _windows(monkeypatch, _engines(pid, [1]), _engines(pid, [99]), dxgi=same)
    device = _litertlm_script()["device_report"]
    assert gpu_observed("gpu", _arc(_run_report(device(), device()))) is False


def test_on_windows_a_counter_that_appeared_after_the_engine_has_no_baseline(monkeypatch):
    pid = os.getpid()
    device = _litertlm_script()["device_report"]
    _windows(monkeypatch, _engines(pid, [5]), _engines(pid, [5, 400]))
    report = _arc(_run_report(device(), device()))
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is False
    # Present at both and flat, one appeared at zero: no growth, a "no".
    _windows(monkeypatch, _engines(pid, [5]), _engines(pid, [5, 0]))
    assert gpu_unused("gpu", _arc(_run_report(device(), device()))) is True
    # One that closed in between: not established.
    _windows(monkeypatch, _engines(pid, [5, 6]), _engines(pid, [5]))
    report = _arc(_run_report(device(), device()))
    assert gpu_unused("gpu", report) is False and gpu_observed("gpu", report) is False


def test_on_windows_an_unread_time_on_the_adapter_establishes_nothing():
    from litetune.evaluate import _windows_work, windows_luid_times

    key = f"luid_{_ARC_LUID}_phys_0_eng_0_engtype_3D"
    reading = {"platform": "win32", "looked": True, "dxgi_adapters": _DXGI}
    report = _arc(
        _run_report({**reading, "gpu_clients": {key: None}}, {**reading, "gpu_clients": {key: 5}})
    )
    assert _windows_work(report) == (None, None)
    assert windows_luid_times(report) == {_ARC_LUID.lower(): {"start": None, "end": 5}}
    # The engine reading did not look: nothing to compare with.
    report["at_engine"] = {**reading, "looked": False, "gpu_clients": {key: 0}}
    assert _windows_work(report) == (None, None)


def test_windows_gpu_time_is_none_with_no_instance_of_this_process(monkeypatch):
    _windows(monkeypatch, _engines(os.getpid(), []))
    report = _litertlm_script()["device_report"]()
    assert report["looked"] is True and report["gpu_time"] is None and report["gpu_clients"] == {}


def test_one_counter_instance_comes_back_as_an_object_and_is_still_read(monkeypatch):
    pid = os.getpid()
    (row,) = _engines(pid, [12], others=False)
    _windows(monkeypatch, row)
    report = _litertlm_script()["device_report"]()
    assert report["looked"] is True and report["gpu_time"] == 12


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        ([], "no GPU engine counter of any process"),
        (subprocess.CompletedProcess(["ps"], 1, b"", b"Get-CimInstance: denied"), "denied"),
        (subprocess.CompletedProcess(["ps"], 0, b"not json", b""), "JSONDecodeError"),
        (FileNotFoundError("no powershell"), "FileNotFoundError"),
        (subprocess.TimeoutExpired(cmd="powershell", timeout=60), "TimeoutExpired"),
    ],
)
def test_a_counter_query_that_fails_did_not_look_and_never_ends_the_run(monkeypatch, answer, said):
    """Read as looked-and-found-none, any of these would say the GPU was not used."""
    _windows(monkeypatch, answer)
    report = _litertlm_script()["device_report"]()
    assert report["looked"] is False
    assert said in report["error"]


def test_a_counter_time_that_is_not_an_integer_is_unread():
    engines_of = _litertlm_script()["gpu_engines_of"]
    rows = [
        {"n": "pid_7_a", "t": True},
        {"n": "pid_7_b", "t": "n/a"},
        {"n": "pid_7_c", "t": None},
        {"n": "pid_7_d", "t": 5},
        "not a row",
        {"n": 7, "t": "1"},
    ]
    assert engines_of(rows, 7) == {"a": None, "b": None, "c": None, "d": 5}


_ONE_GPU = subprocess.CompletedProcess(["smi"], 0, b"GPU-1111\n", b"")


def _linux_smi(monkeypatch, answer, which="/usr/bin/nvidia-smi", pids=None, gpus=_ONE_GPU):
    """nvidia-smi answering the GPU query with `gpus` and the process query with `answer`."""
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        reply = gpus if argv[1] == "--query-gpu=uuid" else answer
        if isinstance(reply, BaseException):
            raise reply
        return reply

    import shutil

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shutil, "which", lambda name: which if name == "nvidia-smi" else None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    if pids is not None:
        monkeypatch.setitem(_litertlm_script(), "own_pids", lambda: set(pids))
    return seen


def test_on_linux_nvidia_smi_is_recorded_and_the_kernel_is_not_asked(monkeypatch):
    pid = os.getpid()
    listing = f"12, 100, GPU-1111\n{pid}, 3105, GPU-1111\n".encode()
    seen = _linux_smi(monkeypatch, subprocess.CompletedProcess(["smi"], 0, listing, b""))
    report = _litertlm_script()["device_report"]()
    assert seen == [
        ["/usr/bin/nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        [
            "/usr/bin/nvidia-smi",
            "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
    ]
    assert report["looked"] is False
    assert report["nvidia_smi"] == {
        "listed": True,
        "used_memory_mib": 3105,
        "gpu_uuid": "GPU-1111",
        "gpus": 1,
        "pids_hidden": False,
    }
    # Listed, and no adapter line: still not observed.
    assert gpu_observed("gpu", _run_report(report, report)) is False


def test_nvidia_smi_finds_the_process_by_its_pid_in_another_namespace(monkeypatch):
    listing = b"[N/A], 50\n77123, 876\n"
    _linux_smi(
        monkeypatch, subprocess.CompletedProcess(["smi"], 0, listing, b""), pids={"12", "77123"}
    )
    assert _litertlm_script()["device_report"]()["nvidia_smi"] == {
        "listed": True,
        "used_memory_mib": 876,
        "gpu_uuid": None,
        "gpus": 1,
        "pids_hidden": True,
    }


def test_own_pids_reads_every_namespace_from_proc(monkeypatch, tmp_path):
    import builtins

    status = tmp_path / "status"
    status.write_text("Name:\tpython\nNSpid:\t77123\t12\n", encoding="utf-8")
    real_open = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda path, *a, **k: real_open(status if path == "/proc/self/status" else path, *a, **k),
    )
    assert _litertlm_script()["own_pids"]() == {str(os.getpid()), "77123", "12"}


@pytest.mark.parametrize(
    ("answer", "which", "recorded"),
    [
        (
            subprocess.CompletedProcess(["smi"], 0, b"12, 100\n", b""),
            "/usr/bin/nvidia-smi",
            {
                "listed": False,
                "used_memory_mib": None,
                "gpu_uuid": None,
                "gpus": 1,
                "pids_hidden": False,
            },
        ),
        (
            subprocess.CompletedProcess(["smi"], 0, b"[N/A], 100\n", b""),
            "/usr/bin/nvidia-smi",
            {
                "listed": False,
                "used_memory_mib": None,
                "gpu_uuid": None,
                "gpus": 1,
                "pids_hidden": True,
            },
        ),
        (
            subprocess.CompletedProcess(["smi"], 9, b"", b"Failed to initialize NVML"),
            "/usr/bin/nvidia-smi",
            {"error": "nvidia-smi exited 9: Failed to initialize NVML"},
        ),
        (
            subprocess.CompletedProcess(["smi"], 9, b"No devices were found", b""),
            "/usr/bin/nvidia-smi",
            {"error": "nvidia-smi exited 9: No devices were found"},
        ),
        (OSError("gone"), "/usr/bin/nvidia-smi", {"error": "OSError: gone"}),
        (None, None, None),
    ],
)
def test_nvidia_smi_that_lists_nothing_or_fails_is_recorded_as_such(
    monkeypatch, answer, which, recorded
):
    _linux_smi(monkeypatch, answer, which=which)
    assert _litertlm_script()["device_report"]()["nvidia_smi"] == recorded


def test_an_unread_memory_figure_is_none(monkeypatch):
    listing = f"{os.getpid()}, [N/A]\n".encode()
    _linux_smi(monkeypatch, subprocess.CompletedProcess(["smi"], 0, listing, b""))
    smi = _litertlm_script()["device_report"]()["nvidia_smi"]
    assert smi["listed"] is True and smi["used_memory_mib"] is None


def _gpu_split(stderr, device=None, preflight_stderr=None, calls=None, returncode=0):
    """`StageEnv.run` for a GPU split: the one-prompt preflight, then the split."""

    def fake_run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        if calls is not None:
            calls.append(len(spec["prompts"]))
        rows = [{"index": i, "text": "a"} for i in range(len(spec["prompts"]))]
        Path(spec["out"]).write_text("".join(json.dumps(r) + "\n" for r in rows))
        if "report" not in spec:
            return subprocess.CompletedProcess(args, 0, "", preflight_stderr or "")
        if device is not None:
            Path(spec["report"]).write_text(json.dumps(device), encoding="utf-8")
        return subprocess.CompletedProcess(args, returncode, "", stderr)

    return fake_run


def test_the_gpu_parent_creates_a_private_cache_and_threads_it_through_the_spec(
    monkeypatch, tmp_path
):
    seen = []

    def fake_run(self, args, timeout=3600, **kwargs):
        spec_path = Path(args[2])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        cache = Path(spec["cache_dir"])
        seen.append((spec_path.parent, cache, cache.is_dir()))
        rows = [json.dumps({"index": i, "text": "a"}) for i in range(len(spec["prompts"]))]
        Path(spec["out"]).write_text("\n".join(rows) + "\n", encoding="utf-8")
        if "report" in spec:
            Path(spec["report"]).write_text(json.dumps(_WORKED), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(envs.StageEnv, "run", fake_run)

    _litertlm(tmp_path, backend_flag="gpu").generate(["one", "two"])

    assert len(seen) == 2  # one preflight driver, then the split driver
    assert all(existed and cache == work / "cache-candidate" for work, cache, existed in seen)
    assert seen[0][1] != seen[1][1]


def test_the_cpu_parent_keeps_the_cli_default_cache_in_the_spec(monkeypatch, tmp_path):
    seen = []

    def fake_run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        seen.append(spec["cache_dir"])
        Path(spec["out"]).write_text('{"index": 0, "text": "a"}\n', encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(envs.StageEnv, "run", fake_run)

    _litertlm(tmp_path, backend_flag="cpu").generate(["one"])

    assert seen == [""]


def test_a_gpu_split_on_a_software_adapter_is_a_harness_failure_on_every_row(
    monkeypatch, tmp_path, nvidia_host
):
    """The runtime answers on llvmpipe and reports success; the answers are the CPU's."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(envs.StageEnv, "run", _gpu_split(_stderr(_LLVMPIPE), _linux()))
    backend = _litertlm(tmp_path, backend_flag="gpu")
    out = backend.generate(["one"])

    assert [g.text for g in out] == [""]
    assert "selected a software adapter, llvmpipe" in out[0].harness_error
    assert "libegl1" in out[0].harness_error
    described = backend.describe()
    assert described["adapters"][0]["adapter_type"] == "CPU / Software"
    assert described[BACKEND_OBSERVED] is False and described["gpu_unused"] is True


def test_a_gpu_split_on_linux_naming_no_adapter_is_a_harness_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(envs.StageEnv, "run", _gpu_split("no adapter here", _linux()))
    out = _litertlm(tmp_path, backend_flag="gpu").generate(["one"])
    assert [g.text for g in out] == [""]
    assert "named no adapter on stderr" in out[0].harness_error


def test_a_failed_gpu_split_on_linux_naming_no_adapter_keeps_its_own_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(envs.StageEnv, "run", _gpu_split("boom", _linux(), returncode=1))
    (out,) = _litertlm(tmp_path, backend_flag="gpu").generate(["one"])
    assert "named no adapter" not in (out.harness_error or "")


def test_the_preflight_stops_a_split_on_a_software_adapter(monkeypatch, tmp_path):
    calls: list[int] = []
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _gpu_split(_stderr(_L4), _linux(), preflight_stderr=_stderr(_LLVMPIPE), calls=calls),
    )
    out = _litertlm(tmp_path, backend_flag="gpu").generate(["one", "two"])

    assert calls == [1]
    assert all("software adapter, llvmpipe" in g.harness_error for g in out)


def test_the_preflight_on_linux_stops_a_split_that_named_no_adapter(monkeypatch, tmp_path):
    calls: list[int] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        envs.StageEnv, "run", _gpu_split(_stderr(_L4), _linux(), preflight_stderr="", calls=calls)
    )
    out = _litertlm(tmp_path, backend_flag="gpu").generate(["one", "two"])
    assert calls == [1]
    assert all("named no adapter" in g.harness_error for g in out)


def test_a_preflight_that_times_out_on_a_software_adapter_says_which(monkeypatch, tmp_path):
    def slow(self, args, timeout=3600, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout, stderr=_stderr(_LLVMPIPE))

    monkeypatch.setattr(envs.StageEnv, "run", slow)
    out = _litertlm(tmp_path, backend_flag="gpu").generate(["one", "two"])
    assert all("software adapter, llvmpipe" in g.harness_error for g in out)


def test_a_split_killed_on_a_software_adapter_scores_no_row(monkeypatch, tmp_path):
    def killed(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(json.dumps({"index": 0, "text": "a"}) + "\n")
        if "report" not in spec:
            return subprocess.CompletedProcess(args, 0, "", "")
        Path(spec["report"]).write_text(json.dumps(_linux()), encoding="utf-8")
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout, stderr=_stderr(_LLVMPIPE))

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(envs.StageEnv, "run", killed)
    backend = _litertlm(tmp_path, backend_flag="gpu")
    out = backend.generate(["one", "two"])
    assert [g.text for g in out] == ["", ""]
    assert all("software adapter, llvmpipe" in g.harness_error for g in out)
    # The killed split's report keeps the adapter its stderr named.
    assert backend.describe()["adapters"][0]["name"].startswith("llvmpipe")


def test_a_cpu_run_is_not_judged_by_an_adapter_line(monkeypatch, tmp_path):
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        _writes_results([{"index": 0, "text": "a"}], stderr=_stderr(_LLVMPIPE)),
    )
    (out,) = _litertlm(tmp_path, backend_flag="cpu").generate(["one"])
    assert out.text == "a" and out.harness_error is None


def test_a_gpu_split_on_linux_with_a_hardware_adapter_and_its_process_listed_is_observed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(envs.StageEnv, "run", _gpu_split(_stderr(_L4), _linux(smi=_smi())))
    backend = _litertlm(tmp_path, backend_flag="gpu")
    (out,) = backend.generate(["one"])

    assert out.text == "a" and out.harness_error is None
    described = backend.describe()
    assert described[BACKEND_OBSERVED] is True
    assert described["adapter_selected"] is False and described["adapter_selected_note"] is None
    assert "NVIDIA L4" in described["gpu_evidence"]
    assert described["adapters"] == [
        {
            "name": "NVIDIA L4",
            "arch": "lovelace",
            "vendor": "nvidia",
            "backend": "Vulkan",
            "adapter_type": "Discrete GPU",
        }
    ]


def test_a_gpu_split_on_linux_whose_process_nvidia_smi_did_not_list_is_selected(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        envs.StageEnv, "run", _gpu_split(_stderr(_L4), _linux(smi=_smi(listed=False)))
    )
    backend = _litertlm(tmp_path, backend_flag="gpu")
    (out,) = backend.generate(["one"])
    assert out.harness_error is None
    described = backend.describe()
    assert described[BACKEND_OBSERVED] is False and described["adapter_selected"] is True
    assert "nvidia-smi did not list this process" in described["adapter_selected_note"]


def test_the_preflight_takes_no_device_reading(monkeypatch, tmp_path):
    """It writes no report; on Windows a reading is a PowerShell query in its budget."""
    calls = []
    ns = _litertlm_script()
    monkeypatch.setitem(ns, "device_report", lambda: calls.append(1) or {})
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "model": "m.litertlm",
                "prompts": ["one"],
                "runtime_rendered": True,
                "backend": "cpu",
                "activation_data_type": None,
                "cache_dir": "",
                "out": str(tmp_path / "out.jsonl"),
            }
        ),
        encoding="utf-8",
    )

    class Engine:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = types.ModuleType("litert_lm")
    fake.Engine = Engine
    interfaces = types.ModuleType("litert_lm.interfaces")
    interfaces.CPU = lambda **k: "cpu"
    interfaces.GPU = lambda: "gpu"
    monkeypatch.setitem(sys.modules, "litert_lm", fake)
    monkeypatch.setitem(sys.modules, "litert_lm.interfaces", interfaces)
    monkeypatch.setitem(ns, "bundle_sampler", lambda path: None)
    monkeypatch.setitem(ns, "score_one", lambda *a, **k: None)
    ns["main"](str(spec_path))
    assert calls == []


def test_on_a_two_gpu_host_nvidia_smis_listing_is_not_tied_to_the_selected_adapter(monkeypatch):
    """The process is listed on the other card; the Vulkan adapter cannot be bound to a row."""
    from litetune.evaluate import adapter_selected, adapter_selected_note, adapters_of, gpu_unbound

    pid = os.getpid()
    listing = f"{pid}, 876, GPU-2222\n".encode()
    two = subprocess.CompletedProcess(["smi"], 0, b"GPU-1111\nGPU-2222\n", b"")
    _linux_smi(monkeypatch, subprocess.CompletedProcess(["smi"], 0, listing, b""), gpus=two)
    reading = _litertlm_script()["device_report"]()
    assert reading["nvidia_smi"]["gpus"] == 2 and reading["nvidia_smi"]["gpu_uuid"] == "GPU-2222"
    report = {**_run_report(reading, reading), "adapters": adapters_of(_L4)}
    assert gpu_observed("gpu", report) is False and adapter_selected("gpu", report) is True
    assert "nvidia-smi lists 2 GPUs" in gpu_unbound("gpu", report)
    assert "nvidia-smi lists 2 GPUs" in adapter_selected_note("gpu", report)
    # One GPU: the listing is tied, and nothing is unbound.
    one = _linux(adapters_of(_L4), smi=_smi())
    assert gpu_observed("gpu", one) is True and gpu_unbound("gpu", one) is None
    # A listing without the count (an older record) is not tied either, and says why.
    legacy = _linux(adapters_of(_L4), smi=_smi(gpus=None))
    assert gpu_observed("gpu", legacy) is False and adapter_selected("gpu", legacy) is True
    note = adapter_selected_note("gpu", legacy)
    assert "did not record how many GPUs the host has or which one" in note
    assert "did not list this process" not in note


def test_on_windows_two_hardware_adapters_of_one_name_are_ambiguous(monkeypatch):
    """Growth only on the second card of the name: neither is observed."""
    from litetune.evaluate import gpu_unbound, windows_selected_luids

    pid = os.getpid()
    dxgi = [
        {"description": "Intel(R) Arc(TM) B390 GPU", "luid": _ARC_LUID, "software": False},
        {"description": "Intel(R) Arc(TM) B390 GPU", "luid": _OTHER_LUID, "software": False},
    ]
    _windows(
        monkeypatch,
        _engines(pid, [3], luid=_ARC_LUID) + _engines(pid, [10], others=False, luid=_OTHER_LUID),
        _engines(pid, [3], luid=_ARC_LUID) + _engines(pid, [900], others=False, luid=_OTHER_LUID),
        dxgi=dxgi,
    )
    device = _litertlm_script()["device_report"]
    report = _arc(_run_report(device(), device()))
    assert windows_selected_luids(report) == []
    assert gpu_observed("gpu", report) is False and gpu_unused("gpu", report) is False
    said = gpu_unbound("gpu", report)
    assert "DXGI lists 2 hardware adapters named Intel(R) Arc(TM) B390 GPU" in said
    assert _ARC_LUID.lower() in said and _OTHER_LUID.lower() in said


_INTEL_UHD = (
    "I0000 environment.cc:532] Selected adapter: Intel(R) UHD Graphics 770, arch=gen-12lp, "
    "vendor=intel, backend=Vulkan, adapterType=Integrated GPU"
)


def test_on_a_hybrid_host_an_intel_adapter_is_not_read_through_nvidia_smi():
    """The process is listed on the one NVIDIA card, and the runtime selected the Intel one."""
    from litetune.evaluate import adapter_selected, adapter_selected_note, adapters_of, gpu_unbound

    hybrid = _linux(adapters_of(_INTEL_UHD), smi=_smi())
    assert gpu_observed("gpu", hybrid) is False and adapter_selected("gpu", hybrid) is True
    said = gpu_unbound("gpu", hybrid)
    assert "vendor intel" in said and "sees NVIDIA's GPUs alone" in said
    assert said in adapter_selected_note("gpu", hybrid)
    # The NVIDIA adapter on a one-GPU host is still observed.
    nvidia = _linux(adapters_of(_L4), smi=_smi())
    assert gpu_observed("gpu", nvidia) is True and gpu_unbound("gpu", nvidia) is None
    # The vendor is read without regard to case.
    upper = _linux(adapters_of(_L4.replace("vendor=nvidia", "vendor=NVIDIA")), smi=_smi())
    assert gpu_observed("gpu", upper) is True


def test_the_picture_header_probe_is_not_named_after_a_standard_module(tmp_path):
    # Python puts a script's own directory first on sys.path, so a script named
    # `inspect.py` is what `import inspect` finds. In 0.8.1 the probe was, and
    # every CPU picture verify ended failed_harness on it.
    seen = []

    class ProbeEnv:
        name = "runtime"

        def provision(self, events=None):
            return None

        def run(self, argv, timeout=None):
            seen.append(Path(argv[1]).stem)
            spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            Path(spec["report"]).write_text(json.dumps({"error": None}), encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

    backend = _litertlm(
        tmp_path,
        backend_flag="cpu",
        images=({"path": "a.jpg", "sha256": "0" * 64},),
        env=ProbeEnv(),
    )
    assert backend.picture_backend_problem() is None
    assert seen and seen[0] not in sys.stdlib_module_names


def test_no_script_named_in_the_source_shadows_a_standard_module():
    # A helper script is written into a work directory and run by path, so that
    # directory is first on its sys.path (see the test above). This reads the
    # names spelled out in the source -- `work / "name.py"` and `NAME = "name.py"`;
    # `embedding_verify.run_script`, which builds its path from a name, refuses
    # one itself (tests/test_embedding_verify.py).
    source = Path(litetune.__file__).parent
    names = set()
    for path in source.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r'/ "([A-Za-z_][A-Za-z0-9_]*)\.py"', text))
        names.update(re.findall(r'^[A-Z_]+ = "([A-Za-z_][A-Za-z0-9_]*)\.py"', text, re.M))
    assert names, "no helper script names found; the pattern no longer matches the code"
    assert not names & SHADOWABLE_MODULES, names & SHADOWABLE_MODULES


def test_the_shadowable_list_names_every_package_the_source_imports():
    source = Path(litetune.__file__).parent
    imported = set()
    for path in source.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        imported.update(re.findall(r"^\s*import ([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
        imported.update(re.findall(r"^\s*from ([A-Za-z_][A-Za-z0-9_]*)[\w.]* import ", text, re.M))
    # The two helpers a script imports from its own directory on purpose.
    siblings = {"repack", "towers_script"}
    missing = imported - SHADOWABLE_MODULES - siblings - {"__future__"}
    assert not missing, f"add these to embedding_verify.SHADOWABLE_MODULES: {sorted(missing)}"
