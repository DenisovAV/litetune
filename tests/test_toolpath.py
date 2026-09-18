"""The runtime-side tool-path script, against a fake `litert_lm`.

The script is a string, so it is `exec`'d and its `main` run against a fake
module -- the same technique `test_rendering.py` uses, and for the same reason:
the contract that matters is what it asks the runtime for and what it writes,
and neither needs a bundle to pin.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from litetune.metrics import ToolCall
from litetune.toolpath import _TOOL_PATH_SCRIPT, ToolPathError, ToolPathProbe, ToolPathRow

TOOLS = [
    {
        "type": "function",
        "function": {"name": "set_colour", "description": "d"},
    }
]


def _exec(source: str, name: str) -> dict:
    namespace: dict[str, Any] = {"__name__": name}
    exec(compile(source, name, "exec"), namespace)  # noqa: S102
    return namespace


@dataclass
class FakeRuntime:
    """What the script asked for, and what it was told."""

    replies: list[Any] = field(default_factory=list)
    conversations: list[dict] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    create_raises: BaseException | None = None
    # What the native runtime writes to file descriptor 2 before a reply fails.
    logs: dict[int, str] = field(default_factory=dict)
    # Execute the declared tools, as the binding does with automatic calling on.
    execute_tools: bool = False


def _fake_litert_lm(runtime: FakeRuntime) -> Any:
    module: Any = types.ModuleType("litert_lm")

    class Conversation:
        def __enter__(self) -> Conversation:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def __init__(self, **kwargs: Any):
            runtime.conversations.append(kwargs)
            self.tools = kwargs.get("tools") or []

        def send_message(self, prompt: str) -> Any:
            runtime.sent.append(prompt)
            index = len(runtime.sent) - 1
            if index in runtime.logs:
                os.write(2, runtime.logs[index].encode("utf-8"))
            if runtime.execute_tools:
                for tool in self.tools:
                    tool.execute({})
            reply = runtime.replies[index]
            if isinstance(reply, BaseException):
                raise reply
            return reply

    class Engine:
        def __init__(self, model: str, backend: object):
            module.opened = model

        def __enter__(self) -> Engine:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def create_conversation(self, **kwargs: Any) -> Conversation:
            if runtime.create_raises is not None:
                raise runtime.create_raises
            return Conversation(**kwargs)

    class Tool:
        def get_tool_description(self) -> dict:
            raise NotImplementedError

        def execute(self, param: Any) -> Any:
            raise NotImplementedError

    module.Engine = Engine
    module.Tool = Tool
    module.Backend = types.SimpleNamespace(CPU=lambda: "cpu")
    module.SamplerConfig = lambda **kwargs: kwargs
    module.ConstrainedDecodingConfig = lambda **kwargs: kwargs
    module.LogSeverity = types.SimpleNamespace(ERROR="error")
    module.set_min_log_severity = lambda level: None
    return module


def _run(runtime: FakeRuntime, tmp_path: Path, monkeypatch, **spec_extra) -> tuple[int, Path]:
    monkeypatch.setitem(sys.modules, "litert_lm", _fake_litert_lm(runtime))
    out = tmp_path / "rows.json"
    spec_path = tmp_path / "spec.json"
    spec: dict[str, Any] = {
        "model": "bundle.litertlm",
        "prompts": ["make it red"],
        "tools": TOOLS,
        "max_tokens": 64,
        "constrained": True,
        "out": str(out),
        "log": str(tmp_path / "runtime.log"),
    }
    spec.update(spec_extra)
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["toolpath.py", str(spec_path)])
    code = _exec(_TOOL_PATH_SCRIPT, "toolpath_script_under_test")["main"]()
    return code, out


def _reply(name: str, arguments: dict) -> dict:
    return {
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
        "content": [],
    }


# -- what it asks the runtime for -------------------------------------------


def test_it_asks_the_runtime_the_question_an_application_asks(tmp_path, monkeypatch):
    """Declarations passed as `Tool` instances, automatic calling off.

    Automatic calling left on would have the runtime execute the declared tools
    and loop until prose -- running somebody's functions in order to measure a
    model, and measuring the loop rather than the call.
    """
    runtime = FakeRuntime(replies=[_reply("set_colour", {"colour": "red"})])

    code, _ = _run(runtime, tmp_path, monkeypatch)

    assert code == 0
    (args,) = runtime.conversations
    assert args["automatic_tool_calling"] is False
    assert args["max_output_tokens"] == 64
    assert [tool.get_tool_description() for tool in args["tools"]] == TOOLS


def test_a_declared_tool_asked_to_run_stops_the_run(tmp_path, monkeypatch):
    """With automatic calling off the binding returns before it can run a tool,
    so `execute` is unreachable. If it is ever reached anyway, the run stops
    and says so: the binding catches what `execute` raises and would hand the
    text back to the model as a tool response, changing the reply."""
    runtime = FakeRuntime(replies=[_reply("set_colour", {})], execute_tools=True)

    code, out = _run(runtime, tmp_path, monkeypatch)

    assert code == 3
    assert not out.exists()


@pytest.mark.parametrize("constrained", [True, False])
def test_the_decoding_mode_reaches_the_runtime(tmp_path, monkeypatch, constrained):
    """Both modes are the point: with the grammar on, a model that writes the
    wrong shape returns clean calls anyway, so a single number cannot say
    whether the model learned the format."""
    runtime = FakeRuntime(replies=[_reply("set_colour", {})])

    _run(runtime, tmp_path, monkeypatch, constrained=constrained)

    (args,) = runtime.conversations
    assert args["constrained_decoding_config"] == {"enable": constrained}


# -- what it writes ----------------------------------------------------------


def _rows_written(out: Path) -> list[dict]:
    written = json.loads(out.read_text(encoding="utf-8"))
    return written["rows"]


def test_a_structured_call_comes_back_with_its_argument_types(tmp_path, monkeypatch):
    """The runtime types the arguments, and that survives to the row.

    Flattening them here would throw away the thing the tool path has that the
    text path does not.
    """
    # A number comes back as a double: `fc_parser.rs` reads every NUMBER as f64.
    runtime = FakeRuntime(replies=[_reply("set_alarm", {"hour": 7.0, "loud": True})])

    _, out = _run(runtime, tmp_path, monkeypatch)

    (row,) = _rows_written(out)
    assert row == {
        "index": 0,
        "calls": [{"name": "set_alarm", "arguments": {"hour": 7.0, "loud": True}}],
        "text": "",
        "error": None,
        "kind": None,
    }


def test_every_call_in_a_reply_is_kept(tmp_path, monkeypatch):
    """An application acts on each call it is handed, so a second one is part of
    the answer. Keeping only the first scored a reply with an extra call as a
    clean match."""
    reply = _reply("set_colour", {"colour": "red"})
    reply["tool_calls"].append({"type": "function", "function": {"name": "open_app"}})
    runtime = FakeRuntime(replies=[reply])

    _, out = _run(runtime, tmp_path, monkeypatch)

    (row,) = _rows_written(out)
    assert [call["name"] for call in row["calls"]] == ["set_colour", "open_app"]


def test_the_runtimes_version_is_written_beside_the_rows(tmp_path, monkeypatch):
    """A manifest that cannot say which runtime produced its rows cannot be
    compared with a run on another one."""
    import importlib.metadata

    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.16.1" if name == "litert-lm" else real(name),
    )
    runtime = FakeRuntime(replies=[_reply("set_colour", {})])

    _, out = _run(runtime, tmp_path, monkeypatch)

    assert json.loads(out.read_text(encoding="utf-8"))["runtime_version"] == "0.16.1"


def test_a_reply_with_no_call_is_a_result_not_a_failure(tmp_path, monkeypatch):
    """A model that answered in prose instead of calling anything is something
    to see, and a row with neither a call nor text could not be told from one
    that failed."""
    runtime = FakeRuntime(replies=[{"content": [{"text": "I cannot do that."}]}])

    _, out = _run(runtime, tmp_path, monkeypatch)

    (row,) = _rows_written(out)
    assert row["calls"] == []
    assert row["error"] is None
    assert row["text"] == "I cannot do that."


def test_every_prompt_gets_a_row_in_order(tmp_path, monkeypatch):
    prompts = ["a", "b", "c"]
    runtime = FakeRuntime(replies=[_reply("t", {"i": i}) for i in range(3)])

    _, out = _run(runtime, tmp_path, monkeypatch, prompts=prompts)

    rows = _rows_written(out)
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert runtime.sent == prompts
    # One conversation per prompt: every row is a first turn, as every other
    # measurement in this project asks it.
    assert len(runtime.conversations) == 3


# -- how it fails ------------------------------------------------------------

# What the binding raises for every failed reply, in LiteRT-LM v0.16.1.
SEND_FAILED = RuntimeError("litert_lm_conversation_send_message failed")
# What the runtime logs when its parser refuses a reply (`parser_utils.cc`,
# `c/conversation.cc`): the model's code block and full response in it.
PARSE_FAILURE_LOG = (
    "E0918 12:00:00.000000 4242 conversation.cc:552] Failed to send message: "
    "INVALID_ARGUMENT: Failed to parse tool calls from code block: call:secret_tool{}\n"
    "full response: SECRET card 4111\nerror: Failed to parse FC tool calls\n"
)
ROW_MARK = _exec(_TOOL_PATH_SCRIPT, "toolpath_script_under_test")["ROW_MARK"]


def test_a_reply_the_runtime_refused_carries_the_kind_of_reason_from_its_log(tmp_path, monkeypatch):
    """The binding says only that the reply failed; the runtime says why, on its
    log. A parse failure's message carries the model's code block and response
    (`parser_utils.cc`), which must not travel into a manifest a bundle ships,
    so it is kept as its kind. The log lines are absl's shape."""
    runtime = FakeRuntime(
        replies=[_reply("set_colour", {"colour": "red"}), SEND_FAILED, SEND_FAILED, SEND_FAILED],
        logs={
            1: "E0918 12:00:00.000000 4242 conversation.cc:552] Failed to send message: "
            "INVALID_ARGUMENT: Failed to parse tool calls from code block: call:secret_tool{}\n"
            "full response: the model's private text\nerror: Failed to parse FC tool calls\n",
            2: "E0918 12:00:00.000000 4242 conversation.cc:552] Failed to send message: "
            "Input token ids are too long. Exceeding the maximum number of tokens allowed: "
            "4101 >= 4096\n",
            3: "E0918 12:00:00.000000 4242 conversation.cc:552] Failed to send message: "
            "INTERNAL: something else\n",
        },
    )

    code, out = _run(runtime, tmp_path, monkeypatch, prompts=["a", "b", "c", "d"])

    assert code == 0
    rows = _rows_written(out)
    assert rows[0]["calls"] and rows[0]["error"] is None and rows[0]["kind"] is None
    assert [row["kind"] for row in rows[1:]] == ["parse", "too_long", "other"]
    assert "secret_tool" not in json.dumps(rows) and "private text" not in json.dumps(rows)
    # Anything else in the runtime's own words, without absl's prefix.
    assert rows[3]["error"] == "Failed to send message: INTERNAL: something else"
    assert all(row["calls"] == [] for row in rows[1:])


@pytest.mark.parametrize(
    "raised",
    [
        TypeError("send_message() got an unexpected keyword"),
        # The binding's own, beside the no-reply one (`conversation.py`).
        RuntimeError("Conversation is closed."),
    ],
)
def test_an_exception_that_is_not_a_missing_reply_ends_the_run(tmp_path, monkeypatch, raised):
    """Found in review: any exception from `send_message` was scored as the
    model's wrong answer -- an API change, memory, a closed conversation. Only
    the binding's own no-reply error is a result about a row."""
    runtime = FakeRuntime(replies=[raised])

    with pytest.raises(type(raised), match=re.escape(str(raised))):
        _run(runtime, tmp_path, monkeypatch)


@pytest.mark.parametrize(
    "log, reason",
    [
        ("", "nothing in its log"),
        # Found in review: a line with neither word was reported as nothing.
        (
            "E0918 12:00:00.000000 4242 conversation.cc:552] RESOURCE_EXHAUSTED: arena\n",
            "RESOURCE_EXHAUSTED: arena",
        ),
        (
            "Failed to open x\nFailed to send message: INTERNAL: y\n",
            "Failed to send message: INTERNAL: y",
        ),
    ],
)
def test_a_reason_litetune_does_not_read_is_the_runtimes_last_word(log, reason):
    reason_of = _exec(_TOOL_PATH_SCRIPT, "toolpath_script_under_test")["reason_of"]

    assert reason_of(log) == ("other", reason)


def test_a_conversation_that_cannot_be_created_ends_the_run(tmp_path, monkeypatch):
    """Not a row failure. If the runtime will not take these declarations at
    all, nothing was measured, and reporting every prompt as a wrong answer
    would be a statement about the model that no measurement supports."""
    runtime = FakeRuntime(
        replies=[], create_raises=RuntimeError("tools are not supported by this bundle")
    )

    code, out = _run(runtime, tmp_path, monkeypatch, prompts=["a", "b"])

    assert code == 3
    assert not out.exists()


# -- the probe around it -----------------------------------------------------


class _WritingEnv:
    """A runtime environment whose script writes `written` as its output."""

    name = "runtime"

    def __init__(self, written: Any):
        self.written = written

    def provision(self, events=None):
        return None

    def run(self, argv, timeout=None):
        spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(json.dumps(self.written), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")


def _row(index: int, **fields: Any) -> dict:
    return {"index": index, "calls": [], "text": "", "error": None, "kind": None} | fields


def test_a_script_that_writes_the_wrong_number_of_rows_is_a_harness_failure(tmp_path):
    """Silent truncation would score the missing prompts as unanswered."""
    probe = ToolPathProbe(
        model=tmp_path / "m.litertlm", declarations=TOOLS, env=_WritingEnv({"rows": [_row(0)]})
    )

    with pytest.raises(ToolPathError, match="wrote 1 rows for 2 prompts"):
        probe.observe(["a", "b"], constrained=True)


@pytest.mark.parametrize(
    "rows",
    [
        [_row(1), _row(0)],  # swapped: scoring pairs by position
        [_row(0, calls={"name": "t"}), _row(1)],
        [_row(0), "not a row"],
        [_row(0, calls=[{"name": "t", "arguments": "oops"}]), _row(1)],
        [_row(0, error="boom"), _row(1)],  # an error needs its kind
        [_row(0, kind="parse"), _row(1)],  # and a kind its error
        [_row(0, error="x", kind="parse", calls=[{"name": "t", "arguments": {}}]), _row(1)],
    ],
)
def test_a_row_that_is_not_the_row_at_its_position_is_a_harness_failure(tmp_path, rows):
    """Found in review: a row was built with `ToolPathRow(**row)` and its index
    never checked, so two swapped rows were scored against each other's
    targets, and an unexpected field escaped as a `TypeError` from a backend
    that promises never to raise."""
    probe = ToolPathProbe(
        model=tmp_path / "m.litertlm", declarations=TOOLS, env=_WritingEnv({"rows": rows})
    )

    with pytest.raises(ToolPathError):
        probe.observe(["a", "b"], constrained=True)


def test_a_native_crash_leaves_its_reason_in_the_error(tmp_path):
    """Found in review: fd 2 pointed at an unlinked temporary file during a
    reply, so a native abort's reason vanished with the process and the run
    said "no stderr". The log is a file the parent reads."""

    class Crashing(_WritingEnv):
        def run(self, argv, timeout=None):
            spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            Path(spec["log"]).write_text(
                f"\n{ROW_MARK}0\nF0918 kv_cache.cc:12] Check failed: kv_cache != nullptr\n"
            )
            return types.SimpleNamespace(returncode=-6, stderr="")

    probe = ToolPathProbe(model=tmp_path / "m.litertlm", declarations=TOOLS, env=Crashing(None))

    with pytest.raises(ToolPathError, match="on prompt 0: .*kv_cache != nullptr"):
        probe.observe(["a"], constrained=False)


@pytest.mark.parametrize(
    "last, reason",
    [
        ("F0918 kv_cache.cc:12] Check failed: kv_cache != nullptr\n", "kv_cache != nullptr"),
        # Died after the parser refused this one: its kind, never its text.
        (PARSE_FAILURE_LOG, "its call parser rejected the generation"),
    ],
)
def test_a_crash_quotes_only_the_prompt_it_died_on_and_never_the_models_text(
    tmp_path, last, reason
):
    """Found in review: the whole log went into the error, and the log holds
    every earlier reply's parse failure, each with the model's code block and
    full response -- into the manifest a bundle ships, and named as the reason
    for a crash on another prompt."""

    class Crashing(_WritingEnv):
        def run(self, argv, timeout=None):
            spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            Path(spec["log"]).write_text(
                f"\n{ROW_MARK}0\n{PARSE_FAILURE_LOG}\n{ROW_MARK}1\n{last}", encoding="utf-8"
            )
            return types.SimpleNamespace(returncode=-6, stderr="")

    probe = ToolPathProbe(model=tmp_path / "m.litertlm", declarations=TOOLS, env=Crashing(None))

    with pytest.raises(ToolPathError) as caught:
        probe.observe(["a", "b"], constrained=True)

    said = str(caught.value)
    assert reason in said.split("on prompt 1: ", 1)[1]
    assert "SECRET" not in said and "secret_tool" not in said


def test_the_script_marks_each_reply_in_the_log_with_the_mark_the_parent_reads(
    tmp_path, monkeypatch
):
    runtime = FakeRuntime(replies=[SEND_FAILED, SEND_FAILED], logs={0: "first\n", 1: "second\n"})

    _run(runtime, tmp_path, monkeypatch, prompts=["a", "b"])

    log = (tmp_path / "runtime.log").read_text(encoding="utf-8")
    assert log.index(f"{ROW_MARK}0") < log.index("first") < log.index(f"{ROW_MARK}1")
    assert log.index(f"{ROW_MARK}1") < log.index("second")


def test_output_that_is_not_json_is_a_harness_failure(tmp_path):
    class Garbled(_WritingEnv):
        def run(self, argv, timeout=None):
            spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            Path(spec["out"]).write_text("{not json", encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stderr="")

    probe = ToolPathProbe(model=tmp_path / "m.litertlm", declarations=TOOLS, env=Garbled(None))

    with pytest.raises(ToolPathError, match="unreadable"):
        probe.observe(["a"], constrained=True)


def test_a_row_hands_an_application_no_reply_or_its_calls():
    one = {"name": "t", "arguments": {"hour": 7.0}}
    assert ToolPathRow(0, error="boom", kind="other").refused is True
    assert ToolPathRow(0, error="x", kind="parse").handed is None
    assert ToolPathRow(0, calls=(one,)).refused is False
    assert ToolPathRow(0, calls=(one,)).handed == [ToolCall("t", {"hour": 7})]
    assert ToolPathRow(0, calls=(one, one)).handed == [ToolCall("t", {"hour": 7})] * 2
    # Prose is a reply without a call, which is not no reply.
    assert ToolPathRow(0).handed == []


# -- the backend, and how `verify` picks it ----------------------------------


from conftest import FakeBackend, call_text, correct_texts, labelled_rows  # noqa: E402

from litetune.prompt_mode import PromptMode  # noqa: E402
from litetune.toolpath import ToolPathBackend  # noqa: E402
from litetune.verify import (  # noqa: E402
    BackendPair,
    ReferenceRole,
    Status,
    VerifyRequest,
    build_backends,
    run_verify,
)

FUNCTIONGEMMA = "google/functiongemma-270m-it"
DECLS = [{"type": "function", "function": {"name": "change_background_color", "description": "d"}}]


@dataclass
class CannedEnv:
    """A runtime environment that answers the script with canned rows per mode."""

    by_mode: dict[str, list[dict]]
    name: str = "runtime"
    fail: str | None = None
    # Fail only these modes, when `fail` is set; every mode otherwise.
    fail_modes: tuple[str, ...] = ("constrained", "unconstrained")
    runtime_version: str = "0.16.1"
    # Every spec the script was handed, so a test can see what reached it.
    specs: list[dict] = field(default_factory=list)

    def provision(self, events=None):
        return None

    def run(self, argv, timeout=None):
        spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        mode = "constrained" if spec["constrained"] else "unconstrained"
        if self.fail is not None and mode in self.fail_modes:
            return types.SimpleNamespace(returncode=3, stderr=self.fail)
        written = {"runtime_version": self.runtime_version, "rows": self.by_mode[mode]}
        Path(spec["out"]).write_text(json.dumps(written), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")


PARSE_REFUSED = "its call parser rejected the generation"


def _rows(rows_, hits: int, refusals: int = 0) -> list[dict]:
    """A row per example: `hits` correct calls, then `refusals` refusals, then misses."""
    out = []
    for i, row in enumerate(rows_):
        target = row["target"]
        if i < hits:
            calls = [{"name": target["name"], "arguments": target["args"]}]
            out.append(_row(i, calls=calls))
        elif i < hits + refusals:
            out.append(_row(i, error=PARSE_REFUSED, kind="parse"))
        else:
            out.append(_row(i, calls=[{"name": target["name"], "arguments": {"color": "no"}}]))
    return out


def marked(texts: list[str]) -> list[str]:
    """Calls in their markers, which is how the reference writes them and the
    only place the runtime reads a call."""
    return [f"<start_function_call>{t}<end_function_call>" for t in texts]


def _verify(tmp_path, rows_, by_mode, reference_texts=None, request_extra=None, **kwargs):
    split = tmp_path / "heldout.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    env = CannedEnv(by_mode=by_mode, **kwargs)
    candidate = ToolPathBackend(model=tmp_path / "m.litertlm", declarations=DECLS, env=env)
    request = VerifyRequest(
        model=tmp_path / "m.litertlm",
        reference="org/reference",
        data=split,
        prompt_mode=PromptMode.RUNTIME_RENDERED,
        **(request_extra or {}),
    )
    reference = FakeBackend(
        model="org/reference",
        texts=reference_texts if reference_texts is not None else marked(correct_texts(rows_)),
        prompt_mode=PromptMode.RUNTIME_RENDERED,
    )
    return run_verify(request, backends=BackendPair(candidate=candidate, reference=reference))


def test_the_score_is_the_call_the_runtime_returned(tmp_path):
    """Not text this re-parsed. The runtime's parser is the one being measured,
    because it is the one an application depends on."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 6), "unconstrained": _rows(rows_, 6)}
    )

    assert result.status in (Status.PASSED, Status.FAILED_GATE, Status.INCONCLUSIVE)
    assert result.manifest["quality"]["candidate"]["exact_match"]["value"] == pytest.approx(0.75)
    assert result.manifest["tool_path"]["modes"]["constrained"]["score"]["n"] == 8


def test_an_integer_the_runtime_returns_as_a_double_scores_as_that_integer(tmp_path):
    """LiteRT-LM v0.16.1 hands `hour:7` back as `7.0`. Scored as the string
    `"7.0"`, every correct integer argument was wrong on the tool path while the
    reference, parsing `hour:7` itself, got it right -- a conversion cost made
    of nothing but a type. mobile-actions has string arguments only, so the
    measured run could not see it."""
    rows_ = [
        {"prompt": f"wake me at {h}", "target": {"name": "set_alarm", "args": {"hour": h}}}
        for h in range(8)
    ]
    returned = [
        _row(h, calls=[{"name": "set_alarm", "arguments": {"hour": float(h)}}]) for h in range(8)
    ]

    result = _verify(tmp_path, rows_, {"constrained": returned, "unconstrained": returned})

    modes = result.manifest["tool_path"]["modes"]
    assert modes["constrained"]["score"]["exact_match"]["value"] == pytest.approx(1.0)
    assert modes["unconstrained"]["score"]["exact_match"]["value"] == pytest.approx(1.0)
    assert result.manifest["attribution"]["conversion_cost"]["value"] == pytest.approx(0.0)


def test_both_decoding_modes_are_reported(tmp_path):
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 4)}
    )

    path = result.manifest["tool_path"]["modes"]
    assert path["constrained"]["score"]["exact_match"]["value"] == pytest.approx(1.0)
    assert path["unconstrained"]["score"]["exact_match"]["value"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "on, off, direction",
    [(8, 4, "The grammar raised the score"), (4, 8, "The grammar lowered the score")],
)
def test_modes_that_disagree_say_which_way(tmp_path, on, off, direction):
    """The first wording said a gap was "the model leaning on the grammar", and
    the one gap ever measured went the other way: the grammar removed arguments."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, on), "unconstrained": _rows(rows_, off)}
    )

    (text,) = [x for x in result.manifest["limitations"] if "did not agree" in x]
    assert direction in text
    assert "grammar-off number is what an application gets by default" in text


def test_modes_that_agree_raise_none(tmp_path):
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 6), "unconstrained": _rows(rows_, 6)}
    )

    assert not any("did not agree" in limitation for limitation in result.manifest["limitations"])


def test_a_prompt_the_runtime_gave_no_reply_to_is_a_wrong_answer(tmp_path):
    """Found in review: such rows were kept out of every score, so the worse the
    candidate the easier the comparison -- 195 of 200 refused read as a cost of
    0.0 over five. An application handed no reply has nothing to act on, and
    the reference's unparseable text is already a wrong answer on its side."""
    rows_ = labelled_rows(8)
    both = _rows(rows_, hits=6, refusals=2)
    result = _verify(tmp_path, rows_, {"constrained": both, "unconstrained": both})

    mode = result.manifest["tool_path"]["modes"]["unconstrained"]
    assert mode["score"]["n"] == 8
    assert mode["score"]["exact_match"]["value"] == pytest.approx(0.75)
    assert (mode["no_reply"], mode["parse_refusals"], mode["too_long"]) == (2, 2, 0)
    assert result.manifest["quality"]["reference"]["n"] == 8
    assert any(
        "gave no reply to 2 prompts: 2 because its call parser rejected the generation" in text
        and "grammar off" in text
        for text in result.manifest["limitations"]
    )


def test_a_worse_candidate_does_not_pass_more_easily(tmp_path):
    rows_ = labelled_rows(40)
    result = _verify(
        tmp_path,
        rows_,
        {
            "constrained": _rows(rows_, hits=3, refusals=37),
            "unconstrained": _rows(rows_, hits=3, refusals=37),
        },
        request_extra={"max_conversion_cost": 0.05},
    )

    cost = result.manifest["attribution"]["conversion_cost"]
    assert cost["value"] == pytest.approx(37 / 40)
    assert result.status is Status.FAILED_GATE


def test_a_model_whose_own_output_never_parses_is_a_verdict_about_the_model(tmp_path):
    """The case this change started from: clean calls with the grammar on,
    refused with it off. It ended as a harness failure -- "says nothing about
    the model" -- which is the one thing it does say something about."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {"constrained": _rows(rows_, hits=8), "unconstrained": _rows(rows_, 0, refusals=8)},
    )

    assert result.status is Status.FAILED_SMOKE
    (check,) = result.manifest["liveness"]["candidate"]["checks"]
    assert "with the grammar off, 0 of 8 prompts returned a call" in check["detail"]
    # Found in review: the run stopped here with the runtime's reasons lost.
    assert "8 because its call parser rejected the generation" in check["detail"]
    modes = result.manifest["tool_path"]["modes"]
    assert modes["unconstrained"]["parse_refusals"] == 8
    assert modes["constrained"]["no_reply"] == 0


def test_a_model_that_answered_in_prose_is_said_to_have(tmp_path):
    """Found in review: only a row with no reply was given a reason, so a
    liveness failure on prose read as if nothing had come back."""
    rows_ = labelled_rows(8)
    prose = [_row(i, text="I cannot do that.") for i in range(8)]

    result = _verify(tmp_path, rows_, {"constrained": prose, "unconstrained": prose})

    assert result.status is Status.FAILED_SMOKE
    (check,) = result.manifest["liveness"]["candidate"]["checks"]
    assert "8 answered without calling anything" in check["detail"]
    assert check["observed"]["without_a_call"] == 8


def test_the_grammar_refusing_every_row_is_measured_not_hidden(tmp_path):
    """The mirror case: the model's own output parses and the grammar breaks it.
    What an application gets by default is compared with the reference, and
    what the grammar does is reported beside it."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {"constrained": _rows(rows_, 0, refusals=8), "unconstrained": _rows(rows_, hits=8)},
    )

    assert result.status is Status.PASSED
    assert result.manifest["tool_path"]["grammar_effect"]["value"] == pytest.approx(1.0)


def test_a_reply_with_a_call_the_target_did_not_ask_for_is_a_wrong_answer(tmp_path):
    rows_ = labelled_rows(8)
    doubled = _rows(rows_, hits=8)
    doubled[0]["calls"].append({"name": "open_app", "arguments": {}})
    result = _verify(tmp_path, rows_, {"constrained": doubled, "unconstrained": doubled})

    mode = result.manifest["tool_path"]["modes"]["unconstrained"]
    assert mode["score"]["exact_match"]["value"] == pytest.approx(7 / 8)
    assert mode["several_calls"] == 1
    assert any("more than one call on 1 of 8" in x for x in result.manifest["limitations"])


def test_a_tool_path_that_cannot_run_is_a_harness_failure_naming_the_runtime(tmp_path):
    """Not a verdict about the model. Nothing was measured, and reporting every
    prompt as a wrong answer would be a claim no measurement supports."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {},
        fail="could not create a conversation with declarations: RuntimeError: no tool support",
    )

    assert result.status is Status.FAILED_HARNESS
    assert "no tool support" in json.dumps(result.manifest)


def test_a_scorer_that_reads_text_is_refused_on_the_tool_path(tmp_path):
    """Found in review: `--scorer exact-text` scored the candidate's calls by the
    tool-call rule and the reference's text by exact-text, two different rules
    under one conversion cost, and crashed outright on a string target."""
    rows_ = labelled_rows(8)
    split = tmp_path / "heldout.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    both = _rows(rows_, 8)
    candidate = ToolPathBackend(
        model=tmp_path / "m.litertlm",
        declarations=DECLS,
        env=CannedEnv(by_mode={"constrained": both, "unconstrained": both}),
    )
    reference = FakeBackend(
        model="org/reference", texts=correct_texts(rows_), prompt_mode=PromptMode.RUNTIME_RENDERED
    )
    request = VerifyRequest(
        model=tmp_path / "m.litertlm",
        reference="org/reference",
        data=split,
        prompt_mode=PromptMode.RUNTIME_RENDERED,
        scorer="exact-text",
    )

    result = run_verify(request, backends=BackendPair(candidate=candidate, reference=reference))

    assert result.status is Status.FAILED_HARNESS
    assert "--scorer tool-call" in json.dumps(result.manifest["checks"])


def test_the_runtime_version_is_in_the_manifest(tmp_path):
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 8)}
    )

    assert result.manifest["measurements"]["candidate"]["engine"]["runtime_version"] == "0.16.1"


def test_the_declarations_reach_the_runtime_in_both_modes(tmp_path):
    """Found in review: every place the declarations are handed on could be
    emptied and no test failed, because the fake environment ignored them."""
    rows_ = labelled_rows(8)
    env = CannedEnv(by_mode={"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 8)})
    backend = ToolPathBackend(model=tmp_path / "m.litertlm", declarations=DECLS, env=env)

    backend.generate([r["prompt"] for r in rows_])

    # Grammar off first: it is the run the reference is compared with.
    assert [(spec["constrained"], spec["tools"]) for spec in env.specs] == [
        (False, DECLS),
        (True, DECLS),
    ]


def test_the_path_is_chosen_from_the_model_and_the_declarations(tmp_path):
    """Never from a flag: a flag can disagree with the model, and the run would
    then measure a path the artifact does not serve."""
    request = VerifyRequest(
        model=tmp_path / "m.litertlm",
        reference=FUNCTIONGEMMA,
        data=tmp_path / "heldout.jsonl",
        prompt_mode=PromptMode.RUNTIME_RENDERED,
    )

    chosen = build_backends(request, declarations=DECLS)
    assert isinstance(chosen.candidate, ToolPathBackend)

    # Same request, no declarations: there is nothing to declare.
    assert not isinstance(build_backends(request).candidate, ToolPathBackend)


@pytest.mark.parametrize(
    "reference, mode",
    [
        ("Qwen/Qwen3-0.6B", PromptMode.RUNTIME_RENDERED),
        (FUNCTIONGEMMA, PromptMode.PRERENDERED),
    ],
)
def test_a_family_or_a_mode_with_no_tool_path_stays_on_the_text_path(tmp_path, reference, mode):
    """Two different reasons, and neither is a preference: litetune records no
    tool channel for Qwen-3, and a pre-rendered prompt routes the runtime past
    the conversation the tool path needs."""
    request = VerifyRequest(
        model=tmp_path / "m.litertlm",
        reference=reference,
        data=tmp_path / "heldout.jsonl",
        prompt_mode=mode,
    )

    assert not isinstance(build_backends(request, declarations=DECLS).candidate, ToolPathBackend)


def test_the_manifest_says_which_path_ran_and_why(tmp_path):
    """The spec's own words: `verify` records that it measured through the tool
    path and why. An earlier version computed the reason and discarded it, while
    the task record claimed the manifest carried it."""
    rows_ = labelled_rows(8)
    split = tmp_path / "heldout.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    decls = tmp_path / "tools.json"
    decls.write_text(json.dumps(DECLS), encoding="utf-8")
    candidate = ToolPathBackend(
        model=tmp_path / "m.litertlm",
        declarations=DECLS,
        env=CannedEnv(by_mode={"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 8)}),
    )
    request = VerifyRequest(
        model=tmp_path / "m.litertlm",
        reference=FUNCTIONGEMMA,
        data=split,
        prompt_mode=PromptMode.RUNTIME_RENDERED,
        declarations=decls,
    )
    reference = FakeBackend(
        model=FUNCTIONGEMMA, texts=correct_texts(rows_), prompt_mode=PromptMode.RUNTIME_RENDERED
    )

    result = run_verify(request, backends=BackendPair(candidate=candidate, reference=reference))

    selection = result.manifest["harness"]["tool_path_selection"]
    assert selection["tool_path"] is True
    assert "functiongemma" in selection["why"] and "Nothing asked for it" in selection["why"]


def test_supplied_backends_are_not_described_as_selected(tmp_path):
    """A test or a caller that hands `run_verify` its own candidate chose the
    path. Naming the rule's answer instead would describe a run that did not
    happen: here the rule says text path, and a tool path ran."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 8)}
    )

    selection = result.manifest["harness"]["tool_path_selection"]
    assert selection["tool_path"] is True
    assert "supplied its own backends" in selection["why"]


def test_the_device_is_named_in_the_vocabulary_verify_reads(tmp_path):
    """`verify` reads the candidate's device from `engine` and `backend`. The
    script opens the engine on `Backend.CPU()`, and a real run's manifest has to
    say so rather than naming the path where the device belongs."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 8)}
    )

    engine = result.manifest["measurements"]["candidate"]["engine"]
    assert (engine["engine"], engine["backend"]) == ("litert-lm", "cpu")
    assert any(
        "measured on the cpu backend of litert-lm" in limitation
        for limitation in result.manifest["limitations"]
    )


def test_the_conversion_cost_is_measured_with_the_grammar_off_like_the_reference(tmp_path):
    """The attribution a real run got wrong.

    FunctionGemma x mobile-actions at n=640: reference 0.9234, tool path with
    the grammar off 0.9172, with it on 0.7422. Compared with the grammar-on
    number, the manifest reported a resolved conversion cost of 0.1812 that was
    almost entirely the grammar. The reference generates with no grammar, so
    the conversion cost is measured against the grammar-off run, and what the
    grammar does is its own paired difference.
    """
    rows_ = labelled_rows(40)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 20), "unconstrained": _rows(rows_, 40)}
    )

    cost = result.manifest["attribution"]["conversion_cost"]
    assert cost["value"] == pytest.approx(0.0)
    path = result.manifest["tool_path"]
    assert path["compared_with_reference"] == "unconstrained"
    assert "what an application gets by default" in path["compared_with_reference_because"]
    assert path["grammar_effect"]["value"] == pytest.approx(0.5)
    assert path["grammar_effect"]["resolved"] is True


def test_a_row_refused_in_one_mode_stays_in_both(tmp_path):
    """Every row is scored in both modes, so the grammar's effect stays paired
    over the same rows and the reference is scored over all of them."""
    rows_ = labelled_rows(8)

    result = _verify(
        tmp_path,
        rows_,
        {"constrained": _rows(rows_, hits=6, refusals=2), "unconstrained": _rows(rows_, hits=8)},
    )

    modes = result.manifest["tool_path"]["modes"]
    assert modes["constrained"]["score"]["n"] == modes["unconstrained"]["score"]["n"] == 8
    assert (modes["constrained"]["no_reply"], modes["unconstrained"]["no_reply"]) == (2, 0)
    assert result.manifest["quality"]["reference"]["n"] == 8
    assert result.manifest["tool_path"]["grammar_effect"]["value"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "same, status",
    # Against an untuned base there is no conversion cost to attribute, so a
    # candidate that differs from it ends unmeasured rather than passed.
    [(True, Status.FAILED_SMOKE), (False, Status.UNMEASURED)],
)
def test_divergence_from_the_base_compares_calls_on_the_tool_path(tmp_path, same, status):
    """Found in review: it compared the candidate's text, which is empty when it
    called something, so it passed a candidate that was the untuned base."""
    rows_ = labelled_rows(8)
    both = _rows(rows_, 8)
    base_texts = marked(
        correct_texts(rows_) if same else [call_text("open_app", app="x") for _ in rows_]
    )

    result = _verify(
        tmp_path,
        rows_,
        {"constrained": both, "unconstrained": both},
        reference_texts=base_texts,
        request_extra={"reference_role": ReferenceRole.UNTUNED_BASE},
    )

    assert result.status is status
    checks = result.manifest["liveness"]["candidate"]["checks"]
    (divergence,) = [c for c in checks if c["name"] == "divergence from baseline"]
    assert divergence["observed"]["divergence_share"] == pytest.approx(0.0 if same else 1.0)


def test_declarations_reach_the_reference_and_the_rendering_check_only_on_the_tool_path(
    tmp_path,
):
    """Found in review: with a family that has no tool channel, the reference and
    the rendering check were handed the declarations while the candidate, on
    `litert-lm run`, never was -- a rendering check vouching for a prompt the
    candidate is never sent."""

    def request(reference: str) -> VerifyRequest:
        return VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference=reference,
            data=tmp_path / "heldout.jsonl",
            prompt_mode=PromptMode.RUNTIME_RENDERED,
        )

    tool_path = build_backends(request(FUNCTIONGEMMA), declarations=DECLS)
    assert tool_path.candidate.declarations == DECLS
    assert tool_path.reference.declarations == DECLS
    assert tool_path.rendering.declarations == DECLS

    text_path = build_backends(request("Qwen/Qwen3-0.6B"), declarations=DECLS)
    assert text_path.reference.declarations is None
    assert text_path.rendering.declarations is None


def _no_reply(rows_, kind: str, reason: str) -> list[dict]:
    return [_row(i, error=reason, kind=kind) for i in range(len(rows_))]


def _other_on(rows_, count: int, answered: int = 0, parsed: int = 0) -> list[dict]:
    """`answered` right calls, `parsed` parse refusals, `count` unread reasons, misses after."""
    out = _rows(rows_, hits=answered, refusals=parsed)
    for i in range(answered + parsed, answered + parsed + count):
        out[i] = _row(i, error="INTERNAL: tensor arena exhausted", kind="other")
    return out


@pytest.mark.parametrize(
    "constrained",
    [
        # A constraint the runtime could not build refuses every prompt.
        lambda rows_: _no_reply(rows_, "other", "Failed to create constraint with tools."),
        # One row is enough: litetune cannot say whose it was.
        lambda rows_: _other_on(rows_, 1, answered=7),
    ],
)
def test_a_grammar_on_run_with_a_reason_litetune_does_not_read_is_not_measured(
    tmp_path, constrained
):
    """Found in review: a grammar-on run where every reply failed for a reason
    that was not the parser -- a constraint it could not build -- passed with
    "the grammar lowered the score". It is a mode not measured, and the
    grammar-off run, the one compared with the reference, stands."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {"constrained": constrained(rows_), "unconstrained": _rows(rows_, hits=8)},
    )

    assert result.status is Status.PASSED
    path = result.manifest["tool_path"]
    assert path["modes"]["constrained"]["available"] is False
    assert "does not read as the model's" in path["modes"]["constrained"]["reason"]
    assert path["grammar_effect"]["available"] is False
    assert "sign" not in path["grammar_effect"]
    assert any("grammar-on run was not measured" in x for x in result.manifest["limitations"])
    assert not any("lowered the score" in x for x in result.manifest["limitations"])


def test_a_grammar_on_run_that_fails_outright_ends_the_run(tmp_path):
    """Found in review: any failure of the grammar-on script was reported as
    that mode not measured, and the run passed -- a declared tool executed, a
    `TypeError` from the binding, a native abort. Anything but the runtime
    replying ends the run in either mode; only a reason it gave for not
    replying is a mode not measured."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {"unconstrained": _rows(rows_, hits=8)},
        fail="TypeError: send_message() got an unexpected keyword argument",
        fail_modes=("constrained",),
    )

    assert result.status is Status.FAILED_HARNESS
    reason = result.manifest["tool_path"]["modes"]["constrained"]["reason"]
    assert "unexpected keyword argument" in reason


@pytest.mark.parametrize(
    "unconstrained",
    [
        lambda rows_: _no_reply(rows_, "other", "INTERNAL: tensor arena exhausted"),
        # Found in review: one answered row turned the other seven into the
        # model's wrong answers, and one parse refusal made it a smoke failure.
        lambda rows_: _other_on(rows_, 7, answered=1),
        lambda rows_: _other_on(rows_, 7, parsed=1),
        lambda rows_: _other_on(rows_, 1, answered=7),
    ],
)
def test_a_grammar_off_run_with_a_reason_litetune_does_not_read_is_a_harness_failure(
    tmp_path, unconstrained
):
    """On the run the reference is compared with, nothing is scored: the
    reason could be the machine's as easily as the model's."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {"unconstrained": unconstrained(rows_), "constrained": _rows(rows_, hits=8)},
    )

    assert result.status is Status.FAILED_HARNESS
    assert "does not read as the model's" in json.dumps(result.manifest)
    assert "tensor arena exhausted" in json.dumps(result.manifest)


def test_every_prompt_over_the_token_limit_is_the_bundles_failure_not_the_harness(tmp_path):
    """A reason litetune reads: the bundle cannot take the prompt an
    application sends, so the application gets nothing. Scored, on every row as
    on one, and said."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path,
        rows_,
        {
            "unconstrained": _no_reply(rows_, "too_long", "reached the bundle's token limit"),
            "constrained": _rows(rows_, hits=8),
        },
    )

    assert result.status is Status.FAILED_SMOKE
    assert "reached the bundle's token limit" in json.dumps(result.manifest["liveness"])


def test_a_reply_with_several_calls_on_every_prompt_is_alive_and_wrong(tmp_path):
    """Found in review: several calls were stored as no call, so liveness said
    nothing came back when every prompt had returned two."""
    rows_ = labelled_rows(8)
    doubled = _rows(rows_, hits=8)
    for row in doubled:
        row["calls"].append({"name": "open_app", "arguments": {}})

    result = _verify(tmp_path, rows_, {"constrained": doubled, "unconstrained": doubled})

    assert result.status is not Status.FAILED_SMOKE
    mode = result.manifest["tool_path"]["modes"]["unconstrained"]
    assert mode["several_calls"] == 8
    assert mode["score"]["exact_match"]["value"] == pytest.approx(0.0)


def test_the_reference_is_held_to_the_one_call_rule_too(tmp_path):
    """Found in review: the reference's parser took the first of two calls
    while the candidate's two were wrong, so the difference between the rules
    was reported as conversion cost."""
    rows_ = labelled_rows(8)
    texts = [f"<start_function_call>{t}<end_function_call>" for t in correct_texts(rows_)]
    texts[0] += "<start_function_call>call:open_app{}<end_function_call>"

    result = _verify(
        tmp_path,
        rows_,
        {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 8)},
        reference_texts=texts,
    )

    assert result.manifest["quality"]["reference"]["exact_match"]["value"] == pytest.approx(7 / 8)


def test_a_runtime_other_than_the_one_the_default_was_read_from_is_said(tmp_path):
    rows_ = labelled_rows(8)
    both = _rows(rows_, 8)

    same = _verify(tmp_path, rows_, {"constrained": both, "unconstrained": both})
    other = _verify(
        tmp_path, rows_, {"constrained": both, "unconstrained": both}, runtime_version="0.17.1"
    )
    unnamed = _verify(
        tmp_path, rows_, {"constrained": both, "unconstrained": both}, runtime_version=None
    )

    assert not any("this run used" in x for x in same.manifest["limitations"])
    said = next(x for x in other.manifest["limitations"] if "this run used" in x)
    assert "litert-lm 0.16.1's source; this run used 0.17.1" in said
    # The kinds of a missing reply are read from that version's log sentences too.
    assert "log sentences" in said
    assert any("a version it could not name" in x for x in unnamed.manifest["limitations"])


def test_the_default_was_read_from_the_runtime_this_project_pins():
    """The constant names a version; the pin is what actually runs."""
    from litetune import envs
    from litetune.toolpath import GRAMMAR_OFF_BY_DEFAULT_IN

    assert f"litert-lm=={GRAMMAR_OFF_BY_DEFAULT_IN}" in envs.RUNTIME.requirements


def test_a_prompt_over_the_token_limit_is_counted_as_the_bundles_limit(tmp_path):
    """Scored wrong -- an application gets no reply -- but named as a capacity
    of the converted bundle rather than an answer the model gave."""
    rows_ = labelled_rows(8)
    mixed = _rows(rows_, hits=6)
    for i in (6, 7):
        mixed[i] = _row(
            i, error="the prompt was longer than the bundle's token limit", kind="too_long"
        )

    result = _verify(tmp_path, rows_, {"constrained": mixed, "unconstrained": mixed})

    mode = result.manifest["tool_path"]["modes"]["unconstrained"]
    assert (mode["no_reply"], mode["too_long"], mode["parse_refusals"]) == (2, 2, 0)
    assert any(
        "2 because the prompt, with the declarations the runtime renders into it, reached the "
        "bundle's token limit" in x
        for x in result.manifest["limitations"]
    )


def test_divergence_compares_calls_as_answers_not_as_text(tmp_path):
    """Found in review: calls were turned into text and read back by the text
    parser, so two different calls carrying the same text collapsed. Compared as
    calls, a double `7.0` from the runtime is the base's `7`, and a different
    tool is a different answer whatever its arguments say."""
    rows_ = [
        {"prompt": f"wake me at {h}", "target": {"name": "set_alarm", "args": {"hour": h}}}
        for h in range(8)
    ]
    returned = [
        _row(h, calls=[{"name": "set_alarm", "arguments": {"hour": float(h)}}]) for h in range(8)
    ]
    same_base = marked([f"call:set_alarm{{hour:{h}}}" for h in range(8)])
    other_base = marked(
        [f"call:send_email{{body:<escape>call:set_alarm{{hour:{h}}}<escape>}}" for h in range(8)]
    )

    same = _verify(
        tmp_path,
        rows_,
        {"constrained": returned, "unconstrained": returned},
        reference_texts=same_base,
        request_extra={"reference_role": ReferenceRole.UNTUNED_BASE},
    )
    other = _verify(
        tmp_path,
        rows_,
        {"constrained": returned, "unconstrained": returned},
        reference_texts=other_base,
        request_extra={"reference_role": ReferenceRole.UNTUNED_BASE},
    )

    def share(result):
        checks = result.manifest["liveness"]["candidate"]["checks"]
        (check,) = [c for c in checks if c["name"] == "divergence from baseline"]
        return check["observed"]["divergence_share"]

    assert share(same) == pytest.approx(0.0)
    assert share(other) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"error": "boom"},  # an error needs its kind
        {"kind": "parse"},  # and a kind its error
        {"error": "x", "kind": "bogus"},
        {"error": "x", "kind": "parse", "calls": ({"name": "t", "arguments": {}},)},
    ],
)
def test_a_row_cannot_be_built_in_a_state_no_reply_is_in(fields):
    """Found in review: only `read` refused these; built directly, a row with
    an error and no kind fell in no count, and one with a call and an error
    was scored right and counted as no reply."""
    with pytest.raises(ToolPathError):
        ToolPathRow(0, **fields)


def test_a_second_run_keeps_nothing_from_the_first(tmp_path):
    """Found in review: `rows` and `unavailable` outlived a `generate`, so a
    mode could be both measured and not."""
    rows_ = labelled_rows(2)
    env = CannedEnv(
        by_mode={"constrained": _rows(rows_, hits=2), "unconstrained": _rows(rows_, hits=2)}
    )
    backend = ToolPathBackend(model=tmp_path / "m.litertlm", declarations=DECLS, env=env)
    backend.generate(["a", "b"])
    assert set(backend.rows) == {"constrained", "unconstrained"}

    env.by_mode["constrained"] = _no_reply(rows_, "other", "INTERNAL: x")
    backend.generate(["a", "b"])
    assert set(backend.rows) == {"unconstrained"}
    assert set(backend.unavailable) == {"constrained"}

    env.fail, env.fail_modes = "boom", ("unconstrained",)
    generations = backend.generate(["a", "b"])
    assert backend.rows == {}
    assert set(backend.unavailable) == {"unconstrained"}
    assert all(g.harness_error for g in generations)


def test_each_mode_says_over_how_many_prompts_it_counted(tmp_path):
    """Found in review: liveness counts every row and the scored block the
    labelled ones, under the same keys."""
    rows_ = labelled_rows(8)
    both = _rows(rows_, hits=6, refusals=2)

    result = _verify(tmp_path, rows_, {"constrained": both, "unconstrained": both})

    assert result.manifest["tool_path"]["modes"]["unconstrained"]["of"] == 8
    liveness = result.manifest["liveness"]["candidate"]["checks"]
    assert liveness[0]["observed"]["of"] == 8


@pytest.mark.parametrize(
    "candidate, base",
    [
        # Found in review: a reply with two calls and a base that answered in
        # prose were both "no call", so a candidate that always called twice
        # read as the base.
        ("two calls", "prose"),
        # And a parse refusal read as the base's prose.
        ("no reply", "prose"),
        # A base that wrote its call outside the markers answered in prose, as
        # the runtime reads it.
        ("prose", "a call outside the markers"),
    ],
)
def test_divergence_tells_what_an_application_is_handed_apart(tmp_path, candidate, base):
    rows_ = labelled_rows(8)
    target = rows_[0]["target"]
    call = {"name": target["name"], "arguments": target["args"]}
    row = {
        "two calls": lambda i: _row(i, calls=[call, call]),
        "no reply": lambda i: _row(i, error=PARSE_REFUSED, kind="parse"),
        "prose": lambda i: _row(i, text="done"),
    }[candidate]
    text = {
        "prose": "I have changed it for you.",
        "a call outside the markers": call_text(target["name"], **target["args"]),
    }[base]
    # One prompt answered alike on both sides, so liveness lets the check run.
    replies = [_row(0, calls=[call])] + [row(i) for i in range(1, 8)]
    texts = marked([call_text(target["name"], **target["args"])]) + [text] * 7

    result = _verify(
        tmp_path,
        rows_,
        {"constrained": replies, "unconstrained": replies},
        reference_texts=texts,
        request_extra={"reference_role": ReferenceRole.UNTUNED_BASE},
    )

    checks = result.manifest["liveness"]["candidate"]["checks"]
    (divergence,) = [c for c in checks if c["name"] == "divergence from baseline"]
    same_shape = (candidate, base) == ("prose", "a call outside the markers")
    assert divergence["observed"]["divergence_share"] == pytest.approx(0.0 if same_shape else 7 / 8)


@pytest.mark.parametrize(
    "text, scored",
    [
        ("{call}", False),  # no markers: text to the runtime
        ("<start_function_call>{call}", False),  # no end marker
        ("{call} <start_function_call><end_function_call>", False),
        ("<start_function_call>{call}{call}<end_function_call>", False),  # two in one block
        ("<start_function_call>{call}<end_function_call>" * 2, False),  # two calls
        ("<start_function_call>{call}<end_function_call>then more", True),
    ],
)
def test_the_reference_is_read_as_the_runtime_reads_a_reply(tmp_path, text, scored):
    """Found in review: the reference counted markers rather than reading
    calls, so a call without markers, two calls in one block or a call beside
    empty markers scored right on the reference side and wrong on the
    candidate's, and the difference was reported as conversion cost."""
    rows_ = labelled_rows(8)
    texts = [text.format(call=call_text(r["target"]["name"], **r["target"]["args"])) for r in rows_]
    both = _rows(rows_, hits=8)

    result = _verify(tmp_path, rows_, {"constrained": both, "unconstrained": both}, texts)

    reference = result.manifest["quality"]["reference"]["exact_match"]["value"]
    assert reference == pytest.approx(1.0 if scored else 0.0)
    counted = result.manifest["tool_path"]["reference"]
    assert counted["of"] == 8
    assert counted["several_calls"] == (8 if text.count("<start") == 2 else 0)
    assert counted["no_reply"] == (8 if "}}call" in text or "{call}{call}" in text else 0)


def test_a_structural_backend_that_keeps_no_rows_is_not_a_model_that_returned_nothing(tmp_path):
    """Found in review: liveness read the rows only from a `ToolPathBackend`,
    so any other backend declaring structured answers failed as a model that
    returned no call on any prompt."""

    rows_ = labelled_rows(8)
    split = tmp_path / "heldout.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    candidate = FakeBackend(
        texts=correct_texts(rows_),
        prompt_mode=PromptMode.RUNTIME_RENDERED,
        scores_structurally=True,
    )
    reference = FakeBackend(
        model="org/reference", texts=correct_texts(rows_), prompt_mode=PromptMode.RUNTIME_RENDERED
    )

    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=split,
            prompt_mode=PromptMode.RUNTIME_RENDERED,
        ),
        backends=BackendPair(candidate=candidate, reference=reference),
    )

    assert result.status is Status.FAILED_HARNESS
    assert "keeps no per-mode rows" in json.dumps(result.manifest["liveness"])
