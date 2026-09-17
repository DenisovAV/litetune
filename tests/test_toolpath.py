"""The runtime-side tool-path script, against a fake `litert_lm`.

The script is a string, so it is `exec`'d and its `main` run against a fake
module -- the same technique `test_rendering.py` uses, and for the same reason:
the contract that matters is what it asks the runtime for and what it writes,
and neither needs a bundle to pin.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

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


def _fake_litert_lm(runtime: FakeRuntime) -> Any:
    module: Any = types.ModuleType("litert_lm")

    class Conversation:
        def __init__(self, **kwargs: Any):
            runtime.conversations.append(kwargs)

        def __enter__(self) -> Conversation:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def send_message(self, prompt: str) -> Any:
            runtime.sent.append(prompt)
            reply = runtime.replies[len(runtime.sent) - 1]
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


def test_a_declared_tool_raises_rather_than_running(tmp_path, monkeypatch):
    """The guard behind the flag above. If automatic calling were ever on, the
    run says so instead of quietly executing a caller's function."""
    runtime = FakeRuntime(replies=[_reply("set_colour", {})])

    _run(runtime, tmp_path, monkeypatch)

    (args,) = runtime.conversations
    with pytest.raises(AssertionError, match="automatic tool calling should be off"):
        args["tools"][0].execute({})


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


def test_a_structured_call_comes_back_with_its_argument_types(tmp_path, monkeypatch):
    """The runtime types the arguments, and that survives to the row.

    Flattening them here would throw away the thing the tool path has that the
    text path does not.
    """
    runtime = FakeRuntime(replies=[_reply("set_alarm", {"hour": 7, "loud": True})])

    _, out = _run(runtime, tmp_path, monkeypatch)

    (row,) = json.loads(out.read_text(encoding="utf-8"))
    assert row == {
        "index": 0,
        "call": {"name": "set_alarm", "arguments": {"hour": 7, "loud": True}},
        "text": "",
        "error": None,
    }


def test_a_reply_with_no_call_is_a_result_not_a_failure(tmp_path, monkeypatch):
    """A model that answered in prose instead of calling anything is something
    to see, and a row with neither a call nor text could not be told from one
    that failed."""
    runtime = FakeRuntime(replies=[{"content": [{"text": "I cannot do that."}]}])

    _, out = _run(runtime, tmp_path, monkeypatch)

    (row,) = json.loads(out.read_text(encoding="utf-8"))
    assert row["call"] is None
    assert row["error"] is None
    assert row["text"] == "I cannot do that."


def test_every_prompt_gets_a_row_in_order(tmp_path, monkeypatch):
    prompts = ["a", "b", "c"]
    runtime = FakeRuntime(replies=[_reply("t", {"i": i}) for i in range(3)])

    _, out = _run(runtime, tmp_path, monkeypatch, prompts=prompts)

    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert runtime.sent == prompts
    # One conversation per prompt: every row is a first turn, as every other
    # measurement in this project asks it.
    assert len(runtime.conversations) == 3


# -- how it fails ------------------------------------------------------------


def test_the_runtimes_refusal_of_one_row_is_recorded_not_raised(tmp_path, monkeypatch):
    """`Failed to parse tool calls from code block` is the runtime rejecting a
    generation. It is a fact about that row, in the runtime's own words, and the
    rows around it are still measurable."""
    runtime = FakeRuntime(
        replies=[
            _reply("set_colour", {"colour": "red"}),
            ValueError("INVALID_ARGUMENT: Failed to parse tool calls from code block"),
            _reply("set_colour", {"colour": "blue"}),
        ]
    )

    code, out = _run(runtime, tmp_path, monkeypatch, prompts=["a", "b", "c"])

    assert code == 0
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[1]["call"] is None
    assert "Failed to parse tool calls" in rows[1]["error"]
    assert rows[0]["call"] is not None and rows[2]["call"] is not None


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


def test_a_script_that_writes_the_wrong_number_of_rows_is_a_harness_failure(tmp_path):
    """Silent truncation would score the missing prompts as unanswered."""

    class ShortEnv:
        name = "runtime"

        def provision(self, events=None):
            return None

        def run(self, argv, timeout=None):
            spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
            Path(spec["out"]).write_text(json.dumps([{"index": 0}]), encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stderr="")

    probe = ToolPathProbe(model=tmp_path / "m.litertlm", declarations=TOOLS, env=ShortEnv())

    with pytest.raises(ToolPathError, match="wrote 1 rows for 2 prompts"):
        probe.observe(["a", "b"], constrained=True)


def test_a_refused_row_says_so(tmp_path):
    assert ToolPathRow(0, error="boom").refused is True
    assert ToolPathRow(0, call={"name": "t", "arguments": {}}).refused is False
