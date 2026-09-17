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


# -- the backend, and how `verify` picks it ----------------------------------


from conftest import FakeBackend, correct_texts, labelled_rows  # noqa: E402

from litetune.prompt_mode import PromptMode  # noqa: E402
from litetune.toolpath import ToolPathBackend  # noqa: E402
from litetune.verify import (  # noqa: E402
    BackendPair,
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

    def provision(self, events=None):
        return None

    def run(self, argv, timeout=None):
        spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        if self.fail is not None:
            return types.SimpleNamespace(returncode=3, stderr=self.fail)
        mode = "constrained" if spec["constrained"] else "unconstrained"
        Path(spec["out"]).write_text(json.dumps(self.by_mode[mode]), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stderr="")


def _rows(rows_, hits: int, refusals: int = 0) -> list[dict]:
    """A row per example: `hits` correct calls, then `refusals` refusals, then misses."""
    out = []
    for i, row in enumerate(rows_):
        target = row["target"]
        if i < hits:
            out.append(
                {
                    "index": i,
                    "call": {"name": target["name"], "arguments": target["args"]},
                    "text": "",
                    "error": None,
                }
            )
        elif i < hits + refusals:
            out.append(
                {
                    "index": i,
                    "call": None,
                    "text": "",
                    "error": "INVALID_ARGUMENT: Failed to parse tool calls from code block",
                }
            )
        else:
            out.append(
                {
                    "index": i,
                    "call": {"name": target["name"], "arguments": {"color": "no"}},
                    "text": "",
                    "error": None,
                }
            )
    return out


def _verify(tmp_path, rows_, by_mode, **kwargs):
    split = tmp_path / "heldout.jsonl"
    split.write_text("\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8")
    candidate = ToolPathBackend(
        model=tmp_path / "m.litertlm",
        declarations=DECLS,
        env=CannedEnv(by_mode=by_mode, **kwargs),
    )
    request = VerifyRequest(
        model=tmp_path / "m.litertlm",
        reference="org/reference",
        data=split,
        prompt_mode=PromptMode.RUNTIME_RENDERED,
    )
    reference = FakeBackend(
        model="org/reference",
        texts=correct_texts(rows_),
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
    assert result.manifest["tool_path"]["constrained"]["score"]["n"] == 8


def test_both_decoding_modes_are_reported(tmp_path):
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 4)}
    )

    path = result.manifest["tool_path"]
    assert path["constrained"]["score"]["exact_match"]["value"] == pytest.approx(1.0)
    assert path["unconstrained"]["score"]["exact_match"]["value"] == pytest.approx(0.5)


def test_modes_that_disagree_raise_a_limitation(tmp_path):
    """The grammar can carry a model that never learned the format. A run that
    reported only the constrained number would report such a model as working."""
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 8), "unconstrained": _rows(rows_, 4)}
    )

    assert any("did not agree" in limitation for limitation in result.manifest["limitations"])


def test_modes_that_agree_raise_none(tmp_path):
    rows_ = labelled_rows(8)
    result = _verify(
        tmp_path, rows_, {"constrained": _rows(rows_, 6), "unconstrained": _rows(rows_, 6)}
    )

    assert not any("did not agree" in limitation for limitation in result.manifest["limitations"])


def test_a_refusal_is_counted_and_kept_out_of_the_score(tmp_path):
    """A model whose output the parser rejected failed differently from one that
    called the wrong tool, and an average over both describes neither."""
    rows_ = labelled_rows(8)
    both = _rows(rows_, hits=6, refusals=2)
    result = _verify(tmp_path, rows_, {"constrained": both, "unconstrained": both})

    path = result.manifest["tool_path"]
    assert path["constrained"]["refused_by_the_runtime"] == 2
    # Six correct out of the six the runtime could read, not out of eight.
    assert path["constrained"]["score"]["n"] == 6
    assert path["constrained"]["score"]["exact_match"]["value"] == pytest.approx(1.0)
    assert any("refused 2 of 8" in limitation for limitation in result.manifest["limitations"])


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
