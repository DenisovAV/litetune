"""Measuring a tool-calling model the way an application calls it.

`evaluate.LiteRtLmBackend` runs `litert-lm run` and reads text off stdout. That
is the text path, and a FunctionGemma model does not answer on it the way it
answers an application: the runtime renders the declarations into the prompt,
parses the model's output itself, and hands back a structured call. A number
measured by scraping text is a number about a path nobody serves.

So this asks the runtime the question an application asks --
`create_conversation(tools=..., automatic_tool_calling=False)` then
`send_message` -- and reads `tool_calls` off the reply. Three things follow from
that, and each of them is the reason for a piece of this module:

**The runtime's parser is the one being measured, not ours.** A model that
writes a call the runtime cannot read is a model that fails for an application.
`metrics.parse_call` still exists and is still used -- for the transformers
reference, which produces text and nothing else. The runtime's Python binding
reports every failed reply as one `RuntimeError("litert_lm_conversation_send_message
failed")` (`conversation.py`, v0.16.1) and writes the reason to its log, so the
script reads the log around each reply to record the reason with the row.

**Constrained decoding is a choice the caller makes, and both are measured.**
LiteRT-LM v0.16.1 leaves it off unless the caller enables it
(`ConstrainedDecodingConfig(enable=True)`; the binding sets nothing otherwise),
so the unconstrained number is what an application gets by default. With it on
the runtime holds the model to the declared grammar -- which can carry a model
that never learned the format, and which, measured on 2026-09-17, removed
arguments written out of declared order. One number from each mode is the only
way to see either, so both are run and neither is reported alone.

**Automatic tool calling is off.** Left on, the runtime would execute the
declared tools and loop until prose. With it off the binding returns the reply
before it reaches `_handle_tool_calls`, so no declared tool can run; if one were
ever asked to, the script stops and says so rather than carry on.

The script runs under `envs.RUNTIME` in the shape `rendering.py` established: a
spec file in, a JSON row file out, one row per prompt in order.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.evaluate import GREEDY, DecodeConfig, Generation
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.metrics import ToolCall
from litetune.prompt_mode import PromptMode

logger = logging.getLogger(__name__)

TOOL_PATH_TIMEOUT_S = 3600


class ToolPathError(RuntimeError):
    """The tool path could not be measured at all. Never a statement about the model."""


_TOOL_PATH_SCRIPT = r'''
"""The runtime's answer to each prompt, through its own tool path."""
import json
import os
import sys
import tempfile
from pathlib import Path


def text_of(reply):
    """Whatever prose came back beside the call, for the record.

    A tool-calling reply usually has none. It is kept because a model that
    answered in prose instead of calling anything is a result worth seeing, and
    a row with no call and no text cannot be told apart from a row that failed.
    """
    parts = []
    for item in reply.get("content") or []:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    return "".join(parts)


def calls_of(reply):
    """Every `tool_calls` entry, flattened to name and arguments.

    The runtime returns the OpenAI shape, `{"type": "function", "function":
    {...}}`. All of them, not the first: an application acts on each call it
    is handed, so a second one the target does not ask for is part of the
    answer.
    """
    calls = []
    for entry in reply.get("tool_calls") or []:
        function = entry.get("function", entry) if isinstance(entry, dict) else {}
        calls.append({"name": function.get("name"), "arguments": function.get("arguments") or {}})
    return calls


def logged(action):
    """Run `action` with the process's stderr captured: (result, exception, log).

    The binding raises one generic `RuntimeError` for any failed reply and the
    runtime writes the reason to file descriptor 2, so the descriptor itself is
    redirected -- `sys.stderr` is not where native code writes.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as capture:
        os.dup2(capture.fileno(), 2)
        try:
            result, error = action(), None
        except Exception as exc:  # noqa: BLE001
            result, error = None, exc
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        capture.seek(0)
        log = capture.read().decode("utf-8", errors="replace")
    return result, error, log


def reason_in(log):
    """The runtime's own sentence for a failed reply: its last error line."""
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    errors = [line for line in lines if "rror" in line or "ailed" in line or "INVALID" in line]
    return (errors or lines or [""])[-1][-400:]


def runtime_version():
    try:
        from importlib.metadata import version

        return version("litert-lm")
    except Exception:  # noqa: BLE001
        return None


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    import litert_lm

    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except AttributeError:
        pass

    executed = []

    class Declared(litert_lm.Tool):
        """A declaration the runtime renders and must never run.

        `create_conversation` takes `Tool` instances, never raw JSON: it asks
        each for `get_tool_description()`. Handing the file's entries back
        unchanged keeps the runtime's own refusal as the refusal. `execute` is
        unreachable with automatic tool calling off; if it is reached it is
        recorded, because the binding catches what it raises and would feed
        the text back to the model as a tool response.
        """

        def __init__(self, description):
            self._description = description

        def get_tool_description(self):
            return self._description

        def execute(self, param):
            executed.append(self._description.get("function", {}).get("name"))
            return {"error": "not executed: litetune is measuring, not running tools"}

    conversation_args = {
        "sampler_config": litert_lm.SamplerConfig(
            top_k=1, top_p=1.0, temperature=1.0, seed=0
        ),
        "max_output_tokens": spec["max_tokens"],
        "tools": [Declared(entry) for entry in spec["tools"]],
        "automatic_tool_calling": False,
        "constrained_decoding_config": litert_lm.ConstrainedDecodingConfig(
            enable=spec["constrained"]
        ),
    }

    rows = []
    with litert_lm.Engine(
        spec["model"], backend=litert_lm.Backend.CPU()
    ) as engine:
        for index, prompt in enumerate(spec["prompts"]):
            # One conversation per prompt: each row is a first turn, the way
            # every other measurement in this project asks it.
            try:
                conversation = engine.create_conversation(**conversation_args)
            except Exception as exc:  # noqa: BLE001
                # Not a row failure. If a conversation with these declarations
                # cannot be created, no row can be measured and reporting the
                # rest as wrong answers would be a lie about the model.
                sys.stderr.write(
                    "could not create a conversation with declarations: "
                    f"{type(exc).__name__}: {exc}\n"
                )
                return 3
            with conversation:
                reply, exc, log = logged(lambda: conversation.send_message(prompt))
            if executed:
                sys.stderr.write(
                    f"the runtime executed the declared tool {executed[0]!r} while measuring, "
                    "with automatic tool calling off\n"
                )
                return 3
            if exc is not None:
                # A result about this row: the runtime gave no reply an
                # application could act on. Its reason is in its log.
                reason = reason_in(log)
                rows.append(
                    {
                        "index": index,
                        "calls": [],
                        "text": "",
                        "error": f"{type(exc).__name__}: {exc}" + (f": {reason}" if reason else ""),
                        "parse_refusal": "Failed to parse tool calls" in log,
                    }
                )
                continue
            rows.append(
                {
                    "index": index,
                    "calls": calls_of(reply),
                    "text": text_of(reply),
                    "error": None,
                    "parse_refusal": False,
                }
            )
    Path(spec["out"]).write_text(
        json.dumps({"runtime_version": runtime_version(), "rows": rows}), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass(frozen=True)
class ToolPathRow:
    """One prompt's answer through the tool path.

    No calls and no error is a real result -- the model answered without
    calling anything. `error` is set when the runtime gave no reply at all,
    with the reason from its log; `parse_refusal` says whether that reason was
    its call parser rejecting the generation. Either way an application got
    nothing it could act on, and the row is scored as a wrong answer, counted
    apart so the reason is visible.
    """

    index: int
    calls: tuple[dict[str, Any], ...] = ()
    text: str = ""
    error: str | None = None
    parse_refusal: bool = False

    @property
    def refused(self) -> bool:
        return self.error is not None

    @property
    def call(self) -> dict[str, Any] | None:
        """The one call this row answered with, or `None` for none or several.

        Every target is one call, so a reply carrying two is not that answer:
        an application would act on both.
        """
        return self.calls[0] if len(self.calls) == 1 else None

    @classmethod
    def read(cls, position: int, row: Any) -> ToolPathRow:
        """A row as the script wrote it. Raises `ToolPathError` for anything else."""
        if not isinstance(row, dict) or row.get("index") != position:
            raise ToolPathError(
                f"the tool-path script wrote row {position} as {str(row)[:200]}, which is not "
                "that row"
            )
        calls = row.get("calls")
        if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
            raise ToolPathError(f"the tool-path script wrote row {position} with calls {calls!r}")
        return cls(
            index=position,
            calls=tuple(calls),
            text=str(row.get("text") or ""),
            error=None if row.get("error") is None else str(row["error"]),
            parse_refusal=bool(row.get("parse_refusal")),
        )


@dataclass
class ToolPathProbe:
    """Runs the script above in the runtime's environment, once per decoding mode."""

    model: Path
    declarations: list
    max_tokens: int = 128
    env: envs.StageEnv = envs.RUNTIME
    auto_provision: bool = True
    timeout_s: int = TOOL_PATH_TIMEOUT_S
    # Filled by `observe`, from the runtime's own metadata: which runtime
    # produced the rows, because a manifest that cannot say so cannot be
    # compared with a run on another one.
    runtime_version: str | None = field(default=None, init=False)

    def observe(
        self,
        prompts: Sequence[str],
        constrained: bool,
        events: EventStream | None = None,
    ) -> list[ToolPathRow]:
        """One row per prompt, in order. Raises only when nothing could be measured."""
        if self.auto_provision:
            self.env.provision(events=events)
        with tempfile.TemporaryDirectory(prefix="litetune-toolpath-") as tmp:
            work = Path(tmp)
            script = work / "toolpath.py"
            script.write_text(_TOOL_PATH_SCRIPT, encoding="utf-8")
            out = work / "rows.json"
            spec = work / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "model": str(self.model),
                        "prompts": list(prompts),
                        "tools": self.declarations,
                        "max_tokens": self.max_tokens,
                        "constrained": constrained,
                        "out": str(out),
                    }
                ),
                encoding="utf-8",
            )
            if events:
                events.note(
                    f"tool path: {len(prompts)} prompts, "
                    f"{'constrained' if constrained else 'unconstrained'}",
                    total=len(prompts),
                    constrained=constrained,
                )
            try:
                proc = self.env.run(["python", str(script), str(spec)], timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                raise ToolPathError(
                    f"the tool-path script gave no result after {self.timeout_s}s"
                ) from None
            except OSError as exc:
                raise ToolPathError(f"could not start the tool-path script: {exc}") from exc
            if proc.returncode != 0 or not out.is_file():
                reading = read_returncode(proc.returncode)
                raise ToolPathError(
                    f"the tool-path script {reading.describe('the model')}: "
                    f"{(proc.stderr or '').strip()[-300:] or 'no stderr'}"
                )
            try:
                written = json.loads(out.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ToolPathError(f"the tool-path script's output is unreadable: {exc}") from exc
        rows = written.get("rows") if isinstance(written, dict) else None
        if not isinstance(rows, list) or len(rows) != len(prompts):
            raise ToolPathError(
                f"the tool-path script wrote {len(rows) if isinstance(rows, list) else 'no'} rows "
                f"for {len(prompts)} prompts"
            )
        self.runtime_version = written.get("runtime_version")
        return [ToolPathRow.read(position, row) for position, row in enumerate(rows)]


def as_tool_call(call: dict[str, Any] | None) -> ToolCall | None:
    """The runtime's call as the scorer's shape, or `None` when there was none.

    The arguments arrive typed -- `1234.0`, `True` -- and `ToolCall` keeps that
    beside the flattened view the comparison uses. Nothing is re-rendered and
    nothing is re-parsed on the way: the point of this path is that the
    runtime's parser is the one being measured.
    """
    if call is None or not isinstance(call.get("name"), str):
        return None
    return ToolCall(name=call["name"], args=dict(call.get("arguments") or {}))


DISAGREEING_MODES = (
    "the two decoding modes did not agree: {constrained:.4f} with the runtime's grammar on and "
    "{unconstrained:.4f} with it off, over {n} rows. {direction} The runtime leaves the grammar "
    "off unless the application enables it, so the grammar-off number is what an application "
    "gets by default and the grammar-on number what it gets if it enables it; neither describes "
    "the other."
)

GRAMMAR_HELPED = (
    "The grammar raised the score: output the runtime reads only when held to the grammar is a "
    "model that did not fully learn the format."
)
GRAMMAR_HURT = (
    "The grammar lowered the score: it removed or changed something the model wrote on its own "
    "-- measured once, arguments written out of declared order were dropped."
)


@dataclass
class ToolPathBackend:
    """The candidate, measured through the runtime's tool path in both modes.

    A `GenerationBackend` because it is one -- one answer per prompt, in order,
    never raising for a failed run. What it adds is on each `Generation`: the
    structured call the runtime returned, and the runtime's refusal when its
    parser rejected that generation.

    `prompt_mode` is `RUNTIME_RENDERED` and cannot be anything else: the runtime
    builds this prompt, declarations and all. `verify._tool_path_reason` keeps a
    `prerendered` run on the text path, so it never builds this.
    """

    model: Path
    declarations: list
    decode: DecodeConfig = GREEDY
    env: envs.StageEnv = envs.RUNTIME
    auto_provision: bool = True
    timeout_s: int = TOOL_PATH_TIMEOUT_S
    # Filled by `generate`: the rows from each mode, kept so the manifest can
    # carry both numbers rather than only the one that was scored.
    rows: dict[str, list[ToolPathRow]] = field(default_factory=dict, init=False)
    runtime_version: str | None = field(default=None, init=False)

    name = "litert-lm tool path"

    @property
    def model_ref(self) -> str:
        return str(self.model)

    @property
    def prompt_mode(self) -> PromptMode:
        return PromptMode.RUNTIME_RENDERED

    @property
    def decode_enforced(self) -> bool:
        # `max_output_tokens` and a greedy sampler are passed to the
        # conversation, so the numbers in `decode` governed this run.
        return True

    @property
    def scores_structurally(self) -> bool:
        """This backend's answers are calls, not text, and are scored as calls.

        On the Protocol rather than sniffed from the rows, for the reason
        `decode_enforced` records: a run where the model answered in prose on
        every prompt produces rows with no call and no refusal, which is
        indistinguishable from the text path by inspection. A backend that
        forgets to declare this must fail to type-check.
        """
        return True

    def describe(self) -> dict[str, Any]:
        return {
            # The same keys and vocabulary `LiteRtLmBackend` uses, because
            # `verify` reads the device from them: the first version put this
            # backend's name under `backend`, and a real run's manifest then
            # said the candidate was measured on "litert-lm tool path
            # (unknown)" instead of on the CPU the script asks for.
            "engine": "litert-lm",
            "backend": "cpu",
            "backend_vocabulary": "litert-lm Python API Backend",
            "path": "tool path",
            "model": self.model_ref,
            "env": self.env.name,
            "declarations": len(self.declarations),
            "automatic_tool_calling": False,
            "modes": sorted(self.rows),
            "runtime_version": self.runtime_version,
        }

    def generate(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> list[Generation]:
        """The unconstrained run's answers. Both runs happen; both are kept.

        Unconstrained is what is returned, because it is what an application
        gets by default and it is what liveness judges: a model whose own output
        never parses is not alive for an application that has not enabled the
        grammar. The constrained rows sit in `self.rows` beside them for the
        manifest and the grammar's paired effect.
        """
        probe = ToolPathProbe(
            model=self.model,
            declarations=self.declarations,
            max_tokens=self.decode.max_tokens,
            env=self.env,
            auto_provision=self.auto_provision,
            timeout_s=self.timeout_s,
        )
        for mode, constrained in (("constrained", True), ("unconstrained", False)):
            try:
                self.rows[mode] = probe.observe(prompts, constrained=constrained, events=events)
            except ToolPathError as exc:
                # Nothing was measured. Every prompt carries the same harness
                # error, which is what keeps these rows out of the score
                # entirely instead of counting as wrong answers.
                logger.warning("the tool path could not be measured: %s", exc)
                return [Generation(i, p, harness_error=str(exc)) for i, p in enumerate(prompts)]
            self.runtime_version = probe.runtime_version
        return [
            Generation(
                index=row.index,
                prompt=prompts[row.index],
                text=row.text,
                returncode=0,
                call=row.call,
                refusal=row.error,
            )
            for row in self.rows["unconstrained"]
        ]
