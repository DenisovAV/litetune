"""Measuring a tool-calling model the way an application calls it.

`evaluate.LiteRtLmBackend` asks litert-lm for text and scores the text. That
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
The transformers reference produces text and nothing else, and `verify` reads
it with `metrics.runtime_calls`, which follows the runtime's reading of a reply.
The runtime's Python binding reports every failed reply as one
`RuntimeError("litert_lm_conversation_send_message failed")` (`conversation.py`,
v0.16.1) and writes the reason to its log, so the script reads the log around
each reply to record the reason with the row.

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

import functools
import json
import logging
import re
import secrets
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


class ConversationRefused(ToolPathError):
    """The runtime would not create a conversation with these declarations in this mode."""


class NotMeasured(ToolPathError):
    """The runtime replied to the run, and not in a way litetune reads as the model's.

    Apart from its parent because it is the one failure a grammar-on run may
    be reported with while the grammar-off run stands: the script ran to its
    end and wrote every row. Anything else ends the run in either mode.
    """


_TOOL_PATH_SCRIPT = r'''
"""The runtime's answer to each prompt, through its own tool path."""
import json
import os
import re
import sys
from pathlib import Path

# The binding's one signal that the runtime gave no reply (`conversation.py`,
# litert-lm 0.16.1). Anything else `send_message` raises is not an answer from
# the model -- an API change, memory, a closed conversation -- and ends the run
# as a harness failure rather than being scored, in either decoding mode.
NO_REPLY = "send_message failed"

# absl's line prefix: severity, date, time, thread, file:line].
ABSL_PREFIX = re.compile(r"^[IWEF][0-9]{4} [0-9:.]+ +[0-9]+ [^ \]]+:[0-9]+\] *")

# Written to the log on lines of their own before and after each reply, so a
# process that dies mid-reply leaves its log cut into prompts and the parent
# reads only the one it died on. Each begins with a token the parent draws for
# the run, which no model output can carry, because the log holds the model's
# text too.
STARTED = "started prompt"
FINISHED = "finished prompt"

# The script's exit when the runtime would not create a conversation at all.
CREATE_REFUSED = 4


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


def logged(action, log_path, row, mark):
    """Run `action` with file descriptor 2 appended to `log_path`.

    Returns (reply, no-reply exception, what the runtime logged). The binding
    raises one generic `RuntimeError` for any failed reply and the runtime
    writes the reason to file descriptor 2, so the descriptor itself is
    redirected -- `sys.stderr` is not where native code writes. Into a file the
    parent can read, not a temporary one: if native code aborts mid-call, the
    reason it wrote would otherwise vanish with the process. The reply's lines
    sit between its two marks, so the parent can tell them from the others.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    with open(log_path, "ab") as capture:
        capture.write(f"\n{mark} {STARTED} {row}\n".encode("utf-8"))
        capture.flush()
        start = capture.tell()
        os.dup2(capture.fileno(), 2)
        try:
            result, error = action(), None
        except RuntimeError as exc:
            if NO_REPLY not in str(exc):
                raise
            result, error = None, exc
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
    with open(log_path, "rb") as written:
        written.seek(start)
        log = written.read().decode("utf-8", errors="replace")
    with open(log_path, "ab") as capture:
        capture.write(f"\n{mark} {FINISHED} {row}\n".encode("utf-8"))
    return result, error, log


def reason_of(log, quote=True):
    """(kind, reason) for a reply the runtime did not give, without the model's text.

    A parse failure's message carries the model's code block and full response
    (`parser_utils.cc`), and that must not travel into a manifest a bundle
    ships, so a parse failure is its kind alone. The two kinds are told apart
    by the runtime's own sentences, both in `liblitert-lm` 0.16.1. Any other
    reason is the runtime's last error line, or its last line -- only where
    `quote` says the runtime is the version those sentences were read from: on
    another, a parse failure worded differently would land here, model text and
    all.
    """
    if "Failed to parse tool calls" in log:
        return "parse", "its call parser rejected the generation"
    if "Input token ids are too long" in log:
        return (
            "too_long",
            "the prompt, with the declarations the runtime renders into it, reached the "
            "bundle's token limit",
        )
    if not quote:
        return "other", "not quoted: litetune read the log of another runtime version"
    lines = [ABSL_PREFIX.sub("", line).strip() for line in log.splitlines()]
    lines = [line for line in lines if line]
    errors = [line for line in lines if "rror" in line or "ailed" in line] or lines
    return "other", (errors[-1] if errors else "nothing in its log")[:200]


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
    version = runtime_version()

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
                return CREATE_REFUSED
            with conversation:
                reply, exc, log = logged(
                    lambda: conversation.send_message(prompt), spec["log"], index, spec["mark"]
                )
            if executed:
                sys.stderr.write(
                    f"the runtime executed the declared tool {executed[0]!r} while measuring, "
                    "with automatic tool calling off\n"
                )
                return 3
            if exc is not None:
                # A result about this row: the runtime gave no reply an
                # application could act on. Its reason is in its log.
                kind, reason = reason_of(log, quote=version == spec["log_read_from"])
                rows.append(
                    {"index": index, "calls": [], "text": "", "error": reason, "kind": kind}
                )
                continue
            rows.append(
                {
                    "index": index,
                    "calls": calls_of(reply),
                    "text": text_of(reply),
                    "error": None,
                    "kind": None,
                }
            )
    Path(spec["out"]).write_text(
        json.dumps({"runtime_version": version, "rows": rows}), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@functools.cache
def _script() -> dict[str, Any]:
    """The script's own definitions, for the parent to apply the same rules.

    Read from the one source rather than copied, so the mark the script writes
    and the reason it would have given are the ones the parent reads a crashed
    run's log with.
    """
    namespace: dict[str, Any] = {"__name__": "litetune_toolpath_script"}
    exec(compile(_TOOL_PATH_SCRIPT, "litetune toolpath script", "exec"), namespace)  # noqa: S102
    return namespace


def _said_before_dying(log: Path, mark: str, quote: bool) -> str:
    """Why a run that died mid-reply died, from that reply's lines only.

    The log holds every earlier reply's lines too, and a parse failure's carry
    the model's code block and full response (`parser_utils.cc`), so they are
    never quoted: the lines after the last reply that started and never
    finished go through the rule the script uses for a row. Empty when the run
    died outside a reply. The marks carry the run's own token, which no line
    the model wrote can.
    """
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    script = _script()
    marks = list(
        re.finditer(
            rf"^{re.escape(mark)} ({script['STARTED']}|{script['FINISHED']}) ([0-9]+)$",
            text,
            re.MULTILINE,
        )
    )
    if not marks or marks[-1].group(1) != script["STARTED"]:
        return ""
    _, reason = script["reason_of"](text[marks[-1].end() :], quote=quote)
    return f" on prompt {marks[-1].group(2)}: {reason}"


@dataclass(frozen=True)
class ToolPathRow:
    """One prompt's answer through the tool path.

    No calls and no error is a real result -- the model answered without
    calling anything. `error` is set when the runtime gave no reply at all,
    with `kind` saying why: `parse` for its call parser rejecting the
    generation, `too_long` for a prompt at the bundle's token limit, `other`
    for anything else it logged. The first two are scored as wrong answers,
    counted apart so the reason is visible: an application got nothing it
    could act on. `other` is not the model's to answer for, and a mode with one
    is not scored (`_refuse_a_mode_with_unread_reasons`).
    """

    index: int
    calls: tuple[dict[str, Any], ...] = ()
    text: str = ""
    error: str | None = None
    kind: str | None = None

    def __post_init__(self) -> None:
        # The states a reply can be in, and no others: every count and every
        # score below reads a row as exactly one of them.
        if (self.error is None) != (self.kind is None) or self.kind not in (
            None,
            *NO_REPLY_KINDS,
        ):
            raise ToolPathError(
                f"row {self.index} has error {self.error!r} and kind {self.kind!r}: a reply the "
                f"runtime did not give has both, and its kind is one of {NO_REPLY_KINDS}"
            )
        if self.error is not None and self.calls:
            raise ToolPathError(f"row {self.index} has both a reply and no reply")

    @property
    def refused(self) -> bool:
        return self.error is not None

    @property
    def handed(self) -> list[ToolCall] | None:
        """What an application is handed: `None` for no reply, else these calls in order.

        The shape `metrics.runtime_calls` gives a text, so the two compare as
        the same kind of thing. The arguments arrive typed -- `1234.0`, `True`
        -- and `ToolCall` keeps that beside the flattened view the comparison
        uses. Nothing is re-rendered and nothing is re-parsed on the way: the
        point of this path is that the runtime's parser is the one being
        measured.
        """
        if self.refused:
            return None
        return [ToolCall(name=c["name"], args=dict(c["arguments"])) for c in self.calls]

    @classmethod
    def read(cls, position: int, row: Any) -> ToolPathRow:
        """A row as the script wrote it. Raises `ToolPathError` for anything else."""
        if not isinstance(row, dict) or row.get("index") != position:
            raise ToolPathError(
                f"the tool-path script wrote row {position} as {str(row)[:200]}, which is not "
                "that row"
            )
        calls = row.get("calls")
        if not isinstance(calls, list) or not all(
            isinstance(c, dict)
            and isinstance(c.get("name"), str)
            and isinstance(c.get("arguments"), dict)
            for c in calls
        ):
            raise ToolPathError(f"the tool-path script wrote row {position} with calls {calls!r}")
        error = row.get("error")
        return cls(
            index=position,
            calls=tuple(calls),
            text=str(row.get("text") or ""),
            error=None if error is None else str(error),
            kind=row.get("kind"),
        )


NO_REPLY_KINDS = ("parse", "too_long", "other")


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
            log = work / "runtime.log"
            spec = work / "spec.json"
            mark = f"litetune-{secrets.token_hex(8)}"
            spec.write_text(
                json.dumps(
                    {
                        "model": str(self.model),
                        "prompts": list(prompts),
                        "tools": self.declarations,
                        "max_tokens": self.max_tokens,
                        "constrained": constrained,
                        "out": str(out),
                        "log": str(log),
                        "mark": mark,
                        "log_read_from": GRAMMAR_OFF_BY_DEFAULT_IN,
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
                # Where a native abort mid-reply leaves its reason: the log,
                # which fd 2 pointed at. Outside a reply it is stderr. The log
                # is quoted only where the environment pins the runtime whose
                # sentences were read, as the script quotes a row's.
                pinned = f"litert-lm=={GRAMMAR_OFF_BY_DEFAULT_IN}" in getattr(
                    self.env, "requirements", ()
                )
                stderr = (proc.stderr or "").strip()[-400:]
                said = (
                    f"the tool-path script {reading.describe('the model')}"
                    f"{_said_before_dying(log, mark, pinned)}; stderr: {stderr or 'empty'}"
                )
                if proc.returncode == _script()["CREATE_REFUSED"]:
                    raise ConversationRefused(said)
                raise ToolPathError(said)
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


def _refuse_a_mode_with_unread_reasons(rows: list[ToolPathRow]) -> None:
    """A mode where the runtime gave no reply for a reason litetune does not read.

    Two reasons are read: its call parser rejecting what the model wrote, which
    is the model's wrong answer, and a prompt reaching the bundle's token limit,
    which is the bundle's and is scored as what an application gets. Any other
    -- a constraint it could not build, memory, a sentence another runtime
    version words differently -- could be the machine as easily as the model,
    so no row of the mode is scored. On one row as on all of them: the first
    version asked whether every row failed, and one answered row turned the
    other 639 into the model's wrong answers.
    """
    unread = [row for row in rows if row.kind == "other"]
    if unread:
        reasons = sorted({row.error or "" for row in unread})
        which = ", ".join(str(row.index) for row in unread[:5]) + (
            f" and {len(unread) - 5} more" if len(unread) > 5 else ""
        )
        raise NotMeasured(
            f"the runtime gave no reply to {len(unread)} of {len(rows)} prompts (prompt "
            f"{which}) for a reason litetune does not read as the model's: "
            f"{'; '.join(reasons)[:300]}"
        )


# The runtime version on which it was established, from its source, that
# constrained decoding is off unless the caller enables it -- and whose log
# sentences `reason_of` reads a missing reply's kind from. `verify` says when a
# run used another.
GRAMMAR_OFF_BY_DEFAULT_IN = "0.16.1"

DISAGREEING_MODES = (
    "the two decoding modes did not agree: {constrained:.4f} with the runtime's grammar on and "
    "{unconstrained:.4f} with it off, over {n} rows. {direction} The runtime leaves the grammar "
    "off unless the application enables it, so the grammar-off number is what an application "
    "gets by default and the grammar-on number what it gets if it enables it; neither describes "
    "the other."
)

GRAMMAR_HELPED = (
    "The grammar raised the score: rows were right with it on that were wrong with it off. "
    "These numbers do not say what it changed on them."
)
GRAMMAR_HURT = (
    "The grammar lowered the score: rows were wrong with it on that were right with it off. "
    "These numbers do not say what it changed on them; measured once, on a model trained in "
    "an argument order other than the declared one, it dropped the arguments written out of "
    "that order."
)


@dataclass
class ToolPathBackend:
    """The candidate, measured through the runtime's tool path in both modes.

    A `GenerationBackend` because it is one -- one answer per prompt, in order,
    never raising for a failed run. What it adds is on each `Generation`: the
    structured call the runtime returned, and why it gave no reply when it gave
    none -- its call parser rejecting the generation being one reason of several.

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
    # A mode that was not measured, and why. Grammar off landing here, or
    # anything ending the run, means nothing is scored: the reason says which.
    unavailable: dict[str, str] = field(default_factory=dict, init=False)
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
        self.rows, self.unavailable, self.runtime_version = {}, {}, None
        # Grammar off first: it is the run the reference is compared with, so
        # the grammar-on run not being measured must not cost it.
        for mode, constrained in (("unconstrained", False), ("constrained", True)):
            state = "on" if constrained else "off"
            try:
                rows = probe.observe(prompts, constrained=constrained, events=events)
                # Before the rows are judged: a mode not measured is where the
                # runtime's version matters most.
                self.runtime_version = probe.runtime_version
                _refuse_a_mode_with_unread_reasons(rows)
            except ToolPathError as exc:
                logger.warning("the tool path could not be measured (%s): %s", mode, exc)
                # A conversation the runtime would not create with the grammar
                # on is the grammar's refusal: with it off, the same
                # declarations made one for every prompt a moment ago.
                if constrained and isinstance(exc, NotMeasured | ConversationRefused):
                    self.unavailable[mode] = str(exc)
                    continue
                self.unavailable[mode] = f"the run ended: {exc}"
                # Nothing is scored. Every prompt carries the same harness
                # error, which is what keeps these rows out of the score
                # entirely instead of counting as wrong answers.
                ended = f"with the grammar {state}: {exc}"
                return [Generation(i, p, harness_error=ended) for i, p in enumerate(prompts)]
            self.rows[mode] = rows
        return [
            Generation(
                index=row.index,
                prompt=prompts[row.index],
                text=row.text,
                returncode=0,
            )
            for row in self.rows["unconstrained"]
        ]
