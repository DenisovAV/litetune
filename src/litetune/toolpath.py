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
writes a call the runtime cannot read is a model that fails for an application,
and the refusal comes back as the runtime's own message rather than as a parse
of a parse. `metrics.parse_call` still exists and is still used -- for the
transformers reference, which produces text and nothing else.

**Constrained decoding hides the defect it is there to prevent.** It is on
whenever tools are passed, and with it on, a model that writes
`call{name{...}}` -- the shape the colab-tuned demo model writes -- returns
clean calls anyway. With it off the same model is refused with
`Failed to parse tool calls from code block`. One number from each mode is the
only evidence that the model learned the format rather than being forced into
it, so both are run and neither is reported alone.

**Automatic tool calling is off.** Left on, the runtime would execute the
declared tools and loop until prose. Measuring a model must not run somebody's
functions, and a declared tool here raises if it is ever called.

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
from litetune.events import EventStream
from litetune.exits import read_returncode

logger = logging.getLogger(__name__)

TOOL_PATH_TIMEOUT_S = 3600


class ToolPathError(RuntimeError):
    """The tool path could not be measured at all. Never a statement about the model."""


_TOOL_PATH_SCRIPT = r'''
"""One structured call per prompt, through the runtime's own tool path."""
import json
import sys
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


def first_call(reply):
    """The first `tool_calls` entry, flattened to name and arguments.

    The runtime returns the OpenAI shape, `{"type": "function", "function":
    {...}}`. One call, because every target in this project is one call and
    scoring a list against a single target would be inventing a rule nobody
    measured.
    """
    calls = reply.get("tool_calls") or []
    if not calls:
        return None
    entry = calls[0]
    function = entry.get("function", entry) if isinstance(entry, dict) else {}
    return {"name": function.get("name"), "arguments": function.get("arguments") or {}}


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    import litert_lm

    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except AttributeError:
        pass

    class Declared(litert_lm.Tool):
        """A declaration the runtime renders and must never run.

        `create_conversation` takes `Tool` instances, never raw JSON: it asks
        each for `get_tool_description()`. Handing the file's entries back
        unchanged keeps the runtime's own refusal as the refusal.
        """

        def __init__(self, description):
            self._description = description

        def get_tool_description(self):
            return self._description

        def execute(self, param):
            raise AssertionError(
                "the runtime executed a declared tool while measuring; automatic tool "
                "calling should be off"
            )

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
                try:
                    reply = conversation.send_message(prompt)
                except Exception as exc:  # noqa: BLE001
                    # The runtime's parser refusing this generation is a result
                    # about this row, recorded with the runtime's own words.
                    rows.append(
                        {
                            "index": index,
                            "call": None,
                            "text": "",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
            rows.append(
                {
                    "index": index,
                    "call": first_call(reply),
                    "text": text_of(reply),
                    "error": None,
                }
            )
    Path(spec["out"]).write_text(json.dumps(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass(frozen=True)
class ToolPathRow:
    """One prompt's answer through the tool path.

    `call is None` with `error is None` is a real result -- the model answered
    without calling anything. `error` is the runtime's refusal for this row,
    which is counted separately and never scored as a wrong answer: a model
    whose output the parser rejected has failed differently from one that called
    the wrong tool, and averaging the two together hides which.
    """

    index: int
    call: dict[str, Any] | None = None
    text: str = ""
    error: str | None = None

    @property
    def refused(self) -> bool:
        return self.error is not None

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "call": self.call, "text": self.text, "error": self.error}


@dataclass
class ToolPathProbe:
    """Runs the script above in the runtime's environment, once per decoding mode."""

    model: Path
    declarations: list
    max_tokens: int = 128
    env: envs.StageEnv = envs.RUNTIME
    auto_provision: bool = True
    timeout_s: int = TOOL_PATH_TIMEOUT_S
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
            rows = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != len(prompts):
            raise ToolPathError(
                f"the tool-path script wrote {len(rows) if isinstance(rows, list) else 'no'} rows "
                f"for {len(prompts)} prompts"
            )
        return [ToolPathRow(**row) for row in rows]
