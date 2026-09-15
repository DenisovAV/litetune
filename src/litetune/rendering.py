"""Whether the runtime and the reference put the same prompt tokens in front of the model.

In `runtime_rendered` each side of a measurement renders the prompt itself: the
reference through its tokenizer's chat template in transformers (Jinja2), the
converted model through the template inside the `.litertlm`, rendered by the
runtime (MiniJinja). A difference there is not a conversion cost, and nothing
downstream could tell it from one. Equal template text proves nothing either:
two engines can render the same template differently, and a tokenizer can add
tokens neither template shows.

So the check compares token ids. For every prompt, a script in `envs.RUNTIME`
renders it with `Conversation.render_message_to_string` -- in LiteRT-LM v0.16.1
that returns `GetSingleTurnText`, the function `SendMessage` renders with --
and tokenizes the result the way prefill does. A script in the reference's
environment produces the ids the reference generates from, through the same
`evaluate.REFERENCE_PROMPT_SOURCE` the generation script uses. The lists must be
equal. On the first few prompts the runtime also sends the message, and the
prefill count it reports must equal the reference's id count: that catches a
wrong mirror of the prefill step, and tokens the runtime adds outside the
rendered text.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from litetune import envs
from litetune.checks import Check
from litetune.evaluate import REFERENCE_PROMPT_SOURCE
from litetune.events import EventStream
from litetune.exits import read_returncode

RENDERING_CHECK = "runtime and reference render the same prompt tokens"

# How many prompts are also sent, to compare the runtime's own prefill count.
# A parameter, not a derived number: every prompt's ids are compared regardless,
# and a send costs a prefill where a render costs nothing.
DEFAULT_PREFILL_SAMPLE = 8

# How much of each rendering a mismatch quotes, from the end. The generation
# prompt is where a template difference shows first.
_TAIL = 120

_RUNTIME_SCRIPT = r'''
"""What the runtime puts in front of the model for each prompt. Writes JSON."""
import json
import sys
from pathlib import Path


def prefill_ids(engine, text):
    """`text` tokenized the way prefill tokenizes it.

    `Engine.tokenize` calls the tokenizer directly. Prefill first strips a
    leading BOS string and inserts the BOS id in its place, because the
    tokenizer does not read that string as the token (LiteRT-LM
    runtime/core/session_utils.cc, StringToProcessedInputText, v0.16.1). This
    mirrors that step, and a mirror can be wrong: the prefill count the parent
    compares is what would say so.
    """
    bos = engine.bos_token_id
    if bos is not None and bos >= 0:
        bos_text = engine.detokenize([bos])
        if bos_text and text.startswith(bos_text):
            return [bos] + list(engine.tokenize(text[len(bos_text) :]))
    return list(engine.tokenize(text))


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    import litert_lm

    try:
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
    except AttributeError:
        pass
    greedy = litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=1.0, seed=0)
    rows = []
    with litert_lm.Engine(
        spec["model"], backend=litert_lm.Backend.CPU(), enable_benchmark=True
    ) as engine:
        # Rendering reads the conversation's history and never adds to it, so
        # one conversation renders every prompt as the first turn it would be.
        with engine.create_conversation(sampler_config=greedy, max_output_tokens=1) as renderer:
            for index, prompt in enumerate(spec["prompts"]):
                rendered = renderer.render_message_to_string(prompt)
                rows.append(
                    {
                        "index": index,
                        "rendered": rendered,
                        "ids": prefill_ids(engine, rendered),
                        "prefill_tokens": None,
                    }
                )
        for row in rows[: spec["prefill_sample"]]:
            with engine.create_conversation(
                sampler_config=greedy, max_output_tokens=1
            ) as conversation:
                conversation.send_message(spec["prompts"][row["index"]])
                row["prefill_tokens"] = conversation.get_benchmark_info().last_prefill_token_count
    Path(spec["out"]).write_text(json.dumps(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_REFERENCE_SCRIPT = (
    r'''
"""The ids the reference generates from, for each prompt. Writes JSON."""
import json
import sys
from pathlib import Path
'''
    + REFERENCE_PROMPT_SOURCE
    + r"""

def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["model"])
    rows = []
    for index, prompt in enumerate(spec["prompts"]):
        text, add_special = reference_prompt(tok, prompt, True)
        ids = tok(text, add_special_tokens=add_special)["input_ids"]
        rows.append({"index": index, "rendered": text, "ids": [int(i) for i in ids]})
    Path(spec["out"]).write_text(json.dumps(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
)


class RenderingProbeError(RuntimeError):
    """A rendering script could not answer. A fact about the harness, not the model."""


@dataclass(frozen=True)
class RenderingMismatch:
    """One prompt the two sides did not put in front of the model identically."""

    index: int
    prompt: str
    # `ids`: the rendered id lists differ. `prefill`: they agree, and the count
    # the runtime prefilled when the prompt was sent does not.
    kind: str
    runtime_tokens: int
    reference_tokens: int
    first_difference: int | None
    prefill_tokens: int | None
    runtime_tail: str
    reference_tail: str

    def describe(self) -> str:
        if self.kind == "prefill":
            return (
                f"prompt {self.index}: the runtime prefilled {self.prefill_tokens} tokens when it "
                f"was sent, where its rendering and the reference both have "
                f"{self.reference_tokens}"
            )
        return (
            f"prompt {self.index}: the runtime renders {self.runtime_tokens} tokens and the "
            f"reference {self.reference_tokens}, first differing at position "
            f"{self.first_difference}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "prompt": self.prompt[:_TAIL],
            "kind": self.kind,
            "runtime_tokens": self.runtime_tokens,
            "reference_tokens": self.reference_tokens,
            "first_difference": self.first_difference,
            "prefill_tokens": self.prefill_tokens,
            "runtime_rendered_tail": self.runtime_tail,
            "reference_rendered_tail": self.reference_tail,
        }


@dataclass(frozen=True)
class RenderingComparison:
    """What the two renderings were found to be, over every prompt."""

    compared: int
    prefill: tuple[Mapping[str, int], ...] = ()
    mismatches: tuple[RenderingMismatch, ...] = field(default_factory=tuple)

    @property
    def agrees(self) -> bool:
        return not self.mismatches

    def check(self) -> Check:
        sampled = len(self.prefill)
        if self.agrees:
            return Check.passed(
                RENDERING_CHECK,
                f"identical token ids for all {self.compared} prompts, and the runtime's prefill "
                f"count equal to the reference's on the {sampled} sent",
                observed=self.as_dict(),
            )
        first = self.mismatches[0]
        return Check.failed(
            RENDERING_CHECK,
            f"{len(self.mismatches)} of {self.compared} prompts differ; {first.describe()}. The "
            "two sides would score different prompts, so no difference between them could be "
            "attributed to conversion",
            observed=self.as_dict(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": True,
            "prompts_compared": self.compared,
            "prefill_sampled": [dict(row) for row in self.prefill],
            "mismatches": len(self.mismatches),
            # The first few, which is what a reader needs to find the cause; the
            # count above says how many there were.
            "first_mismatches": [m.as_dict() for m in self.mismatches[:5]],
        }


def _first_difference(a: Sequence[int], b: Sequence[int]) -> int | None:
    for position, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return position
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_renderings(
    prompts: Sequence[str],
    runtime_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
) -> RenderingComparison:
    """Compare what the two scripts returned. Raises if either did not cover every prompt."""
    runtime = {int(row["index"]): row for row in runtime_rows}
    reference = {int(row["index"]): row for row in reference_rows}
    expected = set(range(len(prompts)))
    for side, rows in (("runtime", runtime), ("reference", reference)):
        if set(rows) != expected:
            raise RenderingProbeError(
                f"the {side} rendering script returned {len(rows)} prompts of {len(prompts)}"
            )

    mismatches: list[RenderingMismatch] = []
    prefill: list[Mapping[str, int]] = []
    for index, prompt in enumerate(prompts):
        ours, theirs = runtime[index], reference[index]
        runtime_ids = [int(i) for i in ours["ids"]]
        reference_ids = [int(i) for i in theirs["ids"]]
        sent = ours.get("prefill_tokens")
        if sent is not None:
            prefill.append(
                {
                    "index": index,
                    "prefill_tokens": int(sent),
                    "reference_tokens": len(reference_ids),
                }
            )
        kind = None
        if runtime_ids != reference_ids:
            kind = "ids"
        elif sent is not None and int(sent) != len(reference_ids):
            kind = "prefill"
        if kind is not None:
            mismatches.append(
                RenderingMismatch(
                    index=index,
                    prompt=prompt,
                    kind=kind,
                    runtime_tokens=len(runtime_ids),
                    reference_tokens=len(reference_ids),
                    first_difference=_first_difference(runtime_ids, reference_ids),
                    prefill_tokens=None if sent is None else int(sent),
                    runtime_tail=str(ours.get("rendered", ""))[-_TAIL:],
                    reference_tail=str(theirs.get("rendered", ""))[-_TAIL:],
                )
            )
    return RenderingComparison(
        compared=len(prompts), prefill=tuple(prefill), mismatches=tuple(mismatches)
    )


class RenderingObserver(Protocol):
    """Anything that can say how both sides render a set of prompts."""

    def observe(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> RenderingComparison: ...


@dataclass
class RenderingProbe:
    """The real observer: one script in the runtime's environment, one in the reference's."""

    model: Path
    reference: str
    prefill_sample: int = DEFAULT_PREFILL_SAMPLE
    runtime_env: envs.StageEnv = envs.RUNTIME
    reference_env: envs.StageEnv = envs.TRAIN
    auto_provision: bool = True
    timeout_s: int = 1800

    def observe(
        self, prompts: Sequence[str], events: EventStream | None = None
    ) -> RenderingComparison:
        runtime_rows = self._run(
            self.runtime_env,
            _RUNTIME_SCRIPT,
            {
                "model": str(self.model),
                "prompts": list(prompts),
                "prefill_sample": self.prefill_sample,
            },
            events,
        )
        reference_rows = self._run(
            self.reference_env,
            _REFERENCE_SCRIPT,
            {"model": self.reference, "prompts": list(prompts)},
            events,
        )
        return compare_renderings(prompts, runtime_rows, reference_rows)

    def _run(
        self,
        env: envs.StageEnv,
        source: str,
        spec: dict[str, Any],
        events: EventStream | None,
    ) -> list[dict[str, Any]]:
        if self.auto_provision:
            env.provision(events=events)
        with tempfile.TemporaryDirectory(prefix="litetune-render-") as tmp:
            work = Path(tmp)
            script = work / "render.py"
            script.write_text(source, encoding="utf-8")
            out = work / "rows.json"
            spec_path = work / "spec.json"
            spec_path.write_text(json.dumps({**spec, "out": str(out)}), encoding="utf-8")
            try:
                proc = env.run(["python", str(script), str(spec_path)], timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                raise RenderingProbeError(
                    f"the rendering script in {env.name!r} gave no result after {self.timeout_s}s"
                ) from None
            reading = read_returncode(proc.returncode)
            if proc.returncode != 0 or not out.is_file():
                raise RenderingProbeError(
                    f"the rendering script in {env.name!r} {reading.describe('the model')}: "
                    f"{(proc.stderr or '').strip()[-300:] or 'no stderr'}"
                )
            rows = json.loads(out.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise RenderingProbeError(f"the rendering script in {env.name!r} wrote no row list")
        return rows
