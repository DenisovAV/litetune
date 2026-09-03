"""Fakes for the verify tests.

No network, no accelerator, no model load, no subprocess. `FakeBackend`
satisfies `evaluate.GenerationBackend` structurally -- it inherits nothing,
which is the reason that interface is a Protocol: a mock would accept a typo in
a method name and this will not.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from litetune.evaluate import (
    GREEDY,
    DecodeConfig,
    Generation,
    GenerationBackend,
    PromptMode,
)


def call_text(name: str, **args: str) -> str:
    """Render FunctionGemma's wire format, the way a model would emit it."""
    body = ",".join(f"{k}:<escape>{v}<escape>" for k, v in args.items())
    return f"call:{name}{{{body}}}"


@dataclass
class FakeBackend:
    """Canned generations. `texts` is either one per prompt or a single repeat."""

    model: str = "fake-model"
    texts: Sequence[str] = ()
    returncode: int = 0
    harness_error: str | None = None
    prompt_mode: PromptMode = PromptMode.PRERENDERED
    decode: DecodeConfig = GREEDY
    # Part of the backend contract, so the double states it. It used to be
    # absent, and a `.get(..., True)` in the production code recorded every fake
    # measurement as enforcing decode parameters it never received.
    decode_enforced: bool = True
    name: str = "fake"
    prompts_seen: list[list[str]] = field(default_factory=list)

    @property
    def model_ref(self) -> str:
        return self.model

    def describe(self) -> dict:
        return {"engine": "fake", "backend": "none"}

    def generate(self, prompts: Sequence[str], events=None) -> list[Generation]:
        self.prompts_seen.append(list(prompts))
        if len(self.texts) not in (1, len(prompts)):
            raise ValueError(f"fake has {len(self.texts)} texts for {len(prompts)} prompts")
        return [
            Generation(
                index=i,
                prompt=prompt,
                text=self.texts[i] if len(self.texts) == len(prompts) else self.texts[0],
                returncode=None if self.harness_error else self.returncode,
                harness_error=self.harness_error,
            )
            for i, prompt in enumerate(prompts)
        ]


@pytest.fixture
def write_split(tmp_path: Path):
    """Write held-out JSONL and return its path."""

    def _write(rows: Sequence[dict], name: str = "heldout.jsonl") -> Path:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    return _write


def labelled_rows(n: int) -> list[dict]:
    """`n` single-call examples of the shape the README's dataset uses."""
    return [
        {
            "prompt": f"set the background to colour {i}",
            "target": {"name": "change_background_color", "args": {"color": f"c{i}"}},
        }
        for i in range(n)
    ]


def correct_texts(rows: Sequence[dict]) -> list[str]:
    return [call_text(r["target"]["name"], **r["target"]["args"]) for r in rows]


# A conformance assertion, not a runtime one: this is what makes the claim above
# ("a mock would accept a typo in a method name and this will not") true. A fake
# that drifts from the Protocol -- a renamed method, a forgotten
# `decode_enforced` -- fails the type check rather than the measurement.
_CONFORMS: GenerationBackend = FakeBackend()
