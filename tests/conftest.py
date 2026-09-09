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
    UNKNOWN_BACKEND,
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
    # Settable, because a double that can only say `UNKNOWN_BACKEND` says the
    # same word the production code falls back to, and a test
    # asserting the recorded backend then passes against code that never reads
    # the manifest at all.
    backend: str = UNKNOWN_BACKEND
    prompts_seen: list[list[str]] = field(default_factory=list)

    @property
    def model_ref(self) -> str:
        return self.model

    def describe(self) -> dict:
        # Defaults to `UNKNOWN_BACKEND`, not to a third word for it: this double
        # runs on no hardware, which is the state that constant names.
        # `evaluate.device_mismatch` reads this key and suppresses on
        # `UNKNOWN_BACKEND`, so two doubles given *different* real-looking
        # backends would manufacture a hardware difference between two fakes
        # that never touched hardware. The same value on both is harmless.
        #
        # Several subclasses override `describe()` with a literal dict, some of
        # them nested inside test functions where a module-level search misses
        # them. On those, setting `backend=` is silently a no-op.
        return {"engine": "fake", "backend": self.backend}

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


@pytest.fixture(autouse=True)
def _isolated_env_cache(monkeypatch, tmp_path):
    """Point the stage-environment cache at this test's own directory.

    `StageEnv.ready` reads a marker file under `LITETUNE_ENV_DIR`, and code now
    branches on it: `HuggingFaceBackend._ensure_env` runs the device probe only
    against a ready environment. Without this, whether a given test probes
    depends on whether the developer running it happens to have a matching
    environment provisioned in `~/.cache/litetune/envs` -- which is not a
    property of the code under test, and on a machine that has one would send a
    real `python -c "import torch"` into a test that faked everything else.
    Tests that want a ready environment write the marker themselves; this only
    guarantees they start from a machine with none.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "litetune-envs"))


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


@dataclass
class _FakeCuda:
    """`torch.cuda`, reduced to the one method `training_device`/`generation_device`
    call."""

    available: bool

    def is_available(self) -> bool:
        return self.available


@dataclass
class FakeTorch:
    """`torch`, reduced to the one attribute either device function reads."""

    cuda: _FakeCuda


def fake_torch(cuda: bool = False) -> FakeTorch:
    """A `torch` double whose `.cuda.is_available()` answers `cuda`.

    One shape, shared by `test_tune.py` and `test_evaluate.py`, and the
    parameter is the point of it. `training_device` and `generation_device`
    both fall back to `torch.cuda.is_available()`, and a double that can only
    say "no GPU" cannot tell that fallback from a hardcoded `"cpu"` -- every
    mutant of it still answers "cpu" against such a double. Being able to say
    "there is a GPU" is what makes those two lines testable at all.
    """
    return FakeTorch(cuda=_FakeCuda(cuda))


# A conformance assertion, not a runtime one: this is what makes the claim above
# ("a mock would accept a typo in a method name and this will not") true. A fake
# that drifts from the Protocol -- a renamed method, a forgotten
# `decode_enforced` -- fails the type check rather than the measurement.
_CONFORMS: GenerationBackend = FakeBackend()


def mark_provisioned(env) -> Path:
    """Leave a `StageEnv` looking the way a finished provision leaves it.

    Both halves, because `StageEnv.ready` is both: the marker says the install
    finished, the interpreter says the tree it finished into is still there. A
    real cache turned up a directory holding only the marker -- `provision`
    short-circuited on it, `run` reached a python that did not exist, and the
    error named the toolchain instead of the empty directory.

    Nine places across seven test files used to write the marker alone, which is
    the old and wrong definition of ready taught nine times. One helper so that
    the next change to what "provisioned" means has one place to land.
    """
    env.python.parent.mkdir(parents=True, exist_ok=True)
    env.python.touch()
    (env.path / ".litetune-ready").write_text(env.identity, encoding="utf-8")
    return env.path
