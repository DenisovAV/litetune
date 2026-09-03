"""Fakes for the composition tests: a spec builder and a stage that does no work.

`FakeStage` satisfies `runner.Stage` structurally and inherits nothing, which is
why that interface is a Protocol. Its default body writes one artifact whose
*content* is derived from the spec slice and the input hashes it was given --
which is what makes the cache tests real: a change upstream shows up as
different bytes downstream, exactly as a retrained checkpoint would.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from litetune.manifest import RunStatus
from litetune.runner import Artifact, StageContext, StageResult
from litetune.spec import Spec

# Two distinct, well-formed digests. Content hashes are what the cache is keyed
# on, so the tests need to be able to change one without changing anything else.
DATA_A = "a" * 64
DATA_B = "b" * 64
HELDOUT_A = "c" * 64
HELDOUT_B = "d" * 64

PINNED_REVISION = "0123456789abcdef0123456789abcdef01234567"


def spec_mapping(**overrides: Any) -> dict[str, Any]:
    """A valid spec, with per-section overrides merged in.

    `None` removes: as a section it removes the whole section, and as a value
    inside one it removes that field. Both are how a test asks for the "missing
    required field" case without hand-writing the whole document.
    """
    base: dict[str, Any] = {
        "base_model": {"id": "google/functiongemma-270m-it", "revision": PINNED_REVISION},
        "dataset": {"uri": "./data/mobile-actions.jsonl", "content_sha256": DATA_A},
        "train": {"mode": "full", "epochs": 1.0},
        "export": {
            "recipes": ["dynamic_wi8_afp32", "weight_only_wi8_afp32"],
            "context_length": 1280,
        },
        "eval": {
            "heldout_uri": "./data/heldout.jsonl",
            "heldout_content_sha256": HELDOUT_A,
            "limit": 640,
        },
        "gates": {"max_conversion_cost": 0.05},
        "toolchain": {"train": "default", "export": "default", "runtime": "default"},
    }
    for name, patch in overrides.items():
        if patch is None:
            base.pop(name, None)
        elif isinstance(patch, dict) and isinstance(base.get(name), dict):
            merged = base[name] | patch
            base[name] = {k: v for k, v in merged.items() if v is not None}
        else:
            base[name] = patch
    return base


def make_spec(**overrides: Any) -> Spec:
    return Spec.from_mapping(spec_mapping(**overrides), source="test-spec.yaml")


@dataclass
class FakeStage:
    """A stage that records every call and writes one deterministic artifact."""

    name: str
    spec_sections: tuple[str, ...] = ()
    input_names: tuple[str, ...] = ()
    env_names: tuple[str, ...] = ()
    body: Callable[[StageContext], StageResult] | None = None
    calls: list[StageContext] = field(default_factory=list)

    @property
    def artifact_name(self) -> str:
        return f"{self.name}.json"

    def run(self, ctx: StageContext) -> StageResult:
        self.calls.append(ctx)
        if self.body is not None:
            return self.body(ctx)
        payload = json.dumps(
            {
                "stage": self.name,
                "slice": ctx.spec_slice(self.spec_sections) if ctx.spec else None,
                "inputs": [[i.name, i.content_hash] for i in ctx.inputs],
            },
            sort_keys=True,
        )
        out = ctx.workspace / self.artifact_name
        out.write_text(payload, encoding="utf-8")
        return StageResult(
            status=RunStatus.PASSED,
            detail=f"{self.name} wrote {self.artifact_name}",
            artifacts=(Artifact(name=self.artifact_name, path=out),),
            output=f"{self.name}: fake toolchain output\n",
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def pipeline() -> tuple[FakeStage, FakeStage, FakeStage]:
    """train -> export -> measure, wired the way the real pipeline is.

    `export` does not name the dataset: it depends on the checkpoint it is
    given, and the dataset reaches it through that checkpoint's content hash.
    """
    train = FakeStage(
        name="train",
        spec_sections=("base_model", "dataset", "train"),
        env_names=("train",),
    )
    export = FakeStage(
        name="export",
        spec_sections=("base_model", "export"),
        input_names=("train.json",),
        env_names=("export",),
    )
    measure = FakeStage(
        name="measure",
        spec_sections=("base_model", "eval"),
        input_names=("export.json",),
        env_names=("runtime", "train"),
    )
    return train, export, measure


def collect_events() -> tuple[Any, list]:
    """An EventStream that prints nothing and the list it records into."""
    from litetune.events import EventStream

    seen: list = []
    stream = EventStream(echo_json=False)
    stream.subscribe(seen.append)
    return stream, seen
