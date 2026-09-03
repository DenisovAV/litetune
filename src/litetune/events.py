"""Structured progress events.

Stages emit JSONL to stderr (see `EventStream`), keeping stdout free for
the result a caller pipes; nothing prints progress directly. The terminal
renderer and the hosted service's status feed are both consumers of the same
stream, which is what lets the identical pipeline code run locally and behind an
API without a fork.

The practical reason this is worth the indirection: during the measurement work,
progress lived in ad-hoc `print` calls and shell `echo`s, and more than once the
message was emitted from a place that had not observed the thing it announced
("image ready" printed after a wait loop that had merely run out of attempts).
An event carries its observed payload, so a claim without data is visible as
such.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class Event:
    kind: str
    stage: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        d = {"ts": round(self.ts, 3), "kind": self.kind}
        if self.stage:
            d["stage"] = self.stage
        d.update(self.data)
        return d


class EventStream:
    """Emits events as JSONL and fans them out to listeners.

    `stream` defaults to stderr so that a command's *result* can own stdout.
    Machine consumers redirect it; the terminal renderer subscribes instead.
    """

    def __init__(self, stream: TextIO | None = None, echo_json: bool = True):
        self._stream = stream if stream is not None else sys.stderr
        self._echo_json = echo_json
        self._listeners: list[Callable[[Event], None]] = []
        self._stage: str | None = None

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._listeners.append(fn)

    def emit(self, kind: str, **data: Any) -> Event:
        event = Event(kind=kind, stage=self._stage, data=data)
        if self._echo_json:
            print(json.dumps(event.as_dict(), default=str), file=self._stream, flush=True)
        for fn in self._listeners:
            fn(event)
        return event

    # -- conveniences the stages actually call ------------------------------

    def stage_started(self, stage: str, **data: Any) -> None:
        self._stage = stage
        self.emit("stage_started", **data)

    def stage_finished(self, status: str, **data: Any) -> None:
        self.emit("stage_finished", status=status, **data)
        self._stage = None

    def metric(self, name: str, value: Any, **data: Any) -> None:
        self.emit("metric", name=name, value=value, **data)

    def artifact(self, path: str, **data: Any) -> None:
        self.emit("artifact_written", path=path, **data)

    def check(self, check) -> None:  # litetune.checks.Check
        self.emit("check", **check.as_dict())

    def note(self, message: str, **data: Any) -> None:
        """Human-facing commentary. Never used to assert a result."""
        self.emit("note", message=message, **data)


class TerminalRenderer:
    """Renders an event stream for a person. Subscribe, don't print."""

    SYMBOL = {"passed": "✓", "failed": "✗", "could_not_check": "?"}

    def __init__(self, out: TextIO | None = None):
        self._out = out if out is not None else sys.stderr

    def __call__(self, event: Event) -> None:
        d = event.data
        if event.kind == "stage_started":
            print(f"→ {event.stage}", file=self._out, flush=True)
        elif event.kind == "stage_finished":
            print(f"  {event.stage}: {d.get('status')}", file=self._out, flush=True)
        elif event.kind == "metric":
            value = d.get("value")
            ci = d.get("ci95")
            shown = f"{value:.4f}" if isinstance(value, float) else value
            suffix = f" ±{ci:.4f}" if isinstance(ci, float) else ""
            print(f"  {d.get('name')}: {shown}{suffix}", file=self._out, flush=True)
        elif event.kind == "check":
            mark = self.SYMBOL.get(d.get("outcome", ""), "?")
            print(f"  {mark} {d.get('name')} — {d.get('detail')}", file=self._out, flush=True)
        elif event.kind == "note":
            print(f"  {d.get('message')}", file=self._out, flush=True)
