"""Three-valued check results.

A check answers `passed`, `failed`, or `could not check` — never the third as
either of the first two.

This is the load-bearing rule of the whole tool, and it is not defensive
programming. During the measurement work behind litetune, ten separate checks
reported a confident result when they had not in fact run. Among them: a binary
missing from the platform, an interactive prompt with no terminal attached, a
download refused for a licence that had not been accepted, a nightly toolchain
that broke overnight, a process killed by the out-of-memory killer and read as a
rejection, a report from a previous run read as the current one. Every one of
them looked exactly like a real answer, and one of them very nearly shipped as a
finding.

If a customer's release gate can only say yes or no, it will sometimes say no
because it could not start — and, more expensively, it will say yes for the same
kind of reason. So `Outcome.UNCHECKED` exists, `Check.failed()` requires you to
state what was observed, and `guard()` converts any exception into UNCHECKED
rather than letting it read as a verdict.
"""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Outcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNCHECKED = "could_not_check"


@dataclass(frozen=True)
class Check:
    """One check and what it observed.

    `detail` is mandatory in every constructor: a bare verdict with no observed
    value is what makes a wrong verdict impossible to audit later.
    """

    name: str
    outcome: Outcome
    detail: str
    observed: Any = None

    @classmethod
    def passed(cls, name: str, detail: str, observed: Any = None) -> Check:
        return cls(name, Outcome.PASSED, detail, observed)

    @classmethod
    def failed(cls, name: str, detail: str, observed: Any = None) -> Check:
        return cls(name, Outcome.FAILED, detail, observed)

    @classmethod
    def unchecked(cls, name: str, detail: str, observed: Any = None) -> Check:
        """The check did not run. This is not a failure of the thing checked."""
        return cls(name, Outcome.UNCHECKED, detail, observed)

    @property
    def conclusive(self) -> bool:
        return self.outcome is not Outcome.UNCHECKED

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "observed": self.observed,
        }


@dataclass
class CheckSet:
    """An ordered set of checks with an honest aggregate.

    The aggregate deliberately does not collapse UNCHECKED into FAILED. A set
    containing an unchecked item is `could not check` as a whole, because the
    caller was promised an answer about the model and did not get one.
    """

    name: str
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    @property
    def outcome(self) -> Outcome:
        if any(c.outcome is Outcome.UNCHECKED for c in self.checks):
            return Outcome.UNCHECKED
        if any(c.outcome is Outcome.FAILED for c in self.checks):
            return Outcome.FAILED
        if not self.checks:
            # An empty set has established nothing. Reporting PASSED here is the
            # exact mistake this module exists to prevent.
            return Outcome.UNCHECKED
        return Outcome.PASSED

    @property
    def first_failure(self) -> Check | None:
        return next((c for c in self.checks if c.outcome is Outcome.FAILED), None)

    @property
    def first_unchecked(self) -> Check | None:
        return next((c for c in self.checks if c.outcome is Outcome.UNCHECKED), None)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "checks": [c.as_dict() for c in self.checks],
        }


@contextmanager
def guard(name: str, detail: str = "") -> Iterator[list[Check]]:
    """Run a check body; turn any escaping exception into UNCHECKED.

    Usage:

        with guard("model loads") as out:
            out.append(Check.passed("model loads", f"{n} weights", n))

    An exception here means the check could not be performed. It says nothing
    about the model, so it must not be recorded as though it did.
    """
    sink: list[Check] = []
    try:
        yield sink
    except Exception as exc:  # noqa: BLE001 - converting to a result is the point
        sink.clear()
        sink.append(
            Check.unchecked(
                name,
                detail or f"check raised {type(exc).__name__}: {exc}",
                observed={"traceback": traceback.format_exc(limit=6)},
            )
        )
