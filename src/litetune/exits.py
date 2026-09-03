"""Reading a subprocess return code, which is two different facts in one integer.

`subprocess` reports a **negative** return code when the child was killed by a
signal: it never chose an exit status, because it never got to the end. So `-9`
is not "the program ran and failed with 9", it is "the program was shot".

This is not a hypothetical. A Gemma 4 export returned `-9`, the result was read
as "ran and failed", and the model was struck from the catalogue for a reason
that had nothing to do with the model: `-9` is SIGKILL, and on Linux, with
nothing else sending it, SIGKILL is the out-of-memory killer. The export had hit
a 32 GiB memory ceiling. Run on a larger machine, the same command produced a
specific, actionable error instead -- which is to say the "failure" was a fact
about the machine and the real answer was still unknown.

So a signalled process is `could not check`, everywhere a return code is
interpreted: `export`, `tune`, and the two generation backends in `evaluate`.
`Check.failed` needs an observation about the thing being checked, and a corpse
is not one.

`128+N` is deliberately *not* folded in here. A shell reports a signalled child
as `128+N`, but litetune never runs a subprocess through a shell -- `StageEnv.run`
execs an argv directly -- so a positive `137` seen here came from a program that
chose to exit 137, and reinterpreting it would invent a signal nobody sent.
"""

from __future__ import annotations

import logging
import signal as signal_module
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The one signal worth naming in prose: nothing in litetune sends it, so a
# process that dies of it on a build machine was almost always killed for
# memory.
SIGKILL = 9

OOM_HINT = (
    "nothing in litetune sends SIGKILL, so on Linux this is almost always the out-of-memory "
    "killer: the process asked for more memory than the machine would give it. A Gemma 4 export "
    "died exactly this way at a 32 GiB ceiling and read as a failed conversion; on a larger "
    "machine the same command produced a specific, actionable error instead. Re-run it with more "
    "memory before concluding anything about the model"
)


@dataclass(frozen=True)
class ExitReading:
    """What a return code says, and whether it says anything at all.

    `conclusive` is the field callers act on: False means the process was killed
    and the work it was doing has no result, which is `Outcome.UNCHECKED` rather
    than a verdict.
    """

    returncode: int
    signal: int | None = None
    signal_name: str | None = None

    @property
    def killed(self) -> bool:
        return self.signal is not None

    @property
    def conclusive(self) -> bool:
        """Whether this code is a statement about the work the process was doing."""
        return self.signal is None

    @property
    def ok(self) -> bool:
        return self.conclusive and self.returncode == 0

    def describe(self, subject: str = "the work") -> str:
        """One sentence naming what happened and what it does not establish."""
        if not self.killed:
            return f"exited {self.returncode}"
        name = self.signal_name or f"signal {self.signal}"
        text = (
            f"killed by {name} (return code {self.returncode}): the process never chose an exit "
            f"status, so this is a fact about the machine and not about {subject}"
        )
        if self.signal == SIGKILL:
            text = f"{text}. {OOM_HINT}"
        return text

    def as_dict(self) -> dict[str, Any]:
        return {
            "returncode": self.returncode,
            "killed_by_signal": self.signal,
            "signal_name": self.signal_name,
            "conclusive": self.conclusive,
        }


def read_returncode(returncode: int) -> ExitReading:
    """Split a return code into "the program answered" and "the program was killed"."""
    if returncode >= 0:
        return ExitReading(returncode=returncode)
    number = -returncode
    try:
        name = signal_module.Signals(number).name
    except ValueError:
        # A signal number this platform does not name. The reading still holds --
        # the process was killed -- and inventing a name would be worse than
        # reporting the number.
        logger.warning(
            "return code %d names signal %d, which this platform does not", returncode, number
        )
        name = None
    return ExitReading(returncode=returncode, signal=number, signal_name=name)
