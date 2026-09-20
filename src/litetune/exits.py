"""Reading a subprocess return code, which is two different facts in one integer.

`subprocess` reports a **negative** return code when the child was killed by a
signal: it never chose an exit status, because it never got to the end. So `-9`
is not "the program ran and failed with 9", it is "the program was shot".

This is not a hypothetical. A Gemma 4 export returned `-9`, the result was read
as "ran and failed", and the model was struck from the catalogue for a reason
that had nothing to do with the model: `-9` is SIGKILL, and on that machine it
came from the out-of-memory killer. The export had hit a 32 GiB memory ceiling.
A supervisor cancelling a job sends the same signal, which is why `OOM_HINT`
below names both and says which to check first. Run on a larger machine, the same command produced a
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

**A negative code is POSIX's spelling, and it is the only one Python has.**
"A negative value -N indicates that the child was terminated by signal N
(POSIX only)" -- `subprocess.Popen.returncode`. Windows has no signals to
report: a process that dies of an unhandled exception exits with that
exception's NTSTATUS as its code, and `TerminateProcess` sets whatever code
its caller passed. So the one distinction this module exists to draw is
invisible there unless a code shaped like a terminating NTSTATUS is read as
one, which is what `read_returncode` does below. Without it every Windows
crash is a verdict about a model -- the same mistake as the Gemma 4 export,
one platform over.
"""

from __future__ import annotations

import logging
import signal as signal_module
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The one signal worth naming in prose. litetune does send it: to a stage's
# process group and to the stage's own child from `envs._kill_tree`, and to
# pip from `provision`'s own `subprocess.run`. None of those returns a code to
# this function -- every one ends in a raise -- so a -9 read here still came
# from outside. That is a whole-program invariant with nothing enforcing it,
# which is worth knowing if `StageEnv.run` is ever restructured again.
SIGKILL = 9

OOM_HINT = (
    "every SIGKILL litetune sends ends in a raised exception rather than a return code, so a -9 "
    "arriving here did not come from litetune. On Linux that leaves the out-of-memory killer -- "
    "the process asked for more memory than the machine would give it -- or a supervisor that "
    "cancelled the job: a CI runner, `docker stop` past its grace, a cgroup limit, Slurm. A "
    "Gemma 4 export died of memory exactly this way at a 32 GiB ceiling and read as a failed "
    "conversion; on a larger machine the same command produced a specific, actionable error "
    "instead. Check for a cancellation, then re-run with more memory, before concluding anything "
    "about the model"
)


# The Windows half of "the process was killed": an exit code shaped like the
# NTSTATUS of the exception that ended the process, handed on by
# `GetExitCodeProcess`. Names from MS-ERREF's NTSTATUS list, so a reader can
# search for one.
#
# This is a classification, not a proof, and the difference is worth naming:
# `ExitProcess` and `TerminateProcess` both take any `UINT`, so a program
# *could* exit 0xC0000005 on purpose. Reading such a code as unperformed risks
# calling a deliberate exit "not checked"; reading it as a verdict risks
# calling a crash a fact about the model, which is what this module exists to
# prevent. Only one of those two mistakes can be recovered by re-running.
#
# The error severity range (0xC0000000) decides, and the table adds the
# terminating statuses outside it -- a breakpoint with no debugger attached,
# the CRT's fatal exit -- which are exceptions with a lower severity rather
# than exit statuses.
#
# `_NTSTATUS_NOT_MICROSOFT` is what keeps `exit(-1)` out of all that. Windows
# hands back the exit code as a DWORD, so `exit(-1)` arrives as 0xFFFFFFFF and
# `exit(-100)` as 0xFFFFFF9C -- inside the error range, and a program plainly
# saying it failed. MS-ERREF 2.3 gives the discriminator: bit 29 is C, "set for
# customer-defined values and clear for Microsoft-defined values", and bit 28
# is N, "reserved, MUST be set to 0". A status Windows raised has both clear;
# a two's-complement `exit(-N)` has at least one set for every N below
# 0x30000000, which is every negative exit code a program plausibly writes.
# All eleven entries below have both clear, which is the check on the rule.
#
# What is left, and cannot be settled from an exit code: `exit(-0x30000001)`
# is 0xCFFFFFFF, which has neither bit and reads as a kill, and a program may
# call `ExitProcess(0xC0000005)` outright. The rule is the conservative half
# of an ambiguity the platform does not resolve -- see the paragraph above on
# which way it errs -- and not a proof about what the process did.
#
# Read unconditionally rather than under `os.name == "nt"`. A POSIX return code
# is 0-255 or negative, so nothing on this side can collide with the range --
# and a rule that only runs on the platform nothing tests would be a rule
# nobody could check.
NTSTATUS_ERROR = 0xC0000000
_NTSTATUS_NOT_MICROSOFT = 0x30000000

# Terminating statuses below the error range. A set of its own, not "whatever
# the name table happens to hold": naming a status is a courtesy to a reader,
# classifying one decides whether a run is a verdict, and a future entry added
# for the first reason must not quietly do the second.
_TERMINATING_NTSTATUS = frozenset({0x40000015, 0x80000003})

# Named here as well as in the table: `describe` gives it the memory hint.
STATUS_NO_MEMORY = 0xC0000017

_NTSTATUS = {
    0x40000015: "STATUS_FATAL_APP_EXIT",
    0x80000003: "STATUS_BREAKPOINT",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC000013A: "STATUS_CONTROL_C_EXIT",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}

WINDOWS_MEMORY_HINT = (
    "STATUS_NO_MEMORY is the Windows spelling of the failure this module was written about: the "
    "process asked for memory the system would not give it. It is not the Linux OOM killer -- "
    "nothing chose this process -- but the reading is the same, and so is what to do: re-run it "
    "with more memory, or a smaller model, before concluding anything about the model."
)

WINDOWS_KILL_HINT = (
    "Windows reports no signal for a process that was terminated: an unhandled exception surfaces "
    "as its NTSTATUS, and `ExitProcess` and `TerminateProcess` take any code their caller passes, "
    "so an exit code alone cannot tell an exception from a program that chose this number. A "
    "status shaped like an exception is read as the first: re-running recovers a run wrongly "
    "called unperformed, and nothing recovers a crash wrongly called a verdict."
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
    # The NTSTATUS that ended a Windows process, named where this module knows
    # the name and spelled in hex where it does not. Separate from `signal`
    # because it is not one: writing 0xC0000005 into `signal` would have
    # `describe` say "killed by signal 3221225477" about a machine that has no
    # signals, and would put that sentence in every manifest.
    status: str | None = None

    @property
    def killed(self) -> bool:
        return self.signal is not None or self.status is not None

    @property
    def conclusive(self) -> bool:
        """Whether this code is a statement about the work the process was doing."""
        return not self.killed

    @property
    def ok(self) -> bool:
        return self.conclusive and self.returncode == 0

    def describe(self, subject: str = "the work") -> str:
        """One sentence naming what happened and what it does not establish."""
        if not self.killed:
            return f"exited {self.returncode}"
        if self.status is not None:
            text = (
                f"terminated by {self.status} (return code {self.returncode}): read as "
                f"unperformed rather than as a verdict about {subject}. {WINDOWS_KILL_HINT}"
            )
            # The same event as a -9, spelled the way Windows spells it. Without
            # this the memory hint would reach a Linux reader and not a Windows
            # one, for the failure the module was written about.
            if self.returncode == STATUS_NO_MEMORY:
                text = f"{text} {WINDOWS_MEMORY_HINT}"
            return text
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
            "terminated_by_status": self.status,
            "conclusive": self.conclusive,
        }


def read_returncode(returncode: int) -> ExitReading:
    """Split a return code into "the program answered" and "the program was killed"."""
    terminating = returncode >= NTSTATUS_ERROR or returncode in _TERMINATING_NTSTATUS
    if terminating and not returncode & _NTSTATUS_NOT_MICROSOFT:
        status = _NTSTATUS.get(returncode) or f"NTSTATUS 0x{returncode:08X}"
        return ExitReading(returncode=returncode, status=status)
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
