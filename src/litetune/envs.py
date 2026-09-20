"""Per-stage Python environments.

Training and conversion cannot share an interpreter. `torch` + `transformers`
against `litert-torch` + `numpy<2.1` conflict irreconcilably, which is why
Google's own FunctionGemma notebooks tell the reader to restart the runtime
between steps, and why one of them opens with "uses ONLY mediapipe (no
ai-edge-torch) to avoid conflicts".

So the user installs one package and litetune provisions the rest: one venv per
stage, created on first use and cached. The same separation the cloud pipeline
achieves with two container images, on a laptop.

Every requirement is pinned. This is not caution for its own sake — the same
unpinned install command produced a working export on 2026-08-26 and
`AttributeError: pad_token` on 2026-08-30. Pinning the environment's *identity*
is useless if its *definition* resolves differently on different days, so
`StageEnv` refuses to build from an unpinned requirement.
"""

from __future__ import annotations

import codecs
import contextlib
import hashlib
import json
import logging
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
import venv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO, Any, cast

from litetune.exits import read_returncode


class UnpinnedRequirement(ValueError):
    """Raised when an environment definition would float."""


class StageInterrupted(BaseException):
    """A termination signal arrived while a stage subprocess was running.

    `BaseException` rather than `Exception` so it unwinds like the interrupt it
    is instead of being caught as a stage failure -- but not `KeyboardInterrupt`,
    which already means "a person pressed Ctrl-C" to every reader of a log and
    to CPython's own exit status. A run ended by a hangup or by `kill` must not
    be reported as one the operator abandoned.
    """

    def __init__(self, signum: int):
        self.signum = signum
        # `set(...)` rather than `signum in signal.Signals`: the direct form is
        # a TypeError before 3.12, and this package supports 3.10.
        name = signal.Signals(signum).name if signum in set(signal.Signals) else str(signum)
        super().__init__(f"{name} arrived while a stage subprocess was running")


def env_cache_root() -> Path:
    """Where provisioned environments live. Public because `litetune env` shows it.

    A path a user is told to inspect or clear should have a name in the API, not
    only a shape in a docstring.
    """
    return _cache_root()


def _cache_root() -> Path:
    override = os.environ.get("LITETUNE_ENV_DIR")
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "litetune" / "envs"


# A requirement is pinned if it fixes an exact version. `>=` is not a pin: it
# admits tomorrow's release, which is precisely the failure mode above.
def _is_pinned(requirement: str) -> bool:
    return "==" in requirement or requirement.startswith(("http://", "https://", "file://"))


# A pip install of torch is minutes, not seconds; a stalled one is forever.
logger = logging.getLogger(__name__)

PROVISION_TIMEOUT_S = 1800

# Variables that let the host reach inside a stage environment and outrank its
# pins. `PYTHONPATH` is prepended ahead of the venv's own `site-packages`, so a
# directory holding `numpy.py` wins over the `numpy==2.0.2` this module went to
# such lengths to fix: measured 2026-09-09, the same interpreter reporting
# `2.0.2` with the variable unset and a planted module with it set. That is not
# a crash. It is a run recorded under a provenance it did not have -- the
# manifest still names the pin -- and the cache then replays it under the
# pinned identity. `PIP_TARGET` and `PIP_PREFIX` are the same shape one step
# earlier: they redirect the install out of the venv while `.litetune-ready` is
# still written over it.
#
# The line is drawn at *silent structural corruption*, not at everything that
# could change a result, and two cases decided where it falls.
#
# `PIP_INDEX_URL` and its relatives are not here, though they are the sharper
# hole on paper: measured 2026-09-09, pip reports `:env:.index-url` and fetches
# `pyyaml==6.0.2` from whatever host is named, so an environment whose identity
# hashes `numpy==2.0.2` could hold another index's idea of that name. They stay
# because a corporate mirror or an air-gapped installer is exactly how they are
# normally set, and dropping them would break machines that install correctly
# today in order to close a hole that requires an attacker already inside the
# user's environment. `PIP_CONFIG_FILE` settles it: unsetting it does not close
# the channel, it falls back to `/etc/pip.conf` and the user's own config,
# which can set `index-url` just as well. What actually answers this is the
# record, not the strip -- `export.resolve_toolchain` reads the resolved
# closure back with `pip freeze --all`, so what was installed is stated rather
# than assumed.
#
# `PIP_TARGET`, `PIP_PREFIX` and `PIP_ROOT` are a different case and do belong
# here: they do not change what is fetched, they put it somewhere else while
# pip still exits 0 and `.litetune-ready` is still written over a venv that
# now holds nothing. That is the failure this file exists to make impossible.
#
# `PYTHONEXECUTABLE` is here for the workers rather than the stage: it
# overrides `sys.executable`, which `multiprocessing`'s spawn start method --
# the default on macOS, and what a torch `DataLoader` uses -- relaunches
# workers with. The stage would hold its pins while the processes doing the
# work booted off another interpreter.
#
# `LD_PRELOAD` and `DYLD_INSERT_LIBRARIES` inject code under a correctly
# pinned package, which is the same false provenance one layer down. They are
# the least comfortable entries here, because preloading `libgomp` or an
# allocator is a documented torch workaround and stripping one turns a working
# machine into a crashing one. Kept anyway, and the reason is the direction of
# the failure: a dropped preload fails loudly and the warning below names it,
# while an honoured one produces a number nothing can tell apart from a real
# one. This file exists for the second kind.
# `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` are deliberately *not* here: they
# are the same shape, but unlike the rest they have ordinary legitimate uses --
# CUDA and MKL on clusters are routinely reached that way -- so stripping them
# would break working machines to close a hole nobody has been bitten by. The
# line is drawn at injection, not at search paths.
#
# Dropped rather than emptied. `PYTHONPATH=""` is not the same as unset on
# every platform, and an empty entry has meant "the current directory" often
# enough to be worth not relying on.
_HOST_OVERRIDES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONEXECUTABLE",
    "PIP_TARGET",
    "PIP_PREFIX",
    "PIP_ROOT",
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
)

# What a stage's text is encoded in, decided here rather than by the machine.
# `_run_guarded` reads the pipes as UTF-8, and these are the child's side of
# that: `PYTHONUTF8=1` puts it in UTF-8 mode, and `PYTHONIOENCODING` is set
# too because it wins over the mode -- measured on 3.12, `PYTHONUTF8=1
# PYTHONIOENCODING=cp1252` gives `sys.stdout.encoding == "cp1252"`. Both are
# overwritten rather than defaulted: a host value would decide what a
# measurement's text looks like, which is the same argument `_HOST_OVERRIDES`
# makes about the pins, and `PYTHONUTF8=0` from a host would leave the child
# on the locale's encoding -- the ANSI code page on Windows.
#
# `utf-8:surrogateescape` and not bare `utf-8`: naming an encoding with no error
# handler sets the child's stdout to `strict`, where UTF-8 mode alone gives
# `surrogateescape` (PEP 540; measured on 3.12). Strict would make a child raise
# on a lone surrogate -- anything that came back through `os.fsdecode`, a path in
# a traceback -- and the runtime CLI catches that, prints "An error occurred" and
# exits zero, which is the failure this pair exists to prevent. The parent reads
# the pipe with `errors="surrogateescape"`, so the two agree: neither side
# raises on text the other could not represent.
_CHILD_TEXT = (("PYTHONUTF8", "1"), ("PYTHONIOENCODING", "utf-8:surrogateescape"))

# How long to wait for the pipes after killing a process group. A grandchild
# that called `setsid()` itself escapes the group, and if it still holds the
# inherited stdout it can keep an unbounded read open forever -- which would
# put back the unbounded wait this whole change exists to remove, one level
# out from where the original bug was.
_DRAIN_AFTER_KILL_S = 10

# How long to wait for the child to become reapable once it has been SIGKILLed
# and its pipes are done. Short because by this point the only thing that can
# still delay it is an uninterruptible sleep, which no wait would survive
# either; the alternative is leaving a zombie for the life of the run.
_REAP_AFTER_KILL_S = 5

# One read. The size is CPython's own in `Popen._communicate`, and it is
# copied rather than chosen: nothing here measured a better one, and matching
# the code this stands in for is the cheapest way to be sure the difference
# between them is not the read size. Chunking is not observable in the output
# -- both paths join before decoding -- so this is about syscall count alone.
_READ_CHUNK = 32768

# How many times in a row a read may answer EINTR, per pipe, before that pipe
# is given up on. PEP 475 has `os.read` retry a signal itself unless a Python
# handler raised, so even one is unusual and a run of them means something
# else is wrong. The bound is what stops the drain spinning: measured with a
# read that always raises, 34,862 attempts in 30 ms, because the descriptor
# stays ready and `select` hands it straight back.
_MAX_INTERRUPTED_READS = 8

# How long a stage gets between SIGTERM and SIGKILL. Long enough for torch to
# finish a `save_pretrained` it had started, short enough that a wedged process
# cannot meaningfully extend the timeout it already blew through.
_TERM_GRACE_S = 5

# Windows has no `SIGKILL`, and reading the attribute unconditionally is enough
# to raise there -- inside the very handler that was reporting a timeout, so
# the timeout would never be reported at all. `None` means "kill through
# `Popen.kill`", which is what `subprocess` does on that platform anyway.
_SIGKILL = getattr(signal, "SIGKILL", None)

# Names already named in a warning during the current run. `run` is called once
# per prompt during evaluation and Colab sets `PYTHONPATH` by default, so a
# per-call warning is hundreds of identical lines through a progress report --
# but "once per process" is the wrong grain for a library, where a notebook or a
# service is one process spanning many independent runs. `forget_reported_drops`
# is what a run boundary calls to get its own record.
_REPORTED_DROPS: set[str] = set()


def forget_reported_drops() -> None:
    """Start a fresh record of which host variables have been reported dropped.

    Called at the top of a run so the second run in one process says what the
    first one did, rather than inheriting its silence.
    """
    _REPORTED_DROPS.clear()


def _interpreter_in(root: Path) -> Path:
    """Where `venv` puts the interpreter inside `root`.

    One function because there were two spellings: `provision` built
    `Scripts/python.exe` on Windows while `StageEnv.python` looked for
    `Scripts/python`, which `venv` never creates there. That cost nothing
    while `ready` read only the marker file; once `ready` also checks the
    interpreter, the mismatch would make every Windows environment read as
    unprovisioned forever -- reinstalling torch on every invocation while
    `litetune env` called each working environment `incomplete`.
    """
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _console_script_in(root: Path, name: str) -> Path | None:
    """The console script `name` inside `root`, or None to fall back to `-m`.

    Windows entry points are `.exe` wrappers, so looking only for the bare name
    there finds nothing and every stage silently takes the `python -m` path --
    which works for `pip` and not for `litert-torch`, whose module is not
    executable that way.
    """
    bindir = root / ("Scripts" if os.name == "nt" else "bin")
    candidates = [bindir / f"{name}.exe", bindir / name] if os.name == "nt" else [bindir / name]
    return next((c for c in candidates if c.exists()), None)


def _group_of(proc: subprocess.Popen) -> int | None:
    """The child's process group, when signalling it is safe and meaningful.

    `None` where there is no `killpg`, where the lookup is refused, or where
    the group turns out to be our own -- `start_new_session=True` means it
    cannot be, but a wrong answer here SIGKILLs the session running litetune,
    so the invariant is checked rather than trusted.
    """
    if os.name == "nt" or not hasattr(os, "killpg"):
        return None
    # A host that ignores SIGCHLD has the kernel reap children the moment they
    # exit, and `Popen` never learns: `returncode` stays None while the pid --
    # which *is* the group id -- is already free to be handed out again. No
    # check closes that window, because any observation is stale by the time
    # the signal goes out. So the group is not used there at all, and the
    # direct child is what can be reached safely. Losing the sweep on such a
    # host is the price of not signalling a stranger's group on every other.
    sigchld = getattr(signal, "SIGCHLD", None)
    if sigchld is not None and signal.getsignal(sigchld) == signal.SIG_IGN:
        logger.debug("SIGCHLD is ignored, so a stage's descendants cannot be collected safely")
        return None
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        # Not recovered from `proc.pid`. That fallback was added because macOS
        # answers ESRCH for a zombie leader where Linux still reports its
        # group, and it was justified by "we only get here with the child
        # unreaped". `SA_NOCLDWAIT`, set through `sigaction`, breaks that: the
        # kernel reaps, `getsignal` cannot see the flag, `returncode` stays
        # None, and the pid is free -- so the fallback could aim a SIGKILL at
        # whoever holds that number now. The zombie case is solved instead by
        # reading the group at spawn, where the child is certainly alive; see
        # `StageEnv.run`.
        return None
    return None if pgid == os.getpgid(0) else pgid


class Reach(Enum):
    """How far a kill got. A bool had room for two of these four.

    The distinction is `checks.Outcome`'s, one layer down: "the group is gone"
    and "nothing could be signalled" are both *not* "killed", and only one of
    them means something is still running. A caller that logs the difference
    can say what it knows instead of guessing at a rogue grandchild.

    Ordered worst-to-best so `max()` can take the best reach achieved rather
    than the last one attempted -- a SIGTERM that reached the whole group must
    not be reported as "nothing could be signalled" because the SIGKILL after
    it was refused.

    A plain `Enum`, not `str, Enum`: the values are never read, and the mixin
    makes `f"{Reach.GROUP}"` render as `group` on 3.10 and `Reach.GROUP` on
    3.11+, which is a difference waiting for the first person to log one
    directly.
    """

    NOTHING = 0  # every attempt was refused; it is still out there
    ALREADY_GONE = 1  # nothing to signal, and nothing left running
    CHILD_ONLY = 2  # only the direct child; descendants are not covered
    GROUP = 3  # the whole process group was signalled

    def __lt__(self, other: Reach) -> bool:
        return self.value < other.value


def _signal_tree(proc: subprocess.Popen, pgid: int | None, sig: int | None) -> Reach:
    """Deliver `sig` to the group if there is one, else to the child alone.

    `sig` is `None` where the platform has no such signal -- Windows has no
    `SIGKILL` -- and the child is then killed through `Popen.kill`, which is
    what `subprocess` itself does there.

    A process that has already gone reports `ALREADY_GONE` rather than success:
    between a timeout firing and this call is exactly where a process finishes
    on its own, so it is the ordinary case and not a failure, but it is also
    not evidence that anything was signalled.
    """
    if pgid is not None and sig is not None:
        try:
            os.killpg(pgid, sig)
            return Reach.GROUP
        except ProcessLookupError:
            return Reach.ALREADY_GONE
        except OSError:
            # `PermissionError` among them: a setuid child cannot be signalled
            # by its parent. Fall through and try the child directly.
            pass
    try:
        if sig is None:
            proc.kill()
        else:
            proc.send_signal(sig)
    except ProcessLookupError:
        return Reach.ALREADY_GONE
    except (OSError, ValueError):
        # `ValueError` is Windows' answer to a signal it does not support.
        return Reach.NOTHING
    # `Popen.send_signal` polls first and returns silently for a child already
    # reaped, and it swallows `ProcessLookupError` from `os.kill` besides -- so
    # the arm above cannot fire on this path. Read what that poll learned
    # rather than polling again: a second poll of our own would reap on the
    # EPERM fallback, freeing the group id the caller is still holding.
    if proc.returncode is not None:
        return Reach.ALREADY_GONE
    return Reach.CHILD_ONLY


def _kill_tree(
    proc: subprocess.Popen,
    grace: float = _TERM_GRACE_S,
    stop: Callable[[], bool] | None = None,
    pgid: int | None = None,
    pump: Callable[[float], None] | None = None,
) -> Reach:
    """End the child and everything it spawned: SIGTERM, then SIGKILL.

    Returns the best reach achieved, so a caller can say "and it could not be
    killed" rather than leaving a leaked process to be diagnosed later as
    somebody else's out-of-memory kill.

    SIGTERM first because this replaced something gentler. Before the child had
    a session of its own, Ctrl-C reached it as SIGINT through the terminal and
    torch got to run its own handlers -- which for `tune` is the difference
    between a checkpoint written and a file truncated mid-`save_pretrained`.
    Going straight to SIGKILL would have been a quiet downgrade of that, so the
    grace is the part that keeps the replacement honest; SIGKILL still follows,
    because a wedged process must not be able to outlast its own timeout.
    """
    if proc.returncode is not None:
        return Reach.ALREADY_GONE
    # Read before anything can poll, and `poll()` is deliberately not called
    # here. It reaps, and reaping the leader does two bad things at once: it
    # frees the pid that *is* the group id, and it makes `getpgid` answer
    # ESRCH so the id cannot be recovered. A guard that polled first therefore
    # skipped the group kill in exactly the state this module is written for --
    # the child exited, a grandchild still holds the pipe -- and the descendant
    # survived. Measured as a leak. The auto-reaping hazard that guard was
    # aiming at is handled where it can be handled without reaping, in
    # `_group_of`.
    # A group read at spawn beats one looked up now: by the time a kill runs,
    # the leader may be a zombie whose group macOS will not report.
    pgid = pgid if pgid is not None else _group_of(proc)
    reached = _signal_tree(proc, pgid, signal.SIGTERM)
    if reached is Reach.ALREADY_GONE or proc.returncode is not None:
        # `Popen.send_signal` polls before it signals, so the direct-child leg
        # -- reached whenever `killpg` was refused -- can reap the leader on
        # its way through. Once that happens the group id is no longer ours,
        # and on macOS this is the *ordinary* path: a group whose only member
        # is a zombie answers EPERM, which looks exactly like the setuid case.
        return reached
    if reached is not Reach.NOTHING and grace > 0:
        try:
            still_ours = _wait_without_reaping(proc, pgid, grace, stop, pump)
        except BaseException:
            # The grace is the one part of this function that can be
            # interrupted -- a Ctrl-C during it raises straight through --
            # and every exception there used to take the SIGKILL with it.
            # Measured on a real tree: SIGTERM sent, SIGKILL not, the leader
            # unreaped and the group still alive. That is the leak this
            # module exists to prevent, produced by the keystroke that was
            # asking for the stage to stop sooner rather than to outlive us.
            # Guarded the way the ordinary path at the top of this function
            # is: once the leader has been reaped the group id is no longer
            # ours, and a second termination signal can have reaped it while
            # this grace was being interrupted. Signalling anyway aims
            # SIGKILL at whatever now owns that number.
            #
            # It does skip the `still_ours` check below, and that is the
            # trade rather than an oversight: that check is what the grace
            # returns, and the grace is what just raised.
            #
            # `returncode` is the same evidence the ordinary path at the top
            # of this function uses -- no better, and no worse. It is not
            # proof: `_group_of` spells out that a host using `SA_NOCLDWAIT`
            # has the kernel reap with `returncode` still None and the pid
            # already free, and that neither `Popen` nor `getsignal` can see
            # it. That hole is open on both paths and is not closed here;
            # what this check does close is the ordinary case, where a
            # second termination signal reaped the leader through
            # `send_signal` while this grace was being interrupted.
            #
            # `returncode`, not `poll()`. Polling reaps, and reaping the
            # leader here is the very hazard the top of this function is
            # arranged around -- it would free the number and then aim the
            # SIGKILL at it.
            if proc.returncode is None:
                _signal_tree(proc, pgid, _SIGKILL)
            raise
        # Withheld only where there is a group to mis-target. With no group the
        # final signal goes through `Popen`, which polls first and cannot reach
        # a stranger -- and on Linux `waitid` answers ECHILD for any pid that
        # is not our child, so withholding it there stopped the SIGKILL
        # unconditionally. CI caught that; this machine could not, because
        # CPython has no `os.waitid` on macOS before 3.13.
        if not still_ours and pgid is not None:
            return reached
    # Unconditionally, even where the child exited on the SIGTERM: what it
    # spawned is not covered by its own exit, and this is the signal that
    # collects them. Measured 2026-09-09: a group whose leader has exited but
    # is not yet reaped still exists and still takes a signal, which is what
    # makes this safe -- and why the wait above must not reap.
    return max(reached, _signal_tree(proc, pgid, _SIGKILL))


def _wait_without_reaping(
    proc: subprocess.Popen,
    pgid: int | None,
    grace: float,
    stop: Callable[[], bool] | None = None,
    pump: Callable[[float], None] | None = None,
) -> bool:
    """Wait for the child to exit, leaving it reapable.

    `proc.wait()` would be the obvious call and is the wrong one: it reaps, and
    a process group id *is* the leader's pid, so reaping frees the number this
    function's caller is about to send SIGKILL to. Measured 2026-09-09: with
    the leader reaped and no other member, `killpg` answers `ESRCH` and the id
    is free for reuse; unreaped, the group is still there.

    Two ways to watch without reaping, because neither covers every host.
    `waitid` with `WNOWAIT` reports the exit and leaves the child reapable, but
    it does not exist on macOS -- measured, on the Python this project is
    developed against, where the guard has to stay on `waitid` itself because
    `os.P_PID` and `os.WNOWAIT` are both present there anyway. Failing that,
    signalling the group with 0 asks whether anything in it is still there,
    which is the question this wait is really about: it ends early when the
    whole group is gone, and that is also how a *second* termination signal
    cuts this wait short, since its own hard kill empties the group.

    With neither, the grace is slept out. Slower when the child obeys at once,
    never wrong.

    `pump` is given the waiting interval instead of `time.sleep`, so the
    pipes are emptied while the child shuts down. Without it a stage that
    writes more than a pipe buffer on its way out blocks in `write` and
    spends the whole grace blocked -- see `_PipeDrain`.
    """
    deadline = time.monotonic() + grace
    waitid = getattr(os, "waitid", None)
    while True:
        if stop is not None and stop():
            # A second termination signal has already killed outright, so there
            # is nothing left for this wait to give.
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        if waitid is not None:
            try:
                if waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    # Exited, and still reapable: the pid stays pinned, so the
                    # caller's group id is still its own.
                    return True
            except ChildProcessError:
                # Somebody else reaped it: the pid, and so the group id, is
                # free *now*. Reported rather than swallowed, because the
                # caller's next act is a `killpg` on that number.
                return False
            except OSError as exc:
                # We could not observe, which is not the same as "it exited".
                # Sleeping out the rest of the grace is what this function
                # promises; returning here would silently make the grace zero.
                logger.debug("cannot watch pid %d without reaping it: %s", proc.pid, exc)
                waitid = None
        elif pgid is not None:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                # Unambiguous: the group is empty.
                return True
            except PermissionError:
                # Ambiguous, and only on one platform. macOS answers EPERM for
                # a group whose sole member is an unreaped zombie -- measured
                # -- so reading it as "nothing live left" is what keeps the
                # grace from being slept out in full there, which matters
                # because `os.waitid` is missing from CPython on macOS before
                # 3.13. On Linux EPERM has only the ordinary meaning: a live
                # group we may not signal, which is a stage that execed a
                # setuid helper -- and returning there would take away the
                # grace from exactly the shutdown it exists to protect.
                if sys.platform == "darwin":
                    return True
                logger.debug("cannot probe process group %d, so waiting out the grace", pgid)
                # Stop probing, not stop waiting: the grace is what the setuid
                # descendant needs, and it is the one thing we can still give.
                pgid = None
            except OSError as exc:
                logger.debug("cannot probe process group %d: %s", pgid, exc)
                pgid = None
        interval = min(0.05, remaining)
        if pump is not None:
            pump(interval)
        else:
            time.sleep(interval)


@contextlib.contextmanager
def _kill_child_if_we_are_told_to_exit(proc: subprocess.Popen, pgid: int | None = None):
    """Take the child down with us on SIGTERM or SIGHUP.

    `start_new_session=True` is what lets a timeout kill everything the stage
    spawned, but it also took the child out of the terminal's process group --
    and a hangup on that group, an SSH session dropping or a window closing, is
    how an interactive run used to be cleaned up. Python turns only SIGINT into
    an exception, so the `except BaseException` in `run` cannot stand in for
    these two: without this, `kill <litetune>` or a dropped connection leaves a
    six-hour torch run holding its memory, which is the very thing the process
    group was introduced to prevent.

    A signal the process was already ignoring is left alone. `nohup` sets
    SIGHUP to `SIG_IGN` for exactly one purpose -- to survive the hangup that
    ends the SSH session -- and installing over it would turn the documented
    way to run a six-hour `tune` into the one way to lose it. The same holds
    for a supervisor that ignores SIGTERM on purpose.

    The previous disposition is restored before the signal is re-raised, so the
    process still dies the way it would have. Signals can only be installed on
    the main thread; elsewhere this is a no-op rather than an error, because a
    library that refused to run off the main thread would be worse than one
    that cleans up in fewer cases.
    """
    installed: dict[int, Any] = {}
    handling = False
    # Set by a second termination signal and read by the first invocation's
    # grace. Needed because SIGKILL does not empty a group -- the leader stays
    # an unreaped zombie -- so "the wait notices on its own" holds only where
    # there is a group to watch, and not on the child-only fallback.
    escalated = False

    def take_the_child_with_us(signum, _frame):
        nonlocal handling, escalated
        # Our handler stays installed while it runs, so a second signal
        # re-enters here. Measured: three SIGTERMs a second apart against a
        # child that ignores them took 8.0s to give up instead of 5.0s, because
        # each nested call started its own grace.
        #
        # The second signal means "stop harder", so it skips the grace and
        # kills outright -- and then *returns*, rather than raising. Raising
        # through the first invocation was the earlier attempt and it was
        # worse: the first frame never reached its restore or its re-delivery,
        # so the process died of an uncaught exception instead of the signal,
        # and `run`'s handler then started a fresh grace of its own. Returning
        # leaves the first invocation to finish, and it finishes at once,
        # because emptying the group is exactly what its wait is watching for.
        if handling:
            escalated = True
            _kill_tree(proc, grace=0, pgid=pgid)
            return
        handling = True
        # The same drain the timeout path uses, for the same reason: without
        # something emptying the pipes, a stage that writes more than a pipe
        # buffer while shutting down blocks in `write` and is SIGKILLed part
        # way through its `save_pretrained`. Measured with a stage writing
        # 200,000 bytes on SIGTERM: it never reached the end of its handler.
        # This is the path that matters most for that, because it is the one
        # `kill <litetune>` and a dropped SSH session take.
        #
        # What it reads is dropped. The process is on its way out and no
        # caller reads a stage's output on this path; the point is to unblock
        # the child, not to keep what it says.
        drain = _PipeDrain(proc, seed=False)
        try:
            _kill_tree(proc, stop=lambda: escalated, pgid=pgid, pump=drain.pump)
        finally:
            drain.close()
            # Reset before the re-delivery below, and before `run`'s cleanup
            # runs under this same guard, so a later TERM or HUP re-enters
            # this handler rather than being swallowed.
            #
            # An earlier version never reset it, and a `finally` rather than a
            # plain assignment after the `try` is what this has to be. The
            # grace just above is not the damage either way -- `handling` is
            # set during it in every version, and a second signal there takes
            # the escalation branch at the top of this handler on purpose.
            #
            # The two routes out of the `try` differ in what is left pointing
            # here. When the grace returns, the signal being handled is put
            # back on its previous disposition below, and only the other of
            # TERM and HUP can still reach this handler. When the grace is
            # interrupted -- a Ctrl-C while it waits -- the restore below never
            # runs, and both of them can. A plain reset would be skipped on
            # exactly that second route, which is why it is a `finally`.
            #
            # With `handling` left set, whichever of them arrived took the
            # escalation branch and returned instead of ending litetune, for
            # the whole of `run`'s cleanup -- a grace, the drain and the reap,
            # `_TERM_GRACE_S` plus `_DRAIN_AFTER_KILL_S` plus
            # `_REAP_AFTER_KILL_S`, up to 5 + 10 + 5 = 20 seconds on either
            # route. (SIGKILL always got through, and SIGINT was never this
            # guard's to swallow -- it installs only TERM and HUP.)
            handling = False
        # `None` means the handler in place was installed from C and cannot be
        # restored from Python -- `signal.signal(sig, None)` is a `TypeError`.
        # Falling back to the default is what "die the way we would have" means
        # when the previous disposition is not expressible here.
        previous = installed.get(signum)
        signal.signal(signum, signal.SIG_DFL if previous is None else previous)
        os.kill(os.getpid(), signum)
        # Reached only where that disposition returns instead of ending the
        # process -- litetune embedded in a larger application whose own
        # handler carries on. Returning here would resume `communicate()` over
        # a child we just killed and hand the caller a `-9`, which `exits`
        # reports as the out-of-memory killer: our own kill, blamed on the
        # machine. This carries the signal number instead, because
        # `KeyboardInterrupt` already means one specific thing to every reader
        # and to CPython's exit status, and nobody typed Ctrl-C.
        raise StageInterrupted(signum)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            if signal.getsignal(sig) == signal.SIG_IGN:
                continue
            installed[sig] = signal.signal(sig, take_the_child_with_us)
        except (ValueError, OSError) as exc:
            # Off the main thread, or a platform without this signal -- the
            # message must not pick one. `warning`, not `debug`: the
            # consequence is a six-hour torch run outliving the SIGTERM that
            # was meant to end it, and `cli` leaves the level at WARNING unless
            # asked for more.
            logger.warning(
                "no %s guard for %s (%s): a termination signal will not reach it",
                name,
                _describe(proc),
                exc,
            )

    try:
        yield
    finally:
        for sig, previous in installed.items():
            # `None` is a handler installed from C, which Python cannot express
            # and so cannot restore. Leaving ours in place is the worse of the
            # two: it would outlive this call, closed over a process that no
            # longer exists. The default is at least the platform's own answer.
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, signal.SIG_DFL if previous is None else previous)


class _PipeDrain:
    """Read a child's pipes without reaping it, and keep what was read.

    `communicate()` is the obvious call and cannot do this job, for two
    reasons the module has measured.

    It reaps. A process group id *is* the leader's pid, so a reap before the
    SIGKILL frees the number the kill is aimed at -- the same hazard
    `_wait_without_reaping` exists for, one function further out.

    And it can only run *after* the kill, which left nothing emptying the
    pipes during the grace. A pipe holds 65,536 bytes -- measured on the
    machine this was written on, by writing to one until EAGAIN; a stage that writes
    more than that on its way out -- a torch traceback, a progress bar
    flushing its last frames, `save_pretrained` logging each shard -- blocks
    in `write`, never reaches the end of its shutdown, and is SIGKILLed at the
    end of a grace it spent blocked. The grace helped a quiet stage and not a
    talkative one, which is the wrong way round: the talkative one is the one
    with something to say.

    Reads are raw `os.read` on the file descriptors, the way CPython's own
    `Popen._communicate` does it, and the decode happens once at the end. A
    text-mode pipe is a `TextIOWrapper`; reading one incrementally either
    blocks waiting to complete a multi-byte character or raises when the
    descriptor is non-blocking.
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        already: tuple[str | bytes | None, str | bytes | None] = (None, None),
        *,
        seed: bool = True,
    ) -> None:
        self._out: list[bytes] = []
        self._err: list[bytes] = []
        self._encoding = _encoding_of(proc.stdout)
        self._err_encoding = _encoding_of(proc.stderr)
        # Whatever a `communicate` before us already took off the pipes. Those
        # bytes are gone from the descriptors and this is the only place left
        # that has them, so without picking them up the salvaged output starts
        # at the moment of the timeout and drops everything before it -- which
        # on a stage that died early is all of it.
        #
        # `seed=False` skips it for a caller that says it will throw the text
        # away. This is a `b"".join` over everything the stage has said -- a
        # second full copy of the transcript -- and the callers that do not
        # want it are a signal handler, which builds its drain before its own
        # SIGTERM goes out, and `run`'s interrupt cleanup, which runs after a
        # handler has already signalled on the route that raises
        # `StageInterrupted`. A `MemoryError` there would also escape a
        # constructor that promises not to raise.
        carried_in = (
            (
                (proc.stdout, already[0], self._out, self._encoding),
                (proc.stderr, already[1], self._err, self._err_encoding),
            )
            if seed
            else ()
        )
        for stream, carried, sink, encoding in carried_in:
            recovered = _buffered_by_communicate(proc, stream)
            if recovered is None:
                # No `communicate` ran, or this CPython keeps its partial
                # reads somewhere else. The caller's exception is then the
                # only account of them there is.
                recovered = (
                    carried.encode(encoding, "replace") if isinstance(carried, str) else carried
                )
            if recovered:
                sink.append(recovered)
        self._selector: selectors.BaseSelector | None = None
        # Whether this host can watch pipes at all, which is a different
        # question from whether any are left to watch. Conflating them sent
        # the one case the seeding exists for -- `communicate`'s trailing
        # `wait()`, which only raises after both pipes hit EOF and were
        # closed -- down the fallback branch, where the seed is not used.
        self._can_watch = False
        self._open = 0
        # Per descriptor, not one shared count: with a single counter,
        # interruptions alternating between the two pipes close one of them
        # early, and a steady stream of data on one resets the count for the
        # other so the bound is never reached at all.
        self._interrupted: dict[int, int] = {}
        # Nothing here may raise. This runs while a timeout or an interrupt is
        # already being handled, and an exception would replace the reason the
        # caller is here with a reason about reading pipes. A drain that
        # cannot watch anything still keeps the bytes it was seeded with, and
        # `pump` still spends the grace -- which is what happened before this
        # class existed.
        try:
            selector = selectors.DefaultSelector()
        except OSError as exc:  # pragma: no cover - needs a descriptor to spare
            # `epoll`/`kqueue` each need a descriptor of their own, and the
            # host is out of them.
            logger.debug("no usable selector, so the stage goes unheard: %s", exc)
            return
        refused = 0
        for stream, sink in ((proc.stdout, self._out), (proc.stderr, self._err)):
            if stream is None or stream.closed:
                # Nothing to watch is not the same as being unable to watch.
                # A pipe already at EOF and closed is the ordinary state after
                # `communicate` gave up, and the seed is the account of it;
                # treating that as "this host cannot watch" sent it to the
                # fallback, which has no seed to return.
                continue
            try:
                selector.register(stream, selectors.EVENT_READ, sink)
            except (ValueError, OSError) as exc:
                # A refusal, which is the other thing entirely: `epoll_ctl`
                # out of room, or a descriptor this selector will not take.
                # Measured against the version that set the flag before this
                # loop: with both registrations refused the fallback was
                # skipped and a stage that had a traceback to give returned
                # two empty strings.
                logger.debug("cannot watch a pipe of pid %d: %s", proc.pid, exc)
                refused += 1
                continue
            self._open += 1
        # All or nothing -- for a drain whose text will be read. A drain that
        # watches one pipe and not the other reads the one, and is then
        # overridden by a fallback that replaces its result rather than
        # merging, so what it read is discarded. Measured with stdout
        # registered and stderr refused: `out == ''` for a stage that had
        # spoken on both. Giving up on both instead lets `communicate` read
        # both.
        #
        # That rule costs the grace, and for a drain whose text is thrown away
        # it buys nothing that the fallback does not still provide: the leak
        # warning comes from the fallback reading both streams, and a partial
        # drain reaches the fallback just the same. With nothing watched, nothing empties the pipes
        # while the stage shuts down, and a stage that writes more than a
        # buffer on its way out blocks and is SIGKILLed -- measured: 5.0 s,
        # exit -9, its handler never finished, where a partial watch let it
        # finish in 0.1 s. `seed=False` is exactly the two callers that
        # discard: the signal handler and `run`'s own interrupt cleanup, and
        # the first of those is the `kill <litetune>` path. So they keep
        # whatever did register, and the rule applies only where the output
        # is the point.
        self._can_watch = refused == 0
        if self._open and (self._can_watch or not seed):
            self._selector = selector
        else:
            # Kept truthful rather than load-bearing: with the selector gone
            # nothing reads, whatever this says, but `draining` should not
            # claim a pipe nobody is watching.
            self._open = 0
            selector.close()

    @property
    def encoding(self) -> str:
        """The encoding stdout was opened with."""
        return self._encoding

    @property
    def err_encoding(self) -> str:
        """The encoding stderr was opened with."""
        return self._err_encoding

    @property
    def draining(self) -> bool:
        """Whether a pipe this drain is watching has not reached its end.

        Only about the pipes it watches. A pipe it could not register, or one
        it stopped watching when `select` refused, contributes nothing here --
        not "held" and not "closed", because nothing was read from it. A
        drain in that state is not `watching`, and `_kill_and_drain` then asks
        the fallback instead, which reads both streams.

        A pipe can also be closed without the drain giving up watching, and
        then this reports it done while a writer may still hold it: when a
        read fails -- any `OSError` other than an interruption, or
        `_MAX_INTERRUPTED_READS` interruptions in a row. Both are treated as
        the end of that pipe because reading it again cannot succeed and
        leaving it registered spins the loop. Neither has been seen outside a
        test; PEP 475 has `os.read` retry EINTR itself unless a Python
        handler raised.
        """
        return self._open > 0

    @property
    def watching(self) -> bool:
        """Whether this drain's result can be trusted as the whole account.

        Not "is it reading": a discarding drain that registered one pipe of
        two reads that one while this is False. What False means is that some
        pipe went unwatched -- a registration refused, or `select` refusing
        on a host where it takes sockets only -- so a caller that wants the
        output, or wants to know whether a stray process still holds the
        pipes, has to ask the fallback, which reads both. (`close()` makes it
        False as well, on a drain that watched everything; nothing asks after
        that.)

        Not "is anything left to read" either: a drain whose pipes both
        reached EOF is still watching, and its seed is still the account of
        what the stage said. That distinction is the whole of the
        trailing-`wait()` case.

        Answered only after a read has been attempted, never before one.
        `SelectSelector.register` checks the event mask, the descriptor and
        whether it is registered already -- and not whether `select` on this
        host can do anything with it. Measured: it accepts a pipe and keeps
        the descriptor, so where `select` takes sockets only both pipes
        register happily and the refusal arrives on the first `select`.
        """
        return self._can_watch

    def pump(self, budget: float) -> None:
        """Read whatever is ready, spending exactly `budget` seconds.

        Shaped to drop into a wait loop in place of its `time.sleep`: it
        consumes the whole budget, so the cadence of the loop it replaces
        does not change with how talkative the stage is.

        "Consumes" and not "sleeps through": with a child writing steadily it
        reads for the whole budget without yielding, which is the point --
        the alternative is letting the pipe refill and blocking the child
        again. It sleeps only when there is nothing to read.
        """
        self._read_for(budget, until_eof=False)

    def finish(self, budget: float) -> None:
        """Read to the end of both pipes, or until `budget` runs out.

        Returns as soon as the pipes close, which `pump` deliberately does
        not: measured, spending the whole ten seconds here added ten seconds
        to every timeout the moment the drain started working.
        """
        self._read_for(budget, until_eof=True)

    def _read_for(self, budget: float, until_eof: bool) -> None:
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._selector is None or not self._open:
                if until_eof:
                    return
                time.sleep(remaining)
                return
            try:
                ready = self._selector.select(remaining)
            except OSError as exc:
                # Nothing may escape this. It runs between the SIGTERM and
                # the SIGKILL, and an exception here leaves the group
                # unsignalled, the child unreaped, and the caller holding an
                # errno where its own timeout should be.
                #
                # This is the Windows shape and the reason `watching` cannot
                # be answered before a read: `select` there takes sockets
                # only, so both pipes register and the first `select` refuses
                # them. Giving up on watching leaves the caller free to fall
                # back to `communicate`, which does work there.
                logger.debug("cannot watch the stage's pipes: %s", exc)
                self._unwatch()
                continue
            for key, _ in ready:
                self._read(key)

    def _unwatch(self) -> None:
        """Give up on watching, without touching what has already been read."""
        if self._selector is not None:
            self._selector.close()
            self._selector = None
        self._can_watch = False
        self._open = 0

    def _read(self, key: selectors.SelectorKey) -> None:
        try:
            chunk = os.read(key.fd, _READ_CHUNK)
        except InterruptedError:
            # A signal arrived mid-read. PEP 475 has `os.read` retry this
            # itself unless a Python handler raised, so it is close to
            # unreachable -- but the cost of getting it wrong is not
            # proportionate: the branch below closes the pipe, so one EINTR
            # would drop everything the stage says from then on. Verified by
            # planting one: the read end closed and the next write to the
            # pipe raised `BrokenPipeError`.
            self._interrupted[key.fd] = self._interrupted.get(key.fd, 0) + 1
            if self._interrupted[key.fd] > _MAX_INTERRUPTED_READS:
                logger.debug("giving up on a pipe that answers only EINTR")
                self._close(cast("IO[Any]", key.fileobj))
            return
        except OSError as exc:
            # Treated as end of stream. The descriptor is unreadable, so no
            # amount of further selecting will produce anything, and leaving
            # it registered would spin this loop for the rest of the budget.
            logger.debug("cannot read a pipe of the stage: %s", exc)
            chunk = b""
        self._interrupted.pop(key.fd, None)
        if chunk:
            key.data.append(chunk)
            return
        # Narrowed here because here is where it is known: `SelectorKey`
        # types this as `int | HasFileno`, and the register loop above is
        # what establishes that only the two pipe objects are ever in it.
        self._close(cast("IO[Any]", key.fileobj))

    def _close(self, stream: IO[Any]) -> None:
        # `unregister` is not guarded, and that is the invariant rather than
        # an oversight: it is only ever reached with a key `select` has just
        # handed back, so the registration exists. `_fileobj_lookup` falls
        # back to an identity scan for a stream whose `fileno()` now raises,
        # and the kqueue and epoll implementations swallow `OSError` for a
        # stale descriptor, so only `KeyError` could escape and only for a
        # key that was never registered.
        if self._selector is not None:
            self._selector.unregister(stream)
        self._open -= 1
        with contextlib.suppress(OSError):
            stream.close()

    def close(self) -> None:
        self._unwatch()

    def text(self) -> tuple[str, str]:
        r"""What was read, decoded and newline-translated like `communicate`.

        The translation matters because both paths describe the same child:
        `Popen._translate_newlines` turns `\r\n` and a lone `\r` into `\n`,
        so without it the same stage's output changed shape depending on
        whether it finished or was killed -- a `\r`-heavy progress bar
        rendering as overwrites inside an error report rather than as lines.
        """
        return (
            _translate_newlines(b"".join(self._out), self._encoding),
            _translate_newlines(b"".join(self._err), self._err_encoding),
        )


def _buffered_by_communicate(proc: subprocess.Popen, stream: IO[Any] | None) -> bytes | None:
    """What `communicate` read off `stream` before it gave up, or `None`.

    `Popen.communicate` accumulates into `_fileobj2output` and only assembles
    the result at the end, so a timeout raised from its trailing `wait()`
    carries no output at all while every byte sits in that dict -- and a
    timeout raised from the pipe read carries exactly what is in there. One
    source covers both, which is why this is preferred over the exception.

    It is a private attribute, so this asks for it rather than assuming it:
    a CPython that renames or drops it leaves the caller on the exception,
    which is where this module was before.
    """
    buffered = getattr(proc, "_fileobj2output", None)
    if not isinstance(buffered, dict) or stream is None:
        return None
    chunks = buffered.get(stream)
    if chunks is None:
        return None
    try:
        return b"".join(chunks)
    except TypeError:
        # Text-mode chunks on some other implementation. Not ours to join.
        return None


def _encoding_of(stream: object) -> str:
    """The encoding a pipe was opened with, or UTF-8 if it will not say.

    Asked once, while the stream is certainly open: a closed `TextIOWrapper`
    still answers, but a stream this module replaced with `None` does not.
    """
    return getattr(stream, "encoding", None) or "utf-8"


def _reap(proc: subprocess.Popen) -> None:
    """Collect the exit status, so a killed stage does not stay a zombie.

    Bounded, and only ever called after `_kill_tree` has returned -- which is
    not the same as after a SIGKILL, since that returns early both when the
    child had already gone and when the group stopped being ours to signal.
    Either way nothing else is going to collect it: `communicate` used to do
    this as a side effect, and dropping it in favour of reading the pipes
    directly would otherwise leave one zombie per timed-out stage for the life
    of the run.
    """
    try:
        proc.wait(timeout=_REAP_AFTER_KILL_S)
    except subprocess.TimeoutExpired:
        logger.debug("%s did not become reapable after being killed", _describe(proc))


def _kill_and_drain(
    proc: subprocess.Popen,
    pgid: int | None = None,
    already: tuple[str | bytes | None, str | bytes | None] = (None, None),
    *,
    seed: bool = True,
) -> tuple[str, str]:
    """Kill the group and collect what it wrote -- reading throughout.

    The drain runs *during* the grace as well as after the kill, which is the
    difference between a grace a talkative stage can use and one it spends
    blocked on a full pipe. `_PipeDrain` says why that could not be
    `communicate`.

    `already` is what the caller's own read took off the pipes before it gave
    up; it is carried rather than re-read, because those bytes are gone from
    the descriptors. `_PipeDrain` prefers the `Popen`'s own buffer to it and
    says why.

    Both phases are bounded, because the kill is not guaranteed to have
    reached everything: a grandchild that called `setsid()` is outside the
    group and can hold the inherited stdout open. Giving up on the output is
    the right trade, since the caller is already on its way to reporting a
    timeout or re-raising; hanging instead would reinstate the unbounded wait.
    """
    drain = _PipeDrain(proc, already, seed=seed)
    try:
        reached = _kill_tree(proc, pgid=pgid, pump=drain.pump)
        # `finish` before the decision, not after it. `_kill_tree` returns
        # without pumping on several paths -- an already-exited leader is the
        # one that matters, because a grandchild holding the pipes is exactly
        # what this function is for -- so on those the first read of the whole
        # run happens here, and asking `watching` any earlier asks it before
        # anything has tried.
        #
        # But only for a drain that is still watching. A drain that was
        # partial from its constructor already knows it will hand over to the
        # fallback, and letting it `finish` first made it wait the budget
        # twice when something held its pipe -- measured, 21.4 s where the
        # other shapes took 11.3 s -- for bytes the fallback's own read then
        # replaced. The shape the reorder above exists for still gets here:
        # where `select` takes sockets only, both pipes register, `watching`
        # stays true until the first `select`, and that happens in here.
        if drain.watching:
            drain.finish(_DRAIN_AFTER_KILL_S)
        if drain.watching:
            out, err, still_held = (*drain.text(), drain.draining)
        else:
            # Nothing on this host can be watched: `select` on Windows takes
            # sockets and not pipes, and a POSIX host can run out of the
            # descriptor `epoll` needs. Fall back to the read this replaced.
            # It is safe *here* and was not before, because `communicate`
            # reaps and a reap ahead of the kill frees the group id the kill
            # is aimed at. `_kill_tree` has returned by this line, which is
            # not always the same as "the SIGKILL was sent" -- it returns
            # early when the child had already gone, or when the group
            # stopped being ours -- but on exactly those paths there is no
            # further signal for a reap to spoil.
            #
            # Taken by discarding callers as well, and that is deliberate. An
            # earlier revision skipped it for them, on the argument that the
            # read costs up to `_DRAIN_AFTER_KILL_S` for text nobody reads.
            # The read is not where that time goes. `communicate(timeout=...)`
            # returns once both pipes reach EOF and the child is reapable, and
            # after a SIGKILL those normally come together, so it spends the
            # whole budget in two cases. One is something still holding a pipe
            # -- exactly the case the warning below exists for, and a drain
            # that watched every pipe spends one budget in `finish` then too.
            # The other is a child that does not become reapable: nothing could
            # be signalled and it kept running, or the killed leader is in the
            # uninterruptible sleep `_REAP_AFTER_KILL_S` names. The skip did
            # save that budget without silencing anything, on a stop that had
            # already failed. At most one budget either way: a drain that goes
            # to the fallback does not spend one in `finish` as well.
            #
            # What the skip bought in the first case was silence. On Windows,
            # with no process group and a `select` that refuses pipes, an
            # interrupt that finds the child alive and its pipes still held
            # reaches only the child, and "only the child was signalled" --
            # true, and the only clue that its descendants are loose -- stopped
            # being printed.
            #
            # One cost comes back with it: `communicate` assembles the whole
            # transcript, so a discarding caller on this path makes the copy
            # `seed=False` spares it in the constructor. That is a host that
            # could not watch its pipes, being interrupted.
            out, err, still_held = _drain_by_communicate(proc, drain)
        if still_held:
            # Said out loud, because a leaked process is otherwise diagnosed
            # hours later as somebody else's out-of-memory kill. The sentence
            # says only what `reached` supports: blaming a stray grandchild
            # when in fact nothing could be signalled would send the reader
            # hunting for the wrong process.
            logger.warning(
                "output pipes for %s are still open: %s",
                _describe(proc),
                {
                    # Not "a descendant left the group": `killpg` returning 0
                    # says the signal was accepted, not that anything died. A
                    # member wedged in an uninterruptible ioctl is unkillable
                    # and still holds the pipe, so naming one cause would send
                    # the reader hunting for the wrong process.
                    Reach.GROUP: (
                        "the group was signalled, so either something outside it is holding "
                        "them or something in it has not died"
                    ),
                    Reach.CHILD_ONLY: (
                        "only the child was signalled, which does not cover what it spawned"
                    ),
                    Reach.NOTHING: (
                        "nothing could be signalled at all, so the child itself may be "
                        "holding them"
                    ),
                    Reach.ALREADY_GONE: (
                        "the child had already exited, so something it spawned holds them"
                    ),
                }[reached],
            )
        # Whatever was read, however far it got. The stage's stderr is the
        # single most useful artifact in a timeout report, so it is worth
        # carrying the partial one rather than returning two empty strings.
        return out, err
    finally:
        # The selector goes back first. `_reap` waits, and a wait is
        # interruptible: with the two the other way round, a signal arriving
        # during the reap skipped the close entirely and leaked a kqueue or
        # epoll descriptor per interrupted stage. Before the reap moved in
        # here, `drain.close()` was the whole `finally` and could not be
        # skipped -- which is the property to keep.
        try:
            drain.close()
        finally:
            # In a `finally` of its own because the kill above can be
            # interrupted, and a child that was killed and never collected
            # is a zombie for the life of the run. Bounded, so an interrupt
            # is not answered by hanging.
            _reap(proc)


def _pipes_open(proc: subprocess.Popen) -> bool:
    """Whether either read end is still open on our side.

    Used only where a read has just timed out, which is what makes it
    evidence: `communicate` closes each stream as it reaches EOF, so a
    timeout with one still open means the end never came and something is
    still holding the write end. On its own the answer would be weaker than
    that -- an unclosed wrapper says only that nobody closed it.

    This is the fallback's counterpart to `_PipeDrain.draining`, which
    answers the same question on the watched path from what the drain saw.
    They answer over different sets: this asks both streams, while
    `draining` speaks only for the pipes the drain watched. `_kill_and_drain`
    uses `draining` only when the drain watched every pipe, which is when the
    two sets are the same.
    """
    return any(stream is not None and not stream.closed for stream in (proc.stdout, proc.stderr))


def _drain_by_communicate(proc: subprocess.Popen, drain: _PipeDrain) -> tuple[str, str, bool]:
    """Read the pipes the old way, for a host where they cannot be watched.

    Kept because the alternative on such a host is not "a slower drain" but
    "no output at all": a stage would time out and say nothing about why.
    What `communicate` returns already includes everything *it* read earlier,
    so it replaces the drain's seed rather than being added to it. It does
    not include what the drain itself took off a pipe with `os.read` -- a
    discarding drain that watched one pipe during the grace -- and that is
    dropped here; it is dropped only on a path whose caller throws the text
    away.

    Returns the third value the caller needs from the drain it is standing in
    for: whether the pipes are still held by something.
    """
    try:
        out, err = proc.communicate(timeout=_DRAIN_AFTER_KILL_S)
    except subprocess.TimeoutExpired as expired:
        out, err = expired.output, expired.stderr
        still_held = _pipes_open(proc)
    except (OSError, ValueError) as exc:
        # No warning from here. The read failed, so nothing was established
        # about who holds what -- our own wrappers being unclosed says only
        # that we did not close them, not that a writer is alive. A warning
        # that fires when nothing is known is the warning that gets ignored
        # when something is.
        logger.debug("could not read the pipes of %s: %s", _describe(proc), exc)
        return (*drain.text(), False)
    else:
        still_held = False
    if out is None and err is None:
        # The trailing `wait()` raised, so this call assembled nothing -- but
        # it did read, and into the same `_fileobj2output` the drain was
        # seeded from. The seed was taken in the drain's constructor, before
        # this read ran, so returning it dropped whatever the read found:
        # measured, `part1` returned while the buffer held `part1` and
        # `part2`. Asking the buffer again is what picks that up; the seed is
        # the account only where there is no buffer to ask.
        out = _buffered_by_communicate(proc, proc.stdout)
        err = _buffered_by_communicate(proc, proc.stderr)
        if out is None and err is None:
            return (*drain.text(), still_held)
    # In the encoding each pipe was opened with, not the default: a stage on a
    # non-UTF-8 host would otherwise have its salvaged stderr come back as
    # mojibake, which is the one artifact this exists to preserve. And
    # newline-translated, for the same reason `_PipeDrain.text` is: on the
    # `TimeoutExpired` exit these are raw, untranslated bytes straight off
    # the pipe. `_translate_newlines` passes an already-translated `str`
    # through unchanged, so the successful exit is unaffected.
    #
    # Not a Windows claim, though this is the path that exists for Windows:
    # `communicate` there reads on threads and its `TimeoutExpired` carries
    # no output at all, so the bytes-on-timeout shape is a POSIX one -- a
    # host that lost its selector, not a host that never had one.
    return (
        _translate_newlines(out, drain.encoding),
        _translate_newlines(err, drain.err_encoding),
        still_held,
    )


def _run_guarded(
    argv: Sequence[str],
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess so that whatever ends litetune ends it too.

    Shared by `StageEnv.run` and `StageEnv.provision`, which have the same
    problem and used to have two different answers. `run` grew a session, a
    termination guard and a group kill; `provision` stayed on `subprocess.run`,
    whose only cleanup is `Popen.kill` on its direct child and only on paths
    that raise. A SIGTERM to litetune raises nothing, so a half-hour
    `pip install torch` and every compiler it had spawned carried on against a
    directory nobody was going to keep.

    The session is what makes the group killable, and the group is what makes
    the kill worth anything: killing only the direct child means a six-hour
    `tune` timeout records an honest "not checked", the run continues, and an
    abandoned torch process keeps its memory -- the next stage is then
    SIGKILLed and `exits` reads that as the OOM killer, blaming the machine
    for litetune's own orphan.

    Returns the completed process rather than raising on non-zero: callers
    decide whether a non-zero exit is a failed check or an unperformed one,
    and that distinction is the whole point of litetune.checks.

    `encoding="utf-8"` rather than the locale's: `text=True` alone decodes with
    `locale.getpreferredencoding(False)`, which on Windows is the machine's ANSI
    code page. A stage's stdout is where a generation comes back from the
    runtime (`evaluate.LiteRtLmBackend`), so on a cp1252 host every character
    outside that page is mangled on the way in and the row scores wrong -- a
    conversion cost that measures the console. The child is put in UTF-8 mode
    by `_child_env` for the same reason, and the two have to agree: one without
    the other still loses the text.

    `errors="surrogateescape"` on the decode: `text=True` decodes stdout/stderr
    eagerly, and a stray non-UTF-8 byte -- a CUDA or driver banner ahead of a
    probe's own JSON line, in the same class of weird environment the
    last-non-empty-line rule in `resolve_device` exists to survive -- raised
    `UnicodeDecodeError` there uncaught, which is a `ValueError`, not the
    `OSError` every caller was written to expect. That escaped
    `resolve_device` and both of its callers and ended a `tune` or `verify`
    run with a traceback instead of an unanswered probe.

    `surrogateescape` and not `replace`, which is what it used to be: both
    keep the decode from raising, but `replace` writes U+FFFD, and U+FFFD is
    an ordinary character a model may generate. Erasing the difference means a
    generation that arrived broken cannot be told from one that says so -- and
    `evaluate` must tell them apart, because one is a harness failure and the
    other is an answer. A surrogate in U+DC80-U+DCFF is a byte that was not
    UTF-8 and nothing else.

    `env` is *overrides*, not a whole environment: the base is the host's
    minus `_HOST_OVERRIDES`, so a caller cannot accidentally hand the pins
    back through `PYTHONPATH`, and `PIP_TARGET` cannot redirect an install out
    of the venv it is supposed to fill.
    """
    proc = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        # The timeout message names a prompt as a cause; this is what stops
        # one from being waited on at all.
        stdin=subprocess.DEVNULL,
        env=_child_env(env),
        # POSIX only; accepted and ignored on Windows, where `_kill_tree`
        # falls back to killing the child alone. Not overridable: a caller
        # reaching past it with `process_group=` would put the child in *our*
        # group, and `_kill_tree` would then aim SIGKILL at the session that
        # is running litetune. There was a `**kwargs` passthrough here with no
        # user in the package; it is gone.
        start_new_session=True,
    )
    # The guard wraps the cleanup as well as the wait, and that is the whole
    # point of where it sits. With the `try` outside it, the context manager
    # restored the default dispositions *before* either handler ran -- and
    # `_kill_and_drain` is a grace, a drain and a reap, so for up to twenty
    # seconds of every timeout and every interrupt litetune was unprotected.
    # Reproduced: a third signal arriving in that window killed litetune
    # mid-kill and the stage outlived its parent, which is precisely what the
    # guard exists to prevent.
    #
    # The group is read here, once, while the child is certainly alive. By the
    # time a kill runs the leader may be a zombie, and macOS will not report
    # the group of one -- which made the group unreachable in exactly the case
    # the sweep exists for. Recovering it from `proc.pid` later was the other
    # way, and it is unsafe: a host using `SA_NOCLDWAIT` has the kernel reap
    # without `Popen` or `getsignal` ever knowing, and the number is then free
    # for somebody else.
    pgid = _group_of(proc)
    with _kill_child_if_we_are_told_to_exit(proc, pgid):
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as expired:
            # Kill first, then drain: reading first would wait on a pipe the
            # group is still holding open. What `communicate` already read is
            # handed over rather than re-read -- those bytes are off the
            # descriptors now, and `_PipeDrain` says where they are found.
            out, err = _kill_and_drain(proc, pgid, (expired.output, expired.stderr))
            raise subprocess.TimeoutExpired(argv, timeout, output=out, stderr=err) from None
        except BaseException:
            # Ctrl-C reaches the group through the terminal only while the
            # child shares it, and `start_new_session=True` is what stopped it
            # doing so. This is that signal's replacement, and it covers every
            # other way out of the `try` as well.
            #
            # `seed=False` for the same reason the handler uses it: this
            # result is discarded -- the exception is on its way out and no
            # caller reads a stage's output on this path -- and seeding means
            # a `b"".join` over the whole transcript, which is the one thing
            # in the drain's constructor that can raise.
            _kill_and_drain(proc, pgid, seed=False)
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _translate_newlines(chunk: str | bytes | None, encoding: str) -> str:
    """Decode, and end lines the way a completed `communicate` would.

    Spelled out rather than borrowed: `Popen._translate_newlines` is private,
    and reaching into it would tie this to a name CPython owes nobody. It is
    two statements there and two operations here, and the pair is checked
    against it by test rather than by assertion.
    """
    return _as_text(chunk, encoding).replace("\r\n", "\n").replace("\r", "\n")


def _as_text(chunk: str | bytes | None, encoding: str = "utf-8") -> str:
    """`TimeoutExpired` carries bytes from a text-mode pipe on some paths.

    `_check_timeout` attaches the raw chunks without decoding even when the
    pipe was opened in text mode, so the decode has to happen here -- in the
    encoding the pipe was actually opened with, or the salvaged stderr this
    exists to preserve comes back as mojibake on a non-UTF-8 host.
    """
    if chunk is None:
        return ""
    return chunk if isinstance(chunk, str) else chunk.decode(encoding, "surrogateescape")


def _describe(proc: subprocess.Popen) -> str:
    """The first word of a subprocess's command line, for a log message."""
    args = proc.args
    if isinstance(args, str | bytes | os.PathLike):
        return os.fsdecode(args)
    return os.fsdecode(next(iter(args), "the stage subprocess"))


def _child_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a stage subprocess runs in: the host's, minus the reach.

    Callers that need their own variables pass them in `overrides`; they are
    applied after the sanitation, so a caller can still set `PYTHONPATH`
    deliberately -- `export`'s repack does not, but the shape leaves the door
    open for one that must, rather than making every caller rebuild this.
    """
    # Only what is actually being taken away: a caller that passes one of these
    # as an override gets it, so naming it as dropped would state the opposite
    # of what happens.
    dropped = [k for k in _HOST_OVERRIDES if k in os.environ and k not in (overrides or {})]
    # Said out loud rather than done quietly -- but once. `run` is called per
    # prompt during evaluation, and Colab sets `PYTHONPATH` by default, so a
    # per-call warning is several hundred identical lines through the middle of
    # a progress report on the platform this tool names as supported.
    unreported = [k for k in dropped if k not in _REPORTED_DROPS]
    if unreported:
        _REPORTED_DROPS.update(unreported)
        logger.warning(
            "not passing %s to the stage subprocess: it would outrank the "
            "environment's own pins, which every measurement is recorded against",
            ", ".join(unreported),
        )
    env = {k: v for k, v in os.environ.items() if k not in _HOST_OVERRIDES}
    # The other half of `_run_guarded`'s `encoding="utf-8"`. A Python child
    # writing to a pipe encodes with the locale's encoding -- the ANSI code
    # page on Windows -- and litert-lm's CLI prints a generation through it. A
    # character the page cannot hold raises inside that CLI, whose own handler
    # prints "An error occurred" and returns zero, so litetune scores the
    # apology as the model's answer. UTF-8 mode also settles the encoding of
    # every file the generated stage scripts read and write, which is the same
    # question one layer down. Applied before `overrides`, so a caller who
    # means to choose something else still can.
    # `k not in overrides` for the same reason the drop above has it: a caller
    # who chose this key gets it, and saying it was not passed would state the
    # opposite of what happens.
    replaced = [
        k
        for k, v in _CHILD_TEXT
        if k in os.environ and k not in (overrides or {}) and os.environ[k] != v
        if k not in _REPORTED_DROPS
    ]
    if replaced:
        _REPORTED_DROPS.update(replaced)
        logger.warning(
            "not passing %s to the stage subprocess: the text a stage hands back is a "
            "measurement, and its encoding is litetune's to fix rather than the machine's",
            ", ".join(sorted(replaced)),
        )
    env.update(_CHILD_TEXT)
    # An override wins, as every override does -- but the parent's pipe is
    # opened UTF-8 whatever the child is told, so a caller choosing another
    # encoding splits the two sides, and what comes back is mojibake in a
    # scored generation. Nothing in this package does it; a caller that starts
    # hears about it rather than finding out from the numbers.
    # Only a different *codec* splits the two sides. `PYTHONIOENCODING` wins
    # over UTF-8 mode, so an override of `PYTHONUTF8` alone leaves the child's
    # stdout UTF-8 and is nobody's business here; an error handler is a choice
    # about what the child does with text it cannot encode, not about the bytes
    # the parent will read.
    chosen = (overrides or {}).get("PYTHONIOENCODING")
    if chosen is not None and codecs.lookup(chosen.split(":", 1)[0]) != codecs.lookup("utf-8"):
        logger.warning(
            "PYTHONIOENCODING=%s was overridden for this stage: the pipe litetune reads is "
            "decoded as UTF-8 whatever the child writes, so a different encoding on one side "
            "is mojibake in whatever the stage hands back",
            chosen,
        )
    if overrides:
        env.update(overrides)
    return env


# The stage environments are built from the *host* interpreter, so its version
# is theirs -- and each has its own ceiling, set by whichever of its pins stops
# publishing wheels first. Past that, pip falls back to a source build and fails
# after minutes of compiler output that never names the version.
#
# One shared ceiling was tried and was wrong: it refused `train` on 3.13 for a
# `numpy==2.0.2` limit that `train` does not have, in a project whose README
# said `tune` works there. The constraint belongs to the pin, so it is recorded
# with the pin.
STAGE_PYTHON_MIN = (3, 10)


def _try_lock(handle) -> bool | None:
    """One non-blocking attempt. True taken, False held elsewhere, None no locking.

    Imported here rather than at module scope: `fcntl` does not exist on
    Windows, and an unconditional import made `import litetune.envs` -- and so
    the whole CLI -- fail there. This module has always had `os.name == "nt"`
    branches, so that platform is in scope.
    """
    try:
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        except OSError:
            # Advisory locking is unavailable on some network filesystems.
            return None
    except ImportError:
        pass
    if sys.platform == "win32":
        # A platform check rather than a try/except, so the type checker narrows
        # instead of reporting `msvcrt` has no attributes on a POSIX host.
        try:
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    return None


def _refuse_unsupported_interpreter(env: StageEnv) -> None:
    """Say which interpreter is needed, before pip spends minutes not saying it.

    Named per environment, because the ceilings differ: `numpy==2.0.2` stops at
    cp312 and `torch==2.5.1` at cp313, so `convert` and `verify` refuse a
    3.13 host that `tune` is fine on. A message naming the wrong pin sends the
    reader to fix the wrong thing.
    """
    current = sys.version_info[:2]
    if current < STAGE_PYTHON_MIN:
        raise RuntimeError(
            f"environment {env.name!r} needs at least python"
            f"{STAGE_PYTHON_MIN[0]}.{STAGE_PYTHON_MIN[1]}; this is python"
            f"{current[0]}.{current[1]}"
        )
    if env.python_ceiling is None or current <= env.python_ceiling:
        return
    ceiling = env.python_ceiling
    raise RuntimeError(
        f"environment {env.name!r} cannot be provisioned on python"
        f"{current[0]}.{current[1]}: it pins {env.ceiling_pin}, which publishes wheels up to "
        f"python{ceiling[0]}.{ceiling[1]}, and stage environments are built from the "
        "interpreter running litetune. Run litetune under one of those (Colab's default "
        "works), or install it into a virtualenv on one."
    )


@contextlib.contextmanager
def _provision_lock(path: Path, timeout: float = PROVISION_TIMEOUT_S):
    """Serialise provisioning of one environment across processes.

    A file lock rather than a directory sentinel: a sentinel left behind by a
    killed process blocks every later run, while a file lock is released by the
    kernel when the holder dies.

    Where no lock can be taken this proceeds unserialised -- which is what this
    did before the lock existed, and is better than refusing to run -- but says
    so, because "the environments raced" is otherwise indistinguishable from a
    corrupt install.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / f"{path.name}.lock"
    deadline = time.monotonic() + timeout
    with open(lock, "w") as handle:
        while True:
            taken = _try_lock(handle)
            if taken is True:
                break
            if taken is None:
                logger.warning(
                    "no advisory locking for %s; concurrent provisioning of %s is unserialised",
                    lock,
                    path.name,
                )
                break
            if time.monotonic() >= deadline:
                # Bounded, because the first version blocked forever on
                # `LOCK_EX` while the docstring advertised a timeout that only
                # covered pip. A holder that hangs must not hang everyone.
                raise RuntimeError(
                    f"waited {timeout}s for another process to finish provisioning "
                    f"{path.name} (lock: {lock}); it is still holding the lock"
                )
            time.sleep(0.5)
        yield


@dataclass(frozen=True)
class CachedEnv:
    """One directory in the environment cache, and what is known about it."""

    path: Path
    bytes: int
    ready: bool
    # The stage it belongs to, when the name still matches one this version
    # defines. A directory whose identity no longer matches any stage is not
    # junk -- it is what an older litetune, or a different interpreter, built --
    # so it is reported as unclaimed rather than as an error.
    stage: str | None

    @property
    def claimed(self) -> bool:
        return self.stage is not None


def _tree_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            # A file that vanished mid-walk is not a reason to refuse a total;
            # it is a reason for the total to be approximate, which it always is.
            continue
    return total


def cached_environments() -> list[CachedEnv]:
    """Everything in the cache, largest first.

    There is no other way to find out. The path is not in the documentation, the
    sizes differ by an order of magnitude between stages, and a failed provision
    leaves a directory behind that looks exactly like a working one from the
    outside.
    """
    root = _cache_root()
    if not root.is_dir():
        return []
    claimed = {env.path.name: env.name for env in (RUNTIME, TRAIN, EXPORT)}
    out = [
        CachedEnv(
            path=child,
            bytes=_tree_bytes(child),
            # The same two-part test `StageEnv.ready` applies, so the listing
            # and the stage cannot disagree about the same directory. Reading
            # only the marker made `litetune env` print `ready` for one holding
            # nothing else, which is where this was first seen.
            ready=(child / ".litetune-ready").is_file() and _interpreter_in(child).exists(),
            stage=claimed.get(child.name),
        )
        for child in sorted(root.iterdir())
        if child.is_dir()
    ]
    return sorted(out, key=lambda e: e.bytes, reverse=True)


def remove_cached(entries: Sequence[CachedEnv]) -> tuple[int, list[str]]:
    """Delete these directories. Returns the bytes freed and what would not go.

    Failures are collected rather than raised: removing four of five caches and
    saying which one resisted is more useful than stopping at the first.
    """
    freed = 0
    failures: list[str] = []
    for entry in entries:
        try:
            shutil.rmtree(entry.path)
            freed += entry.bytes
        except OSError as exc:
            failures.append(f"{entry.path}: {exc}")
    return freed, failures


@dataclass(frozen=True)
class StageEnv:
    """A named, pinned environment that stage commands run inside."""

    name: str
    requirements: tuple[str, ...]
    # System packages this environment needs that pip cannot provide. Recorded
    # so a missing one produces a named diagnosis rather than a dlopen error.
    system_requirements: tuple[str, ...] = field(default=())
    # The highest Python this environment's pins publish wheels for, and which
    # pin sets it. `None` means no pin here has a known ceiling. Recorded rather
    # than derived: pip only tells you by failing to build from source, minutes
    # in, without naming the version.
    python_ceiling: tuple[int, int] | None = None
    ceiling_pin: str = ""

    def __post_init__(self) -> None:
        unpinned = [r for r in self.requirements if not _is_pinned(r)]
        if unpinned:
            raise UnpinnedRequirement(
                f"environment {self.name!r} has unpinned requirements: {unpinned}. "
                "Pin them with '==': an unchanged definition that resolves "
                "differently over time silently changes pipeline behaviour."
            )

    @property
    def identity(self) -> str:
        """Content hash of the definition. Participates in stage cache keys."""
        payload = "\n".join((sys.version.split()[0], *sorted(self.requirements)))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def path(self) -> Path:
        return _cache_root() / f"{self.name}-{self.identity}"

    @property
    def python(self) -> Path:
        return _interpreter_in(self.path)

    @property
    def ready(self) -> bool:
        """The install finished *and* the tree it finished into is still here.

        The marker alone is not enough. A directory holding nothing but
        `.litetune-ready` was found in a real cache (12 bytes, no `bin/`):
        `provision` short-circuits on it and builds nothing, and `run` then
        reaches a python that does not exist and reports "it could not be
        started" -- which names the wrong cause and sends the reader looking at
        their toolchain instead of at an empty directory.

        Checking `python` rather than the marker's contents is deliberate: the
        path already carries the identity hash, so a marker here cannot belong
        to a different build. What is missing is the tree, not the identity.

        A file, not merely something by that name. `provision` only ever
        writes it with `write_text`, so a directory there was put there by
        something else -- and `.exists()` answered yes for it, which let an
        environment nobody finished read as ready.
        """
        return (self.path / ".litetune-ready").is_file() and self.python.exists()

    def provision(
        self, events=None, force: bool = False, timeout: int = PROVISION_TIMEOUT_S
    ) -> Path:
        """Create the environment if absent. Idempotent, and safe to race.

        Two invocations sharing one cache directory used to be able to see the
        same unready environment and both proceed -- one `rmtree`-ing the venv
        the other was installing into. The lock makes the second wait and then
        find the first one's work, which is what "idempotent" was already
        claiming.

        `timeout` is not optional in practice: this was the only unbounded
        external call in the package, and a stalled `pip install torch` hung
        forever with no output.
        """
        if self.ready and not force:
            return self.path
        _refuse_unsupported_interpreter(self)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One budget, not two. Passing `timeout` to both the wait and the
        # install let a caller who waited the full 1800s then get another
        # 1800s of pip -- an advertised half-hour limit taking an hour.
        deadline = time.monotonic() + timeout
        with _provision_lock(self.path, timeout):
            # Re-check inside the lock: the holder we waited for may have just
            # built exactly what we came to build.
            if self.ready and not force:
                return self.path
            if events:
                events.note(
                    f"provisioning environment {self.name!r} ({len(self.requirements)} pinned)"
                )
            # A virtualenv cannot be built somewhere and moved here. `venv`
            # writes the absolute path of the interpreter *at creation time*
            # into the shebang of every console script it installs, and nothing
            # rewrites them on a rename. Building in `.incoming` and renaming
            # into place therefore produced an environment whose `pip`,
            # `litert-torch` and `litert-lm` all died with `bad interpreter` --
            # naming a path truncated at the kernel's 127-byte shebang limit,
            # so the message pointed at a path that had never existed. Every
            # `convert` and every `verify` failed, and because `StageEnv.run`
            # prefers a console script whenever the file is present, the
            # `python -m` fallback below never got a chance.
            #
            # So the new environment is built at its final path, and it is the
            # *old* one that steps aside. Same invariant, opposite direction:
            # nothing is removed until a replacement exists, and the
            # replacement is born where it will live.
            previous = self.path.parent / f".{self.path.name}.previous"
            moved_aside = False
            if self.path.exists():
                shutil.rmtree(previous, ignore_errors=True)
                self.path.rename(previous)
                moved_aside = True
            elif previous.exists():
                # A provision killed between the two renames left the working
                # environment here. Reclaim it rather than leaving gigabytes
                # nobody looks at; if this build succeeds it is replaced
                # anyway, and if it fails the restore below wants it.
                moved_aside = True
            try:
                # `EnvBuilder.create` spawns `ensurepip` as its own subprocess
                # and takes no timeout, so the budget covers what follows it and
                # not this. Named rather than left as a silent hole.
                venv.EnvBuilder(with_pip=True, clear=True).create(self.path)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"provisioning environment {self.name!r} exhausted its {timeout}s "
                        "budget building the virtualenv; nothing was installed, and any "
                        "existing environment is untouched"
                    )
                # The same spelling `ready` and `run` use, from one function:
                # two of them disagreeing on Windows is what made this worth
                # naming at all.
                cmd = [str(self.python), "-m", "pip", "install", "--quiet", *self.requirements]
                try:
                    # Through the same guard the stages use, and for the same
                    # reason. `subprocess.run` kills its direct child, and
                    # only from an `except`: a SIGTERM to litetune raises
                    # nothing at all, and a timeout reaches pip and not the
                    # compilers pip spawned. Half an hour of `pip install
                    # torch` therefore carried on writing into a directory the
                    # restore below had already decided to throw away.
                    #
                    # `_run_guarded` also puts `PIP_TARGET` and `PIP_PREFIX`
                    # out of reach, which would otherwise install somewhere
                    # other than this venv while `.litetune-ready` was written
                    # over it regardless.
                    proc = _run_guarded(cmd, remaining)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"provisioning environment {self.name!r} exceeded its {timeout}s budget "
                        "(shared between waiting for another provisioner and the install "
                        "itself); the install may be waiting on the network"
                    ) from None
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"could not provision environment {self.name!r} on "
                        f"{sys.platform}/python{sys.version_info.major}."
                        f"{sys.version_info.minor}. It pins {', '.join(self.requirements)}"
                        + (
                            " and needs the system package(s) "
                            + ", ".join(self.system_requirements)
                            if self.system_requirements
                            else ""
                        )
                        + ". The export and runtime toolchains are published for Linux; on "
                        "other platforms the install fails here rather than later. pip "
                        f"said:\n{proc.stderr[-2000:]}"
                    )
                # Written last, and the only thing `ready` consults: an
                # environment interrupted anywhere above this line has no
                # marker, so the next run rebuilds it instead of trusting a
                # half-installed tree.
                (self.path / ".litetune-ready").write_text(self.identity, encoding="utf-8")
            except BaseException:
                # Put the working environment back. This covers the timeout and
                # pip-failure paths above, and Ctrl-C, which is why it catches
                # BaseException rather than Exception.
                if moved_aside:
                    shutil.rmtree(self.path, ignore_errors=True)
                    try:
                        previous.rename(self.path)
                    except OSError as restore_failure:
                        raise RuntimeError(
                            f"provisioning environment {self.name!r} failed, and the working "
                            f"environment could not be moved back from {previous}. It is intact "
                            "there; move it into place by hand, or delete it and let the next "
                            f"run rebuild. Restore failed with: {restore_failure}"
                        ) from restore_failure
                raise
            else:
                if moved_aside:
                    shutil.rmtree(previous, ignore_errors=True)
        if events:
            events.note(f"environment {self.name!r} ready at {self.path}")
        return self.path

    def run(
        self,
        args: list[str],
        timeout: int = 3600,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a console script or module inside this environment.

        A console script is preferred over `python -m` wherever the file is
        present, because that is what `convert` and `verify` invoke
        `litert-torch` and `litert-lm` as.

        Everything about how the process is started, killed and read is in
        `_run_guarded`, which `provision` shares -- including why the child
        gets a session of its own, what `env` may and may not reach, and why
        the decode replaces bad bytes rather than raising.

        Returns the completed process rather than raising on non-zero:
        callers decide whether a non-zero exit is a failed check or an
        unperformed one, and that distinction is the whole point of
        litetune.checks.
        """
        exe = _console_script_in(self.path, args[0])
        argv = [str(exe), *args[1:]] if exe is not None else [str(self.python), "-m", *args]
        return _run_guarded(argv, timeout, env)


# Where a stage's subprocess will run is decided here, once, in the parent,
# before the subprocess starts rather than inside it. Both generated scripts
# (`tune._TRAIN_SCRIPT`, `evaluate._HF_GENERATE_SCRIPT`) still carry a
# `torch.cuda.is_available()` fallback, and it is reached only when this could
# not answer or was never asked -- see `training_device` and
# `generation_device`, which take the parent's answer when there is one.
# Asking here is what lets the parent say something about a slow combination
# before a six-hour run instead of reading the device out of a metrics file
# afterwards.
DEVICE_PROBE_TIMEOUT_S = 30

# Sub-second: import torch, ask it three questions, print one JSON line.
# Nothing else this package runs inside a stage environment is this cheap,
# which is what makes asking it *before* the real, expensive subprocess
# worthwhile.
#
# `is_available()` alone cannot tell "this machine has no GPU" from "this
# torch cannot reach the one it has": a CPU-only wheel, a container started
# without `--gpus` and a driver too old for the runtime all answer False. So
# the build's CUDA version and the device count come back too, and
# `DeviceProbe` says which of the two it saw.
_DEVICE_PROBE_CODE = (
    "import json, torch; "
    "print(json.dumps({"
    "'device': 'cuda' if torch.cuda.is_available() else 'cpu', "
    "'cuda_build': torch.version.cuda, "
    "'device_count': torch.cuda.device_count()}))"
)


@dataclass(frozen=True)
class DeviceProbe:
    """What one environment's own torch answered about its accelerator.

    `device` is the part a caller acts on: `"cuda"`, `"cpu"`, or `None`. `None`
    is not a device and must not be read as one -- `tune.TrainingMetrics.device`
    defaults to `None` for the same reason ("absent is absent, not 'cpu'").
    `detail` says which `None` it is, because they are different facts: the
    probe was never asked, it could not be started, it was killed, it timed
    out, it exited non-zero, or it answered something unusable.
    """

    device: str | None
    detail: str
    # Whether a probe was run at all. Three states share `device is None` and
    # they are different facts: nobody asked, it was asked and could not
    # answer, and there was nothing to ask. Only the middle one implies a
    # generation script will start and write a run report, and a consumer that
    # has to tell them apart should read this rather than the wording of
    # `detail` -- a sentence built somewhere else is not a protocol.
    attempted: bool = True
    # `torch.version.cuda`: the CUDA version this wheel was built against, or
    # `None` for a CPU-only wheel. Reported so that "cpu" from a CUDA build --
    # torch is there and cannot reach a device -- is distinguishable from "cpu"
    # from a CPU-only build, which is the machine having none.
    cuda_build: str | None = None
    device_count: int | None = None

    @property
    def answered(self) -> bool:
        return self.device is not None

    @property
    def cuda_build_without_a_device(self) -> bool:
        """A CUDA build that reports no usable device: torch cannot reach one."""
        return self.device == "cpu" and bool(self.cuda_build)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "detail": self.detail,
            "cuda_build": self.cuda_build,
            "device_count": self.device_count,
        }


# What a caller gets when it never ran a probe at all. Distinct from every
# unanswered probe below, because "nobody asked" and "it was asked and could
# not say" are different facts about a run and the report has to carry which.
NOT_PROBED = DeviceProbe(device=None, detail="no device probe was run")


def _unanswered(env: StageEnv, reason: str, events=None) -> DeviceProbe:
    detail = f"the device probe on environment {env.name!r} could not answer: {reason}"
    logger.warning("%s", detail)
    if events is not None:
        events.note(detail, environment=env.name)
    return DeviceProbe(device=None, detail=detail)


def resolve_device(
    env: StageEnv, timeout: int = DEVICE_PROBE_TIMEOUT_S, events=None
) -> DeviceProbe:
    """Ask this environment's own torch whether it has CUDA.

    Not a guess and not a default: if the probe cannot be started, is killed,
    times out, exits non-zero, or answers with anything other than exactly
    "cuda" or "cpu", `DeviceProbe.device` is `None` rather than a quiet "cpu",
    and `DeviceProbe.detail` carries which of those happened. A caller that
    gets `None` back still has to run something; the training and generation
    scripts fall back to asking `torch.cuda.is_available()` themselves in that
    case, which is the one situation left where they decide for themselves
    rather than being told.

    `events`, when given, is where an unanswered probe is reported. `logging`
    alone reaches nothing the report can carry, and a `None` device that is
    never explained is indistinguishable from a device nobody asked about.
    """
    try:
        proc = env.run(["python", "-c", _DEVICE_PROBE_CODE], timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # Distinguished from the start-failure wording below: a hang read as
        # "did not run" looks identical to a process that never started at
        # all, and a reader debugging a wedged environment needs to know this
        # one ran for the full `timeout` seconds before anything gave up on it.
        return _unanswered(
            env, f"it timed out after {timeout}s ({type(exc).__name__}: {exc})", events
        )
    except (OSError, ValueError) as exc:
        # `ValueError` alongside `OSError`: `env.run`'s `text=True` decode
        # raises `UnicodeDecodeError` -- a `ValueError`, not an `OSError` -- on
        # a stray non-UTF-8 byte ahead of the probe's own line, which is
        # exactly the "banner before the answer" case the last-non-empty-line
        # rule below exists to survive. `env.run` now passes `errors="surrogateescape"`
        # so that byte no longer raises there, but this catch is what stops
        # this function's contract -- it never raises -- from depending on
        # every caller of `env.run` getting that right.
        return _unanswered(env, f"it could not be started ({type(exc).__name__}: {exc})", events)

    reading = read_returncode(proc.returncode)
    if not reading.conclusive:
        # A killed probe never chose an exit status, so `exited -9` would be a
        # sentence about something that did not happen. See `litetune.exits`.
        return _unanswered(env, reading.describe("this environment"), events)
    if proc.returncode != 0:
        return _unanswered(
            env,
            f"it exited {proc.returncode} ({(proc.stderr or '').strip()[-200:] or 'no stderr'})",
            events,
        )

    # The last non-empty line, not the whole of stdout: a stage environment is
    # free to print a banner, a deprecation warning or a loader message ahead
    # of the answer, and the answer is the last thing the probe writes. Reading
    # all of stdout threw away a good answer because something else spoke
    # first.
    lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    if not lines:
        return _unanswered(env, "it printed nothing on stdout", events)
    try:
        answer = json.loads(lines[-1])
        device = answer["device"]
        cuda_build = answer["cuda_build"]
        device_count = answer["device_count"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return _unanswered(
            env, f"its last stdout line is not an answer ({exc}): {lines[-1][:200]!r}", events
        )
    if device not in ("cuda", "cpu"):
        return _unanswered(env, f"it answered {device!r}, which is neither device", events)

    probe = DeviceProbe(
        device=device,
        detail=f"this environment's torch reports {device}",
        cuda_build=cuda_build if isinstance(cuda_build, str) else None,
        device_count=device_count if isinstance(device_count, int) else None,
    )
    if probe.cuda_build_without_a_device:
        probe = DeviceProbe(
            device=device,
            detail=(
                f"this environment's torch is a CUDA {probe.cuda_build} build reporting "
                f"{probe.device_count} devices, so it will run on the CPU: torch cannot reach a "
                "GPU here, which is not the same observation as this machine having none"
            ),
            cuda_build=probe.cuda_build,
            device_count=probe.device_count,
        )
        if events is not None:
            events.note(probe.detail, environment=env.name, device=device)
    return probe


# ---------------------------------------------------------------------------
# The environments litetune actually uses.
#
# Versions are the ones every measurement in the README was produced with. They
# are part of the result, not maintenance trivia: changing them re-runs the
# suite rather than being a routine bump.
# ---------------------------------------------------------------------------

RUNTIME = StageEnv(
    name="runtime",
    python_ceiling=(3, 12),
    ceiling_pin="numpy==2.0.2",
    requirements=(
        "litert-lm==0.16.1",
        "numpy==2.0.2",  # last of the 2.0 line; `<2.1` is a bound, not a pin
    ),
    # litert-lm dlopen()s a native library that links vulkan unconditionally,
    # even for the CPU backend. Without it every invocation -- `--help`
    # included -- dies in under a second.
    system_requirements=("libvulkan1",),
)

EXPORT = StageEnv(
    name="export",
    python_ceiling=(3, 12),
    ceiling_pin="numpy==2.0.2",
    requirements=(
        "litert-torch-nightly==0.10.0.dev20260826",
        "litert-lm==0.16.1",
        # Provides `litert-lm-builder` and `litert-lm-peek`, which `export`
        # runs to write the GPU activation type into each bundle and to read
        # the result back. Pinned by name: `litert-lm` happens to require
        # this exact version today and `litert-torch` accepts any, so without
        # this line the builder floats the moment the other two pins move.
        "litert-lm-builder==0.16.1",
        "numpy==2.0.2",  # last of the 2.0 line; `<2.1` is a bound, not a pin
    ),
    system_requirements=("libvulkan1",),
)

# transformers 5.16.1, not the 4.57.3 the reference notebooks pin.
#
# That pin was taken for FunctionGemma reproducibility and quietly became a
# ceiling on which models exist at all: every 4.x release from 4.55.0 to 4.57.6
# crashes loading a Gemma 4 or Qwen3.5 tokenizer, because `extra_special_tokens`
# ships as a list where 4.x calls `.keys()` on it. Fixed in 5.0.0, never
# backported. Each family's `min_transformers` in `models.py` encodes the rule;
# leaving the pin below
# it would have made litetune fail its own check.
#
# Raising it changes the measured baseline, so it was not done on reasoning. A
# six-model probe re-ran the three previously measured families on 5.16.1 —
# gemma3_text, qwen2 and qwen3 all stayed alive on both recipes with
# byte-identical artifacts — and Qwen3.5, which had failed at tokenizer load,
# passed the full LoRA→merge→export→liveness path.
TRAIN = StageEnv(
    name="train",
    python_ceiling=(3, 13),
    ceiling_pin="torch==2.5.1",
    requirements=(
        "torch==2.5.1",
        "transformers==5.16.1",
        "peft==0.20.0",
        "sentencepiece==0.2.0",
    ),
)
