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

import contextlib
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import venv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune.exits import read_returncode


class UnpinnedRequirement(ValueError):
    """Raised when an environment definition would float."""


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
# pinned package, which is the same false provenance one layer down.
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

# How long to wait for the pipes after killing a process group. A grandchild
# that called `setsid()` itself escapes the group, and if it still holds the
# inherited stdout it can keep an unbounded `communicate()` open forever --
# which would put back the unbounded wait this whole change exists to remove,
# one level out from where the original bug was.
_DRAIN_AFTER_KILL_S = 10

# How long a stage gets between SIGTERM and SIGKILL. Long enough for torch to
# finish a `save_pretrained` it had started, short enough that a wedged process
# cannot meaningfully extend the timeout it already blew through.
_TERM_GRACE_S = 5


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
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return None
    return None if pgid == os.getpgid(0) else pgid


def _signal_tree(proc: subprocess.Popen, pgid: int | None, sig: int) -> bool:
    """Deliver `sig` to the group if there is one, else to the child alone.

    A process that has already gone counts as delivered: between a timeout
    firing and this call is exactly where a process finishes on its own, and
    that is the ordinary case rather than a failure.
    """
    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return True
        except OSError:
            # `PermissionError` among them: a setuid child cannot be signalled
            # by its parent. Fall through and try the child directly.
            pass
    try:
        proc.send_signal(sig)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _kill_tree(proc: subprocess.Popen, grace: float = _TERM_GRACE_S) -> bool:
    """End the child and everything it spawned: SIGTERM, then SIGKILL.

    Returns whether a signal was delivered, so a caller can say "and it could
    not be killed" rather than leaving a leaked process to be diagnosed later
    as somebody else's out-of-memory kill.

    SIGTERM first because this replaced something gentler. Before the child had
    a session of its own, Ctrl-C reached it as SIGINT through the terminal and
    torch got to run its own handlers -- which for `tune` is the difference
    between a checkpoint written and a file truncated mid-`save_pretrained`.
    Going straight to SIGKILL would have been a quiet downgrade of that, so the
    grace is the part that keeps the replacement honest; SIGKILL still follows,
    because a wedged process must not be able to outlast its own timeout.
    """
    # `Popen.send_signal` makes this check for the same reason, and the reason
    # is severe: once the child has been reaped its pid is free for the kernel
    # to hand out again, and `killpg` on a recycled pid signals a stranger's
    # process group. The timeout path cannot reach that state, but the
    # `BaseException` path can -- a KeyboardInterrupt landing inside
    # `communicate`'s trailing `wait()` arrives after the reap.
    if proc.returncode is not None:
        return True
    # Read once, while the child is certainly unreaped: after the wait below it
    # may be gone, and `getpgid` on its pid would then answer about whoever
    # inherited the number.
    pgid = _group_of(proc)
    delivered = _signal_tree(proc, pgid, signal.SIGTERM)
    if delivered and grace > 0:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace)
    # Unconditionally, even where the child exited on the SIGTERM: what it
    # spawned is not covered by its own exit, and this is the signal that
    # collects them.
    return _signal_tree(proc, pgid, signal.SIGKILL) or delivered


@contextlib.contextmanager
def _kill_child_if_we_are_told_to_exit(proc: subprocess.Popen):
    """Take the child down with us on SIGTERM or SIGHUP.

    `start_new_session=True` is what lets a timeout kill everything the stage
    spawned, but it also took the child out of the terminal's process group --
    and a hangup on that group, an SSH session dropping or a window closing, is
    how an interactive run used to be cleaned up. Python turns only SIGINT into
    an exception, so the `except BaseException` in `run` cannot stand in for
    these two: without this, `kill <litetune>` or a dropped connection leaves a
    six-hour torch run holding its memory, which is the very thing the process
    group was introduced to prevent.

    The previous disposition is restored before the signal is re-raised, so the
    process still dies the way it would have. Signals can only be installed on
    the main thread; elsewhere this is a no-op rather than an error, because a
    library that refused to run off the main thread would be worse than one
    that cleans up in fewer cases.
    """
    installed: dict[int, Any] = {}

    def take_the_child_with_us(signum, _frame):
        _kill_tree(proc)
        # `None` means the handler in place was installed from C and cannot be
        # restored from Python -- `signal.signal(sig, None)` is a `TypeError`.
        # Falling back to the default is what "die the way we would have" means
        # when the previous disposition is not expressible here.
        signal.signal(signum, installed.get(signum) or signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        # Reached only if that disposition was `SIG_IGN`, or a Python handler
        # that returns -- litetune embedded in a larger application. Returning
        # here would resume `communicate()` over a child we just SIGKILLed and
        # hand the caller a `-9`, which `exits` reports as the out-of-memory
        # killer: our own kill, blamed on the machine. Raising keeps the run's
        # own account of itself honest.
        raise KeyboardInterrupt(f"signal {signum} arrived while a stage subprocess was running")

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            installed[sig] = signal.signal(sig, take_the_child_with_us)
        except (ValueError, OSError):
            # Not the main thread, or the platform has no such signal.
            pass
    try:
        yield
    finally:
        for sig, previous in installed.items():
            if previous is None:
                # Same `TypeError` as above, and here it would fire inside a
                # `finally` and mask whatever was already propagating.
                continue
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, previous)


def _kill_and_drain(proc: subprocess.Popen) -> tuple[str, str]:
    """Kill the group, then collect what it wrote -- without waiting forever.

    The drain is bounded because the kill is not guaranteed to have reached
    everything: a grandchild that called `setsid()` is outside the group and
    can hold the inherited stdout open. Giving up on the output is the right
    trade here, since the caller is already on its way to reporting a timeout
    or re-raising; hanging instead would reinstate the unbounded wait.
    """
    killed = _kill_tree(proc)
    try:
        return proc.communicate(timeout=_DRAIN_AFTER_KILL_S)
    except subprocess.TimeoutExpired as expired:
        # Said out loud, because a leaked process is otherwise diagnosed hours
        # later as somebody else's out-of-memory kill.
        logger.warning(
            "output pipes stayed open after %s the process group for %s; something it "
            "spawned has left the group and is still running",
            "killing" if killed else "failing to kill",
            _describe(proc),
        )
        # Whatever was read before giving up. The stage's stderr is the single
        # most useful artifact in a timeout report, so it is worth carrying the
        # partial one rather than returning two empty strings.
        return _as_text(expired.output), _as_text(expired.stderr)


def _as_text(chunk: str | bytes | None) -> str:
    """`TimeoutExpired` carries bytes from a text-mode pipe on some paths."""
    if chunk is None:
        return ""
    return chunk if isinstance(chunk, str) else chunk.decode("utf-8", "replace")


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
    dropped = [k for k in _HOST_OVERRIDES if k in os.environ]
    if dropped:
        # Said out loud rather than done quietly. A user who set `PYTHONPATH`
        # following a workaround in an issue thread would otherwise get numbers
        # the workaround does not predict, with nothing anywhere recording that
        # the variable existed. `PYTHONSTARTUP` is excluded from the message
        # because it does nothing to a non-interactive child either way.
        named = [k for k in dropped if k != "PYTHONSTARTUP"]
        if named:
            logger.warning(
                "not passing %s to the stage subprocess: it would outrank the "
                "environment's own pins, which every measurement is recorded against",
                ", ".join(named),
            )
    env = {k: v for k, v in os.environ.items() if k not in _HOST_OVERRIDES}
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
            ready=(child / ".litetune-ready").exists() and _interpreter_in(child).exists(),
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
        """
        return (self.path / ".litetune-ready").exists() and self.python.exists()

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
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=remaining,
                        # The timeout message names a prompt as a cause; this is
                        # what stops one from being waited on at all.
                        stdin=subprocess.DEVNULL,
                        # `PIP_TARGET`/`PIP_PREFIX` in the host environment
                        # would install somewhere other than this venv, and
                        # `.litetune-ready` would still be written over it.
                        env=_child_env(),
                    )
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
                (self.path / ".litetune-ready").write_text(self.identity)
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

        Returns the completed process rather than raising on non-zero: callers
        decide whether a non-zero exit is a failed check or an unperformed one,
        and that distinction is the whole point of litetune.checks.

        `errors="replace"` on the decode: `text=True` decodes stdout/stderr
        eagerly, and a stray non-UTF-8 byte -- a CUDA or driver banner ahead of
        a probe's own JSON line, in the same class of weird environment the
        last-non-empty-line rule in `resolve_device` exists to survive --
        raised `UnicodeDecodeError` there uncaught, which is a `ValueError`,
        not the `OSError` every caller of this method was written to expect.
        That escaped `resolve_device` and both of its callers and ended a
        `tune` or `verify` run with a traceback instead of an unanswered
        probe. Replacing the byte keeps the rest of the line readable, which
        is what a caller that only reads the last line needs; this is shared
        by every stage, so the fix protects all of them, not only the probe.

        `env` is *overrides*, not a whole environment: the base is the host's
        minus `_HOST_OVERRIDES`, so a caller cannot accidentally hand the pins
        back to `PYTHONPATH` by starting from `os.environ`.

        The process is started in its own session so that a timeout can kill
        what it spawned. `subprocess.run` kills only the direct child, which
        for `tune` means a six-hour timeout records an honest "not checked",
        the run continues, and an abandoned torch process keeps its memory --
        the next stage is then SIGKILLed and `exits` reads that as the OOM
        killer, blaming the machine for litetune's own orphan.
        """
        exe = _console_script_in(self.path, args[0])
        argv = [str(exe), *args[1:]] if exe is not None else [str(self.python), "-m", *args]
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            stdin=subprocess.DEVNULL,
            env=_child_env(env),
            # POSIX only; accepted and ignored on Windows, where `_kill_tree`
            # falls back to killing the child alone. Not overridable: a caller
            # reaching past it with `process_group=` would put the child in
            # *our* group, and `_kill_tree` would then aim SIGKILL at the
            # session that is running litetune. There was a `**kwargs`
            # passthrough here with no user in the package; it is gone.
            start_new_session=True,
        )
        try:
            with _kill_child_if_we_are_told_to_exit(proc):
                out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill first, then drain: `communicate()` with no timeout would
            # otherwise wait on a pipe the process group is still holding open.
            out, err = _kill_and_drain(proc)
            raise subprocess.TimeoutExpired(argv, timeout, output=out, stderr=err) from None
        except BaseException:
            # Ctrl-C reaches the group through the terminal only while the
            # child shares it, and `start_new_session=True` is what stopped it
            # doing so. This is that signal's replacement, and it covers every
            # other way out of the `try` as well.
            _kill_and_drain(proc)
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, out, err)


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
        # rule below exists to survive. `env.run` now passes `errors="replace"`
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
