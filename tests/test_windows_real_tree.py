"""The Windows process model against real processes, on the Windows CI job.

Each test builds a small real tree -- a stage child that starts a grandchild,
which writes its pid to a file and sleeps -- and stops it the way litetune
does. Each also runs the same tree with the job API taken away
(`_windows_job_api` returning None, which is the start every Windows run had
before jobs), in the same test, and asserts the grandchild *survives* that: a
pass cannot then come from a check that could not have failed.

Every process a test needs to judge is held by a handle from the moment its
pid is known, checked to be running and to have been created after the test
began; liveness, exit codes and cleanup all go through that handle, never a
pid read again later. Everything a test started is ended in its `finally`.

Stages run two ways where it matters: on the interpreter running the tests,
and through a virtualenv's `python.exe` as production builds one
(`envs._StageEnvBuilder`, no symlinks on Windows) -- the venv redirector,
which runs the base interpreter as a child of its own, in a job of its own
(`PC/launcher.c` with `VENV_REDIRECT` for 3.10 to 3.12, `PC/venvlauncher.c`
from 3.13, per `PCbuild/venvlauncher.vcxproj`).

The first versions of tests 1 to 6 passed on windows-latest, CPython 3.12, the
toolcache interpreter (PR #77, on 23bfc53), and every test here on 3.10, 3.12
and 3.13 on 3400921. What else is asserted here about Windows is what these
tests exist to measure.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from litetune import envs
from litetune.exits import read_returncode

on_windows = pytest.mark.skipif(
    sys.platform != "win32",
    reason="real job objects, Ctrl-Break and process trees: the windows-latest CI job runs these",
)

# `GetExitCodeProcess`: "If the process has not terminated and the function
# succeeds, the status returned is STILL_ACTIVE"; the note under it gives 259.
STILL_ACTIVE = 259
# Process Security and Access Rights.
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
# `WaitForSingleObject`.
WAIT_OBJECT_0 = 0x00000000
# `FILETIME` counts 100-nanosecond intervals since 1601-01-01; this is
# 1970-01-01 in those units.
FILETIME_AT_UNIX_EPOCH = 116444736000000000

# Long enough for two interpreters to start on a CI runner, which nothing has
# measured; a tree that is not up by then fails the test by saying so.
STARTUP_S = 30.0

# What `envs` logs when a stage falls back or a step of the job fails. A
# real run that logs any of these did not test the job, whatever else it
# showed.
FALLBACKS = (
    "could not create a job object",
    "could not put",
    "could not load litetune.winjob",
    "no job object for",
    "could not resume",
    "could not terminate the job",
    "could not close the job object",
    "cannot count the processes",
    "still holds processes",
    "no Ctrl-Break for",
)

# The grandchild: with "ignore" it ignores Ctrl-Break first, so only the job
# can end it; then its pid where the test will look, written atomically; then
# sleep in short steps.
GRANDCHILD = """
import os, signal, sys, time
if sys.argv[2] == "ignore":
    signal.signal(signal.SIGBREAK, signal.SIG_IGN)
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
while True:
    time.sleep(0.1)
"""

# The stage: optionally a Ctrl-Break handler that writes a marker and exits,
# its own pid on file, the grandchild started without the stage's pipes (so a
# pipe held open is never what keeps a wait going), then sleep in short steps
# so a Python handler gets to run.
CHILD = """
import os, signal, subprocess, sys, time
grand_pidfile, child_pidfile, grand_mode, marker, seconds = sys.argv[1:6]
if marker != "-":
    def bye(signum, frame):
        with open(marker, "w") as f:
            f.write("handled")
        sys.exit(0)
    signal.signal(signal.SIGBREAK, bye)
with open(child_pidfile + ".tmp", "w") as f:
    f.write(str(os.getpid()))
os.replace(child_pidfile + ".tmp", child_pidfile)
subprocess.Popen(
    [sys.executable, "-c", GRANDCHILD, grand_pidfile, grand_mode],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("started", flush=True)
deadline = time.monotonic() + float(seconds)
while time.monotonic() < deadline:
    time.sleep(0.1)
""".replace("GRANDCHILD", repr(GRANDCHILD))

# A stage that says three things on each pipe across several slices, then
# hangs -- with its pipes open, or after closing them.
CHATTY = """
import os, sys, time
started, done, mode = sys.argv[1:4]
open(started, "w").close()
time.sleep(0.5)
for n in (1, 2, 3):
    sys.stdout.write(f"out-{n}\\n")
    sys.stdout.flush()
    sys.stderr.write(f"err-{n}\\n")
    sys.stderr.flush()
    time.sleep(0.6)
if mode == "closed":
    os.close(1)
    os.close(2)
open(done, "w").close()
while True:
    time.sleep(0.1)
"""

SLEEPER = "import time\nwhile True: time.sleep(0.1)"

# A process running `_run_guarded` the way the CLI does, so the test can do to
# it what cannot be done to pytest. `control` takes the job API away. Every
# record `envs` logs goes to stderr, where the test reads it.
DRIVER_PREAMBLE = """
import _thread, logging, os, sys, threading, time
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
from litetune import envs
mode, grand_pidfile, child_pidfile, grand_mode, seconds, timeout = sys.argv[1:7]
if mode == "control":
    envs._windows_job_api = lambda: None
child = [sys.executable, "-c", CHILD, grand_pidfile, child_pidfile, grand_mode, "-", seconds]
""".replace("CHILD", repr(CHILD))

# Waits; `TerminateProcess` on it is the test.
DRIVER_WAITS = (
    DRIVER_PREAMBLE
    + """
envs._run_guarded(child, timeout=float(timeout))
"""
)

# Reports `StageInterrupted` as a marker and exit code 42.
DRIVER_INTERRUPTED = (
    DRIVER_PREAMBLE
    + """
try:
    envs._run_guarded(child, timeout=float(timeout))
except envs.StageInterrupted as exc:
    print("STAGE-INTERRUPTED", exc.signum, flush=True)
    sys.exit(42)
print("NOT-INTERRUPTED", flush=True)
sys.exit(1)
"""
)

# Interrupts its own main thread a second after the grandchild is up, and
# reports when it started, when it interrupted, when `_kill_and_drain` was
# entered and whether the child was still running then; exit code 43.
DRIVER_KEYBOARD = (
    DRIVER_PREAMBLE
    + """
sent = []
entered = []
real_kill_and_drain = envs._kill_and_drain
def noted(proc, *args, **kwargs):
    entered.append((time.monotonic(), proc.poll() is None))
    return real_kill_and_drain(proc, *args, **kwargs)
envs._kill_and_drain = noted
def interrupt_once_the_tree_is_up():
    while not os.path.exists(grand_pidfile):
        time.sleep(0.05)
    time.sleep(1.0)
    sent.append(time.monotonic())
    _thread.interrupt_main()
threading.Thread(target=interrupt_once_the_tree_is_up, daemon=True).start()
t_start = time.monotonic()
try:
    envs._run_guarded(child, timeout=float(timeout))
except KeyboardInterrupt:
    print("T_START", t_start, "SENT", sent[0], "ENTERED", entered[0][0],
          "RUNNING", entered[0][1], flush=True)
    sys.exit(43)
print("NOT-INTERRUPTED", flush=True)
sys.exit(1)
"""
)

# `send_ctrl_break(0)` must be refused before it reaches
# `GenerateConsoleCtrlEvent`; run in a console of its own, so a refusal that
# did not happen ends this process and not pytest.
DRIVER_GROUP_ZERO = """
import sys
from litetune import winjob
try:
    winjob.API.send_ctrl_break(0)
except ValueError:
    print("REFUSED", flush=True)
    sys.exit(0)
print("SENT", flush=True)
sys.exit(1)
"""


def test_the_real_tree_scripts_compile():
    """Caught here rather than by the first Windows run."""
    for source in (
        GRANDCHILD,
        CHILD,
        CHATTY,
        SLEEPER,
        DRIVER_WAITS,
        DRIVER_INTERRUPTED,
        DRIVER_KEYBOARD,
        DRIVER_GROUP_ZERO,
    ):
        compile(source, "<windows tree>", "exec")


# ---------------------------------------------------------------------------
# Processes, held by handle
# ---------------------------------------------------------------------------


def _kernel32() -> Any:
    if sys.platform != "win32":
        raise RuntimeError("kernel32 exists only on Windows")
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.GetExitCodeProcess.restype = wintypes.BOOL
    k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k.TerminateProcess.restype = wintypes.BOOL
    k.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k.GetProcessTimes.restype = wintypes.BOOL
    k.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    return k


def _open(pid: int) -> int:
    if sys.platform != "win32":
        raise RuntimeError("kernel32 exists only on Windows")
    handle = _kernel32().OpenProcess(
        PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _exit_code(handle: int) -> int:
    if sys.platform != "win32":
        raise RuntimeError("kernel32 exists only on Windows")
    from ctypes import wintypes

    status = wintypes.DWORD()
    if not _kernel32().GetExitCodeProcess(handle, ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.value)


def _created_at(handle: int) -> float:
    """When the process was created, in `time.time()` seconds."""
    if sys.platform != "win32":
        raise RuntimeError("kernel32 exists only on Windows")
    from ctypes import wintypes

    times = [wintypes.FILETIME() for _ in range(4)]
    if not _kernel32().GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    created = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    return (created - FILETIME_AT_UNIX_EPOCH) / 1e7


class _Held:
    """A process held by handle from the moment its pid was read.

    The pid is checked once, at that moment: still running, and created after
    the test began -- so the number did not already belong to something
    else. Everything afterwards asks the handle, which keeps naming that one
    process however long it has been dead.
    """

    def __init__(self, pid: int, since: float) -> None:
        self.pid = pid
        self.handle = _open(pid)
        if _exit_code(self.handle) != STILL_ACTIVE:
            pytest.fail(f"pid {pid} had already exited when the test first held it")
        if _created_at(self.handle) < since - 1.0:
            pytest.fail(f"pid {pid} is older than the test: the number was reused")

    def alive(self) -> bool:
        return _exit_code(self.handle) == STILL_ACTIVE

    def gone_within(self, seconds: float) -> bool:
        waited = _kernel32().WaitForSingleObject(self.handle, int(seconds * 1000))
        return bool(waited == WAIT_OBJECT_0)

    def exit_code(self) -> int:
        return _exit_code(self.handle)

    def end(self) -> None:
        if self.alive():
            _kernel32().TerminateProcess(self.handle, 1)
        _kernel32().CloseHandle(self.handle)


def _pid_in(path: Path, within: float = STARTUP_S) -> int:
    deadline = time.monotonic() + within
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(
                f"{path.name} never appeared: the tree did not start, so nothing was tested"
            )
        time.sleep(0.05)
    return int(path.read_text())


class _Tree:
    """One stage tree's pid files, and its processes once they are up."""

    def __init__(self, root: Path, name: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.since = time.time()
        self.grandchild_file = root / f"{name}-grandchild.pid"
        self.child_file = root / f"{name}-child.pid"
        self.marker = root / f"{name}-marker"
        self.held: list[_Held] = []
        self.grandchild: _Held | None = None
        self.child: _Held | None = None

    def stage(self, interpreter: str, grand_mode: str, seconds: float, marker: bool = False):
        return [
            interpreter,
            "-c",
            CHILD,
            str(self.grandchild_file),
            str(self.child_file),
            grand_mode,
            str(self.marker) if marker else "-",
            str(seconds),
        ]

    def up(self) -> None:
        """Wait for both pid files and hold both processes."""
        if self.grandchild is None:
            self.grandchild = _Held(_pid_in(self.grandchild_file), self.since)
            self.held.append(self.grandchild)
        if self.child is None:
            self.child = _Held(_pid_in(self.child_file), self.since)
            self.held.append(self.child)

    def end(self) -> None:
        for held in self.held:
            held.end()


def _popen_once_the_tree_is_up(monkeypatch, tree: _Tree) -> list[Any]:
    """Make `_run_guarded`'s `Popen` wait for `tree` in its first `communicate`.

    So a stage timeout can only fire on a tree that is known to be up --
    whatever the runner's interpreter start-up costs. Returns the `Popen`s
    made, so a test can read their return codes.
    """
    made: list[Any] = []

    # Defined before the patch below, so this is the real `Popen`.
    class OnceUp(subprocess.Popen):
        _waited = False

        def communicate(self, *args: Any, **kwargs: Any) -> Any:
            if not self._waited:
                self._waited = True
                tree.up()
            return super().communicate(*args, **kwargs)

    def popen(*args: Any, **kwargs: Any) -> Any:
        proc = OnceUp(*args, **kwargs)
        made.append(proc)
        return proc

    monkeypatch.setattr(envs.subprocess, "Popen", popen)
    return made


def _no_job(monkeypatch) -> None:
    monkeypatch.setattr(envs, "_windows_job_api", lambda: None)


def _assert_the_job_was_made(said: str) -> None:
    fell = [text for text in FALLBACKS if text in said]
    assert not fell, f"the real run fell back or a job step failed ({fell}):\n{said}"


@pytest.fixture(scope="module")
def venv_python(tmp_path_factory) -> Path:
    """A stage environment's interpreter, built as `provision` builds one, without pip."""
    root = tmp_path_factory.mktemp("stage-venv")
    envs._StageEnvBuilder(with_pip=False, clear=True, symlinks=False).create(root)
    return envs._interpreter_in(root)


@pytest.fixture(params=["base", "venv"])
def interpreter(request) -> str:
    """The interpreter a stage runs on: this one, or a virtualenv's redirector."""
    if request.param == "base":
        return sys.executable
    return str(request.getfixturevalue("venv_python"))


@pytest.fixture
def spy_on_job_termination(monkeypatch) -> list[tuple[int, int]]:
    """Every `TerminateJobObject` litetune asks for, still made."""
    if sys.platform != "win32":
        raise RuntimeError("winjob exists only on Windows")
    from litetune import winjob

    calls: list[tuple[int, int]] = []
    real = winjob.API.terminate

    def terminate(job: int, exit_code: int) -> None:
        calls.append((job, exit_code))
        real(job, exit_code)

    monkeypatch.setattr(winjob.API, "terminate", terminate)
    return calls


@pytest.fixture
def jobs_left_open(monkeypatch) -> Iterator[list[Any]]:
    """Keep every stage's job open until the test has looked, then close it.

    So what ended a process inside the stage can only have been the stage's
    own stop, and not the kill-on-close that closing the job would add.
    """
    kept: list[Any] = []
    real = envs._StageJob.close
    monkeypatch.setattr(envs._StageJob, "close", lambda job: kept.append(job))
    try:
        yield kept
    finally:
        for job in kept:
            real(job)


# ---------------------------------------------------------------------------
# The six behaviours, each with its control
# ---------------------------------------------------------------------------


@on_windows
def test_a_stage_timeout_ends_the_grandchild(
    tmp_path, monkeypatch, caplog, interpreter, jobs_left_open
):
    """The grandchild ignores Ctrl-Break, so what ends it is the job's
    termination -- and the job is still open when that is checked, so it was
    not the kill-on-close."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    try:
        with monkeypatch.context() as patched:
            _no_job(patched)
            _popen_once_the_tree_is_up(patched, control)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(control.stage(interpreter, "ignore", 60), timeout=1)
        assert (
            control.grandchild is not None and control.grandchild.alive()
        ), "without the job the grandchild must survive the timeout, or this test cannot fail"

        with monkeypatch.context() as patched:
            _popen_once_the_tree_is_up(patched, real)
            with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
                with pytest.raises(subprocess.TimeoutExpired) as caught:
                    envs._run_guarded(real.stage(interpreter, "ignore", 60), timeout=1)
        _assert_the_job_was_made(caplog.text)
        assert jobs_left_open, "the stage's job was not let go of"
        assert caught.value.timeout == 1
        assert "started" in (caught.value.output or ""), "what the stage said is still carried"
        assert real.grandchild is not None and real.grandchild.gone_within(
            5
        ), "the timeout left the grandchild running"
        assert (
            real.grandchild.exit_code() == envs._JOB_KILL_EXIT_CODE
        ), "ended by something other than litetune's TerminateJobObject"
    finally:
        control.end()
        real.end()


@on_windows
def test_a_forced_kill_of_litetune_ends_the_stage_tree(tmp_path):
    """`TerminateProcess` on the process running `_run_guarded` runs no handler
    at all; the job handle closes with the process, and kill-on-close ends the
    tree. This is the case nothing in litetune's own code can cover."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    driver_file = tmp_path / "driver_waits.py"
    driver_file.write_text(DRIVER_WAITS)
    drivers: list[subprocess.Popen] = []
    said: dict[str, str] = {}

    def run(mode: str, tree: _Tree) -> None:
        driver = subprocess.Popen(
            [
                sys.executable,
                str(driver_file),
                mode,
                str(tree.grandchild_file),
                str(tree.child_file),
                "default",
                "120",
                "300",
            ],
            stderr=subprocess.PIPE,
            text=True,
        )
        drivers.append(driver)
        tree.up()
        driver.kill()
        _, said[mode] = driver.communicate(timeout=30)

    try:
        run("control", control)
        run("real", real)

        assert all(
            held.alive() for held in control.held
        ), "without the job the stage outlives a killed litetune, or this test cannot fail"
        _assert_the_job_was_made(said["real"])
        for held in real.held:
            assert held.gone_within(5), f"pid {held.pid} outlived the litetune that started it"
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        control.end()
        real.end()


@on_windows
def test_the_grace_is_real(tmp_path, monkeypatch, caplog, interpreter, spy_on_job_termination):
    """A stage with a Ctrl-Break handler gets to run it on a timeout, and its
    tree leaves inside the grace: the job is never terminated. The control
    stops it the way Windows did before jobs, `TerminateProcess`, which runs
    no handler -- so a marker there would mean the test could not tell the
    two apart."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    try:
        with monkeypatch.context() as patched:
            _no_job(patched)
            _popen_once_the_tree_is_up(patched, control)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(control.stage(interpreter, "default", 60, True), timeout=1)
        assert (
            not control.marker.exists()
        ), "the handler ran under TerminateProcess, so this test cannot see a grace"

        with monkeypatch.context() as patched:
            _popen_once_the_tree_is_up(patched, real)
            with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
                with pytest.raises(subprocess.TimeoutExpired):
                    envs._run_guarded(real.stage(interpreter, "default", 60, True), timeout=1)
        if "no Ctrl-Break for" in caplog.text:
            pytest.fail(
                "GenerateConsoleCtrlEvent failed in this runner, so no grace was given -- this "
                "is the runner's console, not a pass or a fail of the grace:\n" + caplog.text
            )
        _assert_the_job_was_made(caplog.text)
        assert real.marker.exists() and real.marker.read_text() == "handled", (
            "the stage's Ctrl-Break handler did not get to run:\n" + caplog.text
        )
        assert spy_on_job_termination == [], "the tree had not left by the end of the grace"
        assert real.grandchild is not None and real.grandchild.gone_within(5)
    finally:
        control.end()
        real.end()


@on_windows
def test_ctrl_break_to_litetune_reports_an_interrupted_stage(tmp_path):
    """SIGBREAK in the guard: the driver ends on `StageInterrupted` -- what
    `cli` turns into exit 4 -- and not on `TerminateProcess(self, 21)`, and
    the tree is gone. The grandchild ignores Ctrl-Break, so the job ends it."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    driver_file = tmp_path / "driver_interrupted.py"
    driver_file.write_text(DRIVER_INTERRUPTED)
    drivers: list[subprocess.Popen] = []

    def run(mode: str, tree: _Tree) -> tuple[int, str, str]:
        driver = subprocess.Popen(
            [
                sys.executable,
                str(driver_file),
                mode,
                str(tree.grandchild_file),
                str(tree.child_file),
                "ignore",
                "120",
                "300",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        drivers.append(driver)
        tree.up()
        try:
            os.kill(driver.pid, getattr(signal, "CTRL_BREAK_EVENT", 1))
        except OSError as exc:
            pytest.fail(f"GenerateConsoleCtrlEvent failed in this runner: {exc}")
        out, err = driver.communicate(timeout=60)
        return driver.returncode, out, err

    try:
        run("control", control)
        assert (
            control.grandchild is not None and control.grandchild.alive()
        ), "without the job the grandchild survives the interrupt, or this test cannot fail"

        code, out, err = run("real", real)
        _assert_the_job_was_made(err)
        assert code == 42, f"the driver did not end on StageInterrupted: exit {code}, said {out!r}"
        assert f"STAGE-INTERRUPTED {int(getattr(signal, 'SIGBREAK', 21))}" in out
        for held in real.held:
            assert held.gone_within(5), f"pid {held.pid} outlived the interrupted stage"
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        control.end()
        real.end()


def _run_keyboard_driver(
    driver_file: Path,
    mode: str,
    tree: _Tree,
    seconds: int,
    drivers: list[subprocess.Popen],
    capsys: Any,
) -> tuple[dict[str, float], str]:
    """One `DRIVER_KEYBOARD` run: its times, and what it said on stderr."""
    driver = subprocess.Popen(
        [
            sys.executable,
            str(driver_file),
            mode,
            str(tree.grandchild_file),
            str(tree.child_file),
            "default",
            str(seconds),
            "120",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    drivers.append(driver)
    tree.up()
    out, err = driver.communicate(timeout=90)
    assert driver.returncode == 43, f"no KeyboardInterrupt: exit {driver.returncode}, {out!r}"
    words = out.split()
    said = dict(zip(words[0::2], words[1::2], strict=False))
    with capsys.disabled():
        print(f"\n[measured] {mode}: {out.strip()}")
    times = {key: float(said[key]) for key in ("T_START", "SENT", "ENTERED")}
    times["RUNNING"] = 1.0 if said["RUNNING"] == "True" else 0.0
    return times, err


@on_windows
def test_an_interrupt_reaches_the_wait(tmp_path, capsys):
    """`_thread.interrupt_main()` a second after the tree is up, while
    `_run_guarded` waits on a long stage. What is asserted is how long the
    interrupt took to reach `_kill_and_drain`, not a bound that includes the
    grace. The control takes the job away, which is also the single
    `communicate` call; its stage sleeps 20 s rather than 60 so the run stays
    short. Its lower bound holds only where `Thread.join` cannot be
    interrupted on Windows, which is before 3.14 (gh-125541), so it is not
    asserted on 3.14 and later."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    driver_file = tmp_path / "driver_keyboard.py"
    driver_file.write_text(DRIVER_KEYBOARD)
    drivers: list[subprocess.Popen] = []

    try:
        times, _ = _run_keyboard_driver(driver_file, "control", control, 20, drivers, capsys)
        if sys.version_info < (3, 14):
            floor = 20 - (times["SENT"] - times["T_START"]) - 0.5
            assert times["ENTERED"] - times["SENT"] >= floor, (
                f"without the slices the interrupt still arrived in "
                f"{times['ENTERED'] - times['SENT']:.2f} s"
            )
        assert (
            control.grandchild is not None and control.grandchild.alive()
        ), "without the job the grandchild survives the interrupt, or this test cannot fail"

        times, err = _run_keyboard_driver(driver_file, "real", real, 60, drivers, capsys)
        _assert_the_job_was_made(err)
        assert times["RUNNING"] == 1.0, "the stage had already ended, so nothing was interrupted"
        assert (
            times["ENTERED"] - times["SENT"] < 2
        ), f"the interrupt took {times['ENTERED'] - times['SENT']:.2f} s to reach the wait"
        for held in real.held:
            assert held.gone_within(5), f"pid {held.pid} outlived the interrupted stage"
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        control.end()
        real.end()


INTERRUPT_REPEATS = 10


@on_windows
def test_an_interrupt_reaches_the_wait_every_time(tmp_path, capsys):
    """The same interrupt, ten times. On Python 3.10 it was lost in 10 of 43
    CI runs read, reaching the wait only when the 60-second stage ended
    (`envs._handle_pending_signals`): one run per job let that pass about
    three times in four, ten runs about once in fourteen."""
    driver_file = tmp_path / "driver_keyboard.py"
    driver_file.write_text(DRIVER_KEYBOARD)
    drivers: list[subprocess.Popen] = []
    trees: list[_Tree] = []
    try:
        for n in range(INTERRUPT_REPEATS):
            tree = _Tree(tmp_path, f"real-{n}")
            trees.append(tree)
            times, err = _run_keyboard_driver(driver_file, "real", tree, 60, drivers, capsys)
            _assert_the_job_was_made(err)
            took = times["ENTERED"] - times["SENT"]
            assert times["RUNNING"] == 1.0 and took < 2, (
                f"run {n + 1} of {INTERRUPT_REPEATS}: the interrupt took {took:.2f} s to reach "
                "the wait"
            )
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        for tree in trees:
            tree.end()


@on_windows
def test_a_stage_ended_by_our_ctrl_break_exits_status_control_c_exit(
    monkeypatch, caplog, interpreter
):
    """What `exits` is told about a stage litetune stopped with Ctrl-Break: a
    child with the default handler, and no grandchild. Through the venv the
    code is the redirector's, which passes on its child's. The control is the
    same child ended by `Popen.kill`, whose code is litetune's own kill code,
    so the two are not the same either way."""
    started: list[Any] = []
    real_popen = subprocess.Popen

    def keep(*args: Any, **kwargs: Any) -> Any:
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(envs.subprocess, "Popen", keep)
    sleeper = [interpreter, "-c", SLEEPER]
    try:
        with monkeypatch.context() as patched:
            _no_job(patched)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(sleeper, timeout=3)
        assert started[0].returncode == envs._JOB_KILL_EXIT_CODE

        with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(sleeper, timeout=3)
        if "no Ctrl-Break for" in caplog.text:
            pytest.fail("GenerateConsoleCtrlEvent failed in this runner:\n" + caplog.text)
        _assert_the_job_was_made(caplog.text)
        code = started[1].returncode
        assert code == 0xC000013A, f"ended with {code} ({code:#x})"
        reading = read_returncode(code)
        assert not reading.conclusive
        assert "The process was stopped, not crashed" in reading.describe("the model")
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()


# ---------------------------------------------------------------------------
# Measurements the code is written around
# ---------------------------------------------------------------------------


@on_windows
def test_a_grace_ends_when_the_tree_has_exited(
    tmp_path, monkeypatch, capsys, spy_on_job_termination, jobs_left_open
):
    """`ActiveProcesses` is decremented, says Microsoft, "when the terminated
    process exits and all references to the process are released" -- and
    `Popen` holds a reference to the child. A tree that exits on Ctrl-Break at
    once must end the grace at once whichever way that goes. The job is kept
    open until both processes are seen to have exited, so a check that said
    "empty" too soon cannot pass on the kill-on-close that closing it would
    add. How many processes the count shows for an exited child whose handle
    is still open is printed: 0, on windows-latest, CPython 3.10.11, on the
    first run of this test."""
    from litetune import winjob

    tree = _Tree(tmp_path, "tree")
    lasted: list[float] = []
    real_kill_job = envs._kill_job

    def timed(*args: Any, **kwargs: Any) -> Any:
        began = time.monotonic()
        try:
            return real_kill_job(*args, **kwargs)
        finally:
            lasted.append(time.monotonic() - began)

    monkeypatch.setattr(envs, "_kill_job", timed)
    try:
        with monkeypatch.context() as patched:
            _popen_once_the_tree_is_up(patched, tree)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(tree.stage(sys.executable, "default", 60), timeout=1)
        assert jobs_left_open, "the stage's job was not let go of"
        assert tree.child is not None and tree.child.gone_within(
            0
        ), "the grace ended while the child was still running"
        assert tree.grandchild is not None and tree.grandchild.gone_within(
            0
        ), "the grace ended while the grandchild was still running"
        assert spy_on_job_termination == [], "the grace ran out on a tree that had left"
        assert lasted and lasted[0] < envs._TERM_GRACE_S / 2, f"the stop took {lasted[0]:.2f} s"

        job = winjob.API.create_job()
        child = subprocess.Popen(
            [sys.executable, "-c", "pass"], creationflags=envs._CREATE_SUSPENDED
        )
        try:
            winjob.API.assign(job, child)
            winjob.API.resume(child)
            child.wait(timeout=30)
            active = winjob.API.active_processes(job)
            ids = winjob.API.process_ids(job)
            with capsys.disabled():
                print(
                    f"\n[measured] an exited child, its handle still open: "
                    f"ActiveProcesses={active}, process ids={ids}, child pid={child.pid}"
                )
            assert envs._job_holds_nothing(envs._StageJob(winjob.API, job), child)
        finally:
            winjob.API.close(job)
    finally:
        tree.end()


@on_windows
def test_the_job_api_on_one_suspended_sleeper(tmp_path, capsys):
    """`winjob.API` step by step on a real process: assigned while suspended,
    counted, resumed once and refused a second time, terminated with the
    code it was given, and empty afterwards."""
    from litetune import winjob

    api = winjob.API
    sleeper = subprocess.Popen(
        [sys.executable, "-c", SLEEPER], creationflags=envs._CREATE_SUSPENDED
    )
    job = api.create_job()
    try:
        api.assign(job, sleeper)
        assert api.active_processes(job) == 1
        assert api.process_ids(job) == [sleeper.pid]
        api.resume(sleeper)
        with pytest.raises(OSError):
            api.resume(sleeper)
        api.terminate(job, envs._JOB_KILL_EXIT_CODE)
        assert sleeper.wait(timeout=10) == envs._JOB_KILL_EXIT_CODE
        deadline = time.monotonic() + 5
        stage_job = envs._StageJob(api, job)
        while not envs._job_holds_nothing(stage_job, sleeper):
            assert time.monotonic() < deadline, "the terminated job never emptied"
            time.sleep(0.05)
        left = api.active_processes(job)
        with capsys.disabled():
            print(f"\n[measured] after TerminateJobObject: ActiveProcesses={left}")
    finally:
        api.close(job)
        if sleeper.poll() is None:
            sleeper.kill()

    driver_file = tmp_path / "group_zero.py"
    driver_file.write_text(DRIVER_GROUP_ZERO)
    driver = subprocess.run(
        [sys.executable, str(driver_file)],
        capture_output=True,
        text=True,
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    assert driver.stdout.strip() == "REFUSED", (driver.returncode, driver.stdout, driver.stderr)


@on_windows
@pytest.mark.parametrize("pipes", ["open", "closed"])
def test_a_timed_out_stage_hands_back_exactly_what_it_said(tmp_path, monkeypatch, pipes):
    """Three lines on each pipe across several slices, then a hang -- with the
    pipes open, or after closing them, when Windows' `communicate` has read
    everything into its own buffers and closed the streams. What comes back
    is exactly what was said, once."""
    started, done = tmp_path / "started", tmp_path / "done"
    real_slices = envs._communicate_in_slices

    def once_it_has_started(proc: Any, timeout: float, **kwargs: Any) -> Any:
        _appears(started)
        return real_slices(proc, timeout, **kwargs)

    monkeypatch.setattr(envs, "_communicate_in_slices", once_it_has_started)
    stage = [sys.executable, "-c", CHATTY, str(started), str(done), pipes]

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        envs._run_guarded(stage, timeout=4)

    assert done.exists(), "the stage had not finished writing, so nothing was measured"
    assert caught.value.output == "out-1\nout-2\nout-3\n"
    assert caught.value.stderr == "err-1\nerr-2\nerr-3\n"


def _appears(path: Path, within: float = STARTUP_S) -> None:
    deadline = time.monotonic() + within
    while not path.exists():
        if time.monotonic() >= deadline:
            pytest.fail(f"{path.name} never appeared: the stage did not start")
        time.sleep(0.05)
