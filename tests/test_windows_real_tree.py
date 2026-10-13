"""The Windows process model against real processes, on the Windows CI job.

Each test builds a small real tree -- a stage child that starts a grandchild,
which writes its pid to a file and sleeps -- and stops it the way litetune
does. Each also runs the same tree with the job API taken away
(`_windows_job_api` returning None, which is the start every Windows run had
before jobs), in the same test, and asserts the grandchild *survives* that: a
pass cannot then come from a check that could not have failed.

Liveness is read with `OpenProcess` and `GetExitCodeProcess` (`STILL_ACTIVE`)
through ctypes. Everything a test started is terminated in its `finally`,
whatever the assertions found.

What is asserted about Windows here is what these tests exist to measure; until
the windows-latest job has run them, none of it is established.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import subprocess
import sys
import time
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
# What `OpenProcess` is expected to answer for a pid that names no process.
# Not stated for that case in its reference page; a different answer makes
# `_alive` raise rather than guess.
ERROR_INVALID_PARAMETER = 87

# Long enough for two interpreters to start on a CI runner, which nothing has
# measured; a tree that is not up by then fails the test by saying so.
STARTUP_S = 30.0

# The grandchild: write its pid where the test will look, atomically, then
# sleep in short steps. With "ignore" it ignores Ctrl-Break first, so only the
# job can end it.
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

# A process running `_run_guarded` the way the CLI does, so the test can do to
# it what cannot be done to pytest. `control` takes the job API away.
DRIVER_PREAMBLE = """
import _thread, os, sys, threading, time
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
# reports how long the KeyboardInterrupt took to arrive, with exit code 43.
DRIVER_KEYBOARD = (
    DRIVER_PREAMBLE
    + """
sent = []
def interrupt_once_the_tree_is_up():
    while not os.path.exists(grand_pidfile):
        time.sleep(0.05)
    time.sleep(1.0)
    sent.append(time.monotonic())
    _thread.interrupt_main()
threading.Thread(target=interrupt_once_the_tree_is_up, daemon=True).start()
try:
    envs._run_guarded(child, timeout=float(timeout))
except KeyboardInterrupt:
    print("KEYBOARD-INTERRUPT-AFTER", round(time.monotonic() - sent[0], 2), flush=True)
    sys.exit(43)
print("NOT-INTERRUPTED", flush=True)
sys.exit(1)
"""
)


def test_the_real_tree_scripts_compile():
    """Caught here rather than by the first Windows run."""
    for source in (GRANDCHILD, CHILD, DRIVER_WAITS, DRIVER_INTERRUPTED, DRIVER_KEYBOARD):
        compile(source, "<windows tree>", "exec")


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
    return k


def _alive(pid: int) -> bool:
    if sys.platform != "win32":
        raise RuntimeError("kernel32 exists only on Windows")
    from ctypes import wintypes

    k = _kernel32()
    handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        code = ctypes.get_last_error()
        if code == ERROR_INVALID_PARAMETER:
            return False
        raise ctypes.WinError(code)
    try:
        status = wintypes.DWORD()
        if not k.GetExitCodeProcess(handle, ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return status.value == STILL_ACTIVE
    finally:
        k.CloseHandle(handle)


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _terminate(pid: int) -> None:
    """Cleanup for whatever a test left running; a pid already gone is fine."""
    if sys.platform != "win32":
        raise RuntimeError("kernel32 exists only on Windows")
    k = _kernel32()
    handle = k.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not handle:
        return
    try:
        k.TerminateProcess(handle, 1)
    finally:
        k.CloseHandle(handle)


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
    """The pid files of one tree, and every pid it turned out to have."""

    def __init__(self, root: Path, name: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.grandchild_file = root / f"{name}-grandchild.pid"
        self.child_file = root / f"{name}-child.pid"
        self.marker = root / f"{name}-marker"
        self.started: list[int] = []

    def stage(self, grand_mode: str, seconds: float, marker: bool = False) -> list[str]:
        return [
            sys.executable,
            "-c",
            CHILD,
            str(self.grandchild_file),
            str(self.child_file),
            grand_mode,
            str(self.marker) if marker else "-",
            str(seconds),
        ]

    def grandchild(self) -> int:
        pid = _pid_in(self.grandchild_file)
        self.started.append(pid)
        return pid

    def child(self) -> int:
        pid = _pid_in(self.child_file)
        self.started.append(pid)
        return pid

    def clean_up(self) -> None:
        for path in (self.grandchild_file, self.child_file):
            if path.exists():
                self.started.append(int(path.read_text()))
        for pid in self.started:
            _terminate(pid)


def _no_job(monkeypatch) -> None:
    monkeypatch.setattr(envs, "_windows_job_api", lambda: None)


def _assert_ctrl_break_was_delivered(caplog) -> None:
    if "no Ctrl-Break for" in caplog.text:
        pytest.fail(
            "GenerateConsoleCtrlEvent failed in this runner, so no grace was given -- this is "
            "the runner's console, not a pass or a fail of the grace:\n" + caplog.text
        )


def _assert_the_job_was_made(caplog) -> None:
    # Both spellings: the warning is given once per process, and a later stage
    # that also fell back says so at debug level only.
    assert "could not put" not in caplog.text and "no job object for" not in caplog.text, (
        "litetune could not create or assign a job here, so the real run fell back:\n" + caplog.text
    )


@on_windows
def test_a_stage_timeout_ends_the_grandchild(tmp_path, monkeypatch, caplog):
    """The grandchild ignores Ctrl-Break, so what ends it is the job."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    try:
        with monkeypatch.context() as patched:
            _no_job(patched)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(control.stage("ignore", 60), timeout=6)
        survivor = control.grandchild()
        assert _alive(
            survivor
        ), "without the job the grandchild must survive the timeout, or this test cannot fail"

        with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
            with pytest.raises(subprocess.TimeoutExpired) as caught:
                envs._run_guarded(real.stage("ignore", 60), timeout=6)
        _assert_the_job_was_made(caplog)
        assert caught.value.timeout == 6
        assert "started" in (caught.value.output or ""), "what the stage said is still carried"
        assert _gone_within(real.grandchild(), 5), "the timeout left the grandchild running"
        assert _gone_within(real.child(), 5)
    finally:
        control.clean_up()
        real.clean_up()


@on_windows
def test_a_forced_kill_of_litetune_ends_the_stage_tree(tmp_path):
    """`TerminateProcess` on the process running `_run_guarded` runs no handler
    at all; the job handle closes with the process, and kill-on-close ends the
    tree. This is the case nothing in litetune's own code can cover."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    driver_file = tmp_path / "driver_waits.py"
    driver_file.write_text(DRIVER_WAITS)
    drivers: list[subprocess.Popen] = []

    def start(mode: str, tree: _Tree) -> subprocess.Popen:
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
            ]
        )
        drivers.append(driver)
        return driver

    try:
        for mode, tree in (("control", control), ("real", real)):
            driver = start(mode, tree)
            tree.grandchild()
            tree.child()
            driver.kill()
            driver.wait(timeout=30)

        assert all(
            _alive(pid) for pid in control.started
        ), "without the job the stage outlives a killed litetune, or this test cannot fail"
        for pid in real.started:
            assert _gone_within(pid, 5), f"pid {pid} outlived the litetune that started it"
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        control.clean_up()
        real.clean_up()


@on_windows
def test_the_grace_is_real(tmp_path, monkeypatch, caplog):
    """A stage with a Ctrl-Break handler gets to run it on a timeout. Without
    the job it is `TerminateProcess`, and the handler never runs."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    try:
        with monkeypatch.context() as patched:
            _no_job(patched)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(control.stage("default", 60, marker=True), timeout=6)
        control.grandchild()
        assert (
            not control.marker.exists()
        ), "the handler ran without a grace, so this test cannot tell a grace from none"

        with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(real.stage("default", 60, marker=True), timeout=6)
        _assert_the_job_was_made(caplog)
        _assert_ctrl_break_was_delivered(caplog)
        assert real.marker.exists() and real.marker.read_text() == "handled", (
            "the stage's Ctrl-Break handler did not get to run:\n" + caplog.text
        )
        assert _gone_within(real.grandchild(), 5)
    finally:
        control.clean_up()
        real.clean_up()


@on_windows
def test_ctrl_break_to_litetune_reports_an_interrupted_stage(tmp_path):
    """SIGBREAK in the guard: the driver ends on `StageInterrupted` -- what
    `cli` turns into exit 4 -- and not on `TerminateProcess(self, 21)`, and
    the tree is gone. The grandchild ignores Ctrl-Break, so the job ends it."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    driver_file = tmp_path / "driver_interrupted.py"
    driver_file.write_text(DRIVER_INTERRUPTED)
    drivers: list[subprocess.Popen] = []

    def run(mode: str, tree: _Tree) -> tuple[int, str]:
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
            text=True,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        drivers.append(driver)
        tree.grandchild()
        tree.child()
        try:
            os.kill(driver.pid, getattr(signal, "CTRL_BREAK_EVENT", 1))
        except OSError as exc:
            pytest.fail(f"GenerateConsoleCtrlEvent failed in this runner: {exc}")
        out, _ = driver.communicate(timeout=60)
        return driver.returncode, out

    try:
        run("control", control)
        assert _alive(
            control.started[0]
        ), "without the job the grandchild survives the interrupt, or this test cannot fail"

        code, out = run("real", real)
        assert code == 42, f"the driver did not end on StageInterrupted: exit {code}, said {out!r}"
        assert f"STAGE-INTERRUPTED {int(getattr(signal, 'SIGBREAK', 21))}" in out
        for pid in real.started:
            assert _gone_within(pid, 5), f"pid {pid} outlived the interrupted stage"
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        control.clean_up()
        real.clean_up()


@on_windows
def test_an_interrupt_reaches_the_wait(tmp_path):
    """`_thread.interrupt_main()` a second after the tree is up, while
    `_run_guarded` waits on a long stage. Through the job's sliced wait it
    arrives within a few seconds and the tree is gone. The control takes the
    job away, which is also the single `communicate` call; its stage sleeps
    20 s rather than 60 so the run stays short, and how long its interrupt
    took is asserted as the measurement it is."""
    control, real = _Tree(tmp_path, "control"), _Tree(tmp_path, "real")
    driver_file = tmp_path / "driver_keyboard.py"
    driver_file.write_text(DRIVER_KEYBOARD)
    drivers: list[subprocess.Popen] = []

    def run(mode: str, tree: _Tree, seconds: int) -> float:
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
            text=True,
        )
        drivers.append(driver)
        tree.grandchild()
        tree.child()
        out, _ = driver.communicate(timeout=90)
        assert driver.returncode == 43, f"no KeyboardInterrupt: exit {driver.returncode}, {out!r}"
        return float(out.split("KEYBOARD-INTERRUPT-AFTER", 1)[1].split()[0])

    try:
        waited = run("control", control, 20)
        assert waited >= 8, f"without the slices the interrupt still arrived in {waited} s"
        assert _alive(
            control.started[0]
        ), "without the job the grandchild survives the interrupt, or this test cannot fail"

        waited = run("real", real, 60)
        assert waited < 5, f"the interrupt took {waited} s to reach the wait"
        for pid in real.started:
            assert _gone_within(pid, 5), f"pid {pid} outlived the interrupted stage"
    finally:
        for driver in drivers:
            if driver.poll() is None:
                driver.kill()
        control.clean_up()
        real.clean_up()


@on_windows
def test_a_stage_ended_by_our_ctrl_break_exits_status_control_c_exit(monkeypatch, caplog):
    """What `exits` is told about a stage litetune stopped with Ctrl-Break: a
    child with the default handler, and no grandchild. The control is the same
    child ended by `Popen.kill`, whose 1 shows the code is not the same either
    way."""
    started: list[Any] = []
    real_popen = subprocess.Popen

    def keep(*args: Any, **kwargs: Any) -> Any:
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(envs.subprocess, "Popen", keep)
    sleeper = [sys.executable, "-c", "import time\nwhile True: time.sleep(0.1)"]
    try:
        with monkeypatch.context() as patched:
            _no_job(patched)
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(sleeper, timeout=3)
        assert started[0].returncode == 1, "`Popen.kill` on Windows ends a child with 1"

        with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
            with pytest.raises(subprocess.TimeoutExpired):
                envs._run_guarded(sleeper, timeout=3)
        _assert_the_job_was_made(caplog)
        _assert_ctrl_break_was_delivered(caplog)
        code = started[1].returncode
        assert code == 0xC000013A, f"ended with {code} ({code:#x})"
        reading = read_returncode(code)
        assert not reading.conclusive
        assert "not as a crash" in reading.describe("the model")
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()
