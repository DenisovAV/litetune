"""The Windows process model, run on any host against a fake job API.

`envs` decides the order of the Win32 calls and what each failure means;
`litetune.winjob` only makes them. So every Windows branch is reachable here by
handing `_run_guarded` and `_kill_tree` a recording `_JobApi` in place of
kernel32 -- the injection #74 used for `msvcrt` -- and the real calls are
exercised by `test_windows_real_tree.py`, on the Windows job alone.

Nothing here starts a process. The stage is a `Popen` stand-in, and where the
POSIX half of `_kill_tree` could be reached the platform is set to win32 first,
so no fake pid is ever handed to `getpgid` or `killpg` on the host.
"""

from __future__ import annotations

import _thread
import logging
import signal
import subprocess
import sys
import time
import types
from typing import Any

import pytest
from conftest import FAKE_JOB, FakeJobApi

import litetune
from litetune import envs
from litetune.exits import read_returncode

JOB = FAKE_JOB
PID = 4242
FLAGS = 0x00000004 | 0x00000200  # CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP

# Ten seconds of slices: no test here gives a stage that long, so a wait still
# going after this many is one that would never stop.
_MOST_WAITS = 40


class FakeClock:
    """`time.monotonic` for `_communicate_in_slices`, moved only by the stage."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeStage:
    """A stage child on Windows: never a real process, and every touch recorded.

    `answers` is what successive `communicate` calls do, the last one repeated:
    a tuple is returned (and the child has exited); `"wait"` sleeps the timeout
    out -- or moves `clock` on by it -- and raises `TimeoutExpired`, which is
    what `Popen.communicate` does on Windows while the child runs;
    `"interrupt"` does the same after scheduling a SIGINT, the way a Ctrl-C
    lands while a `join` that cannot be interrupted is waiting; an exception
    is raised.

    `kill` leaves `returncode` alone, as `Popen.kill` does on Windows: it is
    `TerminateProcess`, and only a later `poll` or `wait` reads the code.
    """

    args = ["python"]
    stdout: Any = None
    stderr: Any = None

    def __init__(
        self,
        log: list[tuple[Any, ...]] | None = None,
        *,
        pid: int = PID,
        answers: tuple[Any, ...] = (("out", "err"),),
        returncode: int | None = None,
        clock: FakeClock | None = None,
    ) -> None:
        self.log = log if log is not None else []
        self.pid = pid
        self.answers = list(answers)
        self.returncode = returncode
        self.clock = clock
        self.waits: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[Any, Any]:
        if len(self.waits) >= _MOST_WAITS:
            raise AssertionError("still waiting in slices long past every deadline here")
        self.log.append(("communicate", timeout))
        self.waits.append(timeout)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if answer in ("wait", "interrupt"):
            if answer == "interrupt":
                _thread.interrupt_main()
            if self.clock is not None:
                self.clock.now += timeout or 0
            else:
                time.sleep(timeout or 0)
            raise subprocess.TimeoutExpired(self.args, timeout or 0)
        if isinstance(answer, BaseException):
            raise answer
        self.returncode = 0
        return answer

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.log.append(("wait",))
        if self.returncode is None:
            self.returncode = 1
        return self.returncode

    def kill(self) -> None:
        self.log.append(("kill",))

    def send_signal(self, sig: int) -> None:
        # `Popen.send_signal(SIGTERM)` is `terminate()` on Windows; the code is
        # read here at once so the route under test ends without a grace.
        self.log.append(("send_signal", sig))
        self.returncode = 1


@pytest.fixture
def windows(monkeypatch):
    """Run `_run_guarded` the Windows way: a fake API, and a fresh "warned" flag.

    Yields a function that installs a stage and an API sharing one log, so a
    test can read the order of everything that happened. `before_popen` runs
    inside the fake `Popen`, where a signal can be raised to land between
    `CreateProcess` and the job.
    """
    monkeypatch.setattr(envs, "_NO_JOB_WARNED", False)

    def install(stage: FakeStage, api: FakeJobApi, before_popen: Any = None) -> dict[str, Any]:
        started: dict[str, Any] = {}

        def popen(argv, **kwargs):
            started.update(kwargs)
            stage.log.append(("popen", kwargs.get("creationflags")))
            if before_popen is not None:
                before_popen()
            return stage

        monkeypatch.setattr(envs.subprocess, "Popen", popen)
        monkeypatch.setattr(envs, "_windows_job_api", lambda: api)
        return started

    return install


def _job(api: FakeJobApi) -> envs._StageJob:
    return envs._StageJob(api, JOB)


def _steps(log: list[tuple[Any, ...]]) -> list[Any]:
    return [call[0] for call in log]


def _raise_sigint() -> None:
    signal.raise_signal(signal.SIGINT)


# ---------------------------------------------------------------------------
# Starting a stage: the job, then the child suspended, assigned, resumed
# ---------------------------------------------------------------------------


def test_a_windows_stage_is_in_its_job_before_it_runs(windows):
    """The job comes first, so a host that cannot make one starts the stage as
    before; and the child runs only once it is in the job, because a child
    assigned after it started can start a grandchild outside it."""
    log: list[tuple[Any, ...]] = []
    stage, api = FakeStage(log), FakeJobApi(log)
    started = windows(stage, api)

    result = envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert started["creationflags"] == FLAGS, "suspended, and the root of a process group"
    assert started["start_new_session"] is True, "unchanged for POSIX, ignored on Windows"
    steps = _steps(log)
    assert steps[:4] == ["create_job", "popen", "assign", "resume"], steps
    assert ("assign", JOB, PID) in log and ("resume", PID) in log
    assert steps.index("communicate") > steps.index("resume"), "nothing waits on a suspended child"
    assert log[-1] == ("close", JOB), "the job is let go of last, once the stage is done"
    assert (result.returncode, result.stdout, result.stderr) == (0, "out", "err")


def test_a_host_that_cannot_make_a_job_starts_its_stages_as_before(windows, caplog, monkeypatch):
    """No job, so no process group either: CREATE_NEW_PROCESS_GROUP disables
    Ctrl-C for the group, and only a job's Ctrl-Break would make up for it.
    Stopping one is `Popen`'s, as it always was."""
    monkeypatch.setattr(envs.sys, "platform", "win32")
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, fail={"create_job": OSError("access denied")})

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        first = windows(FakeStage(log), api)
        assert envs._run_guarded(["python", "-c", "pass"], timeout=5).stdout == "out"
        assert first["creationflags"] == 0, "the start every Windows stage had before jobs"
        windows(FakeStage(log, answers=("wait",)), api)
        with pytest.raises(subprocess.TimeoutExpired):
            envs._run_guarded(["python", "-c", "pass"], timeout=0.3)

    steps = _steps(log)
    assert "assign" not in steps and "resume" not in steps, "nothing was suspended"
    assert "send_ctrl_break" not in steps and "terminate" not in steps and "close" not in steps
    assert ("send_signal", signal.SIGTERM) in log, "today's route: `Popen.send_signal`"
    warned = [r.getMessage() for r in caplog.records if "could not create a job" in r.getMessage()]
    assert len(warned) == 1, "once per process, not once per stage"
    assert "ends its own process and not what it started" in warned[0]


def test_a_job_that_refuses_the_child_leaves_its_group_to_ctrl_break(windows, caplog):
    """Created in a process group already, the child gets a Ctrl-Break and a
    grace on a stop, observed through its own handle, and the warning says
    what the job would have added."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))

    def exits_on_break() -> None:
        stage.returncode = 0xC000013A

    api = FakeJobApi(log, fail={"assign": OSError("access denied")}, on_break=exits_on_break)
    started = windows(stage, api)

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        with pytest.raises(subprocess.TimeoutExpired):
            envs._run_guarded(["python", "-c", "pass"], timeout=0.3)

    assert started["creationflags"] == FLAGS
    steps = _steps(log)
    assert steps[steps.index("assign") + 1] == "close", "the job that refused is let go of at once"
    assert ("resume", PID) in log, "resumed anyway: a suspended child would wait for ever"
    assert ("send_ctrl_break", PID) in log
    assert "terminate" not in steps and "kill" not in steps, "it left inside the grace"
    assert "whatever ignores Ctrl-Break is not collected" in caplog.text


def test_a_group_without_a_job_is_killed_through_the_child_after_its_grace():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log)
    stage = FakeStage(log)

    started = time.monotonic()
    reached = envs._kill_tree(stage, grace=0.2, pgid=envs._BreakOnly(api), pump=time.sleep)

    assert time.monotonic() - started >= 0.19, "the grace is given"
    assert _steps(log) == ["send_ctrl_break", "kill"]
    assert reached is envs.Reach.GROUP


def test_a_child_that_cannot_be_resumed_is_terminated_and_never_waited_on(windows):
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)
    pipes = _Pipe(), _Pipe()
    stage.stdout, stage.stderr = pipes
    refused = OSError("ResumeThread failed")
    windows(stage, FakeJobApi(log, fail={"resume": refused}))

    with pytest.raises(OSError, match="could not resume python") as caught:
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert caught.value.__cause__ is refused
    steps = _steps(log)
    assert "communicate" not in steps, "a suspended child waits for ever"
    assert steps.index("kill") > steps.index("resume")
    assert steps.index("wait") > steps.index("kill"), "and the terminated child is reaped"
    assert log[-1] == ("close", JOB), "its job goes with it"
    assert all(pipe.closed for pipe in pipes), "nothing else will close our ends of its pipes"


class _Pipe:
    """Our end of a stage pipe, as far as closing it goes."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_a_suspended_child_that_cannot_be_killed_is_said_and_its_job_still_closed(windows, caplog):
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)

    def refuse() -> None:
        log.append(("kill",))
        raise PermissionError("access denied")

    stage.kill = refuse
    windows(stage, FakeJobApi(log, fail={"resume": OSError("ResumeThread failed")}))

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        with pytest.raises(OSError, match="could not resume"):
            envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert "could not terminate python" in caplog.text
    assert "closing its job ends it" in caplog.text
    assert log[-1] == ("close", JOB)


def test_a_popen_without_a_process_handle_is_refused_not_guessed_at(windows):
    """What the real API does with a `Popen` that has no `_handle`: neither the
    assignment nor the resume will act on a bare pid."""
    log: list[tuple[Any, ...]] = []
    missing = OSError("pid 4242 has no open process handle in its Popen")
    windows(FakeStage(log), FakeJobApi(log, fail={"assign": missing, "resume": missing}))

    with pytest.raises(OSError, match="could not resume"):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    steps = _steps(log)
    assert "communicate" not in steps
    assert "kill" in steps
    assert steps.count("close") == 1, "the job that refused is closed once, at the refusal"


@pytest.mark.parametrize("step", ["create_job", "assign", "resume"])
def test_an_exception_while_the_stage_starts_leaves_nothing_suspended(windows, step):
    """A Ctrl-C between the spawn and the resume, raised as the exception
    itself: the one `finally` that owns the start ends the child and the job,
    and the interrupt stays an interrupt."""
    log: list[tuple[Any, ...]] = []
    windows(FakeStage(log), FakeJobApi(log, fail={step: KeyboardInterrupt()}))

    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    steps = _steps(log)
    assert "communicate" not in steps
    if step == "create_job":
        assert "popen" not in steps, "nothing was started"
    else:
        assert "kill" in steps and log[-1] == ("close", JOB)


@pytest.mark.parametrize("where", ["create_job", "popen", "assign", "resume"])
def test_a_signal_while_the_stage_starts_is_held_until_it_can_be_acted_on(windows, where):
    """SIGINT raised for real at each step of the start -- after `CreateProcess`
    inside `Popen`, before the job exists, before the child runs. Held, not
    acted on: the start completes, the child runs in its job, and the
    interrupt is then delivered through the guard's path, which stops the
    stage through the job and closes it."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))
    api = FakeJobApi(log, counts=(1, 0))
    if where == "popen":
        windows(stage, api, before_popen=_raise_sigint)
    else:
        real = getattr(api, where)

        def then_signal(*args: Any) -> Any:
            answer = real(*args)
            _raise_sigint()
            return answer

        setattr(api, where, then_signal)
        windows(stage, api)

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert time.monotonic() - started < 2, "held past the first wait, until the stage's deadline"
    steps = _steps(log)
    assert ("resume", PID) in log, "the child was not left suspended"
    assert steps.index("communicate") > steps.index("resume"), "and was running when it came"
    assert ("send_ctrl_break", PID) in log, "stopped through its job"
    assert log[-1] == ("close", JOB)
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "handed back"


# The three trees a Windows stage can have, by what the job API refuses: a
# job; a process group and no job, when the job will not take the child; and
# neither, when no job can be made.
_TREES = {
    "job": {},
    "group only": {"assign": OSError("access denied")},
    "plain": {"create_job": OSError("access denied")},
}


@pytest.mark.parametrize("tree", list(_TREES))
def test_an_exception_before_the_guard_stops_the_running_stage(windows, monkeypatch, tree):
    """Between the start handing the child over and the guard owning it, an
    exception that is not a held signal: the start's `finally` owns the
    running child, and stops it without a grace through whatever tree it
    has -- nothing else would, the caller's own kill being inside the guard."""
    log: list[tuple[Any, ...]] = []
    windows(FakeStage(log, answers=("wait",)), FakeJobApi(log, counts=(1, 0), fail=_TREES[tree]))

    def raises_first(proc, pgid=None, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(envs, "_kill_child_if_we_are_told_to_exit", raises_first)

    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    steps = _steps(log)
    assert "communicate" not in steps
    stopped = steps[steps.index("popen") + 1 :]
    if tree == "job":
        assert ("terminate", JOB, envs._JOB_KILL_EXIT_CODE) in log
        assert log[-1] == ("close", JOB)
    elif tree == "group only":
        assert ("send_ctrl_break", PID) in log and "kill" in stopped
    else:
        assert "kill" in stopped and "send_ctrl_break" not in steps
    assert "wait" in stopped, "and reaped"


@pytest.mark.parametrize("tree", list(_TREES))
def test_a_signal_before_the_guard_is_held_until_the_guard_owns_the_stage(
    windows, monkeypatch, caplog, tree
):
    """A real SIGINT between the start and the guard is still held -- the
    recorders stay installed until the guard is in place -- and is then
    delivered inside the guard, which stops the stage through its tree."""
    monkeypatch.setattr(envs.sys, "platform", "win32")
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))

    def exits() -> None:
        stage.returncode = 0xC000013A

    windows(stage, FakeJobApi(log, counts=(1, 0), fail=_TREES[tree], on_break=exits))
    real_guard = envs._kill_child_if_we_are_told_to_exit

    def signalled_first(proc, pgid=None, **kwargs):
        _raise_sigint()
        return real_guard(proc, pgid, **kwargs)

    monkeypatch.setattr(envs, "_kill_child_if_we_are_told_to_exit", signalled_first)

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        with pytest.raises(KeyboardInterrupt):
            envs._run_guarded(["python", "-c", "pass"], timeout=5)

    steps = _steps(log)
    assert steps.count("communicate") == 1, "delivered after the first wait, inside the guard"
    after = steps[steps.index("communicate") :]
    if tree == "job":
        assert "send_ctrl_break" in after and log[-1] == ("close", JOB)
    elif tree == "group only":
        assert "send_ctrl_break" in after
    else:
        assert "send_signal" in after, "the stop `Popen` had before jobs"
    assert "finished with" not in caplog.text, "an interrupted stage did not finish"
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "handed back"


def test_the_guard_takes_a_held_signal_over_and_puts_the_original_back(monkeypatch):
    """`inherited` is what a held signal was bound to before: the guard decides
    on it and restores it, and never puts the recorder back."""
    sig = _a_sigbreak(monkeypatch)
    original = signal.getsignal(sig)
    held = envs._DeferredSignals()
    try:
        assert signal.getsignal(sig) is held._recorder, "held while the stage starts"
        with envs._kill_child_if_we_are_told_to_exit(FakeStage(), inherited=held.originals):
            assert signal.getsignal(sig) is not held._recorder, "the guard replaced the recorder"
            held.hand_to_guard()
            assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        assert signal.getsignal(sig) == original
        assert held.originals == {}
    finally:
        held.restore()


def test_a_held_signal_that_was_ignored_stays_ignored_under_the_guard(monkeypatch):
    sig = _a_sigbreak(monkeypatch)
    previous = signal.signal(sig, signal.SIG_IGN)
    held = envs._DeferredSignals()
    try:
        with envs._kill_child_if_we_are_told_to_exit(FakeStage(), inherited=held.originals):
            held.hand_to_guard()
            assert signal.getsignal(sig) == signal.SIG_IGN, "not installed over an ignore"
        assert signal.getsignal(sig) == signal.SIG_IGN
    finally:
        held.restore()
        signal.signal(sig, previous)


def test_an_interrupted_restore_is_finished_by_the_next_one(monkeypatch):
    """A Ctrl-C between putting SIGBREAK back and putting SIGINT back must not
    leave either bound to a recorder nobody reads: each is forgotten only once
    it is back, and the `finally` that calls `restore` again finishes it."""
    sig = _a_sigbreak(monkeypatch)
    original_break = signal.getsignal(sig)
    held = envs._DeferredSignals()
    real_signal = signal.signal
    calls: list[int] = []

    def interrupted_the_first_time(signum: int, handler: Any) -> Any:
        calls.append(signum)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_signal(signum, handler)

    monkeypatch.setattr(envs.signal, "signal", interrupted_the_first_time)
    try:
        with pytest.raises(KeyboardInterrupt):
            held.restore()
    finally:
        monkeypatch.setattr(envs.signal, "signal", real_signal)
    assert calls == [sig], "SIGINT is put back last"
    assert signal.getsignal(sig) is held._recorder, "the interruption came before it was put back"

    held.restore()

    assert signal.getsignal(sig) == original_break
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert held.originals == {}


def test_an_ignored_sigint_is_not_held_or_delivered(windows, monkeypatch):
    log: list[tuple[Any, ...]] = []
    windows(FakeStage(log), FakeJobApi(log), before_popen=_raise_sigint)
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        result = envs._run_guarded(["python", "-c", "pass"], timeout=5)
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)

    assert result.stdout == "out"


# ---------------------------------------------------------------------------
# Stopping a stage: Ctrl-Break, a grace, then the job
# ---------------------------------------------------------------------------


def _sleeping_pump(log: list[tuple[Any, ...]]) -> Any:
    def pump(seconds: float) -> None:
        log.append(("pump", seconds))
        time.sleep(seconds)

    return pump


def test_stopping_a_job_breaks_then_waits_pumping_then_terminates():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0))
    stage = FakeStage(log)

    reached = envs._kill_tree(stage, grace=0.2, pgid=_job(api), pump=_sleeping_pump(log))

    assert reached is envs.Reach.GROUP
    steps = _steps(log)
    assert ("send_ctrl_break", PID) in log
    assert steps.index("pump") > steps.index("send_ctrl_break"), "the grace comes after the break"
    assert ("terminate", JOB, 1) in log
    assert steps.index("terminate") > steps.index("pump")
    assert all(
        s <= 0.05 for (name, *rest) in log if name == "pump" for s in rest
    ), "pumped in short slices, as the POSIX grace is"


def test_the_exit_code_of_a_job_kill_is_the_one_popen_kill_uses():
    """`Popen.terminate` passes 1 to `TerminateProcess` on Windows, and `kill`
    is the same method -- so a stage killed through the job and one killed
    through `Popen` end with the same code."""
    assert envs._JOB_KILL_EXIT_CODE == 1


def test_a_job_that_empties_during_the_grace_is_not_terminated():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2, 2, 0))

    started = time.monotonic()
    reached = envs._kill_tree(FakeStage(log), grace=5, pgid=_job(api), pump=time.sleep)

    assert reached is envs.Reach.GROUP
    assert "terminate" not in api.steps()
    assert time.monotonic() - started < 1, "the grace ends when the job is empty"


def test_a_job_whose_only_process_is_our_exited_child_holds_nothing():
    """`ActiveProcesses` may stay at 1 while `Popen` holds the exited child's
    handle. The pid list naming only our child, and the child having exited,
    ends the grace all the same."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)

    def exits() -> None:
        stage.returncode = 0xC000013A

    api = FakeJobApi(log, counts=(1,), ids=(PID,), on_break=exits)

    started = time.monotonic()
    reached = envs._kill_tree(stage, grace=5, pgid=_job(api), pump=time.sleep)

    assert time.monotonic() - started < 1
    assert reached is envs.Reach.GROUP
    assert "terminate" not in api.steps()


def test_a_pid_list_with_anything_else_in_it_is_not_empty():
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, returncode=0)
    api = FakeJobApi(log, counts=(2,), ids=(PID, 4343))

    assert not envs._job_holds_nothing(_job(api), stage)
    assert envs._job_holds_nothing(_job(FakeJobApi(counts=(1,), ids=(PID,))), stage)
    running = FakeStage(log)
    assert not envs._job_holds_nothing(_job(FakeJobApi(counts=(1,), ids=(PID,))), running)


def test_the_child_is_known_dead_before_the_pid_list_is_trusted():
    """Read the list first and the child could start a grandchild and exit
    between the two looks: a list taken while it was alone would then vouch
    for a job that is not empty. Here the child does exactly that the moment
    the list has been read."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)
    api = FakeJobApi(log, counts=(2,), ids=(PID,))
    listed = api.process_ids

    def then_it_spawns_and_exits(job: int) -> list[int]:
        answer = listed(job)
        stage.returncode = 0
        return answer

    api.process_ids = then_it_spawns_and_exits

    assert not envs._job_holds_nothing(_job(api), stage)
    stage.returncode = None
    assert envs._still_in_job(_job(api), stage) == 1, "our child, alive when it was counted"


def test_without_a_console_there_is_no_grace(caplog, monkeypatch):
    """`GenerateConsoleCtrlEvent` fails when litetune has no console, and a
    grace with no signal to use it is only a delay."""
    monkeypatch.setattr(envs, "_REAP_AFTER_KILL_S", 0.2)
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,), fail={"send_ctrl_break": OSError("no console")})

    started = time.monotonic()
    with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
        reached = envs._kill_tree(FakeStage(log), grace=5, pgid=_job(api), pump=_sleeping_pump(log))

    assert time.monotonic() - started < 1
    steps = _steps(log)
    assert "pump" not in steps[: steps.index("terminate")], "no grace before the job is ended"
    assert ("terminate", JOB, 1) in log
    assert reached is envs.Reach.GROUP
    assert "no Ctrl-Break" in caplog.text


@pytest.mark.parametrize(
    ("pid", "returncode"),
    [(PID, 0), (0, None)],
    ids=["the child has exited", "a group id of 0"],
)
def test_ctrl_break_goes_only_to_a_live_child_of_ours(monkeypatch, pid, returncode):
    """An exited child's pid is no longer known to be ours, and group 0 is
    every process on litetune's console, litetune included."""
    monkeypatch.setattr(envs, "_REAP_AFTER_KILL_S", 0.2)
    log: list[tuple[Any, ...]] = []
    # Something besides the child is in the job, so it is not empty either way.
    api = FakeJobApi(log, counts=(2,), ids=(PID, 4343))

    started = time.monotonic()
    envs._kill_tree(
        FakeStage(log, pid=pid, returncode=returncode), grace=5, pgid=_job(api), pump=time.sleep
    )

    assert "send_ctrl_break" not in api.steps()
    assert time.monotonic() - started < 1, "no grace without a signal"
    assert ("terminate", JOB, 1) in log, "what the child started is still ended"


def test_a_job_that_is_already_empty_is_left_alone():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(0,))

    assert envs._kill_tree(FakeStage(log), pgid=_job(api)) is envs.Reach.ALREADY_GONE
    assert api.steps() == ["active_processes"]


def test_a_closed_job_is_already_gone_and_asked_nothing():
    """A guard left installed could otherwise act on a handle value that
    belongs to something else by now."""
    api = FakeJobApi()
    job = _job(api)
    job.close()

    assert envs._kill_tree(FakeStage(), pgid=job) is envs.Reach.ALREADY_GONE
    assert api.steps() == ["close"]


def test_a_job_that_cannot_be_counted_is_waited_on_for_the_whole_grace(monkeypatch):
    """Silence is not an empty job: the grace is waited out, as a POSIX
    grace that cannot observe the group is slept out."""
    monkeypatch.setattr(envs, "_REAP_AFTER_KILL_S", 0.1)
    log: list[tuple[Any, ...]] = []
    refused = OSError("refused")
    api = FakeJobApi(log, fail={"active_processes": refused, "process_ids": refused})

    started = time.monotonic()
    envs._kill_tree(FakeStage(log), grace=0.3, pgid=_job(api), pump=time.sleep)

    assert time.monotonic() - started >= 0.29
    assert ("terminate", JOB, 1) in log


def test_a_terminated_job_is_waited_on_until_it_is_empty(caplog, monkeypatch):
    """`TerminateJobObject` returns before the processes have gone, and the
    next step can be `provision` removing the directory they run from."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)
    api = FakeJobApi(log, counts=(2,), fail={"send_ctrl_break": OSError("no console")})
    real_terminate = api.terminate

    def terminate(job: int, code: int) -> None:
        real_terminate(job, code)
        api.counts = [2, 2, 0]

    api.terminate = terminate

    envs._kill_tree(stage, grace=5, pgid=_job(api), pump=time.sleep)

    after = _steps(log)[_steps(log).index("terminate") :]
    assert after.count("active_processes") == 3, "asked until it said empty"

    monkeypatch.setattr(envs, "_REAP_AFTER_KILL_S", 0.2)
    stuck = FakeJobApi(counts=(2,), fail={"send_ctrl_break": OSError("no console")})
    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        envs._kill_tree(FakeStage(), grace=5, pgid=_job(stuck), pump=time.sleep)
    assert "still holds processes" in caplog.text


def test_an_interrupted_grace_still_terminates_the_job():
    """The POSIX rescue, in Windows terms: a Ctrl-C during the grace must not
    take the hard kill with it."""
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,))

    def interrupted(_seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        envs._kill_tree(FakeStage(log), grace=5, pgid=_job(api), pump=interrupted)

    assert ("terminate", JOB, 1) in log


def test_a_second_signal_ends_the_grace_and_terminates_now(monkeypatch):
    """The guard's escalation: `stop` says a second signal has already killed."""
    monkeypatch.setattr(envs, "_REAP_AFTER_KILL_S", 0.1)
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,))

    started = time.monotonic()
    envs._kill_tree(FakeStage(log), grace=5, stop=lambda: True, pgid=_job(api), pump=time.sleep)

    assert time.monotonic() - started < 1
    assert ("terminate", JOB, 1) in log


def test_an_interrupted_grace_without_a_job_still_kills_the_child():
    log: list[tuple[Any, ...]] = []

    def interrupted(_seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        envs._kill_tree(
            FakeStage(log), grace=5, pgid=envs._BreakOnly(FakeJobApi(log)), pump=interrupted
        )

    assert _steps(log) == ["send_ctrl_break", "kill"]


def test_a_refused_job_termination_falls_back_to_the_child(caplog):
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,), fail={"terminate": OSError("access denied")})

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        reached = envs._kill_tree(FakeStage(log), grace=0, pgid=_job(api))

    assert reached is envs.Reach.GROUP, "the best reach achieved: the break reached the group"
    assert ("kill",) in log, "`Popen.kill` is what is left"
    assert "could not terminate the job" in caplog.text


def test_when_neither_the_break_nor_the_job_works_only_the_child_was_reached():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(
        log,
        counts=(2,),
        fail={"send_ctrl_break": OSError("no console"), "terminate": OSError("access denied")},
    )

    assert envs._kill_tree(FakeStage(log), grace=5, pgid=_job(api)) is envs.Reach.CHILD_ONLY
    assert ("kill",) in log


# ---------------------------------------------------------------------------
# The job's lifetime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answers",
    [
        (("out", "err"),),
        ("wait",),
        (KeyboardInterrupt(),),
        (envs.StageInterrupted(signal.SIGTERM),),
    ],
    ids=["completed", "timed out", "Ctrl-C", "interrupted by the guard"],
)
def test_the_job_is_closed_once_on_every_route_and_after_the_guard(windows, answers):
    """Last, and after the guard has handed its signals back: closing the job
    ends what it holds, so nothing may still be using the stage then."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=answers)
    api = FakeJobApi(log, counts=(1, 0))
    before = signal.getsignal(signal.SIGTERM)
    at_close: list[Any] = []
    real_close = api.close

    def close(job: int) -> None:
        at_close.append(signal.getsignal(signal.SIGTERM))
        real_close(job)

    api.close = close
    windows(stage, api)

    try:
        envs._run_guarded(["python", "-c", "pass"], timeout=0.3)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, envs.StageInterrupted):
        pass

    assert log.count(("close", JOB)) == 1
    assert log[-1] == ("close", JOB), "after the reap and the drain, not before"
    assert at_close == [before], "the guard was still installed when the job was closed"


def test_a_job_is_closed_once_however_often_it_is_asked():
    """A handle value can be reused once closed, so a second `CloseHandle` on
    it could close somebody else's handle."""
    api = FakeJobApi()
    job = _job(api)

    job.close()
    job.close()

    assert api.steps() == ["close"]


def test_a_job_that_will_not_close_is_said_and_does_not_replace_the_result(windows, caplog):
    log: list[tuple[Any, ...]] = []
    windows(FakeStage(log), FakeJobApi(log, fail={"close": OSError("invalid handle")}))

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        result = envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert result.stdout == "out"
    assert "could not close the job object" in caplog.text


def test_a_stage_that_finishes_with_processes_left_is_told_they_are_ended(windows, caplog):
    """Closing the job ends them, which a POSIX stage's process group would
    not do, so it is said -- with how many."""
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(3,), ids=(PID, 7, 8))
    real_terminate = api.terminate

    def terminate(job: int, code: int) -> None:
        real_terminate(job, code)
        api.counts, api.ids = [0], []

    api.terminate = terminate
    windows(FakeStage(log), api)

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        result = envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert result.stdout == "out"
    assert "finished with 2 processes it started still running" in caplog.text
    steps = _steps(log)
    assert steps.index("terminate") < steps.index("close") == len(steps) - 1


def test_a_stage_that_did_not_finish_is_not_said_to_have_finished(windows, caplog, monkeypatch):
    """A job left holding processes after an interrupted stage is ended all
    the same, but the warning about a *finished* stage leaving processes
    behind would be false there."""
    monkeypatch.setattr(envs, "_REAP_AFTER_KILL_S", 0.1)
    log: list[tuple[Any, ...]] = []
    stuck = FakeJobApi(
        log, counts=(2,), ids=(PID, 7), fail={"send_ctrl_break": OSError("no console")}
    )
    windows(FakeStage(log, answers=(KeyboardInterrupt(),)), stuck)

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        with pytest.raises(KeyboardInterrupt):
            envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert "still holds processes" in caplog.text
    assert "finished with" not in caplog.text
    assert log[-1] == ("close", JOB)


def test_a_timed_out_stage_is_killed_through_its_job_and_reports_its_own_timeout(
    windows, monkeypatch
):
    """The caller-visible contract does not move: the `TimeoutExpired` names the
    stage's timeout, not a slice's, and carries what the drain read."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))
    windows(stage, FakeJobApi(log, counts=(1, 0)))
    handed: list[Any] = []

    def drained(proc, pgid=None, already=(None, None), *, seed=True):
        handed.append(pgid)
        return "said before the end", "and on stderr"

    monkeypatch.setattr(envs, "_kill_and_drain", drained)

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        envs._run_guarded(["python", "-c", "pass"], timeout=0.6)

    assert caught.value.timeout == 0.6
    assert caught.value.output == "said before the end"
    assert caught.value.stderr == "and on stderr"
    assert isinstance(handed[0], envs._StageJob), "the kill goes through the job"


class _Read:
    """A stream `communicate` has read to its end and closed."""

    closed = True
    encoding = "utf-8"


def test_what_windows_communicate_read_survives_pipes_closed_before_the_hang(monkeypatch):
    """On Windows `_communicate` reads each pipe on a thread into `_stdout_buff`
    and `_stderr_buff` and closes the streams once both end -- even when its
    trailing `wait()` then times out. A stage that closed its pipes and hung
    has said everything there and nothing on the descriptors."""
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    stage = FakeStage()
    stage.stdout, stage.stderr = _Read(), _Read()
    stage._stdout_buff = ["out-1\nout-2\n\udc80\n"]
    stage._stderr_buff = ["err-1\n"]

    out, err = envs._kill_and_drain(stage, None, (None, None))

    assert (out, err) == ("out-1\nout-2\n\udc80\n", "err-1\n")


def test_a_reader_that_has_not_finished_holds_nothing_yet():
    stage = FakeStage()
    stage.stdout = _Read()
    stage._stdout_buff = []

    assert envs._buffered_by_communicate(stage, stage.stdout) is None
    assert envs._buffered_by_communicate(FakeStage(), _Read()) is None, "not one of its pipes"


# ---------------------------------------------------------------------------
# The wait: in slices, so a signal handler gets to run
# ---------------------------------------------------------------------------


def test_the_wait_comes_back_to_the_interpreter_and_keeps_what_was_said():
    clock = FakeClock()
    stage = FakeStage(answers=("wait", "wait", "wait", ("all of it", "and stderr")), clock=clock)

    out, err = envs._communicate_in_slices(stage, timeout=30, clock=clock)

    assert (out, err) == ("all of it", "and stderr")
    assert stage.waits == [0.25, 0.25, 0.25, 0.25]


def test_the_slices_stop_at_the_stages_own_deadline():
    clock = FakeClock()
    stage = FakeStage(answers=("wait",), clock=clock)

    with pytest.raises(subprocess.TimeoutExpired):
        envs._communicate_in_slices(stage, timeout=0.6, clock=clock)

    assert stage.waits == pytest.approx([0.25, 0.25, 0.1])


def test_what_was_held_back_is_delivered_once_after_the_first_wait():
    clock = FakeClock()
    delivered: list[int] = []
    stage = FakeStage(answers=("wait", "wait", ("out", "err")), clock=clock)

    def deliver() -> None:
        delivered.append(len(stage.waits))

    envs._communicate_in_slices(stage, timeout=5, after_first_wait=deliver, clock=clock)
    assert delivered == [1]

    done = FakeStage(clock=clock)
    after: list[int] = []
    envs._communicate_in_slices(
        done, timeout=5, after_first_wait=lambda: after.append(len(done.waits)), clock=clock
    )
    assert after == [1], "a stage done in the first wait still gets them"


def test_an_interrupt_reaches_a_windows_wait_within_a_slice(windows):
    """A `communicate` that cannot be interrupted -- the Windows `join` before
    3.14, and a SIGBREAK handler on any version -- is stood in for by one that
    sleeps its whole timeout. `interrupt_main` schedules the SIGINT handler
    without sending anything, so the sleep is not cut short, exactly as the
    join is not: only coming back to the interpreter lets it run."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("interrupt", "wait"))
    windows(stage, FakeJobApi(log, counts=(1, 0)))

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert time.monotonic() - started < 2, "the interrupt waited for the stage's deadline"
    assert len(stage.waits) == 1, "it landed as the first slice returned"
    assert ("send_ctrl_break", PID) in log, "and the stage was stopped on its way out"


# ---------------------------------------------------------------------------
# The guard: SIGBREAK, and no re-delivery on Windows
# ---------------------------------------------------------------------------


def _a_sigbreak(monkeypatch) -> int:
    """SIGBREAK where the platform has it; elsewhere a stand-in under that name."""
    real = getattr(signal, "SIGBREAK", None)
    if real is not None:
        return int(real)
    stand_in = int(signal.SIGUSR1)  # type: ignore[attr-defined,unused-ignore]
    monkeypatch.setattr(signal, "SIGBREAK", stand_in, raising=False)
    return stand_in


def test_sigbreak_is_guarded_where_it_exists(monkeypatch):
    sig = _a_sigbreak(monkeypatch)
    before = signal.getsignal(sig)

    with envs._kill_child_if_we_are_told_to_exit(FakeStage()):
        installed = signal.getsignal(sig)

    assert callable(installed) and installed is not before
    assert signal.getsignal(sig) == before, "restored on the way out"


def test_an_ignored_sigbreak_stays_ignored(monkeypatch):
    sig = _a_sigbreak(monkeypatch)
    previous = signal.signal(sig, signal.SIG_IGN)
    try:
        with envs._kill_child_if_we_are_told_to_exit(FakeStage()):
            assert signal.getsignal(sig) == signal.SIG_IGN
    finally:
        signal.signal(sig, previous)


def test_on_windows_the_guard_raises_instead_of_ending_litetune(monkeypatch):
    """`os.kill(os.getpid(), SIGBREAK)` on Windows is `TerminateProcess(self,
    21)`: litetune would exit 21, past `cli`'s exit 4. So the handler kills
    the tree and raises `StageInterrupted` there, and re-delivers nowhere."""
    sig = _a_sigbreak(monkeypatch)
    monkeypatch.setattr(envs.sys, "platform", "win32")
    redelivered: list[tuple[int, int]] = []
    monkeypatch.setattr(envs.os, "kill", lambda pid, s: redelivered.append((pid, s)))
    killed: list[Any] = []
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: killed.append(kwargs["pgid"]))
    job = _job(FakeJobApi())

    with envs._kill_child_if_we_are_told_to_exit(FakeStage(), job):
        handler = signal.getsignal(sig)
        assert callable(handler)
        with pytest.raises(envs.StageInterrupted) as caught:
            handler(sig, None)

    assert caught.value.signum == sig
    assert redelivered == []
    assert killed == [job], "the tree is taken down first, through the job"


def test_a_held_sigbreak_reaches_the_guard_once_it_is_installed(windows, monkeypatch):
    """Delivered to whatever handles it by then -- the guard -- so a Ctrl-Break
    during the start ends as one during the wait does: `StageInterrupted`.

    A handler of the test's own stands where the default would be, so a
    signal that reached anything but the guard fails here rather than ending
    the test run."""
    sig = _a_sigbreak(monkeypatch)
    monkeypatch.setattr(envs.sys, "platform", "win32")
    monkeypatch.setattr(envs.os, "kill", lambda pid, s: None)
    log: list[tuple[Any, ...]] = []
    windows(
        FakeStage(log, answers=("wait",)),
        FakeJobApi(log, counts=(1, 0)),
        before_popen=lambda: signal.raise_signal(sig),
    )
    elsewhere: list[int] = []
    previous = signal.signal(sig, lambda signum, frame: elsewhere.append(signum))
    try:
        with pytest.raises(envs.StageInterrupted) as caught:
            envs._run_guarded(["python", "-c", "pass"], timeout=5)
    finally:
        signal.signal(sig, previous)

    assert elsewhere == [], "the held Ctrl-Break reached something other than the guard"
    assert caught.value.signum == sig
    assert ("send_ctrl_break", PID) in log and log[-1] == ("close", JOB)


# ---------------------------------------------------------------------------
# Where the job API comes from
# ---------------------------------------------------------------------------


def test_only_windows_gets_a_job_api(monkeypatch):
    monkeypatch.setattr(envs.sys, "platform", "linux")
    assert envs._windows_job_api() is None

    api = FakeJobApi()
    fake = types.SimpleNamespace(API=api)
    monkeypatch.setattr(envs.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "litetune.winjob", fake)
    monkeypatch.setattr(litetune, "winjob", fake, raising=False)
    assert envs._windows_job_api() is api


def test_a_windows_host_that_cannot_load_winjob_runs_stages_as_before(monkeypatch, caplog):
    """A CPython built without `_ctypes` cannot import `winjob`: the host is
    told once, and its stages start as they did before jobs."""
    monkeypatch.setattr(envs, "_NO_JOB_WARNED", False)
    monkeypatch.setattr(envs.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "litetune.winjob", None)
    monkeypatch.delattr(litetune, "winjob", raising=False)

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        assert envs._windows_job_api() is None
        assert envs._windows_job_api() is None

    warned = [r for r in caplog.records if "could not load litetune.winjob" in r.getMessage()]
    assert len(warned) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="the refusal is what other platforms get")
def test_winjob_refuses_to_import_off_windows():
    with pytest.raises(ImportError, match="only on Windows"):
        import litetune.winjob  # noqa: F401


# ---------------------------------------------------------------------------
# How a stopped stage's code reads
# ---------------------------------------------------------------------------


def test_status_control_c_exit_reads_as_a_stop_not_a_crash_or_memory():
    detail = read_returncode(0xC000013A).describe("the model")

    assert detail.startswith(
        "terminated by STATUS_CONTROL_C_EXIT (return code 3221225786): read as unperformed "
        "rather than as a verdict about the model. STATUS_CONTROL_C_EXIT is how a console "
        "process ends on a Ctrl-C or Ctrl-Break it does not handle. Every Ctrl-Break litetune "
        "sends a stage ends in a raised exception rather than a return code, so this did not "
        "come from litetune stopping it."
    ), detail
    assert "The process was stopped, not crashed" in detail
    assert "unhandled exception" not in detail, "the exception hint is for exceptions"
    assert "out-of-memory" not in detail and "more memory" not in detail
    assert detail.endswith(".")
    assert "unhandled exception" in read_returncode(0xC0000005).describe("the model")
