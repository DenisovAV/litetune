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
import threading
import time
import types
from typing import Any

import pytest

import litetune
from litetune import envs
from litetune.exits import read_returncode

JOB = 100
PID = 4242
FLAGS = 0x00000004 | 0x00000200  # CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP


class FakeJobApi:
    """The Win32 job calls, recorded in one log with whatever else the test records.

    A step named in `fail` raises what it maps to. `counts` is what
    `active_processes` answers, one entry per call, the last one repeated.
    """

    def __init__(
        self,
        log: list[tuple[Any, ...]] | None = None,
        *,
        fail: dict[str, BaseException] | None = None,
        counts: tuple[int, ...] = (1,),
    ) -> None:
        self.log = log if log is not None else []
        self.fail = fail or {}
        self.counts = list(counts)

    def _step(self, *call: Any) -> None:
        self.log.append(call)
        if call[0] in self.fail:
            raise self.fail[call[0]]

    def create_job(self) -> int:
        self._step("create_job")
        return JOB

    def assign(self, job: int, proc: subprocess.Popen) -> None:
        self._step("assign", job, proc.pid)

    def resume(self, proc: subprocess.Popen) -> None:
        self._step("resume", proc.pid)

    def active_processes(self, job: int) -> int:
        self._step("active_processes", job)
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]

    def terminate(self, job: int, exit_code: int) -> None:
        self._step("terminate", job, exit_code)

    def close(self, job: int) -> None:
        self._step("close", job)

    def send_ctrl_break(self, pid: int) -> None:
        self._step("send_ctrl_break", pid)

    def steps(self) -> list[Any]:
        return [call[0] for call in self.log]


# Checked by mypy: a renamed or re-typed `_JobApi` method fails here, not in CI.
_CONFORMS: envs._JobApi = FakeJobApi()


_MOST_WAITS = 40


class FakeStage:
    """A stage child on Windows: never a real process, and every touch recorded.

    `answers` is what successive `communicate` calls do: a tuple is returned
    (and the child has exited), `"wait"` is a `TimeoutExpired` after sleeping
    the timeout out -- which is what `Popen.communicate` does on Windows while
    the child runs -- and an exception is raised.
    """

    args = ["python"]
    stdout = None
    stderr = None

    def __init__(
        self,
        log: list[tuple[Any, ...]] | None = None,
        *,
        pid: int = PID,
        answers: tuple[Any, ...] = (("out", "err"),),
        returncode: int | None = None,
    ) -> None:
        self.log = log if log is not None else []
        self.pid = pid
        self.answers = list(answers)
        self.returncode = returncode
        self.waits: list[float | None] = []

    def communicate(self, timeout: float | None = None) -> tuple[Any, Any]:
        if len(self.waits) >= _MOST_WAITS:
            # Ten seconds of slices: no test here gives a stage that long, so
            # a wait still going is one that would never stop.
            raise AssertionError("still waiting in slices long past every deadline here")
        self.log.append(("communicate", timeout))
        self.waits.append(timeout)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if answer == "wait":
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
        self.returncode = 1

    def send_signal(self, sig: int) -> None:
        # `Popen.send_signal(SIGTERM)` is `terminate()` on Windows.
        self.log.append(("send_signal", sig))
        self.returncode = 1


@pytest.fixture
def windows(monkeypatch):
    """Run `_run_guarded` the Windows way: a fake API, and a fresh "warned" flag.

    Yields a function that installs a stage and an API sharing one log, so a
    test can read the order of everything that happened.
    """
    monkeypatch.setattr(envs, "_NO_JOB_WARNED", False)

    def install(stage: FakeStage, api: FakeJobApi) -> dict[str, Any]:
        started: dict[str, Any] = {}

        def popen(argv, **kwargs):
            started.update(kwargs)
            stage.log.append(("popen", kwargs.get("creationflags")))
            return stage

        monkeypatch.setattr(envs.subprocess, "Popen", popen)
        monkeypatch.setattr(envs, "_windows_job_api", lambda: api)
        return started

    return install


def _job(api: FakeJobApi) -> envs._StageJob:
    return envs._StageJob(api, JOB)


# ---------------------------------------------------------------------------
# Starting a stage: suspended, assigned, resumed
# ---------------------------------------------------------------------------


def test_a_windows_stage_is_in_its_job_before_it_runs(windows):
    """The order is the whole point. Assigned after an unsuspended start, a
    child has a window in which a grandchild is created outside the job, and
    nothing would ever collect that one."""
    log: list[tuple[Any, ...]] = []
    stage, api = FakeStage(log), FakeJobApi(log)
    started = windows(stage, api)

    result = envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert started["creationflags"] == FLAGS, "suspended, and the root of a process group"
    assert started["start_new_session"] is True, "unchanged for POSIX, ignored on Windows"
    steps = [call[0] for call in log]
    assert steps[:4] == ["popen", "create_job", "assign", "resume"], steps
    assert ("assign", JOB, PID) in log and ("resume", PID) in log
    assert steps.index("communicate") > steps.index("resume"), "nothing waits on a suspended child"
    assert log[-1] == ("close", JOB), "the job is let go of last, once the stage is done"
    assert (result.returncode, result.stdout, result.stderr) == (0, "out", "err")


class _Pipe:
    """Our end of a stage pipe, as far as closing it goes."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_a_child_that_cannot_be_resumed_is_terminated_and_never_waited_on(windows):
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)
    pipes = _Pipe(), _Pipe()
    stage.stdout, stage.stderr = pipes
    refused = OSError("ResumeThread failed")
    api = FakeJobApi(log, fail={"resume": refused})
    windows(stage, api)

    with pytest.raises(OSError, match="could not resume python") as caught:
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert caught.value.__cause__ is refused
    steps = [call[0] for call in log]
    assert "communicate" not in steps, "a suspended child waits for ever"
    assert steps.index("kill") > steps.index("resume")
    assert ("close", JOB) in log, "the job goes with it"
    assert steps.index("wait") > steps.index("kill"), "and the terminated child is reaped"
    assert all(pipe.closed for pipe in pipes), "nothing else will close our ends of its pipes"


def test_an_interrupt_before_the_resume_terminates_the_child_and_says_so_as_itself(windows):
    """Anything that cuts the start short is answered the same way, but a
    Ctrl-C stays a Ctrl-C: it is not dressed up as a failed resume."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log)
    api = FakeJobApi(log, fail={"assign": KeyboardInterrupt()})
    windows(stage, api)

    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    steps = [call[0] for call in log]
    assert "resume" not in steps and "communicate" not in steps
    assert "kill" in steps and ("close", JOB) in log


@pytest.mark.parametrize("step", ["create_job", "assign"])
def test_a_host_without_jobs_warns_once_and_runs_the_stage_anyway(windows, caplog, step):
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, fail={step: OSError("access denied")})

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        for _ in range(2):
            stage = FakeStage(log)
            windows(stage, api)
            result = envs._run_guarded(["python", "-c", "pass"], timeout=5)
            assert result.stdout == "out"

    warned = [r for r in caplog.records if "could not put" in r.getMessage()]
    assert len(warned) == 1, "once per process, not once per stage"
    assert "descendants cannot be collected" in warned[0].getMessage()
    assert log.count(("resume", PID)) == 2, "resumed anyway: a suspended child would wait for ever"
    if step == "assign":
        # The job that could not take the child is let go of before it runs,
        # and nothing is closed at the end, because there is no job then.
        assert log.count(("close", JOB)) == 2
        for first, second in zip(log, log[1:], strict=False):
            if first[0] == "assign":
                assert second == ("close", JOB)
    else:
        assert ("close", JOB) not in log


def test_without_a_job_a_stage_is_stopped_the_way_it_always_was(windows, monkeypatch):
    """No job, no Ctrl-Break and no job termination: `Popen` alone, as before."""
    monkeypatch.setattr(envs.sys, "platform", "win32")
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))
    api = FakeJobApi(log, fail={"assign": OSError("access denied")})
    windows(stage, api)

    with pytest.raises(subprocess.TimeoutExpired):
        envs._run_guarded(["python", "-c", "pass"], timeout=0.3)

    steps = [call[0] for call in log]
    assert "send_ctrl_break" not in steps and "terminate" not in steps
    assert ("send_signal", signal.SIGTERM) in log, "today's route: `Popen.send_signal`"


# ---------------------------------------------------------------------------
# Stopping a stage: Ctrl-Break, a grace with the pipes pumped, then the job
# ---------------------------------------------------------------------------


def test_stopping_a_job_breaks_then_waits_pumping_then_terminates():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,))
    stage = FakeStage(log)

    reached = envs._kill_tree(
        stage, grace=0.2, pgid=_job(api), pump=lambda s: log.append(("pump", s))
    )

    assert reached is envs.Reach.GROUP
    steps = [call[0] for call in log]
    assert ("send_ctrl_break", PID) in log
    assert steps.index("pump") > steps.index("send_ctrl_break"), "the grace comes after the break"
    assert steps[-1] == "terminate" and log[-1] == ("terminate", JOB, 1)
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


def test_without_a_console_there_is_no_grace(caplog):
    """`GenerateConsoleCtrlEvent` fails when litetune has no console, and a
    grace with no signal to use it is only a delay."""
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,), fail={"send_ctrl_break": OSError("no console")})
    pumped: list[float] = []

    started = time.monotonic()
    with caplog.at_level(logging.DEBUG, logger="litetune.envs"):
        reached = envs._kill_tree(FakeStage(log), grace=5, pgid=_job(api), pump=pumped.append)

    assert time.monotonic() - started < 1
    assert pumped == []
    assert ("terminate", JOB, 1) in log
    assert reached is envs.Reach.GROUP
    assert "no Ctrl-Break" in caplog.text


@pytest.mark.parametrize(
    ("pid", "returncode"),
    [(PID, 0), (0, None)],
    ids=["the child has exited", "a group id of 0"],
)
def test_ctrl_break_goes_only_to_a_live_child_of_ours(pid, returncode):
    """An exited child's pid is no longer known to be ours, and group 0 is
    every process on litetune's console, litetune included."""
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(1,))
    pumped: list[float] = []

    envs._kill_tree(
        FakeStage(log, pid=pid, returncode=returncode), grace=5, pgid=_job(api), pump=pumped.append
    )

    assert "send_ctrl_break" not in api.steps()
    assert pumped == [], "no grace without a signal"
    assert ("terminate", JOB, 1) in log, "what the child started is still ended"


def test_a_job_that_is_already_empty_is_left_alone():
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(0,))

    assert envs._kill_tree(FakeStage(log), pgid=_job(api)) is envs.Reach.ALREADY_GONE
    assert api.steps() == ["active_processes"]


def test_a_job_that_cannot_be_counted_is_waited_on_for_the_whole_grace():
    """Silence is not an empty job: the grace is waited out, as a POSIX
    grace that cannot observe the group is slept out."""
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, fail={"active_processes": OSError("refused")})

    started = time.monotonic()
    envs._kill_tree(FakeStage(log), grace=0.3, pgid=_job(api), pump=time.sleep)

    assert time.monotonic() - started >= 0.29
    assert ("terminate", JOB, 1) in log


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


def test_a_second_signal_ends_the_grace_and_terminates_now():
    """The guard's escalation: `stop` says a second signal has already killed."""
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,))

    started = time.monotonic()
    envs._kill_tree(FakeStage(log), grace=5, stop=lambda: True, pgid=_job(api), pump=time.sleep)

    assert time.monotonic() - started < 1
    assert ("terminate", JOB, 1) in log


def test_a_refused_job_termination_falls_back_to_the_child(caplog):
    log: list[tuple[Any, ...]] = []
    api = FakeJobApi(log, counts=(2,), fail={"terminate": OSError("access denied")})
    stage = FakeStage(log)

    with caplog.at_level(logging.WARNING, logger="litetune.envs"):
        reached = envs._kill_tree(stage, grace=0, pgid=_job(api))

    assert reached is envs.Reach.GROUP, "the best reach achieved: the break reached the group"
    assert ("kill",) in log, "`Popen.kill` is what is left"
    assert "could not terminate the job" in caplog.text


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
def test_the_job_is_closed_once_on_every_route_and_last(windows, answers):
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=answers)
    api = FakeJobApi(log, counts=(1, 0))
    windows(stage, api)

    try:
        envs._run_guarded(["python", "-c", "pass"], timeout=0.3)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, envs.StageInterrupted):
        pass

    assert log.count(("close", JOB)) == 1
    assert log[-1] == ("close", JOB), "after the reap and the drain, not before"


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


def test_a_timed_out_stage_is_killed_through_its_job_and_reports_its_own_timeout(
    windows, monkeypatch
):
    """The caller-visible contract does not move: the `TimeoutExpired` names the
    stage's timeout, not a slice's, and carries what the drain read."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))
    windows(stage, FakeJobApi(log))
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


# ---------------------------------------------------------------------------
# The wait: in slices, so a signal handler gets to run
# ---------------------------------------------------------------------------


def test_the_wait_comes_back_to_the_interpreter_and_keeps_what_was_said():
    stage = FakeStage(answers=("wait", "wait", "wait", ("all of it", "and stderr")))

    out, err = envs._communicate_in_slices(stage, timeout=30)

    assert (out, err) == ("all of it", "and stderr")
    assert len(stage.waits) == 4
    assert all(w is not None and w <= envs._INTERRUPT_POLL_S for w in stage.waits)


def test_the_slices_stop_at_the_stages_own_deadline():
    stage = FakeStage(answers=("wait",))

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        envs._communicate_in_slices(stage, timeout=0.6)

    assert 0.55 <= time.monotonic() - started < 1.5
    assert len(stage.waits) >= 3


def test_an_interrupt_reaches_a_windows_wait_within_a_slice(windows):
    """A `communicate` that cannot be interrupted -- the Windows `join` before
    3.14, and a SIGBREAK handler on any version -- is stood in for by one that
    sleeps its whole timeout. `interrupt_main` schedules the SIGINT handler
    without sending anything, so the sleep is not cut short, exactly as the
    join is not: only coming back to the interpreter lets it run."""
    log: list[tuple[Any, ...]] = []
    stage = FakeStage(log, answers=("wait",))
    windows(stage, FakeJobApi(log, counts=(1, 0)))
    threading.Timer(0.3, _thread.interrupt_main).start()

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert time.monotonic() - started < 2, "the interrupt waited for the stage's deadline"
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


@pytest.mark.skipif(sys.platform == "win32", reason="the refusal is what other platforms get")
def test_winjob_refuses_to_import_off_windows():
    with pytest.raises(ImportError, match="only on Windows"):
        import litetune.winjob  # noqa: F401


# ---------------------------------------------------------------------------
# How a stopped stage's code reads
# ---------------------------------------------------------------------------


def test_status_control_c_exit_reads_as_a_stop_not_a_crash_or_memory():
    detail = read_returncode(0xC000013A).describe("the model")

    assert "STATUS_CONTROL_C_EXIT" in detail
    assert "unperformed rather than as a verdict about the model" in detail
    assert "litetune sends Ctrl-Break to a stage it is stopping" in detail
    assert "not as a crash" in detail
    assert "unhandled exception" not in detail, "the exception hint is for exceptions"
    assert "out-of-memory" not in detail and "more memory" not in detail
    assert detail.endswith(".")
    assert "unhandled exception" in read_returncode(0xC0000005).describe("the model")
