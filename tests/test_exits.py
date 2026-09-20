"""A negative return code is a signal, not an exit status.

The measured failure these tests pin down: a Gemma 4 export returned `-9`, which
was read as "ran and failed" and struck the model from the catalogue. `-9` is
SIGKILL -- the out-of-memory killer at a 32 GiB ceiling -- and on a larger
machine the same command produced a specific, actionable error. So every place
litetune interprets a return code has to report `could not check` for a
signalled process, and each of those places is exercised below.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from conftest import mark_provisioned

from litetune import envs
from litetune.checks import Outcome
from litetune.evaluate import LiteRtLmBackend
from litetune.exits import (
    _NTSTATUS,
    _TERMINATING_NTSTATUS,
    NTSTATUS_ERROR,
    SIGKILL,
    STATUS_NO_MEMORY,
    read_returncode,
)
from litetune.export import ExportRequest, export_recipe, run_export
from litetune.tune import TRAINING_CHECK, run_tune

# ---------------------------------------------------------------------------
# The reading itself
# ---------------------------------------------------------------------------


def test_a_clean_exit_is_conclusive():
    reading = read_returncode(0)
    assert reading.conclusive
    assert reading.ok
    assert not reading.killed


def test_a_non_zero_exit_is_still_the_program_answering():
    reading = read_returncode(1)
    assert reading.conclusive
    assert not reading.ok
    assert reading.signal is None
    assert "exited 1" in reading.describe()


def test_minus_nine_is_a_kill_and_names_the_oom_killer():
    reading = read_returncode(-SIGKILL)
    assert not reading.conclusive
    assert reading.killed
    assert reading.signal == SIGKILL
    assert reading.signal_name == "SIGKILL"
    detail = reading.describe("the model")
    assert "SIGKILL" in detail
    assert "out-of-memory" in detail
    assert "not about the model" in detail


def test_other_signals_are_named_too():
    reading = read_returncode(-15)
    assert reading.signal_name == "SIGTERM"
    assert not reading.conclusive
    # Only SIGKILL carries the memory story; SIGTERM has other senders.
    assert "out-of-memory" not in reading.describe()


def test_a_shell_style_137_is_not_reinterpreted():
    # litetune never runs a subprocess through a shell, so 137 here came from a
    # program that chose to exit 137. Inventing a signal would be worse.
    reading = read_returncode(137)
    assert reading.conclusive
    assert reading.signal is None


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


# The two spellings of "this process was killed": POSIX's negative signal and
# the Windows exit code that carries the exception's NTSTATUS. Every stage
# below is exercised in both, because the reading is the same fact and the
# callers must not learn to recognise only one of them.
KILL_CODES = pytest.mark.parametrize(
    "kill_code, named",
    [(-SIGKILL, "SIGKILL"), (0xC0000005, "STATUS_ACCESS_VIOLATION")],
    ids=["sigkill", "ntstatus"],
)


@pytest.fixture
def killing_toolchain(monkeypatch, tmp_path, request):
    """A stage environment whose every command is killed.

    Parametrised through `KILL_CODES` where the test takes `kill_code`, and
    SIGKILL for the tests that do not.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    params = getattr(request.node, "callspec", None)
    kill_code = params.params["kill_code"] if params else -SIGKILL

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    def fake_run(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, 0, "transformers==5.5.0\n", "")
        return subprocess.CompletedProcess(args, kill_code, "", "")

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake_run)


@KILL_CODES
def test_a_killed_export_is_could_not_check_not_a_failed_recipe(
    killing_toolchain, tmp_path, kill_code, named
):
    request = ExportRequest(
        model="google/functiongemma-270m-it",
        output_dir=tmp_path / "out",
        recipes=("dynamic_wi8_afp32",),
    )
    export = export_recipe(request, "dynamic_wi8_afp32")
    assert export.check.outcome is Outcome.UNCHECKED
    assert not export.ok
    assert named in export.check.detail
    exit_record = export.check.observed["exit"]
    assert exit_record["killed_by_signal"] == (SIGKILL if kill_code < 0 else None)
    assert exit_record["terminated_by_status"] == (None if kill_code < 0 else named)
    assert exit_record["conclusive"] is False


@KILL_CODES
def test_a_killed_sweep_does_not_report_a_verdict(killing_toolchain, tmp_path, kill_code, named):
    result = run_export(
        ExportRequest(
            model="google/functiongemma-270m-it",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32", "weight_only_wi8_afp32"),
        )
    )
    assert result.outcome is Outcome.UNCHECKED
    assert result.failed == []


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------


@KILL_CODES
def test_a_killed_training_run_is_could_not_check(killing_toolchain, tmp_path, kill_code, named):
    from litetune.prompt_mode import PromptMode
    from litetune.tune import TuneRequest

    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "a", "completion": "call:a{}"}\n', encoding="utf-8")
    result = run_tune(
        TuneRequest(
            model="google/functiongemma-270m-it",
            data=data,
            output_dir=tmp_path / "run",
            # Bare text, which is what this mode trains; `tune` refuses
            # `prerendered` on it before anything else is checked.
            prompt_mode=PromptMode.RUNTIME_RENDERED,
        )
    )
    training = next(c for c in result.checks.checks if c.name == TRAINING_CHECK)
    assert training.outcome is Outcome.UNCHECKED
    assert named in training.detail
    assert result.outcome is Outcome.UNCHECKED


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@KILL_CODES
def test_a_killed_generation_is_not_performed(monkeypatch, tmp_path, kill_code, named):
    def fake_run(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, kill_code, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    generation = LiteRtLmBackend(model=tmp_path / "model.litertlm", auto_provision=False).generate(
        ["hello"]
    )[0]
    # `ran` is False, so the liveness tier reports could_not_check rather than
    # scoring an empty generation as a failure.
    assert not generation.ran
    assert generation.harness_error is not None
    assert named in generation.harness_error


def test_an_ntstatus_is_a_kill_even_though_it_is_positive():
    """Windows has no negative return codes, and the distinction still has to hold.

    `Popen.returncode` says "a negative value -N indicates that the child was
    terminated by signal N (POSIX only)". On Windows an unhandled exception
    ends the process with its NTSTATUS as the exit code, so 0xC0000005 is the
    same fact `-11` is here: the process never chose a status. Read as a
    verdict it says "the model does not generate" about a machine fault, which
    is the Gemma 4 mistake this module exists to prevent.
    """
    reading = read_returncode(0xC0000005)

    assert not reading.conclusive
    assert reading.killed
    assert reading.status == "STATUS_ACCESS_VIOLATION"
    assert reading.signal is None, "there is no signal on Windows to name"
    detail = reading.describe("the model")
    assert "STATUS_ACCESS_VIOLATION" in detail
    assert "unperformed rather than as a verdict about the model" in detail
    assert (
        "cannot tell an exception from a program that chose this number" in detail
    ), "the reading is a classification, not a proof: `ExitProcess` takes any code"
    assert reading.as_dict()["terminated_by_status"] == "STATUS_ACCESS_VIOLATION"


def test_an_unnamed_ntstatus_is_still_not_a_verdict():
    """The range decides, not the table: a status nobody listed is still a kill."""
    reading = read_returncode(0xC0000123)

    assert not reading.conclusive
    assert reading.status == "NTSTATUS 0xC0000123"
    assert (
        "0xC0000123" in reading.describe()
    ), "uppercase, so an unnamed status greps against the same list the names came from"


def test_a_code_below_the_ntstatus_range_is_a_program_that_chose_it():
    """The boundary, so the rule cannot grow into ordinary exit statuses.

    Everything a program chooses is far below this, and the codes litetune's
    own table gives meaning to -- 2, 3, 4 -- are the ones that must never be
    read as a crash.
    """
    for code in (0, 1, 3, 4, 255, NTSTATUS_ERROR - 1):
        reading = read_returncode(code)
        assert reading.conclusive, code
        assert reading.status is None, code


def test_the_first_code_in_the_error_range_is_already_a_kill():
    """The floor, from above. Without this `>=` and `>` are the same rule."""
    assert not read_returncode(NTSTATUS_ERROR).conclusive
    assert read_returncode(NTSTATUS_ERROR - 1).conclusive


def test_a_negative_exit_code_is_the_program_talking_not_a_crash():
    """`exit(-1)` arrives as 0xFFFFFFFF, inside the range and not a crash.

    Windows hands the exit code back as a DWORD, so every `exit(-N)` lands in
    the error-severity range while plainly being a program that chose to fail.
    MS-ERREF 2.3 separates them: a Microsoft-defined status has bit 29 (C,
    customer) and bit 28 (N, reserved) clear, and a two's-complement negative
    has both set. Reading these as kills would turn an ordinary failure into
    "not measured", which is the quieter of the two wrong answers and still a
    wrong answer.
    """
    for n in (1, 2, 100, 255, 1000):
        code = (-n) & 0xFFFFFFFF
        reading = read_returncode(code)
        assert reading.conclusive, f"exit(-{n}) -> {code:#x}"
        assert reading.status is None
    assert all(
        status & 0x30000000 == 0 for status in _NTSTATUS
    ), "every status this module names is Microsoft-defined, which is what the rule tests"


def test_what_the_rule_cannot_separate_is_read_the_conservative_way():
    """Where the two spellings overlap, and which way the rule falls.

    `exit(-0x30000001)` is 0xCFFFFFFF, which carries neither bit and is read
    as a kill; `ExitProcess` can be handed 0xC0000005 outright. Neither is
    decidable from an exit code, and the module says so -- this pins the
    direction, so a future edit choosing the other one has to say why.
    """
    assert not read_returncode((-0x30000001) & 0xFFFFFFFF).conclusive
    assert not read_returncode(0xC0000005).conclusive


def test_the_windows_spelling_of_a_memory_kill_carries_the_memory_hint():
    """`-9` and STATUS_NO_MEMORY are the same event; the hint must reach both readers."""
    detail = read_returncode(0xC0000017).describe("the model")

    assert "STATUS_NO_MEMORY" in detail
    assert "more memory" in detail
    assert "Slurm" not in detail, (
        "the Linux hint sends the reader after a supervisor that cancelled the job; "
        "on Windows nothing chose this process, and the hint says so"
    )


def test_the_hint_ends_a_sentence_so_a_caller_can_add_its_own():
    """`describe()` is interpolated mid-sentence by three callers.

    `prepare` appends "without counts: <stderr>", `rendering` and `toolpath`
    append their own tails. A hint ending on a dangling "is not" made the
    joined sentence say the opposite of what this module means.
    """
    for code in (0xC0000005, STATUS_NO_MEMORY):
        detail = read_returncode(code).describe("the model")
        assert detail.endswith("."), detail[-80:]


def test_a_terminating_status_below_the_error_range_is_a_kill_too():
    """Severity is not the whole story: a breakpoint is an exception, not an exit.

    `STATUS_BREAKPOINT` has warning severity, and a process that hits one with
    no debugger attached ends with it as its code. The range test alone would
    read that as a program that chose 2147483651.
    """
    for code, name in ((0x80000003, "STATUS_BREAKPOINT"), (0x40000015, "STATUS_FATAL_APP_EXIT")):
        reading = read_returncode(code)
        assert not reading.conclusive, name
        assert reading.status == name


def test_every_status_the_rule_classifies_below_the_range_has_a_name():
    """The set decides and the table names; a member of one belongs in the other.

    They are separate on purpose -- a name added for a reader must not change
    a verdict -- and this is what that separation costs: a classified status
    with no name would be described as a bare number.
    """
    assert set(_TERMINATING_NTSTATUS) <= set(_NTSTATUS)
    assert all(
        code < NTSTATUS_ERROR for code in _TERMINATING_NTSTATUS
    ), "a status in the error range is already classified by the range"
