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

from litetune import envs
from litetune.checks import Outcome
from litetune.evaluate import LiteRtLmBackend
from litetune.exits import SIGKILL, read_returncode
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


@pytest.fixture
def killing_toolchain(monkeypatch, tmp_path):
    """A stage environment whose every command is killed by SIGKILL."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".litetune-ready").write_text(self.identity)
        return self.path

    def fake_run(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, 0, "transformers==5.5.0\n", "")
        return subprocess.CompletedProcess(args, -SIGKILL, "", "")

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake_run)


def test_a_killed_export_is_could_not_check_not_a_failed_recipe(killing_toolchain, tmp_path):
    request = ExportRequest(
        model="google/functiongemma-270m-it",
        output_dir=tmp_path / "out",
        recipes=("dynamic_wi8_afp32",),
    )
    export = export_recipe(request, "dynamic_wi8_afp32")
    assert export.check.outcome is Outcome.UNCHECKED
    assert not export.ok
    assert "SIGKILL" in export.check.detail
    assert export.check.observed["exit"]["killed_by_signal"] == SIGKILL


def test_a_killed_sweep_does_not_report_a_verdict(killing_toolchain, tmp_path):
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


def test_a_killed_training_run_is_could_not_check(killing_toolchain, tmp_path):
    from litetune.evaluate import PromptMode
    from litetune.tune import TuneRequest

    data = tmp_path / "train.jsonl"
    data.write_text('{"prompt": "a", "completion": "call:a{}"}\n', encoding="utf-8")
    result = run_tune(
        TuneRequest(
            model="google/functiongemma-270m-it",
            data=data,
            output_dir=tmp_path / "run",
            prompt_mode=PromptMode.PRERENDERED,
        )
    )
    training = next(c for c in result.checks.checks if c.name == TRAINING_CHECK)
    assert training.outcome is Outcome.UNCHECKED
    assert "SIGKILL" in training.detail
    assert result.outcome is Outcome.UNCHECKED


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def test_a_killed_generation_is_not_performed(monkeypatch, tmp_path):
    def fake_run(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, -SIGKILL, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    generation = LiteRtLmBackend(model=tmp_path / "model.litertlm", auto_provision=False).generate(
        ["hello"]
    )[0]
    # `ran` is False, so the liveness tier reports could_not_check rather than
    # scoring an empty generation as a failure.
    assert not generation.ran
    assert generation.harness_error is not None
    assert "SIGKILL" in generation.harness_error
