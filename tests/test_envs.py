"""Environment definitions must not be able to float.

An unchanged Dockerfile produced a working export on 2026-08-26 and
`AttributeError: pad_token` on 2026-08-30, because the requirement was
unpinned. The constructor refuses that shape so the failure cannot recur
silently.
"""

import contextlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import types

import pytest

from litetune import envs
from litetune.envs import EXPORT, RUNTIME, TRAIN, StageEnv, UnpinnedRequirement


def _fake_venv(self, path):
    """Stand in for `EnvBuilder.create` -- including the interpreter.

    A bare `mkdir` was enough while `ready` read only the marker file. It is
    not a virtualenv, and a fake that is missing the one thing `ready` now
    looks for would assert the opposite of the invariant under test.

    Built through `envs._interpreter_in` rather than by spelling the filename
    here: writing an extensionless `python` on Windows is what would hide the
    very asymmetry that made `ready` unsatisfiable there.
    """
    interpreter = envs._interpreter_in(pathlib.Path(path))
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()


def test_unpinned_requirement_is_refused():
    with pytest.raises(UnpinnedRequirement) as e:
        StageEnv(name="bad", requirements=("litert-lm",))
    assert "litert-lm" in str(e.value)


def test_lower_bound_is_not_a_pin():
    # '>=' admits tomorrow's release, which is exactly the failure mode.
    with pytest.raises(UnpinnedRequirement):
        StageEnv(name="bad", requirements=("torch>=2.0",))


def test_shipped_environments_are_pinned():
    for env in (RUNTIME, EXPORT, TRAIN):
        assert env.identity, env.name


def test_identity_changes_with_requirements():
    a = StageEnv(name="x", requirements=("litert-lm==0.16.1",))
    b = StageEnv(name="x", requirements=("litert-lm==0.16.2",))
    assert a.identity != b.identity
    assert a.path != b.path


def test_train_and_runtime_are_separate_environments():
    # They conflict irreconcilably; sharing a path would defeat the split.
    assert TRAIN.path != RUNTIME.path


def test_provisioning_is_serialised_without_fcntl(tmp_path, monkeypatch, caplog):
    """`fcntl` does not exist on Windows, and this module has `os.name == "nt"`
    branches, so that platform is in scope.

    An unconditional `import fcntl` at module scope made `import litetune.envs`
    -- and therefore every CLI command -- fail there. Where no lock can be taken
    the run proceeds unserialised, which is what it did before the lock existed,
    but it says so: "the environments raced" must not be indistinguishable from
    a corrupt install.
    """
    import builtins

    real_import = builtins.__import__

    def without_locking(name, *args, **kwargs):
        if name in {"fcntl", "msvcrt"}:
            raise ImportError(f"no {name} on this platform")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_locking)

    with caplog.at_level("WARNING"):
        with envs._provision_lock(tmp_path / "env"):
            pass

    assert "unserialised" in caplog.text


def test_provisioning_takes_a_lock_where_it_can(tmp_path):
    """And the lock file is a sibling, not a sentinel inside the environment.

    A sentinel left behind by a killed process blocks every later run; a file
    lock is released by the kernel when the holder dies.
    """
    target = tmp_path / "env"

    with envs._provision_lock(target):
        assert (tmp_path / "env.lock").exists()
        assert not target.exists()


def test_waiting_for_another_provisioner_is_bounded(tmp_path):
    """`LOCK_EX` blocks forever; the advertised timeout only covered pip.

    A holder that hangs must not hang everyone, so the wait has the same
    deadline as the install it is waiting for.
    """
    import time

    target = tmp_path / "env"
    (tmp_path / "env.lock").touch()
    with open(tmp_path / "env.lock", "w") as holder:
        if envs._try_lock(holder) is not True:
            pytest.skip("no advisory locking on this filesystem")
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="still holding the lock"):
            with envs._provision_lock(target, timeout=0.6):
                pass
        assert time.monotonic() - started < 5


def test_the_install_does_not_start_once_the_budget_is_gone(tmp_path, monkeypatch):
    """`max(1.0, remaining)` started pip with a second after the budget expired.

    The virtualenv build is the unbounded step -- it spawns `ensurepip` and
    takes no timeout -- so the budget has to be re-read after it, not only
    before.
    """
    import subprocess
    import venv

    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    # A supported interpreter, so the budget is what this test is about: the
    # version guard runs first and would otherwise be the thing that fires.
    monkeypatch.setattr(envs.sys, "version_info", (3, 12, 0, "final", 0))
    env = StageEnv(name="slow", requirements=("pyyaml==6.0.2",))

    def slow_create(self, path):
        import time as _time

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        _time.sleep(0.4)

    called = []
    monkeypatch.setattr(venv.EnvBuilder, "create", slow_create)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: called.append(a) or None)

    # A working environment must survive a provision that refuses to run.
    ready_marker = env.path / ".litetune-ready"
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    ready_marker.write_text(env.identity, encoding="utf-8")
    (env.path / "payload").write_text("the existing environment", encoding="utf-8")

    with pytest.raises(RuntimeError, match="budget"):
        env.provision(timeout=0.3, force=True)

    assert called == [], "pip must not start once the budget is gone"
    assert (env.path / "payload").read_text(encoding="utf-8") == "the existing environment"


def test_an_unsupported_interpreter_is_refused_before_pip_runs(monkeypatch, tmp_path):
    """The stage environments inherit the host's Python, and the pins have a ceiling.

    `numpy==2.0.2` publishes wheels for cp39-cp312 only. Past that, pip falls
    back to a source build and fails after minutes of compiler output that never
    names the version. litetune itself runs fine on 3.13+; only what it
    provisions does not, and that distinction is invisible from the traceback.

    The ceiling belongs to the environment, not to the package: `train` pins no
    numpy and stops a version later. A shared constant refused `train` on 3.13
    and blamed numpy for it, which sends the reader to fix a pin that is not
    there. So the message must name the pin that actually set the limit.
    """
    import subprocess
    import venv

    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(
        name="probe",
        requirements=("pyyaml==6.0.2",),
        python_ceiling=(3, 12),
        ceiling_pin="numpy==2.0.2",
    )

    started = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: started.append(a))
    monkeypatch.setattr(envs.sys, "version_info", (3, 13, 0, "final", 0))

    with pytest.raises(RuntimeError, match=r"pins numpy==2\.0\.2.*up to python3\.12"):
        env.provision()
    assert started == [], "the refusal must come before pip"

    # An environment whose pins reach further is not caught by another's limit.
    reaches_further = StageEnv(
        name="probe2",
        requirements=("torch==2.5.1",),
        python_ceiling=(3, 13),
        ceiling_pin="torch==2.5.1",
    )
    monkeypatch.setattr(venv.EnvBuilder, "create", _fake_venv)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    reaches_further.provision()
    assert reaches_further.ready, "3.13 is inside torch's range and must be allowed"

    assert started == [], "the refusal must come before pip, not after it"


def test_a_provisioned_environment_can_run_its_console_scripts(monkeypatch, tmp_path):
    """A virtualenv is not relocatable, and every check we had was blind to it.

    `venv` writes the absolute path of the interpreter *at creation time* into
    the shebang of every console script, and nothing rewrites them on a rename.
    An environment built in `.incoming.<pid>` and renamed into place therefore
    had a working `pip` file whose interpreter did not exist -- and because the
    kernel truncates a shebang at 127 bytes, the error named a path that had
    never existed at all. `convert` and `verify` invoke `litert-torch` and
    `litert-lm` exactly this way, so both failed on every fresh install.

    Nothing caught it because every other test in this file replaces
    `EnvBuilder.create` with `mkdir`, so no real virtualenv is ever built. This
    one builds one. It stubs only the pip install -- the defect is in *where the
    environment is created*, not in what is installed into it -- which keeps the
    test offline and under two seconds while still exercising the real builder.
    """
    import subprocess as real_subprocess

    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name="relocatable", requirements=("pyyaml==6.0.2",))

    installs = []
    real_run = real_subprocess.run

    def skip_only_our_install(cmd, **kwargs):
        # `EnvBuilder.create` shells out to `ensurepip` through this same
        # function, and that call is the one installing the script under test.
        # Stubbing indiscriminately removes the evidence.
        if isinstance(cmd, list | tuple) and "install" in cmd:
            installs.append(list(cmd))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(envs.subprocess, "run", skip_only_our_install)
    path = env.provision(timeout=300)
    monkeypatch.undo()

    assert installs, "the install must still have been attempted"

    bindir = path / ("Scripts" if os.name == "nt" else "bin")
    script = bindir / ("pip.exe" if os.name == "nt" else "pip")
    assert script.exists(), "ensurepip installs a pip console script"

    if os.name != "nt":
        shebang = script.read_text(encoding="utf-8").splitlines()[0]
        assert shebang.startswith("#!"), shebang
        interpreter = pathlib.Path(shebang[2:].strip().split()[0])
        assert interpreter.exists(), (
            f"the console script points at {interpreter}, which does not exist -- "
            "the environment was built somewhere else and moved here"
        )
        assert (
            path in interpreter.parents
        ), f"{interpreter} lives outside {path}; a rename would strand it"

    # The assertion that matters: it runs.
    done = real_subprocess.run(
        [str(script), "--version"], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert "pip" in done.stdout


def test_the_cache_is_inventoried_with_sizes_and_readiness(monkeypatch, tmp_path):
    """There is no other way to find out what the stages left on disk.

    The path is in no documentation, the sizes differ by an order of magnitude
    between stages, and a provision that died halfway leaves a directory that
    looks exactly like a working one from the outside.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))

    big = tmp_path / "export-deadbeef"
    (big / "lib").mkdir(parents=True)
    (big / "lib" / "payload").write_bytes(b"x" * 5000)
    (big / "bin").mkdir()
    (big / "bin" / "python").touch()
    (big / ".litetune-ready").write_text("deadbeef", encoding="utf-8")

    half = tmp_path / "runtime-cafe"
    half.mkdir()
    (half / "payload").write_bytes(b"x" * 10)

    # The marker over nothing. This shape was found in a real cache and the
    # listing called it `ready`, which is the one word that would stop a reader
    # from suspecting it -- and `provision` short-circuits on the same test.
    hollow = tmp_path / "train-hollow"
    hollow.mkdir()
    (hollow / ".litetune-ready").write_text("hollow", encoding="utf-8")

    entries = envs.cached_environments()

    # Largest first: the reason to look is usually to find what to delete.
    assert [e.path.name for e in entries] == ["export-deadbeef", "runtime-cafe", "train-hollow"]
    assert entries[0].bytes > 5000 and entries[0].ready
    assert entries[1].bytes == 10 and not entries[1].ready
    assert not entries[2].ready, "a marker over an empty directory is not a ready environment"


def test_an_environment_for_another_interpreter_is_not_reported_as_junk():
    """`identity` hashes the running Python along with the pins.

    So an environment built by 3.12 looks foreign from 3.14 -- and it is not
    junk, it is the one that works over there. Reporting it as "unused" would
    invite deleting exactly the caches worth keeping.
    """
    from litetune.envs import EXPORT, RUNTIME, TRAIN

    identities = {env.path.name for env in (RUNTIME, TRAIN, EXPORT)}
    assert len(identities) == 3, "each stage owns a distinct directory"
    for env in (RUNTIME, TRAIN, EXPORT):
        assert env.identity in env.path.name
        # The interpreter is in the hash, which is why the CLI says "not for
        # pythonX.Y" rather than "unclaimed".
        assert env.identity != _identity_ignoring_interpreter(env)


def _identity_ignoring_interpreter(env) -> str:
    import hashlib

    return hashlib.sha256("\n".join(sorted(env.requirements)).encode()).hexdigest()[:12]


def test_removing_the_cache_reports_what_would_not_go(monkeypatch, tmp_path):
    """Four of five removed, and which one resisted, beats stopping at the first."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    for name in ("export-aaa", "runtime-bbb"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "f").write_bytes(b"x" * 100)

    entries = envs.cached_environments()
    freed, failures = envs.remove_cached(entries)

    assert freed == 200
    assert failures == []
    assert envs.cached_environments() == []


# ---------------------------------------------------------------------------
# The device probe
# ---------------------------------------------------------------------------
#
# `resolve_device` had no direct test at all: everything about it was reached
# through `run_tune`, where the fake could not answer it, so every one of its
# branches ran only in the "could not answer" direction. Each of them is a
# different fact about a run and the report has to carry which.


@pytest.fixture
def probe_env(monkeypatch, tmp_path) -> StageEnv:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    return StageEnv(name="probe", requirements=("torch==2.5.1",))


def _answers(monkeypatch, *, stdout="", returncode=0, stderr="", raises=None):
    def fake_run(self, args, timeout=3600, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    monkeypatch.setattr(StageEnv, "run", fake_run)


def _json_answer(device="cpu", cuda_build=None, device_count=0) -> str:
    import json

    return json.dumps({"device": device, "cuda_build": cuda_build, "device_count": device_count})


def test_the_probe_asks_this_environments_own_torch(probe_env, monkeypatch):
    """The question has to be asked inside the environment that will run the
    job. Any other interpreter answers about a machine this run is not using.
    """
    seen: list = []

    def fake_run(self, args, timeout=3600, **kwargs):
        seen.append((self.name, list(args), timeout))
        return subprocess.CompletedProcess(args, 0, _json_answer("cuda"), "")

    monkeypatch.setattr(StageEnv, "run", fake_run)

    assert envs.resolve_device(probe_env).device == "cuda"
    name, argv, timeout = seen[0]
    assert name == "probe"
    assert argv[:2] == ["python", "-c"]
    assert "cuda.is_available" in argv[2]
    assert timeout == envs.DEVICE_PROBE_TIMEOUT_S


def test_a_cpu_answer_is_a_cpu_answer(probe_env, monkeypatch):
    _answers(monkeypatch, stdout=_json_answer("cpu"))
    probe = envs.resolve_device(probe_env)
    assert probe.device == "cpu"
    assert probe.answered


def test_a_banner_before_the_answer_does_not_destroy_it(probe_env, monkeypatch):
    """Any line a stage environment prints on startup lands on stdout ahead of
    the answer. The answer is the last thing written, and reading all of stdout
    turned a perfectly good "cuda" into "could not answer"."""
    _answers(
        monkeypatch,
        stdout=f"WARNING: pip is out of date\n\n{_json_answer('cuda', '12.4', 1)}\n",
    )
    assert envs.resolve_device(probe_env).device == "cuda"


def test_an_answer_that_is_neither_device_is_not_a_device(probe_env, monkeypatch):
    _answers(monkeypatch, stdout=_json_answer("mps"))
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "'mps'" in probe.detail


def test_unparseable_stdout_is_not_a_device(probe_env, monkeypatch):
    _answers(monkeypatch, stdout="cuda\n")
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "not an answer" in probe.detail


def test_silence_is_not_a_device(probe_env, monkeypatch):
    _answers(monkeypatch, stdout="   \n\n")
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "printed nothing" in probe.detail


def test_a_non_zero_exit_is_not_a_device(probe_env, monkeypatch):
    _answers(monkeypatch, returncode=1, stderr="ModuleNotFoundError: No module named 'torch'")
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "exited 1" in probe.detail
    assert "No module named" in probe.detail


def test_a_killed_probe_is_read_as_a_signal_not_a_status(probe_env, monkeypatch):
    """`-9` is not an exit status the probe chose. See `litetune.exits`; the
    same reading that turned a memory ceiling into a verdict about a model."""
    _answers(monkeypatch, returncode=-9)
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "SIGKILL" in probe.detail
    assert "exited -9" not in probe.detail


def test_a_timeout_is_not_a_device(probe_env, monkeypatch):
    _answers(monkeypatch, raises=subprocess.TimeoutExpired(cmd="python", timeout=30))
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "TimeoutExpired" in probe.detail


def test_a_probe_that_cannot_start_is_not_a_device(probe_env, monkeypatch):
    _answers(monkeypatch, raises=FileNotFoundError("python"))
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert "FileNotFoundError" in probe.detail


def test_an_invalid_byte_ahead_of_the_answer_does_not_crash_the_run(probe_env):
    """A garbled byte on the probe's stdout used to end the whole run.

    `StageEnv.run` decodes with `text=True`, which reads
    stdout eagerly, and a stray non-UTF-8 byte -- a CUDA, driver or vendor
    banner ahead of the probe's own line, the same class of weird environment
    the last-non-empty-line rule exists to survive -- raised
    `UnicodeDecodeError`. That is a `ValueError`, not the `OSError` this
    function used to catch, so it escaped `resolve_device` entirely and ended
    a `tune` or `verify` run with a traceback instead of an unanswered probe.

    No mock of `StageEnv.run`: the fake "python" below is a real executable at
    the real path `run` looks for, so this exercises the actual
    `errors="replace"` decode the fix lives in, not a
    stand-in that hands back an already-decoded string and could not have
    caught the regression.
    """
    probe_env.path.mkdir(parents=True, exist_ok=True)
    bindir = probe_env.python.parent
    bindir.mkdir(parents=True, exist_ok=True)
    fake_python = bindir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stdout.buffer.write(b'\\xff\\n')\n"
        'sys.stdout.buffer.write(b\'{"device": "cuda", "cuda_build": null, '
        '"device_count": 0}\\n\')\n'
    )
    fake_python.chmod(0o755)

    probe = envs.resolve_device(probe_env, timeout=5)

    assert probe.device == "cuda"
    assert probe.answered


def test_a_decode_error_out_of_env_run_does_not_escape_resolve_device(probe_env, monkeypatch):
    """The other half of the same fix, pinned independently of `errors="replace"`.

    `env.run` no longer raises `UnicodeDecodeError` for this, but
    `resolve_device`'s own contract -- it never raises -- must not depend on
    every caller of `env.run` getting that right, so it catches `ValueError`
    too. Reproduces the exact shape from the field: `UnicodeDecodeError` out
    of `StageEnv.run`, uncaught by `except (TimeoutExpired, OSError)` because
    it is a `ValueError`, not an `OSError`.
    """
    _answers(
        monkeypatch,
        raises=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    )
    probe = envs.resolve_device(probe_env)
    assert probe.device is None
    assert not probe.answered
    assert "UnicodeDecodeError" in probe.detail


def test_a_cuda_build_that_sees_no_device_says_so(probe_env, monkeypatch):
    """`is_available()` answers False for a CPU-only wheel, a container started
    without `--gpus`, and a driver too old for the runtime. The first is "this
    machine has no GPU"; the others are "torch cannot reach the one it has".
    Both used to become a confident, identical "cpu".
    """
    _answers(monkeypatch, stdout=_json_answer("cpu", cuda_build="12.4", device_count=0))
    probe = envs.resolve_device(probe_env)
    assert probe.device == "cpu"
    assert probe.cuda_build_without_a_device
    assert "CUDA 12.4 build" in probe.detail
    assert "0 devices" in probe.detail


def test_a_cpu_only_wheel_reporting_cpu_is_an_ordinary_cpu_machine(probe_env, monkeypatch):
    _answers(monkeypatch, stdout=_json_answer("cpu", cuda_build=None))
    probe = envs.resolve_device(probe_env)
    assert probe.device == "cpu"
    assert not probe.cuda_build_without_a_device


def test_an_unanswered_probe_reaches_the_event_stream(probe_env, monkeypatch):
    """`logging` alone reaches nothing a report can carry."""
    from litetune.events import EventStream

    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)
    _answers(monkeypatch, returncode=1, stderr="boom")

    envs.resolve_device(probe_env, events=events)

    assert [e for e in seen if e.kind == "note" and "could not answer" in e.data["message"]]


def test_the_probe_source_reports_the_build_and_the_count(monkeypatch, capsys):
    """`_DEVICE_PROBE_CODE` is a string of source that runs in another
    interpreter, so the only way to pin what it prints is to run it here
    against a fake torch. `torch.version.cuda` and `device_count()` are the
    two facts that separate "this machine has no GPU" from "this torch cannot
    reach the one it has" -- dropping either leaves `is_available()` alone,
    which answers False for both.
    """
    import json
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda="12.4")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", fake)

    # `exec` on this package's own constant, not on external input.
    exec(compile(envs._DEVICE_PROBE_CODE, "device_probe.py", "exec"), {})

    assert json.loads(capsys.readouterr().out.strip()) == {
        "device": "cpu",
        "cuda_build": "12.4",
        "device_count": 0,
    }


def test_the_probe_source_says_cuda_when_torch_can_reach_one(monkeypatch, capsys):
    import json
    import sys
    import types

    fake = types.ModuleType("torch")
    fake.version = types.SimpleNamespace(cuda="12.4")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    monkeypatch.setitem(sys.modules, "torch", fake)

    exec(compile(envs._DEVICE_PROBE_CODE, "device_probe.py", "exec"), {})

    assert json.loads(capsys.readouterr().out.strip()) == {
        "device": "cuda",
        "cuda_build": "12.4",
        "device_count": 2,
    }


def test_no_probe_at_all_is_a_distinct_state():
    """ "Nobody asked" and "it was asked and could not say" are different facts
    about a run, and `None` alone cannot tell them apart."""
    assert envs.NOT_PROBED.device is None
    assert not envs.NOT_PROBED.answered
    assert "no device probe was run" in envs.NOT_PROBED.detail


# ---------------------------------------------------------------------------
# What the host can reach into a stage with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(envs._HOST_OVERRIDES))
def test_the_host_cannot_reach_past_the_pins(name, monkeypatch):
    """A pinned environment that imports something else is the ledger's case.

    `PYTHONPATH` is prepended ahead of the venv's own `site-packages`, so a
    directory holding `numpy.py` outranks `numpy==2.0.2`. The run does not
    crash; it records a measurement under a provenance it did not have, and the
    cache then replays it under the pinned identity.
    """
    monkeypatch.setenv(name, "/somewhere/of/the/hosts/own")
    assert name not in envs._child_env()


def _symlinked_env(tmp_path, monkeypatch, name: str) -> StageEnv:
    """A stage environment whose interpreter is the one running the tests.

    Enough for `run` to start a real process without provisioning anything,
    which is what the tests below are about: they ask the child what it got.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name=name, requirements=("pyyaml==6.0.2",))
    env.python.parent.mkdir(parents=True, exist_ok=True)
    env.python.symlink_to(sys.executable)
    return env


def test_the_child_process_gets_the_host_environment_minus_the_reach(tmp_path, monkeypatch):
    """The helper being right is not the claim; the subprocess is.

    Every other test here calls `_child_env` directly, so all of them would
    still pass if `run` stopped calling it. This one starts a real process and
    reads its whole environment back.

    Asserting the absences alone is not enough either: `env=dict(env or {})`
    passes a test that only checks `PYTHONPATH` is gone, and it launches every
    stage with no `PATH`, no `HOME` and no Hugging Face cache. Sanitation is
    the claim, not isolation, so what survives is asserted too.
    """
    monkeypatch.setenv("PYTHONPATH", "/planted")
    monkeypatch.setenv("LITETUNE_CANARY", "kept")
    env = _symlinked_env(tmp_path, monkeypatch, "realchild")

    proc = env.run(
        ["python", "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
        timeout=60,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    seen = json.loads(proc.stdout.strip().splitlines()[-1])

    assert "PYTHONPATH" not in seen
    assert seen["CUDA_VISIBLE_DEVICES"] == "", "a caller's override still reaches the child"
    assert seen["LITETUNE_CANARY"] == "kept", "the rest of the host environment is not isolation"
    assert seen["PATH"] == os.environ["PATH"], "the base is the host's, not an empty dict"


def test_the_child_gets_a_session_of_its_own(tmp_path, monkeypatch):
    """`_kill_tree` SIGKILLs the child's process group.

    If the child ever shared ours, that group is the test runner's -- so a
    regression here would not fail a test, it would kill pytest with signal 9,
    which this project's own `read_returncode` then reads as the OOM killer.
    `_kill_tree` refuses to signal its own group for that reason; this pins the
    other half, that the child is genuinely somewhere else.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "ownsession")
    proc = env.run(["python", "-c", "import os; print(os.getsid(0))"], timeout=30)
    assert int(proc.stdout.strip()) != os.getsid(0)


def test_a_signal_still_reads_as_a_signal_through_the_new_run(tmp_path, monkeypatch):
    """`Popen` replaced `subprocess.run`, and the whole check model rides on this.

    `exits.read_returncode` tells "killed" from "exited non-zero" by the sign of
    the return code, and that is what keeps a SIGKILLed stage `unchecked`
    rather than `failed`. A rewrite of `run` that returned an unsigned status
    would collapse the two silently.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "signalled")
    proc = env.run(
        ["python", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"], timeout=60
    )
    assert proc.returncode == -9
    assert not envs.read_returncode(proc.returncode).conclusive


def test_a_timeout_carries_what_the_child_managed_to_say(tmp_path, monkeypatch):
    """And pins the order: kill first, then drain.

    Draining before the kill is the shape that hangs, and it is an easy thing
    to reorder while tidying. The output is the only artifact that would
    explain a six-hour non-result, so it is worth having in hand even though
    no caller reads it today.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "talkative")
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        env.run(
            ["python", "-c", "import time; print('partial', flush=True); time.sleep(30)"],
            timeout=1,
        )
    assert "partial" in (caught.value.output or "")
    assert caught.value.timeout == 1


class _FakeProc:
    """A `Popen` stand-in that records what was signalled, and never dies.

    `poll` answers from `returncode` the way the real one does, because
    `_kill_tree` and `_signal_tree` both ask it: a host that ignores SIGCHLD
    has the kernel reap for us, and `Popen` only learns through `poll`.
    """

    def __init__(self, pid=4242, returncode=None):
        self.pid = pid
        self.returncode = returncode
        self.direct_signals: list[int] = []

    def send_signal(self, sig):
        self.direct_signals.append(sig)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("fake", timeout)


def test_a_stage_is_asked_to_stop_before_it_is_killed(monkeypatch):
    """SIGKILL alone would be a quiet downgrade of what this replaced.

    Ctrl-C used to reach the child as SIGINT through the shared terminal group,
    so torch ran its own handlers -- for `tune` that is a checkpoint written
    rather than a file truncated mid-save. The grace keeps that; SIGKILL still
    follows, so a wedged process cannot outlast its own timeout.
    """
    sent = []
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    # Signal 0 is the liveness probe the grace uses, not a kill; recording it
    # would make this test about the wait rather than about the order.
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: sig and sent.append(sig))

    assert envs._kill_tree(_FakeProc(), grace=0.01) is envs.Reach.GROUP
    assert sent == [signal.SIGTERM, signal.SIGKILL], "polite first, then certain"


def test_a_group_that_cannot_be_killed_falls_back_to_the_child(monkeypatch):
    """`killpg` returns EPERM in some sandboxes and containers.

    That is exactly where an orphaned trainer would come back, so the fallback
    is not decoration. It never runs in this suite otherwise.
    """

    def refuse(*_args):
        raise PermissionError("operation not permitted")

    # A real pid is not needed and would be a different test; what matters is
    # that the group exists, is not ours, and refuses the signal.
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", refuse)

    proc = _FakeProc()
    assert envs._kill_tree(proc, grace=0.01) is envs.Reach.CHILD_ONLY
    assert proc.direct_signals == [signal.SIGTERM, signal.SIGKILL]


def test_a_process_that_already_left_counts_as_killed(monkeypatch):
    """Finishing between the timeout firing and the kill is the ordinary case."""

    def gone(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4243 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", gone)
    assert envs._kill_tree(_FakeProc(pid=4243), grace=0.01) is envs.Reach.ALREADY_GONE


def test_a_reaped_process_is_never_signalled(monkeypatch):
    """Its pid is free for the kernel to hand to somebody else.

    `killpg` on a recycled pid would take out a stranger's process group, and
    the interrupt path can reach this state: a KeyboardInterrupt landing inside
    `communicate`'s trailing `wait()` arrives after the child was reaped.
    """
    signalled = []
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: signalled.append(sig))

    proc = _FakeProc(returncode=0)
    assert envs._kill_tree(proc) is envs.Reach.ALREADY_GONE
    assert signalled == [] and proc.direct_signals == []


def test_the_stage_refuses_to_kill_its_own_process_group(monkeypatch):
    """Otherwise a caller reaching past `start_new_session` takes the shell out.

    `os.getpgid(proc.pid)` returning our own group can only happen if someone
    reaches past `start_new_session=True`, which nothing can now that the
    `**kwargs` passthrough is gone -- but the cost of the invariant being wrong
    once is the user's terminal.
    """
    signalled = []
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: signalled.append(pgid))
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 999)

    proc = _FakeProc(pid=4244)
    envs._kill_tree(proc, grace=0.01)

    assert signalled == [], "our own process group must never be signalled"
    assert proc.direct_signals == [signal.SIGTERM, signal.SIGKILL]


def test_the_rest_of_the_host_environment_survives(monkeypatch):
    """Sanitation, not isolation: a stage still needs HOME, PATH and the rest."""
    monkeypatch.setenv("LITETUNE_CANARY", "kept")
    assert envs._child_env()["LITETUNE_CANARY"] == "kept"


def test_a_caller_override_is_applied_after_the_sanitation(monkeypatch):
    """`export` passes `CUDA_VISIBLE_DEVICES=""` and must still win."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert envs._child_env({"CUDA_VISIBLE_DEVICES": ""})["CUDA_VISIBLE_DEVICES"] == ""


def test_the_cpu_only_override_is_overrides_and_not_an_environment(monkeypatch):
    """It returned `dict(os.environ) | ...`, which handed PYTHONPATH back.

    The base is `StageEnv.run`'s to build. A caller that starts from
    `os.environ` re-adds exactly what the sanitation just removed, and this is
    the one caller that passes `env=` at all.
    """
    from litetune.export import _cpu_only_environ

    monkeypatch.setenv("PYTHONPATH", "/host")
    assert "PYTHONPATH" not in _cpu_only_environ()


def test_the_install_does_not_inherit_a_redirected_pip(tmp_path, monkeypatch):
    """`PIP_TARGET` installs elsewhere while `.litetune-ready` is written here.

    The result is an environment that reports ready and holds nothing, which is
    the same false pass from one step earlier in the pipeline.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    monkeypatch.setenv("PIP_TARGET", "/somewhere/else")
    monkeypatch.setattr(envs.venv.EnvBuilder, "create", _fake_venv)

    seen = {}

    def capture(cmd, **kwargs):
        seen.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(envs.subprocess, "run", capture)
    StageEnv(name="pipenv", requirements=("pyyaml==6.0.2",)).provision()
    assert "PIP_TARGET" not in seen


# ---------------------------------------------------------------------------
# What "ready" is evidence of
# ---------------------------------------------------------------------------


def test_a_marker_over_an_empty_directory_is_not_ready(tmp_path, monkeypatch):
    """Found in a real cache: 12 bytes, the marker alone, no `bin/`.

    `provision` short-circuits on `ready` and builds nothing, and `run` then
    reaches an interpreter that does not exist and reports "it could not be
    started" -- which sends the reader to look at their toolchain instead of at
    an empty directory.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name="hollow", requirements=("pyyaml==6.0.2",))
    env.path.mkdir(parents=True)
    (env.path / ".litetune-ready").write_text("whatever", encoding="utf-8")

    assert not env.ready


def test_an_interpreter_with_no_marker_is_not_ready_either(tmp_path, monkeypatch):
    """The marker still means "the install finished"; it did not lose its job."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name="half", requirements=("pyyaml==6.0.2",))
    env.python.parent.mkdir(parents=True)
    env.python.touch()

    assert not env.ready


# ---------------------------------------------------------------------------
# What a timeout takes with it
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_a_timeout_kills_what_the_stage_spawned(tmp_path, monkeypatch, request):
    """`subprocess.run(timeout=)` kills the direct child and nothing below it.

    `tune` waits six hours; when that fires it records an honest "not checked"
    and the run *continues*, so an abandoned torch process keeps its memory.
    The next stage is SIGKILLed and `exits` reads that as the OOM killer --
    blaming the machine for litetune's own orphan.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "grandchild")

    # Liveness is a held lock, not a pid. `kill(pid, 0)` succeeds against a
    # zombie, and after the group dies the grandchild is reparented to init and
    # reaped whenever init gets round to it -- which, in a container whose PID 1
    # is a shell that never reaps, is never. A lock is released by the kernel
    # when the process dies, zombie or not.
    lockfile = tmp_path / "grandchild.lock"
    pidfile = tmp_path / "grandchild.pid"
    held = tmp_path / "grandchild.held"
    grandchild_code = (
        # Deaf to the polite signal on purpose: a grandchild that dies on the
        # SIGTERM would let this pass without the SIGKILL sweep ever running,
        # and the sweep is the thing the docstring claims.
        "import fcntl, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        f"h = open({str(lockfile)!r}, 'w'); fcntl.flock(h, fcntl.LOCK_EX);"
        # Announced, because a lock that was never taken is indistinguishable
        # from one that was released -- and the poll below would then report
        # success having observed nothing at all.
        f"open({str(held)!r}, 'w').write('1'); time.sleep(30)"
    )
    child = (
        "import subprocess, sys, time;"
        f"p = subprocess.Popen([sys.executable, '-c', {grandchild_code!r}]);"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid));"
        "time.sleep(30)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        env.run(["python", "-c", child], timeout=3)

    grandchild = int(pidfile.read_text())
    # Registered before the assertion below: a failure there must not leak the
    # process for the next half minute.
    request.addfinalizer(lambda: _reap(grandchild))
    assert held.exists(), "the grandchild never took the lock; this test proved nothing"

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with open(lockfile, "w") as handle:
            taken = envs._try_lock(handle)
        if taken is None:
            pytest.skip("no advisory locking on this filesystem; liveness cannot be observed")
        if taken is True:
            return
        time.sleep(0.05)
    pytest.fail(f"grandchild {grandchild} still holds its lock 10s after the group was killed")


def _reap(pid: int) -> None:
    """Best effort, for a pid we never had a `Popen` for -- a grandchild."""
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGKILL)


def _reap_proc(proc: subprocess.Popen) -> None:
    """The same, through the object, for a process we started ourselves.

    `os.kill` on the bare pid is the recycling hazard the production code goes
    to lengths to avoid, and these finalisers run *after* `wait()` has reaped
    the process -- so the number may already belong to somebody else. `Popen`
    knows it was reaped and does nothing.
    """
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# What the termination guard does and does not take over
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX")
def test_a_signal_the_process_already_ignores_is_left_alone(monkeypatch):
    """`nohup` sets SIGHUP to SIG_IGN for one purpose: surviving the hangup.

    Installing over it turns the documented way to run a six-hour `tune` over
    SSH into the one way to lose it -- litetune would kill the trainer the
    moment the connection dropped, and then report it as a Ctrl-C nobody typed.
    """
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_IGN)
    installed = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.append(sig))

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc()):
        pass

    assert installed == [], "an inherited SIG_IGN is the caller's decision, not ours to override"


def test_the_guard_hands_back_the_signals_it_took(monkeypatch):
    """Including where the previous handler came from C and cannot be restored.

    `signal.signal` returns `None` for those, and leaving ours in place would
    outlive the call, closed over a process that no longer exists.
    """
    # Every call recorded in order, not just the first: `setdefault` kept only
    # the install and discarded the restore, so replacing the whole `finally`
    # body with `pass` left this green.
    calls: list[tuple[int, object]] = []

    # `None` is what `signal.signal` returns when the handler it displaced came
    # from C -- the case the docstring is entirely about, and one a `SIG_DFL`
    # stub never reaches.
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: calls.append((sig, handler)))

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc()):
        installed = dict(calls)
        assert installed, "the guard installed nothing at all"
        assert all(callable(h) for h in installed.values())

    handed_back = dict(calls[len(installed) :])
    assert set(handed_back) == set(installed), "every signal taken must be handed back"
    assert all(h is signal.SIG_DFL for h in handed_back.values()), (
        "a C handler cannot be restored from Python, so the default is what goes back -- "
        "leaving ours would outlive this call over a process that no longer exists"
    )


def test_a_second_signal_does_not_buy_the_child_more_time(monkeypatch):
    """Our handler stays installed while it runs, so a second signal re-enters.

    Measured before the guard: three SIGTERMs a second apart against a child
    that ignores them took 8.0s to give up instead of 5.0s, and deeper nesting
    compounds. "Stop harder" must not mean "wait longer".

    The second signal kills with no grace and returns, rather than raising
    through the first invocation. Raising was the earlier attempt: it left the
    first frame without its restore or its re-delivery, so the process died of
    an uncaught exception instead of the signal.
    """
    _TERM_GRACE_SENTINEL = object()
    entries = []

    def kill_and_signal_again(proc, grace=_TERM_GRACE_SENTINEL, stop=None, pgid=None):
        entries.append(grace)
        if len(entries) == 1:
            handler(signal.SIGTERM, None)  # a second signal, mid-grace
        return envs.Reach.GROUP

    monkeypatch.setattr(envs, "_kill_tree", kill_and_signal_again)
    captured = {}
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc()):
        handler = captured[signal.SIGTERM]
        # `os.kill` is a no-op here, so the re-delivery returns and the first
        # invocation reaches its "the host handler carried on" raise -- which is
        # the correct end for that path, and not what the second signal did.
        with pytest.raises(envs.StageInterrupted):
            handler(signal.SIGTERM, None)

    assert entries == [
        _TERM_GRACE_SENTINEL,
        0,
    ], "the second signal must kill outright rather than start another grace"


def test_an_interrupted_stage_is_not_reported_as_a_ctrl_c():
    """`KeyboardInterrupt` means one specific thing to every reader of a log.

    A run ended by a hangup or by `kill` is not one the operator abandoned, and
    CPython's own exit status for the two is the same 130 unless they are
    different types.
    """
    interrupted = envs.StageInterrupted(signal.SIGTERM)
    assert not isinstance(interrupted, KeyboardInterrupt)
    assert isinstance(interrupted, BaseException)
    assert "SIGTERM" in str(interrupted)
    assert interrupted.signum == signal.SIGTERM


def test_a_platform_without_sigkill_still_kills(monkeypatch):
    """Windows has no `SIGKILL`, and reading the name is enough to raise.

    That `AttributeError` fired inside the handler that was reporting a
    timeout, so the timeout was never reported at all -- and CI is Linux-only,
    so nothing in this suite could have caught it.
    """
    monkeypatch.setattr(envs, "_SIGKILL", None)
    monkeypatch.setattr(envs, "_group_of", lambda proc: None)

    class NoSignals(_FakeProc):
        def __init__(self):
            super().__init__()
            self.hard_killed = False

        def send_signal(self, sig):
            if sig is not signal.SIGTERM:
                raise ValueError(f"Unsupported signal: {sig}")
            super().send_signal(sig)

        def kill(self):
            self.hard_killed = True

    proc = NoSignals()
    assert envs._kill_tree(proc, grace=0.01) is envs.Reach.CHILD_ONLY
    assert proc.hard_killed, "the hard kill must go through Popen.kill where there is no SIGKILL"


def test_the_stage_is_given_its_grace(monkeypatch):
    """Deleting the wait entirely leaves SIGTERM immediately followed by
    SIGKILL, which is behaviourally the SIGKILL-only shape the grace exists to
    replace -- and every signal-order assertion here would still pass."""
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: None)
    waited = []
    monkeypatch.setattr(
        envs,
        "_wait_without_reaping",
        # Returns True: "the group is still ours to signal", which is what the
        # real one says unless the child was reaped out from under it.
        lambda proc, pgid, grace, stop=None: waited.append(grace) or True,
    )

    envs._kill_tree(_FakeProc(), grace=0.25)
    assert waited == [0.25], "the child must be given time between SIGTERM and SIGKILL"


def test_the_group_is_killed_even_when_the_child_went_quietly(monkeypatch):
    """Its own exit does not cover what it spawned.

    A trainer that obeys SIGTERM and leaves its DataLoader workers running is
    the case the unconditional second signal exists for.
    """
    sent = []
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: sent.append(sig))

    proc = _FakeProc()

    def exits_during_the_grace(target, pgid, grace, stop=None):
        # Exited, not reaped: `_wait_without_reaping` is named for what it does
        # not do, so `returncode` stays None and the group id stays pinned. A
        # stub that set it was modelling something the real one cannot do.
        return True

    monkeypatch.setattr(envs, "_wait_without_reaping", exits_during_the_grace)
    envs._kill_tree(proc, grace=0.01)
    assert sent == [signal.SIGTERM, signal.SIGKILL]


def test_a_stage_that_cannot_be_signalled_at_all_says_so(monkeypatch):
    """The difference between the two halves of the leaked-process warning.

    `NOTHING` is the state where the child itself is still there and could not
    be touched -- EPERM in a sandbox -- and the message must not then blame a
    grandchild that never left the group.
    """
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError))

    class Untouchable(_FakeProc):
        def send_signal(self, sig):
            raise PermissionError("operation not permitted")

    assert envs._kill_tree(Untouchable(), grace=0.01) is envs.Reach.NOTHING


def test_the_partial_output_of_a_wedged_stage_is_kept(monkeypatch):
    """The stderr of a timed-out stage is the one artifact that explains it.

    This is the path where a grandchild left the group and still holds the
    pipes: the drain gives up, and what the stage managed to say before that
    must survive rather than being replaced by two empty strings.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, grace=None, pgid=None: envs.Reach.GROUP)

    class Wedged(_FakeProc):
        args = ["python", "-c", "..."]
        stdout = None
        stderr = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("python", timeout, output=b"partial", stderr=b"boom")

    out, err = envs._kill_and_drain(Wedged())
    assert out == "partial" and err == "boom"


def test_bytes_from_a_text_mode_pipe_are_decoded():
    """`_check_timeout` attaches the raw chunks without decoding, even in text
    mode, so the decode has to happen on the way out."""
    assert envs._as_text(None) == ""
    assert envs._as_text("already text") == "already text"
    assert envs._as_text(b"a \xff byte") == "a � byte"


def test_every_variable_the_comment_argues_for_is_actually_dropped():
    """The parametrised test above generates its cases *from* the tuple, so
    deleting an entry removes a case rather than failing one.

    The injection pair is the one the module comment argues hardest about, and
    it is the one a later reader is most likely to take back out.
    """
    # Set equality, not membership: the parametrised test above generates its
    # cases *from* this tuple, so deleting an entry removed a case rather than
    # failing one -- measured, with `PYTHONHOME` gone and the suite green.
    # Both directions now need a deliberate edit here.
    assert set(envs._HOST_OVERRIDES) == {
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONEXECUTABLE",
        "PIP_TARGET",
        "PIP_PREFIX",
        "PIP_ROOT",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    }
    # And the ones the module comment argues are deliberately absent.
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PIP_INDEX_URL", "PIP_CONFIG_FILE"):
        assert name not in envs._HOST_OVERRIDES


def test_the_interpreter_is_where_venv_actually_puts_it(monkeypatch):
    """Anchored to literals on both platforms, because the test fake now builds
    through this same function -- so nothing else can catch it drifting."""
    # Built before the monkeypatch: `os.name` is one global, and `pathlib`
    # reads it to decide which flavour of path to construct.
    root = pathlib.Path("/x")
    posix = pathlib.Path("/x/bin/python")
    windows = pathlib.Path("/x/Scripts/python.exe")

    assert envs._interpreter_in(root) == posix
    assert envs._console_script_in(root, "pip") is None  # nothing there to find

    monkeypatch.setattr(envs.os, "name", "nt")
    assert envs._interpreter_in(root) == windows


def test_a_windows_console_script_is_found_by_its_exe(tmp_path, monkeypatch):
    """Entry points there are `.exe` wrappers. Looking only for the bare name
    finds nothing and silently takes the `python -m` path, which works for
    `pip` and not for `litert-torch`."""
    monkeypatch.setattr(envs.os, "name", "nt")
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "litert-torch.exe").touch()

    assert envs._console_script_in(tmp_path, "litert-torch") == scripts / "litert-torch.exe"


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM dispositions are POSIX")
def test_a_terminated_litetune_takes_the_stage_with_it(tmp_path, request):
    """`kill <litetune>` must not leave a six-hour torch run holding its memory.

    Every other test of the guard calls the context manager directly, so none
    of them notices if `run` stops entering it -- measured: replacing it with
    `nullcontext()` left the whole suite green. The only way to cover that is
    to deliver a real signal to a real litetune process, so this starts one of
    its own rather than signalling pytest.
    """
    lockfile = tmp_path / "stage.lock"
    held = tmp_path / "stage.held"
    stage_code = (
        f"import fcntl, time; h = open({str(lockfile)!r}, 'w'); fcntl.flock(h, fcntl.LOCK_EX);"
        f"open({str(held)!r}, 'w').write('1'); time.sleep(60)"
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(pathlib.Path(envs.__file__).parents[1])!r})\n"
        f"os.environ['LITETUNE_ENV_DIR'] = {str(tmp_path)!r}\n"
        "from litetune.envs import StageEnv\n"
        "env = StageEnv(name='terminated', requirements=('pyyaml==6.0.2',))\n"
        "env.python.parent.mkdir(parents=True, exist_ok=True)\n"
        "env.python.symlink_to(sys.executable)\n"
        f"env.run(['python', '-c', {stage_code!r}], timeout=120)\n",
        encoding="utf-8",
    )

    litetune = subprocess.Popen([sys.executable, str(runner)])
    request.addfinalizer(lambda: _reap_proc(litetune))

    # Signalling before the stage holds the lock would test nothing, and a
    # fixed sleep would either flake or be slow.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not held.exists():
        if litetune.poll() is not None:
            pytest.fail(f"the runner died before the stage started (exit {litetune.returncode})")
        time.sleep(0.05)
    assert held.exists(), "the stage never started; nothing was under test"

    os.kill(litetune.pid, signal.SIGTERM)
    assert litetune.wait(timeout=30) == -signal.SIGTERM, "litetune must still die as it would have"

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with open(lockfile, "a") as handle:
            taken = envs._try_lock(handle)
        if taken is None:
            pytest.skip("no advisory locking on this filesystem")
        if taken is True:
            return
        time.sleep(0.05)
    pytest.fail("the stage outlived the litetune process that was told to exit")


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery to a process group is POSIX")
def test_an_interrupted_litetune_takes_the_stage_with_it(tmp_path, request):
    """The `except BaseException` arm of `run`, which nothing else reaches.

    Measured: narrowing it to `except SystemExit` left the suite green. It is
    the replacement for the SIGINT that used to reach the child through the
    shared terminal, so a Ctrl-C during a six-hour `tune` not landing here
    means the whole torch tree keeps running.
    """
    lockfile = tmp_path / "stage.lock"
    held = tmp_path / "stage.held"
    stage_code = (
        f"import fcntl, time; h = open({str(lockfile)!r}, 'w'); fcntl.flock(h, fcntl.LOCK_EX);"
        f"open({str(held)!r}, 'w').write('1'); time.sleep(60)"
    )
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import os, signal, sys\n"
        f"sys.path.insert(0, {str(pathlib.Path(envs.__file__).parents[1])!r})\n"
        f"os.environ['LITETUNE_ENV_DIR'] = {str(tmp_path)!r}\n"
        # A backgrounded pytest hands its children SIGINT=SIG_IGN, and
        # CPython keeps an inherited SIG_IGN for SIGINT rather than
        # installing its own. Without this the signal is dropped, the test
        # times out, and a mutation run reads that as a killed mutant.
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        "from litetune.envs import StageEnv\n"
        "env = StageEnv(name='interrupted', requirements=('pyyaml==6.0.2',))\n"
        "env.python.parent.mkdir(parents=True, exist_ok=True)\n"
        "env.python.symlink_to(sys.executable)\n"
        f"env.run(['python', '-c', {stage_code!r}], timeout=120)\n",
        encoding="utf-8",
    )

    litetune = subprocess.Popen([sys.executable, str(runner)])
    request.addfinalizer(lambda: _reap_proc(litetune))

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not held.exists():
        if litetune.poll() is not None:
            pytest.fail(f"the runner died before the stage started (exit {litetune.returncode})")
        time.sleep(0.05)
    assert held.exists(), "the stage never started; nothing was under test"

    os.kill(litetune.pid, signal.SIGINT)
    assert (
        litetune.wait(timeout=30) == -signal.SIGINT
    ), "a real Ctrl-C must stay a KeyboardInterrupt and end the process on the signal"

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with open(lockfile, "a") as handle:
            taken = envs._try_lock(handle)
        if taken is None:
            pytest.skip("no advisory locking on this filesystem")
        if taken is True:
            return
        time.sleep(0.05)
    pytest.fail("the stage outlived the interrupted litetune process")


def test_both_termination_signals_are_guarded(monkeypatch):
    """Dropping SIGHUP from the loop left the whole suite green.

    SIGHUP is the case the guard is chiefly for: a dropped SSH session is how a
    six-hour `tune` ends in practice, and SIGTERM alone does not cover it.
    """
    taken = []
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: taken.append(sig))

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc()):
        installed = set(taken)

    assert {signal.SIGTERM, signal.SIGHUP} <= installed


def test_the_drain_says_only_what_the_kill_established(monkeypatch, caplog):
    """The four sentences are the reason `Reach` exists; swapping them was green.

    `killpg` returning 0 says the signal was accepted, not that anything died,
    so the group case must not name a departed grandchild as the cause.
    """

    class Wedged(_FakeProc):
        args = ["python"]
        stdout = None
        stderr = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("python", timeout, output=b"", stderr=b"")

    for reach, expected in (
        (envs.Reach.GROUP, "the group was signalled"),
        (envs.Reach.CHILD_ONLY, "only the child was signalled"),
        (envs.Reach.NOTHING, "nothing could be signalled at all"),
        (envs.Reach.ALREADY_GONE, "the child had already exited"),
    ):
        monkeypatch.setattr(envs, "_kill_tree", lambda proc, grace=None, pgid=None, r=reach: r)
        caplog.clear()
        with caplog.at_level("WARNING"):
            envs._kill_and_drain(Wedged())
        assert expected in caplog.text, f"{reach} must not be described as something else"


def test_the_wait_ends_when_the_group_does(tmp_path):
    """Nothing exercised this directly, and both obvious wrong versions passed.

    A version that never waits, and one that uses `proc.wait()` -- which reaps,
    freeing the pid that *is* the group id the next SIGKILL targets -- were both
    green. This one watches a real process and checks the child is left
    reapable, which is what makes the group id safe to reuse afterwards.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"], start_new_session=True
    )
    started = time.monotonic()
    envs._wait_without_reaping(proc, os.getpgid(proc.pid), 10.0)
    elapsed = time.monotonic() - started

    # Both bounds, because each alone is satisfied by a wrong version: a body
    # replaced by `return` is fast and reaps nothing, and one that sleeps the
    # whole grace waits long enough.
    assert elapsed >= 0.4, f"returned after {elapsed:.2f}s without waiting for the child"
    assert elapsed < 5, f"sat out {elapsed:.1f}s of a 10s grace after the child had exited"
    assert proc.returncode is None, "the child must be left unreaped, or its group id is freed"
    proc.communicate()


def test_a_dropped_variable_is_said_once_per_run(monkeypatch, caplog):
    """Once per subprocess was hundreds of lines through a progress report.

    Once per *process* was the other extreme: a notebook or a service is one
    process spanning many runs, and the second run had no record at all.
    """
    monkeypatch.setenv("PYTHONPATH", "/planted")
    envs.forget_reported_drops()

    with caplog.at_level("WARNING"):
        envs._child_env()
        envs._child_env()
    assert caplog.text.count("not passing PYTHONPATH") == 1, "said twice in one run"

    caplog.clear()
    envs.forget_reported_drops()
    with caplog.at_level("WARNING"):
        envs._child_env()
    assert "not passing PYTHONPATH" in caplog.text, "a second run inherited the first one's silence"


def test_a_variable_the_caller_asked_for_is_not_reported_as_dropped(monkeypatch, caplog):
    """Overrides are applied after the strip, so naming one is the opposite of true."""
    monkeypatch.setenv("PYTHONPATH", "/planted")
    envs.forget_reported_drops()

    with caplog.at_level("WARNING"):
        env = envs._child_env({"PYTHONPATH": "/deliberate"})

    assert env["PYTHONPATH"] == "/deliberate"
    assert "not passing PYTHONPATH" not in caplog.text


@pytest.mark.skipif(os.name == "nt", reason="SIGTERM dispositions are POSIX")
def test_the_grace_lets_a_stage_finish_what_it_started(tmp_path, monkeypatch):
    """The grace is only worth having if it has an effect, and nothing saw one.

    Every other test passes a grace explicitly or stubs the wait, so setting
    `_TERM_GRACE_S` to zero -- switching the grace off for the whole package --
    left the suite green. This one runs a stage that catches SIGTERM and writes
    a file on its way out, which is `save_pretrained` in miniature: with no
    grace, SIGKILL follows immediately and the file never appears.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "graceful")
    marker = tmp_path / "shut-down-cleanly"
    ready = tmp_path / "handler-installed"
    # A real source file, not a one-liner: the shutdown needs a `def`, and it
    # needs to take long enough that the scheduler cannot slip it in between an
    # immediate SIGTERM and SIGKILL. Without that this passes with the grace
    # switched off whenever the child happens to win the race, which is luck
    # rather than the property under test. A real `save_pretrained` is slower
    # than this by orders of magnitude.
    stage = "\n".join(
        (
            "import signal, sys, time",
            "def bye(*_a):",
            "    time.sleep(0.75)",
            f"    open({str(marker)!r}, 'w').write('1')",
            "    sys.exit(0)",
            "signal.signal(signal.SIGTERM, bye)",
            # Announced after the handler is in place: signalling before that
            # is answered by the default disposition, and the test would fail
            # for a reason that has nothing to do with the grace.
            f"open({str(ready)!r}, 'w').write('1')",
            "time.sleep(60)",
        )
    )

    proc = subprocess.Popen(
        [str(env.python), "-c", stage],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not ready.exists():
        if proc.poll() is not None:
            pytest.fail(f"the stage died before it was ready (exit {proc.returncode})")
        time.sleep(0.02)
    assert ready.exists(), "the stage never installed its handler; nothing was under test"

    envs._kill_tree(proc)
    proc.communicate()

    assert marker.exists(), "the stage was killed before it could finish shutting down"


def test_the_grace_is_given_even_where_only_the_child_could_be_signalled(monkeypatch):
    """Narrowing the guard to the group path drops it on the fallback silently.

    That fallback is the sandbox and setuid case -- exactly where an orphaned
    trainer comes back -- so it is the last place the grace should quietly go.
    """
    waited = []
    monkeypatch.setattr(envs, "_group_of", lambda proc: None)  # no group: child only
    monkeypatch.setattr(
        envs,
        "_wait_without_reaping",
        # Returns True: "the group is still ours to signal", which is what the
        # real one says unless the child was reaped out from under it.
        lambda proc, pgid, grace, stop=None: waited.append(grace) or True,
    )

    assert envs._kill_tree(_FakeProc(), grace=0.25) is envs.Reach.CHILD_ONLY
    assert waited == [0.25]


def test_a_delivered_sigterm_is_not_erased_by_a_refused_sigkill(monkeypatch):
    """`Reach` must carry the best reach, not the last attempt.

    A stage that execs a setuid helper mid-flight takes the SIGTERM and refuses
    the SIGKILL. Reporting that as "nothing could be signalled at all" sends the
    reader looking for a permissions problem that did not affect the first
    signal.
    """
    sent = []

    def group(pgid, sig):
        if sig == signal.SIGTERM:
            sent.append(sig)
            return
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", group)

    class Untouchable(_FakeProc):
        def send_signal(self, sig):
            raise PermissionError("operation not permitted")

    assert envs._kill_tree(Untouchable(), grace=0.01) is envs.Reach.GROUP
    assert sent == [signal.SIGTERM]


def test_a_host_that_reaps_for_us_gets_no_group_signal(monkeypatch):
    """A host that ignores SIGCHLD has the kernel reap children as they exit.

    `Popen` never learns, so `returncode` stays `None` while the pid -- which
    *is* the group id -- is already free for reuse. No observation closes that
    window, because any answer is stale by the time the signal goes out, so the
    group is not addressed there at all.

    An earlier attempt asked `poll()` before reading the group. That reaps, and
    reaping the leader both frees the id and makes `getpgid` answer ESRCH, so
    it skipped the sweep in the one case the module is written for -- the child
    exited and a grandchild still holds the pipe. It leaked.
    """
    if not hasattr(signal, "SIGCHLD"):
        pytest.skip("SIGCHLD is POSIX")
    group_signals = []
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: group_signals.append(sig))
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_IGN)

    proc = _FakeProc()
    assert envs._kill_tree(proc, grace=0.01) is envs.Reach.CHILD_ONLY
    assert group_signals == [], "the group id is not ours to signal where the kernel reaps"
    assert proc.direct_signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_a_grandchild_is_collected_after_its_parent_has_exited(tmp_path, monkeypatch):
    """The case the whole module exists for, and it leaked.

    The child exits and a descendant keeps running, holding the inherited
    pipes. Two separate things broke this: a `poll()` guard that reaped the
    leader and then skipped the group kill, and -- on macOS -- `getpgid`
    answering ESRCH for a zombie leader, which made the group unreachable
    exactly when it still had a live member.
    """
    child = (
        "import subprocess, sys;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
        "sys.exit(0)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and proc.poll() is None:
        time.sleep(0.02)
    # `poll` above reaped the leader, which is what the production path must
    # never do -- so put the state back the way `run` sees it.
    proc.returncode = None

    # The group as `run` captured it: at spawn, while the child was alive.
    # Looking it up now would fail on macOS, which does not report the group
    # of a zombie -- and that is the whole reason `run` reads it early.
    assert envs._kill_tree(proc, grace=0.2, pgid=pgid) is envs.Reach.GROUP
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)
