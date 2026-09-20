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
    import venv

    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(
        name="probe",
        requirements=("pyyaml==6.0.2",),
        python_ceiling=(3, 12),
        ceiling_pin="numpy==2.0.2",
    )

    started = []
    # `_run_guarded`, not `subprocess.run`: the install moved onto the same
    # guarded path the stages use, and a tripwire left on the old function
    # would report "the refusal came before pip" whatever pip did.
    monkeypatch.setattr(envs, "_run_guarded", lambda *a, **k: started.append(a))
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
        envs, "_run_guarded", lambda *a, **k: types.SimpleNamespace(returncode=0, stderr="")
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

    def skip_our_install(cmd, *args, **kwargs):
        installs.append(list(cmd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    # Only litetune's own install is stubbed. `EnvBuilder.create` shells out
    # to `ensurepip` through `subprocess.run`, and that call is the one
    # installing the console script under test -- it has to really run. The
    # two used to share a function and the stub had to tell them apart by
    # looking for "install" in the command line; they no longer do.
    monkeypatch.setattr(envs, "_run_guarded", skip_our_install)
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
    `errors="surrogateescape"` decode the fix lives in, not a
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
    """The other half of the same fix, pinned independently of the decode's error handler.

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


class _WedgedProc:
    """A child that never exits, with real pipes that never close.

    Real descriptors, because the drain reads them with `os.read`: a stub
    `communicate` would only prove the stub was called, and it is exactly the
    call the drain stopped making. The write ends stay open until `close`, so
    the pipes never reach EOF -- the state the "are still open" warning is
    about, and the one a grandchild outside the killed group produces.
    """

    args = ["python"]
    returncode = None

    def __init__(self, on_stdout=b"", on_stderr=b"", pid=4242, encoding="utf-8", err_encoding=None):
        self.pid = pid
        self.direct_signals: list[int] = []
        self._writers: list[int] = []
        self.stdout = self._pipe(on_stdout, encoding)
        self.stderr = self._pipe(on_stderr, err_encoding or encoding)

    def _pipe(self, payload, encoding="utf-8"):
        read_fd, write_fd = os.pipe()
        if payload:
            os.write(write_fd, payload)
        self._writers.append(write_fd)
        return open(read_fd, encoding=encoding, errors="replace")

    def send_signal(self, sig):
        self.direct_signals.append(sig)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("fake", timeout)

    def close(self):
        for fd in self._writers:
            with contextlib.suppress(OSError):
                os.close(fd)

    def as_communicate_left_it(self):
        """The state a completed `_communicate` leaves: both pipes at EOF and
        both stream objects closed. It only returns when every descriptor has
        reached EOF, so a timeout raised by the `wait()` after it can only
        ever be seen in this state -- which is what makes it the shape the
        seeding has to be tested against."""
        self.close()
        for stream in (self.stdout, self.stderr):
            with contextlib.suppress(OSError):
                stream.close()
        return self


class _FakePopen:
    """A child that starts, says nothing and exits zero.

    Enough for `_run_guarded` to get through: it reads the group, installs the
    guard, and calls `communicate`. Used where the test is about what was
    handed to `Popen` rather than about anything the child does.
    """

    args = ["python"]
    pid = 4242
    returncode = 0
    stdout = None
    stderr = None

    def communicate(self, timeout=None):
        return "", ""

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class _FakeProc:
    """A `Popen` stand-in that records what was signalled, and never dies.

    `poll` answers from `returncode` the way the real one does, because
    `_kill_tree` and `_signal_tree` both ask it: a host that ignores SIGCHLD
    has the kernel reap for us, and `Popen` only learns through `poll`.
    """

    # A real `Popen` always has these, whether or not it was given pipes, and
    # the termination guard now builds a drain around them.
    args = ["python"]
    stdout = None
    stderr = None

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
    # The wait is not what these pin, and a fake pid is not anyone's child --
    # `waitid` says ECHILD for it on Linux, which is right and which made the
    # real wait decide the outcome here. Stubbed so the assertion is about the
    # signals.
    monkeypatch.setattr(envs, "_wait_without_reaping", lambda *a, **k: True)
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
    # The wait is not what these pin, and a fake pid is not anyone's child --
    # `waitid` says ECHILD for it on Linux, which is right and which made the
    # real wait decide the outcome here. Stubbed so the assertion is about the
    # signals.
    monkeypatch.setattr(envs, "_wait_without_reaping", lambda *a, **k: True)
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

    # At `Popen`, which is where the environment is actually handed over.
    # Capturing at a wrapper only proves the wrapper was called.
    def capture(argv, **kwargs):
        seen.update(kwargs.get("env") or {})
        return _FakePopen()

    monkeypatch.setattr(envs.subprocess, "Popen", capture)
    StageEnv(name="pipenv", requirements=("pyyaml==6.0.2",)).provision()
    assert seen, "the install must have started a process"
    assert "PIP_TARGET" not in seen


# ---------------------------------------------------------------------------
# What "ready" is evidence of
# ---------------------------------------------------------------------------


def test_a_directory_named_like_the_marker_is_not_ready(tmp_path, monkeypatch):
    """The marker is a file `provision` writes; nothing else is evidence.

    `.exists()` answered yes for a directory of the same name, so an
    environment nobody finished read as ready, `provision` short-circuited on
    it, and the cache listing agreed.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name="counterfeit", requirements=("pyyaml==6.0.2",))
    env.python.parent.mkdir(parents=True, exist_ok=True)
    env.python.symlink_to(sys.executable)
    (env.path / ".litetune-ready").mkdir()

    assert not env.ready
    listed = [entry for entry in envs.cached_environments() if entry.path == env.path]
    assert listed and not listed[0].ready, "and the cache listing must agree"


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

    def kill_and_signal_again(proc, grace=_TERM_GRACE_SENTINEL, stop=None, pgid=None, pump=None):
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


def test_a_second_signal_cuts_the_first_grace_short(monkeypatch):
    """Killing outright is half of it; the first grace must also stop waiting.

    The grace `_kill_tree` gives the first signal ends early when its `stop`
    answers yes, and the handler's escalation is what makes it answer yes.
    Without that, the second signal's SIGKILL goes out and the first frame
    still sits out the rest of its `_TERM_GRACE_S` before it can end
    litetune.
    """
    asked: list = []

    def kill(proc, grace=None, stop=None, pgid=None, pump=None):
        if grace == 0:
            return envs.Reach.GROUP
        asked.append(stop is not None and stop())
        handler(signal.SIGHUP, None)  # a second signal, mid-grace
        asked.append(stop is not None and stop())
        return envs.Reach.GROUP

    monkeypatch.setattr(envs, "_kill_tree", kill)
    captured: dict = {}
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc()):
        handler = captured[signal.SIGTERM]
        with pytest.raises(envs.StageInterrupted):
            handler(signal.SIGTERM, None)

    assert asked == [False, True], "the grace was not told that a second signal arrived"


def test_a_handler_whose_grace_was_interrupted_can_still_be_signalled(monkeypatch):
    """The route the `finally` around the handler's grace exists for.

    A Ctrl-C during the grace raises out of `_kill_tree`, so the disposition
    restore and the re-delivery after it never run, and both TERM and HUP
    still point at this handler during `run`'s cleanup. If the handler were
    still marked as running, whichever arrived next would take the
    escalation branch -- a zero grace, and a return instead of ending
    litetune -- for up to twenty seconds of that cleanup.
    """
    graces: list = []
    default = object()

    def kill(proc, grace=default, stop=None, pgid=None, pump=None):
        graces.append(grace)
        if len(graces) == 1:
            raise KeyboardInterrupt
        return envs.Reach.GROUP

    monkeypatch.setattr(envs, "_kill_tree", kill)
    captured: dict = {}
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc()):
        with pytest.raises(KeyboardInterrupt):
            captured[signal.SIGTERM](signal.SIGTERM, None)
        # `os.kill` is a no-op here, so a handler that is not stuck ends in
        # the raise it uses when the process survives its own re-delivery.
        with pytest.raises(envs.StageInterrupted):
            captured[signal.SIGHUP](signal.SIGHUP, None)

    assert graces == [default, default], "the next signal was treated as a second one"


@pytest.mark.parametrize("interrupted", [False, True], ids=["returned", "interrupted"])
def test_the_handler_gives_its_selector_back(monkeypatch, interrupted):
    """The handler's drain holds a selector over the stage's pipes.

    Its grace can end two ways -- by returning, or by an interrupt raised out
    of it -- and neither may keep that descriptor open for the rest of the
    run: `run`'s own cleanup builds another drain over the same pipes next.
    """
    drains: list = []
    real_drain = envs._PipeDrain

    def note(proc_, already=(None, None), *, seed=True):
        drains.append(real_drain(proc_, already, seed=seed))
        return drains[-1]

    def kill(proc, **kwargs):
        assert drains[-1]._selector is not None, "nothing to give back; the test proves nothing"
        if interrupted:
            raise KeyboardInterrupt
        return envs.Reach.GROUP

    monkeypatch.setattr(envs, "_PipeDrain", note)
    monkeypatch.setattr(envs, "_kill_tree", kill)
    captured: dict = {}
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    proc = _WedgedProc()
    try:
        with envs._kill_child_if_we_are_told_to_exit(proc):
            with pytest.raises(KeyboardInterrupt if interrupted else envs.StageInterrupted):
                captured[signal.SIGTERM](signal.SIGTERM, None)
    finally:
        proc.close()

    assert len(drains) == 1
    assert drains[0]._selector is None, "the handler's selector outlived its grace"


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
        lambda proc, pgid, grace, stop=None, pump=None: waited.append(grace) or True,
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

    def exits_during_the_grace(target, pgid, grace, stop=None, pump=None):
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


def test_the_grace_is_usable_by_a_stage_that_has_something_to_say(tmp_path, monkeypatch):
    """The grace was only ever usable by a quiet stage, which is backwards.

    Nothing emptied the pipes between SIGTERM and SIGKILL. A pipe here holds
    65,536 bytes, measured by writing to one until EAGAIN;
    a stage that writes more than that on the way out -- a torch traceback, the
    last frames of a progress bar, `save_pretrained` logging each shard --
    blocks in `write` partway through, never reaches the end of its shutdown,
    and is SIGKILLed at the end of a grace it spent blocked. The stage with the
    most to say lost the most of it.

    Measured against the code this replaced, same child, same timeout: 65,536
    bytes arrived -- exactly one pipe buffer -- and the whole five-second grace
    was spent blocked. Both halves matter, so both are asserted: a version that
    drained only after the kill would still truncate, and one that drained
    without letting the child finish would still take the full grace.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "chatty")
    said = 200_000  # a little over three 65,536-byte buffers
    child = (
        "import signal, sys, time\n"
        "def bye(signum, frame):\n"
        f"    sys.stderr.write('X' * {said})\n"
        "    sys.stderr.flush()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, bye)\n"
        "sys.stdout.write('running\\n'); sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        env.run(["python", "-c", child], timeout=1)
    elapsed = time.monotonic() - started

    assert len(caught.value.stderr or "") == said, (
        "the stage was cut off mid-sentence: the pipes were not being read while "
        "it shut down, so it blocked on a full one"
    )
    assert "running" in (caught.value.output or ""), "stdout was drained too"
    assert elapsed < 1 + envs._TERM_GRACE_S, (
        "the child finished inside the grace and the wait must end with it; "
        "spending the whole grace means it was blocked for it"
    )


def test_a_drain_that_reached_the_end_does_not_sit_out_its_budget(monkeypatch):
    """`finish` returns at EOF and `pump` does not, and the difference is ten
    seconds on every single timeout.

    `pump` stands in for the sleep inside the grace, so it must spend what it
    is given whatever the pipes do. The drain after the kill must not: when
    that one was shaped the same way, the fix for the defect above worked and
    added `_DRAIN_AFTER_KILL_S` to the wall-clock of every timed-out stage.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    proc = _WedgedProc(on_stdout=b"done")
    proc.close()  # both pipes are at EOF before the drain starts

    drain = envs._PipeDrain(proc)
    started = time.monotonic()
    drain.finish(5)
    assert time.monotonic() - started < 1, "finish must return once the pipes are done"

    started = time.monotonic()
    drain.pump(0.3)
    assert time.monotonic() - started >= 0.25, "pump must spend its budget, EOF or not"


def test_the_install_is_killed_with_litetune(tmp_path, monkeypatch):
    """`provision` was the last unguarded subprocess in the package.

    `subprocess.run` kills its direct child, and only on a path that raises. A
    SIGTERM to litetune raises nothing, and a timeout reaches pip and not the
    compilers pip spawned -- so half an hour of `pip install torch` carried on
    writing into a directory the restore path had already decided to discard.
    Its own session is what makes the group killable, and the guard is what
    kills it.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    monkeypatch.setattr(envs.venv.EnvBuilder, "create", _fake_venv)

    started = {}
    guarded = []
    real_guard = envs._kill_child_if_we_are_told_to_exit

    @contextlib.contextmanager
    def note_the_guard(proc, pgid=None):
        guarded.append(proc)
        with real_guard(proc, pgid):
            yield

    def capture(argv, **kwargs):
        started.update(kwargs)
        return _FakePopen()

    monkeypatch.setattr(envs.subprocess, "Popen", capture)
    monkeypatch.setattr(envs, "_kill_child_if_we_are_told_to_exit", note_the_guard)
    StageEnv(name="guarded", requirements=("pyyaml==6.0.2",)).provision()

    assert started.get("start_new_session") is True, (
        "without a session of its own the install shares litetune's process "
        "group, and there is no group left to kill that is only pip's"
    )
    assert guarded, "the install must run inside the termination guard"


def test_a_selector_that_accepts_and_then_refuses_does_not_escape(monkeypatch):
    """`SelectSelector.register` checks the event mask, the descriptor and
    duplicate registration -- and not whether `select` on this host can do
    anything with what it is handed. Measured: it accepts a pipe and keeps
    the descriptor. So where `select` takes sockets only, both pipes register
    and the refusal arrives on the first `select`, which runs inside
    `_kill_tree` between the SIGTERM and the SIGKILL.

    Letting it out of there costs three things at once: the group is never
    SIGKILLed, the child is never reaped, and the caller gets an errno where
    its own `TimeoutExpired` should be -- so `provision`'s `except
    TimeoutExpired` stops matching and a pip timeout surfaces as a socket
    error. Deciding the fallback from registration alone could not catch this,
    because registration succeeded.
    """
    killed: list[str] = []

    class RefusesOnSelect(envs.selectors.SelectSelector):
        def select(self, timeout=None):
            raise OSError(10038, "not a socket")

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesOnSelect)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.2)
    monkeypatch.setattr(
        envs, "_kill_tree", lambda proc, pump=None, **kwargs: (pump(0.05), killed.append("yes"))[1]
    )

    class Talkative(_WedgedProc):
        def communicate(self, timeout=None):
            return "everything it said", "and why"

    proc = Talkative()
    try:
        out, err = envs._kill_and_drain(proc)
    finally:
        proc.close()

    assert killed, "the kill must have run to completion"
    assert (out, err) == (
        "everything it said",
        "and why",
    ), "the drain gave up on watching, so the fallback had to take over"


def test_a_host_that_cannot_watch_its_pipes_still_gets_the_output(monkeypatch):
    """Windows `select` handles sockets, not pipes, and a POSIX host can run
    out of the descriptor `epoll` needs.

    Without a fallback the new drain would register nothing there and read
    nothing, so a timed-out stage would say nothing at all about why -- worse
    than the code this replaced, which used `communicate` and did read. The
    fallback is only safe after the kill, and that is where it is called from.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors,
        "DefaultSelector",
        lambda: (_ for _ in ()).throw(OSError("no descriptors to spare")),
    )

    class Talkative(_FakePopen):
        def communicate(self, timeout=None):
            return "everything it said", "and why"

    out, err = envs._kill_and_drain(Talkative())
    assert (out, err) == ("everything it said", "and why")


def test_the_fallback_keeps_the_seed_when_communicate_assembles_nothing(monkeypatch):
    """Its trailing `wait()` can raise after the pipes are done, and then the
    call returns nothing while the earlier read is all there is."""
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    class Wedged(_FakePopen):
        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("python", timeout)

    out, err = envs._kill_and_drain(Wedged(), already=(b"said before", None))
    assert out == "said before" and err == ""


def test_the_partial_output_of_a_wedged_stage_is_kept(monkeypatch):
    """The stderr of a timed-out stage is the one artifact that explains it.

    This is the path where a grandchild left the group and still holds the
    pipes: the drain gives up, and what the stage managed to say before that
    must survive rather than being replaced by two empty strings.
    """
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.2)
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)

    proc = _WedgedProc(on_stdout=b"partial", on_stderr=b"boom")
    try:
        out, err = envs._kill_and_drain(proc)
    finally:
        proc.close()
    assert out == "partial" and err == "boom"


def test_what_the_earlier_read_took_off_the_pipes_is_not_lost(monkeypatch):
    """The narrow case, and the reason the drain does not trust the exception.

    `communicate` assembles its result at the very end, so a timeout raised
    from its trailing `wait()` -- pipes at EOF, process still alive -- carries
    no output at all while every byte sits in the `Popen`. Those bytes are off
    the descriptors, so no amount of reading gets them back; taking the
    exception's word for it returned two empty strings for a stage that had
    said plenty.
    """
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.2)
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)

    proc = _WedgedProc()
    proc._fileobj2output = {proc.stdout: [b"said"], proc.stderr: [b"complained"]}
    # In the state `_communicate` actually leaves: EOF on both, both closed.
    # With the pipes left open instead, this test passed while exercising a
    # different branch entirely -- the drain still had something to watch, so
    # it never showed that a closed-pipe drain keeps its seed.
    proc.as_communicate_left_it()
    try:
        # As `_run_guarded` calls it on that path: an exception carrying None.
        out, err = envs._kill_and_drain(proc, already=(None, None))
    finally:
        proc.close()
    assert out == "said" and err == "complained"


def test_the_exception_is_used_when_the_popen_kept_nothing(monkeypatch):
    """`_fileobj2output` is a CPython private, so it is asked for, not assumed.

    An implementation that keeps its partial reads elsewhere leaves the
    caller's exception as the only account of them, which is where this module
    was before -- so that path has to keep working rather than silently
    returning nothing.
    """
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.2)
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)

    proc = _WedgedProc()
    assert not hasattr(proc, "_fileobj2output")
    try:
        out, err = envs._kill_and_drain(proc, already=(b"from the exception", None))
    finally:
        proc.close()
    assert out == "from the exception" and err == ""


def test_bytes_from_a_text_mode_pipe_are_decoded():
    """`_check_timeout` attaches the raw chunks without decoding, even in text
    mode, so the decode has to happen on the way out.

    A byte that is not UTF-8 survives as the lone surrogate `surrogateescape`
    gives it, the same handler the pipe itself is opened with: `replace` would
    write U+FFFD, and U+FFFD is a character a model may legitimately generate,
    so erasing the difference costs `evaluate` the one signal that says this
    text never decoded.
    """
    assert envs._as_text(None) == ""
    assert envs._as_text("already text") == "already text"
    assert envs._as_text(b"a \xff byte") == "a \udcff byte"


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
def test_a_terminated_litetune_lets_the_stage_finish_its_sentence(tmp_path, request):
    """The grace is only worth having on the path `kill <litetune>` takes.

    The timeout path was pumped first and the signal path was not, which left
    the fix on the wrong side: a timeout is a report, while a SIGTERM during a
    six-hour `tune` is the case where the grace exists so torch can finish the
    `save_pretrained` it had started. Measured with the pump missing, a stage
    writing 200,000 bytes in its SIGTERM handler before writing its marker:
    litetune exited in 5.00s and the marker was never written -- the stage
    spent the whole grace blocked on a full pipe and was SIGKILLed.

    A quiet stage cannot see this. `test_a_terminated_litetune_takes_the_stage
    _with_it` above holds a lock and sleeps, so it passes either way.
    """
    started = tmp_path / "stage.started"
    finished = tmp_path / "stage.finished"
    stage_code = (
        "import signal, sys, time\n"
        "def bye(signum, frame):\n"
        "    sys.stderr.write('X' * 200_000)\n"
        "    sys.stderr.flush()\n"
        f"    open({str(finished)!r}, 'w').write('1')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, bye)\n"
        f"open({str(started)!r}, 'w').write('1')\n"
        "time.sleep(60)\n"
    )
    runner = tmp_path / "chatty_runner.py"
    runner.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(pathlib.Path(envs.__file__).parents[1])!r})\n"
        f"os.environ['LITETUNE_ENV_DIR'] = {str(tmp_path)!r}\n"
        "from litetune.envs import StageEnv\n"
        "env = StageEnv(name='chattykill', requirements=('pyyaml==6.0.2',))\n"
        "env.python.parent.mkdir(parents=True, exist_ok=True)\n"
        "env.python.symlink_to(sys.executable)\n"
        f"env.run(['python', '-c', {stage_code!r}], timeout=120)\n",
        encoding="utf-8",
    )

    litetune = subprocess.Popen([sys.executable, str(runner)])
    request.addfinalizer(lambda: _reap_proc(litetune))

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not started.exists():
        if litetune.poll() is not None:
            pytest.fail(f"the runner died before the stage started (exit {litetune.returncode})")
        time.sleep(0.05)
    assert started.exists(), "the stage never started; nothing was under test"

    os.kill(litetune.pid, signal.SIGTERM)
    assert litetune.wait(timeout=30) == -signal.SIGTERM, "litetune must still die as it would have"

    assert finished.exists(), (
        "the stage was SIGKILLed part way through its shutdown: nothing emptied "
        "the pipes during the grace, so it blocked after one buffer"
    )


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
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    for reach, expected in (
        (envs.Reach.GROUP, "the group was signalled"),
        (envs.Reach.CHILD_ONLY, "only the child was signalled"),
        (envs.Reach.NOTHING, "nothing could be signalled at all"),
        (envs.Reach.ALREADY_GONE, "the child had already exited"),
    ):
        monkeypatch.setattr(envs, "_kill_tree", lambda proc, r=reach, **kwargs: r)
        proc = _WedgedProc()
        caplog.clear()
        try:
            with caplog.at_level("WARNING"):
                envs._kill_and_drain(proc)
        finally:
            proc.close()
        assert expected in caplog.text, f"{reach} must not be described as something else"


def test_nothing_is_said_when_the_pipes_did_close(monkeypatch, caplog):
    """The warning is about pipes still held, and it is the thing that sends a
    reader hunting for a leaked process. A drain that reached the end of both
    pipes has nothing to report, and saying so anyway would make the warning
    worthless by making it constant."""
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.2)
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)

    proc = _WedgedProc(on_stdout=b"all of it")
    proc.close()  # both write ends gone: the pipes end where the data ends
    with caplog.at_level("WARNING"):
        out, _ = envs._kill_and_drain(proc)
    assert out == "all of it"
    # Asserted against the words the warning actually uses. This line read
    # "still holding them" -- the old wording -- for four review rounds after
    # the warning was rewritten, so it could not fail: that phrase is no
    # longer anywhere in the output, and a drain that warned on every closed
    # pipe passed.
    assert "are still open" not in caplog.text


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
        lambda proc, pgid, grace, stop=None, pump=None: waited.append(grace) or True,
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
def test_a_stage_reads_the_same_whether_it_finished_or_was_killed(tmp_path, monkeypatch):
    r"""Two paths describing one child must not disagree about what it said.

    `communicate` runs `_translate_newlines` and the raw read does not, so a
    `\r`-heavy stage -- any torch progress bar -- came back as lines when it
    finished and as terminal overwrites when it timed out, inside the error
    report that exists to explain the timeout.
    """
    env = _symlinked_env(tmp_path, monkeypatch, "crlf")
    emit = r"import sys; sys.stdout.write('a\r\nb\rc\n'); sys.stdout.flush()"

    finished = env.run(["python", "-c", emit], timeout=30)
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        env.run(["python", "-c", emit + "; import time; time.sleep(30)"], timeout=1)

    assert finished.stdout == "a\nb\nc\n"
    assert caught.value.output == finished.stdout


def test_an_interrupted_read_does_not_end_the_stream(monkeypatch):
    """A single EINTR must not cost every byte after it.

    The read's `except OSError` closes the pipe, and `InterruptedError` is an
    `OSError` -- so one interrupted read was indistinguishable from end of
    stream and dropped the descriptor for good. Verified before the fix by
    planting one: the read end closed, and the next write to the pipe raised
    `BrokenPipeError`. PEP 475 makes `os.read` retry EINTR itself unless a
    Python handler raised, so this is close to unreachable -- and the cost of
    it being reachable is out of all proportion to the two lines.
    """
    proc = _WedgedProc(on_stdout=b"before")
    drain = envs._PipeDrain(proc)
    real_read = envs.os.read
    reads = {"n": 0}

    def interrupted_once(fd, size):
        reads["n"] += 1
        if reads["n"] == 1:
            raise InterruptedError(4, "Interrupted system call")
        return real_read(fd, size)

    try:
        monkeypatch.setattr(envs.os, "read", interrupted_once)
        drain.pump(0.1)
        monkeypatch.undo()
        os.write(proc._writers[0], b" and after")
        drain.pump(0.1)
    finally:
        drain.close()
        proc.close()

    assert drain.text()[0] == "before and after", (
        "the pipe was closed on an interrupted read, so everything the stage "
        "said after it was lost"
    )


def test_an_interrupted_reap_still_gives_the_selector_back(monkeypatch):
    """The reap waits, and a wait can be interrupted.

    With the reap ahead of the close in the same `finally`, a signal landing
    in `proc.wait` skipped the close and leaked a kqueue or epoll descriptor
    for every interrupted stage. Before the reap moved into that `finally`,
    the close was the whole of it and could not be skipped; that is the
    property, and nothing was watching it.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)
    monkeypatch.setattr(envs, "_reap", lambda proc: (_ for _ in ()).throw(KeyboardInterrupt))

    opened: list[int] = []
    real_drain = envs._PipeDrain

    def remember(proc_, already=(None, None), *, seed=True):
        drain = real_drain(proc_, already, seed=seed)
        if drain._selector is not None:
            opened.append(drain._selector.fileno())
        return drain

    monkeypatch.setattr(envs, "_PipeDrain", remember)

    proc = _WedgedProc(on_stdout=b"x")
    proc.close()
    with pytest.raises(KeyboardInterrupt):
        envs._kill_and_drain(proc)

    assert opened, "no selector was opened, so there is nothing under test"
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_the_handlers_grace_is_pumped_with_the_one_that_sleeps(monkeypatch):
    """The same choice as on the timeout path, at the other call site.

    `finish` returns the moment there is nothing left to read, so handing it
    to a wait loop turns the loop into a busy one for the rest of the grace
    -- here, inside a signal handler, on every `kill <litetune>`. The timeout
    path's call site is covered; this one was not, and the mutation survived.
    """
    handed: list = []
    monkeypatch.setattr(
        envs,
        "_kill_tree",
        lambda proc, pump=None, **kwargs: (handed.append(pump), envs.Reach.GROUP)[1],
    )
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    captured: dict = {}
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    with envs._kill_child_if_we_are_told_to_exit(_FakeProc(), pgid=4242):
        with pytest.raises(envs.StageInterrupted):
            captured[signal.SIGTERM](signal.SIGTERM, None)

    assert handed and handed[0] is not None, "the handler's grace must pump"
    started = time.monotonic()
    handed[0](0.3)
    assert (
        time.monotonic() - started >= 0.25
    ), "the handler was handed a pump that returns early, so its grace spins"


def test_a_pipe_open_on_either_side_is_reported(monkeypatch, caplog):
    """`_pipes_open` has to ask about both, and an asymmetric pair is the
    ordinary state rather than an edge case: a leader that closed its stdout
    while a grandchild kept stderr, or the other way round.

    With both pipes always in the same state, `any`, `all`, and either one of
    them alone are indistinguishable -- three mutations that survived.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    def timed_out(timeout=None):
        raise subprocess.TimeoutExpired("python", timeout, output=b"", stderr=b"")

    for closed, label in (("stdout", "only stderr held"), ("stderr", "only stdout held")):
        proc = _WedgedProc(on_stdout=b"o", on_stderr=b"e")
        getattr(proc, closed).close()
        proc.communicate = timed_out
        caplog.clear()
        try:
            with caplog.at_level("WARNING"):
                envs._kill_and_drain(proc)
        finally:
            proc.close()
        assert "are still open" in caplog.text, label


def test_the_drain_gives_its_selector_descriptor_back():
    """`kqueue` and `epoll` each cost a descriptor of their own.

    One per timed-out stage, held for the life of the run, is a leak that
    surfaces much later as an unrelated `OSError` about too many open files.
    Asserted on the selector's own descriptor rather than by counting the
    process's: everything else in reach here is a pipe belonging to the fake.
    """
    proc = _WedgedProc(on_stdout=b"x")
    drain = envs._PipeDrain(proc)
    try:
        assert drain.watching, "nothing to assert about if it never opened one"
        fd = drain._selector.fileno()
        assert fd >= 0 and os.fstat(fd), "the selector should be holding a descriptor"
        drain.close()
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        proc.close()


def test_the_fallback_reads_like_the_stage_wrote(monkeypatch, caplog):
    """The `communicate` fallback, with a real pipe and a real encoding.

    Both existing tests of this branch hand it a fake whose pipes are `None`,
    so the encoding is always the UTF-8 default and the stub returns ASCII
    `str` -- nothing is ever decoded, and nothing is ever translated. Four
    separate mutations survived the suite behind that: decoding in the
    default encoding instead of the pipe's, stderr decoded in stdout's, and
    both hard-coded answers to "are the pipes still held".

    It matters most exactly here. This branch exists for hosts whose `select`
    cannot take a pipe -- Windows -- which is where a non-UTF-8 console
    encoding and `\r\n` are the ordinary case, not the exotic one.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    said = "RuntimeError: n\u2019a pas pu charger le mod\u00e8le"
    printed = (
        "\u0437\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u043c\u043e\u0434\u0435\u043b\u0438"  # noqa: RUF001
    )
    # Two encodings, not one: with both pipes the same, a stderr decoded in
    # stdout's encoding is indistinguishable from a correct one.
    proc = _WedgedProc(encoding="cp1251", err_encoding="cp1252")

    def timed_out(timeout=None):
        raise subprocess.TimeoutExpired(
            "python",
            timeout,
            output=printed.encode("cp1251") + b"\r\nb\rc\n",
            stderr=said.encode("cp1252") + b"\r\n",
        )

    proc.communicate = timed_out
    try:
        with caplog.at_level("WARNING"):
            out, err = envs._kill_and_drain(proc)
    finally:
        proc.close()

    assert err == said + "\n", "stderr must be decoded in the encoding its pipe was opened with"
    assert out == printed + "\nb\nc\n", (
        "stdout too: with an ASCII stdout, decoding it in the wrong encoding "
        "is invisible and the stderr assertion carries the whole test"
    )
    assert "are still open" in caplog.text, (
        "the pipes were open, so the leak warning has to fire -- it is the "
        "only thing that names a stray process at the time it happens"
    )


def test_the_fallback_reports_pipes_that_really_are_held(monkeypatch, caplog):
    """The other half of the same claim.

    A test that only checks the warning stays quiet is satisfied by a
    `still_held` hard-coded to `False`, which is how the previous version of
    this pair let that mutation through. This is the case where the pipes are
    open -- a grandchild outside the killed group -- and the warning is the
    only thing that names a stray process while it is still findable.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    # Only stderr is held, so an `all` in place of the `any` reports nothing
    # for a grandchild that inherited one pipe and not the other.
    proc = _WedgedProc(on_stdout=b"said")
    proc.stdout.close()

    def timed_out(timeout=None):
        raise subprocess.TimeoutExpired("python", timeout, output=b"said", stderr=b"")

    proc.communicate = timed_out
    try:
        with caplog.at_level("WARNING"):
            envs._kill_and_drain(proc)
    finally:
        proc.close()
    assert "are still open" in caplog.text


def test_the_interrupt_bound_counts_only_a_run_of_them(monkeypatch):
    """The bound is on *consecutive* interruptions, and the reset is what
    makes that word true.

    Without it the count only ever grows, so a pipe that is interrupted now
    and then -- which is the case the retry exists for -- is eventually
    closed anyway, and the data after it is lost.

    Asserted on the text that came through, not on `watching`. `watching`
    describes the host, and giving one pipe up does not change it -- so the
    previous version's assertion could not fail for the reason it named, and
    the mutation it was meant to catch was caught by accident, when the
    test's own next write hit a closed pipe.
    """
    proc = _WedgedProc()
    drain = envs._PipeDrain(proc)
    real_read = envs.os.read
    flips = {"n": 0}

    def every_other_one(fd, size):
        flips["n"] += 1
        if flips["n"] % 2:
            raise InterruptedError(4, "Interrupted system call")
        return real_read(fd, size)

    written = []
    try:
        monkeypatch.setattr(envs.os, "read", every_other_one)
        # A fixed count, not one derived from the bound: tied to the
        # constant, a bound of 100,000 turned this into half an hour.
        for i in range(24):
            chunk = f"<{i}>".encode()
            written.append(chunk)
            os.write(proc._writers[0], chunk)
            drain.pump(0.02)
    finally:
        monkeypatch.undo()
        text = drain.text()[0]
        drain.close()
        proc.close()

    assert text == b"".join(written).decode(), (
        "scattered interruptions closed the pipe part way through: the count "
        "is not being reset by the reads between them"
    )


def test_the_rescue_kill_does_not_signal_a_group_that_is_no_longer_ours(monkeypatch):
    """The rescue SIGKILL has to obey the same guard as the ordinary one.

    Once the leader has been reaped the group id is the leader's freed pid,
    and a second termination signal can reap it while this grace is being
    interrupted. Signalling anyway aims SIGKILL at whatever now owns that
    number -- which is the accident the whole module is arranged to avoid,
    arriving through the rescue added to prevent a different one.
    """
    sent: list[int] = []
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: sent.append(sig))

    proc = _FakeProc()

    def reaped_by_somebody_else(*args, **kwargs):
        proc.returncode = 0  # as `Popen.send_signal`'s own poll would leave it
        raise KeyboardInterrupt

    monkeypatch.setattr(envs, "_wait_without_reaping", reaped_by_somebody_else)

    with pytest.raises(KeyboardInterrupt):
        envs._kill_tree(proc, grace=0.25)

    assert sent == [signal.SIGTERM], (
        "the leader was reaped during the grace, so the group id is no longer "
        "ours and the rescue must not use it"
    )


def test_the_discarded_drain_is_not_seeded_either(monkeypatch):
    """`run`'s own cleanup discards its drain, exactly as the handler does.

    Seeding it means a `b"".join` over everything the stage has said, on the
    way out of an interrupt, and the result is thrown away -- the same cost
    and the same `MemoryError` exposure that `seed=False` removed one frame
    in.
    """
    seen: dict = {}
    real = envs._kill_and_drain

    def note(proc, pgid=None, already=(None, None), *, seed=True):
        seen["seed"] = seed
        return real(proc, pgid, already, seed=seed)

    monkeypatch.setattr(envs, "_kill_and_drain", note)

    class Interrupted(_FakePopen):
        def communicate(self, timeout=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(envs.subprocess, "Popen", lambda *a, **k: Interrupted())
    with pytest.raises(KeyboardInterrupt):
        envs._run_guarded(["python", "-c", "pass"], timeout=5)
    assert seen.get("seed") is False, "the discarded drain must not pay for a seed"


def test_the_rescue_asks_returncode_and_does_not_poll(monkeypatch):
    """`returncode`, not `poll()`, and the difference is the whole hazard.

    Every fake in this file has a `poll()` with no side effect, so the two
    were indistinguishable and swapping them passed. A real `poll()` reaps.
    For a leader that has exited but not been collected, that frees the pid
    which *is* the group id -- and the rescue then either skips the SIGKILL
    that would have collected its grandchildren, or aims it at a number that
    is no longer ours. `returncode` reads without reaping.
    """
    sent: list[int] = []
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: sent.append(sig))

    class ExitedButUncollected(_FakeProc):
        """The leader is a zombie: gone, but its pid still pinned."""

        def poll(self):
            self.returncode = 0  # as `Popen.poll` does: it collects, and says so
            return self.returncode

    proc = ExitedButUncollected()
    monkeypatch.setattr(
        envs, "_wait_without_reaping", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        envs._kill_tree(proc, grace=0.25)

    assert proc.returncode is None, "the rescue reaped the leader, freeing its group id"
    assert sent == [signal.SIGTERM, signal.SIGKILL], (
        "an uncollected leader still pins its group, so the SIGKILL that "
        "collects its descendants must go out"
    )


def test_the_fallback_keeps_what_its_own_read_found(monkeypatch):
    """The trailing `wait()` of the fallback's `communicate` can raise too.

    When it does the call assembles nothing, and the code returned the
    drain's seed -- taken in the drain's constructor, before this read ran.
    Everything the fallback's read found was dropped. Measured against a real
    `Popen`: `part1` came back while `_fileobj2output` held `part1` and
    `part2`. This is the wedged-in-exit case, a stage that wrote its last
    traceback during the grace and then could not be reaped.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    proc = _WedgedProc()
    proc._fileobj2output = {proc.stdout: [b"part1\n"], proc.stderr: [b"err1\n"]}

    def reads_more_then_the_wait_raises(timeout=None):
        proc._fileobj2output[proc.stdout].append(b"part2\n")
        proc._fileobj2output[proc.stderr].append(b"err2\n")
        raise subprocess.TimeoutExpired("python", timeout)  # carries no output

    proc.communicate = reads_more_then_the_wait_raises
    try:
        out, err = envs._kill_and_drain(proc)
    finally:
        proc.close()
    assert out == "part1\npart2\n", "the fallback's own read must not be dropped"
    assert err == "err1\nerr2\n", "stderr especially: it is where the traceback is"


def test_a_read_failure_makes_no_claim_and_keeps_the_seed(monkeypatch, caplog):
    """The exit where nothing at all was established.

    A read that failed says nothing about who holds the pipes -- our own
    wrappers being unclosed says only that we did not close them -- so this
    exit warns about nothing. What it must still do is hand back the seed:
    the account of the stage is all that is left, and returning two empty
    strings instead was not caught by anything.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    proc = _WedgedProc()
    proc._fileobj2output = {proc.stdout: [b"everything it said"], proc.stderr: []}

    def refuses(timeout=None):
        raise ValueError("I/O operation on closed file")

    proc.communicate = refuses
    try:
        with caplog.at_level("WARNING"):
            out, _ = envs._kill_and_drain(proc)
    finally:
        proc.close()
    assert out == "everything it said", "the seed is the only account left; it must survive"
    assert "are still open" not in caplog.text


def test_one_pipe_refused_gives_up_on_both(monkeypatch):
    """Watching one pipe and not the other is worse than watching neither.

    The drain reads the one it got, and is then overridden by a fallback that
    replaces its result rather than merging -- so the half it read is thrown
    away and only the other half survives. Measured before the fix, with
    stdout registered and stderr refused: stdout came back empty for a stage
    that had written to both. Giving up on both leaves `communicate` able to
    read both.

    That is more than a half-drain returns *when nothing is written during
    the grace* -- which is this test's setup, writers closed first -- and it
    is not free: nothing is emptied while the stage shuts down. Which is why
    the rule is scoped to drains whose text is read; see the next test.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    proc = _WedgedProc(on_stdout=b"OUT\n", on_stderr=b"ERR\n")
    proc.close()  # both at EOF, so `communicate` can read them to the end

    class RefusesStderr(envs.selectors.SelectSelector):
        def register(self, fileobj, events, data=None):
            if fileobj is proc.stderr:
                raise OSError(1, "not permitted")
            return super().register(fileobj, events, data)

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesStderr)

    def reads_what_is_left(timeout=None):
        return (
            "" if proc.stdout.closed else proc.stdout.read(),
            "" if proc.stderr.closed else proc.stderr.read(),
        )

    proc.communicate = reads_what_is_left
    out, err = envs._kill_and_drain(proc)
    assert (out, err) == (
        "OUT\n",
        "ERR\n",
    ), "the drain must not consume a pipe it is about to be overridden for"


def test_interruptions_on_two_pipes_are_counted_apart(monkeypatch):
    """The other half of "per pipe", and the half nothing exercised.

    With one counter shared between the descriptors, interruptions that
    alternate between them add up: five on each makes ten against a bound of
    eight, and a pipe that never saw more than five in a row is given up on.
    Measured: the per-pipe code returns both streams' data; a shared counter
    returns two empty strings.
    """
    proc = _WedgedProc()
    drain = envs._PipeDrain(proc)
    out_fd, err_fd = proc.stdout.fileno(), proc.stderr.fileno()
    real_read = envs.os.read
    left = {out_fd: 5, err_fd: 5}

    def five_each_then_real(fd, size):
        if left.get(fd, 0) > 0:
            left[fd] -= 1
            raise InterruptedError(4, "Interrupted system call")
        return real_read(fd, size)

    os.write(proc._writers[0], b"OUT")
    os.write(proc._writers[1], b"ERR")
    try:
        monkeypatch.setattr(envs.os, "read", five_each_then_real)
        drain.pump(0.3)
    finally:
        monkeypatch.undo()
        text = drain.text()
        drain.close()
        proc.close()

    assert text == (
        "OUT",
        "ERR",
    ), "five interruptions on each pipe were counted together and both pipes were given up on"


def test_the_interrupt_bound_is_per_pipe_and_consecutive(monkeypatch):
    """One counter for two descriptors was wrong in both directions.

    Interruptions alternating between the pipes closed one of them after
    fewer than its own bound; and a steady stream of data on one reset the
    count for the other, so a pipe interrupted forever was never let go.

    "Steady" is the part that has to be tested. A first version wrote one
    byte to stderr, so a shared reset fired at most once and hid inside the
    slack of the bound. Measured with stderr fed on every stdout
    interruption: the fixed code lets stdout go after nine attempts; the
    shared reset made 125,267 in 0.2 s and never let it go.

    And giving up on stdout must leave stderr alone. That is checked by
    writing to stderr *after* stdout was dropped and seeing it arrive --
    an earlier assertion that stderr "was read throughout" was satisfied by
    what arrived before the bound was reached, so a drain that abandoned both
    pipes at once passed.
    """
    proc = _WedgedProc()
    drain = envs._PipeDrain(proc)
    out_fd = proc.stdout.fileno()
    os.write(proc._writers[0], b"x")

    real_read = envs.os.read
    attempts = {"out": 0}

    def stdout_interrupted_while_stderr_talks(fd, size):
        if fd == out_fd:
            attempts["out"] += 1
            os.write(proc._writers[1], b"y")  # stderr always has more to say
            raise InterruptedError(4, "Interrupted system call")
        return real_read(fd, size)

    try:
        monkeypatch.setattr(envs.os, "read", stdout_interrupted_while_stderr_talks)
        drain.pump(0.2)
        monkeypatch.undo()
        let_go_after = attempts["out"]
        before = drain.text()[1]
        os.write(proc._writers[1], b"AFTER")
        drain.pump(0.1)
        after = drain.text()[1]
    finally:
        monkeypatch.undo()
        drain.close()
        proc.close()

    assert let_go_after <= 50, (
        f"stdout was interrupted {let_go_after} times in a row and never let "
        "go: data arriving on stderr is resetting a bound that belongs to stdout"
    )
    assert after == before + "AFTER", (
        "stderr stopped being read when stdout was given up on; the bound "
        "belongs to one pipe and must not close the other"
    )


def test_a_drain_whose_text_is_discarded_keeps_what_did_register(monkeypatch):
    """All-or-nothing is for drains whose output is read.

    Applied everywhere, it cost the grace on the two paths that throw the text
    away -- the signal handler and `run`'s interrupt cleanup -- and bought
    them nothing, since nothing there reads what the fallback would recover.
    Measured with stderr refused and a child writing 200 KB to stdout on
    SIGTERM: 5.0 s, exit -9, handler unfinished, where watching stdout let it
    finish in 0.1 s. The handler is the `kill <litetune>` path.
    """
    proc = _WedgedProc()

    class RefusesStderr(envs.selectors.SelectSelector):
        def register(self, fileobj, events, data=None):
            if fileobj is proc.stderr:
                raise OSError(1, "not permitted")
            return super().register(fileobj, events, data)

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesStderr)
    try:
        kept = envs._PipeDrain(proc, seed=False)
        read = envs._PipeDrain(proc)
        # Written before each pump, and `read` first: one write shared between
        # them let whichever pumped first empty the pipe, so the second's
        # assertion held whatever it did.
        os.write(proc._writers[0], b"x" * 1000)
        read.pump(0.05)
        os.write(proc._writers[0], b"y" * 1000)
        kept.pump(0.05)
        kept_got, read_got = kept.text()[0], read.text()[0]
        kept.close()
        read.close()
    finally:
        proc.close()

    assert kept_got, "a discarding drain must keep emptying the pipe it could register"
    assert not read_got, "a drain whose text is read must still give up on both"


def test_a_discarding_drain_still_gives_a_half_registered_stage_its_grace(tmp_path, monkeypatch):
    """The measurement the scoped all-or-nothing rule rests on, kept in the tree.

    With one pipe refused, a drain that gives up on both empties nothing while
    the stage shuts down, and a stage writing more than a pipe buffer on its
    way out blocks and is SIGKILLed. Measured before the rule was scoped, with
    exactly this child: 5.0 s, exit -9, and the SIGTERM handler never
    finished; after, 0.1 s, exit 0, finished. Asserted on the handler's own
    marker and on the exit, not on timing.
    """
    finished = tmp_path / "handler.finished"
    child = (
        "import signal, sys, time\n"
        "def bye(signum, frame):\n"
        "    sys.stdout.write('X' * 200_000)\n"
        "    sys.stdout.flush()\n"
        f"    open({str(finished)!r}, 'w').write('1')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, bye)\n"
        "print('up', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = os.getpgid(proc.pid)
    try:
        assert proc.stdout.readline() == "up\n"

        class RefusesStderr(envs.selectors.SelectSelector):
            def register(self, fileobj, events, data=None):
                if fileobj is proc.stderr:
                    raise OSError(1, "not permitted")
                return super().register(fileobj, events, data)

        monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesStderr)
        envs._kill_and_drain(proc, pgid, seed=False)

        assert finished.exists(), (
            "the stage was SIGKILLed before its SIGTERM handler finished: with "
            "one pipe refused, nothing emptied the other during the grace"
        )
        assert proc.returncode == 0
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pgid, signal.SIGKILL)


def test_a_drain_that_gave_up_watching_hands_the_question_on(monkeypatch, caplog):
    """After `select` refuses, the drain knows nothing about the pipes.

    It must say so rather than keep a count from before -- `draining` stays
    truthful -- and whoever wants the answer then gets it from the fallback,
    which reads both streams. Here nothing holds them, so the warning must
    stay quiet -- on the evidence of the fallback's read reaching the end,
    which the test checks was actually asked for.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    class RefusesOnSelect(envs.selectors.SelectSelector):
        def select(self, timeout=None):
            raise OSError(10038, "not a socket")

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesOnSelect)

    proc = _WedgedProc()
    drain = envs._PipeDrain(proc, seed=False)
    try:
        assert drain._open == 2, "both pipes register; the refusal comes later"
        drain.pump(0.05)
        assert not drain.watching
        assert not drain.draining, "a drain that gave up must not report pipes it stopped watching"
    finally:
        drain.close()

    # EOF with the read ends still open, so the drain inside `_kill_and_drain`
    # registers both pipes and meets the refusal. Closing the streams
    # instead -- as an earlier version did -- left nothing to register, kept
    # the drain watching, and reached neither `_unwatch` nor the fallback.
    proc.close()
    asked = []
    proc.communicate = lambda timeout=None: asked.append(timeout) or ("", "")
    with caplog.at_level("WARNING"):
        envs._kill_and_drain(proc, seed=False)
    assert asked, "the refusal must hand the question to the fallback"
    assert "are still open" not in caplog.text


def test_a_partial_discarding_drain_warns_for_the_pipe_it_could_not_watch(monkeypatch, caplog):
    """The refused pipe is exactly the one a half-registered drain cannot see.

    With stdout registered and stderr refused, a stray process holding only
    stderr is invisible to `draining`. The drain is not `watching`, so the
    fallback is asked, and its read of both streams is what finds the pipe
    still held.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    proc = _WedgedProc(on_stdout=b"done")

    class RefusesStderr(envs.selectors.SelectSelector):
        def register(self, fileobj, events, data=None):
            if fileobj is proc.stderr:
                raise OSError(1, "not permitted")
            return super().register(fileobj, events, data)

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesStderr)
    os.close(proc._writers[0])  # stdout reaches EOF; stderr's writer stays held
    proc._writers[0] = -1

    def stderr_still_held(timeout=None):
        raise subprocess.TimeoutExpired("python", timeout, output=b"", stderr=b"")

    proc.communicate = stderr_still_held
    try:
        with caplog.at_level("WARNING"):
            envs._kill_and_drain(proc, seed=False)
    finally:
        proc.close()
    assert "are still open" in caplog.text


def test_a_drain_that_hands_over_does_not_wait_first(monkeypatch):
    """A drain partial from its constructor goes straight to the fallback.

    It used to `finish` first, and a watched pipe that something still held
    made that wait its whole budget before the fallback waited again --
    measured, 21.4 s where the other shapes took 11.3 s. The fallback's own
    read replaces whatever `finish` would have gathered, so the first wait
    bought nothing.

    Asserted directly -- `finish` is not asked for any time at all -- rather
    than by timing. A bound of half a 0.6 s budget failed on correct code
    whenever the process was descheduled for 300 ms, and passed for any
    change that merely made the drain stop watching early.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.6)

    proc = _WedgedProc()  # both writers held: `finish` on stdout would wait it out

    class RefusesStderr(envs.selectors.SelectSelector):
        def register(self, fileobj, events, data=None):
            if fileobj is proc.stderr:
                raise OSError(1, "not permitted")
            return super().register(fileobj, events, data)

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesStderr)

    asked_to_finish: list[float] = []
    real_finish = envs._PipeDrain.finish
    monkeypatch.setattr(
        envs._PipeDrain,
        "finish",
        lambda self, budget: (asked_to_finish.append(budget), real_finish(self, budget))[1],
    )
    fallback_asked: list = []
    proc.communicate = lambda timeout=None: fallback_asked.append(timeout) or ("", "")
    try:
        envs._kill_and_drain(proc, seed=False)
    finally:
        proc.close()

    assert not asked_to_finish, "a drain that was never going to be believed waited first"
    assert fallback_asked, "and it still has to hand over to the fallback"


def test_a_pipe_that_fails_to_read_is_let_go(monkeypatch, caplog):
    """An `os.read` that fails for good must end that pipe, not repeat.

    The descriptor stays ready, so leaving it registered hands it straight
    back to the next `select`: measured with a read that raised EIO, 1,048,105
    attempts in one second, and then a warning that the pipes were still held
    -- by nothing but the drain's own refusal to stop asking.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.3)

    proc = _WedgedProc()
    out_fd = proc.stdout.fileno()
    # Both writers closed, so both pipes are readable. Readiness is what makes
    # `select` hand stdout to `_read` at all -- a first version left stdout's
    # writer open with nothing written, the descriptor was never ready, the
    # failing read was never attempted, and the test failed on correct code
    # for a reason that had nothing to do with it.
    proc.close()
    real_read = envs.os.read
    attempts = {"n": 0}

    def eio_on_stdout(fd, size):
        if fd == out_fd:
            attempts["n"] += 1
            raise OSError(5, "Input/output error")
        return real_read(fd, size)

    monkeypatch.setattr(envs.os, "read", eio_on_stdout)
    try:
        with caplog.at_level("WARNING"):
            envs._kill_and_drain(proc)
    finally:
        monkeypatch.undo()
        proc.close()

    # Exactly one: none would mean the failing read was never reached, which
    # is how this test's first version passed on nothing.
    assert attempts["n"] == 1, f"{attempts['n']} reads of a descriptor that fails every time"
    assert "are still open" not in caplog.text


def test_a_discarding_caller_still_hears_about_an_unwatched_pipe(monkeypatch, caplog):
    """A discarding caller asks the fallback too, for the warning's sake.

    An earlier revision skipped the fallback for callers that throw the text
    away, to save the read. But with the child killed and reapable,
    `communicate` only runs to its timeout when something still holds the
    pipes -- which is the one case the leak warning is for -- so there the
    skip saved time only by silencing it. The shape
    here is the Windows one: `select` refuses both pipes, the drain watches
    nothing, and an interrupt that finds the child alive and its pipes still
    held reaches only the child. "Only the child was signalled" is then true,
    and it was never printed.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.CHILD_ONLY)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    class RefusesOnSelect(envs.selectors.SelectSelector):
        def select(self, timeout=None):
            raise OSError(10038, "not a socket")

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesOnSelect)

    proc = _WedgedProc()  # a stray process still holds both write ends

    def still_held(timeout=None):
        raise subprocess.TimeoutExpired("python", timeout, output=b"", stderr=b"")

    proc.communicate = still_held
    try:
        with caplog.at_level("WARNING"):
            envs._kill_and_drain(proc, seed=False)
    finally:
        proc.close()
    assert "only the child was signalled" in caplog.text, (
        "the pipes were held and only the child had been signalled; a "
        "discarding caller has to say so as loudly as any other"
    )


def test_a_selector_that_refuses_every_pipe_falls_back(monkeypatch):
    """Registration refused is not the same as nothing to register.

    Both leave no pipe watched, and only one of them means this host cannot
    watch pipes at all. Deciding from the count alone sent a drain whose
    pipes were merely already closed -- the ordinary state after a
    `communicate` gave up -- to a fallback with no seed to return; deciding
    before the loop sent a host that refused both registrations to the
    watched path, where it read nothing and reported nothing.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    class RefusesRegistration(envs.selectors.SelectSelector):
        def register(self, fileobj, events, data=None):
            raise OSError(1, "not permitted")

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesRegistration)

    proc = _WedgedProc()

    def spoke(timeout=None):
        return "the traceback that explains the timeout\n", ""

    proc.communicate = spoke
    try:
        out, _ = envs._kill_and_drain(proc)
    finally:
        proc.close()
    assert out == "the traceback that explains the timeout\n", (
        "both registrations were refused, so the drain cannot read and the " "fallback has to run"
    )


def test_a_refusal_first_seen_after_the_kill_still_falls_back(monkeypatch):
    """`_kill_tree` returns without pumping on three of its four exits.

    An already-exited leader is the one that matters, because a grandchild
    holding the pipes is the case this whole function exists for -- and on
    that exit the first read of the entire run happens in `finish`. Sampling
    `watching` before it therefore asked the question before anything had
    tried, and a host that cannot watch silently returned two empty strings
    with no warning: a visible crash traded for invisible loss.
    """
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)
    # Returns without ever calling the pump, as the ALREADY_GONE exit does.
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, pump=None, **kwargs: envs.Reach.GROUP)

    class RefusesOnSelect(envs.selectors.SelectSelector):
        def select(self, timeout=None):
            raise OSError(10038, "not a socket")

    monkeypatch.setattr(envs.selectors, "DefaultSelector", RefusesOnSelect)

    proc = _WedgedProc()

    def spoke(timeout=None):
        return "what the stage said before it stopped\n", ""

    proc.communicate = spoke
    try:
        out, _ = envs._kill_and_drain(proc)
    finally:
        proc.close()
    assert (
        out == "what the stage said before it stopped\n"
    ), "the refusal arrived inside finish, and the fallback must still run"


def test_the_fallback_does_not_claim_pipes_it_cannot_see(monkeypatch, caplog):
    """The warning must not assert what the code cannot check.

    Both failure exits used to hard-code "still held". The likeliest way to
    reach the second is a read on an already-closed file, where the claim is
    exactly backwards.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    proc = _WedgedProc(on_stdout=b"said")
    proc.as_communicate_left_it()  # both pipes closed, as after a completed read

    def refuses(timeout=None):
        raise ValueError("I/O operation on closed file")

    proc.communicate = refuses
    with caplog.at_level("WARNING"):
        envs._kill_and_drain(proc)
    assert "are still open" not in caplog.text, (
        "the pipes are closed; saying they are held sends the reader after a "
        "process that does not exist"
    )


def test_the_handler_does_not_copy_what_it_is_about_to_discard(monkeypatch):
    """The signal handler's drain is built to unblock the child, not to keep
    its words -- its own comment says so.

    Seeding it means a `b"".join` over everything the stage has said so far,
    inside a signal handler, before the first SIGTERM goes out -- a second
    full copy of the transcript. What this asserts is the behaviour: that
    the seed is skipped, and that the handler is the caller asking for it.
    The copy is also the one operation in that constructor that can raise
    where the constructor promises not to, so a `MemoryError` there would
    replace the kill entirely.
    """
    proc = _WedgedProc()
    proc._fileobj2output = {proc.stdout: [b"expensive"], proc.stderr: [b"also expensive"]}
    try:
        assert envs._PipeDrain(proc, seed=False).text() == ("", "")
        assert envs._PipeDrain(proc).text() == ("expensive", "also expensive")
    finally:
        proc.close()

    handed: dict = {}
    real_drain = envs._PipeDrain

    def note(proc_, already=(None, None), *, seed=True):
        handed["seed"] = seed
        return real_drain(proc_, already, seed=seed)

    monkeypatch.setattr(envs, "_PipeDrain", note)
    monkeypatch.setattr(envs, "_kill_tree", lambda *a, **k: envs.Reach.GROUP)
    monkeypatch.setattr(signal, "getsignal", lambda sig: signal.SIG_DFL)
    captured: dict = {}
    monkeypatch.setattr(signal, "signal", lambda sig, h: captured.setdefault(sig, h))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    victim = _FakeProc()
    with envs._kill_child_if_we_are_told_to_exit(victim, pgid=4242):
        # The re-delivery is stubbed out, so the handler falls through to the
        # raise it uses when the process is expected to survive the signal.
        with pytest.raises(envs.StageInterrupted):
            captured[signal.SIGTERM](signal.SIGTERM, None)
    assert handed.get("seed") is False, "the handler must not pay for a seed it discards"


def test_the_grace_is_pumped_with_the_one_that_sleeps(monkeypatch):
    """Which of the two the grace is given is not interchangeable.

    `pump` spends its budget; `finish` returns the moment the pipes are done.
    Handing `finish` to the wait turns a loop that sleeps in 50 ms steps into
    one that spins at full speed for the whole grace as soon as the stage has
    stopped writing -- a stage that closes its output early and then takes its
    time shutting down is exactly that case. Asserting the contract at the
    call site rather than the identity of the method, so a differently-named
    replacement still has to sleep.
    """
    handed: list = []
    monkeypatch.setattr(
        envs,
        "_kill_tree",
        lambda proc, pump=None, **kwargs: (handed.append(pump), envs.Reach.GROUP)[1],
    )
    monkeypatch.setattr(envs, "_DRAIN_AFTER_KILL_S", 0.05)

    proc = _WedgedProc()
    proc.as_communicate_left_it()  # nothing left to read, as after a long stage
    try:
        envs._kill_and_drain(proc)
    finally:
        proc.close()

    assert handed and handed[0] is not None, "the grace must be given something to pump"
    started = time.monotonic()
    handed[0](0.3)
    assert (
        time.monotonic() - started >= 0.25
    ), "the grace was handed a pump that returns early, so the wait loop spins"


def test_the_drain_kills_the_group_and_reaps_the_leader(tmp_path):
    """`_kill_and_drain` end to end on a real process tree, not a fake.

    Every other test of this function stubs `_kill_tree` and hands it a fake
    whose `wait` always raises, so nothing observes the kill or the reap. A
    `proc.poll()` inserted before the kill survived the whole suite because
    of that -- and it is the exact thing the module is built to avoid, since
    polling reaps the leader and frees the pid that *is* the group id, after
    which the group kill is aimed at nothing and the grandchild is left
    running.

    The reap is asserted with `os.waitpid(..., WNOHANG)` and not with
    `proc.returncode`. `returncode` is set by any poll from any direction --
    including `Popen.send_signal`'s own -- so asserting on it passed six
    times in eight whether or not anything reaped, which is a test that
    reports the weather. `ECHILD` from `waitpid` means the leader was
    collected by us and by nobody else.

    The reap half of this only bites on Linux, and that is not a flaw to be
    tidied away on a Mac. Here a group whose only member is a zombie answers
    EPERM, so the SIGKILL falls to the direct-child leg, `send_signal` polls
    on its way through and collects the leader before `_reap` is reached --
    measured. On Linux `killpg` reaches that group, nothing polls, and
    deleting `_reap` fails this test: verified in the 3.13 container.
    """
    gone = tmp_path / "leader.gone"
    child = (
        "import subprocess, sys;"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']);"
        f"open({str(gone)!r}, 'w').write('1');"
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
    pid = proc.pid
    try:
        # Waited for through a file the leader writes on its way out, never
        # with `poll`. Polling reaps, and a leader the *test* collected makes
        # `waitpid` answer ECHILD whether or not the code under test ever
        # reaped anything -- which is how the first version of this test
        # passed with `_reap` deleted.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not gone.exists():
            time.sleep(0.02)
        assert gone.exists(), "the leader never got as far as exiting"
        time.sleep(0.2)

        envs._kill_and_drain(proc, pgid)

        time.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)


def test_the_fallback_stays_quiet_when_a_timeout_left_nothing_open(monkeypatch, caplog):
    """The `TimeoutExpired` exit had "still held" written into it too.

    Its sibling test covers pipes that are held; without this one, hard-coding
    the answer the other way round passes both. The warning is the only thing
    that names a stray process while it is still findable, and a warning that
    fires either way names nothing.

    The shape is synthetic and has to be: CPython cannot produce a
    `TimeoutExpired` carrying output with both pipes already closed, because
    once both are closed `_communicate` registers nothing and the only
    timeout left is the trailing `wait()`'s, which carries none. It is built
    by hand because the predicate has to be exercised in both directions and
    this is the only way to reach one of them -- not because it is a state
    the code will meet.
    """
    monkeypatch.setattr(envs, "_kill_tree", lambda proc, **kwargs: envs.Reach.GROUP)
    monkeypatch.setattr(
        envs.selectors, "DefaultSelector", lambda: (_ for _ in ()).throw(OSError("none"))
    )

    proc = _WedgedProc(on_stdout=b"said")
    proc.as_communicate_left_it()

    def timed_out(timeout=None):
        raise subprocess.TimeoutExpired("python", timeout, output=b"said", stderr=b"")

    proc.communicate = timed_out
    with caplog.at_level("WARNING"):
        out, _ = envs._kill_and_drain(proc)
    assert out == "said"
    assert "are still open" not in caplog.text


def test_a_pipe_that_only_ever_answers_eintr_is_let_go(monkeypatch):
    """Retrying an interrupted read is right; retrying it forever is not.

    The descriptor stays ready, so `select` hands it straight back and the
    drain reads at full speed for the rest of its budget. Measured before the
    bound: 372,988 attempts in 0.3 s. The sibling test covers one interruption
    followed by data, which a version with no bound at all also passes.

    What is pinned here is that a bound exists and is under fifty, not that
    it is eight. The number is a judgement rather than a measurement, so
    asserting it exactly would only detect that somebody changed it.
    Consecutive-only and per-pipe are separate properties with tests of
    their own.
    """
    proc = _WedgedProc(on_stdout=b"unreachable")
    drain = envs._PipeDrain(proc)
    attempts = {"n": 0}

    def always_interrupted(fd, size):
        attempts["n"] += 1
        raise InterruptedError(4, "Interrupted system call")

    try:
        monkeypatch.setattr(envs.os, "read", always_interrupted)
        drain.pump(0.2)
    finally:
        monkeypatch.undo()
        drain.close()
        proc.close()

    # A literal ceiling, not one derived from the bound. Compared with the
    # constant itself, any value passed -- a bound of 1000 did.
    assert attempts["n"] <= 50, (
        f"{attempts['n']} attempts in 0.2s: the pipe is never let go, so the "
        "drain spins for the rest of its budget"
    )


def test_an_interrupted_grace_still_kills_the_group(monkeypatch):
    """The grace is the one interruptible part of the kill.

    A second Ctrl-C during cleanup used to take the SIGKILL with it, leaving
    the process group alive -- from the keystroke that was asking for the
    stage to stop sooner. Measured on a real tree before the fix: SIGTERM
    sent, SIGKILL not, the leader unreaped and the group still there.
    """
    sent: list[int] = []
    # 4242 for the child and something else for our own group: `_group_of`
    # refuses to signal a group that is also ours, and a stub answering the
    # same id to both makes it refuse.
    monkeypatch.setattr(envs.os, "getpgid", lambda pid: 4242 if pid else 1)
    monkeypatch.setattr(envs.os, "killpg", lambda pgid, sig: sent.append(sig))
    monkeypatch.setattr(
        envs,
        "_wait_without_reaping",
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        envs._kill_tree(_FakeProc(), grace=0.25)

    assert sent == [
        signal.SIGTERM,
        signal.SIGKILL,
    ], "the interrupt escaped before the SIGKILL, so the group outlives us"


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


def test_a_stage_pipe_is_decoded_as_utf8_whatever_the_host_calls_its_encoding(monkeypatch):
    """The generation comes back through this pipe, so the locale must not shape it.

    `text=True` alone decodes with the locale's encoding -- the ANSI code page
    on Windows -- and `evaluate.LiteRtLmBackend` reads the model's own answer
    off that stream. A character outside the page then arrives mangled, the
    row scores wrong, and the loss is reported as a conversion cost. Pinned as
    the argument rather than as behaviour because there is no CI runner whose
    locale is not already UTF-8: the point is that litetune does not ask.
    """
    started = {}

    def capture(argv, **kwargs):
        started.update(kwargs)
        return _FakePopen()

    monkeypatch.setattr(envs.subprocess, "Popen", capture)
    envs._run_guarded(["python", "-c", "pass"], timeout=5)

    assert started.get("encoding") == "utf-8"
    assert started.get("errors") == "surrogateescape", (
        "a byte that is not UTF-8 must neither raise out of the decode nor be erased: "
        "U+FFFD is a character a model may write, a lone surrogate is not"
    )


def test_a_stage_child_writes_utf8_whatever_the_host_calls_its_encoding():
    """The other half: the child has to be able to write what the parent reads.

    A Python child writing to a pipe encodes with its own locale, and the
    runtime CLI that prints a generation catches the resulting
    `UnicodeEncodeError` itself, prints an apology and exits zero -- which
    litetune would score as the model's answer.
    """
    env = envs._child_env()

    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"].startswith("utf-8")


def test_the_child_s_error_handler_matches_the_parent_s():
    """`utf-8` with no handler means `strict`, where UTF-8 mode means `surrogateescape`.

    Measured on 3.12. Strict would make a child raise on a lone surrogate --
    a path that came back through `os.fsdecode`, a filename in a traceback --
    and the runtime CLI catches that, prints an apology and exits zero, which
    is the failure this pair exists to prevent. The parent reads the pipe with
    `errors="surrogateescape"`, so neither side raises on what the other cannot hold.
    """
    assert envs._child_env()["PYTHONIOENCODING"] == "utf-8:surrogateescape"


def test_a_host_cannot_choose_the_encoding_a_measurement_comes_back_in(monkeypatch):
    """`PYTHONIOENCODING` wins over UTF-8 mode, so setting the mode is not enough.

    Measured on CPython 3.12: `PYTHONUTF8=1 PYTHONIOENCODING=cp1252` gives
    `sys.stdout.encoding == "cp1252"`. A host that carries either variable
    would otherwise decide what a generation looks like by the time litetune
    scores it -- the same argument `_HOST_OVERRIDES` makes about the pins,
    about the one value that cannot be recovered afterwards.
    """
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")

    env = envs._child_env()

    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"].startswith("utf-8")


def test_a_host_value_a_caller_asked_for_is_not_reported_as_dropped(monkeypatch, caplog):
    """The warning names what was taken away, and an override was not taken away.

    The drop warning beside it has excluded `overrides` since it was written,
    for the same reason: a caller who asked for this value gets it, so saying
    it was not passed states the opposite of what happens.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    envs.forget_reported_drops()

    with caplog.at_level("WARNING"):
        env = envs._child_env({"PYTHONIOENCODING": "cp1252"})

    assert env["PYTHONIOENCODING"] == "cp1252"
    assert "not passing PYTHONIOENCODING" not in caplog.text, (
        "the drop warning must not name a key the caller chose; the override warning, "
        "which says something else and is tested next door, may"
    )


def test_the_warning_names_the_host_encoding_it_replaced_once(monkeypatch, caplog):
    """Said out loud, and said once.

    `_child_env` runs per prompt during evaluation, so a per-call warning is
    several hundred identical lines through a progress report -- the reason
    `_REPORTED_DROPS` exists for the drop beside this one.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")
    envs.forget_reported_drops()

    with caplog.at_level("WARNING"):
        envs._child_env()
        envs._child_env()

    assert caplog.text.count("not passing PYTHONIOENCODING") == 1


def test_an_override_that_splits_the_two_sides_is_said_out_loud(caplog):
    """The parent's pipe is UTF-8 whatever the child is told.

    A caller may still choose -- every override may -- but choosing another
    encoding for the child alone is mojibake in a scored generation, and that
    is worth hearing before the numbers rather than from them.
    """
    with caplog.at_level("WARNING"):
        env = envs._child_env({"PYTHONIOENCODING": "cp1252"})

    assert env["PYTHONIOENCODING"] == "cp1252"
    assert "PYTHONIOENCODING=cp1252 was overridden" in caplog.text


def test_an_override_that_keeps_utf8_is_not_warned_about(caplog):
    """Only a different codec splits the two sides.

    `PYTHONIOENCODING` wins over UTF-8 mode, so overriding `PYTHONUTF8` alone
    leaves the child writing UTF-8; naming the same codec another way, or
    choosing another error handler, changes what the child does with text it
    cannot encode and not the bytes the parent reads.
    """
    with caplog.at_level("WARNING"):
        envs._child_env({"PYTHONUTF8": "0"})
        envs._child_env({"PYTHONIOENCODING": "UTF8"})
        envs._child_env({"PYTHONIOENCODING": "utf-8:strict"})

    assert "was overridden" not in caplog.text


def test_a_caller_can_still_choose_the_child_s_encoding():
    """Set, not imposed: `overrides` are applied after, as they are for every other key."""
    env = envs._child_env({"PYTHONUTF8": "0"})

    assert env["PYTHONUTF8"] == "0"


def test_a_stage_pipe_carries_a_non_ascii_generation_whole(monkeypatch, tmp_path):
    """The round trip through a real child, with the host asking for something else.

    Without `PYTHONIOENCODING=ascii` this passes on any machine litetune is
    developed on, fix or no fix: the host is UTF-8 already, so the child
    encodes UTF-8 by default and the parent decodes it. Forcing the host is
    what makes it a test of litetune rather than of the laptop -- reverted,
    the child dies of `UnicodeEncodeError` and this comes back empty.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    answer = "перевод — 変換 — ✅"

    result = envs._run_guarded([sys.executable, "-c", f"print({answer!r})"], timeout=30)

    assert result.returncode == 0, result.stderr[-300:]
    assert result.stdout.strip() == answer


def test_a_pin_with_no_wheel_for_this_machine_is_named_as_that():
    """pip's "no matching distribution" is a different fact from a broken install.

    A pin can exist on PyPI and publish nothing for this platform, Python
    version or architecture. Without this the reader gets 2000 characters of
    pip's resolution trace and has to find the one line that says so.
    """
    message = envs.EXPORT._provisioning_failed(
        "ERROR: No matching distribution found for nosuchpkg==1.0"
    )

    assert "no distribution of nosuchpkg for this machine" in message
    assert "litert-torch#968" not in message, "only the converter gets the converter's answer"


def test_the_missing_converter_is_explained_rather_than_dumped():
    """The one pin this project knows has no wheel for three platforms.

    Read off pip's output rather than `sys.platform`: the day `litert-converter`
    publishes a wheel for a platform it does not build for today, this stops
    firing without anyone editing a table.
    """
    for said in (
        "ERROR: Could not find a version that satisfies the requirement "
        "litert-converter>=0.0.0.dev0 (from versions: none)",
        "ERROR: No matching distribution found for litert_converter",
    ):
        message = envs.EXPORT._provisioning_failed(said)

        assert "convert` cannot run without it" in message
        assert "litert-torch#968" in message
        assert "verify" in message, "the reader is told what still works and how to use it"


def test_the_converter_is_recognised_in_any_spelling_of_its_name():
    """`litert_converter` and `litert.converter` are the same project.

    PyPA's name specification normalises runs of `-`, `_` and `.` to one `-`,
    and pip prints whichever spelling the requirement used -- so a comparison
    against a literal misses two thirds of the time without normalising.
    """
    for spelling in (
        "litert-converter",
        "litert_converter",
        "litert.converter",
        "LiteRT-Converter",
    ):
        message = envs.EXPORT._provisioning_failed(
            f"ERROR: No matching distribution found for {spelling}>=0.0.0.dev0"
        )
        assert "litert-torch#968" in message, spelling


def test_the_converter_message_keeps_what_pip_said_too():
    """The explanation replaces reading pip's trace, not having it.

    The resolver's own output carries the index it consulted and the versions
    it saw, which is what tells a reader whether they were pointed at a
    private mirror.
    """
    message = envs.EXPORT._provisioning_failed(
        "ERROR: No matching distribution found for litert-converter\n"
        "Looking in indexes: https://example.invalid/simple"
    )

    assert "litert-torch#968" in message
    assert "example.invalid" in message


def test_an_unrecognised_pip_failure_still_hands_over_what_pip_said():
    """No pattern matched is not a reason to hide the output."""
    message = envs.EXPORT._provisioning_failed("ERROR: something else went wrong")

    assert "something else went wrong" in message


def test_a_system_package_is_named_as_the_debian_package_it_is():
    """`libvulkan1` is a Debian name, printed on every platform.

    Unqualified, it reads as an instruction a macOS or Windows reader cannot
    follow, in the message that is already telling them their install failed.
    """
    message = envs.EXPORT._provisioning_failed("ERROR: something else went wrong")

    assert "Debian package(s) libvulkan1" in message
