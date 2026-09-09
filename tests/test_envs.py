"""Environment definitions must not be able to float.

An unchanged Dockerfile produced a working export on 2026-08-26 and
`AttributeError: pad_token` on 2026-08-30, because the requirement was
unpinned. The constructor refuses that shape so the failure cannot recur
silently.
"""

import os
import pathlib
import subprocess
import sys
import types

import pytest

from litetune import envs
from litetune.envs import EXPORT, RUNTIME, TRAIN, StageEnv, UnpinnedRequirement


def _fake_venv(self, path):
    """Stand in for `EnvBuilder.create` -- including the interpreter.

    A bare `mkdir` was enough while `ready` read only the marker file. It is
    not a virtualenv, and a fake that is missing the one thing `ready` now
    looks for would assert the opposite of the invariant under test.
    """
    root = pathlib.Path(path)
    bindir = root / ("Scripts" if os.name == "nt" else "bin")
    bindir.mkdir(parents=True, exist_ok=True)
    (bindir / "python").touch()


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

    `StageEnv.run` calls `subprocess.run(..., text=True)`, which decodes
    stdout eagerly, and a stray non-UTF-8 byte -- a CUDA, driver or vendor
    banner ahead of the probe's own line, the same class of weird environment
    the last-non-empty-line rule exists to survive -- raised
    `UnicodeDecodeError`. That is a `ValueError`, not the `OSError` this
    function used to catch, so it escaped `resolve_device` entirely and ended
    a `tune` or `verify` run with a traceback instead of an unanswered probe.

    No mock of `StageEnv.run`: the fake "python" below is a real executable at
    the real path `run` looks for, so this exercises the actual
    `subprocess.run(..., errors="replace")` call the fix lives in, not a
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


def test_the_child_process_really_does_not_see_it(tmp_path, monkeypatch):
    """The helper being right is not the claim; the subprocess is.

    Every other test here calls `_child_env` directly, so all of them would
    still pass if `run` stopped calling it. This one starts a real process
    through `StageEnv.run` and asks it what it inherited.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "/planted")
    env = StageEnv(name="realchild", requirements=("pyyaml==6.0.2",))
    env.python.parent.mkdir(parents=True, exist_ok=True)
    env.python.symlink_to(sys.executable)

    seen = env.run(
        ["python", "-c", "import os; print(os.environ.get('PYTHONPATH', '<unset>'))"], timeout=60
    )
    assert seen.stdout.strip() == "<unset>"

    # And a caller's own override still reaches it, through the same path.
    override = env.run(
        ["python", "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES', '<none>'))"],
        timeout=60,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert override.stdout.strip() == ""


def test_a_signal_still_reads_as_a_signal_through_the_new_run(tmp_path, monkeypatch):
    """`Popen` replaced `subprocess.run`, and the whole check model rides on this.

    `exits.read_returncode` tells "killed" from "exited non-zero" by the sign of
    the return code, and that is what keeps a SIGKILLed stage `unchecked`
    rather than `failed`. A rewrite of `run` that returned an unsigned status
    would collapse the two silently.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name="signalled", requirements=("pyyaml==6.0.2",))
    env.python.parent.mkdir(parents=True, exist_ok=True)
    env.python.symlink_to(sys.executable)

    proc = env.run(
        ["python", "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"], timeout=60
    )
    assert proc.returncode == -9
    assert not envs.read_returncode(proc.returncode).conclusive


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
def test_a_timeout_kills_what_the_stage_spawned(tmp_path, monkeypatch):
    """`subprocess.run(timeout=)` kills the direct child and nothing below it.

    `tune` waits six hours; when that fires it records an honest "not checked"
    and the run *continues*, so an abandoned torch process keeps its memory.
    The next stage is SIGKILLed and `exits` reads that as the OOM killer --
    blaming the machine for litetune's own orphan.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path))
    env = StageEnv(name="grandchild", requirements=("pyyaml==6.0.2",))
    env.python.parent.mkdir(parents=True)
    env.python.symlink_to(sys.executable)

    pidfile = tmp_path / "grandchild.pid"
    child = (
        "import subprocess, sys, time;"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid));"
        "time.sleep(60)"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        env.run(["python", "-c", child], timeout=3)

    grandchild = int(pidfile.read_text())
    # Signal 0 asks "is it there" without sending anything.
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
