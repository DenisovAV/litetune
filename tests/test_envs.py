"""Environment definitions must not be able to float.

An unchanged Dockerfile produced a working export on 2026-08-26 and
`AttributeError: pad_token` on 2026-08-30, because the requirement was
unpinned. The constructor refuses that shape so the failure cannot recur
silently.
"""

import os
import pathlib
import types

import pytest

from litetune import envs
from litetune.envs import EXPORT, RUNTIME, TRAIN, StageEnv, UnpinnedRequirement


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
    monkeypatch.setattr(
        venv.EnvBuilder,
        "create",
        lambda self, path: pathlib.Path(path).mkdir(parents=True, exist_ok=True),
    )
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
    (big / ".litetune-ready").write_text("deadbeef", encoding="utf-8")

    half = tmp_path / "runtime-cafe"
    half.mkdir()
    (half / "payload").write_bytes(b"x" * 10)

    entries = envs.cached_environments()

    # Largest first: the reason to look is usually to find what to delete.
    assert [e.path.name for e in entries] == ["export-deadbeef", "runtime-cafe"]
    assert entries[0].bytes > 5000 and entries[0].ready
    assert entries[1].bytes == 10 and not entries[1].ready


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
