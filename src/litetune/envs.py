"""Per-stage Python environments.

Training and conversion cannot share an interpreter. `torch` + `transformers`
against `litert-torch` + `numpy<2.1` conflict irreconcilably, which is why
Google's own FunctionGemma notebooks tell the reader to restart the runtime
between steps, and why one of them opens with "uses ONLY mediapipe (no
ai-edge-torch) to avoid conflicts".

So the user installs one package and litetune provisions the rest: one venv per
stage, created on first use and cached. The same separation the cloud pipeline
achieves with two container images, on a laptop.

Every requirement is pinned. This is not caution for its own sake — the same
unpinned install command produced a working export on 2026-08-26 and
`AttributeError: pad_token` on 2026-08-30. Pinning the environment's *identity*
is useless if its *definition* resolves differently on different days, so
`StageEnv` refuses to build from an unpinned requirement.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
import venv
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


class UnpinnedRequirement(ValueError):
    """Raised when an environment definition would float."""


def env_cache_root() -> Path:
    """Where provisioned environments live. Public because `litetune env` shows it.

    A path a user is told to inspect or clear should have a name in the API, not
    only a shape in a docstring.
    """
    return _cache_root()


def _cache_root() -> Path:
    override = os.environ.get("LITETUNE_ENV_DIR")
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "litetune" / "envs"


# A requirement is pinned if it fixes an exact version. `>=` is not a pin: it
# admits tomorrow's release, which is precisely the failure mode above.
def _is_pinned(requirement: str) -> bool:
    return "==" in requirement or requirement.startswith(("http://", "https://", "file://"))


# A pip install of torch is minutes, not seconds; a stalled one is forever.
logger = logging.getLogger(__name__)

PROVISION_TIMEOUT_S = 1800

# The stage environments are built from the *host* interpreter, so its version
# is theirs -- and each has its own ceiling, set by whichever of its pins stops
# publishing wheels first. Past that, pip falls back to a source build and fails
# after minutes of compiler output that never names the version.
#
# One shared ceiling was tried and was wrong: it refused `train` on 3.13 for a
# `numpy==2.0.2` limit that `train` does not have, in a project whose README
# said `tune` works there. The constraint belongs to the pin, so it is recorded
# with the pin.
STAGE_PYTHON_MIN = (3, 10)


def _try_lock(handle) -> bool | None:
    """One non-blocking attempt. True taken, False held elsewhere, None no locking.

    Imported here rather than at module scope: `fcntl` does not exist on
    Windows, and an unconditional import made `import litetune.envs` -- and so
    the whole CLI -- fail there. This module has always had `os.name == "nt"`
    branches, so that platform is in scope.
    """
    try:
        import fcntl

        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        except OSError:
            # Advisory locking is unavailable on some network filesystems.
            return None
    except ImportError:
        pass
    if sys.platform == "win32":
        # A platform check rather than a try/except, so the type checker narrows
        # instead of reporting `msvcrt` has no attributes on a POSIX host.
        try:
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    return None


def _refuse_unsupported_interpreter(env: StageEnv) -> None:
    """Say which interpreter is needed, before pip spends minutes not saying it.

    Named per environment, because the ceilings differ: `numpy==2.0.2` stops at
    cp312 and `torch==2.5.1` at cp313, so `convert` and `verify` refuse a
    3.13 host that `tune` is fine on. A message naming the wrong pin sends the
    reader to fix the wrong thing.
    """
    current = sys.version_info[:2]
    if current < STAGE_PYTHON_MIN:
        raise RuntimeError(
            f"environment {env.name!r} needs at least python"
            f"{STAGE_PYTHON_MIN[0]}.{STAGE_PYTHON_MIN[1]}; this is python"
            f"{current[0]}.{current[1]}"
        )
    if env.python_ceiling is None or current <= env.python_ceiling:
        return
    ceiling = env.python_ceiling
    raise RuntimeError(
        f"environment {env.name!r} cannot be provisioned on python"
        f"{current[0]}.{current[1]}: it pins {env.ceiling_pin}, which publishes wheels up to "
        f"python{ceiling[0]}.{ceiling[1]}, and stage environments are built from the "
        "interpreter running litetune. Run litetune under one of those (Colab's default "
        "works), or install it into a virtualenv on one."
    )


@contextlib.contextmanager
def _provision_lock(path: Path, timeout: float = PROVISION_TIMEOUT_S):
    """Serialise provisioning of one environment across processes.

    A file lock rather than a directory sentinel: a sentinel left behind by a
    killed process blocks every later run, while a file lock is released by the
    kernel when the holder dies.

    Where no lock can be taken this proceeds unserialised -- which is what this
    did before the lock existed, and is better than refusing to run -- but says
    so, because "the environments raced" is otherwise indistinguishable from a
    corrupt install.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.parent / f"{path.name}.lock"
    deadline = time.monotonic() + timeout
    with open(lock, "w") as handle:
        while True:
            taken = _try_lock(handle)
            if taken is True:
                break
            if taken is None:
                logger.warning(
                    "no advisory locking for %s; concurrent provisioning of %s is unserialised",
                    lock,
                    path.name,
                )
                break
            if time.monotonic() >= deadline:
                # Bounded, because the first version blocked forever on
                # `LOCK_EX` while the docstring advertised a timeout that only
                # covered pip. A holder that hangs must not hang everyone.
                raise RuntimeError(
                    f"waited {timeout}s for another process to finish provisioning "
                    f"{path.name} (lock: {lock}); it is still holding the lock"
                )
            time.sleep(0.5)
        yield


@dataclass(frozen=True)
class CachedEnv:
    """One directory in the environment cache, and what is known about it."""

    path: Path
    bytes: int
    ready: bool
    # The stage it belongs to, when the name still matches one this version
    # defines. A directory whose identity no longer matches any stage is not
    # junk -- it is what an older litetune, or a different interpreter, built --
    # so it is reported as unclaimed rather than as an error.
    stage: str | None

    @property
    def claimed(self) -> bool:
        return self.stage is not None


def _tree_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            # A file that vanished mid-walk is not a reason to refuse a total;
            # it is a reason for the total to be approximate, which it always is.
            continue
    return total


def cached_environments() -> list[CachedEnv]:
    """Everything in the cache, largest first.

    There is no other way to find out. The path is not in the documentation, the
    sizes differ by an order of magnitude between stages, and a failed provision
    leaves a directory behind that looks exactly like a working one from the
    outside.
    """
    root = _cache_root()
    if not root.is_dir():
        return []
    claimed = {env.path.name: env.name for env in (RUNTIME, TRAIN, EXPORT)}
    out = [
        CachedEnv(
            path=child,
            bytes=_tree_bytes(child),
            ready=(child / ".litetune-ready").exists(),
            stage=claimed.get(child.name),
        )
        for child in sorted(root.iterdir())
        if child.is_dir()
    ]
    return sorted(out, key=lambda e: e.bytes, reverse=True)


def remove_cached(entries: Sequence[CachedEnv]) -> tuple[int, list[str]]:
    """Delete these directories. Returns the bytes freed and what would not go.

    Failures are collected rather than raised: removing four of five caches and
    saying which one resisted is more useful than stopping at the first.
    """
    freed = 0
    failures: list[str] = []
    for entry in entries:
        try:
            shutil.rmtree(entry.path)
            freed += entry.bytes
        except OSError as exc:
            failures.append(f"{entry.path}: {exc}")
    return freed, failures


@dataclass(frozen=True)
class StageEnv:
    """A named, pinned environment that stage commands run inside."""

    name: str
    requirements: tuple[str, ...]
    # System packages this environment needs that pip cannot provide. Recorded
    # so a missing one produces a named diagnosis rather than a dlopen error.
    system_requirements: tuple[str, ...] = field(default=())
    # The highest Python this environment's pins publish wheels for, and which
    # pin sets it. `None` means no pin here has a known ceiling. Recorded rather
    # than derived: pip only tells you by failing to build from source, minutes
    # in, without naming the version.
    python_ceiling: tuple[int, int] | None = None
    ceiling_pin: str = ""

    def __post_init__(self) -> None:
        unpinned = [r for r in self.requirements if not _is_pinned(r)]
        if unpinned:
            raise UnpinnedRequirement(
                f"environment {self.name!r} has unpinned requirements: {unpinned}. "
                "Pin them with '==': an unchanged definition that resolves "
                "differently over time silently changes pipeline behaviour."
            )

    @property
    def identity(self) -> str:
        """Content hash of the definition. Participates in stage cache keys."""
        payload = "\n".join((sys.version.split()[0], *sorted(self.requirements)))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def path(self) -> Path:
        return _cache_root() / f"{self.name}-{self.identity}"

    @property
    def python(self) -> Path:
        bindir = "Scripts" if os.name == "nt" else "bin"
        return self.path / bindir / "python"

    @property
    def ready(self) -> bool:
        return (self.path / ".litetune-ready").exists()

    def provision(
        self, events=None, force: bool = False, timeout: int = PROVISION_TIMEOUT_S
    ) -> Path:
        """Create the environment if absent. Idempotent, and safe to race.

        Two invocations sharing one cache directory used to be able to see the
        same unready environment and both proceed -- one `rmtree`-ing the venv
        the other was installing into. The lock makes the second wait and then
        find the first one's work, which is what "idempotent" was already
        claiming.

        `timeout` is not optional in practice: this was the only unbounded
        external call in the package, and a stalled `pip install torch` hung
        forever with no output.
        """
        if self.ready and not force:
            return self.path
        _refuse_unsupported_interpreter(self)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One budget, not two. Passing `timeout` to both the wait and the
        # install let a caller who waited the full 1800s then get another
        # 1800s of pip -- an advertised half-hour limit taking an hour.
        deadline = time.monotonic() + timeout
        with _provision_lock(self.path, timeout):
            # Re-check inside the lock: the holder we waited for may have just
            # built exactly what we came to build.
            if self.ready and not force:
                return self.path
            if events:
                events.note(
                    f"provisioning environment {self.name!r} ({len(self.requirements)} pinned)"
                )
            # A virtualenv cannot be built somewhere and moved here. `venv`
            # writes the absolute path of the interpreter *at creation time*
            # into the shebang of every console script it installs, and nothing
            # rewrites them on a rename. Building in `.incoming` and renaming
            # into place therefore produced an environment whose `pip`,
            # `litert-torch` and `litert-lm` all died with `bad interpreter` --
            # naming a path truncated at the kernel's 127-byte shebang limit,
            # so the message pointed at a path that had never existed. Every
            # `convert` and every `verify` failed, and because `StageEnv.run`
            # prefers a console script whenever the file is present, the
            # `python -m` fallback below never got a chance.
            #
            # So the new environment is built at its final path, and it is the
            # *old* one that steps aside. Same invariant, opposite direction:
            # nothing is removed until a replacement exists, and the
            # replacement is born where it will live.
            previous = self.path.parent / f".{self.path.name}.previous"
            moved_aside = False
            if self.path.exists():
                shutil.rmtree(previous, ignore_errors=True)
                self.path.rename(previous)
                moved_aside = True
            elif previous.exists():
                # A provision killed between the two renames left the working
                # environment here. Reclaim it rather than leaving gigabytes
                # nobody looks at; if this build succeeds it is replaced
                # anyway, and if it fails the restore below wants it.
                moved_aside = True
            try:
                # `EnvBuilder.create` spawns `ensurepip` as its own subprocess
                # and takes no timeout, so the budget covers what follows it and
                # not this. Named rather than left as a silent hole.
                venv.EnvBuilder(with_pip=True, clear=True).create(self.path)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"provisioning environment {self.name!r} exhausted its {timeout}s "
                        "budget building the virtualenv; nothing was installed, and any "
                        "existing environment is untouched"
                    )
                bindir = "Scripts" if os.name == "nt" else "bin"
                python = self.path / bindir / ("python.exe" if os.name == "nt" else "python")
                cmd = [str(python), "-m", "pip", "install", "--quiet", *self.requirements]
                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=remaining,
                        # The timeout message names a prompt as a cause; this is
                        # what stops one from being waited on at all.
                        stdin=subprocess.DEVNULL,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"provisioning environment {self.name!r} exceeded its {timeout}s budget "
                        "(shared between waiting for another provisioner and the install "
                        "itself); the install may be waiting on the network"
                    ) from None
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"could not provision environment {self.name!r} on "
                        f"{sys.platform}/python{sys.version_info.major}."
                        f"{sys.version_info.minor}. It pins {', '.join(self.requirements)}"
                        + (
                            " and needs the system package(s) "
                            + ", ".join(self.system_requirements)
                            if self.system_requirements
                            else ""
                        )
                        + ". The export and runtime toolchains are published for Linux; on "
                        "other platforms the install fails here rather than later. pip "
                        f"said:\n{proc.stderr[-2000:]}"
                    )
                # Written last, and the only thing `ready` consults: an
                # environment interrupted anywhere above this line has no
                # marker, so the next run rebuilds it instead of trusting a
                # half-installed tree.
                (self.path / ".litetune-ready").write_text(self.identity)
            except BaseException:
                # Put the working environment back. This covers the timeout and
                # pip-failure paths above, and Ctrl-C, which is why it catches
                # BaseException rather than Exception.
                if moved_aside:
                    shutil.rmtree(self.path, ignore_errors=True)
                    try:
                        previous.rename(self.path)
                    except OSError as restore_failure:
                        raise RuntimeError(
                            f"provisioning environment {self.name!r} failed, and the working "
                            f"environment could not be moved back from {previous}. It is intact "
                            "there; move it into place by hand, or delete it and let the next "
                            f"run rebuild. Restore failed with: {restore_failure}"
                        ) from restore_failure
                raise
            else:
                if moved_aside:
                    shutil.rmtree(previous, ignore_errors=True)
        if events:
            events.note(f"environment {self.name!r} ready at {self.path}")
        return self.path

    def run(self, args: list[str], timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        """Run a console script or module inside this environment.

        Returns the completed process rather than raising on non-zero: callers
        decide whether a non-zero exit is a failed check or an unperformed one,
        and that distinction is the whole point of litetune.checks.
        """
        bindir = "Scripts" if os.name == "nt" else "bin"
        exe = self.path / bindir / args[0]
        argv = [str(exe), *args[1:]] if exe.exists() else [str(self.python), "-m", *args]
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# The environments litetune actually uses.
#
# Versions are the ones every measurement in the README was produced with. They
# are part of the result, not maintenance trivia: changing them re-runs the
# suite rather than being a routine bump.
# ---------------------------------------------------------------------------

RUNTIME = StageEnv(
    name="runtime",
    python_ceiling=(3, 12),
    ceiling_pin="numpy==2.0.2",
    requirements=(
        "litert-lm==0.16.1",
        "numpy==2.0.2",  # last of the 2.0 line; `<2.1` is a bound, not a pin
    ),
    # litert-lm dlopen()s a native library that links vulkan unconditionally,
    # even for the CPU backend. Without it every invocation -- `--help`
    # included -- dies in under a second.
    system_requirements=("libvulkan1",),
)

EXPORT = StageEnv(
    name="export",
    python_ceiling=(3, 12),
    ceiling_pin="numpy==2.0.2",
    requirements=(
        "litert-torch-nightly==0.10.0.dev20260826",
        "litert-lm==0.16.1",
        "numpy==2.0.2",  # last of the 2.0 line; `<2.1` is a bound, not a pin
    ),
    system_requirements=("libvulkan1",),
)

# transformers 5.16.1, not the 4.57.3 the reference notebooks pin.
#
# That pin was taken for FunctionGemma reproducibility and quietly became a
# ceiling on which models exist at all: every 4.x release from 4.55.0 to 4.57.6
# crashes loading a Gemma 4 or Qwen3.5 tokenizer, because `extra_special_tokens`
# ships as a list where 4.x calls `.keys()` on it. Fixed in 5.0.0, never
# backported. Each family's `min_transformers` in `models.py` encodes the rule;
# leaving the pin below
# it would have made litetune fail its own check.
#
# Raising it changes the measured baseline, so it was not done on reasoning. A
# six-model probe re-ran the three previously measured families on 5.16.1 —
# gemma3_text, qwen2 and qwen3 all stayed alive on both recipes with
# byte-identical artifacts — and Qwen3.5, which had failed at tokenizer load,
# passed the full LoRA→merge→export→liveness path.
TRAIN = StageEnv(
    name="train",
    python_ceiling=(3, 13),
    ceiling_pin="torch==2.5.1",
    requirements=(
        "torch==2.5.1",
        "transformers==5.16.1",
        "peft==0.20.0",
        "sentencepiece==0.2.0",
    ),
)
