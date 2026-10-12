"""Fakes for the verify tests.

No network, no accelerator, no model load, no subprocess. `FakeBackend`
satisfies `evaluate.GenerationBackend` structurally -- it inherits nothing,
which is the reason that interface is a Protocol: a mock would accept a typo in
a method name and this will not. The one subprocess is `run_compose_script`,
which the convert tests' toolchain fakes share: it runs litetune's own script in
the host interpreter, against a fake `transformers`.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from packaging.tags import mac_platforms

from litetune.evaluate import (
    BACKEND_OBSERVED,
    GREEDY,
    UNKNOWN_BACKEND,
    DecodeConfig,
    Generation,
    GenerationBackend,
)
from litetune.prompt_mode import PromptMode
from litetune.wheels import Host


def call_text(name: str, **args: str) -> str:
    """Render FunctionGemma's wire format, the way a model would emit it."""
    body = ",".join(f"{k}:<escape>{v}<escape>" for k, v in args.items())
    return f"call:{name}{{{body}}}"


@dataclass
class FakeBackend:
    """Canned generations. `texts` is either one per prompt or a single repeat."""

    model: str = "fake-model"
    texts: Sequence[str] = ()
    returncode: int = 0
    harness_error: str | None = None
    prompt_mode: PromptMode = PromptMode.PRERENDERED
    decode: DecodeConfig = GREEDY
    # Part of the backend contract, so the double states it. It used to be
    # absent, and a `.get(..., True)` in the production code recorded every fake
    # measurement as enforcing decode parameters it never received.
    decode_enforced: bool = True
    # Likewise part of the contract: a double that did not say this would be
    # scored down whichever path the production code happens to check for.
    scores_structurally: bool = False
    name: str = "fake"
    # Settable, because a double that can only say `UNKNOWN_BACKEND` says the
    # same word the production code falls back to, and a test
    # asserting the recorded backend then passes against code that never reads
    # the manifest at all.
    backend: str = UNKNOWN_BACKEND
    # Settable too, and `False` by default for the same reason `backend` is
    # `UNKNOWN_BACKEND`: a double read nothing back off any device. A test that
    # needs the other state sets it, so that state is visible in the test
    # rather than assumed by the double.
    backend_observed: bool = False
    prompts_seen: list[list[str]] = field(default_factory=list)

    @property
    def model_ref(self) -> str:
        return self.model

    def describe(self) -> dict:
        # Defaults to `UNKNOWN_BACKEND`, not to a third word for it: this double
        # runs on no hardware, which is the state that constant names.
        # `evaluate.device_mismatch` reads this key and suppresses on
        # `UNKNOWN_BACKEND`, so two doubles given *different* real-looking
        # backends would manufacture a hardware difference between two fakes
        # that never touched hardware. The same value on both is harmless.
        #
        # Several subclasses override `describe()` with a literal dict, some of
        # them nested inside test functions where a module-level search misses
        # them. On those, setting `backend=` is silently a no-op.
        return {
            "engine": "fake",
            "backend": self.backend,
            BACKEND_OBSERVED: self.backend_observed,
        }

    def generate(self, prompts: Sequence[str], events=None) -> list[Generation]:
        self.prompts_seen.append(list(prompts))
        if len(self.texts) not in (1, len(prompts)):
            raise ValueError(f"fake has {len(self.texts)} texts for {len(prompts)} prompts")
        return [
            Generation(
                index=i,
                prompt=prompt,
                text=self.texts[i] if len(self.texts) == len(prompts) else self.texts[0],
                returncode=None if self.harness_error else self.returncode,
                harness_error=self.harness_error,
            )
            for i, prompt in enumerate(prompts)
        ]


@pytest.fixture(autouse=True)
def _forget_reported_drops():
    """Clear the once-per-run warning state around every test.

    `envs._REPORTED_DROPS` is module level, so a test that triggers the
    "not passing X" warning silences it for every test that runs after it in
    the same process -- including any test whose subject *is* that warning.
    """
    from litetune import envs

    envs.forget_reported_drops()
    yield
    envs.forget_reported_drops()


@pytest.fixture(autouse=True)
def _isolated_env_cache(monkeypatch, tmp_path):
    """Point the stage-environment cache at this test's own directory.

    `StageEnv.ready` reads a marker file under `LITETUNE_ENV_DIR`, and code now
    branches on it: `HuggingFaceBackend._ensure_env` runs the device probe only
    against a ready environment. Without this, whether a given test probes
    depends on whether the developer running it happens to have a matching
    environment provisioned in `~/.cache/litetune/envs` -- which is not a
    property of the code under test, and on a machine that has one would send a
    real `python -c "import torch"` into a test that faked everything else.
    Tests that want a ready environment write the marker themselves; this only
    guarantees they start from a machine with none.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "litetune-envs"))


@pytest.fixture(autouse=True)
def _no_host_device_settings(monkeypatch):
    """Keep the operator's device settings out of every test, and sysctl too.

    `LITETUNE_DEVICE` and the three MPS variables are read from the
    environment `tune` and `verify` run in, so a developer who exported one
    would change what the suite asserts. `devices.read_sysctl` would start a
    real `sysctl` and answer about this machine; the fake answers the same
    figures everywhere. A test that wants other figures passes its own.
    """
    from litetune import devices

    for name in (devices.DEVICE_VARIABLE, *devices.MPS_VARIABLES):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(devices, "read_sysctl", fake_sysctl(FAKE_SYSCTL))


# What `fake_sysctl` answers by default: a Mac at normal memory pressure with
# 16 GiB available in 16 KiB pages -- 6 free and 10 pageable external -- and
# a speculative and a purgeable GiB the sum must not count.
FAKE_SYSCTL = {
    "hw.pagesize": "16384",
    "vm.page_free_count": str(6 * 65536),
    "vm.page_speculative_count": str(65536),
    "vm.page_pageable_external_count": str(10 * 65536),
    "vm.page_purgeable_count": str(65536),
    "kern.memorystatus_vm_pressure_level": "1",
    "vm.swapusage": "total = 0.00M  used = 0.00M  free = 0.00M",
    "machdep.cpu.brand_string": "Apple M-test",
}


def sysctl_available(available_bytes: int, pressure: int = 1) -> dict[str, str]:
    """`FAKE_SYSCTL` with `available_bytes` available, all of it free pages."""
    pages = available_bytes // int(FAKE_SYSCTL["hw.pagesize"])
    return dict(
        FAKE_SYSCTL,
        **{
            "vm.page_free_count": str(pages),
            "vm.page_speculative_count": "0",
            "vm.page_pageable_external_count": "0",
            "vm.page_purgeable_count": "0",
            "kern.memorystatus_vm_pressure_level": str(pressure),
        },
    )


def fake_sysctl(values: dict[str, str]):
    """A `devices.Sysctl` that answers from `values` and raises for anything else."""
    from litetune.devices import HostReadError

    def read(name: str) -> str:
        if name not in values:
            raise HostReadError(f"sysctl {name} exited 1: unknown oid {name!r}")
        return values[name]

    return read


# Every field `envs._DEVICE_PROBE_CODE` prints, in one place. `tune`'s
# `FakeTrainer`, the probe fakes in `test_evaluate.py` and `test_verify.py`, and
# `test_cli.py`'s `FakeToolchain` all answer the probe through this, and
# `test_envs.py` pins it against what the probe source itself prints -- so a
# field added to the probe reaches every fake at once instead of one at a time.
PROBE_FIELDS = (
    "device",
    "cuda_build",
    "device_count",
    "mps_built",
    "mps_available",
    "mps_recommended_max_memory",
    "os",
    "os_version",
    "os_version_source",
    "machine",
)


def probe_answer(device: str | None = "cpu", **fields: Any) -> str:
    """One line of JSON, the way the device probe prints it.

    A "cpu" or "cuda" answer comes from a Linux x86-64 machine without MPS
    unless a field says otherwise, which is what every probe fake answered
    before MPS was asked about. An "mps" answer comes from an arm64 Mac with
    16 GiB of recommended working set.
    """
    answer: dict[str, Any] = {
        "device": device,
        "cuda_build": None,
        "device_count": 0,
        "mps_built": False,
        "mps_available": False,
        "mps_recommended_max_memory": None,
        "os": "Linux",
        "os_version": "6.8.0-test",
        "os_version_source": "platform.release()",
        "machine": "x86_64",
    }
    if device == "mps":
        answer.update(
            mps_built=True,
            mps_available=True,
            mps_recommended_max_memory=16 * 1024**3,
            os="Darwin",
            os_version="15.0",
            os_version_source="platform.mac_ver()",
            machine="arm64",
        )
    unknown = set(fields) - set(PROBE_FIELDS)
    if unknown:
        raise TypeError(f"the probe prints no {sorted(unknown)}")
    answer.update(fields)
    return json.dumps(answer)


@pytest.fixture
def windows_text_writes(monkeypatch):
    """Make text writes translate `\n` the way Windows does.

    `open()` with `newline=None` writes `os.linesep` for every `\n`, so on
    Windows a file written in text mode holds CRLF. Every digest litetune
    publishes over a file it wrote itself is therefore platform-dependent
    unless the write says `newline=""`. There is no Windows runner to prove
    that on -- CI is ubuntu-only -- so this reproduces the one behaviour that
    matters, which is the same thing the `newline` argument of `open()`
    documents.

    Deliberately not selective: on Windows *every* text write translates, the
    fixtures in a test included, which is what makes a digest taken over one
    of those files differ from the digest taken over its text. Both routes are
    patched for the same reason -- a review found that patching
    `Path.write_text` alone let the same bug back in through `open(..., "w")`
    with both CRLF tests still green.
    """
    real_write_text = Path.write_text
    real_open = builtins.open

    def translating(self, data, encoding=None, errors=None, newline=None):
        if newline is None:
            data = data.replace("\n", "\r\n")
            newline = ""
        return real_write_text(self, data, encoding=encoding, errors=errors, newline=newline)

    def translating_open(file, mode="r", *args, **kwargs):
        writing_text = "b" not in mode and any(m in mode for m in "wax+")
        if writing_text and kwargs.get("newline") is None:
            kwargs["newline"] = "\r\n"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", translating)
    monkeypatch.setattr(builtins, "open", translating_open)


@pytest.fixture
def write_split(tmp_path: Path):
    """Write held-out JSONL and return its path."""

    def _write(rows: Sequence[dict], name: str = "heldout.jsonl") -> Path:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    return _write


def labelled_rows(n: int) -> list[dict]:
    """`n` single-call examples of the shape the README's dataset uses."""
    return [
        {
            "prompt": f"set the background to colour {i}",
            "target": {"name": "change_background_color", "args": {"color": f"c{i}"}},
        }
        for i in range(n)
    ]


def correct_texts(rows: Sequence[dict]) -> list[str]:
    return [call_text(r["target"]["name"], **r["target"]["args"]) for r in rows]


@dataclass
class _FakeCuda:
    """`torch.cuda`, reduced to the one method `training_device`/`generation_device`
    call."""

    available: bool

    def is_available(self) -> bool:
        return self.available


@dataclass
class FakeTorch:
    """`torch`, reduced to the one attribute either device function reads."""

    cuda: _FakeCuda


def fake_torch(cuda: bool = False) -> FakeTorch:
    """A `torch` double whose `.cuda.is_available()` answers `cuda`.

    One shape, shared by `test_tune.py` and `test_evaluate.py`, and the
    parameter is the point of it. `training_device` and `generation_device`
    both fall back to `torch.cuda.is_available()`, and a double that can only
    say "no GPU" cannot tell that fallback from a hardcoded `"cpu"` -- every
    mutant of it still answers "cpu" against such a double. Being able to say
    "there is a GPU" is what makes those two lines testable at all.
    """
    return FakeTorch(cuda=_FakeCuda(cuda))


# A conformance assertion, not a runtime one: this is what makes the claim above
# ("a mock would accept a typo in a method name and this will not") true. A fake
# that drifts from the Protocol -- a renamed method, a forgotten
# `decode_enforced` -- fails the type check rather than the measurement.
_CONFORMS: GenerationBackend = FakeBackend()


def mark_provisioned(env) -> Path:
    """Leave a `StageEnv` looking the way a finished provision leaves it.

    Both halves, because `StageEnv.ready` is both: the marker says the install
    finished, the interpreter says the tree it finished into is still there. A
    real cache turned up a directory holding only the marker -- `provision`
    short-circuited on it, `run` reached a python that did not exist, and the
    error named the toolchain instead of the empty directory.

    Nine places across seven test files used to write the marker alone, which is
    the old and wrong definition of ready taught nine times. One helper so that
    the next change to what "provisioned" means has one place to land.
    """
    env.python.parent.mkdir(parents=True, exist_ok=True)
    env.python.touch()
    (env.path / ".litetune-ready").write_text(env.identity, encoding="utf-8")
    return env.path


def _manylinux(arch: str, glibc_minor: int) -> list[str]:
    """The platform tags a glibc 2.`glibc_minor` Linux on `arch` accepts, most specific first.

    In the order `packaging._manylinux.platform_tags` yields them (packaging
    26.3): each glibc from this one down to 2.17 -- 2.5 on x86 -- with the legacy
    alias beside the version it names, then the bare `linux_` tag pip adds.
    """
    oldest = 4 if arch in ("x86_64", "i686") else 16
    legacy = {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}
    tags: list[str] = []
    for minor in range(glibc_minor, oldest, -1):
        tags.append(f"manylinux_2_{minor}_{arch}")
        if minor in legacy:
            tags.append(f"{legacy[minor]}_{arch}")
    return [*tags, f"linux_{arch}"]


# Machines nobody has here, by name, as the platform tags pip would accept on
# each: `packaging.tags.mac_platforms` for macOS, `_manylinux` above and
# `_musllinux.platform_tags`'s order for Linux, the bare tag for Windows.
MACHINES: dict[str, list[str]] = {
    "linux_x86_64": _manylinux("x86_64", 35),
    "linux_aarch64": _manylinux("aarch64", 35),
    "linux_x86_64_glibc_2_16": _manylinux("x86_64", 16),
    "linux_x86_64_glibc_2_17": _manylinux("x86_64", 17),
    "linux_x86_64_glibc_2_26": _manylinux("x86_64", 26),
    "linux_x86_64_glibc_2_27": _manylinux("x86_64", 27),
    "musl_x86_64": [f"musllinux_1_{m}_x86_64" for m in (2, 1, 0)] + ["linux_x86_64"],
    "macosx_arm64": list(mac_platforms((14, 0), "arm64")),
    "macosx_11_arm64": list(mac_platforms((11, 0), "arm64")),
    "macosx_12_arm64": list(mac_platforms((12, 0), "arm64")),
    "macosx_x86_64": list(mac_platforms((14, 0), "x86_64")),
    "win_amd64": ["win_amd64"],
    "win_arm64": ["win_arm64"],
    "win32": ["win32"],
}


def machine(name: str, python: tuple[int, int]) -> Host:
    """A regular CPython `python` on the machine `MACHINES` names."""
    return Host.of(MACHINES[name], python)


@pytest.fixture(autouse=True)
def _missing_wheels_refused(monkeypatch):
    """The shell running the suite does not decide which tests refuse.

    `LITETUNE_ALLOW_MISSING_WHEELS=1` turns the wheel table's refusal into a
    note, so a developer who set it for their own installs would otherwise see
    every refusal test fail. Tests about the opt-out set it themselves.
    """
    from litetune import envs

    monkeypatch.delenv(envs.ALLOW_MISSING_WHEELS, raising=False)


def buildable_here(monkeypatch) -> None:
    """Let a faked `provision` stand for the whole of provisioning.

    `export` asks `StageEnv.refusal` before anything else, and that reads this
    machine's platform and Python against PyPI's file lists: on Windows
    `litert-converter` stops it, on 3.13 `numpy==2.0.2` does. A test that fakes
    the install is about what follows one, so a fake environment is one this
    machine can build, wherever the suite runs.
    """
    from litetune.envs import StageEnv

    monkeypatch.setattr(StageEnv, "refusal", lambda self: None)


# What `export._COMPOSE_SCRIPT` imports in place of the export environment's
# transformers: an `AutoTokenizer` whose tokenizer carries `_TEMPLATE` as its
# `chat_template`, writes each path it is asked to load to `_LOADS`, and renders
# with `apply_chat_template` the way transformers 5.16.1 does for a string
# template and `tokenize=False`: Jinja2's `ImmutableSandboxedEnvironment` with
# `trim_blocks`, `lstrip_blocks` and the loop-controls extension, its own
# `tojson`, `raise_exception` and `strftime_now`, and the messages handed to
# `render` as given (`utils/chat_template_utils.py:478-495` and `:581-587`;
# `tokenization_utils_base.py:3085-3117`). The `{% generation %}` tracker it
# also registers is left out: no template rendered here uses it. The template
# also sees the tokenizer's special tokens (`:3107`), Qwen's here: `eos_token`
# and `pad_token` as both checkpoints' `tokenizer_config.json` set them, BOS and
# UNK unset. `_RAISE` makes `apply_chat_template` raise as transformers does
# without a usable jinja2 (`chat_template_utils.py:426-428`).
_FAKE_TRANSFORMERS = """
import json
from datetime import datetime

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )


class _Tokenizer:
    special_tokens_map = {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}

    def apply_chat_template(
        self, conversation, chat_template=None, tokenize=True, add_generation_prompt=False
    ):
        assert tokenize is False, "only the text path is faked"
        if _RAISE:
            raise ImportError(_RAISE)
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        template = chat_template if chat_template is not None else self.chat_template
        return env.from_string(template).render(
            messages=conversation,
            tools=None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **self.special_tokens_map,
        )


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path):
        with open(_LOADS, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(path) + "\\n")
        tokenizer = _Tokenizer()
        tokenizer.chat_template = _TEMPLATE
        return tokenizer
"""

# And `huggingface_hub` 1.33.0, for the `chat_template.json` a Hub id falls
# back to: `_HUB` maps a file name to its text, and any other is missing, which
# `hf_hub_download` raises as `RemoteEntryNotFoundError`; `_HUB_FAULT` names the
# error it raises instead when the Hub cannot be reached, as it raises
# `LocalEntryNotFoundError` (`file_download.py:1961`), or any other.
_FAKE_HUB = """
import os

from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError


def hf_hub_download(repo_id, filename):
    if _HUB_FAULT == "LocalEntryNotFoundError":
        raise LocalEntryNotFoundError("cannot reach the Hub and the file is not cached")
    if _HUB_FAULT:
        raise ConnectionResetError(_HUB_FAULT)
    if filename not in _HUB:
        raise RemoteEntryNotFoundError(f"404: {filename} is not in {repo_id}")
    path = os.path.join(_DIR, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_HUB[filename])
    return path
"""

_FAKE_HUB_ERRORS = """
class EntryNotFoundError(Exception):
    pass


class RemoteEntryNotFoundError(OSError, EntryNotFoundError):
    pass


class LocalEntryNotFoundError(FileNotFoundError, EntryNotFoundError):
    pass
"""


def run_compose_script(
    args: Sequence[str],
    timeout: int,
    chat_template: object,
    version: str = "5.16.1",
    hub_files: dict[str, str] | None = None,
    hub_fault: str = "",
    render_raises: str = "",
) -> tuple[subprocess.CompletedProcess, list[str]]:
    """`export._COMPOSE_SCRIPT`, run for real with the host interpreter.

    The script's own reading of the template -- a dict, an empty one, the
    `chat_template.json` fallback, the bytes it writes, the probe it renders --
    is exercised; only `transformers` and `huggingface_hub` are fakes, through
    packages of those names first on `PYTHONPATH`. Rendering is real Jinja2,
    which the dev extras pin at the export environment's version. Returns the
    completed process and the paths `AutoTokenizer.from_pretrained` was given,
    in order. The toolchain fakes in the test files each call this for the one
    command it answers.
    """
    script, spec, result = str(args[1]), str(args[2]), str(args[3])
    shim = Path(spec).parent / "shim"
    package = shim / "transformers"
    package.mkdir(parents=True, exist_ok=True)
    loads = shim / "loads.jsonl"
    (package / "__init__.py").write_text(
        f"__version__ = {version!r}\n_TEMPLATE = {chat_template!r}\n_LOADS = {str(loads)!r}\n"
        f"_RAISE = {render_raises!r}\n" + _FAKE_TRANSFORMERS,
        encoding="utf-8",
    )
    hub = shim / "huggingface_hub"
    hub.mkdir(exist_ok=True)
    (hub / "errors.py").write_text(_FAKE_HUB_ERRORS, encoding="utf-8")
    (hub / "__init__.py").write_text(
        f"_HUB = {dict(hub_files or {})!r}\n_DIR = {str(shim)!r}\n_HUB_FAULT = {hub_fault!r}\n"
        + _FAKE_HUB,
        encoding="utf-8",
    )
    # The caller's path after the shim's, so a Jinja2 that is not installed in
    # this interpreter can be supplied on PYTHONPATH as well.
    path = os.pathsep.join(filter(None, [str(shim), os.environ.get("PYTHONPATH", "")]))
    # The environment `StageEnv.run` gives a stage, with the shim on the path:
    # in UTF-8 mode among the rest, so the script reads the checkpoint's files
    # as it does in the export environment. With the host's own environment it
    # read them in the ANSI code page on Windows, where a byte that is not
    # UTF-8 decodes to something -- the first windows-latest run.
    from litetune.envs import _child_env

    proc = subprocess.run(
        [sys.executable, script, spec, result],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env({"PYTHONPATH": path}),
    )
    seen = loads.read_text(encoding="utf-8").splitlines() if loads.is_file() else []
    return proc, [json.loads(line) for line in seen]
