"""Which machines a stage environment's pins can be installed on, as PyPI lists them.

`StageEnv.provision` builds an environment with pip, from the interpreter
running litetune. Where a pin publishes no wheel for that interpreter's
platform or Python version, pip finds nothing to install, or falls back to a
source build and fails minutes later without naming the version -- so the
environments are checked before pip is asked, against the files each pin
actually publishes.

**The facts are data, with their source.** Each `Release` below is what PyPI's
JSON API returned for one pinned release, read on 2026-10-12: its file names
(`urls[].filename`) copied verbatim, the URL they were read from, and the `==`
requirements its metadata names without a marker (`info.requires_dist`), which
pip can only satisfy with exactly that version. Nothing here is inferred from a
project's other releases: a wheel is in the list or it is not.

**Keyed by the pinned release, which is what makes a table tolerable.** #45 read
the converter's absence off pip's output rather than the platform, so that the
day `litert-converter` publishes a Windows wheel nothing would go on refusing.
A table keyed by project would. This one is keyed by the exact release an
environment pins: a wheel published under a later version changes nothing
until a pin moves to it, and moving a pin fails
`test_every_pin_has_its_wheels_on_record` until the new release is read in
here. What it cannot see is a file uploaded to a pinned release after it was
read here, and for that #45's reading of pip's output still stands behind it.

**What is not read, so that nothing is refused on a guess.** Requirements the
pins leave open or put behind a marker, which pip resolves itself and whose
failures reach the reader through pip's own output; the glibc and macOS
versions inside a platform tag; musl; interpreters other than CPython, which
are judged as if they were one; and any index but PyPI -- this is PyPI's file
list, not `PIP_EXTRA_INDEX_URL`'s.
"""

from __future__ import annotations

import platform as platform_module
import re
import struct
import sysconfig
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# A wheel built for no platform in particular: `py3-none-any`.
ANY = "any"


def canonical(name: str) -> str:
    """A distribution name in the one spelling that compares equal.

    `litert_converter`, `litert.converter` and `LiteRT-Converter` are the same
    project: PyPA's name specification normalises runs of `-`, `_` and `.` to a
    single `-` and lowercases. pip prints whichever spelling the requirement
    used, and a wheel's file name spells it with `_`, so a comparison against a
    literal has to normalise or it silently misses.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


# Platform tags, read the way PEP 425 and its successors spell them, and kept in
# that spelling: a refusal that says `win_arm64` names what a reader will find
# in the file list on PyPI. Each manylinux generation is the same architecture
# as far as this question goes; a universal2 macOS wheel is both of its halves.
_MANYLINUX = re.compile(r"manylinux(?:1|2010|2014|_\d+_\d+)_(\w+)")
_MUSLLINUX = re.compile(r"musllinux_\d+_\d+_(\w+)")
_MACOSX = re.compile(r"macosx_\d+_\d+_(\w+)")
_MACOSX_FORMATS = {"arm64": ("arm64",), "x86_64": ("x86_64",), "universal2": ("arm64", "x86_64")}
_ANDROID = re.compile(r"android_\d+_(\w+)")
_WINDOWS = frozenset({"win32", "win_amd64", "win_arm64"})


def _platforms(tag: str) -> frozenset[str]:
    """The platforms one wheel's platform tag installs on.

    Raises on a tag this does not know, rather than reading it as nothing: a pin
    moved to a release with an unfamiliar tag must fail the test that reads
    every file on record, not refuse a machine the wheel was built for.
    """
    found: set[str] = set()
    for part in tag.split("."):
        if part == ANY or part in _WINDOWS:
            found.add(part)
        elif m := _MANYLINUX.fullmatch(part):
            found.add(f"linux_{m.group(1)}")
        elif m := _MUSLLINUX.fullmatch(part):
            found.add(f"musllinux_{m.group(1)}")
        elif (m := _MACOSX.fullmatch(part)) and m.group(1) in _MACOSX_FORMATS:
            found.update(f"macosx_{arch}" for arch in _MACOSX_FORMATS[m.group(1)])
        elif m := _ANDROID.fullmatch(part):
            found.add(f"android_{m.group(1)}")
        else:
            raise ValueError(f"unrecognised platform tag {part!r} in {tag!r}")
    return frozenset(found)


@dataclass(frozen=True)
class _Wheel:
    """What one wheel file installs on."""

    platforms: frozenset[str]
    # CPython 3 minor versions it is built for. Empty with `every_python` set
    # for a `py3` wheel, which any Python 3 installs: `litert-lm-api` ships its
    # native library that way, as `py3-none-<platform>`.
    minors: frozenset[int]
    every_python: bool


_CPYTHON = re.compile(r"cp3(\d+)")
# The ABIs a regular CPython build installs: none, its own, and its own with the
# `m` suffix older releases carried. `abi3` and the free-threaded `t` builds are
# refused below until a pin needs them and somebody reads what they mean.
_ABI = re.compile(r"none|cp3\d+m?")


def _wheel(filename: str) -> _Wheel | None:
    """Read a wheel file name's tags. None for a wheel no CPython installs."""
    if not filename.endswith(".whl"):
        raise ValueError(f"{filename} is not a wheel")
    python, abi, tag = filename[: -len(".whl")].split("-")[-3:]
    minors: set[int] = set()
    every_python = False
    cpython = False
    for part in python.split("."):
        if part == "py3":
            every_python = cpython = True
        elif m := _CPYTHON.fullmatch(part):
            minors.add(int(m.group(1)))
            cpython = True
        elif part == "py2" or part.startswith("pp"):
            # Python 2, or PyPy: nothing a stage environment is built from.
            continue
        else:
            raise ValueError(f"unrecognised python tag {part!r} in {filename}")
    if not cpython:
        return None
    if not _ABI.fullmatch(abi):
        raise ValueError(f"unrecognised ABI tag {abi!r} in {filename}")
    return _Wheel(_platforms(tag), frozenset(minors), every_python)


@dataclass(frozen=True)
class Release:
    """One pinned release, as PyPI's JSON API listed it."""

    requirement: str
    # Where the files below were read: `https://pypi.org/pypi/<name>/<version>/json`.
    source: str
    files: tuple[str, ...]
    # The `==` requirements without a marker in this release's own metadata.
    requires: tuple[str, ...] = ()

    @property
    def sdist(self) -> bool:
        """Whether a source distribution was published, which pip would try to build."""
        return any(not name.endswith(".whl") for name in self.files)

    def _wheels(self) -> list[_Wheel]:
        return [w for name in self.files if name.endswith(".whl") if (w := _wheel(name))]

    def platforms(self) -> list[str]:
        """Every platform some wheel of this release installs on, sorted."""
        return sorted({p for w in self._wheels() for p in w.platforms})

    def minors_on(self, platform: str) -> frozenset[int] | None:
        """The Python 3 minors with a wheel for `platform`; None for every one of them."""
        minors: set[int] = set()
        for w in self._wheels():
            if platform in w.platforms or ANY in w.platforms:
                if w.every_python:
                    return None
                minors |= w.minors
        return frozenset(minors)


@dataclass(frozen=True)
class Gap:
    """A pin with no wheel for a host: for its platform at all, or for its Python."""

    release: Release
    platform: str
    # The Python 3 minors `release` has wheels for on `platform`. Empty when it
    # has none there for any Python, which is a different sentence.
    minors: frozenset[int]
    # The pin that brought this one in, when it is not one of the environment's own.
    required_by: str | None = None

    @property
    def platform_missing(self) -> bool:
        return not self.minors

    def describe(self) -> str:
        """The gap as a clause naming the pin, what is missing, and where that was read."""
        name = self.release.requirement
        if self.required_by:
            name = f"{name} (required by {self.required_by})"
        if self.platform_missing:
            builds = (
                "only a source distribution, which pip would try to build"
                if self.release.sdist
                else "no source distribution either, so pip has nothing to install"
            )
            return (
                f"{name} publishes no wheel for {self.platform} and {builds} -- its wheels are "
                f"for {', '.join(self.release.platforms())} ({self.release.source})"
            )
        return (
            f"{name}, which publishes wheels for {self.platform} for {_span(self.minors)} "
            f"only ({self.release.source})"
        )


def _span(minors: frozenset[int]) -> str:
    ordered = sorted(minors)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"python3.{ordered[0]} to python3.{ordered[-1]}"
    return ", ".join(f"python3.{m}" for m in ordered)


def _key(requirement: str) -> str | None:
    """`name==version` in canonical spelling, or None for a requirement that is not one."""
    name, sep, version = requirement.partition("==")
    if not sep:
        return None
    return f"{canonical(name.strip())}=={version.strip()}"


def gaps(
    requirements: Iterable[str],
    platform: str,
    minor: int,
    published: Mapping[str, Release] | None = None,
) -> list[Gap]:
    """Every pin with no wheel for `platform` and CPython 3.`minor`.

    Follows each release's own exact requirements, so a pin whose wheel exists
    but whose `==` dependency's does not is caught and named as that
    dependency. A requirement with no release on record is not judged at all:
    this says what PyPI's lists establish and nothing past them.
    """
    table = PUBLISHED if published is None else published
    found: list[Gap] = []
    seen: set[str] = set()
    pending: list[tuple[str, str | None]] = [(r, None) for r in requirements]
    while pending:
        requirement, parent = pending.pop(0)
        key = _key(requirement)
        if key is None or key in seen or key not in table:
            continue
        seen.add(key)
        release = table[key]
        pending.extend((r, release.requirement) for r in release.requires)
        minors = release.minors_on(platform)
        if minors is None or minor in minors:
            continue
        found.append(Gap(release, platform, minors, required_by=parent))
    return found


def platform_key(system: str, sysconfig_platform: str, mac_arch: str, is_32bit: bool) -> str:
    """The platform a wheel must be built for to install on an interpreter.

    The same inputs pip reads, in the same order (`packaging.tags.platform_tags`
    as pip 26.2.1 vendors it): on macOS the CPU from `platform.mac_ver()`,
    `i386` for a 32-bit interpreter; on Linux `sysconfig.get_platform()`, moved
    to its 32-bit name for a 32-bit interpreter on a 64-bit kernel, which is
    what puts 32-bit Python on x86-64 Linux on `linux_i686`; everywhere else
    `sysconfig.get_platform()` itself, which on Windows is `win_amd64`,
    `win_arm64` or `win32` for the interpreter's build, whatever the machine.
    `.` and `-` become `_`, as in a wheel's tag.
    """
    if system == "Darwin":
        if is_32bit:
            return "macosx_ppc" if mac_arch.startswith("ppc") else "macosx_i386"
        return f"macosx_{mac_arch}"
    key = re.sub(r"[-. ]", "_", sysconfig_platform)
    if system == "Linux" and is_32bit:
        key = {"linux_x86_64": "linux_i686", "linux_aarch64": "linux_armv8l"}.get(key, key)
    return key


def host_platform() -> str:
    """`platform_key` for the interpreter running this."""
    return platform_key(
        platform_module.system(),
        sysconfig.get_platform(),
        platform_module.mac_ver()[2],
        struct.calcsize("P") == 4,
    )


# ---------------------------------------------------------------------------
# What PyPI lists for every pin in `envs.RUNTIME`, `envs.EXPORT` and
# `envs.TRAIN`, and for every `==` requirement behind one. Read from the URL in
# each entry on 2026-10-12. To read a new pin in: its `urls[].filename`, sorted,
# and the entries of `info.requires_dist` that are `name==version` with no `;`.
# ---------------------------------------------------------------------------

_RELEASES = (
    Release(
        "litert-lm==0.18.0",
        "https://pypi.org/pypi/litert-lm/0.18.0/json",
        requires=("litert-lm-api==0.18.0", "litert-lm-builder==0.18.0"),
        files=("litert_lm-0.18.0-py3-none-any.whl",),
    ),
    Release(
        "litert-lm-api==0.18.0",
        "https://pypi.org/pypi/litert-lm-api/0.18.0/json",
        files=(
            "litert_lm_api-0.18.0-py3-none-android_23_arm64_v8a.whl",
            "litert_lm_api-0.18.0-py3-none-android_23_x86_64.whl",
            "litert_lm_api-0.18.0-py3-none-macosx_12_0_arm64.whl",
            "litert_lm_api-0.18.0-py3-none-manylinux_2_27_aarch64.whl",
            "litert_lm_api-0.18.0-py3-none-manylinux_2_27_x86_64.whl",
            "litert_lm_api-0.18.0-py3-none-win_amd64.whl",
        ),
    ),
    Release(
        "litert-lm-builder==0.18.0",
        "https://pypi.org/pypi/litert-lm-builder/0.18.0/json",
        files=("litert_lm_builder-0.18.0-py3-none-any.whl",),
    ),
    Release(
        "numpy==2.0.2",
        "https://pypi.org/pypi/numpy/2.0.2/json",
        files=(
            "numpy-2.0.2-cp310-cp310-macosx_10_9_x86_64.whl",
            "numpy-2.0.2-cp310-cp310-macosx_11_0_arm64.whl",
            "numpy-2.0.2-cp310-cp310-macosx_14_0_arm64.whl",
            "numpy-2.0.2-cp310-cp310-macosx_14_0_x86_64.whl",
            "numpy-2.0.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "numpy-2.0.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "numpy-2.0.2-cp310-cp310-musllinux_1_1_x86_64.whl",
            "numpy-2.0.2-cp310-cp310-musllinux_1_2_aarch64.whl",
            "numpy-2.0.2-cp310-cp310-win32.whl",
            "numpy-2.0.2-cp310-cp310-win_amd64.whl",
            "numpy-2.0.2-cp311-cp311-macosx_10_9_x86_64.whl",
            "numpy-2.0.2-cp311-cp311-macosx_11_0_arm64.whl",
            "numpy-2.0.2-cp311-cp311-macosx_14_0_arm64.whl",
            "numpy-2.0.2-cp311-cp311-macosx_14_0_x86_64.whl",
            "numpy-2.0.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "numpy-2.0.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "numpy-2.0.2-cp311-cp311-musllinux_1_1_x86_64.whl",
            "numpy-2.0.2-cp311-cp311-musllinux_1_2_aarch64.whl",
            "numpy-2.0.2-cp311-cp311-win32.whl",
            "numpy-2.0.2-cp311-cp311-win_amd64.whl",
            "numpy-2.0.2-cp312-cp312-macosx_10_9_x86_64.whl",
            "numpy-2.0.2-cp312-cp312-macosx_11_0_arm64.whl",
            "numpy-2.0.2-cp312-cp312-macosx_14_0_arm64.whl",
            "numpy-2.0.2-cp312-cp312-macosx_14_0_x86_64.whl",
            "numpy-2.0.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "numpy-2.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "numpy-2.0.2-cp312-cp312-musllinux_1_1_x86_64.whl",
            "numpy-2.0.2-cp312-cp312-musllinux_1_2_aarch64.whl",
            "numpy-2.0.2-cp312-cp312-win32.whl",
            "numpy-2.0.2-cp312-cp312-win_amd64.whl",
            "numpy-2.0.2-cp39-cp39-macosx_10_9_x86_64.whl",
            "numpy-2.0.2-cp39-cp39-macosx_11_0_arm64.whl",
            "numpy-2.0.2-cp39-cp39-macosx_14_0_arm64.whl",
            "numpy-2.0.2-cp39-cp39-macosx_14_0_x86_64.whl",
            "numpy-2.0.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "numpy-2.0.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "numpy-2.0.2-cp39-cp39-musllinux_1_1_x86_64.whl",
            "numpy-2.0.2-cp39-cp39-musllinux_1_2_aarch64.whl",
            "numpy-2.0.2-cp39-cp39-win32.whl",
            "numpy-2.0.2-cp39-cp39-win_amd64.whl",
            "numpy-2.0.2-pp39-pypy39_pp73-macosx_10_9_x86_64.whl",
            "numpy-2.0.2-pp39-pypy39_pp73-macosx_14_0_x86_64.whl",
            "numpy-2.0.2-pp39-pypy39_pp73-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "numpy-2.0.2-pp39-pypy39_pp73-win_amd64.whl",
            "numpy-2.0.2.tar.gz",
        ),
    ),
    Release(
        "litert-torch-nightly==0.10.0.dev20260926",
        "https://pypi.org/pypi/litert-torch-nightly/0.10.0.dev20260926/json",
        files=("litert_torch_nightly-0.10.0.dev20260926-py3-none-any.whl",),
    ),
    Release(
        "ai-edge-litert-nightly==2.3.0.dev20260925",
        "https://pypi.org/pypi/ai-edge-litert-nightly/2.3.0.dev20260925/json",
        files=(
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp310-cp310-macosx_12_0_arm64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp310-cp310-manylinux_2_27_aarch64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp310-cp310-manylinux_2_27_x86_64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp310-cp310-win_amd64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp311-cp311-macosx_12_0_arm64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp311-cp311-manylinux_2_27_aarch64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp311-cp311-manylinux_2_27_x86_64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp311-cp311-win_amd64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp312-cp312-macosx_12_0_arm64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp312-cp312-manylinux_2_27_aarch64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp312-cp312-manylinux_2_27_x86_64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp312-cp312-win_amd64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp313-cp313-macosx_12_0_arm64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp313-cp313-manylinux_2_27_aarch64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp313-cp313-manylinux_2_27_x86_64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp313-cp313-win_amd64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp314-cp314-macosx_12_0_arm64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp314-cp314-manylinux_2_27_aarch64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp314-cp314-manylinux_2_27_x86_64.whl",
            "ai_edge_litert_nightly-2.3.0.dev20260925-cp314-cp314-win_amd64.whl",
        ),
    ),
    Release(
        "ai-edge-quantizer-nightly==0.10.0.dev20260926",
        "https://pypi.org/pypi/ai-edge-quantizer-nightly/0.10.0.dev20260926/json",
        files=(
            "ai_edge_quantizer_nightly-0.10.0.dev20260926-py3-none-any.whl",
            "ai_edge_quantizer_nightly-0.10.0.dev20260926.tar.gz",
        ),
    ),
    Release(
        "litert-converter==0.4.0",
        "https://pypi.org/pypi/litert-converter/0.4.0/json",
        requires=("xdsl==0.28.0",),
        files=(
            "litert_converter-0.4.0-cp310-cp310-macosx_12_0_arm64.whl",
            "litert_converter-0.4.0-cp310-cp310-manylinux_2_27_x86_64.whl",
            "litert_converter-0.4.0-cp311-cp311-macosx_12_0_arm64.whl",
            "litert_converter-0.4.0-cp311-cp311-manylinux_2_27_x86_64.whl",
            "litert_converter-0.4.0-cp312-cp312-macosx_12_0_arm64.whl",
            "litert_converter-0.4.0-cp312-cp312-manylinux_2_27_x86_64.whl",
            "litert_converter-0.4.0-cp313-cp313-macosx_12_0_arm64.whl",
            "litert_converter-0.4.0-cp313-cp313-manylinux_2_27_x86_64.whl",
            "litert_converter-0.4.0-cp314-cp314-macosx_12_0_arm64.whl",
            "litert_converter-0.4.0-cp314-cp314-manylinux_2_27_x86_64.whl",
        ),
    ),
    Release(
        "xdsl==0.28.0",
        "https://pypi.org/pypi/xdsl/0.28.0/json",
        requires=("ordered-set==4.1.0",),
        files=(
            "xdsl-0.28.0-py3-none-any.whl",
            "xdsl-0.28.0.tar.gz",
        ),
    ),
    Release(
        "ordered-set==4.1.0",
        "https://pypi.org/pypi/ordered-set/4.1.0/json",
        files=(
            "ordered-set-4.1.0.tar.gz",
            "ordered_set-4.1.0-py3-none-any.whl",
        ),
    ),
    Release(
        "transformers==5.16.1",
        "https://pypi.org/pypi/transformers/5.16.1/json",
        files=(
            "transformers-5.16.1-py3-none-any.whl",
            "transformers-5.16.1.tar.gz",
        ),
    ),
    Release(
        "torch==2.5.1",
        "https://pypi.org/pypi/torch/2.5.1/json",
        files=(
            "torch-2.5.1-cp310-cp310-manylinux1_x86_64.whl",
            "torch-2.5.1-cp310-cp310-manylinux2014_aarch64.whl",
            "torch-2.5.1-cp310-cp310-win_amd64.whl",
            "torch-2.5.1-cp310-none-macosx_11_0_arm64.whl",
            "torch-2.5.1-cp311-cp311-manylinux1_x86_64.whl",
            "torch-2.5.1-cp311-cp311-manylinux2014_aarch64.whl",
            "torch-2.5.1-cp311-cp311-win_amd64.whl",
            "torch-2.5.1-cp311-none-macosx_11_0_arm64.whl",
            "torch-2.5.1-cp312-cp312-manylinux1_x86_64.whl",
            "torch-2.5.1-cp312-cp312-manylinux2014_aarch64.whl",
            "torch-2.5.1-cp312-cp312-win_amd64.whl",
            "torch-2.5.1-cp312-none-macosx_11_0_arm64.whl",
            "torch-2.5.1-cp313-cp313-manylinux1_x86_64.whl",
            "torch-2.5.1-cp39-cp39-manylinux1_x86_64.whl",
            "torch-2.5.1-cp39-cp39-manylinux2014_aarch64.whl",
            "torch-2.5.1-cp39-cp39-win_amd64.whl",
            "torch-2.5.1-cp39-none-macosx_11_0_arm64.whl",
        ),
    ),
    Release(
        "peft==0.20.0",
        "https://pypi.org/pypi/peft/0.20.0/json",
        files=(
            "peft-0.20.0-py3-none-any.whl",
            "peft-0.20.0.tar.gz",
        ),
    ),
    Release(
        "sentencepiece==0.2.0",
        "https://pypi.org/pypi/sentencepiece/0.2.0/json",
        files=(
            "sentencepiece-0.2.0-cp310-cp310-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.0-cp310-cp310-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp310-cp310-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp310-cp310-win32.whl",
            "sentencepiece-0.2.0-cp310-cp310-win_amd64.whl",
            "sentencepiece-0.2.0-cp311-cp311-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.0-cp311-cp311-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp311-cp311-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp311-cp311-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp311-cp311-win32.whl",
            "sentencepiece-0.2.0-cp311-cp311-win_amd64.whl",
            "sentencepiece-0.2.0-cp312-cp312-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.0-cp312-cp312-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp312-cp312-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp312-cp312-win32.whl",
            "sentencepiece-0.2.0-cp312-cp312-win_amd64.whl",
            "sentencepiece-0.2.0-cp36-cp36m-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp36-cp36m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp36-cp36m-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp36-cp36m-win32.whl",
            "sentencepiece-0.2.0-cp36-cp36m-win_amd64.whl",
            "sentencepiece-0.2.0-cp37-cp37m-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp37-cp37m-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp37-cp37m-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp37-cp37m-win32.whl",
            "sentencepiece-0.2.0-cp37-cp37m-win_amd64.whl",
            "sentencepiece-0.2.0-cp38-cp38-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.0-cp38-cp38-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp38-cp38-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp38-cp38-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp38-cp38-win32.whl",
            "sentencepiece-0.2.0-cp38-cp38-win_amd64.whl",
            "sentencepiece-0.2.0-cp39-cp39-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.0-cp39-cp39-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.0-cp39-cp39-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            "sentencepiece-0.2.0-cp39-cp39-manylinux_2_17_i686.manylinux2014_i686.whl",
            "sentencepiece-0.2.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "sentencepiece-0.2.0-cp39-cp39-win32.whl",
            "sentencepiece-0.2.0-cp39-cp39-win_amd64.whl",
            "sentencepiece-0.2.0.tar.gz",
        ),
    ),
)


def _index(releases: Iterable[Release]) -> dict[str, Release]:
    table: dict[str, Release] = {}
    for release in releases:
        key = _key(release.requirement)
        if key is None:
            raise ValueError(f"{release.requirement} is not a name==version pin")
        table[key] = release
    return table


PUBLISHED: Mapping[str, Release] = _index(_RELEASES)
