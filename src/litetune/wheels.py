"""Which machines a stage environment's pins can be installed on, as PyPI lists them.

`StageEnv.provision` builds an environment with pip, from the interpreter
running litetune. Where a pin publishes no wheel that interpreter can install,
pip finds nothing to install, or falls back to a source build and fails minutes
later without naming the version -- so the environments are checked before pip
is asked, against the files each pin actually publishes.

**The facts are data, with their source.** Each `Release` below is what PyPI's
JSON API returned for one pinned release, read on 2026-10-12: its file names
(`urls[].filename`) copied verbatim, the URL they were read from, and the `==`
requirements its metadata names without a marker (`info.requires_dist`), which
pip can only satisfy with exactly that version. Nothing here is inferred from a
project's other releases: a wheel is in the list or it is not.

**A wheel is matched the way pip matches it.** Its file name's tags are read
with `packaging.utils.parse_wheel_filename` and compared with the
interpreter's compatible tags, `packaging.tags.sys_tags()` -- the same
`cpython_tags` and `compatible_tags` pip's own `get_supported` joins
(`pip/_internal/utils/compatibility_tags.py`, pip 26.2.1). So the minimum
macOS in `macosx_12_0_arm64`, the glibc in `manylinux_2_27_x86_64`, musl,
free-threaded builds and `abi3` all count, as they do for pip. Reducing a tag
to its architecture admitted macOS 11 and glibc 2.26, where pip has no wheel.

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
failures reach the reader through pip's own output; and any index but PyPI.
This is PyPI's file list: `envs` refuses by it and says so, and
`LITETUNE_ALLOW_MISSING_WHEELS=1` is how someone installing from an index or a
wheelhouse that has the wheel turns the refusal into a note.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags, platform_tags, sys_tags
from packaging.utils import parse_wheel_filename


def canonical(name: str) -> str:
    """A distribution name in the one spelling that compares equal.

    `litert_converter`, `litert.converter` and `LiteRT-Converter` are the same
    project: PyPA's name specification normalises runs of `-`, `_` and `.` to a
    single `-` and lowercases. pip prints whichever spelling the requirement
    used, and a wheel's file name spells it with `_`, so a comparison against a
    literal has to normalise or it silently misses.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Host:
    """An interpreter as pip sees it: the tags of every wheel it would install."""

    python: tuple[int, int]
    # Most specific first, in pip's order: `platforms[0]` is what a refusal calls
    # this machine -- `win_amd64`, `macosx_15_0_arm64`, `manylinux_2_39_x86_64`.
    platforms: tuple[str, ...]
    tags: frozenset[Tag]

    @property
    def platform(self) -> str:
        return self.platforms[0]

    @classmethod
    def of(cls, platforms: Sequence[str], python: tuple[int, int]) -> Host:
        """A regular CPython build of `python` that accepts `platforms`: a machine nobody has here.

        The tags `get_supported` would compute for it -- `cpython_tags` with the
        build's own ABI, then `compatible_tags` -- which is how a POSIX runner
        asks what Windows, an older macOS or the next Python would be told.
        """
        interpreter = f"cp{python[0]}{python[1]}"
        tags = frozenset(cpython_tags(python, [interpreter], platforms)) | frozenset(
            compatible_tags(python, interpreter, platforms)
        )
        return cls(python, tuple(platforms), tags)


def current_host() -> Host:
    """The interpreter running this, which is the one a stage environment is built from."""
    return Host(
        (sys.version_info[0], sys.version_info[1]),
        tuple(platform_tags()),
        frozenset(sys_tags()),
    )


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

    def wheels(self) -> list[frozenset[Tag]]:
        """Each wheel's tags, as `parse_wheel_filename` reads its file name."""
        return [parse_wheel_filename(name)[3] for name in self.files if name.endswith(".whl")]

    def platforms(self) -> list[str]:
        """Every platform tag some wheel of this release names, sorted."""
        return sorted({tag.platform for tags in self.wheels() for tag in tags})

    def installs_on(self, host: Host) -> bool:
        return any(tags & host.tags for tags in self.wheels())

    def minors_on(self, host: Host) -> frozenset[int]:
        """The Python 3 minors a regular build on `host`'s platforms has a wheel for.

        Asked the way the match itself is: for each minor some wheel names, the
        tags `Host.of` gives that minor on these platforms.
        """
        named = {
            int(m.group(1))
            for tags in self.wheels()
            for tag in tags
            if (m := re.fullmatch(r"(?:cp|py)3(\d+)", tag.interpreter))
        }
        return frozenset(
            minor
            for minor in named
            if any(tags & Host.of(host.platforms, (3, minor)).tags for tags in self.wheels())
        )


@dataclass(frozen=True)
class Gap:
    """A pin with no wheel for a host: for its platform at all, or for its Python."""

    release: Release
    host: Host
    # The Python 3 minors `release` has wheels for on `host`'s platforms. Empty
    # when it has none there for any Python, which is a different sentence.
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
                f"{name} publishes no wheel for {self.host.platform} and {builds} -- its wheels "
                f"are for {', '.join(self.release.platforms())} ({self.release.source})"
            )
        return (
            f"{name}, which publishes wheels for {self.host.platform} for {_span(self.minors)} "
            f"only ({self.release.source})"
        )


def _span(minors: frozenset[int]) -> str:
    ordered = sorted(minors)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"python3.{ordered[0]} to python3.{ordered[-1]}"
    return ", ".join(f"python3.{m}" for m in ordered)


def marker_environment(host: Host | None = None) -> dict[str, str]:
    """What a requirement's marker is evaluated against, for `host` or this interpreter.

    `packaging.markers.default_environment()`, the values pip evaluates a
    marker with, with the Python version replaced by `host`'s when it asks
    about another one. Only the version follows a machine nobody has here: the
    pins' markers name nothing else.
    """
    environment = {key: str(value) for key, value in default_environment().items()}
    if host is not None and environment["python_version"] != f"{host.python[0]}.{host.python[1]}":
        environment["python_version"] = f"{host.python[0]}.{host.python[1]}"
        environment["python_full_version"] = f"{host.python[0]}.{host.python[1]}.0"
    return environment


def pin(requirement: str) -> tuple[str, str] | None:
    """`(name, version)` for a requirement that fixes one exact version, else None.

    Read as pip reads it (`packaging.requirements.Requirement`), so a marker is
    not part of the version and an `==` inside a marker is not a pin:
    `torch>=2.0; python_version == "3.12"` floats. A URL is not a version and
    answers None here; whether it counts as pinned is the caller's rule.
    """
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return None
    specifiers = list(parsed.specifier)
    if parsed.url or len(specifiers) != 1:
        return None
    (only,) = specifiers
    if only.operator not in ("==", "===") or only.version.endswith(".*"):
        return None
    return parsed.name, only.version


def applies(requirement: str, environment: Mapping[str, str]) -> bool:
    """Whether pip would install `requirement` under `environment`: no marker, or a true one."""
    try:
        marker = Requirement(requirement).marker
    except InvalidRequirement:
        return True
    return marker is None or marker.evaluate(environment)


def without_marker(requirement: str) -> str:
    """The requirement without its marker, read as pip reads it.

    Unchanged when it has no marker -- so an identity over unmarked pins hashes
    the very strings it always did -- and when it is a direct URL, or not a
    requirement packaging can read: a `;` inside a URL's query is part of the
    URL, and splitting there made two different URLs one identity.
    """
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return requirement
    if parsed.marker is None or parsed.url:
        return requirement
    parsed.marker = None
    return str(parsed)


def _key(requirement: str) -> str | None:
    """`name==version` in canonical spelling, or None for a requirement that is not one."""
    pinned = pin(requirement)
    if pinned is None:
        return None
    return f"{canonical(pinned[0])}=={pinned[1]}"


def named_minors(published: Mapping[str, Release] | None = None) -> list[int]:
    """Every CPython 3 minor some wheel on record is built for, in order."""
    table = PUBLISHED if published is None else published
    return sorted(
        {
            int(m.group(1))
            for release in table.values()
            for tags in release.wheels()
            for tag in tags
            if (m := re.fullmatch(r"cp3(\d+)", tag.interpreter))
        }
    )


def gaps(
    requirements: Iterable[str],
    host: Host,
    published: Mapping[str, Release] | None = None,
) -> list[Gap]:
    """Every pin that applies to `host` with no wheel `host` installs.

    Follows each release's own exact requirements, so a pin whose wheel exists
    but whose `==` dependency's does not is caught and named as that
    dependency. A requirement with no release on record is not judged at all:
    this says what PyPI's lists establish and nothing past them.
    """
    table = PUBLISHED if published is None else published
    found: list[Gap] = []
    seen: set[str] = set()
    # Only what pip would install there: a pin whose marker is false for
    # `host`'s Python is not asked for, so its wheels are not this host's
    # business -- `sentencepiece==0.2.1; python_version >= "3.13"` on 3.12.
    environment = marker_environment(host)
    pending: list[tuple[str, str | None]] = [
        (r, None) for r in requirements if applies(r, environment)
    ]
    while pending:
        requirement, parent = pending.pop(0)
        key = _key(requirement)
        if key is None or key in seen or key not in table:
            continue
        seen.add(key)
        release = table[key]
        pending.extend((r, release.requirement) for r in release.requires)
        if release.installs_on(host):
            continue
        found.append(Gap(release, host, release.minors_on(host), required_by=parent))
    return found


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
    Release(
        "sentencepiece==0.2.1",
        "https://pypi.org/pypi/sentencepiece/0.2.1/json",
        files=(
            "sentencepiece-0.2.1-cp310-cp310-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.1-cp310-cp310-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.1-cp310-cp310-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp310-cp310-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp310-cp310-win32.whl",
            "sentencepiece-0.2.1-cp310-cp310-win_amd64.whl",
            "sentencepiece-0.2.1-cp310-cp310-win_arm64.whl",
            "sentencepiece-0.2.1-cp311-cp311-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.1-cp311-cp311-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.1-cp311-cp311-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp311-cp311-win32.whl",
            "sentencepiece-0.2.1-cp311-cp311-win_amd64.whl",
            "sentencepiece-0.2.1-cp311-cp311-win_arm64.whl",
            "sentencepiece-0.2.1-cp312-cp312-macosx_10_13_universal2.whl",
            "sentencepiece-0.2.1-cp312-cp312-macosx_10_13_x86_64.whl",
            "sentencepiece-0.2.1-cp312-cp312-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp312-cp312-win32.whl",
            "sentencepiece-0.2.1-cp312-cp312-win_amd64.whl",
            "sentencepiece-0.2.1-cp312-cp312-win_arm64.whl",
            "sentencepiece-0.2.1-cp313-cp313-macosx_10_13_universal2.whl",
            "sentencepiece-0.2.1-cp313-cp313-macosx_10_13_x86_64.whl",
            "sentencepiece-0.2.1-cp313-cp313-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp313-cp313-win32.whl",
            "sentencepiece-0.2.1-cp313-cp313-win_amd64.whl",
            "sentencepiece-0.2.1-cp313-cp313-win_arm64.whl",
            "sentencepiece-0.2.1-cp313-cp313t-macosx_10_13_universal2.whl",
            "sentencepiece-0.2.1-cp313-cp313t-macosx_10_13_x86_64.whl",
            "sentencepiece-0.2.1-cp313-cp313t-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp313-cp313t-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp313-cp313t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp313-cp313t-win32.whl",
            "sentencepiece-0.2.1-cp313-cp313t-win_amd64.whl",
            "sentencepiece-0.2.1-cp313-cp313t-win_arm64.whl",
            "sentencepiece-0.2.1-cp314-cp314-macosx_10_13_universal2.whl",
            "sentencepiece-0.2.1-cp314-cp314-macosx_10_13_x86_64.whl",
            "sentencepiece-0.2.1-cp314-cp314-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp314-cp314-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp314-cp314-win32.whl",
            "sentencepiece-0.2.1-cp314-cp314-win_amd64.whl",
            "sentencepiece-0.2.1-cp314-cp314-win_arm64.whl",
            "sentencepiece-0.2.1-cp314-cp314t-macosx_10_13_universal2.whl",
            "sentencepiece-0.2.1-cp314-cp314t-macosx_10_13_x86_64.whl",
            "sentencepiece-0.2.1-cp314-cp314t-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp314-cp314t-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp314-cp314t-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp314-cp314t-win32.whl",
            "sentencepiece-0.2.1-cp314-cp314t-win_amd64.whl",
            "sentencepiece-0.2.1-cp314-cp314t-win_arm64.whl",
            "sentencepiece-0.2.1-cp39-cp39-macosx_10_9_universal2.whl",
            "sentencepiece-0.2.1-cp39-cp39-macosx_10_9_x86_64.whl",
            "sentencepiece-0.2.1-cp39-cp39-macosx_11_0_arm64.whl",
            "sentencepiece-0.2.1-cp39-cp39-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
            "sentencepiece-0.2.1-cp39-cp39-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            "sentencepiece-0.2.1-cp39-cp39-win32.whl",
            "sentencepiece-0.2.1-cp39-cp39-win_amd64.whl",
            "sentencepiece-0.2.1-cp39-cp39-win_arm64.whl",
            "sentencepiece-0.2.1.tar.gz",
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
