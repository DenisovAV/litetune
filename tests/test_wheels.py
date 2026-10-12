"""What PyPI lists for each pin, and what that means for each machine.

The table in `litetune.wheels` is copied from PyPI's JSON API, file name by
file name. These tests hold it to that shape -- every pin has an entry, every
entry is the release it claims to be -- and pin what it decides, platform by
platform, so a pin moved without reading its files in fails here rather than
refusing or admitting a machine on stale facts.
"""

from __future__ import annotations

import pytest

from litetune import envs, wheels
from litetune.wheels import PUBLISHED, Release, canonical, gaps, platform_key

STAGES = (envs.RUNTIME, envs.EXPORT, envs.TRAIN)


def _blocking(env: envs.StageEnv, platform: str, minor: int) -> list[str]:
    return sorted(g.release.requirement for g in gaps(env.requirements, platform, minor))


# ---------------------------------------------------------------------------
# The table is what it says it is
# ---------------------------------------------------------------------------


def test_every_pin_has_its_wheels_on_record():
    """A pin moved without reading its release in would be judged on nothing.

    `gaps` does not judge a requirement it has no release for -- that is what
    keeps a test's `pyyaml==6.0.2` out of it -- so for the stages' own pins,
    absence has to fail here instead. The exact requirements behind each pin
    are followed too: `litert-lm-api` is in no environment's list, and it is
    the package Windows on ARM64 has no wheel for.
    """
    pending = [r for env in STAGES for r in env.requirements]
    seen: set[str] = set()
    while pending:
        requirement = pending.pop()
        name, _, version = requirement.partition("==")
        key = f"{canonical(name)}=={version}"
        assert key in PUBLISHED, f"{requirement}: read https://pypi.org/pypi/{name}/{version}/json"
        if key not in seen:
            seen.add(key)
            pending.extend(PUBLISHED[key].requires)


@pytest.mark.parametrize("release", list(PUBLISHED.values()), ids=lambda r: r.requirement)
def test_every_release_on_record_is_the_one_it_names(release: Release):
    """Its source is its own JSON URL, and every file is a file of that release."""
    name, _, version = release.requirement.partition("==")
    assert release.source == f"https://pypi.org/pypi/{name}/{version}/json"
    assert release.files, "a release with no files would refuse everything"
    for filename in release.files:
        if filename.endswith(".whl"):
            # A wheel's name has its `-` replaced, so the first two fields are it.
            dist, file_version = filename.split("-")[:2]
        else:
            # An sdist's may not (`ordered-set-4.1.0.tar.gz`): the version is last.
            dist, file_version = filename.removesuffix(".tar.gz").rsplit("-", 1)
        assert canonical(dist) == canonical(name), filename
        assert file_version == version, filename
    # Every wheel's tags are ones the reader knows; an unknown one raises.
    release.platforms()


# ---------------------------------------------------------------------------
# What it decides, machine by machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "platform", "minor", "blocking"),
    [
        # Linux x86-64: everything to 3.12; numpy stops runtime and export at
        # 3.13, sentencepiece stops train there -- torch alone would not.
        (envs.RUNTIME, "linux_x86_64", 12, []),
        (envs.EXPORT, "linux_x86_64", 12, []),
        (envs.TRAIN, "linux_x86_64", 12, []),
        (envs.RUNTIME, "linux_x86_64", 13, ["numpy==2.0.2"]),
        (envs.EXPORT, "linux_x86_64", 13, ["numpy==2.0.2"]),
        (envs.TRAIN, "linux_x86_64", 13, ["sentencepiece==0.2.0"]),
        # Apple Silicon: the same, except torch has no cp313 wheel there.
        (envs.EXPORT, "macosx_arm64", 12, []),
        (envs.TRAIN, "macosx_arm64", 13, ["sentencepiece==0.2.0", "torch==2.5.1"]),
        # Windows x64: verify and tune install; convert does not, for one pin.
        (envs.RUNTIME, "win_amd64", 12, []),
        (envs.TRAIN, "win_amd64", 12, []),
        (envs.EXPORT, "win_amd64", 12, ["litert-converter==0.4.0"]),
        (envs.TRAIN, "win_amd64", 13, ["sentencepiece==0.2.0", "torch==2.5.1"]),
        # Windows on ARM64 and 32-bit Windows: no stage at all.
        (envs.RUNTIME, "win_arm64", 12, ["litert-lm-api==0.18.0", "numpy==2.0.2"]),
        (envs.TRAIN, "win_arm64", 12, ["sentencepiece==0.2.0", "torch==2.5.1"]),
        (envs.RUNTIME, "win32", 12, ["litert-lm-api==0.18.0"]),
        (envs.TRAIN, "win32", 12, ["torch==2.5.1"]),
        (
            envs.EXPORT,
            "win32",
            12,
            [
                "ai-edge-litert-nightly==2.3.0.dev20260925",
                "litert-converter==0.4.0",
                "litert-lm-api==0.18.0",
            ],
        ),
        # ARM Linux: the runtime installs, the converter does not.
        (envs.RUNTIME, "linux_aarch64", 12, []),
        (envs.EXPORT, "linux_aarch64", 12, ["litert-converter==0.4.0"]),
        # Intel macOS: neither the runtime's native library nor torch.
        (envs.RUNTIME, "macosx_x86_64", 12, ["litert-lm-api==0.18.0"]),
        (envs.TRAIN, "macosx_x86_64", 12, ["torch==2.5.1"]),
    ],
)
def test_which_pins_stop_which_stage_where(env, platform, minor, blocking):
    assert _blocking(env, platform, minor) == blocking


def test_the_converter_has_no_windows_wheel_and_no_source_to_build():
    converter = PUBLISHED["litert-converter==0.4.0"]
    assert not converter.sdist
    assert converter.platforms() == ["linux_x86_64", "macosx_arm64"]
    for platform in ("win_amd64", "win_arm64", "win32", "linux_aarch64", "macosx_x86_64"):
        assert converter.minors_on(platform) == frozenset()


def test_an_exact_requirement_is_named_with_the_pin_that_brought_it():
    (gap,) = gaps(["litert-lm==0.18.0"], "win32", 12)
    assert gap.release.requirement == "litert-lm-api==0.18.0"
    assert gap.required_by == "litert-lm==0.18.0"
    assert "litert-lm-api==0.18.0 (required by litert-lm==0.18.0)" in gap.describe()


def test_a_requirement_with_no_release_on_record_is_not_judged():
    """Nothing here is refused on a guess: no file list, no claim."""
    assert gaps(["pyyaml==6.0.2", "https://example.invalid/x.whl"], "win_arm64", 13) == []


def test_a_wheel_built_for_no_platform_installs_everywhere():
    assert PUBLISHED["transformers==5.16.1"].minors_on("freebsd_14_0_amd64") is None


def test_a_pin_is_followed_through_any_spelling_of_its_name():
    assert [g.release.requirement for g in gaps(["LiteRT_Converter==0.4.0"], "win_amd64", 12)] == [
        "litert-converter==0.4.0"
    ]


def test_a_source_distribution_is_said_to_be_one():
    """pip would try to build it, which is a different failure from finding nothing."""
    (numpy,) = gaps(["numpy==2.0.2"], "win_arm64", 12)
    assert "only a source distribution, which pip would try to build" in numpy.describe()
    (converter,) = gaps(["litert-converter==0.4.0"], "win_arm64", 12)
    assert "no source distribution either, so pip has nothing to install" in converter.describe()


def test_a_python_gap_names_the_versions_that_do_have_wheels():
    (gap,) = gaps(["torch==2.5.1"], "win_amd64", 13)
    assert not gap.platform_missing
    assert "for win_amd64 for python3.9 to python3.12 only" in gap.describe()
    assert gap.release.source in gap.describe()


def test_a_python_range_with_a_hole_is_listed_rather_than_bridged():
    assert wheels._span(frozenset({10, 11, 12})) == "python3.10 to python3.12"
    assert wheels._span(frozenset({10, 12})) == "python3.10, python3.12"


# ---------------------------------------------------------------------------
# Reading a wheel's tags
# ---------------------------------------------------------------------------


def test_a_universal2_macos_wheel_is_both_of_its_architectures():
    assert wheels._platforms("macosx_10_9_universal2") == {"macosx_arm64", "macosx_x86_64"}


def test_a_compressed_manylinux_tag_is_one_platform():
    tag = "manylinux_2_17_aarch64.manylinux2014_aarch64"
    assert wheels._platforms(tag) == {"linux_aarch64"}


@pytest.mark.parametrize(
    ("filename", "minors", "every"),
    [
        ("x-1-py3-none-any.whl", set(), True),
        ("x-1-py2.py3-none-any.whl", set(), True),
        ("x-1-cp310-none-macosx_11_0_arm64.whl", {10}, False),
        ("x-1-cp36-cp36m-win32.whl", {6}, False),
    ],
)
def test_python_tags_are_read_as_cpython(filename, minors, every):
    wheel = wheels._wheel(filename)
    assert wheel is not None
    assert wheel.minors == minors
    assert wheel.every_python is every


def test_a_pypy_wheel_installs_on_no_cpython():
    assert wheels._wheel("numpy-2.0.2-pp39-pypy39_pp73-win_amd64.whl") is None


@pytest.mark.parametrize(
    "filename",
    [
        "x-1-cp312-abi3-win_amd64.whl",
        "x-1-cp313-cp313t-win_amd64.whl",
        "x-1-cp312-cp312-linux_x86_64.whl",
        "x-1-cp312-cp312-macosx_10_9_intel.whl",
        "x-1-jy27-none-any.whl",
    ],
)
def test_a_tag_nobody_has_read_is_refused_rather_than_guessed(filename):
    """A pin moved to a release with such a tag fails the table's tests instead
    of being read as nothing -- which would refuse a machine it was built for."""
    with pytest.raises(ValueError):
        wheels._wheel(filename)


# ---------------------------------------------------------------------------
# The host, as pip would see it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "sysconfig_platform", "mac_arch", "is_32bit", "key"),
    [
        ("Darwin", "macosx-11.0-arm64", "arm64", False, "macosx_arm64"),
        ("Darwin", "macosx-10.9-universal2", "x86_64", False, "macosx_x86_64"),
        ("Darwin", "macosx-10.9-universal2", "x86_64", True, "macosx_i386"),
        ("Linux", "linux-x86_64", "", False, "linux_x86_64"),
        ("Linux", "linux-x86_64", "", True, "linux_i686"),
        ("Linux", "linux-aarch64", "", True, "linux_armv8l"),
        ("Linux", "linux-aarch64", "", False, "linux_aarch64"),
        ("Windows", "win-amd64", "", False, "win_amd64"),
        ("Windows", "win-arm64", "", False, "win_arm64"),
        ("Windows", "win32", "", True, "win32"),
    ],
)
def test_the_host_platform_is_read_as_pip_reads_it(
    system, sysconfig_platform, mac_arch, is_32bit, key
):
    assert platform_key(system, sysconfig_platform, mac_arch, is_32bit) == key


def test_the_running_interpreter_is_where_pip_says_it_is():
    """Checked against pip's own reading, on whatever machine runs this.

    pip's first platform tag is the most specific one this interpreter installs
    (`packaging.tags.platform_tags`), and read through this module's tag reader
    it must name the platform `host_platform` derived. On the Windows job that
    is the one check here made by a real Windows interpreter. pip's vendored
    `packaging` is private to pip, so its absence skips rather than fails.
    """
    tags = pytest.importorskip("pip._vendor.packaging.tags")

    first = next(iter(tags.platform_tags()))
    assert wheels.host_platform() in wheels._platforms(first), first
