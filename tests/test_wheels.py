"""What PyPI lists for each pin, and what that means for each machine.

The table in `litetune.wheels` is copied from PyPI's JSON API, file name by
file name. These tests hold it to that shape -- every pin has an entry, every
entry is the release it claims to be -- and pin what it decides, machine by
machine, so a pin moved without reading its files in fails here rather than
refusing or admitting a machine on stale facts.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
from conftest import MACHINES, machine
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

from litetune import envs, wheels
from litetune.wheels import PUBLISHED, Host, Release, canonical, gaps

STAGES = (envs.RUNTIME, envs.EXPORT, envs.TRAIN)


def _blocking(env: envs.StageEnv, name: str, minor: int) -> list[str]:
    host = machine(name, (3, minor))
    return sorted(g.release.requirement for g in gaps(env.requirements, host))


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
            # Read as pip reads it, which also refuses a name that is not a wheel's.
            wheel_name, file_version, _, _ = parse_wheel_filename(filename)
            assert str(file_version) == version, filename
            assert canonical(wheel_name) == canonical(name), filename
        else:
            # An sdist's may keep its `-` (`ordered-set-4.1.0.tar.gz`): the version is last.
            sdist_name, sdist_version = filename.removesuffix(".tar.gz").rsplit("-", 1)
            assert sdist_version == version, filename
            assert canonical(sdist_name) == canonical(name), filename


# ---------------------------------------------------------------------------
# What it decides, machine by machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "name", "minor", "blocking"),
    [
        # Linux x86-64 on glibc 2.35: everything to 3.12, and numpy stops
        # runtime and export at 3.13. Train reaches 3.13 here and nowhere else --
        # torch 2.5.1's one cp313 wheel is manylinux x86_64 -- and stops at 3.14.
        (envs.RUNTIME, "linux_x86_64", 12, []),
        (envs.EXPORT, "linux_x86_64", 12, []),
        (envs.TRAIN, "linux_x86_64", 12, []),
        (envs.RUNTIME, "linux_x86_64", 13, ["numpy==2.0.2"]),
        (envs.EXPORT, "linux_x86_64", 13, ["numpy==2.0.2"]),
        (envs.TRAIN, "linux_x86_64", 13, []),
        (envs.TRAIN, "linux_x86_64", 14, ["torch==2.5.1"]),
        (envs.TRAIN, "linux_aarch64", 13, ["torch==2.5.1"]),
        # glibc: litert-lm-api, the converter, ai-edge-litert and sentencepiece
        # 0.2.1 are all manylinux_2_27, so 2.27 is every stage's floor.
        (
            envs.EXPORT,
            "linux_x86_64_glibc_2_26",
            12,
            [
                "ai-edge-litert-nightly==2.3.0.dev20260925",
                "litert-converter==0.4.0",
                "litert-lm-api==0.18.0",
            ],
        ),
        (envs.TRAIN, "linux_x86_64_glibc_2_26", 12, ["sentencepiece==0.2.1"]),
        (envs.RUNTIME, "linux_x86_64_glibc_2_27", 12, []),
        (envs.EXPORT, "linux_x86_64_glibc_2_27", 12, []),
        (envs.TRAIN, "linux_x86_64_glibc_2_27", 12, []),
        # musl: nothing, not even the runtime's own library.
        (envs.RUNTIME, "musl_x86_64", 12, ["litert-lm-api==0.18.0"]),
        (envs.TRAIN, "musl_x86_64", 12, ["sentencepiece==0.2.1", "torch==2.5.1"]),
        # Apple Silicon: macOS 12 for what carries litert-lm-api, 11 for train.
        (envs.EXPORT, "macosx_arm64", 12, []),
        (envs.TRAIN, "macosx_arm64", 13, ["torch==2.5.1"]),
        (envs.RUNTIME, "macosx_11_arm64", 12, ["litert-lm-api==0.18.0"]),
        (envs.TRAIN, "macosx_11_arm64", 12, []),
        (envs.RUNTIME, "macosx_12_arm64", 12, []),
        (envs.EXPORT, "macosx_12_arm64", 12, []),
        # Windows x64: verify and tune install; convert does not, for one pin.
        (envs.RUNTIME, "win_amd64", 12, []),
        (envs.TRAIN, "win_amd64", 12, []),
        (envs.EXPORT, "win_amd64", 12, ["litert-converter==0.4.0"]),
        (envs.TRAIN, "win_amd64", 13, ["torch==2.5.1"]),
        # Windows on ARM64 and 32-bit Windows: no stage at all.
        (envs.RUNTIME, "win_arm64", 12, ["litert-lm-api==0.18.0", "numpy==2.0.2"]),
        (envs.TRAIN, "win_arm64", 12, ["torch==2.5.1"]),
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
def test_which_pins_stop_which_stage_where(env, name, minor, blocking):
    assert _blocking(env, name, minor) == blocking


def test_the_converter_has_no_windows_wheel_and_no_source_to_build():
    converter = PUBLISHED["litert-converter==0.4.0"]
    assert not converter.sdist
    assert converter.platforms() == ["macosx_12_0_arm64", "manylinux_2_27_x86_64"]
    for name in ("win_amd64", "win_arm64", "win32", "linux_aarch64", "macosx_x86_64"):
        assert converter.minors_on(machine(name, (3, 12))) == frozenset(), name


def test_an_exact_requirement_is_named_with_the_pin_that_brought_it():
    (gap,) = gaps(["litert-lm==0.18.0"], machine("win32", (3, 12)))
    assert gap.release.requirement == "litert-lm-api==0.18.0"
    assert gap.required_by == "litert-lm==0.18.0"
    assert "litert-lm-api==0.18.0 (required by litert-lm==0.18.0)" in gap.describe()


def test_a_requirement_with_no_release_on_record_is_not_judged():
    """Nothing here is refused on a guess: no file list, no claim."""
    requirements = ["pyyaml==6.0.2", "https://example.invalid/x.whl"]
    assert gaps(requirements, machine("win_arm64", (3, 13))) == []


def test_a_wheel_built_for_no_platform_installs_everywhere():
    host = Host.of(["freebsd_14_0_amd64"], (3, 12))
    assert PUBLISHED["transformers==5.16.1"].installs_on(host)


def test_a_pin_is_followed_through_any_spelling_of_its_name():
    found = gaps(["LiteRT_Converter==0.4.0"], machine("win_amd64", (3, 12)))
    assert [g.release.requirement for g in found] == ["litert-converter==0.4.0"]


def test_a_source_distribution_is_said_to_be_one():
    """pip would try to build it, which is a different failure from finding nothing."""
    host = machine("win_arm64", (3, 12))
    (numpy,) = gaps(["numpy==2.0.2"], host)
    assert "only a source distribution, which pip would try to build" in numpy.describe()
    (converter,) = gaps(["litert-converter==0.4.0"], host)
    assert "no source distribution either, so pip has nothing to install" in converter.describe()


def test_a_python_gap_names_the_versions_that_do_have_wheels():
    (gap,) = gaps(["torch==2.5.1"], machine("win_amd64", (3, 13)))
    assert not gap.platform_missing
    assert "for win_amd64 for python3.9 to python3.12 only" in gap.describe()
    assert gap.release.source in gap.describe()


def test_a_platform_gap_lists_the_tags_that_say_why():
    """On macOS 11 the wheel exists for the architecture; its tag says macOS 12."""
    (gap,) = gaps(["litert-lm-api==0.18.0"], machine("macosx_11_arm64", (3, 12)))
    said = gap.describe()
    assert "publishes no wheel for macosx_11_0_arm64" in said
    assert "macosx_12_0_arm64" in said


def test_a_python_range_with_a_hole_is_listed_rather_than_bridged():
    assert wheels._span(frozenset({10, 11, 12})) == "python3.10 to python3.12"
    assert wheels._span(frozenset({10, 12})) == "python3.10, python3.12"


# ---------------------------------------------------------------------------
# Matched as pip matches
# ---------------------------------------------------------------------------


def _release(*files: str) -> Release:
    return Release("x==1", "https://pypi.org/pypi/x/1/json", files=files)


def test_a_free_threaded_wheel_installs_on_no_regular_build():
    """`sentencepiece==0.2.1` ships `cp313t` beside `cp313`; only one is a regular build's."""
    only_free_threaded = _release("x-1-cp313-cp313t-win_arm64.whl")
    assert not only_free_threaded.installs_on(machine("win_arm64", (3, 13)))


def test_an_abi3_wheel_installs_on_every_later_python():
    stable_abi = _release("x-1-cp38-abi3-win_amd64.whl")
    assert stable_abi.installs_on(machine("win_amd64", (3, 13)))
    assert not stable_abi.installs_on(Host.of(["win_amd64"], (3, 7)))


def test_a_pypy_wheel_installs_on_no_cpython():
    assert not _release("x-1-pp39-pypy39_pp73-win_amd64.whl").installs_on(
        machine("win_amd64", (3, 9))
    )


# Every tag a wheel on record carries: what a host's tags are asked about.
_TAGS_ON_RECORD = sorted(
    {str(tag) for release in PUBLISHED.values() for tags in release.wheels() for tag in tags}
)


def _accepted(tags: Iterable[object]) -> set[str]:
    return {str(tag) for tag in tags if str(tag) in _TAGS_ON_RECORD}


@pytest.mark.parametrize("name", sorted(MACHINES))
def test_a_machine_nobody_has_accepts_what_pip_would_accept_there(name):
    """`Host.of` against pip's own `get_supported`, for every machine the tests use.

    Compared on the tags the files on record carry, which is the decision this
    makes: the `packaging` library and pip's vendored copy are versions apart
    (26.3 spells a legacy macOS format `fat3` where 26.2 spells it `fat32`), and
    no pinned wheel is one of those. pip's module is private to pip, so only a
    test reads it.
    """
    compatibility = pytest.importorskip("pip._internal.utils.compatibility_tags")
    host = machine(name, (3, 12))
    # `get_supported` expands every macOS tag it is handed into that version's
    # whole list (`_mac_platforms`, for `--platform`), so macOS is handed its
    # most specific tag alone; other tags it takes as given.
    given = [host.platform] if host.platform.startswith("macosx") else list(host.platforms)
    pips = compatibility.get_supported(version="312", platforms=given, impl="cp", abis=["cp312"])
    assert _accepted(host.tags) == _accepted(pips)


def test_this_interpreter_answers_for_its_own_build(monkeypatch):
    """`current_host` takes `sys_tags` whole rather than rebuilding them.

    On a regular CPython the two agree; on a free-threaded or debug build, or
    another interpreter, only `sys_tags` knows the ABI -- `cp313t` here -- and a
    host rebuilt from its platforms would accept wheels the build cannot load.
    """
    from packaging.tags import Tag

    built = [Tag("cp313", "cp313t", "win_amd64")]
    monkeypatch.setattr(wheels, "sys_tags", lambda: iter(built))

    assert wheels.current_host().tags == frozenset(built)


def test_this_interpreter_is_seen_as_pip_sees_it():
    """`current_host` against pip's own answer, on whichever runner this is.

    On the Windows job this is a Windows interpreter answering for itself.
    """
    compatibility = pytest.importorskip("pip._internal.utils.compatibility_tags")
    host = wheels.current_host()

    assert _accepted(host.tags) == _accepted(compatibility.get_supported())
    assert host.tags == frozenset(sys_tags())
