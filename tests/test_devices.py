"""The parent's side of a stage's device: LITETUNE_DEVICE, host facts, the MPS policy.

No torch, no sysctl, no subprocess: every input is passed in, which is why
`devices.mps_memory_policy` takes its figures as arguments.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from conftest import FAKE_SYSCTL, fake_sysctl

from litetune import devices, envs
from litetune.devices import (
    DEVICE_VARIABLE,
    GIB,
    HIGH_WATERMARK,
    LOW_WATERMARK,
    MPS_FALLBACK,
    DeviceSettingError,
    HostReadError,
    MpsMemoryRefused,
    apply_device_setting,
    device_setting,
    host_record,
    mps_memory_policy,
    prepare_mps,
)

# Bound at import, before `conftest` replaces `devices.read_sysctl` for every
# test with a fake: these tests are about the real one, with `subprocess.run`
# faked under it instead.
from litetune.devices import read_sysctl as real_read_sysctl  # noqa: E402


def _probe(device: str | None = "mps", **fields) -> envs.DeviceProbe:
    defaults: dict[str, Any] = {
        "detail": f"this environment's torch reports {device}",
        "source": "probe" if device is not None else None,
        "mps_recommended_max_memory": 16 * GIB if device == "mps" else None,
        "os": "Darwin",
        "os_version": "15.0",
        "machine": "arm64",
    }
    defaults.update(fields)
    return envs.DeviceProbe(device=device, **defaults)


# ---------------------------------------------------------------------------
# LITETUNE_DEVICE
# ---------------------------------------------------------------------------


def test_unset_is_auto():
    assert device_setting({}) == "auto"


@pytest.mark.parametrize("value", ["auto", "cpu"])
def test_the_accepted_values_are_taken_as_they_are(value):
    assert device_setting({DEVICE_VARIABLE: value}) == value


@pytest.mark.parametrize("value", ["mps", "cuda", "CPU", " cpu", "", "gpu"])
def test_anything_else_is_refused_naming_what_is_accepted(value):
    """A misspelt "cpu" read as "auto" would put the run on the GPU it was
    meant to avoid. "mps" and "cuda" are refused too: the variable can force
    the CPU, and choosing an accelerator is the probe's."""
    with pytest.raises(DeviceSettingError) as raised:
        device_setting({DEVICE_VARIABLE: value})
    message = str(raised.value)
    assert DEVICE_VARIABLE in message
    assert repr(value) in message
    assert "auto, cpu" in message


def test_auto_leaves_the_probe_alone():
    probe = _probe("mps")
    assert apply_device_setting(probe, "auto") is probe


def test_cpu_replaces_an_mps_answer_and_says_who_chose_it():
    forced = apply_device_setting(_probe("mps"), "cpu")
    assert forced.device == "cpu"
    assert forced.source == DEVICE_VARIABLE
    assert forced.as_dict()["source"] == DEVICE_VARIABLE
    assert "LITETUNE_DEVICE=cpu" in forced.detail
    # What the probe found stays on the record.
    assert "reports mps" in forced.detail
    # A forced CPU on a Mac is not the CPU the probe has to explain.
    assert not forced.cpu_on_macos


def test_cpu_establishes_a_device_where_the_probe_could_not():
    unanswered = envs.DeviceProbe(device=None, detail="the device probe could not answer: boom")
    forced = apply_device_setting(unanswered, "cpu")
    assert forced.answered
    assert forced.device == "cpu"
    assert "boom" in forced.detail


# ---------------------------------------------------------------------------
# Host facts
# ---------------------------------------------------------------------------


def test_the_host_record_takes_the_os_from_the_probe_and_the_chip_from_sysctl():
    record = host_record(_probe("mps"), platform="darwin", sysctl=fake_sysctl(FAKE_SYSCTL))
    assert record == {
        "os": "Darwin",
        "os_version": "15.0",
        "machine": "arm64",
        "chip": "Apple M-test",
    }


def test_the_chip_is_read_only_on_macos():
    def never(name: str) -> str:
        raise AssertionError(f"sysctl {name} read off macOS")

    record = host_record(_probe("cpu", os="Linux", os_version=None), platform="linux", sysctl=never)
    assert record["chip"] is None
    assert record["os"] == "Linux"


def test_a_chip_that_cannot_be_read_is_recorded_with_the_reason():
    record = host_record(_probe("cpu"), platform="darwin", sysctl=fake_sysctl({}))
    assert record["chip"] is None
    assert "machdep.cpu.brand_string" in record["chip_error"]


def test_read_sysctl_returns_the_stripped_value(monkeypatch):
    seen: list = []

    def run(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "Apple M4 Pro\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert real_read_sysctl("machdep.cpu.brand_string") == "Apple M4 Pro"
    argv, kwargs = seen[0]
    assert argv == ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
    assert kwargs["timeout"] == devices.SYSCTL_TIMEOUT_S


@pytest.mark.parametrize(
    ("result", "raises", "words"),
    [
        (subprocess.CompletedProcess([], 1, "", "unknown oid"), None, "exited 1"),
        (subprocess.CompletedProcess([], 0, "  \n", ""), None, "printed nothing"),
        (None, FileNotFoundError("sysctl"), "FileNotFoundError"),
        (None, subprocess.TimeoutExpired("sysctl", 10), "TimeoutExpired"),
    ],
)
def test_read_sysctl_raises_rather_than_answering_nothing(monkeypatch, result, raises, words):
    def run(argv, **kwargs):
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(HostReadError, match=words):
        real_read_sysctl("hw.memsize")


# ---------------------------------------------------------------------------
# The MPS memory policy
# ---------------------------------------------------------------------------


PAGE = 16384


def _reading(available: int = 16 * GIB, pressure: int = 1) -> devices.MemoryReading:
    """`available` bytes, spread over the four counts so each one is summed."""
    pages = available // PAGE
    quarter = pages // 4
    return devices.memory_reading(
        page_size=PAGE,
        pages={
            "vm.page_free_count": quarter,
            "vm.page_speculative_count": quarter,
            "vm.page_pageable_external_count": quarter,
            "vm.page_purgeable_count": pages - 3 * quarter,
        },
        pressure_level=pressure,
    )


def _policy(environ=None, recommended=16 * GIB, available=16 * GIB, pressure=1):
    return mps_memory_policy(
        recommended_max_memory=recommended,
        reading=_reading(available, pressure),
        environ=environ or {},
        swapusage="total = 0.00M",
    )


def test_available_memory_is_the_four_page_counts_times_the_page_size():
    reading = devices.memory_reading(
        page_size=PAGE,
        pages={
            "vm.page_free_count": 1,
            "vm.page_speculative_count": 10,
            "vm.page_pageable_external_count": 100,
            "vm.page_purgeable_count": 1000,
        },
        pressure_level=1,
    )
    assert reading.available_bytes == 1111 * PAGE


def test_a_memory_reading_takes_exactly_the_four_counts():
    with pytest.raises(ValueError):
        devices.memory_reading(PAGE, {"vm.page_free_count": 1}, 1)


def test_the_budget_is_the_available_memory_less_the_headroom_when_that_is_smaller():
    # 16 GiB available, less 3 GiB: 13 GiB, under Metal's 16.
    policy = _policy()
    assert policy.budget == 13 * GIB
    assert policy.variables[HIGH_WATERMARK] == "0.812500"
    assert policy.variables[LOW_WATERMARK] == "0.650000"
    assert policy.variables[MPS_FALLBACK] == "0"
    assert policy.child_env == policy.variables
    assert policy.user_set == ()


def test_the_budget_is_metals_recommendation_when_that_is_smaller():
    # 57.6 GiB available less 3 is far above 16: the ratio is exactly 1.
    policy = _policy(recommended=16 * GIB, available=576 * GIB // 10)
    assert policy.budget == 16 * GIB
    assert policy.variables[HIGH_WATERMARK] == "1.000000"
    assert policy.variables[LOW_WATERMARK] == "0.800000"


def test_the_low_ratio_is_never_above_the_high_one():
    for tenths in range(40, 200, 7):
        policy = _policy(available=tenths * GIB // 10)
        assert float(policy.variables[LOW_WATERMARK]) <= float(policy.variables[HIGH_WATERMARK])


@pytest.mark.parametrize("available", [32 * GIB // 10, 3 * GIB, 2 * GIB, 0])
def test_a_budget_under_one_gib_is_refused_with_both_ways_out(available):
    with pytest.raises(MpsMemoryRefused) as raised:
        _policy(available=available)
    message = str(raised.value)
    assert "Quit what is holding memory" in message
    assert "LITETUNE_DEVICE=cpu" in message
    assert f"{available / GIB:.2f} GiB available" in message
    # The measure is named, so the reader can recompute it.
    assert "vm.page_pageable_external_count" in message
    assert "free memory" not in message.lower()


def test_a_budget_of_exactly_one_gib_runs():
    # 4 GiB available, less 3: the floor itself is not below the floor.
    policy = _policy(available=4 * GIB)
    assert policy.budget == GIB
    assert policy.variables[HIGH_WATERMARK] == "0.062500"


@pytest.mark.parametrize(("level", "name"), [(2, "warn"), (4, "critical")])
def test_a_mac_under_memory_pressure_is_refused_whatever_is_available(level, name):
    with pytest.raises(MpsMemoryRefused) as raised:
        _policy(available=64 * GIB, pressure=level)
    message = str(raised.value)
    assert f"kern.memorystatus_vm_pressure_level is {level} ({name})" in message
    assert "LITETUNE_DEVICE=cpu" in message


@pytest.mark.parametrize("level", [0, 3, 8])
def test_a_pressure_level_litetune_does_not_know_is_refused(level):
    with pytest.raises(MpsMemoryRefused, match="not a level litetune knows"):
        _policy(pressure=level)


def test_the_operators_variables_are_kept_and_recorded_as_theirs():
    environ = {HIGH_WATERMARK: "0.5", MPS_FALLBACK: "1"}
    policy = _policy(environ)
    assert policy.variables[HIGH_WATERMARK] == "0.5"
    assert policy.variables[MPS_FALLBACK] == "1"
    # The low ratio follows the high one that will actually be in force.
    assert policy.variables[LOW_WATERMARK] == "0.400000"
    assert set(policy.user_set) == {HIGH_WATERMARK, MPS_FALLBACK}
    assert policy.child_env == {LOW_WATERMARK: "0.400000"}
    recorded = policy.as_dict()["variables"]
    assert recorded[HIGH_WATERMARK] == {"value": "0.5", "set_by": "user"}
    assert recorded[LOW_WATERMARK] == {"value": "0.400000", "set_by": "litetune"}
    assert recorded[MPS_FALLBACK] == {"value": "1", "set_by": "user"}


def test_an_operators_low_ratio_above_the_high_one_is_refused():
    with pytest.raises(MpsMemoryRefused, match="only between 0 and the high one"):
        _policy({LOW_WATERMARK: "1.4"})


def test_an_operators_ratio_that_is_not_a_number_is_refused():
    with pytest.raises(MpsMemoryRefused, match="not a number"):
        _policy({HIGH_WATERMARK: "most of it"})


def test_the_record_carries_every_input_and_both_ratios():
    record = _policy().as_dict()
    assert record["budget_bytes"] == 13 * GIB
    assert record["recommended_max_memory_bytes"] == 16 * GIB
    assert record["available_bytes"] == 16 * GIB
    assert record["available_measure"] == (
        "(vm.page_free_count + vm.page_speculative_count + vm.page_pageable_external_count "
        "+ vm.page_purgeable_count) * hw.pagesize"
    )
    reading = record["memory_reading"]
    assert reading["hw.pagesize"] == PAGE
    assert sum(reading[name] for name in devices.AVAILABLE_PAGES) * PAGE == 16 * GIB
    assert reading["kern.memorystatus_vm_pressure_level"] == 1
    assert record["pressure_level"] == 1
    assert record["swapusage"] == "total = 0.00M"
    assert record["headroom_bytes"] == 3 * GIB
    assert record["minimum_budget_bytes"] == GIB
    assert record["variables"][HIGH_WATERMARK]["value"] == "0.812500"
    assert record["variables"][LOW_WATERMARK]["value"] == "0.650000"
    # Nothing in the record is named for free memory: what it holds is not.
    assert not [key for key in record if "free" in key]


def test_a_zero_recommendation_is_refused_rather_than_divided_by():
    with pytest.raises(MpsMemoryRefused):
        _policy(recommended=0)


def test_prepare_mps_reads_this_macs_memory():
    policy = prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(FAKE_SYSCTL))
    assert policy.reading.available_bytes == 16 * GIB
    assert policy.reading.pressure_level == 1
    assert policy.swapusage == FAKE_SYSCTL["vm.swapusage"]
    assert policy.budget == 13 * GIB


def test_prepare_mps_refuses_without_metals_recommendation():
    with pytest.raises(MpsMemoryRefused, match="recommended_max_memory"):
        prepare_mps(_probe("mps", mps_recommended_max_memory=None), {}, sysctl=fake_sysctl({}))


@pytest.mark.parametrize("missing", devices.MEMORY_SYSCTLS)
def test_prepare_mps_refuses_when_memory_cannot_be_read(missing):
    values = {k: v for k, v in FAKE_SYSCTL.items() if k != missing}
    with pytest.raises(MpsMemoryRefused, match=missing):
        prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(values))


@pytest.mark.parametrize("key", devices.MEMORY_SYSCTLS)
def test_prepare_mps_refuses_a_reading_that_is_not_a_number(key):
    values = dict(FAKE_SYSCTL, **{key: "lots"})
    with pytest.raises(MpsMemoryRefused, match="not a number"):
        prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(values))


def test_swap_that_cannot_be_read_is_recorded_not_refused():
    values = {k: v for k, v in FAKE_SYSCTL.items() if k != "vm.swapusage"}
    policy = prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(values))
    assert policy.swapusage is None
    assert "vm.swapusage" in (policy.swapusage_error or "")


# -- this Mac's own sysctl ------------------------------------------------------
# The only tests here that start a process: `/usr/sbin/sysctl -n`, read-only,
# for exactly the keys the policy reads. They pin that the keys exist and
# answer integers on the macOS running the suite, which no fake can.


@pytest.mark.skipif(sys.platform != "darwin", reason="sysctl's memory keys are macOS's")
@pytest.mark.parametrize("key", devices.MEMORY_SYSCTLS)
def test_this_macs_sysctl_answers_every_key_the_policy_reads(key):
    assert int(real_read_sysctl(key)) >= 0


@pytest.mark.skipif(sys.platform != "darwin", reason="sysctl's memory keys are macOS's")
def test_this_macs_memory_reads_into_a_reading():
    reading = devices.read_memory(real_read_sysctl)
    assert reading.page_size > 0
    assert reading.available_bytes >= 0
    assert reading.pressure_level in devices.PRESSURE_NAMES
