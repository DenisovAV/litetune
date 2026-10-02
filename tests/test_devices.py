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
        "os_version_source": "platform.mac_ver()",
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


def test_cpu_on_a_working_cuda_box_is_not_a_cuda_build_without_a_device():
    cuda = _probe("cuda", cuda_build="12.4", device_count=1, os="Linux")
    assert not cuda.cuda_build_without_a_device
    forced = apply_device_setting(cuda, "cpu")
    assert forced.device == "cpu"
    assert forced.cuda_build == "12.4"
    assert not forced.cuda_build_without_a_device
    # The probe's own CPU answer on the same build still is one.
    assert _probe("cpu", cuda_build="12.4", device_count=0).cuda_build_without_a_device


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
        "os_version_source": "platform.mac_ver()",
        "machine": "arm64",
        "chip": "Apple M-test",
    }


def test_off_macos_the_version_is_the_release_and_says_so():
    probe = _probe(
        "cpu", os="Linux", os_version="6.8.0-45-generic", os_version_source="platform.release()"
    )
    record = host_record(probe, platform="linux", sysctl=fake_sysctl({}))
    assert record["os_version"] == "6.8.0-45-generic"
    assert record["os_version_source"] == "platform.release()"


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
    assert kwargs["encoding"] == "utf-8"


@pytest.mark.parametrize(
    ("result", "raises", "words"),
    [
        (subprocess.CompletedProcess([], 1, "", "unknown oid"), None, "exited 1"),
        (subprocess.CompletedProcess([], 0, "  \n", ""), None, "printed nothing"),
        (None, FileNotFoundError("sysctl"), "FileNotFoundError"),
        (None, subprocess.TimeoutExpired("sysctl", 10), "TimeoutExpired"),
        (None, UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"), "UnicodeDecode"),
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


@pytest.mark.parametrize("name", [HIGH_WATERMARK, LOW_WATERMARK])
@pytest.mark.parametrize(
    "value",
    # `strtod` would read the first as 0.5 and the rest as 0.0 -- no limit --
    # or as inf and nan; Python's `float` reads "0_5" as 5 and " 1" as 1.
    ["0.5x", "most of it", "", "inf", "nan", "-0.5", "0_5", " 1", "0x1p-1", "1,5"],
)
def test_an_operators_ratio_that_is_not_a_plain_decimal_is_refused(name, value):
    with pytest.raises(MpsMemoryRefused) as raised:
        _policy({name: value})
    message = str(raised.value)
    assert repr(value) in message or value in message
    assert "could not check" in message or "outside 0 to 2.0" in message
    # Nothing is claimed about what torch would make of it.
    assert "torch" not in message


@pytest.mark.parametrize("value", ["2.0001", "3", "1e1", "1e999"])
def test_a_high_ratio_above_two_is_refused(value):
    with pytest.raises(MpsMemoryRefused, match="outside 0 to 2.0"):
        _policy({HIGH_WATERMARK: value})


@pytest.mark.parametrize("value", ["2", "2.0", "1.5", ".5", "5e-1", "0.000", "0"])
def test_a_high_ratio_in_range_is_taken_as_it_is(value):
    policy = _policy({HIGH_WATERMARK: value})
    assert policy.variables[HIGH_WATERMARK] == value
    assert policy.budget_source == "user"


def test_an_invalid_high_ratio_is_refused_before_a_low_one_is_derived_from_it():
    with pytest.raises(MpsMemoryRefused, match=HIGH_WATERMARK):
        _policy({HIGH_WATERMARK: "5"})


def test_the_operators_high_ratio_owns_the_limit_and_skips_the_budget_floor():
    # 2 GiB available would be refused for a budget litetune computed.
    policy = _policy({HIGH_WATERMARK: "0.5"}, available=2 * GIB)
    assert policy.budget_source == "user"
    assert policy.budget == -1 * GIB
    record = policy.as_dict()
    assert record["budget_source"] == "user"
    assert record["computed_budget_bytes"] == -1 * GIB
    assert record["effective_high_bytes"] == 8 * GIB
    assert record["effective_low_bytes"] == int(0.4 * 16 * GIB)
    assert policy.limitations == []


def test_pressure_is_refused_even_under_the_operators_high_ratio():
    with pytest.raises(MpsMemoryRefused, match="pressure_level is 2"):
        _policy({HIGH_WATERMARK: "0.5"}, pressure=2)


@pytest.mark.parametrize("value", ["0", "0.0", "0.000"])
def test_a_high_ratio_of_zero_is_unlimited_and_said_so(value):
    policy = _policy({HIGH_WATERMARK: value})
    record = policy.as_dict()
    assert record["effective_high_bytes"] == "unlimited"
    # No low ratio is derived from no limit: torch's own default applies.
    assert policy.variables[LOW_WATERMARK] is None
    assert LOW_WATERMARK not in policy.child_env
    assert record["variables"][LOW_WATERMARK] == {"value": None, "set_by": None}
    assert "torch's default" in record["effective_low_bytes"]
    (said,) = policy.limitations
    assert f"{HIGH_WATERMARK}={value}" in said
    assert "no upper limit" in said


def test_under_an_unlimited_high_ratio_a_low_one_up_to_two_is_accepted():
    policy = _policy({HIGH_WATERMARK: "0.0", LOW_WATERMARK: "1.9"})
    assert policy.variables[LOW_WATERMARK] == "1.9"
    assert policy.as_dict()["effective_low_bytes"] == int(1.9 * 16 * GIB)
    with pytest.raises(MpsMemoryRefused):
        _policy({HIGH_WATERMARK: "0.0", LOW_WATERMARK: "2.1"})


def test_the_operators_own_valid_low_ratio_is_recorded_as_theirs_and_not_resent():
    policy = _policy({LOW_WATERMARK: "0.5"})
    assert policy.budget_source == "litetune"
    assert policy.variables[LOW_WATERMARK] == "0.5"
    assert LOW_WATERMARK not in policy.child_env
    assert policy.child_env == {HIGH_WATERMARK: "0.812500", MPS_FALLBACK: "0"}
    recorded = policy.as_dict()["variables"][LOW_WATERMARK]
    assert recorded == {"value": "0.5", "set_by": "user"}
    assert policy.as_dict()["effective_low_bytes"] == 8 * GIB


def test_a_low_ratio_of_zero_is_recorded_as_disabled():
    assert _policy({LOW_WATERMARK: "0"}).as_dict()["effective_low_bytes"] == "disabled"


def test_a_low_ratio_above_litetunes_high_one_names_litetune_as_its_author():
    with pytest.raises(MpsMemoryRefused) as raised:
        _policy({LOW_WATERMARK: "0.9"})
    assert "which litetune computed from the available memory" in str(raised.value)


def test_a_low_ratio_above_the_operators_high_one_names_the_environment():
    with pytest.raises(MpsMemoryRefused) as raised:
        _policy({HIGH_WATERMARK: "0.5", LOW_WATERMARK: "0.6"})
    assert "(set in this environment)" in str(raised.value)


@pytest.mark.parametrize("value", ["1", "2", "-1", "+1", "01"])
def test_an_operators_nonzero_fallback_is_allowed_and_said(value):
    policy = _policy({MPS_FALLBACK: value})
    assert policy.fallback_enabled
    (said,) = policy.limitations
    assert f"{MPS_FALLBACK}={value}" in said
    assert "ran on the CPU" in said
    assert "`device: mps` does not mean every operation" in said


@pytest.mark.parametrize("value", ["0", "00", "-0"])
def test_an_operators_zero_fallback_is_theirs_and_says_nothing(value):
    policy = _policy({MPS_FALLBACK: value})
    assert not policy.fallback_enabled
    assert policy.limitations == []
    assert policy.as_dict()["variables"][MPS_FALLBACK] == {"value": value, "set_by": "user"}


@pytest.mark.parametrize("value", ["yes", "1x", "", " 1", "1.0", "2147483648", "true"])
def test_an_operators_fallback_that_is_not_an_integer_is_refused(value):
    with pytest.raises(MpsMemoryRefused, match="not a plain integer"):
        _policy({MPS_FALLBACK: value})


def test_litetunes_fallback_is_zero_and_says_nothing():
    policy = _policy()
    assert policy.variables[MPS_FALLBACK] == "0"
    assert not policy.fallback_enabled
    assert policy.limitations == []


def test_the_summary_and_the_oom_advice_say_who_set_the_limit():
    ours = _policy()
    assert "set by litetune" in ours.summary()
    advice = devices.mps_oom_advice(ours)
    assert "budget litetune set from the memory this Mac had available" in advice
    assert "LITETUNE_DEVICE=cpu" in advice

    theirs = _policy({HIGH_WATERMARK: "0.5"})
    assert "set in this environment" in theirs.summary()
    advice = devices.mps_oom_advice(theirs)
    assert "not by litetune" in advice
    assert "budget litetune set" not in advice
    assert "LITETUNE_DEVICE=cpu" in advice

    unlimited = _policy({HIGH_WATERMARK: "0"})
    assert "no limit" in unlimited.summary()
    assert "no upper limit" in devices.mps_oom_advice(unlimited)

    # Nobody set a policy: nothing is claimed about one.
    advice = devices.mps_oom_advice(None)
    assert "LITETUNE_DEVICE=cpu" in advice
    assert "litetune set" not in advice
    for text in (devices.mps_oom_advice(ours), devices.mps_oom_advice(theirs), advice):
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in text


def test_the_record_carries_every_input_and_both_ratios():
    record = _policy().as_dict()
    assert record["budget_source"] == "litetune"
    assert record["computed_budget_bytes"] == 13 * GIB
    assert record["effective_high_bytes"] == int(0.8125 * 16 * GIB)
    assert record["effective_low_bytes"] == int(0.65 * 16 * GIB)
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
