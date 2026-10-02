"""The parent's side of a stage's device: LITETUNE_DEVICE, host facts, the MPS policy.

No torch, no sysctl, no subprocess: every input is passed in, which is why
`devices.mps_memory_policy` takes its figures as arguments.
"""

from __future__ import annotations

import subprocess
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


def _policy(environ=None, recommended=16 * GIB, memsize=32 * GIB, level=50):
    return mps_memory_policy(
        recommended_max_memory=recommended,
        memsize=memsize,
        memorystatus_level=level,
        environ=environ or {},
        swapusage="total = 0.00M",
    )


def test_the_budget_is_the_free_memory_less_the_headroom_when_that_is_smaller():
    # 50% of 32 GiB is 16 GiB available, less 3 GiB: 13 GiB, under Metal's 16.
    policy = _policy()
    assert policy.budget == 13 * GIB
    assert policy.variables[HIGH_WATERMARK] == "0.812500"
    assert policy.variables[LOW_WATERMARK] == "0.650000"
    assert policy.variables[MPS_FALLBACK] == "0"
    assert policy.child_env == policy.variables
    assert policy.user_set == ()


def test_the_budget_is_metals_recommendation_when_that_is_smaller():
    # 90% of 64 GiB less 3 is far above 16: the ratio is exactly 1.
    policy = _policy(recommended=16 * GIB, memsize=64 * GIB, level=90)
    assert policy.budget == 16 * GIB
    assert policy.variables[HIGH_WATERMARK] == "1.000000"
    assert policy.variables[LOW_WATERMARK] == "0.800000"


def test_the_low_ratio_is_never_above_the_high_one():
    for level in range(10, 101, 7):
        try:
            policy = _policy(level=level)
        except MpsMemoryRefused:
            continue
        assert float(policy.variables[LOW_WATERMARK]) <= float(policy.variables[HIGH_WATERMARK])


@pytest.mark.parametrize(("memsize", "level"), [(8 * GIB, 40), (16 * GIB, 20), (2 * GIB, 100)])
def test_a_budget_under_one_gib_is_refused_with_both_ways_out(memsize, level):
    # 40% of 8 GiB and 20% of 16 GiB are both 3.2 GiB available, 0.2 GiB once
    # 3 GiB is left to everything else; 2 GiB wholly free is negative.
    with pytest.raises(MpsMemoryRefused) as raised:
        _policy(memsize=memsize, level=level)
    message = str(raised.value)
    assert "Free memory" in message
    assert "LITETUNE_DEVICE=cpu" in message
    assert f"{level}%" in message


def test_a_budget_of_exactly_one_gib_runs():
    # 4 GiB wholly free, less 3: the floor itself is not below the floor.
    policy = _policy(memsize=4 * GIB, level=100)
    assert policy.budget == GIB
    assert policy.variables[HIGH_WATERMARK] == "0.062500"


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
    assert record["memsize_bytes"] == 32 * GIB
    assert record["memorystatus_level"] == 50
    assert record["swapusage"] == "total = 0.00M"
    assert record["headroom_bytes"] == 3 * GIB
    assert record["minimum_budget_bytes"] == GIB
    assert record["variables"][HIGH_WATERMARK]["value"] == "0.812500"
    assert record["variables"][LOW_WATERMARK]["value"] == "0.650000"


def test_a_zero_recommendation_is_refused_rather_than_divided_by():
    with pytest.raises(MpsMemoryRefused):
        _policy(recommended=0)


def test_prepare_mps_reads_this_macs_memory():
    policy = prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(FAKE_SYSCTL))
    assert policy.memsize == 32 * GIB
    assert policy.memorystatus_level == 50
    assert policy.swapusage == FAKE_SYSCTL["vm.swapusage"]
    assert policy.budget == 13 * GIB


def test_prepare_mps_refuses_without_metals_recommendation():
    with pytest.raises(MpsMemoryRefused, match="recommended_max_memory"):
        prepare_mps(_probe("mps", mps_recommended_max_memory=None), {}, sysctl=fake_sysctl({}))


@pytest.mark.parametrize("missing", ["hw.memsize", "kern.memorystatus_level"])
def test_prepare_mps_refuses_when_memory_cannot_be_read(missing):
    values = {k: v for k, v in FAKE_SYSCTL.items() if k != missing}
    with pytest.raises(MpsMemoryRefused, match=missing):
        prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(values))


def test_prepare_mps_refuses_a_reading_that_is_not_a_number():
    values = dict(FAKE_SYSCTL, **{"kern.memorystatus_level": "lots"})
    with pytest.raises(MpsMemoryRefused, match="not a number"):
        prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(values))


def test_swap_that_cannot_be_read_is_recorded_not_refused():
    values = {k: v for k, v in FAKE_SYSCTL.items() if k != "vm.swapusage"}
    policy = prepare_mps(_probe("mps"), {}, sysctl=fake_sysctl(values))
    assert policy.swapusage is None
    assert "vm.swapusage" in (policy.swapusage_error or "")
