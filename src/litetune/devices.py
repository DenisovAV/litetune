"""What the parent decides around a stage's device, before the stage starts.

`envs.resolve_device` asks a stage environment's own torch what it can reach:
CUDA, then Apple's GPU through Metal (MPS), then the CPU. This module holds the
three things the parent adds to that answer, each of them before a training or
reference-generation process is started rather than inside it:

- the operator's `LITETUNE_DEVICE`, which can force the CPU and is refused for
  any value it does not know -- an environment variable rather than a CLI flag,
  so the README's "there is no `--device` flag" stays true of the commands;
- the host facts a report records beside the device: operating system, its
  version, the machine, and on macOS the chip;
- the memory policy for a child that will use MPS, which sets torch's MPS
  allocator watermarks from the memory this Mac has available when the run
  starts.

The script-side fallbacks in `tune._TRAIN_SCRIPT` and
`evaluate._HF_GENERATE_SCRIPT` -- used only when the parent could not probe --
never choose MPS, because the memory policy lives here and a child that chose
MPS for itself would run without it.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from litetune import envs

logger = logging.getLogger(__name__)

# The operator's escape hatch. Unset or "auto" leaves the device to the probe;
# "cpu" puts the run on the CPU whatever the probe says. Anything else is
# refused rather than read as "auto": a misspelt "cpu" that quietly trained on
# the GPU is exactly the run the variable exists to prevent.
DEVICE_VARIABLE = "LITETUNE_DEVICE"
DEVICE_SETTINGS = ("auto", "cpu")


class DeviceSettingError(ValueError):
    """`LITETUNE_DEVICE` holds a value litetune does not accept."""


def device_setting(environ: Mapping[str, str]) -> str:
    """The operator's `LITETUNE_DEVICE`: "auto" when unset. Raises on anything else.

    Exact, not case-folded or stripped: the accepted values are named in the
    refusal, and a value that is almost one of them is still not one.
    """
    value = environ.get(DEVICE_VARIABLE)
    if value is None:
        return "auto"
    if value not in DEVICE_SETTINGS:
        raise DeviceSettingError(
            f"{DEVICE_VARIABLE}={value!r} is not a setting litetune accepts. Accepted: "
            f"{', '.join(DEVICE_SETTINGS)} (or leave it unset, which is auto: CUDA, then MPS, "
            "then the CPU, by what this environment's torch can reach)"
        )
    return value


def apply_device_setting(probe: envs.DeviceProbe, setting: str) -> envs.DeviceProbe:
    """The probe with the operator's setting applied. Records that it was.

    "cpu" replaces whatever the probe answered, including no answer at all --
    the operator has said where the run goes, so a probe that could not say is
    no longer the reason the device is unknown. The probe's own finding stays
    in `detail`, so a Mac forced off a working MPS still says it had one.
    """
    if setting != "cpu":
        return probe
    return replace(
        probe,
        device="cpu",
        source=DEVICE_VARIABLE,
        detail=f"{DEVICE_VARIABLE}=cpu places this run on the CPU; the probe found: {probe.detail}",
    )


# ---------------------------------------------------------------------------
# What the host is
# ---------------------------------------------------------------------------

SYSCTL = "/usr/sbin/sysctl"
SYSCTL_TIMEOUT_S = 10


class HostReadError(RuntimeError):
    """A host fact could not be read."""


def read_sysctl(name: str) -> str:
    """One `sysctl -n` value, stripped. Raises `HostReadError` if there is none."""
    try:
        done = subprocess.run(
            [SYSCTL, "-n", name],
            capture_output=True,
            text=True,
            timeout=SYSCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HostReadError(f"sysctl {name}: {type(exc).__name__}: {exc}") from exc
    if done.returncode != 0:
        raise HostReadError(
            f"sysctl {name} exited {done.returncode}: {(done.stderr or '').strip()[:200]}"
        )
    value = (done.stdout or "").strip()
    if not value:
        raise HostReadError(f"sysctl {name} printed nothing")
    return value


Sysctl = Callable[[str], str]


def host_record(
    probe: envs.DeviceProbe, platform: str | None = None, sysctl: Sysctl | None = None
) -> dict[str, Any]:
    """Operating system, version, machine and chip, for a report.

    The first three are what the probe's interpreter answered, so they are
    `None` where the probe could not answer. The chip is read here, in the
    parent, and only on macOS -- `machdep.cpu.brand_string`, the key
    `evaluate.cpu_report` reads for the runtime side. A chip that cannot be
    read is recorded with the reason, not left out: this is a description,
    and a run is not refused for an incomplete one.
    """
    read = sysctl if sysctl is not None else read_sysctl
    platform = platform if platform is not None else sys.platform
    record: dict[str, Any] = {
        "os": probe.os,
        "os_version": probe.os_version,
        "machine": probe.machine,
        "chip": None,
    }
    if platform == "darwin":
        try:
            record["chip"] = read("machdep.cpu.brand_string")
        except HostReadError as exc:
            logger.warning("could not read this Mac's chip: %s", exc)
            record["chip_error"] = str(exc)
    return record


# ---------------------------------------------------------------------------
# The MPS memory policy
# ---------------------------------------------------------------------------
#
# On Apple silicon the GPU allocates from the same memory as everything else,
# and torch's MPS allocator by default lets a process go past Metal's
# recommended working set: its default high watermark ratio is 1.7 and its
# default low watermark ratio on unified memory 1.4
# (torch/include/ATen/mps/MPSAllocator.h, `default_high_watermark_ratio` and
# `default_low_watermark_ratio_unified`, torch 2.5.1 as `envs.TRAIN` pins it).
# So the parent sets both ratios from the memory the machine has available
# when the run starts, in the child's environment, before the child exists.
# The header records the low limit as fixed "at the time we initialize the
# allocator"; where the variables are read is MPSAllocator.mm, which the wheel
# does not ship.
#
# budget = min(R, available - headroom)
#
# where R is `torch.mps.recommended_max_memory()` from the probe and
# `available` is
#
#   (vm.page_free_count + vm.page_speculative_count
#    + vm.page_pageable_external_count + vm.page_purgeable_count) * hw.pagesize
#
# the sum XNU's jetsam counts as available (osfmk/vm/vm_page.h, around lines
# 1533-1539, in the XNU source the review read; not checked against a copy
# here). It is not `kern.memorystatus_level`, which an earlier draft used: that
# is XNU's pressure level, computed from active + inactive + free + speculative
# pages over the total (vm_pageout.c, `vm_pressure_response`, and vm_page.h's
# `AVAILABLE_NON_COMPRESSED_MEMORY`, same reading), so it counts other
# processes' active memory as available. Read on 2026-10-02 on a 24 GiB Mac
# (macOS 26.5.1, arm64): `kern.memorystatus_level` 32, which the old formula
# made 7.68 GiB available and a 4.68 GiB budget, while the sum above came to
# 2.87 GiB, `kern.memorystatus_vm_pressure_level` was 2 and `vm.swapusage`
# showed 23993 of 25600 MB in use. Every key above answered `sysctl -n` there.
#
# A Mac the kernel already reports under memory pressure is refused outright,
# whatever the sum says: `kern.memorystatus_vm_pressure_level` answers 1, 2 or 4
# for normal, warn and critical -- the values `DISPATCH_MEMORYPRESSURE_NORMAL`,
# `_WARN` and `_CRITICAL` carry in the SDK's dispatch/source.h; that the sysctl
# reports in those values is from XNU's kern_memorystatus_notify.c and
# sys/event.h, as the review read them. Anything but 1 is refused.
#
# The high ratio is budget / R, never above 1 since the budget is at most R.
# The low ratio has to be set as well, and not above the high one: the header
# documents it as a value "between 0 to m_high_watermark_ratio", the 1.4
# default is above any high ratio this policy produces, and libtorch_cpu in
# that wheel carries the message "invalid low watermark ratio".
#
# `PYTORCH_ENABLE_MPS_FALLBACK=0`: an operator MPS does not implement must
# raise, not run on the CPU, which would leave a run recorded as "mps" that
# computed part of itself somewhere else.
#
# None of this has been measured on a training run. It bounds what the
# allocator may take; whether a given model then fits is the run's to find out.
GIB = 1024**3
MPS_HEADROOM_BYTES = 3 * GIB
MPS_MIN_BUDGET_BYTES = 1 * GIB
MPS_LOW_OVER_HIGH = 0.8
HIGH_WATERMARK = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
LOW_WATERMARK = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
MPS_FALLBACK = "PYTORCH_ENABLE_MPS_FALLBACK"
MPS_VARIABLES = (HIGH_WATERMARK, LOW_WATERMARK, MPS_FALLBACK)
_RATIO_FORMAT = "{:.6f}"

PAGE_SIZE = "hw.pagesize"
# The four page counts whose sum is the available memory, in the order above.
AVAILABLE_PAGES = (
    "vm.page_free_count",
    "vm.page_speculative_count",
    "vm.page_pageable_external_count",
    "vm.page_purgeable_count",
)
PRESSURE_LEVEL = "kern.memorystatus_vm_pressure_level"
PRESSURE_NORMAL = 1
PRESSURE_NAMES = {1: "normal", 2: "warn", 4: "critical"}
# Every key `read_memory` reads, for a test that asks this Mac's own sysctl.
MEMORY_SYSCTLS = (PAGE_SIZE, *AVAILABLE_PAGES, PRESSURE_LEVEL)


class MpsMemoryRefused(RuntimeError):
    """The memory policy could not be applied, so the run does not start on MPS."""


@dataclass(frozen=True)
class MemoryReading:
    """This Mac's memory as the kernel counted it when the run was about to start."""

    page_size: int
    # Each of `AVAILABLE_PAGES`, by name, in pages.
    pages: Mapping[str, int]
    pressure_level: int

    @property
    def available_bytes(self) -> int:
        return sum(self.pages[name] for name in AVAILABLE_PAGES) * self.page_size

    def as_dict(self) -> dict[str, Any]:
        return {
            PAGE_SIZE: self.page_size,
            **{name: self.pages[name] for name in AVAILABLE_PAGES},
            PRESSURE_LEVEL: self.pressure_level,
        }


def memory_reading(page_size: int, pages: Mapping[str, int], pressure_level: int) -> MemoryReading:
    """A `MemoryReading`, refusing one that names a page count it should not or misses one."""
    if set(pages) != set(AVAILABLE_PAGES):
        raise ValueError(
            f"a memory reading takes exactly {list(AVAILABLE_PAGES)}, got {list(pages)}"
        )
    return MemoryReading(page_size=page_size, pages=dict(pages), pressure_level=pressure_level)


@dataclass(frozen=True)
class MpsMemory:
    """The policy one MPS run started under, and the inputs it was computed from."""

    recommended_max_memory: int
    reading: MemoryReading
    budget: int
    # The value of each of `MPS_VARIABLES` that reaches the child.
    variables: Mapping[str, str]
    # Which of them the operator had already set. Those are passed through as
    # they were, never overridden, and recorded as theirs.
    user_set: tuple[str, ...] = ()
    swapusage: str | None = None
    swapusage_error: str | None = None

    @property
    def child_env(self) -> dict[str, str]:
        """What litetune adds to the child's environment: everything not the operator's."""
        return {k: v for k, v in self.variables.items() if k not in self.user_set}

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget,
            "recommended_max_memory_bytes": self.recommended_max_memory,
            "available_bytes": self.reading.available_bytes,
            "available_measure": AVAILABLE_MEASURE,
            "memory_reading": self.reading.as_dict(),
            "pressure_level": self.reading.pressure_level,
            "headroom_bytes": MPS_HEADROOM_BYTES,
            "minimum_budget_bytes": MPS_MIN_BUDGET_BYTES,
            "swapusage": self.swapusage,
            "swapusage_error": self.swapusage_error,
            "variables": {
                name: {
                    "value": value,
                    "set_by": "user" if name in self.user_set else "litetune",
                }
                for name, value in self.variables.items()
            },
        }


# What `available_bytes` is, in the record beside it.
AVAILABLE_MEASURE = f"({' + '.join(AVAILABLE_PAGES)}) * {PAGE_SIZE}"


def _ratio(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise MpsMemoryRefused(
            f"{name}={value!r} is set in this environment and is not a number; torch's MPS "
            "allocator would refuse it when the run starts. Unset it, or set a ratio"
        ) from None


def _pressure_refusal(level: int) -> str:
    name = PRESSURE_NAMES.get(level)
    said = f"{level} ({name})" if name else f"{level}, which is not a level litetune knows"
    return (
        f"{PRESSURE_LEVEL} is {said}: litetune starts an MPS run only at level "
        f"{PRESSURE_NORMAL} ({PRESSURE_NAMES[PRESSURE_NORMAL]}). Quit what is holding memory "
        f"and run again, or set {DEVICE_VARIABLE}=cpu to run on the CPU"
    )


def mps_memory_policy(
    *,
    recommended_max_memory: int,
    reading: MemoryReading,
    environ: Mapping[str, str],
    swapusage: str | None = None,
    swapusage_error: str | None = None,
) -> MpsMemory:
    """The watermark ratios for a child about to use MPS. Pure: every input is passed in.

    Raises `MpsMemoryRefused` when the kernel reports memory pressure, when
    the budget is under `MPS_MIN_BUDGET_BYTES`, or when a ratio the operator
    set cannot be one torch accepts.
    """
    if recommended_max_memory <= 0:
        raise MpsMemoryRefused(
            f"torch reported a recommended MPS working set of {recommended_max_memory} bytes, "
            "which no budget can be a share of"
        )
    if reading.pressure_level != PRESSURE_NORMAL:
        raise MpsMemoryRefused(_pressure_refusal(reading.pressure_level))
    available = reading.available_bytes
    budget = min(recommended_max_memory, available - MPS_HEADROOM_BYTES)
    if budget < MPS_MIN_BUDGET_BYTES:
        raise MpsMemoryRefused(
            f"the MPS memory budget is {budget / GIB:.2f} GiB, under the "
            f"{MPS_MIN_BUDGET_BYTES / GIB:.0f} GiB litetune starts an MPS run with: this Mac "
            f"has {available / GIB:.2f} GiB available ({AVAILABLE_MEASURE}), less "
            f"{MPS_HEADROOM_BYTES / GIB:.0f} GiB left to everything else, against the "
            f"{recommended_max_memory / GIB:.1f} GiB Metal recommends. Quit what is holding "
            f"memory and run again, or set {DEVICE_VARIABLE}=cpu to run on the CPU"
        )

    user_set = tuple(name for name in MPS_VARIABLES if name in environ)
    high_text = environ.get(HIGH_WATERMARK, _RATIO_FORMAT.format(budget / recommended_max_memory))
    high = _ratio(HIGH_WATERMARK, high_text)
    if LOW_WATERMARK in environ:
        low_text = environ[LOW_WATERMARK]
        low = _ratio(LOW_WATERMARK, low_text)
        if low > high:
            owner = (
                "set in this environment"
                if HIGH_WATERMARK in user_set
                else "computed from the available memory"
            )
            raise MpsMemoryRefused(
                f"{LOW_WATERMARK}={low_text} is set in this environment and is above the high "
                f"watermark ratio {high_text} ({owner}); torch's MPS allocator takes a low ratio "
                f"only between 0 and the high one. Unset it, or set it at or below {high_text}"
            )
    else:
        low_text = _RATIO_FORMAT.format(high * MPS_LOW_OVER_HIGH)
    return MpsMemory(
        recommended_max_memory=recommended_max_memory,
        reading=reading,
        budget=budget,
        variables={
            HIGH_WATERMARK: high_text,
            LOW_WATERMARK: low_text,
            MPS_FALLBACK: environ.get(MPS_FALLBACK, "0"),
        },
        user_set=user_set,
        swapusage=swapusage,
        swapusage_error=swapusage_error,
    )


def _sysctl_int(read: Sysctl, name: str) -> int:
    try:
        text = read(name)
    except HostReadError as exc:
        raise MpsMemoryRefused(
            f"{exc}, so the MPS memory budget could not be computed. Set {DEVICE_VARIABLE}=cpu "
            "to run on the CPU"
        ) from exc
    try:
        return int(text)
    except ValueError:
        raise MpsMemoryRefused(
            f"sysctl {name} printed {text[:80]!r}, not a number, so the MPS memory budget could "
            f"not be computed. Set {DEVICE_VARIABLE}=cpu to run on the CPU"
        ) from None


def read_memory(sysctl: Sysctl | None = None) -> MemoryReading:
    """This Mac's page size, available page counts and pressure level, from sysctl.

    Raises `MpsMemoryRefused` for any key that cannot be read as an integer.
    """
    read = sysctl if sysctl is not None else read_sysctl
    return memory_reading(
        page_size=_sysctl_int(read, PAGE_SIZE),
        pages={name: _sysctl_int(read, name) for name in AVAILABLE_PAGES},
        pressure_level=_sysctl_int(read, PRESSURE_LEVEL),
    )


def prepare_mps(
    probe: envs.DeviceProbe, environ: Mapping[str, str], sysctl: Sysctl | None = None
) -> MpsMemory:
    """Read this Mac's memory and compute the policy for an MPS child.

    Raises `MpsMemoryRefused` rather than starting the child without a policy:
    a budget that could not be computed is not a reason to let the allocator
    take its defaults, which is what the policy exists to replace.
    """
    read = sysctl if sysctl is not None else read_sysctl
    recommended = probe.mps_recommended_max_memory
    if recommended is None:
        raise MpsMemoryRefused(
            "the device probe answered mps without torch.mps.recommended_max_memory(), so the "
            f"MPS memory budget could not be computed. Set {DEVICE_VARIABLE}=cpu to run on the CPU"
        )
    reading = read_memory(read)
    swapusage: str | None = None
    swapusage_error: str | None = None
    try:
        swapusage = read("vm.swapusage")
    except HostReadError as exc:
        # Recorded, not refused: swap use at the start describes the run and
        # takes no part in the budget.
        logger.warning("could not read swap usage: %s", exc)
        swapusage_error = str(exc)
    return mps_memory_policy(
        recommended_max_memory=recommended,
        reading=reading,
        environ=environ,
        swapusage=swapusage,
        swapusage_error=swapusage_error,
    )
