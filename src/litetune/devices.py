"""What the parent decides around a stage's device, before the stage starts.

`envs.resolve_device` asks a stage environment's own torch what it can reach:
CUDA, then Apple's GPU through Metal (MPS), then the CPU. This module holds the
three things the parent adds to that answer, each of them before a training or
reference-generation process is started rather than inside it:

- the operator's `LITETUNE_DEVICE`, which can force the CPU and is refused for
  any value it does not know;
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
import math
import re
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
            encoding="utf-8",
            timeout=SYSCTL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError) as exc:
        # A value that is not UTF-8 is a value this could not read, the same
        # as one sysctl did not print.
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
    `None` where the probe could not answer. The version is macOS's product
    version on a Mac and `platform.release()` elsewhere -- on Linux the
    kernel's release, not the distribution's -- and `os_version_source` says
    which. The chip is read here, in the
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
        "os_version_source": probe.os_version_source,
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
# The header describes the low limit as the "low watermark size limit (in
# Bytes) at the time we initialize the allocator"; where the variables are read
# is MPSAllocator.mm, which the wheel does not ship.
#
# budget = min(R, available - headroom)
#
# where R is `torch.mps.recommended_max_memory()` from the probe and
# `available` is
#
#   (vm.page_free_count + vm.page_speculative_count
#    + vm.page_pageable_external_count + vm.page_purgeable_count) * hw.pagesize
#
# pages nothing else is holding on to: free, speculative, file-backed pageable
# and purgeable. It is modelled on, and not the same as, what XNU's jetsam
# counts as available. In the copy of osfmk/vm/vm_page.h the review read
# (lines 1531-1539), jetsam's `VM_CHECK_MEMORYSTATUS` sums
# `vm_page_pageable_external_count + vm_page_free_count` and the secluded
# pages over target, and adds `vm_page_purgeable_count` only when dynamic
# paging is off; it counts no speculative pages. Jetsam is an embedded option
# (XNU config/MASTER: "enable jetsam - used on embedded"), so macOS computes
# none of this itself. The sum here adds speculative pages and counts
# purgeable ones whether or not dynamic paging is on; both choices are this
# policy's, not XNU's.
#
# It is not `kern.memorystatus_level`, which an earlier draft used. On macOS
# that is XNU's pressure level: `vm_pressure_response` in vm_pageout.c sets it
# to available pages over total pages, and without jetsam the available count
# is `AVAILABLE_NON_COMPRESSED_MEMORY` -- active + inactive + free +
# speculative (vm_page.h lines 1528 and 1549 in the same copy). So it counts
# other processes' active memory as available. Read on 2026-10-02 on a 24 GiB Mac
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
# `PYTORCH_ENABLE_MPS_FALLBACK=0` is torch's own default made explicit:
# libtorch_cpu's message for an operator MPS does not implement says "you can
# set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU
# as a fallback for this op", so unset, it raises. Written out so the record
# says which it was rather than leaving it to the environment.
#
# The operator may set any of the three, and what they set is theirs: passed
# through, never overridden, and recorded as theirs. A high ratio they set is
# the limit, so the 1 GiB floor on the computed budget does not apply to it;
# the computed budget is still recorded beside it. Each value is checked
# before anything starts, and one litetune cannot read is refused as a fact
# about the configuration, not passed on. The ranges are torch 2.5.1's. The
# upper bound of 2.0 on the high ratio is `default_high_watermark_upper_bound`
# in the shipped MPSAllocator.h, and 0.0 meaning "no limit" is that header's
# comment on `m_high_watermark_ratio`. That both are read with `strtod`, that
# the low ratio must be within [0, high] -- [0, 2.0] when high is 0.0 -- and
# that the fallback is read with `std::stoi` is from upstream source at
# v2.5.1 (aten/src/ATen/mps/MPSAllocator.mm and MPSFallback.mm), which the
# wheel does not ship. `strtod` reads "0.5x" as 0.5 and "most" as 0.0 -- no
# limit at all -- so only a plain decimal is accepted here, a form on which
# `strtod` and Python's `float` agree; `std::stoi` likewise reads "1x" as 1
# and throws on "x", so only a plain integer is.
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
HIGH_RATIO_UPPER_BOUND = 2.0
_DECIMAL = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INTEGER = re.compile(r"[+-]?[0-9]+")
# `std::stoi` returns an `int`; a value outside a 32-bit one throws.
_INT_MIN, _INT_MAX = -(2**31), 2**31 - 1

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
    # min(R, available - headroom), computed whoever set the high ratio. When
    # the operator set it, this is what litetune would have set, not the limit.
    budget: int
    # The value of each of `MPS_VARIABLES` that reaches the child. `None` for a
    # low ratio nobody sets: under an operator's high ratio of 0.0 litetune
    # derives none, and torch's own default applies.
    variables: Mapping[str, str | None]
    # Which of them the operator had already set. Those are passed through as
    # they were, never overridden, and recorded as theirs.
    user_set: tuple[str, ...] = ()
    swapusage: str | None = None
    swapusage_error: str | None = None

    @property
    def child_env(self) -> dict[str, str]:
        """What litetune adds to the child's environment: everything not the operator's."""
        return {k: v for k, v in self.variables.items() if k not in self.user_set and v is not None}

    @property
    def budget_source(self) -> str:
        """Who set the limit in force: "user" when the operator set the high ratio."""
        return "user" if HIGH_WATERMARK in self.user_set else "litetune"

    def _ratio(self, name: str) -> float | None:
        value = self.variables.get(name)
        return float(value) if value is not None else None

    @property
    def effective_high(self) -> int | str:
        """The high limit in bytes, ratio x R truncated, or "unlimited" at 0.0."""
        high = self._ratio(HIGH_WATERMARK)
        if high is None:
            # `mps_memory_policy` always sets one; a hand-built record may not.
            raise ValueError(f"this MPS policy carries no {HIGH_WATERMARK}")
        return "unlimited" if high == 0.0 else int(high * self.recommended_max_memory)

    @property
    def effective_low(self) -> int | str:
        """The low limit in bytes; "disabled" at 0.0; torch's default when nobody set one."""
        low = self._ratio(LOW_WATERMARK)
        if low is None:
            return "not set: torch's default low ratio applies"
        # MPSAllocator.h: "setting 0.0 disables adaptive commit and garbage
        # collection".
        return "disabled" if low == 0.0 else int(low * self.recommended_max_memory)

    @property
    def fallback_enabled(self) -> bool:
        return int(self.variables[MPS_FALLBACK] or "0") != 0

    @property
    def limitations(self) -> list[str]:
        """What a report must say about this policy beyond its record."""
        said = []
        if self.effective_high == "unlimited":
            said.append(
                f"{HIGH_WATERMARK}={self.variables[HIGH_WATERMARK]} was set in this environment, "
                "which disables torch's MPS memory limit (MPSAllocator.h: \"disables high "
                'watermark limit (may cause system failure if system-wide OOM occurs)"). '
                "This run's MPS allocations had no upper limit, from litetune or from torch; "
                f"litetune's computed budget, {self.budget / GIB:.2f} GiB, was not applied"
            )
        if self.fallback_enabled:
            said.append(
                f"{MPS_FALLBACK}={self.variables[MPS_FALLBACK]} was set in this environment, so "
                "any operation MPS does not implement ran on the CPU instead of raising (torch's "
                'own message offers the variable "to use the CPU as a fallback for this op"). '
                "`device: mps` does not mean every operation of this run ran on MPS"
            )
        return said

    def summary(self) -> str:
        """One line for the event stream, worded by who set the limit."""
        if self.budget_source == "litetune":
            return (
                f"MPS memory budget {self.budget / GIB:.1f} GiB of the "
                f"{self.recommended_max_memory / GIB:.1f} GiB Metal recommends, set by litetune "
                "from the memory available now"
            )
        limit = self.effective_high
        said = "no limit" if isinstance(limit, str) else f"{limit / GIB:.1f} GiB"
        return (
            f"MPS high watermark set in this environment ({HIGH_WATERMARK}="
            f"{self.variables[HIGH_WATERMARK]}): {said}; litetune's computed budget would have "
            f"been {self.budget / GIB:.1f} GiB"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "budget_source": self.budget_source,
            "computed_budget_bytes": self.budget,
            "effective_high_bytes": self.effective_high,
            "effective_low_bytes": self.effective_low,
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
                    "set_by": (
                        "user"
                        if name in self.user_set
                        else "litetune"
                        if value is not None
                        else None
                    ),
                }
                for name, value in self.variables.items()
            },
        }


# What `available_bytes` is, in the record beside it.
AVAILABLE_MEASURE = f"({' + '.join(AVAILABLE_PAGES)}) * {PAGE_SIZE}"


def mps_oom_advice(memory: MpsMemory | None) -> str:
    """What to do about an MPS out-of-memory ending, worded by who set the limit."""
    cpu = f"set {DEVICE_VARIABLE}=cpu to run on the CPU"
    if memory is None:
        return cpu[0].upper() + cpu[1:]
    if memory.budget_source == "litetune":
        return (
            f"the {memory.budget / GIB:.1f} GiB budget litetune set from the memory this Mac had "
            f"available when the run started was not enough; quit what is holding memory and "
            f"run again, or {cpu}"
        )
    high = memory.variables[HIGH_WATERMARK]
    if memory.effective_high == "unlimited":
        return (
            f"{HIGH_WATERMARK}={high} was set in this environment, so torch's MPS allocator had "
            f"no upper limit, and the memory still ran out; quit what is holding memory, or {cpu}"
        )
    return (
        f"the high watermark ratio was set in this environment ({HIGH_WATERMARK}={high}), not "
        f"by litetune; raise or unset it, or {cpu}"
    )


def _decimal(name: str, text: str, upper: float) -> float:
    """`text` as a ratio between 0 and `upper`, or a refusal naming why it is not one."""
    if not _DECIMAL.fullmatch(text):
        raise MpsMemoryRefused(
            f"{name}={text!r} is set in this environment and is not a plain decimal number, so "
            "litetune could not check it before the run. Unset it, or set a number between 0 "
            f"and {upper}"
        )
    value = float(text)
    if not (math.isfinite(value) and 0.0 <= value <= upper):
        raise MpsMemoryRefused(
            f"{name}={text} is set in this environment and is outside 0 to {upper}, the range "
            "litetune accepts for it. Unset it, or set a number in that range"
        )
    return value


def _fallback(text: str) -> int:
    if not _INTEGER.fullmatch(text) or not _INT_MIN <= int(text) <= _INT_MAX:
        raise MpsMemoryRefused(
            f"{MPS_FALLBACK}={text!r} is set in this environment and is not a plain integer, "
            "so litetune could not tell whether it turns the CPU fallback on. Unset it, or set "
            "0 or 1"
        )
    return int(text)


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
    litetune sets the limit and the budget is under `MPS_MIN_BUDGET_BYTES`, or
    when a value the operator set is not one litetune can check.
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
    user_set = tuple(name for name in MPS_VARIABLES if name in environ)

    # The high ratio first: the low one is checked against it, and is never
    # derived from one that has not been checked.
    if HIGH_WATERMARK in environ:
        high_text = environ[HIGH_WATERMARK]
        high = _decimal(HIGH_WATERMARK, high_text, HIGH_RATIO_UPPER_BOUND)
    else:
        if budget < MPS_MIN_BUDGET_BYTES:
            raise MpsMemoryRefused(
                f"the MPS memory budget is {budget / GIB:.2f} GiB, under the "
                f"{MPS_MIN_BUDGET_BYTES / GIB:.0f} GiB litetune starts an MPS run with: this Mac "
                f"has {available / GIB:.2f} GiB available ({AVAILABLE_MEASURE}), less "
                f"{MPS_HEADROOM_BYTES / GIB:.0f} GiB left to everything else, against the "
                f"{recommended_max_memory / GIB:.1f} GiB Metal recommends. Quit what is holding "
                f"memory and run again, or set {DEVICE_VARIABLE}=cpu to run on the CPU"
            )
        high_text = _RATIO_FORMAT.format(budget / recommended_max_memory)
        high = float(high_text)

    low_text: str | None
    if LOW_WATERMARK in environ:
        low_text = environ[LOW_WATERMARK]
        low = _decimal(LOW_WATERMARK, low_text, HIGH_RATIO_UPPER_BOUND)
        ceiling = high if high > 0.0 else HIGH_RATIO_UPPER_BOUND
        if low > ceiling:
            owner = (
                "set in this environment"
                if HIGH_WATERMARK in user_set
                else "which litetune computed from the available memory"
            )
            raise MpsMemoryRefused(
                f"{LOW_WATERMARK}={low_text} is set in this environment and is above the high "
                f"watermark ratio {high_text} ({owner}); litetune accepts a low ratio only "
                f"between 0 and the high one. Unset it, or set it at or below {high_text}"
            )
    elif high == 0.0:
        # No limit to be a share of: whatever litetune derived would be its
        # own choice of a soft limit the operator did not ask for.
        low_text = None
    else:
        low_text = _RATIO_FORMAT.format(high * MPS_LOW_OVER_HIGH)

    fallback_text = environ.get(MPS_FALLBACK, "0")
    _fallback(fallback_text)
    return MpsMemory(
        recommended_max_memory=recommended_max_memory,
        reading=reading,
        budget=budget,
        variables={
            HIGH_WATERMARK: high_text,
            LOW_WATERMARK: low_text,
            MPS_FALLBACK: fallback_text,
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


# ---------------------------------------------------------------------------
# What the child saw and what it used
# ---------------------------------------------------------------------------
#
# Source concatenated into both scripts that can run on mps --
# `tune._TRAIN_SCRIPT` and `evaluate._HF_GENERATE_SCRIPT` -- because neither
# may import litetune: each runs in a stage environment that has only torch.
#
# `environment_seen` is what the child itself read for the three MPS
# variables, so the parent can compare it with what it sent and what the
# child inherited (`environment_mismatch`) instead of assuming the two agree.
#
# The memory record holds the largest of torch's two MPS counters over the
# samples taken, and names where they were taken: `current_allocated_memory()`
# is what tensors occupy, `driver_allocated_memory()` everything Metal holds
# for the process, cached blocks included. Between samples the counters were
# not read, so this is the largest value seen at those points, not a peak.
MPS_SCRIPT_SOURCE = (
    f"\nMPS_VARIABLES = {MPS_VARIABLES!r}\n"
    + r'''

def environment_seen():
    """The three MPS variables as this process sees them; `None` for an unset one."""
    import os

    return {name: os.environ.get(name) for name in MPS_VARIABLES}


def new_mps_samples(sampled_at):
    return {
        "current_allocated_bytes": 0,
        "driver_allocated_bytes": 0,
        "samples": 0,
        "sampled_at": list(sampled_at),
    }


def sample_mps_memory(torch, record):
    """Fold one reading of torch's two MPS counters into `record`'s maxima."""
    record["current_allocated_bytes"] = max(
        record["current_allocated_bytes"], int(torch.mps.current_allocated_memory())
    )
    record["driver_allocated_bytes"] = max(
        record["driver_allocated_bytes"], int(torch.mps.driver_allocated_memory())
    )
    record["samples"] += 1


def finished_mps_samples(record):
    """The record, or `None` when there is none or nothing was sampled."""
    return record if record is not None and record["samples"] else None
'''
)


def environment_mismatch(memory: MpsMemory, seen: Any, who: str) -> str | None:
    """A limitation when the child did not see the MPS variables the policy records.

    `seen` is the child's own `environment_seen()`. `None` -- no report, from
    a child that stopped before writing one -- is no observation and returns
    `None`; anything else that is not exactly what reached the child by the
    policy's account is a mismatch.
    """
    if seen is None:
        return None
    expected = {name: memory.variables.get(name) for name in MPS_VARIABLES}
    if isinstance(seen, Mapping) and dict(seen) == expected:
        return None
    return (
        f"the {who} did not see the MPS variables litetune sent or the environment passed "
        f"on: expected {expected}, it reported {seen!r}. The memory limit it ran under is not "
        "the one recorded in mps_memory"
    )
