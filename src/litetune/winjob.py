"""A Windows job object per stage subprocess, over ctypes and kernel32.

What a process group is to `envs._run_guarded` on POSIX, a job is here: the
set of processes a stage started, which one call can end. `envs` decides when
and in what order each call is made, and what a failure means; this module only
makes the calls. Every function raises `OSError` when the call it wraps
reports failure, naming the call, and does nothing else about it.

Imported only on win32, by `envs._windows_job_api`. Elsewhere the import
itself refuses, which also keeps `mypy` on a POSIX host from type-checking
kernel32 names that do not exist there; `mypy --platform win32` checks all
of it.

Constants and structure layouts are Microsoft's, named as in the reference page
each comes from (learn.microsoft.com/windows/win32/api/...), so a reader can
search for them.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from ctypes import wintypes
from typing import Any

if sys.platform != "win32":
    raise ImportError("litetune.winjob calls kernel32 and exists only on Windows")

# `CreateToolhelp32Snapshot`: "Includes all threads in the system in the
# snapshot."
TH32CS_SNAPTHREAD = 0x00000004
# Thread Security and Access Rights: "Required to suspend or resume a thread",
# and the right `GetProcessIdOfThread` asks for.
THREAD_SUSPEND_RESUME = 0x0002
THREAD_QUERY_LIMITED_INFORMATION = 0x0800
# `GenerateConsoleCtrlEvent`: "CTRL_BREAK_EVENT 1 Generates a CTRL+BREAK signal."
CTRL_BREAK_EVENT = 1
# `JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags`: "Causes all processes
# associated with the job to terminate when the last handle to the job is
# closed." That page also says the flag "requires use of a
# JOBOBJECT_EXTENDED_LIMIT_INFORMATION structure", hence `_ExtendedLimit`.
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
# `JOBOBJECTINFOCLASS` values, from `QueryInformationJobObject` and
# `SetInformationJobObject`.
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
# How many process ids one `JobObjectBasicProcessIdList` query asks room for
# at first; `_process_ids` doubles it while the list does not fit.
_FIRST_ID_ROOM = 16
# And the most it asks room for before giving up, so the doubling has an end:
# 65,536 ids is a 512 KiB buffer.
_MOST_ID_ROOM = 1 << 16
# `Thread32First` / `Thread32Next`: "The ERROR_NO_MORE_FILES error value is
# returned by the GetLastError function if no threads exist or the snapshot
# does not contain thread information."
ERROR_NO_MORE_FILES = 18
# What a query answers when its buffer is too small for what it returns
# (System Error Codes).
ERROR_MORE_DATA = 234
# `ResumeThread`: "If the function fails, the return value is (DWORD) -1."
_RESUME_FAILED = 0xFFFFFFFF
# `CreateToolhelp32Snapshot` fails with `INVALID_HANDLE_VALUE`, `(HANDLE)-1`.
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", wintypes.LARGE_INTEGER),
        ("TotalKernelTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


def _process_id_list(room: int) -> Any:
    """`JOBOBJECT_BASIC_PROCESS_ID_LIST` with room for `room` ids.

    Its `ProcessIdList` is "a variable-length array", declared with one
    element in C, so the structure is built at the size the query needs.
    """

    class _ProcessIdList(ctypes.Structure):
        _fields_ = [
            ("NumberOfAssignedProcesses", wintypes.DWORD),
            ("NumberOfProcessIdsInList", wintypes.DWORD),
            ("ProcessIdList", ctypes.c_size_t * room),
        ]

    return _ProcessIdList()


class _ThreadEntry32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


def _prototype(name: str, restype: Any, *argtypes: Any) -> Any:
    """`kernel32.<name>` with its argument and result types declared.

    Declared rather than left to ctypes' defaults, which pass and return a C
    `int`: a 64-bit `HANDLE` would be truncated on its way in or out.
    """
    function = getattr(_kernel32, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


_CreateJobObjectW = _prototype(
    "CreateJobObjectW", wintypes.HANDLE, ctypes.c_void_p, wintypes.LPCWSTR
)
_SetInformationJobObject = _prototype(
    "SetInformationJobObject",
    wintypes.BOOL,
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
)
_QueryInformationJobObject = _prototype(
    "QueryInformationJobObject",
    wintypes.BOOL,
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
)
_AssignProcessToJobObject = _prototype(
    "AssignProcessToJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.HANDLE
)
_TerminateJobObject = _prototype(
    "TerminateJobObject", wintypes.BOOL, wintypes.HANDLE, wintypes.UINT
)
_CloseHandle = _prototype("CloseHandle", wintypes.BOOL, wintypes.HANDLE)
_GetProcessId = _prototype("GetProcessId", wintypes.DWORD, wintypes.HANDLE)
_CreateToolhelp32Snapshot = _prototype(
    "CreateToolhelp32Snapshot", wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD
)
_Thread32First = _prototype(
    "Thread32First", wintypes.BOOL, wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)
)
_Thread32Next = _prototype(
    "Thread32Next", wintypes.BOOL, wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32)
)
_OpenThread = _prototype(
    "OpenThread", wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
)
_ResumeThread = _prototype("ResumeThread", wintypes.DWORD, wintypes.HANDLE)
_GetProcessIdOfThread = _prototype("GetProcessIdOfThread", wintypes.DWORD, wintypes.HANDLE)
_GenerateConsoleCtrlEvent = _prototype(
    "GenerateConsoleCtrlEvent", wintypes.BOOL, wintypes.DWORD, wintypes.DWORD
)


def _failed(call: str, code: int | None = None) -> OSError:
    """The `OSError` for a failed kernel32 call, naming the call.

    `use_last_error=True` above is what makes `get_last_error` the error of
    *this* call: ctypes keeps a private copy of the Windows error code that is
    exchanged with the real one before and after each foreign call (the
    `use_errno` and `use_last_error` parameters, the ctypes documentation), so
    nothing Python does in between can overwrite it.
    """
    code = ctypes.get_last_error() if code is None else code
    return ctypes.WinError(code, f"{call} failed: {ctypes.FormatError(code)}")


def _process_handle(proc: subprocess.Popen) -> int:
    """The handle `Popen` holds to its child, asked for rather than assumed.

    `Popen` keeps the handle `CreateProcess` returned as `_handle`, an `int`
    subclass closed only by its own `Close`/`__del__` (`Handle` and
    `_execute_child` in CPython's `Lib/subprocess.py`, 3.10 to 3.14). It is
    private, so a `Popen` without one is refused here rather than guessed at:
    the alternative, `OpenProcess` on `proc.pid`, would act on whatever owns
    that number if this is not the process it appears to be.
    """
    handle = getattr(proc, "_handle", None)
    if not isinstance(handle, int) or getattr(handle, "closed", False):
        raise OSError(f"pid {proc.pid} has no open process handle in its Popen")
    return int(handle)


def _threads_of(pid: int) -> list[int]:
    """The identifiers of every thread whose owner is `pid`, from one snapshot.

    `CreateToolhelp32Snapshot`: "To identify the threads that belong to a
    specific process, compare its process identifier to the
    th32OwnerProcessID member of the THREADENTRY32 structure".
    """
    snapshot = _CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    if snapshot is None or snapshot == _INVALID_HANDLE_VALUE:
        raise _failed("CreateToolhelp32Snapshot")
    try:
        found: list[int] = []
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        more = _Thread32First(snapshot, ctypes.byref(entry))
        while more:
            if entry.th32OwnerProcessID == pid:
                found.append(int(entry.th32ThreadID))
            # `Thread32First` "changes dwSize to the number of bytes written";
            # set back before each call so every entry is asked for in full.
            entry.dwSize = ctypes.sizeof(entry)
            more = _Thread32Next(snapshot, ctypes.byref(entry))
        code = ctypes.get_last_error()
        if code != ERROR_NO_MORE_FILES:
            raise _failed("Thread32Next", code)
        return found
    finally:
        _CloseHandle(snapshot)


class Win32JobApi:
    """The kernel32 calls `envs._JobApi` names, one method per step."""

    def create_job(self) -> int:
        """A new job that ends every process in it when its last handle closes.

        `CreateJobObjectW` with no security attributes: "If lpJobAttributes is
        NULL, the job object gets a default security descriptor and the handle
        cannot be inherited" -- so no child holds it open, and closing ours is
        closing the last one.
        """
        job = _CreateJobObjectW(None, None)
        if not job:
            raise _failed("CreateJobObjectW")
        limits = _ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = _failed("SetInformationJobObject")
            _CloseHandle(job)
            raise error
        return int(job)

    def assign(self, job: int, proc: subprocess.Popen) -> None:
        if not _AssignProcessToJobObject(job, _process_handle(proc)):
            raise _failed("AssignProcessToJobObject")

    def resume(self, proc: subprocess.Popen) -> None:
        """Start the primary thread of a child created with `CREATE_SUSPENDED`.

        `Popen` closes the thread handle `CreateProcess` returned (`_winapi.
        CloseHandle(ht)` in `_execute_child`), so the thread is found again by
        its owner's pid. That pid is checked against the process handle first,
        so the snapshot is read for the process `Popen` started and not for
        whatever else holds the number.

        A thread id from the snapshot is checked again once it is open: "The
        identifier is valid until the thread terminates" (Microsoft, "Thread
        Handles and Identifiers"), and a child terminated by something else
        after the snapshot would free its thread's id for another thread. Once
        a handle is open the thread it names cannot change, so
        `GetProcessIdOfThread` on that handle has to answer our child's pid
        before anything is resumed.

        Exactly one thread is required, and a previous suspend count of
        exactly 1. Neither is a documented property of a new suspended process
        -- `CREATE_SUSPENDED` says only that "the primary thread of the new
        process is created in a suspended state" -- so both are checked here
        and anything else is refused: a second thread could be running already,
        and a count above 1 leaves the thread suspended ("If the return value
        is greater than 1, the specified thread is still suspended",
        `ResumeThread`). Both held for every stage the real-tree tests started
        on windows-latest, CPython 3.12, the toolcache interpreter
        (`tests/test_windows_real_tree.py`).
        """
        handle = _process_handle(proc)
        owner = _GetProcessId(handle)
        if owner != proc.pid:
            raise OSError(f"the process handle of pid {proc.pid} names pid {owner}")
        threads = _threads_of(proc.pid)
        if len(threads) != 1:
            raise OSError(
                f"pid {proc.pid} was created suspended and has {len(threads)} threads, not one"
            )
        thread = _OpenThread(
            THREAD_QUERY_LIMITED_INFORMATION | THREAD_SUSPEND_RESUME, False, threads[0]
        )
        if not thread:
            raise _failed("OpenThread")
        try:
            belongs_to = _GetProcessIdOfThread(thread)
            if belongs_to != proc.pid:
                raise OSError(
                    f"thread {threads[0]} of pid {proc.pid} now belongs to pid {belongs_to}"
                )
            previous = _ResumeThread(thread)
            if previous == _RESUME_FAILED:
                raise _failed("ResumeThread")
            if previous != 1:
                raise OSError(
                    f"ResumeThread on pid {proc.pid} found a suspend count of {previous}, not 1"
                )
        finally:
            _CloseHandle(thread)

    def active_processes(self, job: int) -> int:
        """`JOBOBJECT_BASIC_ACCOUNTING_INFORMATION.ActiveProcesses`.

        "The total number of processes currently associated with the job."
        """
        info = _BasicAccounting()
        if not _QueryInformationJobObject(
            job,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            raise _failed("QueryInformationJobObject")
        return int(info.ActiveProcesses)

    def process_ids(self, job: int) -> list[int]:
        """`JOBOBJECT_BASIC_PROCESS_ID_LIST`: the pids the job holds now.

        "If this number is less than NumberOfAssignedProcesses, increase the
        size of the buffer to accommodate the complete list" -- so the query
        is repeated with twice the room until the whole list fits. Whether a
        short buffer makes the call fail with ERROR_MORE_DATA or succeed with
        part of the list is not stated there; either is taken as "more room".
        """
        room = _FIRST_ID_ROOM
        while room <= _MOST_ID_ROOM:
            listed = _process_id_list(room)
            if not _QueryInformationJobObject(
                job,
                JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                ctypes.byref(listed),
                ctypes.sizeof(listed),
                None,
            ):
                code = ctypes.get_last_error()
                if code != ERROR_MORE_DATA:
                    raise _failed("QueryInformationJobObject", code)
            elif listed.NumberOfProcessIdsInList >= listed.NumberOfAssignedProcesses:
                return [int(pid) for pid in listed.ProcessIdList[: listed.NumberOfProcessIdsInList]]
            room *= 2
        raise OSError(f"the job holds more than {_MOST_ID_ROOM} processes")

    def terminate(self, job: int, exit_code: int) -> None:
        """`TerminateJobObject`: "as if TerminateProcess were called for each
        process associated with the job", with `exit_code` for each."""
        if not _TerminateJobObject(job, exit_code):
            raise _failed("TerminateJobObject")

    def close(self, job: int) -> None:
        if not _CloseHandle(job):
            raise _failed("CloseHandle")

    def send_ctrl_break(self, pid: int) -> None:
        """`CTRL_BREAK_EVENT` to the console process group `pid` leads.

        `GenerateConsoleCtrlEvent` itself, rather than `os.kill`: on CPython
        3.10 and 3.11 `os.kill` sets the error from a failed
        `GenerateConsoleCtrlEvent` and then falls through to `OpenProcess` and
        `TerminateProcess(handle, 1)` on the same pid, returning with the error
        still set -- a `SystemError` after the process was killed outright
        (`os_kill_impl`, `Modules/posixmodule.c`, 3.10 and 3.11; 3.12 returns
        at the failure). This raises `OSError` on every version and touches
        nothing else.

        A group id of 0 is refused before it is sent: `GenerateConsoleCtrlEvent`
        documents 0 as "all processes that share the console of the calling
        process", litetune among them.
        """
        if pid <= 0:
            raise ValueError(f"refusing to send CTRL_BREAK_EVENT to process group {pid}")
        if not _GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, pid):
            raise _failed("GenerateConsoleCtrlEvent")


API = Win32JobApi()
