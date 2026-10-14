"""The FIFO watchdog in `conftest`, which the FIFO tests rely on not to hang."""

from __future__ import annotations

import os
import threading
import time

import pytest
from conftest import fifo_released_after


@pytest.mark.posix
def test_a_reader_that_arrives_after_the_deadline_is_released(tmp_path):
    fifo = tmp_path / "litetune.json"
    os.mkfifo(fifo)
    read = {}

    def reader():
        with open(fifo, "rb") as handle:  # waits for a writer
            read["bytes"] = handle.read()

    with fifo_released_after(fifo, 0.1, b"payload") as released:
        time.sleep(0.5)  # the deadline passes with nothing reading
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout=10)
        if thread.is_alive():
            # The watchdog gave up. Release the reader here, then fail.
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            thread.join(timeout=5)
            pytest.fail("the watchdog did not release a reader that arrived after the deadline")
    assert read == {"bytes": b"payload"}
    assert released == [True]


@pytest.mark.posix
def test_what_fails_in_the_watchdog_is_raised(tmp_path):
    missing = tmp_path / "never-made"  # opening it fails, and not for want of a reader
    with pytest.raises(FileNotFoundError):
        with fifo_released_after(missing, 0, b""):
            time.sleep(0.5)
