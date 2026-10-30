"""Reading a regular file without blocking, pinned before the opener moves.

`read_file` and `open_regular` are what freshness and citations read through. Only the Reader's
side of opening was pinned (`test_reader.py`'s FIFO and device tests): these pin the other side,
so that merging the two openers into one cannot change what either answers.

Each case returns at once. A FIFO with no writer is the one that would hang a blocking open, so
those calls run under a watchdog thread rather than trusting the open to be non-blocking.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

import chickadee.context.files as files_module
from chickadee.context.files import open_regular, read_file


def _within(seconds: float, call: Callable[[], object]) -> object:
    """`call()`'s result, or a failure if it has not returned in `seconds`: a blocked open
    is the defect these pins exist to catch, and it must fail rather than hang the suite."""
    out: list[object] = []
    worker = threading.Thread(target=lambda: out.append(call()), daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"did not return within {seconds}s: the open blocked")
    return out[0]


@pytest.fixture
def fifo(tmp_path: Path) -> Path:
    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs on this platform")
    path = tmp_path / "pipe"
    os.mkfifo(path)
    return path


def test_a_regular_file_reads_whole(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_bytes(b"x = 1\n")
    assert read_file(path) == b"x = 1\n"
    fd = open_regular(path)
    assert fd is not None
    os.close(fd)


def test_a_fifo_with_no_writer_is_no_file_and_does_not_block(fifo: Path) -> None:
    assert _within(5, lambda: read_file(fifo)) is None
    assert _within(5, lambda: open_regular(fifo)) is None


def test_a_device_is_no_file(tmp_path: Path) -> None:
    """`/dev/zero` never ends: reading it would fill memory before any bound applied."""
    if not Path("/dev/zero").exists():
        pytest.skip("no /dev/zero on this platform")
    assert _within(5, lambda: read_file(Path("/dev/zero"))) is None
    assert open_regular(Path("/dev/zero")) is None


def test_a_directory_is_no_file(tmp_path: Path) -> None:
    assert read_file(tmp_path) is None
    assert open_regular(tmp_path) is None


def test_a_missing_path_is_no_file(tmp_path: Path) -> None:
    assert read_file(tmp_path / "gone.py") is None
    assert open_regular(tmp_path / "gone.py") is None


def test_a_file_past_the_cap_is_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_file` holds a file whole, so the cap is on the file. At the cap it reads; one byte
    past it, it does not."""
    monkeypatch.setattr(files_module, "FILE_BYTES", 8)
    at = tmp_path / "at.txt"
    at.write_bytes(b"12345678")
    past = tmp_path / "past.txt"
    past.write_bytes(b"123456789")
    assert read_file(at) == b"12345678"
    assert read_file(past) is None
