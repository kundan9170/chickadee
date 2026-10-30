"""One append-only log, and how its readers split it.

The run journal and the session log each had this, with the cut-and-stop of `T-3.90` (#276
B2) and a reader that splits bytes before decoding (#276 B3). These hold the rule for the
one implementation both now use; their own tests still hold each caller's side.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from pathlib import Path

import pytest

from chickadee.durable import AppendLog, LogStopped, complete_lines, repair_tail


def _log(tmp_path: Path, *lines: bytes) -> Path:
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"".join(line + b"\n" for line in lines))
    return path


def _full_disk(monkeypatch: pytest.MonkeyPatch, keep: int) -> None:
    """The next write lands `keep` bytes and the disk is full."""
    writing: Callable[[int, bytes | memoryview], int] = os.write

    def write(fd: int, data: bytes | memoryview) -> int:
        writing(fd, bytes(data)[:keep])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "write", write)


def _failing_fsync(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Every fsync of `path` fails, as on a disk that reported a write-back error."""
    syncing = os.fsync
    target = path.stat()

    def fsync(fd: int) -> None:
        if os.path.samestat(os.fstat(fd), target):
            raise OSError(errno.EIO, "Input/output error")
        syncing(fd)

    monkeypatch.setattr(os, "fsync", fsync)


# -- a failure stops the log ------------------------------------------------------


def test_a_failed_append_is_cut_back_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _log(tmp_path, b"one")
    before = path.read_bytes()
    log = AppendLog.open(path)
    _full_disk(monkeypatch, keep=3)
    with pytest.raises(OSError, match="No space left"):
        log.append(b"two\n", sync=False)
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert log.cut is True
    with pytest.raises(LogStopped, match="No space left"):
        log.append(b"three\n", sync=False)
    with pytest.raises(LogStopped):
        log.sync()
    assert path.read_bytes() == before


def test_a_failed_append_never_lands_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A buffered handle kept the failed line and wrote it on the next flush or on close
    # (#276 B2). Nothing is buffered here, so closing writes nothing.
    path = _log(tmp_path, b"one")
    before = path.read_bytes()
    log = AppendLog.open(path)
    _full_disk(monkeypatch, keep=0)
    with pytest.raises(OSError):
        log.append(b"two\n", sync=False)
    monkeypatch.undo()
    log.close()
    assert path.read_bytes() == before


def test_a_line_whose_fsync_fails_is_cut_back_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # All of it reached the file, but whether it would outlive a crash is unknown, and the
    # caller is told it failed: so the log does not keep it.
    path = _log(tmp_path, b"one")
    before = path.read_bytes()
    log = AppendLog.open(path)
    _failing_fsync(monkeypatch, path)
    with pytest.raises(OSError, match="Input/output error"):
        log.append(b"two\n", sync=True)
    monkeypatch.undo()
    assert path.read_bytes() == before
    with pytest.raises(LogStopped):
        log.append(b"three\n", sync=False)


def test_a_failed_sync_stops_the_log_and_cuts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every line a sync covers was already handed out, so a failed one removes none of them.
    path = _log(tmp_path)
    log = AppendLog.open(path)
    log.append(b"one\n", sync=False)
    log.append(b"two\n", sync=False)
    before = path.read_bytes()
    _failing_fsync(monkeypatch, path)
    with pytest.raises(OSError, match="Input/output error"):
        log.sync()
    monkeypatch.undo()
    assert log.cut is None
    assert path.read_bytes() == before
    with pytest.raises(LogStopped):
        log.append(b"three\n", sync=False)
    log.close()


def test_stop_names_the_owners_reason_on_every_later_call(tmp_path: Path) -> None:
    log = AppendLog.open(_log(tmp_path))
    log.stop("a blob did not land")
    assert log.stopped == "a blob did not land"
    with pytest.raises(LogStopped, match=r"^a blob did not land$"):
        log.append(b"one\n", sync=False)
    log.close()


def test_a_reopen_writes_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _log(tmp_path, b"one")
    log = AppendLog.open(path)
    _full_disk(monkeypatch, keep=2)
    with pytest.raises(OSError):
        log.append(b"two\n", sync=True)
    monkeypatch.undo()
    log.close()
    again = AppendLog.open(path)
    again.append(b"two\n", sync=True)
    again.close()
    assert complete_lines(path.read_bytes()) == [b"one", b"two"]


def test_close_after_a_failure_is_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failure was reported when it happened; closing is how the owner gets to reopen.
    path = _log(tmp_path, b"one")
    log = AppendLog.open(path)
    _failing_fsync(monkeypatch, path)
    with pytest.raises(OSError):
        log.append(b"two\n", sync=True)
    log.close()
    log.close()


def test_a_failed_fsync_on_close_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _log(tmp_path)
    log = AppendLog.open(path)
    log.append(b"one\n", sync=False)
    _failing_fsync(monkeypatch, path)
    with pytest.raises(OSError, match="Input/output error"):
        log.close()
    monkeypatch.undo()
    log.close()


# -- open repairs the tail ---------------------------------------------------------------


def test_a_reopen_repairs_a_torn_tail_and_appends_cleanly(tmp_path: Path) -> None:
    # Appending after the partial bytes would glue the next line onto them, turning the one
    # survivable crash into a corrupt line mid-file.
    path = _log(tmp_path, b"one", b"two")
    intact = path.stat().st_size
    with path.open("ab") as fh:
        fh.write(b'{"torn')
    log = AppendLog.open(path)
    assert (log.repaired_bytes, log.size, path.stat().st_size) == (6, intact, intact)
    log.append(b"three\n", sync=True)
    log.close()
    assert path.read_bytes() == b"one\ntwo\nthree\n"


def test_a_tear_that_leaves_no_complete_line_empties_the_file(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"first')
    assert repair_tail(path) == 7
    assert path.read_bytes() == b""
    assert repair_tail(path) == 0


def test_a_clean_log_is_not_touched(tmp_path: Path) -> None:
    path = _log(tmp_path, b"one")
    log = AppendLog.open(path)
    assert log.repaired_bytes == 0
    log.close()
    assert path.read_bytes() == b"one\n"


def test_open_creates_only_when_asked(tmp_path: Path) -> None:
    path = tmp_path / "new.jsonl"
    with pytest.raises(FileNotFoundError):
        AppendLog.open(path)
    log = AppendLog.open(path, create=True)
    assert (log.size, log.repaired_bytes) == (0, 0)
    log.close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


# -- readers split bytes first -----------------------------------------------------


def test_complete_lines_drops_a_tail_torn_inside_a_utf8_character() -> None:
    # Payloads are raw UTF-8. Decoding before splitting raised at a tear inside a character
    # (#276 B3); split first and the partial line never reaches a decoder.
    torn = '{"goal": "café'.encode()[:-1]
    assert complete_lines(b"one\ntwo\n" + torn) == [b"one", b"two"]


def test_complete_lines_keeps_every_terminated_line() -> None:
    assert complete_lines(b"") == []
    assert complete_lines(b"\n") == [b""]
    assert complete_lines(b"one\n\ntwo\n") == [b"one", b"", b"two"]
    assert complete_lines(b"partial") == []
