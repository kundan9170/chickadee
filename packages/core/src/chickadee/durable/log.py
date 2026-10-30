"""An append-only log of `\\n`-terminated lines, and how its readers split it (§3.5 of
`37-durable.md`).

A crash can only lose the tail: a line torn part-way. So the tail is the one legal repair,
and only a writer's open makes it, because a reader may be looking at someone else's live
log. Anything else wrong in the file is the caller's to refuse, since a log edited to make
it parse is no longer the record.

A line is one unbuffered write. A buffered handle kept a line whose write had failed and
flushed it on the next append or on close (#276 B2), so the log recorded an event its
caller had been told did not happen. Here a failed write or fsync cuts the file back to the
last line handed out and stops the log; only a reopen, which repairs and lets the caller
re-read what is on disk, writes again.

The run journal, the session log and `gc.log` were three of these, with two tail repairs
and three append disciplines. The line format, the seq and when to fsync stay the
caller's.
"""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from chickadee.durable.files import NEW_FILE_MODE, sync_dir, write_all

__all__ = ["AppendLog", "LogStopped", "complete_lines", "complete_lines_from", "repair_tail"]


class LogStopped(OSError):
    """An earlier write failed, or the owner stopped the log: what is durable is no longer
    known, so nothing more is written through this handle. Reopen to write again."""


class AppendLog:
    """One append-only file, open for appending. One writer: the caller holds whatever lock
    makes that true."""

    def __init__(self, path: Path, fd: int, *, repaired: int) -> None:
        self.path = path
        self._fd: int | None = fd
        self.repaired_bytes = repaired
        """Bytes of a torn tail removed on open; 0 when the log was clean."""
        self.size = os.fstat(fd).st_size
        """The file's length through the last line handed out: where a failed append is cut
        back to."""
        self.cut: bool | None = None
        """After a failed append, whether its bytes were cut back. `None` when the log
        stopped without a cut (a failed sync, or `stop`), or has not stopped."""
        self._stopped: str | None = None

    @staticmethod
    def open(path: Path, *, create: bool = False) -> AppendLog:
        """Repair a torn tail, then open for appending. `create` makes a missing file
        (`NEW_FILE_MODE`) and syncs its directory, so the entry is as durable as the first
        line written into it; without it a missing file is `FileNotFoundError`."""
        if create and not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT, NEW_FILE_MODE)
            os.close(fd)
            sync_dir(path.parent)
        repaired = repair_tail(path)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            return AppendLog(path, fd, repaired=repaired)
        except BaseException:
            os.close(fd)
            raise

    @property
    def stopped(self) -> str | None:
        """Why the log stopped, as every later `append` and `sync` reports it; `None` while
        it can write."""
        return self._stopped

    def append(self, line: bytes, *, sync: bool) -> None:
        """Write one whole line, then fsync it if `sync`. When to sync is the caller's policy.

        On an `OSError` the file is cut back to `size` (best effort, `cut` says whether it
        was), the log stops, and the error is re-raised: the line the caller is told failed
        is not in the file, and nothing later can put it there."""
        fd = self._writable()
        try:
            write_all(fd, line)
            if sync:
                os.fsync(fd)
        except OSError as exc:
            self._stopped = str(exc)
            try:
                os.ftruncate(fd, self.size)
            except OSError:
                # The file ends in part of the line, a torn tail the reopen removes, or in
                # all of it, which the reopen keeps: the file, not this process, is the
                # record.
                self.cut = False
            else:
                self.cut = True
                with contextlib.suppress(OSError):
                    os.fsync(fd)
            raise
        self.size += len(line)

    def sync(self) -> None:
        """Fsync what has been appended. A failure stops the log without a cut: every line a
        sync covers was already handed out, so none is removed."""
        fd = self._writable()
        try:
            os.fsync(fd)
        except OSError as exc:
            self._stopped = str(exc)
            raise

    def stop(self, why: str) -> None:
        """Stop writing, with the owner's words for why: every later `append` and `sync`
        raises `LogStopped(why)`. After a failure, this is the owner's account of it, naming
        what the log cannot know (a seq, how to resume). On a log that has not failed, it
        stops one whose owner can no longer vouch for the next line, as when the blob a line
        would name did not land."""
        self._stopped = why

    def close(self) -> None:
        """Fsync and close. Idempotent. Quiet on a stopped log: its failure was reported when
        it happened, and closing is how the owner gets to reopen."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if self._stopped is None:
                os.fsync(fd)
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _writable(self) -> int:
        if self._fd is None:
            raise LogStopped(f"{self.path} is closed")
        if self._stopped is not None:
            raise LogStopped(self._stopped)
        return self._fd


def repair_tail(path: Path) -> int:
    """Cut a torn trailing line off `path` and fsync, returning the bytes removed. Bytes, not
    text: the tear can land inside a character. Called by `AppendLog.open`, never by a
    reader."""
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return 0
    cut = data.rfind(b"\n") + 1  # 0 when no complete line survives
    fd = os.open(path, os.O_WRONLY)
    try:
        os.ftruncate(fd, cut)
        os.fsync(fd)
    finally:
        os.close(fd)
    return len(data) - cut


def complete_lines(data: bytes) -> list[bytes]:
    """Every `\\n`-terminated line in `data`, without its newline. A trailing partial line is
    dropped: it is the one thing a crash produces, and it never reaches a decoder (#276 B3).
    `complete_lines_from` over the bytes, so the rule has one home."""
    return [line[:-1] for line in complete_lines_from(io.BytesIO(data))]


def complete_lines_from(fh: BinaryIO) -> Iterator[bytes]:
    """`complete_lines`, read from `fh`'s position a line at a time: each `\\n`-terminated
    line with its newline, so a reader can count the bytes it consumed, stopping before a
    trailing partial line. For a reader that stops early (a page of a journal) and so must not
    read, or split, the rest of the file."""
    for line in fh:
        if not line.endswith(b"\n"):
            return
        yield line
