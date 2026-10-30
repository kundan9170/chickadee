"""Durable file writes: atomic replace, directory sync, durable rename and unlink (§3.1 of
`37-durable.md`).

temp → write → `fsync(file)` → `rename` → `fsync(dir)`. A crash at any point leaves the old
file or the new one and never a hybrid, because `rename` within a directory is atomic and the
two fsyncs are what make that ordering survive a power loss rather than merely a process kill.

The one home for this. Every part that keeps state on disk calls here for how a file
reaches it, and keeps for itself what the file means. Nothing here imports from `chickadee`:
errors are `OSError`, and each caller wraps them in its own typed error.

The `Fault` seam exists so the property can be *tested* rather than asserted. An atomicity
claim nobody has interrupted is a claim about code that has never been interrupted.
"""

from __future__ import annotations

import contextlib
import errno
import os
import stat
import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

__all__ = [
    "NEW_FILE_MODE",
    "Fault",
    "FaultHook",
    "make_dir",
    "rename_durably",
    "sync_dir",
    "unlink_durably",
    "write_all",
    "write_atomic",
]


class Fault(StrEnum):
    """Where a test may inject a crash. Never set in production."""

    BEFORE_WRITE = "before_write"
    AFTER_WRITE = "after_write"
    """Bytes written, not yet fsynced."""
    AFTER_FSYNC = "after_fsync"
    """File fsynced, not yet renamed. For `rename_durably`: about to rename."""
    AFTER_RENAME = "after_rename"
    """Renamed, directory not yet synced."""


FaultHook = Callable[[Fault], None]

NEW_FILE_MODE = 0o644
"""Mode for a file this creates. `mkstemp`'s 0600 would make every new file owner-only,
which is not what writing a repository file, or a file git reads, means."""


def _no_fault(_: Fault) -> None:
    return None


def write_all(fd: int, data: bytes) -> None:
    """`os.write` until every byte is out. A regular file takes a short write only on the
    way to an error, which the next call raises."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written == 0:
            raise OSError(errno.EIO, "write made no progress")
        view = view[written:]


def sync_dir(directory: Path) -> None:
    """Persist a directory's entries: a rename, a create or an unlink in it. Without this the
    new inode is durable but the entry naming it may not be, so the file can come back as
    the old one after a crash the caller was told had succeeded.

    A failure raises. Swallowing it reports a write durable whose rename might not
    survive a crash, which is the hybrid this module exists to prevent."""
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_dir(directory: Path) -> None:
    """`mkdir -p`, then sync the parent of every directory it created, so the new entries
    survive a crash before anything is written into them."""
    missing: list[Path] = []
    at = directory
    while not at.exists():
        missing.append(at)
        at = at.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        sync_dir(created.parent)


def write_atomic(
    path: Path, data: bytes, *, fault: FaultHook | None = None, mode: int | None = None
) -> None:
    """Write `data` to `path` so that a crash leaves old or new, never a partial file.

    The temp file is created **in the same directory** as the target, not in `/tmp`: a
    rename is only atomic within one filesystem, and a cross-device rename degrades to
    copy-then-delete, which is exactly the hybrid this prevents.

    A replaced file keeps its mode and a new one is `NEW_FILE_MODE`: `os.replace`
    carries the temp file's mode onto the target, so without this a 0755 script would stop
    running and a 0644 file would become owner-only. The mode is set before the fsync, so it
    is durable with the bytes.

    `mode` is a new file's mode in place of `NEW_FILE_MODE`, for a file that must never be
    readable by others even for an instant (a server's tokens): the temp file has it before a
    byte is written. A replaced file keeps its own mode either way.
    """
    hit = fault or _no_fault
    path.parent.mkdir(parents=True, exist_ok=True)
    new_mode = NEW_FILE_MODE if mode is None else mode
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else new_mode
    hit(Fault.BEFORE_WRITE)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, mode)
            write_all(fd, data)
            hit(Fault.AFTER_WRITE)
            os.fsync(fd)
        finally:
            os.close(fd)
        hit(Fault.AFTER_FSYNC)
        os.replace(tmp, path)
        hit(Fault.AFTER_RENAME)
    except BaseException:
        tmp.unlink(missing_ok=True)
        # Best effort on the way out: the error already propagating is the one the caller
        # needs, and a second one raised here would replace it.
        with contextlib.suppress(OSError):
            sync_dir(path.parent)
        raise
    sync_dir(path.parent)


def write_new(path: Path, data: bytes) -> bool:
    """Create `path` holding `data`, only if nothing is there: True when it wrote, False when
    a file was already there, which is left untouched.

    The bytes go to a temp file in the same directory, are synced, and are then *linked* into
    place. `link` fails when the name exists, so the check and the create are one step: a
    file that appears between them is kept rather than replaced, which `write_atomic`'s
    rename would do. A crash leaves the name absent or whole, never partial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        try:
            os.fchmod(fd, NEW_FILE_MODE)
            write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
    finally:
        tmp.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            sync_dir(path.parent)
    return True


def rename_durably(src: Path, dst: Path, *, fault: FaultHook | None = None) -> None:
    """Rename a file or a directory and persist the rename: `dst`'s parent is synced, and
    `src`'s too when it differs, because a move between directories changes both.

    `Fault.AFTER_FSYNC` fires before the rename (there is nothing to write, so the file is as
    ready as it will be) and `Fault.AFTER_RENAME` after it, before the syncs."""
    hit = fault or _no_fault
    hit(Fault.AFTER_FSYNC)
    os.replace(src, dst)
    hit(Fault.AFTER_RENAME)
    sync_dir(dst.parent)
    if src.parent != dst.parent:
        sync_dir(src.parent)


def unlink_durably(path: Path) -> None:
    """Remove `path` if it exists and persist the removal, the way `write_atomic` persists a
    rename: an unlink the caller is told is durable must be, or the file can come back after
    a crash."""
    path.unlink(missing_ok=True)
    sync_dir(path.parent)
