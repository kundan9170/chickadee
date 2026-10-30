"""Reading a regular file without blocking, bounded.

`open_nonblocking` is the one place a file is opened for reading: non-blocking, so a FIFO with
no writer returns at once rather than holding the caller. `open_regular` adds the one check
every reader wants before a byte is read, that the descriptor holds a regular file, since a
FIFO or a device such as `/dev/zero` can block or never end. The Reader opens with
`open_nonblocking` and asks its own questions of the descriptor (`reader.py`), because it
answers a directory and another non-regular file differently; freshness and citations read
through `read_file`. Imports nothing from `chickadee.context`.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Final

__all__ = [
    "FILE_BYTES",
    "open_nonblocking",
    "open_regular",
    "read_bounded",
    "read_file",
    "same_file",
]


FILE_BYTES: Final = 64 * 1024 * 1024
"""The most any file read takes into memory. A handle's bytes are held whole — hashed,
sniffed, sliced — so the bound is on the file, before any of the caps on what is sent."""


def open_nonblocking(path: Path) -> int:
    """A read-only descriptor on `path`, opened non-blocking, so opening a FIFO returns at
    once rather than waiting for a writer. Raises `OSError` as `os.open` does. The caller
    checks what the descriptor holds before reading it, and closes it."""
    return os.open(path, os.O_RDONLY | os.O_NONBLOCK)


def open_regular(path: Path) -> int | None:
    """A descriptor on the regular file at `path`, or None. Anything but a regular file — a
    FIFO, a device such as `/dev/zero`, a directory — is refused before a byte is read, since
    reading one can block forever or never end."""
    try:
        fd = open_nonblocking(path)
    except OSError:
        return None
    try:
        if stat.S_ISREG(os.fstat(fd).st_mode):
            return fd
    except OSError:
        pass
    os.close(fd)
    return None


def read_bounded(fd: int, limit: int) -> bytes | None:
    """At most `limit` bytes from `fd`, or None when it holds more. The file may grow while
    it is read, so the bound is on what was read, not only on what `fstat` said."""
    chunks: list[bytes] = []
    got = 0
    while got <= limit:
        chunk = os.read(fd, min(1 << 20, limit + 1 - got))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        got += len(chunk)
    return None


def read_file(path: Path) -> bytes | None:
    """A regular file's bytes, or None when there is no such file to read: gone, a
    directory or other non-regular file, one this process may not open, or one past
    `FILE_BYTES`. Freshness reads each as stale, which it is — the handle no longer
    resolves to what it named."""
    fd = open_regular(path)
    if fd is None:
        return None
    try:
        return read_bounded(fd, FILE_BYTES)
    except OSError:
        return None
    finally:
        os.close(fd)


def same_file(path: Path, fd: int) -> bool:
    """Whether `path` reaches the file open at `fd` — the same file, not merely the same
    name. A path that is gone or cannot be stat'd reaches nothing."""
    try:
        return os.path.samestat(os.stat(path), os.fstat(fd))
    except OSError:
        return False
