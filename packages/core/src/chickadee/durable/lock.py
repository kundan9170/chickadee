"""A file lock held by the kernel.

`flock` on an open descriptor, never the presence of a file. A create-then-check protocol
needs a notion of a *stale* lock, and reclaiming one cannot be made atomic with the tools a
filesystem offers: two processes that both judge a lock stale can unlink each other's
replacement. An advisory lock on a descriptor has no such concept: the kernel drops it when
the holder dies, so a crashed holder needs no cleanup and a live one cannot be evicted.

The file carries the holder's pid and host so a refusal can name who holds it. It is never
the mechanism, which is why unreadable contents are harmless, and it is never unlinked:
between another process opening it and locking it, an unlink would let a third lock a fresh
inode, putting two holders on two files.

POSIX advisory locking. Over NFS it degrades, which is the shared-storage case already out of
scope until isolation hardens.

The run journal's LOCK, the session store's LOCK and init's repository lease were three copies
of this, with the contention errnos copied three times; `gc`'s probe was a fourth.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import socket
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

__all__ = ["FileLock", "LockHeld", "LockUnavailable", "lock_is_held", "read_owner"]

# `flock` reports contention as EWOULDBLOCK; EACCES appears on some platforms. Any other
# errno means the lock could not be evaluated at all, which is not the same answer.
_CONTENTION = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


class LockHeld(OSError):
    """Another process holds the lock. `pid` and `host` are what the file records:
    diagnostics, `(-1, "?")` when it cannot be read."""

    def __init__(self, path: Path, pid: int, host: str) -> None:
        super().__init__(f"{path} is held by pid {pid} on {host}")
        self.path = path
        self.pid = pid
        self.host = host


class LockUnavailable(OSError):
    """The lock could not be evaluated: not the same as being held. Reporting it as held
    would name an owner that does not exist, permanently on a filesystem without `flock`."""


class FileLock:
    """An exclusive `flock` on `path`, held until `release` or until the process dies."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd: int | None = fd

    @staticmethod
    def acquire(path: Path, *, owner: Mapping[str, str | int] | None = None) -> FileLock:
        """Take the lock without waiting. `LockHeld` if another process has it,
        `LockUnavailable` if it cannot be evaluated. On success the file records `pid`,
        `host` and `owner`'s fields, fsynced, best effort: a stale record only weakens a
        later refusal's message."""
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as exc:
            raise LockUnavailable(f"cannot open {path}: {exc}") from exc
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in _CONTENTION:
                pid, host = read_owner(path)
                raise LockHeld(path, pid, host) from None
            raise LockUnavailable(f"cannot lock {path}: {exc}") from exc
        record: dict[str, str | int] = {"pid": os.getpid(), "host": socket.gethostname()}
        record.update(owner or {})
        with contextlib.suppress(OSError):
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps(record).encode())
            os.fsync(fd)
        return FileLock(path, fd)

    def release(self) -> None:
        """Unlock and close. Idempotent. Closing the descriptor drops the lock on its own;
        unlocking first keeps the intent visible and is harmless."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    def __enter__(self) -> FileLock:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def lock_is_held(path: Path) -> bool:
    """Whether a live process holds the lock on `path`, asked the way `FileLock` asks: the
    lock on a descriptor, never the presence of the file (a leftover is inert). Taking and
    releasing it here is harmless, since nothing is written through it.

    "Cannot tell" reads as held: a file that exists but cannot be opened or locked must not
    read as free, because a caller would then act as if no holder were alive."""
    if not path.is_file():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def read_owner(path: Path) -> tuple[int, str]:
    """Who the file says holds the lock. Diagnostics only: `flock` decides, not this."""
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        return int(owner["pid"]), str(owner["host"])
    except (ValueError, KeyError, OSError, TypeError):
        return -1, "?"
