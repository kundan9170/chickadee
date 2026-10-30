"""Repository-wide init/WRITE lease.

Cross-process, keyed by repository identity, and living under the Git common directory so
every linked worktree of one repository contends for the same lease. An in-memory
`PacketEngine` lock would not: two processes each hold their own, both publish, and the
second overwrites a generation the first is still reading.

A `durable.FileLock`, for the reason D55's journal lock is one: a protocol
built on a file's *presence* needs a notion of staleness that cannot be evaluated atomically,
and two processes that both judge a lease stale can unlink each other's replacement. The
kernel releases this when the holder dies, so a crashed init needs no manual cleanup. The
lease adds the repository id to the record, and its errors in init's terms.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

from chickadee.durable import FileLock, LockHeld, LockUnavailable
from chickadee.errors import InfraError, RepoLeaseBusy
from chickadee.init.detect import RepoIdentity

LEASE_NAME = "chickadee-init.lock"

_BUILDING: dict[str, threading.Event] = {}
"""This process's background index builds, by repository id, each set when its build ends.
Guarded by `_BUILDING_LOCK`, a leaf."""
_BUILDING_LOCK = threading.Lock()


class LeaseBusy(Exception):
    """Another process holds the repository lease. Typed, so `init` can say who."""

    def __init__(self, pid: int, host: str) -> None:
        super().__init__(f"repository is locked by pid {pid} on {host}")
        self.pid, self.host = pid, host


class LeaseError(Exception):
    """The lease could not be evaluated — not the same as being held."""


def lease_path(git_common_dir: Path) -> Path:
    return git_common_dir / LEASE_NAME


class RepoLease:
    """Held for the whole of init, released only after the final directory fsync."""

    def __init__(self, lock: FileLock) -> None:
        self._lock = lock
        self.path = lock.path

    @staticmethod
    def acquire(git_common_dir: Path, repository_id: str) -> RepoLease:
        path = lease_path(git_common_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return RepoLease(FileLock.acquire(path, owner={"repo": repository_id}))
        except LockHeld as held:
            raise LeaseBusy(held.pid, held.host) from None
        except LockUnavailable as exc:
            raise LeaseError(str(exc)) from exc

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> RepoLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


@contextmanager
def building(repository_id: str) -> Generator[None]:
    """This process building the repository's index in the background
    (`PacketEngine.index_in_background`): while it holds the lease, `take_write_lease` in this
    process waits for it to end rather than refusing as busy, so a run, a handoff, an undo or
    a write the person asked for meanwhile goes ahead once the index is there."""
    done = threading.Event()
    with _BUILDING_LOCK:
        _BUILDING[repository_id] = done
    try:
        yield
    finally:
        with _BUILDING_LOCK:
            _BUILDING.pop(repository_id, None)
        done.set()


def take_write_lease(identity: RepoIdentity) -> RepoLease:
    """The lease in the engine's terms: a busy lease is `RepoLeaseBusy`,
    naming the holder, and one that cannot be evaluated is `InfraError`. `init` keeps
    `LeaseBusy`, which its own surface renders; everything that takes the lease for the
    engine, a run's workspace or a reindex, takes it here. Held by this process's own
    background index build (`building`), it is waited for, never refused."""
    while True:
        try:
            return RepoLease.acquire(identity.git_common_dir, identity.repository_id)
        except LeaseBusy as busy:
            with _BUILDING_LOCK:
                build = _BUILDING.get(identity.repository_id)
            if build is not None and busy.pid == os.getpid():
                build.wait()
                continue
            raise _busy(busy) from busy
        except LeaseError as exc:
            raise InfraError(f"repository lease could not be evaluated: {exc}") from exc


def _busy(busy: LeaseBusy) -> RepoLeaseBusy:
    return RepoLeaseBusy(
        f"another process holds the repository WRITE lease: pid {busy.pid} on {busy.host} (D39)"
    )
