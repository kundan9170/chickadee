"""DU-6: a lock is `flock` on a descriptor, and its errors say what they mean."""

from __future__ import annotations

import errno
import fcntl
import json
import os
from pathlib import Path

import pytest

from chickadee.durable import FileLock, LockHeld, LockUnavailable, lock_is_held, read_owner


def _hold(path: Path, owner: dict[str, object]) -> int:
    """Another holder: a separate open file description, as another process would have."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps(owner).encode())
    return fd


def test_contention_names_the_owner(tmp_path: Path) -> None:
    path = tmp_path / "LOCK"
    fd = _hold(path, {"pid": 4242, "host": "elsewhere"})
    try:
        with pytest.raises(LockHeld) as held:
            FileLock.acquire(path)
        assert (held.value.pid, held.value.host) == (4242, "elsewhere")
    finally:
        os.close(fd)


def test_the_record_names_this_holder_and_the_callers_fields(tmp_path: Path) -> None:
    path = tmp_path / "LOCK"
    with FileLock.acquire(path, owner={"repo": "abc123"}):
        record = json.loads(path.read_text())
        assert record["pid"] == os.getpid() and record["repo"] == "abc123"
        assert read_owner(path) == (os.getpid(), record["host"])


def test_a_non_contention_errno_is_unavailable_not_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_locks(fd: int, op: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_locks)
    with pytest.raises(LockUnavailable, match="cannot lock"):
        FileLock.acquire(tmp_path / "LOCK")


def test_a_dead_holders_lock_is_dropped(tmp_path: Path) -> None:
    """The kernel releases the lock when the holder's descriptor closes, as it does when the
    process dies: no stale-lock reclaim, and no manual cleanup."""
    path = tmp_path / "LOCK"
    os.close(_hold(path, {"pid": 999_999, "host": "long-dead-host"}))
    FileLock.acquire(path).release()


def test_release_never_unlinks_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "LOCK"
    lock = FileLock.acquire(path)
    lock.release()
    lock.release()
    assert path.exists()
    FileLock.acquire(path).release()


def test_the_probe_reads_a_held_lock_as_held_and_a_free_one_as_free(tmp_path: Path) -> None:
    path = tmp_path / "LOCK"
    assert not lock_is_held(path), "no file is no holder"
    with FileLock.acquire(path):
        assert lock_is_held(path)
    assert not lock_is_held(path), "a leftover file is inert"


def test_the_probe_reads_cannot_tell_as_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "LOCK"
    path.write_text("{}")

    def refuse(*_: object, **__: object) -> int:
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(os, "open", refuse)
    assert lock_is_held(path)


def test_an_unreadable_record_is_an_unknown_owner(tmp_path: Path) -> None:
    path = tmp_path / "LOCK"
    path.write_text("not json")
    assert read_owner(path) == (-1, "?")
