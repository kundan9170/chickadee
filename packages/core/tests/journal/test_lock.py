"""LOCK: a second writer is refused; a dead holder blocks nothing.

Exclusion is `flock` on an open descriptor, so the kernel releases it when the holder
dies. There is no stale-lock concept to get wrong, which is what the file-presence
protocol could not make atomic.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest
from helpers import payload_for

from chickadee.journal import (
    Journal,
    JournalCorrupt,
    JournalWriteError,
    ResumeRefused,
    paths,
    read_events,
)
from chickadee.schema import EventKind
from chickadee.state.roots import Roots


def test_second_open_refused_while_live(repo: Roots, run_id: str) -> None:
    j = Journal.create(repo, run_id, {})
    try:
        with pytest.raises(ResumeRefused):
            Journal.create(repo, run_id, {})
    finally:
        j.close()


def test_lock_left_by_a_dead_process_does_not_block(repo: Roots, run_id: str) -> None:
    """A crashed run must not need manual cleanup — that would make every crash a
    support ticket.

    Under `flock` there is nothing to reclaim: the kernel released the lock when the
    holder died, and the leftover file is inert. Its contents can name any pid at all.
    """
    Journal.create(repo, run_id, {}).close()
    lock = paths.lock_path(repo, run_id)
    lock.write_text(json.dumps({"pid": 999_999, "host": socket.gethostname()}))
    Journal.open_for_resume(repo, run_id).close()


def test_close_releases_the_lock(repo: Roots, run_id: str) -> None:
    """Asserted behaviour, not the artifact: the LOCK file is deliberately left in place
    (unlinking it would let a third process lock a fresh inode while a second waits on
    the old one), so "released" means the next open succeeds."""
    j = Journal.create(repo, run_id, {})
    assert paths.lock_path(repo, run_id).exists()
    j.close()
    Journal.open_for_resume(repo, run_id).close()


def test_unreadable_lock_is_treated_as_stale(repo: Roots, run_id: str) -> None:
    Journal.create(repo, run_id, {}).close()
    paths.lock_path(repo, run_id).write_text("garbage")
    Journal.open_for_resume(repo, run_id).close()


def test_failed_resume_releases_the_lock(repo: Roots, run_id: str) -> None:
    """A corrupt journal is exactly when the caller wants to retry.

    The lock is taken before the journal is read, so an exception in between used to
    strand it — holding our own live pid, which made the process refuse itself for the
    rest of its life. Recovering from corruption then required exiting the program.
    """
    Journal.create(repo, run_id, {}).close()
    paths.journal_path(repo, run_id).write_text("{not json at all\n")

    # Twice: the second attempt reaching the same JournalCorrupt — rather than
    # ResumeRefused — is the proof that the first released. Asserting on the file would
    # test the artifact, which is deliberately left behind.
    for _ in range(2):
        with pytest.raises(JournalCorrupt):
            Journal.open_for_resume(repo, run_id)


def test_a_held_lock_refuses_regardless_of_file_contents(repo: Roots, run_id: str) -> None:
    """The kernel decides, not the text.

    An outside descriptor holds the lock while the file claims a long-dead owner. A
    contents-based protocol would call that stale and reclaim it, admitting a second
    writer — the exact race that made unlink-and-recreate unfixable.
    """
    lock = paths.lock_path(repo, run_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.write(fd, json.dumps({"pid": 999_999, "host": "long-dead-host"}).encode())
    try:
        with pytest.raises(ResumeRefused):
            Journal.create(repo, run_id, {})
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    # Once the holder lets go, the same run opens without manual cleanup.
    Journal.create(repo, run_id, {}).close()


def test_release_does_not_unlink(repo: Roots, run_id: str) -> None:
    """Keeping the file is the point: an unlink between another process opening it and
    locking it would put two writers on two different inodes."""
    j = Journal.create(repo, run_id, {})
    j.close()
    assert paths.lock_path(repo, run_id).exists()


_CHILD = """
import sys, time, pathlib
sys.path.insert(0, {tests!r})
from helpers import payload_for
from chickadee.journal.writer import Journal
from chickadee.journal.errors import ResumeRefused
from chickadee.schema import EventKind

from chickadee.state.roots import roots as resolve_roots
roots = resolve_roots(pathlib.Path(sys.argv[1]))
run_id = sys.argv[2]
barrier, name = pathlib.Path(sys.argv[3]), sys.argv[4]
(barrier / ('ready.' + name)).touch()
while not (barrier / 'go').exists():
    time.sleep(0.005)

try:
    journal = Journal.create(roots, run_id, {{}})
except ResumeRefused:
    sys.exit(3)
for _ in range(3):
    journal.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL))
    time.sleep(0.1)
journal.close()
"""

CHILDREN = 8


@pytest.mark.writes_worktree("_barrier", "_barrier/*")
def test_concurrent_processes_produce_one_writer(
    worktree: pathlib.Path, repo: Roots, run_id: str
) -> None:
    """The invariant, under real contention.

    Single-process tests cannot see the failure this guards: two writers each hold their
    own sequence counter, so the damage only appears when both actually run.

    The barrier is what makes that happen. Spawning eight interpreters and hoping they
    collide depends on startup skew — about a millisecond on an idle laptop, but longer
    than the whole lock hold on a loaded CI runner, where children then take the lock in
    sequence. That direction fails loudly; the other one does not, and a run where nobody
    overlapped would pass while testing nothing. So every child announces itself and waits
    for `go`, and contention is arranged rather than hoped for.
    """
    paths.run_dir(repo, run_id).mkdir(parents=True, exist_ok=True)
    barrier = worktree / "_barrier"
    barrier.mkdir()
    script = _CHILD.format(tests=str(pathlib.Path(__file__).parent))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(worktree), run_id, str(barrier), str(i)]
        )
        for i in range(CHILDREN)
    ]
    try:
        deadline = time.monotonic() + 60
        while len(list(barrier.glob("ready.*"))) < CHILDREN:
            assert time.monotonic() < deadline, "children never reached the barrier"
            time.sleep(0.01)
        (barrier / "go").touch()
        codes = [p.wait(timeout=60) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()

    winners = codes.count(0)
    assert winners == 1, f"{winners} processes wrote the same run concurrently"
    assert codes.count(3) == CHILDREN - 1, f"unexpected child exits: {codes}"

    seqs = [e.seq for e in read_events(paths.journal_path(repo, run_id))]
    assert seqs == list(range(1, len(seqs) + 1))
    assert len(seqs) == 3 * winners


def test_operational_flock_failure_is_not_reported_as_contention(
    repo: Roots, run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem that cannot lock is not a busy run.

    Reporting `ENOLCK` as `ResumeRefused` would name an owner that does not exist. On a
    filesystem without flock support the lie would be permanent — every run would look
    locked, and the recorded pid would send whoever debugs it after a live process.
    """
    paths.run_dir(repo, run_id).mkdir(parents=True, exist_ok=True)

    def _no_locks(fd: int, op: int) -> None:
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", _no_locks)
    with pytest.raises(JournalWriteError, match="cannot lock"):
        Journal.create(repo, run_id, {})


def test_lock_is_released_when_close_fails(repo: Roots, run_id: str) -> None:
    """`close()` must free the run even when flushing it does not.

    Skipping the release on a failing close holds the kernel lock for the life of the
    process, so every later create or resume on this run is refused — the same leak the
    resume path had, on the way out instead of the way in.
    """
    journal = Journal.create(repo, run_id, {})
    journal.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL))
    fd = journal._log._fd  # the run's `durable.AppendLog` holds the descriptor
    assert fd is not None
    os.close(fd)  # the fsync and close will now both fail

    with pytest.raises(OSError):
        journal.close()
    Journal.open_for_resume(repo, run_id).close()  # the run is free again
