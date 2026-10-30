"""A failed append never lands later, and the journal stops until resumed (D149 DU-7).

The record the caller was told failed must not be in the journal: not on the next append,
not on `close()`. A buffered handle kept the failed line and flushed it later (#276 B2), so
the journal recorded an event that "did not happen", and a retry at the same seq wrote two
lines with one number, which no reader accepts.
"""

from __future__ import annotations

import pathlib
import resource
import signal

import pytest
from helpers import payload_for, scope_args

from chickadee.journal import Journal, JournalWriteError, paths, read_events
from chickadee.schema import EventKind
from chickadee.state.roots import Roots


def _append(j: Journal, kind: EventKind) -> int:
    return j.append(kind, payload_for(kind), **scope_args(kind))


class _DiskFull:
    """Cap the file size just past the journal's end, as a full disk does: the next append
    lands a few bytes and fails. The kernel enforces it, so it reaches the write however the
    writer makes it, buffered or not. `SIGXFSZ` is ignored so the write fails with `EFBIG`
    instead of killing the process."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self._saved = resource.getrlimit(resource.RLIMIT_FSIZE)

    def __enter__(self) -> None:
        _, hard = self._saved
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.path.stat().st_size + 8, hard))

    def __exit__(self, *exc: object) -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, self._saved)


@pytest.fixture(autouse=True)
def _no_sigxfsz():
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    yield
    signal.signal(signal.SIGXFSZ, previous)


def test_a_failed_append_never_reaches_the_file(
    repo: Roots,
    run_id: str,
) -> None:
    j = Journal.create(repo, run_id, {})
    _append(j, EventKind.RUN_START)
    with (
        _DiskFull(paths.journal_path(repo, run_id)),
        pytest.raises(JournalWriteError, match="seq=2"),
    ):
        _append(j, EventKind.TASK_REPORTED)
    j.close()
    events = list(read_events(paths.journal_path(repo, run_id), repair_tail=False))
    assert [(e.seq, e.kind) for e in events] == [(1, EventKind.RUN_START)]


def test_every_append_after_a_failure_raises_until_resume(
    repo: Roots,
    run_id: str,
) -> None:
    j = Journal.create(repo, run_id, {})
    _append(j, EventKind.RUN_START)
    with _DiskFull(paths.journal_path(repo, run_id)), pytest.raises(JournalWriteError):
        _append(j, EventKind.TASK_REPORTED)
    # What is durable after a failed write is uncertain; writing on would build on a guess.
    with pytest.raises(JournalWriteError, match="seq=2"):
        _append(j, EventKind.SCOUT_TOOL)
    j.close()
    assert [e.seq for e in read_events(paths.journal_path(repo, run_id))] == [1]


def test_resume_after_a_failure_reads_what_is_on_disk_and_continues(
    repo: Roots,
    run_id: str,
) -> None:
    j = Journal.create(repo, run_id, {})
    _append(j, EventKind.RUN_START)
    with _DiskFull(paths.journal_path(repo, run_id)), pytest.raises(JournalWriteError):
        _append(j, EventKind.TASK_REPORTED)
    j.close()

    resumed = Journal.open_for_resume(repo, run_id)
    state = resumed.projection()
    assert state.last_seq == 1
    assert _append(resumed, EventKind.SCOUT_TOOL) == 2
    resumed.close()
    events = list(read_events(paths.journal_path(repo, run_id), repair_tail=False))
    assert [(e.seq, e.kind) for e in events] == [
        (1, EventKind.RUN_START),
        (2, EventKind.SCOUT_TOOL),
    ]
