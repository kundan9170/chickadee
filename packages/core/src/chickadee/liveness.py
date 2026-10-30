"""Liveness: whether anyone is still working on a session.

*Busy* and *the process died mid-turn* leave the same records, so no reading of the log can
tell them apart. Liveness is therefore a fact about a **process**, read from the thing that
already tracks one: a lock that records its holder's pid and host, which the kernel releases
when the holder dies.

**The session's lock is that lease.** A session is written by the one process that holds its
log's `LOCK` (`context/store.py`), for as long as it has the session open. The repository's
write lease could not answer for one session: another session's run may be what holds it.

A projection, computed on every read and stored nowhere. It writes nothing, and it never
repairs what it sees: reclaiming a dead holder's lock is the next writer's, never a watcher's.

- `IDLE`: nothing is in flight (no question waiting, no call waiting for its result, no run
  without its end, no user turn without a reply). It carries the last run's outcome, when a
  run ended. It is also what a session reads when the process hosting it says its last turn
  ended on an error (`failed`) before replying: the unanswered user turn is then not in
  flight, since nobody is working on it, and `why` says what ended it. That is the
  process's own knowledge, so a reader in another process, or after a restart, lacks it.
- `AWAITING`: a question is waiting for an answer, and a live process on this host holds the
  lock to receive it. "Waiting for you" is the more useful truth than `WORKING`, but only
  while someone is there to take the answer: a question its process died asking is
  `ABANDONED`.
- `WORKING`: something else is in flight, and a live process on this host holds the lock.
- `UNKNOWN`: something is in flight, and the lock's holder is on another host, which cannot be
  inspected (invariant 20). Never `ABANDONED` on a guess.
- `ABANDONED`: something is in flight, and nobody alive holds the lock: the process died.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from chickadee.context.project import RunRecords, paused_runs, run_ended
from chickadee.context.rewind import safe_point_refusal
from chickadee.durable import lock_is_held, read_owner
from chickadee.graph.indexer import IndexProgress
from chickadee.journal.errors import JournalCorrupt

if TYPE_CHECKING:
    from chickadee.context.questions import QuestionAskedLine
    from chickadee.context.session import SessionState

__all__ = ["OPENING", "Liveness", "LivenessKind", "liveness"]


class LivenessKind(StrEnum):
    IDLE = "idle"
    AWAITING = "awaiting"
    WORKING = "working"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Liveness:
    """Computed on every read, stored nowhere. It carries why: `abandoned` with no `last_seq`
    would be an accusation rather than a finding."""

    kind: LivenessKind
    last_seq: int
    outcome: str | None = None
    """`IDLE` only: the last run's outcome, when a run has ended."""
    question_id: str | None = None
    """`AWAITING` only."""
    owner_host: str | None = None
    """`UNKNOWN` only: the host the lock's holder is on."""
    why: str | None = None
    """`WORKING`, `UNKNOWN` and `ABANDONED`: what is in flight. `IDLE`: why the last turn
    ended without a reply, when the process hosting the session knows it did."""
    paused_run: str | None = None
    """Any kind: the latest run whose `coder` call was answered only that it paused, and that
    has not ended since (`project.paused_runs`): what opening the session again
    resumes. Not one that reported how it ended, nor one whose journal will not read, nor the
    run working in the background now (`team_run`)."""
    team_run: str | None = None
    """Any kind: the session's run working in the background now (attended), which
    only the process hosting it knows; its journal alone cannot tell it from a paused one."""
    indexing: str | None = None
    """Any kind: the repository's index being built in the background by the process hosting
    the session, or why that failed; None once it is built, or when none was."""
    index_progress: IndexProgress | None = None
    """Any kind: how far that build has got, set while it runs and never otherwise:
    `indexing` set with no progress is a build that failed."""


def liveness(
    state: SessionState,
    waiting: tuple[QuestionAskedLine, ...],
    records: RunRecords,
    lock: Path,
    *,
    host: str | None = None,
    failed: str | None = None,
    opening: bool = False,
    team: str | None = None,
    indexing: str | None = None,
    index_progress: IndexProgress | None = None,
) -> Liveness:
    """The session's liveness from its fold, its open questions, its runs' records and its
    log's lock. `failed` is why the last turn ended on an error, given by the process hosting
    the session while its loop is idle. `opening` is that process saying a turn waits on the
    session's build: someone is working on it before its log says so, and its lock
    may not be taken yet. `team` is the run working in the background now, given by the same
    process: the run's state beside Zeus's. `host` is this machine's name; given
    only by a test."""
    found = _liveness(state, waiting, records, lock, host=host, failed=failed, opening=opening)
    # A run handed over in the background was answered `RUNNING`, with no outcome, as a paused
    # one was: the one working now is the team's, never paused.
    paused = [r for r in paused_runs(state) if r != team and run_ended(records, r) is False]
    found = replace(found, team_run=team, indexing=indexing, index_progress=index_progress)
    return replace(found, paused_run=paused[-1]) if paused else found


def _liveness(
    state: SessionState,
    waiting: tuple[QuestionAskedLine, ...],
    records: RunRecords,
    lock: Path,
    *,
    host: str | None,
    failed: str | None,
    opening: bool,
) -> Liveness:
    if opening:
        return Liveness(LivenessKind.WORKING, state.last_seq, why=OPENING)
    in_flight = (
        f"question {waiting[0].question_id} is waiting for an answer"
        if waiting
        else _in_flight(state, records)
    )
    why: str | None = None
    if failed is not None and in_flight == _NO_REPLY:
        # Nobody is working on the unanswered turn: it ended. A turn that did reply and then
        # reported an error (an answer cut at max_tokens) was answered, and stays plain IDLE.
        why = f"the last turn ended without a reply: {failed}"
        in_flight = None
    if in_flight is None:
        return Liveness(
            LivenessKind.IDLE, state.last_seq, outcome=_outcome(state, records), why=why
        )
    if not lock_is_held(lock):
        return Liveness(LivenessKind.ABANDONED, state.last_seq, why=in_flight)
    pid, owner = read_owner(lock)
    if owner != (host if host is not None else socket.gethostname()):
        return Liveness(LivenessKind.UNKNOWN, state.last_seq, owner_host=owner, why=in_flight)
    if not _alive(pid):
        return Liveness(LivenessKind.ABANDONED, state.last_seq, why=in_flight)
    if waiting:
        return Liveness(
            LivenessKind.AWAITING, state.last_seq, question_id=waiting[0].question_id
        )
    return Liveness(LivenessKind.WORKING, state.last_seq, why=in_flight)


OPENING: Final = "opening the session"
"""Why a session whose turn waits on its build is working."""

_NO_REPLY: Final = "the last user turn has no reply yet"


def _in_flight(state: SessionState, records: RunRecords) -> str | None:
    """What is unfinished, or None: a call or a run (the safe-point rule, one home), or a user
    turn the model has not answered."""
    why = safe_point_refusal(state, records)
    if why is not None:
        return why
    turns = state.turns()
    if turns and max(turns, key=lambda t: t.seq).role == "user":
        return _NO_REPLY
    return None


def _outcome(state: SessionState, records: RunRecords) -> str | None:
    episodes = state.episodes()
    if not episodes:
        return None
    try:
        end = records.projection(episodes[-1].run_id).end
    except (FileNotFoundError, OSError, JournalCorrupt):
        # The outcome is a courtesy on IDLE: a run journal that cannot be read says nothing
        # about whether anyone is working on the session, so it costs only the outcome.
        return None
    return end.outcome if end is not None else None


def _alive(pid: int) -> bool:
    """Whether `pid` names a live process. A pid the lock could not record is not alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
