"""A journal scope's records as a host reads them: a session or a run.

**One read, two logs.** A session's log (`context/store.py`) and a run's journal
(`journal/reader.py`) each number their lines from 1 with no gaps, and each is read here past
a cursor into one record type. That is the seek §5 is built on: a reader says where it got
to, and the file is read from there. Nothing is kept for it between reads.

**A cursor is two numbers, never a buffer.** `after` is the last seq the reader holds. For a
run, `offset` is how far into the journal that seq ends, so the next read starts there instead
of at byte 0; a cursor without one is still correct, only slower. A session's log is read by
its holder (the engine's fold cache extends it once for every reader), so a session's cursor
needs only `after`.

**Payloads cross as stored**: a session line's payload is its stored
JSON object, a run event's is the payload its kind's model wrote. Nothing is re-rendered from
richer state, which is what keeps unscrubbed content off a socket.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from pydantic import JsonValue

from chickadee.context.store import SessionLog
from chickadee.errors import HarnessBug
from chickadee.journal.reader import read_after
from chickadee.journal.writer import AS_STORED
from chickadee.schema import JournalEvent

__all__ = ["PAGE_DEFAULT", "Cursor", "Page", "ScopeRecord", "run_page", "session_page"]

PAGE_DEFAULT: Final = 256
"""Records one read returns at most: `RECORD_PAGE`. A read size, not a
semantic bound; a reader asks again from the cursor it was given."""


@dataclass(frozen=True)
class ScopeRecord:
    """One durable line of a scope, the same shape for a session and a run."""

    scope: str
    """The session id or the run id the line is numbered in."""
    seq: int
    at: str
    """When it was written, for people. Ordering is `seq`."""
    kind: str
    run_id: str | None
    """A session line's run, when it has one; a run event's own run."""
    payload: Mapping[str, JsonValue]
    task_id: str | None = None
    """A run event's task, when it names one. Always None for a session line."""
    round: int | None = None
    """The worker's round within `task_id`."""


@dataclass(frozen=True)
class Cursor:
    after: int = 0
    """The last seq the reader holds; `0` is the start, which the log always has."""
    offset: int | None = None
    """Where `after` ends in a run's journal, when known. None reads from the start."""


@dataclass(frozen=True)
class Page:
    records: tuple[ScopeRecord, ...]
    cursor: Cursor
    """Where to read from next: past the last record returned."""


def session_page(log: SessionLog, cursor: Cursor, limit: int = PAGE_DEFAULT) -> Page:
    """The lines of `log` past `cursor`, at most `limit`, each with its payload.

    The log is the caller's: the engine passes the one its fold cache keeps, extended once for
    every reader, so a session's seek needs no offset of its own and the returned
    cursor carries none."""
    _check(cursor, limit)
    # A log's seqs run from 1 with no gap, a fork's ancestry included (`SessionLog.event`
    # reads by index the same way), so the page is a slice: an attached client's poll costs
    # its page, never the log's length.
    taken = log.events()[cursor.after : cursor.after + limit]
    records = tuple(
        ScopeRecord(
            scope=log.session_id,
            seq=e.seq,
            at=e.at,
            kind=e.kind,
            run_id=e.run_id,
            payload=_object(log.payload(e), log.session_id, e.seq),
        )
        for e in taken
    )
    return Page(records, Cursor(taken[-1].seq if taken else cursor.after))


def run_page(journal: Path, run_id: str, cursor: Cursor, limit: int = PAGE_DEFAULT) -> Page:
    """A run journal's events past `cursor`, at most `limit`, and a cursor that names the byte
    offset its last record ends at, so the next page reads only its own lines. A journal not
    yet written reads as empty, so a reader that asks before the run's first event waits
    rather than failing. Damage is served as every reader serves it: the records before it
    first, then `JournalCorrupt` once a page would hold nothing else."""
    _check(cursor, limit)
    if not journal.is_file():
        return Page((), cursor)
    tail = read_after(journal, cursor.after, cursor.offset, limit)
    if tail.damage is not None and not tail.events:
        raise tail.damage
    records = tuple(_run_record(run_id, e) for e in tail.events)
    after = tail.events[-1].seq if tail.events else cursor.after
    return Page(records, Cursor(after, tail.end))


def _run_record(run_id: str, e: JournalEvent) -> ScopeRecord:
    if e.run_id != run_id:
        raise HarnessBug(f"run {run_id}'s journal holds an event of run {e.run_id}")
    return ScopeRecord(
        scope=run_id,
        seq=e.seq,
        at=e.ts,
        kind=e.kind.value,
        run_id=e.run_id,
        payload=cast("dict[str, JsonValue]", e.payload.model_dump(mode="json", **AS_STORED)),
        task_id=e.task_id,
        round=e.round,
    )


def _object(raw: bytes, session_id: str, seq: int) -> Mapping[str, JsonValue]:
    """A stored payload is a JSON object: the store wrote it so, and checked it on read."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise HarnessBug(f"session {session_id} line {seq}'s payload is not an object")
    return cast("dict[str, JsonValue]", data)


def _check(cursor: Cursor, limit: int) -> None:
    if cursor.after < 0 or limit < 1:
        raise ValueError(
            f"a cursor starts at 0 and a page holds a record; got {cursor}, {limit}"
        )
