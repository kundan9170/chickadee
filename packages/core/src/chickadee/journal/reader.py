"""Reading a journal back: line validation and corruption refusal.

The asymmetry here is the whole design. A **trailing** partial line is the one thing a
crash can produce, so ignoring it is the one legal repair. Anything else — a seq gap,
unparseable content mid-file, an event that will not validate — refuses, because a journal
we edited to make it parse is no longer the source of truth invariant 4 depends on.

The bytes of a torn tail are removed only by the writer's open (`durable.AppendLog`),
which only a resume makes.
"""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chickadee.durable import complete_lines_from
from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.paths import journal_path
from chickadee.schema import JournalEvent

if TYPE_CHECKING:
    from chickadee.state.roots import Roots


@dataclass(frozen=True)
class Tail:
    """The complete lines of a journal past an offset, read up to its first damage."""

    events: tuple[JournalEvent, ...]
    end: int
    """The byte offset just past the last line read: where the next read starts."""
    damage: JournalCorrupt | None
    """The first damaged complete line's refusal, or None. A complete line never changes, so
    a reader that keeps this need never read that line again."""


def read_tail(
    path: pathlib.Path, offset: int = 0, first_seq: int = 1, limit: int | None = None
) -> Tail:
    """The validated lines of `path` past byte `offset`, the first expected to be `first_seq`,
    at most `limit` of them when it is given.

    A torn trailing line is not read, and `end` stops before it, so the next read starts
    there once it is complete. Damage stops the read: the lines before it are returned and
    the damage beside them. A read that stops at `limit` parses nothing past it, and its
    `end` is just past its last line, so the next read starts there (a
    page costs its own lines, never the journal's). The one parser of a journal line
    (`read_events` reads through it). In memory only: a reader never mutates the journal,
    because it also serves eval archives and other people's runs."""
    events: list[JournalEvent] = []
    expected, end = first_seq, offset
    with path.open("rb") as fh:
        fh.seek(offset)
        # Split bytes, then decode each complete line, so a tail torn inside a multi-byte
        # character is dropped like any other torn tail (#276 B3).
        for line in complete_lines_from(fh):
            if limit is not None and len(events) >= limit:
                break
            try:
                event = _event(path, line[:-1], expected)
            except JournalCorrupt as exc:
                return Tail(tuple(events), end, exc)
            events.append(event)
            expected += 1
            end += len(line)
    return Tail(tuple(events), end, None)


def read_after(
    path: pathlib.Path, after: int, offset: int | None = None, limit: int | None = None
) -> Tail:
    """The lines past seq `after`, at most `limit`: read from byte `offset` when the reader
    knows where `after` ends. Else the first `after` complete lines are stepped over as bytes,
    unparsed, since a valid journal's line number is its seq; the first line read must then
    carry seq `after + 1`, so a journal that disagrees still refuses. A host's seek on a run
    (`scopes.run_page`): the lines as written, never folded."""
    if offset is None:
        offset, skipped = _skip_lines(path, after)
        if skipped < after:
            return Tail((), offset, None)
    return read_tail(path, offset, after + 1, limit)


def _skip_lines(path: pathlib.Path, count: int) -> tuple[int, int]:
    """The byte offset just past the first `count` complete lines, and how many there were."""
    offset = skipped = 0
    with path.open("rb") as fh:
        for line in complete_lines_from(fh):
            if skipped >= count:
                break
            offset += len(line)
            skipped += 1
    return offset, skipped


def _event(path: pathlib.Path, chunk: bytes, expected: int) -> JournalEvent:
    """One complete line, as the event with seq `expected`. A valid journal's line number is
    its seq, so the refusal names the line by it."""
    try:
        line = chunk.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JournalCorrupt(
            expected, str(path), f"line {expected} is not UTF-8: {exc}"
        ) from exc
    if not line.strip():
        raise JournalCorrupt(expected, str(path), f"blank line at {expected}")
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise JournalCorrupt(expected, str(path), f"line {expected}: {exc}") from exc
    try:
        event = JournalEvent.model_validate(data)
    except Exception as exc:  # pydantic ValidationError and anything it wraps
        raise JournalCorrupt(expected, str(path), f"line {expected}: {exc}") from exc
    if event.seq != expected:
        raise JournalCorrupt(
            expected, str(path), f"seq gap: expected {expected}, got {event.seq}"
        )
    return event


def read_events(path: pathlib.Path, *, repair_tail: bool = True) -> Iterator[JournalEvent]:
    """Yield validated events in seq order, repairing at most a torn trailing line."""
    if not path.exists():
        return
    if not repair_tail and path.stat().st_size:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                raise JournalCorrupt(None, str(path), "file does not end with a newline")
    tail = read_tail(path)
    yield from tail.events
    if tail.damage is not None:
        raise tail.damage


def read_run(roots: Roots, run_id: str) -> Iterator[JournalEvent]:
    """Run `run_id`'s journal, as `read_events` reads it: a torn tail dropped in memory and the
    file never touched. The one place a reader outside the journal package reaches a run's
    file (`context.project.JournalRecords`, `budget.family`).

    An absent journal raises `FileNotFoundError` here, at once, rather than reading as an
    empty one. The lines are read as they are iterated, so a reader that keeps what it read
    up to any damage (`projection.project_readable`) can."""
    path = journal_path(roots, run_id)
    if not path.is_file():
        raise FileNotFoundError(f"run {run_id} has no journal at {path}")
    return read_events(path)


def follow(
    roots: Roots, run_id: str, ended: threading.Event, poll_s: float = 0.05
) -> Iterator[JournalEvent]:
    """Run `run_id`'s journal from its first line, then live, until `ended` is set and every
    line written by then is read: a watcher's view of a run as it goes (`CoderHandle.events`).
    Read as written, never folded, so what a watcher sees is what
    replay reads. A journal that does not exist by the end yields nothing: the run was
    stopped before it began. Damage raises, after the lines before it."""
    path = journal_path(roots, run_id)
    offset, seq = 0, 1
    while True:
        finished = ended.is_set()
        if path.exists():
            tail = read_tail(path, offset, seq)
            yield from tail.events
            offset, seq = tail.end, seq + len(tail.events)
            if tail.damage is not None:
                raise tail.damage
        if finished:
            return
        ended.wait(poll_s)
