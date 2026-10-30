"""`ScoutSink`: the port a caller supplies for the scout's records.

The scout does not choose where its records go. A `locate` from Zeus writes the session log
; inside a coder episode it writes
the run journal. Either way the scout emits the same frozen payloads and never learns which.

What it emits is closed: the four live `scout_*` kinds (`scout_escalated` is retired, not
removed — the scout never escalates), a granted escape's D40 effect phases, and the
Executor's `exec` record for it. `record` returns only once the record is durable, because an
escape's `effect_intended` must be on disk before the command is dispatched: a crash
between the two is then a started effect recovery can see, never one it cannot.
"""

from __future__ import annotations

from typing import Protocol

from chickadee.errors import HarnessBug
from chickadee.journal.effects import EFFECT_EVENTS
from chickadee.journal.writer import Journal
from chickadee.schema.journal import KIND_TABLE, EventKind, P

__all__ = ["RECORD_KINDS", "JournalSink", "ScoutSink", "check_record"]

RECORD_KINDS: frozenset[EventKind] = (
    frozenset(
        {
            EventKind.SCOUT_START,
            EventKind.SCOUT_TOOL,
            EventKind.SCOUT_ESCAPE,
            EventKind.SCOUT_REPORT,
            EventKind.EXEC,
        }
    )
    | EFFECT_EVENTS
)
"""Everything the scout may record. `SCOUT_ESCALATED` is deliberately absent."""


class ScoutSink(Protocol):
    def record(self, kind: EventKind, payload: P) -> int:
        """Write one record and return its sequence number in the log it went to. Returns
        only once the record is durable."""
        ...


def check_record(kind: EventKind, payload: P) -> None:
    """A kind outside `RECORD_KINDS`, or a payload that is not its kind's model, is a harness
    bug in the scout — raised here so a sink cannot write it."""
    if kind not in RECORD_KINDS:
        raise HarnessBug(f"the scout does not record {kind.value!r} (14-scout.md §9)")
    model, _ = KIND_TABLE[kind]
    if type(payload) is not model:
        raise HarnessBug(f"{kind.value} carries {model.__name__}, not {type(payload).__name__}")


class JournalSink:
    """The run journal as the sink — a `locate` inside a coder episode.

    Durability is the Journal's own policy: every line is flushed, and the D40 effect
    phases are fsynced before `append` returns, which is the order the escape needs.
    """

    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def record(self, kind: EventKind, payload: P) -> int:
        check_record(kind, payload)
        return self._journal.append(kind, payload)
