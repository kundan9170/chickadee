"""`locate()`: the discovery rung as an on-demand tool.

Zeus calls `locate(question)` when it can only *describe* what it wants. A snapshot built by
code (`snapshot`) orients a cheap model (`rung`) that searches with the ladder's shared tools
and its own `search` (`tools`), may request a read-only command it cannot run itself
(`escapes`), and answers in pointers that a deterministic gate checks before Zeus sees them
(`findings`). The loop (`loop`) never escalates: a short report goes back to Zeus as it is,
and a scout that cannot search says `UNAVAILABLE` rather than guessing. Nothing here reads
names out of the question: a name Zeus can write, Zeus looks up itself.
Records go through a sink the caller supplies (`sink`); recurring escapes are promoted into
the snapshot (`hints`).
"""

from __future__ import annotations

from chickadee.scout.escapes import EscapeOutcome, EscapeRequest, Escapes, action_in_disguise
from chickadee.scout.findings import Checker, Report, gate
from chickadee.scout.hints import Hints
from chickadee.scout.loop import Scout
from chickadee.scout.rung import ScoutRung, scout_rung
from chickadee.scout.search import Search
from chickadee.scout.sink import JournalSink, ScoutSink
from chickadee.scout.snapshot import Snapshot, SnapshotCache, build_snapshot
from chickadee.scout.tools import LocateCaps, ScoutTools

__all__ = [
    "Checker",
    "EscapeOutcome",
    "EscapeRequest",
    "Escapes",
    "Hints",
    "JournalSink",
    "LocateCaps",
    "Report",
    "Scout",
    "ScoutRung",
    "ScoutSink",
    "ScoutTools",
    "Search",
    "Snapshot",
    "SnapshotCache",
    "action_in_disguise",
    "build_snapshot",
    "gate",
    "scout_rung",
]
