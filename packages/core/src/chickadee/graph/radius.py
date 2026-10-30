"""Blast radius: recursive CTE over reverse edges.

Two rules that look like bugs and are not.

**`unknown` edges are included.** An unknown edge means "something here references
something and the graph cannot say what". Excluding it would shrink the verifier's affected
set on exactly the edges least understood, which is the wrong direction: over-inclusion
costs a few extra checks, under-inclusion ships an unverified change (matches scout SC-4).

**`stale` edges are included too**, for the same reason — GR-2 marks an edge stale precisely
when its target vanished, which is when the caller most needs to be looked at.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass

UPGRADE_TRIGGER_MS = 200
"""GR-1: hop-3 averaging over this on a real repository is the trigger to move to a graph
store. Measured and reported rather than assumed — the decision says *not before*."""


@dataclass(frozen=True)
class RadiusResult:
    files: tuple[str, ...]
    symbols: tuple[str, ...]
    wall_ms: int
    over_trigger: bool
    """True when this traversal exceeded GR-1's threshold. Surfaced so the trigger fires on
    evidence instead of on someone remembering to time it."""
    reached: tuple[tuple[str, str | None, str], ...] = ()
    """Each reached symbol as `(symbol, file, edge_class)`: the class of its best path, where
    a path is as confident as its weakest step. A symbol any `resolved` path reaches is
    affected; one only `unknown` steps reach is possibly affected. `impact` reports it so;
    the seeds are `resolved`."""


_ORDER = ("resolved", "fallback", "unknown")

_CTE = """
WITH RECURSIVE reach(sym, depth, cls) AS (
    SELECT value, 0, 'resolved' FROM json_each(?)
  UNION
    SELECT e.src, reach.depth + 1,
      CASE
        WHEN reach.cls = 'unknown' OR e.edge_class = 'unknown' THEN 'unknown'
        WHEN reach.cls = 'fallback' OR e.edge_class = 'fallback' THEN 'fallback'
        ELSE 'resolved'
      END
    FROM edges e JOIN reach ON e.dst = reach.sym
    WHERE reach.depth < ?
)
SELECT DISTINCT reach.sym AS sym, s.file AS file, reach.cls AS cls
FROM reach LEFT JOIN symbols s ON s.qualified_name = reach.sym
"""


def blast_radius(conn: sqlite3.Connection, syms: Sequence[str], depth: int = 2) -> RadiusResult:
    """Everything that could be affected by changing `syms`, to `depth` hops backwards."""
    started = time.perf_counter()
    rows = conn.execute(_CTE, (json.dumps(list(syms)), depth)).fetchall()
    wall_ms = int((time.perf_counter() - started) * 1000)
    files = sorted({r["file"] for r in rows if r["file"]})
    symbols = sorted({r["sym"] for r in rows if r["sym"]})
    best: dict[tuple[str, str | None], str] = {}
    for r in rows:
        if not r["sym"]:
            continue
        key = (r["sym"], r["file"])
        held = best.get(key)
        if held is None or _ORDER.index(r["cls"]) < _ORDER.index(held):
            best[key] = r["cls"]
    return RadiusResult(
        files=tuple(files),
        symbols=tuple(symbols),
        wall_ms=wall_ms,
        over_trigger=depth >= 3 and wall_ms > UPGRADE_TRIGGER_MS,
        reached=tuple(
            (sym, file, cls) for (sym, file), cls in sorted(best.items(), key=_by_sym)
        ),
    )


def _by_sym(item: tuple[tuple[str, str | None], str]) -> tuple[str, str]:
    (sym, file), _ = item
    return sym, file or ""
