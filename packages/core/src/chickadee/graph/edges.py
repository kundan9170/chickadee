"""Edge resolution and classification.

D19's rule in one line: **prefer `unknown` over a wrong edge.** Every consumer downstream
treats classes differently — the packer renders `# callers: 2 resolved, 1 unknown`, and the
verifier puts `unknown` edges *inside* the affected set because over-inclusion is the safe
direction. All of that depends on the class being honest, so the only thing this
module must never do is record a guess as `resolved`.

**The LSP seam.** `Indexer` asks `Resolver.definition` while classifying edges, after a
`prefetch` batch when the resolver offers one. Confirmed → `resolved`. `None` → name-match
`fallback` / `unknown`. Gate A5 is not this path: it is diagnostics, later, on a patch.
`NoResolver` is the opt-out (and the test default so goldens do not depend on pyright).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from chickadee.errors import InfraError
from chickadee.graph.parse import Reference

if TYPE_CHECKING:
    from chickadee.graph.diagnostics import Verdict
    from chickadee.graph.lsp import DefQuery

EdgeClass = Literal["resolved", "fallback", "unknown"]

SQL_BATCH = 900
"""Upper bound on names bound per statement, capped further by what the connection allows."""

_BIND_MARGIN = 8
"""Headroom for placeholders that are not names. `mark_stale_for_deleted` binds one extra
(the edge class); the rest is deliberate slack so a future statement gaining a parameter
does not silently start overflowing."""


def batch_size(conn: sqlite3.Connection) -> int:
    """Names per statement, read from the connection rather than assumed.

    SQLite caps bound variables per statement: 32766 on current builds, 999 on older ones,
    and a caller may lower it further. A hard-coded constant is a guess that is wrong on
    both sides — too large on an old build, needlessly small on a new one — so ask.

    Raises when the cap cannot fit even one name plus the fixed placeholders. Clamping to 1
    looked safe and was not: `mark_stale_for_deleted` binds the edge class as well, so at a
    limit of 1 a single-name batch still needs two bindings and SQLite rejects it with
    "too many SQL variables" — a message naming neither the limit nor the caller.
    """
    try:
        limit = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    except (AttributeError, sqlite3.Error):  # pragma: no cover - very old bindings
        limit = 999
    usable = limit - _BIND_MARGIN
    if usable < 1:
        raise InfraError(
            f"this SQLite connection allows {limit} bound variables per statement, which "
            f"cannot fit one symbol name plus {_BIND_MARGIN} fixed placeholders. The graph "
            "cannot mark edges stale under that limit; raise it or rebuild the index."
        )
    return min(SQL_BATCH, usable)


def chunks(items: Sequence[str], size: int) -> Iterator[tuple[str, ...]]:
    for start in range(0, len(items), size):
        yield tuple(items[start : start + size])


RESOLVED: EdgeClass = "resolved"
FALLBACK: EdgeClass = "fallback"
UNKNOWN: EdgeClass = "unknown"


@dataclass(frozen=True)
class CandidateEdge:
    src: str
    dst: str
    type: str
    edge_class: EdgeClass
    file: str
    line: int


class Resolver(Protocol):
    """An authoritative answer to "what does this name refer to, here?".

    Returning `None` means *this resolver could not say* — never "no such symbol". The
    difference decides `fallback` versus `unknown`, so a resolver that reports failure as an
    answer would manufacture confident wrong edges.
    """

    def definition(self, name: str, file: str, line: int) -> str | None: ...


@runtime_checkable
class Prefetching(Resolver, Protocol):
    """A resolver that asks its questions in a batch before `definition` is read, and returns
    the verdicts its servers published on the way (`graph/lsp.CheckerResolver`)."""

    def prefetch(
        self,
        queries: Sequence[DefQuery],
        lookup: Callable[[str, int], str | None],
        files: Sequence[str] = (),
    ) -> tuple[Verdict, ...]: ...


@runtime_checkable
class Rechecking(Protocol):
    """A resolver whose servers outlive a pass, so the files a change marked stale can be
    asked their verdicts after it (`graph/servers.EpisodeServers`)."""

    def recheck(self, paths: Sequence[str], asked_at: int) -> None: ...


class NoResolver:
    """Opt out of LSP. Tests use this so goldens do not depend on a checker binary."""

    def definition(self, name: str, file: str, line: int) -> str | None:
        return None


def classify(
    references: tuple[Reference, ...],
    file: str,
    known_symbols: dict[str, str],
    resolver: Resolver | None = None,
) -> tuple[CandidateEdge, ...]:
    """Turn parsed references into classified edges.

    `known_symbols` maps a bare name to its qualified name, and is the *fallback* evidence:
    a name that matches exactly one declaration in the repository is a good guess and a
    guess is what `fallback` means. A name matching several is **not** downgraded to a
    coin-flip — it becomes `unknown`, because picking one would be precisely the wrong edge
    D19 forbids.
    """
    resolver = resolver or NoResolver()
    edges: list[CandidateEdge] = []
    for ref in references:
        target = resolver.definition(ref.name, file, ref.line)
        if target is not None:
            edge_class, dst = RESOLVED, target
        else:
            match = known_symbols.get(ref.name)
            if match is None:
                edge_class, dst = UNKNOWN, ref.name
            else:
                edge_class, dst = FALLBACK, match
        edges.append(
            CandidateEdge(
                src=ref.from_symbol,
                dst=dst,
                type=ref.kind,
                edge_class=edge_class,
                file=file,
                line=ref.line,
            )
        )
    return tuple(edges)


def name_index(conn: sqlite3.Connection, lang: str) -> dict[str, str]:
    """Bare name → qualified name, for names unambiguous **within one language**.

    Two restrictions, and both exist to keep `fallback` from meaning "wrong".

    *Ambiguous names are absent* rather than mapped to a first match, so their references
    become `unknown`. That is the honest answer and the safe one: the verifier over-includes
    on unknown, so an ambiguous call still pulls its file into the affected set instead of
    pointing confidently at the wrong definition.

    *Matching is scoped to the language*, because a name is only a candidate if it could
    actually be the target. Without this a JavaScript `helper()` in a repository whose only
    `helper` is a Python function produced a confident cross-language edge — a wrong edge
    wearing the `fallback` label, which is what D19 forbids however it is labelled.
    """
    rows = conn.execute(
        "SELECT s.name, s.qualified_name FROM symbols s JOIN files f ON f.path = s.file "
        "WHERE f.lang = ? GROUP BY s.name "
        "HAVING COUNT(DISTINCT s.qualified_name) = 1",
        (lang,),
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def insert(conn: sqlite3.Connection, edges: tuple[CandidateEdge, ...]) -> None:
    conn.executemany(
        "INSERT INTO edges(src, dst, type, edge_class, file, line, stale) "
        "VALUES(?, ?, ?, ?, ?, ?, 0)",
        [(e.src, e.dst, e.type, e.edge_class, e.file, e.line) for e in edges],
    )


def mark_stale_for_deleted(conn: sqlite3.Connection, deleted: set[str]) -> int:
    """Incoming edges to a deleted symbol become `stale=1` **and** `unknown`.

    Deleting a file's own rows is the easy half. Edges pointing *into* it from files that
    did not change are the hard half, and the three obvious options are all worse:

    - leave them → a dangling edge that still claims `resolved`, the exact lie D19 bans;
    - delete them → the caller silently loses a dependency it really does have;
    - keep the class → a `resolved` edge to a symbol that no longer exists.

    Marking them `unknown` says what is actually true: something is referenced here and the
    graph can no longer tell you what. They re-resolve on the next pass that touches them.
    """
    if not deleted:
        return 0
    # Chunked because SQLite caps bound variables per statement (32766 on current builds,
    # 999 on older ones). A full index of a large repository produces far more symbol names
    # than that, so a single statement would raise on exactly the repositories this is for.
    marked = 0
    for batch in chunks(sorted(deleted), batch_size(conn)):
        marks = ",".join("?" * len(batch))
        cur = conn.execute(
            f"UPDATE edges SET stale = 1, edge_class = ? WHERE dst IN ({marks}) AND stale = 0",
            (UNKNOWN, *batch),
        )
        marked += cur.rowcount
    return marked


def class_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """D19 health, reported on every index so a collapse in `resolved` is visible."""
    rows = conn.execute("SELECT edge_class, COUNT(*) FROM edges GROUP BY edge_class").fetchall()
    return {r[0]: int(r[1]) for r in rows}
