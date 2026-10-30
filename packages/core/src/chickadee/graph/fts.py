"""Subword expansion and FTS5 rows.

The expansion is computed **at insert** into an ordinary `tokens` column, so search works on
any stock SQLite build. A custom FTS5 tokenizer would be tidier and would also make the
index unreadable anywhere the extension is not loaded — including a user's own `sqlite3`
session, which is where someone debugging a bad search result will actually look.
"""

from __future__ import annotations

import re
import sqlite3

_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+")


def subwords(name: str) -> tuple[str, ...]:
    """`getUserName` → `get user name`; `refresh_session` → `refresh session`.

    Both conventions expand to the same shape, which is the point: a repository that mixes
    them (every JavaScript+Python repository) is otherwise searchable only if you already
    know which convention the author used for the symbol you are looking for.
    """
    parts: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z]+", name):
        if not chunk:
            continue
        parts.extend(m.group(0).lower() for m in _CAMEL.finditer(chunk))
    seen: dict[str, None] = {}
    for part in parts:
        seen.setdefault(part, None)
    return tuple(seen)


def tokens_for(qualified_name: str) -> str:
    """The `tokens` column: the name itself, then its subwords, so both an exact query and
    a fragment hit the same row."""
    tail = qualified_name.rsplit(".", 1)[-1]
    words = subwords(qualified_name)
    return " ".join((qualified_name, tail, *words))


def index_symbol(conn: sqlite3.Connection, rowid: int, qualified_name: str) -> None:
    """Index one symbol, keyed by its `symbols.id`.

    The table is **contentless** (`content=''`): FTS5 indexes the columns without
    storing them, so selecting `qualified_name` back out returns NULL. Every read therefore
    goes through `rowid`, and that rowid has to be the symbol's own id or there is nothing
    to join back to.
    """
    conn.execute(
        "INSERT INTO symbols_fts(rowid, qualified_name, tokens) VALUES(?, ?, ?)",
        (rowid, qualified_name, tokens_for(qualified_name)),
    )


def delete_symbol(conn: sqlite3.Connection, rowid: int, qualified_name: str) -> None:
    """Remove one symbol's FTS row.

    A contentless table cannot answer `DELETE... WHERE`, because it does not hold the
    values to match on. FTS5's `'delete'` command takes them from the caller instead — and
    they must be **byte-identical to what was inserted** or the index silently keeps a
    phantom entry. That is safe here only because `tokens_for` is pure: the same name always
    reconstructs the same tokens.

    `contentless_delete=1` would avoid the dance, but it needs SQLite 3.43+, and GR-3 chose
    this design specifically so the index stays readable on a stock build.
    """
    conn.execute(
        "INSERT INTO symbols_fts(symbols_fts, rowid, qualified_name, tokens) "
        "VALUES('delete', ?, ?, ?)",
        (rowid, qualified_name, tokens_for(qualified_name)),
    )


def drop_file(conn: sqlite3.Connection, path: str) -> None:
    """Drop every FTS row for a file, before its symbols are deleted.

    Order matters: the symbol rows are the only source of the ids and names this needs, so
    calling it after the cascade leaves the FTS index holding rows for symbols that no
    longer exist.
    """
    rows = conn.execute(
        "SELECT id, qualified_name FROM symbols WHERE file = ?", (path,)
    ).fetchall()
    for row in rows:
        delete_symbol(conn, int(row[0]), str(row[1]))


def clear(conn: sqlite3.Connection) -> None:
    """Empty the index. `DELETE FROM symbols_fts` raises on a contentless table — FTS5
    exposes `'delete-all'` for exactly this, and a full reindex is the one caller."""
    conn.execute("INSERT INTO symbols_fts(symbols_fts) VALUES('delete-all')")


def search(
    conn: sqlite3.Connection, query: str, limit: int | None = None
) -> tuple[tuple[int, float], ...]:
    """`(id, bm25)` pairs. Lower BM25 is better; equal scores are a recency tie for
    `queries.search`, which is the layer that knows about git.

    Returning the score (not the FTS row order) is what makes two equally relevant hits
    share a primary key. An enumerate of this result would make every rank unique and
    recency would never run.

    `limit` is a safety cap on the MATCH set, not the result ranking. `Graph.search`
    applies the caller's limit after recency reorders equal scores, so a SQL LIMIT of
    that size would drop the rest of a BM25 tie group before recency could see it.
    """
    terms = " OR ".join(f'"{w}"' for w in subwords(query)) or f'"{query}"'
    sql = (
        "SELECT rowid, bm25(symbols_fts) FROM symbols_fts WHERE symbols_fts MATCH ? "
        "ORDER BY bm25(symbols_fts)"
    )
    params: tuple[object, ...] = (terms,)
    if limit is not None:
        sql += " LIMIT ?"
        params = (terms, limit)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # A malformed FTS expression is a bad query, not a broken index. Returning nothing
        # beats propagating a SQL error to a model-facing search.
        return ()
    return tuple((int(r[0]), float(r[1])) for r in rows)
