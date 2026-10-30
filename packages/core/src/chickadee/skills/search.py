"""`find_skill`'s ranking: BM25 over every indexed skill.

A skill is found by what it says it is for: its name, its description and its
`chickadee-keywords`. They are ranked by SQLite FTS5's BM25, the same ranking and the same
subword split (`graph/fts.subwords`) the symbol search uses, so `pdf-forms`, `pdfForms` and
`pdf forms` are one query. No embeddings and no model: the same catalog and query give the same
answer every time, and ties are broken by name.

**Nothing outlives a call.** A session's catalog is tens of skills, so each query builds an
in-memory table, ranks and drops it. There is no connection to share between threads or to
keep in step with the catalog. Only indexed skills are ranked: a shadowed one is not loadable,
so it is never offered.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from chickadee.graph.fts import subwords
from chickadee.skills.discover import Skill, SkillCatalog

__all__ = ["SkillHit", "rank"]


@dataclass(frozen=True)
class SkillHit:
    skill: Skill
    score: float
    """BM25; lower is better, as FTS5 reports it."""


def rank(catalog: SkillCatalog, query: str, limit: int) -> tuple[SkillHit, ...]:
    """The `limit` skills that best match `query`, best first, ties by name. Nothing matches
    an empty query, or one with no word in it."""
    words = subwords(query)
    if not words or limit <= 0 or not catalog.skills:
        return ()
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE skills USING fts5(name, description, keywords)")
        conn.executemany(
            "INSERT INTO skills(rowid, name, description, keywords) VALUES (?, ?, ?, ?)",
            [
                (
                    n,
                    " ".join(subwords(s.name)),
                    s.front.description,
                    " ".join(w for k in s.front.keywords for w in subwords(k)),
                )
                for n, s in enumerate(catalog.skills)
            ],
        )
        terms = " OR ".join(f'"{w}"' for w in words)
        rows = conn.execute(
            "SELECT rowid, bm25(skills) FROM skills WHERE skills MATCH ?", (terms,)
        ).fetchall()
    finally:
        conn.close()
    hits = [SkillHit(catalog.skills[int(r[0])], float(r[1])) for r in rows]
    hits.sort(key=lambda h: (h.score, h.skill.name))
    return tuple(hits[:limit])
