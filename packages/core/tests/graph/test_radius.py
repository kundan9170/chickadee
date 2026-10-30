"""Blast radius includes unknown edges; depth cap; the GR-1 timing probe."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.graph.indexer import Indexer
from chickadee.graph.radius import UPGRADE_TRIGGER_MS, blast_radius


def test_radius_reaches_callers(indexed) -> None:
    files = indexed.blast_radius(["helpers.helper"], depth=2)
    assert "app.py" in files


def test_radius_is_depth_capped(tmp_path: Path) -> None:
    root = tmp_path / "chain"
    root.mkdir()
    (root / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def b():\n    return a()\n", encoding="utf-8")
    (root / "c.py").write_text("def c():\n    return b()\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    from chickadee.graph.queries import Graph

    g = Graph(root, ix.conn)
    assert "b.py" in g.blast_radius(["a.a"], depth=1)
    assert "c.py" not in g.blast_radius(["a.a"], depth=1)
    assert "c.py" in g.blast_radius(["a.a"], depth=2)


def test_unknown_edges_are_included(tmp_path: Path) -> None:
    """Over-inclusion is the safe direction: an unknown edge is the one you least want to
    drop from the verifier's affected set."""
    root = tmp_path / "u"
    root.mkdir()
    (root / "caller.py").write_text("def go():\n    return mystery()\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    classes = {r["edge_class"] for r in ix.conn.execute("SELECT edge_class FROM edges")}
    assert classes == {"unknown"}, classes
    result = blast_radius(ix.conn, ["mystery"], depth=2)
    assert "caller.py" in result.files


@pytest.mark.writes_worktree("helpers.py")
def test_unknown_edges_after_a_delete_still_pull_the_caller(repo: Path) -> None:
    """The caller file did not change; the next pass still reclassifies it. The new unknown
    edge (bare name) must remain in the radius — that is when the caller needs review.
    """
    ix = Indexer(repo)
    ix.full_index()
    (repo / "helpers.py").unlink()
    ix.reindex()
    result = blast_radius(ix.conn, ["helper"], depth=2)
    assert "app.py" in result.files


def test_the_timing_probe_reports_rather_than_assumes(indexed) -> None:
    """GR-1's upgrade trigger fires on a measurement, not on someone remembering to time."""
    result = blast_radius(indexed.conn, ["helpers.helper"], depth=3)
    assert result.wall_ms >= 0
    assert result.over_trigger == (result.wall_ms > UPGRADE_TRIGGER_MS)


def test_each_reached_symbol_carries_its_best_paths_class(tmp_path: Path) -> None:
    """A path is as confident as its weakest step, and a symbol keeps its best path's
    class, so `impact` says *possibly affected* only where no better path reaches it."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE edges (src TEXT, dst TEXT, type TEXT, edge_class TEXT, file TEXT, "
        "line INTEGER, stale INTEGER DEFAULT 0)"
    )
    conn.execute("CREATE TABLE symbols (qualified_name TEXT, file TEXT)")
    edges = [
        ("b", "a", "fallback"),  # a <- b fallback
        ("c", "b", "unknown"),  # b <- c unknown, so c is unknown
        ("d", "a", "resolved"),  # a <- d resolved
        ("c", "d", "resolved"),  # d <- c resolved: c's best path is resolved too
        ("e", "b", "unknown"),  # b <- e unknown only
    ]
    conn.executemany(
        "INSERT INTO edges(src, dst, type, edge_class, file, line) "
        "VALUES (?, ?, 'calls', ?, 'x', 1)",
        edges,
    )
    reached = {sym: cls for sym, _, cls in blast_radius(conn, ["a"], depth=2).reached}
    assert reached == {
        "a": "resolved",
        "b": "fallback",
        "c": "resolved",
        "d": "resolved",
        "e": "unknown",
    }
