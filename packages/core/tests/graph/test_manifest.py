"""PL-10: every dependency and global is a resolved contract or an explicit gap."""

from __future__ import annotations

from pathlib import Path

from chickadee.graph.indexer import Indexer
from chickadee.graph.manifest import build
from chickadee.graph.queries import Graph


def test_the_manifest_collects_imports_globals_and_callers(indexed) -> None:
    inputs = build(indexed, ["app.py"])
    assert inputs.target_files == ("app.py",)
    assert any(i.module == "helpers.helper" for i in inputs.imports)
    assert {g.name for g in inputs.globals} == {"MAX_SIZE", "counter"}
    assert any(c.name == "build_widget" for c in inputs.public_contracts)


def test_an_unindexed_target_is_a_gap_not_an_omission(indexed) -> None:
    inputs = build(indexed, ["app.py", "ghost.py"])
    assert "ghost.py" not in inputs.target_files
    assert any(g.name == "ghost.py" and g.kind == "file" for g in inputs.graph_gaps)
    assert not inputs.complete


def test_an_unresolvable_import_is_an_explicit_gap(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text(
        "from nowhere import thing\n\n\ndef go():\n    return 1\n", encoding="utf-8"
    )
    ix = Indexer(root)
    ix.full_index()
    inputs = build(Graph(root, ix.conn), ["a.py"])
    assert any(g.kind == "import" for g in inputs.graph_gaps)
    assert not inputs.complete


def test_an_unresolved_reference_is_a_gap(tmp_path: Path) -> None:
    """PL-10: a name the target calls but the graph cannot resolve is a dependency, so it
    must appear as an explicit gap rather than simply be absent."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "b.py").write_text("def go():\n    return mystery()\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    inputs = build(Graph(root, ix.conn), ["b.py"])
    assert any(g.kind == "symbol" and g.name == "mystery" for g in inputs.graph_gaps)
    assert not inputs.complete


def test_an_unknown_incoming_caller_is_a_gap(tmp_path: Path) -> None:
    """Reachable through GR-2: deleting a symbol downgrades its incoming edges to unknown,
    so the graph knows something calls this and cannot say what.

    Stale marking is visible *before* the next pass reclassifies the caller. The indexer
    reclassifies in the same `reindex` as the delete (item 6), so this test marks stale
    without reparsing the caller — that is the gap the manifest must still name.
    """
    from chickadee.graph.edges import mark_stale_for_deleted

    root = tmp_path / "r"
    root.mkdir()
    (root / "h.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "a.py").write_text("def go():\n    return helper()\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    mark_stale_for_deleted(ix.conn, {"h.helper"})
    ix.conn.commit()

    inputs = build(Graph(root, ix.conn), ["h.py"])
    assert any(g.kind == "caller" for g in inputs.graph_gaps), inputs.graph_gaps
    assert not inputs.complete


def test_restoring_a_unique_callee_re_resolves_the_unchanged_caller(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "h.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "a.py").write_text("def go():\n    return helper()\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    (root / "h.py").unlink()
    ix.reindex()
    (root / "h.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    ix.reindex()

    inputs = build(Graph(root, ix.conn), ["h.py"])
    assert not any(g.kind == "caller" for g in inputs.graph_gaps), inputs.graph_gaps
    assert any(c.src == "a.go" and c.edge_class != "unknown" for c in inputs.callers)


def test_complete_is_named_rather_than_recomputed(indexed) -> None:
    inputs = build(indexed, ["helpers.py"])
    assert inputs.complete == (not inputs.graph_gaps)
