"""Gitignore-aware walk: generated trees stay out; tracked source stays in."""

from __future__ import annotations

from pathlib import Path

from chickadee.graph.ignore import Ignore
from chickadee.graph.indexer import Indexer


def test_gitignore_skips_a_generated_tree(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / ".gitignore").write_text("generated/\n*.tmp\n", encoding="utf-8")
    (root / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (root / "generated").mkdir()
    (root / "generated" / "x.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    (root / "noise.tmp").write_text("nope\n", encoding="utf-8")

    ix = Indexer(root)
    ix.full_index()
    paths = {r[0] for r in ix.conn.execute("SELECT path FROM files")}
    assert "keep.py" in paths
    assert "generated/x.py" not in paths
    assert "noise.tmp" not in paths


def test_negation_unignores_a_file(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / ".gitignore").write_text("*.py\n!keep.py\n", encoding="utf-8")
    (root / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (root / "drop.py").write_text("def drop():\n    return 1\n", encoding="utf-8")
    rules = Ignore(root)
    assert not rules.skipped(root / "keep.py")
    assert rules.skipped(root / "drop.py")


def test_dir_only_rule_does_not_ignore_a_file_of_the_same_name(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (root / "generated").write_text("def keep():\n    return 1\n", encoding="utf-8")
    rules = Ignore(root)
    assert not rules.skipped(root / "generated")


def test_chickadee_dir_is_skipped_even_without_gitignore(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    nested = root / ".chickadee" / "worktree"
    nested.mkdir(parents=True)
    (nested / "dup.py").write_text("def dup():\n    return 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    paths = {r[0] for r in ix.conn.execute("SELECT path FROM files")}
    assert "keep.py" in paths
    assert not any("chickadee" in p for p in paths)
