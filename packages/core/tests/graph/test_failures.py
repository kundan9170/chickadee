"""GR-4 and §6: the index always completes and reports what it missed."""

from __future__ import annotations

from pathlib import Path

from chickadee.graph import store
from chickadee.graph.indexer import Indexer


def test_a_parse_error_keeps_the_file_and_records_a_gap(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "good.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    (root / "bad.py").write_text("def broken(\n", encoding="utf-8")

    report = Indexer(root).full_index()

    assert report.files_indexed == 2, "the index completes despite the bad file"
    assert any(g.path == "bad.py" for g in report.gaps)
    ix = Indexer(root)
    row = ix.conn.execute("SELECT parse_ok FROM files WHERE path='bad.py'").fetchone()
    assert row is not None, "the file stays known even though it did not parse"
    assert row["parse_ok"] == 0


def test_an_unprofiled_language_is_counted_not_crashed(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    (root / "README.md").write_text("# hello\n", encoding="utf-8")
    # Zig has no profile.
    (root / "main.zig").write_text("pub fn main() void {}\n", encoding="utf-8")

    report = Indexer(root).full_index()

    reasons = {g.path: g.reason for g in report.gaps}
    assert reasons["README.md"] == "unprofiled language"
    assert reasons["main.zig"] == "unprofiled language"
    assert report.files_indexed == 1


def test_a_corrupt_database_rebuilds(tmp_path: Path) -> None:
    """The graph is a derived artifact — always safely reconstructible."""
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    Indexer(root).full_index()

    store.db_path(root).write_bytes(b"this is not a database")
    report = Indexer(root).full_index()
    assert report.symbols == 1


def test_a_schema_version_mismatch_rebuilds(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    ix.conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    ix.conn.commit()
    ix.close()

    conn = store.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0, "rebuilt empty"
    assert store._meta_int(store.db_path(root), "schema_version") == store.SCHEMA_VERSION


def test_an_unreadable_file_does_not_stop_the_index(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
    blocked = root / "locked.py"
    blocked.write_text("def x(): pass\n", encoding="utf-8")
    blocked.chmod(0o000)
    try:
        report = Indexer(root).full_index()
        assert report.symbols >= 1
    finally:
        blocked.chmod(0o644)
