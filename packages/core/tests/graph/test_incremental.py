"""GR-2: incremental reindex, and the deleted-symbol rule that is the hard half."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from chickadee.errors import InfraError
from chickadee.graph.indexer import Indexer


@pytest.mark.writes_worktree("app.py")
def test_only_changed_files_are_reparsed(repo: Path) -> None:
    ix = Indexer(repo)
    ix.full_index()
    report = ix.reindex()
    assert report.files_indexed == 0, "nothing changed, so nothing should reparse"

    (repo / "app.py").write_text(
        (repo / "app.py").read_text(encoding="utf-8") + "\n\ndef extra():\n    return 1\n",
        encoding="utf-8",
    )
    report = ix.reindex()
    assert report.files_indexed == 1


@pytest.mark.writes_worktree("helpers.py")
def test_incoming_edges_to_deleted_symbols_go_unknown_not_dangle(repo: Path) -> None:
    """The rule: prefer `unknown` over a dangling edge.

    The caller file did not change. Item 6 still reclassifies it on this pass, so the
    edge is a fresh unknown to the bare name — not a leftover `helpers.helper` that
    claims to know where `helper` went.
    """
    ix = Indexer(repo)
    ix.full_index()
    before = ix.conn.execute(
        "SELECT edge_class, stale FROM edges WHERE dst = 'helpers.helper' "
        "AND src = 'app.Widget.render'"
    ).fetchone()
    assert before["edge_class"] == "fallback"
    assert before["stale"] == 0

    (repo / "helpers.py").unlink()
    ix.reindex()

    dangling = ix.conn.execute("SELECT 1 FROM edges WHERE dst = 'helpers.helper'").fetchone()
    assert dangling is None, "must not keep a classified edge to a name that no longer exists"
    after = ix.conn.execute(
        "SELECT edge_class, dst, stale FROM edges WHERE file = 'app.py' "
        "AND type = 'calls' AND src = 'app.Widget.render'"
    ).fetchone()
    assert after is not None, "the caller still calls something"
    assert after["edge_class"] == "unknown"
    assert after["dst"] == "helper"
    assert after["stale"] == 0


@pytest.mark.writes_worktree("helpers.py", "lib.py")
def test_moving_a_unique_name_re_resolves_the_unchanged_caller(repo: Path) -> None:
    """A rename must not leave a permanent gap while the caller file is unchanged."""
    ix = Indexer(repo)
    ix.full_index()
    src = (repo / "helpers.py").read_text(encoding="utf-8")
    (repo / "helpers.py").unlink()
    (repo / "lib.py").write_text(src, encoding="utf-8")
    ix.reindex()

    row = ix.conn.execute(
        "SELECT dst, edge_class, stale FROM edges WHERE file = 'app.py' "
        "AND type = 'calls' AND src = 'app.Widget.render'"
    ).fetchone()
    assert row["dst"] == "lib.helper"
    assert row["edge_class"] == "fallback"
    assert row["stale"] == 0


@pytest.mark.writes_worktree("helpers.py")
def test_editing_a_file_does_not_mark_surviving_symbols_stale(repo: Path) -> None:
    """Reparsing is not deleting.

    `_drop_file` removes a file's rows on every reindex, so treating those names as
    deleted downgraded every incoming edge to unknown+stale on an ordinary edit. The
    original stale test only deleted a file, so it could not fail on this.
    """
    ix = Indexer(repo)
    ix.full_index()
    (repo / "helpers.py").write_text(
        (repo / "helpers.py").read_text(encoding="utf-8") + "\n\ndef extra():\n    return 2\n",
        encoding="utf-8",
    )
    ix.reindex()

    edge = ix.conn.execute(
        "SELECT edge_class, stale FROM edges WHERE dst = 'helpers.helper'"
    ).fetchone()
    assert edge["stale"] == 0, "helper still exists — the edge is not stale"
    assert edge["edge_class"] == "fallback"


def test_a_path_outside_the_repository_is_skipped_not_raised(
    repo: Path, tmp_path: Path
) -> None:
    """GR-4: the index always completes, including when a watcher hands over a stray path."""
    outside = tmp_path / "elsewhere.py"
    outside.write_text("def x():\n    return 1\n", encoding="utf-8")
    ix = Indexer(repo)
    ix.full_index()
    report = ix.reindex([outside])
    assert report.files_indexed == 0


@pytest.mark.writes_worktree("helpers.py")
def test_a_deleted_file_loses_its_own_rows(repo: Path) -> None:
    ix = Indexer(repo)
    ix.full_index()
    (repo / "helpers.py").unlink()
    ix.reindex()
    assert ix.conn.execute("SELECT 1 FROM files WHERE path='helpers.py'").fetchone() is None
    assert ix.conn.execute("SELECT 1 FROM symbols WHERE file='helpers.py'").fetchone() is None


@pytest.mark.writes_worktree("app.py")
def test_reindexing_does_not_duplicate_rows(repo: Path) -> None:
    ix = Indexer(repo)
    first = ix.full_index()
    (repo / "app.py").write_text(
        (repo / "app.py").read_text(encoding="utf-8") + "\n\ndef extra():\n    return 1\n",
        encoding="utf-8",
    )
    ix.reindex()
    second = ix.reindex()
    assert second.files_indexed == 0
    assert (
        ix.conn.execute("SELECT COUNT(*) FROM symbols WHERE name='helper'").fetchone()[0] == 1
    )
    assert ix.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == first.symbols + 1


def test_the_generation_advances_on_every_pass(repo: Path) -> None:
    """PRECOND downstream uses this to notice it read a stale graph."""
    ix = Indexer(repo)
    ix.full_index()
    from chickadee.graph.store import generation

    first = generation(ix.conn)
    ix.reindex()
    assert generation(ix.conn) == first + 1


def test_stale_marking_respects_the_sqlite_bind_variable_limit(tmp_path: Path) -> None:
    """SQLite caps bound variables per statement — 32766 here, but 999 on older builds.

    The limit is lowered on the connection rather than by generating tens of thousands of
    symbols: it makes the portability claim directly, runs in milliseconds, and fails on the
    unchunked implementation, which a fixture sized for *this* machine's limit would not.
    """
    root = tmp_path / "many"
    root.mkdir()
    for n in range(4):
        body = "".join(f"def f_{n}_{k}():\n    return {k}\n\n\n" for k in range(50))
        (root / f"m_{n}.py").write_text(body, encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()
    assert ix.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 200

    ix.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 50)
    for n in range(4):
        (root / f"m_{n}.py").unlink()
    ix.reindex()  # must not raise "too many SQL variables"
    assert ix.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 0


def test_an_unusable_bind_limit_is_named_rather_than_hit(tmp_path: Path) -> None:
    """Clamping the batch to 1 looked safe and was not.

    `mark_stale_for_deleted` binds the edge class too, so at a limit of 1 a single-name
    batch still needs two bindings. SQLite then says "too many SQL variables", which names
    neither the limit nor the caller.
    """
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_text("def gone():\n    return 1\n", encoding="utf-8")
    ix = Indexer(root)
    ix.full_index()

    ix.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 4)
    (root / "a.py").unlink()
    with pytest.raises(InfraError, match="bound variables per statement"):
        ix.reindex()


def test_the_write_guard_is_armed_in_this_suite(repo: Path) -> None:
    """#85's whole point, asserted from outside `tests/applier/`.

    This test carries no `writes_worktree` marker, so the root conftest has the guard armed
    around its body. The write raising is what proves the guard reaches this directory —
    demonstrating it from inside the applier suite would prove only what was already true.
    """
    from watchdog import WorktreeWriteError

    with pytest.raises(WorktreeWriteError, match="without the Applier"):
        (repo / "sneaky.py").write_text("x = 1\n", encoding="utf-8")


@pytest.mark.writes_worktree("declared.py")
def test_a_declaration_permits_only_what_it_names(repo: Path) -> None:
    """The marker declares a set; it does not switch the guard off.

    A marked test doing one extra undeclared write is the case this has to catch, or the
    marker would be an opt-out with a friendly name.
    """
    from watchdog import WorktreeWriteError

    (repo / "declared.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(WorktreeWriteError, match=r"without the Applier"):
        (repo / "undeclared.py").write_text("x = 2\n", encoding="utf-8")
