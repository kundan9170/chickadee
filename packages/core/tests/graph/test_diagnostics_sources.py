"""Two sources for one file: a language server's verdict and a linter's side by side.

What these pin: a verdict replaces only its own server's rows, so a linter's never erases the
language server's; a read shows both, each row with its label; a dependency's change outdates
the language server's verdict and never the linter's, which is the file's own; the file reads
as fresh only when every source is; a task's start is kept only when every source was fresh.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from chickadee.graph import diagnostics, store
from chickadee.graph.diagnostics import DiagCoverage, Diagnostic, Verdict

TEXT = b"import os\nX = undefined\n"
SHA = hashlib.sha256(TEXT).hexdigest()


def _row(message: str, label: str, severity: str = "error") -> Diagnostic:
    return Diagnostic(2, 1, severity, "E1", message, label)  # type: ignore[arg-type]


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_bytes(TEXT)
    c = store.connect(root)
    c.execute(
        "INSERT INTO files(path, sha256, lang, parse_ok, indexed_at) VALUES(?,?,?,?,?)",
        ("a.py", SHA, "python", 1, "now"),
    )
    yield c
    c.close()


def _both(conn: sqlite3.Connection) -> None:
    diagnostics.record(
        conn,
        (
            Verdict("a.py", SHA, "pyright-langserver", (_row("undefined X", "Pyright"),)),
            Verdict("a.py", SHA, "ruff", (_row("unused os", "Ruff", "warning"),), local=True),
        ),
    )


def test_a_linter_s_verdict_never_erases_the_language_server_s(
    conn: sqlite3.Connection,
) -> None:
    _both(conn)
    diagnostics.record(conn, (Verdict("a.py", SHA, "ruff", (), local=True),))
    found = diagnostics.diagnostics_in(conn, "a.py")
    assert found.coverage is DiagCoverage.TOTAL
    assert [(d.message, d.source) for d in found.in_span] == [("undefined X", "Pyright")]


def test_a_read_shows_every_source_s_rows(conn: sqlite3.Connection) -> None:
    _both(conn)
    found = diagnostics.diagnostics_in(conn, "a.py")
    assert sorted(d.source for d in found.in_span) == ["Pyright", "Ruff"]


def test_a_dependency_outdates_the_language_server_and_never_the_linter(
    conn: sqlite3.Connection,
) -> None:
    _both(conn)
    assert diagnostics.mark_stale(conn, {"a.py": "b.py"}, 2) == ["a.py"]
    stale = {
        str(r[0]): r[1]
        for r in conn.execute("SELECT source, stale_by FROM analyzed WHERE path = 'a.py'")
    }
    assert stale == {"pyright-langserver": "b.py", "ruff": None}
    found = diagnostics.diagnostics_in(conn, "a.py")
    assert (found.coverage, found.stale_by) == (DiagCoverage.STALE, "b.py")


def test_the_start_is_kept_only_when_every_source_was_fresh(conn: sqlite3.Connection) -> None:
    _both(conn)
    diagnostics.keep(conn, "T1", {"a.py": SHA})
    diagnostics.mark_stale(conn, {"a.py": "b.py"}, 2)
    diagnostics.keep(conn, "T2", {"a.py": SHA})
    judged = dict(conn.execute("SELECT task, judged FROM baseline_files").fetchall())
    assert judged == {"T1": 1, "T2": 0}
    kept = conn.execute("SELECT COUNT(*) FROM baseline_diags WHERE task = 'T1'").fetchone()
    assert kept[0] == 2
