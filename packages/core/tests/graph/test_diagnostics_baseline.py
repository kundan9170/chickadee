"""A task's start, kept, and its files compared with it at the stop.

What these pin: the first time a task's pass changes a file or outdates a dependent, its
verdict is copied aside before the pass or its re-check replaces it; a file with no fresh
verdict then is "no record", never clean; a file kept once keeps its first; at the stop a
problem that moved a line is not new, an error there at the start is told apart, and a file
with no fresh verdict now is unchecked; with no task kept nothing is copied. A changed file
never judged has its bytes at the task's start tree asked of its server first; one the task
created starts empty; a start tree git cannot read leaves "no record".
"""

from __future__ import annotations

from pathlib import Path

from judging import Judging
from repos import git, init_repo
from test_diagnostics_recorded import APP, MAIN, UTIL, _error, _repo
from test_servers import _episode, fake  # noqa: F401 # pyright: ignore[reportUnusedImport]
from test_servers import _repo as _server_repo

from chickadee.graph.diagnostics import DiagCoverage, Diagnostic, since_start
from chickadee.graph.indexer import Indexer
from chickadee.graph.lsp import CheckerResolver


def _warning(line: int, message: str = "unused") -> Diagnostic:
    return Diagnostic(line, 1, "warning", "reportUnused", message, "fake")


def _kept(ix: Indexer, task: str) -> dict[str, bool]:
    return {
        str(r[0]): bool(r[1])
        for r in ix.conn.execute(
            "SELECT path, judged FROM baseline_files WHERE task = ?", (task,)
        )
    }


def _start(tmp_path: Path) -> tuple[Path, Judging, Indexer]:
    """util.py clean, app.py (which imports it) with one error, main.py never judged."""
    root = _repo(tmp_path, {"util.py": UTIL, "app.py": APP, "main.py": MAIN})
    judge = Judging(root, {"util.py": (), "app.py": (_error(5, "old problem"),)})
    ix = Indexer(root, judge)
    ix.full_index()
    ix.keeping = "T1"
    return root, judge, ix


def test_the_start_is_kept_before_a_pass_replaces_it(tmp_path: Path) -> None:
    root, judge, ix = _start(tmp_path)
    (root / "util.py").write_text(UTIL + "\nY = 2\n", encoding="utf-8")
    judge.problems["util.py"] = (_error(3, "new problem"),)
    ix.reindex([root / "util.py"])

    assert _kept(ix, "T1") == {"util.py": True, "app.py": True, "main.py": False}
    found = since_start(ix.conn, "T1")
    assert [str(f) for f in found.new_errors] == ["util.py:3 fake reportCallIssue new problem"]
    # app.py is stale until it is judged again; main.py was never judged.
    assert found.unchecked == ("app.py", "main.py")


def test_a_moved_problem_is_not_new_and_one_already_there_is_told_apart(
    tmp_path: Path,
) -> None:
    root, judge, ix = _start(tmp_path)
    (root / "app.py").write_text("\n" + APP, encoding="utf-8")
    judge.problems["app.py"] = (
        _error(6, "old problem"),
        _error(2, "another"),
        _warning(1),
    )
    ix.reindex([root / "app.py"])

    found = since_start(ix.conn, "T1")
    assert [str(f) for f in found.already] == ["app.py:6 fake reportCallIssue old problem"]
    assert [str(f) for f in found.new_errors] == ["app.py:2 fake reportCallIssue another"]
    assert [str(f) for f in found.new_warnings] == ["app.py:1 fake reportUnused unused"]


def test_a_file_kept_once_keeps_its_first(tmp_path: Path) -> None:
    root, judge, ix = _start(tmp_path)
    for n, problem in enumerate(("first edit", "second edit"), start=1):
        (root / "app.py").write_text(APP + f"\nZ = {n}\n", encoding="utf-8")
        judge.problems["app.py"] = (_error(5, "old problem"), _error(1, problem))
        ix.reindex([root / "app.py"])
    found = since_start(ix.conn, "T1")
    assert [f.message for f in found.new_errors] == ["second edit"]
    assert [f.message for f in found.already] == ["old problem"]


def test_a_file_with_no_verdict_at_the_start_has_no_record(tmp_path: Path) -> None:
    root, judge, ix = _start(tmp_path)
    (root / "main.py").write_text(MAIN + "\nW = 1\n", encoding="utf-8")
    judge.problems["main.py"] = (_error(1, "anything"),)
    ix.reindex([root / "main.py"])
    found = since_start(ix.conn, "T1")
    assert (found.no_record, found.new_errors) == (("main.py",), ())


def test_with_no_task_kept_nothing_is_copied(tmp_path: Path) -> None:
    root, _, ix = _start(tmp_path)
    ix.keeping = None
    (root / "util.py").write_text(UTIL + "\nY = 2\n", encoding="utf-8")
    ix.reindex([root / "util.py"])
    assert ix.conn.execute("SELECT COUNT(*) FROM baseline_files").fetchone()[0] == 0


def _git_start(tmp_path: Path) -> tuple[Path, Judging, Indexer]:
    """A repository whose app.py nothing judged at the full index; its start tree kept."""
    root = init_repo(tmp_path / "g", {"util.py": UTIL, "app.py": APP})
    judge = Judging(root, {"util.py": ()})
    ix = Indexer(root, judge)
    ix.full_index()
    ix.keeping, ix.keeping_from = "T1", git(root, "rev-parse", "HEAD^{tree}")
    return root, judge, ix


def test_a_changed_file_never_judged_has_its_start_asked(tmp_path: Path) -> None:
    root, judge, ix = _git_start(tmp_path)
    judge.started["app.py"] = (_error(5, "old problem"),)
    judge.problems["app.py"] = (_error(6, "old problem"), _error(1, "new problem"))
    (root / "app.py").write_text("\n" + APP, encoding="utf-8")
    ix.reindex([root / "app.py"])

    assert judge.asked_start == [["app.py"]]
    found = since_start(ix.conn, "T1")
    assert [f.message for f in found.already] == ["old problem"]
    assert [f.message for f in found.new_errors] == ["new problem"]
    assert found.no_record == ()


def test_a_file_the_task_created_starts_empty(tmp_path: Path) -> None:
    root, judge, ix = _git_start(tmp_path)
    (root / "new.py").write_text("Q = 1\n", encoding="utf-8")
    judge.problems["new.py"] = (_error(1, "brand new"),)
    ix.reindex([root / "new.py"])

    assert judge.asked_start == []
    assert [f.message for f in since_start(ix.conn, "T1").new_errors] == ["brand new"]


def test_a_start_tree_git_cannot_read_leaves_no_record(tmp_path: Path) -> None:
    root, judge, ix = _git_start(tmp_path)
    ix.keeping_from = "0" * 40
    judge.problems["app.py"] = (_error(1, "anything"),)
    (root / "app.py").write_text("\n" + APP, encoding="utf-8")
    ix.reindex([root / "app.py"])

    assert judge.asked_start == []
    found = since_start(ix.conn, "T1")
    assert (found.no_record, found.new_errors) == (("app.py",), ())


def test_a_server_judges_the_bytes_given_and_the_next_pass_the_disks(
    tmp_path: Path,
    fake: None,  # noqa: F811
) -> None:
    """Through a live server: the start's bytes are judged as given, and the next pass's
    refresh brings the file back to what is on disk."""
    root = _server_repo(tmp_path)
    ix, graph, servers = _episode(root)
    try:
        assert isinstance(ix.resolver, CheckerResolver)
        [then] = ix.resolver.judge({"util.py": b"BAD = 1\n"})
        assert [d.message for d in then.diagnostics] == ["bad"]
        ix.reindex([root / "util.py"])
        now = graph.diagnostics_in("util.py")
        assert now.coverage is DiagCoverage.TOTAL and now.in_span == ()
    finally:
        servers.close()
