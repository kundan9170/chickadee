"""Diagnostics recorded from the index's language servers, and dependents stale when what they
depend on changes.

What a read says of a file: its problems with their lines, `none` only where a server finished
judging these very bytes and nothing they depend on has changed since, `STALE` with the file
that changed, `unavailable` where nothing judged it. Never `none` for a file nothing judged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
from judging import Judging
from test_lsp import ScriptedSupervisor, _resolver
from test_lsp_wired import PYRIGHT, _unsealed

from chickadee.box.spawn import index_resolver
from chickadee.context.diagnostics import render_diagnostics
from chickadee.graph.diagnostics import DiagCoverage, Diagnostic
from chickadee.graph.indexer import Indexer
from chickadee.graph.lsp import frame_message
from chickadee.graph.queries import Graph
from chickadee.profiles import ProfileSet
from chickadee.secrets.denylist import Denylist

UTIL = "def helper(x):\n    return x\n"
APP = "from util import helper\n\n\ndef go():\n    return helper(1)\n"
MAIN = "from app import go\n\n\ndef main():\n    return go()\n"
LONE = "def alone():\n    return 0\n"


def _sha(root: Path, rel: str) -> str:
    return hashlib.sha256((root / rel).read_bytes()).hexdigest()


def _error(line: int, message: str = "Argument missing") -> Diagnostic:
    return Diagnostic(line, 12, "error", "reportCallIssue", message, "fake")


def _repo(tmp_path: Path, files: Mapping[str, str] | None = None) -> Path:
    root = tmp_path / "r"
    root.mkdir()
    for rel, text in (files or {"util.py": UTIL, "app.py": APP, "main.py": MAIN}).items():
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _read(graph: Graph, root: Path, rel: str, shown: range | None = None) -> tuple[str, ...]:
    """The read's diagnostics lines, as `context/views` renders them for the bytes on disk."""
    return render_diagnostics(
        graph.diagnostics_in(rel, shown),
        shown=None,
        current_sha=_sha(root, rel),
        name=Path(rel).suffix,
    )


def _indexed(root: Path, judge: Judging) -> tuple[Indexer, Graph]:
    ix = Indexer(root, judge)
    ix.full_index()
    return ix, Graph(root, ix.conn)


def test_a_type_error_a_clean_file_an_unjudged_one_and_an_edited_one(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"util.py": UTIL, "app.py": APP, "lone.py": LONE})
    judge = Judging(root, {"app.py": (_error(5),), "util.py": ()})
    _, graph = _indexed(root, judge)
    assert _read(graph, root, "app.py") == (
        "[diagnostics: 1 in this file]",
        "  5:12 error reportCallIssue: Argument missing (fake)",
    )
    assert _read(graph, root, "util.py") == ("[diagnostics: none]",)
    assert _read(graph, root, "lone.py") == (
        "[diagnostics: unavailable — no analyzer verdict for .py]",
    )
    judged = _sha(root, "app.py")
    (root / "app.py").write_text(APP + "\n\nX = 1\n", encoding="utf-8")
    [line, *_] = _read(graph, root, "app.py")
    assert line.startswith(f"[diagnostics: STALE — judged at {judged[:6]}, file is now ")


def test_lines_partition_the_problems_rather_than_filter_them(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    _, graph = _indexed(root, Judging(root, {"app.py": (_error(1), _error(5))}))
    found = graph.diagnostics_in("app.py", range(4, 6))
    assert [d.line for d in found.in_span] == [5]
    assert [d.line for d in found.out_of_span] == [1]
    assert found.coverage is DiagCoverage.TOTAL


def test_a_changed_signature_makes_its_dependents_stale_however_far(tmp_path: Path) -> None:
    """`util.py` changes; `app.py` imports it, `main.py` imports `app.py`. Neither was edited,
    and neither reads `none` until a server judges them again."""
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, graph = _indexed(root, judge)
    assert _read(graph, root, "main.py") == ("[diagnostics: none]",)

    judge.problems = {"util.py": ()}  # the server judges only the file that changed
    (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
    ix.reindex([root / "util.py"])
    assert judge.judged[-1] == ["util.py"]
    for rel in ("app.py", "main.py"):
        assert _read(graph, root, rel) == (
            "[diagnostics: STALE — util.py changed since this was judged · none]",
        )
        assert graph.diagnostics_in(rel).coverage is DiagCoverage.STALE
    assert _read(graph, root, "util.py") == ("[diagnostics: none]",), "judged afresh"


def test_a_fresh_verdict_clears_the_mark(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, graph = _indexed(root, judge)
    judge.problems = {"util.py": ()}
    (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
    ix.reindex([root / "util.py"])
    judge.problems = {"app.py": (_error(5, 'Argument missing for parameter "y"'),)}
    ix.reindex([root / "app.py"])
    assert _read(graph, root, "app.py")[0] == "[diagnostics: 1 in this file]"
    assert graph.diagnostics_in("app.py").stale_by is None
    assert graph.diagnostics_in("main.py").stale_by == "util.py", "the first change named"


def test_a_file_judged_in_the_same_pass_as_its_dependency_is_not_marked(
    tmp_path: Path,
) -> None:
    """The server judged it with the changed file already on disk."""
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, graph = _indexed(root, judge)
    (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
    ix.reindex([root / "util.py", root / "app.py"])
    assert graph.diagnostics_in("app.py").coverage is DiagCoverage.TOTAL
    assert graph.diagnostics_in("main.py").stale_by == "util.py"


def test_a_deleted_file_marks_its_dependents_and_its_verdict_goes(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, graph = _indexed(root, judge)
    judge.problems = {}
    (root / "util.py").unlink()
    ix.reindex()
    assert graph.diagnostics_in("app.py").stale_by == "util.py"
    assert graph.diagnostics_in("util.py").coverage is DiagCoverage.UNAVAILABLE


def test_a_file_judged_in_the_pass_is_not_rechecked_after_it(tmp_path: Path) -> None:
    """Review of #612: `app.py` was marked and then judged in the same pass, so only `main.py`
    is left to ask again."""
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, _ = _indexed(root, judge)
    (root / "util.py").write_text("def helper(x, y):\n    return x\n", encoding="utf-8")
    ix.reindex([root / "util.py", root / "app.py"])
    assert judge.rechecked[-1] == ["main.py"]


def test_a_file_newly_denied_marks_what_depended_on_it(tmp_path: Path) -> None:
    """Review of #612: denying an indexed file takes its rows away, as deleting it does, so its
    dependents are marked first, while the rows still say who they are."""
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, graph = _indexed(root, judge)
    judge.problems = {}
    ix.denylist = Denylist(patterns=(*Denylist().patterns, "util.py"))
    ix.reindex([root / "util.py"])
    assert graph.diagnostics_in("app.py").stale_by == "util.py"
    assert graph.diagnostics_in("util.py").coverage is DiagCoverage.UNAVAILABLE


def test_an_edit_that_leaves_the_bytes_as_they_were_marks_nothing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (), "main.py": ()})
    ix, graph = _indexed(root, judge)
    judge.problems = {}
    ix.reindex([root / "util.py"])
    assert graph.diagnostics_in("app.py").coverage is DiagCoverage.TOTAL


def test_a_full_index_judges_afresh(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    judge = Judging(root, {"util.py": (), "app.py": (_error(5),), "main.py": ()})
    ix, graph = _indexed(root, judge)
    judge.problems = {}
    ix.full_index()
    assert graph.diagnostics_in("app.py").coverage is DiagCoverage.UNAVAILABLE


# -- from a session ----------------------------------------------------------------------------


def _frame(fid: int, result: object = None, *, error: bool = False) -> bytes:
    if error:
        return frame_message(
            {"jsonrpc": "2.0", "id": fid, "error": {"code": -32601, "message": "no"}}
        )
    return frame_message({"jsonrpc": "2.0", "id": fid, "result": result})


def _pushed(root: Path, rel: str, items: list[dict[str, object]]) -> bytes:
    return frame_message(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {"uri": root.resolve().joinpath(rel).as_uri(), "diagnostics": items},
        }
    )


def _problem(line_0: int, severity: int | None, message: str) -> dict[str, object]:
    found: dict[str, object] = {
        "range": {
            "start": {"line": line_0, "character": 4},
            "end": {"line": line_0, "character": 5},
        },
        "message": message,
        "code": "reportCallIssue",
        "source": "Pyright",
    }
    if severity is not None:
        found["severity"] = severity
    return found


INIT = _frame(0, {"capabilities": {}})
"""The ids a pass asks, for `app.py` (one question: its call of `helper`), `lone.py` and
`util.py`: the definition 1, then each file's verdict, the asked-about file first, 2 to 4."""


def _session(tmp_path: Path, *replies: bytes) -> tuple[Graph, ScriptedSupervisor]:
    root = _repo(tmp_path, {"util.py": UTIL, "app.py": APP, "lone.py": LONE})
    supervisor = ScriptedSupervisor(b"".join((INIT, *replies)))
    ix = Indexer(root, _resolver(root, supervisor))
    ix.full_index()
    return Graph(root, ix.conn), supervisor


def test_a_full_report_is_a_verdict_and_anything_else_is_none(tmp_path: Path) -> None:
    items = [
        _problem(4, 1, "Argument missing"),
        _problem(0, 3, "information"),
        _problem(1, None, "x"),
    ]
    graph, supervisor = _session(
        tmp_path,
        _frame(1),
        _frame(2, {"kind": "full", "items": items}),
        _frame(3, error=True),
        _frame(4, {"kind": "full", "items": []}),
    )
    root = tmp_path / "r"
    app = graph.diagnostics_in("app.py")
    assert app.coverage is DiagCoverage.TOTAL and app.analyzed_sha == _sha(root, "app.py")
    assert [(d.line, d.col, d.severity, d.message, d.source) for d in app.in_span] == [
        (1, 5, "hint", "information", "Pyright"),
        (2, 5, "error", "x", "Pyright"),
        (5, 5, "error", "Argument missing", "Pyright"),
    ], "1-based; information is a hint; a severity left out is an error"
    assert graph.diagnostics_in("util.py").coverage is DiagCoverage.TOTAL
    assert graph.diagnostics_in("lone.py").coverage is DiagCoverage.UNAVAILABLE, "it erred"
    asked = [
        f["params"]["textDocument"]["uri"].rsplit("/", 1)[-1]  # type: ignore[index]
        for f in supervisor.sent()
        if f.get("method") == "textDocument/diagnostic"
    ]
    assert asked == ["app.py", "lone.py", "util.py"], (
        "every file of the pass, asked about or not"
    )


def test_a_pushed_empty_list_is_never_a_verdict(tmp_path: Path) -> None:
    """The false clean of D205 Amendment 1: a server pushes an empty list for a file it has not
    analysed yet. Only an answer to the question is a verdict."""
    graph, _ = _session(
        tmp_path,
        _pushed(tmp_path / "r", "app.py", []),
        _pushed(tmp_path / "r", "util.py", []),
        _frame(1),
        _frame(2, error=True),
    )
    for rel in ("app.py", "lone.py", "util.py"):
        assert graph.diagnostics_in(rel).coverage is DiagCoverage.UNAVAILABLE


def test_a_session_cut_short_keeps_the_verdicts_that_landed(tmp_path: Path) -> None:
    """Each answer is final for its file, so one that landed stands; the rest are none."""
    graph, _ = _session(tmp_path, _frame(1), _frame(2, {"kind": "full", "items": []}))
    assert graph.diagnostics_in("app.py").coverage is DiagCoverage.TOTAL
    for rel in ("lone.py", "util.py"):
        assert graph.diagnostics_in(rel).coverage is DiagCoverage.UNAVAILABLE


@pytest.mark.skipif(not PYRIGHT, reason="pyright cannot run here")
def test_pyrights_verdicts_are_recorded(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"util.py": UTIL, "app.py": APP.replace("helper(1)", "helper()")})
    profiles = ProfileSet.load(repo=root)
    resolver = index_resolver(root, _unsealed(require_sandbox=False), profiles)
    ix = Indexer(root, resolver, profiles=profiles)
    ix.full_index()
    graph = Graph(root, ix.conn)
    app, util = graph.diagnostics_in("app.py"), graph.diagnostics_in("util.py")
    assert app.coverage is DiagCoverage.TOTAL, (app, util)
    assert [(d.line, d.severity) for d in app.in_span] == [(5, "error")], app
    assert util.coverage is DiagCoverage.TOTAL, util
