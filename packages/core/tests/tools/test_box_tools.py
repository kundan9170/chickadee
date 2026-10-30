"""The task worker's four new tools, each through its port as the box
gives it: `create`, `delete` and `move` through `WorktreeEdits` over a real git worktree, and
`tests_for` through `GraphNavigator` over a real index."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from repos import init_repo

from chickadee.box.edits import WorktreeEdits
from chickadee.box.syntax import SyntaxChecker
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.profiles.loader import load_profile
from chickadee.secrets.denylist import Denylist
from chickadee.tools.builtin import create, delete, move, tests_for
from chickadee.tools.catalogue import answer_text
from chickadee.tools.context import ToolContext
from chickadee.zeus.tools.graphnav import GraphNavigator, opened

from .tools_helpers import call, dispatcher

CORE = "def helper(x: int) -> int:\n    return x + 1\n"
API = "from app.core import helper\n\n\ndef serve() -> int:\n    return helper(1)\n"
TEST_API = "from app.api import serve\n\n\ndef test_serve() -> None:\n    assert serve() == 2\n"
TEST_OTHER = "def test_nothing() -> None:\n    assert True\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {
            "app/__init__.py": "",
            "app/core.py": CORE,
            "app/api.py": API,
            "tests/test_api.py": TEST_API,
            "tests/test_other.py": TEST_OTHER,
        },
    )


@pytest.fixture
def graph(root: Path) -> Iterator[Graph]:
    Indexer(root).full_index()
    found = opened(root)
    assert found is not None
    yield found
    found.close()


def _ctx(root: Path, navigator: GraphNavigator | None = None) -> ToolContext:
    python = load_profile("python")
    edits = WorktreeEdits(
        root,
        Denylist(),
        SyntaxChecker(root, (python,)),
        untracked_max_bytes=1024,
    )
    return ToolContext(role="worker", edits=edits, navigator=navigator)


def _said(root: Path, tool: object, name: str, **args: str) -> str:
    d = dispatcher([tool], ctx=_ctx(root))  # type: ignore[list-item]
    return answer_text(d.call(call(name, text="", **args)))


def test_create_writes_a_new_file_and_refuses_an_existing_one(root: Path) -> None:
    said = _said(root, create.TOOL, "create", path="app/new.py", content="N = 1\n")
    assert said.startswith("ok: app/new.py written") and "syntax ok: app/new.py" in said
    assert (root / "app" / "new.py").read_text(encoding="utf-8") == "N = 1\n"
    again = _said(root, create.TOOL, "create", path="app/core.py", content="x = 1\n")
    assert "already exists. Do: edit it" in again
    assert (root / "app" / "core.py").read_text(encoding="utf-8") == CORE


def test_delete_removes_a_tracked_file(root: Path) -> None:
    said = _said(root, delete.TOOL, "delete", path="tests/test_other.py")
    assert said == "ok: tests/test_other.py deleted"
    assert not (root / "tests" / "test_other.py").exists()


def test_move_carries_the_file_to_a_new_directory(root: Path) -> None:
    said = _said(root, move.TOOL, "move", path="app/api.py", destination="app/web/api.py")
    assert said.startswith("ok: app/api.py moved to app/web/api.py")
    assert (root / "app" / "web" / "api.py").read_text(encoding="utf-8") == API
    assert not (root / "app" / "api.py").exists()


def test_tests_for_names_the_tests_that_reach_a_change_and_its_limit(
    root: Path, graph: Graph
) -> None:
    navigator = GraphNavigator(graph, [load_profile("python")])
    d = dispatcher([tests_for.TOOL], ctx=_ctx(root, navigator))
    said = answer_text(d.call(call("tests_for", text="", paths=["app/core.py"])))
    assert "tests/test_api.py    reaches app/core.py" in said
    assert "tests/test_other.py" not in said
    assert "run: pytest -q tests/test_api.py" in said
    assert said.endswith("dynamic imports, fixtures, data files and subprocesses are not seen")


def test_tests_for_says_when_nothing_reaches_a_change(root: Path, graph: Graph) -> None:
    navigator = GraphNavigator(graph, [load_profile("python")])
    said = navigator.tests_for(["app/__init__.py"])
    assert isinstance(said, str) and "no test file within 3 import hops" in said


def test_grep_and_glob_hand_the_call_s_stop_to_the_search() -> None:
    """Review: no `Search` was built with a stop, so a hard stop waited out a running `rg`
    for its timeout. Each call hands the search its own."""
    import threading

    from chickadee.tools.builtin import glob, grep
    from chickadee.tools.tool import Invocation
    from chickadee.tools.validate import Arguments

    stop = threading.Event()
    given: list[object] = []

    def search(
        query: str, *, kind: str = "text", paths: object = (), cancel: object = None
    ) -> str:
        given.append(cancel)
        return "ok"

    ctx = ToolContext(role="test", search=search, cancel=stop)  # type: ignore[arg-type]
    grep.TOOL.run(Invocation("c1", "op1", 1), Arguments("grep", {"pattern": "x"}), ctx)
    glob.TOOL.run(Invocation("c2", "op2", 2), Arguments("glob", {"pattern": "*"}), ctx)
    assert given == [stop, stop]
