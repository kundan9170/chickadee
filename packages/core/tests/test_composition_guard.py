"""The composition guard fails a second home for a session's lifecycle.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI.
Loaded by file path, as the other guard tests load the runner.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDS = REPO_ROOT / "scripts" / "guards"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_composition_guard_runner", GUARDS / "run.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
composition: Any = importlib.import_module("guards.composition")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


_HANDLE = (
    "from chickadee.journal.reader import read_run\n"
    "class JournalRecords:\n"
    "    def journal_text(self, run_id):\n"
    "        return tuple(read_run(self._roots, run_id))\n"
)
"""The one exempt read: a handle shows a journal's lines as written."""

_GROUP = (
    "from concurrent.futures import ThreadPoolExecutor\n"
    "class Dispatcher:\n"
    "    def execute_group(self, calls):\n"
    "        with ThreadPoolExecutor(len(calls)) as pool:\n"
    "            yield from pool.map(self.run, calls)\n"
)
"""The one exempt pool: one group's threaded calls, joined before the call returns."""


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(composition, "REPO_ROOT", root)
    # The exempt read and pool, so a tree that is about something else holds no stale
    # exemption.
    _write(root, "context/project.py", _HANDLE)
    _write(root, "tools/dispatch.py", _GROUP)
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0])
        for v in composition.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(composition.GUARD, find=composition.find, baseline={})


@pytest.mark.parametrize(
    "source",
    [
        "from concurrent.futures import ThreadPoolExecutor\nw = ThreadPoolExecutor(1)\n",
        "import concurrent.futures\nw = concurrent.futures.ThreadPoolExecutor(max_workers=1)\n",
        "from concurrent import futures\nw = futures.ThreadPoolExecutor()\n",
        "from concurrent.futures import ThreadPoolExecutor as Pool\nw = Pool(1)\n",
    ],
    ids=["by-name", "by-module", "by-submodule", "renamed"],
)
def test_a_worker_made_outside_the_conversation_fails(tree: Path, source: str) -> None:
    _write(tree, "engine.py", source)
    assert _found(tree) == [("engine.py", 2, "C1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}engine.py", 1, 0),)


def test_the_conversations_own_worker_passes(tree: Path) -> None:
    _write(
        tree,
        "zeus/conversation.py",
        "from concurrent.futures import ThreadPoolExecutor\nw = ThreadPoolExecutor(1)\n",
    )
    assert _found(tree) == []
    assert not runner.evaluate(_guard(), tree).failed


def test_naming_the_class_without_making_one_passes(tree: Path) -> None:
    _write(
        tree,
        "engine.py",
        "from concurrent.futures import ThreadPoolExecutor\n"
        "def f(w: ThreadPoolExecutor) -> None:\n    w.shutdown()\n",
    )
    assert _found(tree) == []


def test_the_live_tree_holds_only_its_baseline() -> None:
    """The guard against the repository as it stands: no executor outside the conversation,
    no read cache outside the engine but the unwired `Sessions`', and no
    journal read outside `journal/runs.py` but a handle's."""
    files = sorted((REPO_ROOT / SRC).rglob("*.py"))
    assert [(v.path, v.text.split(":", 1)[0]) for v in composition.find(files)] == [
        (f"{CORE}sessions.py", "C2")
    ]


# -- C2: one cache of each per engine ---------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "from chickadee.context.replay import ReplayCache\nc = ReplayCache(root)\n",
        "from chickadee.journal.runs import RunProjections as Runs\nr = Runs(roots)\n",
        "import chickadee.journal.runs as runs\nr = runs.RunProjections(roots)\n",
    ],
    ids=["session-folds", "renamed", "by-module"],
)
def test_a_read_cache_built_outside_the_engine_fails(tree: Path, source: str) -> None:
    _write(tree, "run_header.py", source)
    assert _found(tree) == [("run_header.py", 2, "C2")]


def test_the_engine_builds_its_pair(tree: Path) -> None:
    _write(
        tree,
        "engine.py",
        "from chickadee.context.replay import ReplayCache\n"
        "from chickadee.journal.runs import RunProjections\n"
        "pair = (ReplayCache(root), RunProjections(roots))\n",
    )
    assert _found(tree) == []


# -- C3: a journal folded once ----------------------------------------------------


def test_a_journal_read_outside_runs_fails(tree: Path) -> None:
    _write(
        tree,
        "budget/family.py",
        "from chickadee.journal.reader import read_run, read_tail\n"
        "def spend(roots, run_id, path):\n"
        "    read_run(roots, run_id)\n"
        "    return read_tail(path)\n",
    )
    assert _found(tree) == [("budget/family.py", 3, "C3"), ("budget/family.py", 4, "C3")]


def test_the_runs_module_and_the_reader_read_journals(tree: Path) -> None:
    _write(tree, "context/project.py", _HANDLE)
    _write(
        tree,
        "journal/runs.py",
        "from x import read_tail, read_run\nread_tail(p)\nread_run(r, i)\n",
    )
    _write(tree, "journal/reader.py", "def read_events(p):\n    return read_tail(p)\n")
    assert _found(tree) == []


def test_a_handle_read_elsewhere_in_its_module_is_not_exempt(tree: Path) -> None:
    _write(
        tree,
        "context/project.py",
        _HANDLE
        + "    def projection(self, run_id):\n        return read_run(self._roots, run_id)\n",
    )
    assert _found(tree) == [("context/project.py", 6, "C3")]


def test_an_exemption_that_exempts_nothing_fails(tree: Path) -> None:
    _write(tree, "context/project.py", "class JournalRecords:\n    pass\n")
    assert _found(tree) == [("context/project.py", 0, "stale exemption")]


def test_a_pool_elsewhere_in_the_dispatcher_is_not_exempt(tree: Path) -> None:
    """Only `execute_group`'s pool, made and joined in one call, is exempt."""
    _write(
        tree,
        "tools/dispatch.py",
        _GROUP + "    def start(self):\n        self._pool = ThreadPoolExecutor(4)\n",
    )
    assert _found(tree) == [("tools/dispatch.py", 7, "C1")]
