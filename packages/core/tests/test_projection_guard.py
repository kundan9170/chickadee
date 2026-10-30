"""The projection guard fails a second fold of the run journal.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI, and
a test pinned to today's count would fail whenever a theme legitimately moves it. Loaded by
file path, as `test_records_guard.py` loads the runner: `scripts/` is not a package core
imports.
"""

from __future__ import annotations

import ast
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
    spec = importlib.util.spec_from_file_location("_projection_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
projection: Any = importlib.import_module("guards.projection")
scope_of: Any = importlib.import_module("guards._scope")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with core's layout, no exemptions, and the guard pointed at it."""
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(projection, "REPO_ROOT", root)
    monkeypatch.setattr(projection, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0])
        for v in projection.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(projection.GUARD, find=projection.find, baseline={})


# -- P1 --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "test",
    [
        "e.kind is EventKind.RUN_END",
        "e.kind is not schema.EventKind.RUN_END",
        "e.kind == EventKind.RUN_END",
        "e.kind in {EventKind.RUN_END, EventKind.ESCALATED}",
        "e.kind in frozenset({EventKind.RUN_END})",
        "e.kind not in (EventKind.RUN_END,)",
    ],
)
def test_a_lines_kind_against_an_event_kind_fails(tree: Path, test: str) -> None:
    _write(tree, "context/ended.py", f"def ended(e):\n    return {test}\n")
    assert _found(tree) == [("context/ended.py", 2, "P1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}context/ended.py", 1, 0),)


def test_membership_in_a_module_level_set_of_kinds_fails(tree: Path) -> None:
    source = (
        "_TERMINALS = frozenset({EventKind.PACKET_RESOLVED, EventKind.ESCALATED})\n"
        "def terminal(e):\n"
        "    return e.kind in _TERMINALS\n"
    )
    _write(tree, "boxctx/records.py", source)
    assert _found(tree) == [("boxctx/records.py", 3, "P1")]


def test_a_match_on_a_lines_kind_fails(tree: Path) -> None:
    source = (
        "def count(e):\n"
        "    match e.kind:\n"
        "        case EventKind.EXEC:\n"
        "            return 1\n"
        "    return 0\n"
    )
    _write(tree, "journal/metrics.py", source)
    assert _found(tree) == [("journal/metrics.py", 2, "P1")]


def test_one_lines_kind_against_another_fails_where_a_line_is_a_parameter(tree: Path) -> None:
    source = (
        "def same(lines, recorded: Append):\n"
        "    return lines[0].kind is recorded.kind\n"
        "def same_kind(packet, stub):\n"
        "    return packet.kind is not stub.kind\n"
    )
    _write(tree, "orchestrator/run.py", source)
    assert _found(tree) == [("orchestrator/run.py", 2, "P1")]


@pytest.mark.parametrize(
    "test",
    [
        # The kind a writer is handed, never a line's: the write path's policy.
        "kind in FSYNC_KINDS",
        "kind not in {EventKind.SCOUT_START}",
        # Another enum's kind.
        "packet.kind is PacketKind.ACTION",
        "handle.kind == 'file'",
        # A payload's type: how a line is read.
        "isinstance(e.payload, RunEndP)",
    ],
)
def test_what_is_not_a_lines_kind_is_not_p1(tree: Path, test: str) -> None:
    source = (
        "FSYNC_KINDS = frozenset({EventKind.CLASSIFY})\n"
        f"def check(kind, packet, handle, e):\n    return {test}\n"
    )
    _write(tree, "journal/other.py", source)
    assert _found(tree) == []


def test_writing_a_line_names_its_kind_and_is_not_p1(tree: Path) -> None:
    _write(
        tree,
        "orchestrator/driver.py",
        "def ask(journal, p):\n    journal.append(EventKind.QUESTION_ASKED, p)\n",
    )
    assert _found(tree) == []


def test_the_fold_reads_kinds(tree: Path) -> None:
    _write(tree, "journal/projection.py", "def add(e):\n    return e.kind is EventKind.EXEC\n")
    assert _found(tree) == []


# -- P2 --------------------------------------------------------------------------------------


def test_reading_a_journal_file_outside_the_journal_package_fails(tree: Path) -> None:
    _write(tree, "run_header.py", "def episode(path):\n    return list(read_events(path))\n")
    assert _found(tree) == [("run_header.py", 2, "P2")]


@pytest.mark.parametrize(
    "home", ["journal/reader.py", "journal/writer.py", "journal/gc.py", "journal/projection.py"]
)
def test_the_journal_package_reads_its_files(tree: Path, home: str) -> None:
    _write(tree, home, "def read(path):\n    return list(read_events(path))\n")
    assert _found(tree) == []


# -- P3 --------------------------------------------------------------------------------------


def test_iterating_a_held_journal_fails(tree: Path) -> None:
    source = (
        "def cost(journal):\n"
        "    return sum(1 for e in journal.events())\n"
        "def finish(self):\n"
        "    return list(self.ports.journal.events())\n"
    )
    _write(tree, "orchestrator/run.py", source)
    assert _found(tree) == [("orchestrator/run.py", 2, "P3"), ("orchestrator/run.py", 4, "P3")]


def test_a_held_journal_folded_by_the_projection_is_not_p3(tree: Path) -> None:
    source = (
        "def whole(journal):\n"
        "    return project(journal.events())\n"
        "def before(self, seq):\n"
        "    return project_before(self.orch.ports.journal.events(), seq)\n"
    )
    _write(tree, "orchestrator/run.py", source)
    assert _found(tree) == []


def test_events_held_by_something_else_are_not_p3(tree: Path) -> None:
    _write(tree, "context/session.py", "def fold(log):\n    return list(log.events())\n")
    assert _found(tree) == []


def test_a_loop_over_a_journal_file_fails_outside_the_fold(tree: Path) -> None:
    source = "def walk(path):\n    for e in read_events(path):\n        print(e)\n"
    _write(tree, "journal/gc.py", source)
    _write(tree, "journal/writer.py", source)
    assert _found(tree) == [("journal/gc.py", 2, "P3")]


# -- exemptions and the baseline -------------------------------------------------------------


def test_exemptions_are_by_function(tree: Path, monkeypatch: Any) -> None:
    rel = f"{CORE}journal/gc.py"
    monkeypatch.setattr(projection, "EXEMPT", {(rel, "classify"): "the blob walk"})
    source = (
        "def classify(path):\n"
        "    for e in read_events(path):\n"
        "        print(e)\n"
        "def other(path):\n"
        "    for e in read_events(path):\n"
        "        print(e)\n"
    )
    _write(tree, "journal/gc.py", source)
    assert _found(tree) == [("journal/gc.py", 5, "P3")]


def test_an_exemption_that_exempts_nothing_fails(tree: Path, monkeypatch: Any) -> None:
    rel = f"{CORE}orchestrator/run.py"
    monkeypatch.setattr(projection, "EXEMPT", {(rel, "_without_recorded"): "legality"})
    _write(tree, "orchestrator/run.py", "def _without_recorded(lines):\n    return lines\n")
    assert _found(tree) == [("orchestrator/run.py", 0, "stale exemption")]


def test_a_count_within_its_baseline_passes(tree: Path) -> None:
    _write(tree, "budget/family.py", "def spend(path):\n    return list(read_events(path))\n")
    guard = dataclasses.replace(_guard(), baseline={f"{CORE}budget/family.py": 1})
    assert not runner.evaluate(guard, tree).failed


def test_the_live_baseline_is_empty() -> None:
    # The guard landed tolerating only the spend `T-3.75` had not moved, and
    # `T-3.75` moved it.
    assert projection.BASELINE == {}


def test_every_live_exemption_names_a_function_that_exists() -> None:
    for (rel, function), reason in projection.EXEMPT.items():
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        scopes = {scope for scope, _ in scope_of.functions(tree)}
        assert function in scopes, f"{rel}: no function {function}"
        assert reason
