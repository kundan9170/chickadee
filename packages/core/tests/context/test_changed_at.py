"""`read(path, at="@changed")` through the box's real reader, against the run's base.

Pinned before a reader was given a `ChangedLines`, when nothing implemented one: after an edit,
`@changed` said there was no base to compare against. Moved with T-3.185, listed in its Log:
the box's readers are given the run's base, and `@changed` returns exactly the lines that
differ from it (`context/changes.py`); a reader given no base still says none is set."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from context_helpers import python_module
from repos import git, init_repo

from chickadee.box.compose import _reader  # pyright: ignore[reportPrivateUsage]
from chickadee.context.changes import GitChanges, changed_spans, worktree_changes
from chickadee.context.reader import Reader
from chickadee.context.results import InvalidArgument, ToolResult
from chickadee.context.spans import Span
from chickadee.context.terms import Vision
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.journal.port import JournalPort
from chickadee.secrets.denylist import Denylist

RUN = cast(JournalPort, SimpleNamespace(run_id="2026-10-20-00000000"))
"""The one field a box reader reads of its journal before a stored handle opens."""


MOD, _ = python_module(10)
"""Fifty lines: ten functions of four lines, each with a blank after."""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"mod.py": MOD})


def _box_reader(repo: Path, base: str | None) -> Reader:
    return _reader(repo, Denylist(), Vision("worker"), RUN, None, base)


def _edit(repo: Path, old: str, new: str) -> None:
    path = repo / "mod.py"
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


@pytest.mark.writes_worktree("mod.py")
def test_with_no_base_at_changed_says_none_is_set(repo: Path) -> None:
    _edit(repo, "return 3", "return 33")
    got = _box_reader(repo, None).read("mod.py", at="@changed")
    assert isinstance(got, InvalidArgument), got
    assert got.why == "@changed needs a base to compare against; none is set"


@pytest.mark.writes_worktree("mod.py")
def test_at_changed_after_an_edit_returns_exactly_the_changed_lines(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    _edit(repo, "return 3", "return 33")
    _edit(repo, "    y1 = x + 801\n", "    y1 = x + 801\n    y9 = x\n")
    got = _box_reader(repo, base).read("mod.py", at="@changed")
    assert isinstance(got, ToolResult), got
    assert got.view == (Span(14, 14), Span(38, 38))
    assert got.content is not None
    assert "14│    return 33" in got.content and "38│    y9 = x" in got.content


@pytest.mark.writes_worktree("mod.py")
def test_a_committed_change_still_differs_from_the_base(repo: Path) -> None:
    """A task's commit lands on the run's branch; the base is where the run started."""
    base = git(repo, "rev-parse", "HEAD")
    _edit(repo, "return 3", "return 33")
    git(repo, "commit", "-qam", "T1")
    got = _box_reader(repo, base).read("mod.py", at="@changed")
    assert isinstance(got, ToolResult), got
    assert got.view == (Span(14, 14),)


def test_with_no_change_at_changed_says_so(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    got = _box_reader(repo, base).read("mod.py", at="@changed")
    assert isinstance(got, InvalidArgument), got
    assert got.why == "nothing in mod.py differs from the base"


@pytest.mark.writes_worktree("new.py")
def test_a_file_new_since_the_base_differs_on_every_line(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    (repo / "new.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    got = _box_reader(repo, base).read("new.py", at="@changed")
    assert isinstance(got, ToolResult), got
    assert got.view == (Span(1, 2),)


def test_a_base_git_does_not_know_is_no_answer(repo: Path) -> None:
    assert GitChanges(repo, "0" * 40).changed_lines("mod.py") is None


def test_hunks_name_the_new_sides_lines() -> None:
    diff = (
        "@@ -3 +3 @@\n-a\n+b\n"
        "@@ -10,0 +11,2 @@\n+c\n+d\n"
        "@@ -20,2 +21,0 @@\n-e\n-f\n"
        "@@ -1,1 +0,0 @@\n-g\n"
    )
    assert changed_spans(diff) == (Span(1, 1), Span(3, 3), Span(11, 12), Span(21, 21))


def test_zeus_compares_only_the_session_worktree(tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    ref = WorkspaceRef(
        mode="worktree",
        lease_id="L1",
        source_ref="main",
        source_head="a" * 40,
        branch="chickadee/s1",
        worktree_path=str(worktree),
        handoff="pending",
    )
    at_worktree = worktree_changes(worktree, ref)
    assert at_worktree is not None and at_worktree.base == "a" * 40
    assert worktree_changes(tmp_path / "checkout", ref) is None
    assert worktree_changes(worktree, None) is None
