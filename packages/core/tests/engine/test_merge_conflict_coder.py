"""The coder resolves a conflict on a checkout mid-merge (Amendment 2 decision
6).

Decision 8 sends a merge's conflicts to the coder: Zeus's git line stops on them, the coder's
one-task plan resolves them in the checkout where the merge stopped, and Zeus concludes the
merge with a git-only line the person confirms. Each step is established here before anything
relies on it: the box runs in place on a checkout whose index holds unmerged entries, the
worker edits a file full of conflict markers, and concluding the merge passes D229's guard.

The real box and workspace, the model scripted, no sandbox: the person allowed an unsandboxed
shell for Zeus and approves every confirmation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    Approving,
    athena_accepts,
    build,
    make_ignoring_repo,
    worker_edit,
)
from engines import answer, call, calls
from repos import git
from zeus_turn_helpers import WAIT_S, result

from chickadee.engine import PacketEngine

CONFLICT = "<<<<<<< HEAD\n    return 2\n=======\n    return 3\n>>>>>>> feature\n"
RESOLVE = {
    "title": "Resolve the merge",
    "brief": "## Goal\nResolve the conflict in `app.py`, keeping the feature's `return 3`.\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """`main` and `feature` both changed `go()`'s one line, so merging stops on it."""
    made = make_ignoring_repo(tmp_path / "repo")
    git(made, "checkout", "-q", "-b", "feature")
    (made / "app.py").write_text("def go() -> int:\n    return 3\n")
    git(made, "commit", "-qam", "feature: return 3")
    git(made, "checkout", "-q", "main")
    (made / "app.py").write_text("def go() -> int:\n    return 2\n")
    git(made, "commit", "-qam", "main: return 2")
    return made


def _resolved(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    *then: Any,
    person: Approving | None = None,
) -> PacketEngine:
    raw["sandbox"] = {"allow_unsandboxed_shell": {"zeus": True}}
    script = [
        calls(call("shell", "s1", command="git merge feature")),
        calls(call("coder", "c1", **RESOLVE)),
        *worker_edit(CONFLICT, "    return 3\n"),
        *athena_accepts(),
        *then,
        answer("merged"),
    ]
    engine, _ = build(monkeypatch, repo, raw, script, interaction=person or Approving())
    engines.append(engine)
    engine.turn(SID, "merge feature into main").wait(timeout=WAIT_S)
    return engine


@pytest.mark.writes_worktree("app.py")
def test_the_coder_resolves_the_conflict_where_the_merge_stopped(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """In place, on a checkout whose index holds the unmerged entries: the box takes its
    snapshot, the worker edits the file of markers, and Athena accepts."""
    engine = _resolved(monkeypatch, repo, raw, engines)

    assert "git stopped on conflicts in app.py" in result(engine, "s1", SID)
    assert "COMPLETED" in result(engine, "c1", SID)
    body = (repo / "app.py").read_text()
    assert "<<<<<<<" not in body and "return 3" in body
    assert (repo / ".git" / "MERGE_HEAD").exists(), "the merge is still the person's to finish"


@pytest.mark.writes_worktree("app.py")
def test_zeus_concludes_the_merge_with_a_confirmed_git_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D229's guard keeps git from committing source a command wrote; the coder wrote this
    file, so the git line that concludes the merge goes through."""
    main = git(repo, "rev-parse", "HEAD")
    feature = git(repo, "rev-parse", "feature")
    person = Approving()
    engine = _resolved(
        monkeypatch,
        repo,
        raw,
        engines,
        calls(call("shell", "s2", command="git add app.py && git commit --no-edit")),
        person=person,
    )

    assert any("git add app.py" in a.display for a in person.shown), "the person confirmed it"
    said = result(engine, "s2", SID)
    assert "DENIED" not in said and "put back" not in said, said
    assert git(repo, "rev-parse", "HEAD^1") == main
    assert git(repo, "rev-parse", "HEAD^2") == feature, "a merge commit, both parents kept"
    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert "return 3" in git(repo, "show", "HEAD:app.py")
    assert git(repo, "status", "--porcelain", "--untracked-files=no") == ""
