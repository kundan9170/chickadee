"""`bring`, the fifth handoff choice, and the landing refused by overlap.

`bring` writes the run's changes into the person's checkout as uncommitted edits, as Codex's
Handoff and Cursor's Apply leave them: no branch moves, nothing is staged or committed. Every
landing is refused only by overlap, a path the run changed that also changed in the checkout
since the run started, never by a dirty checkout as such.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import commit_in, git, init_repo

from chickadee.errors import HandoffError
from chickadee.executor.reconcile import Landed
from chickadee.handoff import describe, finalize, landed, reconcile
from chickadee.schema.isolation import GitHandoffChoice
from chickadee.workspace import Workspace, acquire

OP = "op-b41f0001"
BRING = GitHandoffChoice.BRING


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo", {"app.py": "x = 1\n", "notes.txt": "mine\n"}, commit="first"
    )


@pytest.fixture
def run(repo: Path) -> tuple[Workspace, str, str]:
    """A finished worktree run: `app.py` changed, `new.py` added."""
    base = git(repo, "rev-parse", "HEAD")
    ws = acquire(repo, run_id="r-0000b41f", goal="return 2")
    ws.release()
    commit_in(ws.root, "app.py", "x = 2\n")
    tip = commit_in(ws.root, "pkg/new.py", "y = 1\n")
    return ws, base, tip


def _bring(repo: Path, ws: Workspace, base: str) -> object:
    return finalize(repo, ws.spec, BRING, operation_id=OP, base=base)


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py")
def test_bring_writes_the_runs_changes_uncommitted_and_moves_no_branch(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    ws, base, tip = run

    result = _bring(repo, ws, base)

    assert git(repo, "rev-parse", "HEAD") == base, "no branch moved"
    assert (repo / "app.py").read_text() == "x = 2\n"
    assert (repo / "pkg/new.py").read_text() == "y = 1\n"
    assert git(repo, "diff", "--cached", "--name-only") == "", "nothing staged"
    assert git(repo, "rev-parse", str(ws.spec.branch)) == tip, "the branch keeps the work"
    assert Path(str(ws.spec.worktree_path)).is_dir(), "the worktree stays, as after apply"
    assert result.choice is BRING  # type: ignore[attr-defined]
    assert result.commits_applied == ()  # type: ignore[attr-defined]


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py", "notes.txt", "scratch.txt")
def test_a_dirty_checkout_elsewhere_refuses_nothing(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    """Uncommitted changes are never a reason to refuse. Only the run's own
    paths matter, and the person's other edits are left exactly as they were."""
    ws, base, _ = run
    (repo / "notes.txt").write_text("mine, edited\n")
    (repo / "scratch.txt").write_text("untracked\n")

    view = describe(repo, ws.spec, base=base)
    assert view.overlapping == ()
    assert BRING in view.accepted and GitHandoffChoice.APPLY in view.accepted
    _bring(repo, ws, base)

    assert (repo / "notes.txt").read_text() == "mine, edited\n"
    assert (repo / "scratch.txt").read_text() == "untracked\n"
    assert (repo / "app.py").read_text() == "x = 2\n"


@pytest.mark.parametrize(
    "theirs",
    [
        pytest.param(("app.py", "x = 9\n", False), id="uncommitted"),
        pytest.param(("app.py", "x = 9\n", True), id="committed-since"),
        pytest.param(("pkg/new.py", "mine\n", False), id="untracked"),
    ],
)
@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py")
def test_a_path_both_sides_changed_is_refused_and_named(
    repo: Path, run: tuple[Workspace, str, str], theirs: tuple[str, str, bool]
) -> None:
    ws, base, _ = run
    name, body, commit = theirs
    if commit:
        commit_in(repo, name, body)
    else:
        if not (repo / name).parent.is_dir():
            (repo / name).parent.mkdir(parents=True)
        (repo / name).write_text(body)
    head = git(repo, "rev-parse", "HEAD")

    view = describe(repo, ws.spec, base=base)
    assert view.overlapping == (name,)
    why = view.refusal(BRING)
    assert why is not None and name in why and f"git merge {ws.spec.branch}" in why
    with pytest.raises(HandoffError, match="nothing was changed"):
        _bring(repo, ws, base)

    assert (repo / name).read_text() == body, "the person's change is untouched"
    assert git(repo, "rev-parse", "HEAD") == head


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py", "notes.txt")
def test_bring_lands_even_after_the_persons_branch_moved(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    """A commit of the person's on paths the run did not change: `apply` cannot fast-forward
    and names `bring`; `bring` lands, the person's commit kept."""
    ws, base, _ = run
    moved = commit_in(repo, "notes.txt", "mine, committed\n")

    view = describe(repo, ws.spec, base=base)
    apply_why = view.refusal(GitHandoffChoice.APPLY)
    assert apply_why is not None and "bring" in apply_why
    _bring(repo, ws, base)

    assert git(repo, "rev-parse", "HEAD") == moved
    assert (repo / "app.py").read_text() == "x = 2\n"


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py", "notes.txt")
def test_apply_no_longer_refuses_a_checkout_dirty_elsewhere(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    """The answer that moves: `apply` refused any uncommitted change; it now refuses
    only a path the run changed."""
    ws, base, tip = run
    (repo / "notes.txt").write_text("mine, edited\n")

    finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == tip
    assert (repo / "notes.txt").read_text() == "mine, edited\n"


def test_an_in_place_run_has_nothing_to_bring(repo: Path) -> None:
    from chickadee.config import WorkspacePolicy

    base = git(repo, "rev-parse", "HEAD")
    ws = acquire(repo, run_id="r-0000b41e", goal="g", policy=WorkspacePolicy(mode="in_place"))
    ws.release()

    view = describe(repo, ws.spec, base=base)
    why = view.refusal(BRING)
    assert why is not None and "keep" in why
    assert BRING not in view.accepted


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py")
def test_bring_is_reconciled_by_what_the_checkout_holds(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    ws, base, tip = run

    before = reconcile(repo, ws.spec, BRING, operation_id=OP, tip=tip, base=base)
    assert before.landed is Landed.NO
    assert landed(repo, base, tip) is None

    _bring(repo, ws, base)

    after = reconcile(repo, ws.spec, BRING, operation_id=OP, tip=tip, base=base)
    assert after.landed is Landed.YES
    assert landed(repo, base, tip) is BRING, "an undo then reads the code as in the checkout"


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py")
def test_a_bring_cut_short_is_unknown_never_a_licence_to_retry(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    """One path at the tip's content, one still at the base's: what landed cannot be told
    apart from what the person did since, so it is neither yes nor no."""
    ws, base, tip = run
    (repo / "app.py").write_text("x = 2\n")

    seen = reconcile(repo, ws.spec, BRING, operation_id=OP, tip=tip, base=base)

    assert seen.landed is Landed.UNKNOWN


@pytest.mark.writes_worktree("notes.txt")
def test_a_worktree_holding_only_its_seed_has_not_landed(repo: Path) -> None:
    """Final review of 2026-10-25: from the person's HEAD, the seed's paths hold the seed's
    content in the checkout, since they are the person's own edits, so nothing committed after
    the seed read as brought, and the work dropped out of the open works."""
    base = git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text("mine, edited\n")
    ws = acquire(repo, run_id="r-0000b41d", goal="g", session="S-seed")
    ws.release()
    tip = git(ws.root, "rev-parse", "HEAD")
    assert tip != base, "the worktree was seeded"

    assert landed(repo, base, tip) is None


@pytest.mark.writes_worktree("app.py", "pkg", "pkg/new.py")
def test_deleting_what_never_reached_the_checkout_has_not_landed(
    repo: Path, run: tuple[Workspace, str, str]
) -> None:
    """Final review of 2026-10-25: a later turn that only deletes a file an earlier, unlanded
    turn added finds the file absent from the checkout as at its tip, which read as brought, so
    an undo of it wrote into the checkout."""
    ws, _, added = run
    (Path(ws.root) / "pkg/new.py").unlink()
    git(ws.root, "commit", "-qam", "drop new.py")
    deleted = git(ws.root, "rev-parse", "HEAD")

    assert landed(repo, added, deleted) is None
