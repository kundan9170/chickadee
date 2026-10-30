"""T-2.11 — reviewed apply/squash/keep/discard primitives.

The claims worth testing here are the ones that are invisible when they break: a handoff
that mutates something before it refuses, a squash whose commit reconciliation cannot
identify afterwards, and a "never pushes" that holds only because nobody happened to write
`push`. Each of those looks exactly like success from the outside.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest
from repos import commit_in, git, init_repo

from chickadee.config import WorkspacePolicy
from chickadee.errors import HandoffError, InfraError
from chickadee.executor.reconcile import TRAILER, Landed, by_operation_trailer
from chickadee.handoff import (  # pyright: ignore[reportPrivateUsage]
    HandoffPreview,
    _discard,
    describe,
    finalize,
    reconcile,
)
from chickadee.process import Completed, ProcessSupervisor
from chickadee.schema.isolation import GitHandoffChoice, IsolationSpec
from chickadee.workspace import (
    OPS_REF_PREFIX,
    Workspace,
    acquire,
    snapshot_operation,
)

OP = "op-2f9c41d8"


class Recorder(ProcessSupervisor):
    """A real supervisor that keeps every argv it was asked to spawn.

    Real rather than fake on purpose: a stub would prove only that the module does not
    build a push argv against a repository that does not exist. This one runs the commands
    for real *and* keeps the list, so the push assertion is made about the same execution
    that produced the observed git state.
    """

    def __init__(self) -> None:
        self.argv: list[tuple[str, ...]] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        self.argv.append(argv)
        return super().run(argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"kept.txt": "original\n"}, commit="first")


@pytest.fixture
def run(repo: Path) -> tuple[Workspace, str, tuple[str, str]]:
    """A finished worktree run: two packet commits on the run's internal branch."""
    base = git(repo, "rev-parse", "HEAD")
    ws = acquire(repo, run_id="r-0000abcd", goal="add session refresh")
    ws.release()
    first = commit_in(ws.root, "one.py", "one\n")
    second = commit_in(ws.root, "two.py", "two\n")
    return ws, base, (first, second)


@pytest.fixture
def in_place(repo: Path) -> tuple[Workspace, str]:
    base = git(repo, "rev-parse", "HEAD")
    ws = acquire(
        repo,
        run_id="r-0000beef",
        goal="quick fix",
        policy=WorkspacePolicy(mode="in_place"),
    )
    ws.release()
    commit_in(repo, "quick.py", "quick\n")
    return ws, base


# --- describe: what §3.5.1 says the handoff must identify --------------------


def test_describe_identifies_the_worktree_branch_lease_and_commit_range(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, commits = run
    view = describe(repo, ws.spec, base=base)

    assert view.branch == ws.spec.branch
    assert view.worktree_path == ws.spec.worktree_path
    assert view.lease_id == ws.spec.lease_id
    assert view.base == base
    assert view.tip == commits[-1]
    assert view.commits == commits, "oldest first, so the range reads like history"
    assert view.source_branch == "main"
    assert view.source_head == base
    assert view.refusal(GitHandoffChoice.APPLY) is None
    assert set(view.accepted) == set(GitHandoffChoice)


@pytest.mark.writes_worktree("user.py")
def test_describe_reports_drift_instead_of_raising(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run
    moved = commit_in(repo, "user.py", "theirs\n")

    view = describe(repo, ws.spec, base=base)

    assert view.drift is not None
    refused = view.refusal(GitHandoffChoice.APPLY)
    assert refused is not None and refused.startswith(view.drift)
    assert "bring" in refused, "the exit the harness owns (D230 Amendment 2)"
    assert view.refusal(GitHandoffChoice.KEEP) is None, "keep touches nothing"
    assert base[:12] in view.drift and moved[:12] in view.drift
    assert view.source_head == moved


# --- apply -------------------------------------------------------------------


def test_apply_fast_forwards_and_keeps_every_packet_commit(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, commits = run

    result = finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == commits[-1]
    assert result.commits_applied == commits
    # The identities survive, not just the shape: each packet is a rollback unit someone
    # may already have named by sha, and a cherry-pick would silently renumber them.
    landed = git(repo, "log", "--format=%H", f"{base}..HEAD").split()
    assert sorted(landed) == sorted(commits)
    assert (repo / "one.py").exists() and (repo / "two.py").exists()
    assert result.pushed is False


def test_apply_with_nothing_to_replay_leaves_the_branch_alone(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    ws = acquire(repo, run_id="r-0000dead", goal="no op")
    ws.release()

    result = finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)

    assert result.commits_applied == ()
    assert git(repo, "rev-parse", "HEAD") == base
    assert "no commits" in result.detail


# --- squash ------------------------------------------------------------------


def test_squash_writes_one_commit_carrying_the_operation_trailer(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, commits = run

    result = finalize(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP, base=base)

    head = git(repo, "rev-parse", "HEAD")
    assert head == result.source_head
    assert result.commits_applied == (head,)
    assert git(repo, "log", "--format=%H", f"{base}..HEAD").split() == [head]
    assert f"{TRAILER}: {OP}" in git(repo, "log", "-1", "--format=%B")
    # The point of a squash is that the *content* is the run's, only the history differs.
    run_tree = git(repo, "rev-parse", f"{commits[-1]}^{{tree}}")
    assert git(repo, "rev-parse", "HEAD^{tree}") == run_tree
    assert (repo / "two.py").read_text(encoding="utf-8") == "two\n"


def test_the_squash_commit_is_what_the_existing_trailer_reader_looks_for(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    """`reconcile.by_operation_trailer` has had no writer since it was built in T-2.3, so
    it answered `NO` — the one verdict that permits a retry — for every operation ever
    performed. This is the test that the two halves finally meet."""
    ws, base, _ = run
    assert by_operation_trailer(repo, OP).landed is Landed.NO

    finalize(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP, base=base)

    found = by_operation_trailer(repo, OP)
    assert found.landed is Landed.YES
    assert not found.may_retry
    assert by_operation_trailer(repo, "op-someone-else").landed is Landed.NO


def test_squash_does_not_attribute_the_commit_to_the_user(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run

    finalize(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP, base=base)

    assert git(repo, "log", "-1", "--format=%an") == "chickadee"
    assert git(repo, "log", "-1", "--format=%ae") == "chickadee@localhost"


# --- keep --------------------------------------------------------------------


def test_keep_changes_nothing_and_says_where_the_work_is(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, commits = run

    result = finalize(repo, ws.spec, GitHandoffChoice.KEEP, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == base
    assert git(repo, "rev-parse", str(ws.spec.branch)) == commits[-1]
    assert Path(str(ws.spec.worktree_path)).exists()
    assert result.branch == ws.spec.branch
    assert result.commits_applied == ()


# --- discard -----------------------------------------------------------------


def test_discard_removes_the_worktree_and_branch_and_leaves_the_source_alone(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run

    result = finalize(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == base
    assert not Path(str(ws.spec.worktree_path)).exists()
    assert str(ws.spec.branch) not in git(repo, "branch", "--format=%(refname:short)").split()
    assert result.branch is None and result.worktree_path is None
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "original\n"


def test_discard_deletes_the_uncommitted_work_it_was_shown_and_names_it(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    """The person chose `discard` after the review listed the work no commit holds, so
    it goes with the worktree, and the result names what was deleted."""
    ws, base, _ = run
    (ws.root / "unsaved.py").write_text("not committed anywhere\n", encoding="utf-8")
    assert describe(repo, ws.spec, base=base).refusal(GitHandoffChoice.DISCARD) is None

    result = finalize(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)

    assert not Path(str(ws.spec.worktree_path)).exists()
    assert str(ws.spec.branch) not in git(repo, "branch", "--format=%(refname:short)").split()
    assert "deleted work no commit held: unsaved.py" in result.detail
    assert git(repo, "rev-parse", "HEAD") == base


def test_discard_without_listed_work_keeps_gits_refusal_of_a_dirty_worktree(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    """`--force` is passed only over work the preview listed: a file written after it is
    refused by `git worktree remove` itself, never deleted unseen."""
    ws, base, _ = run
    clean = describe(repo, ws.spec, base=base)
    assert clean.uncommitted == ()
    (ws.root / "late.py").write_text("written after the review\n", encoding="utf-8")

    with pytest.raises(HandoffError) as caught:
        _discard(ProcessSupervisor(), repo, clean, operation_id=OP)

    assert "worktree remove" in str(caught.value)
    assert (ws.root / "late.py").exists()


def test_discard_refuses_to_delete_the_branch_the_source_is_standing_on(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    spec = IsolationSpec(
        mode="worktree",
        lease_id="repo:1",
        branch="main",
        worktree_path=str(repo),
    )

    with pytest.raises(HandoffError, match="user's own branch"):
        finalize(repo, spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == base
    assert "main" in git(repo, "branch", "--format=%(refname:short)").split()


# --- source drift: refuse, and mutate nothing --------------------------------


@pytest.mark.parametrize("choice", [GitHandoffChoice.APPLY, GitHandoffChoice.SQUASH])
@pytest.mark.writes_worktree("user.py")
def test_a_moved_source_branch_refuses_without_mutating_anything(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]], choice: GitHandoffChoice
) -> None:
    ws, base, commits = run
    moved = commit_in(repo, "user.py", "theirs\n")

    with pytest.raises(HandoffError, match="nothing was changed"):
        finalize(repo, ws.spec, choice, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == moved
    assert git(repo, "rev-parse", str(ws.spec.branch)) == commits[-1]
    assert not (repo / "one.py").exists()
    assert Path(str(ws.spec.worktree_path)).exists()


@pytest.mark.parametrize(
    "choice", [GitHandoffChoice.APPLY, GitHandoffChoice.SQUASH, GitHandoffChoice.BRING]
)
@pytest.mark.writes_worktree("one.py")
def test_a_source_change_on_the_runs_paths_refuses_without_mutating_anything(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]], choice: GitHandoffChoice
) -> None:
    """Was: any uncommitted change refused (D230 Amendment 2 moves it). Now only a
    path the run changed, which a landing would write over."""
    ws, base, _ = run
    (repo / "one.py").write_text("the person's own, not committed\n", encoding="utf-8")

    with pytest.raises(HandoffError, match="also changed in the checkout"):
        finalize(repo, ws.spec, choice, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == base
    assert (repo / "one.py").read_text(encoding="utf-8") == "the person's own, not committed\n"
    assert not (repo / "two.py").exists()


@pytest.mark.parametrize("choice", [GitHandoffChoice.KEEP, GitHandoffChoice.DISCARD])
@pytest.mark.writes_worktree("kept.txt")
def test_a_dirty_source_does_not_block_the_choices_that_never_touch_it(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]], choice: GitHandoffChoice
) -> None:
    """`discard` never touches the source. A precondition about the source checkout
    would refuse a safe operation on evidence that has nothing to do with it."""
    ws, base, _ = run
    (repo / "kept.txt").write_text("edited but not committed\n", encoding="utf-8")

    result = finalize(repo, ws.spec, choice, operation_id=OP, base=base)

    assert result.choice is choice
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "edited but not committed\n"
    assert git(repo, "rev-parse", "HEAD") == base


def test_a_detached_source_head_refuses_to_be_applied_onto(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run
    git(repo, "checkout", "-q", "--detach", "HEAD")

    with pytest.raises(HandoffError, match="detached HEAD"):
        finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == base
    assert not (repo / "one.py").exists()


def test_a_branch_that_is_already_gone_is_named_rather_than_guessed_at(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    spec = IsolationSpec(
        mode="worktree",
        lease_id="repo:1",
        branch="chickadee/never-existed",
        worktree_path=str(repo / "nowhere"),
    )

    with pytest.raises(HandoffError, match="no longer exists"):
        describe(repo, spec, base=base)


# --- never pushes ------------------------------------------------------------


@pytest.mark.parametrize(
    "choice",
    [
        GitHandoffChoice.APPLY,
        GitHandoffChoice.SQUASH,
        GitHandoffChoice.KEEP,
        GitHandoffChoice.DISCARD,
    ],
)
def test_no_choice_ever_builds_a_push(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]], choice: GitHandoffChoice
) -> None:
    """ "Chickadee never silently pushes." The argv this module builds is its entire
    effect on the repository, so asserting over all of it is the whole claim."""
    ws, base, _ = run
    rec = Recorder()

    finalize(repo, ws.spec, choice, operation_id=OP, base=base, supervisor=rec)

    assert rec.argv, "the recorder saw nothing, so this asserts about nothing"
    for argv in rec.argv:
        assert argv[0] == "git"
        assert "push" not in argv
        assert not any(a.startswith("--repo") or a == "remote" for a in argv)


# --- reconciliation ----------------------------------------------------------


def test_apply_is_reconciled_by_ancestry_since_it_writes_no_commit(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, commits = run
    before = reconcile(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP)
    assert before.landed is Landed.NO and before.may_retry

    finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)

    after = reconcile(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, tip=commits[-1])
    assert after.landed is Landed.YES
    assert not after.may_retry


def test_squash_is_reconciled_by_its_trailer(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run
    assert reconcile(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP).may_retry

    finalize(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP, base=base)

    after = reconcile(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP)
    assert after.landed is Landed.YES


def test_discard_is_reconciled_by_the_branch_being_gone(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run
    before = reconcile(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP)
    assert before.landed is Landed.NO

    finalize(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)

    after = reconcile(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP)
    assert after.landed is Landed.YES


def test_a_missing_branch_after_keep_is_unknown_not_a_licence_to_retry(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    """`NO` is the only verdict that permits a retry, so it is reserved for observations
    that positively show the operation did not happen. A vanished branch is not one:
    retrying a `keep` cannot bring it back."""
    ws, base, _ = run
    finalize(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)

    verdict = reconcile(repo, ws.spec, GitHandoffChoice.KEEP, operation_id=OP)

    assert verdict.landed is Landed.UNKNOWN
    assert not verdict.may_retry


def test_apply_without_a_tip_or_a_branch_is_unknown_rather_than_no(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run
    finalize(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)

    verdict = reconcile(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP)

    assert verdict.landed is Landed.UNKNOWN
    assert not verdict.may_retry


# --- in-place has no worktree branch, so it gets fewer choices ---------------


@pytest.mark.parametrize("choice", [GitHandoffChoice.APPLY, GitHandoffChoice.KEEP])
@pytest.mark.writes_worktree("quick.py")
def test_in_place_apply_and_keep_are_honest_no_ops(
    repo: Path, in_place: tuple[Workspace, str], choice: GitHandoffChoice
) -> None:
    ws, base = in_place
    head = git(repo, "rev-parse", "HEAD")

    result = finalize(repo, ws.spec, choice, operation_id=OP, base=base)

    assert git(repo, "rev-parse", "HEAD") == head
    assert result.commits_applied == ()
    assert "in-place" in result.detail


@pytest.mark.parametrize("choice", [GitHandoffChoice.SQUASH, GitHandoffChoice.DISCARD])
@pytest.mark.writes_worktree("quick.py")
def test_in_place_refuses_to_rewrite_the_branch_the_user_is_standing_on(
    repo: Path, in_place: tuple[Workspace, str], choice: GitHandoffChoice
) -> None:
    """`01-orchestrator` §3.5.2: in-place recovery "uses those durable preimages, not
    `git reset --hard`". The refusal hands the user the snapshot rather than the reset."""
    ws, base = in_place
    head = git(repo, "rev-parse", "HEAD")

    with pytest.raises(HandoffError) as caught:
        finalize(repo, ws.spec, choice, operation_id=OP, base=base)

    assert "refs/chickadee/snapshots" in str(caught.value)
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "quick.py").exists()


@pytest.mark.writes_worktree("quick.py")
def test_in_place_drift_is_the_branch_no_longer_descending_from_the_base(
    repo: Path, in_place: tuple[Workspace, str]
) -> None:
    ws, base = in_place
    git(repo, "checkout", "-q", "-b", "elsewhere", base)
    git(repo, "checkout", "-q", "--orphan", "unrelated")
    git(repo, "commit", "-qm", "unrelated root", "--allow-empty")

    view = describe(repo, ws.spec, base=base)

    assert view.drift is not None
    assert "no longer descends" in view.drift


# --- git that does not answer ------------------------------------------------


class Hanging(ProcessSupervisor):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        return Completed(b"", b"", 124, True)


def test_a_git_that_times_out_is_infrastructure_not_a_handoff_verdict(repo: Path) -> None:
    """`InfraError`, never `HandoffError`: git failing to finish says nothing about
    whether the source drifted, and reporting it as a drift would send the user looking
    at their branch for a problem that is in their machine."""
    spec = IsolationSpec(
        mode="worktree", lease_id="repo:1", branch="b", worktree_path=str(repo / "wt")
    )

    with pytest.raises(InfraError, match="timed out"):
        describe(repo, spec, base="0" * 40, supervisor=Hanging())


def test_the_refusal_rule_is_one_implementation_not_two() -> None:
    """T-5.5 renders from `refusal`/`accepted` and `finalize` enforces from the same call,
    so a choice the preview offers is one `finalize` will carry out. This pins the matrix
    directly, without a repository, so a change to the rule has to come here."""

    def preview(**over: object) -> HandoffPreview:
        fields: dict[str, object] = dict(
            mode="worktree",
            lease_id="repo:1",
            branch="chickadee/x-0000abcd",
            worktree_path="/tmp/wt",
            base="a" * 40,
            tip="b" * 40,
            commits=("b" * 40,),
            source_branch="main",
            source_head="a" * 40,
            source_clean=True,
            drift=None,
        )
        fields.update(over)
        return HandoffPreview(**fields)  # type: ignore[arg-type]

    assert set(preview().accepted) == set(GitHandoffChoice)

    # Drift stops the two that move the source branch, and only those; `bring` moves none.
    drifted = preview(drift="the source branch moved")
    assert set(drifted.accepted) == {
        GitHandoffChoice.KEEP,
        GitHandoffChoice.DISCARD,
        GitHandoffChoice.BRING,
    }

    # A detached HEAD has no branch to apply onto, and `drift` does not know that.
    detached = preview(source_branch=None)
    assert detached.drift is None
    assert set(detached.accepted) == {
        GitHandoffChoice.KEEP,
        GitHandoffChoice.DISCARD,
        GitHandoffChoice.BRING,
    }

    # A path both sides changed stops every landing, and nothing else.
    overlapping = preview(overlapping=("one.py",))
    assert set(overlapping.accepted) == {GitHandoffChoice.KEEP, GitHandoffChoice.DISCARD}

    # In-place: the two that would rewrite the user's branch, and nothing else.
    inplace = preview(mode="in_place", branch=None, worktree_path=None)
    assert set(inplace.accepted) == {GitHandoffChoice.APPLY, GitHandoffChoice.KEEP}

    # A dirty in-place checkout is the ordinary state of such a run, not a refusal.
    dirty = preview(
        mode="in_place",
        branch=None,
        worktree_path=None,
        source_clean=False,
        drift="the source checkout has uncommitted changes",
    )
    assert set(dirty.accepted) == {GitHandoffChoice.APPLY, GitHandoffChoice.KEEP}

    # Discarding the branch the source is standing on is refused however clean things are.
    standing = preview(branch="main")
    assert GitHandoffChoice.DISCARD not in standing.accepted


# --- review findings on PR #126 ---------------------------------------------


@pytest.mark.writes_worktree("quick.py", "kept.txt")
def test_in_place_apply_is_not_blocked_by_the_users_own_uncommitted_work(
    repo: Path, in_place: tuple[Workspace, str]
) -> None:
    """In-place is *specified* to leave pre-existing changes alone — `workspace.snapshot`
    records them and D39 says the mode never auto-commits user work. So the dirty tree the
    guard refuses on is the normal state of an in-place run, and `apply` there mutates
    nothing at all. The source preconditions belong to the choices that move the branch."""
    ws, base = in_place
    (repo / "kept.txt").write_text("the user's own edit\n", encoding="utf-8")
    head = git(repo, "rev-parse", "HEAD")

    result = finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)

    assert result.choice is GitHandoffChoice.APPLY
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "the user's own edit\n"


@pytest.mark.writes_worktree("quick.py")
def test_the_preview_refuses_exactly_what_finalize_refuses_in_place(
    repo: Path, in_place: tuple[Workspace, str]
) -> None:
    """T-5.5 renders from the preview and must not re-derive the rule: a surface that
    offered `squash` on an in-place run would fail at `finalize`."""
    ws, base = in_place
    view = describe(repo, ws.spec, base=base)

    assert view.refusal(GitHandoffChoice.SQUASH) is not None
    assert view.refusal(GitHandoffChoice.DISCARD) is not None
    assert view.refusal(GitHandoffChoice.APPLY) is None
    assert set(view.accepted) == {GitHandoffChoice.APPLY, GitHandoffChoice.KEEP}


def test_the_preview_refuses_the_mutating_choices_on_a_detached_head(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run
    git(repo, "checkout", "-q", "--detach", "HEAD")

    view = describe(repo, ws.spec, base=base)

    assert view.refusal(GitHandoffChoice.APPLY) is not None
    assert view.refusal(GitHandoffChoice.SQUASH) is not None
    assert set(view.accepted) == {
        GitHandoffChoice.KEEP,
        GitHandoffChoice.DISCARD,
        GitHandoffChoice.BRING,
    }


@pytest.mark.writes_worktree("quick.py")
def test_a_refused_in_place_squash_is_never_reconciled_as_retryable(
    repo: Path, in_place: tuple[Workspace, str]
) -> None:
    """`NO` permits a retry. An in-place `squash` can never succeed, so answering `NO` —
    which is what the trailer reader says about a commit that was never written — puts
    recovery in a loop on an operation that is refused by design."""
    ws, base = in_place
    with pytest.raises(HandoffError):
        finalize(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP, base=base)

    verdict = reconcile(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP)

    assert not verdict.may_retry


@pytest.fixture
def empty_in_place(tmp_path: Path) -> tuple[Path, Workspace, str]:
    root = tmp_path / "empty"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "T")
    ws = acquire(
        root,
        run_id="r-0000cafe",
        goal="first commit",
        policy=WorkspacePolicy(mode="in_place"),
    )
    ws.release()
    return root, ws, str(ws.spec.pre_image_sha)


def test_an_in_place_run_that_started_before_the_first_commit_can_still_hand_off(
    empty_in_place: tuple[Path, Workspace, str],
) -> None:
    """`workspace.snapshot` supports an empty repository and names the empty tree as the
    pre-image, so `acquire` produces this spec. Every choice goes through `describe`, so
    refusing it here means such a run can never be completed at all."""
    root, ws, base = empty_in_place

    view = describe(root, ws.spec, base=base)

    assert view.commits == ()
    assert view.tip is None
    assert view.refusal(GitHandoffChoice.KEEP) is None
    result = finalize(root, ws.spec, GitHandoffChoice.KEEP, operation_id=OP, base=base)
    assert result.commits_applied == ()


@pytest.mark.writes_worktree("first.py")
def test_the_range_of_a_run_that_started_from_nothing_is_its_whole_history(
    empty_in_place: tuple[Path, Workspace, str],
) -> None:
    root, ws, base = empty_in_place
    made = commit_in(root, "first.py", "first\n")

    view = describe(root, ws.spec, base=base)

    assert view.commits == (made,)
    assert view.tip == made
    assert view.drift is None


@pytest.mark.writes_worktree("first.py")
def test_a_refusal_hands_over_a_command_that_actually_runs(
    empty_in_place: tuple[Path, Workspace, str],
) -> None:
    """A refusal exists to leave the user able to act, so the command in it has to work in
    the state that produced the refusal. `EMPTY_TREE` is a tree, so `git log <it>..HEAD` is
    rejected by git — and that is precisely the state an unborn in-place run is in."""
    root, ws, base = empty_in_place

    no_commits = describe(root, ws.spec, base=base).refusal(GitHandoffChoice.SQUASH)
    assert no_commits is not None
    assert "added no commits" in no_commits
    assert base[:12] not in no_commits

    commit_in(root, "first.py", "first\n")
    whole = describe(root, ws.spec, base=base).refusal(GitHandoffChoice.SQUASH)
    assert whole is not None
    assert "`git log HEAD`" in whole
    assert base[:12] not in whole

    # Every command a refusal offers is one git accepts.
    for message in (no_commits, whole):
        for command in re.findall(r"`(git [^`]+)`", message):
            git(root, *command.split()[1:])


@pytest.mark.writes_worktree("quick.py")
def test_an_ordinary_in_place_refusal_still_names_the_two_dot_range(
    repo: Path, in_place: tuple[Workspace, str]
) -> None:
    ws, base = in_place
    message = describe(repo, ws.spec, base=base).refusal(GitHandoffChoice.DISCARD)

    assert message is not None
    assert f"`git log {base[:12]}..HEAD`" in message
    git(repo, "log", f"{base}..HEAD")


# --- D107, D162: snapshot refs outlive the handoff; only retention removes them -----------


def _ops_refs(repo: Path, branch: str) -> list[str]:
    out = git(repo, "for-each-ref", "--format=%(refname)", f"{OPS_REF_PREFIX}/{branch}/")
    return [ln for ln in out.splitlines() if ln]


@pytest.fixture
def run_with_ops(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> tuple[Workspace, str, tuple[str, str]]:
    """The finished run, plus two declared-effect snapshots taken on its branch."""
    ws, _, _ = run
    snapshot_operation(ws.root, "op-a")
    snapshot_operation(ws.root, "op-b")
    assert len(_ops_refs(repo, str(ws.spec.branch))) == 2
    return run


@pytest.mark.parametrize(
    "choice", [GitHandoffChoice.APPLY, GitHandoffChoice.SQUASH, GitHandoffChoice.DISCARD]
)
def test_no_handoff_choice_deletes_a_snapshot_ref(
    repo: Path,
    run_with_ops: tuple[Workspace, str, tuple[str, str]],
    choice: GitHandoffChoice,
) -> None:
    """D162 decision 7, amending D107: an undo needs the refs after the handoff, so they live
    until retention removes them (`workspace.sweep_snapshots`). Before T-5.11, `apply`,
    `squash` and `discard` deleted them here."""
    ws, base, _ = run_with_ops
    finalize(repo, ws.spec, choice, operation_id=OP, base=base)
    assert len(_ops_refs(repo, str(ws.spec.branch))) == 2


def test_keep_leaves_the_refs_with_the_branch_they_explain(
    repo: Path, run_with_ops: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run_with_ops
    finalize(repo, ws.spec, GitHandoffChoice.KEEP, operation_id=OP, base=base)
    assert len(_ops_refs(repo, str(ws.spec.branch))) == 2


@pytest.mark.writes_worktree("drift.py")
def test_a_refused_apply_leaves_the_refs(
    repo: Path, run_with_ops: tuple[Workspace, str, tuple[str, str]]
) -> None:
    """Drift refuses the apply before any git work, so nothing is retired."""
    ws, base, _ = run_with_ops
    commit_in(repo, "drift.py", "moved\n")
    with pytest.raises(HandoffError):
        finalize(repo, ws.spec, GitHandoffChoice.APPLY, operation_id=OP, base=base)
    assert len(_ops_refs(repo, str(ws.spec.branch))) == 2


def test_another_sessions_refs_are_not_this_handoffs_to_retire(
    repo: Path, run_with_ops: tuple[Workspace, str, tuple[str, str]]
) -> None:
    ws, base, _ = run_with_ops
    other = "refs/chickadee/ops/some-other-branch/op-z"
    git(repo, "update-ref", other, git(repo, "rev-parse", "HEAD"))
    finalize(repo, ws.spec, GitHandoffChoice.DISCARD, operation_id=OP, base=base)
    assert git(repo, "for-each-ref", "--format=%(refname)", other) == other


def test_a_squash_message_lists_each_commit(
    repo: Path, run: tuple[Workspace, str, tuple[str, str]]
) -> None:
    """D204 decision 3 (was: a count and the range, pinned before T-3.166 part 4): each commit
    of the run by its subject, so none is reduced to a number; the trailer last."""
    ws, base, _ = run
    finalize(repo, ws.spec, GitHandoffChoice.SQUASH, operation_id=OP, base=base)
    assert git(repo, "log", "-1", "--format=%B") == (
        f"chickadee: 2 commits from {ws.spec.branch}\n\n"
        f"- chickadee: one.py\n- chickadee: two.py\n\n{TRAILER}: {OP}"
    )


def test_a_squash_of_three_tasks_keeps_each_goal_and_what_ran() -> None:
    """A task's `Goal` and `Ran` lines, a continuation line with them; its
    other lines (`Done when`, `Reviewed`, the trailer) left in the branch's commits."""
    from chickadee.handoff import squash_message

    def task(n: int, ran: str) -> tuple[str, str]:
        body = (
            f"Goal: T{n} done.\nDone when: it passes.\nReviewed: accepted by Athena, round 1\n"
            f"Ran: {ran}\n\nChickadee-Operation: task-commit:T{n}:1"
        )
        return (f"chickadee: T{n} task {n}", body)

    got = squash_message(
        "chickadee/b",
        [task(1, "nothing"), task(2, "`pytest` (exit 0)"), task(3, "`make` (exit 2)")],
        "op-1",
    )
    assert got == (
        "chickadee: 3 commits from chickadee/b\n\n"
        "- chickadee: T1 task 1\n  Goal: T1 done.\n  Ran: nothing\n"
        "- chickadee: T2 task 2\n  Goal: T2 done.\n  Ran: `pytest` (exit 0)\n"
        "- chickadee: T3 task 3\n  Goal: T3 done.\n  Ran: `make` (exit 2)\n\n"
        f"{TRAILER}: op-1\n"
    )
