"""T-2.1 — repository lease, worktree isolation, in-place snapshots.

The claims worth testing here are the ones that are invisible when they break: a lease that
does not actually exclude another *process*, a snapshot that `gc` can collect, and a
worktree failure that quietly continues in the user's checkout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.config import WorkspacePolicy
from chickadee.errors import ConfigError, InfraError
from chickadee.init.detect import resolve
from chickadee.workspace import (
    FIRST_COMMIT_MESSAGE,
    OPS_REF_PREFIX,
    SNAPSHOT_REF_PREFIX,
    acquire,
    ops_ref,
    restore_operation,
    run_branch_name,
    snapshot,
    snapshot_operation,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"kept.txt": "original\n"}, commit="first")


@pytest.fixture
def empty_repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "empty", commit=None)


# --- worktree is the default -----------------------------------------------


def test_the_default_is_a_dedicated_worktree(repo: Path) -> None:
    with acquire(repo, run_id="r-0000abcd", goal="add session refresh") as ws:
        assert ws.spec.mode == "worktree"
        assert ws.isolated
        assert ws.root != ws.source_root
        assert ws.root.is_dir()
        # A linked worktree has a `.git` *file* pointing at the common dir, not a directory.
        assert (ws.root / ".git").is_file()
        assert ws.spec.branch == "chickadee/add-session-refresh-0000abcd"


def test_a_worktree_run_reports_no_degradation(repo: Path) -> None:
    with acquire(repo, run_id="r-0000abcd", goal="g") as ws:
        assert ws.degraded == ()


def test_writes_in_the_worktree_do_not_touch_the_source_checkout(repo: Path) -> None:
    with acquire(repo, run_id="r-0000abcd", goal="g") as ws:
        (ws.root / "kept.txt").write_text("changed by the run\n", encoding="utf-8")
        (ws.root / "new.txt").write_text("new\n", encoding="utf-8")

    assert (repo / "kept.txt").read_text(encoding="utf-8") == "original\n"
    assert not (repo / "new.txt").exists()
    assert git(repo, "status", "--porcelain") == ""


def test_an_empty_repository_gets_a_first_commit_and_still_a_worktree(empty_repo: Path) -> None:
    """D202 decision 4 moved this answer: no commits was refused ("nothing to branch from"),
    and the owner's run had Zeus make one itself. The harness makes an empty first commit,
    says why in its message, and branches the worktree from it: never in place."""
    with acquire(empty_repo, run_id="r-0000abcd", goal="g") as ws:
        assert ws.spec.mode == "worktree" and ws.root != ws.source_root
    assert git(empty_repo, "log", "--format=%s").splitlines() == [FIRST_COMMIT_MESSAGE]
    assert git(empty_repo, "ls-tree", "HEAD") == "", "the first commit holds nothing"
    assert git(empty_repo, "log", "--format=%an", "-1") == "chickadee", "the commit is ours"


@pytest.mark.writes_worktree("notes.txt")
def test_the_first_commit_moves_no_file_and_no_index_entry(empty_repo: Path) -> None:
    """Files the person has not committed stay as they were: untracked, in their checkout."""
    (empty_repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    with acquire(empty_repo, run_id="r-0000abcd", goal="g") as ws:
        assert not (ws.root / "notes.txt").exists()
    assert git(empty_repo, "status", "--porcelain") == "?? notes.txt"


@pytest.mark.writes_worktree("notes.txt")
def test_a_session_in_an_empty_repository_with_files_is_seeded(empty_repo: Path) -> None:
    """Was: a dirty checkout, refused (D188 decision 4.1). D230 Amendment 2 decision 4 moves
    it: the empty first commit, then the seed holding the person's files on top of it; their
    checkout keeps the files untracked."""
    (empty_repo / "notes.txt").write_text("mine\n", encoding="utf-8")
    with acquire(empty_repo, run_id="r-0000abcd", goal="g", session="S-1") as ws:
        assert (ws.root / "notes.txt").read_text() == "mine\n"
        first = git(empty_repo, "rev-parse", "HEAD")
        assert git(ws.root, "rev-parse", "HEAD~1") == first
    assert git(empty_repo, "log", "--format=%s").splitlines() == [FIRST_COMMIT_MESSAGE]
    assert git(empty_repo, "status", "--porcelain") == "?? notes.txt"


def test_a_failed_worktree_never_falls_back_to_in_place(repo: Path) -> None:
    """The whole point of D39: reduced isolation is chosen, never arrived at."""
    branch = run_branch_name("g", "r-0000abcd")
    git(repo, "branch", branch)  # the name is taken, so `worktree add -b` must fail

    with pytest.raises(InfraError):
        acquire(repo, run_id="r-0000abcd", goal="g")

    # and the failure released the lease rather than stranding the repository
    with acquire(repo, run_id="r-0000beef", goal="g") as ws:
        assert ws.spec.mode == "worktree"


def test_a_failed_worktree_leaves_no_registration_behind(repo: Path) -> None:
    """Removing the directory is not enough: a half-registered worktree makes the *next*
    run fail with "already registered" instead of the real cause."""
    branch = run_branch_name("g", "r-0000abcd")
    git(repo, "branch", branch)
    with pytest.raises(InfraError):
        acquire(repo, run_id="r-0000abcd", goal="g")

    listed = git(repo, "worktree", "list", "--porcelain")
    assert "r-0000abcd" not in listed, f"stale worktree registration survived:\n{listed}"

    # The point of pruning is that the *retry* reports the real cause. Asserting only that
    # the registration is gone leaves that untested.
    with pytest.raises(InfraError) as retry:
        acquire(repo, run_id="r-0000abcd", goal="g")
    message = str(retry.value)
    # The retry must report the *original* cause — the branch name being taken — not
    # leftover state from the first attempt. Matched on the distinctive phrases of the two
    # wrong answers, because git's own "fatal: <branch> already exists" legitimately
    # contains "already exists" and a looser assertion fails on the correct message.
    assert "already registered" not in message
    assert "a worktree for run" not in message
    assert branch in message


def test_an_existing_worktree_path_is_refused_not_deleted(repo: Path) -> None:
    """A directory here belongs to an earlier run and may hold unsaved work. The cleanup
    path used to delete it, destroying data this run never created."""
    from chickadee.workspace import WORKTREES_DIR

    identity = resolve(repo)
    victim = identity.git_common_dir / WORKTREES_DIR / "r-0000abcd"
    victim.mkdir(parents=True)
    (victim / "unsaved.txt").write_text("precious\n", encoding="utf-8")

    with pytest.raises(InfraError, match="already exists"):
        acquire(repo, run_id="r-0000abcd", goal="g")

    assert (victim / "unsaved.txt").read_text(encoding="utf-8") == "precious\n"


# --- in place by default ----------


IN_PLACE = WorkspacePolicy(mode="in_place")


def test_in_place_is_the_configured_default_and_needs_no_asking(repo: Path) -> None:
    """D230 decision 3. Was: a policy alone never put a run in the person's checkout, and an
    unasked in-place policy was refused."""
    assert WorkspacePolicy().mode == "in_place"
    with acquire(repo, run_id="r-0000abcd", goal="g", policy=WorkspacePolicy()) as ws:
        assert ws.spec.mode == "in_place"


def test_no_policy_is_a_worktree(repo: Path) -> None:
    """A host's own request and a fork's child are no person's conversation: with no policy,
    `acquire` makes a worktree, as `CoderRequest` defaults to one."""
    with acquire(repo, run_id="r-0000abcd", goal="g") as ws:
        assert ws.spec.mode == "worktree"


def test_in_place_is_accepted_for_any_job(repo: Path) -> None:
    """D162 decision 10 lifted D39's quick-only bound."""
    with acquire(repo, run_id="r-0000abcd", goal="g", policy=IN_PLACE) as ws:
        assert ws.spec.mode == "in_place"
        assert ws.root == ws.source_root
        assert not ws.isolated


def test_in_place_reports_its_reduced_isolation(repo: Path) -> None:
    with acquire(repo, run_id="r-0000abcd", goal="g", policy=IN_PLACE) as ws:
        assert ws.degraded, "in-place must say it is less isolated"
        assert any("writes directly into your checkout" in d for d in ws.degraded)
        sha = ws.spec.dirty_snapshot_sha
        assert sha is not None  # contract 11 guarantees it for in_place
        assert any(sha[:12] in d for d in ws.degraded)


def test_require_lease_false_is_refused_not_ignored(repo: Path) -> None:
    policy = WorkspacePolicy(require_lease=False)
    with pytest.raises(ConfigError, match="require_lease"):
        acquire(repo, run_id="r-0000abcd", goal="g", policy=policy)


# --- the snapshot ----------------------------------------------------------


@pytest.mark.writes_worktree("kept.txt", "staged.txt", "loose.txt")
def test_the_snapshot_captures_staged_unstaged_and_untracked(repo: Path) -> None:
    (repo / "kept.txt").write_text("unstaged edit\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    git(repo, "add", "staged.txt")
    (repo / "loose.txt").write_text("untracked\n", encoding="utf-8")

    _, dirty = snapshot(repo, "r-0000abcd")
    listed = git(repo, "ls-tree", "-r", "--name-only", dirty).split("\n")
    assert set(listed) == {"kept.txt", "staged.txt", "loose.txt"}
    assert git(repo, "show", f"{dirty}:kept.txt") == "unstaged edit"
    assert git(repo, "show", f"{dirty}:loose.txt") == "untracked"


@pytest.mark.writes_worktree("kept.txt", "staged.txt")
def test_snapshotting_leaves_the_working_tree_and_index_alone(repo: Path) -> None:
    """`git stash push` would have moved their work. This must not."""
    (repo / "kept.txt").write_text("unstaged edit\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    git(repo, "add", "staged.txt")
    before = git(repo, "status", "--porcelain")

    snapshot(repo, "r-0000abcd")

    assert git(repo, "status", "--porcelain") == before
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "unstaged edit\n"


@pytest.mark.writes_worktree("loose.txt")
def test_the_snapshot_does_not_need_an_ambient_git_identity(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI usually has no `user.name`, and `commit-tree` refuses without one.

    Deliberately tested by *removing* the identity rather than by configuring it in the
    fixture: a fixture that sets `user.name` hides this failure instead of catching it,
    which is how the defect reached review in the first place.
    """
    for var in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    git(repo, "config", "--unset-all", "user.name")
    git(repo, "config", "--unset-all", "user.email")
    (repo / "loose.txt").write_text("work\n", encoding="utf-8")

    _, dirty = snapshot(repo, "r-0000abcd")
    assert git(repo, "show", f"{dirty}:loose.txt") == "work"


@pytest.mark.writes_worktree("loose.txt")
def test_the_snapshot_survives_garbage_collection(repo: Path) -> None:
    """A snapshot with no ref is unreachable, and unreachable is what `gc` collects."""
    (repo / "loose.txt").write_text("untracked\n", encoding="utf-8")
    _, dirty = snapshot(repo, "r-0000abcd")

    assert git(repo, "rev-parse", f"{SNAPSHOT_REF_PREFIX}/r-0000abcd") == dirty
    git(repo, "gc", "--prune=now", "--quiet")
    assert git(repo, "cat-file", "-t", dirty) == "commit"
    assert git(repo, "show", f"{dirty}:loose.txt") == "untracked"


@pytest.mark.writes_worktree("kept.txt", "loose.txt")
def test_the_snapshot_is_restorable(repo: Path) -> None:
    """The point of the snapshot is getting the work back, so test that, not its shape."""
    (repo / "kept.txt").write_text("user work\n", encoding="utf-8")
    (repo / "loose.txt").write_text("user file\n", encoding="utf-8")
    _, dirty = snapshot(repo, "r-0000abcd")

    (repo / "kept.txt").write_text("destroyed by the run\n", encoding="utf-8")
    (repo / "loose.txt").unlink()

    git(repo, "checkout", dirty, "--", ".")
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "user work\n"
    assert (repo / "loose.txt").read_text(encoding="utf-8") == "user file\n"


@pytest.mark.writes_worktree("loose.txt")
def test_a_snapshot_does_not_swallow_a_previous_runs_worktree(repo: Path) -> None:
    """Regression: worktrees under `.chickadee/` were inside the tree `git add -A` walks.

    An in-place run following a worktree run would have snapshotted a whole second copy of
    the repository as "the user's pre-existing changes".
    """
    with acquire(repo, run_id="r-0000abcd", goal="earlier run"):
        pass
    (repo / "loose.txt").write_text("the only untracked file\n", encoding="utf-8")

    _, dirty = snapshot(repo, "r-0000beef")
    listed = git(repo, "ls-tree", "-r", "--name-only", dirty).split("\n")
    assert set(listed) == {"kept.txt", "loose.txt"}


@pytest.mark.writes_worktree("loose.txt")
def test_an_empty_repository_snapshots_against_the_empty_tree(empty_repo: Path) -> None:
    (empty_repo / "loose.txt").write_text("first file\n", encoding="utf-8")
    pre, dirty = snapshot(empty_repo, "r-0000abcd")
    assert pre == "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
    assert git(empty_repo, "show", f"{dirty}:loose.txt") == "first file"


# --- the lease -------------------------------------------------------------


LEASE_PROBE = """
import sys
from pathlib import Path
from chickadee.errors import RepoLeaseBusy
from chickadee.workspace import acquire

try:
    acquire(Path(sys.argv[1]), run_id="r-0000beef", goal="second")
except RepoLeaseBusy:
    print("BUSY")
else:
    print("ACQUIRED")
"""


def _second_process(repo: Path) -> str:
    out = subprocess.run(
        [sys.executable, "-c", LEASE_PROBE, str(repo)], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_the_lease_excludes_another_process(repo: Path) -> None:
    """In-process locking would pass a same-process test and still let two engines run."""
    with acquire(repo, run_id="r-0000abcd", goal="first"):
        assert _second_process(repo) == "BUSY"


def test_releasing_the_lease_lets_the_next_process_in(repo: Path) -> None:
    with acquire(repo, run_id="r-0000abcd", goal="first"):
        pass
    assert _second_process(repo) == "ACQUIRED"


def test_a_linked_worktree_contends_for_the_same_lease(repo: Path) -> None:
    """One repository, one lease — a lease per checkout would exclude nobody."""
    linked = repo.parent / "linked"
    git(repo, "worktree", "add", "-q", str(linked), "-b", "side")
    with acquire(repo, run_id="r-0000abcd", goal="first"):
        assert _second_process(linked) == "BUSY"


# --- branch naming ---------------------------------------------------------


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("Add session refresh endpoint", "chickadee/add-session-refresh-endpoint-0000abcd"),
        ("  Fix   THE bug!!  ", "chickadee/fix-the-bug-0000abcd"),
        ("", "chickadee/run-0000abcd"),
        ("!!!", "chickadee/run-0000abcd"),
    ],
)
def test_branch_names_are_descriptive(goal: str, expected: str) -> None:
    assert run_branch_name(goal, "r-0000abcd") == expected


def test_a_long_goal_is_cut_on_a_word_boundary() -> None:
    name = run_branch_name("refactor the authentication middleware entirely now", "r-0000abcd")
    slug = name.removeprefix("chickadee/").removesuffix("-0000abcd")
    assert len(slug) <= 40
    assert not slug.endswith("-")
    assert slug == "refactor-the-authentication-middleware"


def test_a_single_long_word_still_produces_a_branch() -> None:
    name = run_branch_name("x" * 80, "r-0000abcd")
    assert name == f"chickadee/{'x' * 40}-0000abcd"


# --- D107 / T-2.28: a declared effect is an undo unit --------------------------------


def _op_ref(repo: Path, operation_id: str) -> str:
    return f"{OPS_REF_PREFIX}/{git(repo, 'symbolic-ref', '--short', 'HEAD')}/{operation_id}"


@pytest.mark.writes_worktree("kept.txt", "staged.txt", "loose.txt")
def test_snapshot_operation_parks_the_tree_under_a_ref_and_leaves_the_index_alone(
    repo: Path,
) -> None:
    (repo / "kept.txt").write_text("edited but unstaged\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    git(repo, "add", "staged.txt")
    (repo / "loose.txt").write_text("untracked\n", encoding="utf-8")
    index_before = git(repo, "ls-files", "--stage")

    tree = snapshot_operation(repo, "op-1")

    assert git(repo, "rev-parse", f"{_op_ref(repo, 'op-1')}^{{tree}}") == tree
    # Untracked files are in the tree — an operation that overwrites one must be undoable.
    assert "loose.txt" in git(repo, "ls-tree", "--name-only", tree)
    assert git(repo, "ls-files", "--stage") == index_before, "the user's index is untouched"
    assert git(repo, "status", "--porcelain").count("\n") >= 2, "the working tree is untouched"


def test_snapshot_operation_refuses_a_second_snapshot_for_the_same_id(repo: Path) -> None:
    snapshot_operation(repo, "op-2")
    with pytest.raises(InfraError, match="already has a snapshot"):
        snapshot_operation(repo, "op-2")


@pytest.mark.writes_worktree("kept.txt", "loose.txt", "made.txt", "undeclared.txt")
def test_restore_operation_restores_declared_paths_and_deletes_created_ones(
    repo: Path,
) -> None:
    (repo / "loose.txt").write_text("untracked before\n", encoding="utf-8")
    snapshot_operation(repo, "op-3")

    # "The operation": edits a tracked file, an untracked one, creates a new one, and
    # touches a file it did not declare.
    (repo / "kept.txt").write_text("changed by the operation\n", encoding="utf-8")
    (repo / "loose.txt").write_text("changed too\n", encoding="utf-8")
    (repo / "made.txt").write_text("created by the operation\n", encoding="utf-8")
    (repo / "undeclared.txt").write_text("not the operation's to undo\n", encoding="utf-8")
    index_before = git(repo, "ls-files", "--stage")

    touched = restore_operation(repo, "op-3", ("kept.txt", "loose.txt", "made.txt"))

    assert set(touched) == {"kept.txt", "loose.txt", "made.txt"}
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "original\n"
    assert (repo / "loose.txt").read_text(encoding="utf-8") == "untracked before\n"
    assert not (repo / "made.txt").exists(), "created by the operation, so deleted"
    assert (repo / "undeclared.txt").exists(), "bounded to the declared surface"
    assert git(repo, "ls-files", "--stage") == index_before, "restore never touches the index"


def test_restore_operation_refuses_a_path_outside_the_worktree(repo: Path) -> None:
    snapshot_operation(repo, "op-4")
    with pytest.raises(InfraError, match="outside the worktree"):
        restore_operation(repo, "op-4", ("../elsewhere.txt",))


def test_restore_operation_refuses_an_unknown_operation(repo: Path) -> None:
    with pytest.raises(InfraError, match="has no snapshot"):
        restore_operation(repo, "never-taken", ("kept.txt",))


def test_operation_refs_survive_gc(repo: Path) -> None:
    tree = snapshot_operation(repo, "op-5")
    git(repo, "gc", "-q", "--prune=now")
    assert git(repo, "cat-file", "-t", tree) == "tree"


# --- an operation id is not a ref name until it is encoded -----------------------


@pytest.mark.parametrize(
    "operation_id",
    ["bootstrap:run-01JABCDEF", "verify:P-1:A3:status", "verify:A1:src/app.py"],
    ids=["bootstrap", "gate", "gate-with-path"],
)
@pytest.mark.writes_worktree("kept.txt")
def test_an_operation_id_in_this_codebases_own_shape_snapshots(
    repo: Path, operation_id: str
) -> None:
    """`<kind>:<id>` is the convention everywhere — `engine.bootstrap_worktree` mints
    `bootstrap:<run_id>`, the A-gates mint `verify:...` — and `:` is a character git
    refuses in a ref name. Interpolating it raw failed on every real caller."""
    tree = snapshot_operation(repo, operation_id)
    assert git(repo, "rev-parse", f"{ops_ref(repo, operation_id)}^{{tree}}") == tree

    (repo / "kept.txt").write_text("mutated by the effect\n", encoding="utf-8")
    assert restore_operation(repo, operation_id, ("kept.txt",)) == ("kept.txt",)
    assert (repo / "kept.txt").read_text(encoding="utf-8") == "original\n"


def test_two_operation_ids_never_share_a_ref(repo: Path) -> None:
    """Sanitising `:` to `-` would alias `a:b` onto `a-b`, and an undo unit that restores
    another operation's snapshot is worse than the error it replaced."""
    assert ops_ref(repo, "a:b") != ops_ref(repo, "a-b")
    snapshot_operation(repo, "a:b")
    snapshot_operation(repo, "a-b")  # refused as a duplicate if the two collided


def test_the_encoded_ref_is_one_git_accepts(repo: Path) -> None:
    ref = ops_ref(repo, "verify:P-1:A3:status")
    git(repo, "check-ref-format", ref)  # raises if git refuses the name


def test_an_empty_operation_id_is_refused(repo: Path) -> None:
    with pytest.raises(InfraError, match="cannot name a ref"):
        snapshot_operation(repo, "")


# --- a declared directory is restored exactly, contents included --------------


@pytest.mark.writes_worktree(
    "pkg",
    "pkg/sub",
    "pkg/kept.py",
    "pkg/new.py",
    "pkg/gone.py",
    "pkg/sub/deep.py",
    "pkg/sub/also-new.py",
)
def test_restoring_a_directory_removes_what_the_operation_created(repo: Path) -> None:
    """`git restore` only puts back what the tree holds, so on its own it left every file
    the operation *created* inside a declared directory in place — a directory that is
    neither its before state nor its after. The snapshot is a full `write-tree` including
    untracked files, so it already knows which paths existed; the sweep asks it."""
    pkg = repo / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "kept.py").write_text("original\n", encoding="utf-8")
    (pkg / "gone.py").write_text("deleted by the effect\n", encoding="utf-8")
    (pkg / "sub" / "deep.py").write_text("nested original\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "pkg")

    snapshot_operation(repo, "op:dir")
    (pkg / "kept.py").write_text("changed\n", encoding="utf-8")
    (pkg / "gone.py").unlink()
    (pkg / "new.py").write_text("created by the effect\n", encoding="utf-8")
    (pkg / "sub" / "also-new.py").write_text("created, nested\n", encoding="utf-8")

    assert restore_operation(repo, "op:dir", ("pkg",)) == ("pkg",)

    assert (pkg / "kept.py").read_text(encoding="utf-8") == "original\n", "edit undone"
    assert (pkg / "gone.py").read_text(encoding="utf-8") == "deleted by the effect\n", (
        "a file the effect deleted comes back"
    )
    assert (pkg / "sub" / "deep.py").read_text(encoding="utf-8") == "nested original\n"
    assert not (pkg / "new.py").exists(), "a file the effect created is removed"
    assert not (pkg / "sub" / "also-new.py").exists(), "...at any depth"


@pytest.mark.writes_worktree("pkg", "pkg/kept.py", "pkg/untracked.txt", "pkg/new.py")
def test_the_sweep_keeps_a_file_that_was_untracked_but_present(repo: Path) -> None:
    """The snapshot stages with `git add -A`, so a file never committed is still in the
    tree and is therefore something that existed — it is kept, not swept."""
    pkg = repo / "pkg"
    pkg.mkdir()
    (pkg / "kept.py").write_text("original\n", encoding="utf-8")
    (pkg / "untracked.txt").write_text("never committed\n", encoding="utf-8")

    snapshot_operation(repo, "op:untracked")
    (pkg / "new.py").write_text("created by the effect\n", encoding="utf-8")

    restore_operation(repo, "op:untracked", ("pkg",))
    assert (pkg / "untracked.txt").read_text(encoding="utf-8") == "never committed\n"
    assert not (pkg / "new.py").exists()


@pytest.mark.writes_worktree(
    "pkg", "pkg/.venv", "pkg/kept.py", "pkg/new.py", "pkg/.venv/site.py"
)
def test_the_sweep_never_deletes_under_a_directory_snapshots_leave_out(repo: Path) -> None:
    """A snapshot holds no untracked file under `.venv` or `__pycache__`, so it
    cannot prove one was not there; restoring a directory around them keeps them."""
    pkg = repo / "pkg"
    (pkg / ".venv").mkdir(parents=True)
    (pkg / "kept.py").write_text("original\n", encoding="utf-8")
    (pkg / ".venv" / "site.py").write_text("the person's environment\n", encoding="utf-8")

    snapshot_operation(repo, "op:venv")
    (pkg / "new.py").write_text("created by the effect\n", encoding="utf-8")

    restore_operation(repo, "op:venv", ("pkg",))
    assert (pkg / ".venv" / "site.py").read_text(encoding="utf-8") == (
        "the person's environment\n"
    )
    assert not (pkg / "new.py").exists()


@pytest.mark.writes_worktree("pkg", "pkg/kept.py", "outside.txt")
def test_a_declared_file_beside_a_directory_is_unaffected(repo: Path) -> None:
    """The sweep is bounded to the declared directory; a sibling is not its business."""
    pkg = repo / "pkg"
    pkg.mkdir()
    (pkg / "kept.py").write_text("original\n", encoding="utf-8")
    snapshot_operation(repo, "op:bounded")
    (repo / "outside.txt").write_text("made meanwhile\n", encoding="utf-8")

    restore_operation(repo, "op:bounded", ("pkg",))
    assert (repo / "outside.txt").exists(), "a path nobody declared is never touched"


def test_restoring_the_worktree_root_is_refused(repo: Path) -> None:
    """Sweeping under the root would delete every file created since the snapshot, the
    person's own work included — `/undo <operation>` becoming `reset the repository`."""
    snapshot_operation(repo, "op:root")
    for rel in (".", ""):
        with pytest.raises(InfraError, match="worktree root"):
            restore_operation(repo, "op:root", (rel,))
