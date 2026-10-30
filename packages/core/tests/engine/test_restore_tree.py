"""One restore for every undo, and the plan of what an undo would restore and ask about:
`restore_tree`, `changed_paths`, `harness_paths`, `plan_undo`."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.errors import InfraError
from chickadee.workspace import (
    IN_PLACE_KEY,
    SnapshotSite,
    TurnTrees,
    changed_paths,
    harness_paths,
    plan_undo,
    restore_tree,
    snapshot,
    snapshot_turn,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "chickadee"
BOUND = 1024


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {"a.py": "a = 1\n", "b.py": "b = 1\n", "pkg/x.py": "x = 1\n"},
    )


def tree_of(repo: Path, turn: int, phase: str) -> str:
    site = SnapshotSite(repo, "in_place", IN_PLACE_KEY)
    return snapshot_turn(
        site,
        session_id="S",
        turn=turn,
        phase=phase,  # type: ignore[arg-type]
        untracked_max_bytes=BOUND,
    ).tree


def write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    if not path.parent.is_dir():
        path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")


# -- restore_tree -------------------------------------------------------------------------


@pytest.mark.writes_worktree("a.py", "made.py", "pkg/new.py")
def test_restore_tree_puts_back_what_the_tree_holds_and_deletes_what_it_lacks(
    repo: Path,
) -> None:
    before = tree_of(repo, 1, "before")
    write(repo, "a.py", "a = 2\n")
    write(repo, "made.py", "new\n")
    write(repo, "pkg/new.py", "new\n")
    touched = restore_tree(repo, before, ("a.py", "made.py", "pkg"))
    assert touched == ("a.py", "made.py", "pkg")
    assert (repo / "a.py").read_text(encoding="utf-8") == "a = 1\n"
    assert not (repo / "made.py").exists()
    assert not (repo / "pkg" / "new.py").exists(), "a directory is restored exactly"
    assert (repo / "pkg" / "x.py").exists()


@pytest.mark.writes_worktree("b.py")
def test_restore_tree_touches_only_the_paths_it_is_given(repo: Path) -> None:
    before = tree_of(repo, 1, "before")
    write(repo, "b.py", "the person's\n")
    restore_tree(repo, before, ("a.py",))
    assert (repo / "b.py").read_text(encoding="utf-8") == "the person's\n"


@pytest.mark.parametrize("rel", ["../elsewhere.py", "."], ids=["outside", "root"])
def test_restore_tree_refuses_outside_the_worktree_and_the_root(repo: Path, rel: str) -> None:
    before = tree_of(repo, 1, "before")
    with pytest.raises(InfraError, match="refusing to restore it"):
        restore_tree(repo, before, (rel,))


# -- changed_paths --------------------------------------------------------------------------


@pytest.mark.writes_worktree("a.py", "c.py", "b.py", "renamed.py")
def test_changed_paths_names_every_path_that_differs_both_ends_of_a_rename(repo: Path) -> None:
    before = tree_of(repo, 1, "before")
    assert changed_paths(repo, before, before) == ()
    write(repo, "a.py", "a = 2\n")
    write(repo, "c.py", "c = 1\n")
    (repo / "b.py").rename(repo / "renamed.py")
    after = tree_of(repo, 1, "after")
    assert changed_paths(repo, before, after) == ("a.py", "b.py", "c.py", "renamed.py")


def test_changed_paths_is_the_only_tree_diff_under_chickadee() -> None:
    """Every reader asks it rather than diffing two trees itself. The one
    `diff-tree` is `workspace._diff_tree`, which `changed_paths` and the task facts'
    `diff_stats` (its counts) share."""
    where: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                for node in ast.walk(fn):
                    if isinstance(node, ast.Constant) and node.value == "diff-tree":
                        where.append(f"{path.relative_to(SRC)}:{fn.name}")
    assert sorted(set(where)) == ["workspace.py:_diff_tree"]


# -- plan_undo ------------------------------------------------------------------------------


@pytest.fixture
def recorded(repo: Path) -> list[TurnTrees]:
    """Two turns that wrote, with the person editing between them and after the last."""
    b1 = tree_of(repo, 1, "before")
    write(repo, "a.py", "a = 2  # harness, turn 1\n")
    a1 = tree_of(repo, 1, "after")
    write(repo, "b.py", "b = 2  # the person, between turns\n")
    b2 = tree_of(repo, 2, "before")
    write(repo, "c.py", "c = 1  # harness, turn 2\n")
    a2 = tree_of(repo, 2, "after")
    write(repo, "a.py", "a = 3  # the person, after the last turn\n")
    write(repo, "d.py", "d = 1  # the person, after the last turn\n")
    return [TurnTrees(1, b1, a1), TurnTrees(2, b2, a2)]


@pytest.mark.writes_worktree("a.py", "b.py", "c.py", "d.py")
def test_plan_undo_attributes_every_changed_path(repo: Path, recorded: list[TurnTrees]) -> None:
    plan = plan_undo(repo, recorded, untracked_max_bytes=BOUND)
    assert plan.target == recorded[0].before
    assert plan.restore == ("a.py", "c.py"), "what the harness wrote"
    assert plan.person == ("a.py", "b.py", "d.py"), "between the turns, and since the last"
    assert plan.conflicts == ("a.py",)
    assert harness_paths(repo, recorded) == plan.restore


@pytest.mark.writes_worktree("a.py", "b.py", "c.py", "d.py")
def test_plan_undo_writes_nothing_a_conflict_included(
    repo: Path, recorded: list[TurnTrees]
) -> None:
    files = {
        p: (repo / p).read_text(encoding="utf-8") for p in ("a.py", "b.py", "c.py", "d.py")
    }
    status = git(repo, "status", "--porcelain")
    plan = plan_undo(repo, recorded, untracked_max_bytes=BOUND)
    assert plan.conflicts == ("a.py",)
    assert {p: (repo / p).read_text(encoding="utf-8") for p in files} == files
    assert git(repo, "status", "--porcelain") == status, "the person's index is untouched"


@pytest.mark.writes_worktree("big.bin", "a.py")
def test_a_skipped_path_is_unknown_not_deleted_by_the_harness(repo: Path) -> None:
    """In place, a turn's `before` is the pre-run tree, which skips nothing; its `after`
    applies the bound. A large untracked file of the person's is in one and not the other,
    and must not read as the harness's."""
    (repo / "big.bin").write_bytes(b"x" * (BOUND * 4))
    _, dirty = snapshot(repo, "R-1")
    before = git(repo, "rev-parse", f"{dirty}^{{tree}}")
    write(repo, "a.py", "a = 2\n")
    site = SnapshotSite(repo, "in_place", IN_PLACE_KEY)
    after = snapshot_turn(
        site, session_id="S", turn=1, phase="after", untracked_max_bytes=BOUND
    )
    assert after.skipped == ("big.bin",)
    plan = plan_undo(
        repo, [TurnTrees(1, before, after.tree, after.skipped)], untracked_max_bytes=BOUND
    )
    assert plan.restore == ("a.py",)
    assert "big.bin" not in plan.person
    assert plan.skipped == ("big.bin",)


def test_plan_undo_needs_a_turn(repo: Path) -> None:
    with pytest.raises(InfraError, match="at least one"):
        plan_undo(repo, [], untracked_max_bytes=BOUND)


@pytest.mark.writes_worktree("a.py", "out.bin")
def test_a_harness_path_whose_current_state_is_unknown_is_a_conflict(repo: Path) -> None:
    """The harness wrote `out.bin`; the person then grew it past the bound, so the current tree
    cannot say what it holds. An undo must ask, never overwrite it on a guess (decision 3)."""
    before = tree_of(repo, 1, "before")
    write(repo, "out.bin", "small\n")
    after = tree_of(repo, 1, "after")
    (repo / "out.bin").write_bytes(b"x" * (BOUND * 4))
    plan = plan_undo(repo, [TurnTrees(1, before, after)], untracked_max_bytes=BOUND)
    assert "out.bin" in plan.restore
    assert plan.conflicts == ("out.bin",)
    assert plan.skipped == ("out.bin",)
