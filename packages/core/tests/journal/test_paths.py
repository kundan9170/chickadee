"""ST-2: run history is repo-local, and the layout no longer derives from a repo path.

The defect these guard (an earlier security review): every path derived from
`chickadee_root(repo) = repo / ".chickadee"`, and every caller passed a repository path —
which, under the **default** isolation mode, is a linked worktree.
"""

from __future__ import annotations

import pathlib

from repos import git, init_repo

from chickadee.journal import paths
from chickadee.state.roots import roots as resolve_roots


def test_runs_root_takes_roots_not_a_path(tmp_path: pathlib.Path) -> None:
    """The path-taking entry point is **gone**, not deprecated. While both existed a stale
    caller still compiled; now the only way to ask is with a root someone resolved."""
    init_repo(tmp_path, commit=None)
    r = resolve_roots(tmp_path)
    assert paths.runs_root(r) == r.repo_local / "runs"


def test_one_session_does_not_split_across_worktrees(tmp_path: pathlib.Path) -> None:
    """The property that matters: a journal written from a linked worktree and one written
    from the main tree are the SAME journal, because both resolve to the same repo_local."""
    main = init_repo(tmp_path / "main", {"f.txt": "x\n"}, commit="c")
    wt = tmp_path / "wt"
    git(main, "worktree", "add", "-q", "-b", "side", str(wt))

    from_main = paths.run_dir(resolve_roots(main), "r1")
    from_wt = paths.run_dir(resolve_roots(wt), "r1")
    assert from_main == from_wt


def test_the_root_is_inside_the_git_dir(tmp_path: pathlib.Path) -> None:
    """So no exclude line is needed at all — git never tracks its own directory."""
    init_repo(tmp_path, commit=None)
    assert ".git" in paths.runs_root(resolve_roots(tmp_path)).parts
