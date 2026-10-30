"""A fresh worktree has no index, builds one, and can say so before it does.

The second horn of an earlier security review. If `.chickadee/` had been resolved
per worktree with no other change, a run in a new worktree could not start: no config, no
index, and `24-init.md` §3 is explicit that the index build is the slow step which exists as
a separate command precisely so a run does not do it silently.

Under ST-3 it is a stated event instead: the checkout has no index, that is knowable before
the pause, and building one is announced rather than discovered.
"""

from __future__ import annotations

import pathlib

import pytest
from repos import git, init_repo

from chickadee.graph import store
from chickadee.graph.indexer import Indexer


def build(checkout: pathlib.Path) -> None:
    """An index is a completed pass, not a file `connect` created."""
    indexer = Indexer(checkout)
    try:
        indexer.full_index()
    finally:
        indexer.close()


@pytest.fixture
def clone(tmp_path: pathlib.Path) -> pathlib.Path:
    return init_repo(tmp_path / "main", {"app.py": "def go():\n    return 1\n"})


def test_a_fresh_worktree_reports_no_index(clone: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Knowable BEFORE the slow step, which is what lets a surface announce it."""
    build(clone)
    assert store.index_exists(clone) is True

    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))
    assert store.index_exists(wt) is False


def test_it_builds_one_rather_than_failing(clone: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The alternative horn: a run in a fresh worktree simply could not start."""
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))
    build(wt)
    assert store.index_exists(wt) is True


def test_the_main_index_is_not_borrowed(clone: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """The failure this guards is the quiet one: reading ANOTHER tree's index looks like it
    works, and is stale about a checkout no generation counter describes."""
    build(clone)
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))
    assert store.index_exists(wt) is False, "a fresh worktree must not see the main index"


def test_removing_a_worktree_takes_its_index_with_it(
    clone: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The lifetime that falls out of putting it in the worktree's git dir: `git worktree
    remove` cleans up the index for free, and leaves repo-local history alone."""
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))
    store.connect(wt).close()
    wt_db = store.db_path(wt)
    assert wt_db.exists()

    git(clone, "worktree", "remove", "--force", str(wt))
    assert not wt_db.exists()
