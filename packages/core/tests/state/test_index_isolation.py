"""ST-3: the graph index is checkout-local, so a worktree cannot corrupt the main tree's.

This is the half of an earlier security review that **bites**. The other half — no
document saying where `.chickadee/` lives — was merely undefined. This one is a data loss:
row 18a reindexes after every packet commit, and the default isolation mode is a linked
worktree, so an index keyed to the *repository* means every isolated run rewrites the index
the person is working against.
"""

from __future__ import annotations

import pathlib

import pytest
from repos import git, init_repo

from chickadee.graph import store
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def clone(tmp_path: pathlib.Path) -> pathlib.Path:
    return init_repo(tmp_path / "main", {"app.py": "def go():\n    return 1\n"})


def test_the_index_paths_differ_between_trees(
    clone: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))
    assert store.db_path(clone) != store.db_path(wt)


def test_indexing_a_worktree_leaves_the_main_index_byte_identical(
    clone: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The assertion finding 6 is really about, and it is about BYTES rather than paths:
    a test that only compared locations would pass against an implementation that wrote
    both."""
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))

    store.connect(clone).close()
    main_db = store.db_path(clone)
    before = main_db.read_bytes()

    (wt / "app.py").write_text("def go():\n    return 2\n", encoding="utf-8")
    store.connect(wt).close()

    assert main_db.read_bytes() == before
    assert store.db_path(wt).exists()


def test_both_indexes_live_inside_the_git_directory(clone: pathlib.Path) -> None:
    """So neither needs an exclude line, and neither shows up in the user's diffs."""
    assert ".git" in store.db_path(clone).parts


def test_a_directory_that_is_not_a_checkout_still_gets_an_index(
    tmp_path: pathlib.Path,
) -> None:
    """ "Checkout-local" presupposes a checkout. A plain directory has none — and no worktree
    for a stray index to pollute, which is the whole hazard ST-3 removes."""
    plain = tmp_path / "plain"
    plain.mkdir()
    store.connect(plain).close()
    assert store.db_path(plain).exists()


def test_journals_go_the_other_way(clone: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Two roots, two questions. History belongs to the CLONE — one session must not split
    across worktrees — while the index belongs to the TREE it describes. Answering both the
    same way is what the single `.chickadee/` did."""
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", "-b", "side", str(wt))
    assert resolve_roots(clone).repo_local == resolve_roots(wt).repo_local
    assert resolve_roots(clone).checkout_local != resolve_roots(wt).checkout_local
