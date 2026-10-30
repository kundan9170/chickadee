"""The four scopes, and the one distinction that only a linked worktree reveals.

These tests use a **real** linked worktree, never a mock. The bug they guard
(an earlier security review) survived precisely because `--git-dir` and
`--git-common-dir` are the same directory in the main working tree, so any fixture that
stubs git reproduces the blindness instead of catching it.
"""

from __future__ import annotations

import pathlib

import pytest
from repos import git, init_repo

from chickadee.state import LOCAL_DIR, PROJECT_DIR, NotARepository, roots, xdg_config_home


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    return init_repo(tmp_path / "main", {"f.txt": "x\n"}, commit="first")


@pytest.fixture
def linked(repo: pathlib.Path, tmp_path: pathlib.Path) -> pathlib.Path:
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "-b", "side", str(wt))
    return wt


def test_main_tree_collapses_the_two_local_scopes(repo: pathlib.Path) -> None:
    """In the main tree repo_local == checkout_local — and that identity is the trap.

    Asserting it here is not redundant: it documents that the property the design relies on
    is *absent* in the only case most tests would cover.
    """
    r = roots(repo)
    assert r.repo_local == r.checkout_local
    assert r.is_linked_worktree is False


def test_linked_worktree_shares_repo_local_and_splits_checkout_local(
    repo: pathlib.Path, linked: pathlib.Path
) -> None:
    """The whole fix, in one assertion pair."""
    main_roots = roots(repo)
    wt_roots = roots(linked)

    # History and leases are one per CLONE: a session must not split across worktrees.
    assert wt_roots.repo_local == main_roots.repo_local

    # The index describes THESE bytes, so it is one per TREE. Without this, a reindex in a
    # worktree would overwrite the index the person is working against.
    assert wt_roots.checkout_local != main_roots.checkout_local
    assert wt_roots.is_linked_worktree is True


def test_project_scope_follows_the_working_tree(
    repo: pathlib.Path, linked: pathlib.Path
) -> None:
    """Committed files travel with the checkout, so `project` is per working tree."""
    assert roots(repo).project == repo.resolve() / PROJECT_DIR
    assert roots(linked).project == linked.resolve() / PROJECT_DIR


def test_local_roots_live_inside_the_git_dir(repo: pathlib.Path) -> None:
    """Inside `.git`, which git never tracks — so no exclude line is needed at all."""
    r = roots(repo)
    assert r.repo_local.name == LOCAL_DIR
    assert ".git" in r.repo_local.parts
    assert ".git" in r.checkout_local.parts


def test_roots_are_absolute_regardless_of_process_cwd(
    repo: pathlib.Path, linked: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--git-dir` answers `.git` relatively in the main tree; unresolved, that would mean
    a different directory for every caller's working directory."""
    for path in (repo, linked):
        r = roots(path)
        for got in (r.project, r.repo_local, r.checkout_local, r.user):
            assert got.is_absolute()


@pytest.mark.writes_worktree("a")
def test_subdirectory_resolves_to_the_same_roots(repo: pathlib.Path) -> None:
    """Declared to the fs-watchdog: the `repo` fixture IS a worktree, so creating a
    directory in it is a real invariant-1 write. It is arrangement rather than simulation,
    and the marker is how that is said out loud instead of worked around."""
    sub = repo / "a"
    sub.mkdir()
    assert roots(sub) == roots(repo)


def test_user_scope_honours_xdg(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert xdg_config_home() == tmp_path / "cfg"


def test_empty_xdg_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty value would otherwise resolve the user scope relative to the process cwd."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    assert xdg_config_home() == pathlib.Path.home() / ".config"


def test_outside_a_repository_is_typed(tmp_path: pathlib.Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(NotARepository):
        roots(plain)
