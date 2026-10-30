"""`.git` file resolution and D39 linked-worktree identity."""

from __future__ import annotations

import pathlib

import pytest
from repos import git, init_repo

from chickadee.init import detect, init, publish


def _linked(repo: pathlib.Path) -> pathlib.Path:
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "seed")
    linked = repo.parent / "linked"
    git(repo, "worktree", "add", "-q", str(linked))
    return linked


def test_a_linked_worktree_has_a_git_file_not_a_directory(repo: pathlib.Path) -> None:
    """The assumption this guards against: `.git` is a directory. In a linked worktree it
    is a file, and D39 makes linked worktrees the harness's default."""
    linked = _linked(repo)
    assert (linked / ".git").is_file()
    identity = detect.resolve(linked)
    assert identity.is_linked_worktree
    assert identity.git_common_dir.is_dir()


def test_identity_is_stable_across_worktrees(repo: pathlib.Path) -> None:
    linked = _linked(repo)
    assert detect.resolve(linked).repository_id == detect.resolve(repo).repository_id


def test_a_separate_repository_gets_a_different_identity(
    repo: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    other = init_repo(tmp_path / "other", commit=None)
    assert detect.resolve(other).repository_id != detect.resolve(repo).repository_id


def test_exclude_resolves_through_git_not_a_literal_path(repo: pathlib.Path) -> None:
    """`.git/info/exclude` does not exist inside a linked worktree; writing it there would
    create a file git ignores while the real exclude went unedited."""
    linked = _linked(repo)
    path = detect.exclude_path(linked)
    assert path.is_absolute()
    init(linked)
    assert publish.GIT_EXCLUDE_LINE in path.read_text()


def test_init_from_a_subdirectory_resolves_the_root(repo: pathlib.Path) -> None:
    sub = repo / "pkg" / "deep"
    sub.mkdir(parents=True)
    report = init(sub)
    assert report.root == repo.resolve() or report.root == repo
    assert publish.is_initialised(report.root)


def test_a_bare_directory_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(detect.NotAGitRepository):
        detect.resolve(tmp_path / "nope")
