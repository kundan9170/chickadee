"""The repository's own programs never run in the harness's git. No
`core.fsmonitor`, and no hooks, except on a call made for the person onto their branch, which
runs their hooks as their own `git` would."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.process import run_git
from chickadee.workspace import commit_paths, current_tree


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    hook = root / ".git" / "hooks" / "post-commit"
    hook.write_text(f"#!/bin/sh\ntouch {tmp_path / 'hook-ran'}\n")
    hook.chmod(0o755)
    git(root, "config", "core.fsmonitor", f"touch {tmp_path / 'fsmonitor-ran'};false")
    return root


@pytest.mark.writes_worktree("*")
def test_the_harness_s_own_git_runs_no_hook_and_no_fsmonitor(
    tmp_path: Path, repo: Path
) -> None:
    current_tree(repo, untracked_max_bytes=None)
    (repo / "a.py").write_text("a = 2\n")
    commit_paths(repo, ["a.py"], "the harness's own unit", own_unit=True)
    assert not (tmp_path / "fsmonitor-ran").exists()
    assert not (tmp_path / "hook-ran").exists()


@pytest.mark.writes_worktree("*")
def test_a_commit_made_for_the_person_on_their_branch_runs_their_hooks(
    tmp_path: Path, repo: Path
) -> None:
    (repo / "a.py").write_text("a = 3\n")
    commit_paths(repo, ["a.py"], "an undo, on their branch", own_unit=False)
    assert (tmp_path / "hook-ran").exists()
    assert not (tmp_path / "fsmonitor-ran").exists(), "fsmonitor stays off even then"


def test_reading_the_person_s_hooks_path_is_never_the_harness_s_override(repo: Path) -> None:
    git(repo, "config", "core.hooksPath", ".husky")
    off = run_git(repo, "config", "--get", "core.hooksPath")
    theirs = run_git(repo, "config", "--get", "core.hooksPath", hooks=True)
    assert off.stdout.strip() == b"/dev/null"
    assert theirs.stdout.strip() == b".husky"
