"""`git_guard`: which change to git's settings stands. Only a branch's upstream, to a
remote the config already names, is kept; everything else is put back."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.git_guard import dump, guarded, held, load, put_back


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = init_repo(tmp_path / "repo", {"a.txt": "a\n"})
    git(root, "remote", "add", "origin", "https://example.com/x.git")
    return root


def _changed(repo: Path, *config: str) -> tuple[tuple[str, ...], bytes]:
    """`git config` with `config`, then the guard: what it put back, and the config after."""
    before = held(repo)
    git(repo, "config", *config)
    restored = put_back(before, held(repo))
    return restored, (repo / ".git" / "config").read_bytes()


@pytest.mark.parametrize(
    "config",
    [("branch.main.remote", "origin"), ("branch.main.merge", "refs/heads/main")],
    ids=["remote", "merge"],
)
def test_a_branch_s_upstream_to_a_named_remote_is_kept(
    repo: Path, config: tuple[str, str]
) -> None:
    restored, after = _changed(repo, *config)
    assert restored == ()
    assert config[1].encode() in after


@pytest.mark.parametrize(
    "config",
    [
        ("branch.main.remote", "https://elsewhere.example/x.git"),
        ("branch.main.remote", "mirror"),
        ("branch.main.merge", "HEAD"),
        ("branch.main.pushRemote", "origin"),
        ("core.hooksPath", ".husky"),
        ("core.sshCommand", "touch pwned"),
        ("remote.origin.url", "https://elsewhere.example/x.git"),
        ("include.path", "../evil"),
    ],
    ids=[
        "url-remote",
        "unknown-remote",
        "not-a-branch",
        "push-remote",
        "hooks-path",
        "ssh",
        "remote-url",
        "include",
    ],
)
def test_every_other_change_is_put_back(repo: Path, config: tuple[str, str]) -> None:
    original = (repo / ".git" / "config").read_bytes()
    restored, after = _changed(repo, *config)
    assert restored == (str((repo / ".git" / "config").resolve()),)
    assert after == original


def test_an_upstream_beside_another_change_is_put_back_with_it(repo: Path) -> None:
    original = (repo / ".git" / "config").read_bytes()
    before = held(repo)
    git(repo, "config", "branch.main.remote", "origin")
    git(repo, "config", "core.fsmonitor", "evil")
    assert put_back(before, held(repo)) != ()
    assert (repo / ".git" / "config").read_bytes() == original


def test_a_hook_is_never_kept(repo: Path) -> None:
    before = held(repo)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\n")
    assert put_back(before, held(repo)) == (str(hook.resolve()),)
    assert not hook.exists()


def test_guarded_names_git_s_settings_as_they_exist(repo: Path) -> None:
    git_dir = (repo / ".git").resolve()
    (git_dir / "info").mkdir(exist_ok=True)
    found = guarded(repo)
    assert git_dir / "config" in found and git_dir / "hooks" in found
    assert git_dir / "info" in found
    assert git_dir / "config.worktree" not in found, "only what exists is bound read-only"


@pytest.mark.writes_worktree("*")
def test_guarded_names_what_a_tool_outside_the_sandbox_runs_from_the_worktree(
    repo: Path,
) -> None:
    """The hooks folder git is told to use, and the editors' folders."""
    for folder in ("hooks-here", ".husky", ".vscode", ".idea"):
        (repo / folder).mkdir()
    git(repo, "config", "core.hooksPath", "hooks-here")
    found = guarded(repo)
    for folder in ("hooks-here", ".husky", ".vscode", ".idea"):
        assert (repo / folder).resolve() in found, folder


@pytest.mark.writes_worktree("*")
def test_a_change_under_an_editor_s_folder_is_put_back(repo: Path) -> None:
    (repo / ".vscode").mkdir()
    tasks = repo / ".vscode" / "tasks.json"
    tasks.write_text("{}\n")
    before = held(repo)
    tasks.write_text('{"tasks": [{"runOptions": {"runOn": "folderOpen"}}]}\n')
    assert put_back(before, held(repo)) == (str(tasks.resolve()),)
    assert tasks.read_text() == "{}\n"


def test_a_hold_survives_its_blob(repo: Path) -> None:
    """Resume reads the hold a step made before its command ran."""
    kept = held(repo)
    assert load(dump(kept)) == kept
