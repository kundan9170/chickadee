"""Git repositories for tests, made one way (one concept, one home).

Every suite used to keep its own `git` helper and its own way of making a repository: about
thirty copies across the core, cli and eval tests, differing in whether they set an
identity, named a branch or raised on failure. This module is the one. It is shared by every
package's tests, so it lives in `packages/testing`, which the root `pyproject.toml` puts on
pytest's path.

It runs git directly, never through the harness's `ProcessSupervisor`: arranging a test is not
an effect the harness makes, and the spawn inventories (`g2`) must not count it.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from chickadee.state.roots import Roots

IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}
"""Who a test's commits are by. Through the environment, so no test depends on the machine's
git config, and a commit made by any later `git(...)` call has an author too."""


def git(root: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    """Run `git -C root...` and return its stripped stdout; a failure raises
    `subprocess.CalledProcessError` with git's own message. `env` adds to the environment
    (`GIT_INDEX_FILE` for a throwaway index)."""
    done = git_result(root, *args, env=env)
    done.check_returncode()
    return done.stdout.strip()


def git_command(argv: Sequence[str]) -> tuple[str, ...]:
    """A recorded git argv as `git <subcommand> …`, past git's own `-c` options: the harness
    runs every git call with some, and a test asks which subcommand ran."""
    out = [argv[0]]
    rest = list(argv[1:])
    while rest[:1] == ["-c"]:
        rest = rest[2:]
    return (*out, *rest)


def git_result(
    root: Path, *args: str, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `git -C root...` and return what happened, whatever its exit code: for a test
    whose subject is git's own answer (`check-ignore` exits 1 for "not ignored", and 128
    outside a repository)."""
    return subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **IDENTITY, **(env or {})},
    )


def init_repo(
    root: Path,
    files: Mapping[str, str] | None = None,
    *,
    commit: str | None = "seed",
) -> Path:
    """A git repository at `root` on branch `main`, holding `files`, with one commit named
    `commit` when it is not `None` (an unborn repository otherwise). Returns `root`."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    # Also in the repository's own config: harness code that commits with plain `git` (and
    # no identity of its own) finds one there, as it would in a person's clone.
    git(root, "config", "user.name", IDENTITY["GIT_AUTHOR_NAME"])
    git(root, "config", "user.email", IDENTITY["GIT_AUTHOR_EMAIL"])
    for rel, text in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if commit is not None:
        git(root, "add", "-A")
        git(root, "commit", "-q", "--allow-empty", "-m", commit)
    return root


def layer_roots(tmp_path: Path) -> Roots:
    """A scope root for each configuration layer under `tmp_path`, each made: what a test of
    the layers' loading reads and writes, with no repository behind it."""
    for name in ("project", "user", "repo_local", "checkout_local"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    return Roots(
        project=tmp_path / "project",
        repo_local=tmp_path / "repo_local",
        checkout_local=tmp_path / "checkout_local",
        user=tmp_path / "user",
    )


def commit_in(root: Path, name: str, body: str) -> str:
    """`name` written with `body` under `root`, its folder made when missing, and committed
    with everything else changed; the new commit."""
    if not (root / name).parent.is_dir():
        (root / name).parent.mkdir(parents=True)
    (root / name).write_text(body, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", f"chickadee: {name}")
    return git(root, "rev-parse", "HEAD")


def clone_roots(tmp_path: Path) -> Roots:
    """The scope roots a clone at `tmp_path/repo` has, laid out as `roots()` finds them in a
    main worktree, none made: for a test of what is written under them."""
    repo = tmp_path / "repo"
    return Roots(
        project=repo / ".chickadee",
        repo_local=repo / ".git" / "chickadee",
        checkout_local=repo / ".git" / "chickadee",
        user=tmp_path / "user",
    )
