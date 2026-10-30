"""A folder with no git repository made into one: `init`'s first step there.

Chickadee keeps every change in git: a task is a commit, a worktree branches from `HEAD`, undo
puts a commit back. A folder that is not a repository was refused (`NotAGitRepository`). Now
`init` makes it one, at the owner's word ("instead of refusing it should do it on its own"):

1. `git init`;
2. the disposable directories (`UNTRACKED`) written to `.git/info/exclude`;
3. everything else staged, then every staged file the deny list denies (`.env`, keys,
   `.ssh/`, …) unstaged and excluded by its path: the one secret rule (`name_denied`), never
   a second one translated into gitignore syntax;
4. one local commit, `chickadee: start`. Nothing is pushed anywhere.

Never in the person's home directory, the filesystem's root, or a folder holding the home
directory: a repository there would take in everything under it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

from chickadee.durable.files import write_atomic
from chickadee.errors import InfraError
from chickadee.init.detect import exclude_path
from chickadee.process import git, run_git
from chickadee.secrets.denylist import Denylist, name_denied

__all__ = ["COMMIT_MESSAGE", "UNTRACKED", "NotHere", "make_repository"]

COMMIT_MESSAGE: Final = "chickadee: start"

UNTRACKED: Final = (
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".DS_Store",
)
"""What no project keeps in history: installed packages, caches. Not `init`'s `SKIP_DIRS`
whole: a `build/` or `dist/` can be a project's own sources."""

_IDENTITY: Final = {"user.name": "Chickadee", "user.email": "chickadee@localhost"}
"""The first commit's author, each part only where git has none configured; the person's own
otherwise."""


class NotHere(Exception):
    """A folder Chickadee will not make a repository of: the home directory, or above it."""


def make_repository(
    folder: Path,
    denylist: Denylist | None = None,
    *,
    say: Callable[[str], None] = lambda _line: None,
) -> tuple[str, ...]:
    """Make `folder` a git repository holding its files as one first commit, secrets and
    disposables left out of it; returns the paths left out for being secret, so the person
    is told. `NotHere` for the home directory, the root, or a folder holding the home."""
    folder = folder.resolve()
    home = Path.home().resolve()
    if folder == Path(folder.anchor) or folder == home or home.is_relative_to(folder):
        raise NotHere(
            f"{folder} is not a git repository, and Chickadee will not make one there: it "
            "would take in everything under it. Run it in a project's folder"
        )
    deny = denylist or Denylist()
    say(f"{folder} is not a git repository; making one (local only, nothing is pushed)")
    git(folder, "init", "--quiet")
    exclude = exclude_path(folder)
    exclude.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Written by Chickadee when it made this repository.", *UNTRACKED]
    write_atomic(exclude, ("\n".join(lines) + "\n").encode("utf-8"))
    git(folder, "add", "--all")
    staged = git(folder, "ls-files", "--cached", "-z")
    secret = tuple(p for p in staged.split("\0") if p and name_denied(p, deny) is not None)
    if secret:
        # On stdin, NUL-separated: a folder of key fixtures can hold more than an argv may.
        done = run_git(
            folder,
            "rm",
            "--cached",
            "--quiet",
            "--pathspec-from-file=-",
            "--pathspec-file-nul",
            stdin=b"\0".join(p.encode("utf-8") for p in secret),
        )
        if done.exit_code != 0:
            raise InfraError(
                "could not leave the secret files out of the first commit: "
                + done.stderr.decode("utf-8", "replace").strip()
            )
        lines += [f"/{_literal(p)}" for p in secret]
        write_atomic(exclude, ("\n".join(lines) + "\n").encode("utf-8"))
        say(f"left out of history as secret: {', '.join(secret)}")
    identity = _missing_identity(folder)
    try:
        git(folder, *identity, "commit", "--quiet", "--allow-empty", "-m", COMMIT_MESSAGE)
    except InfraError as exc:
        raise InfraError(f"the first commit failed: {exc}") from exc
    return secret


def _missing_identity(folder: Path) -> tuple[str, ...]:
    """`-c` settings for each part of an author git has none of, so the commit is made."""
    out: list[str] = []
    for key, value in _IDENTITY.items():
        if not run_git(folder, "config", key).stdout.strip():
            out += ["-c", f"{key}={value}"]
    return tuple(out)


def _literal(path: str) -> str:
    """`path` as an exclude line that matches it and only it: git's pattern characters, and a
    leading `#` or `!`, escaped."""
    out = "".join(f"\\{c}" if c in "*?[]\\" else c for c in path)
    return f"\\{out}" if out[:1] in ("#", "!") else out
