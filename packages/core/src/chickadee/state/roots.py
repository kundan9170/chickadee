"""The four state scopes, resolved once.

Every path the harness writes belongs to exactly one scope, and the scope is decided by
one question: *what breaks if this is shared, and what breaks if it is not?*

| scope | holds | shared across worktrees? |
|------------------|-----------------------------------------|--------------------------|
| `project` | committed team policy and project shape | it is committed, so yes |
| `repo_local` | this clone's history and leases | **yes** — one per clone |
| `checkout_local` | anything describing *these* bytes | **no** — one per tree |
| `user` | this person, every repository | not repo-scoped at all |

**`checkout_local` is the row the design was missing**, and it was missing because in the
main working tree it is indistinguishable from `repo_local`: git reports the same directory
for `--git-dir` and `--git-common-dir` there, and nowhere else. A linked worktree is the case
that separates them — and a linked worktree is the *default* isolation mode, so the
ambiguity sat under every autonomous run. See an earlier security review.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

PROJECT_DIR = ".chickadee"
"""§8. Tracked, in the working tree. The name people already know."""

LOCAL_DIR = "chickadee"
"""§8. A directory *inside* the git dir, so no leading dot and no exclude line: git never
tracks its own directory, which is what lets ST-2 delete J-2's exclusion rather than narrow
it. A rule that does not exist cannot drift out of step with what it excludes."""

USER_DIR = "chickadee"
"""§8. Under `XDG_CONFIG_HOME`, defaulting to `~/.config` — the convention every other tool
on the machine follows, and finding 2's missing operand."""


class NotARepository(Exception):
    """`roots()` was asked about a directory that is not inside a git repository.

    Typed rather than a bare OS error because the caller's remedy is a sentence ("run this
    inside a git repository"), not a stack trace.
    """


@dataclasses.dataclass(frozen=True)
class Roots:
    """Resolved once, at engine construction, and passed down.

    Never re-derived per component: a component that resolves its own root is a component
    that can resolve a *different* root than its neighbour, which is the class of bug this
    type exists to make impossible.
    """

    project: pathlib.Path
    repo_local: pathlib.Path
    checkout_local: pathlib.Path
    user: pathlib.Path

    @property
    def is_linked_worktree(self) -> bool:
        """True when this is a linked worktree rather than the main working tree.

        Exposed because the honest behaviours differ — a fresh worktree has no index and
        must be told so rather than failing obscurely — and because a test asserting
        the two local roots differ needs to say *which* case it is in.
        """
        return self.repo_local != self.checkout_local


def xdg_config_home() -> pathlib.Path:
    """`$XDG_CONFIG_HOME`, or `~/.config` when unset or empty.

    The spec says *defaulting to* `~/.config`; an empty string is the same as unset for
    every other XDG consumer, and treating `""` as a path would resolve the user scope
    relative to the current directory, which is the one place it must never be.
    """
    raw = os.environ.get("XDG_CONFIG_HOME")
    if not raw:
        return pathlib.Path.home() / ".config"
    return pathlib.Path(raw)


def _find_dot_git(start: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Walk up for `.git`, returning (working tree, the `.git` entry).

    Read from disk rather than asked of `git rev-parse`, and that is not a shortcut:
    **invariant 2 gives ProcessSupervisor the monopoly on spawning**, and layout resolution
    runs *before* a supervisor exists — it is what decides where the supervisor's own state
    lives. Rather than claim a bootstrap exception to an invariant, read the format git
    documents. It is also faster, and it works where git is not installed.
    """
    here = start.resolve()
    for candidate in (here, *here.parents):
        entry = candidate / ".git"
        if entry.exists():
            return candidate, entry
    raise NotARepository(f"{start} is not inside a git repository (no .git found above it)")


def _git_dirs(worktree: pathlib.Path, entry: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Return (git_dir, common_dir) from git's on-disk layout.

    Two shapes, both documented by git:

    - **main working tree** — `.git` is a directory, and it is both the git dir and the
      common dir. *That identity is exactly why finding 6 was invisible.*
    - **linked worktree** — `.git` is a FILE holding `gitdir: <path>`, and that directory
      holds a `commondir` file pointing at the shared `.git`.
    """
    if entry.is_dir():
        return entry, entry

    text = entry.read_text(encoding="utf-8").strip()
    if not text.startswith("gitdir:"):
        raise NotARepository(f"{entry} is not a git link file: {text[:60]!r}")
    git_dir = pathlib.Path(text.split(":", 1)[1].strip())
    if not git_dir.is_absolute():
        git_dir = (worktree / git_dir).resolve()

    common_file = git_dir / "commondir"
    if not common_file.is_file():
        # A link file with no commondir is a worktree git dir that stands alone.
        return git_dir, git_dir
    raw = common_file.read_text(encoding="utf-8").strip()
    common = pathlib.Path(raw)
    if not common.is_absolute():
        common = (git_dir / common).resolve()
    return git_dir, common


def roots(cwd: pathlib.Path | None = None) -> Roots:
    """Resolve the four scopes for `cwd`.

    The **only** place in the codebase that resolves repository layout. Three answers, and
    they are not interchangeable:

    - the **common** git dir — shared by every linked worktree -> `repo_local`
    - **this** worktree's git dir -> `checkout_local`
    - this worktree's working directory -> `project`

    In the main working tree the first two are the same directory, and nowhere else. A
    linked worktree is the case that separates them — and it is the default isolation mode,
    so the ambiguity sat under every autonomous run.
    """
    here = pathlib.Path(cwd) if cwd is not None else pathlib.Path.cwd()
    worktree, entry = _find_dot_git(here)
    git_dir, common = _git_dirs(worktree, entry)
    return Roots(
        project=worktree / PROJECT_DIR,
        repo_local=common.resolve() / LOCAL_DIR,
        checkout_local=git_dir.resolve() / LOCAL_DIR,
        user=xdg_config_home() / USER_DIR,
    )
