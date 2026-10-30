"""Gitignore-aware index walk.

`.git` and `.chickadee` are never source, even if a user un-ignores them — walking them
indexes the harness's own journals and a nested worktree. Everything else follows the
repository's `.gitignore` files the way git does: a rule is relative to the directory that
contains it, later matching rules override earlier ones, and `!` un-ignores.

Hard-coded vendor/cache directory names stay skipped as well. Gitignore is additive: a
repo that forgot the file must not suddenly index `node_modules`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from chickadee.paths import glob_match
from chickadee.secrets.denylist import Denylist, read_denied

ALWAYS_SKIP = {
    ".git",
    ".chickadee",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass(frozen=True)
class _Rule:
    glob: str
    """As `paths.glob_match` reads it: the line, less its `!` and trailing `/`."""
    negate: bool
    dir_only: bool


def _parse_gitignore(text: str) -> tuple[_Rule, ...]:
    rules: list[_Rule] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        if line.strip("/"):
            rules.append(_Rule(glob=line, negate=negate, dir_only=dir_only))
    return tuple(rules)


def _load(gitignore: Path) -> tuple[_Rule, ...]:
    try:
        return _parse_gitignore(gitignore.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ()


def _matches(rule: _Rule, rel: str, *, is_dir: bool) -> bool:
    """Git: a path is ignored when it, or a directory it is in, matches the rule's glob, and
    a trailing `/` matches directories only: `generated/` ignores `generated/a.py` because
    `generated` is a directory it is in, never a file named `generated`. Matching a glob is
    `paths.glob_match`'s; what is a candidate is this reader's."""
    parts = rel.split("/")
    within = ["/".join(parts[:i]) for i in range(1, len(parts))]
    candidates = within if rule.dir_only and not is_dir else [*within, rel]
    return any(glob_match(c, (rule.glob,)) is not None for c in candidates)


class Ignore:
    """Cached gitignore evaluation for one repository root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._files: dict[Path, tuple[_Rule, ...]] = {}

    def _rules(self, directory: Path) -> tuple[_Rule, ...]:
        gi = directory / ".gitignore"
        cached = self._files.get(gi)
        if cached is None:
            cached = _load(gi) if gi.is_file() else ()
            self._files[gi] = cached
        return cached

    def skipped(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            rel = resolved.relative_to(self.root)
        except ValueError:
            return True
        if any(part in ALWAYS_SKIP for part in rel.parts):
            return True
        is_dir = resolved.is_dir()
        ignored = False
        current = self.root
        chain = [self.root]
        for part in rel.parts[:-1]:
            current = current / part
            chain.append(current)
        for directory in chain:
            rel_to = resolved.relative_to(directory).as_posix()
            for rule in self._rules(directory):
                if _matches(rule, rel_to, is_dir=is_dir):
                    ignored = not rule.negate
        return ignored


def walk(root: Path) -> Iterator[str]:
    """Every file under `root` the repository does not ignore, repository-relative and POSIX,
    in a stable order: what `glob` searches (`scout/search.py`)."""
    rules = Ignore(root)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not rules.skipped(current / d))
        for name in sorted(filenames):
            full = current / name
            if not rules.skipped(full):
                yield full.relative_to(root).as_posix()


def listed(root: Path, denylist: Denylist) -> tuple[str, ...]:
    """What `glob` would list under `root`: `walk`'s files less those the read rule refuses
    (`secrets.read_denied`, by name and by landing), so a path suggested from it never says a
    secret exists."""
    real = Path(os.path.realpath(root))
    return tuple(
        rel
        for rel in walk(root)
        if read_denied(rel, root / rel, denylist, workspace=real) is None
    )
