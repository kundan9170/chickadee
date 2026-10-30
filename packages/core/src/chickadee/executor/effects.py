"""The effect snapshot/diff bracket.

What a command actually touched, measured rather than trusted. A declaration is a claim;
this is the check that the claim held.
"""

from __future__ import annotations

import os
from pathlib import Path

from chickadee.executor.types import Violation
from chickadee.paths import glob_match

SKIP_DIRS = {".git", ".chickadee", "node_modules", "__pycache__", ".venv", ".pytest_cache"}


Fingerprint = tuple[int, int, int]
"""`(mtime_ns, ctime_ns, size)`.

mtime and size alone are forgeable: a command can rewrite a file with identical length and
restore the timestamp with `utime`, and the diff then reports nothing — verified. `ctime`
changes on any inode modification and **cannot be set from userland**, so it is the field
that makes the bracket a check rather than a courtesy.
"""

DIR_MARKER: Fingerprint = (-1, -1, -1)
"""Stands in for a directory's stat. Directories are recorded because an action can create
or remove an **empty** one, and a file-only snapshot sees nothing either way."""


def snapshot_paths(root: Path) -> dict[str, Fingerprint]:
    """mtime and size per file, plus a marker per directory.

    Not content hashes: the bracket runs around every command, and hashing a large tree
    twice per execution would cost more than the check is worth.
    """
    seen: dict[str, Fingerprint] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in dirnames:
            full = Path(dirpath) / name
            seen[str(full.relative_to(root))] = DIR_MARKER
        for name in filenames:
            full = Path(dirpath) / name
            try:
                stat = full.stat()
            except OSError:
                continue
            seen[str(full.relative_to(root))] = (
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_size,
            )
    return seen


def diff_against(
    before: dict[str, Fingerprint],
    after: dict[str, Fingerprint],
    effects: tuple[str, ...],
) -> tuple[Violation, ...]:
    """Paths that changed and were not declared.

    A directory created or removed to hold a declared change is part of that change:
    `build/**` names what is inside `build`, and making `build` is how it gets
    written. An empty directory is judged by its own path."""
    changed = {
        p: "created" if p not in before else "modified"
        for p, s in after.items()
        if before.get(p) != s
    }
    changed |= {p: "deleted" for p in before if p not in after}
    declared = {p for p in changed if glob_match(p, effects) is not None}
    holders = {d.rsplit("/", i)[0] for d in declared for i in range(1, d.count("/") + 1)}
    violations = [
        Violation(path, reason)
        for path, reason in changed.items()
        if path not in declared
        and not (path in holders and DIR_MARKER in (before.get(path), after.get(path)))
    ]
    return tuple(sorted(violations, key=lambda v: v.path))
