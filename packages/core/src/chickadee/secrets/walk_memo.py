"""What `denylist.resolve` found in each directory, kept so an unchanged one is not walked
again.

`resolve` masks every file and directory under the workspace and `$HOME` whose name the deny
list denies, and it ran at every session open, every loop rebuild and every run: a walk of the
whole home directory, each name matched. In a home with a package cache that was seconds each
time, and it grew with the person's home.

The verdict on a directory's entries depends only on their names, where the directory sits
under the root, the list, and the trust-store files the environment names (`_ca_files`). A
directory's `mtime` moves whenever an entry is added, removed or renamed in it. So a directory
seen before, whose `mtime_ns` is the same, has the same verdict, and its walk is one `stat`.

Kept exact by three rules:

- **Racy entries are never kept.** A directory changed within `RACY_NS` of its walk may be
  changed again with no visible `mtime` move (a coarse clock): it is walked every time until
  it is older.
- **A directory holding a symlink is never kept**: a link's verdict reads its target's kind
  (`is_dir` follows it, as `os.walk` does), which can change with no move in this directory.
- **The key is the whole question**: the list's entries and switches, the root, and the
  trust-store files; any change is a new memo, never a stale answer.

A cache, not a record: a memo that cannot be read is no memo, and the walk is done whole. It
is written atomically (`durable.files.write_atomic`), so a torn write is never read.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from chickadee.durable.files import make_dir, write_atomic
from chickadee.isolation.types import harness_cache

__all__ = ["RACY_NS", "Seen", "WalkMemo", "memo_key"]

RACY_NS: Final = 2_000_000_000
"""How old a directory's change must be before its verdict is kept: 2 s, past the coarsest
`mtime` a common filesystem keeps (FAT's 2 s)."""

_VERSION: Final = 1


@dataclass(frozen=True)
class Seen:
    """One directory as the walk judged it."""

    mtime_ns: int
    denied: tuple[str, ...]
    """Its entries the list denies: each is masked, and a denied directory is not entered."""
    descend: tuple[str, ...]
    """Its subdirectories walked into."""
    keepable: bool
    """False when it holds a symlink: never kept."""


class WalkMemo:
    """The memo, one file per question under `where`, and held in memory once read. Shared by
    every caller in a process (`shared`); its lock guards `_held` only."""

    def __init__(self, where: Path) -> None:
        self._where = where
        self._lock = threading.Lock()
        self._held: dict[str, dict[str, Seen]] = {}

    @staticmethod
    def shared() -> WalkMemo:
        """The process's one memo, under the harness's cache."""
        return _shared()

    def of(self, key: str) -> Mapping[str, Seen]:
        """What was kept for `key`, by directory path; empty when nothing was, or it cannot
        be read."""
        with self._lock:
            held = self._held.get(key)
        if held is not None:
            return held
        try:
            raw = json.loads((self._where / f"{key}.json").read_text(encoding="utf-8"))
            found = _parsed(raw)
        except (OSError, ValueError, TypeError, KeyError):
            found = {}
        with self._lock:
            self._held.setdefault(key, found)
            return self._held[key]

    def keep(self, key: str, seen: Mapping[str, Seen]) -> None:
        """`seen` is what `key` holds now: in memory, and on disk when it changed."""
        kept = dict(seen)
        with self._lock:
            before = self._held.get(key)
            self._held[key] = kept
        if before == kept:
            return
        body = {
            "version": _VERSION,
            "dirs": {
                path: [s.mtime_ns, list(s.denied), list(s.descend)] for path, s in kept.items()
            },
        }
        try:
            make_dir(self._where)
            write_atomic(self._where / f"{key}.json", json.dumps(body).encode("utf-8"))
        except OSError:
            # A cache that cannot be written is no cache: the next walk is whole.
            return


def memo_key(question: object) -> str:
    """The memo's name for one question (the list, the root, the trust stores)."""
    text = json.dumps([_VERSION, question], sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _parsed(raw: object) -> dict[str, Seen]:
    """The memo as written, or empty when it is not (a `ValueError` or `TypeError` from a
    malformed row is the caller's, which reads it as no memo)."""
    if not isinstance(raw, dict):
        return {}
    body = cast("dict[str, object]", raw)
    dirs = body.get("dirs")
    if body.get("version") != _VERSION or not isinstance(dirs, dict):
        return {}
    out: dict[str, Seen] = {}
    for path, row in cast("dict[str, list[Any]]", dirs).items():
        mtime, denied, descend = row
        out[str(path)] = Seen(
            int(mtime), tuple(str(n) for n in denied), tuple(str(n) for n in descend), True
        )
    return out


_SHARED: list[WalkMemo] = []
_SHARED_LOCK = threading.Lock()


def _shared() -> WalkMemo:
    with _SHARED_LOCK:
        if not _SHARED:
            _SHARED.append(WalkMemo(harness_cache() / "deny-walk"))
        return _SHARED[0]
