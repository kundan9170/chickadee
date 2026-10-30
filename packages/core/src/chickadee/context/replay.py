"""Folds of whole sessions read without their locks, kept and advanced.

Its own module because it is a reader's cache, not the fold or the writer `session.py` holds,
and `session.py` is held at its size by the context package's guard.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from chickadee.context.session import SessionState, parsed
from chickadee.context.store import LOG_NAME, SessionEvent, SessionLog, session_dir

__all__ = ["ReplayCache"]

Stamp = tuple[int, int, int]
"""A log file's inode, size and modification time, taken before it is read."""


@dataclass(frozen=True)
class _Held:
    """One session's entry: immutable, swapped whole under the cache's guard, never edited."""

    stamp: Stamp
    log: SessionLog
    lines: tuple[tuple[SessionEvent, object | None], ...]
    """Every line the log reads, its payload parsed: what a later advance folds again."""
    state: SessionState


class ReplayCache:
    """Folds of whole sessions read without their locks, kept while their logs are unchanged,
    and advanced by the lines appended since when they grow.

    A log below any seq already written is immutable, and a fork's read-through lines are its
    parent's lines below the fork point, so a session's own log file is the only thing whose
    change can change its fold. Its stat is taken *before* the read: a line appended during
    the read leaves the stamp stale and the next call reads again, never the reverse.

    **Advancing.** When the stamp moved and the file is the same file (same inode, not shorter
    than what was read), only its new complete lines are read (`SessionLog.extend`), and only
    their payloads parsed. Every line is then checked and applied from the first by the one
    fold (`SessionState.fold`), in memory, into a new state: a state handed out never changes,
    so a reader on another thread needs no lock. A payload already read is not read again
    while its log only grows, as a stamp hit already trusts an unchanged log's. Otherwise (a
    new inode, a shorter file, no line of its own read yet) the session is read whole."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._guard = threading.Lock()
        """Guards `_held`. Entries are swapped under it, never edited, and nothing is read or
        folded while it is held."""
        self._held: dict[str, _Held] = {}

    def get(self, session_id: str) -> tuple[SessionLog, SessionState]:
        try:
            info = (session_dir(self._root, session_id) / LOG_NAME).stat()
            stamp: Stamp | None = (info.st_ino, info.st_size, info.st_mtime_ns)
        except FileNotFoundError:
            stamp = None  # the read below says what is missing, as it always has
        with self._guard:
            if stamp is None:
                self._held.pop(session_id, None)
            hit = self._held.get(session_id)
        if hit is not None and hit.stamp == stamp:
            return hit.log, hit.state
        if hit is not None and stamp is not None and _same_file(hit, stamp):
            log, kept = hit.log.extend(), hit.lines
        else:
            log, kept = SessionLog.read(self._root, session_id), ()
        new: list[tuple[SessionEvent, object | None]] = []
        state = SessionState.fold(session_id, self._root, self._lines(log, kept, new))
        if stamp is not None:
            entry = _Held(stamp, log, (*kept, *new), state)
            with self._guard:
                held = self._held.get(session_id)
                # Two threads may advance one session at once; the later read is kept. An
                # entry nobody swapped since this read began is replaced whatever its length,
                # so a log replaced by a shorter one is cached again.
                if held is None or held is hit or held.log.last_seq <= log.last_seq:
                    self._held[session_id] = entry
        return log, state

    def _lines(
        self,
        log: SessionLog,
        kept: tuple[tuple[SessionEvent, object | None], ...],
        new: list[tuple[SessionEvent, object | None]],
    ) -> Iterator[tuple[SessionEvent, object | None]]:
        """The lines parsed before, then each new one parsed as the fold reaches it, so a
        refusal and its seq are a full replay's."""
        yield from kept
        for event in log.events()[len(kept) :]:
            line = parsed(self._root, event, log.payload)
            new.append(line)
            yield line


def _same_file(hit: _Held, stamp: Stamp) -> bool:
    """The file the entry read, only grown: the same inode, not shorter than what was read.
    And at least one line of its own read: until a fork's first line is complete, the log
    reads as a root's, and what it reads through is not known yet."""
    return stamp[0] == hit.stamp[0] and 0 < hit.log.consumed <= stamp[1]
