"""Run journals folded once and kept, each advanced by the lines appended since.

A journal is append-only below what has been written: a torn tail is never read as a line,
and only the writer truncates, past the last complete line. So the bytes a reader has
consumed never change while the file is the same file, and a run's fold
(`projection.Fold`, which takes one line at a time and hands out projections that never
change) can be kept and advanced instead of read again from the first line.

Per run this keeps the file's inode, the bytes consumed, the fold, the next seq and the
first damage. A damaged complete line cannot change, so once recorded it is never read
again. A torn tail is not consumed: its line is read once it completes. A new inode or a
shorter file is folded again from the start, and a missing journal drops the entry.

One lock per run, so two threads never advance one fold; different runs advance in
parallel. The registry's guard is a leaf: nothing is taken while it is held.
"""

from __future__ import annotations

import stat
import threading
from dataclasses import dataclass, field

from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.paths import journal_path
from chickadee.journal.projection import Fold, RunProjection
from chickadee.journal.reader import read_tail
from chickadee.state.roots import Roots

__all__ = ["RunProjections"]


@dataclass
class _RunFold:
    """One run's fold and how far it has read. Every field is guarded by `lock`."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    inode: int | None = None
    consumed: int = 0
    next_seq: int = 1
    fold: Fold = field(default_factory=Fold)
    damage: JournalCorrupt | None = None


class RunProjections:
    """Each run's journal folded once and kept, advanced by the lines appended since. One per
    engine, shared by every reader it composes."""

    def __init__(self, roots: Roots) -> None:
        self._roots = roots
        self._guard = threading.Lock()
        """Guards `_runs`, the registry only. A leaf: nothing else is taken under it."""
        self._runs: dict[str, _RunFold] = {}

    def projection(self, run_id: str) -> RunProjection:
        """What `project(read_run(...))` returns, or raises: `FileNotFoundError` for a run
        with no journal, the damage at its line, or the read's own `OSError`."""
        folded, refusal = self._read(run_id)
        if refusal is not None:
            raise refusal
        return folded

    def readable(self, run_id: str) -> RunProjection:
        """What `project_readable(read_run(...))` returns: every line before any damage or
        failed read, folded. A run with no journal is `FileNotFoundError`, as `read_run`
        raises it."""
        return self._read(run_id)[0]

    def _read(self, run_id: str) -> tuple[RunProjection, Exception | None]:
        path = journal_path(self._roots, run_id)
        with self._guard:
            run = self._runs.setdefault(run_id, _RunFold())
        with run.lock:
            try:
                info = path.stat()
            except FileNotFoundError:
                info = None
            if info is None or not stat.S_ISREG(info.st_mode):
                with self._guard:
                    if self._runs.get(run_id) is run:
                        del self._runs[run_id]
                raise FileNotFoundError(f"run {run_id} has no journal at {path}")
            if info.st_ino != run.inode or info.st_size < run.consumed:
                # Not the file this fold read: a new one, or one cut shorter. Read it whole.
                run.inode, run.consumed, run.next_seq = info.st_ino, 0, 1
                run.fold, run.damage = Fold(), None
            if run.damage is None and info.st_size > run.consumed:
                try:
                    tail = read_tail(path, run.consumed, run.next_seq)
                except OSError as exc:
                    # Not damage: the next read tries again from where this one stood. A read
                    # that fails yields nothing, so `project_readable` of it is an empty fold.
                    return Fold().projection(), exc
                for event in tail.events:
                    run.fold.add(event)
                run.consumed, run.next_seq = tail.end, run.next_seq + len(tail.events)
                run.damage = tail.damage
            return run.fold.projection(), _again(run.damage)


def _again(damage: JournalCorrupt | None) -> JournalCorrupt | None:
    """The recorded damage as a new exception each time it is raised, so two threads raising
    it never share one traceback."""
    if damage is None:
        return None
    again = JournalCorrupt(damage.seq, damage.path, damage.detail)
    again.__cause__ = damage.__cause__
    return again
