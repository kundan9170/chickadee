"""Turn snapshots: a turn that changes code is snapshotted before its first write and when it
ends.

Lazily: nothing is taken until the turn's first `coder` dispatch, after the run's workspace is
acquired and before the box runs, so a turn that only talks costs nothing, and an edit the
person made during the turn, before the harness wrote, is inside its `before`. Each workspace
the turn writes in gets one pair: a worktree run's own, or the checkout in place.

The git work is `workspace.snapshot_turn`, the retention `workspace.sweep_snapshots`: this
module says when, for which turn, and records the line. The `code_snapshot` line's type lives
in `context/rewind.py`, where the fork and the undo read it through `taken()`; this writer
checks what the context fold would.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from chickadee.config import UndoPolicy
from chickadee.context.lines import PayloadError
from chickadee.context.rewind import CodeSnapshotLine, taken
from chickadee.context.session import Session
from chickadee.workspace import Phase, SnapshotSite, snapshot_turn, sweep_snapshots

__all__ = ["TurnSnapshots", "record"]


def record(session: Session, line: CodeSnapshotLine) -> None:
    """A snapshot's line, checked as the fold checks its own kinds, then synced: the ref it
    names already exists, and an undo after a power cut must still find it. Three rules: it
    names a user turn, a workspace has one `before` in a turn's part, and its `after` follows
    that `before` once. Under the session's lock: a run's `after` is written from its
    watcher's thread while the conversation writes its own lines."""
    with session.lock:
        opened = session.state.turn(line.turn)
        if opened is None or opened.role != "user":
            raise PayloadError(f"a code snapshot's turn {line.turn} is not a user turn")
        mine = (line.turn, line.part, line.key)
        phases = {s.phase for s in taken(session) if (s.turn, s.part, s.key) == mine}
        of = f"turn {line.turn}" + (f" part {line.part}" if line.part else "")
        if line.phase in phases:
            raise PayloadError(f"{of} already has a {line.phase} snapshot of {line.key}")
        if line.phase == "after" and "before" not in phases:
            raise PayloadError(f"{of}'s after snapshot of {line.key} has no before")
        session.append(line)
        session.sync()


class TurnSnapshots:
    """One conversation's turn snapshots. `before` at each `coder` dispatch, `ended` when a turn
    ends, both on the conversation's worker; and `run_ended` when a run handed over in the
    background ends, on its watcher's thread: the run's changes are the run's, so
    a turn that ends while the run goes on leaves that workspace alone, and the run's end takes
    its `after`, under the turn and part its `before` named. A run that ends while its turn is
    still going leaves the `after` to the turn's end, as an unattended run does: a second run
    the turn starts in the same workspace is then inside it too."""

    def __init__(
        self,
        session: Session,
        policy: UndoPolicy,
        repo: Path,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session = session
        self._policy = policy
        self._repo = repo
        self._clock = clock
        self._woken: tuple[int, int] | None = None
        """The wake turn in progress, as its person's turn and its part, from `woken`
        until the person's next turn."""
        self._lock = threading.Lock()
        """Guards `_held`, and makes each take and its line one step: `ended` runs on the
        worker, `run_ended` on a run's watcher. A leaf but for the session's own lock."""
        self._held: dict[str, tuple[int, int]] = {}
        """Each workspace a live run writes in, by key: the turn and part of its `before`."""
        self._closed: set[tuple[int, int]] = set()
        """The turns and parts `ended` has run for: a run held at one of them outlived it, and
        its end takes the `after` the turn left."""

    def woken(self) -> None:
        """A wake turn begins: it has no user line, so what it writes is
        its person's last turn's next part, snapshotted as a pair of its own."""
        turn = self._turn()
        if turn is None:
            return
        parts = [s.part for s in taken(self._session) if s.turn == turn]
        self._woken = (turn, max(parts, default=0) + 1)

    def before(self, site: SnapshotSite) -> None:
        """The turn's `before` of `site`, unless it already has one: a second run in the same
        checkout writes into a tree the first `before` already holds."""
        turn = self._turn()
        if turn is None:
            return
        part = self._part(turn)
        if any((s.turn, s.part, s.key) == (turn, part, site.key) for s in taken(self._session)):
            return
        self._take(site, turn, "before", part)

    def hold(self, site: SnapshotSite) -> None:
        """`site` is written by a run handed over in the background: its `after` waits for
        `run_ended`, under the turn and part of the `before` just taken."""
        turn = self._turn()
        if turn is None:
            return
        with self._lock:
            self._held[site.key] = (turn, self._part(turn))

    def run_ended(self, site: SnapshotSite) -> None:
        """The run writing in `site` ended, or was abandoned before it started: its `after`,
        under the turn and part it was held at, when that turn has ended; while it is still
        going, the turn's own end takes it. Nothing for a site no run held (a resumed run's,
        whose turn took none)."""
        with self._lock:
            at = self._held.pop(site.key, None)
            if at is None or at not in self._closed:
                return
            turn, part = at
            if any(
                (s.turn, s.part, s.key, s.phase) == (turn, part, site.key, "after")
                for s in taken(self._session)
            ):
                return
            if site.root.is_dir():
                self._take(site, turn, "after", part)

    def ended(self) -> None:
        """Each workspace with a `before` in the turn, or wake, that just ended gets its
        `after`, from the root its `before` names. One with none takes nothing, and one a live
        run still writes in waits for its `run_ended`."""
        turn = self._turn()
        if turn is None:
            return
        part = self._part(turn)
        with self._lock:
            # One step with the mark: a run ending after it takes its own `after`; one that
            # ended before it is no longer held, so its workspace's `after` is taken here.
            self._closed.add((turn, part))
            held = {key for key, at in self._held.items() if at == (turn, part)}
        mine = [s for s in taken(self._session) if (s.turn, s.part) == (turn, part)]
        done = {s.key for s in mine if s.phase == "after"} | held
        for line in (s for s in mine if s.phase == "before" and s.key not in done):
            root = Path(line.root)
            if not root.is_dir():
                # A worktree the turn discarded (`worktree discard`): there is
                # no tree left to take, and its last state is the discard's own snapshot. The
                # turn's code undo then says the pair is incomplete, rather than guessing.
                continue
            site = SnapshotSite(root, line.mode, line.key)
            self._take(site, turn, "after", part)

    def _part(self, turn: int) -> int:
        """The part being written: a wake's, until the person's next turn; else the turn's."""
        woken = self._woken
        return woken[1] if woken is not None and woken[0] == turn else 0

    def _turn(self) -> int | None:
        """The turn in progress: the latest user turn's seq."""
        users = [t.seq for t in self._session.state.turns() if t.role == "user"]
        return max(users) if users else None

    def _take(self, site: SnapshotSite, turn: int, phase: Phase, part: int = 0) -> None:
        policy = self._policy
        session_id = self._session.session_id
        shot = snapshot_turn(
            site,
            session_id=session_id,
            turn=turn,
            phase=phase,
            untracked_max_bytes=policy.untracked_max_bytes,
            part=part,
        )
        record(
            self._session,
            CodeSnapshotLine(
                turn=turn,
                part=part,
                phase=phase,
                key=site.key,
                mode=site.mode,
                root=str(site.root),
                tree=shot.tree,
                ref=shot.ref,
                head=shot.head,
                skipped=shot.skipped,
            ),
        )
        # The sweep runs when a snapshot is taken.
        sweep_snapshots(
            self._repo,
            session_id,
            keep_days=policy.keep_days,
            keep_turns=policy.keep_turns,
            now=self._clock(),
        )
