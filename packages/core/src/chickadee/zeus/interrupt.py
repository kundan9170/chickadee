"""Queued input, and the soft and hard boundaries.

Three things get called "an interrupt", and they need different answers:

- **The person types while work is in flight.** That is input, not a stop. It is queued and
  delivered at the next turn boundary — never dropped, never injected into a call that is
  running. A message that arrives mid-run may contradict the approved design, which is
  why it waits for a boundary: Zeus reconciles it as an ordinary next turn.
- **A soft stop** takes effect at the next tool boundary. A call in flight completes; the
  calls after it in the same response are answered `CANCELLED` without running.
- **A hard stop** also sets the session's cancellation token, which every dispatch
  receives: a batch `shell` is reaped, `locate` stops between rungs, the model request in
  flight is abandoned.

Neither stop discards queued input. An interrupt pauses work; it never disposes of it,
and a person's words are the last thing it would be right to lose.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from chickadee.context.attachment import CheckedFile
from chickadee.schema.isolation import WorkspaceMode

__all__ = ["Interrupts", "Words"]


@dataclass(frozen=True)
class Words:
    """One input as the person sent it: the text, and where they said its work runs, if they
    did (`--worktree`, `--in-place`). Queued together, so a flag
    is never stamped on another turn's words."""

    text: str
    workspace: WorkspaceMode | None = None
    attachments: tuple[CheckedFile, ...] = ()
    """What the person sent with the words, already checked: queued with them, so a
    queued turn keeps its files, and recorded when the turn is delivered."""


class Interrupts:
    """One session's interrupt state. Written from the person's side (another thread, the
    terminal), read by the loop at its boundaries — so every part of it is thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: deque[Words] = deque()
        self._soft = threading.Event()
        self.cancel = threading.Event()
        """D95's token. Handed to every dispatch and to the model request; set only by a hard
        stop, and cleared when the next turn starts."""

    # -- the person's side ------------------------------------------------------------------

    def type(
        self,
        text: str,
        workspace: WorkspaceMode | None = None,
        attachments: tuple[CheckedFile, ...] = (),
    ) -> None:
        """Input typed while a turn runs. Queued as typed, whatever it says."""
        with self._lock:
            self._queue.append(Words(text, workspace, attachments))

    def interrupt(self, hard: bool = False) -> None:
        """Soft: stop at the next tool boundary. Hard: that, and the token is set now."""
        self._soft.set()
        if hard:
            self.cancel.set()

    # -- the loop's side --------------------------------------------------------------------

    def stopping(self) -> bool:
        """Whether a stop, of either kind, has been asked for since the turn began."""
        return self._soft.is_set() or self.cancel.is_set()

    def begin_turn(self) -> None:
        """A new turn starts unstopped. The queue is untouched: what was typed before this turn
        is delivered by it, not lost to it."""
        self._soft.clear()
        self.cancel.clear()

    def pending(self) -> tuple[str, ...]:
        """The queued inputs' text, oldest first."""
        with self._lock:
            return tuple(w.text for w in self._queue)

    def has_input(self) -> bool:
        """Whether anything is queued, without copying the queue to say so."""
        with self._lock:
            return bool(self._queue)

    def requeue(self, words: list[Words]) -> None:
        """Put undelivered input back at the front, in order: it was typed before anything
        queued since."""
        with self._lock:
            self._queue.extendleft(reversed(words))

    def next_input(self) -> Words | None:
        """The oldest queued input, removed — the loop takes it only at a turn boundary."""
        with self._lock:
            return self._queue.popleft() if self._queue else None
