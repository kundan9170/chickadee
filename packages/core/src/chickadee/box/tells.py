"""A tell's delivery to the team: the one home of it.

Zeus's `tell` writes a `tell` line on the session log naming the run (`context/lines.TellLine`)
and wakes the run (`Tells.wake`). The run delivers it at its next boundary:

- **the worker's round is held**: a call in flight finishes, except a shell command,
  which is cut (`RoundToken`) and its step put back (`box/steps.StepRecord`); the calls after it
  in the same response are answered `not run` (`tools.loop.HELD_NOT_RUN`); no next model call
  is made, and the round reports `held`;
- **Athena reads it**: with the held round's facts, as the answer to the `delegate` or
  `review` that opened the round, or, with no round open, before her next model call.

`pending()` is the session's `tell` lines for this run, and for the runs it continues, that no
run of them delivered: a tell written while a run was paused is delivered by the run that goes
on with its work. `deliver()` journals each `tell_delivered` and clears the token;
`owed()` is the delivered ones Athena has not replied to.

**Concurrency.** `wake` runs on Zeus's thread and only sets an event; everything that reads the
journal runs on the run's own thread.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Final

from chickadee.journal.port import JournalPort
from chickadee.journal.projection import RunProjection
from chickadee.schema.journal import EventKind, TellDeliveredP, TellOutcome, TellReplyP

__all__ = [
    "CUT",
    "RoundToken",
    "Tells",
    "Told",
    "to_athena",
    "told_text",
    "unanswered",
]

CUT: Final = "interrupted for a message from the team lead"
"""Why a shell command a tell cut was answered without its result."""


@dataclass(frozen=True)
class Told:
    """A tell as the run reads it: its id, Zeus's words, and the person's turns it came from."""

    id: str
    text: str
    sources: tuple[int, ...]
    run_id: str


class Tells:
    """The run's side of a tell. `read` is the session's `tell` lines, newest last;
    `runs` the run journals of the runs this one continues (`continues.prior_runs`), so a tell
    one of them delivered is never delivered again."""

    def __init__(
        self,
        run_id: str,
        journal: JournalPort,
        read: Callable[[], tuple[Told, ...]],
        *,
        prior: tuple[str, ...] = (),
        runs: Callable[[str], RunProjection] | None = None,
        token: threading.Event | None = None,
        replied: Callable[[str, TellOutcome, str], None] | None = None,
    ) -> None:
        """`token` is the episode's (`BoxEpisode.told` sets it); `replied` tells Zeus of a
        reply."""
        self._run_id = run_id
        self._journal = journal
        self._read = read
        self._prior = prior
        self._runs = runs
        self._replied = replied
        self._token = token if token is not None else threading.Event()
        if self.pending():
            # Written while the run was paused, or before it opened: its first boundary.
            self._token.set()

    @property
    def token(self) -> threading.Event:
        """Set while a tell may be waiting: what holds the worker's round."""
        return self._token

    def wake(self) -> None:
        """A tell was written for the session's run: the next boundary looks."""
        self._token.set()

    def pending(self) -> tuple[Told, ...]:
        """The tells for this run, or for a run it continues, that no run of them delivered."""
        mine = {self._run_id, *self._prior}
        delivered = _delivered(self._journal.projection())
        for run_id in self._prior:
            if self._runs is not None:
                try:
                    delivered |= _delivered(self._runs(run_id))
                except FileNotFoundError:
                    continue
        return tuple(t for t in self._read() if t.run_id in mine and t.id not in delivered)

    def deliver(
        self, *, task: str | None = None, round_: int | None = None
    ) -> tuple[Told, ...]:
        """The pending tells, each journaled `tell_delivered` now, oldest first; the token
        cleared, so a tell written after this holds the next boundary."""
        self._token.clear()
        due = self.pending()
        for told in due:
            self._journal.append(
                EventKind.TELL_DELIVERED,
                TellDeliveredP(tell=told.id, task_id=task, round=round_ if task else None),
            )
        return due

    def delivered_at(self, task: str, round_: int) -> tuple[Told, ...]:
        """The tells delivered with `task`'s round `round_` (a resume reads them again)."""
        run = self._journal.projection()
        ids = [
            p.payload.tell
            for p in run.lines(TellDeliveredP)
            if (p.payload.task_id, p.payload.round) == (task, round_)
        ]
        found = {t.id: t for t in self._read()}
        return tuple(found[i] for i in ids if i in found)

    def owed(self) -> tuple[str, ...]:
        """The tells delivered in this run that Athena has not replied to."""
        return unanswered(self._journal.projection())

    def reply(self, tell: str, outcome: TellOutcome, text: str, *, call_id: str) -> None:
        """Athena's reply to a delivered tell: its `tell_reply` line, then Zeus told
        (a trigger-10 notice). A reply already written for `call_id` (a resume) is told
        again only if nothing told it, which the session's notice cannot say, so it is not."""
        run = self._journal.projection()
        if any(p.payload.call_id == call_id for p in run.lines(TellReplyP)):
            return
        sha = self._journal.put_blob(text.encode("utf-8"))
        self._journal.append(
            EventKind.TELL_REPLY,
            TellReplyP(tell=tell, outcome=outcome, text_sha=sha, call_id=call_id),
        )
        if self._replied is not None:
            self._replied(tell, outcome, text)


def unanswered(run: RunProjection) -> tuple[str, ...]:
    """The tells delivered in `run` with no reply: owed while it runs, and listed in its
    report when it ends with them, never silently."""
    replied = {p.payload.tell for p in run.lines(TellReplyP)}
    return tuple(
        p.payload.tell for p in run.lines(TellDeliveredP) if p.payload.tell not in replied
    )


def _delivered(run: RunProjection) -> set[str]:
    return {p.payload.tell for p in run.lines(TellDeliveredP)}


def to_athena(due: tuple[Told, ...]) -> str:
    """Zeus's messages with no worker round open: the same words, without a round's
    part, and what she owes them."""
    return (
        f"{told_text(due)}\n\nDecide what the work needs, and answer each message (reply): "
        "applied, planned, needs_design or declined."
    )


def told_text(due: tuple[Told, ...]) -> str:
    """Zeus's messages as Athena reads them, oldest first."""
    blocks: list[str] = []
    for told in due:
        turns = ", ".join(str(n) for n in told.sources)
        said = "\n".join(f"  {line}" for line in told.text.splitlines() or [""])
        # In her prompt's words: the project's names for its parts mean nothing to a model.
        blocks.append(f"Message ({told.id}), passed on from the user's turn {turns}:\n{said}")
    return "\n\n".join(blocks)


class RoundToken(threading.Event):
    """The token a worker's round hands its tools: set by the run's stop, or by a
    tell while a call it may cut runs (`cuttable`: a shell command that opened no network, or
    any command while the person is asked about it, `asking`). A read, a search, an edit or
    `tests_for` finishes; a networked command that started finishes, since its outcome would
    be unknown if cut. `tripped` says a tell cut the call in flight: its step is
    put back, and it is answered `interrupted`.

    Everything that waits on a token polls `is_set`, so `wait` polls too."""

    _POLL_S: Final = 0.05

    def __init__(self, stop: threading.Event | None, tells: threading.Event) -> None:
        super().__init__()
        self._stop = stop
        self._tells = tells
        self._cuttable = 0
        self.tripped = False

    def is_set(self) -> bool:
        if super().is_set() or (self._stop is not None and self._stop.is_set()):
            return True
        if self._cuttable and self._tells.is_set():
            self.tripped = True
            return True
        return False

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if end is not None and time.monotonic() >= end:
                return False
            time.sleep(self._POLL_S)
        return True

    @contextmanager
    def cuttable(self) -> Generator[None]:
        """While a call a tell may cut runs; `tripped` starts clear, and stays as it is in a
        span already open."""
        if not self._cuttable:
            self.tripped = False
        self._cuttable += 1
        try:
            yield
        finally:
            self._cuttable -= 1

    def begin_call(self) -> None:
        """A call starts: nothing has cut it yet. Every call clears it, a networked one that
        enters no cuttable span included, so a cut of an earlier call on the same token (hers
        spans a request) is never read as this one's."""
        if not self._cuttable:
            self.tripped = False

    def asking(self) -> AbstractContextManager[None]:
        """While the person is asked about a command (`shell.AskCut`): cuttable, a networked
        command's question too, since nothing has run."""
        return self.cuttable()
