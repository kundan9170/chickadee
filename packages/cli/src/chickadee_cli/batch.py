"""`chickadee run`, `ask` and `resume`: one turn, followed to its end,
without a REPL. On the wire, like the REPL.

**A turn ends when the session is idle again after it began.** The turn route answers
before the person's words are written, so liveness can still read `idle` from before them.
The turn has begun when its user record arrives on the stream; from then, liveness
is read until it says `idle`, which it does only once the reply is
written too. The command then exits `0` whatever the run's terminal was, and prints that
terminal. A turn that ends any other way exits `2`: on an error (liveness reads `idle` with
its reason as `why`), stopped by Ctrl-C, a question nobody can answer, or a session
whose process died (`abandoned`).

**Questions** are answered three ways, and never silently:
- with `--yes`, by their defaults, through the automation token (an approval's default is
  `deny`);
- on a terminal, by the person, as the REPL asks them (`interaction.py`);
- otherwise not at all. The turn is stopped and the command exits `2`, naming `--yes`.

**`--json`** writes each record of the session from the turn's start to stdout, one per line,
as `/journal` serves it (`frames.encode_frame`), and each record of a run the turn opened
(`runs.RunFollower`); every frame names its scope. Every human line goes to stderr.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from chickadee_cli.interaction import Interaction
from chickadee_cli.interrupt import Interrupter
from chickadee_cli.render.stream import Verbosity, frame_lines, user_turn_text
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.repl import InputSource
from chickadee_cli.runs import RunFollower, pump
from chickadee_cli.surface import DONE, STOPPED
from chickadee_serve.frames import ProgressFrame, RecordFrame, TurnWorkspace, encode_frame
from chickadee_wire.client import Client
from chickadee_wire.events import Events, Frame
from chickadee_wire.models import Liveness

__all__ = ["POLL_S", "BatchIO", "follow", "prepend"]

POLL_S: Final = 0.1
DRAIN_S: Final = 5.0
"""How long the last records may take to arrive after liveness reads idle."""


@dataclass(frozen=True)
class BatchIO:
    """How one batch command talks to the person. `answerer` answers questions as `--yes`'s
    automation token when given; `source` asks the person on a terminal when given."""

    json: bool
    out: Callable[[str], None]
    err: Callable[[str], None]
    style: Style
    answerer: Client | None = None
    source: InputSource | None = None

    def say(self, text: str, tone: Tone = Tone.PLAIN) -> None:
        human = self.err if self.json else self.out
        for line in self.style.lines(text, tone):
            human(line)


def prepend(piped: str, goal: str) -> str:
    """Non-TTY stdin before the goal, a blank line between (`git diff | chickadee run
    "review this"`). Nothing piped leaves the goal as it is."""
    piped = piped.strip("\n")
    return f"{piped}\n\n{goal}" if piped else goal


def follow(
    client: Client,
    sid: str,
    how: BatchIO,
    *,
    after: int,
    send: str | None,
    workspace: TurnWorkspace | None = None,
    attachments: Sequence[tuple[str, bytes]] = (),
) -> int:
    """Send `send` as a turn (none for `resume`, which the caller started), with where its
    work runs when the person said (`workspace`), render the session from `after` until it is
    idle, and return the exit code."""
    return _Follow(client, sid, how, after, workspace, attachments).run(send)


class _Follow:
    def __init__(
        self,
        client: Client,
        sid: str,
        how: BatchIO,
        after: int,
        workspace: TurnWorkspace | None = None,
        attachments: Sequence[tuple[str, bytes]] = (),
    ) -> None:
        self._client = client
        self._workspace: TurnWorkspace | None = workspace
        self._attachments = tuple(attachments)
        """`--file`'s files, sent with the turn."""
        self._sid = sid
        self._how = how
        self._after = after
        self._seen = after
        self._seen_lock = threading.Lock()
        human = how.err if how.json else how.out
        self._questions = Interaction(how.answerer or client, sid, how.style, human)
        self._interrupter = Interrupter(client, sid)
        self._stopped = False
        self._answered: set[str] = set()
        self._out_lock = threading.Lock()
        self._runs = RunFollower(client, self._show, self._lost_run)
        self._done = threading.Event()
        self._begun = threading.Event()
        self._errored = False
        """An `error` frame was shown: its reason is said, so the verdict need not repeat it."""
        self._lost: Exception | None = None
        """What ended the session's stream early, if anything: the turn cannot be followed."""
        self._events: Events | None = None

    def run(self, send: str | None) -> int:
        if send is None:
            self._begun.set()  # resume: the run is already in flight
        self._events = self._client.session_events(self._sid, self._after)
        reader = threading.Thread(target=self._render, name="batch-events", daemon=True)
        reader.start()
        try:
            live = self._turn(send)
        except KeyboardInterrupt:
            # Outside the wait's own handling (the send, or the drain): stop the turn and go.
            self._interrupter.press()
            self._how.say("stopping the turn at its next boundary", Tone.BAD)
            return STOPPED
        finally:
            self._runs.close()
            # The reader stops here; its read ends when the caller closes the client, as the
            # REPL's does (an `Events` is closed only from the thread iterating it).
            self._done.set()
        return self._verdict(live)

    def _turn(self, send: str | None) -> Liveness:
        if send is not None:
            accepted = self._client.turn(
                self._sid, send, workspace=self._workspace, attachments=self._attachments
            )
            if accepted.queued:
                self._how.say("(queued behind the running turn)", Tone.DIM)
        live = self._until_idle()
        self._drain(live.last_seq)
        self._runs.wait(DRAIN_S)
        return live

    def _verdict(self, live: Liveness) -> int:
        if self._lost is not None:
            self._how.say(f"lost the session's stream: {self._lost}", Tone.BAD)
            return STOPPED
        if live.kind == "idle" and live.why is not None:
            # The turn ended on an error. Its frame is droppable, so the reason is said here
            # unless the frame already said it.
            if not self._errored:
                self._how.say(live.why, Tone.BAD)
            return STOPPED
        if live.kind != "idle":
            self._how.say(
                f"the session is {live.kind}: {live.why or 'no reason given'}", Tone.BAD
            )
            return STOPPED
        if live.outcome and self._runs.started:
            # Liveness names the session's last run; only a run this turn opened is its own.
            self._how.say(f"outcome: {live.outcome}", Tone.DIM)
        return STOPPED if self._stopped else DONE

    def _until_idle(self) -> Liveness:
        while True:
            try:
                live = self._client.liveness(self._sid)
                if live.kind == "idle" and self._begun.is_set():
                    return live
                if live.kind in ("abandoned", "unknown") or self._lost is not None:
                    return live
                if live.kind == "awaiting" and not self._answer():
                    return self._client.liveness(self._sid)
                time.sleep(POLL_S)
            except KeyboardInterrupt:
                self._stopped = True
                stop = self._interrupter.press()
                self._how.say(
                    "stopping now" if stop == "hard" else "stopping at the next boundary",
                    Tone.BAD,
                )

    def _answer(self) -> bool:
        """Answer the open question, or stop the turn when nobody can. False when stopped."""
        asked = self._questions.current
        if asked is None or asked.id in self._answered:
            return True  # its record, or its answer's, has not reached the reader yet
        if self._how.answerer is not None or self._how.source is not None:
            # `--yes` takes the default as the automation token; a person answers on the
            # terminal. A line naming no option leaves the question open, asked again.
            source = self._how.source
            line = source.read("? ") if self._how.answerer is None and source else ""
            closed = self._questions.answer(line)
            if closed is not None:
                self._answered.add(closed)
            return True
        self._how.say(
            f"question {asked.id} needs an answer and nobody is here to give it; rerun with "
            "--yes to take every default, or in a terminal to answer",
            Tone.BAD,
        )
        self._stopped = True
        self._client.interrupt(self._sid, hard=True)
        self._until_stopped()
        return False

    def _until_stopped(self) -> None:
        while self._client.liveness(self._sid).kind in ("working", "awaiting"):
            time.sleep(POLL_S)

    def _drain(self, last_seq: int) -> None:
        deadline = time.monotonic() + DRAIN_S
        while time.monotonic() < deadline:
            with self._seen_lock:
                if self._seen >= last_seq:
                    return
            time.sleep(POLL_S / 2)

    def _show(self, frame: Frame) -> None:
        """One frame of the session or of a run it opened, from either reader's thread."""
        human = self._how.err if self._how.json else self._how.out
        with self._out_lock:
            if self._how.json and isinstance(frame, RecordFrame):
                self._how.out(encode_frame(frame).decode("utf-8").rstrip("\n"))
            for line in frame_lines(frame, Verbosity.NORMAL, self._how.style):
                human(line)

    def _lost_run(self, run_id: str, why: str) -> None:
        with self._out_lock:
            self._how.say(f"lost run {run_id}'s stream: {why}", Tone.BAD)

    def _render(self) -> None:
        assert self._events is not None
        self._lost = pump(self._events, self._take, self._done)

    def _take(self, frame: Frame) -> None:
        if isinstance(frame, ProgressFrame) and frame.kind == "error":
            self._errored = True
        self._show(frame)
        if isinstance(frame, RecordFrame):
            if user_turn_text(frame) is not None:
                self._begun.set()
            self._questions.feed(frame)
            self._runs.feed(frame)
            with self._seen_lock:
                self._seen = max(self._seen, frame.seq)
