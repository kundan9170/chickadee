"""The REPL: a line loop over one session, on the wire.

**A line is one of three things, decided once.** While a question is open, a line that does
not name a command is its answer (`interaction.py`), so a free-text answer may start with a
slash (`/etc/app`); a refused answer is said and the question asked again. Otherwise
`/name args` is a command (`commands.py`); `!cmd` is a shell line, which this build does not
run (a process the person starts belongs outside the harness's spawning rule, and no decision
says how yet); anything else is said to the session (`client.turn`).

**Steering is the engine's.** Words sent while a turn runs are queued by the engine and
delivered at the turn's next boundary; the REPL only says so.

**The person's words are shown once**, on the prompt line they typed them on. The session's
record of a turn this REPL sent is not printed back; one another client sent is.

**What the session does arrives on its own thread.** One `Events` iterator, from the
session's last seq, renders each frame (`render/stream.py`) and feeds each record to the
question fold, and each `episode` to `runs.RunFollower`, which streams that run's records
beside it. Output goes through one lock and to the transcript (`session.py`).

**Ctrl-C** during a turn stops it (`interrupt.py`); at an idle prompt it only clears the line.
Ctrl-D, or `/exit`, leaves; with a worktree whose work is not handed off, leaving first shows
`/handoff`'s review and asks, and Enter leaves it pending. A run still going
is paused, not ended, and opening the session again (`--continue`, `--session`) resumes it:
its permission question, if it was waiting on one, is asked again.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal, Protocol

from chickadee_cli.attach import Staging, ready
from chickadee_cli.commands import (
    Context,
    Exit,
    NoAnswer,
    dispatch,
    is_command,
    resume,
    review,
)
from chickadee_cli.interaction import Interaction, answering
from chickadee_cli.interrupt import Interrupter, stop_if_working
from chickadee_cli.render.live import LiveView
from chickadee_cli.render.status import StatusLine
from chickadee_cli.render.stream import (
    Verbosity,
    attachment_lines,
    frame_lines,
    user_turn,
    user_turn_text,
)
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.runs import RunFollower, pump
from chickadee_cli.session import Transcript, transcript_path
from chickadee_serve.frames import ProgressFrame, RecordFrame, TurnWorkspace
from chickadee_wire.client import Client
from chickadee_wire.errors import Unanswered, WireError
from chickadee_wire.events import Events, Frame

__all__ = [
    "NO_ANSWER",
    "PROMPT",
    "SHELL_NOT_RUN",
    "InputSource",
    "LineKind",
    "Repl",
    "classify",
    "handle_line",
]

PROMPT: Final = "> "
ASKING_PROMPT: Final = "? "
"""The prompt while a question waits: the next line is its answer."""

SHELL_NOT_RUN: Final = "shell lines are not run in this build"
NO_ANSWER: Final = "no answer given: nothing more was sent"
"""Said when Ctrl-C or Ctrl-D ends a command's question (`commands.NoAnswer`): the command
stopped there, and a question the server holds stays open, answered at the prompt."""

type LineKind = Literal["command", "answer", "shell", "words", "nothing"]


def classify(line: str, *, asking: bool) -> LineKind:
    """What a line the person typed is, decided once (the module's first rule): with a
    question open, a line that does not name a command is its answer, so a free-text answer
    may start with a slash; otherwise `/name` is a command, `!cmd` a shell line, and anything
    else words to the session. Public so every surface a person types into reads a line the
    same way (the TUI's composer)."""
    stripped = line.strip()
    if stripped.startswith("/") and (not asking or is_command(stripped)):
        return "command"
    if asking:
        return "answer"
    if stripped.startswith("!"):
        return "shell"
    return "words" if stripped else "nothing"


def handle_line(ctx: Context, interaction: Interaction, line: str) -> None:
    """One line, read once (`classify`), and done: a command run, a question answered, a shell
    line declined, or words told to the session. Every surface a person types into does
    this, the REPL and the TUI alike."""
    stripped = line.strip()
    match classify(line, asking=interaction.typed is not None):
        case "command":
            dispatch(ctx, stripped)
        case "answer":
            try:
                interaction.answer(line)
            except WireError as exc:
                ctx.say(f"the answer was refused: {exc.message}", Tone.BAD)
        case "shell":
            ctx.say(SHELL_NOT_RUN, Tone.DIM)
        case "words":
            ctx.tell(stripped)
        case "nothing" if ctx.staging.items():
            # Files with no words: a turn's text is never empty, so they are said to be
            # what was sent.
            ctx.tell(ATTACHED_ONLY)
        case "nothing":
            pass


ATTACHED_ONLY: Final = "(attached)"
"""The words of a turn sent with files and nothing typed."""


class InputSource(Protocol):
    """Where the person's lines come from. `read` raises `EOFError` at the end (Ctrl-D) and
    `KeyboardInterrupt` for Ctrl-C. `default` is text put in the line to edit (a terminal's),
    and a source that cannot edit a line ignores it."""

    def read(self, prompt_text: str, default: str = "") -> str: ...


class Repl:
    """One session's conversation. `run` returns the exit code (`0`)."""

    def __init__(
        self,
        client: Client,
        session_id: str,
        *,
        source: InputSource,
        write: Callable[[str], None],
        style: Style,
        state_dir: Path | None = None,
        verbosity: Verbosity = Verbosity.NORMAL,
        resume_paused: bool = False,
        turn_ended: Callable[[], None] | None = None,
        workspace: TurnWorkspace | None = None,
    ) -> None:
        """`resume_paused`: the session was opened again, so a run left paused continues;
        `turn_ended` is called as each turn's end arrives: the `@` list is asked again;
        `workspace` (`--worktree`, `--in-place`) goes with the first turn the server accepts,
        which moves the session; later words move it themselves."""
        self.client = client
        self.style = style
        self._sid = session_id
        self.staging = Staging(client, session_id)
        """The files `/attach` staged for the next line: its prompt shows how many."""
        self._source = source
        self._write = write
        self._verbosity = verbosity
        self._resume_paused = resume_paused
        self._turn_ended = turn_ended
        self._workspace: TurnWorkspace | None = workspace
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._transcript = (
            Transcript(transcript_path(state_dir, session_id))
            if state_dir is not None
            else None
        )
        # What a turn is doing, as it does it, and the status line; a terminal's only.
        self._live = LiveView(self._emit, style, lambda: self._verbosity) if style.tty else None
        self._status = (
            StatusLine(
                style, lambda: client.budget(session_id), lambda: client.jobs(session_id)
            )
            if style.tty
            else None
        )
        self._interaction = Interaction(client, session_id, style, self._emit)
        self._interrupter = Interrupter(client, session_id)
        self._runs = RunFollower(client, self._render, self._lost_run)
        self._sent: deque[str] = deque()
        """Words this REPL sent whose record has not arrived; guarded by `_lock`."""
        self._events: Events | None = None
        self._reader: threading.Thread | None = None
        self._put_back = ""
        """Words to edit at the next prompt: a rewound turn's. Read and set on the REPL's
        own thread only."""

    # -- the loop ----------------------------------------------------------------------------

    def run(self) -> int:
        self._attach()
        if self._status is not None:
            self._status.refresh(now=True)
            self._status.recount()
        self.say(f"session {self._sid}; /help for commands, Ctrl-D to leave", Tone.DIM)
        if self._resume_paused:
            self._resume()
        try:
            while True:
                try:
                    line = self._source.read(PROMPT, self._put_back)
                    self._put_back = ""
                except KeyboardInterrupt:
                    self._ctrl_c()
                    continue
                except EOFError:
                    break
                if self._transcript is not None:
                    self._transcript.write(PROMPT + line)
                try:
                    self.handle(line)
                except Exit:
                    break
                except NoAnswer:
                    self.say(NO_ANSWER, Tone.DIM)
            self._leave()
        finally:
            self.close()
        return 0

    def _resume(self) -> None:
        try:
            resume(self, attaching=True)
        except WireError as exc:
            self.say(f"resume: {exc.message}", Tone.BAD)
        except NoAnswer:
            self.say(NO_ANSWER, Tone.DIM)

    def _leave(self) -> None:
        """The handoff question, once, as the person leaves."""
        try:
            review(self, leaving=True)
        except WireError as exc:
            self.say(f"/handoff: {exc.message}", Tone.BAD)
        except Exception as exc:
            self.say(f"/handoff: the session could not be asked ({exc})", Tone.BAD)

    def handle(self, line: str) -> None:
        handle_line(self, self._interaction, line)

    def close(self) -> None:
        self._stopping.set()
        self._runs.close()
        if self._transcript is not None:
            self._transcript.close()

    # -- what commands may do (commands.Context) ---------------------------------------------

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    def set_verbosity(self, verbosity: Verbosity) -> None:
        self._verbosity = verbosity

    def say(self, text: str, tone: Tone = Tone.PLAIN, *, indent: str = "") -> None:
        for line in self.style.lines(text, tone, indent=indent):
            self._emit(line)

    def show(self, lines: Sequence[str]) -> None:
        for line in lines:
            self._emit(line)

    def ask(self, prompt: str) -> str:
        try:
            return self._source.read(prompt)
        except (EOFError, KeyboardInterrupt):
            raise NoAnswer from None  # never "", which would take the question's default

    def answering[T](self, call: Callable[[], T]) -> T:
        """`call()`, while the person answers each question it raises (an undo's conflicts, a
        fork's restore or keep), as they answer a turn's: one line each."""
        return answering(call, self._interaction, self.ask)

    def put_back(self, text: str) -> None:
        self._put_back = text

    def tell(self, text: str) -> None:
        staged = ready(self.staging, self.ask)
        if isinstance(staged, str):
            self.say(staged, Tone.BAD)
            self.put_back(text)
            return
        # Noted before the send: the record can reach the reader before `turn` returns.
        with self._lock:
            self._sent.append(text)
        try:
            accepted = self.client.turn(
                self._sid,
                text,
                workspace=self._workspace,
                attachments=Staging.files(staged),
            )
        except WireError as exc:
            if not (isinstance(exc, Unanswered) and exc.maybe_delivered):
                # Refused, or never sent: no record of these words is coming. One that may
                # have arrived stays noted, so its record shows as this person's,
                # and its files are not staged again, since they may have gone with it.
                with self._lock:
                    self._sent.remove(text)
                self.staging.put_back(staged)
            self.say(exc.message, Tone.BAD)
            return
        # Said once: a flag typed at the start never overrides a later word.
        self._workspace = None
        if accepted.queued:
            self.say("(queued: delivered at the running turn's next boundary)", Tone.DIM)

    def clear(self) -> None:
        if self.style.tty:
            self._emit("\x1b[2J\x1b[H")

    def follow_run(self, run_id: str, after: int) -> None:
        self._runs.follow(run_id, after)

    # -- output and events -------------------------------------------------------------------

    def _emit(self, line: str) -> None:
        if self._stopping.is_set():
            return
        with self._lock:
            self._write(line)
            if self._transcript is not None:
                self._transcript.write(line)

    def _attach(self) -> None:
        """Fold what the session already holds, then follow it from there: an open question
        is shown at once, and nothing already seen is printed again."""
        history = self.client.session_journal(self._sid)
        self._interaction.fold(history)
        after = history[-1].seq if history else 0
        episodes = [r for r in history if r.kind == "episode"]
        if episodes and self.client.liveness(self._sid).kind in ("working", "awaiting"):
            # A run already under way when this REPL joined: followed from its start, since
            # its records so far are what the person has not seen.
            self._runs.feed(episodes[-1])
        self._events = self.client.session_events(self._sid, after)
        self._reader = threading.Thread(target=self._follow, name="repl-events", daemon=True)
        self._reader.start()

    def _follow(self) -> None:
        assert self._events is not None
        lost = pump(self._events, self._take, self._stopping)
        if lost is not None:
            self.say(f"lost the session's stream: {lost}", Tone.BAD)

    def _take(self, frame: Frame) -> None:
        self._render(frame)
        if isinstance(frame, RecordFrame):
            self._interaction.feed(frame)
            self._runs.feed(frame)

    def _render(self, frame: Frame) -> None:
        if isinstance(frame, RecordFrame) and self._echo_of_own(frame):
            # The person sees the words they typed, but not the ids their files were given.
            own = user_turn(frame)
            for line in attachment_lines(own, self.style) if own is not None else ():
                self._emit(line)
            return
        frame = self._lived(frame)
        for line in frame_lines(frame, self._verbosity, self.style):
            self._emit(line)

    def _lived(self, frame: Frame) -> Frame:
        """`frame` through the live view and the status line, which keep what they draw from
        it; Zeus's reply comes back less what was already printed of it."""
        ended = isinstance(frame, ProgressFrame) and frame.kind in ("answer", "error")
        if ended and self._turn_ended is not None:
            self._turn_ended()
        if self._live is None or self._status is None:
            return frame
        if isinstance(frame, ProgressFrame):
            self._live.progress(frame)
            self._status.progress(frame)
            return frame
        if isinstance(frame, RecordFrame):
            self._status.record(frame)
            return self._live.record(frame)
        return frame

    def toolbar(self) -> str:
        """The prompt's bottom toolbar: the open question, if one
        waits, then the live lines, then the status line. Empty to a pipe."""
        if self._live is None or self._status is None:
            return ""
        waiting = self._interaction.waiting()
        asked = [self.style.verbatim(waiting, Tone.ASK)] if waiting is not None else []
        return "\n".join([*asked, *self._live.bottom(), self._status.line()])

    def prompt_mark(self) -> str:
        """The prompt to show now: `? ` while a question waits, since the next line answers it,
        else `> `, with `[2 attached]` before it while `/attach` has staged files. Read
        at every redraw, so it changes while a line is being typed."""
        if self._interaction.typed is not None:
            return ASKING_PROMPT
        n = len(self.staging.items())
        return f"[{n} attached] {PROMPT}" if n else PROMPT

    def _echo_of_own(self, frame: RecordFrame) -> bool:
        """Whether `frame` records words this REPL sent and the person already sees."""
        words = user_turn_text(frame)
        with self._lock:
            if words is None or words not in self._sent:
                return False
            self._sent.remove(words)
            return True

    def _lost_run(self, run_id: str, why: str) -> None:
        self.say(f"lost run {run_id}'s stream: {why}; /status shows its board", Tone.BAD)

    def _ctrl_c(self) -> None:
        stop = stop_if_working(self.client, self._sid, self._interrupter)
        if stop is None:
            self.say("(Ctrl-D or /exit to leave)", Tone.DIM)
            return
        if stop == "hard" and self._live is not None:
            self._live.stopped()
        if stop == "hard":
            self.say("stopping now", Tone.BAD)
        else:
            self.say("stopping at the next boundary; Ctrl-C again to stop now", Tone.DIM)
