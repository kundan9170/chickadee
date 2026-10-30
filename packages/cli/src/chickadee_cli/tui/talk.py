"""What a line the person types does: the screen's `commands.Context`,
so the REPL's slash commands, its question answering and its sending run here unchanged
("slash commands work exactly as in the REPL").

**Read and done as the REPL does it** (`repl.handle_line`, one home for both): with a
question open, a line that is not a command is its answer (`interaction.Interaction`, the
REPL's own); else `/name` is a command (`commands.dispatch`), `!cmd` a shell line this build
does not run, and anything else words to Zeus.

**On one thread, in order.** A command may wait on the server or on the person (`ask`, an
undo's conflicts), so lines are handled on a thread of their own, one after another, never on
the screen's loop. Everything they change on screen goes through `Screen` methods, which the
app runs on its loop; nothing here touches the feed.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from typing import Final, Protocol

from chickadee_cli.attach import Staging, ready
from chickadee_cli.commands import Exit, NoAnswer, resume
from chickadee_cli.interaction import Interaction, answering
from chickadee_cli.interrupt import Interrupter, stop_if_working
from chickadee_cli.render.stream import Verbosity
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.repl import handle_line
from chickadee_wire.client import Client
from chickadee_wire.errors import Unanswered, WireError
from chickadee_wire.questions import OpenQuestion

__all__ = ["OVERLAYS", "QUEUED", "READ_ONLY", "SCREEN_COMMANDS", "Screen", "Talk"]

QUEUED: Final = "(queued: delivered at the running turn's next boundary)"
"""The REPL's words for a line sent while a turn runs (`repl.tell`)."""
READ_ONLY: Final = "this is a replay: read-only (30-serve §7); it takes answers only"


class Screen(Protocol):
    """What a line may do to the screen; the app runs each on its own loop."""

    def lines(self, lines: Sequence[str]) -> None:
        """Lines in Zeus's pane, styled as the REPL styles them for a terminal."""
        ...

    def sent(self, text: str) -> None: ...

    def unsent(self, text: str) -> None: ...

    def prompt(self, text: str | None) -> None:
        """The composer's prompt while a command asks something (None: back to Zeus)."""
        ...

    def put_back(self, text: str) -> None: ...

    def verbosity(self, verbosity: Verbosity) -> None: ...

    def clear(self) -> None: ...

    def leave(self) -> None: ...

    def open_diff(self) -> None:
        """The diff screen (`^D`), for a bare `/diff`."""
        ...

    def open_overlay(self, name: str) -> None:
        """An overlay (`overlays/`) by its name in `OVERLAYS`."""
        ...

    def toggle_fullscreen(self) -> None:
        """The header to one line and back (`^F`), for a bare `/fullscreen`."""
        ...


OVERLAYS: Final = ("budget", "handoff", "help", "resume", "rung", "undo")
"""The commands whose bare form opens an overlay (the mockup's table: `/budget`,
`/handoff`, `/resume`, `/rung`, `/undo`; `/help` is "keys and commands")."""

SCREEN_COMMANDS: Final[dict[str, Callable[[Screen], None]]] = {
    "/diff": lambda screen: screen.open_diff(),
    "/fullscreen": lambda screen: screen.toggle_fullscreen(),
    **{f"/{name}": lambda screen, name=name: screen.open_overlay(name) for name in OVERLAYS},
}
"""The lines the screen answers with a view of its own, before the REPL's `dispatch`
(the mockup's table: "reached by ^D or /diff", and each overlay's command). With
arguments, a command is the REPL's (`/budget 5`, `/diff <run>`); and an overlay's own actions
reach the REPL past this table (`Talk.command`), so `/resume` from the sessions overlay
continues the run rather than opening it again."""


class Talk:
    """One session's typed lines. `submit` from the screen's loop; the rest on `talk`'s own
    thread. `client` is shared with the screen's readers: httpx's client is thread-safe."""

    def __init__(
        self,
        client: Client,
        session_id: str,
        screen: Screen,
        interaction: Interaction,
        *,
        follow: Callable[[str, int], None],
        resume_paused: bool = False,
    ) -> None:
        """`follow(run_id, after)`: the screen's readers follow a run no `episode` names
        (`commands.Context.follow_run`). `resume_paused`: the session was opened again, so a
        run left paused continues, as the REPL does."""
        self.client = client
        self.style = Style(tty=True)
        self._sid = session_id
        self.staging = Staging(client, session_id)
        """The files on the draft: the composer's chips, `/attach`'s list. The screen
        sets `staging.changed` to redraw the chips."""
        self._screen = screen
        self._interaction = interaction
        self._verbosity = Verbosity.NORMAL
        self._lines: queue.Queue[tuple[str, bool] | None] = queue.Queue()
        """Each line and whether it was typed (so the screen's own commands may take it), in
        order."""
        self._answers: queue.Queue[str | None] = queue.Queue()
        """Each answer typed while `ask` waits; None once the screen closed."""
        self._asking = threading.Event()
        """Set while `ask` waits: the next line typed is its answer, not a new line."""
        self._interrupter = Interrupter(client, session_id)
        self._follow = follow
        self._resume_paused = resume_paused
        self._thread = threading.Thread(target=self._run, name="tui-talk", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._lines.put(None)
        self._answers.put(None)  # a waiting `ask` raises `NoAnswer`, never reads an answer

    def submit(self, line: str) -> None:
        """A line the person typed: an `ask`'s answer when one waits, else the next line."""
        if self._asking.is_set():
            self._answers.put(line)
        else:
            self._lines.put((line, True))

    def command(self, line: str) -> None:
        """An overlay's action, a slash command: to the REPL's `dispatch` in turn, past the
        screen's own commands, and never taken as an `ask`'s answer."""
        self._lines.put((line, False))

    def interrupt(self) -> None:
        """`Esc`: stop the turn in flight, as the REPL's Ctrl-C does (`stop_if_working`); a
        second press soon after stops it hard. Off the loop: it asks the server twice."""
        threading.Thread(target=self._stop, name="tui-interrupt", daemon=True).start()

    def _stop(self) -> None:
        try:
            stop = stop_if_working(self.client, self._sid, self._interrupter)
        except WireError as exc:
            self.say(f"the interrupt was refused: {exc.message}", Tone.BAD)
            return
        if stop == "hard":
            self.say("stopping now", Tone.BAD)
        elif stop == "boundary":
            self.say("stopping at the next boundary; Esc again to stop now", Tone.DIM)

    def answer(
        self,
        asked: OpenQuestion,
        choice: str,
        note: str | None,
        text: str | None,
        failed: Callable[[], None],
    ) -> None:
        """A modal's answer, sent through the REPL's own `Interaction.send` on a thread of its
        own: never on the screen's loop (it asks the server), and never queued behind a line
        whose command may be waiting on the person (`ask`). A refusal is said, and `failed`
        runs so the modal's keys work again."""

        def send() -> None:
            try:
                self._interaction.send(asked, choice, note, text=text)
            except WireError as exc:
                self.say(f"{asked.id}: {exc.message}", Tone.BAD)
                failed()

        threading.Thread(target=send, name="tui-answer", daemon=True).start()

    @property
    def replay(self) -> bool:
        server = self.client.server
        return server is not None and server.replay

    # -- the thread ------------------------------------------------------------------------

    def _run(self) -> None:
        if self._resume_paused and not self.replay:
            try:
                resume(self, attaching=True)
            except WireError as exc:
                self.say(f"resume: {exc.message}", Tone.BAD)
            except NoAnswer:
                return  # closed while resuming asked
        while (item := self._lines.get()) is not None:
            line, typed = item
            try:
                self.handle(line, typed=typed)
            except NoAnswer:
                return  # the screen closed while the line's command asked
            except Exit:
                self._screen.leave()
                return
            except WireError as exc:
                self.say(exc.message, Tone.BAD)
            except Exception as exc:  # a command that broke says so; the screen goes on
                self.say(f"that line could not be handled: {exc}", Tone.BAD)

    def handle(self, line: str, *, typed: bool = True) -> None:
        """One line, done as the REPL does it (`repl.handle_line`), but for the commands the
        screen draws itself (`SCREEN_COMMANDS`), when typed: a bare `/diff` shows the diff
        screen; `/diff <run>` is still the REPL's, naming that run's files."""
        if typed and line.strip() in SCREEN_COMMANDS:
            SCREEN_COMMANDS[line.strip()](self._screen)
            return
        handle_line(self, self._interaction, line)

    # -- commands.Context ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def verbosity(self) -> Verbosity:
        return self._verbosity

    def set_verbosity(self, verbosity: Verbosity) -> None:
        self._verbosity = verbosity
        self._screen.verbosity(verbosity)

    def say(self, text: str, tone: Tone = Tone.PLAIN, *, indent: str = "") -> None:
        self._screen.lines(self.style.lines(text, tone, indent=indent))

    def show(self, lines: Sequence[str]) -> None:
        self._screen.lines(lines)

    def ask(self, prompt: str) -> str:
        """`prompt` in the composer; the person's next line is the answer. `NoAnswer` when the
        screen closes first."""
        self._asking.set()
        self._screen.prompt(prompt)
        try:
            answer = self._answers.get()
            if answer is None:
                self._answers.put(None)  # and every later `ask` of this command
                raise NoAnswer
            return answer
        finally:
            self._asking.clear()
            self._screen.prompt(None)

    def answering[T](self, call: Callable[[], T]) -> T:
        return answering(call, self._interaction, self.ask)

    def put_back(self, text: str) -> None:
        self._screen.put_back(text)

    def tell(self, text: str) -> None:
        if self.replay:
            self.say(READ_ONLY, Tone.DIM)
            return
        staged = ready(self.staging, self.ask)
        if isinstance(staged, str):
            self.say(staged, Tone.BAD)
            self.put_back(text)
            return
        # Pending before the send: the record can reach the screen before `turn` returns.
        self._screen.sent(text)
        try:
            accepted = self.client.turn(self._sid, text, attachments=Staging.files(staged))
        except WireError as exc:
            self._screen.unsent(text)
            if not (isinstance(exc, Unanswered) and exc.maybe_delivered):
                # A turn that may have arrived took its files with it: staged again, they
                # would go a second time with the next message, as the REPL also holds.
                self.staging.put_back(staged)
            self.say(exc.message, Tone.BAD)
            return
        if accepted.queued:
            self.say(QUEUED, Tone.DIM)

    def clear(self) -> None:
        self._screen.clear()

    def follow_run(self, run_id: str, after: int) -> None:
        # From its first record, not `after`: the panes fold the whole run (the board, the
        # spend), and the feed drops a record it has folded.
        del after
        self._follow(run_id, 0)
