"""Zeus's input box (`ui/` `composer()`): the only input on the screen.

**Enter sends; Alt+Enter adds a line** (the mockup's key, the owner's choice). Most terminals
send Alt+Enter and Shift+Enter as a plain Enter: only those speaking the kitty keyboard
protocol tell them apart, so `Ctrl+J`, a line feed every terminal sends distinctly, adds a line
too, and the hint names it.

**History is the session's own words**: Up on the first line steps back through what the
person said to Zeus, Down on the last steps forward, `Ctrl+R` finds the last one holding what
is typed. A history of the records, not of this process, so it survives a restart.

**`?` in an empty box asks for help**: a TextArea takes every printable key
before any binding sees it (`TextArea.check_consume_key`), so the box itself asks
(`HelpRequested`); with words in it, `?` is a character like any other.

**`/` lists the REPL's commands** (`commands.COMMANDS`, its usage and one line each), filtered
as the person types; **`@` lists the workspace's paths**, as the REPL's prompt ranks them
(`prompt.ranked_paths`, over `/session/{sid}/files`). `Tab` completes the one
highlighted: a command's name, or the path in place of the `@word`. The paths are the ones last
read (`paths`), never fetched here: the screen reads them off its loop.

**On the front page, Up and Down choose a recent session and Enter opens it**, while
the box is empty and the screen says it offers some (`browses`): there the history is empty,
since landing means nothing was said. Words typed drop the choice (`Browsed(0)`), so Enter
sends them as ever.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

from textual import events
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.widgets import TextArea

from chickadee_cli.attach import dropped_path
from chickadee_cli.commands import COMMANDS, Command
from chickadee_cli.prompt import ranked_paths

__all__ = ["HINT", "NEWLINE_KEYS", "Composer", "ComposerChoice", "matches", "options"]

NEWLINE_KEYS: Final = frozenset({"alt+enter", "shift+enter", "ctrl+j"})
HINT: Final = "Enter send · Alt+Enter or ^J newline · / commands · @ files · ^V image"


def matches(text: str) -> tuple[Command, ...]:
    """The commands a line beginning `/` may be heading for, in the registry's order; none once
    it holds a space (the command is chosen, its arguments are being typed)."""
    if not text.startswith("/") or " " in text or "\n" in text:
        return ()
    prefix = text[1:]
    return tuple(c for name, c in COMMANDS.items() if name.startswith(prefix))


@dataclass(frozen=True)
class ComposerChoice:
    """One line of the completion list: what it shows, and what `Tab` puts from `start` on."""

    shown: str
    start: int
    insert: str


def options(text: str, paths: Sequence[str]) -> tuple[ComposerChoice, ...]:
    """What the list offers for `text`: the commands a `/` line may be heading for, or the
    paths an `@word` at its end may name."""
    commands = matches(text)
    if commands:
        return tuple(ComposerChoice(f"{c.usage}  {c.help}", 0, f"/{c.name} ") for c in commands)
    word = text.rsplit(None, 1)[-1] if text and not text[-1].isspace() else ""
    if not word.startswith("@"):
        return ()
    start = len(text) - len(word)
    return tuple(ComposerChoice(p, start, p) for p in ranked_paths(word[1:], paths))


class Composer(TextArea):
    """What the person types to Zeus. Sends `Submitted`; asks the pane to show or move the
    command list with `Completing`; asks the app for `$EDITOR` with `EditRequested`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+e", "edit", "edit in $EDITOR", show=False),
        Binding("ctrl+r", "search", "search what you said", show=False),
        Binding("ctrl+v", "image", "attach the clipboard's image", show=False),
    ]

    DEFAULT_CSS = """
    Composer { height: auto; max-height: 6; min-height: 1; border: none; padding: 0; }
    """

    @dataclass
    class Submitted(Message):
        text: str

    @dataclass
    class Completing(Message):
        options: tuple[ComposerChoice, ...]
        move: int = 0
        """+1 or -1 to move the highlight; 0 to show the list for what is typed."""

    @dataclass
    class Completed(Message):
        """Tab: complete to the highlighted option."""

    @dataclass
    class Browsed(Message):
        """Up or Down in the empty box on the front page: +1 or -1 moves the choice
        among the recent sessions, 0 drops it."""

        move: int

    @dataclass
    class Opened(Message):
        """Enter in the empty box on the front page: open the session chosen."""

    @dataclass
    class EditRequested(Message):
        text: str

    @dataclass
    class HelpRequested(Message):
        """`?` typed into the empty box."""

    @dataclass
    class ImageRequested(Message):
        """`Ctrl+V`: attach the clipboard's image. The app reads it off the loop."""

    @dataclass
    class FileDropped(Message):
        """A paste that is one existing file, as a terminal pastes a dragged one."""

        path: Path

    def __init__(self, history: Callable[[], Sequence[str]]) -> None:
        super().__init__(soft_wrap=True, show_line_numbers=False, compact=True, id="composer")
        self._history = history
        self.paths: Callable[[], Sequence[str]] = tuple
        """The workspace's paths as last read, for `@`; the screen sets it once connected."""
        self.browses: Callable[[], bool] = lambda: False
        """Whether the front page offers recent sessions to choose; the screen sets
        it."""
        self._back = 0
        """How far back Up has gone: 0 is the draft."""
        self._draft = ""

    def offered(self) -> tuple[ComposerChoice, ...]:
        return options(self.text, self.paths())

    @property
    def listing(self) -> bool:
        return bool(self.offered())

    async def _on_key(self, event: events.Key) -> None:
        if event.key in ("up", "down", "enter") and not self.text and self.browses():
            self._stop(event)
            if event.key == "enter":
                self.post_message(self.Opened())
            else:
                self.post_message(self.Browsed(-1 if event.key == "up" else 1))
        elif event.key == "enter":
            self._stop(event)
            self._send()
        elif event.character == "?" and not self.text:
            self._stop(event)
            self.post_message(self.HelpRequested())
        elif event.key in NEWLINE_KEYS:
            self._stop(event)
            self.insert("\n")
        elif event.key == "tab" and self.listing:
            self._stop(event)
            self.post_message(self.Completed())
        elif event.key == "tab":
            # With no command list open, `Tab` moves the focus on, as everywhere else: the box
            # never holds a tab character.
            self._stop(event)
            self.screen.focus_next()
        elif event.key in ("up", "down") and self.listing:
            self._stop(event)
            self.post_message(self.Completing(self.offered(), -1 if event.key == "up" else 1))
        elif event.key == "up" and self.cursor_location[0] == 0:
            self._stop(event)
            self._step(+1)
        elif event.key == "down" and self.cursor_location[0] == self.document.line_count - 1:
            self._stop(event)
            self._step(-1)
        else:
            await super()._on_key(event)

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self.post_message(self.Completing(self.offered()))
        if self.text:
            self.post_message(self.Browsed(0))

    @staticmethod
    def _stop(event: events.Key) -> None:
        event.prevent_default()
        event.stop()

    def _send(self) -> None:
        text = self.text
        self.load_text("")
        self._back, self._draft = 0, ""
        self.post_message(self.Submitted(text))

    def _step(self, by: int) -> None:
        said = list(self._history())
        if self._back == 0:
            self._draft = self.text
        back = max(0, min(self._back + by, len(said)))
        if back == self._back:
            return
        self._back = back
        self.load_text(self._draft if back == 0 else said[-back])
        self.move_cursor(self.document.end)

    def action_search(self) -> None:
        """The last thing said holding what was typed when the search began, then on each
        press the one before it: the draft is the query, and Down steps back to it."""
        said = list(self._history())
        if self._back == 0:
            self._draft = self.text
        for back in range(self._back + 1, len(said) + 1):
            if self._draft in said[-back]:
                self._back = back
                self.load_text(said[-back])
                self.move_cursor(self.document.end)
                return

    def action_edit(self) -> None:
        self.post_message(self.EditRequested(self.text))

    def action_image(self) -> None:
        self.post_message(self.ImageRequested())

    async def _on_paste(self, event: events.Paste) -> None:
        """A paste naming one existing file is given, not typed: the app attaches it, or
        types it back as a mention when it is not an attachment."""
        path = dropped_path(event.text)
        if path is None:
            await super()._on_paste(event)
            return
        event.prevent_default()
        event.stop()
        self.post_message(self.FileDropped(path))
