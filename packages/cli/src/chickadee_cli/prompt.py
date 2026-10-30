"""The REPL's input: what the prompt completes, what it keeps,
and its keys. `main.py` builds the `PromptSession` from `prompt_session`.

- **`/` completes commands,** each with its one-line help, filtered as typed, from
  `commands.COMMANDS`, the table `/help` prints; after a command's name, its fixed words
  (`Command.words`).
- **`@` completes paths,** fuzzily, inserting the plain path. The list is the server's
  (`Client.files`): what `glob` lists, so a completion never says an ignored or denied file
  exists. It is asked once, and again after each turn (`Files.stale`).
- **History is kept across sessions,** per repository, in its state directory, never in the
  repository: Up and Ctrl-R search it. A line the server says holds a secret value
  (`Client.holds_secret`) is never written, nor is one when the server cannot say.
- **Enter sends; Alt+Enter is a new line.** A pasted block is one message: the terminal's
  bracketed paste inserts it whole, its newlines included.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Final

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent

from chickadee_cli.commands import COMMANDS
from chickadee_wire.errors import WireError

__all__ = [
    "FILES_SHOWN",
    "REFRESH_S",
    "Files",
    "InputCompleter",
    "ScreenedHistory",
    "history_path",
    "prompt_session",
    "ranked_paths",
]

_log = logging.getLogger(__name__)

FILES_SHOWN: Final = 50
"""The most paths `@` offers at once: the best matches, never the whole tree."""
REFRESH_S: Final = 0.25
"""How often the toolbar is drawn again while nothing is typed: the live lines move."""


def history_path(state_dir: Path) -> Path:
    """The repository's REPL history: in its state directory, never in the repository."""
    return state_dir / "history"


class Files:
    """The workspace's paths for `@`, asked of the server once and again after `stale`.
    Read on the prompt's thread; `stale` is called from the REPL's reader."""

    def __init__(self, fetch: Callable[[], Sequence[str]]) -> None:
        self._fetch = fetch
        self._lock = threading.Lock()
        """Guards `_paths` and `_stale`."""
        self._paths: tuple[str, ...] = ()
        self._stale = True

    def stale(self) -> None:
        """A turn ended: the next `@` asks again."""
        with self._lock:
            self._stale = True

    def cached(self) -> tuple[str, ...]:
        """The paths as last read, never asking: for a reader that must not wait on the
        server (the TUI's input, read on its screen's loop), which calls `paths` elsewhere."""
        with self._lock:
            return self._paths

    def paths(self) -> tuple[str, ...]:
        with self._lock:
            if not self._stale:
                return self._paths
        try:
            fresh = tuple(self._fetch())
        except WireError:
            _log.debug("the workspace's files could not be read", exc_info=True)
            return self._paths
        with self._lock:
            self._paths, self._stale = fresh, False
            return fresh


def ranked_paths(query: str, paths: Iterable[str]) -> list[str]:
    """The paths `@query` offers, best first, at most `FILES_SHOWN`: one ranking for the REPL's
    prompt and the TUI's input."""
    ranked = sorted((rank, p) for p in paths if (rank := _fuzzy(query, p)) is not None)
    return [p for _rank, p in ranked[:FILES_SHOWN]]


def _fuzzy(query: str, path: str) -> tuple[int, int, int] | None:
    """A rank for `path` against `query`, lower first, or None when the query's letters are not
    in it in order: a match in the file's name before one in its folder, a run of letters
    before scattered ones, a short path before a long one."""
    lowered, wanted = path.lower(), query.lower()
    if not wanted:
        return (1, 0, len(path))
    name = lowered.rsplit("/", 1)[-1]
    if wanted in name:
        return (0, name.index(wanted), len(path))
    if wanted in lowered:
        return (1, lowered.index(wanted), len(path))
    at = 0
    for letter in wanted:
        at = lowered.find(letter, at)
        if at < 0:
            return None
        at += 1
    return (2, at, len(path))


class InputCompleter(Completer):
    """`/` for commands and their words, `@` for paths."""

    def __init__(self, files: Callable[[], Sequence[str]]) -> None:
        self._files = files

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if text.startswith("/") and "\n" not in text:
            yield from self._command(text)
            return
        word = text.rsplit(None, 1)[-1] if text and not text[-1].isspace() else ""
        if word.startswith("@"):
            yield from self._paths(word)

    def _command(self, text: str) -> Iterable[Completion]:
        name, space, arg = text.removeprefix("/").partition(" ")
        if not space:
            for command in COMMANDS.values():
                if command.name.startswith(name):
                    yield Completion(
                        f"/{command.name}",
                        start_position=-len(text),
                        display_meta=command.help,
                    )
            return
        command = COMMANDS.get(name)
        if command is None or " " in arg:
            return
        for word in command.words:
            if word.startswith(arg):
                yield Completion(word, start_position=-len(arg))

    def _paths(self, word: str) -> Iterable[Completion]:
        for path in ranked_paths(word[1:], self._files()):
            yield Completion(path, start_position=-len(word), display=path)


class ScreenedHistory(FileHistory):
    """`FileHistory` that writes no line `secret` says holds a secret value, and none when it
    cannot say: a credential typed or pasted at the prompt never reaches the file."""

    def __init__(self, path: Path, secret: Callable[[str], bool]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(str(path))
        self._secret = secret

    def store_string(self, string: str) -> None:
        try:
            if self._secret(string):
                return
        except WireError:
            return
        super().store_string(string)


def _keys() -> KeyBindings:
    keys = KeyBindings()

    @keys.add("enter")
    def _send(event: KeyPressEvent) -> None:
        event.current_buffer.validate_and_handle()

    @keys.add("escape", "enter")
    def _newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return keys


def prompt_session(
    *,
    toolbar: Callable[[], str],
    files: Files,
    history: Path,
    secret: Callable[[str], bool],
) -> PromptSession[str]:
    """The REPL's prompt: completion, screened history, the keys, and `toolbar` drawn below it,
    redrawn every `REFRESH_S` so the live lines move."""

    def bottom() -> ANSI | None:
        drawn = toolbar()
        return ANSI(drawn) if drawn else None

    return PromptSession(
        completer=InputCompleter(files.paths),
        complete_while_typing=True,
        history=ScreenedHistory(history, secret),
        enable_history_search=True,
        multiline=True,
        key_bindings=_keys(),
        bottom_toolbar=bottom,
        refresh_interval=REFRESH_S,
    )
