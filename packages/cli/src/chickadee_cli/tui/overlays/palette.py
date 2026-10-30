"""The command palette (`^P`): the REPL's commands, filtered as the
person types.

**The REPL's list, whole and in its order**: `commands.COMMANDS`, filtered
by the composer's own rule (`composer.matches`), each with its one line and, where a bar key
does the same, that key (`keys.command_keys`). A command no route serves is drawn dimmed and
named so; `Enter` on it runs it all the same, and the REPL says it is not on the wire.

**`Enter` runs it as typed** (`Host.submit`): a bare `/budget` opens the budget overlay, as
typing it does. A command that cannot run without an argument (`/run <goal>`) is completed into
the composer instead, its arguments to type there; `Tab` completes any.
"""

from __future__ import annotations

import re
from typing import ClassVar, Final

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal
from textual.content import Content
from textual.widgets import Input, Static

from chickadee_cli.commands import Command
from chickadee_cli.tui.composer import matches
from chickadee_cli.tui.keys import command_keys
from chickadee_cli.tui.overlays import CLOSE, MOVE, Overlay, OverlayKey, bindings, fit, row

__all__ = ["NOT_SERVED", "Palette", "needs_argument"]

NOT_SERVED: Final = "not on the wire"
_NAME_W: Final = 14

_OPTIONAL: Final = re.compile(r"\[[^\[\]]*\]")


def needs_argument(command: Command) -> bool:
    """Whether the command's usage requires an argument: a `<...>` outside every `[...]`
    (`/run <goal>` does; `/journal [run]` and `/undo [... task <run> <task>...]` do not)."""
    usage = command.usage
    while _OPTIONAL.search(usage):
        usage = _OPTIONAL.sub("", usage)
    return "<" in usage


class Palette(Overlay):
    WORD = "COMMANDS"
    COMMAND = ""
    WIDTH = 62
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey((), "Type", "to filter"),
        MOVE,
        OverlayKey(("enter",), "Enter", "run", ("run",)),
        OverlayKey(("tab",), "Tab", "complete", ("complete",)),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def head(self) -> ComposeResult:
        with Horizontal(id="prompt"):
            yield Static(">")  # the composer's own prompt mark (`panes/zeus.py`)
            yield Input("/", id="filter")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#filter", Input).focus()

    def on_input_changed(self, _event: Input.Changed) -> None:
        self.cursor = 0
        self.redraw()

    def typed(self) -> str:
        if not self.is_mounted:
            return "/"
        text = self.query_one("#filter", Input).value.strip()
        return text if text.startswith("/") else f"/{text}"

    def shown(self) -> tuple[Command, ...]:
        return matches(self.typed())

    def rows(self) -> int:
        return len(self.shown())

    def chosen(self) -> Command | None:
        shown = self.shown()
        return shown[self.cursor] if 0 <= self.cursor < len(shown) else None

    def right(self) -> str:
        return str(len(self.shown()))

    def body(self, width: int) -> Content:
        c = self.colours
        keys = command_keys()
        lines: list[Content] = []
        for at, command in enumerate(self.shown()):
            name = f"/{command.name}"
            key = keys.get(name, "")
            words = command.help if command.on_the_wire else f"{command.help} · {NOT_SERVED}"
            room = max(width - _NAME_W - len(key) - 2, 1)
            text = f"{fit(name, _NAME_W)}{fit(words, room)} {key}"
            if at == self.cursor:
                lines.append(row(Content(text), width, c, chosen=True))
                continue
            colour = c["fg"] if command.on_the_wire else c["dim"]
            lines.append(
                Content.assemble(
                    (fit(name, _NAME_W), c["shell"] if command.on_the_wire else c["dim"]),
                    (f"{fit(words, room)} ", colour),
                    (key, c["zeus"]),
                )
            )
        if not lines:
            lines.append(Content.styled("no command starts so; /help lists them", c["dim"]))
        return Content("\n").join(lines)

    def applies(self, action: str) -> bool:
        return action not in ("run", "complete") or self.chosen() is not None

    def action_run(self) -> None:
        command = self.chosen()
        if command is None:
            return
        self.dismiss()
        if needs_argument(command):
            self.host.put_in_composer(f"/{command.name} ")
        else:
            self.host.submit(f"/{command.name}")

    def action_complete(self) -> None:
        command = self.chosen()
        if command is not None:
            self.dismiss()
            self.host.put_in_composer(f"/{command.name} ")
