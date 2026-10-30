"""Help (`?`, `/help`): every key the screen has and every command, as the mockup's
table describes `/help`, "keys and commands". No picture draws it; it takes the palette's look.

**Drawn from the tables that bind them** (`keys.py`, `commands.COMMANDS`), so help never names
a key the screen does not have: a built key is lit, one not built yet is dimmed and names what
builds it, and a command no route serves is dimmed and named so. Whether a built key applies
this moment is the bar's to show.
"""

from __future__ import annotations

from typing import ClassVar, Final

from textual.binding import BindingType
from textual.content import Content

from chickadee_cli.commands import COMMANDS
from chickadee_cli.tui.keys import BAR, DIFF_BAR, DIFF_OFF_BAR, OFF_BAR, Key
from chickadee_cli.tui.overlays import CLOSE, Overlay, OverlayKey, bindings, fit
from chickadee_cli.tui.overlays.palette import NOT_SERVED

__all__ = ["Help"]

_KEY_W: Final = 8
_SECTIONS: Final = (
    ("THE SCREEN", (*(k for g in BAR for k in g), *OFF_BAR)),
    ("THE DIFF SCREEN", (*(k for g in DIFF_BAR for k in g), *DIFF_OFF_BAR)),
)


class Help(Overlay):
    WORD = "HELP"
    COMMAND = "?"
    KEYS: ClassVar[tuple[OverlayKey, ...]] = (
        OverlayKey(("up", "down"), "Up/Down", "scroll", ("scroll(-1)", "scroll(1)")),
        CLOSE,
    )
    BINDINGS: ClassVar[list[BindingType]] = bindings(KEYS)

    def right(self) -> str:
        return "keys and commands"

    def body(self, width: int) -> Content:
        c = self.colours
        lines: list[Content] = []
        for title, keys in _SECTIONS:
            lines.append(Content.styled(title, f"bold {c['fg']}"))
            lines.extend(self._key(key) for key in keys)
            lines.append(Content(""))
        lines.append(Content.styled("COMMANDS", f"bold {c['fg']}"))
        usage_w = max(len(command.usage) for command in COMMANDS.values())
        usage_w = min(usage_w, max(width // 2, 12))
        for command in COMMANDS.values():
            usage = fit(command.usage, usage_w)
            words = command.help if command.on_the_wire else f"{command.help} · {NOT_SERVED}"
            colour = c["fg"] if command.on_the_wire else c["dim"]
            lines.append(Content.assemble((f"{usage}  ", colour), (words, c["dim"])))
        return Content("\n").join(lines)

    def _key(self, key: Key) -> Content:
        c = self.colours
        built = key.action is not None
        why = "" if built else f"  (not built: {key.builds_in})"
        return Content.assemble(
            (fit(key.shown, _KEY_W), c["zeus"] if built else c["faint"]),
            (key.label, c["fg"] if built else c["dim"]),
            (why, c["dim"]),
        )

    def action_scroll(self, step: int) -> None:
        self.scroll.scroll_relative(y=step, animate=False)
