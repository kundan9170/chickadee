"""Which panes the screen shows, by its size, the role in focus and the zoom: plain functions of
those, so each layout is tested without a terminal; the app applies
them.

- **100 columns or more:** layout E, every pane.
- **80 to 99:** stacked (the owner's choice): Zeus full width on top, and under him one of the
  CODE, ATHENA and WORKER panes as tabs, following the focus (CODE while Zeus has it, Athena
  on F2, the worker on F3).
- **Below 80:** one pane at a time: Zeus, Athena, the worker or CODE by F1 to F4; the diff on
  `^D`.
- **Below 60 by 20:** too small to draw legibly: one line saying so, and where the line view
  is.
- **Zoomed** (`Ctrl+Up`, `Ctrl+Down` back): the focused pane alone, in any of them.

**An event is never missed**: whatever a layout hides, the ticker names the latest entry
of each hidden role, drawn by that role's own pane (`athena_line`, `worker_line`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from textual.content import Content

from chickadee_cli.render.stream import preview
from chickadee_cli.tui.model import AthenaKind, TuiState, ZeusKind
from chickadee_cli.tui.names import PANE_WORDS, ROLE_WORDS
from chickadee_cli.tui.panes.athena import athena_line
from chickadee_cli.tui.panes.worker import worker_line
from chickadee_cli.tui.theme import Glyphs
from chickadee_wire.models import AssistantTurnP

__all__ = [
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SINGLE_BELOW",
    "STACKED_BELOW",
    "TOO_SMALL",
    "Layout",
    "Mode",
    "layout",
    "mode_for",
    "tab_line",
    "ticker_line",
]

STACKED_BELOW: Final = 100
""""below 100 columns the right column becomes tabs under the Zeus pane"."""
SINGLE_BELOW: Final = 80
""""below 80, one pane at a time"."""
MIN_WIDTH: Final = 60
MIN_HEIGHT: Final = 20
"""Below these not even one pane, its title and the composer fit legibly (the owner's 60x20)."""
TOO_SMALL: Final = (
    f"the terminal is too small for the screen ({MIN_WIDTH}x{MIN_HEIGHT} at least); widen "
    "it, or use chickadee for the line view"
)

ROLES: Final = ("zeus", "athena", "worker")
"""The roles a ticker can name; the CODE pane is the worker's work, not a role."""


class Mode(StrEnum):
    WIDE = "wide"
    STACKED = "stacked"
    SINGLE = "single"
    TOO_SMALL = "too-small"


def mode_for(width: int, height: int) -> Mode:
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return Mode.TOO_SMALL
    if width < SINGLE_BELOW:
        return Mode.SINGLE
    if width < STACKED_BELOW:
        return Mode.STACKED
    return Mode.WIDE


@dataclass(frozen=True)
class Layout:
    panes: frozenset[str]
    """The frames shown: `zeus`, `diff`, `background`, `athena`, `worker`."""
    tabs: tuple[str, ...] = ()
    """The tab line's names, the shown one first-class (`Layout.tab`); none in layout E."""
    tab: str | None = None
    ticker: tuple[str, ...] = ()
    """The roles whose latest entry the ticker names: those the layout hides."""


_CENTRE: Final = {"diff": ("background",), "background": ("diff",)}
"""The centre column's two panes, shown together: BACKGROUND sits under CODE while anything
runs, and the app hides it when nothing does."""


def layout(mode: Mode, focus: str, zoomed: str | None) -> Layout:
    """The layout for `mode`, with `focus` the role in focus (`zeus` while the composer has it)
    and `zoomed` the pane zoomed, if any."""
    if mode is Mode.TOO_SMALL:
        return Layout(frozenset())
    if zoomed is not None:
        return Layout(frozenset({zoomed}), ticker=_hidden({zoomed}))
    if mode is Mode.WIDE:
        return Layout(frozenset({"zeus", "diff", "background", "athena", "worker"}))
    if mode is Mode.STACKED:
        tab = focus if focus in ("athena", "worker") else "diff"
        return Layout(
            frozenset({"zeus", tab, *_CENTRE.get(tab, ())}),
            ("diff", "athena", "worker"),
            tab,
            _hidden({"zeus", tab}),
        )
    shown = focus if focus in (*ROLES, *_CENTRE) else "zeus"
    return Layout(frozenset({shown, *_CENTRE.get(shown, ())}), ROLES, shown, _hidden({shown}))


def _hidden(shown: set[str]) -> tuple[str, ...]:
    return tuple(r for r in ROLES if r not in shown)


_TAB_WORDS: Final = {name: word.upper() for name, word in (PANE_WORDS | ROLE_WORDS).items()}
"""Each pane's tab, as its frame's title names it (`names.ROLE_WORDS`)."""


def tab_line(lay: Layout, colours: dict[str, str]) -> Content:
    """`[CODE] · ATHENA · WORKER`, the shown tab marked, with the key that shows each."""
    keys = {"zeus": "F1", "athena": "F2", "worker": "F3", "diff": "F4", "background": "F5"}
    parts: list[Content] = []
    for name in lay.tabs:
        word = _TAB_WORDS[name]
        if name == lay.tab:
            parts.append(Content.styled(f"[{word}]", f"bold {colours[_colour(name)]}"))
        else:
            parts.append(Content.styled(f"{word} {keys[name]}", colours["dim"]))
    return Content.styled(" · ", colours["dim"]).join(parts)


def _colour(name: str) -> str:
    return "worker" if name == "diff" else name


def ticker_line(
    state: TuiState, roles: tuple[str, ...], glyphs: Glyphs, colours: dict[str, str]
) -> Content:
    """The latest entry of each of `roles`, as its own pane draws it, on one line."""
    parts: list[Content] = []
    for role in roles:
        line = _latest(state, role, glyphs, colours)
        if line is not None:
            parts.append(Content.assemble((f"{role} ", f"bold {colours[role]}"), line))
    return Content.styled("  ·  ", colours["dim"]).join(parts)


def _latest(
    state: TuiState, role: str, glyphs: Glyphs, colours: dict[str, str]
) -> Content | None:
    if role == "zeus":
        said = [
            e.record
            for e in state.zeus.entries
            if e.kind is ZeusKind.ZEUS and isinstance(e.record, AssistantTurnP)
        ]
        if not said:
            return None
        return Content.styled(preview(said[-1].text, whole=False), colours["fg"])
    if role == "athena":
        tail = [e for e in state.athena.tail if e.kind is not AthenaKind.MODEL]
        return athena_line(tail[-1], glyphs, colours) if tail else None
    task = state.worker.task
    if task is None or not task.entries:
        return None
    return worker_line(task.entries[-1], glyphs, colours)
