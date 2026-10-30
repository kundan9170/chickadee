"""How the TUI looks: the role colours, the
glyphs and their ASCII fallbacks, and the word every state is shown with.

**The pictures' colours**, with a
16-colour fallback for a terminal that has no truecolor. **No painted background**: the
screen is the terminal's own, so a light theme works.

**Colour is never the only signal**. Every state has a word, and every glyph an ASCII
fallback for a terminal whose encoding cannot print it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from textual.content import Content

from chickadee_cli.tui.model import SpendLevel, Stage

UNKNOWN: Final = "—"
"""What a fact not read yet shows on the screen: unknown, never a guess (the header's rows, the
models overlay)."""

__all__ = [
    "ASCII",
    "PALETTE",
    "PALETTE_16",
    "SPEND_WORDS",
    "STAGE_WORDS",
    "TASK_GLYPH",
    "UNICODE",
    "UNKNOWN",
    "Depth",
    "Glyphs",
    "glyphs_for",
    "meter",
    "palette",
]

type Depth = Literal["truecolor", "16"]

PALETTE: Final = {
    "zeus": "#ffb000",
    "gold": "#ffd23f",
    "athena": "#3fd0c0",
    "worker": "#6be36b",
    "bad": "#ff5f6d",
    "shell": "#6aa0ff",
    "fg": "#c9ccd3",
    "dim": "#7a8090",
    "faint": "#3a3f4c",
    "you": "#1c2540",
    "run": "#0f2f2d",
    "ink": "#0b0b0f",
    "btn": "#262a35",
    "add_bg": "#10260f",
    "del_bg": "#2a1013",
}
"""The mockup's `C`, by role. Zeus amber, Athena teal, the worker green; `gold` the
name, the header's and the front page's; `bad` for an error or a removed line, `shell`
for a command; `you` is the person's rows' band, `run` the running task's on Athena's board,
`btn` a modal's answer and command band, `add_bg` and `del_bg` an added and a removed line's
band on the diff screen (`ui/` `codeRow`)."""

PALETTE_16: Final = {
    "zeus": "ansi_yellow",
    "gold": "ansi_bright_yellow",
    "athena": "ansi_cyan",
    "worker": "ansi_green",
    "bad": "ansi_red",
    "shell": "ansi_blue",
    "fg": "ansi_default",
    "dim": "ansi_bright_black",
    "faint": "ansi_bright_black",
    "you": "ansi_blue",
    "run": "ansi_bright_black",
    "ink": "ansi_black",
    "btn": "ansi_bright_black",
    "add_bg": "ansi_default",
    "del_bg": "ansi_default",
}
"""The bracketed fallbacks: yellow, cyan, green, and the rest by the same rule; a diff line's
band is none, its colour alone."""


def palette(depth: Depth) -> dict[str, str]:
    return dict(PALETTE if depth == "truecolor" else PALETTE_16)


@dataclass(frozen=True)
class Glyphs:
    """The glyphs, and the stage strip's and spend bar's (`ui/` `FLOW_*`, `header`)."""

    accepted: str
    running: str
    waiting: str
    dropped: str
    re_cut: str
    call: str
    step: str
    in_flight: str
    skipped: str
    done_join: str
    next_join: str
    bar_full: str
    bar_empty: str
    minus: str
    """The mockup's minus sign (U+2212) before a count of removed lines."""
    mark: str
    """The brand mark where the bird cannot be drawn (the header)."""
    hook: str
    """A hook that fired: its card's title and its line."""


UNICODE: Final = Glyphs(
    accepted="✔", running="▶", waiting="○", dropped="✖", re_cut="↻", call="▸", step="⟲",
    in_flight="●", skipped="⊘", done_join="━━", next_join="┈┈", bar_full="█", bar_empty="░",
    minus="\u2212", mark="◖●◗", hook="↪",
)  # fmt: skip
ASCII: Final = Glyphs(
    accepted="v", running=">", waiting="o", dropped="x", re_cut="r", call=">", step="~",
    in_flight="*", skipped="-", done_join="==", next_join="..", bar_full="#", bar_empty="-",
    minus="-", mark="<o)", hook="~>",
)  # fmt: skip
"""The fallbacks `v > o x r > ~ *`, and the same kind for the rest."""

_PROBE: Final = "✔▶○✖↻▸⟲●⊘━┈█░◖◗▀▄─═│║↪"
"""Every non-ASCII character the screen draws, glyphs and box lines both."""


def meter(share: float, cells: int, glyphs: Glyphs, fill: str, empty: str) -> Content:
    """A bar `cells` wide, `share` of it full (clamped to 0..1): the spend bar, the context
    bar, the budget's and the index build's, drawn one way."""
    full = round(cells * min(max(share, 0.0), 1.0))
    return Content.assemble(
        (glyphs.bar_full * full, fill), (glyphs.bar_empty * (cells - full), empty)
    )


def glyphs_for(encoding: str | None) -> Glyphs:
    """The glyphs a terminal of `encoding` can print: Unicode when it encodes every one."""
    if not encoding:
        return ASCII
    try:
        _PROBE.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return ASCII
    return UNICODE


TASK_GLYPH: Final = {
    "running": "running",
    "taken over": "running",
    "accepted": "accepted",
    "committed": "accepted",
    "kept": "accepted",
    "dropped": "dropped",
    "re-cut": "re_cut",
}
"""Each board state (`render/board.py`'s `STATES`) to its `Glyphs` field; the state's word is
the board's own, drawn beside it."""

STAGE_WORDS: Final = {
    Stage.DESIGN: "Design",
    Stage.PLAN: "Plan",
    Stage.WORK: "Work",
    Stage.CHECK: "Check",
    Stage.HANDOFF: "Hand off",
}

SPEND_WORDS: Final = {SpendLevel.OK: "", SpendLevel.WARNING: "warning", SpendLevel.CAP: "cap"}
"""What the spend line says beside its colour: amber at a warning, red at the cap."""
