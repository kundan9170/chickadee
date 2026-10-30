"""The panes of layout E: one frame for each, its title on the
left of the top border and its spend or count on the right (`ui/` `G.box`).

**The focused pane is drawn with a double border in its role's colour; every other pane keeps
updating but is quiet**, and a pane with nothing to show yet is drawn faint
(the mockup). A pane's body is its own; the frame is shared.

**A pane with rows to open** (Athena's tasks, the worker's commands) names its keys in
`keys.py` (`KEYS`, the bar's rule: bound and drawn from one table); while it has the focus,
`↑`/`↓` choose a row, `Enter` asks the screen to open it (`Details`), and its hint says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, VerticalScroll
from textual.content import Content
from textual.events import Blur, Focus, Resize
from textual.message import Message
from textual.widgets import Static

from chickadee_cli.render.live import TOOL_AFTER_S, StreamingReply, size_text
from chickadee_cli.tui.keys import Key, lit
from chickadee_cli.tui.model import TuiState
from chickadee_cli.tui.theme import Glyphs, meter

__all__ = [
    "CONTEXT_CELLS",
    "IDLE",
    "Following",
    "RoleFrame",
    "border_title",
    "follow_overflow",
    "live_lines",
    "spend_label",
    "window_text",
]

IDLE = "-idle"
"""The class of a pane with nothing to show yet: drawn faint."""

_CORNERS = 6
"""What a border title cannot use of the pane's width: the two corners, a border cell and a
space each side of the title (Textual draws ` title ` inside `═ … ═`)."""
_LABEL_MIN = 4
"""The narrowest a cut label is still worth drawing: three cells and its `…`."""
CONTEXT_CELLS = 8
"""A pane label's context bar, in cells."""
_CONTEXT_FULL = 0.9
"""Where a context bar turns red: the next prompts may not fit."""


def window_text(tokens: int) -> str:
    """A count of tokens as the picture writes it: `200K`, `1M`."""
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    return f"{round(tokens / 1000)}K" if tokens >= 1000 else str(tokens)


def spend_label(
    cost: float | None,
    context: tuple[int, int | None] | None,
    colour: str,
    glyphs: Glyphs,
    colours: dict[str, str],
) -> tuple[Content, ...]:
    """A pane's right label, widest first, for `border_title` to take the first that fits: its
    role's spend, then how full its latest prompt was against its model's window
    (`$0.31 · ctx ██░░░░░░ 17%`, then `$0.31 · ctx 17%`, then `$0.31`; `ctx 34K` where the
    catalog does not give the window). The spend first, so a narrow pane drops the context,
    never the cost. What is not known yet is left out: unknown is never `$0` or `0%`."""
    dim, faint = colours["dim"], colours["faint"]
    spend = [Content.styled(f"${cost:.2f}", dim)] if cost is not None else []
    if context is None:
        return (spend[0],) if spend else (Content(""),)
    tokens, window = context
    if not window:
        known = Content.styled(f"ctx {window_text(tokens)}", dim)
        return (Content.styled(" · ", faint).join([*spend, known]), *spend)
    share = min(tokens / window, 1.0)
    fill = colours["bad"] if share >= _CONTEXT_FULL else colour
    percent = (f" {share:.0%}", colours["bad"] if share >= _CONTEXT_FULL else dim)
    bar = Content.assemble(
        ("ctx ", dim), meter(share, CONTEXT_CELLS, glyphs, fill, faint), percent
    )
    short = Content.assemble(("ctx", dim), percent)
    joined = Content.styled(" · ", faint)
    return (joined.join([*spend, bar]), joined.join([*spend, short]), *spend)


def follow_overflow(pane: ScrollableContainer) -> None:
    """Keep `pane` at its newest line while its content overflows it, and top-aligned
    while it fits. Textual's anchor follows content however late it grows, but it also draws
    content shorter than the pane from the pane's bottom, so it is held only while there is
    more than fits. Scrolling up releases it; back at the end, or `End`, takes it again."""
    overflows = pane.virtual_size.height > pane.scrollable_content_region.height
    if overflows and not pane.is_anchored:
        pane.anchor()
    elif not overflows and pane.is_anchored:
        pane.anchor(False)


def live_lines(
    reply: StreamingReply,
    glyphs: Glyphs,
    colours: dict[str, str],
    *,
    whole: bool = False,
    now: float,
) -> list[Content]:
    """A reply being written, as a pane draws it: its text so far (`whole`) or its line
    being written, how long it has thought while it has no text, and each tool call being
    written once it has been for `TOOL_AFTER_S`, as the REPL's live lines say them."""
    dim = colours["dim"]
    lines: list[Content] = []
    if reply.thinking_s is not None and not reply.text:
        lines.append(
            Content.styled(f"{glyphs.in_flight} thinking… {reply.thinking_s:.0f}s", dim)
        )
    text = reply.text.rstrip("\n")
    if text:
        shown = text if whole else text.rsplit("\n", 1)[-1]
        lines.append(Content.styled(f"{glyphs.in_flight} {shown}", colours["fg"]))
    for call in reply.tools.values():
        if now - call.since >= TOOL_AFTER_S:
            where = f" {call.path}" if call.path else ""
            said = f"{glyphs.call} {call.tool}{where} · writing… {size_text(call.size)}"
            lines.append(Content.styled(said, dim))
    return lines


def border_title(
    left: Content, right: Content | Sequence[Content], width: int, *, double: bool
) -> Content:
    """The title on the left, the label on the right, the border's own line between, for a
    frame `width` cells wide: Textual has one title per edge, so the gap is drawn in the
    border's character. A label given in several forms, widest first, is the first that fits
    whole. When none fits, the last is cut, marked `…` (invariant 16), never the title."""
    fill = "═" if double else "─"
    room = width - _CORNERS
    lefts = cell_len(left.plain)
    forms = [right] if isinstance(right, Content) else list(right) or [Content("")]
    right = next((f for f in forms if lefts + cell_len(f.plain) + 3 <= room), forms[-1])
    rights = cell_len(right.plain)
    if not rights or room - lefts < _LABEL_MIN + 2:
        return left
    label = right
    if lefts + rights + 3 > room:
        label = label.truncate(room - lefts - 2, ellipsis=True)
    gap = room - lefts - cell_len(label.plain) - 2
    return Content.assemble(left, " ", fill * max(gap, 1), " ", label)


class Following(VerticalScroll):
    """The part of a pane that scrolls, following its newest line (`follow_overflow`), with
    what is pinned below it kept outside: docked inside a scroll, a pinned block covers the
    newest lines, which no scroll reaches (found in Zeus's pane, and again in the worker's)."""

    DEFAULT_CSS = """
    Following { height: 1fr; scrollbar-size-vertical: 1; }
    """

    def watch_virtual_size(self) -> None:
        follow_overflow(self)

    def on_resize(self) -> None:
        follow_overflow(self)


class RoleFrame(VerticalScroll):
    """A pane: a frame titled `ROLE · function`, a right label, and a body below."""

    role: ClassVar[str]
    """`zeus`, `athena`, `worker` or `diff`: its id, and its colour variable (`worker` for the
    diff, as the mockup's CODE pane)."""
    colour: ClassVar[str]
    follows: ClassVar[bool] = False
    """Newest at the bottom and followed, by `follow_overflow`."""
    KEYS: ClassVar[tuple[Key, ...]] = ()
    """Its own keys (`keys.ATHENA_KEYS`); a subclass binds them from the same table."""

    @dataclass
    class Details(Message):
        """`Enter` on a row: the screen opens what it holds."""

        role: str
        chosen: str

    DEFAULT_CSS = """
    RoleFrame { border: round $faint; padding: 0 1; scrollbar-size-vertical: 1; }
    RoleFrame.-idle { border: round $faint; }
    RoleFrame > .body { width: 1fr; }
    """

    def __init__(self, colours: dict[str, str], glyphs: Glyphs) -> None:
        super().__init__(id=self.role)
        self._colours, self._glyphs = colours, glyphs
        self._left = Content("")
        self._right: Content | tuple[Content, ...] = Content("")
        self.body = Static("", classes="body")
        self.state = TuiState()
        """The state it last drew, for its keys' rule."""
        self.cursor: int | None = None
        """The chosen row, by its place in `rows`; None for the pane's default."""

    def compose(self) -> ComposeResult:
        yield self.body

    def watch_virtual_size(self) -> None:
        if self.follows:
            follow_overflow(self)

    # -- what the pane shows ---------------------------------------------------------------

    def show(self, state: TuiState) -> None:
        """Redraw from `state`: the title, the right label, idle or not, and the body."""
        self.state = state
        idle = self.idle(state)
        self.set_class(idle, IDLE)
        self._left, self._right = self.title(state), self.label(state)
        self.body.update(self.idle_text(state) if idle else self.content(state))
        self._retitle()

    def idle(self, state: TuiState) -> bool:
        raise NotImplementedError

    def title(self, state: TuiState) -> Content:
        raise NotImplementedError

    def label(self, state: TuiState) -> Content | tuple[Content, ...]:
        """The right label, or its forms widest first (`border_title` takes one that fits)."""
        return Content("")

    def idle_text(self, state: TuiState) -> Content:
        return Content("")

    def content(self, state: TuiState) -> Content:
        """The pane's live body: Part 3 (Athena, the worker, the diff) and Part 4 (Zeus)."""
        return Content("")

    # -- the frame -------------------------------------------------------------------------

    def on_resize(self, _event: Resize) -> None:
        self._retitle()
        if self.follows:
            follow_overflow(self)

    def on_focus(self, _event: Focus) -> None:
        self.call_after_refresh(self._refocused)

    def on_blur(self, _event: Blur) -> None:
        self.call_after_refresh(self._refocused)

    def _refocused(self) -> None:
        """Its border, and with keys of its own its chosen row and hint, once the focus has
        moved (not before: in the event the focus is not yet the pane's)."""
        self._retitle()
        if self.KEYS:
            self.show(self.state)

    # -- rows to open ----------------------------------------------------------------------

    def rows(self, state: TuiState) -> list[str]:
        """The ids of the rows `Enter` opens, in the order drawn."""
        return []

    def default_row(self, rows: list[str]) -> int:
        return len(rows) - 1

    def chosen(self) -> str | None:
        """The chosen row's id, while the pane has the focus and has rows."""
        rows = self.rows(self.state)
        if not rows or not self.has_focus:
            return None
        at = self.default_row(rows) if self.cursor is None else self.cursor
        return rows[min(max(at, 0), len(rows) - 1)]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        for key in self.KEYS:
            if key.action is not None and key.action.split("(")[0] == action:
                return lit(key, self.state)
        return True

    def action_choose(self, step: int) -> None:
        rows = self.rows(self.state)
        if not rows:
            return
        at = self.default_row(rows) if self.cursor is None else self.cursor
        self.cursor = (at + step) % len(rows)
        self.show(self.state)

    def action_details(self) -> None:
        chosen = self.chosen()
        if chosen is not None:
            self.post_message(self.Details(self.role, chosen))

    def _retitle(self) -> None:
        double = self.styles.border_top[0] == "double"
        self.border_title = border_title(
            self._left, self._right, self.outer_size.width, double=double
        )
