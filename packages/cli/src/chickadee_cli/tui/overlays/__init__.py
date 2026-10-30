"""The screens drawn over the panes: the
command palette, help, the budget, the models, the sessions, undo and the handoff.

**An overlay is a view with actions; a modal answers a question**: the
modals (`modals/`) put keys over a question's answers and close when its record lands; an
overlay shows what routes and records say, and what it does is a slash command.

**What it does is the REPL's command** (one home): an action builds the command line
(`/budget 5`, `/rung mid-param qwen3`) and the host hands it to the REPL's own `dispatch`
(`Host.command`), so a command's parsing, its refusals, its questions and the lines it prints
stay `commands.py`'s. The overlay closes first: what the command prints, and anything it asks,
land in Zeus's pane, as if typed.

**One table binds and draws its keys** (the status bar's rule, `keys.py`): each `OverlayKey` is
bound and drawn in the foot line, lit when built and applying now, dimmed (and doing nothing)
when not; one not built yet names the wire request that would build it.

**The panes keep updating beneath**: the host draws them on the base screen whichever
screen is on top, and hands the open overlay each new state (`show`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import Static

from chickadee_cli.tui.model import TuiState
from chickadee_cli.tui.panes import border_title
from chickadee_cli.tui.theme import Glyphs
from chickadee_wire.client import Client
from chickadee_wire.models import RunBlob

__all__ = [
    "CLOSE",
    "MOVE",
    "NAMED_KEYS",
    "READING",
    "Blobs",
    "Host",
    "Overlay",
    "OverlayKey",
    "bindings",
    "chip",
    "fit",
    "foot_line",
    "money",
    "row",
]

NAMED_KEYS: Final = frozenset(
    {"up", "down", "left", "right", "enter", "escape", "tab", "shift+tab", "pageup", "pagedown"}
)
"""Keys no text input types: bound with priority, so a filter's input does not take them. A
letter is not, so it types where an input has the focus and acts where none has."""


class Host(Protocol):
    """What an overlay may ask of the screen it is drawn over (the app)."""

    @property
    def session_id(self) -> str: ...

    @property
    def state(self) -> TuiState: ...

    @property
    def colours(self) -> dict[str, str]: ...

    @property
    def glyphs(self) -> Glyphs: ...

    @property
    def header_height(self) -> int:
        """The header's rows now: where the box starts, so the stage and spend stay seen."""
        ...

    def command(self, line: str) -> None:
        """`line` to the REPL's `dispatch`, as typed, past the screen's own views."""
        ...

    def submit(self, line: str) -> None:
        """`line` as the person would type it: a bare `/budget` opens its overlay."""
        ...

    def put_in_composer(self, text: str) -> None: ...

    def read[T](
        self,
        read: Callable[[Client], T],
        deliver: Callable[[T], None],
        failed: Callable[[str], None],
    ) -> None:
        """A route read once off the screen's loop (`SessionReader.ask`), its answer or its
        failure handed back on it."""
        ...

    def refresh_routes(self, routes: Iterable[str]) -> None:
        """Routes the screen holds, read again (`SessionReader.refresh`)."""
        ...


@dataclass(frozen=True)
class OverlayKey:
    """One entry of an overlay's foot line, and the keys it binds."""

    keys: tuple[str, ...]
    """Textual's names (`up`, `down`); empty for one only drawn."""
    shown: str
    """How the foot prints it (`Up/Down`)."""
    label: str
    actions: tuple[str, ...] | None = None
    """The action of each key, in order; None while it is not built."""
    builds_in: str | None = None
    """For one not built, the wire request that would build it."""


CLOSE: Final = OverlayKey(("escape",), "Esc", "close", ("close",))

READING: Final = "reading…"


class Blobs:
    """The run blobs an overlay shows (a brief, a report's words, a review, a command's
    output), each read once from `/run/{run_id}/blob/{sha}` and said as
    the route answers: its text, why it is withheld, or why it could not be read. Never
    guessed, and never read twice while the overlay is open."""

    def __init__(self, host: Host, run_id: str | None, redraw: Callable[[], None]) -> None:
        self._host = host
        self._run_id = run_id
        self._redraw = redraw
        self._said: dict[str, str] = {}
        self._asking = False
        """A read is being started: an answer delivered at once (a fake reader's) is the
        caller's to return, never a redraw from inside the drawing that asked."""

    def said(self, sha: str | None) -> str | None:
        """`sha`'s text as it reads now; None for no blob. The first ask starts its read."""
        if not sha:
            return None
        if self._run_id is None:
            return "no run on screen"
        if sha not in self._said:
            self._said[sha] = READING
            run_id = self._run_id
            self._asking = True
            try:
                self._host.read(
                    lambda c: c.run_blob(run_id, sha),
                    lambda blob: self._got(sha, blob),
                    lambda why: self._got(sha, f"not read: {why}"),
                )
            finally:
                self._asking = False
        return self._said[sha]

    def _got(self, sha: str, blob: RunBlob | str) -> None:
        if isinstance(blob, str):
            self._said[sha] = blob
        else:
            self._said[sha] = blob.text if blob.text is not None else f"({blob.withheld})"
        if not self._asking:
            self._redraw()


MOVE: Final = OverlayKey(("up", "down"), "Up/Down", "move", ("move(-1)", "move(1)"))


def bindings(keys: tuple[OverlayKey, ...]) -> list[BindingType]:
    """An overlay's bindings, from its table: every built key, none of the rest."""
    return [
        Binding(key, action, show=False, priority=key in NAMED_KEYS)
        for entry in keys
        if entry.actions is not None
        for key, action in zip(entry.keys, entry.actions, strict=True)
    ]


def foot_line(
    keys: tuple[OverlayKey, ...], colours: dict[str, str], lit: Callable[[OverlayKey], bool]
) -> Content:
    """`Up/Down role · Enter switch · Esc close`: each lit or dimmed, never hidden."""
    parts: list[Content] = []
    for entry in keys:
        # An entry with no key says how the overlay is used (`Type to filter`): never dimmed.
        on = not entry.keys or (entry.actions is not None and lit(entry))
        shown = Content.styled(entry.shown, colours["fg"] if on else colours["faint"])
        parts.append(Content.assemble(shown, " ", (entry.label, colours["dim"])))
    return Content.styled(" · ", colours["dim"]).join(parts)


def chip(key: str, label: str, colours: dict[str, str], *, on: bool) -> Content:
    """An action as the mockup draws it: the key in amber, the label on a button band;
    faint when it does nothing now."""
    if not on:
        return Content.styled(f" {key} {label} ", f"{colours['faint']} on {colours['btn']}")
    return Content.assemble(
        (f" {key} ", f"bold {colours['zeus']} on {colours['btn']}"),
        (f"{label} ", f"{colours['fg']} on {colours['btn']}"),
    )


def fit(text: str, width: int) -> str:
    """`text` padded or cut to `width` cells, a cut marked `…` (invariant 16)."""
    if cell_len(text) <= width:
        return text + " " * (width - cell_len(text))
    return Content(text).truncate(max(width, 1), ellipsis=True).plain


def money(usd: float | None) -> str:
    """A spend as every overlay shows it: unknown when the route has not said, never $0."""
    return "unknown" if usd is None else f"${usd:.2f}"


def row(text: Content, width: int, colours: dict[str, str], *, chosen: bool) -> Content:
    """A list's row, the chosen one on the highlight band across the overlay's width."""
    if not chosen:
        return text
    pad = max(width - cell_len(text.plain), 0)
    band = f"{colours['ink']} on {colours['shell']}"
    return Content.assemble(Content.styled(text.plain, band), (" " * pad, band))


class Overlay(ModalScreen[None]):
    """One overlay over the screen. A subclass names itself (`WORD`, `COMMAND`), its keys
    (`KEYS`) and draws its body (`body`)."""

    AUTO_FOCUS = "#body"
    """The list's keys, not a hidden input (an amount, a filter), have the focus as it opens."""
    WORD: ClassVar[str]
    """The title's chip (`MODELS`)."""
    COMMAND: ClassVar[str]
    """The command it is reached by, beside the chip (`/rung`)."""
    KEYS: ClassVar[tuple[OverlayKey, ...]]
    WIDTH: ClassVar[int] = 76
    """Its box's share of the screen's width, in percent, as its picture draws it (`05` 62,
    `06` 79, `08` 76, `09` 83, `10` 75, `11` 72). Its height is the screen's between the
    header and the composer's last line, so the header's stage strip and spend stay in view,
    as in every picture."""

    DEFAULT_CSS = """
    Overlay { align: center top; }
    /* Spaced by the box's margin, not the screen's padding: a padded screen draws the screen
       beneath it shifted by that padding. */
    Overlay > #box {
        height: 1fr; margin: 0 0 2 0; border: double $shell; background: ansi_default;
        padding: 0 1;
        min-width: 58; max-width: 100%;
    }
    Overlay #body { height: 1fr; scrollbar-size-vertical: 1; }
    Overlay #body > Static { height: auto; }
    Overlay #foot { height: 2; border-top: solid $faint; color: $dim; }
    Overlay #foot { text-wrap: nowrap; text-overflow: ellipsis; }
    Overlay Input, Overlay Input:focus {
        border: none; height: 1; padding: 0; background: ansi_default;
    }
    Overlay #prompt { height: 2; border-bottom: solid $faint; }
    Overlay #prompt > Static { width: 2; color: $zeus; }
    """

    def __init__(self, host: Host) -> None:
        super().__init__()
        self.host = host
        self.cursor = 0
        """The chosen row of the overlay's list, if it has one."""
        self.scroll = VerticalScroll(id="body")
        self.content = Static("", id="content")
        self.foot = Static("", id="foot")

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield from self.head()
            with self.scroll:
                yield self.content
            yield from self.tail()
            yield self.foot

    def head(self) -> ComposeResult:
        """What stands above the scrolling body (the palette's filter)."""
        yield from ()

    def tail(self) -> ComposeResult:
        """What stands below it, above the foot (the budget's actions and their input)."""
        yield from ()

    def on_mount(self) -> None:
        box = self.query_one("#box")
        box.styles.width = f"{self.WIDTH}%"
        self.below_header()
        self.redraw()

    def on_resize(self, _event: Resize) -> None:
        self.below_header()
        self.redraw()

    def below_header(self) -> None:
        """The box starts under the header, however tall the screen or full screen makes it."""
        self.query_one("#box").styles.margin = (self.host.header_height, 0, 2, 0)

    # -- what it shows ---------------------------------------------------------------------

    @property
    def state(self) -> TuiState:
        return self.host.state

    @property
    def colours(self) -> dict[str, str]:
        return self.host.colours

    def right(self) -> str:
        """The title's label on the right (`this session only`)."""
        return ""

    def heading(self) -> str:
        """What stands beside the title's chip: the command it is reached by, by default."""
        return self.COMMAND

    def body(self, width: int) -> Content:
        raise NotImplementedError

    def foot_keys(self) -> tuple[OverlayKey, ...]:
        """The keys its foot line draws: all of them, unless the body draws some as buttons."""
        return self.KEYS

    def rows(self) -> int:
        """How many rows its list has, for `move`; none by default."""
        return 0

    def show(self, state: TuiState) -> None:
        """A new state from the host: drawn again, the panes beneath already were."""
        del state  # read through the host, so a draw between two shows reads the latest
        if self.is_mounted:
            self.redraw()

    def redraw(self) -> None:
        if not self.is_mounted:
            return
        box = self.query_one("#box")
        width = max(box.content_size.width, 20)
        c = self.colours
        left = Content.assemble(
            (f" {self.WORD} ", f"bold {c['ink']} on {c['shell']}"),
            " ",
            (self.heading(), c["shell"]),
        )
        box.border_title = border_title(
            left, Content.styled(self.right(), c["dim"]), box.outer_size.width, double=True
        )
        self.content.update(self.body(width))
        self.foot.update(foot_line(self.foot_keys(), c, self.lit))

    # -- keys ------------------------------------------------------------------------------

    def applies(self, action: str) -> bool:
        """Whether `action` does something now; one that does not is dimmed and does nothing.
        `close` too: an overlay waiting on what it sent may keep itself open."""
        del action
        return True

    def lit(self, entry: OverlayKey) -> bool:
        return entry.actions is not None and all(
            self.applies(a.split("(")[0]) for a in entry.actions
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        return action == "move" or self.applies(action)

    def action_close(self) -> None:
        self.dismiss()

    def action_move(self, step: int) -> None:
        count = self.rows()
        if count:
            self.cursor = (self.cursor + step) % count
            self.redraw()
            self.scroll.scroll_to(y=max(self.cursor_line() - 2, 0), animate=False)

    def cursor_line(self) -> int:
        """The line of the body the chosen row is drawn on, to keep it in view."""
        return self.cursor
