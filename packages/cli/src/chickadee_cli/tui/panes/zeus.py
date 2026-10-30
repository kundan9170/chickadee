"""Zeus's pane (`ui/` `zeusRun`): left, large and focused; the
transcript, and the only input box on the screen.

**The REPL's words for what the REPL already says**.
A tool result, a harness note, an answered question, an episode, a cap, an undo is drawn by
`render/stream.py` (`frame_lines`) as the REPL prints it; a command's output and a question as
it opens are the REPL's own lines (`commands`, `interaction`). What the screen adds is the
mockup's: the person's words on their band, Zeus's reply rendered as Markdown, his tool
calls as cards under it (`panes/cards.py`, a command whole, each result in the card's
foot, an approval a tick in its title; `Ctrl+O` adds each result's lines), and a stop's
answer as a rule across the pane ("design approved"). An approval of another role's is a card
of its own where its answer lands.

**What he is doing is said** ("silence is never a mystery"): a turn in flight pulses
(`●`), and while the box runs he is "holding until Athena reports". Words sent and not yet
recorded are shown dimmed until they land.

His reply is drawn as it is written, from the live frames (`TuiState.live`), and
replaced by his record when it lands.

**As the screen first opens** the pane greets the person,
centred: "Hi, I'm Chickadee. / What shall we build today?" (`WELCOME`, the owner's words of
2026-10-26: no suggestions, and no bird, which the front page beside it draws). It is the
screen's greeting, never a `zeus` line: he has said nothing yet.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Final

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widget import Widget
from textual.widgets import Markdown, OptionList, Static

from chickadee_cli.attach import attachment_line
from chickadee_cli.render.stream import TURN_ERROR, Verbosity, frame_lines
from chickadee_cli.render.style import Style
from chickadee_cli.tui.chips import Chips
from chickadee_cli.tui.composer import HINT, Composer, ComposerChoice
from chickadee_cli.tui.model import Approval, TuiState, ZeusEntry, ZeusKind, ZeusPane
from chickadee_cli.tui.names import ROLE_WORDS, VOICE
from chickadee_cli.tui.panes import Following, RoleFrame, live_lines, spend_label
from chickadee_cli.tui.panes.cards import (
    Card,
    CardText,
    approval_card,
    show_cards,
    turn_cards,
)
from chickadee_cli.tui.pulse import PulseDot
from chickadee_cli.tui.theme import Glyphs
from chickadee_wire.models import (
    ActionGrantedRecord,
    AssistantTurnP,
    QuestionAnsweredRecord,
    ToolRecord,
    UserTurnP,
)

__all__ = [
    "ANSWER",
    "TALK",
    "WELCOME",
    "ToolMode",
    "ZeusFrame",
    "ansi",
    "approval_of",
    "entry_text",
    "status_text",
    "tool_mode",
    "welcome_text",
]

_TERMINAL: Final = Style(tty=True)
"""The REPL's lines as a terminal gets them: their tones become the screen's styles."""

_BUSY: Final = {
    "thinking": "thinking",
    "tool_call": "calling a tool",
    "tool_result": "reading a result",
    "delegation": "delegating",
}
"""What a turn in flight is doing, by its progress frame's kind (a hint)."""

WELCOME: Final = (f"Hi, I'm {VOICE}.", "What shall we build today?")
"""The greeting a fresh session opens on, a line each, centred."""
TALK: Final = f"Talk to {VOICE}…"
ANSWER: Final = f"Answer {VOICE}…"
"""The composer's placeholders: words to Zeus, or the answer to his open question."""

_CLASSES: Final = {ZeusKind.YOU: "entry you", ZeusKind.OUTPUT: "entry said"}
"""The person's words on their band; what the screen says, one line after another, as the
REPL prints a command's output and a question (no gap between its lines)."""

_STOP_WORDS: Final = {
    "approve": "approved",
    "edit": "edited",
    "note": "sent back with a note",
    "cancel": "cancelled",
}


class ToolMode(StrEnum):
    """How Zeus's tool cards are drawn: `/verbose q` hides them, `v` shows them whole, and
    `Ctrl+O` switches between collapsed and expanded, each result's lines in its card."""

    QUIET = "quiet"
    COLLAPSED = "collapsed"
    EXPANDED = "expanded"


def tool_mode(verbosity: Verbosity, *, expanded: bool) -> ToolMode:
    if verbosity is Verbosity.QUIET:
        return ToolMode.QUIET
    return (
        ToolMode.EXPANDED if expanded or verbosity is Verbosity.VERBOSE else ToolMode.COLLAPSED
    )


def ansi(lines: Sequence[str]) -> Content:
    """The REPL's lines, styled for a terminal, as the screen's text."""
    return Content("\n").join(Content.from_rich_text(Text.from_ansi(line)) for line in lines)


def approval_of(entry: ZeusEntry, zeus: ZeusPane) -> Approval | None:
    """The approval an entry answers (a question's answer, or a grant's approval), if it is
    one: drawn as a card's tick, never as a line of its own."""
    match entry.record:
        case QuestionAnsweredRecord(id=action_id) | ActionGrantedRecord(action_id=action_id):
            return zeus.approvals.get(action_id)
        case _:
            return None


def _on_a_call(approval: Approval, zeus: ZeusPane) -> bool:
    return approval.call_id is not None and approval.call_id in zeus.calls


def entry_text(
    entry: ZeusEntry,
    mode: ToolMode,
    verbosity: Verbosity,
    colours: dict[str, str],
    zeus: ZeusPane,
) -> Content | None:
    """Any entry but his reply and an approval's card, as drawn; None when this mode does not
    draw it, or when a card says it: a result of his call, an approval's answer."""
    if approval_of(entry, zeus) is not None:
        return None
    if isinstance(entry.record, QuestionAnsweredRecord) and entry.record.id in zeus.questions:
        return None  # his call's question: its card says how it was answered
    match entry.kind:
        case ZeusKind.YOU if isinstance(entry.record, UserTurnP):
            said = Content.assemble(("you  ", "bold"), (entry.record.text, "bold"))
            attached = [
                Content.styled(f"\n     attached {attachment_line(a)}", colours["dim"])
                for a in entry.record.attachments
            ]
            return Content.assemble(said, *attached) if attached else said
        case ZeusKind.RESULT if isinstance(entry.record, ToolRecord) and (
            entry.record.call_id in zeus.calls or mode is not ToolMode.EXPANDED
        ):
            return None
        case ZeusKind.MARKER if entry.stop is not None and isinstance(
            entry.record, QuestionAnsweredRecord
        ):
            word = _STOP_WORDS.get(entry.record.choice, entry.record.choice)
            return Content.assemble(
                ("── ", colours["faint"]), (f"{entry.stop} {word}", colours["dim"]),
                (" ──", colours["faint"]),
            )  # fmt: skip
        case ZeusKind.ERROR:
            said = f"{TURN_ERROR}: {entry.text}" if entry.text else TURN_ERROR
            return Content.styled(said, colours["bad"])
        case ZeusKind.OUTPUT:
            return ansi([entry.text or ""])
        case _ if entry.frame is not None:
            lines = frame_lines(entry.frame, verbosity, _TERMINAL)
            return ansi(lines) if lines else None
        case _:
            return None


def status_text(state: TuiState, glyphs: Glyphs, colours: dict[str, str]) -> Content:
    """Below the transcript: what he is doing, whom he waits on, and words not yet recorded."""
    zeus, dim = state.zeus, colours["dim"]
    lines: list[Content] = []
    now = time.monotonic()
    for reply in state.live.open.values():
        if reply.role == "zeus":
            # His reply as it is written; his record replaces it when it lands.
            lines += live_lines(reply, glyphs, colours, whole=True, now=now)
    if lines:
        pass
    elif zeus.busy is not None:
        lines.append(Content.styled(f"{_BUSY.get(zeus.busy, zeus.busy)}…", dim))
    elif (whom := state.holding_for) is not None:
        lines.append(Content.styled(f"holding until {whom} reports", dim))
    lines += [Content.styled(f"you  {text}  (queued)", dim) for text in zeus.pending]
    return Content("\n").join(lines)


def welcome_text(colours: dict[str, str]) -> Content:
    """The greeting: his name in his colour, centred by the pane."""
    hello, ask = WELCOME
    name = f"{VOICE}"
    before, _, after = hello.partition(name)
    return Content("\n").join(
        [
            Content.assemble(
                (before, colours["fg"]),
                (name, f"bold {colours['zeus']}"),
                (after, colours["fg"]),
            ),
            Content.styled(ask, colours["fg"]),
        ]
    )


class _ReplyView(Vertical):
    """His reply: `chickadee` (the owner's name for the voice the person talks to), his words
    as Markdown, and his calls' cards under them."""

    DEFAULT_CSS = """
    _ReplyView { height: auto; }
    _ReplyView > Markdown { margin: 0; padding: 0; background: transparent; }
    _ReplyView > .cards { height: auto; }
    """

    def __init__(self, text: str) -> None:
        super().__init__(classes="entry reply")
        self._text = text
        self._cards: Sequence[CardText] = ()
        self.cards = Vertical(classes="cards")

    def compose(self) -> ComposeResult:
        yield Static(VOICE.lower(), classes="who")
        yield Markdown(self._text)
        with self.cards:
            yield from (Card(text) for text in self._cards)

    def set_cards(self, cards: Sequence[CardText]) -> None:
        """Its cards, mounted with it when it is new, else shown again in place."""
        self._cards = cards
        if self.is_mounted:
            show_cards(self.cards, cards)

    def on_mount(self) -> None:
        # Cards set between `compose` and the mount (a question landing just after its turn)
        # would be lost: `compose` drew the earlier ones, and `set_cards` skipped the update.
        show_cards(self.cards, self._cards)


class _Transcript(Following):
    """The conversation: the one part of his pane that scrolls, following its newest entry
    until the person scrolls up."""

    DEFAULT_CSS = """
    _Transcript > .entry { height: auto; margin-bottom: 1; }
    _Transcript > .entry.said { margin-bottom: 0; }
    _Transcript > .you { background: $you; padding: 0 1; }
    _Transcript > .welcome { height: 1fr; content-align: center middle; text-align: center; }
    """


class ZeusFrame(RoleFrame):
    """His pane: the transcript above, and below it, outside the scroll, the status line, the
    command list and the composer. The foot is not docked inside the scrolling frame: docked
    there it covered the newest lines, which no scroll could reach (found in the live demo)."""

    role = "zeus"
    colour = "zeus"

    DEFAULT_CSS = """
    ZeusFrame { border: round $zeus; overflow-y: hidden; }
    ZeusFrame:focus-within { border: double $zeus; }
    ZeusFrame .who { color: $zeus; text-style: bold; }
    ZeusFrame > .foot { height: auto; }
    ZeusFrame > .foot > .status { height: auto; }
    ZeusFrame > .foot > OptionList { height: auto; max-height: 8; display: none; }
    ZeusFrame > .foot > OptionList.-open { display: block; }
    ZeusFrame > .foot > .rule { color: $faint; height: 1; }
    ZeusFrame > .foot > .hint {
        color: $dim; height: 1; text-wrap: nowrap; text-overflow: ellipsis;
    }
    ZeusFrame > .foot > .line { height: auto; }
    ZeusFrame > .foot > .line > .mark { width: 2; color: $zeus; text-style: bold; }
    """

    def __init__(
        self, colours: dict[str, str], glyphs: Glyphs, history: Callable[[], Sequence[str]]
    ) -> None:
        super().__init__(colours, glyphs)
        self.transcript = _Transcript()
        self.composer = Composer(history)
        self.chips = Chips()
        """The files on the draft, drawn only while there are any."""
        self.completions = OptionList()
        self.status = Static("", classes="status")
        self.dot = PulseDot(f"{glyphs.in_flight} ")
        self._views: list[Widget] = []
        self._shown: list[ZeusEntry] = []
        self._cleared = 0
        self._refills = 0
        """Counts `redraw`s: a refill can change which views draw nothing (`-undrawn`)."""
        self._carded = -1
        """`ZeusPane.cards` when the cards were last filled: a result or an answer moves it."""
        self._placed: tuple[int, frozenset[int], int, int] | None = None
        """What each view's `display` was last worked out from: the views, what an undo
        hides, `/clear`'s mark and the refills. A live frame moves only the status line, so
        streaming a reply never walks the whole transcript again."""
        self._verbosity = Verbosity.NORMAL
        self._expanded = False
        self._options: tuple[ComposerChoice, ...] = ()
        self._asking: str | None = None
        self._idle = TALK
        """The composer's placeholder while no command asks: `_prompt`'s, from the state."""
        self.welcome = Static("", classes="entry welcome")
        self._landing = False

    def compose(self) -> ComposeResult:
        with self.transcript:
            yield self.welcome
            yield from super().compose()
        with Vertical(classes="foot"):
            with _Line(classes="line"):
                yield self.dot
                yield self.status
            yield self.completions
            yield Static("─" * 200, classes="rule")
            yield self.chips
            with _Line(classes="line"):
                yield Static(">", classes="mark")
                yield self.composer
            yield Static(HINT, classes="hint")

    # -- the frame -------------------------------------------------------------------------

    def idle(self, state: TuiState) -> bool:
        return False  # the conversation is never idle: it is where everything starts

    def title(self, state: TuiState) -> Content:
        return Content.styled(ROLE_WORDS["zeus"].upper(), "bold")

    def label(self, state: TuiState) -> Content | tuple[Content, ...]:
        return spend_label(
            state.zeus.cost_usd,
            state.context("zeus"),
            self._colours["zeus"],
            self._glyphs,
            self._colours,
        )

    @property
    def mode(self) -> ToolMode:
        return tool_mode(self._verbosity, expanded=self._expanded)

    # -- the transcript --------------------------------------------------------------------

    def show(self, state: TuiState) -> None:
        super().show(state)
        entries = state.zeus.entries
        if state.zeus.cards != self._carded:
            self._carded = state.zeus.cards
            for entry, view in zip(self._shown, self._views, strict=True):
                if isinstance(view, _ReplyView | Card):
                    self._fill(entry, view)
        for entry in entries[len(self._views) :]:
            view = self._view(entry)
            self._views.append(view)
            self._shown.append(entry)
            self.transcript.mount(view)
        placed = (len(self._views), state.zeus.hidden, self._cleared, self._refills)
        if placed != self._placed:
            self._placed = placed
            visible = {id(e) for e in state.zeus.visible()}
            for i, (entry, view) in enumerate(zip(self._shown, self._views, strict=True)):
                shown = i >= self._cleared and id(entry) in visible and self._drawn(view)
                view.display = shown
        self._landing = state.landing
        self.welcome.display = self._landing
        if self._landing:
            self.welcome.update(welcome_text(self._colours))
        self.status.update(status_text(state, self._glyphs, self._colours))
        self.dot.display = state.zeus.busy is not None
        self._prompt(state)

    def redraw(self) -> None:
        """Every entry again, for a new tool mode."""
        for entry, view in zip(self._shown, self._views, strict=True):
            self._fill(entry, view)
        self._refills += 1

    def set_verbosity(self, verbosity: Verbosity) -> None:
        self._verbosity = verbosity
        self.redraw()

    def toggle_tools(self) -> None:
        self._expanded = not self._expanded
        self.redraw()

    def clear_view(self) -> None:
        """`/clear`: the view, not the session; every entry so far is put out of sight."""
        self._cleared = len(self._views)
        for view in self._views:
            view.display = False

    def asking(self, prompt: str | None) -> None:
        """A command's question in the composer (`Context.ask`), or None once answered."""
        self._asking = prompt
        self.composer.placeholder = self._idle if prompt is None else prompt

    def _view(self, entry: ZeusEntry) -> Widget:
        zeus = self.state.zeus
        approval = approval_of(entry, zeus)
        if entry.kind is ZeusKind.ZEUS and isinstance(entry.record, AssistantTurnP):
            view: Widget = _ReplyView(entry.record.text)
        elif approval is not None and not _on_a_call(approval, zeus):
            view = Card(approval_card(approval, self._colours), classes="entry")
        else:
            view = Static("", classes=_CLASSES.get(entry.kind, "entry"))
        self._fill(entry, view)
        return view

    def _fill(self, entry: ZeusEntry, view: Widget) -> None:
        zeus, quiet = self.state.zeus, self.mode is ToolMode.QUIET
        if isinstance(view, _ReplyView) and isinstance(entry.record, AssistantTurnP):
            cards = turn_cards(
                entry.record,
                zeus,
                self._colours,
                expanded=self.mode is ToolMode.EXPANDED,
                verbosity=self._verbosity,
            )
            view.set_cards([] if quiet else cards)
            return
        if isinstance(view, Card):
            approval = approval_of(entry, zeus)
            if approval is not None:
                view.show(approval_card(approval, self._colours))
            view.set_class(quiet, "-undrawn")
            return
        if isinstance(view, Static):
            text = entry_text(entry, self.mode, self._verbosity, self._colours, zeus)
            view.update(text or "")
            view.set_class(text is None, "-undrawn")

    @staticmethod
    def _drawn(view: Widget) -> bool:
        return not view.has_class("-undrawn")

    def _prompt(self, state: TuiState) -> None:
        # A stop's modal answers its own question; the composer answers Zeus's.
        self._idle = ANSWER if state.stops.open and state.asking is None else TALK
        if self._asking is None:
            self.composer.placeholder = self._idle

    # -- the command list ------------------------------------------------------------------

    def on_composer_completing(self, message: Composer.Completing) -> None:
        if message.options != self._options:
            self._options = message.options
            self.completions.clear_options()
            self.completions.add_options([o.shown for o in message.options])
            if message.options:
                self.completions.highlighted = 0
        self.completions.set_class(bool(message.options), "-open")
        if message.move and message.options:
            at = self.completions.highlighted or 0
            self.completions.highlighted = (at + message.move) % len(message.options)

    def on_composer_completed(self, _message: Composer.Completed) -> None:
        if not self._options:
            return
        chosen = self._options[self.completions.highlighted or 0]
        self.composer.load_text(self.composer.text[: chosen.start] + chosen.insert)
        self.composer.move_cursor(self.composer.document.end)

    def close_completions(self) -> bool:
        """Esc on an open list closes it; whether there was one."""
        if not self.completions.has_class("-open"):
            return False
        self.completions.set_class(False, "-open")
        return True


class _Line(Widget):
    """The `>` and the composer, side by side (`ui/` `composer()`)."""

    DEFAULT_CSS = "_Line { layout: horizontal; height: auto; }"
