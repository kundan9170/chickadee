"""The TUI's screen (layout E): the header, Zeus on the left and
focused, the diff in the centre, Athena over the worker on the right, and the status bar.

**A fixed layout**: the panes keep the mockup's
proportions, 36: 40: 28 across and 11: 17 down the right column (`ui/` `SCREENS.base`, 104
by 33), as ratios that scale with the terminal.

**It draws the state it is given.** `show(state, changed)` redraws the panes `feed.Changed`
names, and only those ("Motion": a frame budget over ssh).

**Live, the feed is folded here and nowhere else.** `live.SessionReader` reads the server on
threads and hands each frame, route answer and loss to `take`, `take_snapshot` and `take_lost`
through `call_from_thread`, so the `Feed` is only ever touched on this event loop (its
docstring's rule). What the person types is `talk.Talk`'s, on its own thread, and what it
changes on screen comes back the same way (`_Screen`).

**Views, zoom and narrow terminals.** `^D` (or `/diff`) shows the diff screen in place of the
panes (`screens/diff.py`); `Ctrl+Up` zooms the focused pane, `Ctrl+Down` back; below 100
columns the panes stack, below 80 one shows at a time (`narrow.py`). Whatever is hidden, the
ticker names its latest entry. `/handoff` is read only while the diff screen shows.

**Overlays**: `^P` the palette, `?` help, `^U` undo, and a
bare `/budget`, `/rung`, `/resume`, `/undo`, `/handoff` their screens over the panes. The app is
their `Host`: what they do goes to the REPL's `dispatch` (`Talk.command`), what they read once
goes through the reader (`SessionReader.ask`). Leaving with work not handed off opens the
handoff overlay first; choosing another session there
closes the screen and `run` opens it again on that one.

**A stop or a command's confirmation is a modal over the screen** (`modals/`), opened as
it becomes the oldest open question and closed when the record answering it lands. The panes
beneath keep updating: they are drawn on the base screen, which `show` reads whichever screen
is on top.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import Screen
from textual.widgets import Input, Static

from chickadee_cli.attach import NotStaged, Staged, Staging
from chickadee_cli.commands import NO_ATTACHING
from chickadee_cli.interaction import Interaction, open_in_editor
from chickadee_cli.prompt import Files
from chickadee_cli.render.diff import asked_on_leaving
from chickadee_cli.render.stream import Verbosity
from chickadee_cli.render.style import Style, Tone
from chickadee_cli.surface import DONE
from chickadee_cli.tui.composer import Composer
from chickadee_cli.tui.feed import Changed, Feed, NotServed, Snapshot
from chickadee_cli.tui.header import SHORT_BELOW, BrandHeader, banner_line, header_rows
from chickadee_cli.tui.keys import ScreenView, bound, lit
from chickadee_cli.tui.live import NOT_SERVED_HERE, SessionReader
from chickadee_cli.tui.modals import QuestionModal
from chickadee_cli.tui.modals.confirm import ConfirmModal
from chickadee_cli.tui.modals.stop import StopModal
from chickadee_cli.tui.model import (
    Connection,
    Pane,
    StaleRoute,
    TuiState,
    ZeusKind,
    drawn_as_modal,
    drawn_in_card,
)
from chickadee_cli.tui.narrow import (
    TOO_SMALL,
    Layout,
    Mode,
    layout,
    mode_for,
    tab_line,
    ticker_line,
)
from chickadee_cli.tui.overlays import Overlay
from chickadee_cli.tui.overlays.budget import Budget
from chickadee_cli.tui.overlays.command import CommandDetail
from chickadee_cli.tui.overlays.handoff import Handoff
from chickadee_cli.tui.overlays.help import Help
from chickadee_cli.tui.overlays.models import Models
from chickadee_cli.tui.overlays.palette import Palette
from chickadee_cli.tui.overlays.sessions import SessionList
from chickadee_cli.tui.overlays.task import TaskDetail
from chickadee_cli.tui.overlays.undo import Undo
from chickadee_cli.tui.panes import RoleFrame
from chickadee_cli.tui.panes.athena import AthenaFrame
from chickadee_cli.tui.panes.background import BackgroundFrame
from chickadee_cli.tui.panes.diff import DiffFrame
from chickadee_cli.tui.panes.worker import WorkerFrame
from chickadee_cli.tui.panes.zeus import ZeusFrame
from chickadee_cli.tui.pulse import Pulse
from chickadee_cli.tui.screens.diff import FullDiff
from chickadee_cli.tui.statusbar import StatusBar
from chickadee_cli.tui.talk import Talk
from chickadee_cli.tui.theme import Depth, glyphs_for, palette
from chickadee_serve.frames import ProgressFrame, RecordFrame
from chickadee_wire.client import Client
from chickadee_wire.events import Frame, Link
from chickadee_wire.models import (
    EffectCompletedP,
    ExecP,
    HandoffSummary,
    RunDiff,
    SessionInfo,
    UserTurnP,
)
from chickadee_wire.questions import OpenQuestion
from chickadee_wire.records import is_run_record

__all__ = [
    "NOT_SENT_LOST",
    "OVERLAYS",
    "ROLES",
    "ChickadeeApp",
    "Connected",
    "connector",
    "run",
]

HEADER_POLL_S: Final = 15.0
"""How often the header's tree is read again for what no record says: a commit the
person makes in their own terminal, where the terminal never reports its focus. One `git
status` on the server a read."""

INDEXING_POLL_S: Final = 1.0
"""How often liveness is read again while the repository's index is being built."""

NOT_SENT_LOST: Final = (
    "not sent: the wire is lost, so nothing would carry it. It is kept in the box; "
    "chickadee tui --continue opens the session again"
)

ROLES = ("zeus", "athena", "worker")
"""The panes `F1`, `F2` and `F3` focus, in order."""

_PANES_OF: dict[Pane, tuple[str, ...]] = {
    Pane.ZEUS: ("zeus", "background"),
    Pane.BACKGROUND: ("background",),
    Pane.ATHENA: ("athena", "diff", "worker"),
    Pane.WORKER: ("worker", "diff"),
}
"""Which frames a changed pane redraws: the diff is derived from the worker pane and the
board (`TuiState.task_diff`, `panes.diff.live`), and the worker's facts say from the board
whether its task's commit landed; a background item's words are his call's."""


OVERLAYS: dict[str, Callable[[ChickadeeApp], Overlay]] = {
    "palette": Palette,
    "help": Help,
    "budget": Budget,
    "rung": Models,
    "resume": SessionList,
    "undo": Undo,
    "handoff": Handoff,
}
"""Each overlay by the name its key's action or its command opens it with."""


@dataclass(frozen=True)
class Connected:
    """What a live screen holds: its readers, and the session's typed half (`Talk`) with the
    REPL's question renderer it answers through."""

    reader: SessionReader
    talk: Talk
    interaction: Interaction
    files: Files | None = None
    """The workspace's paths for `@`, the REPL's own cache (`prompt.Files`)."""


class ChickadeeApp(App[None]):
    CSS = """
    Screen { background: ansi_default; layers: base; }
    /* After the rule above, which would paint a modal's screen too and hide the panes. */
    ModalScreen { background: $backdrop; }
    #panes { height: 1fr; }
    #zeus { width: 36fr; }
    #centre { width: 40fr; }
    #diff { height: 1fr; }
    #right { width: 28fr; }
    #athena { height: 11fr; }
    #worker { height: 17fr; }
    #banner { height: auto; padding: 0 1; display: none; }
    /* A lost wire freezes the panes under a dimmed veil; the banner says why. */
    .-frozen { opacity: 55%; }
    #tabs, #ticker { height: 1; padding: 0 1; display: none; }
    #tabs, #ticker { text-wrap: nowrap; text-overflow: ellipsis; }
    #too-small { height: 1fr; padding: 1 2; display: none; color: $dim; }
    .-stacked #panes { layout: vertical; }
    .-stacked #zeus, .-stacked #centre, .-stacked #right { width: 100%; height: 1fr; }
    .-single #zeus, .-single #centre, .-single #right { width: 100%; }
    .-narrow BrandHeader #bird { display: none; }
    .-short BrandHeader #bird, .-short BrandHeader .tall { display: none; }
    .-short BrandHeader #brand { padding-left: 1; }
    .-fullscreen BrandHeader #bird, .-fullscreen BrandHeader .row { display: none; }
    .-fullscreen BrandHeader #line { display: block; }
    .-fullscreen BrandHeader #brand { padding-left: 1; }
    .-narrow BrandHeader #brand { padding-left: 1; }
    .-single BrandHeader #spend { width: 20; }
    """

    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False
    """Textual's own palette would open on `^P`, a key the bar draws dimmed until Part 7 builds
    the mockup's: an unbuilt key does nothing (the standing rule)."""

    BINDINGS: ClassVar[list[BindingType]] = [
        # A key off the bar acts whichever widget has focus, the composer's included.
        Binding(k.key, k.action, k.label, show=False, priority=k.priority)
        for k in bound()
        if k.action is not None
    ]

    def __init__(
        self,
        state: TuiState | None = None,
        *,
        feed: Feed | None = None,
        connect: Callable[[ChickadeeApp], Connected] | None = None,
        repo: str | None = None,
        depth: Depth | None = None,
        encoding: str | None = None,
        notice: str | None = None,
        full_screen: bool = False,
        remember: Callable[[bool], None] | None = None,
    ) -> None:
        """Static, `state` is drawn as given; live, `feed` folds what `connect(app)` delivers
        once the screen is up. `repo` is the repository's name, from whoever hosts the
        screen; `depth` and `encoding` default to the terminal's (tests pin them); `notice` is
        said in Zeus's pane as it opens (why it opened on this session). `full_screen` is how
        the header opens, and `remember(on)` keeps the person's choice for the next
        screen: the caller's, which may write it down (`chickadee_cli.tui_prefs`)."""
        self._depth: Depth = depth or "truecolor"
        self._depth_given = depth is not None
        super().__init__()
        self.feed = feed
        self.state = feed.state if feed is not None else state or TuiState()
        self._connect = connect
        self.full_screen = full_screen
        """The header as one line (`^F`)."""
        self._remember = remember
        self._keeping = threading.Lock()
        """Held while `remember` writes, so the writes land one at a time, in order."""
        self.live: SessionReader | None = None
        self.talk: Talk | None = None
        self._interaction: Interaction | None = None
        self._files: Files | None = None
        self._diff_reading = False
        self._diff_queue: list[str] = []
        """`_read_task_diff`'s one read in flight, and the tasks waiting to be read after it,
        each once however many records moved it meanwhile."""
        self._step_reading: str | None = None
        """The step whose diff is being read (`_read_step_diffs`, one at a time)."""
        self._steps_asked: set[str] = set()
        """Steps whose diff was asked for: each once, answered or not."""
        self._loop_thread = threading.get_ident()
        """The thread this app's loop runs on: a call from it runs at once (`post`)."""
        self.repo = repo
        stdout = sys.__stdout__
        self.glyphs = glyphs_for(encoding or (stdout.encoding if stdout is not None else None))
        self.pulse = Pulse(self)
        self.modal: QuestionModal | None = None
        """The question drawn over the screen now (`TuiState.asking`), if any."""
        self.view = ScreenView.BASE
        self.focus_role = "zeus"
        """The role in focus, which the narrow layouts' tab follows."""
        self.zoomed: str | None = None
        self.mode = Mode.WIDE
        self.layout = layout(Mode.WIDE, "zeus", None)
        self.overlay: Overlay | None = None
        """The overlay over the screen now, if any."""
        self.next_session: str | None = None
        """The session to open the screen on next, when the person chose another in the
        sessions overlay."""
        self.farewell: list[str] = []
        """REPL-styled lines printed in the terminal once the screen has gone (`leave`)."""
        self._notice = notice
        self._leaving = False
        self._landed = self.state.landing
        """Whether the CODE pane last drew the front page."""
        """`Ctrl+Q` is asking `/handoff` whether leaving asks first."""

    @property
    def colours(self) -> dict[str, str]:
        return palette(self._depth)

    def get_css_variables(self) -> dict[str, str]:
        if not self._depth_given:
            system = self.console.color_system
            self._depth = "truecolor" if system in ("truecolor", "256") else "16"
        # The screen beneath a modal is dimmed where colours blend; a 16-colour terminal's
        # cannot, so it shows through unchanged.
        backdrop = "black 45%" if self._depth == "truecolor" else "transparent"
        return {**super().get_css_variables(), **palette(self._depth), "backdrop": backdrop}

    def compose(self) -> ComposeResult:
        colours = self.colours
        yield BrandHeader(self.glyphs, colours, ascii_only=self.glyphs.mark.isascii())
        yield Static("", id="banner")
        yield Static("", id="tabs")
        with Horizontal(id="panes"):
            yield ZeusFrame(colours, self.glyphs, self._said)
            with Vertical(id="centre"):
                yield DiffFrame(colours, self.glyphs)
                yield BackgroundFrame(colours, self.glyphs)
            with Vertical(id="right"):
                yield AthenaFrame(colours, self.glyphs)
                yield WorkerFrame(colours, self.glyphs)
        yield FullDiff(colours, self.glyphs)
        yield Static("", id="ticker")
        yield Static(Content(TOO_SMALL), id="too-small")
        yield StatusBar(colours)

    def on_mount(self) -> None:
        self.zeus.composer.focus()
        self.zeus.composer.browses = self._browses
        code = self.code
        code.pictures = not self.glyphs.mark.isascii()
        self.relayout()
        self.show(self.state)
        if self._connect is not None:
            connected = self._connect(self)
            self.live, self.talk = connected.reader, connected.talk
            staging = connected.talk.staging
            staging.changed = lambda: self.post(
                lambda: self.zeus.chips.show(staging.items(), staging.over())
            )
            self._interaction = connected.interaction
            self._files = connected.files
            if self._files is not None:
                self.zeus.composer.paths = self._files.cached
                self.code.paths = self._files.cached
                self._read_files()
            self.talk.start()
            self.live.start()
            # Liveness moves on its own while the index is built: read again until
            # the header no longer says so, since no record arrives when the build ends.
            self.set_interval(INDEXING_POLL_S, self._indexing_poll)
            # What no record says: the person's own commit; `on_app_focus` too.
            self.set_interval(HEADER_POLL_S, self._header_poll)
            # The front page's recent sessions, read once.
            self.read(lambda c: c.sessions(), self._recent, lambda _why: self._recent(()))
        if self._notice is not None:
            self.note(Style(tty=True).lines(self._notice, Tone.DIM))

    def _header_poll(self) -> None:
        if self.live is not None:
            self.live.refresh({StaleRoute.HEADER})

    def on_app_focus(self, _event: events.AppFocus) -> None:
        """The terminal gave the screen its focus back: the person may have committed,
        or changed the tree, in another window; the header's tree is read again."""
        self._header_poll()

    def _indexing_poll(self) -> None:
        # While a build runs only: a failure stays said until `chickadee init`, and
        # nothing moves meanwhile to read again.
        live = self.state.liveness
        if self.live is not None and live is not None and live.index_progress is not None:
            self.live.refresh({StaleRoute.LIVENESS})

    def on_unmount(self) -> None:
        if self.live is not None:
            self.live.close()
        if self.talk is not None:
            self.talk.close()

    @property
    def base(self) -> Screen[None]:
        """The screen the panes are on, beneath any modal: `App.query` reads only the screen on
        top."""
        return self.screen_stack[0]

    @property
    def zeus(self) -> ZeusFrame:
        return self.base.query_one(ZeusFrame)

    @property
    def code(self) -> DiffFrame:
        return self.base.query_one(DiffFrame)

    def _said(self) -> list[str]:
        """What the person said to Zeus, oldest first: the composer's history."""
        return [
            e.record.text
            for e in self.state.zeus.entries
            if e.kind is ZeusKind.YOU and isinstance(e.record, UserTurnP)
        ]

    def post(self, call: Callable[[], None]) -> None:
        """`call()` on this app's loop: at once from it, else handed to it."""
        if threading.get_ident() == self._loop_thread:
            call()
        else:
            self.call_from_thread(call)

    def edit(self, text: str) -> str | None:
        """`text` in `$EDITOR`, the terminal handed to it meanwhile (the REPL's own editor,
        `interaction.open_in_editor`); from any thread."""

        def suspended() -> str | None:
            with self.suspend():
                return open_in_editor(text)

        if threading.get_ident() == self._loop_thread:
            return suspended()
        return self.call_from_thread(suspended)

    # -- what the server sends, on this event loop ----------------------------------------

    def take(self, frame: Frame) -> None:
        self._moved(self._feed().apply(frame))
        ended = isinstance(frame, ProgressFrame) and frame.kind in ("answer", "error")
        if ended and self._files is not None:
            # A turn ended: the workspace may hold new paths, as the REPL reads it then.
            self._files.stale()
            self._read_files()
        if (
            self._interaction is not None
            and isinstance(frame, RecordFrame)
            and not is_run_record(frame)
        ):
            self._interaction.feed(frame)  # a question, shown as it opens, as the REPL does

    def take_snapshot(self, value: Snapshot) -> None:
        self._moved(self._feed().snapshot(value))

    def take_lost(self, what: str, why: str, stream: bool) -> None:
        self._moved(self._feed().lost(what, why, stream=stream))

    def take_link(self, link: Link) -> None:
        self._moved(self._feed().link(link))

    def _feed(self) -> Feed:
        if self.feed is None:
            raise RuntimeError("a static screen is given its state, not frames")
        return self.feed

    def note(self, lines: Sequence[str]) -> None:
        """Lines said in Zeus's pane, from any thread."""
        self.fold(lambda feed: feed.note(lines))

    def fold(self, change: Callable[[Feed], Changed]) -> None:
        """A change to the feed, from any thread, folded and drawn on this loop."""
        self.post(lambda: self._moved(change(self._feed())))

    def _moved(self, changed: Changed) -> None:
        # Shutting down, the panes are being removed before `on_unmount` closes the readers: a
        # frame landing between is folded, not drawn (there is nothing left to draw it on).
        if not self.is_running:
            return
        if changed.panes:
            self.show(self._feed().state, changed.panes)
        refresh = set(changed.refresh)
        if self.view is not ScreenView.DIFF and not isinstance(self.overlay, Handoff):
            # Read only while the diff screen or the handoff overlay shows it (A14).
            refresh.discard(StaleRoute.HANDOFF)
        if StaleRoute.TASK_DIFF in refresh:
            refresh.discard(StaleRoute.TASK_DIFF)
            self._read_task_diff()
        if StaleRoute.STEP_DIFFS in refresh:
            refresh.discard(StaleRoute.STEP_DIFFS)
            self._read_step_diffs()
        if refresh and self.live is not None:
            self.live.refresh(refresh)

    def _read_files(self) -> None:
        """The workspace's paths, read off the loop into the cache `@` reads (`Files`)."""
        files = self._files
        if files is not None:
            self.read(lambda _c: files.paths(), lambda _paths: None, lambda _why: None)

    def _read_task_diff(self, task_id: str | None = None) -> None:
        """A task's diff, read again: `task_id`, else the worker's task and
        the one the CODE pane has open while it may still move or was never read.
        One read in flight; a task waiting is read once after it, so a catch-up of many steps
        is two reads, not one per step. A server that does not build the route is not asked
        again."""
        state = self.state
        if state.athena.run_id is None or StaleRoute.TASK_DIFF in state.not_served:
            return
        if task_id is not None:
            wanted = [task_id]
        else:
            chosen = self.code.open_task(state)
            wanted = [t for t in (state.worker.current, chosen) if t is not None]
            if chosen is not None and chosen != state.worker.current and chosen in state.diffs:
                wanted.remove(chosen)  # a task no longer worked does not move
        self._diff_queue += [t for t in wanted if t not in self._diff_queue]
        self._next_task_diff()

    def _next_task_diff(self) -> None:
        run_id = self.state.athena.run_id
        if self._diff_reading or not self._diff_queue or run_id is None:
            return
        self._diff_reading = True
        task_id = self._diff_queue.pop(0)

        def settled() -> None:
            self._diff_reading = False
            self._next_task_diff()

        def done(diff: RunDiff) -> None:
            self._moved(self._feed().snapshot(diff))
            settled()

        def failed(why: str) -> None:
            if why == NOT_SERVED_HERE:
                self._diff_queue.clear()
                self._moved(self._feed().snapshot(NotServed(StaleRoute.TASK_DIFF)))
            settled()

        self.read(lambda c: c.task_diff(run_id, task_id), done, failed)

    def _read_step_diffs(self) -> None:
        """Each step that changed files, its diff read once: the line counts its card
        shows. One read in flight; the next step's after it. A server that does not build the
        route is not asked again: its cards show the paths alone."""
        state = self.state
        run_id = state.athena.run_id
        if self._step_reading is not None or run_id is None:
            return
        if StaleRoute.STEP_DIFFS in state.not_served:
            return
        wanted = [
            entry.record.operation_id
            for entry in (
                *state.athena.tail,
                *(e for t in state.worker.tasks.values() for e in t.entries),
            )
            if isinstance(entry.record, EffectCompletedP) and entry.record.observed_surface
        ]
        todo = [op for op in wanted if op not in self._steps_asked]
        if not todo:
            return
        op = self._step_reading = todo[0]
        self._steps_asked.add(op)

        def settled() -> None:
            self._step_reading = None
            self._read_step_diffs()

        def done(diff: RunDiff) -> None:
            self._moved(self._feed().snapshot(diff))
            settled()

        def failed(why: str) -> None:
            if why == NOT_SERVED_HERE:
                self._moved(self._feed().snapshot(NotServed(StaleRoute.STEP_DIFFS)))
            settled()

        self.read(lambda c: c.step_diff(run_id, op), done, failed)

    # -- drawing the state -----------------------------------------------------------------

    def show(self, state: TuiState, changed: Iterable[Pane] | None = None) -> None:
        """Draw `state`: every part of the screen, or only those `changed` names."""
        self.state = state
        panes = set(Pane) if changed is None else set(changed)
        frames = {f for p in panes for f in _PANES_OF.get(p, ())}
        if state.landing != self._landed:
            # The front page gives way to the mockup on the first words: a change
            # the feed names as Zeus's, which the CODE pane draws too.
            self._landed = state.landing
            frames.add("diff")
        base = self.base
        for frame in base.query(RoleFrame):
            if frame.role in frames:
                frame.show(state)
        if Pane.BACKGROUND in panes and not self._background_placed():
            self.relayout()  # it comes or goes with the work, and `F5` with it
        if not self.state.background.running and self.focus_role == "background":
            self.action_focus_role("diff")
        if Pane.HEADER in panes:
            base.query_one(BrandHeader).show(state, self.repo)
            self._draw_banner(state)
        diff = self.full_diff
        if self.view is ScreenView.DIFF and panes & {Pane.ATHENA, Pane.WORKER, Pane.DIFF}:
            diff.show(state)
        if self.layout.ticker:
            ticker = ticker_line(state, self.layout.ticker, self.glyphs, self.colours)
            base.query_one("#ticker", Static).update(ticker)
        self._draw_bar()
        self.pulse.follow(state)
        if self.overlay is not None:
            self.overlay.show(state)
        if Pane.STOPS in panes:
            self._follow_question(state.asking)
            self._draw_banner(state)  # a stop put back is the banner's

    def _recent(self, sessions: tuple[SessionInfo, ...]) -> None:
        code = self.base.query_one(DiffFrame)
        code.recent = sessions
        code.show(self.state)

    def _draw_banner(self, state: TuiState) -> None:
        base = self.base
        line = banner_line(state, self.colours)
        banner = base.query_one("#banner", Static)
        banner.display = line is not None and self.mode is not Mode.TOO_SMALL
        banner.update(line or "")
        frozen = state.connection is not Connection.LIVE
        for name in ("#panes", "#fulldiff"):
            base.query_one(name).set_class(frozen, "-frozen")

    def _follow_question(self, asked: OpenQuestion | None) -> None:
        """Close the modal whose question was answered (by anyone), and open the next."""
        if self.modal is not None and (asked is None or asked.id != self.modal.asked.id):
            self.modal.dismiss()
            self.modal = None
        if asked is None or self.modal is not None:
            return
        if isinstance(self.overlay, Handoff) and self.overlay.leaving:
            return  # the person is leaving: the question stays open for the next screen
        colours = self.colours
        deadline = self.state.deadline(asked)
        if asked.stop is not None:
            shown = self.state.stop_shown.get(asked.id, asked.stop.round)
            # No deadline: with the person here a stop waits until answered.
            self.modal = StopModal(
                asked, colours, self.answer, self.edit, shown=shown, later=self._stop_later
            )
        else:
            self.modal = ConfirmModal(asked, colours, self.answer, deadline=deadline)
        self.push_screen(self.modal)

    def answer(
        self,
        asked: OpenQuestion,
        choice: str,
        note: str | None,
        text: str | None,
        failed: Callable[[], None],
    ) -> None:
        """A modal's answer, sent by `Talk`; `failed` back on this loop if it was refused."""
        if self.talk is None:
            self.note(Style(tty=True).lines("not connected: nothing is sent"))
            failed()
            return
        self.talk.answer(asked, choice, note, text, lambda: self.post(failed))

    # -- keys ------------------------------------------------------------------------------

    def action_focus_role(self, role: str) -> None:
        self.focus_role = role
        self.relayout()  # a narrow layout's tab follows the focus
        if role == "zeus":
            self.zeus.composer.focus()  # his pane's focus is the box you type in
        else:
            self.base.query_one(f"#{role}").focus()
            if role == "diff":
                self.code.show(self.state)  # its hint, drawn while it has the focus

    def action_toggle_diff(self) -> None:
        """`^D` (and `/diff`): the diff screen in place of the panes, and back."""
        if self.view is ScreenView.DIFF:
            self.view = ScreenView.BASE
            self.relayout()
            self.action_focus_role(self.focus_role)
            self._draw_bar()
            return
        self.view = ScreenView.DIFF
        self.relayout()
        diff = self.full_diff
        diff.show(self.state)
        diff.focus()
        if self.live is not None:
            self.live.refresh({StaleRoute.HANDOFF})
        self._draw_bar()

    def open_diff(self) -> None:
        if self.view is not ScreenView.DIFF:
            self.action_toggle_diff()

    def on_full_diff_closed(self, _message: FullDiff.Closed) -> None:
        self.action_toggle_diff()

    def on_full_diff_moved(self, _message: FullDiff.Moved) -> None:
        self._draw_bar()

    def action_toggle_fullscreen(self) -> None:
        """`^F`, `/fullscreen` or its button: the header to one line and back, kept
        for the next screen by `remember`, the CLI's (the screen never writes a file itself:
        it reaches no further than the wire)."""
        self.full_screen = not self.full_screen
        self.relayout()
        if self.overlay is not None:
            self.overlay.below_header()
        if self._remember is not None:
            # On a thread: the write syncs to disk, which may be slow, and the screen must not
            # wait on it.
            self.run_worker(self._keep, thread=True, group="prefs")

    def _keep(self) -> None:
        """`remember` the choice as it is when the write starts, one write at a time
        (`_keeping`): two toggles' writes cannot land out of order, and the last to land is
        the last choice. A failure is said, and the toggle stands."""
        assert self._remember is not None
        with self._keeping:
            on = self.full_screen
            try:
                self._remember(on)
            except OSError as exc:
                said = f"full screen {'on' if on else 'off'} for now, not remembered: {exc}"
                self.call_from_thread(self.notify, said, severity="warning")

    async def action_on_diff(self, action: str) -> None:
        """A diff screen key's button in the bar: its action on the diff screen, which binds it
        (`statusbar`), as pressing the key runs it."""
        await self.full_diff.run_action(action)

    @property
    def header_height(self) -> int:
        """The header's rows now (`header.header_rows`): where an overlay's box starts."""
        return header_rows(self.size.height, full_screen=self.full_screen)

    def action_zoom_in(self) -> None:
        """`Ctrl+Up`: the focused pane alone (the zoom); the ticker names the others."""
        self.zoomed = self._focused_pane()
        self.relayout()

    def action_zoom_out(self) -> None:
        self.zoomed = None
        self.relayout()

    def _focused_pane(self) -> str:
        node = self.focused
        while node is not None:
            if isinstance(node, RoleFrame):
                return node.role
            node = node.parent
        return "zeus"

    @property
    def full_diff(self) -> FullDiff:
        return self.base.query_one(FullDiff)

    def get_default_screen(self) -> Screen[None]:
        return _Base()

    def relayout(self) -> None:
        """Show what the size, the view, the focus and the zoom call for (`narrow.layout`)."""
        base = self.base
        size = self.size
        self.mode = mode_for(size.width, size.height)
        lay: Layout = layout(self.mode, self.focus_role, self.zoomed)
        self.layout = lay
        small = self.mode is Mode.TOO_SMALL
        base.set_class(size.width < 100, "-narrow")
        base.set_class(size.height < SHORT_BELOW and not self.full_screen, "-short")
        base.set_class(self.full_screen, "-fullscreen")
        base.set_class(self.mode is Mode.SINGLE, "-single")
        base.set_class(self.mode is Mode.STACKED and self.zoomed is None, "-stacked")
        header = base.query_one(BrandHeader)
        header.compact = size.width < 80
        header.full_screen = self.full_screen
        header.styles.height = self.header_height  # `header_rows`: the one rule for it
        header.display = not small
        diff_view = self.view is ScreenView.DIFF
        base.query_one("#panes").display = not small and not diff_view
        self.full_diff.display = not small and diff_view
        for frame in base.query(RoleFrame):
            frame.display = frame.role in lay.panes and self._shows(frame.role)
        base.query_one("#centre").display = bool(lay.panes & {"diff", "background"})
        base.query_one("#right").display = bool(lay.panes & {"athena", "worker"})
        tabs = base.query_one("#tabs", Static)
        tabs.display = bool(lay.tabs) and not diff_view and not small
        if tabs.display:
            tabs.update(tab_line(lay, self.colours))
        ticker = base.query_one("#ticker", Static)
        ticker.display = bool(lay.ticker) and not diff_view and not small
        if ticker.display:
            ticker.update(ticker_line(self.state, lay.ticker, self.glyphs, self.colours))
        base.query_one("#too-small").display = small
        base.query_one(StatusBar).display = not small
        header.show(self.state, self.repo)
        self._draw_banner(self.state)

    def _shows(self, role: str) -> bool:
        """BACKGROUND only while an agent or a job runs; every other pane always."""
        return role != "background" or self.state.background.running

    def _background_placed(self) -> bool:
        """Whether BACKGROUND is shown exactly when it should be, in this layout."""
        frame = self.base.query_one(BackgroundFrame)
        return frame.display == (
            "background" in self.layout.panes and self._shows("background")
        )

    def _draw_bar(self) -> None:
        self.base.query_one(StatusBar).show(
            self.state, view=self.view, hunks=self.full_diff.hunks_shown
        )

    def action_slash(self) -> None:
        """`/` from any pane: the composer, the command list open."""
        composer = self.zeus.composer
        composer.focus()
        if not composer.text:
            composer.insert("/")

    def action_open_stop(self) -> None:
        """`^S`: the stop put back as a banner, opened again to answer."""
        waiting = self.state.waiting_stop
        if waiting is None:
            return
        self.state.stops_put_back.discard(waiting.id)
        self._follow_question(self.state.asking)
        self._draw_banner(self.state)
        self._draw_bar()

    def _stop_later(self) -> None:
        """`Esc` on an open stop: back to its banner, unanswered; the run goes on waiting."""
        if self.modal is not None:
            self.state.stops_put_back.add(self.modal.asked.id)
        self._follow_question(self.state.asking)
        self._draw_banner(self.state)
        self._draw_bar()

    def action_pause_team(self) -> None:
        """`^G`: pause the team, as `/pause` typed does."""
        if self.talk is not None:
            self.talk.command("/pause")

    def action_interrupt(self) -> None:
        """`Esc`: the command list closes first; otherwise the turn in flight stops."""
        if self.zeus.close_completions():
            return
        if self.talk is not None:
            self.talk.interrupt()

    def action_toggle_tools(self) -> None:
        self.zeus.toggle_tools()

    def action_palette(self) -> None:
        self.open_overlay("palette")

    def action_overlay(self, name: str) -> None:
        self.open_overlay(name)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """A key that does not apply now, in this view, to this state, does nothing (it is
        drawn dimmed). Under an overlay none of the screen's keys applies, so the overlay's
        own are reached. (`?` in the composer is the composer's: `Composer.HelpRequested`.)"""
        if self.overlay is not None and action != "quit":
            return False
        if action == "interrupt" and self.code.takes_escape():
            return False  # the CODE pane's `Esc` closes its search or a read-only file
        for key in bound():
            if key.action is not None and key.action.split("(")[0] == action:
                return lit(key, self.state, view=self.view)
        return True

    def on_key(self, event: events.Key) -> None:
        """Everything typed goes to Zeus. A character typed in a read-only pane lands in
        the composer, which takes the focus."""
        if isinstance(self.focused, Composer | Input) or not event.is_printable:
            return  # the composer's own, and the CODE pane's search box's
        searching = isinstance(self.focused, DiffFrame) and event.key == "slash"
        if not event.character or searching:
            return  # `/` in the CODE pane opens its search
        if self.modal is not None or self.overlay is not None or self.view is ScreenView.DIFF:
            return  # a modal's keys are its own, an overlay's, and the diff screen's
        self.action_focus_role("zeus")  # a narrow layout shows his pane again
        self.zeus.composer.insert(event.character)
        event.stop()

    # -- what the person types -------------------------------------------------------------

    def on_composer_submitted(self, message: Composer.Submitted) -> None:
        if self.state.connection is Connection.LOST:
            # Never queued where nothing would send it; kept to send
            # from a screen opened again.
            self.zeus.composer.load_text(message.text)
            self.note(Style(tty=True).lines(NOT_SENT_LOST, Tone.BAD))
            return
        if self.talk is None:
            if self.feed is not None:
                self.note(Style(tty=True).lines("not connected: nothing is sent"))
            return
        self.talk.submit(message.text)

    def _browses(self) -> bool:
        """Whether the front page offers recent sessions to choose: the screen is
        landing, the CODE pane is shown, and it lists one."""
        return self.state.landing and "diff" in self.layout.panes and bool(self.code.choices())

    def on_composer_browsed(self, message: Composer.Browsed) -> None:
        self.code.choose(message.move)

    def on_composer_opened(self, _message: Composer.Opened) -> None:
        """Enter in the empty composer on the front page's chosen session."""
        chosen = self.code.chosen_session()
        if chosen is not None:
            self.switch_session(chosen)

    def on_diff_frame_session_opened(self, message: DiffFrame.SessionOpened) -> None:
        """Enter on the chosen session with the CODE pane focused, or a click on its row:
        the screen reopens on it, as the sessions overlay's Enter does, its
        records drawn from seq 0."""
        self.switch_session(message.session_id)

    def on_composer_image_requested(self, _message: Composer.ImageRequested) -> None:
        """`Ctrl+V`: the clipboard's image staged, read off the loop, since the reader
        is a process and the preview a request."""
        self._attaching(lambda staging: staging.stage_clipboard())

    def on_composer_file_dropped(self, message: Composer.FileDropped) -> None:
        """A dragged file: attached; but a text file the workspace lists,
        under the directory this screen started in, is mentioned as `@path`, since Zeus reads
        the live file and a copy would go stale."""
        path = message.path
        listed = _listed(path, self.zeus.composer.paths())

        def attach(staging: Staging) -> Staged | NotStaged:
            checked = staging.check_file(path)
            if isinstance(checked, Staged) and checked.preview.kind == "text" and listed:
                mention = f"@{listed} "
                self.post(lambda: self.put_in_composer(self.zeus.composer.text + mention))
                self.note(Style(tty=True).lines(f"{MENTIONED} @{listed}", Tone.DIM))
                return checked
            return staging.add(checked) if isinstance(checked, Staged) else checked

        self._attaching(attach)

    def _attaching(self, stage: Callable[[Staging], Staged | NotStaged]) -> None:
        talk = self.talk
        if talk is None or not talk.staging.offered:
            self.note(Style(tty=True).lines(NO_ATTACHING, Tone.BAD))
            return

        def run() -> None:
            done = stage(talk.staging)
            if isinstance(done, NotStaged):
                self.note(Style(tty=True).lines(f"not attached: {done.why}", Tone.BAD))

        threading.Thread(target=run, name="tui-attach", daemon=True).start()

    def on_composer_edit_requested(self, message: Composer.EditRequested) -> None:
        saved = self.edit(message.text)
        if saved is not None:
            self.zeus.composer.load_text(saved.rstrip("\n"))

    def on_composer_help_requested(self, _message: Composer.HelpRequested) -> None:
        """`?` in the empty composer, under the same rule as the bar's `?`."""
        if self.check_action("overlay", ("help",)):
            self.open_overlay("help")

    # -- overlays (their `Host`) -----------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self.feed.session_id if self.feed is not None else ""

    def open_overlay(self, name: str) -> None:
        """The overlay `name` over the screen."""
        self.open_view(OVERLAYS[name](self))

    def open_view(self, overlay: Overlay) -> None:
        """`overlay` over the screen; one at a time, and none over a modal (a stop has
        the screen)."""
        if self.overlay is not None or self.modal is not None:
            return
        self.overlay = overlay
        self.push_screen(overlay, lambda _closed: self._overlay_closed(overlay))
        self._draw_bar()

    def on_role_frame_details(self, message: RoleFrame.Details) -> None:
        """`Enter` on a task in Athena's pane, or on a command in the worker's."""
        if message.role == "athena":
            self.open_view(TaskDetail(self, message.chosen))
            return
        task = self.state.worker.task
        entry = (
            next((e for e in task.entries if str(e.seq) == message.chosen), None)
            if task is not None
            else None
        )
        if task is not None and entry is not None and isinstance(entry.record, ExecP):
            self.open_view(CommandDetail(self, entry.record, task.task_id))

    def on_diff_frame_task_chosen(self, message: DiffFrame.TaskChosen) -> None:
        """The person opened another task in the CODE pane: its diff, if unread."""
        if message.task_id not in self.state.diffs:
            self._read_task_diff(message.task_id)

    def on_diff_frame_file_chosen(self, message: DiffFrame.FileChosen) -> None:
        """A file no task changed, chosen in the CODE pane's search: read once."""

        def failed(why: str) -> None:
            if self.feed is not None:  # a static screen has nowhere to say it
                said = f"{message.path} could not be read: {why}"
                self.note(Style(tty=True).lines(said, Tone.BAD))

        self.read(
            lambda c: c.file_text(self.session_id, message.path), self.code.show_file, failed
        )

    def _overlay_closed(self, overlay: Overlay) -> None:
        if self.overlay is overlay:
            self.overlay = None
        self._draw_bar()

    def command(self, line: str) -> None:
        if self.talk is None:
            self.note(Style(tty=True).lines("not connected: nothing is sent"))
            return
        self.talk.command(line)

    def submit(self, line: str) -> None:
        if self.talk is None:
            self.note(Style(tty=True).lines("not connected: nothing is sent"))
            return
        self.talk.submit(line)

    def put_in_composer(self, text: str) -> None:
        composer = self.zeus.composer
        self.action_focus_role("zeus")
        composer.load_text(text)
        composer.move_cursor(composer.document.end)

    def read[T](
        self,
        read: Callable[[Client], T],
        deliver: Callable[[T], None],
        failed: Callable[[str], None],
    ) -> None:
        if self.live is None:
            failed("not connected")
            return
        self.live.ask(
            read,
            lambda value: self.post(lambda: deliver(value)),
            lambda why: self.post(lambda: failed(why)),
        )

    def refresh_routes(self, routes: Iterable[str]) -> None:
        if self.live is not None:
            self.live.refresh(routes)

    def switch_session(self, session_id: str) -> None:
        self.next_session = session_id
        self.exit()

    def leave(self, lines: list[str]) -> None:
        self.farewell = list(lines)
        self.exit()

    # -- leaving ---------------------------------------------------------------------------

    async def action_quit(self) -> None:
        """`Ctrl+Q`: with work not handed off, the handoff overlay first, as the REPL asks when
        it closes; a replay has nothing to hand off. If `/handoff` cannot be asked, that is
        said, and a second `Ctrl+Q` leaves."""
        replay = self.talk is not None and self.talk.replay
        if self.live is None or replay or self._leaving:
            self.exit()
            return
        self._leaving = True
        self.read(lambda c: c.handoff(self.session_id), self._leave_over, self._leave_unasked)

    def _leave_over(self, summary: HandoffSummary) -> None:
        if not asked_on_leaving(summary):
            self.exit()
            return
        self._moved(self._feed().snapshot(summary))
        # Leaving puts the handoff over whatever is open; a question stays open on the server.
        for open_now in (self.modal, self.overlay):
            if open_now is not None:
                open_now.dismiss()
        self.modal = self.overlay = None
        overlay = Handoff(self, leaving=True)
        self.overlay = overlay
        self.push_screen(overlay, lambda _closed: self._leave_closed(overlay))

    def _leave_closed(self, overlay: Handoff) -> None:
        """The leaving overlay went without leaving (`d` shows the diff): the next `Ctrl+Q`
        asks again, as the first did. Leaving itself exits the app, so this never undoes it."""
        self._leaving = False
        self._overlay_closed(overlay)

    def _leave_unasked(self, why: str) -> None:
        self.note(
            Style(tty=True).lines(
                f"/handoff: the session could not be asked ({why}); Ctrl+Q again leaves",
                Tone.BAD,
            )
        )


class _Base(Screen[None]):
    """The screen layout E is drawn on. A resize reaches the screen, not the app (Textual's
    `App._on_resize` stops it), so the screen asks the app to lay out again."""

    def on_resize(self, _event: events.Resize) -> None:
        app = self.app
        if isinstance(app, ChickadeeApp):
            app.relayout()


MENTIONED: Final = "in the repository: mentioned, not attached (/attach attaches it):"


def _listed(path: Path, paths: Sequence[str]) -> str | None:
    """`path` as `@` lists it, when it is under the directory this screen started in and the
    workspace lists it; None otherwise, which is every path on another machine's server."""
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return None
    return relative if relative in paths else None


class _Screen:
    """What `Talk` changes on screen (`talk.Screen`), each run on the app's loop."""

    def __init__(self, app: ChickadeeApp) -> None:
        self._app = app

    def lines(self, lines: Sequence[str]) -> None:
        self._app.note(lines)

    def sent(self, text: str) -> None:
        self._app.fold(lambda feed: feed.sent(text))

    def unsent(self, text: str) -> None:
        self._app.fold(lambda feed: feed.unsent(text))

    def prompt(self, text: str | None) -> None:
        app = self._app
        app.post(lambda: app.zeus.asking(text))

    def put_back(self, text: str) -> None:
        app = self._app
        app.post(lambda: app.zeus.composer.load_text(text))

    def verbosity(self, verbosity: Verbosity) -> None:
        app = self._app
        app.post(lambda: app.zeus.set_verbosity(verbosity))

    def clear(self) -> None:
        app = self._app
        app.post(lambda: app.zeus.clear_view())

    def leave(self) -> None:
        self._app.post(self._app.exit)

    def open_diff(self) -> None:
        self._app.post(self._app.open_diff)

    def open_overlay(self, name: str) -> None:
        app = self._app
        app.post(lambda: app.open_overlay(name))

    def toggle_fullscreen(self) -> None:
        self._app.post(self._app.action_toggle_fullscreen)


def connector(
    client: Client, session_id: str, *, resume_paused: bool = False
) -> Callable[[ChickadeeApp], Connected]:
    """How a screen reads and talks to `client`'s session: its readers deliver to the app's own
    loop, from whichever thread they read on; the REPL's question renderer prints into Zeus's
    pane and opens `$EDITOR` through the app; `Talk` handles what the person types."""

    def connect(app: ChickadeeApp) -> Connected:
        reader = SessionReader(
            client,
            session_id,
            on_frame=lambda frame: app.call_from_thread(app.take, frame),
            on_snapshot=lambda value: app.call_from_thread(app.take_snapshot, value),
            on_lost=lambda what, why, stream: app.call_from_thread(
                app.take_lost, what, why, stream
            ),
            on_link=lambda link: app.call_from_thread(app.take_link, link),
        )
        if client.server is not None:
            app.take_snapshot(client.server)  # how long a question waits here
        interaction = Interaction(
            client,
            session_id,
            Style(tty=True),
            lambda line: app.note([line]),
            app.edit,
            # A stop or an approval is a modal; his call's question is drawn in its card.
            draws=lambda asked: drawn_as_modal(asked) or drawn_in_card(asked, app.state.zeus),
            # A modal's answer is its own, never the composer's next line.
            answers=drawn_as_modal,
        )
        talk = Talk(
            client,
            session_id,
            _Screen(app),
            interaction,
            follow=reader.follow,
            resume_paused=resume_paused,
        )
        files = Files(lambda: client.files(session_id))
        return Connected(reader, talk, interaction, files)

    return connect


def run(
    client: Client,
    session_id: str,
    *,
    repo: str | None,
    resume_paused: bool = False,
    full_screen: bool = False,
    remember: Callable[[bool], None] | None = None,
) -> int:
    """The screen over `client`'s session `session_id`, until the person leaves (`Ctrl+Q`); the
    exit
    code is returned. `resume_paused`: the session was opened again (`--continue`, `--session`),
    so
    a run left paused continues, as the REPL's does. Choosing another session in the
    sessions overlay opens the screen again on it, where `/resume` continues its run.
    What leaving did is printed once the screen has gone. The caller owns `client` and closes
    it after. `full_screen` and `remember` are the person's header choice and how it is kept
    (`ChickadeeApp`); a screen opened again on another session opens as the last one left."""
    sid: str | None = session_id
    notice: str | None = None
    while sid is not None:
        connect = connector(client, sid, resume_paused=resume_paused and notice is None)
        app = ChickadeeApp(
            feed=Feed(sid),
            connect=connect,
            repo=repo,
            notice=notice,
            full_screen=full_screen,
            remember=remember,
        )
        app.run()
        full_screen = app.full_screen
        chosen = app.next_session
        for line in app.farewell:
            print(line)
        notice = f"opened on session {chosen}; /resume continues a run it left unfinished"
        sid = chosen
    return DONE
