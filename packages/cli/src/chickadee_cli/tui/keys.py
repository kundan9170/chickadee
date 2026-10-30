"""The TUI's keys (the mockup's status bar, and
the mockup's on the diff screen): one table per view that both binds the keys and draws the
bar, so a key is never bound and drawn dimmed, or drawn lit and not bound. Under an overlay
(`overlays/`) none of them applies: the overlay's own keys do (the app's `check_action`).

**Dimmed, never hidden** ("keys that do not apply in the stage are dimmed, not hidden, so
the bar does not jump"). A key is lit when its action is built and it applies now: in this
view, to this state, and (for the diff screen's hunk keys) where hunks are shown. A key whose
action is not built yet has `action = None`: drawn dimmed, bound to nothing. The part that
builds an action sets it here, with its `when`, and a test that the key lights up where it
applies and is dimmed where it does not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from chickadee_cli.tui.model import TuiState, WorkerKind
from chickadee_cli.tui.names import PANE_WORDS, ROLE_WORDS

__all__ = [
    "ATHENA_KEYS",
    "BACKGROUND_KEYS",
    "BAR",
    "CODE_KEYS",
    "CODE_OFF_KEYS",
    "DIFF_BAR",
    "DIFF_OFF_BAR",
    "FRONT_KEYS",
    "OFF_BAR",
    "WORKER_KEYS",
    "Key",
    "ScreenView",
    "bar_for",
    "bound",
    "command_keys",
    "diff_bound",
    "lit",
]


class ScreenView(StrEnum):
    """What the screen shows between the header and the bar: layout E's panes, or the diff."""

    BASE = "base"
    DIFF = "diff"


_BASE: Final = frozenset({ScreenView.BASE})
_DIFF: Final = frozenset({ScreenView.DIFF})
_BOTH: Final = frozenset(ScreenView)


def _always(_state: TuiState) -> bool:
    return True


def _no_modal(state: TuiState) -> bool:
    """A stop or a command's confirmation has the screen: the keys of the screen beneath
    it are dimmed, as the mockup dims everything under a modal, and do nothing."""
    return state.asking is None


def _stop_waiting(state: TuiState) -> bool:
    """A stop put back waits as a banner, and no modal has the screen: `^S` opens it."""
    return _no_modal(state) and state.waiting_stop is not None


def _background(state: TuiState) -> bool:
    """The BACKGROUND pane shows: an agent or a job is running."""
    return state.background.running


def _background_key(state: TuiState) -> bool:
    return _no_modal(state) and _background(state)


def _diffable(state: TuiState) -> bool:
    """The session has a run whose changes the diff screen shows (the owner's call: dimmed
    before any run), and no modal has the screen."""
    return _no_modal(state) and state.athena.run_id is not None


@dataclass(frozen=True)
class Key:
    key: str
    """Textual's name for it (`f1`, `ctrl+p`, `slash`)."""
    shown: str
    """How the bar prints it (`F1`, `^P`, `/`)."""
    label: str
    action: str | None = None
    """The action it runs; None until the part that builds it."""
    builds_in: str | None = None
    """For a key not built yet, the part (or wire request) that builds it."""
    when: Callable[[TuiState], bool] = _always
    """Whether it applies to the state now; a key that does not is dimmed."""
    views: frozenset[ScreenView] = _BASE
    """The views it applies in; in another it is dimmed and does nothing."""
    hunks: bool = False
    """It applies only where the diff screen shows hunks (`[`, `]`)."""
    priority: bool = False
    """It acts whichever widget has focus, the composer's included (whose own Ctrl+D deletes a
    character, and Ctrl+U the line): Textual's priority binding."""
    command: str | None = None
    """The slash command that does the same (`/diff`), for the command palette's key column."""
    drawn: Callable[[TuiState], bool] = _always
    """Whether the bar draws it at all. Every key is, dimmed or lit, but one: `F5`, drawn only
    while the BACKGROUND pane it focuses shows (the owner's call, 2026-10-26: the pane comes
    and goes, and its key with it)."""


BAR: Final[tuple[tuple[Key, ...], ...]] = (
    (
        Key("f1", "F1", ROLE_WORDS["zeus"], "focus_role('zeus')", when=_no_modal),
        Key("f2", "F2", ROLE_WORDS["athena"], "focus_role('athena')", when=_no_modal),
        Key("f3", "F3", ROLE_WORDS["worker"], "focus_role('worker')", when=_no_modal),
        Key("f4", "F4", PANE_WORDS["diff"], "focus_role('diff')", when=_no_modal),
        Key(
            "f5",
            "F5",
            PANE_WORDS["background"],
            "focus_role('background')",
            when=_background_key,
            drawn=_background,
        ),
    ),
    (
        Key("ctrl+p", "^P", "commands", "palette", when=_no_modal, views=_BOTH, priority=True),
        Key(
            "ctrl+d",
            "^D",
            "diff",
            "toggle_diff",
            when=_diffable,
            views=_BOTH,
            priority=True,
            command="/diff",
        ),
        Key(
            "ctrl+u",
            "^U",
            "undo",
            "overlay('undo')",
            when=_no_modal,
            priority=True,
            command="/undo",
        ),
        Key(
            "ctrl+f",
            "^F",
            "fullscreen",
            "toggle_fullscreen",
            when=_no_modal,
            views=_BOTH,
            priority=True,
            command="/fullscreen",
        ),
    ),
    (
        Key("slash", "/", "slash", "slash", when=_no_modal),
        Key(
            "question_mark",
            "?",
            "help",
            "overlay('help')",
            when=_no_modal,
            views=_BOTH,
            priority=True,
            command="/help",
        ),
    ),
)
"""The base view's bar, in the mockup's order (the mockup's `KEYS3`), parted by
`│`."""

OFF_BAR: Final[tuple[Key, ...]] = (
    Key("escape", "Esc", "interrupt", "interrupt", when=_no_modal, priority=True),
    Key("ctrl+o", "^O", "tool lines", "toggle_tools", when=_no_modal, priority=True),
    Key("ctrl+up", "^↑", "zoom in", "zoom_in", when=_no_modal, priority=True),
    Key("ctrl+down", "^↓", "zoom out", "zoom_out", when=_no_modal, priority=True),
    Key("ctrl+s", "^S", "answer the stop", "open_stop", when=_stop_waiting, priority=True),
    Key(
        "ctrl+g",
        "^G",
        "pause the team",
        "pause_team",
        when=_no_modal,
        priority=True,
        command="/pause",
    ),
)
"""Bound, and not drawn: `Esc` interrupts the turn in flight, `Ctrl+O` collapses and expands
Zeus's tool lines, `Ctrl+Up` zooms the focused pane to the whole screen and `Ctrl+Down`
back (zoom, on the owner's keys: most terminals never pass Ctrl+Z's alternatives on, and
Ctrl+Z is the composer's undo); `Ctrl+S` opens again the stop `Esc` put back as a banner, which
names it, and `Ctrl+G` pauses the team as `/pause` does (neither is the composer's,
nor Textual's text area's); the base screen's bar shows none of them. In this table so one place
still holds every key the screen binds. Under a modal none applies, so a confirm
modal's own `Esc` (deny) is reached: the app's priority binding steps aside when its
`check_action` says no; on the diff screen `Esc` goes back instead."""

DIFF_BAR: Final[tuple[tuple[Key, ...], ...]] = (
    (
        Key("n", "n", "next file", "next_file", views=_DIFF),
        Key("left_square_bracket", "[ ]", "hunk", "hunk(-1)", views=_DIFF, hunks=True),
        Key("t", "t", "this task / all", "toggle_scope", views=_DIFF),
        Key("u", "u", "undo hunk", builds_in="a later release", views=_DIFF),
    ),
    (
        Key("escape", "Esc", "back", "close_diff", views=_DIFF),
        Key("ctrl+p", "^P", "commands", "palette", when=_no_modal, views=_DIFF),
        Key("question_mark", "?", "help", "overlay('help')", when=_no_modal, views=_DIFF),
    ),
)
"""The diff screen's bar, in the mockup's order. `[ ]` is one chip for two keys;
`]` is `DIFF_OFF_BAR`'s."""

DIFF_OFF_BAR: Final[tuple[Key, ...]] = (
    Key("right_square_bracket", "]", "next hunk", "hunk(1)", views=_DIFF, hunks=True),
)
"""Bound on the diff screen, drawn as half of the `[ ]` chip."""


def _tasks(state: TuiState) -> bool:
    board = state.athena.board
    return _no_modal(state) and board is not None and bool(board.tasks)


def _commands(state: TuiState) -> bool:
    task = state.worker.task
    return (
        _no_modal(state)
        and task is not None
        and any(e.kind is WorkerKind.EXEC for e in task.entries)
    )


ATHENA_KEYS: Final[tuple[Key, ...]] = (
    Key("up", "↑", "choose", "choose(-1)", when=_tasks),
    Key("down", "↓", "choose", "choose(1)", when=_tasks),
    Key("enter", "Enter", "details", "details", when=_tasks),
)
"""Athena's pane's own keys, while it has the focus ("`Enter` on a task opens its brief,
the worker's last report and Athena's review verdict"): bound on the pane, drawn in its hint."""

WORKER_KEYS: Final[tuple[Key, ...]] = (
    Key("up", "↑", "choose", "choose(-1)", when=_commands),
    Key("down", "↓", "choose", "choose(1)", when=_commands),
    Key("enter", "Enter", "command", "details", when=_commands),
)
"""The worker's pane's own keys ("a command's output is a first line and its exit code;
`Enter` expands it")."""


def _worked(state: TuiState) -> bool:
    """A task has been delegated: the CODE pane has tasks and files to choose."""
    return _no_modal(state) and bool(state.worker.tasks)


def _finding(state: TuiState) -> bool:
    return _no_modal(state)


CODE_KEYS: Final[tuple[Key, ...]] = (
    Key("tab", "Tab", "task", "next_task(1)", when=_worked),
    Key("left", "←", "file", "next_file(-1)", when=_worked),
    Key("right", "→", "file", "next_file(1)", when=_worked),
    Key("slash", "/", "find", "find", when=_finding),
)
"""The CODE pane's own keys, while it has the focus: `Tab` the next task,
`←`/`→` the open task's files, `/` its search. `/` is the pane's here, and the composer's
slash everywhere else (the owner agreed, 2026-10-26)."""

CODE_OFF_KEYS: Final[tuple[Key, ...]] = (
    Key("shift+tab", "⇧Tab", "task back", "next_task(-1)", when=_worked),
    Key("escape", "Esc", "back", "close", when=_no_modal),
)
"""Bound on the CODE pane, not drawn in its hint: the task before, and `Esc`, which closes its
search or a read-only file. While one is open the app's `Esc` (interrupt) steps aside
(`DiffFrame.takes_escape`); otherwise it interrupts, as everywhere."""


def _landing(state: TuiState) -> bool:
    """The front page shows: nothing said, no run opened, and no modal."""
    return _no_modal(state) and state.landing


FRONT_KEYS: Final[tuple[Key, ...]] = (
    Key("up", "↑", "choose", "front(-1)", when=_landing),
    Key("down", "↓", "choose", "front(1)", when=_landing),
    Key("enter", "Enter", "open", "front_open", when=_landing),
)
"""The front page's keys on the CODE pane, where its sessions are drawn: a click on
the pane, or `F4`, gives it the focus, and Up, Down and Enter still choose and open, as they do
in the empty composer. Not drawn in the pane's hint: the front page draws its own."""


def _items(state: TuiState) -> bool:
    return _no_modal(state) and bool(state.background.items)


BACKGROUND_KEYS: Final[tuple[Key, ...]] = (
    Key("up", "↑", "choose", "choose(-1)", when=_items),
    Key("down", "↓", "choose", "choose(1)", when=_items),
)
"""The BACKGROUND pane's keys: choose a row. No `Enter`: there are no details (the
owner's call, 2026-10-26), so `Esc` keeps its one meaning."""


def lit(
    key: Key, state: TuiState, *, view: ScreenView = ScreenView.BASE, hunks: bool = False
) -> bool:
    """Drawn lit: its action is built and it applies now, in `view`, where hunks are shown or
    not (`hunks`)."""
    return (
        key.action is not None
        and view in key.views
        and key.when(state)
        and (hunks or not key.hunks)
    )


def bar_for(view: ScreenView) -> tuple[tuple[Key, ...], ...]:
    return DIFF_BAR if view is ScreenView.DIFF else BAR


def bound() -> tuple[Key, ...]:
    """The keys the app binds: every built key of the base bar and off it, whatever the state.
    One that does not apply now is dimmed, and the app's `check_action` makes pressing it do
    nothing."""
    return tuple(k for k in (*(k for group in BAR for k in group), *OFF_BAR) if k.action)


def diff_bound() -> tuple[Key, ...]:
    """The keys the diff screen binds: every built key of its bar and off it, except one the
    app binds already to the same action (`^D`). `Esc` is bound by both, to different actions:
    the app's (interrupt) does not apply in this view, so the screen's (back) is reached."""
    keys = (*(k for group in DIFF_BAR for k in group), *DIFF_OFF_BAR)
    app = {(k.key, k.action) for k in bound()}
    return tuple(k for k in keys if k.action and (k.key, k.action) not in app)


def command_keys() -> dict[str, str]:
    """Each slash command a key of the base screen also does, to the key as the bar prints it
    (`/diff`: `^D`, `/pause`: `^G`): the palette's key column."""
    keys = (*(k for group in BAR for k in group), *OFF_BAR)
    return {k.command: k.shown for k in keys if k.command is not None}
