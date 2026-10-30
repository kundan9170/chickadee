"""Athena's pane: right, top. The plan as a board (the fold the
REPL's `/status` prints, `render/board.py`) and, under it, her last calls, steps and verdicts
as cards;
read-only.

**Her calls and verdicts only, never raw model text**: her model calls are counted
in her spend, not drawn. A task's state is always a word beside its glyph. A board line
is cut with `…` where the pane is narrower, never wrapped (`ui/` `athenaRun`; invariant 16).
The plan line names its size, from the tasks the approved plan lists (`plan_written.tasks`), and
a listed task not yet delegated is a waiting row under the board's, matched by the
id it was delegated by (`task_delegated.plan_task`); a plan that lists none names its version
alone.

**`Enter` on a task** asks the screen for its details (`overlays/task.py`): with the pane
focused, `↑`/`↓` choose a task, the chosen one marked `▸`, and the pane's hint names the keys.
"""

from __future__ import annotations

import time
from typing import ClassVar, Final

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from chickadee_cli.render.board import STATES
from chickadee_cli.render.stream import preview
from chickadee_cli.tui.feed import SETTLED
from chickadee_cli.tui.keys import ATHENA_KEYS
from chickadee_cli.tui.model import (
    AthenaEntry,
    AthenaKind,
    PlanState,
    TuiState,
    WorkerEntry,
    WorkerKind,
)
from chickadee_cli.tui.names import ROLE_WORDS
from chickadee_cli.tui.panes import RoleFrame, live_lines, spend_label
from chickadee_cli.tui.panes.cards import CardText, box_cards, show_cards
from chickadee_cli.tui.panes.worker import worker_line
from chickadee_cli.tui.statusbar import pane_hint
from chickadee_cli.tui.theme import TASK_GLYPH, Glyphs
from chickadee_wire.models import (
    EffectCompletedP,
    ExecP,
    HookP,
    PlanTaskP,
    TaskReviewedP,
    ToolCallP,
)

__all__ = [
    "IDLE_TEXT",
    "TAIL_SHOWN",
    "AthenaFrame",
    "athena_body",
    "athena_cards",
    "athena_line",
]

TAIL_SHOWN: Final = 3
"""Her last cards under the board: the board is what the pane is for, and a card is three
to five rows, so three keep the newest in a half-height pane."""

_PLAN_WORDS: Final = {
    PlanState.APPROVED: "approved",
    PlanState.REVISED: "revised",
    PlanState.CANCELLED: "cancelled at stop 2",
}
_BAD_ANSWERS: Final = frozenset({"drop", "abandon", "re_cut"})
"""Athena's answers drawn as bad: `abandon` is `drop`'s name in older journals."""
_TONE: Final = {
    "committed": "worker", "kept": "worker", "accepted": "worker", "dropped": "bad",
    "re-cut": "bad",
}  # fmt: skip
"""A task row's colour by its board state, beside its glyph and word."""
_IN_HAND: Final = STATES - SETTLED - {"accepted"}
"""The task being worked now, drawn on the mockup's band (`ui/` `run`)."""


IDLE_TEXT: Final = "idle\n\nThe plan shows here."
"""Her pane before a run: as the worker's says it waits, never how a run goes (the owner's
call, 2026-10-26)."""


def athena_body(
    state: TuiState, glyphs: Glyphs, colours: dict[str, str], chosen: str | None = None
) -> Content:
    """Her pane's body; `chosen` is the task marked for `Enter`, while the pane is focused."""
    athena = state.athena
    dim, fg = colours["dim"], colours["fg"]
    lines: list[Content] = []
    one_task = athena.plan_by == "zeus"
    if one_task:
        # A one-task plan: Zeus's, with no design and no plan of hers.
        lines.append(Content.assemble(("ONE TASK", dim), (" · no plan", fg)))
    elif athena.plan_version is None:
        lines.append(Content.styled("PLAN not written yet", dim))
    else:
        word = _PLAN_WORDS[athena.plan_state] if athena.plan_state else "written"
        size = f" {len(athena.plan_tasks)} tasks ·" if athena.plan_tasks else ""
        lines.append(
            Content.assemble(("PLAN", dim), (f"{size} v{athena.plan_version} · {word}", fg))
        )
    for done in athena.carried:
        # Settled by an earlier run of the work: done, and this run's board never
        # names it, so it is drawn from the work's fold, said as such.
        tone = colours[_TONE.get(done.state, "fg")]
        glyph = getattr(glyphs, TASK_GLYPH[done.state])
        lines.append(Content.styled(f"{glyph} {done.task_id} {done.title}", tone))
        lines.append(Content.styled(f"  {done.state} in an earlier run", dim))
    board = athena.board
    for task in board.tasks.values() if board is not None else ():
        glyph = getattr(glyphs, TASK_GLYPH[task.state])
        band = f" on {colours['run']}" if task.state in _IN_HAND else ""
        tone = _TONE.get(task.state, "fg")
        mark = "▸ " if task.task_id == chosen else ""
        style = f"bold reverse {colours[tone]}" if mark else f"{colours[tone]}{band}"
        lines.append(Content.styled(f"{mark}{glyph} {task.task_id} {task.title}", style))
        detail = f"  {task.state} · {task.rounds} rd · ${task.cost_usd:.4f}"
        if task.stopped:
            detail += f" · {task.stopped}"
        lines.append(Content.styled(detail, f"{dim}{band}"))
    lines += [Content.styled(f"{glyphs.waiting} {p.id} {p.title}", dim) for p in waiting(state)]
    if one_task:
        # Everything after the report is her review, then the final check.
        lines.append(Content.styled("reviews after the report, then the final check", dim))
    now = time.monotonic()
    for reply in state.live.open.values():
        if reply.role == "athena":
            lines += live_lines(reply, glyphs, colours, now=now)  # as she writes
    return Content("\n").join(lines)


def athena_cards(state: TuiState, colours: dict[str, str]) -> list[CardText]:
    """Her last calls, steps and verdicts as cards, newest last (`TAIL_SHOWN`)."""
    athena = state.athena
    board = athena.board
    titles = {t.task_id: t.title for t in board.tasks.values()} if board is not None else {}
    cards = box_cards(
        [e for e in athena.tail if e.kind is not AthenaKind.MODEL],
        step_diffs=state.step_diffs,
        colours=colours,
        plan=athena.plan_tasks,
        titles=titles,
    )
    return cards[-TAIL_SHOWN:]


def waiting(state: TuiState) -> list[PlanTaskP]:
    """The plan's tasks no delegation has run yet, in the plan's order: neither this run's
    nor settled by an earlier run of the work."""
    board = state.athena.board
    started = {t.plan_task for t in board.tasks.values()} if board is not None else set()
    done = {t.task_id for t in state.athena.carried}
    return [p for p in state.athena.plan_tasks if p.id not in started | done]


def athena_line(entry: AthenaEntry, glyphs: Glyphs, colours: dict[str, str]) -> Content:
    """Her latest entry as one line (the narrow layout's ticker): a step or a command of hers
    as the worker's line says one."""
    athena, dim, bad = colours["athena"], colours["dim"], colours["bad"]
    who = f"{ROLE_WORDS['athena'].lower()} "
    match entry.record:
        case EffectCompletedP() | ExecP() | HookP():
            return worker_line(
                WorkerEntry(entry.seq, WorkerKind.STEP, entry.record, step=entry.step),
                glyphs,
                colours,
            )
        case ToolCallP() as call if call.denied_reason is not None:
            said = preview(call.result_preview or call.denied_reason, whole=False)
            return Content.assemble((who, athena), (f"✗ {call.tool} refused: {said}", bad))
        case ToolCallP() as call:
            args = preview(call.args_preview, whole=False) if call.args_preview else ""
            return Content.assemble((who, athena), (f"{glyphs.call} {call.tool} {args}", dim))
        case TaskReviewedP() as review:
            tone = bad if review.answer in _BAD_ANSWERS else colours["fg"]
            return Content.styled(f"{review.task_id} → {review.answer}", tone)
        case _:
            return Content.styled(f"unreadable record ({entry.text})", bad)


class AthenaFrame(RoleFrame):
    role = "athena"
    colour = "athena"
    KEYS = ATHENA_KEYS
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(k.key, k.action, show=False) for k in ATHENA_KEYS if k.action is not None
    ]

    DEFAULT_CSS = """
    AthenaFrame { border: round $athena; }
    AthenaFrame.-idle { border: round $faint; }
    AthenaFrame:focus, AthenaFrame.-idle:focus { border: double $athena; }
    AthenaFrame > .body { text-wrap: nowrap; text-overflow: ellipsis; }
    AthenaFrame > .cards { height: auto; }
    AthenaFrame > .after { height: auto; }
    """

    def __init__(self, colours: dict[str, str], glyphs: Glyphs) -> None:
        super().__init__(colours, glyphs)
        self.cards = Vertical(classes="cards")
        self.after = Static("", classes="after")
        """Under her cards: the pane's keys while it has the focus."""

    def compose(self) -> ComposeResult:
        yield from super().compose()
        yield self.cards
        yield self.after

    def show(self, state: TuiState) -> None:
        super().show(state)
        idle = self.idle(state)
        show_cards(self.cards, [] if idle else athena_cards(state, self._colours))
        hint = not idle and self.has_focus
        self.after.update(pane_hint(state, self._colours, self.KEYS) if hint else "")
        self.after.display = hint

    def idle(self, state: TuiState) -> bool:
        return state.athena.run_id is None

    def title(self, state: TuiState) -> Content:
        return Content.styled(ROLE_WORDS["athena"].upper(), f"bold {self._colours['athena']}")

    def label(self, state: TuiState) -> Content | tuple[Content, ...]:
        athena = state.athena
        if athena.plan_version is None:
            return Content.styled("no plan", self._colours["dim"])
        # Her model is the header's (`header.models_line`); the pane says what she spent and
        # how full her latest prompt was.
        return spend_label(
            athena.cost_usd,
            state.context("athena"),
            self._colours["athena"],
            self._glyphs,
            self._colours,
        )

    def idle_text(self, state: TuiState) -> Content:
        return Content.styled(IDLE_TEXT, self._colours["dim"])

    def content(self, state: TuiState) -> Content:
        return athena_body(state, self._glyphs, self._colours, self.chosen())

    def rows(self, state: TuiState) -> list[str]:
        board = state.athena.board
        return list(board.tasks) if board is not None else []

    def default_row(self, rows: list[str]) -> int:
        """The task in hand, else the first."""
        board = self.state.athena.board
        tasks = board.tasks if board is not None else {}
        return next((i for i, t in enumerate(rows) if tasks[t].state in _IN_HAND), 0)
