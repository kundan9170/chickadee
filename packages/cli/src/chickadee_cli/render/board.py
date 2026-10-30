"""The task board: a run's tasks, each with its state, rounds,
cost and how its last round ended, from the run's own journal (`/run/{id}/journal`).

**One fold of a run's records for display**, `fold_run`, which `/diff` (`render/diff.py`) reads
too, and `apply_record`, the same fold one record at a time, which the status line and the
TUI's feed keep a `RunView` with.
Every record is read as its kind's type (`chickadee_wire.records.payload`); one this client
cannot read is skipped, as the stream says it is unreadable where it renders it.

Tasks are listed in the order they were delegated. A task's state is Athena's last review of
it, and `committed` once its commit landed, or `kept` once a task accepted in place was
recorded as known good, with nothing committed: what the run kept of it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import ValidationError

from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import (
    EffectCompletedP,
    EffectIntendedP,
    ModelCallP,
    RunEndP,
    RunStartP,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
    WireModel,
)
from chickadee_wire.records import payload

__all__ = [
    "COMMITTED",
    "KEPT",
    "STATES",
    "BoardTask",
    "RunView",
    "apply_record",
    "board_lines",
    "fold_run",
]

COMMIT = "task_commit"
KNOWN_GOOD = "task_known_good"
"""The desk's effect for a task accepted in place: its tree recorded, nothing
committed."""

RUNNING = "running"
COMMITTED = "committed"
KEPT = "kept"
"""Accepted in place and recorded as known good: kept in the checkout, never
`committed`, which would claim a commit there is not (invariant 7)."""

_REVIEWED = {
    "accept": "accepted",
    "feedback": RUNNING,
    # A round held for a message from Zeus goes on: running again, as feedback.
    "continue": RUNNING,
    "take_over": "taken over",
    "re_cut": "re-cut",
    "drop": "dropped",
    # `drop`'s name in older journals: a run written then reads as a drop.
    "abandon": "dropped",
}
"""A task's state after each of Athena's answers."""

STATES = frozenset({RUNNING, *_REVIEWED.values(), COMMITTED, KEPT})
"""Every state a task can show (`BoardTask.state`): delegated and running, each of Athena's
answers, and committed or kept. One home, so a surface that draws a state (the TUI's theme)
covers each one rather than keeping its own list."""


@dataclass
class BoardTask:
    """One task as the board shows it."""

    task_id: str
    title: str = ""
    state: str = RUNNING
    rounds: int = 0
    cost_usd: float = 0.0
    stopped: str | None = None
    """How its last round ended, when not with a reply."""
    changed: tuple[str, ...] = ()
    """The files its last round changed, by its facts."""
    committed: tuple[str, ...] = ()
    """The files its commit wrote: what the run kept of it."""
    plan_task: str | None = None
    """The approved plan's task it runs, when delegated by that id; None
    for a brief Athena wrote, and a one-task plan."""


@dataclass
class RunView:
    run_id: str
    goal: str = ""
    tasks: dict[str, BoardTask] = field(default_factory=dict[str, BoardTask])
    end: RunEndP | None = None
    intended: dict[str, str] = field(default_factory=dict[str, str])
    """Each effect's kind by its operation, as intended: what a completion is read as."""


def fold_run(run_id: str, frames: Iterable[RecordFrame]) -> RunView:
    """A run's records, in seq order, folded into what the board and `/diff` show."""
    view = RunView(run_id)
    for frame in frames:
        apply_record(view, frame)
    return view


def apply_record(view: RunView, frame: RecordFrame) -> bool:
    """One more of the run's records, in seq order: the fold one record at a time, for a view
    kept as the run goes.
    Whether the view changed: False for a record this client cannot read, or a kind that
    changes nothing the board shows. The caller drops any record it has seen."""
    try:
        typed = payload(frame)
    except ValidationError:
        return False
    return _apply(view, _task(view, frame), typed, view.intended)


def _task(view: RunView, frame: RecordFrame) -> BoardTask | None:
    if frame.task_id is None:
        return None
    return view.tasks.setdefault(frame.task_id, BoardTask(frame.task_id))


def _apply(
    view: RunView,
    task: BoardTask | None,
    typed: WireModel | None,
    intended: dict[str, str],
) -> bool:
    """Fold one typed record into `view`; whether the view changed."""
    match typed:
        case RunStartP():
            view.goal = typed.goal
        case RunEndP():
            view.end = typed
        case EffectIntendedP():
            intended[typed.operation_id] = typed.effect_kind
            return False
        case _ if task is None:
            return False
        case TaskDelegatedP():
            task.title, task.plan_task = typed.title, typed.plan_task
        case TaskReportedP():
            task.rounds = max(task.rounds, typed.round)
            task.stopped = None if typed.stopped == "replied" else typed.stopped
            task.changed = tuple(f.path for f in typed.facts.files)
        case TaskReviewedP():
            task.state = _REVIEWED.get(typed.answer, typed.answer)
        case ModelCallP():
            task.cost_usd += typed.cost_usd
        case EffectCompletedP() if (
            typed.outcome == "succeeded" and intended.get(typed.operation_id) == COMMIT
        ):
            task.state, task.committed = COMMITTED, tuple(typed.observed_surface)
        case EffectCompletedP() if (
            typed.outcome == "succeeded" and intended.get(typed.operation_id) == KNOWN_GOOD
        ):
            task.state = KEPT
        case _:
            return False
    return True


_TONE = {"committed": Tone.GOOD, "kept": Tone.GOOD, "dropped": Tone.BAD, "re-cut": Tone.BAD}


def board_lines(view: RunView, style: Style) -> list[str]:
    """The board: the run and its goal, one line per task, and how the run ended."""
    lines = style.lines(f"run {view.run_id}: {view.goal or '(no run_start yet)'}")
    if not view.tasks:
        lines += style.lines("no tasks yet", Tone.DIM, indent="  ")
    for t in view.tasks.values():
        row = f"{t.task_id:<4} {t.state:<11} {t.rounds} round(s)  ${t.cost_usd:.4f}  {t.title}"
        lines += style.lines(row, _TONE.get(t.state, Tone.PLAIN), indent="  ")
        if t.stopped:
            lines += style.lines(f"last round: {t.stopped}", Tone.DIM, indent="       ")
    if view.end is None:
        lines += style.lines("in flight: no run_end yet", Tone.DIM, indent="  ")
    else:
        end = view.end
        lines += style.lines(
            f"ended {end.outcome} · {end.committed}/{end.tasks} task(s) committed · "
            f"${end.cost_usd:.4f}",
            Tone.GOOD if end.outcome == "COMPLETED" else Tone.BAD,
            indent="  ",
        )
    return lines
