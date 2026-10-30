"""How the session's repair words a run a crash left open, from its journal.

Facts only: the run's terminal when it reached one, else how many effects it started and
finished. Zeus's loop records this as the interrupted `coder` call's result, so the next turn
sees what the run did rather than guessing. Moved out of the facade unchanged, which
binds it to the session's run records.

The same words tell Zeus how a run ended whose `coder` call was answered only that it paused:
resumed later, it ends with no call waiting on it. `tell_ended` says so as his next turn
starts (`ZeusLoop._repair`), as a notice, where it never falls between a call and its result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chickadee.context import standing
from chickadee.context.project import RunRecords, paused_runs, run_ended
from chickadee.context.session import Turn
from chickadee.context.standing import NoticeTrigger
from chickadee.journal.errors import JournalError
from chickadee.schema.journal import EffectCompletedP, EffectIntendedP

if TYPE_CHECKING:
    from chickadee.context.session import Session, SessionState

__all__ = ["run_outcome", "tell_ended"]


def tell_ended(session: Session, records: RunRecords) -> tuple[int, ...]:
    """Before every turn: one trigger-4 notice per run answered `PAUSED` that has ended since
    and was not told yet. Returns the notice lines' seqs."""
    told: list[int] = []
    with session.lock:
        for run_id in _paused_untold(session.state):
            if run_ended(records, run_id):
                outcome = run_outcome(records, run_id)
                told.append(
                    standing.notice(session, standing.paused_run_ended(run_id, outcome))
                )
    return tuple(told)


def run_outcome(records: RunRecords, run_id: str) -> str:
    """What `run_id` did, as its journal says, in the words the repair records."""
    try:
        run = records.projection(run_id)
    except FileNotFoundError:
        return f"run {run_id} left no journal: it did no work"
    except (JournalError, OSError, ValueError) as exc:
        # A damaged or unreadable journal must not fail every later turn's repair: the call
        # is answered with what is known, which is that nothing can be said.
        return f"run {run_id}'s journal could not be read ({exc}); what it did is unknown"
    end = run.end
    if end is not None:
        done = f"{end.committed} of {end.tasks} task(s) committed"
        return f"run {run_id} ended {end.outcome}: {done}"
    intended = len(run.lines(EffectIntendedP))
    completed = len(run.lines(EffectCompletedP))
    return (
        f"run {run_id} reached no terminal: {intended} effect(s) intended, {completed} "
        "completed; resume the session to reconcile the rest"
    )


def told_ended(state: SessionState, run_id: str) -> bool:
    """Whether a trigger-4 notice told `run_id`'s end: its `handle` names the run. The one
    check of it, for `tell_ended` and for a background run's watcher, so a run's
    end is told once whichever writes first."""
    return run_id in _told(state)


def _told(state: SessionState) -> set[str]:
    return {
        record.notice.handle
        for _event, record in state.lines()
        if isinstance(record, Turn)
        and record.notice is not None
        and record.notice.trigger is NoticeTrigger.RUN_ENDED
        and record.notice.handle is not None
    }


def _paused_untold(state: SessionState) -> list[str]:
    """The paused runs (`paused_runs`) whose end no trigger-4 notice has told."""
    told = _told(state)
    return [run_id for run_id in paused_runs(state) if run_id not in told]
