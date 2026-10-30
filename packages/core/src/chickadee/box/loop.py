"""The task loop: one
episode on tasks, end to end.

From a design: Athena's request opens on her head (`boxctx/athena.athena_head`) with
the approved design, and everything after is her tool calls over the desk: the plan and
stop 2, each task delegated, reported and reviewed, her final check, her reply. From a one-task
plan: the desk writes Zeus's `plan_written` and `task_delegated` and runs the worker's
first round before Athena is called; her head then holds the brief and that round, and she
begins at `review`.

**How it ends** (`LoopResult`): her reply with no task open is `COMPLETED`, and with one
open `PARTIAL`, naming it; with no plan approved or no task accepted, `PARTIAL`, saying which,
and after her `outgrown` call, `OUTGROWN`; a cancel at stop 2 is
`ABORTED`, the person's; a person's hard stop
is `ABORTED` and the spend cap `BUDGET_CAP`, both resumable; a pause at a task boundary is
`paused`; a request that does not fit is `PARTIAL`; a model that failed or would not stop
calling is `BLOCKED`. `COMPLETED` says the episode ended, never that the change is right: what
ran is in her report and the facts (invariant 7). An exception in the harness is not
caught here: the episode reports it, as it does today.

**Resume** (the episode's rows): her conversation `rebuilt` from the journal; each call
of her last turn the crash left unanswered owed its answer from her steps, or, a desk call,
dispatched again, so the desk continues it from its own line (`box/resume.athena_owed`); a
one-task plan's first round resumed or run; a request already answered finished with no model
call.

It writes the task lines only. The run lines (`run_start`, `run_end`) are the episode's, which
writes them around it (`box/episode.py`); `box/compose.compose_episode` composes it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chickadee.box.athena import Athena, AthenaAnswer
from chickadee.box.commit import Reindexer
from chickadee.box.desk import TaskDesk, render_round
from chickadee.box.outcome import Outcome
from chickadee.box.report import CoderReport
from chickadee.box.resume import athena_owed
from chickadee.box.worker import WorkerStopped
from chickadee.boxctx.athena import athena_head
from chickadee.boxctx.conversation import rebuilt
from chickadee.errors import HarnessBug
from chickadee.journal.port import JournalPort
from chickadee.schema import RetryParams
from chickadee.tools.builtin import BUILTINS

__all__ = [
    "DESK_TOOLS",
    "NO_REPORT",
    "OUTGROWN_WHY",
    "HeadParts",
    "LoopResult",
    "OneTaskPlan",
    "TaskLoop",
]

NO_REPORT: Final = {
    "failed": "the planning model's request failed",
    "unparsed": "the planning model's reply could not be read",
    "kept_calling": "the planning model kept calling tools past its limit",
}
"""Why the planning conversation ended without a report, as the report says it."""

OUTGROWN_WHY: Final = (
    "the one-task plan outgrew its task: Athena's report says what is missing, what is done "
    "and what she proposes"
)
"""The `why` of an episode she ended `OUTGROWN`."""

DESK_TOOLS: Final = frozenset(
    n
    for n in BUILTINS.names()
    if (t := BUILTINS.get(n)) is not None and t.spec.family == "task"
)
"""Athena's calls the desk continues from its own line when a crash cut them off."""


@dataclass(frozen=True)
class OneTaskPlan:
    """Zeus's one-task plan, as approved at stop 2."""

    title: str
    brief: str


@dataclass(frozen=True)
class LoopResult:
    outcome: Outcome
    why: str | None
    athena: str | None
    """Her final reply, verbatim; None when the episode ended without one."""
    paused: bool = False


@dataclass(frozen=True)
class HeadParts:
    """What her head is built from: her instructions and the templates in force."""

    fixed: str
    plan_template: str
    brief_template: str


class TaskLoop:
    """One episode on tasks: `run` from the start, `resume` from the journal."""

    def __init__(
        self,
        athena: Athena,
        desk: TaskDesk,
        journal: JournalPort,
        head: HeadParts,
        *,
        root: Path,
        params: RetryParams,
        untracked_max_bytes: int,
        design: str | None = None,
        one_task: OneTaskPlan | None = None,
        indexer: Reindexer | None = None,
    ) -> None:
        if (design is None) == (one_task is None):
            raise ValueError(
                "an episode runs from a design or from a one-task plan, exactly one"
            )
        self._athena = athena
        self._desk = desk
        self._journal = journal
        self._head = head
        self._root = root
        self._params = params
        self._cap = untracked_max_bytes
        self._design = design
        self._one_task = one_task
        self._indexer = indexer

    def run(self) -> LoopResult:
        # The approved design is the run's first commit, before any task;
        # then the plan a run that continues the work inherits, never asked again.
        self._desk.record_design()
        self._desk.seed_plan()
        return self._ended(self._start)

    def resume(self) -> LoopResult:
        self._desk.record_design()
        self._desk.seed_plan()
        return self._ended(self._resumed)

    def record_report(self, report: CoderReport) -> bool:
        """The work folder's records of how the run ended, as its last
        commit; whether any were written."""
        return self._desk.record_report(report)

    # -- the two ways in ----------------------------------------------------------------------

    def _start(self) -> AthenaAnswer:
        h = self._head
        if self._one_task is not None:
            if self._desk.paused_before_start():
                return AthenaAnswer(None, "paused")
            first = self._desk.delegate_one(self._one_task.title, self._one_task.brief)
            head = athena_head(
                h.fixed,
                h.plan_template,
                h.brief_template,
                one_task=(self._one_task.brief, render_round(first)),
            )
        else:
            head = athena_head(
                h.fixed,
                h.plan_template,
                h.brief_template,
                design=self._design,
                standing=self._desk.standing(),
            )
        return self._athena.start(head)

    def _resumed(self) -> AthenaAnswer:
        run = self._journal.projection()
        conversation = rebuilt(run, "athena", None, self._journal.get_blob, params=self._params)
        if not conversation.head:
            # Her head was never journaled: the crash came before her request began (in a
            # one-task plan, during the worker's first round, which `delegate_one` continues).
            return self._start()
        last = conversation.messages[-1] if conversation.messages else None
        if last is not None and last.role == "assistant" and not last.tool_calls:
            # Her request was answered: her reply is journaled after it ends.
            content = last.content
            return AthenaAnswer(content if isinstance(content, str) else None, "answered")
        owed = athena_owed(
            run,
            conversation,
            self._root,
            self._journal,
            self._indexer,
            untracked_max_bytes=self._cap,
            desk_tools=DESK_TOOLS,
            asking=self._desk.asking,
        )
        return self._athena.resume(conversation, owed)

    # -- how it ended -------------------------------------------------------------------------

    def _ended(self, go: Callable[[], AthenaAnswer]) -> LoopResult:
        try:
            answer = go()
        except WorkerStopped as stopped:
            outcome: Outcome = "BUDGET_CAP" if stopped.by == "spend_cap" else "ABORTED"
            return LoopResult(outcome, stopped.why, None)
        desk = self._desk
        if desk.ended is not None:
            return LoopResult("ABORTED", desk.ended.why, answer.report)
        match answer.end:
            case "paused":
                return LoopResult(
                    "PARTIAL", "paused by the user at a task boundary", None, paused=True
                )
            case "cancelled":
                return LoopResult("ABORTED", "stopped by the user", answer.report)
            case "spend_cap":
                return LoopResult("BUDGET_CAP", "the spend cap stopped the work", None)
            case "context_full":
                return LoopResult(
                    "PARTIAL",
                    "the planning conversation no longer fit the model's context",
                    answer.report,
                )
            case "failed" | "unparsed" | "kept_calling":
                # Said in the report's words: Zeus reads the why, not our names.
                return LoopResult("BLOCKED", NO_REPORT[answer.end], None)
            case "halted":
                # A hook ended her request before a model call.
                return LoopResult("PARTIAL", "a hook halted the planning request", None)
            case "held":
                # Her request reads a tell and goes on (`Athena`): never her end.
                raise HarnessBug("Athena's request ended held for a message")
            case "answered":
                book = desk.book()
                held = book.open()
                if held is not None:
                    why = f"{held.id} {held.title!r} is open, uncommitted ({held.state})"
                    return LoopResult("PARTIAL", why, answer.report)
                if desk.outgrown:
                    return LoopResult("OUTGROWN", OUTGROWN_WHY, answer.report)
                # Decision 20: an episode that changed nothing never reads as one that ended
                # well, whatever her reply says.
                if not desk.plans():
                    return LoopResult("PARTIAL", "no plan was approved", answer.report)
                accepted = any(e.state in ("accepted", "committed") for e in book.entries())
                if not accepted and not desk.carried_committed():
                    return LoopResult("PARTIAL", "no task was accepted", answer.report)
                return LoopResult("COMPLETED", None, answer.report)
