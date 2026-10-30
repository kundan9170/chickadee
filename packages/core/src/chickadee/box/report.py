"""What the box returns: `CoderReport` v2, assembled
from the run's journal after the episode ends, and from nothing else.

It says how the episode ended (`outcome`, and `why` when not `COMPLETED`), each task with its
state, Athena's last review of it, its commit and the facts measured for it (never
read from the worker's words), Athena's own report verbatim, and what changed, counted. It names
no gate, checkpoint or verification: none ran, and a report that claimed one would overstate
what was checked (invariant 7). `COMPLETED` says the episode ended with every task settled,
never that the change is right; what ran and what it showed is in the facts.

The report never carries the diff (invariant 14). It counts what changed, and names the diff,
the journal and every failed command's output by handle, over the run's own blobs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final, TypedDict

from pydantic import BaseModel, ConfigDict, field_serializer, model_validator

from chickadee.box.outcome import Outcome
from chickadee.box.tasks import TaskBook, TaskState
from chickadee.context.handle import Handle, HandleKind, content_sha, handle_id
from chickadee.context.opener import stored_sniff
from chickadee.context.parts import Parts
from chickadee.context.terms import Caps, Vision
from chickadee.journal.projection import RunProjection, RunSpend
from chickadee.schema.journal import ExecP, ReviewAnswer, RoundEnd, TaskFactsP, TaskReportedP

__all__ = [
    "Blob",
    "ChangeSummary",
    "CoderReport",
    "CostWire",
    "ReportHandles",
    "ReportedTask",
    "diff_preview",
    "failing_output",
    "reported_tasks",
    "stored_handle",
    "summarize",
]

WIRE = ConfigDict(extra="forbid", frozen=True)

_PARTS: Final = Parts(Vision(rung="zeus"), None)
"""How stored bytes are shaped for the reader of the report, Zeus. No symbol index: stored
output is read as text (`context.opener.stored_sniff`)."""


Blob = Callable[[str], bytes]
"""`(sha) -> bytes` over the run's blob store."""


class ChangeSummary(BaseModel):
    """Files, lines and symbols: deliberately not the diff, which travels as a handle."""

    model_config = WIRE

    files: tuple[str, ...] = ()
    lines_added: int = 0
    lines_removed: int = 0
    symbols_added: tuple[str, ...] = ()
    symbols_removed: tuple[str, ...] = ()
    symbols_known: bool = True
    """False when a task's symbols were not measured (after a resume, invariant 20): an empty
    list would otherwise claim nothing changed."""


class ReportHandles(BaseModel):
    model_config = WIRE

    journal: Handle
    diff: Handle | None = None
    """`None` when nothing changed: there is no diff to name."""
    failing_output: tuple[Handle, ...] = ()


class ReportedTask(BaseModel):
    """One task as the episode left it: the task book's entry, and its last
    round's end and facts."""

    model_config = WIRE

    id: str
    title: str
    by: str
    state: TaskState
    rounds: int
    answer: ReviewAnswer | None = None
    """Athena's last review of it; None when she never reviewed it."""
    commit: str | None = None
    stopped: RoundEnd | None = None
    """Why its last round ended; None before its first round reported."""
    facts: TaskFactsP | None = None
    """Its last round's facts; None before its first round reported."""


class CostWire(TypedDict):
    """`RunSpend` as the report writes it: the same fields, its read-only maps as objects."""

    tokens_in: int
    tokens_out: int
    cost_usd: float
    by_rung: dict[str, float]
    by_task: dict[str, tuple[int, float]]
    floor_usd: float


class CoderReport(BaseModel):
    model_config = WIRE

    outcome: Outcome
    why: str | None = None
    """One line: why the episode ended as it did; None for `COMPLETED`."""
    growth: str | None = None
    """How a one-task plan grew past its task (`box/tasks.growth`): "1 task
    approved, 3 delegated (T2, T3 from a re-cut of T1)"; None when it did not, or a design."""
    run_id: str | None = None
    """`None` only for an episode that ended before its run began: no journal was written."""
    tasks: tuple[ReportedTask, ...] = ()
    athena: str | None = None
    """Athena's final reply, verbatim; None when the episode ended without one."""
    change_summary: ChangeSummary = ChangeSummary()
    handles: ReportHandles | None = None
    """`None` exactly when `run_id` is."""
    cost: RunSpend
    """Zero for an episode that ended before its run: nothing was spent."""
    harness: tuple[str, ...] = ()
    """What the run's hooks said as it ended (`episode_ended`), each note
    as the hook wrote it; empty where none was bound to say anything."""
    unanswered: tuple[str, ...] = ()
    """Zeus's tells delivered in the run that the team never replied to, by id:
    the run ended with them owed, and says so."""

    @field_serializer("cost")
    def _cost(self, cost: RunSpend) -> CostWire:
        # `RunSpend` keeps its maps read-only (`MappingProxyType`), which pydantic cannot
        # write; the report is what crosses to Zeus, so it writes them as plain objects.
        return {
            "tokens_in": cost.tokens_in,
            "tokens_out": cost.tokens_out,
            "cost_usd": cost.cost_usd,
            "by_rung": dict(cost.by_rung),
            "by_task": dict(cost.by_task),
            "floor_usd": cost.floor_usd,
        }

    @model_validator(mode="after")
    def _consistent(self) -> CoderReport:
        if (self.outcome == "COMPLETED") == (self.why is not None):
            raise ValueError("why is set exactly when the outcome is not COMPLETED")
        if (self.run_id is None) != (self.handles is None):
            raise ValueError("a report names handles exactly when a run was journaled")
        return self

    @staticmethod
    def before_run(outcome: Outcome, why: str) -> CoderReport:
        """The report of an episode that ended before its run began: no journal, no cost."""
        return CoderReport(outcome=outcome, why=why, cost=RunSpend())


# -- reading the run ----------------------------------------------------------------------


def reported_tasks(run: RunProjection) -> tuple[ReportedTask, ...]:
    """Every task of the run, in the order delegated, as the task book and its last report
    leave it."""
    last: dict[str, TaskReportedP] = {}
    for task, lines in run.tasks.items():
        for line in lines:
            if isinstance(line.payload, TaskReportedP):
                last[task] = line.payload
    out: list[ReportedTask] = []
    for entry in TaskBook(run).entries():
        report = last.get(entry.id)
        out.append(
            ReportedTask(
                id=entry.id,
                title=entry.title,
                by=entry.by,
                state=entry.state,
                rounds=entry.rounds,
                answer=entry.answer,
                commit=entry.commit,
                stopped=report.stopped if report is not None else None,
                facts=report.facts if report is not None else None,
            )
        )
    return tuple(out)


def failing_output(run: RunProjection, blob: Blob) -> tuple[Handle, ...]:
    """Every failed or timed-out command's output, as stored handles."""
    shas: list[str] = []
    for line in run.lines(ExecP):
        e = line.payload
        if e.exit_code != 0 or e.timed_out:
            shas.extend(s for s in (e.stderr_sha, e.stdout_sha) if s)
    return tuple(stored_handle("output", run.run_id, blob(sha)) for sha in dict.fromkeys(shas))


def stored_handle(
    kind: HandleKind, run_id: str, data: bytes, *, preview: str | None = None
) -> Handle:
    """A handle on bytes in the run's blob store, with the id the session would mint for the
    same address (`context.handle.handle_id`), so the engine records it as it arrives. Its
    shape, preview and ops are the ones `Reader.store` gives stored bytes: one rule for both.
    `preview` replaces the outline where the outline would carry what the handle is for
    keeping out of the report: the diff's own lines (invariant 14)."""
    sha = content_sha(data)
    shape, outline, ops = _PARTS.of(stored_sniff(data), data, None, Caps().inline_bytes)
    return Handle(
        id=handle_id(kind, path=None, run_id=run_id, sha=sha),
        kind=kind,
        shape=shape,
        preview=outline if preview is None else preview,
        ops=ops,
        content_sha=sha,
        run_id=run_id,
    )


def diff_preview(numstat: Sequence[tuple[int, int, str]]) -> str:
    """A diff's preview: what changed, counted, never its lines."""
    added = sum(a for a, _, _ in numstat)
    removed = sum(r for _, r, _ in numstat)
    return f"{len(numstat)} file(s) changed, +{added} -{removed}"


def summarize(
    tasks: Sequence[ReportedTask], numstat: Sequence[tuple[int, int, str]]
) -> ChangeSummary:
    """`numstat` is `git diff --numstat` read as (added, removed, path); symbols come from
    each task's facts, which diffed the graph against the table taken at its delegation."""
    facts = [t.facts for t in tasks if t.facts is not None]
    added = [s for f in facts for s in f.symbols_added]
    removed = [s for f in facts for s in f.symbols_removed]
    return ChangeSummary(
        files=tuple(path for _, _, path in numstat),
        lines_added=sum(a for a, _, _ in numstat),
        lines_removed=sum(r for _, r, _ in numstat),
        symbols_added=tuple(dict.fromkeys(added)),
        symbols_removed=tuple(dict.fromkeys(removed)),
        symbols_known=all(f.symbols_known for f in facts),
    )
