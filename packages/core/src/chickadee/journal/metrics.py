"""Telemetry derivation.

Computed **only from the journal**, through its one projection. Spend lives in the
`model_call` lines and wall-clock in `run_end`, so there is no side channel to reconcile
against — a number that cannot be derived from the journal is a number this project does
not report.

The task loop's counts: tasks, rounds and why each ended, Athena's reviews by
answer, plans, calls. Nothing here is a solve rate: no gate ran, so a count of tasks accepted is
Athena's judgment counted, never a verification (invariant 7). The eval measures solving against
hidden tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from chickadee.journal.projection import RunProjection
from chickadee.schema.journal import (
    BudgetP,
    EffectCompletedP,
    EffectIntendedP,
    ModelCallP,
    PlanWrittenP,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
    ToolCallP,
)

SCRIPT_TOOL = "python"
"""The Python arm's call: a block, whose own calls are lines of their own."""


def python_note(call: ToolCallP) -> bool:
    """A served `python` line: the Python arm's note that a block ran after a cleanup or a
    repair. The block's calls are lines of their own, so a note is never a
    call: neither the metrics nor a resumed round count it as one. The one reading of it."""
    return call.tool == SCRIPT_TOOL and call.denied_reason is None


@dataclass
class RunMetrics:
    run_id: str = ""
    outcome: str | None = None
    isolation: str = ""

    tasks_total: int = 0
    tasks_accepted: int = 0
    """Accepted by Athena's review: her judgment counted, not a verification."""
    tasks_committed: int = 0
    """From `run_end`; 0 before the run ends."""
    rounds: int = 0
    plans: int = 0

    model_calls: int = 0
    tool_calls: int = 0
    tool_refusals: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0

    round_ends: dict[str, int] = field(default_factory=dict[str, int])
    """Why each worker round ended, read from the loop, never the report."""
    reviews: dict[str, int] = field(default_factory=dict[str, int])
    """Athena's answers to the reports, by answer."""

    operations: int = 0
    operations_ambiguous: int = 0
    operations_incomplete: int = 0

    budget_warnings: int = 0
    budget_hard_caps: int = 0

    @property
    def cost_per_accepted_task(self) -> float:
        return self.cost_usd / self.tasks_accepted if self.tasks_accepted else 0.0

    def to_rows(self) -> list[dict[str, Any]]:
        """Flat dicts for CSV/JSONL. The eval package owns rendering to report.md."""
        row = {k: v for k, v in asdict(self).items() if not isinstance(v, dict)}
        row["cost_per_accepted_task"] = round(self.cost_per_accepted_task, 6)
        return [row]


def derive_metrics(run: RunProjection) -> RunMetrics:
    """The run's metrics, as counts of its typed lines and its spend."""
    m = RunMetrics()
    if run.start is not None:
        m.run_id = run.run_id
        m.isolation = str(run.start.isolation)
    if run.end is not None:
        m.outcome = str(run.end.outcome)
        m.wall_ms = int(run.end.wall_ms)
        m.tasks_committed = int(run.end.committed)
    m.tasks_total = len(run.lines(TaskDelegatedP))
    # A plan cancelled at stop 2 is a line, not a plan (`TaskDesk.plans`): nothing ran by it.
    m.plans = sum(p.payload.outcome == "approved" for p in run.lines(PlanWrittenP))
    for report in run.lines(TaskReportedP):
        m.rounds += 1
        _count(m.round_ends, str(report.payload.stopped))
    for review in run.lines(TaskReviewedP):
        _count(m.reviews, str(review.payload.answer))
    m.tasks_accepted = m.reviews.get("accept", 0)

    m.model_calls = len(run.lines(ModelCallP))
    calls = [line.payload for line in run.lines(ToolCallP) if not python_note(line.payload)]
    m.tool_calls = len(calls)
    m.tool_refusals = sum(c.denied_reason is not None for c in calls)
    m.tokens_in, m.tokens_out = run.spend.tokens_in, run.spend.tokens_out
    m.cost_usd = run.spend.cost_usd

    # Counted from the lines, not `run.operations`: an id intended twice is a defect the fold
    # records once, and the metrics count what the journal says.
    completed = [line.payload for line in run.lines(EffectCompletedP)]
    m.operations = len(run.lines(EffectIntendedP))
    m.operations_incomplete = m.operations - len(completed)
    m.operations_ambiguous = sum(p.outcome == "ambiguous" for p in completed)

    for budget in run.lines(BudgetP):
        if budget.payload.kind == "warning":
            m.budget_warnings += 1
        else:
            m.budget_hard_caps += 1
    return m


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
