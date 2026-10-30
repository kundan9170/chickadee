"""RunMetrics derived only from the journal, through its projection.

The task loop's counts: tasks, rounds and why each ended, Athena's reviews, plans and
calls. No solve rate: no gate ran, so an accepted task is Athena's judgment counted, never a
verification (invariant 7).
"""

from __future__ import annotations

from typing import Any

from helpers import payload_for, scope_args

from chickadee.journal import Journal, derive_metrics, paths, read_events
from chickadee.journal.projection import project
from chickadee.schema import EventKind
from chickadee.schema.journal import BudgetP
from chickadee.state.roots import Roots

SHA = "a" * 64
IN_ROUND_1 = {"task_id": "T1", "round": 1}
IN_ROUND_2 = {"task_id": "T1", "round": 2}


def _mixed_run(repo: Roots, run_id: str) -> list[Any]:
    script: list[tuple[EventKind, dict[str, Any], dict[str, Any]]] = [
        (EventKind.RUN_START, {"isolation": "worktree"}, {}),
        (EventKind.PLAN_WRITTEN, {"by": "athena", "version": 1, "asked": True}, {}),
        (EventKind.TASK_DELEGATED, {"task_id": "T1", "by": "athena"}, {}),
        (
            EventKind.MODEL_CALL,
            {"role": "worker", "tokens_in": 900, "tokens_out": 100, "cost_usd": 0.05},
            IN_ROUND_1,
        ),
        (EventKind.TOOL_CALL, {"role": "worker", "tool": "read"}, IN_ROUND_1),
        (
            EventKind.TOOL_CALL,
            {"role": "worker", "tool": "shell", "denied_reason": "outside the worktree"},
            IN_ROUND_1,
        ),
        (EventKind.TASK_REPORTED, {"task_id": "T1", "round": 1, "stopped": "replied"}, {}),
        (EventKind.TASK_REVIEWED, {"task_id": "T1", "answer": "feedback"}, {}),
        (
            EventKind.MODEL_CALL,
            {"role": "worker", "tokens_in": 950, "tokens_out": 120, "cost_usd": 0.06},
            IN_ROUND_2,
        ),
        (EventKind.TASK_REPORTED, {"task_id": "T1", "round": 2, "stopped": "limit"}, {}),
        (EventKind.TASK_REVIEWED, {"task_id": "T1", "answer": "accept"}, {}),
        (
            EventKind.RUN_END,
            {
                "outcome": "COMPLETED",
                "tasks": 1,
                "committed": 1,
                "wall_ms": 42_000,
                "cost_usd": 0.11,
            },
            {},
        ),
    ]
    with Journal.create(repo, run_id, {}) as j:
        for kind, over, envelope in script:
            j.append(kind, payload_for(kind, **over), **(envelope or scope_args(kind)))
            if kind is EventKind.TASK_REVIEWED and over["answer"] == "accept":
                # `payload_for` takes the field name `kind`, which `BudgetP` has too.
                crossing = BudgetP(
                    scope="run", kind="warning", spent_usd=0.11, cap_usd=0.1, terminal=False
                )
                j.append(EventKind.BUDGET, crossing)
    return list(read_events(paths.journal_path(repo, run_id)))


def test_golden_metrics(repo: Roots, run_id: str) -> None:
    m = derive_metrics(project(_mixed_run(repo, run_id)))
    assert m.outcome == "COMPLETED"
    assert m.isolation == "worktree"
    assert m.rounds == 2
    assert m.model_calls == 2
    assert m.tokens_in == 1850
    assert m.tokens_out == 220
    assert round(m.cost_usd, 4) == 0.11
    assert (m.tasks_total, m.tasks_accepted, m.tasks_committed) == (1, 1, 1)
    assert m.plans == 1
    assert (m.tool_calls, m.tool_refusals) == (2, 1)
    assert (m.budget_warnings, m.budget_hard_caps) == (1, 0)
    assert m.wall_ms == 42_000


def test_why_each_round_ended_and_each_review_are_attributed(repo: Roots, run_id: str) -> None:
    """Invariant 7's legibility: a round cut at its limit never reads as one that replied, and
    a review asking for more never reads as an acceptance."""
    m = derive_metrics(project(_mixed_run(repo, run_id)))
    assert m.round_ends == {"replied": 1, "limit": 1}
    assert m.reviews == {"feedback": 1, "accept": 1}


def test_an_accepted_task_is_not_a_committed_one_before_the_run_ends(
    repo: Roots, run_id: str
) -> None:
    """Accepted is Athena's judgment; committed is `run_end`'s count, and nothing is a solve."""
    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(
            EventKind.TASK_DELEGATED,
            payload_for(EventKind.TASK_DELEGATED),
            **scope_args(EventKind.TASK_DELEGATED),
        )
        j.append(
            EventKind.TASK_REVIEWED,
            payload_for(EventKind.TASK_REVIEWED, answer="accept"),
            **scope_args(EventKind.TASK_REVIEWED),
        )
    m = derive_metrics(project(read_events(paths.journal_path(repo, run_id))))
    assert (m.tasks_accepted, m.tasks_committed, m.outcome) == (1, 0, None)
    assert not hasattr(m, "solve_rate")


def test_to_rows_is_flat(repo: Roots, run_id: str) -> None:
    rows = derive_metrics(project(_mixed_run(repo, run_id))).to_rows()
    assert len(rows) == 1
    assert all(not isinstance(v, dict | list) for v in rows[0].values())
    assert rows[0]["cost_per_accepted_task"] == 0.11


def test_a_plan_cancelled_at_stop_2_is_not_counted_as_a_plan(repo: Roots, run_id: str) -> None:
    """T-3.157 review: the desk says a cancelled plan is not a plan (`TaskDesk.plans`), and the
    metrics agree, so a run cancelled at stop 2 reports no plan."""
    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START, isolation="worktree"))
        cancelled = payload_for(
            EventKind.PLAN_WRITTEN, by="athena", version=1, asked=True, outcome="cancelled"
        )
        j.append(EventKind.PLAN_WRITTEN, cancelled, **scope_args(EventKind.PLAN_WRITTEN))
    m = derive_metrics(project(list(read_events(paths.journal_path(repo, run_id)))))
    assert m.plans == 0
