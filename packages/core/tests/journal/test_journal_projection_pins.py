"""Pins of what the run journal's other readers say about it, over
task-loop journals.

The pins of T-3.61 read journals recorded from the packet loop (`journal_pins.RECORDED`). A v2
reader refuses those journals, so each pin here reads a task-loop journal this
file writes, and states its value by hand from the lines it wrote, never from what a reader
returned. What a reader says about a recorded packet-loop journal is pinned too, last.

- **Metrics:** `derive_metrics`, every field.
- **Family spend:** `family._run_spend` over the whole journal, and over it cut mid-line.
- **The run header:** `run_header._episode`.
- **Reconciliation:** `zeus.reconcile.run_outcome`.

The packet record (`build_packet_record`) parked with the packet loop and is not pinned here.
The context readers (`project_unit`, `_run_effects`, recall, `fork_refusal`) need a session, so
their pins are `tests/context/test_context_projection_pins.py`.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from helpers import payload_for, scope_args
from journal_pins import RECORDED, install, scratch_roots

from chickadee.budget.family import _run_spend  # pyright: ignore[reportPrivateUsage]
from chickadee.context.project import JournalRecords
from chickadee.journal import Journal, paths, read_events
from chickadee.journal.metrics import RunMetrics, derive_metrics
from chickadee.journal.projection import RunSpend, project
from chickadee.journal.runs import RunProjections
from chickadee.run_header import Episode, _episode  # pyright: ignore[reportPrivateUsage]
from chickadee.schema import EventKind
from chickadee.schema.journal import BudgetP
from chickadee.state.roots import Roots
from chickadee.zeus.reconcile import run_outcome

Script = list[tuple[EventKind, Any, dict[str, Any]]]

ROUND_1 = {"task_id": "T1", "round": 1}

EXEC = payload_for(EventKind.EXEC, operation_id="step-1", exit_code=0, wall_ms=5)
SUSPECTED = payload_for(EventKind.SECRET_SUSPECTED, path=".env", line=3, detector="aws")


def _line(kind: EventKind, envelope: dict[str, Any] | None = None, **over: Any) -> Any:
    return (kind, payload_for(kind, **over), scope_args(kind) if envelope is None else envelope)


def _effect(op: str, kind: str, *, through: str, envelope: dict[str, Any]) -> Script:
    lines: Script = [
        _line(EventKind.EFFECT_INTENDED, envelope, operation_id=op, effect_kind=kind),
    ]
    if through in ("started", "completed"):
        lines.append(_line(EventKind.EFFECT_STARTED, envelope, operation_id=op))
    if through == "completed":
        lines.append(
            _line(EventKind.EFFECT_COMPLETED, envelope, operation_id=op, outcome="succeeded")
        )
    return lines


def completed() -> Script:
    """One task, one round, accepted and committed; Zeus's bootstrap first, a command and an
    advisory secret inside the task, and Athena's call outside it. Costs are exact in binary,
    so a sum is the number written."""
    return [
        _line(EventKind.RUN_START, isolation="worktree"),
        _line(EventKind.PLAN_WRITTEN, by="athena", version=1, asked=True),
        *_effect("boot", "bootstrap", through="completed", envelope={}),
        _line(EventKind.TASK_DELEGATED, task_id="T1", by="athena"),
        _line(
            EventKind.MODEL_CALL,
            ROUND_1,
            role="worker",
            rung="low-param",
            tokens_in=100,
            tokens_out=20,
            cost_usd=0.25,
        ),
        (EventKind.EXEC, EXEC, ROUND_1),
        (EventKind.SECRET_SUSPECTED, SUSPECTED, {}),
        _line(EventKind.TASK_REPORTED, task_id="T1", round=1, stopped="replied"),
        _line(
            EventKind.MODEL_CALL, {}, role="athena", tokens_in=50, tokens_out=10, cost_usd=0.5
        ),
        _line(EventKind.TASK_REVIEWED, task_id="T1", answer="accept"),
        (
            EventKind.BUDGET,
            BudgetP(scope="run", kind="warning", spent_usd=0.75, cap_usd=0.5, terminal=False),
            {},
        ),
        _line(
            EventKind.RUN_END,
            outcome="COMPLETED",
            tasks=1,
            committed=1,
            tokens=180,
            cost_usd=0.75,
            wall_ms=1000,
        ),
    ]


def capped() -> Script:
    return [
        _line(EventKind.RUN_START, isolation="in_place"),
        _line(EventKind.TASK_DELEGATED, task_id="T1", by="zeus"),
        _line(
            EventKind.RUN_END, outcome="BUDGET_CAP", why="the hard cap", tasks=2, committed=1
        ),
    ]


def open_step() -> Script:
    """A crash inside a worker's step: intended, never started, and the run never ended."""
    return [
        _line(EventKind.RUN_START),
        *_effect("boot", "bootstrap", through="started", envelope={}),
        _line(EventKind.TASK_DELEGATED, task_id="T1", by="zeus"),
        *_effect("step-1", "worker_step", through="intended", envelope=ROUND_1),
    ]


def write(repo: Roots, run_id: str, script: Script) -> list[bytes]:
    with Journal.create(repo, run_id, {}) as j:
        for kind, payload, envelope in script:
            j.append(kind, payload, **envelope)
    return paths.journal_path(repo, run_id).read_bytes().splitlines(keepends=True)


def projected(repo: Roots, run_id: str, script: Script) -> Any:
    write(repo, run_id, script)
    return project(read_events(paths.journal_path(repo, run_id)))


# -- metrics ---------------------------------------------------------------------------------


def test_derived_metrics_are_pinned_field_by_field(repo: Roots, run_id: str) -> None:
    assert derive_metrics(projected(repo, run_id, completed())) == RunMetrics(
        run_id=run_id,
        outcome="COMPLETED",
        isolation="worktree",
        tasks_total=1,
        tasks_accepted=1,
        tasks_committed=1,
        rounds=1,
        plans=1,
        model_calls=2,
        tool_calls=0,
        tool_refusals=0,
        tokens_in=150,
        tokens_out=30,
        cost_usd=0.75,
        wall_ms=1000,
        round_ends={"replied": 1},
        reviews={"accept": 1},
        operations=1,
        operations_ambiguous=0,
        operations_incomplete=0,
        budget_warnings=1,
        budget_hard_caps=0,
    )


# -- family spend ----------------------------------------------------------------------------


def _as_pinned(spend: RunSpend) -> dict[str, Any]:
    """A run's spend in the shape T-3.61's pin recorded: the same five facts, under the names
    they had. `by_rung` is keyed by `model_call.rung`, or the role for a call naming none."""
    return {
        "tokens_in": spend.tokens_in,
        "tokens_out": spend.tokens_out,
        "cost": spend.cost_usd,
        "floor": spend.floor_usd,
        "by_rung": dict(spend.by_rung),
    }


def _spend_cut(repo: Roots, run_id: str, raw: list[bytes], last: int) -> dict[str, Any]:
    """The spend with the journal cut halfway through line `last` (1-based), lines after it
    gone: a torn tail, which the family counts up to."""
    cut = b"".join(raw[: last - 1]) + raw[last - 1][: len(raw[last - 1]) // 2]
    where = scratch_roots(repo.project / "cut")
    target = paths.run_dir(where, run_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / paths.JOURNAL_NAME).write_bytes(cut)
    return _as_pinned(_run_spend(RunProjections(where), run_id))


def test_a_runs_family_spend_whole_and_cut_mid_line_is_pinned(repo: Roots, run_id: str) -> None:
    raw = write(repo, run_id, completed())
    whole = _as_pinned(_run_spend(RunProjections(repo), run_id))
    assert whole == {
        "tokens_in": 150,
        "tokens_out": 30,
        "cost": 0.75,
        "floor": 0.75,
        "by_rung": {"low-param": 0.25, "athena": 0.5},
    }
    # Cut inside `run_end`: the spend was all written before it.
    assert _spend_cut(repo, run_id, raw, len(raw)) == whole
    # Cut inside Athena's call: only the worker's call, and no floor yet, is counted.
    athena = next(i for i, line in enumerate(raw, 1) if b'"role":"athena"' in line)
    assert _spend_cut(repo, run_id, raw, athena) == {
        "tokens_in": 100,
        "tokens_out": 20,
        "cost": 0.25,
        "floor": 0.0,
        "by_rung": {"low-param": 0.25},
    }


def test_a_task_s_spend_is_its_calls(repo: Roots, run_id: str) -> None:
    """Per task, by the envelope's `task_id`: Athena's call outside the task is the run's."""
    spend = projected(repo, run_id, completed()).spend
    assert dict(spend.by_task) == {"T1": (120, 0.25)}


# -- the run header --------------------------------------------------------------------------


def test_the_run_headers_episode_is_pinned(repo: Roots, run_id: str) -> None:
    assert _episode(run_id, projected(repo, run_id, completed())) == Episode(
        run_id, True, "succeeded", (EXEC,), (SUSPECTED,)
    )


def test_an_episode_with_a_bootstrap_in_flight_is_pending(repo: Roots, run_id: str) -> None:
    assert _episode(run_id, projected(repo, run_id, open_step())) == Episode(
        run_id, True, "pending", (), ()
    )


def test_a_run_that_only_started_did_nothing(repo: Roots, run_id: str) -> None:
    assert _episode(run_id, projected(repo, run_id, completed()[:1])) == Episode(
        run_id, False, "none", (), ()
    )


# -- reconciliation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("script", "said"),
    [
        (completed, "ended COMPLETED: 1 of 1 task(s) committed"),
        (capped, "ended BUDGET_CAP: 1 of 2 task(s) committed"),
        (
            open_step,
            "reached no terminal: 2 effect(s) intended, 0 completed; resume the session to "
            "reconcile the rest",
        ),
    ],
    ids=["completed", "capped", "open_step"],
)
def test_reconciliations_run_outcome_is_pinned(
    repo: Roots, run_id: str, script: Any, said: str
) -> None:
    write(repo, run_id, script())
    assert run_outcome(JournalRecords(repo, RunProjections(repo)), run_id) == (
        f"run {run_id} {said}"
    )


# -- a packet-loop journal -------------------------------------------------------------------


@pytest.mark.parametrize("name", list(RECORDED))
def test_a_recorded_packet_loop_journal_is_refused_not_misread(
    tmp_path: pathlib.Path, name: str
) -> None:
    """A journal of the packet loop is not read. Reconciliation says it could
    not be read and why; the family's spend reads nothing from it rather than a guess."""
    where, rid = scratch_roots(tmp_path), "2026-09-30-0a1b2c3d"
    install(name, where, rid)
    said = run_outcome(JournalRecords(where, RunProjections(where)), rid)
    assert said.startswith(f"run {rid}'s journal could not be read (")
    assert "D196" in said and said.endswith("what it did is unknown")
    assert _run_spend(RunProjections(where), rid) == RunSpend()
