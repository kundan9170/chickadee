"""The pure fold. 20 synthetic task-loop events → an expected state (Phase 0 DoD)."""

from __future__ import annotations

from typing import Any

import pytest
from helpers import payload_for, scope_args

from chickadee.journal import (
    Fold,
    Journal,
    JournalCorrupt,
    RunProjection,
    paths,
    project,
    project_before,
    project_readable,
    read_events,
)
from chickadee.schema import EventKind, JournalEvent
from chickadee.schema.journal import (
    BudgetP,
    EffectIntendedP,
    ModelCallP,
    QuestionAnsweredP,
    QuestionAskedP,
    TaskReviewedP,
)
from chickadee.state.roots import Roots

SHA = "a" * 64
ROUND_1 = {"task_id": "T1", "round": 1}

Script = list[
    tuple[EventKind, dict[str, Any]] | tuple[EventKind, dict[str, Any], dict[str, Any]]
]


def _write(repo: Roots, run_id: str, script: Script):
    """Each line's envelope is its kind's scope (`scope_args`), or the one a line names."""
    with Journal.create(repo, run_id, {}) as j:
        for kind, over, *envelope in script:
            env = envelope[0] if envelope else scope_args(kind)
            j.append(kind, payload_for(kind, **over), **env)
    return list(read_events(paths.journal_path(repo, run_id)))


def _clean_task() -> Script:
    """One task, one round, straight through to its acceptance — the shape every other path
    deviates from."""
    return [
        (EventKind.RUN_START, {"isolation": "worktree"}),
        (EventKind.SCOUT_START, {}),
        (EventKind.SCOUT_REPORT, {}),
        (EventKind.PLAN_WRITTEN, {"by": "athena", "version": 1, "asked": True}),
        (EventKind.TASK_DELEGATED, {"task_id": "T1", "by": "athena", "brief_sha": SHA}),
        (EventKind.BOX_MESSAGE, {"role": "worker", "side": "system"}, ROUND_1),
        (EventKind.BOX_MESSAGE, {"role": "worker", "side": "user"}, ROUND_1),
        (
            EventKind.MODEL_CALL,
            {"role": "worker", "tokens_in": 100, "tokens_out": 20, "cost_usd": 0.01},
            ROUND_1,
        ),
        (EventKind.BOX_MESSAGE, {"role": "worker", "side": "assistant"}, ROUND_1),
        (EventKind.TOOL_CALL, {"role": "worker", "tool": "edit"}, ROUND_1),
        (EventKind.BOX_MESSAGE, {"role": "worker", "side": "tool"}, ROUND_1),
        (
            EventKind.MODEL_CALL,
            {"role": "worker", "tokens_in": 120, "tokens_out": 10, "cost_usd": 0.01},
            ROUND_1,
        ),
        (
            EventKind.EFFECT_INTENDED,
            {"operation_id": "op1", "declared_surface": ["src/a.py"]},
            ROUND_1,
        ),
        (EventKind.EFFECT_STARTED, {"operation_id": "op1"}, ROUND_1),
        (
            EventKind.EFFECT_COMPLETED,
            {"operation_id": "op1", "outcome": "succeeded", "observed_surface": ["src/a.py"]},
            ROUND_1,
        ),
        (EventKind.TASK_REPORTED, {"task_id": "T1", "round": 1, "stopped": "replied"}),
        (EventKind.MODEL_CALL, {"role": "athena", "tokens_in": 50, "cost_usd": 0.02}, {}),
        (EventKind.TASK_REVIEWED, {"task_id": "T1", "answer": "accept"}),
        (EventKind.BOX_MESSAGE, {"role": "athena", "side": "assistant"}, {}),
        (EventKind.RUN_END, {"outcome": "COMPLETED", "tasks": 1, "committed": 1}),
    ]


def test_twenty_event_fold(repo: Roots, run_id: str) -> None:
    events = _write(repo, run_id, _clean_task())
    assert len(events) == 20
    st = project(events)

    assert st.run_id == run_id
    assert st.isolation == "worktree"
    assert st.end is not None and st.end.outcome == "COMPLETED"
    assert st.last_seq == 20
    assert st.defects == ()

    # The task's lines, kept by task in order: everything its envelope names it on.
    task = st.tasks["T1"]
    assert [line.seq for line in task] == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18]
    assert {line.round for line in task[1:-2]} == {1}
    assert set(st.tasks) == {"T1"}

    op = st.operations["op1"]
    assert op.phase == "completed"
    assert op.needs_reconciliation is False
    assert op.outcome == "succeeded"
    assert op.task_id == "T1"


def test_the_fold_is_pure(repo: Roots, run_id: str) -> None:
    """Same events, same projection — no filesystem, no clock."""
    events = _write(repo, run_id, _clean_task())
    assert project(events) == project(events)


def test_the_journal_folds_each_line_it_appends(repo: Roots, run_id: str) -> None:
    """The `Journal`'s projection, folded on append, is the file's, folded on read."""
    with Journal.create(repo, run_id, {}) as j:
        for kind, over, *envelope in _clean_task():
            env = envelope[0] if envelope else scope_args(kind)
            j.append(kind, payload_for(kind, **over), **env)
        held = j.projection()
    assert held == project(read_events(paths.journal_path(repo, run_id)))
    resumed = Journal.open_for_resume(repo, run_id)
    try:
        assert resumed.projection() == held
    finally:
        resumed.close()


def test_a_projection_handed_out_never_changes(repo: Roots, run_id: str) -> None:
    events = _write(repo, run_id, _clean_task())
    fold = Fold()
    for e in events[:10]:
        fold.add(e)
    early = fold.projection()
    for e in events[10:]:
        fold.add(e)
    assert early.last_seq == 10 and early.end is None and "op1" not in early.operations
    assert fold.projection().last_seq == 20


def test_the_run_as_it_stood_before_a_line(repo: Roots, run_id: str) -> None:
    events = _write(repo, run_id, _clean_task())
    before = project_before(events, 14)
    assert before.last_seq == 13
    assert before.operations["op1"].phase == "intended"


def _defects(script: Script, repo: Roots, run_id: str) -> RunProjection:
    return project(_write(repo, run_id, script))


def test_effect_lifecycle_ordering_is_enforced(repo: Roots, run_id: str) -> None:
    """Intent never implies safe-to-redo, so the phases must be reconstructible. A start
    with no intent is a defect, recorded; `require_consistent` refuses it as replay did."""
    st = _defects([(EventKind.EFFECT_STARTED, {"operation_id": "ghost"})], repo, run_id)
    assert [d.why for d in st.defects] == ["effect_started without intended"]
    with pytest.raises(JournalCorrupt, match="without intended"):
        st.require_consistent()


def test_an_effect_with_no_intent_is_still_an_operation(repo: Roots, run_id: str) -> None:
    """A damaged journal's effect is still seen. A start or a completion with no
    intent opens the operation with none, and the defect is recorded all the same: a reader
    shows what happened, and the readers that refuse still refuse."""
    st = _defects(
        [
            (EventKind.EFFECT_STARTED, {"operation_id": "ghost"}),
            (EventKind.EFFECT_COMPLETED, {"operation_id": "ghost", "outcome": "succeeded"}),
            (EventKind.EFFECT_COMPLETED, {"operation_id": "orphan", "outcome": "ambiguous"}),
        ],
        repo,
        run_id,
    )
    ghost, orphan = st.operations["ghost"], st.operations["orphan"]
    assert ghost.intended is None and ghost.started and ghost.outcome == "succeeded"
    assert orphan.intended is None and not orphan.started and orphan.outcome == "ambiguous"
    assert [d.why for d in st.defects] == [
        "effect_started without intended",
        "effect_completed without started",
    ]
    with pytest.raises(JournalCorrupt, match="without intended"):
        st.require_consistent()


def test_an_intent_written_late_is_the_operations_intent(repo: Roots, run_id: str) -> None:
    """An intent after its start is out of order, a defect; it is still what the operation
    declared, so it is kept rather than reported as a reuse."""
    st = _defects(
        [
            (EventKind.EFFECT_STARTED, {"operation_id": "op1"}),
            (EventKind.EFFECT_INTENDED, {"operation_id": "op1"}),
        ],
        repo,
        run_id,
    )
    op = st.operations["op1"]
    assert op.intended is st.lines(EffectIntendedP)[0].payload and op.started
    assert [d.why for d in st.defects] == ["effect_started without intended"]


def test_effect_completed_without_started_refuses(repo: Roots, run_id: str) -> None:
    st = _defects(
        [
            (EventKind.EFFECT_INTENDED, {"operation_id": "op1"}),
            (EventKind.EFFECT_COMPLETED, {"operation_id": "op1", "outcome": "succeeded"}),
        ],
        repo,
        run_id,
    )
    assert [d.seq for d in st.defects] == [2]
    with pytest.raises(JournalCorrupt, match="without started"):
        st.require_consistent()


def test_duplicate_operation_id_refuses(repo: Roots, run_id: str) -> None:
    """Invariant 5: operation_id is stable and unique, or reconciliation cannot pair an
    intent with what actually happened. The first intent is the one kept."""
    st = _defects(
        [
            (EventKind.EFFECT_INTENDED, {"operation_id": "op1"}),
            (EventKind.EFFECT_INTENDED, {"operation_id": "op1"}),
        ],
        repo,
        run_id,
    )
    first = st.lines(EffectIntendedP)[0].payload
    assert st.operations["op1"].intended is first
    with pytest.raises(JournalCorrupt, match="reused"):
        st.require_consistent()


def test_a_defect_does_not_stop_the_fold(repo: Roots, run_id: str) -> None:
    """It records what the journal says and refuses nothing: the lines after a defect fold."""
    st = _defects(
        [
            (EventKind.EFFECT_STARTED, {"operation_id": "ghost"}),
            (EventKind.EFFECT_INTENDED, {"operation_id": "op1"}),
            (EventKind.EFFECT_STARTED, {"operation_id": "op1"}),
        ],
        repo,
        run_id,
    )
    assert len(st.defects) == 1
    assert st.operations["op1"].phase == "started" and st.last_seq == 3


def test_interrupted_effect_is_flagged_for_reconciliation(repo: Roots, run_id: str) -> None:
    """The crash case D40 exists for: started, never completed."""
    st = project(
        _write(
            repo,
            run_id,
            [
                (EventKind.EFFECT_INTENDED, {"operation_id": "op1", "idempotent": False}),
                (EventKind.EFFECT_STARTED, {"operation_id": "op1"}),
            ],
        )
    )
    op = st.operations["op1"]
    assert op.phase == "started"
    assert op.needs_reconciliation is True
    assert op.intended is not None and op.intended.idempotent is False


def test_ambiguous_outcome_is_visible(repo: Roots, run_id: str) -> None:
    """An ambiguous external outcome suspends for the user; it must not look succeeded."""
    st = project(
        _write(
            repo,
            run_id,
            [
                (EventKind.EFFECT_INTENDED, {"operation_id": "op1"}),
                (EventKind.EFFECT_STARTED, {"operation_id": "op1"}),
                (EventKind.EFFECT_COMPLETED, {"operation_id": "op1", "outcome": "ambiguous"}),
            ],
        )
    )
    assert st.operations["op1"].outcome == "ambiguous"


def test_cassetted_kinds_are_named(repo: Roots, run_id: str) -> None:
    """The projection reads exec/mcp results as recorded; it says which, so 'state
    verified' is never read as 're-executed'."""
    st = project(
        _write(
            repo,
            run_id,
            [
                (EventKind.EXEC, {"operation_id": "op1"}),
                (EventKind.MCP_CALL, {"operation_id": "op2"}),
            ],
        )
    )
    assert st.cassetted_kinds == {"exec", "mcp_call"}
    assert len(st.external_effects) == 1


def test_external_effects_are_the_undo_list(repo: Roots, run_id: str) -> None:
    st = project(_write(repo, run_id, [(EventKind.MCP_CALL, {"tool": "create_issue"})]))
    assert st.external_effects[0].tool == "create_issue"


def test_human_assistance_is_recorded(repo: Roots, run_id: str) -> None:
    """A decision a human made must replay as human-made, not as ours. A question asked
    inside a task names the task on its envelope, and the fold files the line under it."""
    st = project(
        _write(
            repo,
            run_id,
            [
                (EventKind.QUESTION_ASKED, {"id": "q1"}, {"task_id": "T1"}),
                (
                    EventKind.QUESTION_ANSWERED,
                    {"id": "q1", "source": "human"},
                    {"task_id": "T1"},
                ),
            ],
        )
    )
    assert [line.seq for line in st.tasks["T1"]] == [1, 2]
    assert st.lines(QuestionAskedP)[0].task_id == "T1"
    answered = st.first(QuestionAnsweredP)
    assert answered is not None and answered.source == "human"


def test_spend_is_summed_once(repo: Roots, run_id: str) -> None:
    """Spend folds the `model_call` lines, per task by the envelope's `task_id`."""
    st = project(
        _write(
            repo,
            run_id,
            [
                *_clean_task()[:10],
                (
                    EventKind.MODEL_CALL,
                    {"role": "worker", "tokens_in": 5, "cost_usd": 0.5},
                    ROUND_1,
                ),
                (EventKind.BUDGET, {"scope": "run", "spent_usd": 2.0}),
            ],
        )
    )
    calls = [line.payload for line in st.lines(ModelCallP)]
    assert st.spend.cost_usd == sum(c.cost_usd for c in calls)
    assert st.spend.by_task["T1"][1] == st.spend.cost_usd
    assert st.spend.floor_usd == max(b.payload.spent_usd for b in st.lines(BudgetP))
    assert st.spend.proven_usd == max(st.spend.cost_usd, st.spend.floor_usd)


def test_spend_by_rung_is_the_calls_rung_else_its_role(repo: Roots, run_id: str) -> None:
    """A call naming its rung is counted by it; one naming none, by its role. A call
    outside any task counts for the run and for no task."""
    st = project(
        _write(
            repo,
            run_id,
            [
                (
                    EventKind.MODEL_CALL,
                    {"role": "worker", "rung": "low-param", "tokens_in": 3, "cost_usd": 0.25},
                    ROUND_1,
                ),
                (
                    EventKind.MODEL_CALL,
                    {"role": "worker", "rung": "low-param", "tokens_out": 4, "cost_usd": 0.25},
                    {"task_id": "T2", "round": 1},
                ),
                (EventKind.MODEL_CALL, {"role": "athena", "cost_usd": 1.0}, {}),
            ],
        )
    )
    assert dict(st.spend.by_rung) == {"low-param": 0.5, "athena": 1.0}
    # The same lines by the role that made each call.
    assert dict(st.spend.by_role) == {"worker": 0.5, "athena": 1.0}
    # `payload_for` fills each count it is not given with 1.
    assert dict(st.spend.by_task) == {"T1": (4, 0.25), "T2": (5, 0.25)}
    assert st.spend.cost_usd == 1.5


def test_a_fold_stopped_at_damage_keeps_what_it_read(repo: Roots, run_id: str) -> None:
    events = _write(repo, run_id, _clean_task())

    def torn() -> Any:
        yield from events[:5]
        raise ValueError("a torn UTF-8 tail")

    held = project_readable(torn())
    assert held.last_seq == 5 and held.tasks["T1"]


def test_lines_answer_by_payload_type(repo: Roots, run_id: str) -> None:
    events: list[JournalEvent] = _write(repo, run_id, _clean_task())
    st = project(events)
    assert [line.seq for line in st.lines(ModelCallP)] == [8, 12, 17]
    assert st.first(TaskReviewedP) is not None
