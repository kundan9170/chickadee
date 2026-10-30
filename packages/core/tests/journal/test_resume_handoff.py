"""Mid-flight detection, parametrized over every task-scoped kind.

The property that matters for resume: killing after *any* event leaves a journal that
replays to a consistent state, and an interrupted effect is visible as interrupted rather
than assumed complete.
"""

from __future__ import annotations

import pytest
from helpers import payload_for, scope_args

from chickadee.journal import Journal, RunProjection, paths, project, read_events
from chickadee.schema import EventKind
from chickadee.schema.journal import KIND_TABLE, Scope
from chickadee.state.roots import Roots

TASK_KINDS = [k for k in EventKind if KIND_TABLE[k][1] is Scope.TASK]


@pytest.mark.parametrize("kind", TASK_KINDS, ids=lambda k: k.value)
def test_kill_after_each_task_kind_replays(kind: EventKind, repo: Roots, run_id: str) -> None:
    """Every task-scoped kind can be the last durable event. None may make replay
    raise — a crash at an awkward moment must not need a human to unpick."""
    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(
            EventKind.TASK_DELEGATED,
            payload_for(EventKind.TASK_DELEGATED),
            task_id="T1",
        )
        if kind is not EventKind.TASK_DELEGATED:
            j.append(kind, payload_for(kind), **scope_args(kind))

    state = project(read_events(paths.journal_path(repo, run_id)))
    assert isinstance(state, RunProjection)
    assert state.end is None, "an interrupted run has no terminal"
    assert "T1" in state.tasks


def test_resume_continues_the_sequence(repo: Roots, run_id: str) -> None:
    """Resume must not restart seq — a gap or a repeat is corruption on the next read."""
    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(EventKind.SCOUT_START, payload_for(EventKind.SCOUT_START))

    j2 = Journal.open_for_resume(repo, run_id)
    state = j2.projection()
    try:
        assert state.last_seq == 2
        assert j2.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL)) == 3
    finally:
        j2.close()
    assert [e.seq for e in read_events(paths.journal_path(repo, run_id))] == [1, 2, 3]


def test_resume_after_torn_tail(repo: Roots, run_id: str) -> None:
    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(EventKind.SCOUT_START, payload_for(EventKind.SCOUT_START))
    path = paths.journal_path(repo, run_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "seq": 3, "kin')

    j2 = Journal.open_for_resume(repo, run_id)
    state = j2.projection()
    try:
        assert state.last_seq == 2
        assert j2.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL)) == 3
    finally:
        j2.close()


@pytest.mark.parametrize("phase", ["intended", "started"])
def test_interrupted_effect_never_looks_complete(phase: str, repo: Roots, run_id: str) -> None:
    """J-7: 'intent means safe to redo' is exactly the assumption that corrupts state."""
    with Journal.create(repo, run_id, {}) as j:
        j.append(
            EventKind.EFFECT_INTENDED,
            payload_for(EventKind.EFFECT_INTENDED, operation_id="op1"),
        )
        if phase == "started":
            j.append(
                EventKind.EFFECT_STARTED,
                payload_for(EventKind.EFFECT_STARTED, operation_id="op1"),
            )

    state = project(read_events(paths.journal_path(repo, run_id)))
    op = state.operations["op1"]
    assert op.phase == phase
    assert op.needs_reconciliation is True
    assert op.outcome is None
