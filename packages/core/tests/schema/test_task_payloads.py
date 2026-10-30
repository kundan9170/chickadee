"""Contracts v2, the additive half: the task loop's line kinds round-trip, match the journal
golden, keep the
envelope's rules (a line names a task and never a packet, a round only with its task, a plan
is the run's), and fold by task."""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest

from chickadee.journal.projection import project
from chickadee.schema.export import GOLDEN_DIR
from chickadee.schema.journal import (
    BoxCompactionP,
    BoxMessageP,
    CommandFactP,
    EventKind,
    FileFactP,
    JournalEvent,
    PlanWrittenP,
    TaskDelegatedP,
    TaskFactsP,
    TaskReportedP,
    TaskReviewedP,
    ToolCallP,
)

RUN_ID = "2026-10-15-0a1b2c3d"
TS = "2026-10-15T09:00:00Z"
SHA = "b" * 64

FACTS = TaskFactsP(
    start_tree="t0",
    end_tree="t1",
    files=(FileFactP(path="src/stats.py", added=4, removed=1, kind="edited"),),
    tests_added=("tests/test_stats.py",),
    symbols_added=("stats.mean",),
    commands=(
        CommandFactP(
            operation_id="step-1",
            command="pytest -q tests/test_stats.py",
            exit_code=1,
            timed_out=False,
            tail="1 failed",
            output_sha=SHA,
            test_outcome="failed",
            failing=("tests/test_stats.py::test_mean",),
        ),
    ),
)

LINES: dict[EventKind, tuple[Any, dict[str, Any]]] = {
    EventKind.TASK_DELEGATED: (
        TaskDelegatedP(
            task_id="T1",
            call_id="call-7",
            title="Use it in stats",
            brief_sha=SHA,
            by="athena",
            start_tree="t0",
            start_commit=None,
        ),
        {"task_id": "T1"},
    ),
    EventKind.TASK_REPORTED: (
        TaskReportedP(task_id="T1", round=1, stopped="replied", report_sha=SHA, facts=FACTS),
        {"task_id": "T1"},
    ),
    EventKind.BOX_MESSAGE: (
        BoxMessageP(role="worker", side="assistant", sha=SHA),
        {"task_id": "T1", "round": 1},
    ),
    EventKind.BOX_COMPACTION: (
        BoxCompactionP(role="athena", steps=("evict",), evict=("call-3",), tokens_before=9),
        {},
    ),
    EventKind.TASK_REVIEWED: (
        TaskReviewedP(task_id="T1", call_id="call-9", answer="feedback", text_sha=SHA),
        {"task_id": "T1"},
    ),
    EventKind.PLAN_WRITTEN: (
        PlanWrittenP(call_id="call-2", sha=SHA, by="athena", version=1, asked=True),
        {},
    ),
}


def _event(seq: int, kind: EventKind, payload: Any, **envelope: Any) -> JournalEvent:
    return JournalEvent.model_validate(
        {
            "run_id": RUN_ID,
            "seq": seq,
            "ts": TS,
            "kind": kind.value,
            "payload": payload.model_dump(by_alias=True),
            **envelope,
        }
    )


def _golden() -> dict[str, object]:
    return json.loads((GOLDEN_DIR / "journal_event.schema.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("kind", list(LINES), ids=lambda k: k.value)
def test_each_new_line_round_trips_and_matches_the_golden(kind: EventKind) -> None:
    payload, envelope = LINES[kind]
    event = _event(1, kind, payload, **envelope)
    line = event.model_dump_json(by_alias=True, exclude_none=True)
    assert JournalEvent.model_validate_json(line) == event
    jsonschema.validate(json.loads(line), _golden())


def test_a_task_line_names_its_task_and_never_a_packet() -> None:
    payload, _ = LINES[EventKind.TASK_DELEGATED]
    with pytest.raises(ValueError, match="task-scoped"):
        _event(1, EventKind.TASK_DELEGATED, payload)
    # v2 has no packet: the envelope refuses the field itself.
    with pytest.raises(ValueError, match="packet_id"):
        _event(1, EventKind.TASK_DELEGATED, payload, task_id="T1", packet_id="p1")


def test_a_round_needs_its_task_and_a_run_line_takes_no_task() -> None:
    message, _ = LINES[EventKind.BOX_MESSAGE]
    with pytest.raises(ValueError, match="requires `task_id`"):
        _event(1, EventKind.BOX_MESSAGE, message, round=1)
    compaction, _ = LINES[EventKind.BOX_COMPACTION]
    _event(1, EventKind.BOX_COMPACTION, compaction)  # Athena's, outside any task: allowed
    with pytest.raises(ValueError, match="packet_id"):
        _event(1, EventKind.BOX_MESSAGE, message, task_id="T1", packet_id="p1")


def test_a_review_is_its_tasks_and_a_plan_is_the_runs() -> None:
    review, _ = LINES[EventKind.TASK_REVIEWED]
    with pytest.raises(ValueError, match="task-scoped"):
        _event(1, EventKind.TASK_REVIEWED, review)
    plan, _ = LINES[EventKind.PLAN_WRITTEN]
    with pytest.raises(ValueError, match="must not carry task_id"):
        _event(1, EventKind.PLAN_WRITTEN, plan, task_id="T1")


def test_a_task_id_is_the_harnesss_shape() -> None:
    payload, _ = LINES[EventKind.TASK_DELEGATED]
    with pytest.raises(ValueError):
        _event(1, EventKind.TASK_DELEGATED, payload, task_id="task-1")


def test_the_fold_keeps_each_tasks_lines_in_order() -> None:
    call = ToolCallP(
        role="worker", tool="read", args_sha=SHA, outcome="served", result_tokens=3
    )
    events = [
        _line(1, EventKind.TASK_DELEGATED),
        _event(2, EventKind.TOOL_CALL, call, task_id="T1", round=1),
        _line(3, EventKind.TASK_REPORTED),
        _line(4, EventKind.BOX_COMPACTION),
        _line(5, EventKind.TASK_REVIEWED),
        _line(6, EventKind.PLAN_WRITTEN),
    ]
    run = project(events)
    assert run.defects == ()
    assert [p.seq for p in run.tasks["T1"]] == [1, 2, 3, 5]
    assert [p.seq for p in run.lines(PlanWrittenP)] == [6]
    assert run.tasks["T1"][1].round == 1
    assert set(run.tasks) == {"T1"}


def _line(seq: int, kind: EventKind) -> JournalEvent:
    payload, envelope = LINES[kind]
    return _event(seq, kind, payload, **envelope)
