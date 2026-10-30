"""The header's stage, from the records alone.

Design while Zeus talks or stop 1 is open; plan once a coder call's run opens, until its plan
is approved, or while stop 2 is open; work while a delegated task is unsettled; check once
a delegated task is unsettled, and on until the run ends, since no record yet says the
plan's last task is done (check is not derived); hand off once the run has
ended and the session's workspace awaits its choice; design again after.
"""

from __future__ import annotations

from tui_frames import (
    SID,
    answered,
    assistant,
    episode,
    run_frames,
    run_record,
    stop_asked,
    user,
    workspace,
)

from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.model import Pane, Stage
from chickadee_serve.frames import JsonValue


def _stage_after(seq: int) -> tuple[Stage, int, int]:
    """The stage and the work counts once the recorded run reached `seq`."""
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames():
        if frame.seq <= seq:
            feed.apply(frame)
    header = feed.state.header
    return header.stage, header.work_done, header.work_delegated


def test_talking_is_design_and_so_is_stop_1() -> None:
    feed = Feed(SID)
    feed.apply(user("add range sums", 1))
    feed.apply(assistant("two questions first", 2))
    assert feed.state.header.stage is Stage.DESIGN
    feed.apply(stop_asked("stop-1", "design", 3))
    assert feed.state.header.stage is Stage.DESIGN


def test_a_run_is_plan_until_its_plan_is_approved() -> None:
    assert _stage_after(0) == (Stage.PLAN, 0, 0)
    assert _stage_after(2) == (Stage.PLAN, 0, 0)


def test_stop_2_open_is_plan() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    feed.apply(stop_asked("stop-2", "plan", 2))
    assert feed.state.header.stage is Stage.PLAN
    feed.apply(answered("stop-2", 3))
    assert feed.state.header.stage is Stage.PLAN, "approved only once the run says so"


def test_work_counts_the_tasks_settled_of_those_delegated() -> None:
    assert _stage_after(3) == (Stage.WORK, 0, 0)
    assert _stage_after(6) == (Stage.WORK, 0, 1)
    assert _stage_after(40) == (Stage.WORK, 0, 1), "accepted, its commit not landed"
    assert _stage_after(44) == (Stage.WORK, 1, 2)


def test_every_task_settled_with_the_run_open_is_still_work_never_a_guessed_check() -> None:
    # Between T1's commit and T2's delegation every task delegated so far has settled, as it
    # has after T2's: nothing on the record tells the two apart, so neither reads as check.
    assert _stage_after(41) == (Stage.WORK, 1, 1)
    assert _stage_after(53) == (Stage.WORK, 2, 2)


def test_a_run_ended_is_hand_off_while_its_workspace_awaits_the_choice() -> None:
    feed = Feed(SID)
    feed.apply(workspace("pending", 1))
    feed.apply(episode(2))
    for frame in run_frames():
        feed.apply(frame)
    assert feed.state.header.stage is Stage.HANDOFF
    changed = feed.apply(workspace("squash", 3))
    assert feed.state.header.stage is Stage.DESIGN
    assert Pane.HEADER in changed.panes


def test_a_run_ended_with_nothing_to_hand_off_is_design_again() -> None:
    assert _stage_after(56)[0] is Stage.DESIGN


def test_a_cancel_at_stop_2_is_design_once_the_run_ends() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    cancelled = {
        "asked": True, "by": "athena", "outcome": "cancelled", "sha": "p" * 64, "version": 1,
    }  # fmt: skip
    feed.apply(run_record("plan_written", cancelled, 1))
    assert feed.state.header.stage is Stage.PLAN
    end = {
        "outcome": "ABORTED", "committed": 0, "tasks": 0, "cost_usd": 0.01, "tokens": 10,
        "wall_ms": 5,
    }  # fmt: skip
    feed.apply(run_record("run_end", end, 2))
    assert feed.state.header.stage is Stage.DESIGN


def test_a_re_cut_task_leaves_the_count() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    plan = {"asked": True, "by": "athena", "outcome": "approved", "sha": "p" * 64, "version": 1}
    feed.apply(run_record("plan_written", plan, 1))
    for seq, task in ((2, "T1"), (3, "T2")):
        delegated: dict[str, JsonValue] = {
            "task_id": task, "title": task, "brief_sha": "b", "by": "athena", "start_tree": "t",
        }  # fmt: skip
        feed.apply(run_record("task_delegated", delegated, seq, task_id=task))
    review: dict[str, JsonValue] = {"task_id": "T1", "call_id": "c", "answer": "re_cut"}
    feed.apply(run_record("task_reviewed", review, 4, task_id="T1"))
    header = feed.state.header
    assert (header.stage, header.work_done, header.work_delegated) == (Stage.WORK, 0, 1)


def test_a_one_task_plan_is_asked_at_stop_2_only() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    feed.apply(stop_asked("stop-2", "plan", 2))
    assert feed.state.header.stage is Stage.PLAN
    plan = {"asked": True, "by": "zeus", "outcome": "approved", "sha": "p" * 64, "version": 1}
    feed.apply(answered("stop-2", 3))
    feed.apply(run_record("plan_written", plan, 1))
    assert feed.state.header.stage is Stage.WORK
    assert feed.state.athena.plan_by == "zeus", "the header shows design and plan skipped"
