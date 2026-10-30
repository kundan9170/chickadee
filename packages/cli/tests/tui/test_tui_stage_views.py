"""The panes' stage views: what Athena's and the CODE
pane say before a run, and Athena's for a one-task plan.

What these pin: before a run the Planner's pane says it is idle and that the plan shows there,
as the worker's says it waits; a one-task plan is headed "ONE TASK · no plan" and says her
review and the final check follow, never "PLAN v1"; the idle CODE pane's key
hint lights `^D` only once there is a run and `^U` (built in Part 7) throughout, the bar's own
rule; and the worker never advertises `/quick`, which is not a command.
"""

from __future__ import annotations

from typing import Any

from recordings import run_frames
from tui_frames import SID, episode, run_record

from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.panes.athena import IDLE_TEXT, athena_body
from chickadee_cli.tui.panes.diff import diff_body
from chickadee_cli.tui.statusbar import key_hint
from chickadee_cli.tui.theme import UNICODE, palette

COLOURS = palette("truecolor")
HINT = (("ctrl+d", "diff"), ("ctrl+u", "undo task"))


def _lines(text: str) -> list[str]:
    return text.split("\n")


def test_before_a_run_the_planner_says_the_plan_shows_here() -> None:
    assert IDLE_TEXT.splitlines() == ["idle", "", "The plan shows here."]
    assert "HOW A RUN GOES" not in IDLE_TEXT, "never how a run goes (the owner's call)"


def test_a_one_task_plan_is_headed_one_task_and_says_her_review_follows() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    plan: dict[str, Any] = {
        "version": 1, "by": "zeus", "sha": "p" * 64, "asked": True,
    }  # fmt: skip
    feed.apply(run_record("plan_written", plan, 1))
    body = athena_body(feed.state, UNICODE, COLOURS).plain
    assert body.startswith("ONE TASK · no plan"), "04-quick.png"
    assert "PLAN v1" not in body
    assert "reviews after the report, then the final check" in body


def test_the_code_pane_s_hint_lights_a_key_only_where_it_applies() -> None:
    before = key_hint(Feed(SID).state, COLOURS, HINT)
    assert before.plain == "^D diff · ^U undo task"
    lit_before = [
        before.plain[s.start : s.end] for s in before.spans if COLOURS["zeus"] in str(s.style)
    ]
    assert lit_before == ["^U"], "no run yet: ^D dimmed; ^U (built in Part 7) takes turns back"
    feed = Feed(SID)
    feed.apply(episode(1))
    for frame in run_frames()[:12]:
        feed.apply(frame)
    hinted = diff_body(feed.state.task_diff, UNICODE, COLOURS, feed.state)
    lit = [
        hinted.plain[s.start : s.end] for s in hinted.spans if COLOURS["zeus"] in str(s.style)
    ]
    assert lit == ["^D", "^U"], "^D lit once there is a run"


def test_the_worker_never_advertises_a_command_that_does_not_exist() -> None:
    from chickadee_cli.tui.panes.worker import WorkerFrame

    frame = WorkerFrame(COLOURS, UNICODE)
    assert "/quick" not in frame.idle_text(Feed(SID).state).plain
