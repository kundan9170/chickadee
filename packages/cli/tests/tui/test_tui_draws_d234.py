"""What the screen draws of D234's facts (T-5.23 Part 8): the model that served a box call, a
rung's switch making `/rung` stale, and the approved plan's tasks, waiting until delegated."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from recordings import recorded_run_id, recording, replayed, run_frames
from tui_frames import SID, approval_asked, assistant, episode, progress, run_record, session

from chickadee_cli.tui.composer import options
from chickadee_cli.tui.feed import Feed
from chickadee_cli.tui.header import banner_line
from chickadee_cli.tui.model import Connection, Pane, SpendLevel, StaleRoute
from chickadee_cli.tui.panes.athena import waiting
from chickadee_cli.tui.panes.diff import diff_body
from chickadee_cli.tui.panes.worker import worker_after
from chickadee_cli.tui.panes.zeus import status_text
from chickadee_cli.tui.theme import UNICODE, palette
from chickadee_serve.frames import Hello, JsonValue
from chickadee_wire.events import Frame, Reconnected, Retrying
from chickadee_wire.models import PathDiff, RunDiff, SessionBudget, TaskDelegatedP

COLOURS = palette("truecolor")


def _call(role: str, model: str | None) -> dict[str, JsonValue]:
    call: dict[str, JsonValue] = {
        "role": role, "request": "turn", "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.01,
        "stop_reason": "end_turn", "rung": "high-param",
    }  # fmt: skip
    if model is not None:
        call["model"] = model
    return call


def test_a_box_call_names_the_model_that_served_it_and_an_older_one_keeps_the_last() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    feed.apply(run_record("model_call", _call("athena", "big-planner"), 1))
    feed.apply(run_record("model_call", _call("athena", None), 2))
    assert feed.state.athena.model == "big-planner"
    feed.apply(run_record("model_call", _call("worker", "small-coder"), 3, task_id="T1"))
    assert feed.state.worker.tasks["T1"].model == "small-coder"


def test_a_rungs_switch_makes_the_wiring_stale() -> None:
    feed = Feed(SID)
    switched: dict[str, JsonValue] = {
        "rung": "low-param", "from_model": "a", "to_model": "b", "model_id": "b-1",
        "rungs_sha": "r" * 64,
    }  # fmt: skip
    changed = feed.apply(session("model_switched", switched, 1))
    assert StaleRoute.RUNG in changed.refresh


def test_the_plans_tasks_wait_until_a_delegation_runs_them() -> None:
    feed = Feed(SID)
    feed.apply(episode(1))
    tasks: list[JsonValue] = [{"id": "T1", "title": "One"}, {"id": "T2", "title": "Two"}]
    plan: dict[str, JsonValue] = {
        "asked": True, "by": "athena", "sha": "p" * 64, "version": 1, "tasks": tasks,
    }  # fmt: skip
    feed.apply(run_record("plan_written", plan, 1))
    assert [p.id for p in waiting(feed.state)] == ["T1", "T2"]
    delegated: dict[str, JsonValue] = {
        "task_id": "T1", "title": "One", "brief_sha": "b" * 64, "by": "athena",
        "start_tree": "t" * 40, "plan_task": "T1",
    }  # fmt: skip
    feed.apply(run_record("task_delegated", delegated, 2, task_id="T1"))
    assert [p.id for p in waiting(feed.state)] == ["T2"]
    assert feed.state.header.work_planned == 2


def test_the_header_is_amber_once_the_family_passes_its_first_warning_level() -> None:
    feed = Feed(SID)
    below = SessionBudget(
        spent_usd=0.7, cap_usd=1.0, cap_source="config", cap_seq=None, warning_fractions=[0.8]
    )
    feed.snapshot(below)
    assert feed.state.header.spend.level is SpendLevel.OK
    feed.snapshot(below.model_copy(update={"spent_usd": 0.85}))
    assert feed.state.header.spend.level is SpendLevel.WARNING


def test_a_question_says_when_it_answers_itself() -> None:
    """Its record's time plus the server's timeout, in local time."""
    feed = Feed(SID)
    assert feed.state.question_timeout_s is None
    feed.snapshot(Hello(wire="4.4", engine="e", capabilities=(), question_timeout_s=900))
    feed.apply(approval_asked("shell-1", 1))
    [asked] = feed.state.stops.open
    at = datetime.fromisoformat(feed.state.asked_at[asked.seq].replace("Z", "+00:00"))
    expected = (at + timedelta(seconds=900)).astimezone().strftime("%H:%M")
    assert feed.state.deadline(asked) == expected


def test_the_stream_retrying_is_reconnecting_until_it_is_live_again() -> None:
    feed = Feed(SID)
    feed.link(Retrying(1, 0.2, "connection reset"))
    assert feed.state.connection is Connection.RECONNECTING
    banner = banner_line(feed.state, COLOURS)
    assert banner is not None and "RECONNECTING" in banner.plain
    assert "try 1: connection reset" in banner.plain
    feed.link(Reconnected())
    assert feed.state.connection is Connection.LIVE and banner_line(feed.state, COLOURS) is None
    feed.lost("the session's stream", "gave up", stream=True)
    feed.link(Reconnected())
    assert feed.state.connection is Connection.LOST, "a lost wire stays lost"


def test_a_replay_serves_its_runs_named_blobs(tmp_path: Path) -> None:
    """D234 decision 10, end to end: a recorded run's brief over `--replay`'s wire."""
    delegated = next(f for f in run_frames() if f.kind == "task_delegated")
    brief = TaskDelegatedP.model_validate(delegated.payload).brief_sha
    with replayed(recording(tmp_path / "recording")) as served:
        got = served.client.run_blob(recorded_run_id(), brief)
        assert got.field == "brief_sha" and got.text


DIFF_TEXT = (
    "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
    "@@ -1,2 +1,2 @@\n def go() -> int:\n-    return 1\n+    return 2\n"
)


def test_the_shown_tasks_diff_is_its_counts_and_its_hunks() -> None:
    """The server's diff of the task, start to its latest step."""
    feed = Feed(SID)
    feed.apply(episode(1))
    delegated: dict[str, JsonValue] = {
        "task_id": "T1", "title": "One", "brief_sha": "b" * 64, "by": "athena",
        "start_tree": "t" * 40,
    }  # fmt: skip
    changed = feed.apply(run_record("task_delegated", delegated, 1, task_id="T1"))
    assert StaleRoute.TASK_DIFF in changed.refresh, "a new task: its diff is read"
    run_id = feed.state.athena.run_id
    assert run_id is not None
    file = PathDiff(path="app.py", kind="edited", added=1, removed=1)
    feed.snapshot(
        RunDiff(
            run_id=run_id,
            task_id="T1",
            operation_id=None,
            base="t" * 40,
            tip="u" * 40,
            files=[file],
            text=DIFF_TEXT,
        )
    )
    view = feed.state.task_diff
    assert view.lines == (1, 1) and view.text == DIFF_TEXT
    body = diff_body(view, UNICODE, COLOURS, feed.state).plain
    assert "   2 + " in body and "   2 - " in body and "@@ -1,2 +1,2 @@" in body
    # An older run's diff, answered late, is not this run's.
    late = RunDiff(
        run_id="other", task_id="T1", operation_id=None, base="x", tip=None, files=[], text=""
    )
    feed.snapshot(late)
    assert feed.state.task_diff.text == DIFF_TEXT


def test_at_offers_the_workspaces_paths_as_the_repl_ranks_them() -> None:
    """D212 decision 9 on the TUI (W9): `@word` at the end lists paths, best first, and the
    chosen one replaces the `@word`; a `/` line still lists commands."""
    paths = ("src/segtree.py", "tests/test_segtree.py", "README.md")
    offered = options("look at @seg", paths)
    assert [o.insert for o in offered] == ["src/segtree.py", "tests/test_segtree.py"]
    chosen = offered[0]
    assert "look at @seg"[: chosen.start] + chosen.insert == "look at src/segtree.py"
    assert options("look at", paths) == ()
    assert options("/und", paths)[0].insert == "/undo "


def _piece(role: str, reply: str, offset: int, piece: str, task: str | None = None) -> Frame:
    payload: dict[str, JsonValue] = {
        "role": role, "reply": reply, "offset": offset, "piece": piece, "tokens": 3,
    }  # fmt: skip
    if task is not None:
        payload["task"] = task
    return progress("text", **payload)


def test_zeus_s_reply_is_drawn_as_it_streams_and_his_record_replaces_it() -> None:
    """W10: the live frames, through the REPL's own fold (`render/live.Replies`)."""
    feed = Feed(SID)
    feed.apply(_piece("zeus", "r1", 0, "Reading segtree"))
    feed.apply(_piece("zeus", "r1", 15, ".py first."))
    assert "Reading segtree.py first." in status_text(feed.state, UNICODE, COLOURS).plain
    feed.apply(assistant("Reading segtree.py first.", 1))
    assert not feed.state.live.open
    assert "Reading" not in status_text(feed.state, UNICODE, COLOURS).plain


def test_a_workers_reply_is_its_tasks_live_line_until_its_model_call() -> None:
    """W7: the box's replies stream to its panes; Zeus's `busy` stays his own."""
    feed = Feed(SID)
    feed.apply(episode(1))
    delegated: dict[str, JsonValue] = {
        "task_id": "T1", "title": "One", "brief_sha": "b" * 64, "by": "athena",
        "start_tree": "t" * 40,
    }  # fmt: skip
    feed.apply(run_record("task_delegated", delegated, 1, task_id="T1"))
    changed = feed.apply(_piece("worker", "w1", 0, "editing segtree.py", task="T1"))
    assert Pane.WORKER in changed.panes and feed.state.zeus.busy is None

    def after() -> str:
        return "\n".join(line.plain for line in worker_after(feed.state, UNICODE, COLOURS))

    assert "editing segtree.py" in after(), "under its cards, as it is written"
    feed.apply(run_record("model_call", _call("worker", None), 2, task_id="T1"))
    assert "editing segtree.py" not in after()


def test_a_turn_that_errors_ends_what_streamed() -> None:
    feed = Feed(SID)
    feed.apply(_piece("zeus", "r1", 0, "half a"))
    feed.apply(progress("error", text="the model went away"))
    assert not feed.state.live.open
