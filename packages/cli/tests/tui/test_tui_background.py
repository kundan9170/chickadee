"""The BACKGROUND pane: Zeus's
agents and jobs, under the CODE pane, only while one runs.

What these pin: an agent is folded from its `agent` records and worded by its `spawn_agent`
call; a job's effect makes `/jobs` stale and the answer is folded, worded by its shell call;
the pane shows while one runs, a finished one stays while another runs, and once none does
the pane goes and its items with it; its bottom border sits on the right column's bottom
row; `F5` is on the bar only while it shows.
"""

from __future__ import annotations

import pytest
from tui_frames import SID, session, user

from chickadee_cli.tui.app import ChickadeeApp
from chickadee_cli.tui.feed import Feed, Jobs
from chickadee_cli.tui.keys import BAR
from chickadee_cli.tui.model import StaleRoute, ZeusKind
from chickadee_cli.tui.panes.background import background_body, background_label
from chickadee_cli.tui.statusbar import bar_line
from chickadee_cli.tui.theme import UNICODE, palette
from chickadee_serve.frames import JsonValue, RecordFrame
from chickadee_wire.models import JobView

pytestmark = pytest.mark.anyio

COLOURS = palette("truecolor")
SIZE = (120, 34)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _turn(seq: int) -> Feed:
    """Zeus spawns two researchers and runs the tests in the background."""
    feed = Feed(SID)
    feed.apply(user("why are the retry tests flaky?", 1))
    calls: list[JsonValue] = [
        {"id": "c1", "name": "spawn_agent", "arguments": {"task": "urllib3 retry docs"}},
        {"id": "c2", "name": "spawn_agent", "arguments": {"task": "last 20 CI runs"}},
        {"id": "c3", "name": "shell", "arguments": {"command": "pytest -q tests/net"}},
    ]
    feed.apply(session("turn", {"role": "assistant", "text": "", "calls": calls}, seq))
    return feed


def _agent(event: str, seq: int, **payload: JsonValue) -> RecordFrame:
    return session("agent", {"event": event, "payload": dict(payload)}, seq)


def _start(agent: str, call_id: str, seq: int) -> RecordFrame:
    return _agent(
        "agent_start", seq, agent=agent, call_id=call_id, model="m", profile="researcher",
        round=0, rung="low-param", source="built-in", task_sha="t" * 64,
    )  # fmt: skip


def _report(agent: str, seq: int) -> RecordFrame:
    return _agent(
        "agent_report", seq, agent=agent, cost_usd=0.01, findings=4, flagged=0,
        report_sha="r" * 64, round=0, status="complete", tokens=900, tool_calls=5,
    )  # fmt: skip


def _job(running: bool) -> JobView:
    return JobView(
        job_id="1", kind="task", operation_id="job:S-tui:c3", report=None,
        row="1 · task · running", running=running, state="running" if running else "exit 0",
        step="S-tui:c3",
    )  # fmt: skip


def test_an_agent_runs_from_its_start_to_its_report_worded_by_its_call() -> None:
    feed = _turn(2)
    feed.apply(_start("A1", "c1", 3))
    feed.apply(_agent("agent_message", 4, agent="A1", round=0, sha="s" * 64, side="tool"))
    state = feed.state
    assert state.background.running and state.background.items == ["A1"]
    body = background_body(state, UNICODE, COLOURS).plain.split("\n")
    assert body == ["● A1 researcher  urllib3 retry docs", "  running · 1 call"]
    feed.apply(_start("A2", "c2", 5))
    feed.apply(_report("A1", 6))
    assert background_label(feed.state) == "1 running · 1 done", "a finished one stays"
    assert "✔ A1" in background_body(feed.state, UNICODE, COLOURS).plain
    feed.apply(_report("A2", 7))
    assert not feed.state.background.running
    assert feed.state.background.items == [], "once none runs, the burst is put away"


def test_a_job_s_effect_reads_jobs_again_and_the_answer_is_worded_by_its_call() -> None:
    feed = _turn(2)
    intended: dict[str, JsonValue] = {
        "event": "effect_intended",
        "payload": {"operation_id": "job:S-tui:c3", "effect_kind": "shell_job"},
    }
    changed = feed.apply(session("shell_effect", intended, 3))
    assert StaleRoute.JOBS in changed.refresh
    step: dict[str, JsonValue] = {
        "event": "effect_intended",
        "payload": {"operation_id": "S-tui:c9", "effect_kind": "shell_step"},
    }
    assert StaleRoute.JOBS not in feed.apply(session("shell_effect", step, 4)).refresh
    feed.snapshot(Jobs((_job(True),)))
    assert feed.state.background.running
    body = background_body(feed.state, UNICODE, COLOURS).plain.split("\n")
    assert body == ["● job 1  pytest -q tests/net", "  task · running"]
    feed.snapshot(Jobs((_job(False),)))
    assert not feed.state.background.running and feed.state.background.items == []


def test_f5_is_on_the_bar_only_while_the_pane_shows() -> None:
    feed = _turn(2)
    assert " F5 " not in bar_line(feed.state, COLOURS).plain
    feed.apply(_start("A1", "c1", 3))
    line = bar_line(feed.state, COLOURS).plain
    assert " F5 " in line and "Background" in line
    f5 = next(k for g in BAR for k in g if k.key == "f5")
    assert f5.drawn(feed.state)


async def test_the_pane_comes_and_goes_with_the_work_its_bottom_on_the_screen_s() -> None:
    feed = _turn(2)
    app = ChickadeeApp(feed=feed, repo="project-chickadee", depth="truecolor", encoding="utf-8")
    async with app.run_test(size=SIZE) as pilot:
        background = app.query_one("#background")
        assert not background.display, "nothing runs: CODE alone"
        app.take(_start("A1", "c1", 3))
        await pilot.pause()
        assert background.display
        code, worker = app.query_one("#diff"), app.query_one("#worker")
        assert background.region.bottom == worker.region.bottom, "the same bottom row"
        assert code.region.bottom == background.region.y, "CODE keeps the rest above"
        await pilot.press("f5")
        assert app.focused is background
        app.take(_report("A1", 4))
        await pilot.pause()
        assert not background.display, "the last one ended: the pane goes"
        assert app.focused is app.code, "its focus passes to CODE"


def test_an_agent_record_that_is_not_its_event_s_type_is_unreadable_not_a_crash() -> None:
    feed = _turn(2)
    feed.apply(_agent("agent_start", 3, agent="A1"))
    assert feed.state.zeus.entries[-1].kind is ZeusKind.UNREADABLE
    assert not feed.state.background.running
