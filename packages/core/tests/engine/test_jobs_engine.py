"""Background jobs through the engine: a command Zeus starts in the background
runs on as a job, its exit is a notice on the session log, and an idle Zeus is woken by it to
take a turn of his own. Closing the engine kills what is still running.

The real engine and loop, the model scripted, no sandbox, the person having lifted the
unsandboxed floor, as `test_declared_write.py` runs its shell.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo
from test_declared_write import Approving
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

from chickadee.context.standing import NoticeTrigger
from chickadee.engine import PacketEngine


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    script: list[Any],
) -> PacketEngine:
    raw["sandbox"] = {"allow_unsandboxed_shell": {"zeus": True}}
    engine, _ = build(monkeypatch, repo, raw, script, interaction=Approving())
    engines.append(engine)
    return engine


def _answers(engine: PacketEngine) -> list[str]:
    return [t.text for t in state(engine, SID).turns() if t.role == "assistant" and t.text]


def _until(check: Any) -> None:
    deadline = time.monotonic() + WAIT_S
    while not check() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert check()


def test_a_job_s_exit_wakes_an_idle_zeus_with_no_user_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(call("shell", "s1", command="sleep 1; echo suite passed", background=True)),
        answer("started the suite as j1"),
        answer("j1 passed"),
    ]
    engine = _engine(monkeypatch, repo, raw, engines, script)
    engine.turn(SID, "run the suite").wait(timeout=WAIT_S)
    assert _answers(engine) == ["started the suite as j1"]
    _until(lambda: _answers(engine) == ["started the suite as j1", "j1 passed"])
    turns = state(engine, SID).turns()
    users = [t for t in turns if t.role == "user"]
    told = [
        t
        for t in turns
        if t.notice is not None and t.notice.trigger is NoticeTrigger.JOB_EXITED
    ]
    assert len(users) == 1, "the harness put no words in the person's mouth"
    assert len(told) == 1 and "suite passed" in told[0].text


def test_closing_the_engine_kills_a_running_job_and_wakes_nobody(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(call("shell", "s1", command="sleep 60", background=True)),
        answer("a server is up as j1"),
    ]
    engine = _engine(monkeypatch, repo, raw, engines, script)
    engine.turn(SID, "start it").wait(timeout=WAIT_S)
    assert state(engine, SID).jobs.ids() == ("j1",)
    engine.close()
    after, _ = engine._on_disk(SID)  # pyright: ignore[reportPrivateUsage]
    assert after.jobs.ids() == ()
    assert not any(
        t.notice is not None and t.notice.trigger is NoticeTrigger.JOB_EXITED
        for t in after.turns()
    ), "a kill by close is recorded, never told"


def test_a_job_runs_on_across_a_model_switch_and_its_exit_wakes_the_rebuilt_loop(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script = [
        calls(call("shell", "s1", command="sleep 3; echo suite passed", background=True)),
        answer("started the suite as j1"),
        answer("on the other model"),
        answer("j1 passed"),
    ]
    engine = _engine(monkeypatch, repo, raw, engines, script)
    engine.turn(SID, "run the suite").wait(timeout=WAIT_S)
    engine.switch_model(SID, "high-param", raw["rungs"]["low-param"])
    # The next word rebuilds the loop; the job, its table and the writer stay.
    engine.turn(SID, "switched").wait(timeout=WAIT_S)
    assert state(engine, SID).jobs.ids() == ("j1",), "a rebuild kills nothing"
    _until(lambda: _answers(engine)[-1:] == ["j1 passed"])
    (view,) = engine.jobs(SID)
    assert view.report is not None
    assert (view.report.ended, view.report.by, view.report.exit_code) == ("exited", None, 0)
    told = [
        t
        for t in state(engine, SID).turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.JOB_EXITED
    ]
    assert len(told) == 1 and "suite passed" in told[0].text, "never settled as lost"
