"""An episode lives under the session family's caps (`T-3.8` items 5, 12 and 15): it charges the
session's own ledger, so it cannot reset the session's spend; it starts
under a lifted cap; and its `run_start` records every cap in force."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo, one_task, solved
from zeus_turn_helpers import WAIT_S, answer

from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.journal.reader import read_run
from chickadee.schema.adapter import Usage
from chickadee.schema.journal import RunStartP
from chickadee.state.roots import roots as resolve_roots


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


def run_start(repo: Path, run_id: str) -> RunStartP:
    first = next(iter(read_run(resolve_roots(repo), run_id)))
    assert isinstance(first.payload, RunStartP)
    return first.payload


def test_an_episode_cannot_reset_the_sessions_spend(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The episode charges the session's ledger; a run's start resets its run
    scope, never the session's. The family's spend after it is what it was, and the run's own
    cost beside it, counted once."""
    with Session.open(sessions_root(resolve_roots(repo)), SID) as session:
        session.record_spend(
            "high-param-model",
            Usage(prompt_tokens=100, completion_tokens=10),
            1.25,
            "high-param",
        )
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    before = engine.budget(SID).spent_usd

    result = engine.coder(one_task(), session_id=SID).result()

    assert before == pytest.approx(1.25)
    assert result.cost_usd > 0, "the scripted episode is priced"
    assert engine.budget(SID).spent_usd == pytest.approx(before + result.cost_usd)


def test_an_episode_starts_under_a_lifted_cap_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Item 15: `_journal_start` refused a lifted total. It records the config's total, as
    always, and the caps in force beside it."""
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    engine.set_cap(SID, None)

    handle = engine.coder(one_task(), session_id=SID)

    assert handle.result().outcome == "COMPLETED"
    start = run_start(repo, handle.run_id)
    assert start.hard_cost_cap_usd == raw["spend"]["hard_cost_usd"]
    assert start.caps is not None
    assert (start.caps.total, start.caps.total_lifted) == (None, True)


def test_run_start_records_each_rungs_cap_in_force(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    engine.set_cap(SID, 1.0, rung="low-param")
    engine.set_cap(SID, None, rung="mid-param")

    handle = engine.coder(one_task(), session_id=SID)

    handle.result()
    caps = run_start(repo, handle.run_id).caps
    assert caps is not None
    assert caps.total == raw["spend"]["hard_cost_usd"] and not caps.total_lifted
    assert caps.rungs.get("low-param") == 1.0
    assert caps.lifted_rungs == ("mid-param",)


def test_a_cap_set_after_a_turn_binds_the_next_episode(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The ledger is restored from the records before an episode, as before a turn,
    so a cap the person moved since the session's last turn is the one the episode runs
    under."""
    engine, _ = build(monkeypatch, repo, raw, [answer("hi"), *solved("return 1", "return 2")])
    engines.append(engine)
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    engine.set_cap(SID, None)

    handle = engine.coder(one_task(), session_id=SID)
    handle.result()

    caps = run_start(repo, handle.run_id).caps
    assert caps is not None and caps.total_lifted
