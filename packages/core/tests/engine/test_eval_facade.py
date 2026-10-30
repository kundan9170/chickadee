"""What the researcher's evaluation reaches through the facade.

`spawn_agent`: a round of a profile, run by the session's own agents as a Zeus `spawn_agent`
call runs it, its report returned as a value to score. A profile the session cannot spawn is
refused, as Zeus's call is; a session that does not exist is a `SessionError`.

`ask_model`: one prompt on a catalog model, outside any session, for the eval's judge; what it
cost by the model's price row, and nothing sent for a model the catalog lacks or cannot price.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import SID, answer, call, calls, config_for, start, with_sessions

from chickadee.agents.report import AgentReport
from chickadee.context.results import InvalidArgument
from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError, SessionError

REPORT = "## Summary\nIt is done.\n## Findings\n- none\n## Not done\nnothing\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo"), SID)


def priced(raw: dict[str, Any]) -> dict[str, Any]:
    """Every model priced: an agent's round is charged to its own rung, and an unpriced one is
    refused before it runs."""
    raw["prices"] = {model: {"price_in": 1.0, "price_out": 2.0} for model in raw["models"]}
    return raw


def test_a_round_returns_its_report_as_a_value(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(
        monkeypatch, repo, config_for(priced(raw), priced=False), [answer(REPORT)]
    )
    engines.append(engine)
    got = engine.spawn_agent(SID, "researcher", "What is the limit?", call_id="eval-q-1")
    assert isinstance(got, AgentReport)
    assert (got.profile, got.status, got.summary) == ("researcher", "complete", "It is done.")
    assert got.tokens > 0
    assert len(model.requests) == 1


def test_a_profile_the_session_cannot_spawn_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(priced(raw), priced=False))
    engines.append(engine)
    got = engine.spawn_agent(SID, "nobody", "What is the limit?", call_id="eval-q-1")
    assert isinstance(got, InvalidArgument)
    assert "nobody" in got.render()
    assert not model.requests


def test_a_session_that_does_not_exist_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(priced(raw), priced=False))
    engines.append(engine)
    with pytest.raises(SessionError):
        engine.spawn_agent("S-nope", "researcher", "What is the limit?", call_id="eval-q-1")


def test_a_model_is_asked_once_outside_any_session_and_priced(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(
        monkeypatch, repo, config_for(priced(raw), priced=False), [answer("graded")]
    )
    engines.append(engine)
    judge = next(iter(raw["models"]))
    asked = engine.ask_model(judge, "Grade this.")
    assert (asked.text, asked.tokens) == ("graded", 110)
    # 100 tokens in at 1.0 and 10 out at 2.0, per million.
    assert asked.cost_usd == pytest.approx(120 / 1_000_000)
    (request,) = model.requests
    assert [(m.role, m.content) for m in request.messages] == [("user", "Grade this.")]
    assert not request.tools
    # The suite's one session, untouched: nothing was written to any log.
    assert [(s.session_id, s.last_seq) for s in engine.sessions()] == [(SID, 0)]


def test_a_model_the_catalog_lacks_or_cannot_price_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    unpriced = next(iter(raw["models"]))
    engine, model = start(monkeypatch, repo, config_for(raw, priced=False))
    engines.append(engine)
    with pytest.raises(ConfigError, match="no model 'nope' in the catalog"):
        engine.ask_model("nope", "Grade this.")
    with pytest.raises(ConfigError, match="no price row"):
        engine.ask_model(unpriced, "Grade this.")
    assert not model.requests


def test_a_reader_narrowed_to_grep_keeps_its_search(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Which ports a round gets is its role's: a reader that kept only `grep` needs no `files`,
    and is a reader still, never given a researcher's web alone."""
    raw["agents"] = {"profiles": {"finder": {"base": "reader", "tools": ["grep"]}}}
    script = [calls(call("grep", "g1", pattern="hello")), answer(REPORT)]
    engine, model = start(monkeypatch, repo, config_for(priced(raw), priced=False), script)
    engines.append(engine)
    got = engine.spawn_agent(SID, "finder", "Where is hello?", call_id="eval-q-1")
    assert isinstance(got, AgentReport)
    assert got.status == "complete", got.why
    assert [t.name for t in model.requests[0].tools] == ["grep"]
