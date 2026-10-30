"""A rung's model switches between turns.

A switch is the person's, checked as a load is, made only while nothing runs, and recorded as
a `model_switched` session-log line; the rung's next call, never the one in flight, goes to
the new model and is priced at it. A pinned (eval) session refuses it, and a new session starts
from the engine config."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from engines import MODEL, with_prices
from repos import init_repo
from zeus_turn_helpers import (
    SID,
    WAIT_S,
    ScriptedRung,
    Step,
    answer,
    config_for,
    held,
    state,
    with_sessions,
)

import chickadee.engine as engine_module
from chickadee.config import RungEndpoint
from chickadee.context.lines import SpendLine
from chickadee.engine import PacketEngine
from chickadee.errors import ConfigError, EngineBusy
from chickadee.state.roots import roots as resolve_roots
from chickadee.wiring import record_switch, rungs_sha

OTHER = "frontier-b"
"""A catalog model no rung names: the switch target."""
ZEUS = "high-param"
"""The rung the `zeus` role runs on by default."""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID, "S-two")


PRICE = {"price_in": 3.0, "price_out": 6.0}


def _engine(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    script: list[Step],
    *,
    prices: dict[str, dict[str, float]] | None = None,
) -> tuple[PacketEngine, list[str]]:
    """An engine on `raw` with `OTHER` in its catalog (priced by default), every catalog
    model's id its own name, and every client recording the model it was made for while it
    answers `script`."""
    for name, model in raw["models"].items():
        model["model_id"] = name
    raw["models"][OTHER] = {**MODEL, "model_id": OTHER}
    with_prices(raw)
    raw["prices"].update({OTHER: PRICE} if prices is None else prices)
    resolved = config_for(raw, priced=False)
    model = ScriptedRung(script)
    made: list[str] = []

    def client_for(endpoint: RungEndpoint) -> ScriptedRung:
        made.append(endpoint.model_id)
        return model

    monkeypatch.setattr(engine_module, "client_for", client_for)
    engine = PacketEngine(
        repo, resolved.engine, trust=resolved.trust(resolve_roots(repo).repo_local)
    )
    return engine, made


def _charged(engine: PacketEngine) -> list[str]:
    """The model each of Zeus's charges names, in order."""
    return [r.model for _, r in state(engine).lines() if isinstance(r, SpendLine)]


def test_a_switch_takes_effect_at_the_rungs_next_call_never_the_one_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    gate = threading.Event()
    engine, made = _engine(monkeypatch, repo, raw, [held(gate), answer("on the new model")])
    engines.append(engine)
    before = raw["rungs"][ZEUS]

    running = engine.turn(SID, "first")
    with pytest.raises(EngineBusy, match="between turns"):
        engine.switch_model(SID, ZEUS, OTHER)
    gate.set()
    running.wait(timeout=WAIT_S)
    assert before in made and OTHER not in made, "the call in flight stayed where it started"

    moved = engine.switch_model(SID, ZEUS, OTHER)
    assert (moved.rung, moved.from_model, moved.to_model, moved.model_id) == (
        ZEUS,
        before,
        OTHER,
        OTHER,
    )
    wiring = engine.models(SID)
    assert wiring.rungs[ZEUS] == OTHER
    assert moved.rungs_sha == rungs_sha(wiring.rungs) != rungs_sha(engine.config.rungs)
    assert wiring.switches == (moved,)
    assert OTHER in wiring.catalog

    composed = len(made)
    engine.turn(SID, "second").wait(timeout=WAIT_S)
    assert OTHER in made[composed:], "the next turn's loop was composed on the new model"
    assert before not in made[composed:], "every part on the rung moved with it"
    assert _charged(engine) == [before, OTHER], "the next call is priced at the new model"
    assert engine.config.rungs[ZEUS] == before, "the engine config, and its sha, do not move"


def test_each_refusal_names_why_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    raw["models"]["tiny"] = {
        **MODEL,
        "model_id": "tiny",
        "context_window": 4096,
        "max_output_tokens": 1024,
    }
    engine, _ = _engine(monkeypatch, repo, raw, [], prices={OTHER: PRICE, "tiny": PRICE})
    engines.append(engine)

    with pytest.raises(ConfigError, match="not in the catalog"):
        engine.switch_model(SID, ZEUS, "no-such-model")
    with pytest.raises(ConfigError, match="not declared"):
        engine.switch_model(SID, "no-such-rung", OTHER)
    with pytest.raises(
        ConfigError, match=r"cannot fit one request .*exceed the window \(4096\)"
    ):
        engine.switch_model(SID, ZEUS, "tiny")
    with pytest.raises(ConfigError, match="already runs on"):
        engine.switch_model(SID, ZEUS, raw["rungs"][ZEUS])
    assert engine.models(SID).switches == ()


def test_a_model_with_no_price_row_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _engine(monkeypatch, repo, raw, [], prices={})
    engines.append(engine)
    with pytest.raises(ConfigError, match=r"no price row.*\[prices\.frontier-b\]"):
        engine.switch_model(SID, ZEUS, OTHER)
    assert engine.models(SID).switches == ()


def test_a_run_in_flight_refuses_a_switch(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _engine(monkeypatch, repo, raw, [])
    engines.append(engine)
    episodes = engine._episodes  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(episodes, "busy", lambda: "run R-1")
    with pytest.raises(EngineBusy, match="run R-1 is in flight"):
        engine.switch_model(SID, ZEUS, OTHER)
    assert engine.models(SID).switches == ()


def test_a_pinned_session_refuses_a_switch_and_a_cap(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _engine(monkeypatch, repo, raw, [])
    engines.append(engine)
    engine.create_session("S-eval", pinned=True)
    with pytest.raises(ConfigError, match="pinned"):
        engine.switch_model("S-eval", ZEUS, OTHER)
    with pytest.raises(ConfigError, match="pinned"):
        engine.set_cap("S-eval", 9.0)
    assert engine.models("S-eval").switches == ()


def test_a_new_session_and_a_restart_read_the_wiring_from_the_records(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _engine(monkeypatch, repo, raw, [])
    engines.append(engine)
    engine.switch_model(SID, ZEUS, OTHER)
    assert engine.models("S-two").rungs == dict(engine.config.rungs), (
        "a new session's is the config's"
    )

    again, made = _engine(monkeypatch, repo, raw, [answer("after the restart")])
    engines.append(again)
    assert again.models(SID).rungs[ZEUS] == OTHER
    again.turn(SID, "hello").wait(timeout=WAIT_S)
    assert OTHER in made, "a restarted engine composes the session on its switched model"


def test_the_header_names_each_rungs_model_and_the_switch(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = _engine(monkeypatch, repo, raw, [])
    engines.append(engine)
    before = raw["rungs"][ZEUS]
    [line] = [h for h in engine.header(SID).lines if h.kind == "models"]
    assert f"{ZEUS} {before}" in line.text and "switched" not in line.text
    engine.switch_model(SID, ZEUS, OTHER)
    [line] = [h for h in engine.header(SID).lines if h.kind == "models"]
    assert f"{ZEUS} {OTHER} (switched from {before} at seq " in line.text


def test_a_turn_sent_while_a_switch_is_written_waits_and_runs_on_the_new_model(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Nothing starts between a switch's "nothing runs" and its line. A turn sent then
    starts after the line is written, so its first call is already on the new model."""
    engine, _ = _engine(monkeypatch, repo, raw, [answer("on the new model")])
    engines.append(engine)
    engine.open_session(SID)  # as the REPL does: the switch writes through its writer
    writing, release = threading.Event(), threading.Event()

    def held(session: Any, switch: Any) -> Any:
        writing.set()
        assert release.wait(WAIT_S)
        return record_switch(session, switch)

    monkeypatch.setattr("chickadee.engine.record_switch", held)
    switching = threading.Thread(target=engine.switch_model, args=(SID, ZEUS, OTHER))
    switching.start()
    assert writing.wait(WAIT_S)
    sent: list[Any] = []
    sending = threading.Thread(target=lambda: sent.append(engine.turn(SID, "hello")))
    sending.start()
    sending.join(0.3)
    assert sending.is_alive(), "the turn waited for the switch's line"
    release.set()
    switching.join(WAIT_S)
    sending.join(WAIT_S)
    sent[0].wait(timeout=WAIT_S)
    assert _charged(engine) == [OTHER]
