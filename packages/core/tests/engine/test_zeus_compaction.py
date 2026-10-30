"""Zeus compacts: the engine runs `compact` before every
request, and once more after a provider refuses one as too long.

Driven through `PacketEngine.turn()`, so what is tested is the engine's own wiring: Zeus's
endpoint, its price row and cache mode, the residue model on the `residue` role's rung. Two
scripted clients stand behind the rungs, told apart by model id. The test models' window is
65,536 tokens with a 16,384-token reserve, so a few turns of ~8,000 tokens each cross the
compact trigger (0.80 of the window).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import (
    SID,
    WAIT_S,
    ScriptedRung,
    Step,
    answer,
    config_for,
    state,
    with_sessions,
)

import chickadee.engine as engine_module
from chickadee.client.errors import ContextOverflow
from chickadee.client.request import ChatRequest
from chickadee.config import RungEndpoint
from chickadee.context.compaction.residue import ModelResidue, ProseTurn, ResidueRequest
from chickadee.context.compaction.session import SessionCompaction
from chickadee.context.compaction.trigger import TriggerKind
from chickadee.context.lines import CutLine, SpendLine
from chickadee.context.request import assemble
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.isolation.none import Passthrough
from chickadee.schema.adapter import CacheMode, Completion, RetryParams, StopReason, Usage
from chickadee.state.roots import roots as resolve_roots
from chickadee.zeus.loop import Event

RESIDUE_ID = "residue-m"
BIG = "word " * 6_300
"""~7,200 tokens at P-2's rate: six of these and their answers cross 0.80 of the window, four
leave room under it. Was 7,000 words until Zeus's prefix carried the design and brief templates,
and 6,500 until it carried the `worktree` tool, each of which took the
margin four turns had."""


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


class Residue:
    """The residue rung: answers each request with `items`, bulleted as a model might."""

    def __init__(self, items: Sequence[str] = ("the person wants tabs, not spaces",)) -> None:
        self.items = items
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(
            text="\n".join(f"- {i}" for i in self.items),
            usage=Usage(prompt_tokens=500, completion_tokens=20),
            latency_ms=1,
            stop_reason=StopReason.END_TURN,
        )

    def close(self) -> None:
        """Nothing to release."""


def _config(raw: dict[str, Any], *, priced_residue: bool) -> Any:
    """The residue moved by `[roles]` onto `mid-param` (its default is Zeus's own rung),
    so its model, and its client, are its own."""
    raw["roles"] = {"residue": "mid-param"}
    raw["models"][raw["rungs"]["mid-param"]]["model_id"] = RESIDUE_ID
    config = config_for(raw)
    if priced_residue:
        raw["prices"][raw["rungs"]["mid-param"]] = {"price_in": 0.1, "price_out": 0.2}
        config = config_for(raw, priced=False)
    return config


def _start(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    script: Sequence[Step] = (),
    *,
    priced_residue: bool = True,
) -> tuple[PacketEngine, ScriptedRung, Residue]:
    zeus, residue = ScriptedRung(script), Residue()

    def client_for(endpoint: RungEndpoint) -> Any:
        return residue if endpoint.model_id == RESIDUE_ID else zeus

    monkeypatch.setattr(engine_module, "client_for", client_for)
    config = _config(raw, priced_residue=priced_residue)
    trust = config.trust(resolve_roots(repo).repo_local)
    engine = PacketEngine(repo, config.engine, trust=trust)
    return engine, zeus, residue


@pytest.fixture
def engines() -> Iterator[list[PacketEngine]]:
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()


def _turn(engine: PacketEngine, text: str) -> list[Event]:
    handle = engine.turn(SID, text)
    handle.wait(timeout=WAIT_S)
    return list(handle.events())


def _errors(events: Sequence[Event]) -> list[str]:
    return [str(e.payload) for e in events if e.kind == "error"]


def _cuts(engine: PacketEngine) -> tuple[CutLine, ...]:
    return state(engine).cuts()


# -- a session longer than its window ------------------------------------------------------


def test_a_session_longer_than_its_window_keeps_running_and_its_cuts_replay(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, zeus, residue = _start(monkeypatch, repo, raw)
    engines.append(engine)
    turns = 12
    for n in range(turns):
        events = _turn(engine, f"turn {n}: {BIG}")
        assert _errors(events) == [], f"turn {n}"
    assert len(zeus.requests) == turns, "one request per turn: every turn was answered"
    cuts = _cuts(engine)
    assert cuts, "the session crossed its window, so it was cut"
    # The residue model ran on its own rung, its answer kept, its bullets stripped.
    assert residue.requests
    assert any("the person wants tabs, not spaces" in c.block for c in cuts)
    assert all("- - the person wants" not in c.block for c in cuts), "one bullet, the block's"
    window = raw["models"][raw["rungs"]["high-param"]]["context_window"]
    reserve = raw["models"][raw["rungs"]["high-param"]]["max_output_tokens"]
    held = assemble(state(engine), _blob(engine)).tokens
    assert held < window - reserve, "what is held now still leaves the answer its room"
    # Replay: the log alone rebuilds what the model reads ("the log is the context").
    live = assemble(state(engine), _blob(engine)).messages
    engine.close()
    root = sessions_root(resolve_roots(repo))
    reopened = Session.open(root, SID)
    try:
        assert assemble(reopened.state, reopened.blob).messages == live
        assert len(reopened.state.cuts()) == len(cuts)
    finally:
        reopened.close()


def test_the_residue_models_calls_are_charged_to_its_rung(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _, residue = _start(monkeypatch, repo, raw)
    engines.append(engine)
    for n in range(12):
        _turn(engine, f"turn {n}: {BIG}")
    assert residue.requests
    spends = [
        r
        for _, r in state(engine).lines()
        if isinstance(r, SpendLine) and r.rung == "mid-param"
    ]
    assert len(spends) == len(residue.requests)
    assert all(s.tokens_in == 500 for s in spends)


def test_an_unpriced_residue_model_turns_layer_3_off_and_every_cut_says_why(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Refusing the session would make a price for the cheap rung a precondition of talking
    to Zeus; charging at zero is what 08-budget §6 refuses. Layer 3 is off, on the record."""
    engine, _, residue = _start(monkeypatch, repo, raw, priced_residue=False)
    engines.append(engine)
    for n in range(12):
        assert _errors(_turn(engine, f"turn {n}: {BIG}")) == []
    cuts = _cuts(engine)
    assert cuts and not residue.requests
    failure = str(cuts[0].facts["residue"]["failure"])  # type: ignore[index]
    assert "has no price row" in failure and "layer 3 is off" in failure


# -- AD-13 step 3: a provider's overflow ---------------------------------------------------


def _overflow(_: ChatRequest) -> Completion:
    raise ContextOverflow("prompt is too long: 70000 tokens > 65536 maximum", 400, 1)


HISTORY = "word " * 5_500
"""A history turn: smaller than `BIG` since Zeus's prefix grew by `spawn_agent` and
`find_skill`, so four of them stay under the trigger as this setup needs."""


def _history(engine: PacketEngine, turns: int = 4) -> None:
    """Under the trigger, but with enough for a cut to take."""
    for n in range(turns):
        assert _errors(_turn(engine, f"turn {n}: {HISTORY}")) == []


def test_a_provider_overflow_is_compacted_once_and_the_turn_sent_again(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script: list[Step] = [answer()] * 4 + [_overflow, answer("after the cut")]
    engine, zeus, _ = _start(monkeypatch, repo, raw, script)
    engines.append(engine)
    _history(engine)
    assert not _cuts(engine), "under the trigger, nothing was cut"
    events = _turn(engine, "one more")
    assert _errors(events) == []
    assert [str(e.payload) for e in events if e.kind == "answer"] == ["after the cut"]
    (cut,) = _cuts(engine)
    assert (cut.facts["trigger"], cut.facts["basis"]) == ("must_fit", "provider_overflow")
    refused, retried = zeus.requests[-2], zeus.requests[-1]
    assert len(retried.messages) < len(refused.messages)


def test_a_second_overflow_ends_the_turn_and_is_never_sent_a_third_time(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script: list[Step] = [answer()] * 4 + [_overflow, _overflow]
    engine, zeus, _ = _start(monkeypatch, repo, raw, script)
    engines.append(engine)
    _history(engine)
    (error,) = _errors(_turn(engine, "one more"))
    assert error.startswith("ContextOverflow:") and "after 1 compact-and-retry" in error
    assert len(zeus.requests) == 6


def test_an_overflow_with_nothing_to_cut_ends_the_turn_without_a_resend(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, zeus, _ = _start(monkeypatch, repo, raw, [_overflow])
    engines.append(engine)
    (error,) = _errors(_turn(engine, "hello"))
    assert error.startswith("ContextOverflow:")
    assert len(zeus.requests) == 1


# -- a turn too large after a full cut -----------------------------------------------------


def test_a_turn_too_large_for_the_window_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """§5.12 step 12b: a fact about the request, never a reason to shorten what the person
    typed (invariant 19)."""
    engine, zeus, _ = _start(monkeypatch, repo, raw)
    engines.append(engine)
    (error,) = _errors(_turn(engine, "word " * 60_000))
    assert error.startswith("TURN_TOO_LARGE:")
    assert zeus.requests == []


# -- the driver's parts --------------------------------------------------------------------


ENDPOINT = RungEndpoint(
    base_url="http://x/v1", model_id="m", context_window=65_536, max_output_tokens=16_384
)


def test_cache_observations_and_the_gap_make_a_free_rebuild(tmp_path: Path) -> None:
    """Two observed responses are a transition, and a gap past the TTL says the cache
    is rebuilt whatever is done, so a cut owes no premium."""
    from chickadee.config import PriceRow

    now = [0.0]
    session = Session.create(tmp_path / "sessions", "S-cache")
    try:
        session.record_user("hello")
        driver = SessionCompaction(
            ENDPOINT,
            PriceRow(price_in=3.0, price_out=15.0, price_cached_in=0.3, cache_ttl_s=300),
            CacheMode.AUTOMATIC,
            resolver=lambda: HandleResolver(tmp_path, session.blob),
            clock=lambda: now[0],
        )
        assert driver.before(session, 100).trigger is None, "nothing observed: no signal"
        driver.observed(Usage(prompt_tokens=100, completion_tokens=5, cached_prompt_tokens=0))
        driver.observed(Usage(prompt_tokens=100, completion_tokens=5, cached_prompt_tokens=90))
        now[0] = 301.0
        done = driver.before(session, 100)
        assert done.trigger is not None and done.trigger.kind is TriggerKind.FREE_REBUILD
    finally:
        session.close()


def test_a_residue_model_without_a_charge_is_refused(tmp_path: Path) -> None:
    from chickadee.config import PriceRow
    from chickadee.errors import HarnessBug

    with pytest.raises(HarnessBug, match="charged"):
        SessionCompaction(
            ENDPOINT,
            PriceRow(price_in=1.0, price_out=2.0),
            CacheMode.NONE,
            resolver=lambda: HandleResolver(tmp_path, lambda sha: b""),
            residue=ModelResidue(Residue(), RetryParams(temperature=0.0, max_tokens=64), "x"),
        )


def test_the_residue_model_renders_its_template_and_holds_its_spend() -> None:
    residue = Residue(items=("a", "", "b"))
    template = "keep {max_items}; skip {exclusion}; {first_seq}-{last_seq}\n{turns}"
    model = ModelResidue(residue, RetryParams(temperature=0.0, max_tokens=64), template)
    request = ResidueRequest(
        "residue", 3, 4, (ProseTurn(3, "user", "use tabs"),), ("already known",), 12
    )
    assert [i for i in model(request) if i] == ["a", "b"]
    sent = residue.requests[0].messages[0].content
    assert sent == "keep 12; skip already known; 3-4\n[3 user] use tabs"
    (usage,) = model.drain()
    assert usage.prompt_tokens == 500
    assert model.drain() == (), "each answer's spend is drained once"


def _blob(engine: PacketEngine) -> Any:
    conversation = engine._conversations.get(SID)  # pyright: ignore[reportPrivateUsage]
    assert conversation is not None
    opened = conversation.opened()
    assert opened is not None
    return opened.session.blob


def test_a_residue_call_in_a_cut_that_still_does_not_fit_is_charged_at_once(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A MUST_FIT cut asks the residue model, then finds the turn still too large and raises.
    The call happened, so it is charged then, not left for a later cut to find."""
    engine, zeus, residue = _start(monkeypatch, repo, raw)
    engines.append(engine)
    _history(engine)
    (error,) = _errors(_turn(engine, "word " * 60_000))
    assert error.startswith("TURN_TOO_LARGE:")
    assert residue.requests, "the cut asked the residue model before it found no room"
    spends = [
        r
        for _, r in state(engine).lines()
        if isinstance(r, SpendLine) and r.rung == "mid-param"
    ]
    assert len(spends) == len(residue.requests)
    assert len(zeus.requests) == 4
