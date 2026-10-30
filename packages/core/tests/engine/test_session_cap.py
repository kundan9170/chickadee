"""The person sets the family's cap mid-session, or lifts it — `T-3.35`.

`set_cap` writes a fsynced `cap` line; the family's latest line is its cap from its next turn,
else the config's. A raise lets a session stopped at `BUDGET_CAP` go on with no restart, a cut
stops the next turn, and `None` stops nothing while every charge is still counted.

Every scripted call costs $1.10: 100 prompt tokens and 10 completion tokens, at $10,000 per
million each way.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo

import chickadee.context.store as store
import chickadee.engine as engine_module
import chickadee.zeus as zeus_package
from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable
from chickadee.budget.spend import Spend
from chickadee.client.request import ChatRequest
from chickadee.config import EngineConfig, PriceRow, SpendPolicy
from chickadee.context.lines import CapLine
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import PacketEngine, RungBudget, SessionBudget
from chickadee.errors import ConfigError, SessionError
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema import StopReason, ToolCall
from chickadee.schema.adapter import Completion, Usage
from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots
from chickadee.tools.spec import SURFACE

SID = "S-cap"
WAIT_S = 20.0
CALL_COST = 1.10


# -- the rung and the engine ------------------------------------------------------------------


def answer() -> Completion:
    return Completion(
        text="done",
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        latency_ms=1,
        stop_reason=StopReason.END_TURN,
    )


def note(cid: str) -> Completion:
    return Completion(
        text="",
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=(ToolCall(id=cid, name="note", arguments={"text": "x"}),),
    )


class CapRung:
    """The zeus rung, scripted; it answers `done` past its script, and counts its calls."""

    def __init__(self, script: list[Completion] | None = None) -> None:
        self._script = list(script or [])
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return self._script.pop(0) if self._script else answer()

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing to release."""


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))


@pytest.fixture(autouse=True)
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """One second per session line, so which `cap` line is the family's latest is decided by
    when it was written — the order D143 names — and never by a tie inside one second."""
    ticks = iter(range(10_000))

    def now() -> str:
        t = next(ticks)
        return f"2026-10-01T{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}Z"

    monkeypatch.setattr(store, "_now", now)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", commit=None)


@pytest.fixture
def where(repo: Path) -> Roots:
    return resolve_roots(repo)


def config(raw: dict[str, Any], cap: float) -> EngineConfig:
    raw["prices"] = {raw["rungs"]["high-param"]: {"price_in": 10_000.0, "price_out": 10_000.0}}
    raw["spend"] = {"hard_cost_usd": cap}
    return EngineConfig.with_defaults(raw)


Engines = list[PacketEngine]


@pytest.fixture
def engines() -> Iterator[Engines]:
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()


def engine_for(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    cfg: EngineConfig,
    engines: Engines,
    rung: CapRung,
) -> PacketEngine:
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: rung)
    engine = PacketEngine(repo, cfg)
    engines.append(engine)
    return engine


def kinds(where: Roots, sid: str = SID) -> list[str]:
    """`sid`'s own lines, read without its lock: an engine may hold it open."""
    return [str(e.kind) for e in SessionLog.read(sessions_root(where), sid).events()]


def caps(where: Roots, sid: str = SID) -> list[CapLine]:
    _log, state = ReplayCache(sessions_root(where)).get(sid)
    return [c for e, c in state.lines() if e.session_id == sid and isinstance(c, CapLine)]


def capped(engine: PacketEngine, sid: str, text: str) -> bool:
    """Whether the turn ended at `BUDGET_CAP`."""
    handle = engine.turn(sid, text)
    handle.wait(timeout=WAIT_S)
    errors = [str(e.payload) for e in handle.events() if e.kind == "error"]
    assert all("BUDGET_CAP" in e for e in errors), errors
    return bool(errors)


def spend_on(where: Roots, sid: str, cost: float = CALL_COST) -> None:
    """One charge written to `sid`'s log, as another process's engine would write it."""
    with Session.open(sessions_root(where), sid) as session:
        session.record_spend(
            "high-param-model",
            Usage(prompt_tokens=100, completion_tokens=10),
            cost,
            "high-param",
        )


def fork_of(where: Roots, parent: str, child: str) -> None:
    root = sessions_root(where)
    with Session.open(root, parent) as s:
        at = s.state.last_seq
    Session.fork(root, parent, at, "try another way", child, code="keep").close()


def created(where: Roots, sid: str = SID) -> None:
    Session.create(sessions_root(where), sid).close()


# -- set_cap ----------------------------------------------------------------------------------


def test_set_cap_writes_one_fsynced_cap_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())
    fsync_kinds: list[str] = []
    real_append = store.SessionStore.append

    def append(self: store.SessionStore, kind: Any, *args: Any, **kwargs: Any) -> Any:
        if kind in store.FSYNC_KINDS:
            fsync_kinds.append(kind)
        return real_append(self, kind, *args, **kwargs)

    monkeypatch.setattr(store.SessionStore, "append", append)

    engine.set_cap(SID, 7.5)

    assert kinds(where).count("cap") == 1
    assert fsync_kinds == ["cap"]
    assert [c.hard_cost_usd for c in caps(where)] == [7.5]


def test_a_raise_lets_a_session_stopped_at_the_cap_go_on_with_no_restart(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    """$2.20 spent against $2.00 stops the next turn; a raise to $10 lets the one after it run,
    on the same engine and the same open session."""
    rung = CapRung([note("n1")])
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    engine.create_session(SID)
    assert not capped(engine, SID, "spend")
    assert capped(engine, SID, "spend more")
    calls = len(rung.requests)

    budget = engine.set_cap(SID, 10.0)

    assert budget.cap_usd == 10.0
    assert not capped(engine, SID, "go on")
    assert len(rung.requests) == calls + 1


def test_a_cut_below_the_spend_stops_the_next_turn(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    rung = CapRung()
    engine = engine_for(monkeypatch, repo, config(raw, cap=100.0), engines, rung)
    engine.create_session(SID)
    assert not capped(engine, SID, "spend")  # $1.10

    engine.set_cap(SID, 1.0)

    assert capped(engine, SID, "spend more")
    assert len(rung.requests) == 1


def test_the_familys_latest_line_wins_across_a_fork_and_a_restart(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """The parent cuts to $1, the child later raises to $10: the raise is the family's cap, in
    the parent too, and still after the engine that wrote it is gone."""
    created(where)
    spend_on(where, SID, 2.2)
    fork_of(where, SID, "S-child")
    first = engine_for(monkeypatch, repo, config(raw, cap=100.0), engines, CapRung())
    first.set_cap(SID, 1.0)
    assert first.budget("S-child").cap_usd == 1.0
    first.set_cap("S-child", 10.0)
    first.close()

    rung = CapRung()
    second = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    budget = second.budget(SID)

    assert (budget.cap_usd, budget.cap_source) == (10.0, "session")
    assert not capped(second, SID, "in the parent")
    assert len(rung.requests) == 1


@pytest.mark.parametrize(
    "bad", [math.inf, -math.inf, math.nan, 0.0, 0, -1.0, True, "5", Decimal("5")]
)
def test_a_cap_that_no_config_could_hold_is_a_config_error_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
    bad: Any,
) -> None:
    created(where)
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())
    before = kinds(where)

    with pytest.raises(ConfigError):
        engine.set_cap(SID, bad)

    assert kinds(where) == before
    assert engine.budget(SID).cap_source == "config"


def test_set_cap_on_an_unknown_session_is_a_session_error(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())
    with pytest.raises(SessionError, match="S-nope"):
        engine.set_cap("S-nope", 3.0)
    with pytest.raises(SessionError, match="S-nope"):
        engine.budget("S-nope")


def test_within_one_log_the_later_line_wins_whatever_the_clock_says(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """The clock steps back between two `set_cap` calls on one session: the second line
    carries the earlier time, and is still the person's latest word."""
    created(where)
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())
    times = iter(["2026-10-01T12:00:05Z", "2026-10-01T12:00:01Z"])
    monkeypatch.setattr(store, "_now", lambda: next(times))

    engine.set_cap(SID, 10.0)
    engine.set_cap(SID, 3.0)

    assert engine.budget(SID).cap_usd == 3.0


def test_a_failed_build_lets_the_session_go_before_it_says_so(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """`set_cap` waits out a build in progress and, if it failed, opens the log itself — so a
    failed build must release the log before it wakes anyone, or `set_cap` finds it locked."""
    created(where)
    building: list[Any] = []
    ready_at_close: list[bool] = []

    def failing(self: PacketEngine, conversation: Any, stack: ExitStack) -> None:
        session = Session.open(sessions_root(where), conversation.session_id)
        stack.callback(session.close)
        building.append(conversation)
        raise RuntimeError("the loop could not be built")

    real_close = Session.close

    def close(self: Session) -> None:
        if building:
            # Whether the build's waiters had been woken when the session let go of the log.
            ready_at_close.append(building[0]._built.is_set())  # pyright: ignore[reportPrivateUsage]
        real_close(self)

    # Patched before the engine is built: its registry holds the build it was given.
    monkeypatch.setattr(PacketEngine, "_compose", failing)
    monkeypatch.setattr(Session, "close", close)
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())

    # The turn returns at once; the build's failure reaches its `wait`.
    with pytest.raises(SessionError, match="could not be built"):
        engine.turn(SID, "hi").wait(timeout=WAIT_S)

    assert ready_at_close == [False]
    assert engine.set_cap(SID, 4.0).cap_usd == 4.0


# -- budget -----------------------------------------------------------------------------------


def test_budget_names_the_source_and_the_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    spend_on(where, SID)
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())

    assert engine.budget(SID) == SessionBudget(
        spent_usd=pytest.approx(CALL_COST),  # pyright: ignore[reportArgumentType]
        cap_usd=2.0,
        cap_source="config",
        cap_seq=None,
        # The session's one charge is Zeus's, and Zeus's rung has no cap of its own.
        rungs={"high-param": RungBudget(pytest.approx(CALL_COST), None, "none", None)},  # pyright: ignore[reportArgumentType]
        # The same charge by its role, and the config's warning levels.
        roles={"zeus": pytest.approx(CALL_COST)},  # pyright: ignore[reportArgumentType]
        warning_fractions=(0.8,),
    )

    set_ = engine.set_cap(SID, 3.0)
    (line,) = caps(where)

    assert set_ == engine.budget(SID)
    assert (set_.cap_usd, set_.cap_source, set_.cap_seq) == (3.0, "session", line.seq)
    assert set_.spent_usd == pytest.approx(CALL_COST)


def test_budget_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, CapRung())
    before = kinds(where)
    engine.budget(SID)
    engine.budget(SID)
    assert kinds(where) == before


# -- only the person --------------------------------------------------------------------------


def test_no_surface_tool_reaches_set_cap() -> None:
    """A cap is the person's. No Zeus tool is named for it, and nothing in Zeus's package
    calls the method or writes the line."""
    names = {spec.name for spec in SURFACE}
    assert not {n for n in names if "cap" in n or "budget" in n}, names

    package = Path(inspect.getfile(zeus_package)).parent
    for source in package.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "set_cap" not in text, source
        assert "record_cap" not in text, source


# -- D144: a session without a cap ------------------------------------------------------------


def test_a_lifted_cap_lets_the_family_run_past_the_old_one(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    rung = CapRung([note("n1")])
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    engine.create_session(SID)
    assert not capped(engine, SID, "spend")  # $2.20
    assert capped(engine, SID, "spend more")

    lifted = engine.set_cap(SID, None)

    assert [c.hard_cost_usd for c in caps(where)] == [None]
    assert (lifted.cap_usd, lifted.cap_source) == (None, "session")
    assert not capped(engine, SID, "go on")
    assert not capped(engine, SID, "and on")
    assert len(rung.requests) == 4


def test_every_charge_is_still_counted_while_uncapped(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    rung = CapRung([note("n1")])
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    engine.set_cap(SID, None)

    assert not capped(engine, SID, "spend")
    assert not capped(engine, SID, "spend more")

    assert kinds(where).count("spend") == len(rung.requests) == 3
    budget = engine.budget(SID)
    assert budget.cap_usd is None
    assert budget.spent_usd == pytest.approx(3 * CALL_COST)


def test_an_unrelated_session_keeps_the_configs_cap(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    created(where, "S-other")
    spend_on(where, SID, 2.2)
    spend_on(where, "S-other", 2.2)
    rung = CapRung()
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)

    engine.set_cap(SID, None)

    assert engine.budget("S-other").cap_usd == 2.0
    assert capped(engine, "S-other", "in the other")
    assert rung.requests == []
    assert not capped(engine, SID, "in the uncapped one")


def test_a_fork_of_an_uncapped_session_is_uncapped_until_a_member_caps_the_family(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    spend_on(where, SID, 2.2)
    rung = CapRung()
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    engine.set_cap(SID, None)
    fork_of(where, SID, "S-child")

    assert engine.budget("S-child").cap_usd is None
    assert not capped(engine, "S-child", "in the child")

    engine.set_cap("S-child", 3.0)

    assert engine.budget(SID).cap_usd == 3.0
    assert capped(engine, SID, "back in the parent")  # $3.30 of the family's $3.00
    assert len(rung.requests) == 1


# -- the ledger -------------------------------------------------------------------------------


NOTHING = Spend("session", 0, 0, 0, 0.0, 0)


def ledger_for(session: Session, cap: float) -> CostLedger:
    cost = PriceTable.from_config({"m": PriceRow(price_in=10_000.0, price_out=10_000.0)})
    caps = BudgetCaps.from_config(SpendPolicy(hard_cost_usd=cap))
    return CostLedger(cost, caps, record=session.record_spend, rungs=DEFAULT_RUNGS)


def test_no_warning_crossing_fires_while_uncapped(where: Roots) -> None:
    """A warning is a fraction of the cap; with none there is nothing to be a fraction of."""
    usage = Usage(prompt_tokens=100, completion_tokens=10)
    with Session.create(sessions_root(where), SID) as session:
        ledger = ledger_for(session, cap=1.25)
        lifted = replace(ledger.caps, hard_cost_cap_usd=None)
        ledger.restore(lifted, NOTHING)

        for _ in range(3):
            ledger.for_rung("high-param").charge_usage("", usage, "m")

        assert ledger.crossings() == ()
        assert ledger.for_rung("high-param").may_start()
        assert ledger.spend("session").cost_usd == pytest.approx(3 * CALL_COST)


def test_a_cap_put_back_fires_its_crossings_once(where: Roots) -> None:
    """Lifted, then capped again at $1.25 with $1.10 spent: the warning already passed is
    marked fired without being raised (as on any resume), and the hard cap still stops."""
    usage = Usage(prompt_tokens=100, completion_tokens=10)
    with Session.create(sessions_root(where), SID) as session:
        ledger = ledger_for(session, cap=1.25)
        capped_at = ledger.caps
        ledger.restore(replace(capped_at, hard_cost_cap_usd=None), NOTHING)
        ledger.for_rung("high-param").charge_usage("", usage, "m")

        ledger.restore(capped_at, ledger.spend("session"))

        assert ledger.crossings() == ()
        assert ledger.for_rung("high-param").may_start()
        ledger.for_rung("high-param").charge_usage("", usage, "m")
        assert [c.kind for c in ledger.crossings()] == ["hard_cap"]
        assert not ledger.for_rung("high-param").may_start()
