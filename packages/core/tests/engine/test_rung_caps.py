"""Each rung may carry its own cap, under the family's total: `T-3.39`, the session side.

A call on a rung starts only if the family's total and that rung's own cap both allow it. The
engine config may cap a rung (`[spend.rungs]`), and the person may set or lift one rung's cap
for a family with `set_cap(..., rung=)`. Every charge names its rung.

Every scripted call costs $1.10: 100 prompt tokens and 10 completion tokens, at $10,000 per
million each way.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from spend_reads import lines_of

import chickadee.context.store as store
import chickadee.engine as engine_module
from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable
from chickadee.budget.spend import Spend
from chickadee.client.request import ChatRequest
from chickadee.config import EngineConfig, PriceRow, SpendPolicy
from chickadee.context.lines import CapLine, EpisodeLine, SpendLine
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import PacketEngine, RungBudget, SessionBudget
from chickadee.errors import ConfigError, HarnessBug
from chickadee.journal.writer import Journal
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema import EventKind, StopReason, ToolCall
from chickadee.schema.adapter import Completion, Usage
from chickadee.schema.journal import ModelCallP, RunStartP
from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots
from chickadee.tools.spec import SURFACE

SID = "S-rung"
WAIT_S = 20.0
CALL_COST = 1.10
SHA = "a" * 64
USAGE = Usage(prompt_tokens=100, completion_tokens=10)


# -- the rung and the engine ------------------------------------------------------------------


def answer() -> Completion:
    return Completion(text="done", usage=USAGE, latency_ms=1, stop_reason=StopReason.END_TURN)


def note(cid: str) -> Completion:
    return Completion(
        text="",
        usage=USAGE,
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=(ToolCall(id=cid, name="note", arguments={"text": "x"}),),
    )


class RungScript:
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
    """One second per session line, so a family's latest `cap` line is decided by when it
    was written, never by a tie inside one second."""
    ticks = iter(range(10_000))

    def now() -> str:
        t = next(ticks)
        return f"2026-10-02T{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}Z"

    monkeypatch.setattr(store, "_now", now)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", commit=None)


@pytest.fixture
def where(repo: Path) -> Roots:
    return resolve_roots(repo)


def config(
    raw: dict[str, Any], total: float, rungs: Mapping[str, float] | None = None
) -> EngineConfig:
    """`rungs` is `[spend.rungs]`, keyed by declared rung: a hyphenated name is no
    keyword argument."""
    raw["prices"] = {raw["rungs"]["high-param"]: {"price_in": 10_000.0, "price_out": 10_000.0}}
    raw["spend"] = {"hard_cost_usd": total, "rungs": dict(rungs or {})}
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
    rung: RungScript,
) -> PacketEngine:
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: rung)
    engine = PacketEngine(repo, cfg)
    engines.append(engine)
    return engine


def capped(engine: PacketEngine, sid: str, text: str) -> bool:
    """Whether the turn ended at `BUDGET_CAP`."""
    handle = engine.turn(sid, text)
    handle.wait(timeout=WAIT_S)
    errors = [str(e.payload) for e in handle.events() if e.kind == "error"]
    assert all("BUDGET_CAP" in e for e in errors), errors
    return bool(errors)


def own(where: Roots, sid: str = SID) -> list[Any]:
    """`sid`'s own records, read without its lock: an engine may hold it open."""
    _log, state = ReplayCache(sessions_root(where)).get(sid)
    return [r for e, r in state.lines() if e.session_id == sid]


def kinds(where: Roots, sid: str = SID) -> list[str]:
    return [str(e.kind) for e in SessionLog.read(sessions_root(where), sid).events()]


def created(where: Roots, sid: str = SID) -> None:
    Session.create(sessions_root(where), sid).close()


def fork_of(where: Roots, parent: str, child: str) -> None:
    root = sessions_root(where)
    with Session.open(root, parent) as s:
        at = s.state.last_seq
    Session.fork(root, parent, at, "try another way", child, code="keep").close()


def charge(where: Roots, sid: str, cost: float, rung: str) -> None:
    """One charge on `rung`, as another process's engine would write it."""
    with Session.open(sessions_root(where), sid) as session:
        session.record_spend(f"{rung}-model", USAGE, cost, rung)


# -- a rung's cap and the total -----------------------------------------------------------


def test_a_rungs_cap_stops_it_while_the_total_has_room(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    rung = RungScript([note("n1")])
    engine = engine_for(
        monkeypatch, repo, config(raw, 100.0, {"high-param": 2.0}), engines, rung
    )
    engine.create_session(SID)

    assert not capped(engine, SID, "spend")  # $2.20 of zeus's $2.00, $100 total
    assert capped(engine, SID, "spend more")
    assert len(rung.requests) == 2


def test_the_total_stops_a_rung_that_still_has_room(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    rung = RungScript([note("n1")])
    engine = engine_for(
        monkeypatch, repo, config(raw, 2.0, {"high-param": 100.0}), engines, rung
    )
    engine.create_session(SID)

    assert not capped(engine, SID, "spend")
    assert capped(engine, SID, "spend more")
    assert len(rung.requests) == 2


# -- the ledger's views ---------------------------------------------------------------------


NOTHING = Spend("session", 0, 0, 0, 0.0, 0)


def ledger_for(session: Session, total: float) -> CostLedger:
    cost = PriceTable.from_config({"m": PriceRow(price_in=10_000.0, price_out=10_000.0)})
    caps = BudgetCaps.from_config(SpendPolicy(hard_cost_usd=total))
    return CostLedger(cost, caps, record=session.record_spend, rungs=DEFAULT_RUNGS)


def test_a_capped_rung_leaves_the_others_running(where: Roots) -> None:
    """`locate`'s rung (`mid-param`) at its cap refuses, while Zeus's rung still starts
    calls."""
    with Session.create(sessions_root(where), SID) as session:
        ledger = ledger_for(session, total=100.0)
        ledger.restore(ledger.caps, NOTHING, {"mid-param": 1.0}, {})
        scout = ledger.for_rung("mid-param")

        assert scout.may_start()
        scout.charge_usage("", USAGE, "m")

        assert not scout.may_start()
        assert ledger.for_rung("high-param").may_start(), "zeus's rung is not locate's"
        assert ledger.for_rung("low-param").may_start()


def test_every_charge_names_its_rung(where: Roots) -> None:
    with Session.create(sessions_root(where), SID) as session:
        ledger = ledger_for(session, total=100.0)
        ledger.for_rung("high-param").charge_usage("", USAGE, "m")
        ledger.for_rung("mid-param").charge_usage("", USAGE, "m")

        assert [c.rung for c in lines_of(session.state, SpendLine)] == [
            "high-param",
            "mid-param",
        ]


def test_a_turns_charges_are_on_zeuss_rung(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    engine = engine_for(monkeypatch, repo, config(raw, 100.0), engines, RungScript([note("n")]))
    engine.create_session(SID)
    assert not capped(engine, SID, "spend")

    lines = [r for r in own(where) if isinstance(r, SpendLine)]
    assert len(lines) == 2
    assert {line.rung for line in lines} == {"high-param"}


def test_a_ledger_is_handed_only_to_a_declared_rung(where: Roots) -> None:
    with (
        Session.create(sessions_root(where), SID) as session,
        pytest.raises(HarnessBug),
    ):
        ledger_for(session, total=1.0).for_rung("mid")


# -- set_cap(rung=) -----------------------------------------------------------------------


def test_set_cap_on_a_rung_writes_one_cap_line_naming_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    created(where)
    engine = engine_for(monkeypatch, repo, config(raw, 10.0), engines, RungScript())

    engine.set_cap(SID, 3.0, rung="low-param")

    assert kinds(where).count("cap") == 1
    (line,) = [r for r in own(where) if isinstance(r, CapLine)]
    assert (line.rung, line.hard_cost_usd) == ("low-param", 3.0)
    assert "cap" in store.FSYNC_KINDS


def test_the_latest_line_per_rung_wins_across_a_fork_and_a_restart(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """The parent caps the worker's rung at $3 and the total at $5; the child later caps it at
    $1. The child's line is the family's cap on that rung; the total and the other rungs are as
    they were, and stay so after the engine that wrote them is gone."""
    created(where)
    first = engine_for(
        monkeypatch, repo, config(raw, 10.0, {"high-param": 4.0}), engines, RungScript()
    )
    first.set_cap(SID, 3.0, rung="low-param")
    first.set_cap(SID, 5.0)
    fork_of(where, SID, "S-child")
    first.set_cap("S-child", 1.0, rung="low-param")
    first.close()

    second = engine_for(
        monkeypatch, repo, config(raw, 10.0, {"high-param": 4.0}), engines, RungScript()
    )
    budget = second.budget(SID)

    assert (budget.cap_usd, budget.cap_source) == (5.0, "session")
    assert budget.rungs["low-param"].cap_usd == 1.0
    assert budget.rungs["low-param"].cap_source == "session"
    assert budget.rungs["high-param"] == RungBudget(0.0, 4.0, "config", None)


def test_lifting_one_rung_leaves_the_total_in_force(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    """Zeus's rung cap ($1) stops the session at $1.10. Lifted, the next turn runs to $2.20,
    and the $2 total stops the one after it."""
    rung = RungScript()
    engine = engine_for(monkeypatch, repo, config(raw, 2.0, {"high-param": 1.0}), engines, rung)
    engine.create_session(SID)
    assert not capped(engine, SID, "spend")
    assert capped(engine, SID, "spend more")

    lifted = engine.set_cap(SID, None, rung="high-param")

    assert lifted.rungs["high-param"].cap_usd is None
    assert lifted.rungs["high-param"].cap_source == "session"
    assert lifted.cap_usd == 2.0, "the total is untouched"
    assert not capped(engine, SID, "go on")
    assert capped(engine, SID, "and on")
    assert len(rung.requests) == 2


@pytest.mark.parametrize(
    ("amount", "rung"),
    [
        (3.0, "nope"),  # well formed, and declared by no [rungs] line
        (3.0, "mid"),  # former, with no successor
        (3.0, "coder"),  # former: its successor is low-param
        (None, ""),
        (math.inf, "low-param"),
        (math.nan, "low-param"),
        (0.0, "low-param"),
        (-1.0, "low-param"),
    ],
)
def test_a_cap_no_config_could_hold_is_a_config_error_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
    amount: float | None,
    rung: str,
) -> None:
    created(where)
    engine = engine_for(monkeypatch, repo, config(raw, 2.0), engines, RungScript())
    before = kinds(where)

    with pytest.raises(ConfigError):
        engine.set_cap(SID, amount, rung=rung)

    assert kinds(where) == before


# -- budget() -------------------------------------------------------------------------------


def run_journal(where: Roots, run_id: str, *done: ModelCallP) -> None:
    start = RunStartP(
        goal="g",
        work_sha=SHA,
        config_sha=SHA,
        repo_head=SHA,
        engine_version="0",
        schema_version=2,
        isolation="worktree",
        hard_cost_cap_usd=5.0,
    )
    with Journal.create(where, run_id, {}) as j:
        j.append(EventKind.RUN_START, start)
        for d in done:
            j.append(EventKind.MODEL_CALL, d, task_id="T1", round=1)


def worker_call(cost: float, rung: str) -> ModelCallP:
    """One of the worker's calls, as its run's journal records it: the rung that served it."""
    return ModelCallP(
        role="worker",
        request="task",
        tokens_in=1,
        tokens_out=1,
        cost_usd=cost,
        stop_reason="end_turn",
        rung=rung,
    )


def open_episode(where: Roots, sid: str, run_id: str) -> None:
    with Session.open(sessions_root(where), sid) as session:
        session.record_user(f"do {run_id}")
        call = ToolCall(id=f"c-{run_id}", name="coder", arguments={"title": "t", "brief": "t"})
        turn = session.record_assistant("", [call])
        session.record_episode(EpisodeLine(run_id, "planned", None, SHA, (), turn.seq))
        session.record_tool(f"c-{run_id}", f"COMPLETED · run {run_id}", run_id=run_id)


def test_budget_breaks_the_spend_down_by_rung_and_names_each_source(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """Zeus's rung spent $1.10 under the config's $4 rung cap; locate's spent $0.50 with no
    rung cap; an episode's worker spent $2 over two calls, each naming its rung; a rung of the
    person's own is capped, then lifted. The total is everything."""
    created(where)
    charge(where, SID, CALL_COST, "high-param")
    charge(where, SID, 0.5, "mid-param")
    run = "2026-10-02-0000000c"
    open_episode(where, SID, run)
    run_journal(where, run, worker_call(1.5, "low-param"), worker_call(0.5, "low-param"))
    raw["rungs"]["nemotron"] = raw["rungs"]["low-param"]
    engine = engine_for(
        monkeypatch, repo, config(raw, 20.0, {"high-param": 4.0}), engines, RungScript()
    )
    engine.set_cap(SID, 2.0, rung="nemotron")
    lifted = engine.set_cap(SID, None, rung="nemotron")
    (_, lift) = [r for r in own(where) if isinstance(r, CapLine)]

    rungs = lifted.rungs
    assert rungs["high-param"] == RungBudget(pytest.approx(CALL_COST), 4.0, "config", None)  # pyright: ignore[reportArgumentType]
    assert rungs["mid-param"] == RungBudget(0.5, None, "none", None)
    assert rungs["low-param"] == RungBudget(2.0, None, "none", None)
    assert rungs["nemotron"] == RungBudget(0.0, None, "session", lift.seq)
    assert set(rungs) == {"high-param", "mid-param", "low-param", "nemotron"}
    assert lifted.spent_usd == pytest.approx(CALL_COST + 0.5 + 2.0)


def test_a_siblings_rung_spend_is_the_familys(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """D142 per rung: a child's charge on Zeus's rung counts against the parent's cap on it."""
    created(where)
    fork_of(where, SID, "S-child")
    charge(where, "S-child", 2.2, "high-param")
    rung = RungScript()
    parent = engine_for(
        monkeypatch, repo, config(raw, 100.0, {"high-param": 2.0}), engines, rung
    )

    assert capped(parent, SID, "in the parent")
    assert rung.requests == []
    assert parent.budget(SID).rungs["high-param"].spent_usd == pytest.approx(2.2)


# -- only the person ------------------------------------------------------------------------


def test_no_surface_tool_sets_a_rungs_cap() -> None:
    """A cap is the person's, a rung's as much as the total."""
    names = {spec.name for spec in SURFACE}
    assert not {n for n in names if "cap" in n or "budget" in n or "rung" in n}, names


# -- the family, forked before anything was said --------------------------------------------


def test_a_fork_taken_at_seq_0_is_its_parents_family_and_so_is_its_own_fork(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """D142, found by this task: a fork of an empty session reads none of its parent's lines,
    so its first line is its own `fork` line, and `_family_root` took it for a family of
    one. A grandchild forked later reads that line first. All three are one family."""
    created(where)
    fork_of(where, SID, "S-child")  # at seq 0
    charge(where, "S-child", 1.0, "high-param")
    fork_of(where, "S-child", "S-grandchild")  # after the child's fork line and charge
    charge(where, "S-grandchild", 1.5, "high-param")
    engine = engine_for(monkeypatch, repo, config(raw, 2.0), engines, RungScript())

    assert engine.budget(SID).spent_usd == pytest.approx(2.5)
    assert engine.budget("S-grandchild").spent_usd == pytest.approx(2.5)
    assert capped(engine, SID, "in the parent")


def test_a_budget_with_a_breakdown_still_hashes_and_equal_ones_hash_equal() -> None:
    """The review of #261: `rungs` is a dict, which cannot be hashed. It stays out of the
    hash and in equality, so `SessionBudget` hashes as it did before the breakdown."""
    share = {"high-param": RungBudget(1.0, 2.0, "config", None)}
    one = SessionBudget(1.0, 5.0, "config", None, share)
    other = SessionBudget(1.0, 5.0, "config", None, dict(share))

    assert one == other
    assert hash(one) == hash(other)
    assert one != SessionBudget(1.0, 5.0, "config", None, {})


def test_spend_on_a_rung_the_config_no_longer_declares_stays_in_the_breakdown(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    where: Roots,
    raw: dict[str, Any],
    engines: Engines,
) -> None:
    """The second review of #261: a rung's lines still replay after the config stops
    declaring it, and its spend is in the total, so the breakdown names it as well,
    with no cap. The breakdown then accounts for every dollar the total counts."""
    created(where)
    charge(where, SID, 0.4, "nemotron")
    charge(where, SID, CALL_COST, "high-param")
    engine = engine_for(monkeypatch, repo, config(raw, 20.0), engines, RungScript())

    budget = engine.budget(SID)

    assert list(budget.rungs) == ["high-param", "nemotron"], "the declared rungs first"
    assert budget.rungs["nemotron"] == RungBudget(0.4, None, "none", None)
    shares = sum(share.spent_usd for share in budget.rungs.values())
    assert shares == pytest.approx(budget.spent_usd)
