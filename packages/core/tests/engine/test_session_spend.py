"""A session's spend survives a restart.

Zeus's charges are `spend` lines on the session log. A session's spend is those lines, its
lineage's included, plus each episode's run journal; the engine restores its ledger from that
projection whenever it builds one. So a cap crossed before a restart is still crossed after
it, and a raised cap lets the session go on.

Every scripted call costs $1.10: 100 prompt tokens and 10 completion tokens, at $10,000 per
million each way.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from spend_reads import family_spent, lines_of

import chickadee.engine as engine_module
from chickadee.budget.family import ledger_inputs
from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable
from chickadee.budget.spend import Spend
from chickadee.client.request import ChatRequest
from chickadee.config import EngineConfig, PriceRow, SpendPolicy
from chickadee.context.lines import EpisodeLine, SpendLine
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import PacketEngine
from chickadee.errors import HarnessBug
from chickadee.journal.paths import journal_path
from chickadee.journal.runs import RunProjections
from chickadee.journal.writer import Journal
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema import EventKind, StopReason, ToolCall
from chickadee.schema.adapter import Completion, Usage
from chickadee.schema.journal import BudgetP, ModelCallP, RunStartP
from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots

SID = "S-spend"
RUN = "2026-10-01-0000000b"
WAIT_S = 20.0
CALL_COST = 1.10
SHA = "a" * 64


# -- the rung and the engine ------------------------------------------------------------------


def answer(text: str = "done") -> Completion:
    return Completion(
        text=text,
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


class ChargedRung:
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


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", commit=None)


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
    rung: ChargedRung,
) -> PacketEngine:
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: rung)
    engine = PacketEngine(repo, cfg)
    engines.append(engine)
    return engine


def lines(repo: Path, sid: str = SID) -> list[EventKind | str]:
    with Session.open(sessions_root(resolve_roots(repo)), sid) as s:
        return [e.kind for e, _ in s.state.lines()]


def errors(engine: PacketEngine, sid: str, text: str) -> list[str]:
    handle = engine.turn(sid, text)
    handle.wait(timeout=WAIT_S)
    return [str(e.payload) for e in handle.events() if e.kind == "error"]


# -- the journal of an episode ----------------------------------------------------------------


def open_episode(where: Roots, sid: str, run_id: str) -> None:
    with Session.open(sessions_root(where), sid) as session:
        open_episode_in(session, run_id)


def run_journal(where: Roots, run_id: str, *, coder_cost: float, crossed_at: float = 0) -> None:
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
    done = ModelCallP(
        role="worker",
        request="task",
        tokens_in=100,
        tokens_out=20,
        cost_usd=coder_cost,
        stop_reason="end_turn",
        rung="low-param",
    )
    with Journal.create(where, run_id, {}) as j:
        j.append(EventKind.RUN_START, start)
        j.append(EventKind.MODEL_CALL, done, task_id="T1", round=1)
        if crossed_at:
            crossing = BudgetP(
                scope="run", kind="warning", spent_usd=crossed_at, cap_usd=5.0, terminal=False
            )
            j.append(EventKind.BUDGET, crossing)


# -- every charge is on the log ---------------------------------------------------------------


def test_every_charge_leaves_one_spend_line(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Each charge Zeus's loop makes — one per completion that came back — is one line. A
    call that raises before a completion is not charged by the loop today, so it leaves
    none; charging it is `zeus/loop.py`'s."""
    rung = ChargedRung([note("n1"), answer()])
    engine = engine_for(monkeypatch, repo, config(raw, cap=100.0), engines, rung)
    engine.create_session(SID)

    assert errors(engine, SID, "hi") == []
    engine.close()

    assert len(rung.requests) == 2
    assert lines(repo).count("spend") == 2
    with Session.open(sessions_root(resolve_roots(repo)), SID) as s:
        spent = family_spent(s.root, s.session_id, s.state)
        assert spent.cost_usd == pytest.approx(2 * CALL_COST)
        assert {c.model for c in lines_of(s.state, SpendLine)} == {raw["rungs"]["high-param"]}
        assert {c.role for c in lines_of(s.state, SpendLine)} == {"zeus"}, "his calls (T-5.28)"


# -- a restart does not reset the cap --------------------------------------------------------


def test_a_second_engine_starts_from_the_first_ones_spend(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """$2.20 spent against a $2.00 cap, then the process ends. The next engine's first turn
    is refused before any model call: the cap was crossed, and a restart does not uncross
    it."""
    first = engine_for(
        monkeypatch, repo, config(raw, cap=2.0), engines, ChargedRung([note("n1")])
    )
    first.create_session(SID)
    errors(first, SID, "spend")
    first.close()

    rung = ChargedRung()
    second = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    said = errors(second, SID, "spend more")

    assert any("BUDGET_CAP" in e for e in said)
    assert rung.requests == []


def test_a_raised_cap_lets_the_session_go_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    first = engine_for(
        monkeypatch, repo, config(raw, cap=2.0), engines, ChargedRung([note("n1")])
    )
    first.create_session(SID)
    errors(first, SID, "spend")
    first.close()

    rung = ChargedRung()
    raised = engine_for(monkeypatch, repo, config(raw, cap=10.0), engines, rung)

    assert errors(raised, SID, "go on") == []
    assert len(rung.requests) == 1


def test_an_episodes_journal_counts_toward_the_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Zeus spent nothing; its one episode's coder calls cost $2.50 against a $2.00 cap."""
    where = resolve_roots(repo)
    Session.create(sessions_root(where), SID).close()
    open_episode(where, SID, RUN)
    run_journal(where, RUN, coder_cost=2.5)

    rung = ChargedRung()
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)

    assert any("BUDGET_CAP" in e for e in errors(engine, SID, "next"))
    assert rung.requests == []


def test_a_budget_crossing_is_a_floor_under_the_journals_coder_costs(repo: Path) -> None:
    """The crossing's amount is the lineage's total when it was written; a crash between it
    and the call's `model_call` loses the line, not the spend (the orchestrator's resume
    reads it the same way)."""
    where = resolve_roots(repo)
    with Session.create(sessions_root(where), SID) as session:
        open_episode_in(session, RUN)
        run_journal(where, RUN, coder_cost=0.5, crossed_at=4.0)
        spent = projection(session, where)

    assert spent.cost_usd == pytest.approx(4.0)


def test_a_journal_damaged_part_way_counts_up_to_the_damage(repo: Path) -> None:
    """A lower bound, never zero, and never a session that will not open."""
    where = resolve_roots(repo)
    with Session.create(sessions_root(where), SID) as session:
        open_episode_in(session, RUN)
        run_journal(where, RUN, coder_cost=1.5)
        path = journal_path(where, RUN)
        path.write_text(path.read_text() + '{"not": json\n', encoding="utf-8")
        spent = projection(session, where)

    assert spent.cost_usd == pytest.approx(1.5)


def projection(session: Session, where: Roots) -> Spend:
    caps = BudgetCaps(hard_cost_cap_usd=None)
    folds = ReplayCache(sessions_root(where))
    return ledger_inputs(
        session.session_id,
        session.state,
        where,
        folds,
        RunProjections(where),
        caps,
        {},
        DEFAULT_RUNGS,
    )[1]


def test_a_journal_torn_inside_a_character_does_not_lock_the_session(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    """A crash mid-write of non-ASCII text leaves half a UTF-8 sequence, which the journal
    reader refuses as `UnicodeDecodeError`. The session still opens and the turn runs."""
    where = resolve_roots(repo)
    Session.create(sessions_root(where), SID).close()
    open_episode(where, SID, RUN)
    run_journal(where, RUN, coder_cost=0.5)
    path = journal_path(where, RUN)
    path.write_bytes(path.read_bytes() + b'{"goal": "caf' + b"\xc3")

    rung = ChargedRung()
    engine = engine_for(monkeypatch, repo, config(raw, cap=100.0), engines, rung)

    assert errors(engine, SID, "next") == []
    assert len(rung.requests) == 1


def open_episode_in(session: Session, run_id: str) -> None:
    session.record_user(f"do {run_id}")
    call = ToolCall(
        id=f"c-{run_id}", name="coder", arguments={"title": run_id, "brief": run_id}
    )
    turn = session.record_assistant("", [call])
    session.record_episode(EpisodeLine(run_id, "planned", None, SHA, (), turn.seq))
    session.record_tool(f"c-{run_id}", f"COMPLETED · run {run_id}", run_id=run_id)


# -- the lineage -----------------------------------------------------------------------------


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


def test_a_fork_made_after_the_cap_is_over_it_too(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    where = resolve_roots(repo)
    Session.create(sessions_root(where), SID).close()
    spend_on(where, SID, 2.2)
    fork_of(where, SID, "S-child")

    rung = ChargedRung()
    child = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)

    assert any("BUDGET_CAP" in e for e in errors(child, "S-child", "in the child"))
    assert rung.requests == []


def test_a_childs_spend_counts_against_its_parent(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    """The parent spent $1.10, forked, and the child spent $1.10 — $2.20 of the
    family's $2.00 cap. The parent's next turn is refused."""
    where = resolve_roots(repo)
    Session.create(sessions_root(where), SID).close()
    spend_on(where, SID)
    fork_of(where, SID, "S-child")
    spend_on(where, "S-child")

    rung = ChargedRung()
    parent = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)

    assert any("BUDGET_CAP" in e for e in errors(parent, SID, "back in the parent"))
    assert rung.requests == []


def test_siblings_share_one_cap_and_each_counts_only_what_it_wrote(repo: Path) -> None:
    """Two children of one parent: the family's spend is each one's own lines once — the
    parent's $1.10 is not counted again for each child that reads it through."""
    where = resolve_roots(repo)
    Session.create(sessions_root(where), SID).close()
    spend_on(where, SID)
    fork_of(where, SID, "S-left")
    fork_of(where, SID, "S-right")
    spend_on(where, "S-left")
    spend_on(where, "S-right")

    with Session.open(sessions_root(where), "S-left") as left:
        assert projection(left, where).cost_usd == pytest.approx(3 * CALL_COST)


def test_a_siblings_spend_is_seen_at_the_next_turn(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any], engines: Engines
) -> None:
    """No restart: the engine reads the family again before each turn."""
    rung = ChargedRung()
    engine = engine_for(monkeypatch, repo, config(raw, cap=2.0), engines, rung)
    engine.create_session(SID)
    assert errors(engine, SID, "first") == []  # $1.10 of $2.00

    where = resolve_roots(repo)
    root = sessions_root(where)
    at = SessionLog.read(root, SID).last_seq
    with Session.fork(root, SID, at, "meanwhile", "S-sibling", code="keep") as sibling:
        sibling.record_spend(
            "high-param-model",
            Usage(prompt_tokens=100, completion_tokens=10),
            1.1,
            "high-param",
        )

    assert any("BUDGET_CAP" in e for e in errors(engine, SID, "second"))
    assert len(rung.requests) == 1


# -- the ledger -------------------------------------------------------------------------------


def ledger_for(session: Session, cap: float) -> CostLedger:
    cost = PriceTable.from_config({"m": PriceRow(price_in=10_000.0, price_out=10_000.0)})
    caps = BudgetCaps.from_config(SpendPolicy(hard_cost_usd=cap))
    return CostLedger(cost, caps, record=session.record_spend, rungs=DEFAULT_RUNGS)


def test_a_warning_crossed_before_the_restart_is_not_raised_again(repo: Path) -> None:
    where = resolve_roots(repo)
    usage = Usage(prompt_tokens=100, completion_tokens=10)
    with Session.create(sessions_root(where), SID) as session:
        # $1.10 is past the 0.8 warning on a $1.25 cap, and short of the cap itself.
        before = ledger_for(session, cap=1.25)
        before.for_rung("high-param").charge_usage("", usage, "m")
        assert [c.kind for c in before.crossings()] == ["warning"]

        after = ledger_for(session, cap=1.25)
        after.resume_from(projection(session, where))

        assert after.spend("session").cost_usd == pytest.approx(CALL_COST)
        assert after.crossings() == ()
        assert after.for_rung("high-param").may_start()


def test_a_spend_line_names_the_role_its_view_was_handed_to(repo: Path) -> None:
    """Two roles may share a rung (`zeus` and `residue` both default to `high-param`),
    so a line says whose call it was; a view handed to no role names none."""
    where = resolve_roots(repo)
    usage = Usage(prompt_tokens=100, completion_tokens=10)
    with Session.create(sessions_root(where), SID) as session:
        ledger = ledger_for(session, cap=100.0)
        ledger.for_rung("high-param", role="zeus").charge_usage("", usage, "m")
        ledger.for_rung("high-param", role="residue").charge_usage("", usage, "m")
        ledger.for_rung("high-param").charge_usage("", usage, "m")
        spent = lines_of(session.state, SpendLine)
        assert [(line.rung, line.role) for line in spent] == [
            ("high-param", "zeus"), ("high-param", "residue"), ("high-param", None),
        ]  # fmt: skip


def test_a_view_is_refused_a_role_the_cast_does_not_have(repo: Path) -> None:
    """A spend line is written for good, so a misspelt role (`resdiue`) would be read by no
    one: the ledger refuses it where it hands the view out."""
    with Session.create(sessions_root(resolve_roots(repo)), SID) as session:
        ledger = ledger_for(session, cap=100.0)
        with pytest.raises(HarnessBug, match="no role 'resdiue'"):
            ledger.for_rung("high-param", role="resdiue")


def test_a_tasks_charge_is_left_to_its_journal(repo: Path) -> None:
    """The box's calls are `model_call` lines in their run's journal; a `spend` line too would
    count them twice. Only a charge no run journals — Zeus's — is written here; a box call says
    it is journaled, Athena's outside a task included."""
    where = resolve_roots(repo)
    usage = Usage(prompt_tokens=100, completion_tokens=10)
    with Session.create(sessions_root(where), SID) as session:
        ledger = ledger_for(session, cap=100.0)
        ledger.for_rung("high-param").charge_usage("T1", usage, "m", journaled=True)
        ledger.for_rung("high-param").charge_usage("", usage, "m", journaled=True)
        assert lines_of(session.state, SpendLine) == ()
        ledger.for_rung("high-param").charge_usage("", usage, "m")
        assert len(lines_of(session.state, SpendLine)) == 1
