"""A turn's family reads equal a from-scratch read, at every turn.

One family, scripted across turns while its records grow:
- a root, driven by a long-lived engine, whose episodes' journals grow between turns: one live
  run appending `model_call` lines, one ending in `run_end`, one gaining a `budget` floor, one
  whose tail is torn and then completed;
- a fork at a seq above 0, holding an episode whose journal is damaged part way, and the
  `cap` lines, for the total and for a rung;
- a fork at seq 0;
- a sibling written by a second engine.

At every step, `budget()` for the root and the fork and `header()` for the root equal what a
fresh engine on the same repository computes, and the inputs each turn restores its ledger
from equal the same read made with nothing cached. The root's and the fork's spend are pinned
too, so a change to both sides at once cannot pass.

Every scripted Zeus call costs $1.10: 100 prompt tokens and 10 completion tokens, at $10,000
per million each way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo

import chickadee.engine as engine_module
from chickadee.budget.family import ledger_inputs
from chickadee.client.request import ChatRequest
from chickadee.config import EngineConfig
from chickadee.context.lines import EpisodeLine
from chickadee.context.replay import ReplayCache
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.journal.paths import journal_path
from chickadee.journal.runs import RunProjections
from chickadee.journal.writer import Journal
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema import EventKind, StopReason, ToolCall
from chickadee.schema.adapter import Completion, Usage
from chickadee.schema.journal import BudgetP, ModelCallP, RunEndP, RunStartP
from chickadee.state.roots import Roots
from chickadee.state.roots import roots as resolve_roots

ROOT, FORK, ZERO, SIBLING = "S-root", "S-fork", "S-zero", "S-sibling"
LIVE, ENDED, FLOORED, TORN = (f"2026-10-05-0000000{n}" for n in range(1, 5))
DAMAGED = "2026-10-05-00000005"
WAIT_S = 20.0
SHA = "a" * 64
MODEL: dict[str, Any] = {
    "base_url": "http://localhost:8000/v1",
    "model_id": "m",
    "context_window": 65_536,
    "max_output_tokens": 16_384,
}


class Rung:
    """Every rung, scripted: each call answers `done` at $1.10."""

    def send(self, request: ChatRequest) -> Completion:
        return Completion(
            text="done",
            usage=Usage(prompt_tokens=100, completion_tokens=10),
            latency_ms=1,
            stop_reason=StopReason.END_TURN,
        )

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


def config() -> EngineConfig:
    rungs = {name: f"{name}-model" for name in DEFAULT_RUNGS}
    return EngineConfig.with_defaults(
        {
            "models": {model: dict(MODEL) for model in rungs.values()},
            "rungs": rungs,
            "prices": {rungs["high-param"]: {"price_in": 10_000.0, "price_out": 10_000.0}},
            "spend": {"hard_cost_usd": 100.0, "rungs": {"low-param": 60.0}},
        }
    )


# -- the records ------------------------------------------------------------------------------


def run_start() -> RunStartP:
    return RunStartP(
        goal="g",
        work_sha=SHA,
        config_sha=SHA,
        repo_head=SHA,
        engine_version="0",
        schema_version=2,
        isolation="worktree",
        hard_cost_cap_usd=5.0,
    )


def model_call(cost: float, rung: str = "coder") -> ModelCallP:
    """A worker's priced call inside a task: the line a run's spend is folded from."""
    return ModelCallP(
        role="worker",
        request="turn",
        tokens_in=100,
        tokens_out=20,
        cost_usd=cost,
        stop_reason="end_turn",
        rung=rung,
    )


def open_episode_in(session: Session, run_id: str) -> None:
    session.record_user(f"do {run_id}")
    call = ToolCall(
        id=f"c-{run_id}", name="coder", arguments={"title": run_id, "brief": f"do {run_id}"}
    )
    turn = session.record_assistant("", [call])
    session.record_episode(EpisodeLine(run_id, "planned", None, None, (), turn.seq))
    session.record_tool(f"c-{run_id}", f"COMPLETED · run {run_id}", run_id=run_id)


def journal(where: Roots, run_id: str, cost: float) -> Journal:
    """A run's journal, open for appending: its start and one worker call."""
    j = Journal.create(where, run_id, {})
    j.append(EventKind.RUN_START, run_start())
    j.append(EventKind.MODEL_CALL, model_call(cost), task_id="T1", round=1)
    return j


def build_family(where: Roots) -> dict[str, Journal]:
    """The family's records before the first turn, and the journals still being written."""
    sessions = sessions_root(where)
    with Session.create(sessions, ROOT) as root:
        root.record_user("start")
        for run_id in (LIVE, ENDED, FLOORED, TORN):
            open_episode_in(root, run_id)
        at = root.state.last_seq
    Session.fork(sessions, ROOT, at, "another way", FORK, code="keep").close()
    Session.fork(sessions, ROOT, 0, "from nothing", ZERO, code="keep").close()
    Session.fork(sessions, ROOT, at, "a sibling", SIBLING, code="keep").close()
    with Session.open(sessions, FORK) as fork:
        open_episode_in(fork, DAMAGED)
    with journal(where, DAMAGED, 0.5) as damaged:
        damaged.append(
            EventKind.MODEL_CALL, model_call(0.25, "frontier"), task_id="T1", round=1
        )
    return {run_id: journal(where, run_id, 1.0) for run_id in (LIVE, ENDED, FLOORED, TORN)}


# -- the oracle --------------------------------------------------------------------------------


Inputs = list[tuple[str, object, object]]


def spy_on_ledger(monkeypatch: pytest.MonkeyPatch) -> Inputs:
    """Each turn's ledger inputs, beside the same read made with nothing cached."""
    seen: Inputs = []
    real = ledger_inputs

    def spy(session_id: str, state: Any, where: Roots, *rest: Any) -> Any:
        got = real(session_id, state, where, *rest)
        _folds, _runs, *config_args = rest
        sessions = sessions_root(where)
        fresh = real(
            session_id, state, where, ReplayCache(sessions), RunProjections(where), *config_args
        )
        seen.append((session_id, got, fresh))
        return got

    monkeypatch.setattr(engine_module, "ledger_inputs", spy)
    return seen


def outcome[T](read: Callable[[], T]) -> T | str:
    """A read's value, or the refusal it raised, so both engines can be compared on it."""
    try:
        return read()
    except SessionError as exc:
        return f"SessionError: {exc}"


def from_scratch(repo: Path, cfg: EngineConfig) -> tuple[object, object, object]:
    fresh = PacketEngine(repo, cfg)
    try:
        return (
            outcome(lambda: fresh.budget(ROOT)),
            outcome(lambda: fresh.budget(FORK)),
            outcome(lambda: fresh.header(ROOT)),
        )
    finally:
        fresh.close()


def turn(engine: PacketEngine, sid: str) -> None:
    handle = engine.turn(sid, "go on")
    handle.wait(timeout=WAIT_S)
    assert [e for e in handle.events() if e.kind == "error"] == []


# -- the script --------------------------------------------------------------------------------


@pytest.fixture
def engines() -> Iterator[list[PacketEngine]]:
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()


def test_every_turns_family_reads_equal_a_read_from_scratch(
    monkeypatch: pytest.MonkeyPatch, repo: Path, engines: list[PacketEngine]
) -> None:
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: Rung())
    inputs = spy_on_ledger(monkeypatch)
    where = resolve_roots(repo)
    cfg = config()
    writing = build_family(where)
    long_lived = PacketEngine(repo, cfg)
    second = PacketEngine(repo, cfg)
    engines.extend([long_lived, second])
    spent: list[tuple[str, float, float]] = []

    def check(step: str) -> None:
        root_budget = long_lived.budget(ROOT)
        fork_budget = long_lived.budget(FORK)
        assert (root_budget, fork_budget, outcome(lambda: long_lived.header(ROOT))) == (
            from_scratch(repo, cfg)
        ), step
        spent.append((step, round(root_budget.spent_usd, 6), round(fork_budget.spent_usd, 6)))

    def torn_then_whole() -> None:
        writing[TORN].append(EventKind.MODEL_CALL, model_call(0.75), task_id="T2", round=1)
        writing[TORN].close()
        path = journal_path(where, TORN)
        whole = path.read_bytes()
        path.write_bytes(whole[:-9])
        check("torn")
        turn(long_lived, ROOT)
        path.write_bytes(whole)
        check("completed")

    def damage() -> None:
        with journal_path(where, DAMAGED).open("ab") as fh:
            fh.write(b'{"seq": 4, "not": "an event"}\n{"seq": 5}\n')

    def cap_from_the_fork() -> None:
        with Session.open(sessions_root(where), FORK) as fork:
            fork.record_cap(80.0)
            fork.record_cap(40.0, "coder")

    def sibling_turn() -> None:
        turn(second, SIBLING)

    def live_run_grows() -> None:
        writing[LIVE].append(EventKind.MODEL_CALL, model_call(2.0), task_id="T2", round=1)

    def run_ends() -> None:
        end = RunEndP(
            outcome="COMPLETED",
            tasks=1,
            committed=1,
            tokens=120,
            cost_usd=1.0,
            wall_ms=1,
            handoff_path="h",
        )
        writing[ENDED].append(EventKind.RUN_END, end)
        writing[ENDED].close()

    def floor_rises() -> None:
        crossing = BudgetP(
            scope="run", kind="warning", spent_usd=13.0, cap_usd=5.0, terminal=False
        )
        writing[FLOORED].append(EventKind.BUDGET, crossing)

    check("before any turn")
    for step, change in (
        ("first turn", None),
        ("live run grows", live_run_grows),
        ("run ends", run_ends),
        ("sibling turns", sibling_turn),
        ("floor rises", floor_rises),
        ("journal damaged", damage),
        ("capped from the fork", cap_from_the_fork),
        ("live run grows again", live_run_grows),
        ("sibling turns again", sibling_turn),
    ):
        if change is not None:
            change()
            check(step)
        turn(long_lived, ROOT)
        check(f"{step}, then a turn" if change is not None else step)
    torn_then_whole()
    turn(long_lived, ROOT)
    check("last turn")
    for journal_writer in writing.values():
        journal_writer.close()

    assert inputs, "every turn restored its ledger"
    for session_id, got, fresh in inputs:
        assert got == fresh, session_id
    assert {sid for sid, _got, _fresh in inputs} == {ROOT, SIBLING}
    # The root's and the fork's spend at each step, pinned on the code in older versions: one
    # family, so one amount. A Zeus turn is $1.10; a worker call, a floor and a completed tail
    # are what each step wrote; a torn tail and a line past damage count nothing.
    assert spent == [
        (step, amount, amount)
        for step, amount in (
            ("before any turn", 4.75),
            ("first turn", 5.85),
            ("live run grows", 7.85),
            ("live run grows, then a turn", 8.95),
            ("run ends", 8.95),
            ("run ends, then a turn", 10.05),
            ("sibling turns", 11.15),
            ("sibling turns, then a turn", 12.25),
            ("floor rises", 13.0),
            ("floor rises, then a turn", 13.35),
            ("journal damaged", 13.35),
            ("journal damaged, then a turn", 14.45),
            ("capped from the fork", 14.45),
            ("capped from the fork, then a turn", 15.55),
            ("live run grows again", 17.55),
            ("live run grows again, then a turn", 18.65),
            ("sibling turns again", 19.75),
            ("sibling turns again, then a turn", 20.85),
            ("torn", 20.85),
            ("completed", 22.7),
            ("last turn", 23.8),
        )
    ]


# -- what a turn opens -------------------------------------------------------------------------


def reads_per_turn(
    monkeypatch: pytest.MonkeyPatch, repo: Path, engines: list[PacketEngine], n: int
) -> list[str]:
    """The files a turn opens for reading, after a first turn has read everything once, in a
    family with one sibling that wrote a line since, `n` untouched siblings and `n` ended
    episodes."""
    where = resolve_roots(repo)
    sessions = sessions_root(where)
    with Session.create(sessions, ROOT) as root:
        for i in range(n):
            run_id = f"2026-10-05-1{i:07d}"
            open_episode_in(root, run_id)
            with journal(where, run_id, 1.0) as ended:
                ended.append(EventKind.RUN_END, run_end())
        at = root.state.last_seq
    for name in ("S-changed", *(f"S-sibling-{i}" for i in range(n))):
        Session.fork(sessions, ROOT, at, "another way", name, code="keep").close()
    engine = PacketEngine(repo, config())
    engines.append(engine)
    turn(engine, ROOT)
    with Session.open(sessions, "S-changed") as changed:
        changed.record_user("a line since the last turn")
    opened: list[str] = []
    real = Path.open

    def counted(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if not set(mode) & {"w", "a", "+", "x"}:
            opened.append(path.relative_to(repo).as_posix())
        return real(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    turn(engine, ROOT)
    monkeypatch.setattr(Path, "open", real)
    return opened


def run_end() -> RunEndP:
    return RunEndP(
        outcome="COMPLETED",
        tasks=1,
        committed=1,
        tokens=120,
        cost_usd=1.0,
        wall_ms=1,
        handoff_path="h",
    )


def test_a_turn_opens_only_what_changed_however_large_the_family(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engines: list[PacketEngine]
) -> None:
    """D158 §5d: a turn opens a changed member's log from where it was read and the blobs of
    its new lines; an untouched sibling and an ended episode are only stat-ed. So ten of each
    cost a turn no more opens than one of each."""
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: Rung())
    repos = []
    for size in (1, 10):
        path = tmp_path / f"repo-{size}"
        init_repo(path, commit=None)
        repos.append(reads_per_turn(monkeypatch, path, engines, size))
    one, many = repos
    assert len(many) == len(one), (one, many)
    assert one and all("/S-changed/" in p for p in one), one
    assert all("/S-changed/" in p for p in many), many
