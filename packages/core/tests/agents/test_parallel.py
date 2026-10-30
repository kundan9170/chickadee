"""D213 Amendment 1 decision 1 (T-3.188 Part 3): agents in one response run at once.

Through the real `spawn_agent` tool and a dispatcher holding Zeus's fan-out rule: three spawns
overlap (each client waits at one barrier, which calls run one after another would break) and
answer in call order; a fourth is refused and not counted; each agent takes its own id, its
own ports on its own thread, and writes whole records; a crossed spend cap and the person's
stop each reach every running agent before its next model call.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from agents_helpers import (
    AgentClient,
    LineReader,
    NoGraph,
    agents_for,
    no_search,
    report,
    turn_calls,
    turn_text,
    write_repo,
)

from chickadee.agents.profiles import Profile
from chickadee.agents.spawn import AgentPorts, Agents
from chickadee.client.request import ChatRequest
from chickadee.context.kept_lines import AgentRecord
from chickadee.context.session import Session
from chickadee.schema.adapter import Completion, ToolCall, Usage
from chickadee.schema.journal import AgentMessageP, AgentReportP, AgentStartP
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import PerResponse, Valid, answer_text
from chickadee.tools.context import ToolContext
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.policy import FileRule, RolePolicy
from chickadee.tools.rules import CallOutcome, Level, Pipeline
from chickadee.tools.sink import Counters
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Invocation

PARALLEL = 3


class _AnswerSink:
    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        return nullcontext()

    def replay(self, level: Level) -> Counters:
        return Counters()

    def render(self, outcome: CallOutcome) -> str:
        return answer_text(outcome)

    def record(self, call: Invocation, outcome: CallOutcome) -> str:
        return answer_text(outcome)


def _zeus(agents: Agents, cancel: threading.Event | None = None) -> Dispatcher[str]:
    """A dispatcher offering `spawn_agent` under Zeus's per-response rule."""
    policy = RolePolicy(
        role="zeus",
        enabled=("spawn_agent", "follow_up"),
        shown_as={},
        files=FileRule("repository", "visible"),
        graph_depth=None,
        pipeline=Pipeline(
            admit=(PerResponse("zeus.locate_fanout"), Valid()),
            counts_at=("turn",),
        ),
    )
    d = Dispatcher[str](
        BUILTINS,
        policy,
        ToolContext(role="zeus", agents=agents, cancel=cancel),
        _AnswerSink(),
        limits={"zeus.locate_fanout": 4, "agents.max_parallel": PARALLEL},
    )
    d.open_scope("turn")
    return d


def _spawns(n: int) -> list[ToolCall]:
    return [
        ToolCall(
            id=f"z{k}",
            name="spawn_agent",
            arguments={"profile": "reader", "task": f"job {k}: find the check"},
        )
        for k in range(n)
    ]


class _Together:
    """Clients whose first request waits at one barrier with every other round's: rounds run
    one after another would break it. `then` runs once, as every round arrives."""

    def __init__(
        self, script: Callable[[], list[Completion]], then: Callable[[], None] = lambda: None
    ) -> None:
        self.barrier = threading.Barrier(PARALLEL, action=then, timeout=20)
        self.script = script

    def __call__(self) -> AgentClient:
        barrier = self.barrier

        class Waiting(AgentClient):
            def send(self, request: ChatRequest) -> Completion:
                if not self.requests:
                    barrier.wait()
                return super().send(request)

        return Waiting(*self.script())


def _session(tmp_path: Path) -> Session:
    return Session.create(tmp_path / "sessions", "S-parallel")


def _by_agent(session: Session) -> dict[str, list[object]]:
    out: dict[str, list[object]] = {}
    for _, r in session.state.lines():
        if isinstance(r, AgentRecord) and r.line is not None:
            payload = r.line.payload
            assert isinstance(payload, AgentStartP | AgentMessageP | AgentReportP)
            out.setdefault(payload.agent, []).append(payload)
    return out


def test_three_agents_run_at_once_answer_in_order_and_a_fourth_is_refused(
    tmp_path: Path,
) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    together = _Together(
        lambda: [
            turn_calls(("read", {"path": "src/auth.py"})),
            turn_text(report("- the check: `src/auth.py:2-2`.")),
        ]
    )
    try:
        agents = agents_for(session, repo, AgentClient(), clients=together)
        d = _zeus(agents)
        pending = d.admit(_spawns(PARALLEL + 1))
        assert [p.decided is None for p in pending] == [True] * PARALLEL + [False]
        assert pending[-1].counted == () and d.state.calls("turn") == PARALLEL
        answers = list(d.execute_group(d.dispatch(pending), stopping=lambda: False))
        heads = [a.split("\n", 1)[0] for a in answers[:PARALLEL]]
        assert all("· reader · complete ·" in h for h in heads)
        assert "fan_out" in answers[-1]
        # Each agent took its own id, and its records are whole: its start first, its
        # report last, every message between.
        records = _by_agent(session)
        assert sorted(records) == ["A1", "A2", "A3"]
        for payloads in records.values():
            assert isinstance(payloads[0], AgentStartP)
            assert isinstance(payloads[-1], AgentReportP)
            assert all(isinstance(p, AgentMessageP) for p in payloads[1:-1])
        started = {
            p.call_id: p.agent
            for ps in records.values()
            for p in ps
            if isinstance(p, AgentStartP)
        }
        assert sorted(started) == ["z0", "z1", "z2"]
        assert [started[f"z{k}"] in heads[k] for k in range(PARALLEL)] == [True] * PARALLEL
    finally:
        session.close()


def test_each_agent_builds_and_closes_its_ports_on_its_own_thread(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    built: list[int] = []
    closed: list[int] = []
    lock = threading.Lock()

    def ports(profile: Profile) -> AgentPorts:
        with lock:
            built.append(threading.get_ident())

        def close() -> None:
            with lock:
                closed.append(threading.get_ident())

        return AgentPorts(LineReader(repo), NoGraph(), no_search, close=close)

    together = _Together(lambda: [turn_text(report())])
    try:
        agents = agents_for(session, repo, AgentClient(), clients=together, ports=ports)
        d = _zeus(agents)
        list(d.execute_group(d.dispatch(d.admit(_spawns(PARALLEL))), stopping=lambda: False))
        assert len(set(built)) == PARALLEL and threading.get_ident() not in built
        assert sorted(closed) == sorted(built)
    finally:
        session.close()


class _SharedCap:
    """Every rung's ledger, one cap: crossed by the test, read by every round."""

    rung = "mid-param"

    def __init__(self) -> None:
        self.crossed = threading.Event()

    def may_start(self) -> bool:
        return not self.crossed.is_set()

    def stopped_by(self) -> str | None:
        return "the session's cap" if self.crossed.is_set() else None

    def charge_usage(
        self, scope_packet: str, usage: Usage, model: str, *, journaled: bool = False
    ) -> float:
        return 0.0


def _reading() -> list[Completion]:
    return [
        turn_calls(("read", {"path": "src/auth.py"})),
        turn_calls(("read", {"path": "src/auth.py"})),
        turn_text(report()),
    ]


def test_a_crossed_spend_cap_stops_every_running_agent(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    cap = _SharedCap()
    together = _Together(_reading, then=cap.crossed.set)
    try:
        agents = agents_for(session, repo, AgentClient(), clients=together, ledger=cap)
        d = _zeus(agents)
        answers = list(
            d.execute_group(d.dispatch(d.admit(_spawns(PARALLEL))), stopping=lambda: False)
        )
        assert all("· failed ·" in a and "spend cap" in a for a in answers), answers
    finally:
        session.close()


def test_the_persons_stop_cancels_every_running_agent(tmp_path: Path) -> None:
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    esc = threading.Event()
    together = _Together(_reading, then=esc.set)
    try:
        agents = agents_for(session, repo, AgentClient(), clients=together)
        d = _zeus(agents, cancel=esc)
        answers = list(
            d.execute_group(d.dispatch(d.admit(_spawns(PARALLEL))), stopping=lambda: False)
        )
        assert all("· cancelled ·" in a for a in answers), answers
    finally:
        session.close()


def _follow(agent: str, cid: str) -> ToolCall:
    return ToolCall(id=cid, name="follow_up", arguments={"agent": agent, "message": "more"})


def test_follow_ups_share_the_agents_cap_and_one_agent_is_followed_up_once(
    tmp_path: Path,
) -> None:
    """Spawns and follow-ups are counted together against
    `agents.max_parallel`, and a second follow-up of one agent in a response is refused,
    naming it; neither refusal counts."""
    repo = write_repo(tmp_path / "repo")
    session = _session(tmp_path)
    try:
        d = _zeus(agents_for(session, repo, AgentClient()))
        twice = d.admit([_follow("A1", "f1"), _follow("A1", "f2"), _follow("A2", "f3")])
        assert [p.decided is None for p in twice] == [True, False, True]
        assert "A1 already has a follow_up call in this response" in str(twice[1].decided)
        assert twice[1].counted == ()
        mixed = d.admit([*_spawns(2), _follow("A1", "f1"), _follow("A2", "f2")])
        assert [p.decided is None for p in mixed] == [True, True, True, False]
        assert "fan_out" in str(mixed[-1].decided)
    finally:
        session.close()
