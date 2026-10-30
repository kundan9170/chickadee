"""`14-scout.md` §4, §8, §10: the loop's caps end it `truncated`, repeated calls
end it early, a scout that cannot search fails `UNAVAILABLE`, cancellation returns a partial
`CANCELLED`, and a failed report goes back as it is — never re-run on another model."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from scout_helpers import (
    CALLER,
    TARGET,
    FileReader,
    GraphNavigate,
    ListSink,
    ScriptedClient,
    answer,
    calls,
)

from chickadee.client.errors import RequestCancelled, TransportExhausted
from chickadee.config import RungEndpoint
from chickadee.errors import HarnessBug
from chickadee.executor.executor import Executor
from chickadee.graph.queries import Graph
from chickadee.hooks.fire import BoundHook, Hooks, HookSet, RoleHooks
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Continue
from chickadee.isolation.none import Passthrough
from chickadee.journal import Journal, paths, project, read_events
from chickadee.schema.adapter import Completion, StopReason, Usage
from chickadee.schema.findings import Findings
from chickadee.schema.journal import (
    EventKind,
    ScoutEscalatedP,
    ScoutReportP,
    ScoutStartP,
    ScoutToolP,
)
from chickadee.scout.escapes import Escapes
from chickadee.scout.findings import Checker
from chickadee.scout.loop import Scout
from chickadee.scout.rung import ScoutRung, nudge
from chickadee.scout.search import Search
from chickadee.scout.sink import JournalSink, check_record
from chickadee.scout.snapshot import Snapshot, build_snapshot
from chickadee.scout.tools import LocateCaps, ScoutTools
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-09-30-5c0e7002"
QUESTION = "where is the timeout for outgoing requests set"


class _Ledger:
    """The scout rung's view of a ledger (`budget.Ledger`), recording what it charged."""

    rung = "scout"

    def __init__(self, *, open_: bool = True) -> None:
        self.open = open_
        self.charged: list[tuple[str, Usage, str]] = []

    def charge_usage(
        self, scope_packet: str, usage: Usage, model: str, *, journaled: bool = False
    ) -> float:
        self.charged.append((scope_packet, usage, model))
        return 0.01

    def may_start(self) -> bool:
        return self.open

    def stopped_by(self) -> str | None:
        return None if self.open else "the scout rung's cap is reached"


def _endpoint(*, native_tools: bool = True) -> RungEndpoint:
    return RungEndpoint(
        base_url="http://localhost:1",
        model_id="m",
        context_window=32_000,
        max_output_tokens=2_000,
        native_tools=native_tools,
    )


def _scout(
    repo: Path,
    graph: Graph,
    client: ScriptedClient | None,
    *,
    caps: LocateCaps | None = None,
    ledger: _Ledger | None = None,
    navigate: GraphNavigate | None = None,
    hooks: RoleHooks | None = None,
) -> Scout:
    tools = ScoutTools(
        navigate or GraphNavigate(graph),
        FileReader(repo),
        Search(root=repo, rg=None),
        hooks=hooks,
    )
    rung = ScoutRung("local-mid", _endpoint(), client, "scout") if client is not None else None
    return Scout(
        rung,
        tools,
        Checker(repo, graph),
        caps or LocateCaps(40, 20_000),
        ledger=ledger,
        system="locate",
        hooks=hooks,
    )


@pytest.fixture
def snap(repo: Path, graph: Graph) -> Snapshot:
    return build_snapshot(repo, [], graph, probe=lambda _b: None)


def test_a_search_that_finds_its_target_answers_zeus(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    client = ScriptedClient(
        calls(("symbol", {"name": "HttpClient"}), ("grep", {"pattern": "timeout"})),
        answer(
            {"targets": [TARGET], "callers": [CALLER], "facts": ["config in config/app.yaml"]}
        ),
    )
    ledger = _Ledger()
    report = _scout(repo, graph, client, ledger=ledger).run(QUESTION, snap, sink)
    assert report.outcome == "found" and not report.findings.truncated
    assert report.findings.targets[0].path == "pkg/net/client.py"
    assert report.calls == 2 and report.searched == ("symbol 'HttpClient'", "grep 'timeout'")
    assert [m for _s, _u, m in ledger.charged] == ["local-mid", "local-mid"]
    assert report.cost_usd == pytest.approx(0.02) and report.tokens == 240
    assert sink.kinds() == [
        EventKind.SCOUT_START,
        EventKind.SCOUT_TOOL,
        EventKind.SCOUT_TOOL,
        EventKind.SCOUT_REPORT,
    ]
    start = sink.of(EventKind.SCOUT_START)[0]
    assert isinstance(start, ScoutStartP) and start.rung == "local-mid"
    assert start.snapshot_sha == snap.sha
    (rep,) = sink.of(EventKind.SCOUT_REPORT)
    assert isinstance(rep, ScoutReportP)
    assert (rep.targets, rep.facts, rep.unresolved, rep.truncated) == (1, 1, 0, False)
    # The first request carries the snapshot and the question, and the tools as schemas.
    first = client.requests[0]
    assert snap.render() in str(first.messages[1].content) and QUESTION in str(
        first.messages[1].content
    )
    assert {"symbol", "grep", "glob"} <= {t.name for t in first.tools}


def test_the_call_cap_ends_the_search_truncated_after_one_turn_to_report(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    client = ScriptedClient(
        calls(("symbol", {"name": "a"}), ("symbol", {"name": "b"}), ("symbol", {"name": "c"})),
        answer({"targets": [TARGET]}),
    )
    report = _scout(repo, graph, client, caps=LocateCaps(2, 20_000)).run(QUESTION, snap, sink)
    assert report.findings.truncated and report.calls == 2
    assert report.outcome == "found", "what it found before the cap still travels"
    tool_texts = [str(m.content) for m in client.requests[1].messages if m.role == "tool"]
    assert tool_texts[-1].startswith("BOUND_REACHED{bound: locate.max_tool_calls")
    assert client.requests[1].tools == client.requests[0].tools, (
        "the tools stay on the last turn: a history holding tool calls needs them, and the "
        "cached prefix is kept"
    )
    assert report.render().splitlines()[-1] == "truncated: true"


def test_the_cumulative_result_cap_ends_the_search_truncated(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    client = ScriptedClient(
        calls(("read", {"path": "pkg/net/client.py"})), answer({"targets": [TARGET]})
    )
    report = _scout(repo, graph, client, caps=LocateCaps(40, 5)).run(QUESTION, snap, sink)
    assert report.findings.truncated
    (tool,) = [m for m in client.requests[1].messages if m.role == "tool"]
    assert str(tool.content).startswith(
        "BOUND_REACHED{bound: locate.max_cumulative_result_tokens"
    )


def test_three_consecutive_repeated_calls_end_the_loop_without_progress(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """Three dedup hits in a row: the first call runs, the next three repeat it."""
    same = ("symbol", {"name": "HttpClient"})
    client = ScriptedClient(
        calls(same), calls(same), calls(same), calls(same), answer({"targets": [TARGET]})
    )
    nav = GraphNavigate(graph)
    report = _scout(repo, graph, client, navigate=nav).run(QUESTION, snap, sink)
    assert report.findings.truncated
    assert "3 consecutive calls repeated earlier ones" in report.render()
    assert nav.calls == ["symbol HttpClient"]
    assert client.replies == [], "the fifth reply was the report, after the early stop"


def test_an_unreachable_rung_is_unavailable_never_a_snapshot_only_report(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """No candidates guessed from the question — Zeus's own ladder is the fallback."""
    client = ScriptedClient(TransportExhausted("connection refused", 3))
    report = _scout(repo, graph, client).run("where is HttpClient.request used", snap, sink)
    assert report.outcome == "unavailable" and report.unavailable == "unreachable"
    assert report.findings == Findings(truncated=True)
    (rec,) = sink.of(EventKind.SCOUT_REPORT)
    assert isinstance(rec, ScoutReportP) and rec.truncated and rec.targets == 0, (
        "the record says the search did not finish, never that it found nothing"
    )
    text = report.render()
    assert text.startswith("UNAVAILABLE{rung: 'local-mid', reason: unreachable")
    assert "connection refused" in text and "FINDINGS" not in text
    assert sink.kinds() == [EventKind.SCOUT_START, EventKind.SCOUT_REPORT], "no record open"


def test_no_rung_at_all_is_unavailable(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    report = _scout(repo, graph, None).run("request timeout", snap, sink)
    assert report.outcome == "unavailable" and report.unavailable == "unreachable"
    assert report.model == "none" and "no scout rung is configured" in report.render()


def test_a_rung_that_fails_mid_search_is_unavailable_not_an_empty_report(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """An empty report would render UNRESOLVED — a finished search that found nothing."""
    client = ScriptedClient(
        calls(("symbol", {"name": "HttpClient"})), TransportExhausted("reset by peer", 3)
    )
    report = _scout(repo, graph, client).run(QUESTION, snap, sink)
    assert report.outcome == "unavailable" and report.unavailable == "failed_mid_search"
    assert "after 1 model call(s) and 1 tool call(s)" in report.detail
    assert "UNRESOLVED" not in report.render()


def test_a_model_without_function_calling_is_unavailable_and_nothing_is_sent(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    client = ScriptedClient()
    tools = ScoutTools(GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None))
    scout = Scout(
        ScoutRung("local-mid", _endpoint(native_tools=False), client, "scout"),
        tools,
        Checker(repo, graph),
        LocateCaps(40, 20_000),
        system="locate",
    )
    report = scout.run(QUESTION, snap, sink)
    assert report.outcome == "unavailable" and report.unavailable == "no_tool_calling"
    assert client.requests == [], "a scout with no tools would only guess"
    assert sink.kinds() == [EventKind.SCOUT_START, EventKind.SCOUT_REPORT]


def test_cancellation_between_calls_returns_a_partial_cancelled_report(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    cancel = threading.Event()

    class _Stopping(GraphNavigate):
        def symbol(self, name: str) -> str:
            cancel.set()
            return super().symbol(name)

    client = ScriptedClient(
        calls(("symbol", {"name": "HttpClient"}), ("symbol", {"name": "HttpClient"}))
    )
    report = _scout(repo, graph, client, navigate=_Stopping(graph)).run(
        QUESTION, snap, sink, cancel
    )
    assert report.outcome == "cancelled" and report.findings.truncated
    assert "CANCELLED" in report.render() and "partial: true" in report.render()
    assert len(client.requests) == 1, "no model call after the stop"
    assert report.calls == 1, "the second call of the turn was not dispatched"


def test_a_cancelled_request_is_cancelled_not_unreachable(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    report = _scout(repo, graph, ScriptedClient(RequestCancelled(0))).run(QUESTION, snap, sink)
    assert report.outcome == "cancelled" and report.unavailable is None


def test_a_failed_report_returns_as_it_is_never_re_run_on_another_model(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """No escalation rung. Dropped pointers are counted in the result Zeus reads, and
    the scout's one model is the only one asked."""
    ghost = {"path": "pkg/ghost.py", "line_start": 1, "line_end": 9}
    client = ScriptedClient(answer({"targets": [ghost]}))
    report = _scout(repo, graph, client).run(QUESTION, snap, sink)
    assert report.outcome == "unresolved" and report.dropped_pointers == 1
    assert len(client.requests) == 1 and client.replies == []
    assert "gate: dropped 1 pointer(s)" in report.render()
    assert EventKind.SCOUT_ESCALATED not in sink.kinds()


def test_the_scout_cannot_record_an_escalation() -> None:
    payload = ScoutEscalatedP(
        reason="x", from_rung="scout", to_rung="planner", dropped_pointers=1, truncated=False
    )
    with pytest.raises(HarnessBug):
        check_record(EventKind.SCOUT_ESCALATED, payload)


def test_findings_that_do_not_parse_get_one_nudge_then_none_are_taken(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    client = ScriptedClient(answer("the client is in pkg/net"), answer("still prose"))
    report = _scout(repo, graph, client).run(QUESTION, snap, sink)
    assert len(client.requests) == 2
    nudge_msg = client.requests[1].messages[-1]
    assert nudge_msg.role == "user" and str(nudge_msg.content).startswith(nudge())
    assert report.outcome == "unresolved" and "did not parse" in report.render()


def test_a_crossed_spend_cap_starts_no_call(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    client = ScriptedClient()
    report = _scout(repo, graph, client, ledger=_Ledger(open_=False)).run(QUESTION, snap, sink)
    assert client.requests == [] and report.findings.truncated
    assert "spend cap" in report.render()


def test_a_question_that_is_only_names_still_runs_the_loop(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """Nothing in the scout reads names out of the question. A name Zeus can write,
    Zeus looks up itself; `locate` searches what it is asked."""
    client = ScriptedClient(
        calls(("symbol", {"name": "HttpClient"})), answer({"targets": [TARGET]})
    )
    report = _scout(repo, graph, client).run("`HttpClient` `NoSuchThing`", snap, sink)
    assert len(client.requests) == 2 and report.outcome == "found"


def test_a_turn_cut_at_max_tokens_is_not_dispatched_and_is_re_asked(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """Output cut at max_tokens is truncated, never a terminal parse — even when what
    was cut happens to parse, and even when it carries tool calls."""
    cut = calls(("symbol", {"name": "HttpClient"})).model_copy(
        update={"stop_reason": StopReason.MAX_TOKENS, "text": '{"targets": []}'}
    )
    nav = GraphNavigate(graph)
    client = ScriptedClient(cut, answer({"targets": [TARGET]}))
    report = _scout(repo, graph, client, navigate=nav).run(QUESTION, snap, sink)
    assert nav.calls == [], "no call from a cut turn runs"
    assert str(client.requests[1].messages[-1].content).startswith(nudge())
    assert "max_tokens" in str(client.requests[1].messages[-1].content)
    assert report.outcome == "found"


def test_a_turn_cut_at_max_tokens_twice_takes_no_findings(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    cut = Completion(
        text='{"targets": [',
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        latency_ms=1,
        stop_reason=StopReason.MAX_TOKENS,
    )
    report = _scout(repo, graph, ScriptedClient(cut, cut)).run(QUESTION, snap, sink)
    assert report.outcome == "unresolved" and "none were taken" in report.render()


def test_tool_calls_after_the_search_ended_are_answered_once_never_run(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    nav = GraphNavigate(graph)
    client = ScriptedClient(
        calls(("symbol", {"name": "a"}), ("symbol", {"name": "b"})),
        calls(("symbol", {"name": "c"})),
        calls(("symbol", {"name": "d"})),
    )
    report = _scout(repo, graph, client, caps=LocateCaps(1, 20_000), navigate=nav).run(
        QUESTION, snap, sink
    )
    assert nav.calls == ["symbol a"]
    (after,) = [m for m in client.requests[2].messages[-1:] if m.role == "tool"]
    assert str(after.content).startswith("BOUND_REACHED{bound: the search ended")
    assert report.findings.truncated and "kept calling tools" in report.render()


def test_an_escape_that_could_not_fit_the_result_allowance_is_refused_before_it_runs(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """An escape is an effect: refused while it can still be, not run and then withheld."""
    escapes = Escapes(
        executor=Executor(backend=Passthrough()),
        cwd=repo,
        output_bytes=4096,
        allow_unsandboxed=True,
    )
    tools = ScoutTools(
        GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None), escapes=escapes
    )
    client = ScriptedClient(
        calls(("request_exec", {"command": "git log -1", "reason": "history"})),
        answer({"targets": [TARGET]}),
    )
    scout = Scout(
        ScoutRung("local-mid", _endpoint(), client, "scout"),
        tools,
        Checker(repo, graph),
        LocateCaps(40, 500),
        system="locate",
    )
    report = scout.run(QUESTION, snap, sink)
    assert escapes.used == 0 and EventKind.EFFECT_INTENDED not in sink.kinds()
    (tool,) = [m for m in client.requests[1].messages if m.role == "tool"]
    assert str(tool.content).startswith("BOUND_REACHED{bound: locate.max_cumulative")
    assert report.findings.truncated


def test_the_scout_s_records_replay_from_the_journal_its_sink_wrote(
    repo: Path, graph: Graph, snap: Snapshot
) -> None:
    client = ScriptedClient(
        calls(("symbol", {"name": "HttpClient"})), answer({"targets": [TARGET]})
    )
    with Journal.create(resolve_roots(repo), RUN, {}) as journal:
        _scout(repo, graph, client).run(QUESTION, snap, JournalSink(journal))
    events = list(read_events(paths.journal_path(resolve_roots(repo), RUN)))
    assert [e.kind for e in events] == [
        EventKind.SCOUT_START,
        EventKind.SCOUT_TOOL,
        EventKind.SCOUT_REPORT,
    ]
    state = project(events)
    assert state.last_seq == 3


def test_a_scout_kept_for_the_session_starts_each_call_fresh(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """Per call: the call count, the dedup cache, the escape budget. Per session: what the
    escapes already learned."""
    escapes = Escapes(
        executor=Executor(backend=Passthrough()), cwd=repo, budget=1, allow_unsandboxed=True
    )
    tools = ScoutTools(
        GraphNavigate(graph), FileReader(repo), Search(root=repo, rg=None), escapes=escapes
    )
    exec_call = ("request_exec", {"command": "git log --format=%s -1", "reason": "history"})
    client = ScriptedClient(
        calls(("symbol", {"name": "HttpClient"}), exec_call),
        answer({"targets": [TARGET]}),
        calls(("symbol", {"name": "HttpClient"}), exec_call),
        answer({"targets": [TARGET]}),
    )
    scout = Scout(
        ScoutRung("local-mid", _endpoint(), client, "scout"),
        tools,
        Checker(repo, graph),
        LocateCaps(40, 20_000),
        system="locate",
    )
    first = scout.run(QUESTION, snap, sink)
    second = scout.run(QUESTION, snap, sink)
    assert first.calls == second.calls == 2
    finds = [p for p in sink.of(EventKind.SCOUT_TOOL) if isinstance(p, ScoutToolP)]
    assert [p.dedup_hit for p in finds] == [False, False, False, False]
    assert first.escapes == 1 and second.escapes == 0, "the second call's answer was known"
    assert escapes.memo["git log --format=%s -1"].startswith("first commit")


def test_a_continue_at_its_findings_is_one_more_search_and_bounded(
    repo: Path, graph: Graph, snap: Snapshot, sink: ListSink
) -> None:
    """D227 `before_stop`: its findings, then the hook's words, and its tools open again; at
    most `hooks.max_continues`, and the report counts both asks."""
    spec = HookSpec.model_validate({"event": "before_stop", "name": "more", "use": "more"})
    bound = HookSet((BoundHook(spec, lambda e: Continue("name the callers too"), "more"),))
    hooks = RoleHooks(Hooks(bound, None), "locate")
    client = ScriptedClient(
        answer({"targets": [TARGET]}),
        calls(("grep", {"pattern": "HttpClient"})),
        answer({"targets": [TARGET], "callers": [CALLER]}),
    )
    ledger = _Ledger()
    report = _scout(repo, graph, client, ledger=ledger, hooks=hooks).run(QUESTION, snap, sink)
    assert report.outcome == "found" and len(report.findings.callers) == 1
    assert len(client.requests) == 3, "one continue, then the bound: it stops"
    assert client.requests[1].messages[-1].content == "name the callers too"
    assert report.cost_usd == pytest.approx(0.03), "both asks' charges"
