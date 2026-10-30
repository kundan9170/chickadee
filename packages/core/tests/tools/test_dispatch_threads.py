"""`Dispatcher.execute_group` (T-3.188 Part 3): a group's calls
whose spec says `threads` run at once, each on a thread of its own, and the rest on the loop's
thread as before; answers come in call order, and every rule, record and observer runs on the
loop's thread. `PerResponse` counts each tool's fan-out under its own key."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from chickadee.errors import HarnessBug
from chickadee.tools.catalogue import PerResponse, Valid
from chickadee.tools.context import ToolContext
from chickadee.tools.rules import CallOutcome, Pipeline
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

from .tools_helpers import KeepSink, call, dispatcher, echo_tool, failure_of, make_spec, said

LOOP = threading.get_ident


def _tool(
    name: str,
    *,
    threads: bool = False,
    parallel: str = "capped",
    barrier: threading.Barrier | None = None,
    ran_on: dict[str, int] | None = None,
    fail: bool = False,
    fanout: str | None = None,
) -> Tool:
    """An echo tool that records the thread each call ran on and, with a barrier, waits for
    the others: calls run one after another would break it."""
    base = echo_tool(name, parallel=parallel)
    inner = base.run

    def run(c: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
        if ran_on is not None:
            ran_on[c.id] = threading.get_ident()
        if barrier is not None:
            barrier.wait()
        if fail:
            raise RuntimeError("the tool broke")
        return inner(c, args, ctx)

    spec = replace(base.spec, threads=threads, fanout=fanout)
    return replace(base, spec=spec, run=run)


class _RecordingSink(KeepSink):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_on: list[int] = []

    def record(self, call: Invocation, outcome: CallOutcome) -> CallOutcome:
        self.recorded_on.append(threading.get_ident())
        return super().record(call, outcome)


def test_threaded_calls_run_at_once_and_answer_in_call_order() -> None:
    barrier = threading.Barrier(3, timeout=10)
    ran_on: dict[str, int] = {}
    sink = _RecordingSink()
    d = dispatcher([_tool("agent", threads=True, barrier=barrier, ran_on=ran_on)], sink=sink)
    calls = [call("agent", text=f"t{n}", cid=f"c{n}") for n in range(3)]
    pending = d.dispatch(d.admit(calls))
    outcomes = list(d.execute_group(pending, stopping=lambda: False))
    assert [said(o) for o in outcomes] == ["agent: t0", "agent: t1", "agent: t2"]
    assert LOOP() not in ran_on.values() and len(set(ran_on.values())) == 3
    assert set(sink.recorded_on) == {LOOP()}, "records stay on the loop's thread"


def test_the_rest_of_a_group_runs_on_the_loops_thread_beside_the_threads() -> None:
    ran_on: dict[str, int] = {}
    d = dispatcher(
        [
            _tool("agent", threads=True, ran_on=ran_on),
            _tool("look", parallel="any", ran_on=ran_on),
        ]
    )
    calls = [call("look", cid="c1"), call("agent", cid="c2"), call("look", cid="c3")]
    outcomes = list(d.execute_group(d.dispatch(d.admit(calls)), stopping=lambda: False))
    assert [o.pending.call.id for o in outcomes] == ["c1", "c2", "c3"]
    assert ran_on["c1"] == ran_on["c3"] == LOOP() != ran_on["c2"]


def test_a_stop_cancels_what_has_not_started_and_the_threads_are_answered() -> None:
    d = dispatcher([_tool("agent", threads=True), _tool("look", parallel="any")])
    calls = [call("agent", cid="c1"), call("look", cid="c2")]
    outcomes = list(d.execute_group(d.dispatch(d.admit(calls)), stopping=lambda: True))
    assert failure_of(outcomes[0]) is None
    assert failure_of(outcomes[1]) == "CANCELLED"


def test_a_threaded_run_that_raises_raises_on_the_loops_thread() -> None:
    d = dispatcher([_tool("agent", threads=True, fail=True)])
    pending = d.dispatch(d.admit([call("agent")]))
    with pytest.raises(RuntimeError, match="the tool broke"):
        list(d.execute_group(pending, stopping=lambda: False))


def test_a_group_with_no_threaded_call_runs_as_execute_does() -> None:
    ran_on: dict[str, int] = {}
    d = dispatcher([_tool("look", parallel="any", ran_on=ran_on)])
    calls = [call("look", cid="c1"), call("look", cid="c2")]
    outcomes = list(d.execute_group(d.dispatch(d.admit(calls)), stopping=lambda: False))
    assert [failure_of(o) for o in outcomes] == [None, None]
    assert set(ran_on.values()) == {LOOP()}


def test_only_a_call_that_runs_in_a_group_may_take_a_thread() -> None:
    with pytest.raises(ValueError, match="runs on a thread"):
        replace(make_spec("x", parallel="serial"), threads=True)


def test_pin_5_moved_each_tool_counts_its_fan_out_under_its_own_key() -> None:
    """Moved: pin 5 counted every `capped` call against one key.
    A tool whose spec names its own row (`ToolSpec.fanout`) counts apart: three agents and four
    locates are each within their own cap, and the fourth agent is refused and not counted."""
    d = dispatcher(
        [_tool("agent", threads=True, fanout="agents"), _tool("locate")],
        pipeline=Pipeline(admit=(PerResponse("fanout"), Valid()), counts_at=("turn",)),
        limits={"fanout": 4, "agents": 3},
    )
    calls = [call("agent", cid=f"a{n}") for n in range(4)] + [
        call("locate", cid=f"l{n}") for n in range(4)
    ]
    pending = d.admit(calls)
    assert [p.decided is None for p in pending] == [True] * 3 + [False] + [True] * 4
    refused = pending[3]
    assert refused.counted == ()
    assert "fan_out" in str(refused.decided)
    assert d.state.calls("turn") == 7


def test_a_tools_own_fan_out_row_is_required_where_fan_out_is_counted() -> None:
    """The row a spec names is the pipeline's to read wherever it offers the tool, so a
    dispatcher counting fan-out without it is a harness bug at composition, never a call
    refused on a missing limit; one that counts no fan-out does not read it."""
    tools = [_tool("agent", threads=True, fanout="agents")]
    counting = Pipeline(admit=(PerResponse("fanout"), Valid()), counts_at=("turn",))
    with pytest.raises(HarnessBug, match="agents"):
        dispatcher(tools, pipeline=counting, limits={"fanout": 4})
    assert dispatcher(tools, pipeline=Pipeline(admit=(Valid(),), counts_at=("turn",)))


def test_only_a_capped_call_has_a_fan_out_row() -> None:
    with pytest.raises(ValueError, match="only a capped call has a fan-out"):
        replace(make_spec("x", parallel="any"), fanout="agents")


def test_one_per_must_name_an_argument_of_the_tools_schema() -> None:
    with pytest.raises(ValueError, match="one_per names no argument"):
        replace(make_spec("x"), one_per="agent")
    assert replace(make_spec("x"), one_per="text").one_per == "text"


def test_the_calls_here_run_while_the_threads_do() -> None:
    """A plain call after an agent's does not wait for it (the review of #640): the two meet at
    one barrier, which a plain call held behind the thread would break."""
    barrier = threading.Barrier(2, timeout=10)
    ran_on: dict[str, int] = {}
    d = dispatcher(
        [
            _tool("agent", threads=True, barrier=barrier, ran_on=ran_on),
            _tool("look", parallel="any", barrier=barrier, ran_on=ran_on),
        ]
    )
    calls = [call("agent", cid="c1"), call("look", cid="c2")]
    outcomes = list(d.execute_group(d.dispatch(d.admit(calls)), stopping=lambda: False))
    assert [o.pending.call.id for o in outcomes] == ["c1", "c2"]
    assert [failure_of(o) for o in outcomes] == [None, None]
    assert ran_on["c2"] == LOOP() != ran_on["c1"]


def test_a_thread_that_raises_never_takes_its_siblings_answers_with_it() -> None:
    """Every other call is recorded before the error is raised: agents that finished were paid
    for, and their reports reach the record.

    Moved (the second review of #640): the record was in call order, `c2` then `c3`. A call
    run here is recorded as it ends, inside its hold, so `c3` is recorded before the agent
    `c2`; the answers still come in call order."""
    sink = _RecordingSink()
    d = dispatcher(
        [
            _tool("broken", threads=True, fail=True),
            _tool("agent", threads=True),
            _tool("look", parallel="any"),
        ],
        sink=sink,
    )
    calls = [call("broken", cid="c1"), call("agent", cid="c2"), call("look", cid="c3")]
    seen: list[str] = []
    with pytest.raises(RuntimeError, match="the tool broke"):
        for outcome in d.execute_group(d.dispatch(d.admit(calls)), stopping=lambda: False):
            seen.append(outcome.pending.call.id)
    assert seen == ["c2", "c3"]
    assert [o.pending.call.id for o in sink.records] == ["c3", "c2"]


def test_a_call_run_here_is_recorded_before_the_next_one_runs() -> None:
    """A read's hold spans its run and its line (`ZeusSink.executing`), so two reads of one
    file never decide against the same view. In a group with an agent, the calls run here
    were all recorded after every run, so the second ran before the first's line. Each is
    recorded as it ends now, while the agent still runs; answers still come in call order."""
    events: list[str] = []
    looked = threading.Event()

    class Sink(KeepSink):
        def record(self, call: Invocation, outcome: CallOutcome) -> CallOutcome:
            events.append(f"recorded {call.id}")
            if call.id == "c3":
                looked.set()
            return super().record(call, outcome)

    def logged(name: str, *, threads: bool = False, wait: bool = False) -> Tool:
        base = _tool(name, threads=threads)

        def run(c: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
            if wait:
                assert looked.wait(10), "the agent outlives the calls run here"
            events.append(f"ran {c.id}")
            return base.run(c, args, ctx)

        return replace(base, run=run)

    d = dispatcher(
        [logged("agent", threads=True, wait=True), logged("look")],
        sink=Sink(),
    )
    calls = [call("agent", cid="c1"), call("look", cid="c2"), call("look", cid="c3")]
    outcomes = list(d.execute_group(d.dispatch(d.admit(calls)), stopping=lambda: False))
    assert [o.pending.call.id for o in outcomes] == ["c1", "c2", "c3"]
    assert events == ["ran c2", "recorded c2", "ran c3", "recorded c3", "ran c1", "recorded c1"]
