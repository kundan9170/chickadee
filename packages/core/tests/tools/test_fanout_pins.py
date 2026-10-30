"""T-3.188's pin 5: how a response's `capped` calls are counted and run before Part 3.
Each test names the answer Part 3 moves, and the task's Log lists the move."""

from __future__ import annotations

import threading
from dataclasses import replace

from chickadee.tools.catalogue import PerResponse, Valid
from chickadee.tools.context import ToolContext
from chickadee.tools.rules import Pipeline
from chickadee.tools.tool import Answer, Invocation, Tool
from chickadee.tools.validate import Arguments

from .tools_helpers import call, dispatcher, echo_tool, failure_of


def test_pin_5_capped_calls_share_one_fan_out_count() -> None:
    """Two `capped` tools are counted together against the one key."""
    a = echo_tool("a", parallel="capped")
    b = echo_tool("b", parallel="capped")
    d = dispatcher(
        [a, b],
        pipeline=Pipeline(admit=(PerResponse("fanout"), Valid())),
        limits={"fanout": 2},
    )
    pending = d.admit([call("a", cid="c1"), call("b", cid="c2"), call("a", cid="c3")])
    assert [p.decided is None for p in pending] == [True, True, False]


def test_pin_5_a_group_runs_on_the_loops_thread_one_after_another() -> None:
    seen: list[str] = []
    ran_on: list[int] = []

    def tool(name: str) -> Tool:
        t = echo_tool(name, parallel="capped", runs=seen)
        inner = t.run

        def run(c: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
            ran_on.append(threading.get_ident())
            return inner(c, args, ctx)

        return replace(t, run=run)

    d = dispatcher([tool("a"), tool("b")])
    pending = d.dispatch(d.admit([call("a", cid="c1"), call("b", cid="c2")]))
    outcomes = [d.execute(p) for p in pending]
    assert seen == ["c1", "c2"]
    assert set(ran_on) == {threading.get_ident()}
    assert [failure_of(o) for o in outcomes] == [None, None]
