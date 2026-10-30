"""The tool-call bound refuses, typed, and the turn goes on.

The 65th call of a turn (cap 64) is `BOUND_REACHED` with `used == limit`, the tool is not
dispatched, and the 66th gets the same value; the turn's text still streams; the session
budget is unchanged by the refusal; `max_tool_calls == 0` never refuses; the session
log has ONE bound line for the turn, not one per refused call. The read bound refuses
the same way."""

from __future__ import annotations

from pathlib import Path

import pytest
from refusals import stated

from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable, RungPrice
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema import ToolCall
from chickadee.zeus.loop import LoopConfig

from .zeus_helpers import ScriptedGraph, answer, call, calls, payloads, rig, zeus_prefix

REFUSED = (
    "BOUND_REACHED{resource: tool_calls, used: 64, limit: 64, scope: turn, "
    "reason: 'bounded for safety, not for spend'}"
)


def _finds(n: int, start: int = 0, prefix: str = "f") -> list[ToolCall]:
    """`n` distinct calls — distinct, so repetition is not what refuses them."""
    return [call("grep", f"{prefix}{i}", pattern=f"q{i}") for i in range(start, start + n)]


def _cfg(cap: int) -> LoopConfig:
    return LoopConfig(max_result_tokens=0, max_tool_calls=cap)


def _ledger() -> CostLedger:
    """Keyed by catalog model, as `EngineConfig.prices` is: `frontier` is the model the
    rig's zeus rung runs on, and no row is keyed by the rung's name."""
    frontier = RungPrice(price_in=1.0, price_out=2.0)
    stronger = RungPrice(price_in=10.0, price_out=20.0)
    return CostLedger(
        PriceTable({"frontier": frontier, "frontier-b": stronger}),
        BudgetCaps(hard_cost_cap_usd=100.0),
        rungs=DEFAULT_RUNGS,
    )


def test_the_65th_call_is_refused_and_the_66th_gets_the_same_value(tmp_path: Path) -> None:
    ledger = _ledger()
    r = rig(
        tmp_path,
        [calls(*_finds(66)), answer("searched 64; the rest next turn")],
        config=_cfg(64),
        ledger=ledger,
    )
    events = r.run("search everything")
    state = r.session.state
    assert len(r.graph.asked) == 64, "the refused calls were not dispatched"
    for cid in ("f64", "f65"):
        line = state.result(cid)
        assert line is not None
        assert stated(line.text) == REFUSED
        assert line.ran is False
    served = state.result("f63")
    assert served is not None and served.ran and "q63" in served.text
    # The turn continues in prose: the refusals reached the model, and it answered.
    assert REFUSED in r.model.wire(1)
    assert (
        events[-1].kind == "answer" and events[-1].payload == "searched 64; the rest next turn"
    )
    # Two model calls, each 100 in and 10 out at $1/$2 per MTok; the refusals cost nothing.
    assert ledger.spend("session").cost_usd == pytest.approx(2 * (100 * 1.0 + 10 * 2.0) / 1e6)


def test_the_call_bound_is_checked_before_the_tool_is_known(tmp_path: Path) -> None:
    """`_plan`'s order (the module docstring's table): the call cap comes first, so a call past
    it is refused by the bound even when it also names no tool. Pinned before the bounds move
    to their own policy."""
    r = rig(
        tmp_path,
        [calls(*_finds(2), call("no_such_tool", "x")), answer("done")],
        config=_cfg(2),
    )
    r.run("search")
    line = r.session.state.result("x")
    assert line is not None and line.ran is False
    assert line.text.startswith("BOUND_REACHED{resource: tool_calls, used: 2, limit: 2")
    assert "UNKNOWN_TOOL" not in line.text


def test_one_bound_line_per_turn_however_many_calls_it_refuses(tmp_path: Path) -> None:
    r = rig(
        tmp_path,
        [
            calls(*_finds(3, prefix="a")),
            calls(*_finds(4, prefix="b")),
            answer("stopped"),
            calls(*_finds(4, prefix="c")),
        ],
        config=_cfg(4),
    )
    r.run("first turn")
    lines = payloads(r.session, "bound")
    assert lines == [
        {"resource": "tool_calls", "outcome": "refused", "used": 4, "limit": 4, "scope": "turn"}
    ]
    refused = [c for c in ("b1", "b2", "b3") if r.session.state.result(c).ran is False]  # type: ignore[union-attr]
    assert refused == ["b1", "b2", "b3"], "the count spans the turn's responses"
    # The next turn starts a fresh allowance, and its own refusal is its own line.
    r.run("second turn")
    assert all(r.session.state.ran(f"c{i}") for i in range(4))
    assert len(payloads(r.session, "bound")) == 1


def test_a_refused_call_does_not_count_and_an_invalid_one_does(tmp_path: Path) -> None:
    broken = call("grep", "bad")  # no query: INVALID_ARGUMENTS, and it cost a round trip
    r = rig(tmp_path, [calls(broken, *_finds(2))], config=_cfg(2))
    r.run("go")
    state = r.session.state
    assert "INVALID_ARGUMENTS" in state.result("bad").text  # type: ignore[union-attr]
    assert state.ran("f0")
    assert "BOUND_REACHED" in state.result("f1").text  # type: ignore[union-attr]


def test_zero_disables_the_bound(tmp_path: Path) -> None:
    r = rig(tmp_path, [calls(*_finds(70))], config=_cfg(0))
    r.run("search")
    assert len(r.graph.asked) == 70
    assert payloads(r.session, "bound") == []


def test_the_read_bound_refuses_once_the_turn_has_taken_in_its_allowance(
    tmp_path: Path,
) -> None:
    """Checked as each group is dispatched: a response's parallel reads are taken in
    together, and the next response meets the bound they reached."""
    graph = ScriptedGraph(big="x" * 400)  # ~115 tokens a result
    cfg = LoopConfig(max_result_tokens=200, max_tool_calls=64)
    script = [calls(*_finds(2, prefix="a")), calls(*_finds(2, prefix="b")), answer("enough")]
    r = rig(tmp_path, script, config=cfg, graph=graph)
    r.run("read a lot")
    state = r.session.state
    assert [state.ran(c) for c in ("a0", "a1", "b0", "b1")] == [True, True, False, False]
    assert len(graph.asked) == 2, "the refused reads were not dispatched"
    text = state.result("b0").text  # type: ignore[union-attr]
    assert text.startswith("BOUND_REACHED{resource: result_tokens, used: ")
    assert "limit: 200, scope: turn" in text
    assert [p["resource"] for p in payloads(r.session, "bound")] == ["result_tokens"]
    # A read refusal is not a call: the turn's count is the two that ran.
    r2 = rig(
        tmp_path / "second",
        [calls(*_finds(2, prefix="a")), calls(*_finds(3, prefix="b"))],
        config=LoopConfig(max_result_tokens=200, max_tool_calls=3),
        graph=ScriptedGraph(big="x" * 400),
    )
    r2.run("again")
    assert not any("tool_calls" in str(p["resource"]) for p in payloads(r2.session, "bound"))


# -- the per-response bounds and the ends of a turn ---------------------------


def test_a_second_coder_in_one_response_is_refused_and_recorded_as_sent(tmp_path: Path) -> None:
    first = call("coder", "c1", title="fix the flaky test", brief="fix the flaky test")
    second = call("coder", "c2", title="and the other one", brief="and the other one")
    r = rig(tmp_path, [calls(first, second), answer("one at a time")])
    r.run("fix both")
    state = r.session.state
    assert len(r.box.started) == 1, "never two in flight"
    assert stated(state.result("c2").text) == (  # type: ignore[union-attr]
        "BOUND_REACHED{resource: coder, used: 1, limit: 1, scope: response, "
        "reason: 'one coder delegation at a time; read the report of the one that ran'}"
    )
    # The fold keeps one coder call to an assistant line, so the second has its own line,
    # after the first's result — and both calls are exactly what was sent.
    c1, c2 = state.call("c1"), state.call("c2")
    assert c1 is not None and c2 is not None and c1.seq < c2.seq
    assert dict(c2.args) == dict(second.arguments)
    assert state.result("c1").seq < c2.seq  # type: ignore[union-attr]


def test_locate_past_the_fan_out_is_refused(tmp_path: Path) -> None:
    asked: list[str] = []

    class Scout:
        def locate(self, question: str, *, cancel: object) -> str:
            asked.append(question)
            return f"src/a.py:1 ({question})"

    locates = [call("locate", f"l{i}", question=f"where is {i}") for i in range(6)]
    r = rig(tmp_path, [calls(*locates)], locator=Scout())  # type: ignore[arg-type]
    r.run("find six things")
    state = r.session.state
    assert len(asked) == 4
    assert [state.ran(f"l{i}") for i in range(6)] == [True] * 4 + [False] * 2
    assert "resource: fan_out, used: 5, limit: 4, scope: response" in state.result("l4").text  # type: ignore[union-attr]


def test_the_spend_cap_stops_the_turn_before_the_next_request(tmp_path: Path) -> None:
    frontier = RungPrice(price_in=1.0, price_out=2.0)
    capped = CostLedger(
        PriceTable({"frontier": frontier}),
        BudgetCaps(hard_cost_cap_usd=0.0001),
        rungs=DEFAULT_RUNGS,
    )
    r = rig(tmp_path, [calls(*_finds(2)), answer("never sent")], ledger=capped)
    events = r.run("go")
    assert len(r.model.requests) == 1, "no request starts once the cap is crossed"
    assert all(r.session.state.ran(f"f{i}") for i in range(2)), "the calls it made are answered"
    assert events[-1].kind == "error" and str(events[-1].payload).startswith("BUDGET_CAP")


def test_a_call_is_charged_at_the_model_behind_the_rung_and_follows_a_switch(
    tmp_path: Path,
) -> None:
    """The ledger is keyed by catalog model, so the loop charges the model its rung
    runs on — never the rung's name, which has no row — and after `/rung zeus` the next call
    is priced at the new model. `answer()` is 100 tokens in and 10 out."""
    ledger = _ledger()
    r = rig(tmp_path, [answer("one"), answer("two")], ledger=ledger)
    r.run("first")
    assert ledger.spend("session").cost_usd == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)
    r.loop.reprefix(zeus_prefix(r.workspace, model="frontier-model-b"), model="frontier-b")
    r.run("second")
    assert ledger.spend("session").cost_usd == pytest.approx(
        (100 * 1.0 + 10 * 2.0 + 100 * 10.0 + 10 * 20.0) / 1e6
    )


def test_a_ledger_without_the_model_it_charges_is_refused(tmp_path: Path) -> None:
    from chickadee.errors import HarnessBug

    with pytest.raises(HarnessBug, match="pass model="):
        rig(tmp_path, ledger=_ledger(), charged_as=None)


def test_an_answer_cut_at_max_tokens_is_recorded_and_said(tmp_path: Path) -> None:
    from chickadee.schema import StopReason
    from chickadee.schema.adapter import Completion, Usage

    cut = Completion(
        text="the answer is",
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        latency_ms=1,
        stop_reason=StopReason.MAX_TOKENS,
    )
    r = rig(tmp_path, [cut])
    events = r.run("explain")
    assert [e.kind for e in events][-2:] == ["answer", "error"]
    assert r.session.state.turns()[-1].text == "the answer is"


def test_the_read_bound_is_asked_again_between_groups_of_one_response(tmp_path: Path) -> None:
    """A serial call splits a response into groups; the reads before it are taken in before
    the calls after it are dispatched."""
    graph = ScriptedGraph(big="x" * 400)  # ~115 tokens a result
    cfg = LoopConfig(max_result_tokens=200, max_tool_calls=64)
    asked = call("ask_user", "q1", question="go on? default: yes")
    later = call("grep", "b0", pattern="later")
    r = rig(tmp_path, [calls(*_finds(2, prefix="a"), asked, later)], config=cfg, graph=graph)
    r.run("read, ask, read")
    state = r.session.state
    assert [state.ran(c) for c in ("a0", "a1", "q1", "b0")] == [True, True, False, False]
    assert r.person.asked == [], "the question was not put once the turn had read its fill"
    assert "resource: result_tokens" in state.result("q1").text  # type: ignore[union-attr]


def test_a_call_noted_at_admit_and_refused_at_dispatch_carries_no_note(tmp_path: Path) -> None:
    """A repetition notice is for a call that is served. One the read bound then refuses as it
    is dispatched is answered with the refusal alone."""
    graph = ScriptedGraph(big="x" * 400)  # ~115 tokens a result
    # `repeat = 1` notes every call from the first, so the last is noted when admitted.
    cfg = LoopConfig(max_result_tokens=200, max_tool_calls=64, repeat=1)
    asked = call("ask_user", "q1", question="go on? default: yes")
    later = call("grep", "b0", pattern="later")
    r = rig(tmp_path, [calls(*_finds(2, prefix="a"), asked, later)], config=cfg, graph=graph)
    r.run("read, ask, read")
    refused = r.session.state.result("b0")
    assert refused is not None and not refused.ran
    assert refused.text.startswith("BOUND_REACHED{resource: result_tokens")
    assert "repetition" not in refused.text
