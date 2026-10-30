"""The dispatcher: a pipeline runs its rules in order per
stage; each verdict's effect; admit before execute; scopes reset their own counters."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

import pytest

from chickadee.errors import HarnessBug
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import Answer as InteractionAnswer
from chickadee.schema.interaction import Question
from chickadee.tools.catalogue import CallCap, Valid
from chickadee.tools.context import ToolContext
from chickadee.tools.rules import (
    Answered,
    AskPerson,
    BoundReached,
    CallOutcome,
    Declined,
    Noted,
    NotOffered,
    Pass,
    Pending,
    Pipeline,
    Refuse,
    ScopeState,
    Slot,
    Stop,
    Verdict,
)
from chickadee.tools.sink import Counters
from chickadee.tools.tool import Answer, TextAnswer

from .tools_helpers import KeepSink, call, dispatcher, echo_tool


@dataclass
class Says:
    """A rule that says one thing and remembers it was asked."""

    name: str
    verdict: Verdict
    asked: list[str] = field(default_factory=list[str])
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        self.asked.append(self.name)
        return self.verdict


def _pipeline(*admit: object, dispatch: tuple[object, ...] = ()) -> Pipeline:
    return Pipeline(
        admit=(*admit, Valid()),  # type: ignore[arg-type]
        dispatch=dispatch,  # type: ignore[arg-type]
        counts_at=("turn",),
    )


def test_admit_rules_run_in_order_and_stop_at_the_first_that_decides() -> None:
    asked: list[str] = []
    rules = (
        Says("a", Pass(), asked),
        Says("b", Noted("n"), asked),
        Says("c", Refuse(BoundReached("x", 1, 1, "turn")), asked),
        Says("d", Pass(), asked),
    )
    runs: list[str] = []
    d = dispatcher([echo_tool(runs=runs)], pipeline=_pipeline(*rules))
    outcome = d.call(call())
    assert asked == ["a", "b", "c"]
    assert runs == []
    assert outcome.answer is None
    assert isinstance(outcome.pending.decided, Refuse)


def test_a_refusal_counts_unless_it_says_it_does_not() -> None:
    counted = dispatcher(
        [echo_tool()], pipeline=_pipeline(Says("r", Refuse(BoundReached("x", 1, 1, "t"))))
    )
    counted.call(call())
    assert counted.state.calls("turn") == 1
    free = dispatcher(
        [echo_tool()],
        pipeline=_pipeline(Says("r", Refuse(BoundReached("x", 1, 1, "t"), counts=False))),
    )
    free.call(call())
    assert free.state.calls("turn") == 0


def test_answered_is_answered_without_running() -> None:
    runs: list[str] = []
    d = dispatcher([echo_tool(runs=runs)], pipeline=_pipeline(Says("a", Answered("cached"))))
    outcome = d.call(call())
    assert runs == [] and outcome.answer is None
    assert isinstance(outcome.pending.decided, Answered)
    assert d.state.calls("turn") == 1


def test_noted_runs_and_carries_its_note() -> None:
    d = dispatcher([echo_tool()], pipeline=_pipeline(Says("n", Noted("seen twice"))))
    outcome = d.call(call())
    assert isinstance(outcome.answer, TextAnswer)
    assert outcome.pending.notes == ("seen twice",)


def test_a_stop_answers_this_call_and_every_later_one_until_a_scope_opens() -> None:
    runs: list[str] = []
    stop = Stop("cap", BoundReached("tool_calls", 1, 1, "call"))
    rule = Says("s", stop)
    d = dispatcher([echo_tool(runs=runs)], pipeline=_pipeline(rule), level="call")
    first = d.call(call(cid="c1"))
    second = d.call(call(cid="c2"))
    assert runs == []
    assert first.pending.decided == stop and second.pending.decided == stop
    assert rule.asked == ["s"], "after a stop no rule is asked: the loop has ended"
    assert d.state.calls("turn") == 0, "a stopped call was never made, so it does not count"
    d.open_scope("call")
    assert d.stopped is None


def test_every_call_of_a_response_is_admitted_and_counted_before_any_runs() -> None:
    runs: list[str] = []
    d = dispatcher([echo_tool(runs=runs)], pipeline=_pipeline())
    pending = d.admit([call(cid="a"), call(cid="b")])
    assert d.state.calls("turn") == 2 and runs == []
    for p in pending:
        d.execute(p)
    assert runs == ["a", "b"]


def test_a_dispatch_refusal_that_does_not_count_uncounts_the_call() -> None:
    late = Says("late", Refuse(BoundReached("result_tokens", 9, 9, "turn"), counts=False))
    late.stage = "dispatch"
    runs: list[str] = []
    d = dispatcher([echo_tool(runs=runs)], pipeline=_pipeline(dispatch=(late,)))
    (p,) = d.admit([call()])
    assert d.state.calls("turn") == 1
    outcome = d.execute(p)
    assert runs == [] and outcome.answer is None
    assert d.state.calls("turn") == 0


def test_scopes_reset_their_own_counters_from_the_sinks_replay() -> None:
    sink = KeepSink(replays={"packet": Counters(calls=5, result_tokens=40)})
    pipeline = Pipeline(admit=(Valid(),), counts_at=("attempt", "packet"))
    d = dispatcher([echo_tool()], pipeline=pipeline, sink=sink, level="packet")
    d.open_scope("attempt")
    d.call(call())
    assert (d.state.calls("packet"), d.state.calls("attempt")) == (6, 1)
    assert d.state.tokens("packet") == 40
    d.open_scope("attempt")
    assert (d.state.calls("packet"), d.state.calls("attempt")) == (6, 0)


def test_a_name_not_offered_is_unknown_with_its_nearest() -> None:
    from chickadee.tools.catalogue import Offered

    d = dispatcher([echo_tool()], pipeline=_pipeline(Offered()))
    outcome = d.call(call("ecko"))
    decided = outcome.pending.decided
    assert isinstance(decided, Refuse) and isinstance(decided.failure, NotOffered)
    assert decided.failure.nearest == "echo" and decided.failure.offered == ("echo",)
    assert d.state.calls("turn") == 1


def test_a_tool_needing_an_absent_port_is_not_offered() -> None:
    reader = echo_tool("reader", needs=frozenset({"files"}))
    d = dispatcher([echo_tool(), reader])
    assert [s.name for s in d.offered()] == ["echo"]


def test_a_scope_can_narrow_the_ports_present() -> None:
    ctx = ToolContext(role="test", cancel=__import__("threading").Event())
    reader = echo_tool("waits", needs=frozenset({"files"}))
    d = dispatcher([reader], ctx=ctx)
    assert d.offered() == ()


def test_a_tool_is_offered_and_called_under_its_shown_name() -> None:
    runs: list[str] = []
    d = dispatcher([echo_tool("find", runs=runs)], shown_as={"find": "grep"})
    assert [s.name for s in d.offered()] == ["grep"]
    outcome = d.call(call("grep"))
    assert runs == ["c1"] and outcome.pending.tool == "find"


def test_a_streaming_tool_runs_through_its_roles_executor() -> None:
    runs: list[str] = []

    def stream(pending: Pending) -> Answer:
        return TextAnswer(f"streamed {pending.call.id}", "bytes")

    d = dispatcher([echo_tool(runs=runs)], executors={"echo": stream})
    outcome = d.call(call())
    assert runs == [] and isinstance(outcome.answer, TextAnswer)
    assert outcome.answer.text == "streamed c1"


def test_schedule_groups_parallel_reads_and_runs_the_rest_alone() -> None:
    d = dispatcher(
        [
            echo_tool("a"),
            echo_tool("b"),
            echo_tool("s", parallel="serial"),
            echo_tool("x", parallel="exclusive"),
        ]
    )
    calls = [call("a", cid="1"), call("b", cid="2"), call("s", cid="3"), call("a", cid="4")]
    pending = d.admit(calls)
    batches = d.schedule(pending)
    assert [(b.parallel, [p.call.id for p in b.pending]) for b in batches] == [
        ("any", ["1", "2"]),
        ("serial", ["3"]),
        ("any", ["4"]),
    ]


def test_an_observer_that_raises_never_fails_the_call() -> None:
    @dataclass
    class Boom:
        def observe(self, outcome: CallOutcome) -> None:
            raise RuntimeError("telemetry is down")

    sink = KeepSink()
    pipeline = Pipeline(admit=(Valid(),), observe=(Boom(),), counts_at=("turn",))
    d = dispatcher([echo_tool()], pipeline=pipeline, sink=sink)
    outcome = d.call(call())
    assert isinstance(outcome.answer, TextAnswer) and sink.records == [outcome]


@pytest.mark.parametrize(("answer", "runs_it"), [(True, True), (False, False)])
def test_a_confirm_is_asked_through_the_user_port(answer: bool, runs_it: bool) -> None:
    @dataclass
    class Person:
        asked: list[str] = field(default_factory=list[str])

        def ask(
            self, question: Question, *, cancel: threading.Event | None = None
        ) -> InteractionAnswer:
            raise AssertionError("a confirmation never asks a question")

        def confirm(self, question: str, tier: Tier) -> bool:
            self.asked.append(question)
            return answer

    person = Person()
    runs: list[str] = []
    d = dispatcher(
        [echo_tool(runs=runs)],
        pipeline=_pipeline(Says("ask", AskPerson("run echo?", Tier.T1))),
        ctx=ToolContext(role="test", user=person),
    )
    outcome = d.call(call())
    assert person.asked == ["run echo?"]
    assert (runs == ["c1"]) is runs_it
    if not runs_it:
        decided = outcome.pending.decided
        assert isinstance(decided, Refuse) and isinstance(decided.failure, Declined)


def test_with_nobody_to_ask_a_confirm_is_declined() -> None:
    d = dispatcher([echo_tool()], pipeline=_pipeline(Says("ask", AskPerson("?", Tier.T1))))
    decided = d.call(call()).pending.decided
    assert isinstance(decided, Refuse) and isinstance(decided.failure, Declined)


def test_a_pipeline_with_an_empty_slot_never_opens() -> None:
    with pytest.raises(HarnessBug, match="unfilled slots"):
        dispatcher([echo_tool()], pipeline=_pipeline(Slot("intent")))


def test_a_filled_slot_runs_the_roles_rule() -> None:
    rule = Says("intent", Pass())
    pipeline = _pipeline(Slot("intent")).fill({"intent": rule})
    dispatcher([echo_tool()], pipeline=pipeline).call(call())
    assert rule.asked == ["intent"]


def test_a_pipeline_reading_a_limit_it_was_not_given_never_opens() -> None:
    with pytest.raises(HarnessBug, match=r"zeus\.max_tool_calls"):
        dispatcher([echo_tool()], pipeline=_pipeline(CallCap("turn", "zeus.max_tool_calls")))


def test_a_pipeline_that_never_validates_never_opens() -> None:
    with pytest.raises(HarnessBug, match="never validates"):
        dispatcher([echo_tool()], pipeline=Pipeline(counts_at=("turn",)))


def test_invalid_arguments_are_refused_and_count() -> None:
    runs: list[str] = []
    d = dispatcher([echo_tool(runs=runs)])
    outcome = d.call(call(text=""))
    decided = outcome.pending.decided
    assert isinstance(decided, Refuse)
    assert decided.failure.failure == "INVALID_ARGUMENTS"
    assert runs == [] and d.state.calls("turn") == 1


def test_every_context_field_but_what_a_call_runs_under_is_a_port() -> None:
    """`ToolContext.has` reads a port by its field's name, so the names are the contract.
    `role`, `cancel`, `effect_log`, `constrained` and `hooks` are context,
    not ports."""
    from dataclasses import fields

    from chickadee.tools.tool import PORTS

    context = {"role", "cancel", "effect_log", "constrained", "hooks"}
    names = {f.name for f in fields(ToolContext)} - context
    assert names <= PORTS


def test_measured_by_reaches_only_rules_that_measure() -> None:
    """`Pipeline.measured_by` sets `measure` on a dataclass rule that declares one, and leaves
    every other result rule as it is, a rule that is not a dataclass included."""
    from chickadee.tools.catalogue import Account

    class Plain:
        def apply(
            self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext
        ) -> CallOutcome:
            return outcome

    def measure(outcome: CallOutcome) -> int:
        return 7

    plain = Plain()
    pipeline = Pipeline(admit=(Valid(),), result=(plain, Account(("turn",)))).measured_by(
        measure
    )
    first, second = pipeline.result
    assert first is plain
    assert isinstance(second, Account) and second.measure is measure
