"""Interruption and continuation.

Input typed during a tool call is queued and delivered at the next turn boundary — never
dropped, never injected mid-call. A soft stop lets the call in flight finish and cancels what
has not started; a hard stop also sets the token every dispatch received, which reaps a
running command. Neither discards queued input."""

from __future__ import annotations

import dataclasses
import errno
import threading
import time
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import Literal, cast

import pytest
from refusals import stated

from chickadee.budget.manager import BudgetCaps, CostLedger
from chickadee.budget.prices import PriceTable, RungPrice
from chickadee.budget.spend import Spend
from chickadee.client.errors import (
    ClientError,
    CredentialError,
    RequestCancelled,
    TransportExhausted,
)
from chickadee.client.request import ChatRequest, text_of
from chickadee.context.request import assemble, est_call, est_text
from chickadee.context.results import ToolFailure
from chickadee.context.session import Turn
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema.adapter import Completion, Usage
from chickadee.schema.interaction import Answer, AnswerSource, Question
from chickadee.zeus.loop import Event, ZeusLoop

from .zeus_helpers import (
    Rig,
    ScriptedGraph,
    ScriptedPerson,
    answer,
    call,
    calls,
    rig,
    zeus_prefix,
)


class TypingGraph(ScriptedGraph):
    """A graph that, while answering, does what the test asks of the person at that moment."""

    def __init__(self) -> None:
        super().__init__()
        self.during: list[object] = []
        self.loop: ZeusLoop | None = None
        self.hook = lambda loop: None

    def search(
        self,
        query: str,
        *,
        kind: Literal["text", "path"] = "text",
        paths: Sequence[str] = (),
        cancel: object = None,
    ) -> str:
        assert self.loop is not None
        self.hook(self.loop)
        return super().search(query, kind=kind, paths=paths, cancel=cancel)

    def impact(self, symbols: Sequence[str]) -> str | ToolFailure:
        return super().impact(symbols)


def _users(turns: Sequence[Turn]) -> list[str]:
    return [t.text for t in turns if t.role == "user"]


def test_input_typed_during_a_call_is_delivered_at_the_next_turn_boundary(
    tmp_path: Path,
) -> None:
    graph = TypingGraph()
    graph.hook = lambda loop: loop.type("also update the CHANGELOG")
    script = [calls(call("grep", "f1", pattern="retry")), answer("found it"), answer("noted")]
    r = rig(tmp_path, script, graph=graph)
    graph.loop = r.loop
    events = r.run("where is the retry logic?")
    state = r.session.state
    assert _users(state.turns()) == ["where is the retry logic?", "also update the CHANGELOG"]
    # Never injected mid-call: the request that followed the call does not carry it, and
    # the call's result sits directly after the call in the log.
    assert "CHANGELOG" not in r.model.wire(1)
    assert "CHANGELOG" in r.model.wire(2)
    answers = [e.payload for e in events if e.kind == "answer"]
    assert answers == ["found it", "noted"]
    assert r.loop.interrupts.pending() == ()


def test_a_soft_stop_finishes_the_call_in_flight_and_cancels_the_rest(tmp_path: Path) -> None:
    graph = TypingGraph()
    graph.hook = lambda loop: loop.interrupt()
    script = [calls(call("grep", "f1", pattern="x"), call("ask_user", "q1", question="which?"))]
    r = rig(tmp_path, script, graph=graph)
    graph.loop = r.loop
    events = r.run("look")
    state = r.session.state
    assert state.ran("f1"), "the call in flight completed"
    assert not state.ran("q1")
    assert stated(state.result("q1").text) == "CANCELLED{tool: ask_user}"  # type: ignore[union-attr]
    assert r.person.asked == [], "the cancelled question was never put"
    assert len(r.model.requests) == 1, "a stop ends the turn: no further model call"
    assert not any(e.kind == "answer" for e in events)
    assert state.pending_calls() == (), "every call is answered"


class StoppedWhileAsked(ScriptedPerson):
    """A person who is still deciding when the hard stop comes: the port is handed the
    stop's token, honours it, and lets go with an answer nobody gave."""

    def __init__(self) -> None:
        super().__init__()
        self.loop: ZeusLoop | None = None
        self.cancels: list[threading.Event | None] = []

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.asked.append(q)
        self.cancels.append(cancel)
        assert self.loop is not None
        self.loop.interrupt(hard=True)
        assert cancel is not None and cancel.wait(5), "the stop's token reached the port"
        return Answer(question_id=q.id, choice="too late", source=AnswerSource.HUMAN)


def test_a_hard_stop_during_ask_user_answers_it_cancelled_and_tells_the_port(
    tmp_path: Path,
) -> None:
    """The port's `ask` is told to cancel, and whatever it returns to let go, the
    question is answered `CANCELLED` — never the words it returned."""
    person = StoppedWhileAsked()
    r = rig(tmp_path, [calls(call("ask_user", "q1", question="which one?"))], person=person)
    person.loop = r.loop
    r.run("ask me")
    assert person.cancels == [r.loop.interrupts.cancel]
    assert stated(r.session.state.result("q1").text) == "CANCELLED{tool: ask_user}"  # type: ignore[union-attr]
    assert len(r.model.requests) == 1, "the stop ended the turn"


def test_a_stop_keeps_queued_input_and_the_next_turn_delivers_it_first(tmp_path: Path) -> None:
    graph = TypingGraph()

    def type_then_stop(loop: ZeusLoop) -> None:
        loop.type("and the docs")
        loop.interrupt(hard=True)

    graph.hook = type_then_stop
    r = rig(tmp_path, [calls(call("grep", "f1", pattern="x"))], graph=graph)
    graph.loop = r.loop
    r.run("look")
    assert r.loop.interrupts.pending() == ("and the docs",), "a stop discards nothing"
    graph.hook = lambda loop: None
    r.run("carry on")
    assert _users(r.session.state.turns()) == ["look", "and the docs", "carry on"]


def test_the_token_every_dispatch_receives_is_the_request_s(tmp_path: Path) -> None:
    seen: list[threading.Event | None] = []

    class Locate:
        def locate(self, question: str, *, cancel: threading.Event | None) -> str:
            seen.append(cancel)
            return "src/a.py:1 (the retry loop)"

    r = rig(
        tmp_path,
        [calls(call("locate", "l1", question="where do we retry?"))],
        locator=Locate(),
    )
    r.run("find the retry loop")
    assert seen == [r.loop.interrupts.cancel]
    assert all(req.cancel is r.loop.interrupts.cancel for req in r.model.requests)


def test_a_hard_stop_reaps_a_running_command(tmp_path: Path) -> None:
    script = [calls(call("shell", "s1", command="sleep 30"))]
    r = rig(tmp_path, script, shell=True)
    timer = threading.Timer(0.5, lambda: r.loop.interrupt(hard=True))
    timer.start()
    started = time.monotonic()
    try:
        r.run("wait for it")
    finally:
        timer.cancel()
    assert time.monotonic() - started < 15, "the command was reaped, not waited out"
    line = r.session.state.result("s1")
    assert line is not None
    assert line.text.startswith("CANCELLED{tool: shell, partial: h_")
    assert len(r.model.requests) == 1


def test_the_frame_and_the_turn_never_split_a_call_from_its_result(tmp_path: Path) -> None:
    """X-29 through the loop: queued input becomes a user line only once every call of the
    turn before it has its result."""
    graph = TypingGraph()
    graph.hook = lambda loop: loop.type("one more thing")
    script = [
        calls(call("grep", "f1", pattern="a"), call("grep", "f2", pattern="b")),
        answer("ok"),
    ]
    r = rig(tmp_path, script, graph=graph)
    graph.loop = r.loop
    r.run("search")
    request = r.model.requests[-1]
    roles = [m.role for m in request.messages]
    first_tool = roles.index("tool")
    assert roles[first_tool : first_tool + 2] == ["tool", "tool"]
    assert "one more thing" in text_of(request.messages[-1])


def test_a_failed_request_ends_the_turn_and_records_nothing_it_did_not_say(
    tmp_path: Path,
) -> None:
    from chickadee.client.errors import TransportExhausted

    def fail(_request: object):
        raise TransportExhausted("503", 3)

    r = rig(tmp_path, [fail])
    events = r.run("hello")
    assert [e.kind for e in events] == ["error"]
    assert "TransportExhausted" in str(events[0].payload)
    assert [t.role for t in r.session.state.turns()] == ["user"]


def test_inputs_not_yet_delivered_when_a_turn_stops_wait_in_order(tmp_path: Path) -> None:
    graph = TypingGraph()
    graph.hook = lambda loop: loop.interrupt()
    r = rig(tmp_path, [calls(call("grep", "f1", pattern="x"))], graph=graph)
    graph.loop = r.loop
    r.loop.type("typed before the turn")
    r.run("the turn's own words")
    assert _users(r.session.state.turns()) == ["typed before the turn"]
    assert r.loop.interrupts.pending() == ("the turn's own words",)
    graph.hook = lambda loop: None
    r.run("and then this")
    assert _users(r.session.state.turns()) == [
        "typed before the turn",
        "the turn's own words",
        "and then this",
    ]


def test_a_stop_during_a_group_cancels_the_calls_of_it_not_yet_started(tmp_path: Path) -> None:
    """A group runs one call after another, so a stop that arrives during its first call
    finds the rest not started: they are answered `CANCELLED` and never run."""
    graph = TypingGraph()
    graph.hook = lambda loop: loop.interrupt()
    script = [calls(call("grep", "f1", pattern="a"), call("grep", "f2", pattern="b"))]
    r = rig(tmp_path, script, graph=graph)
    graph.loop = r.loop
    r.run("look")
    state = r.session.state
    assert state.ran("f1") and not state.ran("f2")
    assert stated(state.result("f2").text) == "CANCELLED{tool: grep}"  # type: ignore[union-attr]
    assert graph.asked == [("grep", ("a", ()))], "the second call never ran"
    assert state.pending_calls() == ()


def test_a_group_runs_on_the_thread_that_runs_the_turn(tmp_path: Path) -> None:
    """The graph's SQLite connection refuses every thread but the one that opened it, and the
    scout holds one conversation: a group of reads is a group, not a thread pool."""
    threads: list[int] = []

    class Graph(ScriptedGraph):
        def search(
            self,
            query: str,
            *,
            kind: Literal["text", "path"] = "text",
            paths: Sequence[str] = (),
            cancel: object = None,
        ) -> str:
            threads.append(threading.get_ident())
            return super().search(query, kind=kind, paths=paths, cancel=cancel)

    group = [call("grep", f"f{n}", pattern=str(n)) for n in range(3)]
    r = rig(tmp_path, [calls(*group)], graph=Graph())
    r.run("look")
    assert threads == [threading.get_ident()] * 3
    assert all(r.session.state.ran(f"f{n}") for n in range(3))


def _typed_during_the_first_answer(r_loop: list[ZeusLoop]) -> Completion:
    r_loop[0].type("A: also check the docs")
    r_loop[0].type("B: and keep the API stable")
    return answer("ok")


def test_input_waiting_behind_a_turn_that_fails_is_kept(tmp_path: Path) -> None:
    """Z-6 holds when a turn ends by an error, not only by a stop: what was typed and not yet
    delivered waits for the next turn, in the order it was typed."""
    holder: list[ZeusLoop] = []

    def broken(request: ChatRequest) -> Completion:
        raise RuntimeError("the transport broke")

    script = [lambda request: _typed_during_the_first_answer(holder), broken]
    r = rig(tmp_path, script)
    holder.append(r.loop)
    with pytest.raises(RuntimeError, match="transport broke"):
        r.run("start")
    assert _users(r.session.state.turns()) == ["start", "A: also check the docs"]
    assert r.loop.interrupts.pending() == ("B: and keep the API stable",)


def test_a_reader_that_stops_reading_loses_no_input(tmp_path: Path) -> None:
    """The stream is the live view and may be dropped; the person's words may not be. A
    consumer that closes the turn's stream mid-way leaves what it had not delivered queued."""
    holder: list[ZeusLoop] = []
    script = [lambda request: _typed_during_the_first_answer(holder), answer("noted A")]
    r = rig(tmp_path, script)
    holder.append(r.loop)
    # A generator: what a consumer that stops reading closes, or drops for the collector to.
    stream = cast("Generator[Event, None, None]", r.loop.turn("start"))
    for event in stream:
        if event.kind == "answer" and event.payload == "noted A":
            break
    stream.close()
    assert _users(r.session.state.turns()) == ["start", "A: also check the docs"]
    assert r.loop.interrupts.pending() == ("B: and keep the API stable",)


def test_a_stream_that_ends_at_the_delegation_hands_the_run_back(tmp_path: Path) -> None:
    """`open` took the repository lease and the run has done nothing. A consumer that stops
    reading at the delegation never asks for the report, so the run is abandoned, which
    releases the lease, and it is never run."""
    coder = call("coder", "c1", title="add a retry", brief="add a retry")
    r = rig(tmp_path, [calls(coder)])
    stream = cast("Generator[Event, None, None]", r.loop.turn("go"))
    for event in stream:
        if event.kind == "delegation":
            break
    stream.close()
    assert (r.box.abandoned, r.box.ran) == (["R-1"], [])
    # The call still waits for its answer: the next turn's repair gives it one.
    assert r.session.state.pending_calls() == ("c1",)


def test_a_run_that_reported_is_not_abandoned(tmp_path: Path) -> None:
    coder = call("coder", "c1", title="add a retry", brief="add a retry")
    r = rig(tmp_path, [calls(coder), answer("built")])
    r.run("go")
    assert (r.box.abandoned, r.box.ran) == ([], ["R-1"])


# -- a call that fails after it was sent is charged ------------------------


def _ledger() -> CostLedger:
    return CostLedger(
        PriceTable({"frontier": RungPrice(price_in=1.0, price_out=2.0)}),
        BudgetCaps(hard_cost_cap_usd=100.0),
        rungs=DEFAULT_RUNGS,
    )


def _raises(exc: ClientError) -> Callable[[ChatRequest], Completion]:
    def send(_request: ChatRequest) -> Completion:
        raise exc

    return send


def test_a_call_that_fails_after_it_was_sent_is_charged_its_estimated_input(
    tmp_path: Path,
) -> None:
    """A provider may bill a request that then times out. Its input is charged, estimated and
    marked so; it streamed nothing, so its output is zero; the turn still ends with the
    error."""
    ledger = _ledger()
    r = rig(
        tmp_path,
        [_raises(TransportExhausted("deadline exceeded", attempts=2, billed_attempts=(0,)))],
        ledger=ledger,
    )

    events = r.run("look at the auth module")

    spent = ledger.spend("session")
    assert spent.tokens_in > 0
    assert spent.tokens_out == 0
    assert spent.estimated_calls == 1, "one charge for the request, not one per attempt"
    assert spent.cost_usd > 0
    assert [e.kind for e in events][-1] == "error"
    assert "TransportExhausted" in str(events[-1].payload)


@pytest.mark.parametrize(
    "exc",
    [
        CredentialError("api_key_env ZEUS_KEY is unset"),
        RequestCancelled(attempts=0),
        TransportExhausted("connect error: refused", attempts=3),
        TransportExhausted("HTTP 400: context too long", attempts=1),
        RequestCancelled(attempts=1),
    ],
    ids=[
        "never-had-a-credential",
        "cancelled-before-any-attempt",
        "never-connected",
        "refused-by-status",
        "cancelled-in-backoff-after-a-503",
    ],
)
def test_a_call_no_provider_billed_is_not_charged(tmp_path: Path, exc: ClientError) -> None:
    """Attempts are not billing: a connect error, a status refused and a cancel after a 503
    were all attempts, and none was billed. Charging them would let an outage spend the
    cap."""
    ledger = _ledger()
    r = rig(tmp_path, [_raises(exc)], ledger=ledger)

    r.run("look at the auth module")

    assert ledger.spend("session").cost_usd == 0
    assert ledger.spend("session").estimated_calls == 0


def test_a_failed_calls_estimate_counts_what_assembly_counted(tmp_path: Path) -> None:
    """The estimate is what was sent: the system prefix, the tool schemas as the box weighs
    them (`request_tokens`), and the history as `assemble` weighed it, tool-call arguments
    included, not only its text."""
    ledger = _ledger()
    expected: dict[str, int] = {}
    held: list[Rig] = []

    def fails(request: ChatRequest) -> Completion:
        r = held[0]
        history = assemble(r.session.state, r.session.blob).tokens
        schemas = sum(
            est_call(t.name, t.schema) + est_text(t.description) for t in request.tools
        )
        expected["sent"] = history + est_text(text_of(request.messages[0])) + schemas
        expected["text_only"] = sum(est_text(text_of(m)) for m in request.messages) + schemas
        raise TransportExhausted("deadline exceeded", attempts=1, billed_attempts=(0,))

    big = "where is the session token refreshed " * 300
    r = rig(tmp_path, [calls(call("grep", pattern=big)), fails], ledger=ledger)
    held.append(r)

    r.run("look at the auth module")

    charged = ledger.spend("session").tokens_in - 100  # the first call's own usage
    assert charged == expected["sent"]
    assert charged > expected["text_only"] + 1000, "the tool call's arguments were counted"


def test_each_billed_attempt_is_charged_in_one_charge(tmp_path: Path) -> None:
    """Three attempts got back a body that was not a completion: each sent the request and
    each may be billed. The input is charged three times over, in one `spend` line.

    Both runs are built by one helper over one workspace, so only `billed_attempts` differs.
    The workspace path is part of the prefix (`env()`), and the estimate counts characters:
    two rigs in directories of different lengths sent requests a token apart."""
    workspace = tmp_path / "ws"

    def spent(billed: int) -> Spend:
        ledger = _ledger()
        rig(
            tmp_path / f"billed-{billed}",
            [
                _raises(
                    TransportExhausted("malformed", attempts=3, billed_attempts=(0,) * billed)
                )
            ],
            ledger=ledger,
            workspace=workspace,
        ).run("look at the auth module")
        return ledger.spend("session")

    once, thrice = spent(1), spent(3)

    assert thrice.tokens_in == 3 * once.tokens_in
    assert thrice.estimated_calls == 1


def test_a_failed_calls_streamed_output_is_charged_as_output(tmp_path: Path) -> None:
    """Streaming makes known what an attempt had produced when it stopped:
    each billed attempt's streamed tokens are charged as output, beside its input."""
    ledger = _ledger()
    failing = TransportExhausted("no token within 60s", attempts=2, billed_attempts=(40, 25))
    rig(tmp_path, [_raises(failing)], ledger=ledger).run("look at the auth module")

    spent = ledger.spend("session")
    assert spent.tokens_out == 65
    assert spent.estimated_calls == 1


def test_a_reprefixed_loop_estimates_its_new_prefix(tmp_path: Path) -> None:
    """The prefix's estimate is taken once per prefix, so `reprefix` must take it
    again: a failed call after it is charged for the prefix it actually sent."""
    full, bare = _ledger(), _ledger()
    failing = TransportExhausted("deadline exceeded", attempts=1, billed_attempts=(0,))
    rig(tmp_path / "full", [_raises(failing)], ledger=full).run("look")
    r = rig(tmp_path / "bare", [_raises(failing)], ledger=bare)
    r.loop.reprefix(dataclasses.replace(zeus_prefix(r.workspace), tools=()), model="frontier")
    r.run("look")

    assert 0 < bare.spend("session").tokens_in < full.spend("session").tokens_in - 500


def _full_disk(
    model: str, usage: Usage, cost_usd: float, rung: str, role: str | None = None
) -> None:
    """The session log's recorder, on a disk that is full: the ledger has counted the charge
    by the time it is called, as the session's does."""
    raise OSError(errno.ENOSPC, "No space left on device")


def test_a_charge_that_cannot_be_written_is_told_beside_the_calls_error(
    tmp_path: Path,
) -> None:
    """The call's failure is what the person is told; the full disk that kept its `spend`
    line off the log is told with it, not in its place, and the turn ends as for any failed
    call."""
    ledger = CostLedger(
        PriceTable({"frontier": RungPrice(price_in=1.0, price_out=2.0)}),
        BudgetCaps(hard_cost_cap_usd=100.0),
        record=_full_disk,
        rungs=DEFAULT_RUNGS,
    )
    r = rig(
        tmp_path,
        [_raises(TransportExhausted("deadline exceeded", attempts=1, billed_attempts=(0,)))],
        ledger=ledger,
    )

    events = r.run("look at the auth module")

    said = str(events[-1].payload)
    assert events[-1].kind == "error"
    assert said.startswith("TransportExhausted: ")
    assert "deadline exceeded" in said
    assert "could not be written to the session log" in said
    assert "No space left on device" in said
    assert ledger.spend("session").estimated_calls == 1, "the cap still counts it"


def test_a_cancel_after_a_send_does_not_say_nothing_was_sent(tmp_path: Path) -> None:
    """Nor how many attempts were made: a retry is silent to the person."""
    ledger = _ledger()
    r = rig(
        tmp_path, [_raises(RequestCancelled(attempts=1, billed_attempts=(0,)))], ledger=ledger
    )

    events = r.run("look at the auth module")

    assert ledger.spend("session").estimated_calls == 1
    said = str(events[-1].payload)
    assert "before it was sent" not in said
    assert said.endswith("CANCELLED: the request was stopped before its reply")
    assert "attempt" not in said


# -- a driver's turns ------------------------------------------------------


def test_a_driver_delivers_queued_words_one_turn_each_in_typed_order(tmp_path: Path) -> None:
    """A driver queues the person's words and calls `turn()` once per word; each call takes
    the oldest, so they arrive in the order they were typed."""
    r = rig(tmp_path, [answer("one"), answer("two"), answer("three")])
    for word in ("first", "second", "third"):
        r.loop.type(word)

    for _ in range(3):
        list(r.loop.turn())

    assert [t.text for t in r.session.state.turns() if t.role == "user"] == [
        "first",
        "second",
        "third",
    ]
    assert r.loop.interrupts.pending() == ()
    assert list(r.loop.turn()) == [], "an empty queue is no turn"


def test_a_stop_between_two_driver_turns_holds_until_the_driver_clears_it(
    tmp_path: Path,
) -> None:
    """The stop is the driver's: a driver's turn never clears it. While it holds, `turn()`
    delivers nothing and the queue keeps its words, in order."""
    r = rig(tmp_path, [answer("one"), answer("two"), answer("three")])
    for word in ("first", "second", "third"):
        r.loop.type(word)

    list(r.loop.turn())
    r.loop.interrupt()

    assert list(r.loop.turn()) == []
    assert r.loop.interrupts.pending() == ("second", "third"), "a stop discards nothing"
    said = [t.text for t in r.session.state.turns() if t.role == "user"]
    assert said == ["first"]

    r.loop.interrupts.begin_turn()  # the person starts a turn
    list(r.loop.turn())
    said = [t.text for t in r.session.state.turns() if t.role == "user"]
    assert said == ["first", "second"]


def test_a_driver_turn_that_fails_before_its_word_is_said_gives_the_word_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A word not yet on the log goes back to the front of the queue."""
    r = rig(tmp_path)
    r.loop.type("first")
    r.loop.type("second")

    def broken(_text: str) -> None:
        raise OSError("the log cannot be written")

    monkeypatch.setattr(r.loop, "_open", broken)
    with pytest.raises(OSError):
        list(r.loop.turn())

    assert r.loop.interrupts.pending() == ("first", "second")


def test_a_hand_driven_turn_still_starts_unstopped(tmp_path: Path) -> None:
    """`turn(text)` keeps its own behaviour: a person driving the loop by hand starts each
    turn unstopped, and words queued before it are delivered first."""
    r = rig(tmp_path, [answer("one"), answer("two")])
    r.loop.interrupt()
    r.loop.type("typed earlier")

    list(r.loop.turn("now"))

    said = [t.text for t in r.session.state.turns() if t.role == "user"]
    assert said == ["typed earlier", "now"]


def test_a_stop_that_lands_while_a_driver_turn_repairs_delivers_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A word is opened only to be answered: a stop landing after the word is taken but
    before it is said leaves it queued, with no user line on the log."""
    r = rig(tmp_path, [answer("one")])
    r.loop.type("first")
    record_prefix = r.loop._record_prefix  # pyright: ignore[reportPrivateUsage]

    def stopped_meanwhile() -> None:
        record_prefix()
        r.loop.interrupt()

    monkeypatch.setattr(r.loop, "_record_prefix", stopped_meanwhile)
    list(r.loop.turn())

    assert [t.text for t in r.session.state.turns() if t.role == "user"] == []
    assert r.loop.interrupts.pending() == ("first",)
