"""The wake: a job's exit, written to the log as a
`JOB_EXITED` notice, opens a turn with no user line, through one gate at every boundary.

Driven on `Conversation` alone, with a build whose word turn, wake turn and readiness the test
holds, as `test_conversation.py` drives the phases; and on the session log, for where a notice
written while a call waits is read.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import pytest

from chickadee.context import standing
from chickadee.context.request import assemble
from chickadee.context.session import Session
from chickadee.context.standing import Notice, NoticeTrigger, job_exited
from chickadee.schema import ToolCall
from chickadee.schema.adapter import Completion
from chickadee.zeus.conversation import Built, Conversation, Conversations, Phase
from chickadee.zeus.jobs import exit_unread
from chickadee.zeus.loop import Event

from .zeus_helpers import answer
from .zeus_helpers import rig as loop_rig

WAIT_S = 20.0
SID = "S-wake"


class WakeRig:
    """`unread` stands for an exit notice after the last answer; a wake turn reads it."""

    def __init__(self, max_wakes: int = 3) -> None:
        self.unread = False
        self.may_start = True
        self.turns: list[str] = []
        self.max_wakes = max_wakes
        self.done = threading.Event()

    def build(self, conversation: Conversation, stack: ExitStack) -> Built:
        def step() -> Iterator[Event]:
            words = conversation.stops.next_input()
            word = words.text if words is not None else None
            self.turns.append(f"word:{word}")
            self.unread = False  # the word's turn reads every notice before it
            self.done.set()
            yield Event("answer", 0, word or "")

        def wake() -> Iterator[Event]:
            self.turns.append("wake")
            self.unread = False
            self.done.set()
            yield Event("answer", 0, "woken")

        return Built(
            step=step,
            session=cast(Any, object()),
            ledger=cast(Any, object()),
            refresh=lambda: None,
            wake=wake,
            wake_ready=lambda: self.unread and self.may_start,
            max_wakes=self.max_wakes,
        )

    def exit(self, conversation: Conversation) -> None:
        """A job's waiter: the notice written, then the conversation told."""
        self.done.clear()
        self.unread = True
        conversation.wake_due()

    def settle(self, conversation: Conversation) -> None:
        deadline = time.monotonic() + WAIT_S
        while conversation.phase is not Phase.IDLE and time.monotonic() < deadline:
            time.sleep(0.01)
        assert conversation.phase is Phase.IDLE


@pytest.fixture
def rig() -> WakeRig:
    return WakeRig()


@pytest.fixture
def opened(rig: WakeRig) -> Iterator[Conversation]:
    registry = Conversations(rig.build)
    conversation = registry.open(SID)
    yield conversation
    registry.close_all()


def test_an_idle_exit_opens_a_turn_with_no_user_line(
    rig: WakeRig, opened: Conversation
) -> None:
    rig.exit(opened)
    assert rig.done.wait(WAIT_S)
    rig.settle(opened)
    assert rig.turns == ["wake"]


def test_a_stop_holds_and_the_exit_waits_for_the_persons_words(
    rig: WakeRig, opened: Conversation
) -> None:
    opened.interrupt()
    rig.exit(opened)
    time.sleep(0.3)
    assert rig.turns == [] and opened.phase is Phase.IDLE
    live, _ = opened.turn("go on")
    live.wait(WAIT_S)
    rig.settle(opened)
    assert rig.turns == ["word:go on"], "the notice is read with the words, no extra wake"


def test_an_exit_during_a_turn_meets_the_gate_at_its_boundary(
    rig: WakeRig, opened: Conversation
) -> None:
    gate = threading.Event()
    original = rig.build

    def build(conversation: Conversation, stack: ExitStack) -> Built:
        built = original(conversation, stack)

        def held() -> Iterator[Event]:
            assert gate.wait(WAIT_S)
            yield from built.step()

        return Built(
            step=held,
            session=built.session,
            ledger=built.ledger,
            refresh=built.refresh,
            wake=built.wake,
            wake_ready=built.wake_ready,
            max_wakes=built.max_wakes,
        )

    registry = Conversations(build)
    conversation = registry.open("S-held")
    try:
        live, _ = conversation.turn("first")
        rig.unread = True
        conversation.wake_due()  # RUNNING: it waits for the boundary
        rig.unread = True
        gate.set()
        live.wait(WAIT_S)
        rig.settle(conversation)
        # The word's turn read the notice it found after it, so the boundary had none left.
        assert rig.turns == ["word:first"]
        rig.exit(conversation)
        rig.settle(conversation)
        assert rig.turns == ["word:first", "wake"]
    finally:
        registry.close_all()


def test_an_exit_told_after_the_boundary_read_the_log_is_read_again(rig: WakeRig) -> None:
    original = rig.build
    late = threading.Event()

    def build(conversation: Conversation, stack: ExitStack) -> Built:
        built = original(conversation, stack)

        def ready() -> bool:
            answered = built.wake_ready()
            if rig.turns == ["word:first"] and not late.is_set():
                # The waiter, between this read and the end of the turn.
                late.set()
                rig.exit(conversation)
            return answered

        return Built(
            step=built.step,
            session=built.session,
            ledger=built.ledger,
            refresh=built.refresh,
            wake=built.wake,
            wake_ready=ready,
            max_wakes=built.max_wakes,
        )

    registry = Conversations(build)
    conversation = registry.open("S-late")
    try:
        live, _ = conversation.turn("first")
        live.wait(WAIT_S)
        rig.settle(conversation)
        assert late.is_set() and rig.turns == ["word:first", "wake"]
    finally:
        registry.close_all()


def test_words_waiting_at_a_boundary_go_first(rig: WakeRig, opened: Conversation) -> None:
    rig.unread = True
    live, _ = opened.turn("mine")
    live.wait(WAIT_S)
    rig.settle(opened)
    assert rig.turns[0] == "word:mine"


def test_the_fourth_wake_since_the_person_spoke_never_opens_and_words_reset_it(
    rig: WakeRig, opened: Conversation
) -> None:
    for _ in range(4):
        rig.exit(opened)
        rig.done.wait(1.0)
        rig.settle(opened)
    assert rig.turns == ["wake", "wake", "wake"]
    live, _ = opened.turn("hello")
    live.wait(WAIT_S)
    rig.settle(opened)
    rig.exit(opened)
    assert rig.done.wait(WAIT_S)
    rig.settle(opened)
    assert rig.turns[-2:] == ["word:hello", "wake"]


def test_a_ledger_that_may_not_start_opens_nothing(rig: WakeRig, opened: Conversation) -> None:
    rig.may_start = False
    rig.exit(opened)
    time.sleep(0.3)
    rig.settle(opened)
    assert rig.turns == []


def test_a_closed_conversation_takes_no_wake(rig: WakeRig) -> None:
    registry = Conversations(rig.build)
    conversation = registry.open("S-closed")
    registry.close_all()
    rig.exit(conversation)
    assert rig.turns == [] and conversation.phase is Phase.CLOSED


# -- the log ---------------------------------------------------------------------------------


def _notice(session: Session) -> None:
    session.record_notice(
        job_exited(
            job_id="j1",
            kind="task",
            line="pytest -q",
            how="exited 1 after 4s",
            moved=False,
            wrote=(),
            overwrote=(),
            alongside=(),
            tail="1 failed",
            handle=None,
        )
    )


def test_a_notice_written_while_a_call_waits_is_read_after_its_result(tmp_path: Path) -> None:
    session = Session.create(tmp_path, SID)
    session.record_user("run the suite")
    session.record_assistant("", [ToolCall(id="c1", name="shell", arguments={})])
    _notice(session)  # the waiter, while the call waits
    session.record_tool("c1", "running as j1", ran=True)
    assembled = assemble(session.state, session.blob)
    roles = [m.role for m in assembled.messages]
    assert roles == ["user", "assistant", "tool", "user"], "nothing between a call and result"
    assert "job j1 · task · pytest -q" in str(assembled.messages[-1].content)


def test_an_exit_is_unread_until_a_request_that_carried_it_is_answered(
    tmp_path: Path,
) -> None:
    session = Session.create(tmp_path, SID)
    session.record_user("hi")
    session.record_assistant("hello")
    carried = session.next_seq
    assert not exit_unread(session.state, carried)
    _notice(session)
    assert exit_unread(session.state, carried)
    session.record_assistant("the answer to a request sent before the notice")
    assert exit_unread(session.state, carried), "a later answer is not a read"
    assert not exit_unread(session.state, session.next_seq)


def test_a_notice_written_while_the_request_is_in_flight_stays_unread(tmp_path: Path) -> None:
    def in_flight(_request: object) -> Completion:
        _notice(r.session)  # the waiter, while the model answers
        return answer("an answer that never saw it")

    r = loop_rig(tmp_path, [in_flight, answer("j1 failed: 1 test")])
    assert not exit_unread(r.session.state, r.loop.read_before), "the log at open is read"
    r.run("hi")
    assert exit_unread(r.session.state, r.loop.read_before)
    r.run("and now?")
    assert not exit_unread(r.session.state, r.loop.read_before)


def test_a_run_s_end_wakes_and_a_pause_does_not(tmp_path: Path) -> None:
    """A run's end, its `handle` naming the run, opens a wake turn; a pause is said
    on the same trigger, naming no run, and wakes nobody, since whoever paused it knows."""
    session = Session.create(tmp_path, SID)
    session.record_user("hi")
    session.record_assistant("hello")
    carried = session.next_seq
    standing.notice(session, Notice(NoticeTrigger.RUN_ENDED, "PAUSED · run r1"))
    assert not exit_unread(session.state, carried), "a pause wakes nobody"
    standing.notice(session, standing.run_reported("r1", "COMPLETED · run r1", None))
    assert exit_unread(session.state, carried)


def test_a_run_s_end_while_a_call_waits_is_read_after_its_result(tmp_path: Path) -> None:
    """A run that ends while Zeus's turn is running is told as a trigger-4 notice,
    held by `assemble` until every waiting result is in, and read at his next request."""
    session = Session.create(tmp_path, SID)
    session.record_user("and the docs?")
    session.record_assistant("", [ToolCall(id="c1", name="read", arguments={})])
    standing.notice(session, standing.run_reported("r1", "COMPLETED · run r1", None))
    session.record_tool("c1", "the docs", ran=True)
    assembled = assemble(session.state, session.blob)
    roles = [m.role for m in assembled.messages]
    assert roles == ["user", "assistant", "tool", "user"], "nothing between a call and result"
    assert "COMPLETED · run r1" in str(assembled.messages[-1].content)
