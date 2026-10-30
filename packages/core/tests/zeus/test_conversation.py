"""`Conversation` and `Conversations`, row by row of D158 decision 2's table.

The engine's pins (`tests/engine/test_session_lifecycle.py`) drive the same table through the
facade with a real loop. This drives the object alone, with a build and a step the test holds
open, so each phase can be reached and named exactly, and checks what the facade cannot see:
the phase itself, an unlisted pair being a `HarnessBug`, that nothing runs under the lock (a
build, a step or an episode that calls back in does not deadlock), and that a failed build
releases exactly what it opened.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from typing import Any, cast

import pytest

from chickadee.errors import HarnessBug, SessionError
from chickadee.zeus.conversation import Built, Conversation, Conversations, LiveTurn, Phase
from chickadee.zeus.loop import Event

WAIT_S = 20.0
SID = "S-conv"


class ConversationRig:
    """A build and a step the test controls. The step takes one queued word, as the loop's
    driver turn does, and may be held open until the test releases it."""

    def __init__(self) -> None:
        self.build_gate = threading.Event()
        self.build_gate.set()
        self.building = threading.Event()
        self.build_error: BaseException | None = None
        self.opened_resources: list[str] = []
        self.released: list[str] = []
        self.delivered: list[str] = []
        self.step_begun = threading.Event()
        self.step_gate = threading.Event()
        self.step_gate.set()
        self.step_error: BaseException | None = None
        self.on_step: Callable[[Conversation], None] | None = None
        self.on_build: Callable[[Conversation], None] | None = None

    def build(self, conversation: Conversation, stack: ExitStack) -> Built:
        self.building.set()
        if self.on_build is not None:
            self.on_build(conversation)
        assert self.build_gate.wait(WAIT_S), "the test never released the build"
        for name in ("session", "client"):
            self.opened_resources.append(name)
            stack.callback(self.released.append, name)
            if self.build_error is not None and name == "session":
                raise self.build_error

        def step() -> Iterator[Event]:
            word = conversation.stops.next_input()
            self.step_begun.set()
            if self.on_step is not None:
                self.on_step(conversation)
            assert self.step_gate.wait(WAIT_S), "the test never released the step"
            if self.step_error is not None:
                raise self.step_error
            if word is not None:
                self.delivered.append(word.text)
                yield Event("answer", 0, word.text)

        return Built(
            step=step,
            session=cast(Any, object()),
            ledger=cast(Any, object()),
            refresh=lambda: None,
        )


class FakeEpisode:
    def __init__(self, on_interrupt: Callable[[bool], None] | None = None) -> None:
        self.interrupted: list[bool] = []
        self._on_interrupt = on_interrupt

    def interrupt(self, hard: bool = False) -> None:
        self.interrupted.append(hard)
        if self._on_interrupt is not None:
            self._on_interrupt(hard)


@pytest.fixture
def rig() -> ConversationRig:
    return ConversationRig()


@pytest.fixture
def registry(rig: ConversationRig) -> Iterator[Conversations]:
    conversations = Conversations(rig.build)
    yield conversations
    conversations.close_all()


def opened_in_thread(registry: Conversations) -> tuple[threading.Thread, list[Any]]:
    out: list[Any] = []

    def run() -> None:
        try:
            out.append(registry.open(SID))
        except BaseException as exc:
            out.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out


def building(
    rig: ConversationRig, registry: Conversations
) -> tuple[Conversation, threading.Thread]:
    """A conversation held in `BUILDING`."""
    rig.build_gate.clear()
    thread, _ = opened_in_thread(registry)
    assert rig.building.wait(WAIT_S)
    conversation = registry.get(SID)
    assert conversation is not None
    return conversation, thread


def running(
    rig: ConversationRig, registry: Conversations, word: str = "go"
) -> tuple[Conversation, LiveTurn]:
    """A conversation in `RUNNING`, its step held open on `word`."""
    conversation = registry.open(SID)
    rig.step_gate.clear()
    live, queued = conversation.turn(word)
    assert queued is False
    assert rig.step_begun.wait(WAIT_S)
    assert conversation.phase is Phase.RUNNING
    return conversation, live


def idle_after(conversation: Conversation, live: LiveTurn, rig: ConversationRig) -> None:
    rig.step_gate.set()
    live.wait(WAIT_S)


# -- open and build ---------------------------------------------------------------------------


def test_none_open_building_then_idle(rig: ConversationRig, registry: Conversations) -> None:
    conversation, thread = building(rig, registry)
    assert conversation.phase is Phase.BUILDING
    rig.build_gate.set()
    thread.join(WAIT_S)
    assert conversation.phase is Phase.IDLE


def test_every_open_waits_for_the_one_build(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, first = building(rig, registry)
    second, out = opened_in_thread(registry)
    second.join(0.2)
    assert second.is_alive(), "a second open returned before the build ended"
    rig.build_gate.set()
    first.join(WAIT_S)
    second.join(WAIT_S)
    assert out == [conversation]


def test_building_stopped_then_held_then_idle(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, thread = building(rig, registry)
    conversation.interrupt()
    assert conversation.phase is Phase.BUILDING_STOPPED
    conversation.interrupt(hard=True)
    assert conversation.phase is Phase.BUILDING_STOPPED
    rig.build_gate.set()
    thread.join(WAIT_S)
    assert conversation.phase is Phase.HELD

    conversation.interrupt()
    assert conversation.phase is Phase.HELD
    live, queued = conversation.turn("kept")
    assert queued is True
    live.wait(WAIT_S)
    assert conversation.phase is Phase.IDLE
    assert rig.delivered == []

    live, queued = conversation.turn("next")
    live.wait(WAIT_S)
    assert queued is False
    assert rig.delivered == ["kept", "next"]


def test_a_failed_build_releases_exactly_what_it_opened_before_any_waiter_wakes(
    rig: ConversationRig, registry: Conversations
) -> None:
    rig.build_error = RuntimeError("no endpoint")
    conversation, opener = building(rig, registry)
    waiter, out = opened_in_thread(registry)
    waiter.join(0.2)
    assert waiter.is_alive(), "the waiter did not wait for the build"
    seen_at_release: list[bool] = []

    real = conversation._forget  # pyright: ignore[reportPrivateUsage]

    def forget(c: Conversation) -> None:
        seen_at_release.append(waiter.is_alive())
        real(c)

    conversation._forget = forget  # pyright: ignore[reportPrivateUsage]
    rig.build_gate.set()
    opener.join(WAIT_S)
    waiter.join(WAIT_S)

    assert rig.opened_resources == ["session"]
    assert rig.released == ["session"]
    assert seen_at_release == [True]
    assert conversation.phase is Phase.FAILED
    (raised,) = out
    assert isinstance(raised, SessionError)
    assert "no endpoint" in str(raised)
    # Forgotten: the next open builds anew.
    assert registry.get(SID) is None
    rig.build_error = None
    assert registry.open(SID) is not conversation


def test_the_opener_gets_the_builds_own_exception(
    rig: ConversationRig, registry: Conversations
) -> None:
    rig.build_error = ValueError("bad config")
    with pytest.raises(ValueError, match="bad config"):
        registry.open(SID)


def test_failed_interrupt_and_close_do_nothing(
    rig: ConversationRig, registry: Conversations
) -> None:
    rig.build_error = RuntimeError("no endpoint")
    conversation, thread = building(rig, registry)
    rig.build_gate.set()
    thread.join(WAIT_S)
    assert conversation.phase is Phase.FAILED

    conversation.interrupt(hard=True)
    conversation.close()
    assert conversation.phase is Phase.FAILED
    assert rig.released == ["session"]
    assert conversation.opened() is None


# -- turn() -----------------------------------------------------------------------------------


def test_idle_turn_running_then_idle(rig: ConversationRig, registry: Conversations) -> None:
    conversation = registry.open(SID)
    live, queued = conversation.turn("hi")
    live.wait(WAIT_S)
    assert queued is False
    assert conversation.phase is Phase.IDLE
    assert [e.payload for e in live.follow()] == ["hi"]


def test_running_turn_running(rig: ConversationRig, registry: Conversations) -> None:
    conversation, live = running(rig, registry)
    again, queued = conversation.turn("two")
    assert (again, queued) == (live, True)
    assert conversation.phase is Phase.RUNNING
    idle_after(conversation, live, rig)
    assert rig.delivered == ["go", "two"]


def test_stopping_turn_continuing_then_running(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, live = running(rig, registry)
    conversation.interrupt()
    assert conversation.phase is Phase.STOPPING
    again, queued = conversation.turn("instead")
    assert (again, queued) == (live, True)
    assert conversation.phase is Phase.CONTINUING
    conversation.turn("and this")
    assert conversation.phase is Phase.CONTINUING
    idle_after(conversation, live, rig)
    assert rig.delivered == ["go", "instead", "and this"]
    assert conversation.phase is Phase.IDLE


def test_closed_turn_is_a_session_error(rig: ConversationRig, registry: Conversations) -> None:
    conversation = registry.open(SID)
    conversation.close()
    assert conversation.phase is Phase.CLOSED
    with pytest.raises(SessionError, match="closed"):
        conversation.turn("late")
    assert not conversation.stops.has_input()


def test_a_word_typed_while_building_is_driven_after_the_build_in_order(
    rig: ConversationRig, registry: Conversations
) -> None:
    """T-3.162 moved this answer: `turn()` in `BUILDING` raised `HarnessBug`. The word is
    queued, the first gets the build's pending turn and the rest join it, and the build's own
    worker job delivers them once it ends."""
    conversation, thread = building(rig, registry)
    first, queued = conversation.turn("early")
    second, queued_too = conversation.turn("earlier still")
    assert (queued, queued_too, second is first) == (False, True, True)
    rig.build_gate.set()
    thread.join(WAIT_S)
    first.wait(WAIT_S)
    assert rig.delivered == ["early", "earlier still"]
    assert conversation.phase is Phase.IDLE


def test_a_word_typed_during_a_stopped_build_is_held(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, thread = building(rig, registry)
    conversation.interrupt()
    live, queued = conversation.turn("early")
    assert queued is True, "typed after the stop: it waits, as a word typed into HELD does"
    rig.build_gate.set()
    thread.join(WAIT_S)
    live.wait(WAIT_S)
    assert rig.delivered == [] and conversation.phase is Phase.IDLE
    assert conversation.stops.has_input(), "the word is kept for the next turn"
    conversation.turn("later")[0].wait(WAIT_S)
    assert rig.delivered == ["early", "later"]


def test_start_returns_before_the_build_ends(
    rig: ConversationRig, registry: Conversations
) -> None:
    """What a turn asks for. `open` still waits, for a caller that needs the loop."""
    rig.build_gate.clear()
    conversation = registry.start(SID)
    assert rig.building.wait(WAIT_S)
    assert conversation.phase is Phase.BUILDING
    live, _ = conversation.turn("hi")
    rig.build_gate.set()
    live.wait(WAIT_S)
    assert rig.delivered == ["hi"]


# -- interrupt() ------------------------------------------------------------------------------


def test_idle_interrupt_idle_stops_nothing(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation = registry.open(SID)
    conversation.interrupt(hard=True)
    assert conversation.phase is Phase.IDLE
    live, queued = conversation.turn("hi")
    live.wait(WAIT_S)
    assert queued is False
    assert rig.delivered == ["hi"]


def test_running_interrupt_stopping_then_idle_keeps_the_words(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, live = running(rig, registry)
    conversation.turn("queued")
    conversation.interrupt()
    assert conversation.phase is Phase.STOPPING
    idle_after(conversation, live, rig)
    assert conversation.phase is Phase.IDLE
    assert rig.delivered == ["go"]
    assert conversation.stops.pending() == ("queued",)
    assert conversation.stops.stopping()
    # The next turn clears the stop and delivers the kept word first.
    live, _ = conversation.turn("now")
    live.wait(WAIT_S)
    assert rig.delivered == ["go", "queued", "now"]


def test_continuing_interrupt_stopping(rig: ConversationRig, registry: Conversations) -> None:
    conversation, live = running(rig, registry)
    conversation.interrupt()
    conversation.turn("then")
    assert conversation.phase is Phase.CONTINUING
    conversation.interrupt()
    assert conversation.phase is Phase.STOPPING
    idle_after(conversation, live, rig)
    assert rig.delivered == ["go"]


def test_closed_interrupt_closed(rig: ConversationRig, registry: Conversations) -> None:
    conversation = registry.open(SID)
    conversation.close()
    conversation.interrupt()
    assert conversation.phase is Phase.CLOSED


# -- the boundary -----------------------------------------------------------------------------


@pytest.mark.parametrize("stop_first", [False, True], ids=["running", "continuing"])
def test_a_step_that_raises_ends_the_turn_idle(
    rig: ConversationRig, registry: Conversations, stop_first: bool
) -> None:
    rig.step_error = RuntimeError("the rung fell over")
    conversation, live = running(rig, registry)
    if stop_first:
        conversation.interrupt()
        conversation.turn("then")
    rig.step_gate.set()
    with pytest.raises(SessionError, match="the rung fell over"):
        live.wait(WAIT_S)
    assert conversation.phase is Phase.IDLE


# -- the episode ------------------------------------------------------------------------------


@pytest.mark.parametrize("hard", [False, True])
def test_attach_after_a_stop_interrupts_the_episode_at_once(
    rig: ConversationRig, registry: Conversations, hard: bool
) -> None:
    conversation, live = running(rig, registry)
    conversation.interrupt(hard=hard)
    episode = FakeEpisode()
    conversation.attach_episode(episode)
    assert episode.interrupted == [hard]
    idle_after(conversation, live, rig)


def test_a_stop_reaches_the_attached_episode_until_it_is_detached(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, live = running(rig, registry)
    episode = FakeEpisode()
    conversation.attach_episode(episode)
    assert episode.interrupted == []
    conversation.interrupt(hard=True)
    assert episode.interrupted == [True]
    conversation.detach_episode()
    conversation.interrupt()
    assert episode.interrupted == [True]
    idle_after(conversation, live, rig)


def test_attach_while_unbuilt_is_a_harness_bug(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, thread = building(rig, registry)
    with pytest.raises(HarnessBug):
        conversation.attach_episode(FakeEpisode())
    rig.build_gate.set()
    thread.join(WAIT_S)


# -- close() ----------------------------------------------------------------------------------


def test_close_stops_hard_interrupts_the_episode_and_releases_in_reverse(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, live = running(rig, registry)
    conversation.turn("queued")
    episode = FakeEpisode(on_interrupt=lambda _hard: rig.step_gate.set())
    conversation.attach_episode(episode)

    conversation.close()

    assert conversation.phase is Phase.CLOSED
    assert episode.interrupted == [True]
    assert conversation.stops.cancel.is_set()
    assert rig.released == ["client", "session"]
    live.wait(WAIT_S)
    assert rig.delivered == ["go"]
    assert conversation.opened() is None
    conversation.close()
    assert rig.released == ["client", "session"]


def test_close_waits_for_a_build_in_progress(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, opener = building(rig, registry)
    closer = threading.Thread(target=conversation.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive(), "close returned while the build ran"
    rig.build_gate.set()
    closer.join(WAIT_S)
    opener.join(WAIT_S)
    assert conversation.phase is Phase.CLOSED


def test_the_registry_forgets_a_closed_session_and_refuses_after_close_all(
    rig: ConversationRig, registry: Conversations
) -> None:
    first = registry.open(SID)
    registry.close(SID)
    assert first.phase is Phase.CLOSED
    assert registry.get(SID) is None
    second = registry.open(SID)
    assert second is not first

    registry.close_all()
    assert second.phase is Phase.CLOSED
    with pytest.raises(SessionError, match="closed"):
        registry.open(SID)


# -- nothing runs under the lock --------------------------------------------------------------


def test_a_step_that_calls_back_in_does_not_deadlock(
    rig: ConversationRig, registry: Conversations
) -> None:
    def back_in(conversation: Conversation) -> None:
        conversation.turn("typed from inside")
        conversation.interrupt()
        assert conversation.phase is Phase.STOPPING

    rig.on_step = back_in
    conversation = registry.open(SID)
    live, _ = conversation.turn("go")
    live.wait(WAIT_S)
    assert rig.delivered == ["go"]


def test_a_build_that_calls_back_in_does_not_deadlock(
    rig: ConversationRig, registry: Conversations
) -> None:
    rig.on_build = lambda conversation: conversation.interrupt()
    conversation = registry.open(SID)
    assert conversation.phase is Phase.HELD


def test_an_episode_that_calls_back_in_does_not_deadlock(
    rig: ConversationRig, registry: Conversations
) -> None:
    conversation, live = running(rig, registry)

    def once_more(hard: bool) -> None:
        if len(episode.interrupted) == 1:
            conversation.interrupt(hard)

    episode = FakeEpisode(on_interrupt=once_more)
    conversation.attach_episode(episode)

    conversation.interrupt()

    assert episode.interrupted == [False, False]
    assert conversation.phase is Phase.STOPPING
    idle_after(conversation, live, rig)


def test_a_word_reaching_a_failed_build_is_told_why_on_its_turn(
    rig: ConversationRig, registry: Conversations
) -> None:
    """A build can fail before the turn that started it arrives. That turn answers as
    the pending one does, so the caller sees one answer however the race fell."""
    rig.build_error = RuntimeError("no endpoint")
    conversation, thread = building(rig, registry)
    rig.build_gate.set()
    thread.join(WAIT_S)
    assert conversation.phase is Phase.FAILED

    live, queued = conversation.turn("late")
    assert queued is False
    assert [e.payload for e in live.follow()] == [
        "the session could not be opened: no endpoint"
    ]
    with pytest.raises(SessionError, match="no endpoint"):
        live.wait(WAIT_S)
    assert not conversation.stops.has_input(), "the word is not queued"


def test_a_release_that_fails_after_a_failed_build_is_noted_and_the_worker_still_stops(
    rig: ConversationRig, registry: Conversations
) -> None:
    """The failure path runs in a worker job whose result nobody reads, so a release
    that raises is noted on the failure every waiter gets, and the worker is shut down anyway.
    """
    rig.build_error = RuntimeError("no endpoint")
    built_for: list[Conversation] = []
    rig.on_build = built_for.append

    def build(conversation: Conversation, stack: ExitStack) -> Built:
        def broken() -> None:
            raise OSError("the lock would not let go")

        stack.callback(broken)
        return rig.build(conversation, stack)

    conversations = Conversations(build)
    try:
        with pytest.raises(RuntimeError, match="no endpoint") as raised:
            conversations.open(SID)
        assert any("the lock would not let go" in note for note in raised.value.__notes__)
        assert rig.released == ["session"], "what it could release, it did"
        (failed,) = built_for
        with pytest.raises(RuntimeError, match="shutdown"):
            failed._worker.submit(lambda: None)  # pyright: ignore[reportPrivateUsage]
    finally:
        conversations.close_all()


def test_opening_is_said_only_while_a_turn_waits_on_the_build(
    rig: ConversationRig, registry: Conversations
) -> None:
    """A host that opens a session to read it (a budget, a header) is not someone
    working on it; liveness says "opening the session" only for a pending turn."""
    conversation, thread = building(rig, registry)
    assert conversation.opening is False
    conversation.turn("hi")
    assert conversation.opening is True
    rig.build_gate.set()
    thread.join(WAIT_S)
