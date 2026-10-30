"""The session's lifecycle, one case per row of D158 decision 2's table.

Landed on the code before the move, and passing unchanged after it: each case drives the facade
(`turn`, `interrupt`, `close`) and asserts only what a host sees — whether a handle is queued,
what the model was asked, which words reached the session log and in what order, whether a stop
holds, the error raised, and each stop an episode receives. `test_zeus_turn.py` covers the same
ground with richer fixtures (real shells, the composition); this file is the table, row by row.

Each case names its row as `phase --event--> phase`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import (
    RUN,
    SID,
    WAIT_S,
    FakeHandle,
    ScriptedRung,
    answer,
    call,
    calls,
    config_for,
    held,
    said,
    start,
    with_sessions,
)

import chickadee.engine as engine_module
from chickadee.client.request import ChatRequest
from chickadee.engine import CoderHandle, PacketEngine, RunResult, TurnHandle
from chickadee.errors import SessionError
from chickadee.schema.adapter import Completion


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


@pytest.fixture
def engines() -> Iterator[list[PacketEngine]]:
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()


class SlowBuild:
    """`client_for` held until the test releases it: the session is still being built."""

    def __init__(self, model: ScriptedRung, *, fail: BaseException | None = None) -> None:
        self.model = model
        self.fail = fail
        self.building = threading.Event()
        self.release = threading.Event()

    def __call__(self, _endpoint: object) -> ScriptedRung:
        self.building.set()
        assert self.release.wait(WAIT_S), "the test never released the build"
        if self.fail is not None:
            raise self.fail
        return self.model


def in_thread(target: Any) -> tuple[threading.Thread, list[Any], list[BaseException]]:
    """Run `target` on its own thread, keeping what it returned or raised."""
    out: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            out.append(target())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out, errors


class Asking:
    """A model call the test holds open, and knows has begun: a stop sent after `begun` lands
    mid-turn, after the word was opened, never before the worker picked it up."""

    def __init__(self, fail: BaseException | None = None) -> None:
        self.begun = threading.Event()
        self.release = threading.Event()
        self.fail = fail

    def __call__(self, _: ChatRequest) -> Completion:
        self.begun.set()
        assert self.release.wait(WAIT_S), "the test never released the model"
        if self.fail is not None:
            raise self.fail
        return answer()

    def started(self) -> None:
        assert self.begun.wait(WAIT_S), "the model was never asked"


def coder_returns(engine: PacketEngine, monkeypatch: pytest.MonkeyPatch, handle: Any) -> None:
    """`coder()` answers `handle`; the work's stop before it takes its unattended default."""
    monkeypatch.setattr(engine, "coder", handle)


# -- open and build ---------------------------------------------------------------------------


def test_none_open_building__every_open_waits_for_the_one_build(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    build = SlowBuild(model)
    monkeypatch.setattr(engine_module, "client_for", build)

    first, firsts, _ = in_thread(lambda: engine.turn(SID, "one"))
    assert build.building.wait(WAIT_S)
    second, seconds, _ = in_thread(lambda: engine.turn(SID, "two"))
    build.release.set()
    first.join(WAIT_S)
    second.join(WAIT_S)

    for handle in (*firsts, *seconds):
        handle.wait(timeout=WAIT_S)
    # One build served both openers, and each word reached the loop once.
    assert sorted(said(engine)) == ["one", "two"]
    assert len(model.requests) == 2


def test_building_returns_idle__the_first_turn_starts(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)

    handle = engine.turn(SID, "hi")

    assert handle.queued is False
    handle.wait(timeout=WAIT_S)
    assert said(engine) == ["hi"]
    assert len(model.requests) == 1


def test_building_stopped_returns_held__the_turn_starts_nothing_and_keeps_its_word(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    build = SlowBuild(model)
    monkeypatch.setattr(engine_module, "client_for", build)

    thread, handles, _ = in_thread(lambda: engine.turn(SID, "refactor all"))
    assert build.building.wait(WAIT_S)
    engine.interrupt(SID)
    build.release.set()
    thread.join(WAIT_S)

    (handle,) = handles
    handle.wait(timeout=WAIT_S)
    # The word typed during the build got the build's pending turn (queued=False),
    # which the stop finished without running it.
    assert handle.queued is False
    assert model.requests == []
    # held --turn--> idle --(the next turn)--> running: the kept word is delivered first.
    engine.turn(SID, "smaller").wait(timeout=WAIT_S)
    assert said(engine) == ["refactor all", "smaller"]


def test_a_turn_answers_while_the_session_still_builds(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The first turn on a session waited out the whole build, longer than the wire
    client's 30 s on a large repository, and the REPL died on the timeout. It returns at once;
    liveness says the session is opening; the reply comes once the build ends."""
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    build = SlowBuild(model)
    monkeypatch.setattr(engine_module, "client_for", build)

    handle = engine.turn(SID, "one")  # on this thread: it must not wait for the build
    assert build.building.wait(WAIT_S)
    assert handle.queued is False
    assert engine.liveness(SID).why == "opening the session"
    more = engine.turn(SID, "two")
    assert more.queued is True
    assert model.requests == []

    build.release.set()
    handle.wait(timeout=WAIT_S)
    assert said(engine) == ["one", "two"]


def test_a_session_opened_to_read_is_not_said_to_be_opening(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A build a host starts to read the session (`open_session`, a budget) has no
    turn waiting on it, so liveness does not say someone is working, and a stop is not sent."""
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    build = SlowBuild(model)
    monkeypatch.setattr(engine_module, "client_for", build)

    thread, _, errors = in_thread(lambda: engine.open_session(SID))
    assert build.building.wait(WAIT_S)
    assert engine.liveness(SID).why != "opening the session"
    build.release.set()
    thread.join(WAIT_S)
    assert errors == []


def test_building_raises_failed__every_waiting_turn_gets_the_cause_and_it_is_forgotten(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """T-3.162 moved this answer: the opener's `turn()` raised the build's exception. It now
    returns at once, and the turn says why on its stream: one `error` event, then the end."""
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    build = SlowBuild(model, fail=RuntimeError("the endpoint is gone"))
    monkeypatch.setattr(engine_module, "client_for", build)

    first = engine.turn(SID, "one")
    assert build.building.wait(WAIT_S)
    second = engine.turn(SID, "two")
    assert (first.queued, second.queued) == (False, True)
    build.release.set()

    for handle in (first, second):
        with pytest.raises(SessionError, match="the endpoint is gone"):
            handle.wait(timeout=WAIT_S)
    errors = [e.payload for e in first.events() if e.kind == "error"]
    assert errors == ["the session could not be opened: the endpoint is gone"]
    assert model.requests == []

    # The registry forgot the failed session: the next turn builds it again.
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: model)
    engine.turn(SID, "three").wait(timeout=WAIT_S)
    assert said(engine)[-1] == "three"


def test_building_raises_failed__an_unpriced_model_is_said_on_the_turns_stream(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw, priced=False))
    engines.append(engine)

    handle = engine.turn(SID, "hi")
    with pytest.raises(SessionError, match="no price row"):
        handle.wait(timeout=WAIT_S)
    assert model.requests == []


# -- turn() -----------------------------------------------------------------------------------


def test_idle_turn_running__queued_false_and_the_word_is_answered(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.turn(SID, "one").wait(timeout=WAIT_S)

    again = engine.turn(SID, "two")

    assert again.queued is False
    again.wait(timeout=WAIT_S)
    assert said(engine) == ["one", "two"]


def test_running_turn_running__queued_on_the_running_turns_handle(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    gate = threading.Event()
    engine, model = start(monkeypatch, repo, config_for(raw), [held(gate, answer("one"))])
    engines.append(engine)

    first = engine.turn(SID, "one")
    second = engine.turn(SID, "two")
    third = engine.turn(SID, "three")
    gate.set()
    second.wait(timeout=WAIT_S)

    assert (first.queued, second.queued, third.queued) == (False, True, True)
    # The queued handles are the running turn's: they stream what it streams.
    answers = [e.payload for e in first.events() if e.kind == "answer"]
    assert [e.payload for e in second.events() if e.kind == "answer"] == answers
    assert said(engine) == ["one", "two", "three"]
    assert len(model.requests) == 3


def test_stopping_turn_continuing__the_stopped_turn_ends_and_the_conversation_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    model_call = Asking()
    engine, model = start(monkeypatch, repo, config_for(raw), [model_call])
    engines.append(engine)

    handle = engine.turn(SID, "go")
    model_call.started()
    engine.interrupt(SID)
    instead = engine.turn(SID, "do this instead")
    model_call.release.set()
    handle.wait(timeout=WAIT_S)

    assert instead.queued is True
    assert said(engine) == ["go", "do this instead"]
    assert len(model.requests) == 2


def test_continuing_turn_continuing__every_word_after_the_stop_is_delivered(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    model_call = Asking()
    engine, _ = start(monkeypatch, repo, config_for(raw), [model_call])
    engines.append(engine)

    handle = engine.turn(SID, "go")
    model_call.started()
    engine.interrupt(SID)
    engine.turn(SID, "then this")
    also = engine.turn(SID, "and this")
    model_call.release.set()
    handle.wait(timeout=WAIT_S)

    assert also.queued is True
    assert said(engine) == ["go", "then this", "and this"]


def test_closed_turn__a_session_error_and_no_model_call(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    engine.close()

    with pytest.raises(SessionError, match="closed"):
        engine.turn(SID, "again")
    assert len(model.requests) == 1


# -- interrupt() ------------------------------------------------------------------------------


def test_idle_interrupt_idle__a_stop_while_nothing_runs_stops_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.turn(SID, "one").wait(timeout=WAIT_S)

    engine.interrupt(SID, hard=True)
    again = engine.turn(SID, "two")
    again.wait(timeout=WAIT_S)

    assert again.queued is False
    assert said(engine) == ["one", "two"]
    assert len(model.requests) == 2


def test_held_interrupt_held__a_second_stop_keeps_the_hold_for_one_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    build = SlowBuild(model)
    monkeypatch.setattr(engine_module, "client_for", build)

    thread, handles, _ = in_thread(lambda: engine.turn(SID, "first"))
    assert build.building.wait(WAIT_S)
    engine.interrupt(SID)
    engine.interrupt(SID, hard=True)
    build.release.set()
    thread.join(WAIT_S)

    (handle,) = handles
    handle.wait(timeout=WAIT_S)
    assert handle.queued is False  # T-3.162: the build's pending turn, finished by the stop
    assert model.requests == []
    engine.turn(SID, "second").wait(timeout=WAIT_S)
    assert said(engine) == ["first", "second"]


def test_running_interrupt_stopping__the_turn_ends_and_later_words_wait_for_the_next(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    model_call = Asking()
    engine, model = start(monkeypatch, repo, config_for(raw), [model_call])
    engines.append(engine)

    handle = engine.turn(SID, "go")
    model_call.started()
    engine.turn(SID, "queued before the stop")
    engine.interrupt(SID)
    model_call.release.set()
    handle.wait(timeout=WAIT_S)

    # stopping --boundary--> idle: the queued word was not delivered, and is not lost.
    assert said(engine) == ["go"]
    assert len(model.requests) == 1
    # The stop stays until the next turn clears it; that turn delivers the kept word first.
    engine.turn(SID, "now").wait(timeout=WAIT_S)
    assert said(engine) == ["go", "queued before the stop", "now"]


def test_continuing_interrupt_stopping__a_second_stop_cancels_the_carry_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    model_call = Asking()
    engine, model = start(monkeypatch, repo, config_for(raw), [model_call])
    engines.append(engine)

    handle = engine.turn(SID, "go")
    model_call.started()
    engine.interrupt(SID)
    engine.turn(SID, "then this")
    engine.interrupt(SID)
    model_call.release.set()
    handle.wait(timeout=WAIT_S)

    assert said(engine) == ["go"]
    assert len(model.requests) == 1
    engine.turn(SID, "now").wait(timeout=WAIT_S)
    assert said(engine) == ["go", "then this", "now"]


def test_interrupt_on_a_session_not_open_here_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)

    engine.interrupt("S-nobody", hard=True)


# -- the boundary -----------------------------------------------------------------------------


def test_running_boundary_idle__nothing_queued_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)

    handle = engine.turn(SID, "hi")
    handle.wait(timeout=WAIT_S)

    assert [e.kind for e in handle.events()][-1] == "answer"
    assert len(model.requests) == 1


def test_running_step_raises_idle__wait_raises_and_the_next_turn_runs(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    def broken(_: ChatRequest) -> Completion:
        raise RuntimeError("the rung fell over")

    engine, _ = start(monkeypatch, repo, config_for(raw), [broken])
    engines.append(engine)

    handle = engine.turn(SID, "hi")
    with pytest.raises(SessionError, match="the rung fell over"):
        handle.wait(timeout=WAIT_S)

    again = engine.turn(SID, "again")
    assert again.queued is False
    again.wait(timeout=WAIT_S)
    assert said(engine) == ["hi", "again"]


def test_continuing_step_raises_idle__the_error_ends_the_carry_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    model_call = Asking(fail=RuntimeError("the rung fell over"))
    engine, model = start(monkeypatch, repo, config_for(raw), [model_call])
    engines.append(engine)

    handle = engine.turn(SID, "go")
    model_call.started()
    engine.interrupt(SID)
    engine.turn(SID, "then this")
    model_call.release.set()
    with pytest.raises(SessionError, match="the rung fell over"):
        handle.wait(timeout=WAIT_S)

    assert said(engine) == ["go"]
    assert len(model.requests) == 1
    again = engine.turn(SID, "now")
    assert again.queued is False
    again.wait(timeout=WAIT_S)
    assert said(engine) == ["go", "then this", "now"]


# -- the episode ------------------------------------------------------------------------------


@pytest.mark.parametrize("hard", [False, True])
def test_attach_episode__a_stop_that_landed_while_the_run_was_named_reaches_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    hard: bool,
) -> None:
    engine, _ = start(
        monkeypatch, repo, config_for(raw), [calls(call("coder", "c1", title="x", brief="x"))]
    )
    engines.append(engine)
    handle = FakeHandle(RUN, repo)

    def coder(req: object, *, session_id: str, caller: str = "host") -> CoderHandle:
        engine.interrupt(SID, hard=hard)
        return handle

    coder_returns(engine, monkeypatch, coder)
    engine.turn(SID, "x").wait(timeout=WAIT_S)

    assert handle.interrupted == [hard]


@pytest.mark.parametrize("hard", [False, True])
def test_running_interrupt__the_episode_in_flight_gets_the_stop(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    hard: bool,
) -> None:
    engine, _ = start(
        monkeypatch, repo, config_for(raw), [calls(call("coder", "c1", title="x", brief="x"))]
    )
    engines.append(engine)
    running, release = threading.Event(), threading.Event()

    class SlowHandle(FakeHandle):
        def result(self) -> RunResult:
            running.set()
            assert release.wait(WAIT_S)
            return RunResult(self.run_id, "ABORTED")

    handle = SlowHandle(RUN, repo)
    coder_returns(engine, monkeypatch, lambda req, *, session_id, caller="host": handle)

    turn = engine.turn(SID, "x")
    assert running.wait(WAIT_S)
    engine.interrupt(SID, hard=hard)
    release.set()
    turn.wait(timeout=WAIT_S)

    assert handle.interrupted == [hard]


def test_detach_episode__a_stop_after_the_episode_reported_reaches_nothing(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(
        monkeypatch, repo, config_for(raw), [calls(call("coder", "c1", title="x", brief="x"))]
    )
    engines.append(engine)
    handle = FakeHandle(RUN, repo)
    coder_returns(engine, monkeypatch, lambda req, *, session_id, caller="host": handle)

    engine.turn(SID, "x").wait(timeout=WAIT_S)
    engine.interrupt(SID, hard=True)

    assert handle.interrupted == []


# -- close() ----------------------------------------------------------------------------------


def test_running_close_closed__the_turn_is_stopped_hard_and_the_episode_too(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, model = start(
        monkeypatch, repo, config_for(raw), [calls(call("coder", "c1", title="x", brief="x"))]
    )
    running, release = threading.Event(), threading.Event()

    class SlowHandle(FakeHandle):
        def result(self) -> RunResult:
            running.set()
            assert release.wait(WAIT_S)
            return RunResult(self.run_id, "ABORTED")

        def interrupt(self, hard: bool = False) -> None:
            super().interrupt(hard)
            release.set()

    handle = SlowHandle(RUN, repo)
    coder_returns(engine, monkeypatch, lambda req, *, session_id, caller="host": handle)

    turn = engine.turn(SID, "x")
    engine.turn(SID, "queued")
    assert running.wait(WAIT_S)
    engine.close()
    turn.wait(timeout=WAIT_S)

    assert handle.interrupted == [True]
    # The word queued behind the closed turn was never delivered.
    assert len(model.requests) == 1
    with pytest.raises(SessionError, match="closed"):
        engine.turn(SID, "again")


def test_closed_close__closing_twice_is_harmless(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engine.turn(SID, "hi").wait(timeout=WAIT_S)

    engine.close()
    engine.close()


def test_building_close__close_waits_for_the_build_and_then_closes(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    build = SlowBuild(model)
    monkeypatch.setattr(engine_module, "client_for", build)

    opener, handles, opener_errors = in_thread(lambda: engine.turn(SID, "hi"))
    assert build.building.wait(WAIT_S)
    closer, _, closer_errors = in_thread(engine.close)
    closer.join(0.3)
    assert closer.is_alive(), "close() returned while the session was still being built"

    build.release.set()
    closer.join(WAIT_S)
    opener.join(WAIT_S)

    assert not closer.is_alive()
    assert closer_errors == []
    # The opener either lost the race to close (a SessionError) or started a turn that the
    # close then stopped; either way nothing is left running.
    if opener_errors:
        (raised,) = opener_errors
        assert isinstance(raised, SessionError)
    else:
        (handle,) = handles
        assert isinstance(handle, TurnHandle)
        handle.wait(timeout=WAIT_S)
    with pytest.raises(SessionError, match="closed"):
        engine.turn(SID, "again")
