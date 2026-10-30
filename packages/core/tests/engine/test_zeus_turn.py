"""A turn reaches Zeus through the facade.

`PacketEngine.turn()` runs the session's loop on a worker thread of its own, queues what is
typed while it runs, and stops on `interrupt()`. The loop is composed from the engine's own
values: the shell's `Executor` and the `Reader` hold the one deny list the engine resolved,
and Zeus's `coder` goes through `PacketEngine.coder`.

A scripted model stands behind the zeus rung. Commands run unwrapped (`Passthrough`), as the
loop's own suite runs them: bubblewrap mounts a private `/tmp`, where `tmp_path` lives.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from refusals import stated
from repos import init_repo
from zeus_turn_helpers import (
    RUN,
    SID,
    WAIT_S,
    Answering,
    FakeHandle,
    ScriptedRung,
    Step,
    answer,
    appears,
    call,
    calls,
    config_for,
    held,
    result,
    said,
    start,
    state,
    with_sessions,
)

import chickadee.engine as engine_module
import chickadee.zeus.tools.graphnav as graphnav
from chickadee.approval.stops import STOP_ROUNDS
from chickadee.client.live import LiveEvent
from chickadee.client.request import ChatRequest
from chickadee.context.reader import Reader
from chickadee.context.store import sessions_root
from chickadee.engine import (
    CoderHandle,
    CoderRequest,
    PacketEngine,
    RunResult,
    ScriptedInteraction,
    TurnHandle,
)
from chickadee.errors import SessionError
from chickadee.executor.executor import Executor
from chickadee.graph.indexer import Indexer
from chickadee.isolation.base import level_of
from chickadee.isolation.none import Passthrough
from chickadee.journal.paths import journal_path
from chickadee.schema import Tier, sha256_hex
from chickadee.schema.adapter import Completion
from chickadee.schema.execution import SandboxLevel
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee.secrets.denylist import Denylist
from chickadee.secrets.walk_memo import WalkMemo
from chickadee.state.roots import roots as resolve_roots
from chickadee.zeus.loop import Event

BRIEF = "## Goal\nAdd a retry.\n"

# -- the engine -------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Each session walks the workspace and `$HOME` for the deny list: point it at a small
    directory, never the developer's own home."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(
        init_repo(tmp_path / "repo", commit=None), SID, "S-one", "S-two", "S-other"
    )


@pytest.fixture
def unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())


@pytest.fixture
def engines() -> Iterator[list[PacketEngine]]:
    """Closed after the test, so no worker thread outlives it."""
    opened: list[PacketEngine] = []
    yield opened
    for engine in opened:
        engine.close()


# -- turn() -----------------------------------------------------------------------------------


def test_turn_returns_before_the_model_answers(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    gate = threading.Event()
    engine, model = start(monkeypatch, repo, config_for(raw), [held(gate, answer("hello"))])
    engines.append(engine)

    handle = engine.turn(SID, "hi")

    assert isinstance(handle, TurnHandle)
    assert handle.session_id == SID
    assert handle.queued is False
    with pytest.raises(TimeoutError):
        handle.wait(timeout=0.2)
    gate.set()
    handle.wait(timeout=WAIT_S)
    assert [e.payload for e in handle.events() if e.kind == "answer"] == ["hello"]
    assert len(model.requests) == 1


def test_a_turn_during_a_turn_is_queued_and_delivered_after_it_in_order(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    gate = threading.Event()
    engine, model = start(monkeypatch, repo, config_for(raw), [held(gate, answer("one"))])
    engines.append(engine)

    first = engine.turn(SID, "first")
    second = engine.turn(SID, "second")
    third = engine.turn(SID, "third")

    assert (second.queued, third.queued) == (True, True)
    gate.set()
    first.wait(timeout=WAIT_S)
    second.wait(timeout=WAIT_S)
    assert said(engine) == ["first", "second", "third"]
    # Each word reached the model, after the one before it was answered.
    assert len(model.requests) == 3


def test_a_turn_after_a_turn_starts_a_new_one(
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


def test_closing_events_mid_turn_loses_no_line_and_no_queued_word(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    gate = threading.Event()
    script: list[Step] = [
        calls(call("note", "n1", text="a finding worth keeping")),
        held(gate, answer("first answered")),
    ]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    handle = engine.turn(SID, "first")
    stream = cast("Generator[Event]", handle.events())
    # The first event that reflects a line: a live one is a hint before it.
    assert next(e for e in stream if not isinstance(e.payload, LiveEvent)).kind == "tool_call"
    stream.close()
    engine.turn(SID, "second")
    gate.set()
    handle.wait(timeout=WAIT_S)

    assert said(engine) == ["first", "second"]
    assert state(engine).result("n1") is not None
    answers = [t.text for t in state(engine).turns() if t.role == "assistant" and t.text]
    assert answers == ["first answered", "done"]


def test_a_loop_error_is_a_session_error_from_wait(
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
    # What was committed before the error is on the log, and the session takes the next word.
    assert said(engine) == ["hi"]
    engine.turn(SID, "again").wait(timeout=WAIT_S)
    assert said(engine) == ["hi", "again"]


def test_an_unpriced_zeus_model_is_refused_before_anything_is_spent(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw, priced=False))
    engines.append(engine)

    # Refused on the turn's stream, the build running off the call.
    with pytest.raises(SessionError, match="no price row"):
        engine.turn(SID, "hi").wait(timeout=WAIT_S)
    assert model.requests == []


# -- interrupt() ------------------------------------------------------------------------------


def test_a_soft_interrupt_cancels_the_calls_not_yet_started_and_keeps_queued_words(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    started, second = repo / "started", repo / "second"
    script: list[Step] = [
        calls(
            call("shell", "s1", command="touch started && sleep 1"),
            call("shell", "s2", command="touch second"),
        )
    ]
    engine, model = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    handle = engine.turn(SID, "go")
    appears(started)
    engine.turn(SID, "afterwards")
    engine.interrupt(SID)
    handle.wait(timeout=WAIT_S)

    assert not result(engine, "s1").startswith("CANCELLED")
    assert stated(result(engine, "s2")) == "CANCELLED{tool: shell}"
    assert not second.exists()
    assert len(model.requests) == 1
    # The stop paused the conversation; the word typed during it is delivered by the next
    # turn, before what that turn was handed.
    assert said(engine) == ["go"]
    engine.turn(SID, "now").wait(timeout=WAIT_S)
    assert said(engine) == ["go", "afterwards", "now"]


def test_a_hard_interrupt_reaps_a_running_shell(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    script: list[Step] = [calls(call("shell", "s1", command="touch started && sleep 30"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    began = time.monotonic()
    handle = engine.turn(SID, "go")
    appears(repo / "started")
    engine.interrupt(SID, hard=True)
    handle.wait(timeout=WAIT_S)

    assert time.monotonic() - began < 15
    assert result(engine, "s1").startswith("CANCELLED{tool: shell")


def test_interrupt_on_a_session_with_no_loop_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.interrupt("S-nobody", hard=True)


# -- the composition --------------------------------------------------------------------------


@pytest.mark.writes_worktree(".npmrc")
def test_the_shell_and_the_reader_hold_the_engines_one_deny_list(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    """A repository that removes `.npmrc` from the list, on a clone whose owner has not
    accepted that: the removal is inert in the engine's list, and so in both readers of it."""
    seen: dict[str, Any] = {}

    class SpyExecutor(Executor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen["executor"] = kwargs
            super().__init__(*args, **kwargs)

    class SpyReader(Reader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen["reader"] = kwargs
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(engine_module, "Executor", SpyExecutor)
    monkeypatch.setattr(engine_module, "Reader", SpyReader)
    raw["sandbox"] = {"require_sandbox": False}
    (repo / ".npmrc").write_text("//registry/:_authToken=secret\n", encoding="utf-8")
    script: list[Step] = [
        calls(
            call("read", "r1", path=".npmrc"),
            call("shell", "s1", command="cat .npmrc"),
        )
    ]
    config = config_for(raw, project={"secrets": {"deny_remove": [".npmrc"]}})
    engine, _ = start(monkeypatch, repo, config, script)
    engines.append(engine)

    engine.turn(SID, "show me the npm config").wait(timeout=WAIT_S)

    listed: Denylist = engine._denylist  # pyright: ignore[reportPrivateUsage]
    assert seen["executor"]["denylist"] is listed
    assert seen["reader"]["denylist"] is listed
    assert not listed.readable(".npmrc")
    assert (repo / ".npmrc").resolve() in seen["executor"]["deny_read"]
    assert result(engine, "r1").startswith("SECRET_PATH")
    assert result(engine, "s1").startswith("SECRET_PATH")
    assert "secret" not in result(engine, "s1").split("{", 1)[1]


def test_with_require_sandbox_and_no_backend_a_shell_call_is_sandbox_required(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    raw["sandbox"] = {"require_sandbox": True}
    script: list[Step] = [calls(call("shell", "s1", command="touch ran"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    text = result(engine, "s1")
    assert text.startswith("SANDBOX_REQUIRED{level: none")
    assert not (repo / "ran").exists()


def test_zeus_coder_call_reaches_packet_engine_coder(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script: list[Step] = [calls(call("coder", "c1", title="Add a retry", brief=BRIEF))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    asked: list[tuple[object, str, str]] = []

    def coder(req: object, *, session_id: str, caller: str = "host") -> CoderHandle:
        asked.append((req, session_id, caller))
        return FakeHandle(RUN, repo)

    monkeypatch.setattr(engine, "coder", coder)
    # The one-task plan is shown at stop 2 first; this person approves it.
    engine.interaction = Answering("yes")

    engine.turn(SID, "add a retry").wait(timeout=WAIT_S)

    ((req, session_id, caller),) = asked
    assert caller == "zeus", "Zeus's dispatch writes his episode line, not the door (D178)"
    assert isinstance(req, CoderRequest)
    assert req.task is not None and req.design is None
    assert (req.task.title, req.task.brief) == ("Add a retry", BRIEF)
    assert req.work_sha == sha256_hex(BRIEF), "the brief as the person approved it"
    assert session_id == SID
    assert result(engine, "c1").startswith(f"COMPLETED · run {RUN}")
    (episode,) = state(engine).episodes()
    assert (episode.run_id, episode.tier, episode.design_sha) == (RUN, None, req.work_sha)


def test_an_interrupt_reaches_the_episode_in_flight(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script: list[Step] = [calls(call("coder", "c1", title="Add a retry", brief=BRIEF))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    running = threading.Event()
    release = threading.Event()

    class SlowHandle(FakeHandle):
        def result(self) -> RunResult:
            running.set()
            assert release.wait(WAIT_S)
            return RunResult(self.run_id, "ABORTED")

    handle = SlowHandle(RUN, repo)
    monkeypatch.setattr(engine, "coder", lambda req, *, session_id, caller="host": handle)

    turn = engine.turn(SID, "add a retry")
    assert running.wait(WAIT_S)
    engine.interrupt(SID, hard=True)
    release.set()
    turn.wait(timeout=WAIT_S)

    assert handle.interrupted == [True]
    assert result(engine, "c1").startswith(f"ABORTED · run {RUN}")


def test_with_no_interaction_port_ask_user_resolves_to_its_default(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """An engine with no port is unattended, and a question takes its default."""
    script: list[Step] = [calls(call("ask_user", "q1", question="go on? default: yes"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "hi").wait(timeout=WAIT_S)

    assert state(engine).result("q1") is not None


# -- the review of 703524c ---------------------------------------------------------------------


def test_a_word_typed_after_a_stop_carries_the_conversation_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    """The stop lands while a shell still runs, and the person types before the stopped turn
    ends: that handle is the stopped turn's, and its `wait()` covers the new word too."""
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    script: list[Step] = [calls(call("shell", "s1", command="touch started && sleep 1"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    handle = engine.turn(SID, "go")
    appears(repo / "started")
    engine.interrupt(SID)
    instead = engine.turn(SID, "do this instead")
    handle.wait(timeout=WAIT_S)

    assert instead.queued is True
    assert said(engine) == ["go", "do this instead"]


class Wrapping(Passthrough):
    """A backend that reports an enforcing level and runs commands as given."""

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE


@pytest.mark.parametrize("allow_secret_read", [False, True])
def test_zeus_is_told_reads_are_masked_only_when_they_are(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    allow_secret_read: bool,
) -> None:
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Wrapping())
    raw["secrets"] = {"allow_secret_read": allow_secret_read}
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    assert engine._denylist.allow_secret_read is allow_secret_read  # pyright: ignore[reportPrivateUsage]

    engine.turn(SID, "hi").wait(timeout=WAIT_S)

    system = model.requests[0].messages[0].content
    assert isinstance(system, str)
    assert ("reads_masked: no" if allow_secret_read else "reads_masked: yes") in system


def test_a_refused_turn_leaves_no_session_behind(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw, priced=False))
    engines.append(engine)

    # An id no session has is refused before anything is built (item 14), and
    # nothing is written for the id.
    with pytest.raises(SessionError, match="does not exist"):
        engine.turn("S-unmade", "hi")

    assert not (sessions_root(resolve_roots(repo)) / "S-unmade").exists()


def test_the_deny_list_is_walked_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    walks: list[int] = []
    real = engine_module._resolve_denylist  # pyright: ignore[reportPrivateImportUsage]

    def counted(
        denylist: Denylist, roots: Sequence[Path], *, memo: WalkMemo | None = None
    ) -> tuple[Path, ...]:
        walks.append(1)
        return real(denylist, roots, memo=memo)

    monkeypatch.setattr(engine_module, "_resolve_denylist", counted)
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)

    engine.turn("S-one", "hi").wait(timeout=WAIT_S)
    engine.turn("S-one", "again").wait(timeout=WAIT_S)
    engine.turn("S-two", "hi").wait(timeout=WAIT_S)

    # Once per session, as `resolve()` asks — never per turn, never once for all: a secret
    # one session writes is masked in the next.
    assert len(walks) == 2


def test_one_session_opening_does_not_hold_up_another(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.turn(SID, "first").wait(timeout=WAIT_S)
    building, release = threading.Event(), threading.Event()

    def slow_client(_endpoint: object) -> ScriptedRung:
        building.set()
        assert release.wait(WAIT_S)
        return model

    monkeypatch.setattr(engine_module, "client_for", slow_client)
    other = threading.Thread(target=lambda: engine.turn("S-other", "hi").wait(timeout=WAIT_S))
    other.start()
    assert building.wait(WAIT_S)

    began = time.monotonic()
    engine.interrupt(SID)
    engine.turn(SID, "second").wait(timeout=WAIT_S)
    # While the other session is still being built: nothing here waited on it.
    assert not release.is_set()
    assert time.monotonic() - began < 5
    assert said(engine) == ["first", "second"]
    release.set()
    other.join(WAIT_S)
    assert not other.is_alive()


def test_turn_after_close_is_a_session_error(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    engine.close()

    with pytest.raises(SessionError, match="closed"):
        engine.turn(SID, "again")


def test_a_soft_stop_while_the_run_is_named_still_reaches_it(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The stop lands inside `coder()`, before the engine holds the episode, so
    `interrupt()` cannot see it; the box forwards it as soon as it does."""
    script: list[Step] = [calls(call("coder", "c1", title="Add a retry", brief=BRIEF))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    handle = FakeHandle(RUN, repo)

    def coder(req: object, *, session_id: str, caller: str = "host") -> CoderHandle:
        engine.interrupt(SID)
        return handle

    monkeypatch.setattr(engine, "coder", coder)

    engine.turn(SID, "add a retry").wait(timeout=WAIT_S)

    assert handle.interrupted == [False]


# -- the review of 7940b98 ---------------------------------------------------------------------


def test_close_is_not_undone_by_a_word_typed_after_an_earlier_stop(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    unwrapped: None,
) -> None:
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    script: list[Step] = [calls(call("shell", "s1", command="touch started && sleep 1"))]
    engine, model = start(monkeypatch, repo, config_for(raw), script)

    handle = engine.turn(SID, "go")
    appears(repo / "started")
    engine.interrupt(SID)
    engine.turn(SID, "then this")
    engine.close()
    handle.wait(timeout=WAIT_S)

    # The resume the word asked for is cancelled by the close: no further model call.
    assert len(model.requests) == 1


def test_the_zeus_role_override_picks_the_model(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    other = raw["rungs"]["mid-param"]
    raw["models"][other]["model_id"] = "the-planner-model"
    raw["roles"] = {"zeus": "mid-param"}
    raw["prices"] = {other: {"price_in": 1.0, "price_out": 2.0}}
    engine, model = start(monkeypatch, repo, config_for(raw, priced=False))
    engines.append(engine)

    engine.turn(SID, "hi").wait(timeout=WAIT_S)

    system = model.requests[0].messages[0].content
    assert isinstance(system, str)
    assert "model: the-planner-model" in system


def test_a_corrupt_run_journal_does_not_lock_the_session(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    """A crash leaves a coder call waiting and its run's journal torn: the next turn answers
    the call that nothing can be said, and goes on."""
    script: list[Step] = [calls(call("coder", "c1", title="Add a retry", brief=BRIEF))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)

    class Crashing(FakeHandle):
        def result(self) -> RunResult:
            raise RuntimeError("the process died mid-run")

    monkeypatch.setattr(
        engine, "coder", lambda req, *, session_id, caller="host": Crashing(RUN, repo)
    )
    with pytest.raises(SessionError):
        engine.turn(SID, "add a retry").wait(timeout=WAIT_S)
    engine.close()
    torn = journal_path(resolve_roots(repo), RUN)
    torn.parent.mkdir(parents=True, exist_ok=True)
    torn.write_text("{not json\n", encoding="utf-8")

    again, _ = start(monkeypatch, repo, config_for(raw))
    try:
        again.turn(SID, "where were we").wait(timeout=WAIT_S)
        text = result(again, "c1")
        assert "could not be read" in text
        assert said(again)[-1] == "where were we"
    finally:
        again.close()


@pytest.mark.parametrize(
    ("answer_", "outcome"),
    [(StopAnswer.NOTE, "ASK"), (StopAnswer.CANCEL, "ABORTED")],
    ids=["note", "cancel"],
)
def test_a_stop_that_is_not_approved_ends_the_call_before_the_lease_is_taken(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    answer_: StopAnswer,
    outcome: str,
) -> None:
    """The work's stop is asked at dispatch, before anything is leased. A note ends
    the call `ASK` and a cancel `ABORTED`, the person's words verbatim in its result, and the
    words bind the work that follows as a design-scope constraint."""
    script: list[Step] = [calls(call("coder", "c1", title="Add a retry", brief=BRIEF))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    leased: list[str] = []

    def coder(req: object, *, session_id: str, caller: str = "host") -> CoderHandle:
        leased.append(session_id)
        return FakeHandle(RUN, repo)

    monkeypatch.setattr(engine, "coder", coder)
    engine.interaction = ScriptedInteraction(
        stops=[StopDecision(answer=answer_, note="keep the public API")]
    )

    engine.turn(SID, "add a retry").wait(timeout=WAIT_S)

    assert leased == [], "nothing was leased"
    text = result(engine, "c1")
    assert text.startswith(f"{outcome} · run not started")
    assert "keep the public API" in text
    # The call's `episode` line names a run nothing was approved for, and that never began.
    (episode,) = state(engine).episodes()
    assert episode.design_sha is None
    assert not journal_path(resolve_roots(repo), episode.run_id).exists()
    live = [c.text for c in state(engine).ledger.live_constraints()]
    assert live == ["keep the public API"]


def test_a_stop_the_rounds_bound_ended_binds_no_words_and_is_not_the_persons_cancel(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """T-3.157 review: past its rounds a stop ends cancelled with the harness's reason, which
    is not the person's words: it binds no constraint and the call does not say they
    cancelled."""
    script: list[Step] = [calls(call("coder", "c1", title="Add a retry", brief=BRIEF))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    engine.interaction = ScriptedInteraction(
        stops=[StopDecision(answer=StopAnswer.NOTE)] * STOP_ROUNDS
    )

    engine.turn(SID, "add a retry").wait(timeout=WAIT_S)

    text = result(engine, "c1")
    assert text.startswith("ABORTED · run not started")
    assert f"shown {STOP_ROUNDS} times without an approval" in text
    assert "the person cancelled" not in text
    assert state(engine).ledger.live_constraints() == ()


def test_a_stop_while_the_session_is_built_stops_the_turn_it_was_built_for(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, model = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    building, release = threading.Event(), threading.Event()

    def slow_client(_endpoint: object) -> ScriptedRung:
        building.set()
        assert release.wait(WAIT_S)
        return model

    monkeypatch.setattr(engine_module, "client_for", slow_client)
    handles: list[TurnHandle] = []
    first = threading.Thread(target=lambda: handles.append(engine.turn(SID, "refactor all")))
    first.start()
    assert building.wait(WAIT_S)
    engine.interrupt(SID)
    release.set()
    first.join(WAIT_S)

    (handle,) = handles
    handle.wait(timeout=WAIT_S)
    assert handle.queued is False  # T-3.162: the build's pending turn, finished by the stop
    assert model.requests == []
    # The word was kept, and the next turn delivers it first.
    engine.turn(SID, "smaller, please").wait(timeout=WAIT_S)
    assert said(engine) == ["refactor all", "smaller, please"]


# -- a budget stop names its cap ---------------------------------


@pytest.mark.parametrize(
    ("spend", "says"),
    [
        ({"hard_cost_usd": 1.0}, "the session family's spend cap of $1.00 is reached"),
        (
            {"hard_cost_usd": 50.0, "rungs": {"high-param": 1.0}},
            "the high-param rung's spend cap of $1.00 is reached",
        ),
    ],
    ids=["the-total", "zeuss-rung"],
)
def test_a_budget_stop_names_the_cap_that_stopped_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    spend: dict[str, Any],
    says: str,
) -> None:
    """Every call costs $1.10 here, so the first turn crosses a $1 cap and the second is
    refused. The refusal says which cap: the person lifts or raises a different one for
    each."""
    raw["prices"] = {raw["rungs"]["high-param"]: {"price_in": 10_000.0, "price_out": 10_000.0}}
    raw["spend"] = spend
    engine, model = start(monkeypatch, repo, config_for(raw, priced=False))
    engines.append(engine)

    engine.turn(SID, "first").wait(timeout=WAIT_S)
    second = engine.turn(SID, "second")
    second.wait(timeout=WAIT_S)

    (error,) = [str(e.payload) for e in second.events() if e.kind == "error"]
    assert error.startswith("BUDGET_CAP: ")
    assert says in error
    assert "$1.10 spent" in error
    assert len(model.requests) == 1


def test_backend_none_runs_the_shell_unwrapped_and_zeus_is_told(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`[sandbox] backend = "none"` is read. Nothing patches the backend here: the
    engine takes the configured one, and Zeus's prefix describes the sandbox it really has.
    The person also set `allow_unsandboxed_shell`, so the command runs unasked."""
    raw["sandbox"] = {
        "backend": "none",
        "require_sandbox": False,
        "allow_unsandboxed_shell": {"zeus": True},
    }
    script: list[Step] = [calls(call("shell", "s1", command="touch ran"))]
    engine, model = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert (repo / "ran").exists(), "ran unwrapped, as configured"
    system = model.requests[0].messages[0].content
    assert isinstance(system, str)
    assert "sandbox: none · reads_masked: no" in system
    # The box's executor, verifier and bootstrap take the same configured backend (T-3.9's
    # constraint), so the header and the box's commands agree.
    inputs = engine._episode_inputs(SID)  # pyright: ignore[reportPrivateUsage]
    assert level_of(inputs.backend) is SandboxLevel.NONE


def test_backend_none_without_the_key_runs_nothing_nobody_confirms(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The engine passes the person's `allow_unsandboxed_shell` to the shell, and here
    it is off. Every command is T2 `unsandboxed`; the engine's port is `NonInteractive`,
    which denies a confirmation nobody authorized, so it does not run."""
    raw["sandbox"] = {"backend": "none", "require_sandbox": False}
    script: list[Step] = [calls(call("shell", "s1", command="touch ran"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert result(engine, "s1").startswith(
        "REFUSED_ABOVE_T0{tier: t2, classification: unsandboxed"
    )
    assert not (repo / "ran").exists()


# -- a person confirms what the shell runs above T0 ---------------------------------


class Person:
    """An interaction port with a person at it: every confirmation gets `verdict`, and a
    question waits until the stop's token lets it go."""

    def __init__(self, verdict: Literal["approve", "deny", "abort"] = "approve") -> None:
        self.verdict: Literal["approve", "deny", "abort"] = verdict
        self.shown: list[ActionApproval] = []
        self.asking = threading.Event()
        self.released: list[bool] = []

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.asking.set()
        assert cancel is not None, "the port is handed the stop's token"
        self.released.append(cancel.wait(WAIT_S))
        return Answer(question_id=q.id, choice=q.default, source=AnswerSource.DEFAULT)

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self.shown.append(approval)
        return ApprovalDecision(
            action_id=approval.action_id, verdict=self.verdict, source="human"
        )


def unprotected(raw: dict[str, Any]) -> None:
    """No sandbox and the floor on: every shell command is T2 and confirmed."""
    raw["sandbox"] = {"require_sandbox": False}


def test_a_shell_command_above_t0_is_confirmed_by_the_person_and_runs(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    unprotected(raw)
    script: list[Step] = [calls(call("shell", "s1", command="touch ran"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    person = Person("approve")
    engine.interaction = person

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert (repo / "ran").exists()
    assert result(engine, "s1").startswith("exit 0")
    (shown,) = person.shown
    assert (shown.kind, shown.tier, shown.task_id) == ("command", Tier.T2, None)
    assert shown.operation_id == f"{SID}:s1", "the call's own operation, nothing else"
    assert "touch ran" in shown.display and "unsandboxed" in shown.display


def test_a_shell_command_the_person_denies_is_refused_and_never_spawns(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    unprotected(raw)
    script: list[Step] = [calls(call("shell", "s1", command="touch ran"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    engine.interaction = Person("deny")

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert result(engine, "s1").startswith(
        "REFUSED_ABOVE_T0{tier: t2, classification: unsandboxed"
    )
    assert not (repo / "ran").exists()


def _fetch() -> list[Step]:
    """A command that asks for the network."""
    return [calls(call("shell", "s1", command="touch ran", network=True))]


def test_under_the_default_config_an_approved_network_runs_with_the_network_open(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D199 moved this answer: under the default config (`require_sandbox` on) an approved
    networked command answered `NETWORK_DENIED{heuristic: false}`. Now the person is asked once
    and, approved, the sandbox opens the network for that command. Moved again with T-3.178:
    approved, the command runs outside the sandbox, so nothing wraps it."""
    from sandboxes import Enforcing

    backend = Enforcing()
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: backend)
    engine, _ = start(monkeypatch, repo, config_for(raw), _fetch())
    engines.append(engine)
    person = Person("approve")
    engine.interaction = person

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    text = result(engine, "s1")
    assert text.startswith("exit 0") and "network on" in text, text
    (shown,) = person.shown
    assert shown.network and "network on" in shown.display
    assert "outside the sandbox, approved" in text and "sandbox none" in text
    assert [i for i in backend.seen if i.argv[-1:] != ("true",)] == []


@pytest.mark.parametrize("verdict", ["deny", None], ids=["declined", "unattended"])
def test_a_network_nobody_approves_never_spawns(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
    verdict: Literal["deny"] | None,
) -> None:
    unprotected(raw)
    engine, _ = start(monkeypatch, repo, config_for(raw), _fetch())
    engines.append(engine)
    if verdict is not None:
        engine.interaction = Person(verdict)

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert result(engine, "s1").startswith("REFUSED_ABOVE_T0")
    assert not (repo / "ran").exists()


def test_never_ask_runs_a_networked_command_unattended_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    """D199 item 8b: the engine's grant check carries the person's `sandbox.never_ask`, so
    with nobody to ask an unsealed, networked command runs, one `action_granted` line."""
    raw["sandbox"] = {"require_sandbox": False, "never_ask": True}
    engine, _ = start(monkeypatch, repo, config_for(raw), _fetch())
    engines.append(engine)

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert result(engine, "s1").startswith("exit 0")
    assert (repo / "ran").exists()
    kinds = [e.kind for e, _ in state(engine).lines()]
    assert "action_granted" in kinds and "question_asked" not in kinds


@pytest.mark.parametrize("required", [False, True])
def test_the_boxs_executors_take_require_sandbox_from_the_same_config(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
    required: bool,
) -> None:
    """The box's executor, which runs the worker's and Athena's shells, takes the
    person's `require_sandbox` as Zeus's does; it was forced on. An
    enforcing backend, so the box offers a shell either way."""
    from dataclasses import replace
    from types import SimpleNamespace

    from sandboxes import Enforcing

    import chickadee.delegation as delegation_module
    from chickadee.delegation import _Composed  # pyright: ignore[reportPrivateUsage]

    raw["sandbox"] = {"require_sandbox": required}
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    made: list[Executor] = []
    real = Executor

    def recording(*args: Any, **kwargs: Any) -> Executor:
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(delegation_module, "Executor", recording)
    inputs = engine._episode_inputs(SID)  # pyright: ignore[reportPrivateUsage]
    composed = _Composed(replace(inputs, backend=Enforcing()), lambda: None)
    composed.run_id = RUN
    try:
        parts = composed._parts(cast(Any, SimpleNamespace(root=repo)))  # pyright: ignore[reportPrivateUsage]
        assert parts.shell_for is not None
        parts.shell_for(
            cast(Any, SimpleNamespace(put_blob=lambda _data: "h")), "worker", lambda: None
        )
        (executor,) = made
        assert executor.require_sandbox is required
    finally:
        composed.stack.close()


@pytest.mark.parametrize("required", [True, False])
def test_where_nothing_seals_the_box_offers_a_shell_only_on_the_persons_opt_out(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
    required: bool,
) -> None:
    """D199 item 6 moved this answer: on a machine with no sandbox the box offered no shell
    whatever the config said. Now `require_sandbox = false` offers it, unsealed, and the
    person's lift for each role reaches that role's shell."""
    from dataclasses import replace
    from types import SimpleNamespace

    from chickadee.delegation import _Composed  # pyright: ignore[reportPrivateUsage]

    raw["sandbox"] = {
        "require_sandbox": required,
        "backend": "auto",
        "allow_unsandboxed_shell": {"athena": True},
    }
    engine, _ = start(monkeypatch, repo, config_for(raw))
    engines.append(engine)
    engine.turn(SID, "hi").wait(timeout=WAIT_S)
    inputs = engine._episode_inputs(SID)  # pyright: ignore[reportPrivateUsage]
    composed = _Composed(replace(inputs, backend=Passthrough()), lambda: None)
    composed.run_id = RUN
    try:
        parts = composed._parts(cast(Any, SimpleNamespace(root=repo)))  # pyright: ignore[reportPrivateUsage]
        if required:
            assert parts.shell_for is None
            return
        assert parts.shell_for is not None and parts.shell_for.sealed is False
        journal = cast(Any, SimpleNamespace(put_blob=lambda _data: "h", run_id=RUN))
        worker = parts.shell_for(journal, "worker", lambda: None)
        athena = parts.shell_for(journal, "athena", lambda: None)
        lifted = [
            s._shell._allow_unsandboxed  # pyright: ignore[reportPrivateUsage]
            for s in (worker, athena)
        ]
        assert lifted == [False, True]
    finally:
        composed.stack.close()


# -- the graph's navigator and `locate` -------------------------------------------


def _indexed(repo: Path) -> None:
    """A function in the repository, and the index over it."""
    (repo / "core.py").write_text(
        "def helper(x: int) -> int:\n    return x + 1\n\n\ndef serve() -> int:\n"
        "    return helper(1)\n",
        encoding="utf-8",
    )
    Indexer(repo).full_index()


@pytest.mark.writes_worktree("core.py")
def test_symbol_through_a_turn_answers_from_the_graph(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    _indexed(repo)
    script: list[Step] = [calls(call("symbol", "y1", name="helper"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "where is helper?").wait(timeout=WAIT_S)

    text = result(engine, "y1")
    assert text.startswith("## SYMBOL core.helper    function · core.py:1-2"), text
    assert "core.serve    core.py:6" in text


def test_without_an_index_the_navigate_family_says_so(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    script: list[Step] = [calls(call("symbol", "y1", name="helper"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "where is helper?").wait(timeout=WAIT_S)

    assert result(engine, "y1").startswith("INDEX_UNAVAILABLE")


@pytest.mark.writes_worktree("core.py")
def test_the_graph_is_opened_used_and_closed_on_the_sessions_worker(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """SQLite refuses a connection used off the thread that made it, so a call that answered
    and a close that did not raise are the proof; the thread names say which thread it was."""
    _indexed(repo)
    threads: list[tuple[str, str]] = []
    real = graphnav.opened

    def watched(path: Path) -> Any:
        graph = real(path)
        assert graph is not None
        threads.append(("open", threading.current_thread().name))
        close = graph.close

        def closing() -> None:
            threads.append(("close", threading.current_thread().name))
            close()

        graph.close = closing  # type: ignore[method-assign]
        return graph

    monkeypatch.setattr(graphnav, "opened", watched)
    script: list[Step] = [calls(call("symbol", "y1", name="helper"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)

    engine.turn(SID, "where is helper?").wait(timeout=WAIT_S)
    assert result(engine, "y1").startswith("## SYMBOL")
    engine.close()

    assert [what for what, _ in threads] == ["open", "close"]
    assert all(name.startswith(f"zeus-{SID}") for _, name in threads), threads


@pytest.mark.writes_worktree("core.py")
def test_locate_reaches_the_scout_with_the_engines_executor_and_deny_list(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    _indexed(repo)
    built: list[Executor] = []
    seen: dict[str, Any] = {}

    class SpyExecutor(Executor):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    class Found:
        def locate(self, question: str, *, cancel: threading.Event | None) -> str:
            return f"## FINDINGS for {question!r}"

    def spy(config: Any, session: Any, **kwargs: Any) -> Found:
        seen.update(kwargs)
        return Found()

    monkeypatch.setattr(engine_module, "Executor", SpyExecutor)
    monkeypatch.setattr(engine_module, "scout_locator", spy)
    raw["prices"] = {model: {"price_in": 1.0, "price_out": 2.0} for model in raw["models"]}
    script: list[Step] = [calls(call("locate", "l1", question="where is helper?"))]
    engine, _ = start(monkeypatch, repo, config_for(raw, priced=False), script)
    engines.append(engine)

    engine.turn(SID, "find it").wait(timeout=WAIT_S)

    assert result(engine, "l1") == "## FINDINGS for 'where is helper?'"
    (executor,) = built
    assert seen["executor"] is executor, "the scout runs under Zeus's shell executor"
    assert seen["executor"].denylist is engine._denylist  # pyright: ignore[reportPrivateUsage]
    assert seen["graph"] is seen["navigator"].graph, "one graph, one navigator"
    assert seen["workspace"] == repo.resolve()


@pytest.mark.writes_worktree("core.py")
def test_an_unpriced_scout_model_makes_locate_unavailable_and_the_session_goes_on(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`config_for` prices Zeus's model alone: a `locate` would be paid for and then refused
    at its charge, so it is `UNAVAILABLE` before it runs."""
    _indexed(repo)
    script: list[Step] = [
        calls(call("locate", "l1", question="where?")),
        calls(call("symbol", "y1", name="helper")),
    ]
    engine, model = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)

    engine.turn(SID, "find it").wait(timeout=WAIT_S)

    text = result(engine, "l1")
    assert text.startswith("UNAVAILABLE{rung: ") and "has no price row" in text
    assert result(engine, "y1").startswith("## SYMBOL")
    assert len(model.requests) == 3, "Zeus's three calls, and none of the scout's"


def test_abort_at_the_confirmation_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
    unwrapped: None,
) -> None:
    """`abort` is the person stopping the turn, as a hard stop does: the call is answered
    cancelled, not refused, the next call never starts and no further model call is made."""
    unprotected(raw)
    script: list[Step] = [
        calls(
            call("shell", "s1", command="touch first"),
            call("shell", "s2", command="touch second"),
        ),
        answer("never reached"),
    ]
    engine, model = start(monkeypatch, repo, config_for(raw), script)
    engines.append(engine)
    person = Person("abort")
    engine.interaction = person

    engine.turn(SID, "go").wait(timeout=WAIT_S)

    assert result(engine, "s1").startswith("CANCELLED{tool: shell")
    assert result(engine, "s2").startswith("CANCELLED{tool: shell")
    assert len(person.shown) == 1, "the second call was never put to the person"
    assert not (repo / "first").exists() and not (repo / "second").exists()
    assert len(model.requests) == 1


def test_close_during_a_pending_ask_user_returns_without_an_answer(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`close()` stops hard, and the stop's token reaches the question waiting on the
    person, so the engine does not wait for an answer that may never come."""
    script: list[Step] = [calls(call("ask_user", "q1", question="which branch?"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    person = Person()
    engine.interaction = person
    engine.turn(SID, "go")
    assert person.asking.wait(WAIT_S), "the question was put"

    closing = threading.Thread(target=engine.close)
    began = time.monotonic()
    closing.start()
    closing.join(timeout=WAIT_S)

    assert not closing.is_alive(), "close() returned"
    assert time.monotonic() - began < WAIT_S
    assert person.released == [True], "released by the token, not by a timeout"
