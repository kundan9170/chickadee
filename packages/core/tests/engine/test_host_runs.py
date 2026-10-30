"""A host's run is recorded in its session as Zeus's are: `PacketEngine.coder()` called
by a host writes the run's `episode` line, with no turn, before the run does any work; the
family's spend counts it; and `resume()` finds it unfinished. A line that cannot be written
gives the lease back, as Zeus's dispatch does, through the same bracket. A session closed under
a host's run does not end the run."""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    SID,
    athena_accepts,
    athena_delegates,
    build,
    designed,
    make_repo,
    one_task,
    solved,
    worker_edit,
)
from repos import git
from zeus_turn_helpers import state

import chickadee.engine as engine_module
from chickadee.context.lines import EpisodeLine
from chickadee.context.session import Session
from chickadee.context.store import SessionLog, sessions_root
from chickadee.engine import CoderHandle, PacketEngine
from chickadee.journal import paths
from chickadee.journal.writer import Journal
from chickadee.schema import EventKind
from chickadee.schema.journal import ModelCallP, RunStartP
from chickadee.state.roots import roots as resolve_roots

LIB = "def h() -> int:\n    return 5\n"
SHA = "a" * 64


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A second file beside `app.py`, for a run that edits both."""
    repo = make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text(LIB)
    git(repo, "add", "lib.py")
    git(repo, "commit", "-qm", "lib")
    engine_module.init_repo(repo)
    return repo


def on_disk(repo: Path) -> SessionLog:
    """The session's log as a later reader finds it, not the live fold."""
    return SessionLog.read(sessions_root(resolve_roots(repo)), SID)


def test_a_hosts_run_is_named_in_its_session_before_it_starts(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    request = one_task()
    handle = engine.coder(request, session_id=SID)

    (episode,) = state(engine, SID).episodes()
    assert (episode.run_id, episode.path, episode.tier) == (handle.run_id, "planned", None)
    assert episode.design_sha == request.work_sha, "the work approved at its stop"
    assert episode.opened_at_seq is None, "no turn of Zeus's called it"
    assert [e.run_id for e in on_disk(repo).events() if e.kind == "episode"] == [
        handle.run_id
    ], "durable before the run does anything"
    assert not paths.run_dir(resolve_roots(repo), handle.run_id).exists()

    assert handle.result().outcome == "COMPLETED"


def test_a_hosts_spend_counts_toward_the_family(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The family's spend is its `spend` lines and its episodes' journals: a run
    with no `episode` line was spent and never counted."""
    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    handle = engine.coder(one_task(), session_id=SID)
    handle.abandon()
    # What the run's model calls would have journaled, at a cost the test chooses.
    start = RunStartP(
        goal="g",
        work_sha=SHA,
        config_sha=SHA,
        repo_head=SHA,
        engine_version="0",
        schema_version=2,
        isolation="worktree",
        hard_cost_cap_usd=5.0,
    )
    done = ModelCallP(
        role="worker",
        request="task",
        tokens_in=100,
        tokens_out=20,
        cost_usd=2.5,
        stop_reason="end_turn",
        rung="low-param",
    )
    with Journal.create(resolve_roots(repo), handle.run_id, {}) as journal:
        journal.append(EventKind.RUN_START, start)
        journal.append(EventKind.MODEL_CALL, done, task_id="T1", round=1)

    assert engine.budget(SID).spent_usd == pytest.approx(2.5)


def test_resume_finds_a_hosts_unfinished_run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """Paused at its first task boundary: Athena's delegation is refused, and she delegates
    again once the run resumes."""
    engine, _ = build(monkeypatch, repo, dict(raw), athena_delegates())
    handle = engine.coder(designed(), session_id=SID)
    handle.interrupt()
    assert handle.result().outcome == "PAUSED"
    engine.close()

    again = [
        *athena_delegates(n=2, plan=None),
        *worker_edit("return 1", "return 2"),
        *athena_accepts(),
    ]
    engine, _ = build(monkeypatch, repo, raw, again)
    engines.append(engine)
    resumed = engine.resume(SID)

    assert resumed.reconciled == (handle.run_id,)
    assert resumed.in_flight is not None and resumed.in_flight.run_id == handle.run_id
    result = resumed.in_flight.result()
    assert (result.outcome, result.tasks_committed) == ("COMPLETED", 1)


def test_a_session_closed_under_a_hosts_run_does_not_end_the_run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A host may close its session while its run goes on (T-3.139's Design)."""
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    handle = engine.coder(one_task(), session_id=SID)
    engine.close_session(SID)

    result = handle.result()
    assert result.outcome == "COMPLETED", result.why


def live(engine: PacketEngine) -> Session:
    """The session's writer, open on `engine`."""
    engine.open_session(SID)
    conversation = engine._conversations.get(SID)  # pyright: ignore[reportPrivateUsage]
    opened = conversation.opened() if conversation is not None else None
    assert opened is not None, f"session {SID} is not built"
    return opened.session


def full_disk(_line: EpisodeLine) -> EpisodeLine:
    raise OSError(errno.ENOSPC, "No space left on device")


def test_a_hosts_line_that_cannot_be_written_gives_the_lease_back(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw, solved("return 1", "return 2"))
    engines.append(engine)
    session = live(engine)
    with monkeypatch.context() as patched:
        patched.setattr(session, "record_episode", full_disk)
        with pytest.raises(OSError, match="No space"):
            engine.coder(one_task(), session_id=SID)

    again = engine.coder(one_task(), session_id=SID)
    assert again.result().outcome == "COMPLETED", "the failed open held nothing"


def test_an_abandon_that_fails_too_does_not_hide_why_a_hosts_open_failed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The bracket Zeus's dispatch uses (`zeus.delegate.given_back_on_failure`): the full disk
    is what the host is told, and the abandon's own failure is a note on it."""
    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    session = live(engine)
    real = CoderHandle.abandon

    def cannot_abandon(handle: CoderHandle) -> None:
        real(handle)  # the lease is given back, so the engine still closes
        raise OSError(errno.ENOSPC, "the run's journal cannot be written either")

    monkeypatch.setattr(session, "record_episode", full_disk)
    monkeypatch.setattr(CoderHandle, "abandon", cannot_abandon)
    with pytest.raises(OSError, match="No space") as err:
        engine.coder(one_task(), session_id=SID)
    assert any("cannot be written either" in n for n in err.value.__notes__)
