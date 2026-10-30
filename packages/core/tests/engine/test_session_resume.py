"""`PacketEngine.resume(session_id)` (`T-3.8` items 4 and 7): a session resumes from
its records, not from memory. Its constraint ledger comes back with its log, and the episode
its `coder` call left unfinished is continued from that run's journal, in a new engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest
from coder_helpers import (
    SID,
    ProcessDied,
    athena_accepts,
    athena_plans_folder,
    box_call,
    build,
    make_repo,
    zeus_delegates,
    zeus_designs,
)
from repos import git
from zeus_turn_helpers import WAIT_S, Answering, answer, calls, state

import chickadee.engine as engine_module
from chickadee.context.ledger import record_refusal_note
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import CoderHandle, CoderRequest, PacketEngine
from chickadee.errors import SessionError
from chickadee.journal.paths import journal_path
from chickadee.journal.reader import read_run
from chickadee.journal.writer import Journal
from chickadee.schema.journal import EventKind
from chickadee.state.roots import roots as resolve_roots

LIB = "def h() -> int:\n    return 5\n"
BOTH = calls(
    box_call("edit", "w1", path="app.py", old="return 1", new="return 2"),
    box_call("edit", "w2", path="lib.py", old="return 5", new="return 6"),
)
"""The worker's one response: both edits."""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = make_repo(tmp_path / "repo")
    (repo / "lib.py").write_text(LIB)
    git(repo, "add", "lib.py")
    git(repo, "commit", "-qm", "lib")
    engine_module.init_repo(repo)
    return repo


def paused_by_a_crash(monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]) -> str:
    """Zeus delegates a design of the two edits; the episode is stopped at once, so it pauses
    at its first task boundary, Athena's first delegation, as a process that died after
    `run_start` leaves it; then the engine goes. Returns the run."""
    script = [*zeus_designs(), *athena_plans_folder()]
    engine, _ = build(monkeypatch, repo, raw, script)
    engine.interaction = Answering("yes")
    real = engine.coder

    def stopped_at_once(
        req: CoderRequest, *, session_id: str, caller: Literal["host", "zeus"] = "host"
    ) -> CoderHandle:
        handle = real(req, session_id=session_id, caller=caller)
        handle.interrupt()
        return handle

    monkeypatch.setattr(engine, "coder", stopped_at_once)
    engine.turn(SID, "edit both").wait(timeout=WAIT_S)
    (episode,) = state(engine, SID).episodes()
    assert state(engine, SID).result("c1").text.startswith("PAUSED")  # type: ignore[union-attr]
    engine.close()
    monkeypatch.undo()
    return episode.run_id


def test_resume_continues_the_unfinished_episode_in_a_new_engine(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    run_id = paused_by_a_crash(monkeypatch, repo, dict(raw))
    kinds = [e.kind for e in read_run(resolve_roots(repo), run_id)]
    assert "run_end" not in kinds, "the run was left unfinished"

    # Athena delegates again, as her refused delegation told her to; the worker edits both.
    again = [
        calls(box_call("delegate", "d2", task="T1")),
        BOTH,
        answer("T1: edited both"),
        *athena_accepts(),
    ]
    engine, _ = build(monkeypatch, repo, raw, again)
    engines.append(engine)
    resumed = engine.resume(SID)

    assert resumed.reconciled == (run_id,)
    handle = resumed.in_flight
    assert handle is not None and handle.run_id == run_id
    result = handle.result()
    assert (result.outcome, result.tasks_committed, result.tasks_total) == ("COMPLETED", 1, 1)
    replayed = [str(e.kind) for e in read_run(resolve_roots(repo), run_id)]
    assert replayed.count("run_start") == 1, "resumed, not started again"
    assert replayed[-1] == "run_end"


def test_resume_restores_the_constraint_ledger_with_the_log(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """D75's failure was a resume that restored the work and lost the constraint ledger."""
    paused_by_a_crash(monkeypatch, repo, dict(raw))
    with Session.open(sessions_root(resolve_roots(repo)), SID) as session:
        cid = record_refusal_note(session, "never touch vendor/", session.state.last_seq)

    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    engine.resume(SID)

    live = [c.id for c in state(engine, SID).ledger.live_constraints()]
    assert cid in live


def test_a_session_with_nothing_unfinished_resumes_with_no_episode(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    resumed = engine.resume(SID)
    assert resumed.in_flight is None and resumed.reconciled == ()
    assert resumed.info().session_id == SID


def test_a_run_that_died_before_its_run_start_is_not_continued(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A journal with a config snapshot and no `run_start` died before the run began:
    nothing was done, and resume finds nothing to continue rather than refusing forever."""
    real = Journal.append

    def dies_at_the_start(self: Journal, kind: EventKind, *args: Any, **kwargs: Any) -> int:
        if kind is EventKind.RUN_START:
            raise ProcessDied
        return real(self, kind, *args, **kwargs)

    engine, _ = build(monkeypatch, repo, raw, zeus_delegates())
    monkeypatch.setattr(Journal, "append", dies_at_the_start)
    with pytest.raises(SessionError):
        engine.turn(SID, "edit").wait(timeout=WAIT_S)
    monkeypatch.setattr(Journal, "append", real)
    engine.close()

    again, _ = build(monkeypatch, repo, raw)
    engines.append(again)
    resumed = again.resume(SID)
    assert resumed.in_flight is None and resumed.reconciled == ()


def test_a_damaged_journal_is_a_session_error_naming_the_run(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """An unfinished run whose journal is damaged cannot be continued (D27 refuses to edit a
    journal into parsing). Resume says which run and why, as a `SessionError`, never a raw
    store error."""
    run_id = paused_by_a_crash(monkeypatch, repo, dict(raw))
    with journal_path(resolve_roots(repo), run_id).open("ab") as fh:
        fh.write(b"not a journal line\n")

    engine, _ = build(monkeypatch, repo, raw)
    engines.append(engine)
    with pytest.raises(SessionError, match=run_id):
        engine.resume(SID)
