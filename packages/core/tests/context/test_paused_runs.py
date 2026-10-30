"""Which runs a session left paused, read once in core (`context/project.paused_runs`):
Zeus's notice of how a resumed run ended (`zeus/reconcile.tell_ended`) and the session's
liveness, which tells the REPL what opening the session again resumes, both read it here.

What these pin: a run whose `coder` results carry no typed outcome is paused; one that reported
how it ended (a harness failure's `BLOCKED`) is not; oldest first. Liveness names the
latest paused run that has not ended, and none whose journal will not read."""

from __future__ import annotations

from pathlib import Path

from context_helpers import FakeRecords, run_end, run_id, run_start

from chickadee.context.lines import EpisodeLine
from chickadee.context.project import paused_runs, run_ended
from chickadee.context.questions import open_in
from chickadee.context.results import CoderOutcome
from chickadee.context.session import Session
from chickadee.context.store import LOCK_NAME, session_dir
from chickadee.liveness import liveness
from chickadee.schema import Tier, ToolCall

BLOCKED = CoderOutcome("BLOCKED", ("the harness failed",))


def _coder(session: Session, name: str, outcome: CoderOutcome | None = None) -> str:
    """A `coder` call and its result: `PAUSED` when `outcome` is None."""
    run = run_id(name)
    call = f"c-{name}"
    dispatch = session.record_assistant(
        "dispatching", [ToolCall(id=call, name="coder", arguments={"request": {"goal": name}})]
    )
    session.record_episode(EpisodeLine(run, "planned", Tier.T1, None, (), dispatch.seq))
    text = f"PAUSED · run {run}" if outcome is None else f"{outcome.terminal} · run {run}"
    session.record_tool(call, text, run_id=run, outcome=outcome)
    return run


def test_a_run_answered_only_paused_is_paused(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        first = _coder(session, "a")
        _coder(session, "b", BLOCKED)
        third = _coder(session, "c")
        assert paused_runs(session.state) == (first, third), "oldest first; BLOCKED is not"


def test_a_session_with_no_coder_call_has_no_paused_run(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        assert paused_runs(session.state) == ()


def test_run_ended_reads_the_journal_and_says_nothing_when_it_cannot() -> None:
    going, done = run_id("going"), run_id("done")
    records = FakeRecords(
        runs={going: (run_start(going, "g"),), done: (run_start(done, "d"), run_end(done, 2))}
    )
    assert run_ended(records, going) is False
    assert run_ended(records, done) is True
    assert run_ended(records, run_id("never")) is None, "no journal: nothing either way"


def test_liveness_names_the_latest_paused_run_not_ended(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        older = _coder(session, "older")
        newer = _coder(session, "newer")
        ended = _coder(session, "ended")
        records = FakeRecords(
            runs={
                older: (run_start(older, "o"),),
                newer: (run_start(newer, "n"),),
                ended: (run_start(ended, "e"), run_end(ended, 2)),
            }
        )
        lock = session_dir(tmp_path, "s1") / LOCK_NAME
        got = liveness(session.state, open_in(session), records, lock)
    assert got.paused_run == newer


def test_liveness_names_no_paused_run_whose_journal_will_not_read(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        _coder(session, "lost")
        lock = session_dir(tmp_path, "s1") / LOCK_NAME
        got = liveness(session.state, open_in(session), FakeRecords(), lock)
    assert got.paused_run is None
