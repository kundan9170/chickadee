"""`liveness()`: one outcome per case of D87, read from the session's fold, its open questions,
its runs' records and its log's lock. Nothing here writes."""

from __future__ import annotations

import socket
from pathlib import Path

from context_helpers import FakeRecords

from chickadee.context.lines import EpisodeLine
from chickadee.context.questions import (
    QuestionAskedLine,
    open_in,
    open_reclaimed,
    record_asked,
)
from chickadee.context.session import Session
from chickadee.context.store import LOCK_NAME, session_dir
from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.projection import RunProjection
from chickadee.liveness import OPENING, Liveness, LivenessKind, liveness
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import Option, Question, QuestionKind

RUN = "2026-10-10-0123abcd"


def _lock(root: Path) -> Path:
    return session_dir(root, "s1") / LOCK_NAME


def _read(session: Session, root: Path, *, host: str | None = None) -> Liveness:
    return liveness(session.state, open_in(session), FakeRecords(), _lock(root), host=host)


def _question() -> Question:
    return Question(
        id="q1",
        kind=QuestionKind.CLARIFY_GOAL,
        text="Go on?",
        options=(Option(key="yes", label="Yes"), Option(key="no", label="No")),
        default="no",
    )


def test_a_replied_turn_is_idle(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        session.record_assistant("hi")
        got = _read(session, tmp_path)
    assert got == Liveness(LivenessKind.IDLE, got.last_seq)
    assert got.last_seq == 2


def test_an_unanswered_turn_under_a_live_lock_is_working(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        got = _read(session, tmp_path)
    assert got.kind is LivenessKind.WORKING
    assert got.why == "the last user turn has no reply yet"


def test_a_waiting_question_under_a_live_lock_is_awaiting(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        record_asked(session, QuestionAskedLine(question=_question()))
        got = _read(session, tmp_path)
    assert (got.kind, got.question_id) == (LivenessKind.AWAITING, "q1")


def test_a_lock_held_on_another_host_is_unknown_never_abandoned(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        got = _read(session, tmp_path, host=f"not-{socket.gethostname()}")
    assert got.kind is LivenessKind.UNKNOWN
    assert got.owner_host == socket.gethostname()


def test_work_in_flight_with_nobody_holding_the_lock_is_abandoned(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "s1")
    session.record_user("hello")
    session.close()
    got = _read(session, tmp_path)
    assert got.kind is LivenessKind.ABANDONED
    assert got.why == "the last user turn has no reply yet"


def test_a_question_its_process_died_asking_is_abandoned(tmp_path: Path) -> None:
    """The open question outlives its asker on the log; nobody is there to take an answer."""
    session = Session.create(tmp_path, "s1")
    record_asked(session, QuestionAskedLine(question=_question()))
    session.close()
    got = _read(session, tmp_path)
    assert got.kind is LivenessKind.ABANDONED
    assert got.why == "question q1 is waiting for an answer"


def test_a_lock_naming_a_dead_pid_is_abandoned(tmp_path: Path) -> None:
    """The holder's record names a pid no process has: the lock may read held while the
    kernel has not caught up, and the pid settles it."""
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        lock = _lock(tmp_path)
        lock.write_text(f'{{"pid": {2**22 + 7}, "host": "{socket.gethostname()}"}}')
        got = _read(session, tmp_path)
    assert got.kind is LivenessKind.ABANDONED


def test_reading_liveness_writes_nothing(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        before = sorted((p, p.stat().st_size) for p in session_dir(tmp_path, "s1").rglob("*"))
        _read(session, tmp_path)
        after = sorted((p, p.stat().st_size) for p in session_dir(tmp_path, "s1").rglob("*"))
    assert before == after


def test_an_unreadable_last_run_costs_only_the_outcome(tmp_path: Path) -> None:
    """A run journal that cannot be read is answered, not raised (a 500 on `/liveness`): no
    outcome, and the safe-point rule's caution that the run may still be in flight."""

    class Damaged(FakeRecords):
        def projection(self, run_id: str) -> RunProjection:
            raise JournalCorrupt(3, run_id, "line 3: not json")

    with Session.create(tmp_path, "s1") as session:
        session.record_episode(EpisodeLine(RUN, "exact", Tier.T0, None, (), None))
        session.record_user("hello")
        session.record_assistant("done")
        got = liveness(session.state, open_in(session), Damaged(), _lock(tmp_path))
    assert got.outcome is None
    assert got.kind is LivenessKind.WORKING, "unread, the run may still be going (safe point)"


def test_a_question_a_dead_holder_left_open_is_withdrawn_by_the_next_writer(
    tmp_path: Path,
) -> None:
    """Reclaiming is the next writer's: the reopened session is not AWAITING a question
    nobody can answer, and the same id may be asked again."""
    session = Session.create(tmp_path, "s1")
    record_asked(session, QuestionAskedLine(question=_question()))
    session.close()
    with open_reclaimed(tmp_path, "s1") as reopened:
        assert open_in(reopened) == ()
        assert _read(reopened, tmp_path).kind is LivenessKind.IDLE
        record_asked(reopened, QuestionAskedLine(question=_question()))
        assert [q.question_id for q in open_in(reopened)] == ["q1"]


def test_a_turn_its_process_says_failed_is_idle_with_why(tmp_path: Path) -> None:
    """An unanswered turn is not in flight once the process hosting it says the turn
    ended on an error. A turn that replied and then reported one (an answer cut at max_tokens)
    was answered: plain idle, so a batch command still exits 0 on it."""
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        got = liveness(
            session.state, open_in(session), FakeRecords(), _lock(tmp_path), failed="refused"
        )
        assert got == Liveness(
            LivenessKind.IDLE, got.last_seq, why="the last turn ended without a reply: refused"
        )
        session.record_assistant("cut")
        got = liveness(
            session.state, open_in(session), FakeRecords(), _lock(tmp_path), failed="max_tokens"
        )
    assert got == Liveness(LivenessKind.IDLE, got.last_seq)


def test_a_session_opening_for_a_turn_is_working_before_its_log_says_so(tmp_path: Path) -> None:
    """The word is not in the log until the build ends and the lock may not be taken
    yet, so the host's word that a turn waits on the build is what says someone works on it."""
    with Session.create(tmp_path, "s1") as session:
        session.record_user("hello")
        session.record_assistant("hi")
        state, waiting = session.state, open_in(session)
    got = liveness(state, waiting, FakeRecords(), _lock(tmp_path), opening=True)
    assert (got.kind, got.why) == (LivenessKind.WORKING, OPENING)
