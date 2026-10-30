"""CL-9 on the session log: a question asked and how it was answered, folded into the questions
still open, which `liveness()` and a late-connecting frontend read."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from chickadee.context.questions import (
    CANCELLED,
    LINES,
    WITHDRAWN,
    QuestionAnsweredLine,
    QuestionAskedLine,
    RecordingPort,
    for_run,
    open_in,
    open_reclaimed,
    record_answered,
    record_asked,
    still_asking,
)
from chickadee.context.rewind import REWIND
from chickadee.context.session import Session
from chickadee.context.store import SessionLog
from chickadee.context.vocabulary import PayloadError
from chickadee.schema import sha256_hex
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    Option,
    Question,
    QuestionKind,
    StopAnswer,
    StopDecision,
    StopKind,
)


def _question(qid: str = "q1") -> Question:
    return Question(
        id=qid,
        kind=QuestionKind.CLARIFY_GOAL,
        text="Raise the limit?",
        options=(Option(key="yes", label="Yes"), Option(key="no", label="No")),
        default="no",
    )


def _approval() -> ActionApproval:
    return ActionApproval(
        action_id="a1",
        operation_id="op1",
        kind="command",
        display="git push",
        spec_sha="0" * 64,
        network=True,
        idempotent=False,
        reversible=False,
        tier=Tier.T2,
    )


def test_a_question_line_carries_the_whole_question(tmp_path: Path) -> None:
    line = QuestionAskedLine(question=_question())
    adapter = TypeAdapter(LINES["question_asked"])
    back = adapter.validate_json(adapter.dump_json(line))
    assert back == line
    assert isinstance(back, QuestionAskedLine)
    assert back.question is not None and back.question.text == "Raise the limit?"
    assert back.question_id == "q1"


def test_an_approval_line_is_named_by_its_action(tmp_path: Path) -> None:
    line = QuestionAskedLine(approval=_approval())
    adapter = TypeAdapter(QuestionAskedLine)
    assert adapter.validate_json(adapter.dump_json(line)) == line
    assert line.question_id == "a1"


def test_open_questions_fold_and_survive_a_reopen(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "s1")
    session.record_user("hi")
    record_asked(session, QuestionAskedLine(question=_question("q1")))
    record_asked(session, QuestionAskedLine(approval=_approval()))
    assert [q.question_id for q in open_in(session)] == ["q1", "a1"]
    record_answered(
        session, QuestionAnsweredLine(id="q1", choice="yes", source="human", elapsed_ms=10)
    )
    session.close()
    reopened = Session.open(tmp_path, "s1")
    assert [q.question_id for q in open_in(reopened)] == ["a1"]
    reopened.close()
    assert [q.question_id for q in open_in(SessionLog.read(tmp_path, "s1"))] == ["a1"]


@pytest.mark.parametrize(
    "line",
    [QuestionAskedLine(), QuestionAskedLine(question=_question(), approval=_approval())],
)
def test_a_line_holds_exactly_one_thing_asked(tmp_path: Path, line: QuestionAskedLine) -> None:
    session = Session.create(tmp_path, "s1")
    with pytest.raises(PayloadError, match="exactly one"):
        record_asked(session, line)
    session.close()


def test_an_answer_to_nothing_open_is_refused(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "s1")
    with pytest.raises(PayloadError, match="was not asked"):
        record_answered(
            session,
            QuestionAnsweredLine(id="q-never", choice="yes", source="human", elapsed_ms=1),
        )
    session.close()


def test_one_question_waits_once(tmp_path: Path) -> None:
    session = Session.create(tmp_path, "s1")
    record_asked(session, QuestionAskedLine(question=_question()))
    with pytest.raises(PayloadError, match="already waiting"):
        record_asked(session, QuestionAskedLine(question=_question()))
    session.close()


def test_a_rewind_keeps_questions() -> None:
    """Who was asked what, and who answered, is provenance whatever is taken back."""
    assert REWIND["question_asked"] == REWIND["question_answered"] == "kept"


# -- RecordingPort: the one home that writes them ---------------------------------------


class _Inner:
    """An inner port that answers, or raises, and checks the question was durable first."""

    def __init__(self, session: Session, *, raises: BaseException | None = None) -> None:
        self.session = session
        self.raises = raises
        self.open_when_asked: list[tuple[str, ...]] = []

    def _seen(self) -> None:
        self.open_when_asked.append(tuple(q.question_id for q in open_in(self.session)))
        if self.raises is not None:
            raise self.raises

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self._seen()
        return Answer(question_id=q.id, choice="yes", source=AnswerSource.HUMAN, note="sure")

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self._seen()
        return ApprovalDecision(action_id=approval.action_id, verdict="deny", source="default")

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        self._seen()
        return StopDecision(
            answer=StopAnswer.NOTE, note="keep the API", source=AnswerSource.HUMAN
        )


def _answers(session: Session) -> list[QuestionAnsweredLine]:
    adapter = TypeAdapter(QuestionAnsweredLine)
    return [
        adapter.validate_json(session.blob(e.payload_sha))
        for e, _ in session.state.lines()
        if e.kind == "question_answered"
    ]


def test_a_question_is_on_the_log_before_it_is_asked_and_closed_after(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        inner = _Inner(session)
        answer = RecordingPort(inner, session).ask(_question())
        assert inner.open_when_asked == [("q1",)], "asked only once durable"
        assert open_in(session) == ()
        [line] = _answers(session)
    assert answer.choice == "yes"
    assert (line.id, line.choice, line.source, line.note) == ("q1", "yes", "human", "sure")


def test_an_approval_closes_with_the_decision_as_given(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        decision = RecordingPort(_Inner(session), session).confirm_action(_approval())
        [line] = _answers(session)
    assert decision.verdict == "deny"
    assert (line.id, line.choice, line.source) == ("a1", "deny", "default")


def test_a_cancelled_approval_closes_as_withdrawn_never_as_its_deny(tmp_path: Path) -> None:
    """The wait fell back to a deny nobody gave; the log says the question was cut off,
    so resume asks it again, and the call is not run."""
    stopped = threading.Event()
    stopped.set()
    with Session.create(tmp_path, "s1") as session:
        decision = RecordingPort(_Inner(session), session).confirm_action(
            _approval(), cancel=stopped
        )
        [line] = _answers(session)
    assert decision.verdict == "deny"
    assert (line.id, line.choice, line.source, line.note) == ("a1", "", WITHDRAWN, CANCELLED)


def test_a_port_that_raises_closes_the_question_as_withdrawn(tmp_path: Path) -> None:
    """Nobody answered, so no choice and no person's source; and the question does not read as
    waiting for as long as the log lives."""
    with Session.create(tmp_path, "s1") as session:
        port = RecordingPort(_Inner(session, raises=KeyboardInterrupt()), session)
        with pytest.raises(KeyboardInterrupt):
            port.ask(_question())
        assert open_in(session) == ()
        [line] = _answers(session)
    assert (line.choice, line.source, line.note) == ("", WITHDRAWN, "KeyboardInterrupt")


def test_the_lines_carry_the_run_they_were_asked_in(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        RecordingPort(_Inner(session), session, run_id="2026-10-10-0123abcd").ask(_question())
        runs = {e.run_id for e, r in session.state.lines() if e.kind in LINES}
    assert runs == {"2026-10-10-0123abcd"}


def test_for_run_tags_a_sessions_port_and_leaves_a_hosts_alone(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        bare = _Inner(session)
        assert for_run(bare, "2026-10-10-0123abcd") is bare, "no session log, nothing to tag"
        for_run(RecordingPort(bare, session), "2026-10-10-0123abcd").ask(_question())
        runs = {e.run_id for e, _ in session.state.lines() if e.kind in LINES}
    assert runs == {"2026-10-10-0123abcd"}


def test_a_cancelled_ask_is_withdrawn_whatever_the_port_returned(tmp_path: Path) -> None:
    """The port contract: a hard stop answers the question cancelled. A remote port's wait
    returns its default when cancelled; that is no one's answer, least of all a timeout."""
    cancel = threading.Event()
    cancel.set()
    with Session.create(tmp_path, "s1") as session:
        RecordingPort(_Inner(session), session).ask(_question(), cancel=cancel)
        [line] = _answers(session)
    assert (line.choice, line.source, line.note) == ("", WITHDRAWN, CANCELLED)


def test_a_withdrawal_the_log_cannot_take_keeps_the_original_error(tmp_path: Path) -> None:
    """The stop that interrupted the ask may have closed the log: the caller still sees the
    stop, never the storage error from recording it."""
    session = Session.create(tmp_path, "s1")

    class ClosesTheLog(_Inner):
        def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
            session.close()
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        RecordingPort(ClosesTheLog(session), session).ask(_question())


def _stop() -> ApprovalStop:
    text = "# Plan\n\n1. T1\n"
    return ApprovalStop(
        id="stop-1",
        kind=StopKind.PLAN,
        author="athena",
        text=text,
        sha=sha256_hex(text),
        workspace="worktree",
    )


def test_a_stop_is_on_the_log_whole_before_it_is_asked_and_closed_with_its_answer(
    tmp_path: Path,
) -> None:
    """A stop is a question, its markdown on the log exactly as shown; its answer
    closes it with the person's note."""
    with Session.create(tmp_path, "s1") as session:
        inner = _Inner(session)
        decision = RecordingPort(inner, session).confirm_stop(_stop())
        assert inner.open_when_asked == [("stop-1",)], "asked only once durable"
        assert open_in(session) == ()
        [line] = _answers(session)
        [asked] = [
            TypeAdapter(QuestionAskedLine).validate_json(session.blob(e.payload_sha))
            for e, _ in session.state.lines()
            if e.kind == "question_asked"
        ]
    assert decision.answer is StopAnswer.NOTE
    assert asked.stop == _stop() and asked.question_id == "stop-1"
    assert (line.id, line.choice, line.source, line.note) == (
        "stop-1",
        "note",
        "human",
        "keep the API",
    )


def test_a_permission_question_its_asker_left_is_still_asked_after_the_reopen(
    tmp_path: Path,
) -> None:
    """Reopening withdraws what a dead holder left open, and resume reads `still_asking`
    after that. The question was answered by nobody, so it is asked again."""
    with Session.create(tmp_path, "s1") as session:
        record_asked(session, QuestionAskedLine(approval=_approval()), run_id="R1")
    with open_reclaimed(tmp_path, "s1") as session:
        assert open_in(session) == (), "withdrawn on the reopen"
        assert still_asking(session, "op1")
        assert not still_asking(session, "op2")


def test_a_permission_question_someone_answered_is_not_asked_again(tmp_path: Path) -> None:
    with Session.create(tmp_path, "s1") as session:
        RecordingPort(_Inner(session), session).confirm_action(_approval())
    with open_reclaimed(tmp_path, "s1") as session:
        assert not still_asking(session, "op1")
