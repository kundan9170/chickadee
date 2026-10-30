"""Questions on the session log: CL-5's budget for every question a run puts
to a person, the reuse of a person's answer, and an approval never reused.

The design gate (Q2) and its CL-6 parity retired with the switch; a stop's record
on the log is `chickadee/context/questions.py`'s, tested in core. The ask points without a
trigger (Q1, Q4 to Q8) join here with the tasks that build them."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from chickadee.context.questions import (
    BUDGET_SPENT,
    QuestionAnsweredLine,
    QuestionAskedLine,
    RecordingPort,
)
from chickadee.context.session import Session
from chickadee.engine import NonInteractive, ScriptedInteraction
from chickadee.schema.interaction import (
    Answer,
    AnswerSource,
    Option,
    Question,
    QuestionKind,
)


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    s = Session.create(tmp_path / "sessions", "S-q")
    yield s
    s.close()


def lines(session: Session, kind: str) -> list[object]:
    adapters = {
        "question_asked": QuestionAskedLine,
        "question_answered": QuestionAnsweredLine,
    }
    from pydantic import TypeAdapter

    adapter = TypeAdapter(adapters[kind])
    return [
        adapter.validate_json(session.blob(e.payload_sha))
        for e, _ in session.state.lines()
        if e.kind == kind
    ]


def ask(qid: str, *, text: str = "which branch?", blocking: bool = True) -> Question:
    return Question(
        id=qid,
        kind=QuestionKind.CLARIFY_GOAL,
        text=text,
        options=(Option(key="main", label="main"), Option(key="dev", label="dev")),
        default="main",
        blocking=blocking,
    )


class Counting(ScriptedInteraction):
    """A person who answers `dev`, counted."""

    def __init__(self) -> None:
        super().__init__(answers=["dev"] * 10)


# -- CL-5: the budget --------------------------------------------------------------------------


def test_a_run_asks_at_most_its_budget_and_then_takes_the_default(session: Session) -> None:
    inner = Counting()
    run = RecordingPort(inner, session, max_questions=1).for_run("R-1")
    first = run.ask(ask("q1"))
    second = run.ask(ask("q2", text="which remote?"))
    assert (first.choice, first.source) == ("dev", AnswerSource.HUMAN)
    assert (second.choice, second.source) == ("main", AnswerSource.DEFAULT)
    assert [q.id for q in inner.asked] == ["q1"], "the second was never put to the person"
    answered = lines(session, "question_answered")
    assert isinstance(answered[-1], QuestionAnsweredLine)
    assert answered[-1].note == BUDGET_SPENT, "the log says why no one was asked"


def test_the_budget_counts_blocking_questions_in_a_run_only(session: Session) -> None:
    inner = Counting()
    port = RecordingPort(inner, session, max_questions=1)
    run = port.for_run("R-1")
    run.ask(ask("q1", text="first"))
    run.ask(ask("q2", text="later", blocking=False))
    port.ask(ask("q3", text="the person's own conversation"))
    assert [q.id for q in inner.asked] == ["q1", "q2", "q3"]


def test_a_question_answered_in_the_session_is_not_asked_again(session: Session) -> None:
    inner = Counting()
    port = RecordingPort(inner, session)
    port.ask(ask("q1"))
    again = port.ask(ask("q2"))
    assert (again.choice, again.source) == ("dev", AnswerSource.SESSION_REUSE)
    assert [q.id for q in inner.asked] == ["q1"]


def test_an_approval_is_never_reused(session: Session) -> None:
    """A person's yes to work above T0 is about that work; the same words over other files
    are asked again."""

    def consent(qid: str) -> Question:
        return Question(
            id=qid,
            kind=QuestionKind.PLAN_APPROVAL,
            text="The change (fix config) needs t2. Run it?",
            options=(Option(key="yes", label="run it"), Option(key="no", label="do not")),
            default="yes",
        )

    inner = ScriptedInteraction(answers=["yes", "no"])
    port = RecordingPort(inner, session)
    port.ask(consent("c1"))
    again = port.ask(consent("c2"))
    assert (again.choice, again.source) == ("no", AnswerSource.HUMAN)
    assert [q.id for q in inner.asked] == ["c1", "c2"]


def test_a_default_is_not_reused_as_an_answer(session: Session) -> None:
    """Only a person's answer is reused: an unattended default was nobody's choice."""
    RecordingPort(NonInteractive(), session).ask(ask("q1"))
    inner = Counting()
    again = RecordingPort(inner, session).ask(ask("q2"))
    assert again.source is AnswerSource.HUMAN and len(inner.asked) == 1


def test_a_cancelled_question_is_answered_cancelled(session: Session) -> None:
    cancel = threading.Event()
    cancel.set()

    class Ignoring(NonInteractive):
        def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
            return Answer(question_id=q.id, choice="dev", source=AnswerSource.HUMAN)

    RecordingPort(Ignoring(), session).ask(ask("q1"), cancel=cancel)
    (answered,) = lines(session, "question_answered")
    assert isinstance(answered, QuestionAnsweredLine) and answered.source == "withdrawn"
