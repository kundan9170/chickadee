"""`web.fetch.approval = "per_domain"` (`external/hosts.py`): the person is
asked once a session for each host, a `SCOPE_EXPANSION` question whose default refuses, so a
session with nobody attached refuses the host and names it; and a refused host is never sent a
request."""

from __future__ import annotations

import threading
from pathlib import Path

from tools.tools_helpers import call, dispatcher
from webs import DictEngine, calls, public, session

from chickadee.external.hosts import ALLOW, DENY, HostApproval, question
from chickadee.external.policy import ExternalPolicy
from chickadee.external.web import Web
from chickadee.schema.interaction import Answer, AnswerSource, Question, QuestionKind
from chickadee.tools.builtin import web_fetch
from chickadee.tools.catalogue import answer_text
from chickadee.tools.context import ToolContext

A = "https://a.example.com/page"


class HostAnswers:
    """Answers every question with `choice`, from `source`, and keeps what was asked."""

    def __init__(self, choice: str, source: AnswerSource = AnswerSource.HUMAN) -> None:
        self.choice = choice
        self.source = source
        self.asked: list[Question] = []
        self.cancels: list[threading.Event | None] = []

    def __call__(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.asked.append(q)
        self.cancels.append(cancel)
        return Answer(question_id=q.id, choice=self.choice, source=self.source)


def test_the_question_is_a_scope_expansion_whose_default_refuses() -> None:
    q = question("docs.example.org", "c1")
    assert (q.kind, q.default) == (QuestionKind.SCOPE_EXPANSION, DENY)
    assert [o.key for o in q.options] == [ALLOW, DENY]
    assert question("docs.example.org", "c9").text == q.text, "worded the same, so reused"


def test_a_host_the_person_allows_is_asked_once() -> None:
    person = HostAnswers(ALLOW)
    hosts = HostApproval(person)
    assert hosts.allows("a.example.com", "c1") and hosts.allows("a.example.com", "c2")
    assert len(person.asked) == 1


def test_an_earlier_answer_in_the_session_allows_it_again() -> None:
    assert HostApproval(HostAnswers(ALLOW, AnswerSource.SESSION_REUSE)).allows(
        "a.example.com", "c1"
    )


def test_a_default_answer_refuses_whatever_it_says() -> None:
    """Nobody attached answers by the default; a default is never a person's yes."""
    for choice in (DENY, ALLOW):
        for source in (AnswerSource.DEFAULT, AnswerSource.TIMEOUT_DEFAULT):
            assert not HostApproval(HostAnswers(choice, source)).allows("a.example.com", "c1")


def test_the_rounds_cancel_reaches_the_question(tmp_path: Path) -> None:
    """A stopped round must not be left waiting on the person: the port withdraws a question
    whose `cancel` is set."""
    stop = threading.Event()
    person = HostAnswers(ALLOW)
    with session(tmp_path) as s:
        w = Web(
            ExternalPolicy(s, rung="researcher"),
            engine=DictEngine({A: "page"}),
            resolve=public,
            approval=HostApproval(person),
        )
        w.saw(A)
        d = dispatcher([web_fetch.TOOL], ctx=ToolContext(role="researcher", web=w, cancel=stop))
        d.call(call("web_fetch", text="", url=A))
    assert person.cancels == [stop]


def test_a_refused_host_is_named_and_sent_nothing(tmp_path: Path) -> None:
    with session(tmp_path) as s:
        engine = DictEngine({A: "page"})
        w = Web(
            ExternalPolicy(s, rung="researcher"),
            engine=engine,
            resolve=public,
            approval=HostApproval(HostAnswers(DENY)),
        )
        w.saw(A)
        d = dispatcher([web_fetch.TOOL], ctx=ToolContext(role="researcher", web=w))
        said = answer_text(d.call(call("web_fetch", text="", url=A)))
        assert said.startswith(f"URL_REFUSED{{url: '{A}', reason: not_approved}}. Do: ")
        assert not engine.asked and not calls(s)
