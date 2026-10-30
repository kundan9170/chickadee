"""The client cannot promote its own answer: it has no way to say who
answered, the server records the token's class, and a body that tries to say is refused."""

from __future__ import annotations

import inspect

import httpx
import pytest
from engines import Step, answer, call, calls
from hosting import SID, hosted, until

from chickadee_serve.frames import ErrorCode, TokenClass
from chickadee_wire import calls as wire_calls
from chickadee_wire.client import AsyncClient, Client
from chickadee_wire.errors import WireError
from chickadee_wire.questions import Questions
from chickadee_wire.transport import base_url, sync_transport


def _asking() -> list[Step]:
    return [calls(call("ask_user", "q1", question="which branch?")), answer()]


@pytest.mark.parametrize("cls", [Client, AsyncClient])
def test_an_answer_cannot_say_who_answered(cls: type) -> None:
    assert "source" not in inspect.signature(cls.answer).parameters
    sent = wire_calls.answer("s", "q", "yes", "n").body
    assert sent is not None
    # D196 §16 adds a stop's edited text (the design gate's edit and subset retired with it);
    # a field naming who answered is never one of them.
    assert set(sent.model_dump()) == {"choice", "note", "text"}
    assert "source" not in sent.model_dump()


@pytest.mark.parametrize(
    ("token_class", "recorded"),
    [(TokenClass.INTERACTIVE, "human"), (TokenClass.AUTOMATION, "default")],
)
def test_the_token_decides_who_answered(
    monkeypatch: pytest.MonkeyPatch, token_class: TokenClass, recorded: str
) -> None:
    with hosted(monkeypatch, _asking()) as host, host.client(token_class) as client:
        client.turn(SID, "go")
        until(client, "question_asked")
        questions = Questions()
        questions.feed_all(client.session_journal(SID))
        (asked,) = questions.open
        assert asked.question is not None and asked.id == asked.question.id
        taken = client.answer(SID, asked.id, asked.question.default, "main")
        assert taken.source == recorded
        answered = until(client, "question_answered")
        questions.feed(answered)
    assert answered.payload["source"] == recorded
    assert questions.open == ()


def test_a_body_naming_a_source_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with hosted(monkeypatch, _asking()) as host, host.client(TokenClass.AUTOMATION) as client:
        client.turn(SID, "go")
        asked = until(client, "question_asked")
        question = asked.payload["question"]
        assert isinstance(question, dict)
        token = host.token(TokenClass.AUTOMATION)
        with httpx.Client(
            transport=sync_transport(host.address),
            base_url=base_url(host.address),
            headers={"authorization": f"Bearer {token}"},
        ) as raw:
            forged = raw.post(
                f"/session/{SID}/answer/{question['id']}",
                json={"choice": question["default"], "source": "human"},
            )
        assert forged.status_code == 422, forged.text
        assert forged.json()["code"] == ErrorCode.BAD_REQUEST
        with pytest.raises(WireError):  # still open: the forged answer was never taken
            client.answer(SID, "no-such-question", "yes")
        assert client.answer(SID, str(question["id"]), str(question["default"])).source == (
            "default"
        )
