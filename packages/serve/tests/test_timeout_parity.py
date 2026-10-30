"""With nobody answering, a remote question resolves to its default as
`timeout_default`, and an approval is denied; a hard stop returns at once."""

from __future__ import annotations

import threading
import time

from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    ActionApproval,
    AnswerSource,
    Option,
    Question,
    QuestionKind,
)
from chickadee_serve.interaction import PendingQuestions, RemoteInteraction


def _question() -> Question:
    return Question(
        id="q1",
        kind=QuestionKind.CLARIFY_GOAL,
        text="Go on?",
        options=(Option(key="yes", label="Yes"), Option(key="no", label="No")),
        default="no",
    )


def test_an_unanswered_question_resolves_to_its_default() -> None:
    answer = RemoteInteraction(PendingQuestions(), timeout_s=0.2).ask(_question())
    assert (answer.choice, answer.source) == ("no", AnswerSource.TIMEOUT_DEFAULT)


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


def test_an_unanswered_approval_is_denied() -> None:
    decision = RemoteInteraction(PendingQuestions(), timeout_s=0.2).confirm_action(_approval())
    assert (decision.verdict, decision.source) == ("deny", "default")


def test_a_cancel_returns_promptly() -> None:
    cancel = threading.Event()
    port = RemoteInteraction(PendingQuestions(), timeout_s=60)
    threading.Timer(0.1, cancel.set).start()
    started = time.monotonic()
    port.ask(_question(), cancel=cancel)
    assert time.monotonic() - started < 2


def test_a_cancel_returns_an_approval_promptly() -> None:
    """A hard stop or a pause never waits on a person at a command's permission."""
    cancel = threading.Event()
    port = RemoteInteraction(PendingQuestions(), timeout_s=60)
    threading.Timer(0.1, cancel.set).start()
    started = time.monotonic()
    port.confirm_action(_approval(), cancel=cancel)
    assert time.monotonic() - started < 2


def test_the_slot_is_gone_after_the_wait() -> None:
    pending = PendingQuestions()
    RemoteInteraction(pending, timeout_s=0.05).ask(_question())
    assert pending.waiting("") == ()
