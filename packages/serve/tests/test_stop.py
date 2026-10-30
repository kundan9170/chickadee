"""A stop over the wire.

`RemoteInteraction.confirm_stop` waits for a client's answer as any question does; a person's
edit travels as its text; only a person may edit or note it; who decided is the token's class.
Nobody answering never approves it (where `timeout_default` used to): it waits
until answered, or until its `cancel` token ends the wait."""

from __future__ import annotations

import threading
import time

import pytest
from pydantic import ValidationError

from chickadee.schema import sha256_hex
from chickadee.schema.interaction import (
    AnswerSource,
    ApprovalStop,
    StopAnswer,
    StopDecision,
    StopKind,
)
from chickadee_serve.auth import TokenClass
from chickadee_serve.frames import STOP_CHOICES, AnswerBody
from chickadee_serve.interaction import (
    STOP_PERSON_ONLY,
    ChoiceRefused,
    Given,
    PendingQuestions,
    RemoteInteraction,
)

SID = "S-stop"
TEXT = "# Design\n\nMake go return 2.\n"
STOP = ApprovalStop(
    id="stop-1",
    kind=StopKind.DESIGN,
    author="zeus",
    text=TEXT,
    sha=sha256_hex(TEXT),
    workspace="worktree",
    source_ref="main",
)


def decide(given: Given | None, timeout_s: float = 5.0) -> StopDecision:
    pending = PendingQuestions()
    port = RemoteInteraction(pending, timeout_s=timeout_s, session_id=SID)
    out: list[StopDecision] = []
    asking = threading.Thread(target=lambda: out.append(port.confirm_stop(STOP)))
    asking.start()
    if given is not None:
        deadline = time.monotonic() + 5
        while STOP.id not in pending.waiting(SID):
            assert time.monotonic() < deadline, "the stop never opened"
            time.sleep(0.01)
        pending.answer(SID, STOP.id, given)
    asking.join(timeout=10)
    (decision,) = out
    return decision


def test_a_person_s_edit_comes_back_as_their_text() -> None:
    decision = decide(Given("edit", TokenClass.INTERACTIVE, text="# Design\n\nReturn 3.\n"))
    assert decision == StopDecision(
        answer=StopAnswer.EDIT, edited="# Design\n\nReturn 3.\n", source=AnswerSource.HUMAN
    )


def test_a_note_comes_back_verbatim() -> None:
    decision = decide(Given("note", TokenClass.INTERACTIVE, "  keep the old API  "))
    assert (decision.answer, decision.note) == (StopAnswer.NOTE, "  keep the old API  ")


def test_an_automation_token_may_approve_and_it_reads_as_the_default() -> None:
    decision = decide(Given("approve", TokenClass.AUTOMATION))
    assert (decision.answer, decision.source) == (StopAnswer.APPROVE, AnswerSource.DEFAULT)


@pytest.mark.parametrize("choice", STOP_PERSON_ONLY)
def test_only_a_person_may_edit_or_note_a_stop(choice: str) -> None:
    pending = PendingQuestions()
    pending.open(SID, STOP.id, STOP_CHOICES, STOP_PERSON_ONLY)
    with pytest.raises(ChoiceRefused, match="only a person"):
        pending.answer(SID, STOP.id, Given(choice, TokenClass.AUTOMATION, "x", text="x"))


def test_nobody_answering_never_approves_it_however_long_it_waits() -> None:
    """Moved by D240 §2.9a: the question timeout once approved a stop nobody answered. The team
    now runs beside the person's conversation, so a stop may wait while they talk to Zeus."""
    pending = PendingQuestions()
    port = RemoteInteraction(pending, timeout_s=0.05, session_id=SID)
    out: list[StopDecision] = []
    asking = threading.Thread(target=lambda: out.append(port.confirm_stop(STOP)))
    asking.start()
    asking.join(timeout=0.5)
    assert asking.is_alive() and out == [], "still waiting, long past the question timeout"
    pending.answer(SID, STOP.id, Given("approve", TokenClass.INTERACTIVE))
    asking.join(timeout=5)
    assert out == [StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.HUMAN)]


def test_its_cancel_token_ends_the_wait_and_approves_nothing() -> None:
    pending = PendingQuestions()
    port = RemoteInteraction(pending, timeout_s=0.05, session_id=SID)
    cancel = threading.Event()
    out: list[StopDecision] = []
    asking = threading.Thread(target=lambda: out.append(port.confirm_stop(STOP, cancel=cancel)))
    asking.start()
    time.sleep(0.2)
    cancel.set()
    asking.join(timeout=5)
    assert not asking.is_alive()
    assert [d.answer for d in out] == [StopAnswer.CANCEL]
    assert STOP.id not in pending.waiting(SID), "the question is closed"


def test_a_stop_s_text_travels_only_with_edit_and_never_beside_a_design() -> None:
    assert AnswerBody(choice="edit", text="# x").text == "# x"
    with pytest.raises(ValidationError, match="only with the choice edit"):
        AnswerBody(choice="note", note="n", text="# x")
    # The design gate's `design` and `subset` retired with it: a body naming either is
    # refused as a field the wire does not declare, never read beside the text.
    for retired in ({"design": {"goal": "g"}}, {"subset": ["t1"]}):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AnswerBody.model_validate({"choice": "edit", "text": "# x", **retired})
