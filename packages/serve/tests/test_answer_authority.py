"""Who answered is the token's class, fixed when it was issued. A request cannot say
otherwise, and an automation token's answer is never recorded as a person's."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    CommandBinding,
    Option,
    Question,
    QuestionKind,
)
from chickadee_serve.app import Served, build_app
from chickadee_serve.auth import TokenClass, TokenStore, source_for
from chickadee_serve.interaction import (
    AUTOMATION_CANNOT_APPROVE,
    PendingQuestions,
    RemoteInteraction,
)


def _question(qid: str = "q1") -> Question:
    return Question(
        id=qid,
        kind=QuestionKind.CLARIFY_GOAL,
        text="Go on?",
        options=(Option(key="yes", label="Yes"), Option(key="no", label="No")),
        default="no",
    )


class _PendingAsk:
    """A port asking on a thread, as an engine worker would, and what it got back."""

    def __init__(self, port: RemoteInteraction, question: Question) -> None:
        self.answer: Answer | None = None
        self._thread = threading.Thread(
            target=lambda: setattr(self, "answer", port.ask(question))
        )
        self._thread.start()
        while question.id not in port.pending.waiting(port.session_id):
            pass

    def result(self) -> Answer:
        self._thread.join(timeout=5)
        assert self.answer is not None
        return self.answer


def _server(tmp: Path) -> tuple[TestClient, RemoteInteraction, dict[TokenClass, str]]:
    del tmp
    pending = PendingQuestions()
    port = RemoteInteraction(pending, timeout_s=5, session_id="s")
    tokens = TokenStore()
    issued = {cls: tokens.issue(cls) for cls in TokenClass}
    served = Served(facade=object(), tokens=tokens, pending=pending)
    return TestClient(build_app(served)), port, issued


@pytest.mark.parametrize(
    ("token_class", "recorded"),
    [
        (TokenClass.INTERACTIVE, AnswerSource.HUMAN),
        (TokenClass.AUTOMATION, AnswerSource.DEFAULT),
    ],
)
def test_the_source_is_the_tokens_class(
    tmp_path: Path, token_class: TokenClass, recorded: AnswerSource
) -> None:
    client, port, issued = _server(tmp_path)
    asking = _PendingAsk(port, _question())
    r = client.post(
        "/session/s/answer/q1",
        json={"choice": "yes"},
        headers={"authorization": f"Bearer {issued[token_class]}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["source"] == recorded.value
    got = asking.result()
    assert (got.choice, got.source) == ("yes", recorded)


def test_a_request_cannot_name_its_own_source(tmp_path: Path) -> None:
    client, port, issued = _server(tmp_path)
    asking = _PendingAsk(port, _question())
    r = client.post(
        "/session/s/answer/q1",
        json={"choice": "yes", "source": "human"},
        headers={"authorization": f"Bearer {issued[TokenClass.AUTOMATION]}"},
    )
    assert r.status_code == 422 and r.json()["code"] == "BAD_REQUEST"
    client.post(
        "/session/s/answer/q1",
        json={"choice": "yes"},
        headers={"authorization": f"Bearer {issued[TokenClass.AUTOMATION]}"},
    )
    assert asking.result().source is AnswerSource.DEFAULT


def test_first_answer_wins_and_a_second_is_told_so(tmp_path: Path) -> None:
    client, port, issued = _server(tmp_path)
    asking = _PendingAsk(port, _question())
    auth = {"authorization": f"Bearer {issued[TokenClass.INTERACTIVE]}"}
    assert (
        client.post("/session/s/answer/q1", json={"choice": "yes"}, headers=auth).status_code
        == 200
    )
    asking.result()
    late = client.post("/session/s/answer/q1", json={"choice": "no"}, headers=auth)
    assert late.status_code == 409 and late.json()["code"] == "ANSWERED"
    unknown = client.post("/session/s/answer/q-never", json={"choice": "no"}, headers=auth)
    assert unknown.status_code == 404 and unknown.json()["code"] == "NOT_FOUND"


def test_a_choice_outside_the_options_is_refused(tmp_path: Path) -> None:
    client, port, issued = _server(tmp_path)
    asking = _PendingAsk(port, _question())
    auth = {"authorization": f"Bearer {issued[TokenClass.INTERACTIVE]}"}
    r = client.post("/session/s/answer/q1", json={"choice": "maybe"}, headers=auth)
    assert r.status_code == 422
    client.post("/session/s/answer/q1", json={"choice": "no"}, headers=auth)
    assert asking.result().choice == "no"


@pytest.mark.parametrize(
    ("token_class", "verdict", "source"),
    [
        (TokenClass.INTERACTIVE, "approve", "human"),
        (TokenClass.AUTOMATION, "deny", "default"),
    ],
)
def test_an_approval_is_approved_only_by_a_persons_answer(
    tmp_path: Path, token_class: TokenClass, verdict: str, source: str
) -> None:
    """An automation token's `approve` is the default it agreed with, and an approval's
    default is deny: it is recorded as the deny it is, and says why."""
    client, port, issued = _server(tmp_path)
    approval = ActionApproval(
        action_id="a1",
        operation_id="op1",
        kind="command",
        display="rm -rf build",
        spec_sha="0" * 64,
        network=False,
        idempotent=False,
        reversible=False,
        tier=Tier.T2,
    )
    decided: list[ApprovalDecision] = []
    worker = threading.Thread(target=lambda: decided.append(port.confirm_action(approval)))
    worker.start()
    deadline = time.monotonic() + 5
    while "a1" not in port.pending.waiting(port.session_id):
        assert time.monotonic() < deadline, "the approval was never put"
        time.sleep(0.01)
    r = client.post(
        "/session/s/answer/a1",
        json={"choice": "approve"},
        headers={"authorization": f"Bearer {issued[token_class]}"},
    )
    assert r.status_code == 200
    worker.join(timeout=5)
    assert (decided[0].verdict, decided[0].source) == (verdict, source)
    if token_class is TokenClass.AUTOMATION:
        assert decided[0].note == AUTOMATION_CANNOT_APPROVE


@pytest.mark.parametrize(
    ("token_class", "choice", "verdict", "scope"),
    [
        (TokenClass.INTERACTIVE, "session", "approve", "session"),
        (TokenClass.INTERACTIVE, "repo", "approve", "repo"),
        (TokenClass.INTERACTIVE, "approve", "approve", "once"),
        (TokenClass.AUTOMATION, "session", "deny", "once"),
    ],
)
def test_an_approval_offering_a_grant_reads_its_scope(
    tmp_path: Path, token_class: TokenClass, choice: str, verdict: str, scope: str
) -> None:
    """A question carrying a grant offers *always for this session* and *always for this repo*
    beside *once*; an automation token grants nothing, as it approves
    nothing."""
    client, port, issued = _server(tmp_path)
    approval = ActionApproval(
        action_id="a1",
        operation_id="op1",
        kind="command",
        display="git push origin main",
        spec_sha="0" * 64,
        network=True,
        idempotent=False,
        reversible=False,
        tier=Tier.T2,
        grant=CommandBinding(cls="push", prefix=("git", "push"), destination="origin"),
    )
    decided: list[ApprovalDecision] = []
    worker = threading.Thread(target=lambda: decided.append(port.confirm_action(approval)))
    worker.start()
    deadline = time.monotonic() + 5
    while "a1" not in port.pending.waiting(port.session_id):
        assert time.monotonic() < deadline, "the approval was never put"
        time.sleep(0.01)
    r = client.post(
        "/session/s/answer/a1",
        json={"choice": choice},
        headers={"authorization": f"Bearer {issued[token_class]}"},
    )
    assert r.status_code == 200
    worker.join(timeout=5)
    assert (decided[0].verdict, decided[0].scope) == (verdict, scope)


def test_an_approval_without_a_grant_offers_once_only(tmp_path: Path) -> None:
    """A choice the question did not offer is refused, never read as a grant."""
    client, port, issued = _server(tmp_path)
    approval = ActionApproval(
        action_id="a1",
        operation_id="op1",
        kind="command",
        display="frobnicate --all",
        spec_sha="0" * 64,
        network=False,
        idempotent=False,
        reversible=False,
        tier=Tier.T2,
    )
    decided: list[ApprovalDecision] = []
    worker = threading.Thread(target=lambda: decided.append(port.confirm_action(approval)))
    worker.start()
    deadline = time.monotonic() + 5
    while "a1" not in port.pending.waiting(port.session_id):
        assert time.monotonic() < deadline, "the approval was never put"
        time.sleep(0.01)
    bearer = {"authorization": f"Bearer {issued[TokenClass.INTERACTIVE]}"}
    refused = client.post("/session/s/answer/a1", json={"choice": "session"}, headers=bearer)
    assert refused.status_code != 200
    r = client.post("/session/s/answer/a1", json={"choice": "approve"}, headers=bearer)
    assert r.status_code == 200
    worker.join(timeout=5)
    assert (decided[0].verdict, decided[0].scope) == ("approve", "once")


def test_source_table() -> None:
    assert source_for(TokenClass.INTERACTIVE) is AnswerSource.HUMAN
    assert source_for(TokenClass.AUTOMATION) is AnswerSource.DEFAULT


def test_an_answer_reaches_only_the_session_that_asked(tmp_path: Path) -> None:
    """Question ids are unique per session (Zeus's is the model's tool-call id): two sessions
    wait on one id at once, and one session's route never answers the other's question."""
    client, port, issued = _server(tmp_path)
    auth = {"authorization": f"Bearer {issued[TokenClass.INTERACTIVE]}"}
    in_s = _PendingAsk(port, _question("call_0"))
    in_t = _PendingAsk(port.for_session("t"), _question("call_0"))
    wrong = client.post("/session/u/answer/call_0", json={"choice": "yes"}, headers=auth)
    assert wrong.status_code == 404, wrong.text
    client.post("/session/t/answer/call_0", json={"choice": "yes"}, headers=auth)
    client.post("/session/s/answer/call_0", json={"choice": "no"}, headers=auth)
    assert (in_s.result().choice, in_t.result().choice) == ("no", "yes")
