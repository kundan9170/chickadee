"""`prompt_tally` (T-3.158 G): each reason and each answer read from a recorded session log,
and a granted action counted as a question saved. Read-only: the log is written by the ports
as the engine composes them, and the tally only reads it."""

from __future__ import annotations

import threading
from pathlib import Path

from repos import clone_roots

from chickadee.approval.grants import GrantingPort
from chickadee.approval.tally import PromptRow, prompt_tally, reason_of
from chickadee.context.questions import RecordingPort
from chickadee.context.session import Session
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    CommandBinding,
    GrantScope,
    Question,
    StopAnswer,
    StopDecision,
)

PUSH = CommandBinding(cls="push", prefix=("git", "push"), destination="origin")
EXACT = CommandBinding(
    cls="push", prefix=("git", "push", "origin", "main"), destination="origin"
)


class _Chooser:
    """Answers each confirmation as the next of `answers` says: (verdict, scope)."""

    def __init__(self, answers: list[tuple[str, GrantScope]]) -> None:
        self.answers = answers

    def ask(self, q: Question, *, cancel: object = None) -> Answer:
        raise AssertionError("not asked here")

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        verdict, scope = self.answers.pop(0)
        return ApprovalDecision.model_validate(
            {
                "action_id": approval.action_id,
                "verdict": verdict,
                "source": "human",
                "scope": scope,
            }
        )


def _approval(
    n: int,
    *,
    grant: CommandBinding | None = None,
    exact: CommandBinding | None = None,
    offer_repo: bool = True,
    display: str = "git push origin main",
) -> ActionApproval:
    return ActionApproval(
        action_id=f"a{n}",
        operation_id=f"op{n}",
        kind="command",
        display=display,
        spec_sha=f"{n:064d}",
        network=True,
        idempotent=False,
        reversible=False,
        tier=Tier.T2,
        grant=grant,
        exact=exact,
        offer_repo=offer_repo,
    )


def test_each_reason_is_read_from_the_fields_the_prompt_carries() -> None:
    assert reason_of(_approval(1)) == "once_only"
    assert reason_of(_approval(1, exact=EXACT)) == "outside_shape"
    assert reason_of(_approval(1, grant=PUSH, exact=EXACT, offer_repo=False)) == (
        "leaves_the_machine"
    )
    assert reason_of(_approval(1, grant=PUSH, exact=EXACT)) == "no_grant_held"


def test_the_tally_counts_each_answer_and_each_question_a_grant_saved(tmp_path: Path) -> None:
    session = Session.create(tmp_path / "sessions", "s-tally")
    person = _Chooser(
        [
            ("approve", "session"),  # a1: granted for the session, exactly
            ("deny", "once"),  # a3: another push, refused
            ("approve", "once"),  # a4: a line no grant covers
            ("abort", "once"),  # a5
            ("approve", "repo"),  # a6: granted for the repository
        ]
    )
    port = GrantingPort(RecordingPort(person, session), session, clone_roots(tmp_path))
    port.confirm_action(_approval(1, grant=PUSH, exact=EXACT))
    port.confirm_action(_approval(2, grant=PUSH, exact=EXACT))  # the exact grant covers it
    port.confirm_action(
        _approval(
            3, grant=PUSH, exact=EXACT.model_copy(update={"prefix": ("git", "push", "x")})
        )
    )
    port.confirm_action(_approval(4, display="git push origin main > /dev/null"))
    port.confirm_action(_approval(5, display="git push origin main 2>&1"))
    pull = CommandBinding(cls="external_fetch", prefix=("docker", "pull"))
    port.confirm_action(_approval(6, grant=pull, exact=pull, display="docker pull x"))
    tally = prompt_tally(session)
    session.close()

    assert tally.granted == 1, "a2 was covered by a1's grant, and asked nobody"
    assert tally.asked == 5
    rows = {(r.reason, r.cls, r.prefix): r for r in tally.rows}
    push = rows[("no_grant_held", "push", ("git", "push"))]
    assert (push.asked, push.session, push.no) == (2, 1, 1)
    once_only = rows[("once_only", "", ())]
    assert (once_only.asked, once_only.once, once_only.no) == (2, 1, 1)
    assert once_only.examples == (
        "git push origin main > /dev/null",
        "git push origin main 2>&1",
    ), "a row keeps its lines, to tell them apart"
    assert rows[("no_grant_held", "external_fetch", ("docker", "pull"))].repo == 1
    assert tally.rows[0].asked >= tally.rows[-1].asked, "most asked first"


def test_an_unanswered_and_a_withdrawn_question_are_counted_as_such(tmp_path: Path) -> None:
    from chickadee.context.questions import WITHDRAWN, QuestionAnsweredLine, QuestionAskedLine

    session = Session.create(tmp_path / "sessions", "s-tally-open")
    session.append(QuestionAskedLine(approval=_approval(1, grant=PUSH, exact=EXACT)))
    session.append(QuestionAskedLine(approval=_approval(2, grant=PUSH, exact=EXACT)))
    session.append(
        QuestionAnsweredLine(id="a2", choice="", source=WITHDRAWN, elapsed_ms=0, note="stop")
    )
    (row,) = prompt_tally(session).rows
    session.close()
    assert row == PromptRow(
        cls="push",
        prefix=("git", "push"),
        reason="no_grant_held",
        asked=2,
        withdrawn=1,
        unanswered=1,
        examples=("git push origin main",),
    )


def test_an_action_never_ask_ran_is_counted_apart_from_the_grants(tmp_path: Path) -> None:
    """D199 item 8b: `never_ask` is no grant, so its runs say nothing about what a grant saves.
    A destructive command, which no grant covers, runs under it too, and is counted."""
    session = Session.create(tmp_path / "sessions", "s-never")
    port = GrantingPort(
        RecordingPort(_Chooser([]), session), session, clone_roots(tmp_path), never_ask=True
    )
    port.confirm_action(_approval(1, grant=PUSH, exact=EXACT))
    port.confirm_action(_approval(2, display="rm -rf build"))
    tally = prompt_tally(session)
    session.close()
    assert (tally.granted, tally.never_asked, tally.asked) == (0, 2, 0)
