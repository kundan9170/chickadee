"""The interaction port, v2: `confirm_stop` on
the engine's own ports, and the asker and task an action's question names.

What these pin: unattended, a stop is approved and says it was the default; a scripted port
answers its stops in order, records what it was shown, and falls back to the default when its
script runs out; a box role's shell claim becomes a question naming who asks and for which task,
and Zeus's names Zeus.
"""

from __future__ import annotations

from chickadee.engine import NonInteractive, ScriptedInteraction
from chickadee.schema import sha256_hex
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    AnswerSource,
    ApprovalStop,
    InteractionPort,
    StopAnswer,
    StopDecision,
    StopKind,
)
from chickadee.shell import ShellClaim
from chickadee.zeus.tools.talk import approval_of

TEXT = "# Design\n\nMake go return 2.\n"


def _stop(sid: str = "stop-1") -> ApprovalStop:
    return ApprovalStop(
        id=sid,
        kind=StopKind.DESIGN,
        author="zeus",
        text=TEXT,
        sha=sha256_hex(TEXT),
        workspace="worktree",
    )


def test_unattended_a_stop_is_approved_and_says_it_was_the_default() -> None:
    port = NonInteractive()
    assert isinstance(port, InteractionPort)
    assert port.confirm_stop(_stop()) == StopDecision(
        answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT
    )


def test_a_scripted_port_answers_its_stops_in_order_then_the_default() -> None:
    port = ScriptedInteraction(
        stops=(StopDecision(answer=StopAnswer.NOTE, note="smaller"),),
    )
    first = port.confirm_stop(_stop("stop-1"))
    second = port.confirm_stop(_stop("stop-2"))
    assert (first.answer, first.note, first.source) == (
        StopAnswer.NOTE,
        "smaller",
        AnswerSource.HUMAN,
    )
    assert (second.answer, second.source) == (StopAnswer.APPROVE, AnswerSource.DEFAULT)
    assert [s.id for s in port.stops] == ["stop-1", "stop-2"]


def _claim(asker: str, task: str | None) -> ShellClaim:
    return ShellClaim(
        "pip install requests",
        ".",
        Tier.T2,
        "install",
        True,
        operation_id="op",
        idempotent=False,
        asker=asker,
        task_id=task,
    )


def test_a_box_roles_question_names_who_asks_and_for_which_task() -> None:
    worker = approval_of(_claim("worker", "T2"))
    assert (worker.asker, worker.task_id) == ("worker", "T2")
    athena = approval_of(_claim("athena", None))
    assert (athena.asker, athena.task_id) == ("athena", None)
    zeus = approval_of(_claim("zeus", None))
    assert (zeus.asker, zeus.task_id) == ("zeus", None)
