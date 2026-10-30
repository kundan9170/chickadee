"""The client half of `RemoteInteraction`: which questions a session is
waiting on, read from its records, and the answer sent back.

**The records are the truth.** A `question_asked` record opens a question; the
`question_answered` naming its id closes it, whoever answered: a person, a script, its timeout,
or a withdrawal (`source: "withdrawn"`). `Questions` folds a session's records, so a frontend
that joins late, or reconnects, sees what is open by reading from seq 0, with no server state.

**An answer names a choice, never who chose**. `Client.answer` takes no source, and the
body it sends has no field for one: the server records the class of the token that sent it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import (
    ActionApproval,
    ApprovalStop,
    Question,
    QuestionAnsweredRecord,
    QuestionAskedRecord,
    RetiredGate,
)

__all__ = ["OpenQuestion", "Questions"]

ASKED = "question_asked"
ANSWERED = "question_answered"


@dataclass(frozen=True)
class OpenQuestion:
    """An open question: an `ask` (`question`), an action awaiting approval (`approval`), a stop
    (`stop`), or an older log's design gate (`gate`), and the seq of the record
    that asked it."""

    id: str
    seq: int
    question: Question | None
    approval: ActionApproval | None
    gate: RetiredGate | None = None
    stop: ApprovalStop | None = None


@dataclass
class Questions:
    """A session's open questions, oldest first, from its records in seq order."""

    _open: dict[str, OpenQuestion] = field(default_factory=dict[str, OpenQuestion])

    def feed(self, frame: RecordFrame) -> None:
        """Apply one record; every kind but the two question kinds is ignored."""
        if frame.kind == ASKED:
            asked = QuestionAskedRecord.model_validate(frame.payload)
            qid = _asked_id(asked)
            self._open[qid] = OpenQuestion(
                qid, frame.seq, asked.question, asked.approval, asked.gate, asked.stop
            )
        elif frame.kind == ANSWERED:
            self._open.pop(QuestionAnsweredRecord.model_validate(frame.payload).id, None)

    def feed_all(self, frames: Iterable[RecordFrame]) -> None:
        for frame in frames:
            self.feed(frame)

    @property
    def open(self) -> tuple[OpenQuestion, ...]:
        return tuple(sorted(self._open.values(), key=lambda a: a.seq))


def _asked_id(asked: QuestionAskedRecord) -> str:
    """The id its answer names: the question's, the approval's `action_id`, the gate's or the
    stop's."""
    if asked.question is not None:
        return asked.question.id
    if asked.approval is not None:
        return asked.approval.action_id
    if asked.gate is not None:
        return asked.gate.id
    if asked.stop is not None:
        return asked.stop.id
    raise ValueError("a question_asked record holds a question, an approval, a gate or a stop")
