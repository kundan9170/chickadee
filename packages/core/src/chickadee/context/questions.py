"""Questions on the session log: what a person was asked, and how it
was answered.

The run journal has carried the box's questions since G0 (`question_asked`,
`question_answered`). The session log carries every question a session puts to a person, the
box's and Zeus's alike, so a frontend reading one session's records sees each one, and one that
connects late reads a waiting question from its cursor like any other record.

**A stop is a question too**: `question_asked` holds the markdown the person was
shown, and `question_answered` their answer, its source and their words. A resumed run reads
the work it was given from its own config snapshot, so nothing reads these back to run. The
design gate and its `plan_decision` line retired with the packet loop: an
older log's gate is read for its id alone (`RetiredGate`), so its answer still pairs with it.

**The question budget is here**, because every asker's question passes through this port:
a run asks at most `max_questions` blocking questions, and a question answered earlier in the
session is answered the same way again, as `session_reuse`, without asking. **An approval is
never reused**: a consent above T0 (`plan_approval`) is about the work in front of the person,
and the same words over other files are another question.

**Its own kinds, so its type is here**. As for `prefix`, `bound` and `scout`, the
session's fold keeps these lines unread, and this module is their writer and reader:
- **`record_asked`** writes the question durably before anyone is asked it. A line holds the
  frozen `Question` or `ActionApproval` whole, so a client renders it from the line alone.
- **`record_answered`** writes the answer, with the source it was given and never upgraded.
- **`open_questions`** is a projection, computed when asked and stored nowhere: the questions
  asked and not yet answered, which `liveness()` reads as `AWAITING`.

Each writer checks the log as it stands before it appends: a question is waited on once, an
answer names a question that is waiting, and a line holds exactly one thing asked.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Protocol, runtime_checkable

from pydantic import Field, TypeAdapter, with_config

from chickadee.context.vocabulary import LINE, PayloadError, SessionKind
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    InteractionPort,
    Question,
    QuestionKind,
    StopDecision,
)

if TYPE_CHECKING:
    from chickadee.context.session import Session
    from chickadee.context.store import SessionEvent, SessionLog

__all__ = [
    "BUDGET_SPENT",
    "LINES",
    "UNRECORDED",
    "WITHDRAWN",
    "QuestionAnsweredLine",
    "QuestionAskedLine",
    "RecordingPort",
    "RetiredGate",
    "SessionScoped",
    "for_run",
    "open_in",
    "open_questions",
    "open_reclaimed",
    "record_answered",
    "record_asked",
    "scoped",
    "still_asking",
]


@with_config(LINE)
@dataclass(frozen=True)
class RetiredGate:
    """A design gate's question as a log written before the stops holds it: its id, so
    its answer still pairs with it. Nothing asks one since the stops replaced it."""

    id: str


@with_config(LINE)
@dataclass(frozen=True)
class QuestionAskedLine:
    """Exactly one of `question` (an `ask`), `approval` (a `confirm_action`), `stop` (a
    `confirm_stop`), or, on an older log, `gate`, each whole: its text, options and
    default, the concrete action and its tier, or the markdown shown at the stop."""

    KIND: ClassVar[SessionKind] = "question_asked"
    question: Question | None = None
    approval: ActionApproval | None = None
    gate: RetiredGate | None = None
    stop: ApprovalStop | None = None

    @property
    def question_id(self) -> str:
        """The id its answer names: the question's, the approval's `action_id`, the gate's."""
        if self.question is not None:
            return self.question.id
        if self.approval is not None:
            return self.approval.action_id
        if self.gate is not None:
            return self.gate.id
        if self.stop is not None:
            return self.stop.id
        raise PayloadError(
            "a question_asked line holds a question, an approval, a gate or a stop"
        )


@with_config(LINE)
@dataclass(frozen=True)
class QuestionAnsweredLine:
    """The choice, who gave it, and how long the wait was. `source` is an `AnswerSource` for a
    question and an `ApprovalDecision.source` for an approval, as given."""

    KIND: ClassVar[SessionKind] = "question_answered"
    id: str
    choice: str
    source: str
    elapsed_ms: Annotated[int, Field(ge=0)]
    note: str | None = None
    scope: str = "once"
    """For an approval, how far the person's answer reached as they gave it (`GrantScope`):
    `once`, `session` or `repo`, so the prompts can be counted by what was chosen.
    `once` for every other question."""


log = logging.getLogger(__name__)

WITHDRAWN: Final = "withdrawn"
"""The source of an answer nobody gave: the port raised (a stop, a closed connection), so the
question closed with no choice. Neither an `AnswerSource` nor an approval's source: no person
and no default decided it."""

RECLAIMED: Final = "its asker stopped before it was answered"
"""A withdrawn answer's note when the next writer closes what a dead holder left open."""

CANCELLED: Final = "cancelled"
"""A withdrawn answer's note when the asker's stop token was set (a hard stop, a close)."""

UNRECORDED: Final = "unrecorded"
"""A stop's source when its port named none: absent, and never read as a person."""

BUDGET_SPENT: Final = "the run's question budget is spent (CL-5): answered by its default"
"""The note on a question a run asked past `max_questions`, answered without asking."""

LINES: Final[Mapping[SessionKind, type]] = {
    "question_asked": QuestionAskedLine,
    "question_answered": QuestionAnsweredLine,
}
"""This module's kinds and their types: the goldens a frontend compiles against read them."""

_ASKED: Final = TypeAdapter(QuestionAskedLine)
_ANSWERED: Final = TypeAdapter(QuestionAnsweredLine)


def open_questions(
    lines: Iterable[tuple[SessionEvent, bytes]], *, withdrawn: bool = False
) -> tuple[QuestionAskedLine, ...]:
    """The questions asked and not yet answered, oldest first, from a session's lines and
    their payloads. Lines of other kinds are passed over. With `withdrawn`, a question closed
    as withdrawn counts as unanswered too: closed, but answered by nobody."""
    waiting: dict[str, QuestionAskedLine] = {}
    for event, raw in lines:
        if event.kind == "question_asked":
            asked = _ASKED.validate_json(raw)
            waiting[asked.question_id] = asked
        elif event.kind == "question_answered":
            answer = _ANSWERED.validate_json(raw)
            if not (withdrawn and answer.source == WITHDRAWN):
                waiting.pop(answer.id, None)
    return tuple(waiting.values())


def _reusable(session: Session, q: Question) -> QuestionAnsweredLine | None:
    """The answer a person gave this question earlier in the session: the same kind
    and words, answered by someone, not withdrawn. Never for an approval."""
    if q.kind is QuestionKind.PLAN_APPROVAL:
        return None
    asked: dict[str, Question] = {}
    found: QuestionAnsweredLine | None = None
    for event, _ in session.state.lines():
        if event.kind == "question_asked":
            line = _ASKED.validate_json(session.blob(event.payload_sha))
            if line.question is not None:
                asked[line.question.id] = line.question
        elif event.kind == "question_answered":
            answer = _ANSWERED.validate_json(session.blob(event.payload_sha))
            earlier = asked.get(answer.id)
            if (
                earlier is not None
                and answer.source == AnswerSource.HUMAN.value
                and (earlier.kind, earlier.text) == (q.kind, q.text)
            ):
                found = answer
    return found


def _asked_in_run(session: Session, run_id: str) -> int:
    """How many blocking questions run `run_id` has asked on this session's log."""
    n = 0
    for event, _ in session.state.lines():
        if event.kind == "question_asked" and event.run_id == run_id:
            line = _ASKED.validate_json(session.blob(event.payload_sha))
            if line.question is not None and line.question.blocking:
                n += 1
    return n


def open_in(
    source: Session | SessionLog, *, withdrawn: bool = False
) -> tuple[QuestionAskedLine, ...]:
    """`open_questions` over a session open for writing, or a log read without its lock."""
    from chickadee.context.session import Session

    if isinstance(source, Session):
        events = [e for e, _ in source.state.lines() if e.kind in LINES]
        return open_questions(
            ((e, source.blob(e.payload_sha)) for e in events), withdrawn=withdrawn
        )
    return open_questions(
        ((e, source.payload(e)) for e in source.events() if e.kind in LINES),
        withdrawn=withdrawn,
    )


def still_asking(session: Session, operation_id: str) -> bool:
    """Whether the permission question of operation `operation_id` was never answered:
    still open, or withdrawn because its asker stopped.
    A step cut off while it was asked never ran, so resume puts it back and asks again. Read
    after `open_reclaimed`, which withdraws whatever a dead holder left open, so open alone
    would never be seen here."""
    with session.lock:
        unanswered = open_in(session, withdrawn=True)
    return any(
        q.approval is not None and q.approval.operation_id == operation_id for q in unanswered
    )


def record_asked(
    session: Session, line: QuestionAskedLine, *, run_id: str | None = None
) -> int:
    """The question, durable before anyone is asked it. Returns its seq."""
    held = [x for x in (line.question, line.approval, line.stop) if x is not None]
    if len(held) != 1 or line.gate is not None:
        raise PayloadError("a question_asked line holds exactly one question, approval or stop")
    with session.lock:
        if any(q.question_id == line.question_id for q in open_in(session)):
            raise PayloadError(f"question {line.question_id} is already waiting for an answer")
        return session.append(line, run_id=run_id).seq


def record_answered(
    session: Session, line: QuestionAnsweredLine, *, run_id: str | None = None
) -> int:
    """How a waiting question was answered. Returns its seq."""
    with session.lock:
        if all(q.question_id != line.id for q in open_in(session)):
            raise PayloadError(f"question {line.id} was not asked, or is already answered")
        return session.append(line, run_id=run_id).seq


class RecordingPort:
    """The session's `InteractionPort`: the one home that puts a session's questions on its log.
    It wraps whichever port the engine was given (a terminal's, `RemoteInteraction`,
    `NonInteractive`) and writes `question_asked` before the inner port is called and
    `question_answered` after it returns, with the source as given and the wait measured. A
    confirmation a standing grant answers never reaches it (`GrantingPort`, over it).

    The engine hands one to every asker a session has: Zeus's `ask_user` and shell authority,
    the box's episode, undo and fork. None of them writes these lines itself. The box's driver
    also journals its own pair in the run journal (both journals)."""

    def __init__(
        self,
        port: InteractionPort,
        session: Session,
        run_id: str | None = None,
        *,
        max_questions: int = 3,
    ) -> None:
        self.inner = port
        self._session = session
        self._run_id = run_id
        self._max_questions = max_questions
        """The blocking questions one run may ask (`interaction.max_questions_per_run`).
        A question asked outside a run (Zeus's, an undo's) is the person's own conversation,
        and is not counted (Q9)."""

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        with self._session.lock:
            reused = _reusable(self._session, q)
            spent = (
                self._run_id is not None
                and q.blocking
                and _asked_in_run(self._session, self._run_id) >= self._max_questions
            )
        if reused is not None or spent:
            # Recorded as asked and answered, so the log shows the question and why no one
            # was asked it; the answer is the earlier person's, or the default.
            record_asked(self._session, QuestionAskedLine(question=q), run_id=self._run_id)
            if reused is not None:
                answer = Answer(
                    question_id=q.id,
                    choice=reused.choice,
                    source=AnswerSource.SESSION_REUSE,
                    note=reused.note,
                )
            else:
                answer = Answer(question_id=q.id, choice=q.default, source=AnswerSource.DEFAULT)
            note = answer.note if reused is not None else BUDGET_SPENT
            self._answered(q.id, answer.choice, answer.source.value, perf_counter(), note)
            return answer
        record_asked(self._session, QuestionAskedLine(question=q), run_id=self._run_id)
        started = perf_counter()
        try:
            answer = self.inner.ask(q, cancel=cancel)
        except BaseException as e:
            self._withdrawn(q.id, started, type(e).__name__)
            raise
        if cancel is not None and cancel.is_set():
            # The port contract: a cancelled ask is answered cancelled, whatever the port
            # returned (a default it fell back to is no one's answer, least of all a timeout).
            self._withdrawn(q.id, started, CANCELLED)
        else:
            self._answered(q.id, answer.choice, answer.source.value, started, answer.note)
        return answer

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        record_asked(self._session, QuestionAskedLine(approval=approval), run_id=self._run_id)
        started = perf_counter()
        try:
            decision = self.inner.confirm_action(approval, cancel=cancel)
        except BaseException as e:
            self._withdrawn(approval.action_id, started, type(e).__name__)
            raise
        if cancel is not None and cancel.is_set():
            # As `ask`'s: a cancelled approval is answered cancelled, never as the deny
            # its wait fell back to, so resume asks it again; the call is not run.
            self._withdrawn(approval.action_id, started, CANCELLED)
            return ApprovalDecision(
                action_id=approval.action_id, verdict="deny", source="default", note=CANCELLED
            )
        self._answered(
            approval.action_id,
            decision.verdict,
            decision.source,
            started,
            decision.note,
            scope=decision.scope,
        )
        return decision

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        """A stop: `question_asked` with the stop, the inner port's decision, then
        `question_answered`, its choice the answer and its note the person's words. Never
        budgeted or reused: every stop is asked, and an edit is shown again as a new stop, so
        the edited text is that showing's `question_asked`."""
        record_asked(self._session, QuestionAskedLine(stop=stop), run_id=self._run_id)
        started = perf_counter()
        try:
            decision = self.inner.confirm_stop(stop, cancel=cancel)
        except BaseException as e:
            self._withdrawn(stop.id, started, type(e).__name__)
            raise
        if cancel is not None and cancel.is_set():
            # As `confirm_action`'s: a stop whose wait was ended is withdrawn, never answered
            # `ask_stop` says so to its caller.
            self._withdrawn(stop.id, started, CANCELLED)
            return decision
        source = decision.source.value if decision.source is not None else UNRECORDED
        self._answered(stop.id, decision.answer.value, source, started, decision.note)
        return decision

    def for_run(self, run_id: str) -> RecordingPort:
        """The same port, its lines tagged with `run_id`: what a run's questions are asked
        through, so a reader of the session log knows which run asked them."""
        return RecordingPort(
            self.inner,
            self._session,
            run_id,
            max_questions=self._max_questions,
        )

    def _answered(
        self,
        qid: str,
        choice: str,
        source: str,
        started: float,
        note: str | None,
        *,
        scope: str = "once",
    ) -> None:
        elapsed = int((perf_counter() - started) * 1000)
        line = QuestionAnsweredLine(
            id=qid, choice=choice, source=source, elapsed_ms=elapsed, note=note, scope=scope
        )
        record_answered(self._session, line, run_id=self._run_id)

    def _withdrawn(self, qid: str, started: float, why: str) -> None:
        """Nobody answered: the question closes as `withdrawn`, with no choice, rather than
        reading as waiting for a person for as long as the log lives. Best effort: this runs
        while a stop or an error is already on its way out, and a log that cannot take the line
        (closed by that same stop) must not replace it with a storage error. The next writer
        withdraws what is left open (`withdraw_open`)."""
        try:
            self._answered(qid, "", WITHDRAWN, started, why)
        except Exception:
            log.warning("question %s could not be marked withdrawn", qid, exc_info=True)


@runtime_checkable
class SessionScoped(Protocol):
    """A port that serves several sessions at once and keys their questions by session (a
    served engine's `RemoteInteraction`): an answer must reach the session that asked, and two
    sessions may ask under one id. A port that is one person's at one terminal is not one."""

    def for_session(self, session_id: str) -> InteractionPort: ...


def scoped(port: InteractionPort, session_id: str) -> InteractionPort:
    """`port` bound to `session_id` when it serves several sessions; any other as it is."""
    return port.for_session(session_id) if isinstance(port, SessionScoped) else port


@runtime_checkable
class RunScoped(Protocol):
    """A port whose lines name the run that asked: the session's `RecordingPort`, and the
    grant check over it (`approval.grants.GrantingPort`)."""

    def for_run(self, run_id: str) -> InteractionPort: ...


def for_run(port: InteractionPort, run_id: str) -> InteractionPort:
    """`port` for run `run_id`'s questions: a session's port tags its lines with the run; any
    other port (a host's own, with no session log) is returned as it is."""
    return port.for_run(run_id) if isinstance(port, RunScoped) else port


def open_reclaimed(root: Path, session_id: str) -> Session:
    """The session open for writing, with every question its previous holder left waiting
    closed as `withdrawn`. Opening takes the log's lock, which no live process holds, so a
    question still open was asked by a process that died asking it: nobody can answer it, a
    run that asks it again must not be refused as a duplicate, and `liveness()` must not read
    it as waiting. Reclaiming a dead holder's leftovers is the next writer's (`liveness.py`).
    Every write-open of an existing session goes through here."""
    from chickadee.context.session import Session

    session = Session.open(root, session_id)
    try:
        for asked in open_in(session):
            line = QuestionAnsweredLine(
                id=asked.question_id, choice="", source=WITHDRAWN, elapsed_ms=0, note=RECLAIMED
            )
            record_answered(session, line)
    except BaseException:
        session.close()
        raise
    return session
