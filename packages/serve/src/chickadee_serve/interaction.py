"""Questions over the wire: the pending table, and `RemoteInteraction`.

`InteractionPort.ask` blocks until it has an `Answer`. Across a socket the two halves travel
apart: the question goes out on the record stream, because whoever asks writes
`question_asked` to the session log before the port is called, and the answer comes back
as `POST /answer/{qid}`, correlated by the question's id. Between the two, the question waits
here, in a slot keyed by that id.

**The asker journals, not the port.** A port is handed a `Question` and no journal, and both
askers write around the call: the box's driver to its run journal, and the session's recording
port to the session log. The guarantee (durable before anyone is told) is the
asker's. This module only waits, and says who answered.

**Who answered is the token's class**: `auth.source_for` turns the class into the
`AnswerSource`, and nothing in the request is read for it.

**A question always gets exactly one answer.** The first `POST` wins; a later one is
`QuestionAnswered` (409), naming the question, for as long as `RECENT` remembers it, and an
id never asked is `QuestionUnknown` (404). With nobody answering, `question_timeout_s`
resolves a question to its default as `timeout_default`, the CLI's knob and behaviour,
so a frontend that vanished cannot hang a session. The served engine passes
`config.interaction.question_timeout_s`; there is no second default here. A stop is the one
exception: it waits until answered or withdrawn, never approving itself.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Final, Literal

from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    GrantScope,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee_serve.auth import TokenClass, approval_source, source_for
from chickadee_serve.frames import STOP_CHOICES, approval_choices

__all__ = [
    "AUTOMATION_CANNOT_APPROVE",
    "RECENT",
    "ChoiceRefused",
    "Given",
    "PendingQuestions",
    "QuestionAnswered",
    "QuestionUnknown",
    "RemoteInteraction",
]

RECENT: Final = 256
"""Answered questions remembered, so a late second answer is `ANSWERED`, not `NOT_FOUND`."""

AUTOMATION_CANNOT_APPROVE: Final = (
    "denied: an automation token cannot approve an action; only a person can (§6.1)"
)
"""The note on an automation token's `approve`, which is recorded as the deny it is."""

_VERDICTS: Final[dict[str, tuple[Literal["approve", "deny", "abort"], GrantScope]]] = {
    "approve": ("approve", "once"),
    "session": ("approve", "session"),
    "repo": ("approve", "repo"),
    "deny": ("deny", "once"),
    "abort": ("abort", "once"),
}
"""Each choice an approval offers (`approval_choices`): its verdict, and how far it reaches."""

STOP_PERSON_ONLY: Final = ("edit", "note")
"""At a stop, the answers that change what is approved or what the author writes next: only a
person gives them. A script may approve or cancel."""

_TICK_S: Final = 0.1
"""How often a wait looks at its `cancel` token: a hard stop is answered within this."""


class QuestionUnknown(LookupError):
    """No question with this id is waiting here, nor was one answered recently."""


class QuestionAnswered(Exception):
    """The question was answered already, by another client or by its timeout."""


class ChoiceRefused(ValueError):
    """The choice is not one of the question's options."""


@dataclass(frozen=True)
class Given:
    """An answer as it arrived: the choice, its note, and the class of the token it came on."""

    choice: str
    token_class: TokenClass
    note: str | None = None
    text: str | None = None
    """At a stop, with `edit`: the person's markdown."""


@dataclass
class _Slot:
    choices: tuple[str, ...]
    """The keys an answer may choose; empty when the question offers free text."""
    person_only: tuple[str, ...] = ()
    """The choices only an interactive token may make: at a design's gate, the ones that
    change what is approved. A script may agree or refuse, never reshape."""
    arrived: threading.Event = field(default_factory=threading.Event)
    given: Given | None = None


_Key = tuple[str, str]
"""A slot's key: the session that asked, and the question's id. Ids are unique only within a
session (Zeus's is the model's tool-call id), so one session's answer never reaches another's
question, and two sessions may wait on the same id at once."""


class PendingQuestions:
    """The questions waiting on a person, by session and id. Guarded by `_lock`: `answer` runs
    on the server's request threads, `wait` on whichever thread is asking."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[_Key, _Slot] = {}
        self._recent: OrderedDict[_Key, None] = OrderedDict()

    def open(
        self,
        session_id: str,
        question_id: str,
        choices: tuple[str, ...] = (),
        person_only: tuple[str, ...] = (),
    ) -> None:
        with self._lock:
            if (session_id, question_id) in self._open:
                raise QuestionAnswered(f"question {question_id} is already waiting")
            self._open[(session_id, question_id)] = _Slot(choices, person_only)

    def waiting(self, session_id: str) -> tuple[str, ...]:
        """The ids of `session_id`'s questions waiting now."""
        with self._lock:
            return tuple(qid for sid, qid in self._open if sid == session_id)

    def answer(self, session_id: str, question_id: str, given: Given) -> None:
        """Deliver `given` to the question's wait. The first answer wins."""
        key = (session_id, question_id)
        with self._lock:
            slot = self._open.get(key)
            if slot is None:
                if key in self._recent:
                    raise QuestionAnswered(f"question {question_id} was already answered")
                raise QuestionUnknown(f"no question {question_id} is waiting")
            if slot.choices and given.choice not in slot.choices:
                raise ChoiceRefused(
                    f"{given.choice!r} is not an option of question {question_id}; "
                    f"choose one of {list(slot.choices)}"
                )
            if (
                given.choice in slot.person_only
                and given.token_class is not TokenClass.INTERACTIVE
            ):
                raise ChoiceRefused(
                    f"only a person may {given.choice} question {question_id}: an automation "
                    "token may approve or cancel, never change what is approved (§6.1, D187, "
                    "D196 §16)"
                )
            if slot.given is not None:
                raise QuestionAnswered(f"question {question_id} was already answered")
            slot.given = given
            slot.arrived.set()

    def wait(
        self,
        session_id: str,
        question_id: str,
        timeout_s: float,
        cancel: threading.Event | None = None,
    ) -> Given | None:
        """The answer, or None when `timeout_s` passed or `cancel` was set first. The slot is
        closed either way, and remembered as answered."""
        key = (session_id, question_id)
        with self._lock:
            slot = self._open.get(key)
        if slot is None:
            raise QuestionUnknown(f"no question {question_id} is waiting")
        deadline = time.monotonic() + timeout_s
        try:
            while not slot.arrived.is_set():
                if cancel is not None and cancel.is_set():
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                slot.arrived.wait(min(left, _TICK_S))
        finally:
            with self._lock:
                self._open.pop(key, None)
                self._recent[key] = None
                while len(self._recent) > RECENT:
                    self._recent.popitem(last=False)
                given = slot.given
        return given


class RemoteInteraction:
    """The `InteractionPort` a served engine is given: each question waits for a client's
    answer, up to `timeout_s`, and resolves to its default when none comes."""

    def __init__(
        self,
        pending: PendingQuestions,
        *,
        timeout_s: float,
        session_id: str = "",
    ) -> None:
        self.pending = pending
        self.timeout_s = timeout_s
        self.session_id = session_id
        """The session whose questions this port waits on. The engine binds one per session
        (`questions.scoped`); the unbound port's questions no session's route can answer, so
        they resolve to their defaults, as unattended ones do."""

    def for_session(self, session_id: str) -> RemoteInteraction:
        """`SessionScoped`: this port, its questions keyed by `session_id`."""
        return RemoteInteraction(self.pending, timeout_s=self.timeout_s, session_id=session_id)

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.pending.open(self.session_id, q.id, tuple(o.key for o in q.options))
        given = self.pending.wait(self.session_id, q.id, self.timeout_s, cancel)
        if given is None:
            return Answer(
                question_id=q.id, choice=q.default, source=AnswerSource.TIMEOUT_DEFAULT
            )
        return Answer(
            question_id=q.id,
            choice=given.choice,
            source=source_for(given.token_class),
            note=given.note,
        )

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        """Approved only by an interactive answer; with none in time, denied, never assumed.


        An automation token's answer is the default it agreed with, and an approval's
        default is deny: unattended, an action runs only under an authorization granted for
        it, which no answer is. So its `approve` is recorded as the deny it
        is, saying why; its `deny` and `abort` stand. Otherwise a script holding the weaker
        token would run what the tier floor holds for a person."""
        choices = approval_choices(
            grant=approval.grant is not None,
            exact=approval.exact is not None,
            offer_repo=approval.offer_repo,
        )
        self.pending.open(self.session_id, approval.action_id, choices)
        given = self.pending.wait(self.session_id, approval.action_id, self.timeout_s, cancel)
        if given is None:
            return ApprovalDecision(
                action_id=approval.action_id, verdict="deny", source="default"
            )
        # The slot admits only `choices`, so the choice is a verdict and a scope.
        (verdict, scope), note = _VERDICTS[given.choice], given.note
        if verdict == "approve" and given.token_class is not TokenClass.INTERACTIVE:
            verdict, scope, note = "deny", "once", AUTOMATION_CANNOT_APPROVE
        return ApprovalDecision(
            action_id=approval.action_id,
            verdict=verdict,
            source=approval_source(given.token_class),
            scope=scope,
            note=note,
        )

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        """A stop: a client renders the markdown from the `question_asked` record,
        and its answer chooses one of `STOP_CHOICES`. Who decided is the token's class:
        only a person may edit or note it (`STOP_PERSON_ONLY`).

        **It never approves itself**. A served session is attended, and the team
        runs beside the person's conversation, so a stop may wait while they talk to Zeus: a
        timeout's default would approve what nobody read. It waits until answered, or until
        `cancel` ends the wait (the run dropped, the process leaving, Zeus's turn stopped hard),
        which the asker reads as withdrawn, never as an answer (`ask_stop`)."""
        self.pending.open(self.session_id, stop.id, STOP_CHOICES, STOP_PERSON_ONLY)
        given = self.pending.wait(self.session_id, stop.id, math.inf, cancel)
        if given is None:
            # Only `cancel` ends this wait unanswered: the caller answers it withdrawn.
            return StopDecision(answer=StopAnswer.CANCEL, source=AnswerSource.DEFAULT)
        return StopDecision(
            answer=StopAnswer(given.choice),
            edited=given.text,
            note=given.note,
            source=source_for(given.token_class),
        )
