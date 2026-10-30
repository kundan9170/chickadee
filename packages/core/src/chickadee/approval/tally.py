"""The permission prompts a session asked, counted.

The tally starts strict on purpose: a line that is not plain is never granted, and a repository
grant covers only its row's shape. A rule is loosened only on evidence and this is
where the evidence comes from: which commands were asked about, how often, what the person
answered, and why no grant could cover them.

**Read-only, for measuring.** It reads the session log and writes nothing; no rule reads it, and
no model sees it. Every fact is one the log already holds: a `question_asked` line carries its
`ActionApproval`, a `question_answered` line the verdict and the scope the person chose, and an
`action_granted` line an action a held grant covered without asking, or one the person's
`sandbox.never_ask` ran unasked, which is no grant and is counted apart. The measurement
runs read it rather than counting a second
way.

**Why a grant could not cover a question** is read from the fields the prompt carries:

- `once_only`: no binding at all. The line is not plain, holds a command no row names, holds a
  destructive command, or no single binding covers its commands.
- `outside_shape`: an exact binding but no repository one. The command fell outside its row's
  grant shape, or is a git command pointed at another repository.
- `leaves_the_machine`: a repository binding the prompt did not offer: its effects leave the
  machine and its row has no shape, so it is granted for the repository only by hand.
- `no_grant_held`: both offered, and no grant was held yet.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import TypeAdapter

from chickadee.approval.grants import ACTION_GRANTED, ActionGrantedLine
from chickadee.context.questions import WITHDRAWN, QuestionAnsweredLine, QuestionAskedLine
from chickadee.context.session import Session
from chickadee.schema.interaction import ActionApproval

__all__ = ["Answer", "PromptRow", "PromptTally", "Reason", "prompt_tally"]

Reason = Literal["once_only", "outside_shape", "leaves_the_machine", "no_grant_held"]
Answer = Literal["once", "session", "repo", "no", "withdrawn", "unanswered"]

EXAMPLES: Final = 3
"""How many of a row's questions it keeps as they were shown, to tell its lines apart."""

_ASKED: Final = TypeAdapter(QuestionAskedLine)
_ANSWERED: Final = TypeAdapter(QuestionAnsweredLine)
_GRANTED: Final = TypeAdapter(ActionGrantedLine)


@dataclass(frozen=True)
class PromptRow:
    """One kind of question: the class and prefix its binding names (its exact line when that
    is all it has; nothing for `once_only`), why no grant covered it, and how it was
    answered."""

    cls: str
    prefix: tuple[str, ...]
    reason: Reason
    asked: int
    once: int = 0
    session: int = 0
    repo: int = 0
    no: int = 0
    withdrawn: int = 0
    unanswered: int = 0
    examples: tuple[str, ...] = field(default=())
    """Up to `EXAMPLES` of its questions, as the prompt showed them."""


@dataclass(frozen=True)
class PromptTally:
    """Every row a session asked, most asked first, and the questions its grants saved."""

    rows: tuple[PromptRow, ...]
    granted: int
    """Actions a held grant covered without asking: questions the grants saved."""
    never_asked: int = 0
    """Actions `sandbox.never_ask` ran unasked: no grant covered them, so they
    say nothing about what a grant could save."""

    @property
    def asked(self) -> int:
        return sum(row.asked for row in self.rows)


def reason_of(approval: ActionApproval) -> Reason:
    """Why no grant covered `approval`, from the fields the prompt carries."""
    if approval.exact is None and approval.grant is None:
        return "once_only"
    if approval.grant is None:
        return "outside_shape"
    if not approval.offer_repo:
        return "leaves_the_machine"
    return "no_grant_held"


def prompt_tally(session: Session) -> PromptTally:
    """The tally of `session`'s permission prompts, read from its log."""
    asked: list[ActionApproval] = []
    answers: dict[str, Answer] = {}
    granted = never_asked = 0
    for kind, payload in _lines(session):
        if kind == "question_asked":
            line = _ASKED.validate_json(payload)
            if line.approval is not None:
                asked.append(line.approval)
        elif kind == "question_answered":
            answer = _ANSWERED.validate_json(payload)
            answers[answer.id] = _answer(answer)
        elif kind == ACTION_GRANTED:
            if _GRANTED.validate_json(payload).scope == "never_ask":
                never_asked += 1
            else:
                granted += 1
    counts: dict[tuple[Reason, str, tuple[str, ...]], dict[Answer, int]] = {}
    examples: dict[tuple[Reason, str, tuple[str, ...]], list[str]] = {}
    for approval in asked:
        key = _key(approval)
        tally = counts.setdefault(key, {})
        got = answers.get(approval.action_id, "unanswered")
        tally[got] = tally.get(got, 0) + 1
        shown = examples.setdefault(key, [])
        if len(shown) < EXAMPLES and approval.display not in shown:
            shown.append(approval.display)
    rows = [
        PromptRow(
            cls=cls,
            prefix=prefix,
            reason=reason,
            asked=sum(by.values()),
            once=by.get("once", 0),
            session=by.get("session", 0),
            repo=by.get("repo", 0),
            no=by.get("no", 0),
            withdrawn=by.get("withdrawn", 0),
            unanswered=by.get("unanswered", 0),
            examples=tuple(examples[(reason, cls, prefix)]),
        )
        for (reason, cls, prefix), by in counts.items()
    ]
    rows.sort(key=lambda r: (-r.asked, r.reason, r.cls, r.prefix))
    return PromptTally(tuple(rows), granted, never_asked)


def _key(approval: ActionApproval) -> tuple[Reason, str, tuple[str, ...]]:
    binding = approval.grant or approval.exact
    if binding is None:
        return "once_only", "", ()
    return reason_of(approval), binding.cls, binding.prefix


def _answer(line: QuestionAnsweredLine) -> Answer:
    if line.source == WITHDRAWN:
        return "withdrawn"
    if line.choice != "approve":
        return "no"
    if line.scope == "session":
        return "session"
    if line.scope == "repo":
        return "repo"
    return "once"


def _lines(session: Session) -> Iterator[tuple[str, bytes]]:
    with session.lock:
        events = [e for e, _ in session.state.lines()]
    for event in events:
        if event.kind in ("question_asked", "question_answered", ACTION_GRANTED):
            yield event.kind, session.blob(event.payload_sha)
