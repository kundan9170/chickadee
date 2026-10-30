"""`web.fetch.approval = "per_domain"`: the person allows each host once a session.

Before a fetch from a host this round has not been allowed, the person is asked, as a
`SCOPE_EXPANSION` question: reaching a new host is a scope-expanding action, which the project
rules
asks a dispatch-time confirmation for. Its default is to refuse, so a session with nobody
attached, or a person who does not answer, refuses the host and names it.

The question goes through the session's port (`engine._port`), which writes it and its answer
to the session log, and which answers the same question again from the person's earlier answer
in the session (`context/questions.py`). So *allow for this session* holds across the
session's rounds and its resumes with no grant kind of its own: the log is the grant.
"""

from __future__ import annotations

import threading
from typing import Final, Protocol

from chickadee.schema.interaction import Answer, AnswerSource, Option, Question, QuestionKind

__all__ = ["ALLOW", "DENY", "Asks", "HostApproval", "question"]

ALLOW: Final = "allow"
DENY: Final = "deny"


class Asks(Protocol):
    """The session port's `ask`: withdrawn when `cancel` is set, so a stopped round is not
    left waiting on the person."""

    def __call__(self, q: Question, *, cancel: threading.Event | None = None) -> Answer: ...


class HostApproval:
    """The hosts one session's researchers may fetch from under `per_domain`, one for the
    session (`engine._agents`). `ask` is the session's port's. Rounds run at once, so the hosts
    allowed are guarded by `_lock`, which is never held while the
    person is asked: two rounds asking for one new host at once may both ask, and the port's
    reuse answers the second."""

    def __init__(self, ask: Asks) -> None:
        self._ask = ask
        self._lock = threading.Lock()
        self._allowed: set[str] = set()

    def allows(self, host: str, call_id: str, cancel: threading.Event | None = None) -> bool:
        """Whether the person allows `host`: asked the first time this session reaches it,
        and answered from the session's earlier answer when there is one. `cancel`, the
        round's, withdraws the question: a stopped round asks nobody."""
        with self._lock:
            if host in self._allowed:
                return True
        answer = self._ask(question(host, call_id), cancel=cancel)
        allowed = answer.choice == ALLOW and answer.source in (
            AnswerSource.HUMAN,
            AnswerSource.SESSION_REUSE,
        )
        if allowed:
            with self._lock:
                self._allowed.add(host)
        return allowed


def question(host: str, call_id: str) -> Question:
    """The one question for `host`, worded the same every time it is asked, so the session's
    earlier answer answers it again."""
    return Question(
        id=f"{call_id}-web-host",
        kind=QuestionKind.SCOPE_EXPANSION,
        text=(
            f"A research agent wants to fetch pages from {host}. Allow {host} for the rest "
            "of this session?"
        ),
        options=(
            Option(key=ALLOW, label=f"Allow {host} for this session"),
            Option(key=DENY, label="Do not fetch from it"),
        ),
        default=DENY,
    )
