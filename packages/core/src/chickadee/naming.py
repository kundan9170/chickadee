"""A session's name, from the `title` role: one call, once, after the first turn the
session takes, on the conversation's worker.

**Once, and never in the way.** The call is made only while the session has no `title` line
(a person's name is never overwritten), at most once per session in this process, and after
the turn's words have all been streamed. A failure (a refused cap, an unpriced model, an error
from the model, an empty answer) leaves the first line as the title: it is
logged, never retried, never said to the person.

**Charged as any call**: the session ledger's `title` view, its spend line naming
the role. A cap that refuses the rung skips the call before anything is sent.

**Guarded state.** `_tried` is read and set only on the conversation's worker, which runs one
turn at a time; it needs no lock.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from chickadee.client.request import ChatRequest, Message, ModelClient
from chickadee.context.session import Session, Turn
from chickadee.schema.adapter import RetryParams, Usage
from chickadee.sessions import SessionTitle, named, record_title, title_of

__all__ = ["ASKED_CHARS", "Namer", "title_from"]

_log = logging.getLogger(__name__)

ASKED_CHARS: Final = 2000
"""How much of the first user turn, and of the first reply, the call is shown: a name needs
the opening, never the whole of a long paste."""

_QUOTES: Final = "\"'`\u201c\u201d\u2018\u2019*_#"
"""What a model wraps a name in, which the name is not: quotes, curly quotes, markdown."""


def title_from(answer: str) -> str | None:
    """The model's answer as a title: its first line, without the quotes, markdown marks or
    `Title:` a model may wrap it in, nor a trailing stop; cut as D100 cuts a first line. None
    for an answer with no words."""
    line = next((ln.strip() for ln in answer.splitlines() if ln.strip()), "")
    if line.lower().startswith("title:"):
        line = line[len("title:") :].strip()
    line = line.strip(_QUOTES).rstrip(".").strip()
    return title_of(line)


@dataclass
class Namer:
    """Called after every turn; names the session the first time it can."""

    session: Session
    client: ModelClient | None
    """The `title` role's rung's client; None where its model has no price row (D239 3)."""
    params: RetryParams | None
    template: str
    charge: Callable[[Usage], object]
    stopped: Callable[[], str | None]
    """The cap that refuses the rung, named, or None (`CostLedger.stopped_by`)."""
    _tried: bool = field(default=False)

    def __call__(self) -> None:
        if self._tried:
            return
        session = self.session
        with session.lock:
            state = session.state
            if named(state, session.session_id, lambda e: session.blob(e.payload_sha)):
                self._tried = True  # named already: the person's, or an earlier process's
                return
            turns = [
                r
                for e, r in state.lines()
                if e.session_id == session.session_id and isinstance(r, Turn)
            ]
        asked = next((t.text for t in turns if t.role == "user"), None)
        if asked is None:
            return  # no turn yet: nothing to name it from
        self._tried = True
        reply = next((t.text for t in turns if t.role == "assistant" and t.text), "")
        if self.client is None or self.params is None:
            return
        why = self.stopped()
        if why is not None:
            _log.info("session %s not named: %s", session.session_id, why)
            return
        prompt = self.template.format(asked=asked[:ASKED_CHARS], reply=reply[:ASKED_CHARS])
        request = ChatRequest(
            messages=(Message(role="user", content=prompt),), params=self.params
        )
        try:
            completion = self.client.send(request)
        except Exception:  # any failure leaves the first line as the title
            _log.warning("session %s not named: the title call failed", session.session_id,
                         exc_info=True)  # fmt: skip
            return
        self.charge(completion.usage)
        title = title_from(completion.text)
        if title is None:
            return
        try:
            record_title(session, SessionTitle(title, "model"))
        except Exception:
            _log.warning("session %s: its name could not be written", session.session_id,
                         exc_info=True)  # fmt: skip
