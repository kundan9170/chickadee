"""An agent's conversation: one transcript, each message on the session log before it joins.


A spawned agent starts from nothing but its prompt, its task and the handles it was passed,
so its conversation is a list that only grows: no compaction, no eviction. It
is held by the shared loop (`tools/loop.py`) through `RequestTranscript`, and `add` writes each
message as an `agent_message` line naming a session blob of its canonical JSON
(`boxctx/conversation.message_bytes`, the one form a message is stored in) before it joins. So a
follow-up after a resume rebuilds it exactly from the log (`agents/resume.py`).

**When it no longer fits** its rung's window, the next request is never sent: `request` raises
`Overflow`, which the spawn reads as the round's end. A conversation an agent has filled is a
round that ran out, and the report says so; a follow-up on it is refused, with the advice to
spawn a fresh agent passing the old report's handle (Amendment 1 decision 2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from chickadee.client.request import ChatRequest, Message, ToolSchema
from chickadee.context.request import request_tokens
from chickadee.schema import RetryParams

__all__ = ["AgentTranscript", "Overflow"]


class Overflow(Exception):
    """The next request would not fit the agent's window."""

    def __init__(self, tokens: int, budget: int) -> None:
        super().__init__(
            f"the conversation is {tokens:,} tokens, past the {budget:,} it may be"
        )
        self.tokens = tokens
        self.budget = budget


@dataclass
class AgentTranscript:
    """`RequestTranscript` for one agent. `record` journals a message before it joins; None
    keeps it in memory only (a test's composition). `budget` is the tokens a request may hold:
    the rung's window less its output reserve. `recorded` is how many of `messages` the log
    already holds: none for a spawn's head, all of them for a follow-up's (`resume.py`)."""

    messages: list[Message]
    params: RetryParams
    budget: int
    tools: tuple[ToolSchema, ...] = ()
    record: Callable[[Message], None] | None = None
    session_key: str | None = None
    recorded: int = 0
    _started: bool = field(default=False, repr=False)

    def start(self) -> None:
        """Journal the head the log does not hold yet, once, before the first request."""
        if self._started:
            return
        self._started = True
        if self.record is not None:
            for message in self.messages[self.recorded :]:
                self.record(message)

    def request(self) -> ChatRequest:
        self.start()
        request = ChatRequest(
            messages=tuple(self.messages),
            params=self.params,
            tools=self.tools,
            session_key=self.session_key,
        )
        tokens = request_tokens(request)
        if tokens > self.budget:
            raise Overflow(tokens, self.budget)
        return request

    def add(self, *messages: Message) -> None:
        self.start()
        for message in messages:
            if self.record is not None:
                self.record(message)
            self.messages.append(message)

    def fits(self, extra: str) -> bool:
        """Whether a request holding the conversation and `extra` more would fit: what a
        follow-up asks before it adds its message."""
        probe = ChatRequest(
            messages=(*self.messages, Message(role="user", content=extra)),
            params=self.params,
            tools=self.tools,
        )
        return request_tokens(probe) <= self.budget
