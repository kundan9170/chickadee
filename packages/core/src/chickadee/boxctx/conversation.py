"""A box role's conversation, journaled so resume rebuilds it exactly.

One value for both box roles, the task worker and Athena: a head that never moves (the fixed
instructions and what the role was given), then messages, appended. Every message is
journaled as a `box_message` line naming a run blob of its canonical JSON before it joins the
conversation, so `rebuilt` folds a role's lines back into the
same messages, in order, with its `box_compaction` lines applied. Nothing is re-rendered on
resume from a template or a record that may have changed since.

`fit` is D196 the placeholder for compaction: when the next request would not fit, the
oldest evictable results (reads and commands; never an edit tool's answer, never the newest
`keep_recent` exchanges) become receipts (`evict.live_receipt`) in one rebuild, recorded as one
`box_compaction` line; still too long is a `BoxExit`, which the transcript raises as
`ContextFull` out of `tools.loop.ask`.

"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from pydantic import BaseModel

from chickadee.boxctx.evict import apply_eviction, exchanges, live_receipt, plan_eviction
from chickadee.boxctx.exits import BoxExit
from chickadee.client.request import (
    ChatRequest,
    DocumentPart,
    ImagePart,
    Message,
    Part,
    TextPart,
    ToolSchema,
)
from chickadee.context.homes import Now
from chickadee.context.request import request_tokens
from chickadee.journal.projection import RunProjection
from chickadee.schema import RetryParams
from chickadee.schema.adapter import ToolCall
from chickadee.schema.journal import BoxCompactionP, BoxMessageP, BoxRoleName, EventKind

__all__ = [
    "BoxConversation",
    "ContextFull",
    "JournaledTranscript",
    "TranscriptJournal",
    "message_bytes",
    "message_of",
    "rebuilt",
]


def message_bytes(message: Message) -> bytes:
    """A message's canonical JSON: its fields, keys sorted, an image's bytes as base64."""
    content: object = (
        message.content
        if isinstance(message.content, str)
        else [_part(p) for p in message.content]
    )
    body = {
        "role": message.role,
        "content": content,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in message.tool_calls
        ],
        "cache_breakpoint": message.cache_breakpoint,
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def message_of(data: bytes) -> Message:
    """`message_bytes` read back."""
    body = json.loads(data)
    raw = body["content"]
    content: str | tuple[Part, ...] = (
        raw if isinstance(raw, str) else tuple(_part_of(p) for p in raw)
    )
    return Message(
        role=body["role"],
        content=content,
        tool_call_id=body["tool_call_id"],
        tool_calls=tuple(
            ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in body["tool_calls"]
        ),
        cache_breakpoint=body["cache_breakpoint"],
    )


def _part(part: Part) -> dict[str, str]:
    if isinstance(part, TextPart):
        return {"text": part.text}
    raw = {
        "media_type": part.media_type,
        "sha": part.sha,
        "data": base64.b64encode(part.data).decode(),
    }
    if isinstance(part, DocumentPart):
        raw["name"] = part.name  # D245: only a document carries a name
    return raw


def _part_of(raw: dict[str, str]) -> Part:
    if "text" in raw:
        return TextPart(raw["text"])
    data = base64.b64decode(raw["data"])
    if "name" in raw:
        return DocumentPart(raw["media_type"], raw["sha"], raw["name"], data)
    return ImagePart(raw["media_type"], raw["sha"], data)


@dataclass(frozen=True)
class BoxConversation:
    """A head, then messages; `evicted` names the calls whose results are receipts."""

    role: BoxRoleName
    head: tuple[Message, ...]
    messages: tuple[Message, ...]
    params: RetryParams
    tools: tuple[ToolSchema, ...] = ()
    evicted: frozenset[str] = frozenset()
    session_key: str | None = None
    commands: frozenset[str] = frozenset()
    """The offered tools that run a command (their spec mutates the environment): an evicted
    result of one keeps its exit line and output handle."""

    def request(self) -> ChatRequest:
        shown = apply_eviction(
            self.messages,
            self.evicted,
            render=lambda ex, result: live_receipt(
                ex, result, command=ex.tool in self.commands
            ),
        )
        return ChatRequest(
            messages=(*self.head, *shown),
            params=self.params,
            tools=self.tools,
            session_key=self.session_key,
        )

    def appended(self, *messages: Message) -> BoxConversation:
        return replace(self, messages=(*self.messages, *messages))

    def fit(
        self, budget: int, *, keep_recent: int, keep_tools: frozenset[str]
    ) -> BoxConversation | BoxExit:
        """This conversation within `budget` tokens: itself when it fits; else with its oldest
        evictable results as receipts; else the exit that says it cannot."""
        tokens = request_tokens(self.request())
        if tokens <= budget:
            return self
        every = exchanges(self.messages, skip=self.evicted)
        recent = every[-keep_recent:] if keep_recent > 0 else ()
        newest: set[str] = {ex.call_id for ex in recent}
        ordered = [ex for ex in every if ex.tool not in keep_tools and ex.call_id not in newest]
        chosen, _ = plan_eviction(ordered, tokens, budget)
        fitted = replace(self, evicted=self.evicted | frozenset(chosen))
        after = request_tokens(fitted.request())
        if after <= budget:
            return fitted
        return BoxExit(
            "worker_overflow_mid" if self.role == "worker" else "athena_exhausted",
            self.role,
            after,
            budget,
            f"{self.role}'s request is {after} tokens after evicting every evictable result",
        )


class ContextFull(Exception):
    """The conversation does not fit even after eviction: `tools.loop.ask` is left by raising
    this, and the role's run reads it as its round's end (`context_full`)."""

    def __init__(self, exit: BoxExit) -> None:
        super().__init__(exit.detail)
        self.exit = exit


class TranscriptJournal(Protocol):
    """What the transcript writes through: the run journal's append and blobs."""

    def append(
        self,
        kind: EventKind,
        payload: BaseModel,
        *,
        task_id: str | None = None,
        round: int | None = None,
    ) -> int: ...

    def put_blob(self, content: bytes) -> str: ...


class JournaledTranscript:
    """`tools.loop.RequestTranscript` over a `BoxConversation`: `request()` fits first and
    journals a compaction once; `add()` journals each message, then appends it. One per role
    per task round, or per Athena request."""

    def __init__(
        self,
        conversation: BoxConversation,
        journal: TranscriptJournal,
        *,
        budget: int,
        keep_recent: int,
        keep_tools: frozenset[str],
        task_id: str | None,
        round: int | None,
    ) -> None:
        self.conversation = conversation
        self._journal = journal
        self._budget = budget
        self._keep_recent = keep_recent
        self._keep_tools = keep_tools
        self._task = task_id
        self._round = round

    def begin(self) -> None:
        """The head journaled, as the conversation's first lines: once, when it opens."""
        for message in self.conversation.head:
            self._line(message)

    def request(self) -> ChatRequest:
        conv = self.conversation
        fitted = conv.fit(
            self._budget, keep_recent=self._keep_recent, keep_tools=self._keep_tools
        )
        if isinstance(fitted, BoxExit):
            raise ContextFull(fitted)
        if fitted.evicted != conv.evicted:
            self._journal.append(
                EventKind.BOX_COMPACTION,
                BoxCompactionP(
                    role=conv.role,
                    steps=("evict",),
                    evict=tuple(sorted(fitted.evicted - conv.evicted)),
                    tokens_before=request_tokens(conv.request()),
                    tokens_after=request_tokens(fitted.request()),
                ),
                task_id=self._task,
                round=self._round,
            )
            self.conversation = fitted
        return self.conversation.request()

    def now(self, key: str) -> Now:
        """The conversation as it stands, for its role's reader: `key`
        names it, so a reader's memory of it is its own."""
        conv = self.conversation
        return Now(
            key,
            conv.messages,
            conv.evicted,
            room=lambda: self._budget - request_tokens(self.conversation.request()),
        )

    def add(self, *messages: Message) -> None:
        for message in messages:
            self._line(message)
            self.conversation = self.conversation.appended(message)

    def _line(self, message: Message) -> None:
        sha = self._journal.put_blob(message_bytes(message))
        self._journal.append(
            EventKind.BOX_MESSAGE,
            BoxMessageP(role=self.conversation.role, side=message.role, sha=sha),
            task_id=self._task,
            round=self._round,
        )


def rebuilt(
    run: RunProjection,
    role: BoxRoleName,
    task: str | None,
    blob: Callable[[str], bytes],
    *,
    params: RetryParams,
    tools: tuple[ToolSchema, ...] = (),
    session_key: str | None = None,
) -> BoxConversation:
    """A role's conversation from its lines, every round of it: the worker's for `task`, or
    Athena's (`task` None) every line of hers. The head is what came before its first reply;
    a call and its result stay one pair, since both are lines in order."""
    lines: Sequence[object]
    if task is not None:
        lines = [p.payload for p in run.tasks.get(task, ())]
    else:
        placed = sorted(
            (*run.lines(BoxMessageP), *run.lines(BoxCompactionP)), key=lambda p: p.seq
        )
        lines = [p.payload for p in placed if p.task_id is None]
    messages: list[Message] = []
    evicted: set[str] = set()
    for line in lines:
        if isinstance(line, BoxMessageP) and line.role == role:
            messages.append(message_of(blob(line.sha)))
        elif isinstance(line, BoxCompactionP) and line.role == role:
            evicted.update(line.evict)
    first_reply = next((i for i, m in enumerate(messages) if m.role == "assistant"), None)
    cut = len(messages) if first_reply is None else first_reply
    return BoxConversation(
        role=role,
        head=tuple(messages[:cut]),
        messages=tuple(messages[cut:]),
        params=params,
        tools=tools,
        evicted=frozenset(evicted),
        session_key=session_key,
    )
