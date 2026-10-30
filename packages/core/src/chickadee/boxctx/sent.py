"""What a box role's model call sent, exactly as its transport got it.

A role's conversation is journaled message by message (`conversation.BoxConversation`), but the
request a provider receives is more than that: the tool schemas, the sampling settings, and,
when a client wraps the transport and rewrites the request, a different set of messages
altogether. `SentRequests` sits directly on the rung's transport, beneath any such wrapper, and
keeps the last request it passed on; `record_sent` journals it as run blobs: each message in
`message_bytes`' canonical form (so a message the conversation already journaled is the same
blob, stored once), the tools, and one small index naming them. The index's sha goes on the
call's `model_call` line as `sent_sha`.

Never parsed by the harness: it is a record for a person reading the run, not state.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from chickadee.boxctx.conversation import message_bytes

if TYPE_CHECKING:
    from chickadee.client.request import ChatRequest, ModelClient
    from chickadee.journal.port import JournalPort
    from chickadee.schema.adapter import Completion

__all__ = ["SentRequests", "record_sent"]


class SentRequests:
    """A `ModelClient` that passes every request on unchanged and keeps the last one.

    One per role, wrapped once around the rung's transport; the role's calls are sequential,
    so the request kept is always the one the latest answer replied to."""

    def __init__(self, inner: ModelClient) -> None:
        self._inner = inner
        self.last: ChatRequest | None = None

    def send(self, request: ChatRequest) -> Completion:
        self.last = request
        return self._inner.send(request)

    def close(self) -> None:
        self._inner.close()

    def take(self) -> ChatRequest | None:
        """The last request, forgotten once taken, so a later line never names it again."""
        last, self.last = self.last, None
        return last


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def record_sent(journal: JournalPort, request: ChatRequest) -> str:
    """`request` as run blobs; the sha of the index that names them."""
    messages = [journal.put_blob(message_bytes(m)) for m in request.messages]
    tools = journal.put_blob(
        _canonical(
            [
                {"name": t.name, "description": t.description, "parameters": dict(t.schema)}
                for t in request.tools
            ]
        )
    )
    return journal.put_blob(
        _canonical(
            {
                "messages": messages,
                "tools": tools,
                "params": request.params.model_dump(mode="json"),
                "force_tool": request.force_tool,
            }
        )
    )
