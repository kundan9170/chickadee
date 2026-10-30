"""`ChatRequest` — the request as a type.

Until D101 the only client method took the coder's two strings, while the assembly pipeline
produced a messages array and the cache table listed `tools` as a frozen region: a request
three documents described and no type carried. D94's resident schemas had nowhere to be
resident, and D47's native `tool_calls` branch could never fire, because no request ever
declared `tools`. This is what stage 1 produces and stage 6 sends.

The client's own type, not `schema/`: `Prompt`, `Completion` and `RetryParams` are
frozen contract 5 and are not edited. `Prompt` becomes a *rendering* into a request
(`to_request`), and the coder path's bytes are unchanged by it.

A message's content is text, or content parts when an image rides. Parts are the
exception: every text-only path keeps a plain string, so its wire bytes do not change.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from chickadee.client.live import Live
from chickadee.schema import Completion, Prompt, RetryParams, ToolCall

__all__ = [
    "IMAGE_MEDIA_TYPES",
    "PDF_MEDIA_TYPE",
    "ChatRequest",
    "DocumentPart",
    "ImagePart",
    "Message",
    "ModelClient",
    "Part",
    "Role",
    "TextPart",
    "ToolGrammar",
    "ToolSchema",
    "has_documents",
    "has_images",
    "text_of",
    "to_request",
]

Role = Literal["system", "user", "assistant", "tool"]

IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
"""What both transports accept. Anything else is `NOT_TEXT` at `read`, upstream."""

PDF_MEDIA_TYPE = "application/pdf"
"""The one document type both transports accept."""


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ImagePart:
    """An image, already downscaled to `image_max_edge` by `read`."""

    media_type: str
    sha: str
    """The handle's content sha (the original's), so a transport can name the image."""
    data: bytes

    def __post_init__(self) -> None:
        if self.media_type not in IMAGE_MEDIA_TYPES:
            raise ValueError(
                f"{self.media_type!r} is not an image either transport accepts "
                f"({sorted(IMAGE_MEDIA_TYPES)})"
            )


@dataclass(frozen=True)
class DocumentPart:
    """A PDF, sent whole to a model that reads PDFs: Anthropic's `document`
    block, OpenAI Chat Completions' `file` part. A model without `supports_documents` is sent
    its text layer instead, upstream, and never one of these."""

    media_type: str
    sha: str
    """The blob's sha, so a transport can name the document."""
    name: str
    """The file's name, which OpenAI's `file` part carries."""
    data: bytes

    def __post_init__(self) -> None:
        if self.media_type != PDF_MEDIA_TYPE:
            raise ValueError(f"{self.media_type!r} is not a document either transport accepts")


Part = TextPart | ImagePart | DocumentPart


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | tuple[Part, ...]
    """Text, or parts when an image rides."""
    tool_call_id: str | None = None
    """Set iff `role == "tool"`: the call this message answers."""
    tool_calls: tuple[ToolCall, ...] = ()
    """Set only when `role == "assistant"`: the calls it made."""
    cache_breakpoint: bool = False
    """Stage 4 marks the last message of a stability class. `EXPLICIT` transports
    turn it into a cache marker; `AUTOMATIC` ones ignore it. Never set by a client — the
    assembler knows where the boundaries are, and the client must not guess."""

    def __post_init__(self) -> None:
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("a tool message, and only a tool message, carries tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only an assistant message carries tool_calls")
        if isinstance(self.content, str):
            return
        if not self.content:
            raise ValueError("parts content carries at least one part; use '' for none")
        # An image reaches a model as something a person attached or a tool returned.
        # Neither vendor takes one in a system or assistant turn, so the refusal is here
        # rather than a 400 after the request was paid for.
        if self.role in ("system", "assistant") and any(
            isinstance(p, ImagePart | DocumentPart) for p in self.content
        ):
            raise ValueError(f"a {self.role} message carries no image or document (D110)")


@dataclass(frozen=True)
class ToolGrammar:
    """A tool decoded under a grammar: the model writes one free text, which the
    grammar holds to its language, and the call's one argument is that text."""

    argument: str
    """The argument the text is: the spec's one free-text property (`patch`)."""
    definition: str
    syntax: Literal["lark"] = "lark"


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    """The stub line or the full description."""
    schema: Mapping[str, Any]
    """JSON schema for the arguments."""
    grammar: ToolGrammar | None = None
    """Set iff the role's policy constrains this tool (`RolePolicy.constrain`): a client sends
    it decoded under the grammar, on a model whose `grammar_tools` says it can."""


@dataclass(frozen=True)
class ChatRequest:
    messages: tuple[Message, ...]
    params: RetryParams
    tools: tuple[ToolSchema, ...] = ()
    """Empty, the coder's four, or `SURFACE`'s sixteen."""
    session_key: str | None = None
    """Cache affinity; best-of-n candidates share one."""
    cancel: threading.Event | None = None
    """The session's token. Checked before every attempt, so a cancelled session
    does not start a request it will discard."""
    live: Live | None = None
    """Handed each piece of the reply as it streams, on the caller's thread
    (`client/live.py`). The `Completion` is the same with it or without."""
    force_tool: bool = False
    """The reply must be a call to one of `tools`. Asked, never promised: the client
    forces it only where its provider enforces it (`openai_client.Forcing`), and sends the
    request unforced otherwise."""

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a request carries at least one message")
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"tool names repeat in one request: {names}")


class ModelClient(Protocol):
    """The one client protocol: every caller that talks to a model types its
    client as this. `OpenAIClient` and `AnthropicClient` satisfy it structurally.

    `close` is part of it so that whoever owns a client releases its connection pool by
    calling it, never by looking for it: a transport dropped without closing is a leak no
    test sees. Both transports tolerate a second close."""

    def send(self, request: ChatRequest) -> Completion: ...

    def close(self) -> None: ...


def to_request(
    prompt: Prompt,
    params: RetryParams,
    *,
    tools: tuple[ToolSchema, ...] = (),
    session_key: str | None = None,
) -> ChatRequest:
    """The coder path: `Prompt{prefix, volatile}` → one system message and one user message,
    content byte-identical. The prefix is the byte-stable part (invariant 8), so it is the
    system message a provider's prefix cache can hold."""
    return ChatRequest(
        messages=(
            Message(role="system", content=prompt.prefix),
            Message(role="user", content=prompt.volatile),
        ),
        params=params,
        tools=tools,
        session_key=session_key,
    )


def text_of(message: Message) -> str:
    """The message's text: its content, or its text parts concatenated."""
    if isinstance(message.content, str):
        return message.content
    return "".join(p.text for p in message.content if isinstance(p, TextPart))


def has_images(request: ChatRequest) -> bool:
    return any(
        isinstance(p, ImagePart)
        for m in request.messages
        if not isinstance(m.content, str)
        for p in m.content
    )


def has_documents(request: ChatRequest) -> bool:
    return any(
        isinstance(p, DocumentPart)
        for m in request.messages
        if not isinstance(m.content, str)
        for p in m.content
    )


def prompt_text(request: ChatRequest) -> str:
    """Every message's text, concatenated — what the usage estimator measures when a
    server reports no usage. For a `to_request` request it is `Prompt.full()`. An
    image counts nothing here; the estimate is crude on purpose and says so."""
    return "".join(text_of(m) for m in request.messages)
