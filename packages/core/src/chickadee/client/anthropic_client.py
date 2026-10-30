"""Anthropic-native transport.

The one case D82 measured where the OpenAI-compatible layer drops caching: Anthropic caches
exactly what a request marks, so the marks have to reach the wire. This client wraps the
official `anthropic` package and nothing else — never a translator between our stage-4
markers and the wire, because a translator is where the byte invariant dies silently.

**It places no cache marker of its own.** Stage 4 sets `Message.cache_breakpoint` on
the last message of each stability class; this client spells each one as `cache_control`
on that message's last block, and refuses a request carrying more than the provider's four.
The assembler knows where the boundaries are; a client that guessed would write cache
entries nothing reads — a surcharge, not a saving.

**Cache writes are counted.** `cache_creation_input_tokens` fills `Usage.cache_write_tokens`,
the budget's third term; `OpenAIClient` leaves it `None`.

**Images are direct.** An image block is legal inside a `tool_result` on this API, so
the OpenAI transport's compensation has no counterpart here.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from typing import Any, cast

import anthropic
import httpx2

from chickadee.client.affinity import AffinityTable
from chickadee.client.errors import ANTHROPIC_HEADERS_ENV, CredentialError
from chickadee.client.live import Piece, Text, Thought, ToolArgs
from chickadee.client.openai_client import (
    MAX_BREAKPOINTS,
    RETRY_POLICY,
    EndpointSpec,
    Forcing,
    Retry,
    RetryPolicy,
    SdkErrors,
    ambient_header_names,
    api_key,
    completion,
    dump,
    failed_attempt,
    probe_usage,
    refuse_ambient_credential,
    refuse_images,
    run_ladder,
)
from chickadee.client.request import (
    ChatRequest,
    DocumentPart,
    Message,
    Part,
    TextPart,
    prompt_text,
)
from chickadee.client.sleep import ModelConnections
from chickadee.client.stream import Silences, StreamEnded, read_stream
from chickadee.errors import HarnessBug
from chickadee.schema import Completion, ReasoningSpec, ToolCall, Usage

__all__ = [
    "MAX_BREAKPOINTS",
    "AnthropicClient",
    "anthropic_messages",
    "anthropic_system",
    "anthropic_tools",
]

_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}

ANTHROPIC_ERRORS = SdkErrors(
    timeout=anthropic.APITimeoutError,
    connection=anthropic.APIConnectionError,
    credential=(anthropic.AuthenticationError, anthropic.PermissionDeniedError),
    status=anthropic.APIStatusError,
    api=anthropic.APIError,
)

_EFFORT = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "max": "max"}
"""D46's levels onto this API's `effort`. It has no `minimal`; `low` is its floor."""


def _mark(blocks: list[dict[str, Any]], m: Message) -> list[dict[str, Any]]:
    if m.cache_breakpoint and blocks:
        blocks[-1] = {**blocks[-1], "cache_control": dict(_EPHEMERAL)}
    return blocks


def _block(part: Part) -> dict[str, Any] | None:
    """One content block. An empty text block is None: this API rejects one with a 400, and
    it carries no bytes a model could read."""
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text} if part.text else None
    if isinstance(part, DocumentPart):
        # A PDF whole, its pages' text and images, on a model with `supports_documents`.
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": part.media_type,
                "data": base64.b64encode(part.data).decode("ascii"),
            },
            "title": part.name,
        }
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": part.media_type,
            "data": base64.b64encode(part.data).decode("ascii"),
        },
    }


def _parts(m: Message) -> tuple[Part, ...]:
    return (TextPart(m.content),) if isinstance(m.content, str) else m.content


def _content_blocks(m: Message) -> list[dict[str, Any]]:
    return [b for b in (_block(p) for p in _parts(m)) if b is not None]


def _blocks(m: Message) -> list[dict[str, Any]]:
    """One message's content blocks. A tool message is a `tool_result` a user turn
    carries; an assistant's calls are `tool_use` blocks after its text."""
    if m.role == "tool":
        result: dict[str, Any] = {"type": "tool_result", "tool_use_id": m.tool_call_id}
        if isinstance(m.content, str):
            result["content"] = m.content
        else:
            result["content"] = _content_blocks(m)
        return [result]
    blocks = _content_blocks(m)
    blocks.extend(
        {"type": "tool_use", "id": c.id, "name": c.name, "input": dict(c.arguments)}
        for c in m.tool_calls
    )
    return blocks


def _check_breakpoints(request: ChatRequest) -> None:
    marks = sum(1 for m in request.messages if m.cache_breakpoint)
    if marks > MAX_BREAKPOINTS:
        # Stage 4's rule is outermost first; a fifth marker reaching the client means it
        # did not run, and dropping one here would be the client guessing which.
        raise ValueError(
            f"{marks} cache breakpoints; the provider takes at most {MAX_BREAKPOINTS} "
            "(04-adapter.md §3.2 — stage 4 drops the innermost)"
        )


def anthropic_system(request: ChatRequest) -> list[dict[str, Any]]:
    """The leading system messages, as the `system` parameter's text blocks."""
    out: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role != "system":
            break
        out.extend(_mark(_content_blocks(m), m))
    return out


def anthropic_messages(request: ChatRequest) -> list[dict[str, Any]]:
    """The conversation after the system prefix. Consecutive messages that land on the same
    wire role merge into one turn — a tool result and the user text after it are one user
    turn to this API, which requires the roles to alternate. A message with no bytes (empty
    text, no calls) adds no block, and so no turn."""
    body = list(request.messages)
    while body and body[0].role == "system":
        body.pop(0)
    out: list[dict[str, Any]] = []
    for m in body:
        if m.role == "system":
            raise ValueError("a system message after the conversation began has no wire form")
        role = "assistant" if m.role == "assistant" else "user"
        blocks = _mark(_blocks(m), m)
        if not blocks:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return out


def anthropic_tools(request: ChatRequest) -> list[dict[str, Any]]:
    """The request's tools. Anthropic decodes none under a grammar: a constrained
    tool reaching here would go out unconstrained, and a run measured as the constrained arm
    would not be one, so it is refused as the load rule's miss."""
    constrained = [t.name for t in request.tools if t.grammar is not None]
    if constrained:
        raise HarnessBug(f"Anthropic decodes no tool under a grammar: {constrained}")
    return [
        {"name": t.name, "description": t.description, "input_schema": dict(t.schema)}
        for t in request.tools
    ]


def anthropic_base_url(base_url: str) -> str:
    """The SDK appends `/v1/messages` itself. A rung written the OpenAI way, ending in `/v1`,
    would otherwise post to `/v1/v1/messages` and read as a 404 from a working endpoint."""
    base = base_url.rstrip("/")
    return base.removesuffix("/v1")


def thinking(reasoning: ReasoningSpec) -> dict[str, Any]:
    """D46's spec in this API's two parameters. A stated budget is the older fixed-budget
    thinking; a level alone is adaptive thinking at that effort."""
    if reasoning.budget_tokens is not None:
        return {"thinking": {"type": "enabled", "budget_tokens": reasoning.budget_tokens}}
    return {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": _EFFORT[reasoning.level]},
    }


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def probe_anthropic_usage(usage: dict[str, Any]) -> Usage | None:
    """`input_tokens` excludes cache reads and writes on this API; `prompt_tokens` is the
    whole prompt, as on every other transport, so the budget subtracts reads and writes
    from one total (`budget/prices.py`). An absent counter is `None`, never 0; with neither
    `input_tokens` nor `output_tokens` there is nothing to read, and the caller estimates."""
    plain = _count(usage.get("input_tokens"))
    out = _count(usage.get("output_tokens"))
    if plain is None and out is None:
        return None
    read = _count(usage.get("cache_read_input_tokens"))
    write = _count(usage.get("cache_creation_input_tokens"))
    return Usage(
        prompt_tokens=(plain or 0) + (read or 0) + (write or 0),
        completion_tokens=out or 0,
        cached_prompt_tokens=read,
        cache_write_tokens=write,
        estimated=False,
    )


_DELTA_TEXT: dict[str, str] = {
    "text_delta": "text",
    "input_json_delta": "partial_json",
    "thinking_delta": "thinking",
}
"""Each streamed delta's kind and the field holding what it streamed: all of it output."""


def _pieces(event: object) -> tuple[Piece, ...]:
    """What one of the message stream helper's events streamed (`client/live.py`): a
    `content_block_delta`'s text, thinking or tool input, and a tool block's name from its
    `content_block_start`. Its derived events (`text`, `input_json`) repeat a delta's, and say
    nothing."""
    kind = getattr(event, "type", None)
    index = getattr(event, "index", 0)
    at = index if isinstance(index, int) else 0
    if kind == "content_block_start":
        block: Any = getattr(event, "content_block", None)
        name = getattr(block, "name", None)
        if getattr(block, "type", None) == "tool_use" and isinstance(name, str):
            return (ToolArgs(at, "", name),)
        return ()
    if kind != "content_block_delta":
        return ()
    delta: Any = getattr(event, "delta", None)
    what = str(getattr(delta, "type", ""))
    field = _DELTA_TEXT.get(what)
    said = getattr(delta, field, None) if field is not None else None
    if not isinstance(said, str) or not said:
        return ()
    if what == "text_delta":
        return (Text(said),)
    if what == "thinking_delta":
        return (Thought(said),)
    return (ToolArgs(at, said),)


def _snapshot(stream: object) -> dict[str, Any] | None:
    """What the message stream helper accumulated, as the provider sent it (`dump`)."""
    return dump(getattr(stream, "current_message_snapshot", None))


class AnthropicClient:
    """One rung's transport over the official `anthropic` package (D82's `explicit` mode).
    Satisfies `ModelClient` (`client/request.py`) through `send` and `close`."""

    def __init__(
        self,
        endpoint: EndpointSpec,
        affinity: AffinityTable | None = None,
        transport: httpx2.BaseTransport | None = None,
        *,
        retry: RetryPolicy = RETRY_POLICY,
    ) -> None:
        self.endpoint = endpoint
        self.affinity = affinity or AffinityTable()
        self._retry = retry
        self._silences = Silences(endpoint.first_token_s, endpoint.token_gap_s)
        self._forcing = Forcing(endpoint)
        # `transport` is the test seam: a mock server with real HTTP semantics. Made anew after
        # the machine sleeps, so no request waits on a connection the sleep left dead.
        self._http = ModelConnections(
            lambda: httpx2.Client(
                timeout=httpx2.Timeout(self._silences.backstop_s), transport=transport
            )
        )

    def close(self) -> None:
        self._http.close()

    def _sdk(self) -> anthropic.Anthropic:
        key = api_key(self.endpoint)
        if key is None:
            raise CredentialError(
                "an Anthropic-native rung needs api_key_env; the API accepts no request "
                "without a key"
            )
        return anthropic.Anthropic(
            api_key=key,
            base_url=anthropic_base_url(self.endpoint.base_url),
            timeout=self._silences.backstop_s,
            max_retries=0,
            http_client=self._http.get(),
        )

    def _build(self, request: ChatRequest, *, forced: bool = False) -> dict[str, Any]:
        _check_breakpoints(request)
        refuse_images(self.endpoint, request)
        params = request.params
        kwargs: dict[str, Any] = {
            "model": self.endpoint.model_id,
            "messages": anthropic_messages(request),
            "max_tokens": params.max_tokens,
        }
        system = anthropic_system(request)
        if system:
            kwargs["system"] = system
        if params.stop:
            kwargs["stop_sequences"] = list(params.stop)
        if request.tools and bool(getattr(self.endpoint, "native_tools", True)):
            kwargs["tools"] = anthropic_tools(request)
            if forced:
                kwargs["tool_choice"] = {"type": "any"}
        if params.reasoning is not None:
            kwargs.update(thinking(params.reasoning))
        extra_body: dict[str, Any] = {}
        # This SDK no longer types the parameter; the API decides per model, and the rung's
        # `supports_temperature` is where a person says it refuses one. Thinking
        # refuses one outright, so a reasoning rung samples at the provider's default —
        # D5's ladder not applied, as on a rung that declares it.
        if params.reasoning is None and bool(
            getattr(self.endpoint, "supports_temperature", True)
        ):
            extra_body["temperature"] = params.temperature
        headers: dict[str, str] = {}
        self.affinity.row(self.endpoint.provider).inject(
            extra_body, headers, request.session_key
        )
        extra_headers: dict[str, Any] = dict(headers)
        ambient = ambient_header_names(ANTHROPIC_HEADERS_ENV)
        refuse_ambient_credential(ANTHROPIC_HEADERS_ENV, ambient, "X-Api-Key", "Authorization")
        for name in ambient:
            extra_headers.setdefault(name, anthropic.Omit())
        kwargs["extra_body"] = extra_body or None
        kwargs["extra_headers"] = extra_headers or None
        return kwargs

    def send(self, request: ChatRequest) -> Completion:
        # Anthropic refuses a forced tool call with thinking on.
        blocked = "thinking" if request.params.reasoning is not None else None
        return self._forcing.send(
            request, lambda forced: self._send(request, forced), blocked=blocked
        )

    def _send(self, request: ChatRequest, forced: bool) -> Completion:
        sdk = self._sdk()
        kwargs = self._build(request, forced=forced)
        started = time.monotonic()
        url = anthropic_base_url(self.endpoint.base_url) + "/v1/messages"
        # The SDK's overloads type each keyword; the request is built as one dict so the
        # byte test can see exactly what was handed over.
        stream = cast(Callable[..., Any], sdk.messages.stream)

        def attempt(n: int) -> Completion:
            try:
                read = read_stream(
                    lambda: stream(**kwargs),
                    pieces=_pieces,
                    final=_snapshot,
                    silences=self._silences,
                    cancel=request.cancel,
                    live=request.live,
                )
            except StreamEnded as ended:
                raise failed_attempt(ended, n, ANTHROPIC_ERRORS, url) from ended
            parsed = self._parse(read.final, request, started)
            if parsed is None:
                raise Retry(
                    "malformed JSON or missing content",
                    "malformed",
                    billed=read.streamed_tokens,
                )
            return parsed

        return run_ladder(request, attempt, self._retry)

    def _parse(
        self, payload: object, request: ChatRequest, started: float
    ) -> Completion | None:
        """Build a Completion from the stream's accumulated message (`_snapshot`), or None to
        signal one more internal retry: a stream that carried no event, or no content."""
        if not isinstance(payload, dict):
            return None
        payload = cast("dict[str, Any]", payload)
        content = payload.get("content")
        if not isinstance(content, list):
            return None
        texts: list[str] = []
        calls: list[ToolCall] = []
        # Thinking blocks are skipped: they are not the answer, and nothing in `Message`
        # carries them back yet.
        for block in cast(list[Any], content):
            if not isinstance(block, dict):
                continue
            b = cast(dict[str, Any], block)
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                texts.append(b["text"])
            elif b.get("type") == "tool_use":
                raw_input = b.get("input")
                calls.append(
                    ToolCall(
                        id=str(b.get("id") or ""),
                        name=str(b.get("name") or ""),
                        arguments=cast(dict[str, object], raw_input)
                        if isinstance(raw_input, dict)
                        else {},
                    )
                )
        text = "".join(texts)
        usage_obj = payload.get("usage")
        usage = (
            probe_anthropic_usage(cast(dict[str, Any], usage_obj))
            if isinstance(usage_obj, dict)
            else None
        )
        if usage is None:
            # The OpenAI transport's estimator, for the same reason: visibly approximate
            # rather than silently precise.
            usage, _ = probe_usage({}, prompt_text(request), text)
        raw_stop = payload.get("stop_reason")
        return completion(
            text=text,
            tool_calls=tuple(calls),
            raw_stop=raw_stop if isinstance(raw_stop, str) else None,
            usage=usage,
            started=started,
        )
