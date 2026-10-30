"""Plain helpers for the client suite.

Deliberately **not** in `conftest.py`. Every suite has a module by that name,
Python caches modules by name, and pytest puts each test directory on `sys.path` —
so `from conftest import...` resolves to whichever loaded first and naming two
suites in one invocation collected the wrong one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import httpx2

Json = dict[str, object]

_DEFAULT_USAGE: Json = {"prompt_tokens": 100, "completion_tokens": 20}


@dataclass(frozen=True)
class FakeEndpoint:
    base_url: str = "https://api.example.test/v1"
    model_id: str = "qwen3-coder-30b-a3b"
    api_key_env: str | None = None
    provider: str = "local"
    first_token_s: float = 30.0
    token_gap_s: float = 30.0
    native_tools: bool = True
    supports_temperature: bool = True
    supports_images: bool = True
    supports_documents: bool = False


def scripted(
    *responses: httpx2.Response | Exception,
) -> tuple[httpx2.MockTransport, list[httpx2.Request]]:
    """Serve the given responses in order; the last repeats if exhausted. Both SDKs run
    on `httpx2`, so one mock serves either transport.

    Every request streams, so a 200 whose body is a whole completion
    (`ok_body`, `anthropic_body`) is served as the event stream its SDK reads (`streamed`);
    any other response, a body that is not a completion included, is served as it is."""
    seen: list[httpx2.Request] = []
    queue = list(responses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return streamed(item)

    return httpx2.MockTransport(handler), seen


def streamed(response: httpx2.Response) -> httpx2.Response:
    """`response` as a stream, when it is a 200 carrying a whole completion: OpenAI's
    `chat.completion.chunk`s or Anthropic's message events, each holding what the whole body
    held, so the SDK's accumulated reply is the body. Anything else, unchanged."""
    if response.status_code != 200:
        return response
    try:
        body: object = json.loads(response.content)
    except ValueError:
        return response
    if not isinstance(body, dict):
        return response
    body = cast(Json, body)
    if isinstance(body.get("choices"), list) and body["choices"]:
        return sse(openai_events(body))
    if body.get("type") == "message" and isinstance(body.get("content"), list):
        return sse(anthropic_events(body), named=True)
    return response


def sse(events: list[Json], *, named: bool = False) -> httpx2.Response:
    """A 200 `text/event-stream` of `events`; `named` puts each event's `type` on an `event:`
    line, as Anthropic's stream does. OpenAI's ends with `[DONE]`."""
    lines = [
        (f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n"
        for e in events
    ]
    if not named:
        lines.append("data: [DONE]\n\n")
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content="".join(lines).encode()
    )


def openai_events(body: Json) -> list[Json]:
    """A chat completion body as the chunks a streaming server sends: the message's content,
    each tool call in its own chunk, the finish reason, then the usage alone."""
    head: Json = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    choice = cast(Json, cast(list[object], body["choices"])[0])
    message = cast(Json, choice.get("message") or {})
    chunks: list[Json] = [
        {
            **head,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": message.get("content")},
                    "finish_reason": None,
                }
            ],
        }
    ]
    for i, call in enumerate(cast(list[Json], message.get("tool_calls") or [])):
        chunks.append(
            {
                **head,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": i, **call}]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    finish = {k: v for k, v in choice.items() if k in ("finish_reason", "stop_reason")}
    chunks.append({**head, "choices": [{"index": 0, "delta": {}, **finish}]})
    if "usage" in body:
        chunks.append({**head, "choices": [], "usage": body["usage"]})
    return chunks


def anthropic_events(body: Json) -> list[Json]:
    """A message body as the events Anthropic streams: the message with its usage and no
    content, each block started, its whole delta, and stopped, then the stop reason and the
    output count."""
    usage = cast(Json, body.get("usage") or {})
    start = {**body, "content": [], "stop_reason": None, "stop_sequence": None}
    events: list[Json] = [{"type": "message_start", "message": start}]
    for i, block in enumerate(cast(list[Json], body["content"])):
        events.extend(_block_events(i, block))
    delta_usage: Json = {"output_tokens": usage.get("output_tokens")}
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": body.get("stop_reason"), "stop_sequence": None},
            "usage": delta_usage,
        }
    )
    events.append({"type": "message_stop"})
    return events


def _block_events(index: int, block: Json) -> Iterator[Json]:
    if block.get("type") == "tool_use":
        yield {
            "type": "content_block_start",
            "index": index,
            "content_block": {**block, "input": {}},
        }
        delta: Json = {
            "type": "input_json_delta",
            "partial_json": json.dumps(block.get("input")),
        }
    else:
        yield {
            "type": "content_block_start",
            "index": index,
            "content_block": {**block, "text": ""},
        }
        delta = {"type": "text_delta", "text": block.get("text", "")}
    yield {"type": "content_block_delta", "index": index, "delta": delta}
    yield {"type": "content_block_stop", "index": index}


def ok_body(
    text: str = "done",
    finish_reason: str = "stop",
    usage: Json | None = _DEFAULT_USAGE,
    tool_calls: list[Json] | None = None,
) -> Json:
    message: Json = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body: Json = {"choices": [{"message": message, "finish_reason": finish_reason}]}
    if usage is not None:
        # Copied, not aliased. `_DEFAULT_USAGE` is one object shared by every call, so a
        # test mutating `body["usage"]` would poison every later `ok_body()`.
        body["usage"] = dict(usage)
    return body


def anthropic_body(
    text: str = "done",
    stop_reason: str = "end_turn",
    usage: Json | None = None,
    tool_uses: list[Json] | None = None,
) -> Json:
    content: list[Json] = [{"type": "text", "text": text}] if text else []
    content.extend({"type": "tool_use", **t} for t in tool_uses or [])
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop_reason,
        "usage": dict(usage)
        if usage is not None
        else {"input_tokens": 100, "output_tokens": 20},
    }
