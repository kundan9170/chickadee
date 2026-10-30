"""A reply's pieces as it streams: each transport hands `live` the
text, the reasoning and each tool call's growing arguments, in order, on the caller's thread;
the `Completion` is the one it returns without `live`; a retry is announced; and a live view
that fails costs the view, never the reply. `LiveReply` folds the pieces into batched
updates."""

from __future__ import annotations

import json
import threading

import httpx2
import pytest
from client_helpers import (
    FakeEndpoint,
    anthropic_body,
    anthropic_events,
    ok_body,
    openai_events,
    scripted,
    sse,
)

from chickadee.client import AnthropicClient, OpenAIClient, RetryPolicy, to_request
from chickadee.client.live import (
    FLUSH_S,
    LiveReply,
    NewAttempt,
    Piece,
    Restarted,
    Text,
    TextUpdate,
    Thought,
    ThoughtUpdate,
    ToolArgs,
    ToolUpdate,
    Update,
    streamed_chars,
)
from chickadee.client.request import ChatRequest
from chickadee.schema import Prompt, RetryParams

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)
KEY = "CHICKADEE_TEST_ANTHROPIC_KEY"
CREATE = {"path": "core/ai.py", "content": "print('hi')\n"}


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "sk-ant-test")


def _no_waits() -> RetryPolicy:
    return RetryPolicy(random=lambda: 0.0, sleep=lambda _delay, _cancel: None)


def _request(live: list[Piece] | None = None) -> ChatRequest:
    req = to_request(PROMPT, PARAMS)
    return ChatRequest(
        messages=req.messages, params=req.params, live=live.append if live is not None else None
    )


def _openai_reply() -> httpx2.Response:
    """Reasoning, then text in two chunks, then a tool call whose arguments arrive in two."""
    body = ok_body(text="")
    events = openai_events(body)
    head = {k: v for k, v in events[0].items() if k != "choices"}

    def chunk(delta: dict[str, object]) -> dict[str, object]:
        return {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}

    args = json.dumps(CREATE)
    call = {"index": 0, "id": "c1", "type": "function"}
    return sse(
        [
            events[0],
            chunk({"reasoning_content": "let me see"}),
            chunk({"content": "Writing "}),
            chunk({"content": "it now."}),
            chunk({"tool_calls": [{**call, "function": {"name": "create", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[:20]}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[20:]}}]}),
            *events[1:],
        ]
    )


def test_openai_hands_live_each_piece_in_order() -> None:
    pieces: list[Piece] = []
    transport, _ = scripted(_openai_reply())
    done = OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits()).send(
        _request(pieces)
    )
    args = json.dumps(CREATE)
    assert pieces == [
        Thought("let me see"),
        Text("Writing "),
        Text("it now."),
        ToolArgs(0, "", "create"),
        ToolArgs(0, args[:20]),
        ToolArgs(0, args[20:]),
    ]
    assert done.text == "Writing it now."
    assert streamed_chars(pieces) == len("let me see") + len(done.text) + len(args)


def test_the_completion_is_the_same_with_live_or_without() -> None:
    transport, _ = scripted(_openai_reply(), _openai_reply())
    client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits())
    without = client.send(_request())
    with_live = client.send(_request([]))
    assert (with_live.text, with_live.tool_calls, with_live.stop_reason, with_live.usage) == (
        without.text,
        without.tool_calls,
        without.stop_reason,
        without.usage,
    )


def test_anthropic_hands_live_text_and_a_tool_calls_name_and_input() -> None:
    pieces: list[Piece] = []
    body = anthropic_body(
        text="On it.", tool_uses=[{"id": "t1", "name": "create", "input": CREATE}]
    )
    transport, _ = scripted(sse(anthropic_events(body), named=True))
    AnthropicClient(FakeEndpoint(api_key_env=KEY), transport=transport, retry=_no_waits()).send(
        _request(pieces)
    )
    assert pieces == [
        Text("On it."),
        ToolArgs(1, "", "create"),
        ToolArgs(1, json.dumps(CREATE)),
    ]


def test_a_retry_is_announced_before_its_pieces() -> None:
    pieces: list[Piece] = []
    transport, _ = scripted(
        httpx2.Response(503), httpx2.Response(200, json=ok_body(text="second"))
    )
    OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits()).send(_request(pieces))
    assert pieces == [NewAttempt(2), Text("second")]


def test_a_live_view_that_fails_costs_the_view_never_the_reply() -> None:
    calls: list[Piece] = []

    def broken(piece: Piece) -> None:
        calls.append(piece)
        raise RuntimeError("the view broke")

    req = to_request(PROMPT, PARAMS)
    transport, _ = scripted(_openai_reply())
    done = OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits()).send(
        ChatRequest(messages=req.messages, params=req.params, live=broken)
    )
    assert done.text == "Writing it now."
    assert len(calls) == 1, "not called again once it raised"


def test_live_is_called_on_the_callers_thread() -> None:
    threads: set[int] = set()
    req = to_request(PROMPT, PARAMS)
    transport, _ = scripted(_openai_reply())
    OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits()).send(
        ChatRequest(
            messages=req.messages,
            params=req.params,
            live=lambda _piece: threads.add(threading.get_ident()),
        )
    )
    assert threads == {threading.get_ident()}


# -- the fold ---------------------------------------------------------------------------------


class _LiveClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_text_is_batched_and_each_update_says_its_offset() -> None:
    clock = _LiveClock()
    out: list[Update] = []
    reply = LiveReply(out.append, clock=clock)
    clock.now = 1.0
    reply.feed(Text("Hel"))  # the first flush is due at once
    reply.feed(Text("lo "))
    clock.now += FLUSH_S / 2
    reply.feed(Text("there"))
    assert out == [TextUpdate(0, "Hel")], "held until the next flush"
    clock.now += FLUSH_S
    reply.feed(Text("!"))
    reply.close()
    assert out == [TextUpdate(0, "Hel"), TextUpdate(3, "lo there!")]


def test_thinking_says_how_long() -> None:
    clock = _LiveClock()
    out: list[Update] = []
    reply = LiveReply(out.append, clock=clock)
    clock.now = 5.0
    reply.feed(Thought("hmm"))
    clock.now = 7.0
    reply.feed(Thought(", yes"))
    assert out == [ThoughtUpdate(0.0, 0, "hmm"), ThoughtUpdate(2.0, 3, ", yes")]


def test_a_tool_call_shows_its_tool_path_and_size_never_its_arguments() -> None:
    clock = _LiveClock()
    out: list[Update] = []
    reply = LiveReply(out.append, clock=clock)
    args = json.dumps(CREATE)
    reply.feed(ToolArgs(0, "", "create"))
    clock.now = 1.0
    reply.feed(ToolArgs(0, args[:6]))
    clock.now = 2.0
    reply.feed(ToolArgs(0, args[6:]))
    assert out == [
        ToolUpdate(0, "create", None, 0),
        ToolUpdate(0, "create", None, 6),
        ToolUpdate(0, "create", "core/ai.py", len(args)),
    ]


def test_a_new_attempt_restarts_what_was_shown() -> None:
    out: list[Update] = []
    reply = LiveReply(out.append)
    reply.feed(Text("first try"))
    reply.feed(NewAttempt(2))
    reply.feed(Text("second"))
    reply.close()
    assert out == [TextUpdate(0, "first try"), Restarted(), TextUpdate(0, "second")]
