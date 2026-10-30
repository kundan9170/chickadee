"""A stream the OpenAI SDK cannot assemble is a malformed reply, never the harness's failure.

Some OpenAI-compatible servers send a tool call's later pieces with `"type": null` or
`"function": null`, or number its calls from 1. The SDK's stream helper then raises from its
own accumulation: an `AssertionError` with no message, an `IndexError`, an `AttributeError`.
Was: `failed_attempt` raised each as it is, so no retry ran, and the worker's round ended as
`harness_error` with the report `the harness failed: ` (2026-10-22). Now each is retried as the
`malformed` kind, and its last failure is a `TransportExhausted` naming what the SDK raised.
"""

from __future__ import annotations

import httpx2
import pytest
from client_helpers import FakeEndpoint, ok_body, scripted, sse

from chickadee.client import OpenAIClient, RetryPolicy, TransportExhausted
from chickadee.client.openai_client import OPENAI_ERRORS, Retry, failed_attempt
from chickadee.client.request import ChatRequest, Message, ToolSchema
from chickadee.client.stream import Broke
from chickadee.errors import HarnessBug
from chickadee.schema import RetryParams

_HEAD: dict[str, object] = {
    "id": "c1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "m",
}
_REQUEST = ChatRequest(
    messages=(Message(role="user", content="read x"),),
    params=RetryParams(temperature=0.2, max_tokens=100),
    tools=(ToolSchema(name="read", description="r", schema={"type": "object"}),),
)
_NO_WAITS = RetryPolicy(random=lambda: 0.0, sleep=lambda _delay, _cancel: None)


def _chunk(delta: dict[str, object], finish: str | None = None) -> dict[str, object]:
    return {**_HEAD, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _call(**fields: object) -> dict[str, object]:
    return _chunk({"tool_calls": [fields]})


_READ = {"name": "read", "arguments": '{"path": "x"}'}

UNREADABLE = {
    "a later piece with type null": [
        _call(index=0, id="a", type="function", function={"name": "read", "arguments": ""}),
        _call(index=0, id=None, type=None, function={"arguments": '{"path": "x"}'}),
    ],
    "a later piece with function null": [
        _call(index=0, id="a", type="function", function=_READ),
        _call(index=0, function=None),
    ],
    "the first call numbered 1": [_call(index=1, id="a", type="function", function=_READ)],
    "id and type before the function": [
        _call(index=0, id="a", type="function"),
        _call(index=0, function=_READ),
    ],
}


def _unreadable(pieces: list[dict[str, object]]) -> list[dict[str, object]]:
    return [_chunk({"role": "assistant"}), *pieces, _chunk({}, "tool_calls")]


@pytest.mark.parametrize("name", list(UNREADABLE))
def test_an_unreadable_stream_is_retried_and_then_a_malformed_reply(name: str) -> None:
    transport, seen = scripted(sse(_unreadable(UNREADABLE[name])))
    client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_NO_WAITS)
    with pytest.raises(TransportExhausted, match=r"malformed stream: [A-Za-z]+Error") as err:
        client.send(_REQUEST)
    assert len(seen) == err.value.attempts == 4


def test_a_good_reply_after_an_unreadable_one_is_the_answer() -> None:
    bad = sse(_unreadable(UNREADABLE["a later piece with type null"]))
    good = ok_body(text="", tool_calls=[{"id": "a", "type": "function", "function": _READ}])
    transport, seen = scripted(bad, httpx2.Response(200, json=good))
    client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_NO_WAITS)
    done = client.send(_REQUEST)
    assert [(c.name, c.arguments) for c in done.tool_calls] == [("read", {"path": "x"})]
    assert len(seen) == 2


def test_an_error_before_the_stream_opened_is_raised_as_it_is() -> None:
    """Not opened, the provider never answered: the request's own error, not a reply's."""
    cause = TypeError("bad keyword")
    ended = Broke(cause, opened=False, started=False, chars=0)
    assert failed_attempt(ended, 1, OPENAI_ERRORS, "u") is cause


def test_a_harness_bug_is_never_a_malformed_reply() -> None:
    cause = HarnessBug("ours")
    ended = Broke(cause, opened=True, started=True, chars=3)
    assert failed_attempt(ended, 1, OPENAI_ERRORS, "u") is cause


def test_an_unreadable_stream_is_billed_with_what_it_streamed() -> None:
    ended = Broke(AssertionError(), opened=True, started=True, chars=70)
    retry = failed_attempt(ended, 1, OPENAI_ERRORS, "u")
    assert isinstance(retry, Retry)
    assert (retry.kind, retry.billed, str(retry)) == (
        "malformed",
        20,
        "malformed stream: AssertionError: ",
    )
