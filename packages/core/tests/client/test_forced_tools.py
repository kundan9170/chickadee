"""The worker's forced tool call, as each client sends it, and its fallback.

A request asks (`ChatRequest.force_tool`); the client forces it only when the request offers
tools, the endpoint declares `forced_tools`, and the provider has not refused it: OpenAI's
`tool_choice: "required"`, Anthropic's `{"type": "any"}`, never with Anthropic's thinking on.
A provider's 400 naming `tool_choice` sends the request again unforced, and every later one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import httpx2
import pytest
from client_helpers import FakeEndpoint, anthropic_body, ok_body, scripted

from chickadee.client import AnthropicClient, OpenAIClient, RetryPolicy
from chickadee.client.request import ChatRequest, Message, ToolSchema
from chickadee.schema import ReasoningSpec, RetryParams

KEY = "CHICKADEE_TEST_ANTHROPIC_KEY"
_NO_WAITS = RetryPolicy(random=lambda: 0.0, sleep=lambda _delay, _cancel: None)
_TOOL = ToolSchema(name="read", description="r", schema={"type": "object"})
_FORCED = ChatRequest(
    messages=(Message(role="user", content="read x"),),
    params=RetryParams(temperature=0.2, max_tokens=100),
    tools=(_TOOL,),
    force_tool=True,
)


@dataclass(frozen=True)
class ForcingEndpoint(FakeEndpoint):
    forced_tools: bool = True


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "sk-ant-test")


def _openai(endpoint: FakeEndpoint, *responses: httpx2.Response) -> tuple[OpenAIClient, list]:
    transport, seen = scripted(*responses)
    return OpenAIClient(endpoint, transport=transport, retry=_NO_WAITS), seen


def _anthropic(*responses: httpx2.Response) -> tuple[AnthropicClient, list]:
    transport, seen = scripted(*responses)
    endpoint = ForcingEndpoint(api_key_env=KEY, provider="anthropic")
    return AnthropicClient(endpoint, transport=transport, retry=_NO_WAITS), seen


def _choice(request: httpx2.Request) -> object:
    return json.loads(request.content).get("tool_choice")


_OK = httpx2.Response(200, json=ok_body())


def test_openai_forces_with_required() -> None:
    client, seen = _openai(ForcingEndpoint(), _OK)
    client.send(_FORCED)
    assert _choice(seen[0]) == "required"


@pytest.mark.parametrize(
    "request_",
    [replace(_FORCED, force_tool=False), replace(_FORCED, tools=())],
    ids=["not asked", "no tools"],
)
def test_nothing_is_forced_unless_asked_with_tools(request_: ChatRequest) -> None:
    client, seen = _openai(ForcingEndpoint(), _OK)
    client.send(request_)
    assert _choice(seen[0]) is None


def test_a_model_without_the_capability_goes_unforced_and_says_so_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, seen = _openai(FakeEndpoint(), _OK)
    with caplog.at_level("WARNING", logger="chickadee.client.openai_client"):
        client.send(_FORCED)
        client.send(_FORCED)
    assert [_choice(r) for r in seen] == [None, None]
    assert [r.getMessage() for r in caplog.records] == [
        "qwen3-coder-30b-a3b: a forced tool call is sent unforced (D220): no_capability"
    ]


def test_a_refused_tool_choice_is_sent_again_unforced_and_never_forced_again() -> None:
    refused = httpx2.Response(
        400, json={"error": {"message": "tool_choice 'required' is not supported"}}
    )
    client, seen = _openai(ForcingEndpoint(), refused, _OK)
    assert client.send(_FORCED).text == "done"
    client.send(_FORCED)
    assert [_choice(r) for r in seen] == ["required", None, None]


def test_another_400_is_not_a_refused_tool_choice() -> None:
    from chickadee.client import TransportExhausted

    client, seen = _openai(
        ForcingEndpoint(), httpx2.Response(400, json={"error": {"message": "bad"}})
    )
    with pytest.raises(TransportExhausted, match="HTTP 400"):
        client.send(_FORCED)
    assert len(seen) == 1


def test_anthropic_forces_with_any() -> None:
    client, seen = _anthropic(httpx2.Response(200, json=anthropic_body()))
    client.send(_FORCED)
    assert _choice(seen[0]) == {"type": "any"}


def test_anthropic_with_thinking_goes_unforced() -> None:
    client, seen = _anthropic(httpx2.Response(200, json=anthropic_body()))
    params = _FORCED.params.model_copy(update={"reasoning": ReasoningSpec(level="medium")})
    client.send(replace(_FORCED, params=params))
    body = json.loads(seen[0].content)
    assert "thinking" in body and "tool_choice" not in body
