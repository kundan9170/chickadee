"""AD-13: a provider's "the request is too long" is `ContextOverflow`, its own failure, through
both transports. It is not retried (a 400 never is), not billed, and an unrecognised 400
is not guessed to be one."""

from __future__ import annotations

import httpx2
import pytest
from client_helpers import FakeEndpoint, scripted

from chickadee.client import (
    AnthropicClient,
    ClientError,
    ContextOverflow,
    OpenAIClient,
    TransportExhausted,
    to_request,
)
from chickadee.schema import Prompt, RetryParams

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)
KEY = "CHICKADEE_TEST_ANTHROPIC_KEY"

REFUSALS = {
    "openai": httpx2.Response(
        400,
        json={
            "error": {
                "message": "This model's maximum context length is 8192 tokens.",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        },
    ),
    "vllm": httpx2.Response(
        400,
        json={
            "object": "error",
            "message": "This model's maximum context length is 32768 tokens. However, you "
            "requested 40000 tokens in the messages.",
        },
    ),
    "anthropic": httpx2.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 210000 tokens > 200000 maximum",
            },
        },
    ),
    "llama.cpp": httpx2.Response(
        400,
        json={
            "error": {"code": 400, "message": "the request exceeds the available context size"}
        },
    ),
    "413": httpx2.Response(413, text="Request Entity Too Large"),
}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "sk-ant-test")
    monkeypatch.setattr("chickadee.client.openai_client.time.sleep", lambda _s: None)


def _send(transport_kind: str, response: httpx2.Response) -> tuple[ClientError, int]:
    transport, seen = scripted(response)
    if transport_kind == "openai":
        client = OpenAIClient(FakeEndpoint(), transport=transport)
    else:
        client = AnthropicClient(FakeEndpoint(api_key_env=KEY), transport=transport)
    with pytest.raises(ClientError) as err:
        client.send(to_request(PROMPT, PARAMS))
    return err.value, len(seen)


@pytest.mark.parametrize("transport_kind", ["openai", "anthropic"])
@pytest.mark.parametrize("provider", sorted(REFUSALS))
def test_each_providers_refusal_is_a_context_overflow(
    transport_kind: str, provider: str
) -> None:
    err, sent = _send(transport_kind, REFUSALS[provider])
    assert isinstance(err, ContextOverflow), err
    assert err.billed_attempts == (), "a refused request is not billed"
    assert sent == 1, "the same request is never sent again at the same size"
    assert err.status == REFUSALS[provider].status_code


def test_the_overflow_carries_the_providers_words() -> None:
    err, _ = _send("openai", REFUSALS["openai"])
    assert isinstance(err, ContextOverflow)
    assert "context_length_exceeded" in err.detail
    assert "context_length_exceeded" in str(err)


@pytest.mark.parametrize("transport_kind", ["openai", "anthropic"])
def test_an_unrecognised_400_is_not_guessed_to_be_an_overflow(transport_kind: str) -> None:
    err, sent = _send(transport_kind, httpx2.Response(400, json={"error": {"message": "bad"}}))
    assert type(err) is TransportExhausted
    assert sent == 1


def test_a_marker_on_a_status_that_is_not_a_refusal_is_not_an_overflow() -> None:
    """A 422 whose body happens to quote the phrase is the request's other fault."""
    body = {"error": {"message": "field 'context_length_exceeded' is not allowed"}}
    err, _ = _send("openai", httpx2.Response(422, json=body))
    assert type(err) is TransportExhausted
