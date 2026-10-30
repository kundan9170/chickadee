"""`AnthropicClient` against a mock server: the behaviour `test_client.py` asserts of the
OpenAI transport — usage, the cached-token probe, retry on 5xx/429/529, 401 as a
`CredentialError`, the deadline — plus the one field only this transport fills,
`Usage.cache_write_tokens` from `cache_creation_input_tokens`."""

from __future__ import annotations

import threading

import httpx2
import pytest
from client_helpers import FakeEndpoint, anthropic_body, scripted

from chickadee.client import (
    AnthropicClient,
    CredentialError,
    RequestCancelled,
    TransportExhausted,
    to_request,
)
from chickadee.schema import Prompt, ReasoningSpec, RetryParams, StopReason

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)
KEY = "CHICKADEE_TEST_ANTHROPIC_KEY"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "sk-ant-test")
    monkeypatch.setattr("chickadee.client.openai_client.time.sleep", lambda _s: None)


def _client(
    *responses: httpx2.Response | Exception,
) -> tuple[AnthropicClient, list[httpx2.Request]]:
    transport, seen = scripted(*responses)
    return AnthropicClient(FakeEndpoint(api_key_env=KEY), transport=transport), seen


def test_usage_and_cache_counters() -> None:
    body = anthropic_body(
        usage={
            "input_tokens": 50,
            "output_tokens": 7,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 300,
        }
    )
    client, seen = _client(httpx2.Response(200, json=body))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.text == "done"
    assert c.stop_reason is StopReason.END_TURN and c.raw_stop_reason == "end_turn"
    # prompt_tokens is the whole prompt, as on every transport; reads and writes are parts.
    assert c.usage.prompt_tokens == 1250
    assert c.usage.cached_prompt_tokens == 900
    assert c.usage.cache_write_tokens == 300
    assert c.usage.completion_tokens == 7
    assert c.cached_prefix is True
    assert seen[0].headers["x-api-key"] == "sk-ant-test"


def test_absent_cache_counters_are_none_not_zero() -> None:
    client, _ = _client(httpx2.Response(200, json=anthropic_body()))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.cached_prompt_tokens is None
    assert c.usage.cache_write_tokens is None
    assert c.cached_prefix is None


def test_tool_use_is_a_tool_call() -> None:
    body = anthropic_body(
        text="",
        stop_reason="tool_use",
        tool_uses=[{"id": "tu_1", "name": "read", "input": {"path": "a.py"}}],
    )
    client, _ = _client(httpx2.Response(200, json=body))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.stop_reason is StopReason.TOOL_USE
    assert [(t.id, t.name, t.arguments) for t in c.tool_calls] == [
        ("tu_1", "read", {"path": "a.py"})
    ]


def test_max_tokens_is_truncation() -> None:
    client, _ = _client(httpx2.Response(200, json=anthropic_body(stop_reason="max_tokens")))
    assert client.send(to_request(PROMPT, PARAMS)).stop_reason is StopReason.MAX_TOKENS


@pytest.mark.parametrize("status", [429, 500, 503, 529])
def test_internal_retry_on_transient(status: int) -> None:
    client, seen = _client(httpx2.Response(status), httpx2.Response(200, json=anthropic_body()))
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"
    assert len(seen) == 2, "the SDK's own retries are off; AD-5's ladder is the only one"


def test_retries_are_bounded() -> None:
    client, seen = _client(httpx2.Response(503))
    with pytest.raises(TransportExhausted) as exc:
        client.send(to_request(PROMPT, PARAMS))
    assert len(seen) == 4 and exc.value.attempts == 4


@pytest.mark.parametrize("status", [401, 403])
def test_credentials_abort_rather_than_retry(status: int) -> None:
    client, seen = _client(httpx2.Response(status))
    with pytest.raises(CredentialError):
        client.send(to_request(PROMPT, PARAMS))
    assert len(seen) == 1


def test_deadline_is_transport_exhausted() -> None:
    client, _ = _client(httpx2.ReadTimeout("too slow"))
    with pytest.raises(TransportExhausted, match="deadline"):
        client.send(to_request(PROMPT, PARAMS))


def test_a_rung_without_a_key_is_refused_before_any_request() -> None:
    transport, seen = scripted(httpx2.Response(200, json=anthropic_body()))
    client = AnthropicClient(FakeEndpoint(), transport=transport)
    with pytest.raises(CredentialError, match="api_key_env"):
        client.send(to_request(PROMPT, PARAMS))
    assert seen == []


def test_a_cancelled_request_is_not_sent() -> None:
    client, seen = _client(httpx2.Response(200, json=anthropic_body()))
    cancel = threading.Event()
    cancel.set()
    req = to_request(PROMPT, PARAMS)
    with pytest.raises(RequestCancelled):
        client.send(type(req)(messages=req.messages, params=req.params, cancel=cancel))
    assert seen == []


def test_an_empty_usage_object_is_estimated_not_zero() -> None:
    client, _ = _client(httpx2.Response(200, json=anthropic_body(usage={})))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.estimated is True
    assert c.usage.prompt_tokens > 0
    assert c.usage.cache_write_tokens is None


def test_a_body_that_says_json_and_is_not_retries() -> None:
    truncated = httpx2.Response(
        200, content=b'{"id"', headers={"content-type": "application/json"}
    )
    client, seen = _client(truncated, httpx2.Response(200, json=anthropic_body()))
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"
    assert len(seen) == 2


def test_a_window_stop_is_truncation() -> None:
    body = anthropic_body(stop_reason="model_context_window_exceeded")
    client, _ = _client(httpx2.Response(200, json=body))
    assert client.send(to_request(PROMPT, PARAMS)).stop_reason is StopReason.MAX_TOKENS


@pytest.mark.parametrize(
    "base_url", ["https://api.example.test", "https://api.example.test/v1/"]
)
def test_the_messages_path_is_the_same_with_or_without_v1(base_url: str) -> None:
    transport, seen = scripted(httpx2.Response(200, json=anthropic_body()))
    ep = FakeEndpoint(base_url=base_url, api_key_env=KEY)
    AnthropicClient(ep, transport=transport).send(to_request(PROMPT, PARAMS))
    assert seen[0].url.path == "/v1/messages"


def _body(params: RetryParams) -> dict[str, object]:
    import json

    client, seen = _client(httpx2.Response(200, json=anthropic_body()))
    client.send(to_request(PROMPT, params))
    return json.loads(seen[0].content)


def test_a_reasoning_level_is_adaptive_thinking_at_that_effort() -> None:
    body = _body(
        RetryParams(temperature=0.2, max_tokens=2048, reasoning=ReasoningSpec(level="minimal"))
    )
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}
    assert "temperature" not in body, "the API refuses a temperature beside thinking"


def test_a_reasoning_budget_is_fixed_budget_thinking() -> None:
    spec = ReasoningSpec(level="high", budget_tokens=1024)
    body = _body(RetryParams(temperature=0.2, max_tokens=2048, reasoning=spec))
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert "output_config" not in body


def test_no_reasoning_sends_no_thinking_and_keeps_temperature() -> None:
    body = _body(PARAMS)
    assert "thinking" not in body and "output_config" not in body
    assert body["temperature"] == 0.2


def test_ambient_custom_headers_put_nothing_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Leak: 1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://elsewhere.test")
    client, seen = _client(httpx2.Response(200, json=anthropic_body()))
    client.send(to_request(PROMPT, PARAMS))
    assert "x-leak" not in seen[0].headers
    assert seen[0].url.host == "api.example.test"


def test_an_ambient_credential_header_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "x-api-key: sk-other")
    client, seen = _client(httpx2.Response(200, json=anthropic_body()))
    with pytest.raises(CredentialError, match="ANTHROPIC_CUSTOM_HEADERS"):
        client.send(to_request(PROMPT, PARAMS))
    assert seen == []
