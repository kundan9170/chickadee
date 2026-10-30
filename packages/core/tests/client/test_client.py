"""Transport behavior against a mock server (§4 — Phase 0 DoD)."""

from __future__ import annotations

import httpx2
import pytest
from client_helpers import FakeEndpoint, ok_body, scripted

from chickadee.client import CredentialError, OpenAIClient, TransportExhausted, to_request
from chickadee.schema import Prompt, RetryParams, StopReason

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)


def _client(
    endpoint: FakeEndpoint, *responses: object
) -> tuple[OpenAIClient, list[httpx2.Request]]:
    transport, seen = scripted(*responses)  # type: ignore[arg-type]
    return OpenAIClient(endpoint, transport=transport), seen


def test_usage_capture(endpoint: FakeEndpoint) -> None:
    client, _ = _client(endpoint, httpx2.Response(200, json=ok_body()))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.text == "done"
    assert c.usage.prompt_tokens == 100
    assert c.usage.completion_tokens == 20
    assert c.usage.estimated is False
    assert c.stop_reason is StopReason.END_TURN
    assert c.raw_stop_reason == "stop"
    assert c.latency_ms >= 0


def test_cached_tokens_probe(endpoint: FakeEndpoint) -> None:
    body = ok_body(
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 900},
        }
    )
    client, _ = _client(endpoint, httpx2.Response(200, json=body))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.cached_prompt_tokens == 900
    assert c.cached_prefix is True


def test_absent_cache_field_is_none_not_zero(endpoint: FakeEndpoint) -> None:
    """Unknown is not zero. Collapsing them corrupts the D18 cache-hit metric, because a
    provider that never reports would read as a permanent 0% hit rate."""
    client, _ = _client(endpoint, httpx2.Response(200, json=ok_body()))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.cached_prompt_tokens is None
    assert c.cached_prefix is None


def test_zero_cached_tokens_is_a_real_miss(endpoint: FakeEndpoint) -> None:
    body = ok_body(
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    client, _ = _client(endpoint, httpx2.Response(200, json=body))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.cached_prompt_tokens == 0
    assert c.cached_prefix is False


def test_absent_usage_is_estimated_and_flagged(endpoint: FakeEndpoint) -> None:
    """Some local servers report no usage at all. Estimating is fine; being silently
    precise about an estimate is not."""
    client, _ = _client(endpoint, httpx2.Response(200, json=ok_body(usage=None)))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.estimated is True
    assert c.usage.prompt_tokens > 0


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_internal_retry_on_transient(endpoint: FakeEndpoint, status: int) -> None:
    client, seen = _client(
        endpoint, httpx2.Response(status), httpx2.Response(200, json=ok_body())
    )
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"
    assert len(seen) == 2


@pytest.mark.usefixtures("no_sleep")
def test_retries_are_bounded(endpoint: FakeEndpoint) -> None:
    client, seen = _client(endpoint, httpx2.Response(503))
    with pytest.raises(TransportExhausted) as exc:
        client.send(to_request(PROMPT, PARAMS))
    assert len(seen) == 4, "1 attempt + 3 retries of its kind (D211 decision 3)"
    assert exc.value.attempts == 4


@pytest.mark.parametrize("status", [401, 403])
def test_credentials_abort_rather_than_retry(endpoint: FakeEndpoint, status: int) -> None:
    """An operator error must not burn the retry budget on a request that cannot succeed."""
    client, seen = _client(endpoint, httpx2.Response(status))
    with pytest.raises(CredentialError):
        client.send(to_request(PROMPT, PARAMS))
    assert len(seen) == 1


@pytest.mark.usefixtures("no_sleep")
def test_malformed_json_retries_then_exhausts(endpoint: FakeEndpoint) -> None:
    client, seen = _client(endpoint, httpx2.Response(200, text="<html>gateway</html>"))
    with pytest.raises(TransportExhausted, match="malformed"):
        client.send(to_request(PROMPT, PARAMS))
    assert len(seen) == 4


@pytest.mark.usefixtures("no_sleep")
def test_missing_choices_retries(endpoint: FakeEndpoint) -> None:
    client, _ = _client(
        endpoint, httpx2.Response(200, json={"usage": {}}), httpx2.Response(200, json=ok_body())
    )
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"


def test_deadline_is_transport_exhausted(endpoint: FakeEndpoint) -> None:
    """A half-completion is not evidence — partial text is discarded."""
    client, _ = _client(endpoint, httpx2.ReadTimeout("too slow"))
    with pytest.raises(TransportExhausted, match="deadline"):
        client.send(to_request(PROMPT, PARAMS))


def test_missing_api_key_env_is_a_config_error(endpoint: FakeEndpoint) -> None:
    ep = FakeEndpoint(api_key_env="CHICKADEE_TEST_KEY_THAT_IS_UNSET")
    client, _ = _client(ep, httpx2.Response(200, json=ok_body()))
    with pytest.raises(CredentialError, match="unset"):
        client.send(to_request(PROMPT, PARAMS))


def test_api_key_is_read_from_env_not_stored(
    endpoint: FakeEndpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan §4.3: the engine never stores secrets — it reads the named variable at call
    time, so the value lives in the process environment and nowhere in our state."""
    monkeypatch.setenv("CHICKADEE_TEST_KEY", "sk-secret")
    ep = FakeEndpoint(api_key_env="CHICKADEE_TEST_KEY")
    client, seen = _client(ep, httpx2.Response(200, json=ok_body()))
    client.send(to_request(PROMPT, PARAMS))
    assert seen[0].headers["authorization"] == "Bearer sk-secret"
    assert "sk-secret" not in repr(ep)


def test_request_shape(endpoint: FakeEndpoint) -> None:
    import json

    client, seen = _client(endpoint, httpx2.Response(200, json=ok_body()))
    client.send(
        to_request(PROMPT, RetryParams(temperature=0.4, max_tokens=1024, stop=("</patch>",)))
    )
    body = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/chat/completions")
    assert body["model"] == endpoint.model_id
    assert body["temperature"] == 0.4
    assert body["max_tokens"] == 1024
    assert body["stop"] == ["</patch>"]
    # The prompt rides as one system and one user message, byte-identical, and
    # together they are still exactly `Prompt.full()`.
    assert body["messages"] == [
        {"role": "system", "content": PROMPT.prefix},
        {"role": "user", "content": PROMPT.volatile},
    ]
    assert "".join(m["content"] for m in body["messages"]) == PROMPT.full()


@pytest.mark.parametrize(
    "wire, expected",
    [
        # The OpenAI schema: a JSON-encoded string. This is what real providers send, and
        # narrowing it as an object yielded `{}` — every tool call reaching the
        # orchestrator stripped of its inputs, with no error anywhere to explain it.
        ('{"path": "src/app.py", "start": 10}', {"path": "src/app.py", "start": 10}),
        # Some compatible servers inline the object instead; both are accepted.
        ({"path": "src/app.py", "start": 10}, {"path": "src/app.py", "start": 10}),
        ("{}", {}),
        ("", {}),
        ("not json", {}),
        ('"a string, validly encoded, but not an object"', {}),
        (None, {}),
    ],
)
# The SDK's stream accumulator types `arguments` as a string and warns when it serialises the
# inline-object shape; the shape still reaches `_arguments` whole, which is what is held here.
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
def test_tool_call_arguments_survive_both_wire_shapes(
    endpoint: FakeEndpoint, wire: object, expected: dict[str, object]
) -> None:
    body = ok_body(
        text="",
        finish_reason="tool_calls",
        tool_calls=[{"id": "c1", "function": {"name": "read_file", "arguments": wire}}],
    )
    client, _ = _client(endpoint, httpx2.Response(200, json=body))
    call = client.send(to_request(PROMPT, PARAMS)).tool_calls[0]
    assert call.name == "read_file"
    assert call.arguments == expected


def test_ok_body_does_not_share_its_usage_mapping() -> None:
    """`_DEFAULT_USAGE` is one object; without a copy, a test mutating `body["usage"]`
    poisons every later `ok_body()`."""
    first = ok_body()
    first["usage"]["prompt_tokens"] = 999  # type: ignore[index]
    assert ok_body()["usage"]["prompt_tokens"] == 100  # type: ignore[index]


def test_an_empty_usage_object_is_estimated_not_zero(endpoint: FakeEndpoint) -> None:
    """`{"usage": {}}` carries no counts. The SDK's model declares every counter, so a plain
    dump turned it into `None`s that read as 0 tokens, precise and wrong."""
    client, _ = _client(endpoint, httpx2.Response(200, json=ok_body(usage={})))
    c = client.send(to_request(PROMPT, PARAMS))
    assert c.usage.estimated is True
    assert c.usage.prompt_tokens > 0


@pytest.mark.usefixtures("no_sleep")
def test_a_body_that_says_json_and_is_not_retries(endpoint: FakeEndpoint) -> None:
    truncated = httpx2.Response(
        200, content=b'{"choices": [', headers={"content-type": "application/json"}
    )
    client, seen = _client(endpoint, truncated, httpx2.Response(200, json=ok_body()))
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"
    assert len(seen) == 2


@pytest.mark.usefixtures("no_sleep")
def test_every_5xx_retries_and_other_4xx_do_not(endpoint: FakeEndpoint) -> None:
    client, seen = _client(endpoint, httpx2.Response(529), httpx2.Response(200, json=ok_body()))
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"
    assert len(seen) == 2
    client, seen = _client(endpoint, httpx2.Response(422, text="bad field"))
    with pytest.raises(TransportExhausted, match="422"):
        client.send(to_request(PROMPT, PARAMS))
    assert len(seen) == 1


def test_an_empty_api_key_env_sends_no_authorization() -> None:
    client, seen = _client(FakeEndpoint(api_key_env=""), httpx2.Response(200, json=ok_body()))
    client.send(to_request(PROMPT, PARAMS))
    assert "authorization" not in seen[0].headers


def test_ambient_sdk_variables_put_nothing_on_the_wire(
    endpoint: FakeEndpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK turns these into headers at construction. The rung and the affinity table
    are the only sources of a header; an organization id set for another tool is not sent
    to a third-party endpoint."""
    monkeypatch.setenv("OPENAI_ORG_ID", "org-leak")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-leak")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Leak: 1\nX-Other: 2")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient")
    client, seen = _client(endpoint, httpx2.Response(200, json=ok_body()))
    client.send(to_request(PROMPT, PARAMS))
    headers = {k.lower() for k in seen[0].headers}
    assert not headers & {"openai-organization", "openai-project", "x-leak", "x-other"}
    assert "authorization" not in headers


def test_an_ambient_credential_header_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHICKADEE_TEST_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer sk-other")
    ep = FakeEndpoint(api_key_env="CHICKADEE_TEST_KEY")
    client, seen = _client(ep, httpx2.Response(200, json=ok_body()))
    with pytest.raises(CredentialError, match="OPENAI_CUSTOM_HEADERS"):
        client.send(to_request(PROMPT, PARAMS))
    assert seen == []


def test_a_cancel_during_backoff_ends_the_ladder(endpoint: FakeEndpoint) -> None:
    """The backoff is the longest wait in the ladder, so it is where a cancel lands."""
    import threading
    import time

    from chickadee.client import RequestCancelled, to_request

    cancel = threading.Event()
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        cancel.set()
        return httpx2.Response(503)

    client = OpenAIClient(endpoint, transport=httpx2.MockTransport(handler))
    req = to_request(PROMPT, PARAMS)
    started = time.monotonic()
    with pytest.raises(RequestCancelled):
        client.send(type(req)(messages=req.messages, params=req.params, cancel=cancel))
    assert len(seen) == 1
    assert time.monotonic() - started < 0.5, "the backoff did not wait out its second"
