"""Which failed requests the provider may have billed.

A failed call is charged only when it was sent. An attempt count cannot say so: it also counts
a connect error, a 429, a 5xx and a 400, none of which a provider bills. So the error that ends
the ladder says it, as `billed_attempts`: each attempt that fell silent past a limit while the
provider was working, got back a body that was not a completion, or was cancelled in flight,
whichever attempt it was, as the output tokens it had streamed.
"""

from __future__ import annotations

import threading

import httpx2
import pytest
from client_helpers import FakeEndpoint, scripted

from chickadee.client import (
    AnthropicClient,
    ClientError,
    CredentialError,
    OpenAIClient,
    RequestCancelled,
    RetryPolicy,
    TransportExhausted,
    to_request,
)
from chickadee.schema import Prompt, RetryParams

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)
KEY = "CHICKADEE_TEST_ANTHROPIC_KEY"
NOT_A_COMPLETION = httpx2.Response(200, text="<html>gateway</html>")


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "sk-ant-test")
    monkeypatch.setattr("chickadee.client.openai_client.time.sleep", lambda _s: None)


def _openai(*responses: httpx2.Response | Exception) -> ClientError:
    transport, _ = scripted(*responses)
    with pytest.raises(ClientError) as err:
        OpenAIClient(FakeEndpoint(), transport=transport).send(to_request(PROMPT, PARAMS))
    return err.value


def _anthropic(*responses: httpx2.Response | Exception) -> ClientError:
    transport, _ = scripted(*responses)
    client = AnthropicClient(FakeEndpoint(api_key_env=KEY), transport=transport)
    with pytest.raises(ClientError) as err:
        client.send(to_request(PROMPT, PARAMS))
    return err.value


@pytest.mark.parametrize(
    "response",
    [
        httpx2.ConnectError("refused"),
        httpx2.Response(429),
        httpx2.Response(503),
        httpx2.Response(400, text="context too long"),
        httpx2.Response(422, text="bad field"),
    ],
    ids=["connect", "429", "503", "400", "422"],
)
def test_a_failure_no_provider_bills_says_so(response: httpx2.Response | Exception) -> None:
    err = _openai(response)
    assert isinstance(err, TransportExhausted)
    assert err.attempts >= 1, "attempts alone would have read this as sent"
    assert err.billed_attempts == ()


def test_a_deadline_before_the_provider_answered_is_not_billed() -> None:
    """Retried 3 times. The read timed out before the stream's headers came
    back, so the provider never took the request on and none is billed: a host that cannot be
    reached must not spend the cap. A silence after the headers is billed
    (`test_streamed.test_no_first_token_is_retried_three_times_then_ends_the_ladder`)."""
    err = _openai(httpx2.ReadTimeout("too slow"))
    assert isinstance(err, TransportExhausted)
    assert err.attempts == 4
    assert err.billed_attempts == ()


def test_each_body_that_is_not_a_completion_may_be_billed() -> None:
    """Four attempts were answered, so four may be billed: one entry per attempt."""
    err = _openai(NOT_A_COMPLETION)
    assert isinstance(err, TransportExhausted)
    assert err.billed_attempts == (0, 0, 0, 0)


def test_a_body_with_no_choices_may_be_billed() -> None:
    err = _openai(httpx2.Response(200, json={"usage": {}}))
    assert err.billed_attempts == (0, 0, 0, 0)


def test_an_earlier_attempt_that_may_be_billed_is_not_forgotten_by_a_later_retry() -> None:
    """The provider answered the first attempt; the retries after it never connected."""
    err = _openai(NOT_A_COMPLETION, httpx2.ConnectError("refused"))
    assert isinstance(err, TransportExhausted)
    assert err.attempts == 5, "the malformed body, then the connect error's own 4"
    assert err.billed_attempts == (0,)


def test_an_earlier_attempt_that_may_be_billed_survives_a_terminal_status() -> None:
    """A 400 ends the ladder at once, from inside an attempt; the earlier answer still
    counts."""
    err = _openai(NOT_A_COMPLETION, httpx2.Response(400, text="context too long"))
    assert isinstance(err, TransportExhausted)
    assert "400" in str(err)
    assert err.billed_attempts == (0,)


def test_a_credential_refused_is_never_billed() -> None:
    err = _openai(httpx2.Response(401))
    assert isinstance(err, CredentialError)
    assert err.billed_attempts == ()


def _cancelled_after(first: httpx2.Response) -> RequestCancelled:
    """The person's stop lands in the backoff after `first`: after the attempt answered, never
    while it was in flight (an attempt cancelled in flight is billed, `test_streamed.py`)."""
    cancel = threading.Event()
    transport, _ = scripted(first)
    client = OpenAIClient(
        FakeEndpoint(),
        transport=transport,
        retry=RetryPolicy(sleep=lambda _delay, _cancel: cancel.set()),
    )
    req = to_request(PROMPT, PARAMS)
    with pytest.raises(RequestCancelled) as err:
        client.send(type(req)(messages=req.messages, params=req.params, cancel=cancel))
    return err.value


def test_a_cancel_after_an_unbilled_attempt_is_not_billed() -> None:
    err = _cancelled_after(httpx2.Response(503))
    assert err.attempts == 1
    assert err.billed_attempts == ()


def test_a_cancel_after_an_answered_attempt_may_be_billed() -> None:
    err = _cancelled_after(NOT_A_COMPLETION)
    assert err.attempts == 1
    assert err.billed_attempts == (0,)


def test_the_anthropic_transport_says_the_same() -> None:
    assert _anthropic(httpx2.Response(529)).billed_attempts == ()
    assert _anthropic(httpx2.ConnectError("refused")).billed_attempts == ()
    assert _anthropic(httpx2.ReadTimeout("too slow")).billed_attempts == ()
    not_a_message = httpx2.Response(200, json={"type": "message"})
    assert _anthropic(not_a_message).billed_attempts == (0, 0, 0, 0)


def test_a_billed_attempt_survives_a_credential_refused_on_the_retry() -> None:
    """The key is revoked between two attempts. The first was answered, and stays counted on
    whatever error ends the ladder."""
    err = _openai(NOT_A_COMPLETION, httpx2.Response(401))
    assert isinstance(err, CredentialError)
    assert err.billed_attempts == (0,)
