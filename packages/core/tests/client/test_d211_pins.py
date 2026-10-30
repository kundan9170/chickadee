"""The five answers D211 moved, pinned through the real client before it changed, and
each moved to D211's answer. Each test's docstring says what it answered before; the task's Log
lists each move and why it is right.
"""

from __future__ import annotations

import threading
import time

import httpx2
from client_helpers import FakeEndpoint, ok_body, scripted

from chickadee.client import (
    OpenAIClient,
    RequestCancelled,
    RetryPolicy,
    TransportExhausted,
    to_request,
)
from chickadee.schema import Prompt, RetryParams

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)


def _recording(waits: list[float]) -> RetryPolicy:
    """No jitter, and each wait recorded rather than slept."""
    return RetryPolicy(random=lambda: 0.0, sleep=lambda delay, _cancel: waits.append(delay))


def test_moved_a_reply_slower_than_the_deadline_is_retried() -> None:
    """Was: one attempt, then `TransportExhausted(deadline exceeded)`, `billed_attempts == 1`.
    Now a silence is retried 3 times (decision 3). This one timed out before the provider
    answered with the stream's headers, so no attempt is billed."""
    waits: list[float] = []
    transport, seen = scripted(httpx2.ReadTimeout("too slow"))
    client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_recording(waits))
    try:
        client.send(to_request(PROMPT, PARAMS))
    except TransportExhausted as err:
        assert "deadline exceeded" in str(err)
        assert err.attempts == 4
        assert err.billed_attempts == ()
    else:
        raise AssertionError("a request that never answers cannot complete")
    assert len(seen) == 4
    assert waits == [1.0, 4.0, 16.0]


def test_moved_503s_get_three_retries_waiting_one_four_and_sixteen() -> None:
    """Was: 3 attempts in all, waiting 1 s and 4 s."""
    waits: list[float] = []
    transport, seen = scripted(httpx2.Response(503))
    client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_recording(waits))
    try:
        client.send(to_request(PROMPT, PARAMS))
    except TransportExhausted as err:
        assert err.attempts == 4
    else:
        raise AssertionError("503s past their retries end the ladder")
    assert len(seen) == 4
    assert waits == [1.0, 4.0, 16.0]


def test_moved_a_429s_retry_after_is_honoured() -> None:
    """Was: retried after 1 s, whatever `Retry-After` said."""
    waits: list[float] = []
    transport, _ = scripted(
        httpx2.Response(429, headers={"Retry-After": "10"}),
        httpx2.Response(200, json=ok_body()),
    )
    client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_recording(waits))
    assert client.send(to_request(PROMPT, PARAMS)).text == "done"
    assert waits == [10.0]


def test_moved_a_cancel_during_a_request_stops_it_at_once() -> None:
    """Was: the request ran to its reply (2 s here, up to `timeout_s` live), and the reply
    was returned. Now the request ends within a fraction of a second, never retried; stopped
    before the provider answered with the stream's headers, it is not billed."""
    cancel = threading.Event()
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        cancel.set()
        time.sleep(2.0)
        return httpx2.Response(200, json=ok_body())

    client = OpenAIClient(FakeEndpoint(), transport=httpx2.MockTransport(handler))
    req = to_request(PROMPT, PARAMS)
    started = time.monotonic()
    try:
        client.send(type(req)(messages=req.messages, params=req.params, cancel=cancel))
    except RequestCancelled as err:
        assert err.attempts == 1
        assert err.billed_attempts == ()
    else:
        raise AssertionError("a cancelled request returns no reply")
    assert time.monotonic() - started < 1.0
    assert len(seen) == 1
