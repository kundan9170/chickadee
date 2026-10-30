"""A reply is streamed, a silence is the timeout, every failure kind is retried, and Esc stops a
request in flight, through the real clients against a mock server
that paces its events."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

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
    streamed,
)

from chickadee.client import (
    AnthropicClient,
    OpenAIClient,
    RequestCancelled,
    RetryPolicy,
    TransportExhausted,
    to_request,
)
from chickadee.client.openai_client import retry_after_s
from chickadee.client.request import ChatRequest
from chickadee.schema import Prompt, RetryParams, StopReason

PROMPT = Prompt(prefix="SYSTEM\n", volatile="TASK t1\n")
PARAMS = RetryParams(temperature=0.2, max_tokens=2048)
KEY = "CHICKADEE_TEST_ANTHROPIC_KEY"
FAST = FakeEndpoint(first_token_s=0.4, token_gap_s=0.25)
"""Limits short enough to cross in a test."""
ROOMY = FakeEndpoint(first_token_s=0.6, token_gap_s=0.4)
"""Limits a stream paced at 0.1 s outlasts in all, with room for a loaded runner."""
SLACK_S = 0.75
"""How soon after the stop a request must end: a fraction of a second, with room for a loaded
runner, against a server that would take seconds."""


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY, "sk-ant-test")


class Paced(httpx2.SyncByteStream):
    """An event stream that waits `first_s` before its first frame and `gap_s` between frames,
    and records being closed, which is how a stopped request ends on the server. `release`
    ends every wait at once, so no reader thread outlives its test."""

    def __init__(
        self, frames: list[bytes], *, first_s: float = 0.0, gap_s: float = 0.0
    ) -> None:
        self.frames = frames
        self.first_s = first_s
        self.gap_s = gap_s
        self.sent = 0
        self.closed = threading.Event()
        self.release = threading.Event()

    def __iter__(self) -> Iterator[bytes]:
        for i, frame in enumerate(self.frames):
            if (
                self.release.wait(self.first_s if i == 0 else self.gap_s)
                or self.closed.is_set()
            ):
                return
            self.sent += 1
            yield frame

    def close(self) -> None:
        self.closed.set()


def _frames(events: list[dict[str, object]], *, named: bool = False) -> list[bytes]:
    frames = [
        ((f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n").encode()
        for e in events
    ]
    return frames if named else [*frames, b"data: [DONE]\n\n"]


def _long_text(pieces: int) -> list[bytes]:
    """A chat reply whose text arrives in `pieces` chunks of 35 characters (10 tokens at P-2's
    rate), as a model writing a long file does."""
    events = openai_events(ok_body(text=""))
    head = {k: v for k, v in events[0].items() if k != "choices"}
    piece = {"index": 0, "delta": {"content": "x" * 35}, "finish_reason": None}
    return _frames(
        [events[0], *({**head, "choices": [piece]} for _ in range(pieces)), *events[1:]]
    )


class Stalls(Paced):
    """`frames`' first `after` at once, then a silence until released."""

    def __init__(self, frames: list[bytes], *, after: int) -> None:
        super().__init__(frames)
        self.after = after

    def __iter__(self) -> Iterator[bytes]:
        for i, frame in enumerate(self.frames):
            if i >= self.after and (self.release.wait(5.0) or self.closed.is_set()):
                return
            self.sent += 1
            yield frame


def _serve(*streams: Paced | httpx2.Response) -> tuple[httpx2.MockTransport, list[Paced]]:
    """Each request gets the next of `streams`, the last repeating: a paced stream as it is, a
    response as `scripted` serves it (a whole completion as its stream)."""
    queue = list(streams)
    used: list[Paced] = []

    def handler(_request: httpx2.Request) -> httpx2.Response:
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, httpx2.Response):
            return streamed(item)
        used.append(item)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=item)

    return httpx2.MockTransport(handler), used


def _no_waits(waits: list[float] | None = None) -> RetryPolicy:
    return RetryPolicy(
        random=lambda: 0.0,
        sleep=lambda delay, _cancel: None if waits is None else waits.append(delay),
    )


def _request(cancel: threading.Event | None = None) -> ChatRequest:
    req = to_request(PROMPT, PARAMS)
    return ChatRequest(messages=req.messages, params=req.params, cancel=cancel)


# -- a reply that keeps arriving ---------------------------------------------------------------


def test_a_reply_longer_than_both_limits_completes_while_tokens_arrive() -> None:
    """Twelve chunks, 0.1 s apart: 1.2 s in all, twice the first-token limit and three times
    the gap limit, and no gap over either."""
    paced = Paced(_long_text(12), first_s=0.1, gap_s=0.1)
    transport, _ = _serve(paced)
    started = time.monotonic()
    done = OpenAIClient(ROOMY, transport=transport, retry=_no_waits()).send(_request())
    assert time.monotonic() - started > 1.0
    assert done.text == "x" * 35 * 12
    assert done.stop_reason is StopReason.END_TURN
    assert done.usage.prompt_tokens == 100, "the usage chunk is read as a whole reply's was"


def test_the_anthropic_stream_completes_the_same() -> None:
    body = anthropic_body(text="hello", stop_reason="max_tokens")
    paced = Paced(_frames(anthropic_events(body), named=True), gap_s=0.05)
    transport, _ = _serve(paced)
    client = AnthropicClient(
        FakeEndpoint(api_key_env=KEY, first_token_s=2.0, token_gap_s=2.0),
        transport=transport,
        retry=_no_waits(),
    )
    done = client.send(_request())
    assert done.text == "hello"
    assert done.stop_reason is StopReason.MAX_TOKENS
    assert (done.usage.prompt_tokens, done.usage.completion_tokens) == (100, 20)


# -- a silence --------------------------------------------------------------------------------


def test_no_first_token_is_retried_three_times_then_ends_the_ladder() -> None:
    silent = Paced(_frames(openai_events(ok_body())), first_s=5.0)
    transport, used = _serve(silent)
    waits: list[float] = []
    with pytest.raises(TransportExhausted, match=r"no first token within 0\.4s") as err:
        OpenAIClient(FAST, transport=transport, retry=_no_waits(waits)).send(_request())
    silent.release.set()
    assert err.value.attempts == 4
    assert err.value.billed_attempts == (0, 0, 0, 0), "sent, and nothing streamed"
    assert waits == [1.0, 4.0, 16.0]
    assert len(used) == 4
    assert silent.closed.is_set(), "a silent request is closed, so the server stops"


def test_a_gap_mid_reply_is_retried_and_the_retrys_reply_returned() -> None:
    """Four chunks arrive, then the server stalls: the attempt is dropped (a half-completion is
    not evidence) and the retry's whole reply is the one returned."""
    stall = Stalls(_long_text(6), after=5)
    transport, _ = _serve(stall, httpx2.Response(200, json=ok_body()))
    done = OpenAIClient(FAST, transport=transport, retry=_no_waits()).send(_request())
    stall.release.set()
    assert done.text == "done"
    assert stall.closed.is_set()


def test_a_stalled_attempt_is_billed_with_the_output_it_streamed() -> None:
    """A stall after four chunks of 35 characters: 140 characters streamed, 40 tokens at P-2's
    rate, billed on each of the four attempts."""
    stall = Stalls(_long_text(6), after=5)
    transport, _ = _serve(stall)
    with pytest.raises(TransportExhausted, match=r"no token within 0\.25s") as err:
        OpenAIClient(FAST, transport=transport, retry=_no_waits()).send(_request())
    stall.release.set()
    assert err.value.billed_attempts == (40, 40, 40, 40)


# -- the ladder -------------------------------------------------------------------------------


def test_each_kind_has_its_own_three_retries() -> None:
    """Two 503s and then a silence: neither kind is past its retries, so the reply after them
    is returned (one shared count of 3 would have ended the ladder before it)."""
    silent = Paced(_frames(openai_events(ok_body())), first_s=5.0)
    transport, _ = _serve(
        httpx2.Response(503),
        httpx2.Response(503),
        silent,
        httpx2.Response(503),
        httpx2.Response(200, json=ok_body(text="at last")),
    )
    waits: list[float] = []
    client = OpenAIClient(FAST, transport=transport, retry=_no_waits(waits))
    assert client.send(_request()).text == "at last"
    silent.release.set()
    assert waits == [1.0, 4.0, 1.0, 16.0], "each kind's own step: 503, 503, silence, 503"


def test_every_kind_is_retried_and_other_refusals_are_not() -> None:
    for retried in (
        httpx2.ConnectError("refused"),
        httpx2.Response(429),
        httpx2.Response(500),
        httpx2.Response(200, text="<html>gateway</html>"),
        httpx2.ReadTimeout("slow"),
    ):
        transport, seen = scripted(retried, httpx2.Response(200, json=ok_body()))
        client = OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits())
        assert client.send(_request()).text == "done", retried
        assert len(seen) == 2
    for refused in (httpx2.Response(400, text="bad"), httpx2.Response(404)):
        transport, seen = scripted(refused, httpx2.Response(200, json=ok_body()))
        with pytest.raises(TransportExhausted):
            OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits()).send(
                _request()
            )
        assert len(seen) == 1


def test_the_waits_carry_up_to_a_quarter_of_jitter() -> None:
    policy = RetryPolicy(random=lambda: 1.0)
    assert [policy.wait_s(n, None) for n in (1, 2, 3)] == [1.25, 5.0, 20.0]
    assert RetryPolicy(random=lambda: 0.5).wait_s(1, None) == 1.125


@pytest.mark.parametrize(
    ("headers", "waited"),
    [
        ({"Retry-After": "7"}, 7.0),
        ({"Retry-After": "120"}, 60.0),
        ({"retry-after-ms": "2500"}, 2.5),
    ],
    ids=["seconds", "capped-at-60", "milliseconds"],
)
def test_a_429s_retry_after_is_waited_up_to_sixty_seconds(
    headers: dict[str, str], waited: float
) -> None:
    waits: list[float] = []
    transport, _ = scripted(
        httpx2.Response(429, headers=headers), httpx2.Response(200, json=ok_body())
    )
    OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits(waits)).send(_request())
    assert waits == [waited]


def test_a_503s_retry_after_date_is_waited_and_a_500s_is_not() -> None:
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    waits: list[float] = []
    transport, _ = scripted(
        httpx2.Response(503, headers={"Retry-After": later}),
        httpx2.Response(500, headers={"Retry-After": "9"}),
        httpx2.Response(200, json=ok_body()),
    )
    OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits(waits)).send(_request())
    assert 25.0 < waits[0] <= 30.0
    assert waits[1] == 4.0, "a 500's Retry-After is not read: the 5xx kind's second step"


# -- Esc --------------------------------------------------------------------------------------


def test_esc_before_the_first_token_ends_the_request_at_once() -> None:
    """The server has sent nothing, and would not for seconds: the request ends within a
    fraction of a second of the stop, is not retried, and is billed with nothing streamed."""
    silent = Paced(_frames(openai_events(ok_body())), first_s=5.0)
    transport, used = _serve(silent)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(RequestCancelled) as err:
        OpenAIClient(FakeEndpoint(), transport=transport, retry=_no_waits()).send(
            _request(cancel)
        )
    elapsed = time.monotonic() - started
    silent.release.set()
    assert elapsed < 0.1 + SLACK_S
    assert (err.value.attempts, err.value.billed_attempts) == (1, (0,))
    assert len(used) == 1, "never retried"
    assert silent.closed.wait(1.0), "the stream is closed, so the server stops"


def test_esc_mid_reply_closes_the_stream_and_bills_what_streamed() -> None:
    paced = Paced(_long_text(40), gap_s=0.05)
    transport, used = _serve(paced)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(RequestCancelled) as err:
        OpenAIClient(ROOMY, transport=transport, retry=_no_waits()).send(_request(cancel))
    elapsed = time.monotonic() - started
    assert elapsed < 0.3 + SLACK_S
    assert paced.closed.wait(1.0)
    assert paced.sent < len(paced.frames), "the server stopped sending"
    [streamed] = err.value.billed_attempts
    assert streamed > 0
    assert len(used) == 1


# -- the review's cases ----------------------------------------------------------------


class Drops(httpx2.SyncByteStream):
    """`frames`, then the connection dropped under the SDK: the transport's own error, which
    Anthropic's stream lets through unwrapped and OpenAI's wraps as a connection error."""

    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames

    def __iter__(self) -> Iterator[bytes]:
        yield from self.frames
        raise httpx2.RemoteProtocolError("peer closed connection without sending a response")


def _anthropic_text(text: str) -> list[bytes]:
    """An Anthropic reply's events up to `text`'s delta: the message started, nothing ended."""
    events = anthropic_events(anthropic_body(text=text))
    return _frames(events[:3], named=True)


def _anthropic_client(transport: httpx2.MockTransport) -> AnthropicClient:
    return AnthropicClient(
        FakeEndpoint(api_key_env=KEY, first_token_s=2.0, token_gap_s=2.0),
        transport=transport,
        retry=_no_waits(),
    )


def _dropping(frames: list[bytes]) -> httpx2.MockTransport:
    return httpx2.MockTransport(
        lambda _r: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Drops(frames)
        )
    )


def test_a_connection_dropped_mid_reply_is_retried_by_both_transports() -> None:
    """Was, on Anthropic: the transport's error escaped the ladder and ended the turn. Now on
    both, a reply that died part way is a token gap, retried, each attempt billed with the 35
    characters (10 tokens) it had streamed."""
    with pytest.raises(TransportExhausted, match="connection lost mid-reply") as err:
        _anthropic_client(_dropping(_anthropic_text("x" * 35))).send(_request())
    assert err.value.billed_attempts == (10, 10, 10, 10)
    with pytest.raises(TransportExhausted, match="connection lost mid-reply") as err:
        OpenAIClient(ROOMY, transport=_dropping(_long_text(1)[:2]), retry=_no_waits()).send(
            _request()
        )
    assert err.value.billed_attempts == (10, 10, 10, 10)


def test_a_connection_dropped_before_the_reply_is_a_connect_error_unbilled() -> None:
    with pytest.raises(TransportExhausted, match="connect error") as err:
        _anthropic_client(_dropping([])).send(_request())
    assert err.value.attempts == 4
    assert err.value.billed_attempts == ()


def test_a_dropped_attempt_then_a_reply_completes() -> None:
    whole = streamed(httpx2.Response(200, json=anthropic_body(text="hello")))
    replies = iter(
        [
            httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Drops(_anthropic_text("hel")),
            ),
            whole,
        ]
    )
    transport = httpx2.MockTransport(lambda _r: next(replies))
    assert _anthropic_client(transport).send(_request()).text == "hello"


def _overloaded() -> httpx2.Response:
    """Anthropic's overload, arriving inside a stream that had started: the SDK raises it as
    a status error carrying the stream's 200."""
    events = anthropic_events(anthropic_body(text="x" * 35))[:3]
    error = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    return sse([*events, error], named=True)


def test_an_overload_inside_the_stream_is_a_server_error_never_billed() -> None:
    """Was: a malformed reply, billed per attempt. Now retried as a 5xx is, and not billed."""
    transport, _ = scripted(_overloaded())
    with pytest.raises(TransportExhausted, match="overloaded") as err:
        _anthropic_client(transport).send(_request())
    assert err.value.attempts == 4
    assert err.value.billed_attempts == ()
    transport, _ = scripted(_overloaded(), httpx2.Response(200, json=anthropic_body()))
    assert _anthropic_client(transport).send(_request()).text == "done"


def test_any_other_error_event_is_a_malformed_reply_and_billed() -> None:
    events = anthropic_events(anthropic_body(text="x" * 35))[:3]
    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    transport, _ = scripted(sse([*events, error], named=True))
    with pytest.raises(TransportExhausted, match="malformed stream") as err:
        _anthropic_client(transport).send(_request())
    assert err.value.billed_attempts == (10, 10, 10, 10)


def test_a_stalled_reasoning_model_is_billed_with_the_reasoning_it_streamed() -> None:
    """Was: an OpenAI-compatible server's `reasoning_content` counted nothing, where
    Anthropic's thinking counted. 70 characters of reasoning streamed: 20 tokens."""
    events = openai_events(ok_body(text=""))
    head = {k: v for k, v in events[0].items() if k != "choices"}
    thinking = {"index": 0, "delta": {"reasoning_content": "y" * 35}, "finish_reason": None}
    reasoning = {**head, "choices": [thinking]}
    frames = _frames([events[0], reasoning, reasoning])
    stall = Stalls(frames, after=3)
    transport, _ = _serve(stall)
    with pytest.raises(TransportExhausted, match="no token") as err:
        OpenAIClient(FAST, transport=transport, retry=_no_waits()).send(_request())
    stall.release.set()
    assert err.value.billed_attempts == (20, 20, 20, 20)


class _Refusal(Exception):
    def __init__(self, headers: dict[str, str]) -> None:
        self.response = httpx2.Response(503, headers=headers)


@pytest.mark.parametrize(
    ("headers", "waited"),
    [
        ({"Retry-After": "nan"}, None),
        ({"Retry-After": "inf"}, None),
        ({"Retry-After": "-inf"}, None),
        ({"retry-after-ms": "nan", "Retry-After": "3"}, 3.0),
        ({"Retry-After": "not a date"}, None),
    ],
    ids=["nan", "inf", "-inf", "ms-nan-falls-back", "garbage"],
)
def test_a_retry_after_that_is_no_wait_is_none(
    headers: dict[str, str], waited: float | None
) -> None:
    """Was: `nan` and `inf` passed as waits, and `sleep(nan)` raised."""
    assert retry_after_s(_Refusal(headers)) == waited


def test_a_retry_after_date_in_the_minus_zero_zone_is_read_as_gmt() -> None:
    """Was: `-0000` parsed to a datetime with no zone, and subtracting it raised `TypeError`
    out of the ladder."""
    later = (datetime.now(UTC) + timedelta(seconds=30)).strftime("%a, %d %b %Y %H:%M:%S -0000")
    waited = retry_after_s(_Refusal({"Retry-After": later}))
    assert waited is not None and 25.0 < waited <= 30.0


def test_a_models_default_limits_are_five_minutes_to_start_and_one_between_tokens() -> None:
    """The owner's call (2026-10-20): five minutes for the first token, over D211's two, so a
    local model loading or a long prefill is not retried as a server that never started."""
    from chickadee.config import RungEndpoint

    model = RungEndpoint(base_url="u", model_id="m", context_window=4096, max_output_tokens=512)
    assert (model.first_token_s, model.token_gap_s) == (300.0, 60.0)
