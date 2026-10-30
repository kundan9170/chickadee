"""What the one `ModelClient` path sends, pinned before `complete()` retires.

`complete(prompt, params, session_key)` was `send(to_request(prompt, params,
session_key=session_key))` in both transports, so every caller moved off it sends the request
pinned here. And `close` joins the protocol, so a caller may call it without knowing whether
someone already has: both transports tolerate a second close.
"""

from __future__ import annotations

import httpx2
import pytest
from client_helpers import FakeEndpoint

from chickadee.client import AnthropicClient, ChatRequest, Message, OpenAIClient, to_request
from chickadee.schema import Prompt, RetryParams


def test_a_prompt_is_one_system_message_and_one_user_message_byte_for_byte() -> None:
    params = RetryParams(temperature=0.3, max_tokens=64)
    request = to_request(
        Prompt(prefix="the stable prefix\n", volatile="the volatile tail\n"),
        params,
        session_key="affinity-1",
    )
    assert request == ChatRequest(
        messages=(
            Message(role="system", content="the stable prefix\n"),
            Message(role="user", content="the volatile tail\n"),
        ),
        params=params,
        session_key="affinity-1",
    )
    assert request.tools == () and request.cancel is None


def _never(request: httpx2.Request) -> httpx2.Response:
    raise AssertionError(f"close sent a request: {request.url}")


@pytest.mark.parametrize("transport", [OpenAIClient, AnthropicClient])
def test_close_tolerates_a_second_close(
    transport: type[OpenAIClient] | type[AnthropicClient],
) -> None:
    client = transport(FakeEndpoint(), transport=httpx2.MockTransport(_never))
    client.close()
    client.close()
