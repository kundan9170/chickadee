"""A mock endpoint and a scriptable transport, so the client is tested against real HTTP
semantics without a network."""

from __future__ import annotations

from collections.abc import Callable

import httpx2
import pytest
from client_helpers import FakeEndpoint


@pytest.fixture
def endpoint() -> FakeEndpoint:
    return FakeEndpoint()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff is real seconds; the ladder's shape is asserted separately."""
    monkeypatch.setattr("chickadee.client.openai_client.time.sleep", lambda _s: None)


Handler = Callable[[httpx2.Request], httpx2.Response]
