"""The client's method set is the route table: `Client` and `AsyncClient`
each have one method per row of `routes.ROUTES`, and none besides. Each method is driven
through a transport that records what it sent, so the test proves which route a method reaches,
not merely that a method of that name exists."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from chickadee_serve.bind import Uds
from chickadee_serve.routes import ROUTES
from chickadee_wire.client import AsyncClient, Client
from chickadee_wire.events import AsyncEvents, Events
from chickadee_wire.models import CoderRequest, OneTask

ARGS: dict[str, object] = {
    "sid": "s",
    "run_id": "r",
    "qid": "q",
    "job_id": "j1",
    "text": "hello",
    "choice": "keep",
    "question": "where?",
    "request": CoderRequest(
        task=OneTask(title="t", brief="do t", sha="a" * 64, template_sha="b" * 64)
    ),
    "to_turn": 1,
    "code": False,
    "conversation": False,
    "at_seq": 1,
    "reason": "why",
    "rung": "zeus",
    "model": "m",
    "hard_cost_usd": None,
    "task_id": "T1",
    "operation_id": "o",
    "sha": "c" * 64,
    "path": "src/app.py",
    "name": "shot.png",
    "data": b"\x89PNG",
}
"""A value for every required parameter any method takes. A new one fails `_required`."""

NOT_ROUTES = {"connect", "close", "aclose"}
"""The client's own lifecycle: the only public methods that are not a route."""

ADDRESS = Uds(Path("/nonexistent/serve.sock"))


class Sent(Exception):
    def __init__(self, method: str, path: str) -> None:
        self.method = method
        self.path = path


def _record(request: httpx.Request) -> httpx.Response:
    raise Sent(request.method, request.url.raw_path.decode().partition("?")[0])


def _expected() -> set[tuple[str, str]]:
    """Each row's method and path, its parameters filled as `ARGS` fills them."""
    return {
        (r.method, re.sub(r"\{(\w+)\}", lambda m: str(ARGS[m.group(1)]), r.path))
        for r in ROUTES
    }


def _route_methods(cls: type) -> dict[str, Callable[..., Any]]:
    return {
        name: member
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_") and name not in NOT_ROUTES
    }


def _required(fn: Callable[..., Any]) -> dict[str, object]:
    params = list(inspect.signature(fn).parameters.values())[1:]
    needed = [p.name for p in params if p.default is inspect.Parameter.empty]
    missing = [name for name in needed if name not in ARGS]
    assert not missing, f"{fn.__name__} takes {missing}: give ARGS a value"
    return {name: ARGS[name] for name in needed}


def test_the_sync_client_reaches_every_route_and_nothing_else() -> None:
    reached: dict[str, tuple[str, str]] = {}
    client = Client(ADDRESS, "t", transport=httpx.MockTransport(_record))
    for name, fn in _route_methods(Client).items():
        with pytest.raises(Sent) as sent:
            result = fn(client, **_required(fn))
            if isinstance(result, Events):
                next(result)
        reached[name] = (sent.value.method, sent.value.path)
    assert sorted(reached.values()) == sorted(_expected()), "one method per route"


def test_the_async_client_reaches_every_route_and_nothing_else() -> None:
    async def drive() -> dict[str, tuple[str, str]]:
        reached: dict[str, tuple[str, str]] = {}
        client = AsyncClient(ADDRESS, "t", transport=httpx.MockTransport(_record))
        for name, fn in _route_methods(AsyncClient).items():
            with pytest.raises(Sent) as sent:
                result = fn(client, **_required(fn))
                if isinstance(result, AsyncEvents):
                    await anext(result)
                else:
                    await result
            reached[name] = (sent.value.method, sent.value.path)
        await client.aclose()
        return reached

    reached = asyncio.run(drive())
    assert sorted(reached.values()) == sorted(_expected()), "one method per route"


def test_both_clients_are_one_table() -> None:
    assert set(_route_methods(Client)) == set(_route_methods(AsyncClient))


def test_no_answer_is_untyped() -> None:
    """Every route answers a declared type; `JsonValue` would be a dict standing in for one."""
    for cls in (Client, AsyncClient):
        untyped = {
            name
            for name, fn in _route_methods(cls).items()
            if "JsonValue" in str(inspect.signature(fn).return_annotation)
        }
        assert untyped == set(), cls.__name__


def test_every_request_carries_the_token() -> None:
    seen: list[str | None] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=[])

    with Client(ADDRESS, "secret", transport=httpx.MockTransport(capture)) as client:
        client.sessions()
    assert seen == ["Bearer secret"]


def test_a_path_parameter_is_one_segment_whatever_it_holds() -> None:
    with pytest.raises(Sent) as sent:
        Client(ADDRESS, "t", transport=httpx.MockTransport(_record)).session_info("a/b c")
    assert sent.value.path == "/session/a%2Fb%20c"
