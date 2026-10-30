"""Finding a server by probing, and what the client makes of a server it
cannot speak to or a refusal it cannot read."""

from __future__ import annotations

import asyncio
import http.server
import threading
from pathlib import Path

import httpx
import pytest
from hosting import hosted

from chickadee_serve.bind import SOCKET_NAME, Tcp, Uds
from chickadee_serve.frames import WIRE_VERSION, ErrorCode, major
from chickadee_wire.client import AsyncClient, Client
from chickadee_wire.errors import NotServing, Unanswered, WireError, WireMismatch
from chickadee_wire.transport import base_url, discover, probe

ADDRESS = Uds(Path("/nonexistent/serve.sock"))


def _hello(wire: str) -> httpx.MockTransport:
    body = {"wire": wire, "engine": "x", "capabilities": [], "replay": False}
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=body))


def test_nothing_serving_names_every_place_it_looked(tmp_path: Path) -> None:
    with pytest.raises(NotServing) as refused:
        discover(tmp_path, Tcp("127.0.0.1", 9))
    assert str(tmp_path / SOCKET_NAME) in str(refused.value)
    assert "http://127.0.0.1:9" in str(refused.value)


def test_a_socket_file_nothing_answers_on_is_skipped_not_removed(tmp_path: Path) -> None:
    (tmp_path / SOCKET_NAME).write_text("")
    assert probe(Uds(tmp_path / SOCKET_NAME)) is None
    assert (tmp_path / SOCKET_NAME).exists(), "reclaiming it is the next server's"


def test_the_socket_is_found_before_a_loopback_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with hosted(monkeypatch) as host:
        assert discover(host.state_dir, Tcp("127.0.0.1", 9)) == host.address


def test_another_major_version_is_refused_and_a_minor_one_is_not() -> None:
    other = f"{major(WIRE_VERSION) + 1}.0"
    with pytest.raises(WireMismatch) as refused:
        Client.connect(ADDRESS, "t", transport=_hello(other))
    assert (refused.value.client, refused.value.server) == (WIRE_VERSION, other)
    ahead = f"{major(WIRE_VERSION)}.99"
    with Client.connect(ADDRESS, "t", transport=_hello(ahead)) as client:
        assert client.server is not None and client.server.wire == ahead


def test_the_async_client_refuses_another_major_version() -> None:
    other = f"{major(WIRE_VERSION) + 1}.0"
    with pytest.raises(WireMismatch):
        asyncio.run(AsyncClient.connect(ADDRESS, "t", transport=_hello(other)))


def test_a_refusal_is_typed_and_one_that_is_not_an_error_body_is_internal() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session":
            return httpx.Response(409, json={"code": "BUSY", "message": "in flight"})
        return httpx.Response(502, text="bad gateway")

    with Client(ADDRESS, "t", transport=httpx.MockTransport(refuse)) as client:
        with pytest.raises(WireError) as busy:
            client.sessions()
        assert (busy.value.status, busy.value.code) == (409, ErrorCode.BUSY)
        with pytest.raises(WireError) as odd:
            client.session_info("s")
        assert (odd.value.status, odd.value.code) == (502, ErrorCode.INTERNAL)
        assert odd.value.message == "bad gateway"


def _counting(valid: str) -> tuple[httpx.MockTransport, list[str]]:
    seen: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent = request.headers["authorization"].removeprefix("Bearer ")
        seen.append(sent)
        if sent != valid:
            return httpx.Response(401, json={"code": "UNAUTHORIZED", "message": "no"})
        return httpx.Response(200, json=[])

    return httpx.MockTransport(answer), seen


def test_a_401_rereads_a_token_source_once() -> None:
    current = ["old"]
    transport, seen = _counting("new")
    with Client(ADDRESS, lambda: current[0], transport=transport) as client:
        current[0] = "new"  # the server restarted and wrote new tokens
        assert client.sessions() == ()
    assert seen == ["old", "new"]


def test_a_401_is_the_answer_when_nothing_new_can_be_read() -> None:
    transport, seen = _counting("valid")
    with Client(ADDRESS, "fixed", transport=transport) as client, pytest.raises(WireError):
        client.sessions()
    assert seen == ["fixed"], "a fixed token is never retried"

    def gone() -> str:
        if seen:
            raise FileNotFoundError("serve.token")
        return "stale"

    seen.clear()
    with Client(ADDRESS, gone, transport=transport) as client, pytest.raises(WireError):
        client.sessions()
    assert seen == ["stale"], "no token file: the 401 stands"


def test_an_ipv6_loopback_address_is_a_valid_url() -> None:
    assert base_url(Tcp("::1", 8765)) == "http://[::1]:8765"
    assert base_url(Tcp("127.0.0.1", 8765)) == "http://127.0.0.1:8765"
    assert httpx.URL(base_url(Tcp("::1", 8765))).host == "::1"


def test_a_server_that_is_not_chickadee_is_not_found() -> None:
    """Something answers `GET /` with 200 on the port, but not with a `Hello`."""

    class Other(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b'{"hello": "world"}'
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Quiet."""

    server = http.server.HTTPServer(("127.0.0.1", 0), Other)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert probe(Tcp("127.0.0.1", server.server_port)) is None
    finally:
        server.shutdown()
        server.server_close()


def _silent(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _broken(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def test_a_server_that_does_not_answer_is_a_wire_error_not_a_crash() -> None:
    """A timeout or a broken connection was the transport's own exception, which every
    caller catching `WireError` let through. It is `Unanswered`, a `WireError`, naming the
    route and the limit."""
    silent = Client(ADDRESS, "t", timeout_s=30, transport=httpx.MockTransport(_silent))
    with silent as client, pytest.raises(Unanswered) as late:
        client.turn("S-1", "hi")
    assert isinstance(late.value, WireError) and late.value.code is None
    assert late.value.message == (
        "the server did not answer POST /session/S-1/turn: no answer came within 30 s; it may"
        " have been received, so check before sending it again"
    )
    assert late.value.maybe_delivered is True
    assert isinstance(late.value.__cause__, httpx.ReadTimeout), "the transport's cause is kept"
    broken = Client(ADDRESS, "t", transport=httpx.MockTransport(_broken))
    with broken as client, pytest.raises(Unanswered, match="the connection failed") as refused:
        client.sessions()
    assert refused.value.maybe_delivered is False, "never sent: known not to have arrived"


@pytest.mark.parametrize(
    ("raised", "said", "maybe"),
    [
        (httpx.ConnectTimeout, "no connection was made within 5 s", False),
        (httpx.PoolTimeout, "no connection was free within 30 s", False),
        (httpx.WriteTimeout, "the request was not sent within 30 s", True),
        (httpx.RemoteProtocolError, "the connection broke (gone)", True),
    ],
)
def test_an_unanswered_request_names_what_happened_and_whether_it_may_have_arrived(
    raised: type[httpx.TransportError], said: str, maybe: bool
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise raised("gone", request=request)

    client = Client(ADDRESS, "t", timeout_s=30, transport=httpx.MockTransport(fail))
    client._http.timeout = httpx.Timeout(30, connect=5)  # pyright: ignore[reportPrivateUsage]
    with client, pytest.raises(Unanswered) as unanswered:
        client.turn("S-1", "hi")
    assert said in unanswered.value.message
    assert unanswered.value.maybe_delivered is maybe


def test_the_async_client_says_an_unanswered_request_too() -> None:
    async def ask() -> None:
        async with AsyncClient(
            ADDRESS, "t", timeout_s=30, transport=httpx.MockTransport(_silent)
        ) as client:
            await client.turn("S-1", "hi")

    with pytest.raises(Unanswered, match="no answer came within 30 s"):
        asyncio.run(ask())
