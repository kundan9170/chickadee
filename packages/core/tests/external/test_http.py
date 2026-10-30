"""The `http` engine against a local server (`external/fetch/http.py`):
it connects to the address the guard checked, says where a redirect points without following
it, cuts a body at its cap, refuses a type that is not text, and names each failure."""

from __future__ import annotations

import contextlib
import logging
import shutil
import socket
import ssl
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from chickadee.context.results import FetchFailed
from chickadee.external.fetch import Fetched, Moved
from chickadee.external.fetch.http import (
    HttpEngine,
    _interleaved,  # pyright: ignore[reportPrivateUsage]
)
from chickadee.external.urls import Admitted

HOST = "docs.example.test"
"""A name nothing resolves: the request can only reach the server by the address it is given."""


class _Handler(BaseHTTPRequestHandler):
    hosts: ClassVar[list[str]] = []
    """Each request's `Host` header, reset by the `port` fixture."""

    def do_GET(self) -> None:
        type(self).hosts.append(self.headers.get("Host", ""))
        if self.path == "/page":
            self._send(200, "text/html", b"<h1>Hi</h1><p>See <a href='/next'>next</a>.</p>")
        elif self.path == "/moved":
            self.send_response(301)
            self.send_header("Location", "/page")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/big":
            self._send(200, "text/plain", b"x" * 5000)
        elif self.path == "/image":
            self._send(200, "image/png", b"\x89PNG")
        elif self.path == "/bad-gzip":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", "8")
            self.end_headers()
            self.wfile.write(b"not gzip")
        elif self.path == "/slow":
            time.sleep(1.0)
            self._send(200, "text/plain", b"late")
        else:
            self._send(404, "text/plain", b"no")

    def _send(self, status: int, kind: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # `/slow`: the client gave up first, which is what that test asks of it.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


@pytest.fixture
def port() -> Iterator[int]:
    _Handler.hosts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def target(port: int, path: str) -> Admitted:
    return Admitted(f"http://{HOST}:{port}{path}", HOST, port, ("127.0.0.1",), False)


def engine(**env: str) -> HttpEngine:
    return HttpEngine(connect_ms=2000, timeout_ms=500, max_bytes=1000, env=env)


def test_a_page_is_its_text_fetched_from_the_checked_address_under_its_own_name(
    port: int,
) -> None:
    got = engine().get(target(port, "/page"))
    assert got == Fetched(
        f"http://{HOST}:{port}/page", f"# Hi\n\nSee [next](http://{HOST}:{port}/next).", False
    )
    assert _Handler.hosts == [f"{HOST}:{port}"]


def test_a_redirect_is_not_followed_and_says_where_it_points(port: int) -> None:
    assert engine().get(target(port, "/moved")) == Moved(
        f"http://{HOST}:{port}/moved", f"http://{HOST}:{port}/page", 301
    )
    assert len(_Handler.hosts) == 1


def test_a_body_past_its_cap_is_cut_and_says_so(port: int) -> None:
    got = engine().get(target(port, "/big"))
    assert isinstance(got, Fetched) and got.truncated and got.text == "x" * 1000


@pytest.mark.parametrize(
    ("path", "failed"),
    [
        ("/image", ("not_text", 200)),
        ("/missing", ("http_status", 404)),
        # Sent, and its own encoding cannot undo it: not a page.
        ("/bad-gzip", ("not_text", None)),
        ("/slow", ("timeout", None)),
    ],
)
def test_each_failure_is_named(port: int, path: str, failed: tuple[str, int | None]) -> None:
    got = engine().get(target(port, path))
    assert isinstance(got, FetchFailed) and (got.why, got.status) == failed


def test_a_host_that_does_not_answer_is_a_connection_failure() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    got = engine().get(target(closed, "/page"))
    assert isinstance(got, FetchFailed) and got.why == "connection"


def test_behind_a_proxy_the_proxy_connects_unless_no_proxy_names_the_host(port: int) -> None:
    """The proxy here answers nothing, so going through it fails; `NO_PROXY` keeps the fetch
    on the checked address."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        proxy = f"http://127.0.0.1:{s.getsockname()[1]}"
    through = engine(HTTP_PROXY=proxy).get(target(port, "/page"))
    assert isinstance(through, FetchFailed) and through.why == "connection"
    for no_proxy in (HOST, ".example.test,other.org", "*"):
        direct = engine(HTTP_PROXY=proxy, NO_PROXY=no_proxy).get(target(port, "/page"))
        assert isinstance(direct, Fetched), no_proxy


# -- TLS: the pinned connection verifies the host's certificate --------------------------------


@pytest.fixture
def tls(tmp_path: Path) -> Iterator[tuple[int, Path]]:
    """A local HTTPS server for `HOST`, its certificate self-signed, made for this test: the
    port, and the certificate a client must be told to trust."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl makes the test's certificate, and none is installed")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            f"/CN={HOST}",
            "-addext",
            f"subjectAltName=DNS:{HOST}",
        ],
        check=True,
        capture_output=True,
    )
    _Handler.hosts = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], cert
    server.shutdown()
    server.server_close()


def secure(port: int, *addresses: str) -> Admitted:
    return Admitted(
        f"https://{HOST}:{port}/page", HOST, port, addresses or ("127.0.0.1",), False
    )


def test_https_is_verified_for_the_host_at_the_checked_address(tls: tuple[int, Path]) -> None:
    port, cert = tls
    got = engine(SSL_CERT_FILE=str(cert)).get(secure(port))
    assert isinstance(got, Fetched)
    assert got.text == f"# Hi\n\nSee [next](https://{HOST}:{port}/next)."
    assert _Handler.hosts == [f"{HOST}:{port}"]


def test_a_certificate_the_machine_does_not_trust_is_said_where_the_person_sees_it(
    tls: tuple[int, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """With no CA named, the machine's own store decides, and it has never seen this test's
    certificate. The answer says only `connection`; the log says why."""
    port, _ = tls
    with caplog.at_level(logging.WARNING, logger="chickadee.external.fetch.http"):
        got = engine().get(secure(port))
    assert isinstance(got, FetchFailed) and got.why == "connection"
    (said,) = caplog.records  # once a fetch, however many addresses it tried
    assert HOST in said.getMessage() and "CERTIFICATE_VERIFY_FAILED" in said.getMessage()


@pytest.fixture
def silent(port: int) -> Iterator[str]:
    """An address that never answers on `port`: a listener whose backlog is full, which drops
    each new connection attempt, so a connect waits out its limit, as on a dead route."""
    listener = socket.socket()
    listener.bind(("127.0.0.2", port))
    listener.listen(0)
    waiting: list[socket.socket] = []
    for _ in range(4):
        s = socket.socket()
        s.setblocking(False)
        with contextlib.suppress(BlockingIOError):
            s.connect(("127.0.0.2", port))
        waiting.append(s)
    yield "127.0.0.2"
    for s in waiting:
        s.close()
    listener.close()


def test_an_address_that_will_not_connect_moves_to_the_next(port: int, silent: str) -> None:
    """The host's first address is silent past the connect limit; its next one answers, so the
    page is fetched rather than the fetch said to have timed out."""
    quick = HttpEngine(connect_ms=300, timeout_ms=5000, max_bytes=1000, env={})
    got = quick.get(
        Admitted(f"http://{HOST}:{port}/page", HOST, port, (silent, "127.0.0.1"), False)
    )
    assert isinstance(got, Fetched)


def test_the_address_families_are_taken_in_turn() -> None:
    """A host listing its IPv6 addresses first reaches IPv4 on its second attempt, so a dead
    IPv6 route costs one connect limit, not the whole fetch. Ordered here, not
    connected: a test runner may have no IPv6 at all."""
    assert _interleaved(
        ("2001:db8::1", "2001:db8::2", "2001:db8::3", "192.0.2.1", "192.0.2.2")
    ) == (
        "2001:db8::1",
        "192.0.2.1",
        "2001:db8::2",
        "192.0.2.2",
        "2001:db8::3",
    )
    assert _interleaved(("192.0.2.1", "2001:db8::1", "192.0.2.2")) == (
        "192.0.2.1",
        "2001:db8::1",
        "192.0.2.2",
    )
    assert _interleaved(("192.0.2.1",)) == ("192.0.2.1",)


def test_a_failure_is_logged_without_its_query(caplog: pytest.LogCaptureFixture) -> None:
    """A search API's URL holds its query, and Mojeek's holds the person's key: the log names
    the address that failed, never the URL."""
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    url = f"http://{HOST}:{port}/search?q=secret-question&api_key=sk-the-key"
    with caplog.at_level(logging.WARNING, logger="chickadee.external.fetch.http"):
        got = engine().get(Admitted(url, HOST, port, ("127.0.0.1",), False))
    assert isinstance(got, FetchFailed) and got.why == "connection"
    assert f"127.0.0.1:{port}" in caplog.text
    assert "sk-the-key" not in caplog.text and "secret-question" not in caplog.text
