"""Where a client finds its server, and how it reaches it.

**A Unix socket by default; loopback TCP where none can exist.** An address is the server's
own bind type (`chickadee_serve.bind`'s `Uds` and `Tcp`), so a client and the server it reaches
name a place one way.

**Chosen by probing, not by platform.** `discover` asks the repository's socket first and a
loopback address the caller names second, and takes the first that answers `GET /`. A socket
file nothing answers on is a server that died; it is skipped, never removed: reclaiming it is
the next server's (`bind.claim_socket`).

**The token is the server's file, beside its socket**, whichever bind the server uses
(`server.running`): `token_in(state_dir)` reads it. A server issues new tokens each time it
starts, so a client that may outlive one presents its token through `TokenAuth` over a
source it can ask again: a 401 re-reads the token once and retries, so a stream reconnecting
to a restarted server carries on.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Final

import httpx
from pydantic import ValidationError

from chickadee_serve.bind import SOCKET_NAME, Tcp, Uds
from chickadee_serve.frames import TOKEN_FILE, Hello, TokenClass, read_token
from chickadee_wire.errors import NotServing

__all__ = [
    "PROBE_TIMEOUT_S",
    "Address",
    "TokenAuth",
    "TokenSource",
    "async_transport",
    "base_url",
    "discover",
    "probe",
    "sync_transport",
    "token_in",
]

type Address = Uds | Tcp

type TokenSource = str | Callable[[], str]
"""A token, or how to read the current one (`token_in` over a state directory)."""

PROBE_TIMEOUT_S: Final = 2.0
"""How long `probe` waits for `GET /`. A live server answers it from memory."""

_UDS_HOST: Final = "http://chickadee"
"""The URL a request over a socket carries: its host names nothing, the socket is the place."""


def base_url(address: Address) -> str:
    return address.url if isinstance(address, Tcp) else _UDS_HOST


def sync_transport(address: Address) -> httpx.HTTPTransport:
    if isinstance(address, Uds):
        return httpx.HTTPTransport(uds=str(address.path))
    return httpx.HTTPTransport()


def async_transport(address: Address) -> httpx.AsyncHTTPTransport:
    if isinstance(address, Uds):
        return httpx.AsyncHTTPTransport(uds=str(address.path))
    return httpx.AsyncHTTPTransport()


def probe(address: Address, timeout_s: float = PROBE_TIMEOUT_S) -> Hello | None:
    """The server's `Hello` if one answers at `address`, else None. Needs no token."""
    if isinstance(address, Uds) and not address.path.exists():
        return None
    try:
        with httpx.Client(
            transport=sync_transport(address), base_url=base_url(address), timeout=timeout_s
        ) as http:
            response = http.get("/")
    except httpx.TransportError:
        return None
    if response.status_code != 200:
        return None
    try:
        return Hello.model_validate_json(response.content)
    except ValidationError:
        return None  # something answers there, but not a chickadee server


def discover(state_dir: Path, tcp: Tcp | None = None) -> Address:
    """The first place a server answers: the socket in `state_dir`, then `tcp`.
    `NotServing`, naming each place tried, when neither does."""
    tried: list[Address] = [Uds(state_dir / SOCKET_NAME)]
    if tcp is not None:
        tried.append(tcp)
    for address in tried:
        if probe(address) is not None:
            return address
    places = ", ".join(str(a.path) if isinstance(a, Uds) else base_url(a) for a in tried)
    raise NotServing(f"no server answers at {places}")


def token_in(state_dir: Path, token_class: TokenClass = TokenClass.INTERACTIVE) -> str:
    """The token of `token_class` the server in `state_dir` issued. `FileNotFoundError` when
    no server is running there."""
    return read_token(state_dir / TOKEN_FILE, token_class)


class TokenAuth(httpx.Auth):
    """`Authorization: Bearer <token>` on every request. When the token came from a source
    and the server answers 401, the source is read again and the request sent once more with
    what it returns: the server restarted and issued new tokens. A fixed token, or a second
    401, is the server's answer and is returned as it is."""

    def __init__(self, token: TokenSource) -> None:
        self._source = token if callable(token) else None
        self._token = token() if callable(token) else token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response]:
        request.headers["authorization"] = f"Bearer {self._token}"
        response = yield request
        if response.status_code != 401 or self._source is None:
            return
        try:
            fresh = self._source()
        except OSError:
            return  # no server has written a token yet: the 401 stands
        if fresh == self._token:
            return
        self._token = fresh
        request.headers["authorization"] = f"Bearer {fresh}"
        yield request
