"""The `http` engine: one request, its body as text. A page is
read by GET (`get`, `web_fetch`'s only method); a search engine's API that takes its query as
JSON is sent a POST (`post`), through the same checks.

**It connects to the address the guard checked.** With no proxy for the host, the request goes
to `Admitted.addresses` with the host in its `Host` header, and TLS is verified for the host
it was asked for (`sni_hostname`), so a name that resolves somewhere private a second time
cannot move the connection there. An address it cannot connect to, refused or silent past
`connect_ms`, moves it to the host's next one, the two address families taken in turn
(`_interleaved`, as RFC 8305 orders them), so a dead IPv6 route costs one attempt, not the
fetch. **Behind a proxy** the proxy makes the
connection: the harness has still refused a host its own resolver finds private, and the rest
is the proxy's.

**Both ways trust what the machine trusts** (`_trust`): a CA file or directory the environment
names, else the operating system's own store, as `httpx2` does for the proxy's client and as
the model clients do.

The body is read to `max_bytes` and cut there; the whole fetch, connection to last byte, has
`timeout_ms`. A type that is not text is refused before its body is read.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import ssl
import time
from collections.abc import Mapping
from typing import Final
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx2

from chickadee.context.results import FetchFailed
from chickadee.external.fetch import CONTACT, Fetched, Moved
from chickadee.external.fetch.proxy import proxy_for
from chickadee.external.fetch.text import is_text, page_text
from chickadee.external.urls import Admitted

__all__ = ["HttpEngine"]

_log = logging.getLogger(__name__)

_HEADERS: Final = {
    "User-Agent": f"chickadee-web-fetch/0 ({CONTACT})",
    "Accept": "text/html, text/markdown;q=0.9, text/plain;q=0.9, application/json;q=0.8, "
    "*/*;q=0.1",
}
_REDIRECTS: Final = frozenset({301, 302, 303, 307, 308})


class HttpEngine:
    """`connect_ms` to connect, `timeout_ms` for the whole fetch, `max_bytes` of body."""

    provider: str = "web.fetch.http"

    def __init__(
        self,
        *,
        connect_ms: int,
        timeout_ms: int,
        max_bytes: int,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._connect_s = connect_ms / 1000
        self._timeout_s = timeout_ms / 1000
        self._max_bytes = max_bytes
        self._env = os.environ if env is None else env

    def get(
        self, target: Admitted, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved | FetchFailed:
        """One GET of `target`; `headers` beside the engine's own, for a search source's API
        (its `Accept`, a person's token), never a model's."""
        return self._request("GET", target, headers, None)

    def post(
        self, target: Admitted, body: bytes, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved | FetchFailed:
        """One POST of `body` to `target`: a search engine's API that takes its query as JSON
        (Tavily, Exa, the keyless trial). Never `web_fetch`'s, which reads only by GET."""
        return self._request("POST", target, headers, body)

    def _request(
        self,
        method: str,
        target: Admitted,
        headers: Mapping[str, str] | None,
        body: bytes | None,
    ) -> Fetched | Moved | FetchFailed:
        deadline = time.monotonic() + self._timeout_s
        timeout = httpx2.Timeout(self._timeout_s, connect=self._connect_s)
        sent = dict(headers or {})
        causes: list[str] = []
        if proxy_for(urlsplit(target.url).scheme, target.host, self._env) is not None:
            with httpx2.Client(timeout=timeout, trust_env=True) as client:
                answer = self._send(
                    client, method, target, target.url, sent, body, {}, deadline, causes
                )
            return _said(target, answer, causes)
        answer = FetchFailed(target.url, "connection")
        with httpx2.Client(
            timeout=timeout, trust_env=False, verify=_trust(self._env)
        ) as client:
            for address in _interleaved(target.addresses):
                if time.monotonic() > deadline:
                    answer = FetchFailed(target.url, "timeout")
                    break
                url, host, extensions = _pinned(target, address)
                answer = self._send(
                    client,
                    method,
                    target,
                    url,
                    {**sent, **host},
                    body,
                    extensions,
                    deadline,
                    causes,
                )
                if not (isinstance(answer, FetchFailed) and answer.why == "connection"):
                    break
        return _said(target, answer, causes)

    def _send(
        self,
        client: httpx2.Client,
        method: str,
        target: Admitted,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        extensions: Mapping[str, object],
        deadline: float,
        causes: list[str],
    ) -> Fetched | Moved | FetchFailed:
        """One request; why it did not connect, when it did not, is added to `causes`, by
        address alone: a search API's URL holds its query, and Mojeek's its key."""
        where = urlsplit(url).netloc
        try:
            with client.stream(
                method,
                url,
                headers={**_HEADERS, **headers},
                content=body,
                extensions=dict(extensions),
            ) as response:
                status = response.status_code
                if status in _REDIRECTS and "location" in response.headers:
                    location = urljoin(target.url, response.headers["location"])
                    return Moved(target.url, location, status)
                if status >= 300:
                    return FetchFailed(target.url, "http_status", status)
                kind = response.headers.get("content-type", "")
                if not is_text(kind):
                    return FetchFailed(target.url, "not_text", status)
                body, cut = self._body(response, deadline)
        except httpx2.ConnectTimeout:
            # Never connected: the host's next address may answer (a dead IPv6 route).
            causes.append(f"{where}: no connection within the connect limit")
            return FetchFailed(target.url, "connection")
        except httpx2.TimeoutException:
            return FetchFailed(target.url, "timeout")
        except httpx2.DecodingError:
            # A body its own `Content-Encoding` cannot undo: sent, and not a page.
            return FetchFailed(target.url, "not_text")
        except httpx2.TransportError as exc:
            causes.append(f"{where}: {exc}")
            return FetchFailed(target.url, "connection")
        if body is None:
            return FetchFailed(target.url, "timeout")
        text = page_text(body, kind, target.url)
        if text is None:
            return FetchFailed(target.url, "not_text", status)
        return Fetched(target.url, text, cut)

    def _body(self, response: httpx2.Response, deadline: float) -> tuple[bytes | None, bool]:
        """The body to `max_bytes`, and whether it was cut there; None past the deadline."""
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            if time.monotonic() > deadline:
                return None, False
            chunks.append(chunk)
            size += len(chunk)
            if size > self._max_bytes:
                return b"".join(chunks)[: self._max_bytes], True
        return b"".join(chunks), False


def _said(
    target: Admitted, answer: Fetched | Moved | FetchFailed, causes: list[str]
) -> Fetched | Moved | FetchFailed:
    """`answer`, and, when it is a failed connection, its last cause logged once for the
    person: a certificate the machine does not trust, a refused port. The model is told only
    that the connection failed, which is all it can act on."""
    if isinstance(answer, FetchFailed) and answer.why == "connection" and causes:
        tried = f" ({len(causes)} addresses tried)" if len(causes) > 1 else ""
        _log.warning("web: no connection to %s%s: %s", target.host, tried, causes[-1])
    return answer


def _interleaved(addresses: tuple[str, ...]) -> tuple[str, ...]:
    """`addresses` with their families taken in turn, the first's first, each family in the
    resolver's order: a host listing four IPv6 addresses before its IPv4 ones
    reaches IPv4 on its second attempt, within the fetch's time."""
    if not addresses:
        return addresses
    six = [a for a in addresses if ipaddress.ip_address(a.split("%", 1)[0]).version == 6]
    four = [a for a in addresses if a not in six]
    first, second = (six, four) if addresses[0] in six else (four, six)
    out: list[str] = []
    for i in range(max(len(first), len(second))):
        out += first[i : i + 1] + second[i : i + 1]
    return tuple(out)


def _trust(env: Mapping[str, str]) -> ssl.SSLContext:
    """What a connection's certificate is verified against: a CA file or directory `env`
    names (`SSL_CERT_FILE`, `SSL_CERT_DIR`), else the operating system's own trust store, the
    one `httpx2` gives the proxy's client (`truststore`). A bare `ssl.create_default_context()`
    reads only OpenSSL's default files, which some Pythons ship without (python.org's on
    macOS) and which miss a root a person's network installs in the system store; every
    handshake then fails, and each fetch and search said only that it could not connect."""
    if env.get("SSL_CERT_FILE"):
        return ssl.create_default_context(cafile=env["SSL_CERT_FILE"])
    if env.get("SSL_CERT_DIR"):
        return ssl.create_default_context(capath=env["SSL_CERT_DIR"])
    return httpx2.create_ssl_context(trust_env=False)


def _pinned(target: Admitted, address: str) -> tuple[str, dict[str, str], dict[str, object]]:
    """The request to `address` that `target` names: its URL, its `Host`, its TLS name."""
    parts = urlsplit(target.url)
    shown = f"[{address}]" if ":" in address else address
    url = urlunsplit((parts.scheme, f"{shown}:{target.port}", parts.path, parts.query, ""))
    extensions: dict[str, object] = (
        {"sni_hostname": target.host} if parts.scheme == "https" else {}
    )
    return url, {"Host": parts.netloc}, extensions
