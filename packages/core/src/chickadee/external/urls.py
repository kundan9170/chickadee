"""The URL guards of `web_fetch`, in the order they are asked. Each refuses with a typed
`UrlRefused` before anything is sent.

1. **It appeared.** `SeenUrls` holds every URL the round was given or found: in its opening, in
   a search result, on a page it fetched. A URL the model wrote itself is how a leak is sent
   (`evil.com/?q=<what you read>`), so only one that appeared, verbatim once normalised, is
   fetched. Anthropic's fetch tool refuses the same (`url_not_in_prior_context`).
2. **Or it is a guess on a documentation domain**, with no query string: `DOCS_DOMAINS`, hosts
   whose pages their vendor writes, plus the person's own. A host that serves pages its users
   write (`github.com`, `*.github.io`, `readthedocs.io`, `docs.rs`, a paste site) is never one:
   an attacker could write the page a guess lands on.
3. **It is well formed:** within `web.url_max_chars`, carrying no credential. Only an `http` or
   `https` URL can have appeared (rule 1), so no other scheme gets this far.
4. **It is public:** every address its host resolves to is a global unicast address. The `http`
   engine connects to the address checked here (`public_addresses`), so a name that resolves
   differently a second time cannot move the connection.

Normalising (`normalise`) lower-cases the scheme and host, drops a default port and the
fragment, and gives an empty path `/`; the query is kept, since it is part of what is fetched.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from chickadee.context.results import UrlReason, UrlRefused
from chickadee.secrets.redact import redact_text

__all__ = [
    "DOCS_DOMAINS",
    "Admitted",
    "Resolver",
    "SeenUrls",
    "addresses_of",
    "check_url",
    "find_urls",
    "normalise",
    "public_addresses",
]

DOCS_DOMAINS: Final = frozenset(
    {
        "docs.python.org",
        "peps.python.org",
        "packaging.python.org",
        "developer.mozilla.org",
        "nodejs.org",
        "docs.npmjs.com",
        "www.typescriptlang.org",
        "doc.rust-lang.org",
        "go.dev",
        "docs.oracle.com",
        "learn.microsoft.com",
        "kotlinlang.org",
        "docs.swift.org",
        "en.cppreference.com",
        "docs.github.com",
        "git-scm.com",
        "docs.docker.com",
        "kubernetes.io",
        "docs.pydantic.dev",
        "fastapi.tiangolo.com",
        "docs.pytest.org",
        "numpy.org",
        "pandas.pydata.org",
        "docs.scipy.org",
        "pytorch.org",
        "react.dev",
        "vuejs.org",
        "angular.dev",
        "nextjs.org",
        "docs.astral.sh",
        "www.postgresql.org",
        "dev.mysql.com",
        "www.sqlite.org",
        "redis.io",
        "nginx.org",
    }
)
"""Hosts a guessed URL may be fetched from (decision 5, rule 2): each a vendor's own
documentation, none serving pages its users write. The person adds their own in
`[web.fetch] docs_domains`; a repository adds none."""

_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_TRAILING = ".,;:!?'\""
_DEFAULT_PORTS: Final = {"http": 80, "https": 443}

Resolver = Callable[[str, int], Sequence[str]]
"""A host and port, to the addresses it resolves to: `socket.getaddrinfo`'s, or a test's."""


def normalise(url: str) -> str | None:
    """`url` in the one form `SeenUrls` compares, or None when it is not an http(s) URL."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    shown = f"[{host}]" if ":" in host else host  # an IPv6 literal keeps its brackets
    netloc = shown if port in (None, _DEFAULT_PORTS[scheme]) else f"{shown}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def find_urls(text: str) -> tuple[str, ...]:
    """The http(s) URLs in `text`, as written, without the punctuation that ends a sentence or
    a closing bracket a link was wrapped in."""
    found: list[str] = []
    for match in _URL.finditer(text):
        url = match.group(0).rstrip(_TRAILING)
        while url.endswith((")", "]")) and url.count(url[-1]) > url.count(
            "(" if url[-1] == ")" else "["
        ):
            url = url[:-1].rstrip(_TRAILING)
        found.append(url)
    return tuple(found)


class SeenUrls:
    """The URLs one round may fetch (rule 1). Not thread-safe, and not shared: a round's tools
    run on its own dispatcher's thread."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def admit(self, text: str) -> None:
        """Every URL in `text`: the round's opening, a result, a page."""
        for url in find_urls(text):
            self.admit_url(url)

    def admit_url(self, url: str) -> None:
        normal = normalise(url)
        if normal is not None:
            self._seen.add(normal)

    def __contains__(self, url: object) -> bool:
        return isinstance(url, str) and normalise(url) in self._seen


@dataclass(frozen=True)
class Admitted:
    """A URL every guard passed: its normal form, its host and port, and the addresses checked,
    which the `http` engine connects to."""

    url: str
    host: str
    port: int
    addresses: tuple[str, ...]
    guessed: bool
    """Fetched on rule 2, a documentation domain, not because it appeared."""


def check_url(
    url: str,
    *,
    seen: SeenUrls,
    docs: Iterable[str] = DOCS_DOMAINS,
    max_chars: int,
    resolve: Resolver | None = None,
) -> Admitted | UrlRefused:
    """`url` through rules 1 to 4, in order: the first that fails decides."""
    normal = normalise(url)
    parts = urlsplit(normal) if normal is not None else None
    host = parts.hostname if parts is not None else None
    if normal is None or parts is None or host is None:
        # Not an http(s) URL with a host: nothing the round read could have shown it, and
        # `SeenUrls` only ever holds http(s) ones.
        return UrlRefused(url, "not_seen")
    appeared = url in seen
    guessed = not appeared and host in frozenset(docs) and not parts.query
    if not appeared and not guessed:
        return UrlRefused(url, "not_seen")
    if len(url) > max_chars:
        return UrlRefused(url, "too_long")
    if parts.username is not None or parts.password is not None or redact_text(url)[1]:
        return UrlRefused(url, "credential")
    port = parts.port or _DEFAULT_PORTS[parts.scheme]
    addresses = public_addresses(host, port, resolve)
    if isinstance(addresses, str):
        return UrlRefused(url, addresses)
    return Admitted(normal, host, port, addresses, guessed)


def public_addresses(
    host: str, port: int, resolve: Resolver | None = None
) -> tuple[str, ...] | UrlReason:
    """Every address `host` resolves to, when each is a public unicast one (rule 4); else why
    not. One private address refuses the host: which one a connection would take is not ours
    to know."""
    found = addresses_of(host, port, resolve)
    if isinstance(found, str):
        return found
    for raw in found:
        address = ipaddress.ip_address(raw.split("%", 1)[0])
        inner = _embedded_ipv4(address) if isinstance(address, ipaddress.IPv6Address) else None
        for one in (address, inner) if inner is not None else (address,):
            if not one.is_global or one.is_multicast:
                return "private_address"
    return found


_NAT64: Final = ipaddress.IPv6Network("64:ff9b::/96")
_COMPATIBLE: Final = ipaddress.IPv6Network("::/96")


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 one carries, which a connection to it may reach: mapped
    (`::ffff:a.b.c.d`), NAT64's well-known prefix (`64:ff9b::a.b.c.d`) or the old compatible
    form (`::a.b.c.d`). Python counts the last two as global whatever they carry, so a private
    address inside them would pass `is_global` alone."""
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64 or address in _COMPATIBLE:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def addresses_of(
    host: str, port: int, resolve: Resolver | None = None
) -> tuple[str, ...] | UrlReason:
    """Every address `host` resolves to, once each, or `unresolvable`: whatever they are.
    `public_addresses` is the check a model's URL meets; only a host the person named for
    themselves (their own SearXNG) is reached by this alone."""
    try:
        found = tuple(dict.fromkeys((resolve or _getaddrinfo)(host, port)))
    except OSError:
        return "unresolvable"
    return found or "unresolvable"


def _getaddrinfo(host: str, port: int) -> tuple[str, ...]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return tuple(str(info[4][0]) for info in infos)
