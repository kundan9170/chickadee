"""What `web_search` asks, and how each source is asked it.

A source is a name, the provider its `external_call` line names, and a function from a query
and a count to results. It asks its API through a `Getter`, the one HTTP path every source
shares (`http_getter`): the `http` engine, with the host's addresses checked public as a fetch's
are, so a search reaches nothing a fetch could not. A result is a title, a URL and a snippet,
never a page; reading the page is `web_fetch`'s.

The built-in sources are the official APIs (`sources.py`); the general engines on a person's key
come in part 6, through the same `Source`.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

from chickadee.context.results import FetchFailed, FetchReason, SearchFailed, SearchReason
from chickadee.external.fetch import Fetched, Moved
from chickadee.external.urls import (
    Admitted,
    Resolver,
    addresses_of,
    normalise,
    public_addresses,
)

__all__ = [
    "ApiEngine",
    "Getter",
    "Parsed",
    "Result",
    "Source",
    "http_getter",
    "plain",
    "read_json",
    "render_results",
]


@dataclass(frozen=True)
class Result:
    title: str
    url: str
    snippet: str
    source: str


class Getter(Protocol):
    """An API's URL, the headers to send and, for an API that takes its query as JSON, the
    body to POST, to its answer as text, or why not. The source names itself in the
    failure."""

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes | None = None
    ) -> str | SearchFailed: ...


Parsed = tuple["Result", ...] | SearchFailed
"""What a source's run answers: its results, or why there are none."""


@dataclass(frozen=True)
class Source:
    name: str
    """What `web_search`'s `source` names it by: `github`, `pypi`."""
    provider: str
    """Its name on the `external_call` line: `web.search.github`."""
    run: Callable[[str, int, Getter], tuple[Result, ...] | SearchFailed]
    private_ok: bool = False
    """It may be on the person's own network: their SearXNG, at the URL they named
    (`http_getter`'s `private_ok`). Every other source reaches public hosts only."""


class ApiEngine(Protocol):
    """An engine that sends a source's own headers, and a body by POST: the `http` engine."""

    def get(
        self, target: Admitted, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved | FetchFailed: ...

    def post(
        self, target: Admitted, body: bytes, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved | FetchFailed: ...


_RATE_LIMITED: Final = frozenset({403, 429})
_AS_SEARCH: Final[dict[FetchReason, SearchReason]] = {
    "timeout": "timeout",
    "connection": "connection",
    "http_status": "http_status",
    # An API that answers with something other than text, or that only a browser's robots
    # rules would refuse, answered in a shape it never had.
    "not_text": "bad_answer",
    "robots": "bad_answer",
}


def http_getter(
    engine: ApiEngine,
    source: str,
    *,
    resolve: Resolver | None = None,
    private_ok: bool = False,
) -> Getter:
    """`source`'s `Getter` over the `http` engine. A redirect is not followed: an API that
    moves is an answer it never gave before, so it is `bad_answer`. `private_ok` is for the
    one host a person names for themselves, their own SearXNG (`[web.search] searxng_url`),
    which may well be on their own network; every other source reaches public hosts only."""

    def get(
        url: str, headers: Mapping[str, str], body: bytes | None = None
    ) -> str | SearchFailed:
        normal = normalise(url)
        host = urlsplit(normal).hostname if normal is not None else None
        if normal is None or host is None:
            return SearchFailed(source, "bad_answer")
        port = urlsplit(normal).port or (443 if normal.startswith("https:") else 80)
        check = addresses_of if private_ok else public_addresses
        addresses = check(host, port, resolve)
        if isinstance(addresses, str):
            return SearchFailed(source, "connection")
        target = Admitted(normal, host, port, addresses, False)
        got = (
            engine.get(target, headers) if body is None else engine.post(target, body, headers)
        )
        if isinstance(got, Fetched):
            return got.text
        if not isinstance(got, FetchFailed):
            return SearchFailed(source, "bad_answer", got.status)
        if got.status in _RATE_LIMITED:
            return SearchFailed(source, "rate_limited", got.status)
        return SearchFailed(source, _AS_SEARCH[got.why], got.status)

    return get


def render_results(results: tuple[Result, ...], snippet_chars: int) -> str:
    """Numbered, one result a block: its title, its URL, its snippet cut visibly at
    `snippet_chars`, and where it came from."""
    if not results:
        return "no results"
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        snippet = " ".join(r.snippet.split())
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + " [...]"
        lines = [f"{i}. {' '.join(r.title.split())}", f"   {r.url}"]
        if snippet:
            lines.append(f"   {snippet}")
        lines.append(f"   ({r.source})")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


_TAGS: Final = re.compile(r"<[^>]+>")


def read_json(
    name: str,
    url: str,
    get: Getter,
    headers: Mapping[str, str],
    read: Callable[[Any], Parsed],
    body: bytes | None = None,
) -> Parsed:
    """`url`'s JSON answer read by `read`, the one way every source reads one: an answer that
    is not JSON, or not in the shape `read` expects, is `bad_answer`, never a guess."""
    answer = get(url, headers, body)
    if isinstance(answer, SearchFailed):
        return answer
    try:
        return read(json.loads(answer))
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return SearchFailed(name, "bad_answer")


def plain(text: object) -> str:
    """An API's HTML-flavoured string as plain text: tags dropped, entities read."""
    return html.unescape(_TAGS.sub("", str(text or "")))
