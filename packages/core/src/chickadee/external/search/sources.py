"""The built-in search sources: each an official, free API.

GitHub's issues and pull requests, Stack Overflow's questions, PyPI's packages, npm's,
crates.io's, Wikipedia's articles and arXiv's papers. None needs a key; GitHub's limit without
one is low, so the person's token is sent when they give one (`official(github_token=...)`),
and only to GitHub. Each answer is read by its documented shape; one in any other shape is
`bad_answer`, never a guess.

PyPI has no search API, only a package's own page, so a PyPI search is a lookup of the query as
a package name: one result or none.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from typing import Any, Final, cast
from urllib.parse import quote, urlencode

from chickadee.context.results import SearchFailed
from chickadee.external.search import Getter, Parsed, Result, Source, plain, read_json

__all__ = ["official"]

_ATOM: Final = "{http://www.w3.org/2005/Atom}"
_JSON: Final = {"Accept": "application/json"}


def official(*, github_token: str | None = None) -> tuple[Source, ...]:
    """The seven, GitHub's with the person's token when there is one."""
    return (
        Source("github", "web.search.github", _github(github_token)),
        Source("stackexchange", "web.search.stackexchange", _stackexchange),
        Source("pypi", "web.search.pypi", _pypi),
        Source("npm", "web.search.npm", _npm),
        Source("crates", "web.search.crates", _crates),
        Source("wikipedia", "web.search.wikipedia", _wikipedia),
        Source("arxiv", "web.search.arxiv", _arxiv),
    )


def _github(token: str | None) -> Callable[[str, int, Getter], Parsed]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def run(query: str, limit: int, get: Getter) -> Parsed:
        url = "https://api.github.com/search/issues?" + urlencode(
            {"q": query, "per_page": limit}
        )

        def read(data: Any) -> Parsed:
            return tuple(
                Result(
                    str(item["title"]),
                    str(item["html_url"]),
                    f"{item.get('state', '')} · {item.get('comments', 0)} comments · "
                    + plain(item.get("body"))[:400],
                    "github",
                )
                for item in cast("list[dict[str, Any]]", data["items"])[:limit]
            )

        return read_json("github", url, get, headers, read)

    return run


def _stackexchange(query: str, limit: int, get: Getter) -> Parsed:
    url = "https://api.stackexchange.com/2.3/search/advanced?" + urlencode(
        {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": "stackoverflow",
            "pagesize": limit,
        }
    )

    def read(data: Any) -> Parsed:
        return tuple(
            Result(
                plain(item["title"]),
                str(item["link"]),
                f"score {item.get('score', 0)} · {item.get('answer_count', 0)} answers · "
                + ("answered" if item.get("is_answered") else "not answered")
                + " · "
                + ", ".join(str(t) for t in item.get("tags", ())),
                "stackexchange",
            )
            for item in cast("list[dict[str, Any]]", data["items"])[:limit]
        )

    return read_json("stackexchange", url, get, _JSON, read)


def _pypi(query: str, limit: int, get: Getter) -> Parsed:
    del limit
    name = query.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        return ()
    body = get(f"https://pypi.org/pypi/{quote(name)}/json", _JSON)
    if isinstance(body, SearchFailed):
        # No package of that name is no result, not a failure of the source.
        return () if body.status == 404 else body

    try:
        info = cast("dict[str, Any]", json.loads(body)["info"])
        return (
            Result(
                f"{info['name']} {info.get('version', '')}".strip(),
                str(info.get("project_url") or f"https://pypi.org/project/{quote(name)}/"),
                str(info.get("summary") or ""),
                "pypi",
            ),
        )
    except (ValueError, KeyError, TypeError):
        return SearchFailed("pypi", "bad_answer")


def _npm(query: str, limit: int, get: Getter) -> Parsed:
    url = "https://registry.npmjs.org/-/v1/search?" + urlencode({"text": query, "size": limit})

    def read(data: Any) -> Parsed:
        out: list[Result] = []
        for item in cast("list[dict[str, Any]]", data["objects"])[:limit]:
            package = cast("dict[str, Any]", item["package"])
            name = str(package["name"])
            links = cast("dict[str, Any]", package.get("links") or {})
            out.append(
                Result(
                    f"{name} {package.get('version', '')}".strip(),
                    str(links.get("npm") or f"https://www.npmjs.com/package/{quote(name)}"),
                    str(package.get("description") or ""),
                    "npm",
                )
            )
        return tuple(out)

    return read_json("npm", url, get, _JSON, read)


def _crates(query: str, limit: int, get: Getter) -> Parsed:
    url = "https://crates.io/api/v1/crates?" + urlencode({"q": query, "per_page": limit})

    def read(data: Any) -> Parsed:
        return tuple(
            Result(
                f"{item['name']} {item.get('max_version', '')}".strip(),
                f"https://crates.io/crates/{quote(str(item['name']))}",
                str(item.get("description") or ""),
                "crates",
            )
            for item in cast("list[dict[str, Any]]", data["crates"])[:limit]
        )

    return read_json("crates", url, get, _JSON, read)


def _wikipedia(query: str, limit: int, get: Getter) -> Parsed:
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": limit,
        }
    )

    def read(data: Any) -> Parsed:
        return tuple(
            Result(
                str(item["title"]),
                "https://en.wikipedia.org/wiki/" + quote(str(item["title"]).replace(" ", "_")),
                plain(item.get("snippet")),
                "wikipedia",
            )
            for item in cast("list[dict[str, Any]]", data["query"]["search"])[:limit]
        )

    return read_json("wikipedia", url, get, _JSON, read)


def _arxiv(query: str, limit: int, get: Getter) -> Parsed:
    url = "https://export.arxiv.org/api/query?" + urlencode(
        {"search_query": f"all:{query}", "max_results": limit}
    )
    body = get(url, {"Accept": "application/atom+xml"})
    if isinstance(body, SearchFailed):
        return body
    if "<!DOCTYPE" in body or "<!ENTITY" in body:
        # The feed declares none; one that does could expand itself without bound.
        return SearchFailed("arxiv", "bad_answer")
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return SearchFailed("arxiv", "bad_answer")
    out: list[Result] = []
    for entry in root.findall(f"{_ATOM}entry")[:limit]:
        link = (entry.findtext(f"{_ATOM}id") or "").strip()
        if not link:
            continue
        out.append(
            Result(
                (entry.findtext(f"{_ATOM}title") or "").strip(),
                link.replace("http://", "https://", 1),
                (entry.findtext(f"{_ATOM}summary") or "").strip(),
                "arxiv",
            )
        )
    return tuple(out)
