"""The general web engines, and the keyless trial.

`web_search`'s `source: "web"` is the whole web. It is answered by the one general engine the
person configured, on their own key: Tavily (its free plan, 1,000 credits a month per account
with no card, is why it is first), Brave, Exa or Mojeek, each through its official API, or a
SearXNG the person hosts. A key is named by the environment variable that holds it, as a rung's
is (`api_key_env`), and read as the session is composed; it is never journaled, never sent
anywhere but its engine, and withheld from every command (`EngineConfig.endpoint_keys`).

With no engine configured, `web` is **Exa's hosted MCP endpoint, keyless, as a trial**. Its free
tier is counted per client address and may change or close; when it refuses (HTTP 429) the
answer is `SEARCH_FAILED{reason: trial_spent}`, naming how to add a key, and the person is told
once a session (`notify`).

Each engine is a `Source`, so a result is the same title, URL and snippet as an official
source's, admitted for a fetch the same way.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from urllib.parse import urlencode

from chickadee.context.results import SearchFailed
from chickadee.errors import ConfigError
from chickadee.external.search import Getter, Parsed, Result, Source, plain, read_json
from chickadee.external.search.sources import official

if TYPE_CHECKING:
    from chickadee.config import WebSearchPolicy

__all__ = ["ENGINES", "TRIAL", "WEB", "EngineName", "general", "session_sources", "trial"]

WEB: Final = "web"
"""The source name `web_search` gives the whole web, whichever engine answers it."""

EngineName = Literal["tavily", "brave", "exa", "mojeek", "searxng"]
ENGINES: Final[tuple[EngineName, ...]] = ("tavily", "brave", "exa", "mojeek", "searxng")

TRIAL: Final = "https://mcp.exa.ai/mcp"
"""Exa's hosted MCP endpoint, which answers `web_search_exa` with no key."""

_JSON: Final = {"Accept": "application/json", "Content-Type": "application/json"}

Run = Callable[[str, int, Getter], Parsed]


def general(name: EngineName, *, key: str | None, base_url: str | None = None) -> Source:
    """The general engine `name`, on the person's `key`; SearXNG at their `base_url`, its key
    optional."""
    runs: dict[EngineName, Callable[[], Run]] = {
        "tavily": lambda: _tavily(_needed(name, key)),
        "brave": lambda: _brave(_needed(name, key)),
        "exa": lambda: _exa(_needed(name, key)),
        "mojeek": lambda: _mojeek(_needed(name, key)),
        "searxng": lambda: _searxng(_needed_url(base_url), key),
    }
    return Source(WEB, f"web.search.{name}", runs[name](), private_ok=name == "searxng")


def trial(notify: Callable[[str], None]) -> Source:
    """Exa's keyless endpoint. `notify` tells the person, once, when its free tier refuses."""
    told = False

    def run(query: str, limit: int, get: Getter) -> Parsed:
        nonlocal told
        got = _exa_trial(query, limit, get)
        if isinstance(got, SearchFailed) and got.why == "rate_limited":
            if not told:
                told = True
                notify(
                    "web search: the keyless Exa trial is refusing searches from this address "
                    "(its free tier is per address). Give the researcher a search engine of "
                    'your own: [web.search] engine = "tavily", api_key_env = "TAVILY_API_KEY" '
                    "(Tavily's free plan is 1,000 credits a month, no card)."
                )
            return SearchFailed(WEB, "trial_spent", got.status)
        return got

    return Source(WEB, "web.search.exa_trial", run)


def session_sources(
    search: WebSearchPolicy,
    *,
    notify: Callable[[str], None],
    env: Mapping[str, str] | None = None,
) -> tuple[Source, ...]:
    """`web_search`'s sources for a session: the official seven, GitHub's with the person's
    token, and `web`, their general engine on their own key or, with none, the keyless trial,
    whose refusal `notify` tells the person once. A key is read from the variable `[web.search]`
    names, now, and kept nowhere but the source that sends it: an unset one is a
    `ConfigError` naming it, so the session's composition says so."""
    found = os.environ if env is None else env

    def read(variable: str | None) -> str | None:
        if variable is None:
            return None
        value = found.get(variable)
        if not value:
            raise ConfigError(
                f"[web.search] names {variable!r} for a key, and that variable is unset"
            )
        return value

    token = read(search.github_token_env)
    if search.engine is None:
        web = trial(notify)
    else:
        web = general(search.engine, key=read(search.api_key_env), base_url=search.searxng_url)
    return (*official(github_token=token), web)


def _needed(name: str, key: str | None) -> str:
    if not key:
        raise ValueError(f"the {name} engine needs its key, and none was given")
    return key


def _needed_url(base_url: str | None) -> str:
    if not base_url:
        raise ValueError("the searxng engine needs its base URL, and none was given")
    return base_url.rstrip("/")


def _tavily(key: str) -> Run:
    def run(query: str, limit: int, get: Getter) -> Parsed:
        body = {"query": query, "max_results": limit, "search_depth": "basic"}

        def read(data: Any) -> Parsed:
            return tuple(
                Result(str(r["title"]), str(r["url"]), str(r.get("content") or ""), "tavily")
                for r in cast("list[dict[str, Any]]", data["results"])[:limit]
            )

        headers = {**_JSON, "Authorization": f"Bearer {key}"}
        url = "https://api.tavily.com/search"
        return read_json(WEB, url, get, headers, read, json.dumps(body).encode())

    return run


def _brave(key: str) -> Run:
    def run(query: str, limit: int, get: Getter) -> Parsed:
        url = "https://api.search.brave.com/res/v1/web/search?" + urlencode(
            {"q": query, "count": limit}
        )

        def read(data: Any) -> Parsed:
            return tuple(
                Result(plain(r["title"]), str(r["url"]), plain(r.get("description")), "brave")
                for r in cast("list[dict[str, Any]]", data["web"]["results"])[:limit]
            )

        headers = {"Accept": "application/json", "X-Subscription-Token": key}
        return read_json(WEB, url, get, headers, read)

    return run


def _exa(key: str) -> Run:
    def run(query: str, limit: int, get: Getter) -> Parsed:
        body = {
            "query": query,
            "numResults": limit,
            "type": "auto",
            "contents": {"highlights": {"numSentences": 3}},
        }

        def read(data: Any) -> Parsed:
            out: list[Result] = []
            for r in cast("list[dict[str, Any]]", data["results"])[:limit]:
                said = r.get("highlights") or [r.get("text") or ""]
                snippet = " ".join(str(h) for h in cast("list[object]", said))
                out.append(
                    Result(str(r.get("title") or r["url"]), str(r["url"]), snippet, "exa")
                )
            return tuple(out)

        headers = {**_JSON, "x-api-key": key}
        url = "https://api.exa.ai/search"
        return read_json(WEB, url, get, headers, read, json.dumps(body).encode())

    return run


def _mojeek(key: str) -> Run:
    def run(query: str, limit: int, get: Getter) -> Parsed:
        # Mojeek takes its key in the query string; the record holds the call's arguments,
        # never this URL.
        url = "https://api.mojeek.com/search?" + urlencode(
            {"q": query, "fmt": "json", "t": limit, "api_key": key}
        )

        def read(data: Any) -> Parsed:
            return tuple(
                Result(plain(r["title"]), str(r["url"]), plain(r.get("desc")), "mojeek")
                for r in cast("list[dict[str, Any]]", data["response"]["results"])[:limit]
            )

        return read_json(WEB, url, get, {"Accept": "application/json"}, read)

    return run


def _searxng(base_url: str, key: str | None) -> Run:
    def run(query: str, limit: int, get: Getter) -> Parsed:
        url = f"{base_url}/search?" + urlencode({"q": query, "format": "json"})

        def read(data: Any) -> Parsed:
            return tuple(
                Result(plain(r["title"]), str(r["url"]), plain(r.get("content")), "searxng")
                for r in cast("list[dict[str, Any]]", data["results"])[:limit]
            )

        headers = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return read_json(WEB, url, get, headers, read)

    return run


def _exa_trial(query: str, limit: int, get: Getter) -> Parsed:
    """One JSON-RPC `tools/call` of `web_search_exa`, the endpoint's one stateless request;
    its answer is a server-sent event whose text holds the results, `---` between them."""
    call = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {"query": query, "numResults": limit, "objective": query},
        },
    }
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    answer = get(TRIAL, headers, json.dumps(call).encode())
    if isinstance(answer, SearchFailed):
        return answer
    try:
        data = _rpc(answer)
        result = cast("dict[str, Any]", data["result"])
        if result.get("isError"):
            return SearchFailed(WEB, "bad_answer")
        text = "\n".join(
            str(part.get("text") or "")
            for part in cast("list[dict[str, Any]]", result["content"])
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return SearchFailed(WEB, "bad_answer")
    return tuple(r for r in map(_trial_result, text.split("\n---\n")) if r is not None)[:limit]


def _rpc(answer: str) -> dict[str, Any]:
    """The JSON-RPC message in `answer`: a server-sent event's `data:` line, or plain JSON."""
    for line in answer.splitlines():
        if line.startswith("data:"):
            return cast("dict[str, Any]", json.loads(line[len("data:") :].strip()))
    return cast("dict[str, Any]", json.loads(answer))


_FIELDS: Final = ("Title:", "URL:", "Published:", "Author:", "Highlights:")


def _trial_result(block: str) -> Result | None:
    """One result block, `Title:`, `URL:` and `Highlights:` by name; None without a URL."""
    title = url = ""
    highlights: list[str] = []
    reading = False
    for line in block.strip().splitlines():
        if line.startswith("Title:"):
            title = line[len("Title:") :].strip()
        elif line.startswith("URL:"):
            url = line[len("URL:") :].strip()
        elif line.startswith("Highlights:"):
            reading = True
        elif reading and not line.startswith(_FIELDS):
            highlights.append(line)
    if not url:
        return None
    return Result(title or url, url, " ".join(" ".join(highlights).split()), "exa_trial")
