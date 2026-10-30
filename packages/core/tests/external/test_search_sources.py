"""The official search sources (`external/search/sources.py`): each asks its
API as documented and reads its answer by its documented shape; an answer in any other shape is
`bad_answer`, never a guess. The answers here are trimmed from each API's real ones."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from chickadee.context.results import FetchFailed, SearchFailed
from chickadee.external.fetch import Fetched, Moved
from chickadee.external.search import Getter, Result, http_getter, render_results
from chickadee.external.search.sources import official
from chickadee.external.urls import Admitted

PUBLIC = "93.184.216.34"


class Canned:
    """A getter answering every URL with `body`, keeping what it was asked."""

    def __init__(self, body: str | SearchFailed) -> None:
        self.body = body
        self.asked: list[tuple[str, dict[str, str]]] = []
        self.sent: list[bytes | None] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes | None = None
    ) -> str | SearchFailed:
        self.asked.append((url, dict(headers)))
        self.sent.append(body)
        return self.body


def run(
    name: str, body: str | SearchFailed, query: str = "q", *, token: str | None = None
) -> tuple[tuple[Result, ...] | SearchFailed, Canned]:
    get = Canned(body)
    (source,) = [s for s in official(github_token=token) if s.name == name]
    return source.run(query, 2, get), get


GITHUB = {
    "items": [
        {
            "title": "SNI is wrong behind a proxy",
            "html_url": "https://github.com/encode/httpx/issues/1",
            "state": "closed",
            "comments": 3,
            "body": "<p>It sends the <b>IP</b>.</p>",
        }
    ]
}
STACK = {
    "items": [
        {
            "title": "How do I use a proxy with &quot;httpx&quot;?",
            "link": "https://stackoverflow.com/questions/1/how",
            "score": 12,
            "answer_count": 2,
            "is_answered": True,
            "tags": ["python", "httpx"],
        }
    ]
}
PYPI = {
    "info": {
        "name": "httpx",
        "version": "0.28.1",
        "summary": "The next generation HTTP client.",
        "project_url": "https://pypi.org/project/httpx/",
    }
}
NPM = {
    "objects": [
        {
            "package": {
                "name": "lodash",
                "version": "4.18.1",
                "description": "Lodash modular utilities.",
                "links": {"npm": "https://www.npmjs.com/package/lodash"},
            }
        }
    ]
}
CRATES = {
    "crates": [
        {"name": "serde", "max_version": "1.0.229", "description": "A serialization framework"}
    ]
}
WIKI = {
    "query": {
        "search": [
            {
                "title": "Proxy server",
                "snippet": 'a <span class="searchmatch">proxy</span> server',
            }
        ]
    }
}
ARXIV = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2201.00978v1</id>
    <title>PyramidTNT: Improved Transformer-in-Transformer Baselines</title>
    <summary>  We build on TNT.  </summary>
  </entry>
</feed>"""


@pytest.mark.parametrize(
    ("name", "body", "result", "asked"),
    [
        (
            "github",
            json.dumps(GITHUB),
            Result(
                "SNI is wrong behind a proxy",
                "https://github.com/encode/httpx/issues/1",
                "closed · 3 comments · It sends the IP.",
                "github",
            ),
            "https://api.github.com/search/issues?q=q&per_page=2",
        ),
        (
            "stackexchange",
            json.dumps(STACK),
            Result(
                'How do I use a proxy with "httpx"?',
                "https://stackoverflow.com/questions/1/how",
                "score 12 · 2 answers · answered · python, httpx",
                "stackexchange",
            ),
            "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q=q"
            "&site=stackoverflow&pagesize=2",
        ),
        (
            "pypi",
            json.dumps(PYPI),
            Result(
                "httpx 0.28.1",
                "https://pypi.org/project/httpx/",
                "The next generation HTTP client.",
                "pypi",
            ),
            "https://pypi.org/pypi/q/json",
        ),
        (
            "npm",
            json.dumps(NPM),
            Result(
                "lodash 4.18.1",
                "https://www.npmjs.com/package/lodash",
                "Lodash modular utilities.",
                "npm",
            ),
            "https://registry.npmjs.org/-/v1/search?text=q&size=2",
        ),
        (
            "crates",
            json.dumps(CRATES),
            Result(
                "serde 1.0.229",
                "https://crates.io/crates/serde",
                "A serialization framework",
                "crates",
            ),
            "https://crates.io/api/v1/crates?q=q&per_page=2",
        ),
        (
            "wikipedia",
            json.dumps(WIKI),
            Result(
                "Proxy server",
                "https://en.wikipedia.org/wiki/Proxy_server",
                "a proxy server",
                "wikipedia",
            ),
            "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=q"
            "&format=json&srlimit=2",
        ),
        (
            "arxiv",
            ARXIV,
            Result(
                "PyramidTNT: Improved Transformer-in-Transformer Baselines",
                "https://arxiv.org/abs/2201.00978v1",
                "We build on TNT.",
                "arxiv",
            ),
            "https://export.arxiv.org/api/query?search_query=all%3Aq&max_results=2",
        ),
    ],
)
def test_each_source_asks_its_api_and_reads_its_answer(
    name: str, body: str, result: Result, asked: str
) -> None:
    got, get = run(name, body)
    assert got == (result,)
    assert [url for url, _ in get.asked] == [asked]


@pytest.mark.parametrize("name", ["github", "stackexchange", "npm", "crates", "wikipedia"])
def test_an_answer_in_another_shape_is_a_bad_answer(name: str) -> None:
    for body in ("not json", json.dumps({"something": "else"}), json.dumps([1, 2])):
        got, _ = run(name, body)
        assert got == SearchFailed(name, "bad_answer"), body


def test_a_feed_that_declares_entities_is_refused_unparsed() -> None:
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><feed>&a;</feed>'
    assert run("arxiv", bomb)[0] == SearchFailed("arxiv", "bad_answer")


def test_github_sends_the_person_s_token_only_to_github() -> None:
    _, get = run("github", json.dumps(GITHUB), token="ghp_example")
    assert get.asked[0][1]["Authorization"] == "Bearer ghp_example"
    _, plain = run("github", json.dumps(GITHUB))
    assert "Authorization" not in plain.asked[0][1]
    _, stack = run("stackexchange", json.dumps(STACK), token="ghp_example")
    assert "Authorization" not in stack.asked[0][1]


def test_pypi_looks_a_name_up_and_no_such_package_is_no_result() -> None:
    assert run("pypi", SearchFailed("pypi", "http_status", 404))[0] == ()
    assert run("pypi", SearchFailed("pypi", "timeout"))[0] == SearchFailed("pypi", "timeout")
    got, get = run("pypi", json.dumps(PYPI), query="how do I parse json")
    assert got == () and not get.asked


def test_a_source_s_failure_is_its_answer() -> None:
    failed = SearchFailed("npm", "rate_limited", 429)
    assert run("npm", failed)[0] == failed


class ApiDouble:
    def __init__(self, answer: Fetched | Moved | FetchFailed) -> None:
        self.answer = answer
        self.sent: list[tuple[Admitted, dict[str, str]]] = []
        self.posted: list[bytes] = []

    def get(
        self, target: Admitted, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved | FetchFailed:
        self.sent.append((target, dict(headers or {})))
        return self.answer

    def post(
        self, target: Admitted, body: bytes, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved | FetchFailed:
        self.posted.append(body)
        return self.get(target, headers)


def getter(answer: Fetched | Moved | FetchFailed, *addresses: str) -> tuple[Getter, ApiDouble]:
    engine = ApiDouble(answer)
    resolved = addresses or (PUBLIC,)
    return http_getter(engine, "npm", resolve=lambda host, port: resolved), engine


def test_the_http_getter_asks_the_checked_address_with_the_source_s_headers() -> None:
    url = "https://registry.npmjs.org/-/v1/search?text=q"
    get, engine = getter(Fetched(url, "{}", False))
    assert get(url, {"Accept": "application/json"}) == "{}"
    ((target, headers),) = engine.sent
    assert (target.host, target.port, target.addresses) == (
        "registry.npmjs.org",
        443,
        (PUBLIC,),
    )
    assert headers == {"Accept": "application/json"}


@pytest.mark.parametrize(
    ("answer", "failed"),
    [
        (FetchFailed("u", "http_status", 429), SearchFailed("npm", "rate_limited", 429)),
        (FetchFailed("u", "http_status", 403), SearchFailed("npm", "rate_limited", 403)),
        (FetchFailed("u", "http_status", 500), SearchFailed("npm", "http_status", 500)),
        (FetchFailed("u", "timeout"), SearchFailed("npm", "timeout")),
        (FetchFailed("u", "not_text", 200), SearchFailed("npm", "bad_answer", 200)),
        (Moved("u", "https://elsewhere.example/", 301), SearchFailed("npm", "bad_answer", 301)),
    ],
)
def test_the_http_getter_names_each_failure_as_the_source_s(
    answer: Fetched | Moved | FetchFailed, failed: SearchFailed
) -> None:
    get, _ = getter(answer)
    assert get("https://registry.npmjs.org/", {}) == failed


def test_the_http_getter_reaches_no_private_address() -> None:
    get, engine = getter(Fetched("u", "{}", False), "10.0.0.1")
    assert get("https://registry.npmjs.org/", {}) == SearchFailed("npm", "connection")
    assert not engine.sent


def test_results_render_numbered_with_each_snippet_cut_visibly() -> None:
    results = (
        Result("A  title", "https://a.example/", "x" * 50, "github"),
        Result("B", "https://b.example/", "", "pypi"),
    )
    assert render_results(results, 10) == (
        "1. A title\n   https://a.example/\n   xxxxxxxxxx [...]\n   (github)\n\n"
        "2. B\n   https://b.example/\n   (pypi)"
    )
    assert render_results((), 10) == "no results"
