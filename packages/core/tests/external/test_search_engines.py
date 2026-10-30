"""The general engines and the keyless trial (`external/search/engines.py`):
each asked as its API documents, on the person's key and only theirs; each answer read by its
documented shape; the trial's refusal named for the model and said to the person once; and the
`[web.search]` table that chooses them."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest
from engines import raw_config
from pydantic import ValidationError

from chickadee.config import EngineConfig, WebSearchPolicy
from chickadee.context.results import SearchFailed
from chickadee.external.fetch import Fetched, Moved
from chickadee.external.search import Getter, Result, http_getter
from chickadee.external.search.engines import (
    TRIAL,
    WEB,
    EngineName,
    general,
    session_sources,
    trial,
)
from chickadee.external.urls import Admitted, Resolver


class Answers:
    def __init__(self, answer: str | SearchFailed) -> None:
        self.answer = answer
        self.asked: list[tuple[str, dict[str, str], object]] = []

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes | None = None
    ) -> str | SearchFailed:
        self.asked.append((url, dict(headers), None if body is None else json.loads(body)))
        return self.answer


def ask(
    name: EngineName, answer: object, *, key: str | None = "k-1", base: str | None = None
) -> tuple[tuple[Result, ...] | SearchFailed, tuple[str, dict[str, str], object]]:
    get = Answers(answer if isinstance(answer, str | SearchFailed) else json.dumps(answer))
    got = general(name, key=key, base_url=base).run("proxy error", 2, get)
    return got, get.asked[0]


@pytest.mark.parametrize(
    ("name", "answer", "url", "auth", "body", "result"),
    [
        (
            "tavily",
            {"results": [{"title": "T", "url": "https://t.example/1", "content": "said"}]},
            "https://api.tavily.com/search",
            ("Authorization", "Bearer k-1"),
            {"query": "proxy error", "max_results": 2, "search_depth": "basic"},
            Result("T", "https://t.example/1", "said", "tavily"),
        ),
        (
            "brave",
            {
                "web": {
                    "results": [
                        {"title": "<b>B</b>", "url": "https://b.example/", "description": "d"}
                    ]
                }
            },
            "https://api.search.brave.com/res/v1/web/search?q=proxy+error&count=2",
            ("X-Subscription-Token", "k-1"),
            None,
            Result("B", "https://b.example/", "d", "brave"),
        ),
        (
            "exa",
            {
                "results": [
                    {"title": "E", "url": "https://e.example/", "highlights": ["a", "b"]}
                ]
            },
            "https://api.exa.ai/search",
            ("x-api-key", "k-1"),
            {
                "query": "proxy error",
                "numResults": 2,
                "type": "auto",
                "contents": {"highlights": {"numSentences": 3}},
            },
            Result("E", "https://e.example/", "a b", "exa"),
        ),
        (
            "mojeek",
            {
                "response": {
                    "results": [{"title": "M", "url": "https://m.example/", "desc": "x"}]
                }
            },
            "https://api.mojeek.com/search?q=proxy+error&fmt=json&t=2&api_key=k-1",
            ("Accept", "application/json"),
            None,
            Result("M", "https://m.example/", "x", "mojeek"),
        ),
    ],
)
def test_each_engine_is_asked_as_its_api_documents(
    name: EngineName,
    answer: object,
    url: str,
    auth: tuple[str, str],
    body: object,
    result: Result,
) -> None:
    got, (asked, headers, sent) = ask(name, answer)
    assert got == (result,)
    assert (asked, sent) == (url, body)
    assert headers[auth[0]] == auth[1]


def test_searxng_is_the_person_s_own_its_key_optional() -> None:
    answer = {"results": [{"title": "S", "url": "https://s.example/", "content": "c"}]}
    got, (asked, headers, _) = ask("searxng", answer, key=None, base="http://10.0.0.5:8080/")
    assert got == (Result("S", "https://s.example/", "c", "searxng"),)
    assert asked == "http://10.0.0.5:8080/search?q=proxy+error&format=json"
    assert "Authorization" not in headers
    assert general("searxng", key=None, base_url="http://10.0.0.5:8080").private_ok
    assert not general("tavily", key="k").private_ok


@pytest.mark.parametrize("name", ["tavily", "brave", "exa", "mojeek"])
def test_every_engine_is_the_web_source_and_needs_its_key(name: EngineName) -> None:
    assert general(name, key="k").name == WEB
    with pytest.raises(ValueError, match="needs its key"):
        general(name, key=None)


def test_an_answer_in_another_shape_is_a_bad_answer() -> None:
    got, _ = ask("tavily", {"something": "else"})
    assert got == SearchFailed(WEB, "bad_answer")


TRIAL_TEXT = (
    "Title: Extensions - HTTPX\nURL: https://www.python-httpx.org/advanced/extensions/\n"
    "Published: N/A\nAuthor: N/A\nHighlights:\nRequest and response extensions.\n\n"
    "The server's hostname.\n\n---\n\n"
    "Title: A PR\nURL: https://github.com/encode/httpcore/pull/507\nHighlights:\nconnect_to."
)


def sse(result: object) -> str:
    return "event: message\ndata: " + json.dumps({"result": result}) + "\n\n"


class Told:
    def __init__(self) -> None:
        self.said: list[str] = []

    def __call__(self, text: str) -> None:
        self.said.append(text)


def test_the_trial_is_one_stateless_tools_call_read_from_its_event() -> None:
    get = Answers(sse({"content": [{"type": "text", "text": TRIAL_TEXT}]}))
    got = trial(Told()).run("httpx sni", 5, get)
    assert got == (
        Result(
            "Extensions - HTTPX",
            "https://www.python-httpx.org/advanced/extensions/",
            "Request and response extensions. The server's hostname.",
            "exa_trial",
        ),
        Result(
            "A PR", "https://github.com/encode/httpcore/pull/507", "connect_to.", "exa_trial"
        ),
    )
    ((url, headers, body),) = get.asked
    assert url == TRIAL and "text/event-stream" in headers["Accept"]
    assert body == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {"query": "httpx sni", "numResults": 5, "objective": "httpx sni"},
        },
    }


def test_the_trial_spent_names_how_to_add_a_key_and_the_person_is_told_once() -> None:
    told = Told()
    source = trial(told)
    refused = SearchFailed(WEB, "rate_limited", 429)
    answers = [source.run("q", 5, Answers(refused)) for _ in range(3)]
    assert answers == [SearchFailed(WEB, "trial_spent", 429)] * 3
    assert "[web.search] engine" in SearchFailed(WEB, "trial_spent", 429).render()
    assert len(told.said) == 1 and "TAVILY_API_KEY" in told.said[0]


def test_a_tool_error_from_the_trial_is_a_bad_answer() -> None:
    get = Answers(sse({"isError": True, "content": [{"type": "text", "text": "no"}]}))
    assert trial(Told()).run("q", 5, get) == SearchFailed(WEB, "bad_answer")


class Api:
    def __init__(self) -> None:
        self.posted: list[tuple[Admitted, bytes]] = []

    def get(self, target: Admitted, headers: Mapping[str, str] | None = None) -> Fetched:
        return Fetched(target.url, "{}", False)

    def post(
        self, target: Admitted, body: bytes, headers: Mapping[str, str] | None = None
    ) -> Fetched | Moved:
        self.posted.append((target, body))
        return Fetched(target.url, "{}", False)


def resolver(*addresses: str) -> Resolver:
    def resolve(host: str, port: int) -> Sequence[str]:
        del host, port
        return addresses

    return resolve


def test_a_body_is_posted_and_only_the_person_s_searxng_may_be_private() -> None:
    api = Api()
    private = resolver("10.0.0.5")
    get: Getter = http_getter(api, "web", resolve=private)
    assert get("http://10.0.0.5:8080/search?q=x", {}) == SearchFailed("web", "connection")
    own: Getter = http_getter(api, "web", resolve=private, private_ok=True)
    assert own("http://10.0.0.5:8080/search?q=x", {}) == "{}"
    public: Getter = http_getter(api, "web", resolve=resolver("93.184.216.34"))
    assert public("https://api.tavily.com/search", {}, b'{"q": 1}') == "{}"
    assert [body for _, body in api.posted] == [b'{"q": 1}']


@pytest.mark.parametrize(
    ("table", "says"),
    [
        ({"engine": "tavily"}, "needs api_key_env"),
        ({"engine": "searxng"}, "needs searxng_url"),
        ({"engine": "searxng", "searxng_url": "search.example.org"}, "http or https URL"),
    ],
)
def test_the_table_refuses_an_engine_it_cannot_ask(table: dict[str, str], says: str) -> None:
    with pytest.raises(ValidationError, match=says):
        WebSearchPolicy.model_validate(table)


def test_the_table_s_key_variables_are_withheld_from_commands() -> None:
    table = WebSearchPolicy(
        engine="tavily", api_key_env="TAVILY_API_KEY", github_token_env="GH"
    )
    assert table.key_variables() == frozenset({"TAVILY_API_KEY", "GH"})
    assert WebSearchPolicy().key_variables() == frozenset()


def test_a_session_reads_each_key_from_its_variable_and_refuses_an_unset_one() -> None:
    from chickadee.errors import ConfigError

    table = WebSearchPolicy(engine="tavily", api_key_env="TAVILY_API_KEY")
    with pytest.raises(ConfigError, match=r"'TAVILY_API_KEY'.*unset"):
        session_sources(table, notify=Told(), env={})
    *named, web = session_sources(table, notify=Told(), env={"TAVILY_API_KEY": "tvly-1"})
    assert web.provider == "web.search.tavily"
    assert [s.name for s in named] == [
        "github",
        "stackexchange",
        "pypi",
        "npm",
        "crates",
        "wikipedia",
        "arxiv",
    ]
    keyless = session_sources(WebSearchPolicy(), notify=Told(), env={})
    assert keyless[-1].provider == "web.search.exa_trial"
    config = EngineConfig.with_defaults(
        {
            **raw_config(),
            "web": {"search": {"engine": "tavily", "api_key_env": "TAVILY_API_KEY"}},
        }
    )
    assert "TAVILY_API_KEY" in config.endpoint_keys
