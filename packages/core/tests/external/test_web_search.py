"""`web_search` through a dispatcher, over a `Web` whose sources answer from tables: one source,
`web` when none is named, bounded, every result's
URL admitted for a fetch, its call recorded, and a failed source's failure the answer."""

from __future__ import annotations

from pathlib import Path

from tools.tools_helpers import call, dispatcher
from webs import DictEngine, calls, public, reads, session

from chickadee.context.results import SearchFailed
from chickadee.context.session import Session
from chickadee.external.policy import ExternalPolicy
from chickadee.external.search import Getter, Result, Source
from chickadee.external.web import Web, WebLimits
from chickadee.tools.builtin import web_fetch, web_search
from chickadee.tools.catalogue import answer_text
from chickadee.tools.context import ToolContext


def source(name: str, *results: Result | SearchFailed) -> Source:
    def run(query: str, limit: int, get: Getter) -> tuple[Result, ...] | SearchFailed:
        del query, get
        if results and isinstance(results[0], SearchFailed):
            return results[0]
        return tuple(r for r in results if isinstance(r, Result))[:limit]

    return Source(name, f"web.search.{name}", run)


def hit(name: str, n: int) -> Result:
    return Result(f"{name} {n}", f"https://{name}.example/{n}", f"about {name} {n}", name)


def web(
    s: Session, *sources: Source, results: int = 5, engine: DictEngine | None = None
) -> Web:
    return Web(
        ExternalPolicy(s, rung="researcher"),
        limits=WebLimits(search_results=results),
        engine=engine or DictEngine({}),
        resolve=public,
        sources=sources,
    )


def ask(w: Web, cid: str = "c1", **args: object) -> str:
    d = dispatcher([web_search.TOOL, web_fetch.TOOL], ctx=ToolContext(role="researcher", web=w))
    name = "web_fetch" if "url" in args else "web_search"
    return answer_text(d.call(call(name, text="", cid=cid, **args)))


def test_one_source_s_results_come_back_as_untrusted_text(tmp_path: Path) -> None:
    with session(tmp_path) as s:
        w = web(s, source("npm", hit("npm", 1), hit("npm", 2)))
        said = ask(w, query="left pad", source="npm")
        assert said == (
            "<web_results query='left pad' trust='untrusted'>\n"
            "1. npm 1\n   https://npm.example/1\n   about npm 1\n   (npm)\n\n"
            "2. npm 2\n   https://npm.example/2\n   about npm 2\n   (npm)\n"
            "</web_results>"
        )
        (p,) = calls(s)
        assert (p.provider, p.tool, p.operation_id, p.ok) == (
            "web.search.npm",
            "web_search",
            "c1",
            True,
        )
        (read,) = reads(s)
        assert (read.path, read.span) == ("search:npm", "2 results")


def test_with_no_source_the_whole_web_is_asked(tmp_path: Path) -> None:
    """`web`, the person's general engine or the keyless trial."""
    with session(tmp_path) as s:
        w = web(s, source("github", hit("github", 1)), source("web", hit("web", 1)))
        said = ask(w, query="proxy")
        assert "1. web 1" in said and "github" not in said
        assert [p.provider for p in calls(s)] == ["web.search.web"]


def test_a_url_an_engine_gave_twice_is_shown_once(tmp_path: Path) -> None:
    with session(tmp_path) as s:
        w = web(
            s,
            source(
                "web",
                Result("Same", "https://same.example/x", "", "web"),
                Result("Same", "https://same.example/x#top", "", "web"),
            ),
        )
        assert ask(w, query="x").count("https://same.example/x") == 1
        # The record holds what the model was shown, the repeat dropped there too.
        (recorded,) = calls(s)
        assert recorded.result_sha is not None
        assert s.blob(recorded.result_sha).decode().count("https://same.example/x") == 1


def test_a_failed_source_is_the_answer_and_is_recorded(tmp_path: Path) -> None:
    with session(tmp_path) as s:
        w = web(s, source("arxiv", SearchFailed("arxiv", "timeout")))
        assert ask(w, query="x", source="arxiv").startswith(
            "SEARCH_FAILED{source: arxiv, reason: timeout}"
        )
        assert [p.ok for p in calls(s)] == [False]
        assert not reads(s)


def test_a_result_s_url_may_be_fetched_next(tmp_path: Path) -> None:
    url = "https://github.example/1"
    with session(tmp_path) as s:
        engine = DictEngine({url: "The issue's page."})
        w = web(s, source("github", hit("github", 1)), engine=engine)
        assert ask(w, url=url).startswith("URL_REFUSED")
        ask(w, "c2", query="bug", source="github")
        assert "\nThe issue's page.\n" in ask(w, "c3", url=url)


def test_a_source_the_tool_does_not_name_is_refused_by_its_schema(tmp_path: Path) -> None:
    with session(tmp_path) as s:
        w = web(s, source("github", hit("github", 1)))
        assert ask(w, query="x", source="google").startswith("INVALID_ARGUMENTS")
        assert not calls(s)


def test_a_source_the_round_was_not_given_is_refused_naming_those_it_was(
    tmp_path: Path,
) -> None:
    """The schema names every official source; a `Web` built with fewer refuses the rest."""
    with session(tmp_path) as s:
        w = web(s, source("github", hit("github", 1)))
        said = ask(w, query="x", source="pypi")
        assert said == (
            "INVALID_ARGUMENTS{field: source, reason: \"no search source is named 'pypi'\"}. "
            "Do: send one of github"
        )
