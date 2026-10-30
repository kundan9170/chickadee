"""The `web` port over one agent round.

`Web` is what `web_fetch` calls: each URL through the guards (`urls.check_url`), each GET
through the external-call policy, each part of a page shown through its untrusted envelope.
What it remembers is the round's: the URLs that appeared (`saw`, a page fetched, a redirect's
target) and the pages fetched, each fetched once and paged from the round's copy, so reading on
is not another request.

A redirect to the same host is followed, its target checked as any URL is, up to `_HOPS` times.
One to another host is the answer: its target is admitted, and the model fetches it if it wants
it, so a page cannot move a fetch somewhere the model never chose.

`search` asks one source, `web` when none is named (the person's general
engine, or the keyless trial), in one `external_call`, at most `web.search_results` results.
Every result's URL is admitted, so what a search found can be fetched next.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Final, Literal
from urllib.parse import urlsplit

from chickadee.context.results import (
    FetchFailed,
    InvalidArgument,
    SearchFailed,
    ToolFailure,
    UrlRefused,
)
from chickadee.external.fetch import Engine, Fetched, Moved
from chickadee.external.fetch.http import HttpEngine
from chickadee.external.fetch.lightpanda import LightpandaEngine, find_lightpanda
from chickadee.external.hosts import HostApproval
from chickadee.external.policy import CallReply, ExternalCall, ExternalPolicy
from chickadee.external.search import Getter, Result, Source, http_getter, render_results
from chickadee.external.search.engines import WEB
from chickadee.external.search.sources import official
from chickadee.external.urls import DOCS_DOMAINS, Resolver, SeenUrls, check_url, normalise
from chickadee.isolation.base import SandboxBackend
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import SandboxLevel
from chickadee.limits import limit

__all__ = ["EngineChoice", "Web", "WebLimits", "find_lightpanda", "page_engine"]

_HOPS: Final = 5
"""Same-host redirects one fetch follows: enough for `http` to `https` and a trailing slash."""


def _row(name: str) -> int:
    return int(limit(name).value)


@dataclass(frozen=True)
class WebLimits:
    """The `web.*` rows, as `config.harness_limit` resolves them."""

    fetch_max_chars: int = _row("web.fetch_max_chars")
    fetch_connect_s: int = _row("web.fetch_connect_s")
    fetch_timeout_s: int = _row("web.fetch_timeout_s")
    url_max_chars: int = _row("web.url_max_chars")
    search_results: int = _row("web.search_results")
    search_snippet_chars: int = _row("web.search_snippet_chars")
    fetch_page_s: int = _row("web.fetch_page_s")

    @staticmethod
    def resolved(row: Callable[[str], int]) -> WebLimits:
        """Every row as `row` resolves it: `EngineConfig.harness_limit`, a person's overrides
        over the register's defaults."""
        return WebLimits(
            fetch_max_chars=row("web.fetch_max_chars"),
            fetch_connect_s=row("web.fetch_connect_s"),
            fetch_timeout_s=row("web.fetch_timeout_s"),
            url_max_chars=row("web.url_max_chars"),
            search_results=row("web.search_results"),
            search_snippet_chars=row("web.search_snippet_chars"),
            fetch_page_s=row("web.fetch_page_s"),
        )

    @property
    def fetch_max_bytes(self) -> int:
        """A body is read to four bytes a character: a page's markup is most of it."""
        return 4 * self.fetch_max_chars


@dataclass(frozen=True)
class _Page:
    url: str
    text: str
    truncated: bool
    retrieved: str


class Web:
    """One round's web. Its tool is `serial` (`web_fetch`), so one call runs at a time and
    what it remembers needs no lock."""

    def __init__(
        self,
        policy: ExternalPolicy,
        *,
        limits: WebLimits | None = None,
        engine: Engine | None = None,
        docs: Iterable[str] = (),
        resolve: Resolver | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sources: Sequence[Source] | None = None,
        getter: Callable[[Source], Getter] | None = None,
        approval: HostApproval | None = None,
    ) -> None:
        """`sources` are the official ones (`sources.official()`) unless given; `getter` is
        each source's way to its API, the `http` engine's unless given; `approval` is the
        person's per-host yes under `web.fetch.approval = "per_domain"`, None under
        `never`."""
        self._policy = policy
        self._limits = limits or WebLimits()
        http = HttpEngine(
            connect_ms=self._limits.fetch_connect_s * 1000,
            timeout_ms=self._limits.fetch_timeout_s * 1000,
            max_bytes=self._limits.fetch_max_bytes,
        )
        self._engine = engine or http
        self._sources = {s.name: s for s in (official() if sources is None else sources)}

        def by_http(source: Source) -> Getter:
            return http_getter(http, source.name, resolve=resolve, private_ok=source.private_ok)

        self._getter: Callable[[Source], Getter] = getter or by_http
        self._approval = approval
        self._docs = DOCS_DOMAINS | frozenset(docs)
        self._resolve = resolve
        self._now = now
        self._seen = SeenUrls()
        self._pages: dict[str, _Page] = {}
        self._earlier: dict[str, str] = {}

    def fetched(self, url: str) -> str | None:
        """`url`'s text as this round fetched it, the same text the `external_call` line's
        blob holds; else what an earlier round of the agent was shown of it (`carry`); else
        None. What a web citation's quote is looked for in (`context/citations.check_web`)."""
        key = normalise(url) or url
        page = self._pages.get(key)
        return page.text if page is not None else self._earlier.get(key)

    def carry(self, url: str, shown: Sequence[str]) -> None:
        """What an earlier round of this agent was shown of `url`, each part as its tool
        result said it: a follow-up's report may quote it still, as a file read in an earlier
        round may be cited. Never fetched from: a new fetch asks again."""
        self._earlier[normalise(url) or url] = "\n".join(shown)

    def saw(self, text: str) -> None:
        """The URLs in `text`, which the round was given, may be fetched (guard 1)."""
        self._seen.admit(text)

    def fetch(
        self, url: str, offset: int, call_id: str, cancel: threading.Event | None = None
    ) -> str | ToolFailure:
        if offset < 0:
            return InvalidArgument(
                "offset", "an offset is a count of characters, from 0", do="send 0 or more"
            )
        page = self._pages.get(normalise(url) or url)
        if page is None:
            got = self._get(url, call_id, cancel)
            if not isinstance(got, _Page):
                return got
            page = got
        return self._part(page, offset)

    def search(self, query: str, source: str | None, call_id: str) -> str | ToolFailure:
        name = WEB if source is None else source
        found = self._sources.get(name)
        if found is None:
            return InvalidArgument(
                "source",
                f"no search source is named {name!r}",
                do=f"send one of {', '.join(sorted(self._sources))}",
            )
        call = ExternalCall(
            call_id, found.provider, "web_search", {"query": query, "source": name}
        )
        run = partial(found.run, query, self._limits.search_results, self._getter(found))
        got = self._policy.run(call, run, self._search_reply)
        if isinstance(got, SearchFailed):
            return got
        results = _distinct(got)
        for r in results:
            self._seen.admit_url(r.url)
        return self._policy.shown(
            "web_results",
            render_results(results, self._limits.search_snippet_chars),
            source=f"search:{name}",
            span=f"{len(results)} results",
            query=query,
            trust="untrusted",
        )

    def _search_reply(self, answer: tuple[Result, ...] | SearchFailed) -> CallReply:
        if isinstance(answer, SearchFailed):
            return CallReply(None, error=f"{answer.failure}:{answer.why}")
        # De-duplicated as the model is shown them, so the record is what it saw.
        return CallReply(render_results(_distinct(answer), self._limits.search_snippet_chars))

    def _get(
        self, url: str, call_id: str, cancel: threading.Event | None
    ) -> _Page | str | ToolFailure:
        current = url
        for hop in range(_HOPS + 1):
            target = check_url(
                current,
                seen=self._seen,
                docs=self._docs,
                max_chars=self._limits.url_max_chars,
                resolve=self._resolve,
            )
            if isinstance(target, UrlRefused):
                return target
            allowed = self._approval is None or self._approval.allows(
                target.host, call_id, cancel
            )
            if not allowed:
                return UrlRefused(current, "not_approved")
            call = ExternalCall(
                call_id, self._engine.provider, "web_fetch", {"url": target.url}
            )
            answer = self._policy.run(call, partial(self._engine.get, target), _reply)
            if isinstance(answer, FetchFailed):
                return answer
            if isinstance(answer, Fetched):
                page = _Page(
                    target.url,
                    answer.text,
                    answer.truncated,
                    self._now().isoformat(timespec="seconds"),
                )
                self._seen.admit(answer.text)
                # Where the page was found: the URL asked, the one checked, and, when an
                # engine followed a same-host redirect itself, the one it landed on.
                self._seen.admit_url(answer.url)
                for key in {
                    normalise(url) or url,
                    target.url,
                    normalise(answer.url) or answer.url,
                }:
                    self._pages[key] = page
                return page
            moved = self._target(answer.location)
            # 0: the browser followed it itself (`lightpanda`), and only says where it landed.
            how = f" ({answer.status})" if answer.status else ""
            if moved is None:
                return (
                    f"{target.url} redirects{how} to a URL web_fetch cannot "
                    "fetch: find the page another way"
                )
            self._seen.admit_url(moved)
            if urlsplit(moved).hostname != target.host:
                return (
                    f"{target.url} redirects{how} to {moved!r}, on another host: "
                    "fetch that URL to read it"
                )
            if hop == _HOPS:
                break
            current = moved
        return (
            f"{url} redirected {_HOPS} times on its own host, last to {current}: fetch "
            "that URL to go on, or find the page another way"
        )

    def _target(self, location: str) -> str | None:
        """A redirect's target as it may be shown and fetched: an http(s) URL within the cap,
        with no space or control character in it. The server chose it, so it is shown outside
        any untrusted region only once it is a URL and nothing else."""
        moved = normalise(location)
        if moved is None or len(moved) > self._limits.url_max_chars:
            return None
        if any(c.isspace() or not c.isprintable() for c in moved):
            return None
        return moved

    def _part(self, page: _Page, offset: int) -> str | ToolFailure:
        total = len(page.text)
        if offset and offset >= total:
            return InvalidArgument(
                "offset",
                f"the page has {total} characters",
                do=f"fetch from an offset below {total}",
            )
        end = min(total, offset + self._limits.fetch_max_chars)
        chars = f"{offset}-{end} of {total}"
        shown = self._policy.shown(
            "web_content",
            page.text[offset:end],
            source=page.url,
            span=f"chars {chars}",
            url=page.url,
            retrieved=page.retrieved,
            chars=chars,
            trust="untrusted",
        )
        notes: list[str] = []
        if end < total:
            notes.append(
                f"{total - end} characters left: web_fetch(url={page.url!r}, offset={end}) "
                "reads on"
            )
        if page.truncated:
            notes.append(
                f"the page was cut at {self._limits.fetch_max_bytes} bytes when it was "
                "fetched: nothing after the cut was read"
            )
        return "\n".join([shown, *notes])


def _reply(answer: Fetched | Moved | FetchFailed) -> CallReply:
    """What the `external_call` line says came back."""
    if isinstance(answer, Fetched):
        return CallReply(answer.text, answer.truncated)
    if isinstance(answer, Moved):
        return CallReply(f"{answer.status} {answer.location}")
    return CallReply(None, error=f"{answer.failure}:{answer.why}")


def _distinct(results: tuple[Result, ...]) -> tuple[Result, ...]:
    """`results` with a URL an engine gave twice shown once, where it came first."""
    seen: set[str] = set()
    out: list[Result] = []
    for r in results:
        key = normalise(r.url) or r.url
        if key not in seen:
            seen.add(key)
            out.append(r)
    return tuple(out)


EngineChoice = Literal["auto", "http", "lightpanda"]


def page_engine(
    choice: EngineChoice,
    limits: WebLimits,
    *,
    backend: SandboxBackend,
    require_sandbox: bool,
    binary: Path | None,
) -> Callable[[tuple[Path, ...]], Engine] | str:
    """`web.fetch.engine` as a factory over what a fetch is masked from, or why it cannot be
    had. `auto` takes `lightpanda` only where it is installed and a sandbox
    seals it, since it runs a page's scripts; else `http`, which runs none. `lightpanda` named
    is refused when it is not installed, or when a sandbox is required and none seals."""
    sealed = backend.available() and backend.level() is not SandboxLevel.NONE
    if choice == "http" or (choice == "auto" and (binary is None or not sealed)):
        return lambda masked: HttpEngine(
            connect_ms=limits.fetch_connect_s * 1000,
            timeout_ms=limits.fetch_timeout_s * 1000,
            max_bytes=limits.fetch_max_bytes,
        )
    if binary is None:
        return (
            "web.fetch.engine is lightpanda, and no lightpanda is installed: install it "
            "(`pip install 'chickadee-core[web]'`, or a `lightpanda` on the PATH), or set "
            'web.fetch.engine = "auto"'
        )
    if require_sandbox and not sealed:
        return (
            "web.fetch.engine is lightpanda, which runs a page's scripts, and "
            "sandbox.require_sandbox is set with no sandbox to seal it: install one, or set "
            'web.fetch.engine = "http"'
        )
    wrap = backend if sealed else Passthrough()
    found = binary
    return lambda masked: LightpandaEngine(
        found,
        wrap=wrap,
        connect_ms=limits.fetch_connect_s * 1000,
        timeout_ms=limits.fetch_timeout_s * 1000,
        page_ms=limits.fetch_page_s * 1000,
        max_bytes=limits.fetch_max_bytes,
        masked=masked,
    )
