"""What a tool returns: `ToolResult`, its `Truncation`, and the typed failures.

A failure is a value naming its fields, never a sentence, because a model handed prose cannot
tell "the file is secret" from "the file is empty". A result carries the
handle it read and content only for a span the caller named and paid for.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final, Literal

from pydantic import with_config

from chickadee.client.request import DocumentPart, ImagePart
from chickadee.context.handle import Handle
from chickadee.context.result_data import ResultData
from chickadee.context.spans import Span
from chickadee.context.vocabulary import LINE

__all__ = [
    "SUPERSEDED_STUB",
    "CallCancelled",
    "CoderOutcome",
    "FetchFailed",
    "FetchReason",
    "HandleExpired",
    "ImageRef",
    "IndexUnavailable",
    "Interrupted",
    "InvalidArgument",
    "JobsFull",
    "NotSupported",
    "NotText",
    "SearchFailed",
    "SearchReason",
    "SecretPath",
    "ToolFailure",
    "ToolResult",
    "Truncation",
    "UnknownJob",
    "UnresolvedSearch",
    "UrlReason",
    "UrlRefused",
    "instructed",
    "size_text",
]

SUPERSEDED_STUB: Final = "[superseded — merged into {call_id}]"
"""What a superseded result renders as (architecture §7.4.2). The CALL it answered is never
touched: both calls say what was asked (invariant 19)."""


# -- typed failures ---------------------------------------------------


def _render(failure: str, fields: Mapping[str, object], do: str | None = None) -> str:
    inner = ", ".join(f"{k}: {v}" for k, v in fields.items() if v is not None)
    return instructed(f"{failure}{{{inner}}}", do)


def instructed(text: str, do: str | None) -> str:
    """A failure as a model reads it: its code and fields, then what to do next. The one
    form of the instruction, for every failure type."""
    return text if do is None else f"{text}. Do: {do}"


@dataclass(frozen=True)
class InvalidArgument:
    """`INVALID_ARGUMENTS{field, reason}` — a call the schema admitted and the meaning did not.

    Its schema-level twin is `tools.validate.InvalidArguments`, which carries every
    violated field and the full schema. This is what a validation *step* refuses: a
    constraint citing a turn the user never said, `lines=` on an image."""

    field: str
    why: str
    failure: Literal["INVALID_ARGUMENTS"] = "INVALID_ARGUMENTS"
    example: str | None = None
    """A minimal valid shape for `field`, from the tool's spec (`ToolSpec.examples`), added by
    the dispatcher's `ShowExample` so a malformed structured argument teaches its fix."""
    did_you_mean: str | None = None
    """For a path that does not resolve, the file it most likely meant (`paths.suggest`)."""
    rule: str | None = None
    """The rule the path broke, beside `did_you_mean`: `paths.ROOT_RULE`."""
    do: str | None = None
    """What to do next. A free-text refusal has no fields to derive it from, so each
    site passes its own; `scripts/guards/instructions.py` holds the sites without one."""

    def render(self) -> str:
        shown = {
            "field": self.field,
            "reason": repr(self.why),
            "example": self.example,
            "did_you_mean": self.did_you_mean,
            "rule": None if self.rule is None else repr(self.rule),
        }
        return _render(self.failure, shown, self.do)


@dataclass(frozen=True)
class HandleExpired:
    """`HANDLE_EXPIRED{handle, reason}`.

    `not_in_session`: the id names nothing this session minted — its session ended, or it
    never existed; the two look the same from here and neither is claimed.
    `not_in_conversation`: a reader of another conversation's (the scout's) was given an
    id it never minted, such as one of Zeus's; its handles are its own, so read by path.
    `pruned`: a person ran `chickadee gc`, with the run and time `gc.log` recorded.
    `absent`: a stored blob is missing for any other reason, or a text file's handle outlived
    the file — a file stores nothing in the session, so once it is gone there is nothing to
    offer."""

    handle: str
    why: Literal["not_in_session", "not_in_conversation", "pruned", "absent"]
    run_id: str | None = None
    gc_at: str | None = None
    failure: Literal["HANDLE_EXPIRED"] = "HANDLE_EXPIRED"
    do: str | None = None

    @property
    def instruction(self) -> str:
        if self.do is not None:
            return self.do
        if self.why == "not_in_conversation":
            return "read the file by its path; handles from another conversation are not yours"
        return "do not use this handle again; read the file by its path, or run the call again"

    def render(self) -> str:
        return _render(
            self.failure,
            {
                "handle": self.handle,
                "reason": self.why,
                "run_id": self.run_id,
                "gc_at": self.gc_at,
            },
            self.instruction,
        )


@dataclass(frozen=True)
class SecretPath:
    """`SECRET_PATH` — the path is on the deny list, the one class of
    path `read` refuses."""

    path: str
    pattern: str
    failure: Literal["SECRET_PATH"] = "SECRET_PATH"
    do: str = "do not read or write this path, and do not try another way to its contents"

    def render(self) -> str:
        fields = {"path": self.path, "pattern": repr(self.pattern)}
        return _render(self.failure, fields, self.do)


@dataclass(frozen=True)
class NotText:
    """`NOT_TEXT{kind, bytes}` — beyond `outline` on a non-text, non-image file."""

    kind: str
    bytes: int
    failure: Literal["NOT_TEXT"] = "NOT_TEXT"
    do: str = (
        'do not read its contents; read it with depth="outline" for what it is, or leave it'
    )

    def render(self) -> str:
        return _render(self.failure, {"kind": self.kind, "bytes": self.bytes}, self.do)


@dataclass(frozen=True)
class NotSupported:
    """`NOT_SUPPORTED{capability, rung}` — an image at `depth="full"` on a rung that cannot
    see. Typed, never a caption the harness invented (invariant 20)."""

    capability: str
    rung: str
    failure: Literal["NOT_SUPPORTED"] = "NOT_SUPPORTED"
    do: str = (
        'do not ask for it at depth="full" again on this model; read it with depth="outline" '
        "for its type and size, and work from that"
    )

    def render(self) -> str:
        fields = {"capability": self.capability, "rung": self.rung}
        return _render(self.failure, fields, self.do)


@dataclass(frozen=True)
class IndexUnavailable:
    """`INDEX_UNAVAILABLE` — with its coverage, so the caller knows what to fall back to."""

    path: str
    coverage: str
    failure: Literal["INDEX_UNAVAILABLE"] = "INDEX_UNAVAILABLE"
    do: str = (
        'find it with grep instead, then read the lines it names with lines="a-b"; '
        "do not ask the index for this path again"
    )

    def render(self) -> str:
        fields = {"path": self.path, "coverage": repr(self.coverage)}
        return _render(self.failure, fields, self.do)


@dataclass(frozen=True)
class UnresolvedSearch:
    """`UNRESOLVED` — nothing found, WITH what was not searched (invariant 20). An empty list
    is a claim about everything, and a partial search is not entitled to make one."""

    query: str
    searched: tuple[str, ...]
    uncovered: tuple[str, ...]
    failure: Literal["UNRESOLVED"] = "UNRESOLVED"

    @property
    def instruction(self) -> str:
        if self.uncovered:
            return (
                "nothing matched where it looked: search the uncovered paths above with grep, "
                "or change the query"
            )
        return "nothing matched anywhere: change the query, or conclude it is not here"

    def render(self) -> str:
        return _render(
            self.failure,
            {
                "query": repr(self.query),
                "searched": len(self.searched),
                "uncovered": "[" + ", ".join(self.uncovered) + "]",
            },
            self.instruction,
        )


@dataclass(frozen=True)
class UnknownJob:
    """`UNRESOLVED{job}`: no job of that id in this session. Names the call
    that lists them."""

    job: str
    failure: Literal["UNRESOLVED"] = "UNRESOLVED"

    def render(self) -> str:
        return _render(
            self.failure,
            {"job": repr(self.job), "send": "job(action='list')"},
            "call job(action='list') for the ids of this session's jobs, then use one of them",
        )


@dataclass(frozen=True)
class JobsFull:
    """`JOBS_FULL{running}`: `zeus.max_jobs` jobs are running, so this one
    cannot start in the background. Names them, and the call that frees one."""

    running: tuple[str, ...]
    failure: Literal["JOBS_FULL"] = "JOBS_FULL"

    def render(self) -> str:
        return _render(
            self.failure,
            {
                "running": "[" + ", ".join(self.running) + "]",
                "send": "job(action='kill', id=…)",
            },
            "wait for one of the running jobs to end, or stop one with job(action='kill', "
            "id=…), then send this call again",
        )


@dataclass(frozen=True)
class CallCancelled:
    """`CANCELLED{tool, partial}`: the person stopped this call. `partial` is the handle
    holding what it produced before it stopped, when it produced anything. Nothing is retried
    unless the person asks."""

    tool: str
    partial: str | None = None
    failure: Literal["CANCELLED"] = "CANCELLED"

    def render(self) -> str:
        fields = [f"tool: {self.tool}"]
        if self.partial is not None:
            fields.append(f"partial: {self.partial}")
        do = "do not call it again unless the person asks; continue from what you have"
        if self.partial is not None:
            do += f", and read {self.partial} for what it produced before it stopped"
        return instructed(f"{self.failure}{{{', '.join(fields)}}}", do)


@dataclass(frozen=True)
class Interrupted:
    """`INTERRUPTED{tool, reason}`: a call answered by the harness, as a
    fact, without having run — the tail repair's answer to a call a crash left waiting, and
    the answer to a call refused before anything was started (a shell before it spawns, a
    `coder` before it opens)."""

    tool: str
    why: str
    failure: Literal["INTERRUPTED"] = "INTERRUPTED"
    do: str = (
        "it may or may not have taken effect: check (read what it would have changed) before "
        "you call it again"
    )
    """For a call a crash left waiting, whose effect is unknown; a site that knows the call
    never started says so instead."""

    def render(self) -> str:
        return instructed(f"{self.failure}{{tool: {self.tool}, reason: {self.why!r}}}", self.do)


UrlReason = Literal[
    "not_seen", "too_long", "credential", "private_address", "unresolvable", "not_approved"
]

_URL_DO: dict[str, str] = {
    "not_seen": (
        "fetch only a URL from your task, a search result or a page you fetched, exactly as it "
        "appeared; search for the page, or follow a link on one you read"
    ),
    "too_long": "send the page's own URL, without its long query",
    "credential": "never put a credential in a URL; fetch the page without it",
    "private_address": "do not fetch it: the harness never reaches a private address",
    "unresolvable": "check the host's name: it does not resolve",
    "not_approved": (
        "find the facts on another site: the person did not allow this one, and say so in "
        "your report"
    ),
}


@dataclass(frozen=True)
class UrlRefused:
    """`URL_REFUSED{url, reason}`: a guard of `web_fetch` refused the URL, so
    nothing was sent."""

    url: str
    why: UrlReason
    failure: Literal["URL_REFUSED"] = "URL_REFUSED"

    def render(self) -> str:
        fields = {"url": repr(self.url), "reason": self.why}
        return _render(self.failure, fields, _URL_DO[self.why])


FetchReason = Literal["timeout", "connection", "http_status", "not_text", "robots"]

_FETCH_DO: dict[str, str] = {
    "timeout": "fetch it once more, or find the same facts on another page",
    "connection": "find the same facts on another page: this host did not answer",
    "http_status": "fetch another page, or search for where this one moved",
    "not_text": "find a text page about it, such as its HTML documentation",
    "robots": "find the same facts on another site: this one's robots.txt does not allow it",
}


@dataclass(frozen=True)
class FetchFailed:
    """`FETCH_FAILED{url, reason}`: the request was sent and did not come back as a
    page: no answer in time, no connection, an HTTP error, or content that is not text."""

    url: str
    why: FetchReason
    status: int | None = None
    failure: Literal["FETCH_FAILED"] = "FETCH_FAILED"

    def render(self) -> str:
        fields = {"url": repr(self.url), "reason": self.why, "status": self.status}
        return _render(self.failure, fields, _FETCH_DO[self.why])


SearchReason = Literal[
    "timeout", "connection", "http_status", "rate_limited", "bad_answer", "trial_spent"
]

_SEARCH_DO: dict[str, str] = {
    "timeout": "search once more, or search another source",
    "connection": "search another source: this one did not answer",
    "http_status": "search another source",
    "rate_limited": (
        "search another source: this one allows few searches without a key, and the person "
        "can give it one"
    ),
    "bad_answer": "search another source: this one answered in a shape it never had",
    "trial_spent": (
        "search a named source (github, stackexchange, wikipedia, pypi, npm, crates, arxiv): "
        "the keyless web trial is spent, and the whole web needs the person's own search key "
        "(`[web.search] engine` and `api_key_env`), which you cannot add"
    ),
}


@dataclass(frozen=True)
class SearchFailed:
    """`SEARCH_FAILED{source, reason}`: a search source did not answer with results:
    no answer in time, no connection, an HTTP error, a rate limit, or an answer it could not
    read."""

    source: str
    why: SearchReason
    status: int | None = None
    failure: Literal["SEARCH_FAILED"] = "SEARCH_FAILED"

    def render(self) -> str:
        fields = {"source": self.source, "reason": self.why, "status": self.status}
        return _render(self.failure, fields, _SEARCH_DO[self.why])


ToolFailure = (
    InvalidArgument
    | HandleExpired
    | SecretPath
    | NotText
    | NotSupported
    | IndexUnavailable
    | UnresolvedSearch
    | UrlRefused
    | FetchFailed
    | SearchFailed
)


# -- results ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Truncation:
    """A value, never a prose notice. Absent when nothing was dropped — and that absence
    is how a caller tells *the file ended* from *the output was cut*.

    For `byte_cap` and `line_cap` the counts are lines: `dropped` is what the request asked
    for and did not get, `total` the lines the file has. For `result_cap` (a search) they
    count matches, and `kept` is the one range the kept matches fall in."""

    kept: tuple[Span, ...]
    """Every span this result carries, in order — several when the request was several (a
    merged view, `at="@changed"`). The gaps between them are lines it does not carry, so
    one range from the first line to the last would claim lines that were never sent."""
    dropped: int
    total: int | None
    """None = coverage unknown (invariant 20)."""
    reason: Literal["byte_cap", "line_cap", "result_cap"]
    continue_with: Handle
    cap: int | None = None
    """The cap that cut, in its unit (bytes for `byte_cap`, lines for `line_cap`), so the
    model is told what one read shows; None where no reader's cap did."""

    def next_span(self) -> Span | None:
        """Where `continue_with` picks up: the same width again, never past what was asked."""
        if self.reason == "result_cap" or self.dropped <= 0 or not self.kept:
            return None
        last = self.kept[-1].end
        start = last + 1
        end = min(start + sum(len(s) for s in self.kept) - 1, last + self.dropped)
        if self.total is not None:
            end = min(end, self.total)
        return Span(start, end) if end >= start else None

    def render(self) -> str:
        """What to do next, in the model's own call form: never the
        reason's name, and never another call syntax than the one it sends."""
        if self.reason == "result_cap":
            total = f"{self.total:,}" if self.total is not None else "an unknown number of"
            return (
                f"[{self.dropped:,} of {total} matches not shown; narrow the pattern, or read "
                f"them by line: {_call('read', self.continue_with, 'lines', 'start-end')}]"
            )
        of = f" of {self.total:,}" if self.total is not None else ""
        shown = ", ".join(str(s) for s in self.kept) if self.kept else "none"
        parts = [f"showing lines {shown}{of}"]
        if self.cap is not None:
            limit = size_text(self.cap) if self.reason == "byte_cap" else f"{self.cap:,} lines"
            parts.append(f"one read shows at most {limit}")
        nxt = self.next_span()
        if nxt is not None:
            parts.append(f"next: {_call('read', self.continue_with, 'lines', str(nxt))}")
        else:
            parts.append(f"{self.dropped:,} more line(s) asked for were not shown")
        return f"[{'; '.join(parts)}]"


def size_text(n: int) -> str:
    """A size as a header and a cut note say it: `800 B`, `16.0 KB`, `1.2 MB`."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _call(tool: str, handle: Handle, key: str, value: str) -> str:
    """A call as the model sends it: the tool's name and its JSON arguments, by path when the
    handle names a file, else by handle."""
    args: dict[str, str] = (
        {"path": handle.path} if handle.path is not None else {"handle": handle.id}
    )
    args[key] = value
    return f"{tool} {json.dumps(args, ensure_ascii=False)}"


@with_config(LINE)
@dataclass(frozen=True)
class ImageRef:
    """The pixels a result sent: the downscaled bytes, in the session's blob store,
    so a resumed session sends the same bytes the live one did. A PDF sent whole is one too,
    by its media type."""

    sha: str
    media_type: str


@with_config(LINE)
@dataclass(frozen=True)
class CoderOutcome:
    """A `coder` result's terminal and why, typed, so a reader shows the reason without
    reading it out of the text. `terminal` is the result's head, the episode's outcome
    (`COMPLETED`, `PARTIAL`, `BLOCKED`, …), or a packet-loop terminal on an older log;
    `why` the lines that say why, as the result states them."""

    terminal: str
    why: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResult:
    """What every read returns: the handle, and content only when it was asked for by
    name. `header`, `diagnostics` and `notes` are the lines around it; `render()` is exactly
    what the model reads."""

    handle: Handle
    content: str | None = None
    """Populated ONLY when the caller asked by name and paid. None means the preview."""
    header: str = ""
    truncation: Truncation | None = None
    image: ImagePart | DocumentPart | None = None
    """The pixels, when an image was read at `depth="full"`; or the PDF itself, on a
    model that reads PDFs. Either is sent as a part beside the text."""
    diagnostics: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    view: tuple[Span, ...] = ()
    """The file lines this result shows, when it shows any — the view it holds."""
    supersedes: tuple[str, ...] = ()
    """Call ids whose results this view absorbed. Their results are stubbed; their
    calls stay exactly as they were sent (invariant 19)."""
    outcome: CoderOutcome | None = None
    """A `coder` result's terminal and why, for its `tool` line."""
    data: ResultData | None = None
    """What it was made from, for the worker's Python blocks: `shell`'s
    `Completed`. None for every other result."""

    def body(self) -> str:
        return self.content if self.content is not None else self.handle.preview

    def render(self) -> str:
        parts = [self.header] if self.header else []
        if self.image is None:
            parts.append(self.body())
        if self.truncation is not None:
            parts.append(self.truncation.render())
        parts.extend(self.diagnostics)
        parts.extend(self.notes)
        return "\n".join(p for p in parts if p)

    def with_notes(self, *notes: str) -> ToolResult:
        return replace(self, notes=(*self.notes, *notes))
