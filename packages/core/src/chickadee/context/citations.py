"""The citation gate: a served claim's evidence LOCATIONS are checked, never its truth.

An answer citing a line that does not say what it claims is worse than no answer, because it
is credible. Checking a location is mechanical and checking a claim is not, so this module
checks exactly the mechanical part and says which part it checked. It is pure code and calls
no model.

Each citation answers three questions, in order:

1. Is it repository evidence at all: a path inside the workspace, and not a secret one?
2. Does the location exist: a text file, with the cited lines inside it?
3. When the lines the citer SAW are known (from the view it read them in), are they
   still there? If they are elsewhere in the file now, the citation MOVED and the new span is
   named; if they are nowhere, the lines CHANGED.

A citation checked only up to 2 passed on its *location*; one checked through 3 passed on its
*content*. The result says which, because an answer states what was proven, not what was
hoped.

**A web citation is a URL and a quote**: `https://docs.python.org/3/x.html
"the words it says"`. It passes `QUOTED` when the round fetched that URL and the quote is in
what the fetch showed, whitespace and Markdown emphasis folded on both sides; `NOT_FETCHED`
when the round never fetched it; `NOT_ON_PAGE` when the words are not there. That the page says
it is all a pass means: quoted from the page, never true.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal

from chickadee.context.files import read_file
from chickadee.context.fit import shown_cut
from chickadee.context.outline import code_language, sniff
from chickadee.context.spans import Span
from chickadee.edits.lines import parse_row
from chickadee.external.urls import find_urls, normalise
from chickadee.paths import within
from chickadee.profiles.loader import ProfileSet
from chickadee.schema.journal import CitationGateP
from chickadee.secrets.denylist import Denial, Denylist, name_denied, read_denied

if TYPE_CHECKING:
    from chickadee.context.session import SessionState

__all__ = [
    "CITATION_RE",
    "Basis",
    "Checked",
    "Citation",
    "CitationGateResult",
    "LineSource",
    "SeenSource",
    "Verdict",
    "WebCitation",
    "WebSource",
    "check",
    "claims",
    "extract",
    "gate",
    "repository_path",
    "seen_in",
    "seen_in_reads",
    "web_extract",
    "workspace_lines",
]

LineSource = Callable[[str], Sequence[str] | Denial | None]
"""A repository path's current lines; the `Denial` when the read rule refuses the file the
path lands on (a link named `notes.txt` to `.env`), so the citation is `SECRET` and never
`NO_FILE`; or None when there is no text file there."""
SeenSource = Callable[[str, Span], Sequence[str] | None]
"""What the citer saw at those lines, or None when that is not known."""
WebSource = Callable[[str], str | None]
"""A URL's text as the round's fetch showed it (`external/web.Web.fetched`), or None when the
round never fetched it."""

WEB_CITATION_RE: Final = re.compile(
    r"(?P<url>https?://[^\s<>\"'`]+)[>`]*:?\s+[\"\u201c](?P<quote>[^\"\u201c\u201d\n]+)[\"\u201d]",
    re.IGNORECASE,
)
"""`<url> "<quote>"`: a URL, then on the same line its quote in straight or curly double
quotes."""

CITATION_RE: Final = re.compile(
    r"(?<![\w/.@+:~-])"
    r"(?P<path>(?:~?/)?(?:[\w.@+-]+/)*\.?[\w@+-][\w.@+-]*)"
    r"(?::(?P<a>\d+)(?:[-\N{EN DASH}](?P<b>\d+))?|#L(?P<c>\d+)(?:-L?(?P<d>\d+))?)"
    r"(?!\w)(?!\.\d)"
)
"""`file:line` (D116's form), `file:12-40`, and `file#L12-L40`, dotfiles included. A trailing
sentence stop is not part of it; a column (`file:12:5`) is ignored."""

_FILE_SUFFIXES: Final = frozenset(
    (
        *(".md", ".rst", ".txt", ".json", ".toml", ".yml", ".yaml", ".ini", ".cfg"),
        *(".csv", ".tsv", ".sh", ".sql", ".html", ".css", ".xml"),
    )
)
"""A bare name (no `/`) with one of these suffixes, or a source file's, which a profile owns
, is a citation even when no such
file exists, so a hallucinated `utils.py:12` is rejected. Any other bare name is a citation
only when the repository has it: `Makefile:12` is one, `example.com:8080` and `1.5:2` are
not."""

_URL: Final = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_FENCE: Final = re.compile(r"^\s*(```|~~~)")
_NOT_A_CLAIM: Final = re.compile(r"^\s*(?:#{1,6}\s|(?:[-*_]\s*){3,}$|\|?[\s:|-]+\|[\s:|-]*$)")
"""A heading, a horizontal rule, or a table's separator row."""
_BULLET: Final = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_SENTENCE: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z`\"(\[])")


# -- extraction -------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    path: str
    """As written: relative, absolute, or `~`-rooted."""
    start: int
    end: int
    text: str
    """As written in the answer."""
    claim: int = 0
    """Which claim of the answer it supports (`claims()`), for coverage."""

    @property
    def span(self) -> Span | None:
        """The lines, or None when the numbers do not form a span (`:0`, `:40-12`)."""
        if self.start < 1 or self.end < self.start:
            return None
        return Span(self.start, self.end)


def claims(answer: str) -> tuple[str, ...]:
    """The answer's claims: its sentences outside code fences, each bullet its own unit, a
    wrapped line joined to the sentence it continues, headings and rules left out. A count,
    not a parse: it feeds `coverage`, which is reported and never gates."""
    blocks: list[str] = []
    current: list[str] = []
    fence = False
    for line in answer.splitlines():
        if _FENCE.match(line):
            fence = not fence
            continue
        if fence or _NOT_A_CLAIM.match(line):
            continue
        if not line.strip() or _BULLET.match(line):
            if current:
                blocks.append(" ".join(current))
            current = [_BULLET.sub("", line).strip()] if line.strip() else []
            continue
        current.append(line.strip())
    if current:
        blocks.append(" ".join(current))
    return tuple(s.strip() for b in blocks for s in _SENTENCE.split(b) if s.strip())


def extract(text: str, *, claim: int = 0) -> tuple[Citation, ...]:
    """Every `file:line` shape in `text`, URLs removed first: a URL is not a citation,
    whatever it ends with. A shape is a candidate; `gate` keeps the ones that name a file."""
    bare = _URL.sub(lambda m: " " * len(m.group(0)), text)
    found: list[Citation] = []
    for m in CITATION_RE.finditer(bare):
        a = m.group("a") or m.group("c")
        b = m.group("b") or m.group("d") or a
        found.append(Citation(m.group("path"), int(a), int(b), m.group(0), claim))
    return tuple(found)


def _names_a_file(
    path: str, read: LineSource, denylist: Denylist, languages: ProfileSet | None = None
) -> bool:
    if "/" in path or name_denied(path, denylist) is not None:
        # A path is a path, and a secret's name is refused as one — never probed to decide.
        return True
    suffix = PurePosixPath(path).suffix
    # Shaped like a file whatever its case: a hallucinated `Utils.PY:12` is still a citation of
    # one, though no profile reads `.PY` as Python (a profile matches an extension as written).
    return (
        suffix.lower() in _FILE_SUFFIXES
        or code_language(path[: len(path) - len(suffix)] + suffix.lower(), languages)
        is not None
        or read(path) is not None
    )


@dataclass(frozen=True)
class WebCitation:
    url: str
    """As written, without what ends a sentence."""
    quote: str
    text: str
    """As written in the answer."""
    claim: int = 0


def web_extract(text: str, *, claim: int = 0) -> tuple[WebCitation, ...]:
    """Every `<url> "<quote>"` in `text`."""
    found: list[WebCitation] = []
    for m in WEB_CITATION_RE.finditer(text):
        urls = find_urls(m.group("url"))
        if urls:
            found.append(WebCitation(urls[0], m.group("quote").strip(), m.group(0), claim))
    return tuple(found)


def _without_web(text: str) -> str:
    """`text` with each web citation blanked, so a quote saying `app.py:12` is not taken for a
    file citation of its own."""
    return WEB_CITATION_RE.sub(lambda m: " " * len(m.group(0)), text)


# -- checking ---------------------------------------------------------------------------


class Verdict(StrEnum):
    OK = "ok"
    MOVED = "moved"
    """The lines the citer saw are elsewhere in the file now; `Checked.now` names where."""
    CHANGED = "changed"
    """The lines the citer saw are nowhere in the file now."""
    NO_FILE = "no_file"
    """No text file the harness may read is at that path."""
    OUT_OF_RANGE = "out_of_range"
    OUTSIDE = "outside"
    """Not repository evidence: absolute, or escaping the workspace."""
    SECRET = "secret"
    """On the deny list. Refused unread."""
    QUOTED = "quoted"
    """A web citation whose quote is on the page the round fetched."""
    NOT_FETCHED = "not_fetched"
    """A web citation of a URL the round never fetched."""
    NOT_ON_PAGE = "not_on_page"
    """A web citation whose quote is not on the page the round fetched."""


Basis = Literal["content", "location", "page"]


@dataclass(frozen=True)
class Checked:
    citation: Citation | WebCitation
    verdict: Verdict
    basis: Basis
    """`content`: the lines were compared with what the citer saw. `location`: only that
    the file and lines exist — nothing recorded what they held when cited. `page`: a web
    citation's quote was looked for in what the round fetched."""
    why: str = ""
    now: Span | None = None
    """Where the cited lines are now, when they MOVED."""

    @property
    def passed(self) -> bool:
        return self.verdict in (Verdict.OK, Verdict.QUOTED)

    def render(self) -> str:
        where = ""
        if self.now is not None and isinstance(self.citation, Citation):
            where = f" (now {self.citation.path}:{self.now})"
        why = f": {self.why}" if self.why else ""
        return f"{self.citation.text} {self.verdict.value.upper()}{where}{why}"


def check(
    citation: Citation,
    *,
    read: LineSource,
    seen: SeenSource | None = None,
    denylist: Denylist | None = None,
) -> Checked:
    """One citation, through the three questions in order. The first that fails decides."""
    path = repository_path(citation.path)
    if path is None:
        return Checked(citation, Verdict.OUTSIDE, "location", "not a path in the repository")
    # The name first, refused unread: a citation names a path it may not read.
    denial = name_denied(path, denylist if denylist is not None else Denylist())
    got = read(path) if denial is None else None
    lines = None if isinstance(got, Denial) else got
    denial = got if isinstance(got, Denial) else denial
    if denial is not None:
        return Checked(citation, Verdict.SECRET, "location", f"matches {denial.pattern!r}")
    if lines is None:
        return Checked(citation, Verdict.NO_FILE, "location", f"no readable text file {path}")
    span = citation.span
    if span is None or span.end > len(lines):
        count = f"{len(lines):,} line" + ("" if len(lines) == 1 else "s")
        return Checked(citation, Verdict.OUT_OF_RANGE, "location", f"{path} has {count}")
    saw = seen(path, span) if seen is not None else None
    if saw is None:
        return Checked(citation, Verdict.OK, "location")
    if _same(lines[span.start - 1 : span.end], saw):
        return Checked(citation, Verdict.OK, "content")
    now = _find(lines, saw, near=span.start)
    if now is None:
        return Checked(citation, Verdict.CHANGED, "content", "the lines cited are gone")
    return Checked(citation, Verdict.MOVED, "content", "the lines cited moved", now)


def check_web(citation: WebCitation, *, fetched: WebSource | None) -> Checked:
    """One web citation: the round fetched its URL, and its quote is on that page."""
    page = None if fetched is None else fetched(normalise(citation.url) or citation.url)
    if page is None:
        return Checked(citation, Verdict.NOT_FETCHED, "page", "this round never fetched it")
    if _folded(citation.quote) not in _folded(page):
        return Checked(citation, Verdict.NOT_ON_PAGE, "page", "the page does not say this")
    return Checked(citation, Verdict.QUOTED, "page")


_LINK: Final = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_EMPHASIS: Final = re.compile(r"[`*_]+")
_QUOTES: Final = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})


def _folded(text: str) -> str:
    """`text` as a quote is compared: a Markdown link as its text, emphasis and code marks
    dropped, curly quotes straightened, every run of whitespace one space. The page was
    converted to Markdown (`external/fetch/text.py`); a quote is of what it says."""
    plain = _EMPHASIS.sub("", _LINK.sub(r"\1", text.translate(_QUOTES)))
    return " ".join(plain.split())


def repository_path(path: str) -> str | None:
    """The workspace-relative form, or None for anything that is not one: what a citation's
    path is checked as, and so what a read is keyed by when what it showed is kept for the
    check (`seen_in_reads`)."""
    if path.startswith(("/", "~")):
        return None
    normal = PurePosixPath(os.path.normpath(path)).as_posix()
    if normal == ".." or normal.startswith("../"):
        return None
    return normal


def _same(now: Sequence[str], saw: Sequence[str]) -> bool:
    """Equal up to trailing whitespace, which an editor may strip and no claim rests on."""
    return len(now) == len(saw) and all(
        a.rstrip() == b.rstrip() for a, b in zip(now, saw, strict=True)
    )


def _find(lines: Sequence[str], saw: Sequence[str], *, near: int) -> Span | None:
    """Where the block `saw` sits in `lines` now: the occurrence nearest to where it was
    cited, the earlier on a tie. None when it occurs nowhere."""
    if not saw:
        return None
    width = len(saw)
    first = saw[0].rstrip()
    best: int | None = None
    for i in range(len(lines) - width + 1):
        if lines[i].rstrip() != first or not _same(lines[i : i + width], saw):
            continue
        start = i + 1
        if best is None or abs(start - near) < abs(best - near):
            best = start
    return Span(best, best + width - 1) if best is not None else None


# -- the gate ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CitationGateResult:
    """The `citation_gate` journal event's three fields, and what produced them."""

    checked: tuple[Checked, ...]
    claims: int
    cited_claims: int

    @property
    def passed(self) -> bool:
        """Every claim cites (every served claim cites current evidence), at least one
        citation was checked, and every one checks out. An answer that cites nothing had
        nothing checked, and one that cites half its claims has half of them unchecked:
        neither is served as checked."""
        return (
            bool(self.checked)
            and self.cited_claims >= self.claims
            and all(c.passed for c in self.checked)
        )

    @property
    def coverage(self) -> float:
        """The fraction of claims that carry a citation."""
        return self.cited_claims / self.claims if self.claims else 0.0

    @property
    def rejected(self) -> tuple[Checked, ...]:
        return tuple(c for c in self.checked if not c.passed)

    @property
    def web(self) -> bool:
        """Whether any web citation was checked."""
        return any(c.basis == "page" for c in self.checked)

    def payload(self) -> CitationGateP:
        """The `citation_gate` event's payload. Validated by its wire name, `pass`."""
        wire = {"pass": self.passed, "checked": len(self.checked), "coverage": self.coverage}
        return CitationGateP.model_validate(wire)

    def render(self) -> str:
        """The limits line a served answer carries: what was checked, and on what basis."""
        content = sum(1 for c in self.checked if c.passed and c.basis == "content")
        location = sum(1 for c in self.checked if c.passed and c.basis == "location")
        quoted = sum(1 for c in self.checked if c.passed and c.basis == "page")
        # Said only where a web citation was checked: a file-only report reads as it did.
        web = "" if not self.web else f", {quoted} quoted from the page fetched"
        head = (
            f"[citations: {len(self.checked)} checked · {content} against the lines read, "
            f"{location} by location only{web} · {len(self.rejected)} rejected · "
            f"{self.cited_claims} of {self.claims} claims cited]"
        )
        return "\n".join((head, *(f"  {c.render()}" for c in self.rejected)))


def gate(
    answer: str,
    *,
    read: LineSource,
    seen: SeenSource | None = None,
    denylist: Denylist | None = None,
    languages: ProfileSet | None = None,
    fetched: WebSource | None = None,
) -> CitationGateResult:
    """Extract, check, and measure coverage: the whole gate, deterministically. `languages`
    is the run's profile set (the shipped one when None): a file its profiles own is code.
    `fetched` is the round's pages, for web citations; without it each one is
    `NOT_FETCHED`."""
    deny = denylist if denylist is not None else Denylist()
    cache: dict[str, Sequence[str] | Denial | None] = {}

    def once(path: str) -> Sequence[str] | Denial | None:
        if path not in cache:
            cache[path] = read(path)
        return cache[path]

    units = claims(answer)
    web = [c for i, unit in enumerate(units) for c in web_extract(unit, claim=i)]
    found = [
        c
        for i, unit in enumerate(units)
        for c in extract(_without_web(unit), claim=i)
        if _names_a_file(c.path, once, deny, languages)
    ]
    checked = (
        *(check(c, read=once, seen=seen, denylist=deny) for c in found),
        *(check_web(c, fetched=fetched) for c in web),
    )
    cited = {c.claim for c in found} | {c.claim for c in web}
    return CitationGateResult(checked, len(units), len(cited))


# -- sources ----------------------------------------------------------------------------


def workspace_lines(
    workspace: Path, *, denylist: Denylist | None = None, languages: ProfileSet | None = None
) -> LineSource:
    """Current lines of a repository path, read from disk. The `Denial` when the read rule
    refuses it, by its name or where it lands: a symlink named `notes.txt` does not make
    `.env` citable. None for a missing, unreadable or non-text file, and for one whose real
    path leaves the workspace."""
    deny = denylist if denylist is not None else Denylist()
    root = workspace.resolve()

    def read(path: str) -> Sequence[str] | Denial | None:
        real = within(root, path)
        if real is None:
            return None
        denial = read_denied(path, root / path, deny, workspace=root, real=real)
        if denial is not None:
            return denial
        data = read_file(real)
        if data is None:
            return None
        sniffed = sniff(path, data, languages)
        return sniffed.lines if sniffed.category == "text" else None

    return read


def seen_in(state: SessionState) -> SeenSource:
    """What the citer saw at the cited lines: the newest live view of that path holding all
    of them, read back from the result text the model was shown. D116 numbers every rendered
    line with its line in the file, which is what makes that exact.

    A view counts only the lines it showed whole (`views.py`), so a line shown cut is never
    compared here and a citation of it is checked on its location only: its head compared
    with the whole line now would call an unchanged line CHANGED."""

    def seen(path: str, span: Span) -> Sequence[str] | None:
        for view in reversed(state.views()):
            handle = state.handles.get(view.handle_id)
            if handle is None or handle.path != path:
                continue
            if not any(s.start <= span.start and span.end <= s.end for s in view.spans):
                continue
            line = state.result(view.call_id)
            if line is None:
                continue
            got = _lines_at(line.text, span)
            if got is not None:
                return got
        return None

    return seen


def seen_in_reads(reads: Mapping[str, Sequence[str]]) -> SeenSource:
    """What the citer saw at the cited lines, where all that is known of its reads is what
    each showed: the text of every read, oldest first, by `repository_path`. A conversation
    that is not Zeus's keeps no views (an agent's), so a line shown cut is told by its
    marker (`fit.shown_cut`), and is never compared, as `seen_in` never compares one."""

    def seen(path: str, span: Span) -> Sequence[str] | None:
        for text in reversed(reads.get(path, ())):
            got = _lines_at(text, span)
            if got is not None:
                return got
        return None

    return seen


def _lines_at(text: str, span: Span) -> tuple[str, ...] | None:
    """The lines `span` covers in a result's text, by the numbers D116 renders them with, or
    None unless it shows every one of them whole."""
    rows: dict[int, str] = {}
    for row in text.split("\n"):
        parsed = parse_row(row)
        if parsed is not None and not shown_cut(parsed[1]):
            rows.setdefault(*parsed)
    wanted = range(span.start, span.end + 1)
    if not all(n in rows for n in wanted):
        return None
    return tuple(rows[n] for n in wanted)
