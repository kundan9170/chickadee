"""`recall(query)`: have we done this before (architecture §7.9.3,
`33-tools.md` row 6; D121 for the box's records).

Memory is what survived a session, not a store of synthesized claims. Everything
recall finds is a record the session logs already keep honestly: the person's own words (a
user turn, an `ask_user` answer), the constraint ledger and the rejected alternatives, the
concerns, each `coder` episode and the run it opened, and the box's `packet_record` and
`design_record` lines. It is a keyword match over those — every term of the query must
appear — explicit and never injected: nothing here runs unless a model calls `recall`. It
reads session logs and, when attached, run journals; it calls no model and writes nothing.

**Pointers, not content (family A).** A hit is one dated line naming its record and the
session line it lives at, `[S#seq]`. Newest first, because a later record is the one most
likely to supersede an earlier. The whole list is stored behind a `findings` handle; what
fits `inline_bytes` is shown and the rest is a `Truncation` away.

**Model-authored text is marked where it appears** (*nothing enters memory that a
model authored, unless it is marked as a claim and carries the turn it came from*). A
concern renders as a claim with its raiser; a constraint names the turns it cites.

**Two words enumerate rather than search**:
`recall("concerns")` lists every concern and `recall("rejected")` every rejected alternative,
because you cannot search for what you have forgotten.

**Nothing found is `UNRESOLVED`, with what was searched and what could not be** (invariant
20). A session that would not open is named in `uncovered`, and a partial answer says it is
partial.

The lines the model can still see verbatim — the calling session's tail past its last cut —
are not recalled: recall answers what the context no longer shows.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

from pydantic import BeforeValidator, TypeAdapter, ValidationError, with_config

from chickadee.context.handle import Handle, HandleKind
from chickadee.context.lines import LINE, Concern, EpisodeLine, PayloadError, RejectedAlt
from chickadee.context.replay import ReplayCache
from chickadee.context.results import InvalidArgument, ToolResult, Truncation, UnresolvedSearch
from chickadee.context.session import (
    ASK_USER,
    CODER,
    ConstraintLine,
    OtherLine,
    Record,
    SessionState,
    ToolLine,
    Turn,
)
from chickadee.context.spans import Span
from chickadee.context.standing import turns_phrase
from chickadee.context.store import (
    SessionCorrupt,
    SessionEvent,
    SessionLog,
    SessionMissing,
    SessionStoreError,
    list_sessions,
)
from chickadee.journal.errors import JournalError
from chickadee.schema.journal import RunEndP, RunStartP

if TYPE_CHECKING:
    from chickadee.context.project import RunRecords

__all__ = [
    "ENUMERATIONS",
    "RECALL",
    "TEXT_WIDTH",
    "Hit",
    "HitKind",
    "Recall",
    "Recalled",
    "RecordHead",
    "record_head",
    "terms",
]

RECALL: Final = "recall"


# -- the box's records, as recall reads them ---------------------------------------------------
# Here, beside their one reader, rather than in `lines.py`: a box record is not a line type of
# this package, and recall only borrows five of its fields.


def _text_or_none(value: object) -> object:
    """A head field that is not text says nothing, as recall always read it: a box record
    is its writer's to validate, and recall only borrows these five fields to show a hit."""
    return value if isinstance(value, str) else None


_Head = Annotated[str | None, BeforeValidator(_text_or_none)]


@with_config(LINE)
@dataclass(frozen=True)
class RecordHead:
    """The top-level fields every box record keeps for `recall()`: what it is, how it ended,
    what it was for, and why it failed. Each box record's
    own type, beside its writer, carries them at its top level."""

    id: _Head = None
    outcome: _Head = None
    goal: _Head = None
    task: _Head = None
    failure: _Head = None


_HEAD: Final = TypeAdapter(RecordHead)


def record_head(raw: bytes) -> RecordHead:
    """A box record as `recall()` reads it, whichever record it is."""
    try:
        return _HEAD.validate_json(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        where = ".".join(str(p) for p in first["loc"]) or "<payload>"
        raise PayloadError(f"{where}: {first['msg']}") from None


HitKind = Literal[
    "turn", "answer", "constraint", "rejected", "concern", "run", "packet", "design"
]

ENUMERATIONS: Final[Mapping[str, HitKind]] = {"concerns": "concern", "rejected": "rejected"}
"""A query that is exactly one of these words lists every record of that kind."""

TEXT_WIDTH: Final = 100
"""A hit quotes at most this many characters of its record, cut with a visible `…` (D100's
rule for a title, for the same reason: the pointer is how to read the rest)."""

_DEPTH: Final = 6
"""How deep into a record's payload its strings are searched."""


def terms(query: str) -> tuple[str, ...]:
    """The query's terms, casefolded: a `"quoted phrase"` is one term, anything else splits
    on whitespace."""
    out: list[str] = []
    for i, part in enumerate(query.split('"')):
        if i % 2:
            if part.strip():
                out.append(" ".join(part.split()).casefold())
        else:
            out.extend(w.casefold() for w in part.split())
    return tuple(out)


# -- hits -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    at: str
    """When its line was written (RFC 3339) — dated, so an old answer reads as old."""
    session_id: str
    """The log the line lives in: an ancestor's, for a line a fork reads through."""
    seq: int
    kind: HitKind
    text: str
    """The rendered record, one line."""

    @property
    def pointer(self) -> str:
        return f"{self.session_id}#{self.seq}"

    def render(self) -> str:
        return f"{self.at[:10]}  {self.text}  [{self.pointer}]"


@dataclass(frozen=True)
class Recalled:
    """What one query found, with the coverage it was found under."""

    query: str
    hits: tuple[Hit, ...]
    searched: tuple[str, ...]
    uncovered: tuple[str, ...]
    """Sessions that could not be read, each with why — never skipped silently."""

    def rows(self) -> tuple[str, ...]:
        return tuple(h.render() for h in self.hits)

    def header(self) -> str:
        n = len(self.hits)
        found = f"{n} record" if n == 1 else f"{n} records"
        where = f"{len(self.searched)} session" + ("" if len(self.searched) == 1 else "s")
        return f"recall({self.query!r}) — {found} in {where}, newest first"

    def notes(self) -> tuple[str, ...]:
        if not self.uncovered:
            return ()
        return (
            f"[recall: partial — {len(self.uncovered)} session(s) could not be read and were "
            f"not searched: {', '.join(self.uncovered)}]",
        )

    def result(
        self, store: Callable[[HandleKind, bytes], Handle], inline_bytes: int
    ) -> ToolResult:
        """The full list stored behind one handle; the head of it inline, up to
        `inline_bytes` and at least one row, and a `Truncation` naming the rest."""
        rows = self.rows()
        handle = store("findings", ("\n".join(rows) + "\n").encode("utf-8"))
        kept: list[str] = []
        used = 0
        for row in rows:
            cost = len(row.encode("utf-8")) + 1
            if kept and used + cost > inline_bytes:
                break
            kept.append(row)
            used += cost
        truncation = None
        if len(kept) < len(rows):
            dropped = len(rows) - len(kept)
            shown = (Span(1, len(kept)),)
            truncation = Truncation(shown, dropped, len(rows), "byte_cap", handle)
        return ToolResult(
            handle,
            content="\n".join(kept),
            header=self.header(),
            truncation=truncation,
            notes=self.notes(),
        )


# -- the search -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    kind: HitKind
    text: str
    haystack: str


class Recall:
    """The `recall` tool over every session under one root, and — when `records` is attached
    — the run journals their episodes opened, for how each run ended and what it cost."""

    def __init__(
        self, sessions: Path, folds: ReplayCache, *, records: RunRecords | None = None
    ) -> None:
        self._sessions = sessions
        self._records = records
        # Recall runs on every `recall` call, under the calling session's lock: replaying
        # every log each time would stall the session's parallel reads behind it. `folds` is
        # the engine's one cache, over the same `sessions` root.
        self._folds = folds

    def __call__(
        self, query: str, *, current: SessionState | None = None
    ) -> Recalled | UnresolvedSearch | InvalidArgument:
        wanted = terms(query)
        if not wanted:
            return InvalidArgument(
                "query",
                "the query is empty",
                do="send what to look for in prior designs, runs and outcomes",
            )
        only = ENUMERATIONS.get(wanted[0]) if len(wanted) == 1 else None
        shown = _verbatim_tail(current)
        seen: set[tuple[str, int]] = set()
        hits: list[Hit] = []
        searched: list[str] = []
        uncovered: list[str] = []
        for sid in list_sessions(self._sessions):
            try:
                log, state = self._folds.get(sid)
                entries = list(self._entries(state, log, seen | shown))
            except (SessionStoreError, OSError) as exc:
                uncovered.append(f"{sid} ({_why(exc)})")
                continue
            searched.append(sid)
            for event, entry in entries:
                seen.add((event.session_id, event.seq))
                if only is not None:
                    if entry.kind != only:
                        continue
                elif not all(t in entry.haystack for t in wanted):
                    continue
                hits.append(Hit(event.at, event.session_id, event.seq, entry.kind, entry.text))
        if not hits:
            return UnresolvedSearch(query, tuple(searched), tuple(uncovered))
        hits.sort(key=lambda h: (h.at, h.session_id, h.seq), reverse=True)
        return Recalled(query, tuple(hits), tuple(searched), tuple(uncovered))

    def _entries(
        self, state: SessionState, log: SessionLog, skip: Set[tuple[str, int]]
    ) -> Iterator[tuple[SessionEvent, _Entry]]:
        """One entry per record line not in `skip`. A line is keyed by the log it lives in,
        so a parent's line a fork reads through is found once, as the parent's."""
        for event, record in state.lines():
            if (event.session_id, event.seq) in skip:
                continue
            entry = self._entry(state, log, event, record)
            if entry is not None:
                yield event, entry

    def _entry(
        self, state: SessionState, log: SessionLog, event: SessionEvent, record: Record
    ) -> _Entry | None:
        if isinstance(record, Turn):
            if record.role != "user":
                return None
            return _entry("turn", f"user: {_quote(record.text)}", record.text)
        if isinstance(record, ToolLine):
            call = state.call(record.call_id)
            if call is None or call.tool != ASK_USER or not record.ran:
                return None
            return _entry("answer", f"user answered: {_quote(record.text)}", record.text)
        if isinstance(record, EpisodeLine):
            return self._run(state, record)
        if isinstance(record, ConstraintLine):
            return _constraint(state, record)
        if isinstance(record, RejectedAlt):
            return _rejected(state, record)
        if isinstance(record, Concern):
            return _concern(state, record)
        if isinstance(record, OtherLine) and record.kind in ("packet_record", "design_record"):
            return _box_record(record, log.payload(event))
        return None

    def _run(self, state: SessionState, episode: EpisodeLine) -> _Entry:
        """One `coder` episode: what was asked, and — from the run's own journal when one is
        attached — how it ended and what it cost."""
        # A host's run has no turn, so no call to read its request from.
        opened = episode.opened_at_seq
        turn = state.turn(opened) if opened is not None else None
        call = next((c for c in turn.calls if c.tool == CODER), None) if turn else None
        args: Mapping[str, object] = call.args if call is not None else {}
        result = state.result(call.call_id) if call is not None else None
        if result is not None:
            said = _first_line(result.text)
        elif opened is None:
            said = "started by the host, not by a coder call"
        else:
            said = "no result recorded"
        journal = self._journal(episode.run_id)
        goal = journal.goal if journal.goal is not None else _pick(args)
        text = f"run {episode.run_id}"
        if episode.design_sha is not None:
            text += f" (design D-{episode.design_sha[:8]})"
        if goal:
            text += f": {_quote(goal)}"
        if journal.outcome:
            text += f" · {journal.outcome}"
        if not journal.ended:
            text += f" · result: {_cut(said)}"
        found = (episode.run_id, *_strings(args), said, journal.goal or "", journal.outcome)
        return _entry("run", text, " ".join(found))

    def _journal(self, run_id: str) -> _Journal:
        if self._records is None:
            return _Journal(None, "")
        try:
            run = self._records.projection(run_id)
        except FileNotFoundError:
            return _Journal(None, "outcome unknown: the run has no journal")
        except (JournalError, OSError) as exc:
            return _Journal(None, f"outcome unknown: its journal did not read ({_why(exc)})")
        start, end = run.first(RunStartP), run.first(RunEndP)
        goal = start.goal if start is not None else None
        if end is None:
            return _Journal(goal, "no RUN_END: the run did not finish")
        return _Journal(goal, f"{end.outcome} · ${end.cost_usd:,.2f}", ended=True)


@dataclass(frozen=True)
class _Journal:
    """What a run's own journal says, when one is attached. Without an ending, the `coder`
    result line in the session is what says how the run went."""

    goal: str | None
    outcome: str
    ended: bool = False


def _verbatim_tail(current: SessionState | None) -> frozenset[tuple[str, int]]:
    """The current session's lines the model already reads verbatim: its turns and tool
    lines past the boundary. Only those — a constraint, a rejection, a concern, an episode
    or a box record is never rendered into messages[], so recall is the one way to see it."""
    if current is None:
        return frozenset()
    return frozenset(
        (e.session_id, e.seq)
        for e, r in current.lines(current.boundary)
        if isinstance(r, Turn | ToolLine) and not isinstance(r, EpisodeLine)
    )


def _entry(kind: HitKind, text: str, haystack: str) -> _Entry:
    return _Entry(kind, text, f"{text} {haystack}".casefold())


def _constraint(state: SessionState, line: ConstraintLine) -> _Entry:
    c = state.ledger.get(line.constraint.id) or line.constraint
    status = ""
    if c.superseded_by is not None:
        status = f" · superseded by {c.superseded_by}"
    elif not c.live:
        status = " · no longer binding"
    source = turns_phrase(c.sources)
    text = f"constraint {c.id} ({c.author}, from {source}): {_quote(c.text)}{status}"
    return _entry("constraint", text, c.text)


def _rejected(state: SessionState, alt: RejectedAlt) -> _Entry:
    again = state.ledger.reconsiderations(alt.id)
    status = f" · reconsidered at {again[-1].seq}" if again else ""
    text = f"rejected {alt.id} ({alt.by}): {_quote(alt.text)} — {_cut(alt.reason)}{status}"
    return _entry("rejected", text, f"{alt.text} {alt.reason}")


def _concern(state: SessionState, concern: Concern) -> _Entry:
    status = ""
    promoted = state.concerns.promoted(concern.id)
    if promoted is not None:
        status = f" · watched by {promoted.check_id}"
    elif state.concerns.dismissed(concern.id) is not None:
        status = " · dismissed"
    anchor = ", ".join(concern.anchor)
    text = (
        f"concern {concern.id} (a claim, raised by {concern.raised_by}) on {anchor}: "
        f"{_quote(concern.text)}{status}"
    )
    return _entry("concern", text, f"{concern.text} {anchor}")


def _box_record(line: OtherLine, raw: bytes) -> _Entry:
    """A `packet_record` or `design_record`, read as the top-level
    fields every box record keeps for recall (`RecordHead`), and searched by every
    string it holds."""
    kind: HitKind = "design" if line.kind == "design_record" else "packet"
    head = record_head(raw)
    strings = list(_strings(json.loads(raw)))
    rid = _said(head.id) or "?"
    outcome = _said(head.outcome)
    what = _said(head.goal) or _said(head.task) or next((s for s in strings if s.strip()), None)
    text = f"{kind} {rid}" + (f" {outcome}" if outcome else "")
    if what:
        text += f": {_quote(what)}"
    failure = _said(head.failure)
    if failure:
        text += f" · {_cut(failure)}"
    return _entry(kind, text, " ".join(strings))


def _said(value: str | None) -> str | None:
    """A field that says something: present and not blank."""
    return value if value is not None and value.strip() else None


# -- text -------------------------------------------------------------------------------


def _strings(value: object, depth: int = 0) -> Iterator[str]:
    """Every string in a JSON value, to a bounded depth."""
    if isinstance(value, str):
        yield value
    elif depth < _DEPTH and isinstance(value, Mapping):
        for item in cast("Mapping[str, object]", value).values():
            yield from _strings(item, depth + 1)
    elif depth < _DEPTH and isinstance(value, list | tuple):
        for item in cast("Sequence[object]", value):
            yield from _strings(item, depth + 1)


def _pick(args: Mapping[str, object]) -> str | None:
    """What a call was about, read from its arguments exactly as the model sent them
    (invariant 19): its `goal` or `task` wherever it sits, else its first string."""
    for key in ("goal", "task"):
        found = _find_key(args, key)
        if found is not None:
            return found
    return next((s for s in _strings(args) if s.strip()), None)


def _find_key(value: object, key: str, depth: int = 0) -> str | None:
    if depth >= _DEPTH or not isinstance(value, Mapping):
        return None
    fields = cast("Mapping[str, object]", value)
    here = fields.get(key)
    if isinstance(here, str) and here.strip():
        return here
    for item in fields.values():
        found = _find_key(item, key, depth + 1)
        if found is not None:
            return found
    return None


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _cut(text: str, width: int = TEXT_WIDTH) -> str:
    line = " ".join(text.split())
    return line if len(line) <= width else f"{line[: width - 1]}…"


def _quote(text: str) -> str:
    return f'"{_cut(_first_line(text))}"'


def _why(exc: BaseException) -> str:
    if isinstance(exc, SessionCorrupt):
        at = f" at seq {exc.seq}" if exc.seq is not None else ""
        return f"corrupt{at}: {_cut(exc.detail, 60)}"
    if isinstance(exc, SessionMissing):
        return "no session log"
    return _cut(str(exc) or type(exc).__name__, 60)
