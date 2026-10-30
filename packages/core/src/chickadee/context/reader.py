"""`read` and `grep_in`: the Reader.

`read(target, lines=, at=, depth=, max_bytes=)` is the one read. With no region it returns
the file's skeleton and a handle, never the file; with a region, those lines, each numbered
as it is in the file. `depth` defaults by whether a region was named. `grep_in` searches
inside a file `read` already opened.

The Reader parses the ask and composes the rest: `opener.py` opens the target under the read
rule, `views.py` decides what it shows, `fit.py` bounds it, `homes.py` holds what it keeps.
It records nothing; the tool that answers a call records the result. Nothing here calls a
model, and nothing spawns a process but `at="@changed"`'s read-only `git diff`, through
`chickadee.process` (`changes.py`), which is what keeps the read path parallel-safe.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, Protocol

import regex as _regex

from chickadee.context.attachment import CheckedFile, Minted, delivery
from chickadee.context.diagnostics import DiagnosticsIndex
from chickadee.context.fit import cost_of, cut_room, fit_lines, header_line, render_lines
from chickadee.context.handle import (
    STORED_KINDS,
    Handle,
    HandleKind,
    Shape,
)
from chickadee.context.homes import ReadHome, Scratch, mint_stored
from chickadee.context.opener import Opened, Opener
from chickadee.context.outline import sniff
from chickadee.context.parts import Parts
from chickadee.context.pattern import (
    GREP_TIMEOUT_S,
    CompiledPattern,
    PatternTooComplex,
    compile_pattern,
    search_lines,
)
from chickadee.context.rereads import (
    RAISED,
    FileReads,
    ReadMemory,
    cover,
    next_unread,
    pointer_note,
    raise_to,
    reread,
    strikes,
    too_big,
)
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import (
    InvalidArgument,
    NotText,
    ToolFailure,
    ToolResult,
    Truncation,
    size_text,
)
from chickadee.context.spans import (
    Span,
    coalesce,
    line_count,
    minus,
    numbers,
    spans_text,
)
from chickadee.context.terms import (
    Caps,
    ChangedLines,
    ReadAsk,
    ReadRules,
    SymbolIndex,
    Vision,
    parse_ask,
)
from chickadee.context.views import Views
from chickadee.errors import HarnessBug
from chickadee.paths import Place
from chickadee.profiles.loader import ProfileSet
from chickadee.secrets.denylist import Denylist

__all__ = [
    "GREP_CONTEXT",
    "GREP_LINE_MAX",
    "ReadAids",
    "Reader",
]


GREP_CONTEXT: Final = 2


GREP_LINE_MAX: Final = 500
"""A `grep_in` line longer than this is shown cut and marked: a hit in a minified file must
not be a whole-file read wearing a costume."""


class ReadAids(SymbolIndex, DiagnosticsIndex, Protocol):
    """`graph.Graph`, structurally: the index and the diagnostics over one connection."""


class Reader:
    """`read` and `grep_in` over a home.

    It mints handles into its home's table, keeps the bytes a handle names there, and asks its
    home what is already held and what a read may merge into. It records nothing: the tool that
    answers a call records the result (Zeus's sink, `zeus/calls.py`), in the same hold of the
    home's lock.
    A handle `store()` mints, or a read mints, lasts beyond this process only when the result
    that carries it is recorded — replay rebuilds the table from result lines.

    Zeus's reader is over its session (`SessionHome`); `scratch()` makes one over a `Scratch`
    for another conversation, sharing the deny list and caps."""

    def __init__(
        self,
        home: ReadHome,
        resolver: HandleResolver,
        *,
        vision: Vision,
        index: SymbolIndex | None = None,
        diagnostics: DiagnosticsIndex | None = None,
        changes: ChangedLines | None = None,
        denylist: Denylist | None = None,
        rules: ReadRules | None = None,
        languages: ProfileSet | None = None,
    ) -> None:
        self._home = home
        self._rules = rules if rules is not None else ReadRules(Caps())
        """The role's own read limits: its caps, how a re-read is answered,
        and when a file's cap is raised."""
        self._memory = ReadMemory()
        self._resolver = resolver
        self._index = index
        self._diagnostics = diagnostics
        self._changes = changes
        self._denylist = denylist if denylist is not None else Denylist()
        self._caps = self._rules.caps
        self._parts = Parts(vision, index, languages)
        self._languages = languages
        """The run's profiles, with the repository's overrides: which files a preview labels as
        code. None is the shipped set, for a reader built with no run around it."""
        self._opener = Opener(
            home,
            resolver,
            self._parts,
            denylist=self._denylist,
            caps=self._caps,
            languages=languages,
        )
        self._views = Views(
            home, self._parts, diagnostics=diagnostics, changes=changes, caps=self._caps
        )

    def scratch(
        self, *, vision: Vision, aids: ReadAids | Literal["shared"] | None = "shared"
    ) -> Reader:
        """A reader of its own for a conversation that is not Zeus's: the scout's, one per
        `locate`, and an agent's.

        `aids` is the index and diagnostics: "shared" is this reader's; a reader used on
        another thread (an agent's, Amendment 1 decision 1) passes a graph connection of its
        own, or None for no index, since a SQLite connection is the thread's that opened it.

        Its handles and bytes are a new `Scratch`, held in memory and dropped with it: nothing
        of that conversation reaches the session, in memory or on disk, and it answers from no
        context but its own. What it shares is what keeps it safe: the deny list itself,
        the caps, the workspace, and the read-only aids (index, diagnostics, changed lines). Its
        `vision` is its own rung's."""
        own = Scratch()
        return Reader(
            own,
            HandleResolver(self._resolver.workspace, own.blob),
            vision=vision,
            index=self._index if aids == "shared" else aids,
            diagnostics=self._diagnostics if aids == "shared" else aids,
            changes=self._changes,
            denylist=self._denylist,
            rules=ReadRules(self._caps),
            languages=self._languages,
        )

    def read(
        self,
        target: str | Path | Handle,
        *,
        lines: Span | str | None = None,
        at: str | None = None,
        depth: str | None = None,
        max_bytes: int | None = None,
        call_id: str | None = None,
    ) -> ToolResult | ToolFailure:
        """The signature. `target` is a path (opens it) or a handle (reuses it); `lines`
        takes the tool's `start-end` form or a `Span`. `call_id` is the call this read
        answers, so a box role's home can say later which call showed which lines."""
        ask = parse_ask(lines, at, depth)
        if isinstance(ask, InvalidArgument):
            return ask
        if max_bytes is not None and max_bytes < 1:
            return InvalidArgument(
                "max_bytes", "a cap is at least one byte", do="send it again without max_bytes"
            )
        with self._home.lock:
            opened = self._opener.open(target)
            if not isinstance(opened, Opened):
                return opened
            self._memory.at(self._home.conversation())
            got, notes = self._read(opened, ask, max_bytes)
            if isinstance(got, ToolResult) and call_id is not None:
                self._home.served(call_id, got.handle, got.view)
        if not isinstance(got, ToolResult):
            return got
        return got.with_notes(*notes, *self._said(opened, ask, max_bytes))

    def repeat_key(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
    ) -> tuple[str, str, str] | None:
        """What a read asks, as a repetition counts it: the file, its bytes
        and the span, so one span asked again in other words is one span asked again. Lines
        already held count as the view that holds them. None when the read would fail, which
        then counts by its words."""
        ask = parse_ask(lines, at, depth)
        if isinstance(ask, InvalidArgument):
            return None
        with self._home.lock:
            opened = self._opener.open(target, mint=False)
            if not isinstance(opened, Opened):
                return None
            h = opened.handle
            depth_ = ask.depth or ("full" if ask.named else "outline")
            if opened.sniffed.category != "text" or depth_ != "full":
                return (h.label, h.content_sha, f"{depth_}:{ask.at or ask.region or ''}")
            wanted = self._views.region(opened, ask)
            if not isinstance(wanted, tuple):
                return None
            held = cover(self._home.held_views(h.id), wanted)
            spans = held.spans if held is not None else wanted
            return (h.label, h.content_sha, spans_text(spans))

    def _said(self, opened: Opened, ask: ReadAsk, max_bytes: int | None) -> tuple[str, ...]:
        """What the read did not do as asked, said: the file ended before
        the lines asked for, and `max_bytes` above the cap changed nothing."""
        notes: list[str] = []
        total = len(opened.sniffed.lines)
        if (
            ask.region is not None
            and ask.region.end > total
            and opened.sniffed.category == "text"
        ):
            notes.append(f"[end of file: {total:,} lines; you asked up to {ask.region.end:,}]")
        if max_bytes is not None and max_bytes > self._caps.inline_bytes:
            cap = size_text(self._caps.inline_bytes)
            notes.append(f"[max_bytes only lowers the limit; it stays {cap}]")
        return tuple(notes)

    def place(self, sent: str) -> Place | None:
        """Where a path a model sent lands in the workspace, or the one it most likely meant
        (`Opener.place`)."""
        return self._opener.place(sent)

    def grep_in(
        self, handle: str | Handle, pattern: str, *, context: int = GREP_CONTEXT
    ) -> ToolResult | ToolFailure:
        """Search inside a file `read` already opened. Every hit is numbered as it is in the
        file, so it is addressable as a slice; bounded by `result_cap` lines, with the
        total named past it. A search of the whole file that finds nothing says so — its
        coverage was total, so the empty answer is a fact."""
        if context < 0:
            return InvalidArgument(
                "context", "context is zero or more lines", do="send context as 0 or more"
            )
        if not pattern:
            # An empty pattern matches every line: never what was meant, and the whole file
            # returned as "matches". The reader refuses it whoever calls it.
            return InvalidArgument(
                "pattern",
                "the pattern is empty, which matches every line",
                do="send a regular expression for the lines you want",
            )
        try:
            regex = compile_pattern(pattern)
        except (re.error, _regex.error) as exc:
            return InvalidArgument(
                "pattern",
                f"not a regular expression: {exc}",
                do="fix it, or escape its special characters to search for the text as written",
            )
        except PatternTooComplex as exc:
            return InvalidArgument(
                "pattern",
                f"refused: {exc}",
                do="send a simpler pattern: a literal word or two from the line you want",
            )
        with self._home.lock:
            opened = self._opener.open_existing(handle)
            if not isinstance(opened, Opened):
                return opened
            if opened.sniffed.category != "text":
                return NotText(opened.handle.shape.kind, opened.handle.shape.bytes)
            return self._grep(opened, regex, context)

    def text_of(self, handle: str) -> str | ToolFailure:
        """A handle this reader's home holds, as its whole text: what a handle passed to an
        agent carries (`spawn_agent`). Opened as `read` opens it, under the
        read rule and with a stale file's note, and minting nothing. Not text is `NotText`."""
        with self._home.lock:
            held = self._home.handles().get(handle)
            if held is None:
                return self._home.never_minted(handle)
            opened = self._opener.open(held, mint=False)
        if not isinstance(opened, Opened):
            return opened
        if opened.sniffed.category != "text":
            return NotText(opened.handle.shape.kind, opened.handle.shape.bytes)
        text = opened.data.decode("utf-8", "replace")
        return "\n".join((*opened.notes, text)) if opened.notes else text

    def attach(self, checked: CheckedFile, *, inline_bytes: int) -> Minted:
        """A file the person sent: its bytes, and a PDF's text layer, kept in
        this reader's home, each under a stored handle of kind `attachment`, and how this
        reader's rung is sent it. A text file is outlined by its name, unlike stored output,
        so `parse.py` gets a Python outline. Numbered when its turn is recorded."""
        with self._home.lock:
            handle = mint_stored(
                self._home,
                "attachment",
                checked.data,
                self._attachment_parts(checked),
                name=checked.name,
            )
            text_handle = None
            if checked.kind == "pdf" and checked.text_layer:
                layer = checked.text_layer.encode("utf-8")
                parts = self._parts.plain(layer, self._caps.inline_bytes)
                text_handle = mint_stored(self._home, "attachment", layer, parts)
        content = text_handle if text_handle is not None else handle
        how, est = delivery(
            checked, self._parts.vision, inline_bytes=inline_bytes, outline=content.preview
        )
        return Minted(checked, handle, how, est, text_handle)

    def _attachment_parts(self, checked: CheckedFile) -> tuple[Shape, str, tuple[str, ...]]:
        data = checked.data
        if checked.kind == "pdf":
            return self._parts.document(data, checked.pages or 0, checked.text_layer or "")
        if checked.kind == "image":
            return self._parts.of(sniff("", data), data, None, self._caps.inline_bytes)
        sniffed = sniff(checked.name, data, self._languages)
        return self._parts.of(sniffed, data, None, self._caps.inline_bytes)

    def store(self, kind: HandleKind, data: bytes) -> Handle:
        """The handle for bytes that existed nowhere else — a command's output, a diff, a log.
        The reader's home keeps them and the handle names the blob,
         so reading it back is exact for as long as the home lives: the session's life for
         Zeus's reader, one `locate` for a scratch."""
        if kind not in STORED_KINDS:
            raise ValueError(f"{kind!r} is not stored: a file handle is an address")
        with self._home.lock:
            return mint_stored(
                self._home, kind, data, self._parts.plain(data, self._caps.inline_bytes)
            )

    def _read(
        self, opened: Opened, ask: ReadAsk, max_bytes: int | None
    ) -> tuple[ToolResult | ToolFailure, tuple[str, ...]]:
        """The read, and the notes it earns beside its own: a raise, or why none was given."""
        category = opened.sniffed.category
        if category == "image":
            return self._views.read_image(opened, ask), ()
        if category == "document":
            return self._views.read_document(opened, ask), ()
        if category == "binary":
            if ask.named or ask.depth in ("signatures", "full"):
                return NotText(opened.handle.shape.kind, opened.handle.shape.bytes), ()
            return ToolResult(
                opened.handle, header=header_line(opened.handle), notes=opened.notes
            ), ()
        reads = self._memory.file(opened.handle.label)
        caps = self._caps
        if reads.raised is not None:
            caps = caps.raised(*reads.raised)
        lowered = caps.lowered(max_bytes)
        if isinstance(lowered, InvalidArgument):
            return lowered, ()
        depth = ask.depth or ("full" if ask.named else "outline")
        if depth == "outline":
            self._memory.moved_on()
            if ask.named:
                return self._views.outline_region(opened, ask, lowered), ()
            return self._views.preview(opened, lowered), ()
        wanted = self._views.region(opened, ask)
        if not isinstance(wanted, tuple):
            return wanted, ()
        if depth == "signatures":
            self._memory.moved_on()
            return self._views.signatures(opened, wanted, lowered), ()
        return self._full(opened, wanted, caps, reads, max_bytes)

    def _full(
        self,
        opened: Opened,
        wanted: tuple[Span, ...],
        base: Caps,
        reads: FileReads,
        max_bytes: int | None,
    ) -> tuple[ToolResult | ToolFailure, tuple[str, ...]]:
        """Lines, through D214's two rules: a file's cap raised once the model has clearly
        asked for it in one view (decision 11), and a re-read answered by what the model can
        still use (decision 10). `base` is the role's cap for this file, raised or not;
        `max_bytes` lowers it last, so it only ever lowers, whenever a raise came."""
        h = opened.handle
        lines = opened.sniffed.lines
        notes: list[str] = []
        raise_ = self._rules.raise_
        raised_now = False
        # A read that fits the cap needs no raise, so it earns no strike: and one that does
        # not earns at most one, whether it restarted, sent `max_bytes` above the cap, or both.
        cut = fit_lines(lines, wanted, base).reason is not None
        if reads.raised is None and raise_.strikes > 0 and cut:
            over = max_bytes is not None and max_bytes > self._caps.inline_bytes
            if over or strikes(reads, wanted, fits=False, held=self._home.held(h.id)):
                reads.strikes += 1
            if reads.strikes >= raise_.strikes:
                raised = raise_to(cost_of(lines, wanted), raise_, self._home.room())
                if raised is None:
                    notes.append(too_big(len(opened.data), base.inline_bytes))
                else:
                    reads.raised = (raised, line_count(wanted))
                    base = base.raised(*reads.raised)
                    notes.append(RAISED.format(n=reads.strikes))
                    raised_now = True
        caps = base.lowered(max_bytes)
        if isinstance(caps, InvalidArgument):
            return caps, ()
        got = self._answer(opened, wanted, caps, raised=raised_now)
        if isinstance(got, ToolResult) and got.view:
            reads.shown.append(got.view)
        return got, tuple(notes)

    def _answer(
        self, opened: Opened, wanted: tuple[Span, ...], caps: Caps, *, raised: bool
    ) -> ToolResult | ToolFailure:
        """The re-read ladder (decision 10) over a full-depth read. A role with no held rows
        answers held lines as in older versions: their outline and where they are."""
        h = opened.handle
        held_rules = self._rules.held
        if held_rules is None or raised:
            self._memory.moved_on()
            return self._views.view(opened, wanted, caps, again=raised)
        answer, view = reread(self._home.held_views(h.id), wanted, held_rules)
        if answer == "show":
            self._memory.moved_on()
            changed = None if opened.key is None else self._home.changed_since(opened.key, h.id)
            note = None if changed is None else f"[changed since your view at {changed}]"
            return self._views.view(opened, wanted, caps, note=note)
        if view is None:
            raise HarnessBug("a held answer names the view it holds")
        if answer == "again":
            self._memory.moved_on()
            note = f"[shown again: your view of these lines at {view.named} is old]"
            return self._views.view(opened, wanted, caps, again=True, note=note)
        if self._memory.insists(h.id, view, held_rules.insist):
            note = "[shown again, as you asked for them again]"
            return self._views.view(opened, wanted, caps, again=True, note=note)
        total = len(opened.sniffed.lines)
        nxt = next_unread(self._home.held(h.id), view.spans[-1], total, line_count(wanted))
        return self._views.pointer(opened, pointer_note(h.id, view, nxt, total))

    def _grep(
        self, opened: Opened, regex: CompiledPattern, context: int
    ) -> ToolResult | InvalidArgument:
        h = opened.handle
        lines = opened.sniffed.lines
        total = len(lines)
        hits = search_lines(regex, lines)
        if hits is None:
            return InvalidArgument(
                "pattern",
                f"the search did not finish in {GREP_TIMEOUT_S:g} s; a pattern that nests "
                f"quantifiers backtracks without end",
                do="send a more specific pattern, without nested quantifiers such as (a+)+",
            )
        shown_pattern = regex.pattern if len(regex.pattern) <= 80 else regex.pattern[:79] + "…"
        if not hits:
            header = (
                f"{h.label} — no match for /{shown_pattern}/ in all {total:,} lines [{h.id}]"
            )
            return ToolResult(h, content="", header=header, notes=opened.notes)
        cap = self._caps.result_lines
        width = min(context, max(cap - 1, 0) // 2)

        def room(n: int) -> int:
            # A matching line shows at most `GREP_LINE_MAX` bytes, and never more than its
            # row can carry inside the cap alone: the first match always shows, so a
            # search progresses, and a fixed cut would carry it past a cap set lower.
            return min(GREP_LINE_MAX, max(cut_room(self._caps, total, lines[n - 1]), 0))

        def long_in(spans: Sequence[Span]) -> dict[int, int]:
            return {
                n: room(n)
                for n in numbers(spans)
                if len(lines[n - 1].encode("utf-8")) > room(n)
            }

        def cost(spans: Sequence[Span]) -> int:
            # The exact rendered bytes — numbers, cut markers, gap markers — since a search
            # is bounded by D78's byte cap too, not only by its line count.
            return len(render_lines(lines, spans, cut=long_in(spans)).encode("utf-8"))

        shown: tuple[Span, ...] = ()
        kept = 0
        for n in hits:
            grown = coalesce((*shown, Span(max(1, n - width), min(total, n + width))))
            if kept == 0 and cost(grown) > self._caps.inline_bytes:
                # The first match always shows, so a search progresses; without its
                # context if the context alone is over the cap.
                grown = (Span(n, n),)
            elif line_count(grown) > cap or cost(grown) > self._caps.inline_bytes:
                break
            shown, kept = grown, kept + 1
        truncation = None
        if kept < len(hits):
            truncation = Truncation(
                (Span(hits[0], hits[kept - 1]),), len(hits) - kept, len(hits), "result_cap", h
            )
        long = long_in(shown)
        at = ", ".join(str(n) for n in hits[: min(kept, 12)]) + (", …" if kept > 12 else "")
        noun = "match" if len(hits) == 1 else "matches"
        found = f"{len(hits):,} {noun} of /{shown_pattern}/ in {total:,} lines · at {at}"
        header = f"{h.label} — {found} [{h.id}]"
        return ToolResult(
            h,
            content=render_lines(lines, shown, cut=long),
            header=header,
            truncation=truncation,
            notes=opened.notes,
            view=minus(shown, [Span(n, n) for n in long]),
        )
