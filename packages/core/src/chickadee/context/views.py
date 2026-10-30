"""What a read shows of an opened target.

A **view** is the file lines one result shows. Reads of one file coalesce into one view under
three rules:

- only the same bytes merge: the same handle, so the same content hash — a view stitched
  from before and after an edit is a file that never existed;
- only while the earlier result is near the tail (the locality rule), because a merge
  rewrites everything after it, and that cost grows with the conversation;
- the earlier result is STUBBED and marked, never deleted, and both calls keep exactly the
  arguments they were sent (invariant 19).

A region already in context is not sent twice. Diagnostics come from the index, never a
live server.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

from chickadee.client.request import DocumentPart, ImagePart
from chickadee.context.diagnostics import DiagnosticsIndex, render_diagnostics, unavailable
from chickadee.context.document import PdfRefused, pdf_tokens, read_pdf
from chickadee.context.fit import (
    bridge,
    cap_of,
    cost_of,
    fit_cut,
    fit_lines,
    header_line,
    no_room,
    preview_text,
    render_lines,
    truncation_of,
)
from chickadee.context.handle import (
    Handle,
)
from chickadee.context.homes import ReadHome, mint_stored
from chickadee.context.opener import Opened
from chickadee.context.outline import (
    Outline,
    code_outline,
    downscale,
    fit_rows,
    image_tokens,
    markdown_outline,
    resolve_at,
    signature_spans,
)
from chickadee.context.parts import WHOLE_FILE, Parts
from chickadee.context.rereads import HELD_HEAD
from chickadee.context.results import (
    IndexUnavailable,
    InvalidArgument,
    NotSupported,
    NotText,
    ToolFailure,
    ToolResult,
    Truncation,
)
from chickadee.context.spans import (
    Span,
    coalesce,
    line_count,
    minus,
    spans_text,
)
from chickadee.context.terms import CHANGED_AT, Caps, ChangedLines, ReadAsk
from chickadee.limits import tokens_for_chars

if TYPE_CHECKING:
    from chickadee.context.session import View

__all__ = [
    "MATERIALIZE_FRACTION",
    "Views",
]


MATERIALIZE_FRACTION: Final = 0.65
"""Cap 2, cumulative per file: once progressive reads would hold this much of a file, the
remaining fragments cost more than the rest of it, so the rest is read at once — a
break-even, not a heuristic."""


class Views:
    """What a read shows of an opened target (architecture §7.4): its outline or preview at
    each depth, the lines a region names, and the merge, locality and materialize rules
    against what the home already holds. Each method answers one depth or one region; the
    Reader decides which."""

    def __init__(
        self,
        home: ReadHome,
        parts: Parts,
        *,
        diagnostics: DiagnosticsIndex | None,
        changes: ChangedLines | None,
        caps: Caps,
    ) -> None:
        self._home = home
        self._parts = parts
        self._diagnostics = diagnostics
        self._changes = changes
        self._caps = caps
        """The reader's own caps: what a held region's outline is fitted to, whatever a call
        lowered its own to."""

    def _coverage(self, opened: Opened) -> str:
        if opened.key is None:
            return "none: outside the workspace, nothing indexes it"
        if self._parts.index is None:
            return "none: no index is attached to this read"
        return "partial: this file is not in the index"

    def _outline_of(self, opened: Opened) -> Outline | IndexUnavailable:
        """The outline `at=` resolves against — the whole of it, never a fitted preview."""
        lines = opened.sniffed.lines
        if opened.sniffed.format == "code":
            symbols = self._parts.symbols(opened.key, opened.data)
            if symbols is None:
                return IndexUnavailable(opened.handle.label, self._coverage(opened))
            return code_outline(symbols, len(lines))
        if opened.sniffed.format == "markdown":
            return markdown_outline(lines)
        return Outline(())

    def read_image(self, opened: Opened, ask: ReadAsk) -> ToolResult | ToolFailure:
        """The shape at outline depth, the pixels at `full` — on a rung that can see."""
        if ask.region is not None:
            return InvalidArgument(
                "lines", "an image has no lines", do='read it again with depth="full" to see it'
            )
        if ask.at is not None:
            return InvalidArgument(
                "at", "an image has no outline entries", do='read it with depth="full" instead'
            )
        if ask.depth == "signatures":
            return InvalidArgument(
                "depth",
                "an image is read at outline or full",
                do='send depth="outline" for its type and size, or depth="full" to see it',
            )
        h = opened.handle
        if ask.depth != "full":
            return ToolResult(h, header=header_line(h), notes=opened.notes)
        if not self._parts.vision.supports_images:
            return NotSupported("images", self._parts.vision.rung)
        scaled = downscale(opened.data, h.shape.kind, self._parts.vision.image_max_edge)
        if scaled is None:
            return NotText(h.shape.kind, h.shape.bytes)
        data, width, height, _, _ = scaled
        header = (
            f"{h.label} — {h.shape.kind} · sent at {width}x{height} · "
            f"~{image_tokens(width, height):,} tokens [{h.id}]"
        )
        image = ImagePart(h.shape.kind, h.content_sha, data)
        return ToolResult(h, header=header, image=image, notes=opened.notes)

    def read_document(self, opened: Opened, ask: ReadAsk) -> ToolResult | ToolFailure:
        """A PDF's shape at outline depth; at `full`, the PDF itself on a model
        that reads PDFs, else its text layer under a handle of its own, which reads by lines
        as any text does. A PDF has no lines of its own, and no outline entries."""
        h = opened.handle
        if ask.region is not None or ask.at is not None:
            return InvalidArgument(
                "lines" if ask.region is not None else "at",
                "a PDF has no lines or outline entries of its own",
                do='read it with depth="full"; its text layer then reads by lines',
            )
        if ask.depth == "signatures":
            return InvalidArgument(
                "depth",
                "a PDF is read at outline or full",
                do='send depth="outline" for its pages, or depth="full" to read it',
            )
        if ask.depth != "full":
            return ToolResult(h, header=header_line(h), notes=opened.notes)
        pdf = read_pdf(opened.data, accepted=h.kind == "attachment")
        if isinstance(pdf, PdfRefused):
            return NotText(h.shape.kind, h.shape.bytes)
        vision = self._parts.vision
        if vision.supports_documents:
            native = pdf_tokens(pdf.pages, pdf.text_layer, native=True)
            noun = "page" if pdf.pages == 1 else "pages"
            cost = f"{pdf.pages} {noun} · ~{native:,} tokens"
            header = f"{h.label} — {h.shape.kind} · {cost} [{h.id}]"
            name = PurePosixPath(h.label).name
            document = DocumentPart(h.shape.kind, h.content_sha, name, opened.data)
            return ToolResult(h, header=header, image=document, notes=opened.notes)
        if not pdf.text_layer:
            return NotSupported("documents", vision.rung)
        layer = pdf.text_layer.encode("utf-8")
        # An attached PDF already keeps its layer under a handle of the attachment's kind
        # (`Reader.attach`): the same address, so the same handle, not a second one.
        kind = "attachment" if h.kind == "attachment" else "output"
        with self._home.lock:
            parts = self._parts.plain(layer, self._caps.inline_bytes)
            text = mint_stored(self._home, kind, layer, parts)
        note = (
            f"text layer of {h.label} [{h.id}] only: no images, no layout; "
            f'read(handle="{text.id}", lines=…) reads it by lines'
        )
        return ToolResult(text, header=header_line(text), notes=(*opened.notes, note))

    def preview(self, opened: Opened, caps: Caps) -> ToolResult:
        """`read(path)`: the skeleton and the handle, never the file — except a file
        small enough that its preview shows every line, which says so. A preview that is
        itself a view — a structureless file's head and tail — is one like any other, and
        is not sent again while it is still held."""
        h = opened.handle
        shape, outline, ops = self._parts.text(
            opened.sniffed, opened.data, opened.key, caps.whole_file, any_whole=caps.any_whole
        )
        preview = preview_text(outline, ops)
        if (shape, preview, ops) != (h.shape, h.preview, h.ops):
            h = replace(h, shape=shape, preview=preview, ops=ops)
        notes = list(opened.notes)
        content: str | None = None
        view = outline.shown
        total = len(opened.sniffed.lines)
        if total and view == (Span(1, total),):
            notes.append(WHOLE_FILE.format(n=total))
        held = self._held(h)
        if held:
            notes.append(_held_note(held, h))
            if view and not minus(view, held):
                content, view = "", ()
        return ToolResult(
            h,
            content=content,
            header=header_line(h),
            diagnostics=self._diagnostics_for(opened, None),
            notes=tuple(notes),
            view=view,
        )

    def outline_region(
        self, opened: Opened, ask: ReadAsk, caps: Caps
    ) -> ToolResult | ToolFailure:
        """`depth="outline"` on a named region: the entries that start inside it."""
        outline = self._outline_of(opened)
        if isinstance(outline, IndexUnavailable):
            return outline
        if not outline.selectable:
            return InvalidArgument(
                "depth",
                "this format has no outline to narrow",
                do='read it with lines="a-b" for the text',
            )
        wanted = self.region(opened, ask)
        if not isinstance(wanted, tuple):
            return wanted
        h = opened.handle
        region = Span(wanted[0].start, wanted[-1].end)
        rows = fit_rows(outline.within(region).rows, caps.inline_bytes) or [
            f"[no outline entries start in lines {region}]"
        ]
        return ToolResult(
            h,
            content="\n".join(rows),
            header=f"{h.label} — outline of lines {region} [{h.id}]",
            notes=opened.notes,
        )

    def region(self, opened: Opened, ask: ReadAsk) -> tuple[Span, ...] | ToolFailure:
        """The lines a read asks for, clipped to the file. Asking past the end is not a
        truncation — the file ended — so the span is clipped, not cut."""
        total = len(opened.sniffed.lines)
        if ask.at is not None:
            return self._at(opened, ask.at)
        if ask.region is not None:
            if ask.region.start > total:
                return InvalidArgument(
                    "lines",
                    f"end of file: {total:,} lines; you asked from {ask.region.start:,}",
                    do=f'ask for lines within 1-{total} (lines="1-{total}" is the whole file)',
                )
            return (Span(ask.region.start, min(ask.region.end, total)),)
        return (Span(1, total),) if total else ()

    def _at(self, opened: Opened, name: str) -> tuple[Span, ...] | ToolFailure:
        if name == CHANGED_AT:
            return self._changed(opened)
        outline = self._outline_of(opened)
        if isinstance(outline, IndexUnavailable):
            return outline
        span = resolve_at(outline, name)
        if isinstance(span, InvalidArgument):
            return span
        total = len(opened.sniffed.lines)
        if span.start > total:
            # The index describes other bytes than these; its span would name the wrong lines.
            return IndexUnavailable(
                opened.handle.label,
                f"stale: the index places {name!r} at line {span.start}, past line {total:,}",
            )
        return (Span(span.start, min(span.end, total)),)

    def _changed(self, opened: Opened) -> tuple[Span, ...] | InvalidArgument:
        found = None
        if self._changes is not None and opened.key is not None:
            found = self._changes.changed_lines(opened.key)
        if found is None:
            return InvalidArgument(
                "at",
                f"{CHANGED_AT} needs a base to compare against; none is set",
                do="read the file without at=, or with lines=",
            )
        total = len(opened.sniffed.lines)
        spans = coalesce(Span(s.start, min(s.end, total)) for s in found if s.start <= total)
        if not spans:
            return InvalidArgument(
                "at",
                f"nothing in {opened.handle.label} differs from the base",
                do="go on: the file is unchanged; read it without at= if you need its text",
            )
        return spans

    def view(
        self,
        opened: Opened,
        wanted: tuple[Span, ...],
        caps: Caps,
        *,
        again: bool = False,
        note: str | None = None,
    ) -> ToolResult | InvalidArgument:
        """The lines asked for, merged into the view already near the tail when there is one
        , and not re-sent when already held unless the reader
         decided they are to be shown `again`, saying why in `note`."""
        h = opened.handle
        lines = opened.sniffed.lines
        total = len(lines)
        if not wanted:
            return ToolResult(
                h, content="", header=f"{h.label} — empty file [{h.id}]", notes=opened.notes
            )
        held = self._held(h)
        if not again and held and not minus(wanted, held):
            return self._already_held(opened, held)
        merge = self._merge_target(h)
        prior = merge.spans if merge is not None else ()
        materialize = self._materialize(lines, prior, wanted, caps)
        if materialize:
            wanted = (Span(1, total),)
        fit = fit_lines(lines, minus(wanted, prior), caps)
        # D78 bounds what this read newly sends. The earlier view's lines are re-rendered
        # but replace the result that carried them, so they are credited, not charged; a
        # gap the merge bridges is new, so it is charged, and bridged only if it fits.
        room = caps.inline_bytes - cost_of(lines, fit.kept)
        shown = bridge(lines, (*prior, *fit.kept), room)
        credit = len(render_lines(lines, prior).encode("utf-8")) if prior else 0
        notes = list(opened.notes)
        if note is not None:
            notes.append(note)
        cut: dict[int, int] = {}
        content = render_lines(lines, shown)
        if fit.cut is not None:
            fitted = fit_cut(
                lines, fit.cut, caps, lambda c: render_lines(lines, shown, cut=c), credit=credit
            )
            if fitted is None:
                return no_room(lines, fit.cut, caps)
            cut, content = fitted
            size = len(lines[fit.cut - 1].encode("utf-8"))
            notes.append(
                f"[line {fit.cut} is {size:,} bytes, over the inline cap alone: shown "
                "cut; grep_in finds text inside it]"
            )
        header = self._view_header(h, shown, total, merge, content, materialize)
        # A line shown cut was not shown: the truncation value says so, as it would for a
        # line not shown at all.
        whole = minus(shown, [Span(n, n) for n in cut])
        return ToolResult(
            h,
            content=content,
            header=header,
            truncation=truncation_of(wanted, whole, h, total, fit.reason, caps),
            diagnostics=self._diagnostics_for(opened, shown),
            notes=tuple(notes),
            view=whole,
            supersedes=(merge.call_id,) if merge is not None else (),
        )

    @staticmethod
    def _materialize(
        lines: Sequence[str], prior: Sequence[Span], wanted: Sequence[Span], caps: Caps
    ) -> bool:
        """Cap 2 (architecture §7.4.2): progressive reads that would hold past
        `MATERIALIZE_FRACTION` of a file take the rest of it at once — but only when the rest
        fits the caps. A remainder that does not fit is no break-even, and the read serves
        what it was asked."""
        total = len(lines)
        if not prior or line_count(coalesce((*prior, *wanted))) < MATERIALIZE_FRACTION * total:
            return False
        return fit_lines(lines, minus((Span(1, total),), prior), caps).reason is None

    def _view_header(
        self,
        h: Handle,
        shown: Sequence[Span],
        total: int,
        merge: View | None,
        content: str,
        materialized: bool,
    ) -> str:
        """`path — view a-b of N`, and, for a merged view, the receipt that restores the
        spend signal merging erases: how many reads it took and what the view costs."""
        parts = [f"view {spans_text(shown)} of {total:,}"]
        if merge is not None:
            reads = 1 + self._home.reads_in(merge.call_id)
            parts.append(f"{reads} reads · ~{tokens_for_chars(len(content)):,} tokens")
        if materialized:
            parts.append(f"the rest of the file at once, past {MATERIALIZE_FRACTION:.0%} read")
        return f"{h.label} — {' · '.join(parts)} [{h.id}]"

    def _already_held(self, opened: Opened, held: Sequence[Span]) -> ToolResult:
        """The outline plus where the lines already are — the bytes are not re-sent. A
        preview that is itself a view is not sent either: that would be the same bytes."""
        h = opened.handle
        _, outline, _ = self._parts.text(
            opened.sniffed,
            opened.data,
            opened.key,
            self._caps.whole_file,
            any_whole=self._caps.any_whole,
        )
        return ToolResult(
            h,
            content="" if outline.shown else None,
            header=header_line(h),
            notes=(*opened.notes, _held_note(held, h)),
        )

    def pointer(self, opened: Opened, note: str) -> ToolResult:
        """D214 decision 10's pointer: where lines the model already holds are, and what to
        read next, instead of the lines."""
        h = opened.handle
        return ToolResult(h, content="", header=header_line(h), notes=(*opened.notes, note))

    def signatures(
        self, opened: Opened, wanted: tuple[Span, ...], caps: Caps
    ) -> ToolResult | ToolFailure:
        """`depth="signatures"`: each symbol's signature and first body line, the bodies
        elided and marked."""
        h = opened.handle
        if opened.sniffed.format != "code":
            return InvalidArgument(
                "depth",
                f"signatures are read from code; this is {h.shape.kind}",
                do='read it with depth="outline" or lines="a-b" instead',
            )
        symbols = self._parts.symbols(opened.key, opened.data)
        if symbols is None:
            return IndexUnavailable(h.label, self._coverage(opened))
        lines = opened.sniffed.lines
        if not wanted:
            return ToolResult(
                h, content="", header=f"{h.label} — empty file [{h.id}]", notes=opened.notes
            )
        region = Span(wanted[0].start, wanted[-1].end)
        spans = signature_spans(symbols, region, len(lines))
        header = f"{h.label} — signatures in lines {region} of {len(lines):,} [{h.id}]"
        if not spans:
            return ToolResult(
                h,
                content=f"[no symbols start in lines {region}]",
                header=header,
                notes=opened.notes,
            )
        fit = fit_lines(lines, spans, caps)
        notes = list(opened.notes)
        truncation = None
        if fit.reason is not None:
            last = fit.kept[-1].end
            # The region's lines are not the ask: `spans` holds the signature lines, and
            # the bodies between them were never going to be shown.
            truncation = Truncation(
                fit.kept,
                line_count(minus(spans, fit.kept)),
                len(lines),
                fit.reason,
                h,
                cap_of(caps, fit.reason),
            )
            rest = {"lines": f"{last + 1}-{region.end}", "depth": "signatures"}
            target = {"path": h.path} if h.path is not None else {"handle": h.id}
            notes.append(f"[at this depth, next: read {json.dumps({**target, **rest})}]")
        edges = region if truncation is None else Span(region.start, fit.kept[-1].end)
        cut: dict[int, int] = {}
        content = render_lines(lines, fit.kept, edges=edges)
        if fit.cut is not None:
            fitted = fit_cut(
                lines,
                fit.cut,
                caps,
                lambda c: render_lines(lines, fit.kept, cut=c, edges=edges),
            )
            if fitted is None:
                return no_room(lines, fit.cut, caps)
            cut, content = fitted
        return ToolResult(
            h,
            content=content,
            header=header,
            truncation=truncation,
            diagnostics=self._diagnostics_for(opened, fit.kept),
            notes=tuple(notes),
            view=minus(fit.kept, [Span(n, n) for n in cut]),
        )

    def _held(self, h: Handle) -> tuple[Span, ...]:
        """The lines of this handle already in context, as its home holds them."""
        return self._home.held(h.id)

    def _merge_target(self, h: Handle) -> View | None:
        """The earlier view a read of this handle may merge into, as its home says."""
        return self._home.merge_target(h.id)

    def _diagnostics_for(self, opened: Opened, shown: Sequence[Span] | None) -> tuple[str, ...]:
        """Every read of a text file carries its diagnostics, from the index. A file
        nothing analyzed renders as unavailable — never as silence."""
        h = opened.handle
        if h.kind != "file" or opened.sniffed.category != "text":
            return ()
        if opened.key is None:
            return unavailable("outside the workspace, nothing analyzes it")
        if self._diagnostics is None:
            return unavailable("no diagnostics index is attached to this read")
        name = PurePosixPath(opened.key).suffix or PurePosixPath(opened.key).name
        return render_diagnostics(
            self._diagnostics.diagnostics_in(opened.key),
            shown=shown,
            current_sha=h.content_sha,
            name=name,
        )


def _held_note(held: Sequence[Span], h: Handle) -> str:
    return f"{HELD_HEAD}{spans_text(held)} ({h.id})]"
