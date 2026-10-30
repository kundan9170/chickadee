"""The parts a file's handle carries: its shape, its preview, the ops it offers.

What a format affords as a preview: code's symbols, markdown's headings, a JSON or TOML file's
keys, a table's head, an image's priced size, a structureless file's head and tail, each
bounded by a budget and fitted to it. A code file whose every line fits the budget is shown
whole instead: its outline would cost a second call to read what the first could have sent.

Code's symbols are the index's while it holds the file's bytes. The box reindexes each path a
step changes (`box/steps.py`) and a landed handoff the checkout's (`session_handoff`), but
nothing does for a file changed outside the harness, such as the person's own edit in their
checkout, or one a step's tree leaves out. Such a file is parsed again here, from the bytes in
hand, by the indexer's own parser (`graph/parse.parse_source`): an outline never names a line
of bytes that are gone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Final

from chickadee.context.document import PDF_MEDIA_TYPE, Pdf, pdf_tokens, read_pdf
from chickadee.context.fit import fitted, preview_text
from chickadee.context.handle import (
    Shape,
    content_sha,
)
from chickadee.context.outline import (
    Outline,
    Sniffed,
    SymbolLike,
    code_outline,
    csv_outline,
    head_tail,
    image_tokens,
    markdown_outline,
    rows_size,
    scaled_size,
    skeleton_outline,
    stored_sniff,
    whole_file,
)
from chickadee.context.spans import (
    line_count,
)
from chickadee.context.terms import SymbolIndex, Vision
from chickadee.errors import HarnessBug
from chickadee.graph.parse import parse_source
from chickadee.profiles.loader import ProfileSet, shipped

__all__ = [
    "PREVIEW_HEAD",
    "PREVIEW_TAIL",
    "WHOLE_FILE",
    "Parts",
]


PREVIEW_HEAD: Final = 20


PREVIEW_TAIL: Final = 20
"""A structureless file's preview: its head and tail, shrunk together to fit the cap."""


_CODE_OPS: Final = (
    "read(lines=…)",
    "read(at=…)",
    'read(depth="signatures")',
    "grep_in(pattern)",
)


_MARKDOWN_OPS: Final = ("read(lines=…)", "read(at=…)", "grep_in(pattern)")


_TEXT_OPS: Final = ("read(lines=…)", "grep_in(pattern)")


_IMAGE_OPS: Final = ('read(depth="full")',)


_NOT_TEXT_PREVIEW: Final = "[not text: the shape is all a read returns]"

_DOCUMENT_OPS: Final = ('read(depth="full")',)


WHOLE_FILE: Final = "[the whole file: all {n} lines fit the preview]"
"""The first bounded exception, said rather than left to be noticed: below the preview's
size the skeleton would cost what the file costs, so the preview is the file (architecture
§7.4.1). A structureless file of at most `PREVIEW_HEAD + PREVIEW_TAIL` lines, a table of a
few rows, or a code file the index knows whose every line fits, within the inline byte cap —
never more. A box role's whole-file threshold extends it to a file of any
format at or below that threshold."""


_NO_INDEX_NOTE: Final = (
    'no index for this file, so no outline: at= and depth="signatures" need one; '
    "read(lines=…) and grep_in work"
)


def _ops(fmt: str | None, symbols: Sequence[SymbolLike] | None) -> tuple[str, ...]:
    """The ops a format offers, as its preview would list them."""
    if fmt == "code" and symbols is not None:
        return _CODE_OPS
    if fmt == "markdown":
        return _MARKDOWN_OPS
    return _TEXT_OPS


def _head_tail(lines: Sequence[str], budget: int) -> Outline:
    return head_tail(lines, PREVIEW_HEAD, PREVIEW_TAIL, budget)


class Parts:
    """The parts a file's handle carries: its shape, its preview and the ops it offers.
    An image's shape is priced at the rung's vision; code's
     preview is its symbols when the index has the file, or the file itself when it fits. The
     opener mints a handle from these, and a view re-derives them at another budget."""

    def __init__(
        self, vision: Vision, index: SymbolIndex | None, languages: ProfileSet | None = None
    ) -> None:
        self.vision = vision
        self.index = index
        self.languages = languages
        """Which grammar parses a file the index holds older bytes of: the run's set, as the
        indexer's; None is the shipped set."""

    def of(
        self,
        sniffed: Sniffed,
        data: bytes,
        index_key: str | None,
        budget: int,
        *,
        any_whole: bool = False,
    ) -> tuple[Shape, str, tuple[str, ...]]:
        if sniffed.category == "image":
            return self.image(sniffed, data)
        if sniffed.category == "document":
            pdf = read_pdf(data)
            if isinstance(pdf, Pdf):
                return self.document(data, pdf.pages, pdf.text_layer)
            preview = f"[not read: {pdf.why}]"
            return Shape(len(data), None, None, sniffed.kind), preview, ()
        if sniffed.category == "binary":
            return Shape(len(data), None, None, sniffed.kind), _NOT_TEXT_PREVIEW, ()
        shape, outline, ops = self.text(sniffed, data, index_key, budget, any_whole=any_whole)
        return shape, preview_text(outline, ops), ops

    def image(self, sniffed: Sniffed, data: bytes) -> tuple[Shape, str, tuple[str, ...]]:
        """The shape is the outline — enough to decide whether to look, and what
        looking costs, before anything is paid."""
        if sniffed.size is None:
            raise HarnessBug("an image was sniffed without its size")
        width, height = sniffed.size
        sent_w, sent_h = scaled_size(width, height, self.vision.image_max_edge)
        est = image_tokens(sent_w, sent_h)
        shape = Shape(len(data), None, None, sniffed.kind, width, height, est)
        preview = f"~{est:,} tokens to view, sent at {sent_w}x{sent_h}\nops: {_IMAGE_OPS[0]}"
        return shape, preview, _IMAGE_OPS

    def plain(self, data: bytes, budget: int) -> tuple[Shape, str, tuple[str, ...]]:
        """Stored bytes read as plain text whatever they say (`stored_sniff`): a command's
        output, a PDF's text layer."""
        return self.of(stored_sniff(data), data, None, budget)

    def document(
        self, data: bytes, pages: int, text_layer: str
    ) -> tuple[Shape, str, tuple[str, ...]]:
        """A PDF's shape is its pages and what reading it costs this rung: the PDF
        whole on a model that reads PDFs, its text layer on one that does not."""
        native = pdf_tokens(pages, text_layer, native=True)
        shape = Shape(len(data), None, None, PDF_MEDIA_TYPE, est_tokens=native, pages=pages)
        noun = "page" if pages == 1 else "pages"
        if self.vision.supports_documents:
            how = f"~{native:,} tokens to read whole"
        elif text_layer:
            text = pdf_tokens(pages, text_layer, native=False)
            how = f"your model does not read PDFs: its text layer, ~{text:,} tokens"
        else:
            how = "no text layer, and your model does not read PDFs"
        return shape, f"{pages} {noun} · {how}\nops: {_DOCUMENT_OPS[0]}", _DOCUMENT_OPS

    def text(
        self,
        sniffed: Sniffed,
        data: bytes,
        index_key: str | None,
        budget: int,
        *,
        any_whole: bool = False,
    ) -> tuple[Shape, Outline, tuple[str, ...]]:
        """The preview a format affords, bounded by `budget`, with its ops. With
        `any_whole`, a file of any format whose every line fits `budget` is the preview: the
        whole-file threshold of a role that has one. Without, only code the
        index knows is."""
        lines = sniffed.lines
        symbols = self.symbols(index_key, data) if sniffed.format == "code" else None
        shape = Shape(
            len(data), len(lines), len(symbols) if symbols is not None else None, sniffed.kind
        )
        if not lines:
            return shape, Outline(("(empty file)",)), _TEXT_OPS
        fmt = sniffed.format
        if any_whole and fmt not in ("minified", "generated"):
            every = whole_file(lines, budget)
            if every is not None:
                return shape, every, _ops(fmt, symbols)
        if fmt == "code" and symbols is not None:
            whole = whole_file(lines, budget)
            if whole is not None:
                return shape, whole, _CODE_OPS
            return (
                shape,
                fitted(code_outline(symbols, len(lines)), budget, "symbols"),
                _CODE_OPS,
            )
        if fmt == "markdown":
            return shape, fitted(markdown_outline(lines), budget, "headings"), _MARKDOWN_OPS
        if fmt in ("json", "toml"):
            skeleton = skeleton_outline("\n".join(lines), "json" if fmt == "json" else "toml")
            if skeleton is not None:
                return shape, fitted(skeleton, budget, "keys"), _TEXT_OPS
            note = f"does not parse as {fmt.upper()}: previewed as text"
            return shape, replace(_head_tail(lines, budget), note=note), _TEXT_OPS
        if fmt in ("csv", "tsv"):
            table = csv_outline(lines, "\t" if fmt == "tsv" else ",")
            if rows_size(table.rows) > budget:
                head = line_count(table.shown)
                table = replace(head_tail(lines, head, 0, budget), note=table.note)
            return shape, table, _TEXT_OPS
        if fmt in ("minified", "generated"):
            return shape, Outline((), note=f"{fmt}: no preview"), _TEXT_OPS
        preview = _head_tail(lines, budget)
        if fmt == "code":
            preview = replace(preview, note=_NO_INDEX_NOTE)
        return shape, preview, _TEXT_OPS

    def symbols(self, index_key: str | None, data: bytes) -> Sequence[SymbolLike] | None:
        """The symbols of `data`, the file's bytes as read, or None when nothing indexes the
        file — which is not "no symbols". The index's own while it holds these bytes; else
        `data` parsed as the indexer would parse it."""
        index = self.index
        if index is None or index_key is None or not index.file_known(index_key):
            return None
        if index.indexed_sha(index_key) == content_sha(data):
            return index.symbols_in(index_key)
        return self._parsed(index_key, data)

    def _parsed(self, index_key: str, data: bytes) -> Sequence[SymbolLike] | None:
        """`data`'s symbols by the grammar its language names, or None when no language or
        grammar can say: then there is no outline, rather than the index's stale one."""
        language = (self.languages or shipped()).language_of(index_key)
        if language is None:
            return None
        parsed = parse_source(index_key, data, language.grammar, language.name)
        if not parsed.parse_ok and not parsed.symbols:
            # No grammar, or nothing recovered: an empty outline would read as "no symbols".
            return None
        return parsed.symbols
