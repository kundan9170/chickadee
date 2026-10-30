"""A PDF, read as the harness can read one: its pages and its text layer.

The text layer is what `pypdf` extracts from each page, under a `--- page N ---` line. It is a
parser's answer, never OCR and never a model's (the harness's own checks call no model): a
scanned PDF has no text layer, and is said to have none. Whether a model reads the PDF itself
or this text is its rung's `supports_documents` (`terms.Vision`), decided where the content is
sent, not here.

Parsing is in this process, bounded by the size and page caps the caller passes
(open point f).
"""

from __future__ import annotations

import io
import logging
import sys
from dataclasses import dataclass
from typing import Final

from chickadee.client.request import PDF_MEDIA_TYPE
from chickadee.limits import LIMITS, tokens_for_chars

__all__ = [
    "PDF_MEDIA_TYPE",
    "PDF_PAGE_TOKENS",
    "Pdf",
    "PdfRefused",
    "document_est",
    "is_pdf",
    "pdf_open",
    "pdf_tokens",
    "read_pdf",
]


PDF_PAGE_TOKENS: Final = 1600
"""What one page costs a model that reads the PDF itself, beside its text: one page image at
`image_max_edge`, since both vendors read a PDF as each page's text and image.
A starting value, like a divisor; a later measurement replaces it."""

_MAGIC: Final = b"%PDF-"


@dataclass(frozen=True)
class Pdf:
    pages: int
    text_layer: str
    """Each page's text under `--- page N ---`, or `""` when the PDF has no text layer."""


@dataclass(frozen=True)
class PdfRefused:
    why: str


def is_pdf(data: bytes) -> bool:
    """Whether the bytes claim to be a PDF. `pdf_open` decides whether they are one."""
    return data.startswith(_MAGIC)


def pdf_open(data: bytes, *, max_pages: int, min_chars: int) -> Pdf | PdfRefused:
    """The PDF's page count and text layer, or why it cannot be read. `min_chars` is the
    average of non-space characters a page must hold for the PDF to have a text layer: below
    it the PDF is a scan, and `text_layer` is `""`."""
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PyPdfError

    # pypdf logs every repair it makes to a damaged file; that is not the person's concern.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            return PdfRefused("the PDF is password-protected")
        pages = len(reader.pages)
    except DependencyError:
        # AES needs `cryptography`, which the harness does not carry: an owner-password PDF
        # (empty user password) is refused, as a password-protected one is, never a crash.
        return PdfRefused("the PDF is AES-encrypted, which this harness cannot open")
    except (PyPdfError, ValueError, KeyError, TypeError, OSError, RecursionError) as exc:
        first = (str(exc).splitlines() or [type(exc).__name__])[0]
        return PdfRefused(f"the PDF could not be read: {first}")
    if pages == 0:
        return PdfRefused("the PDF has no pages")
    if pages > max_pages:
        return PdfRefused(f"the PDF has {pages} pages; at most {max_pages}")
    texts: list[str] = []
    found = 0
    for n, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
            found += _chars(text)
        except (
            PyPdfError,
            DependencyError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            OSError,
            RecursionError,
        ):
            text = f"(page {n} could not be extracted)"
        texts.append(f"--- page {n} ---\n{text}")
    layer = "\n\n".join(texts) if found >= min_chars * pages else ""
    return Pdf(pages, layer)


def read_pdf(data: bytes, *, accepted: bool = False) -> Pdf | PdfRefused:
    """A PDF `read` opens, under the limits register's PDF rows at their defaults
    (`turn.pdf_file_max_bytes`, `turn.pdf_pages_max`, `turn.pdf_text_min_chars`). An
    `accepted` PDF is an attachment the turn's check already held to the session's configured
    rows, which may be wider: its size and pages are not judged again, or a PDF the person was
    told is attached would not read back by its own handle."""
    rows = {lim.name: int(lim.value) for lim in LIMITS if lim.name.startswith("turn.pdf_")}
    if not accepted and len(data) > rows["turn.pdf_file_max_bytes"]:
        return PdfRefused(f"a PDF over {rows['turn.pdf_file_max_bytes'] // (1024 * 1024)} MB")
    max_pages = sys.maxsize if accepted else rows["turn.pdf_pages_max"]
    return pdf_open(data, max_pages=max_pages, min_chars=rows["turn.pdf_text_min_chars"])


def _chars(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def document_est(data: bytes, known: int | None = None) -> int:
    """What one PDF part costs a model, in tokens: the estimate its handle carries when there
    is one, else twice a page image a page, the direction that cuts. The rule a request's
    window estimate counts a document by, as `outline.image_est` counts an image."""
    if known is not None:
        return known
    from pypdf import PdfReader
    from pypdf.errors import DependencyError, PyPdfError

    try:
        pages = len(PdfReader(io.BytesIO(data)).pages)
    except (
        PyPdfError,
        DependencyError,
        ValueError,
        KeyError,
        TypeError,
        OSError,
        RecursionError,
    ):
        return tokens_for_chars(len(data))
    return 2 * PDF_PAGE_TOKENS * max(pages, 1)


def pdf_tokens(pages: int, text_layer: str, *, native: bool) -> int:
    """What a PDF costs to send: its pages' images and its text when a model reads it whole,
    its text layer alone otherwise."""
    text = tokens_for_chars(len(text_layer))
    return pages * PDF_PAGE_TOKENS + text if native else text
