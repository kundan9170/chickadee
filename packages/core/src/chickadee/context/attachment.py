"""What a file the person attaches is: an image, a PDF or a text file.

One check, run by the client to stage a file and by the server again before anything is
written, so the two cannot disagree. The kind is decided from the
bytes, never the name: a `.png` that is text is a text file, and a `.txt` that is a PNG is an
image. Nothing here calls a model or reads a path; the caller hands over the bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from chickadee.client.request import IMAGE_MEDIA_TYPES
from chickadee.context.document import PDF_MEDIA_TYPE, PdfRefused, is_pdf, pdf_open, pdf_tokens
from chickadee.context.handle import Handle
from chickadee.context.lines import Attachment, AttachmentKind, Delivery, attachment_size
from chickadee.context.outline import downscale, image_size, image_tokens, media_type_of
from chickadee.context.terms import IMAGE_MAX_EDGE, Vision
from chickadee.limits import LIMITS, harness_limits, tokens_for_chars

if TYPE_CHECKING:
    from chickadee.config import EngineConfig

__all__ = [
    "TEXT_MEDIA_TYPE",
    "AttachCaps",
    "AttachmentPreview",
    "CheckedFile",
    "Minted",
    "RefusedFile",
    "check",
    "check_turn",
    "delivery",
]

TEXT_MEDIA_TYPE: Final = "text/plain"

_TAKEN: Final = "not an image, a PDF or a text file"


@dataclass(frozen=True)
class AttachCaps:
    """The limits register's `turn.*` rows and the edge images are downscaled to: what a
    server takes, and what it tells a client in its hello."""

    image_max_edge: int
    attachments_max: int
    attachment_bytes_max: int
    attachment_file_max_bytes: int
    pdf_file_max_bytes: int
    pdf_pages_max: int
    pdf_text_min_chars: int
    text_file_max_bytes: int
    text_inline_bytes: int

    @staticmethod
    def defaults() -> AttachCaps:
        """The register's defaults, as `harness_limits` resolves them with nothing set."""
        return AttachCaps.of_rows(harness_limits(LIMITS, {}), IMAGE_MAX_EDGE)

    @staticmethod
    def of(config: EngineConfig) -> AttachCaps:
        """As `config` sets the rows. The edge is D110's, the same for every rung."""
        return AttachCaps.of_rows(
            {name: config.harness_limit(name) for name in _ROWS}, IMAGE_MAX_EDGE
        )

    @staticmethod
    def of_rows(rows: dict[str, int], image_max_edge: int) -> AttachCaps:
        return AttachCaps(image_max_edge, *(rows[name] for name in _ROWS))


_ROWS: Final = (
    "turn.attachments_max",
    "turn.attachment_bytes_max",
    "turn.attachment_file_max_bytes",
    "turn.pdf_file_max_bytes",
    "turn.pdf_pages_max",
    "turn.pdf_text_min_chars",
    "turn.text_file_max_bytes",
    "turn.text_inline_bytes",
)


@dataclass(frozen=True)
class CheckedFile:
    """An attachment that passed the check, as it will be sent and stored."""

    name: str
    """As the person gave it; never used to decide the kind."""
    kind: AttachmentKind
    media_type: str
    """From the bytes: an image's type, `application/pdf` or `text/plain`."""
    data: bytes = field(repr=False)
    """An image downscaled to `image_max_edge`; anything else as given."""
    est_tokens: int
    """What its content costs a model that takes it whole: the pixels, the PDF with its
    pages' images, or the text. A rung that takes less is charged less, at render."""
    width: int | None = None
    height: int | None = None
    pages: int | None = None
    lines: int | None = None
    text_layer: str | None = None
    """A PDF's text layer, `""` when it has none; None for any other kind."""

    @property
    def size(self) -> str:
        return attachment_size(self.kind, self.width, self.height, self.pages, self.lines)


@dataclass(frozen=True)
class AttachmentPreview:
    """What a file would be on this session's turn, before it is sent:
    what the check found, how Zeus's rung would be sent it and what that costs, and whether
    its text holds a secret. What a client's chip shows; nothing is recorded to make it."""

    name: str
    kind: AttachmentKind
    media_type: str
    bytes: int
    size: str
    """Pixels, pages or lines: `1568x523`, `12 pages`, `412 lines`."""
    est_tokens: int
    delivery: Delivery
    secret: bool


@dataclass(frozen=True)
class Minted:
    """An attachment with its handles minted and its blobs stored, not yet numbered: what
    `Reader.attach` makes and `Session.record_user` records."""

    checked: CheckedFile
    handle: Handle
    delivery: Delivery
    est_tokens: int
    """What its content costs as `delivery` sends it."""
    text_handle: Handle | None = None

    def preview(self, *, secret: bool) -> AttachmentPreview:
        c = self.checked
        return AttachmentPreview(
            c.name,
            c.kind,
            c.media_type,
            len(c.data),
            c.size,
            self.est_tokens,
            self.delivery,
            secret,
        )

    def text(self) -> str | None:
        """Its text, for the secret check: a text file's, a PDF's layer; None for an image."""
        c = self.checked
        if c.kind == "text":
            return c.data.decode("utf-8")
        return c.text_layer or None

    def numbered(self, attachment_id: str) -> Attachment:
        c = self.checked
        return Attachment(
            id=attachment_id,
            kind=c.kind,
            name=c.name,
            media_type=c.media_type,
            bytes=len(c.data),
            est_tokens=self.est_tokens,
            delivery=self.delivery,
            handle=self.handle,
            width=c.width,
            height=c.height,
            pages=c.pages,
            lines=c.lines,
            text_handle=self.text_handle,
        )


@dataclass(frozen=True)
class RefusedFile:
    """Why a file is not attached. `index` is its place in the turn, when a turn was
    checked."""

    name: str
    why: str
    index: int | None = None

    def render(self) -> str:
        where = f"attachment {self.index + 1} " if self.index is not None else ""
        return f"{where}({self.name}): {self.why}" if where else f"{self.name}: {self.why}"


def delivery(
    checked: CheckedFile, vision: Vision, *, inline_bytes: int, outline: str
) -> tuple[Delivery, int]:
    """How a reader on `vision`'s rung is sent `checked`, and what that costs: the image or the
    PDF itself when its model takes it; a text file's lines, or a PDF's
    text layer, whole up to `inline_bytes` and as `outline` past it; else the line alone."""
    if checked.kind == "image":
        return ("native", checked.est_tokens) if vision.supports_images else ("none", 0)
    if checked.kind == "pdf" and vision.supports_documents:
        return "native", checked.est_tokens
    text = checked.text_layer if checked.kind == "pdf" else checked.data.decode("utf-8")
    if not text:
        return "none", 0
    if len(text.encode("utf-8")) <= inline_bytes:
        return "text", tokens_for_chars(len(text))
    return "outline", tokens_for_chars(len(outline))


def check(name: str, data: bytes, caps: AttachCaps) -> CheckedFile | RefusedFile:
    """What `data` is, under `caps`, or why it is refused. The order is the design's:
    the file cap before anything decodes it, then an image, a PDF, a text file."""
    if len(data) > caps.attachment_file_max_bytes:
        return RefusedFile(
            name, f"{_mb(len(data))} is over {_mb(caps.attachment_file_max_bytes)}"
        )
    if not data:
        return RefusedFile(name, "the file is empty")
    image = _image(name, data, caps)
    if image is not None:
        return image
    if is_pdf(data):
        return _pdf(name, data, caps)
    text = _text(data)
    if text is not None:
        if len(data) > caps.text_file_max_bytes:
            return RefusedFile(
                name,
                f"a text file of {_mb(len(data))}; at most {_mb(caps.text_file_max_bytes)}",
            )
        lines = text.count("\n") + (0 if text.endswith("\n") else 1)
        return CheckedFile(
            name, "text", TEXT_MEDIA_TYPE, data, tokens_for_chars(len(text)), lines=lines
        )
    return RefusedFile(name, _TAKEN)


def check_turn(
    files: Sequence[tuple[str, bytes]], caps: AttachCaps
) -> tuple[CheckedFile, ...] | RefusedFile:
    """One turn's files: each checked, then their count and their total. The first refusal
    names its file and its place; a turn with one is refused whole."""
    if len(files) > caps.attachments_max:
        name = files[caps.attachments_max][0]
        why = f"{len(files)} attachments; at most {caps.attachments_max} a turn"
        return RefusedFile(name, why, caps.attachments_max)
    checked: list[CheckedFile] = []
    for i, (name, data) in enumerate(files):
        one = check(name, data, caps)
        if isinstance(one, RefusedFile):
            return RefusedFile(one.name, one.why, i)
        checked.append(one)
    total = sum(len(c.data) for c in checked)
    if total > caps.attachment_bytes_max:
        last = len(checked) - 1
        return RefusedFile(
            checked[last].name,
            f"the turn's attachments come to {_mb(total)}; at most "
            f"{_mb(caps.attachment_bytes_max)} a turn",
            last,
        )
    return tuple(checked)


def _image(name: str, data: bytes, caps: AttachCaps) -> CheckedFile | RefusedFile | None:
    """An image, downscaled; a refusal for one that is too large to decode; None for bytes
    that are not an image this harness sends."""
    from PIL import Image

    media = _image_media(data)
    if media is None:
        return None
    # Pillow's own pixel bound, judged from the header before anything decodes: past it the
    # image is a bomb. Not by turning its warning into an error, since the warning filters
    # are the process's and the server checks files on several threads at once.
    bomb = RefusedFile(name, "the image is too large to decode")
    try:
        size = image_size(data)
    except Image.DecompressionBombError:
        return bomb
    bound = Image.MAX_IMAGE_PIXELS
    if size is not None and bound is not None and size[0] * size[1] > bound:
        return bomb
    try:
        scaled = downscale(data, media, caps.image_max_edge)
    except Image.DecompressionBombError:
        return bomb
    if scaled is None:
        return RefusedFile(name, "the image could not be decoded")
    small, width, height, _w, _h = scaled
    return CheckedFile(
        name, "image", media, small, image_tokens(width, height), width=width, height=height
    )


def _image_media(data: bytes) -> str | None:
    media = media_type_of(data)
    return media if media in IMAGE_MEDIA_TYPES else None


def _pdf(name: str, data: bytes, caps: AttachCaps) -> CheckedFile | RefusedFile:
    if len(data) > caps.pdf_file_max_bytes:
        return RefusedFile(
            name, f"a PDF of {_mb(len(data))}; at most {_mb(caps.pdf_file_max_bytes)}"
        )
    pdf = pdf_open(data, max_pages=caps.pdf_pages_max, min_chars=caps.pdf_text_min_chars)
    if isinstance(pdf, PdfRefused):
        return RefusedFile(name, pdf.why)
    return CheckedFile(
        name,
        "pdf",
        PDF_MEDIA_TYPE,
        data,
        pdf_tokens(pdf.pages, pdf.text_layer, native=True),
        pages=pdf.pages,
        text_layer=pdf.text_layer,
    )


def _text(data: bytes) -> str | None:
    """Strict: valid UTF-8 with no NUL byte. A stray invalid byte, which `read` tolerates in a
    log, is refused here, because the person can convert the file and the model cannot."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _mb(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB".replace(".0 MB", " MB")
    if n >= 1024:
        return f"{n // 1024} KB"
    return f"{n} bytes"
