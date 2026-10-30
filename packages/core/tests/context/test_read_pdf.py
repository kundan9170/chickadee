"""`read` on a PDF. The shape at its default depth; at `full`, the PDF itself on a
model that reads PDFs, its text layer under a handle of its own on one that does not, and
`NOT_SUPPORTED` when it has neither."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from context_helpers import pdf, reader_for, write_files

from chickadee.client.request import DocumentPart
from chickadee.context.attachment import AttachCaps, CheckedFile, Minted, check
from chickadee.context.reader import Reader
from chickadee.context.results import InvalidArgument, NotSupported, NotText, ToolResult
from chickadee.context.session import Session
from chickadee.context.terms import Vision

SPEC = pdf(
    "section one: the settings page is blue, with a wide margin",
    "section two: the header is gold, centred above the name",
)
READS = Vision("zeus", supports_documents=True)


def ok(result: object) -> ToolResult:
    assert isinstance(result, ToolResult), result
    return result


def test_the_default_depth_is_the_shape(session: Session, ws: Path) -> None:
    write_files(ws, {"docs/spec.pdf": SPEC})
    shown = ok(reader_for(session, ws).read("docs/spec.pdf"))
    shape = shown.handle.shape
    assert (shape.kind, shape.pages, shape.lines) == ("application/pdf", 2, None)
    assert shown.image is None
    assert "application/pdf · 2 pages" in shown.render()


def test_full_is_the_pdf_itself_on_a_model_that_reads_pdfs(session: Session, ws: Path) -> None:
    write_files(ws, {"spec.pdf": SPEC})
    shown = ok(reader_for(session, ws, vision=READS).read("spec.pdf", depth="full"))
    assert isinstance(shown.image, DocumentPart)
    assert shown.image.name == "spec.pdf"
    assert shown.image.data == SPEC


def test_full_is_the_text_layer_otherwise_and_reads_by_lines(
    session: Session, ws: Path
) -> None:
    write_files(ws, {"spec.pdf": SPEC})
    reader = reader_for(session, ws)
    shown = ok(reader.read("spec.pdf", depth="full"))
    assert shown.image is None
    assert shown.handle.kind == "output"
    assert any("text layer of spec.pdf" in n for n in shown.notes)
    again = ok(reader.read("spec.pdf", depth="full"))
    # One text layer, however often it is read.
    assert again.handle.id == shown.handle.id
    lines = ok(reader.read(shown.handle.id, lines="4-5")).render()
    assert "section two: the header is gold" in lines


def test_a_scanned_pdf_on_a_model_that_does_not_read_pdfs(session: Session, ws: Path) -> None:
    write_files(ws, {"scan.pdf": pdf("")})
    assert reader_for(session, ws).read("scan.pdf", depth="full") == NotSupported(
        "documents", "zeus"
    )


def test_a_pdf_has_no_lines_of_its_own(session: Session, ws: Path) -> None:
    write_files(ws, {"spec.pdf": SPEC})
    refused = reader_for(session, ws).read("spec.pdf", lines="1-3")
    assert isinstance(refused, InvalidArgument)


def test_a_broken_pdf_is_not_text(session: Session, ws: Path) -> None:
    write_files(ws, {"bad.pdf": b"%PDF-1.4 truncated"})
    reader = reader_for(session, ws)
    assert "[not read: the PDF could not be read" in ok(reader.read("bad.pdf")).render()
    assert isinstance(reader.read("bad.pdf", depth="full"), NotText)


def _attached(
    session: Session, ws: Path, name: str, data: bytes, caps: AttachCaps
) -> tuple[Reader, Minted]:
    reader = reader_for(session, ws, vision=READS)
    checked = check(name, data, caps)
    assert isinstance(checked, CheckedFile), checked
    return reader, reader.attach(checked, inline_bytes=caps.text_inline_bytes)


def test_an_attached_pdf_reads_back_whole_under_its_own_name(
    session: Session, ws: Path
) -> None:
    reader, minted = _attached(session, ws, "spec.pdf", SPEC, AttachCaps.defaults())
    shown = ok(reader.read(minted.handle.id, depth="full"))
    assert isinstance(shown.image, DocumentPart)
    assert shown.image.name == "spec.pdf"
    assert shown.render().startswith("spec.pdf — application/pdf")


def test_an_attached_pdf_past_the_default_pages_reads_back(session: Session, ws: Path) -> None:
    # The session's rows took 101 pages; the default rows would refuse its handle's read.
    caps = replace(AttachCaps.defaults(), pdf_pages_max=200)
    many = pdf(*(["a page of words"] * 101))
    reader, minted = _attached(session, ws, "long.pdf", many, caps)
    shown = ok(reader.read(minted.handle.id, depth="full"))
    assert isinstance(shown.image, DocumentPart)
