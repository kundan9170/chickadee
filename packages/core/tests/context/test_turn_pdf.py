"""An attached PDF reaches Zeus in the form his rung takes: whole on a model
that reads PDFs, as its labelled text layer otherwise, and as a line alone when it has neither.
"""

from __future__ import annotations

from pathlib import Path

from context_helpers import pdf
from test_turn_attachment_parts import BLIND, SEES, send, text

from chickadee.client.request import DocumentPart
from chickadee.context.session import Session


def test_a_pdf_whole_to_a_model_that_reads_pdfs(session: Session, ws: Path) -> None:
    m = send(
        session, ws, SEES, ("spec.pdf", pdf("section four says the settings page is blue"))
    )
    assert not isinstance(m.content, str)
    doc = m.content[-1]
    assert isinstance(doc, DocumentPart)
    assert doc.name == "spec.pdf"


def test_a_pdf_as_its_labelled_text_layer_otherwise(session: Session, ws: Path) -> None:
    m = send(
        session, ws, BLIND, ("spec.pdf", pdf("section four says the settings page is blue"))
    )
    body = text(m)
    assert "text layer only: no images, no layout" in body
    assert "1│--- page 1 ---" in body
    assert "section four says the settings page is blue" in body


def test_a_scanned_pdf_on_a_blind_rung_is_a_line_alone(session: Session, ws: Path) -> None:
    m = send(session, ws, BLIND, ("scan.pdf", pdf("")))
    assert text(m).endswith("(no text layer, and your model cannot read PDFs)")
