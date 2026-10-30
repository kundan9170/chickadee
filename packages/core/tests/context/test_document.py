"""A PDF's pages and text layer, as the parser gives them; never OCR, never a model."""

from __future__ import annotations

import pytest
from context_helpers import pdf

from chickadee.context.document import (
    PDF_PAGE_TOKENS,
    Pdf,
    PdfRefused,
    document_est,
    is_pdf,
    pdf_open,
    pdf_tokens,
    read_pdf,
)


def opened(data: bytes, *, max_pages: int = 100, min_chars: int = 32) -> Pdf:
    result = pdf_open(data, max_pages=max_pages, min_chars=min_chars)
    assert isinstance(result, Pdf), result
    return result


def test_each_page_is_marked_in_the_text_layer() -> None:
    first, second = "the first page has all of its own words here", "and so does the second one"
    doc = opened(pdf(first, second), min_chars=10)
    assert doc.pages == 2
    assert doc.text_layer == f"--- page 1 ---\n{first}\n\n--- page 2 ---\n{second}"


def test_a_scan_has_no_text_layer_by_the_average_per_page() -> None:
    assert opened(pdf("", "")).text_layer == ""
    words = "x" * 20  # 20 characters over two pages: an average of 10
    assert opened(pdf(words, ""), min_chars=10).text_layer != ""
    assert opened(pdf(words, ""), min_chars=11).text_layer == ""


def test_the_page_cap_and_a_broken_file_are_refused() -> None:
    assert pdf_open(pdf("a", "b", "c"), max_pages=2, min_chars=1) == PdfRefused(
        "the PDF has 3 pages; at most 2"
    )
    broken = pdf_open(b"%PDF-1.7 nothing more", max_pages=2, min_chars=1)
    assert isinstance(broken, PdfRefused)
    assert broken.why.startswith("the PDF could not be read")


def test_the_magic_and_the_estimates() -> None:
    assert is_pdf(pdf("x"))
    assert not is_pdf(b"plain")
    assert pdf_tokens(3, "", native=True) == 3 * PDF_PAGE_TOKENS
    assert pdf_tokens(3, "", native=False) == 0
    assert document_est(b"anything", 42) == 42
    assert document_est(pdf("a", "b")) == 4 * PDF_PAGE_TOKENS


def test_a_repository_pdf_is_held_to_the_attachment_caps() -> None:
    assert isinstance(read_pdf(pdf(*(["page"] * 101))), PdfRefused)
    assert isinstance(read_pdf(pdf("one page of plenty of words to read")), Pdf)


def test_an_accepted_pdf_is_not_held_to_the_default_caps_again() -> None:
    # An attachment the session's configured rows took: its handle reads it back.
    many = pdf(*(["page"] * 101))
    accepted = read_pdf(many, accepted=True)
    assert isinstance(accepted, Pdf)
    assert accepted.pages == 101


def test_an_aes_pdf_without_the_crypto_library_is_refused_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf
    from pypdf.errors import DependencyError

    class AesReader:
        # An owner-password PDF's reader: encrypted, with an empty user password, and AES.
        is_encrypted = True

        def __init__(self, _stream: object) -> None:
            pass

        def decrypt(self, _password: str) -> object:
            raise DependencyError("cryptography>=3.1 is required for AES algorithm")

    monkeypatch.setattr(pypdf, "PdfReader", AesReader)
    refused = pdf_open(pdf("a page"), max_pages=100, min_chars=1)
    assert isinstance(refused, PdfRefused)
    assert "AES" in refused.why
