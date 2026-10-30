"""What a file the person attaches is, decided from its bytes, under the
`turn.*` caps."""

from __future__ import annotations

import dataclasses

import pytest
from context_helpers import pdf, png

from chickadee.context.attachment import AttachCaps, CheckedFile, RefusedFile, check, check_turn
from chickadee.context.document import PDF_PAGE_TOKENS

CAPS = AttachCaps.defaults()


def ok(result: CheckedFile | RefusedFile) -> CheckedFile:
    assert isinstance(result, CheckedFile), result
    return result


def refused(result: CheckedFile | RefusedFile) -> str:
    assert isinstance(result, RefusedFile), result
    return result.why


def test_the_kind_is_the_bytes_never_the_name() -> None:
    assert ok(check("notes.png", b"just words\n", CAPS)).kind == "text"
    assert ok(check("shot.txt", png(10, 10), CAPS)).kind == "image"
    pdf_named_md = ok(check("spec.md", pdf("a page with plenty of text on it, " * 2), CAPS))
    assert pdf_named_md.kind == "pdf"


def test_an_image_is_downscaled_to_the_edge_and_priced_on_the_small_size() -> None:
    image = ok(check("wide.png", png(3000, 1000), CAPS))
    assert (image.width, image.height) == (1568, 523)
    assert image.media_type == "image/png"
    assert image.size == "1568x523"
    assert image.est_tokens == -(-1568 * 523 // 750)


def test_a_text_file_counts_its_lines() -> None:
    text = ok(check("a.log", b"one\ntwo\nthree\n", CAPS))
    assert (text.kind, text.lines, text.media_type) == ("text", 3, "text/plain")
    assert ok(check("b.txt", b"no newline", CAPS)).lines == 1


def test_a_pdf_carries_its_pages_and_text_layer() -> None:
    doc = ok(
        check("spec.pdf", pdf("first page of the spec, long enough", "second page " * 4), CAPS)
    )
    assert doc.pages == 2
    assert doc.text_layer is not None
    assert doc.text_layer.startswith("--- page 1 ---\nfirst page")
    assert doc.est_tokens >= 2 * PDF_PAGE_TOKENS
    assert doc.size == "2 pages"


def test_a_scanned_pdf_has_no_text_layer() -> None:
    assert ok(check("scan.pdf", pdf("", ""), CAPS)).text_layer == ""


@pytest.mark.parametrize(
    ("data", "why"),
    [
        (bytes(range(256)), "not an image, a PDF or a text file"),
        (b"\xff\xfe broken utf-8 \xff", "not an image, a PDF or a text file"),
        (b"has a \x00 byte", "not an image, a PDF or a text file"),
        (b"", "the file is empty"),
        (b"%PDF-1.4 not really", "the PDF could not be read"),
    ],
)
def test_what_is_refused_and_why(data: bytes, why: str) -> None:
    assert refused(check("f", data, CAPS)).startswith(why)


def test_each_cap_refuses_and_names_itself() -> None:
    small = dataclasses.replace(
        CAPS, attachment_file_max_bytes=50, text_file_max_bytes=5, pdf_pages_max=1
    )
    assert "is over 50 bytes" in refused(check("big", b"x" * 51, small))
    assert "at most 5 bytes" in refused(check("t.txt", b"123456", small))
    two = pdf("page one text " * 3, "page two text " * 3)
    roomy = dataclasses.replace(small, attachment_file_max_bytes=CAPS.attachment_file_max_bytes)
    assert refused(check("p.pdf", two, roomy)) == "the PDF has 2 pages; at most 1"


def test_a_decompression_bomb_is_refused_not_decoded() -> None:
    from PIL import Image

    pixels = Image.MAX_IMAGE_PIXELS
    assert pixels is not None
    side = int((pixels * 1.5) ** 0.5)
    assert (
        refused(check("bomb.png", png(side, side), CAPS)) == "the image is too large to decode"
    )


def test_a_turn_is_refused_whole_naming_the_file_and_its_place() -> None:
    files = [("a.txt", b"fine\n"), ("b.bin", bytes(range(256))), ("c.txt", b"fine\n")]
    result = check_turn(files, CAPS)
    assert isinstance(result, RefusedFile)
    assert (result.index, result.name) == (1, "b.bin")
    assert result.render() == "attachment 2 (b.bin): not an image, a PDF or a text file"


def test_a_turn_holds_its_count_and_total() -> None:
    tight = dataclasses.replace(CAPS, attachments_max=2, attachment_bytes_max=10)
    many = check_turn([("1", b"a"), ("2", b"b"), ("3", b"c")], tight)
    assert isinstance(many, RefusedFile)
    assert "3 attachments; at most 2 a turn" in many.why
    heavy = check_turn([("1", b"aaaaaa"), ("2", b"bbbbbb")], tight)
    assert isinstance(heavy, RefusedFile)
    assert "come to 12 bytes" in heavy.why
    fine = check_turn([("1", b"a"), ("2", b"b")], tight)
    assert isinstance(fine, tuple)
    assert [c.name for c in fine] == ["1", "2"]


def test_the_caps_are_the_registers_rows() -> None:
    from chickadee.limits import LIMITS

    rows = {lim.name: lim.value for lim in LIMITS if lim.name.startswith("turn.")}
    assert CAPS.attachments_max == rows["turn.attachments_max"]
    assert CAPS.text_inline_bytes == rows["turn.text_inline_bytes"]
    assert CAPS.pdf_pages_max == rows["turn.pdf_pages_max"]
    assert len(rows) == len(dataclasses.fields(AttachCaps)) - 1  # and the image edge
