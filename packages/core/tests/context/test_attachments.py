"""A turn's attachments are recorded on its line, numbered per session, each
under a stored handle the fold restores."""

from __future__ import annotations

from pathlib import Path

import pytest
from context_helpers import pdf, png, reader_for, reopen

from chickadee.context.attachment import AttachCaps, CheckedFile, Minted, check
from chickadee.context.lines import PayloadError, UserTurnP, render
from chickadee.context.session import Session
from chickadee.context.terms import Vision

CAPS = AttachCaps.defaults()
SEES = Vision("zeus", supports_images=True, supports_documents=True)


def checked(name: str, data: bytes) -> CheckedFile:
    result = check(name, data, CAPS)
    assert isinstance(result, CheckedFile), result
    return result


def minted(session: Session, ws: Path, name: str, data: bytes, vision: Vision = SEES) -> Minted:
    reader = reader_for(session, ws, vision=vision)
    return reader.attach(checked(name, data), inline_bytes=CAPS.text_inline_bytes)


def test_attachments_are_numbered_per_session_and_never_reused(
    session: Session, ws: Path
) -> None:
    first = session.record_user("look", attachments=(minted(session, ws, "a.png", png(4, 4)),))
    second = session.record_user(
        "and these",
        attachments=(
            minted(session, ws, "b.txt", b"words\n"),
            minted(session, ws, "c.pdf", pdf("a page of text that is long enough")),
        ),
    )
    assert [a.id for a in first.attachments] == ["A-1"]
    assert [a.id for a in second.attachments] == ["A-2", "A-3"]
    assert [a.kind for a in second.attachments] == ["text", "pdf"]
    assert sorted(session.state.attachments) == ["A-1", "A-2", "A-3"]


def test_a_handle_and_its_bytes_come_back_on_replay(session: Session, ws: Path) -> None:
    doc = pdf("the spec's first page, with enough text to count")
    turn = session.record_user("spec", attachments=(minted(session, ws, "spec.pdf", doc),))
    a = turn.attachments[0]
    assert a.handle.kind == "attachment"
    assert a.handle.run_id is None
    assert a.text_handle is not None
    again = reopen(session)
    try:
        state = again.state
        assert state.attachments["A-1"] == a
        assert state.handles.get(a.handle.id) == a.handle
        assert state.handles.get(a.text_handle.id) == a.text_handle
        assert again.blob(a.handle.content_sha) == doc
        layer = again.blob(a.text_handle.content_sha).decode()
        assert layer.startswith("--- page 1 ---")
    finally:
        again.close()


def test_a_turn_without_attachments_keeps_its_bytes() -> None:
    assert render(UserTurnP("hello")) == render(UserTurnP("hello", attachments=()))
    assert b"attachments" not in render(UserTurnP("hello"))


def test_an_out_of_order_id_is_refused(session: Session, ws: Path) -> None:
    good = minted(session, ws, "a.txt", b"x\n").numbered("A-2")
    with pytest.raises(PayloadError, match="out of order; the next is A-1"):
        session.append(UserTurnP("skip", attachments=(good,)))


def test_delivery_is_decided_against_the_rung_when_recorded(session: Session, ws: Path) -> None:
    blind = Vision("zeus")
    image = minted(session, ws, "a.png", png(4, 4), blind)
    doc = minted(
        session,
        ws,
        "b.pdf",
        pdf("a text layer with plenty of characters on its one page"),
        blind,
    )
    scan = minted(session, ws, "c.pdf", pdf(""), blind)
    seen = minted(session, ws, "d.png", png(4, 4))
    read = minted(
        session, ws, "e.pdf", pdf("a text layer with plenty of characters on its one page")
    )
    assert (image.delivery, image.est_tokens) == ("none", 0)
    assert doc.delivery == "text"
    assert scan.delivery == "none"
    assert seen.delivery == "native"
    assert read.delivery == "native"


def test_a_long_text_file_is_sent_as_its_outline(session: Session, ws: Path) -> None:
    long = b"".join(b"line %d of the log\n" % n for n in range(5000))
    assert len(long) > CAPS.text_inline_bytes
    m = minted(session, ws, "big.log", long)
    assert m.delivery == "outline"
    assert m.est_tokens < m.checked.est_tokens


def test_a_number_an_undo_hid_is_never_given_again(session: Session, ws: Path) -> None:
    from chickadee.context.lines import UndoLine
    from chickadee.context.rewind import record_rewind

    session.record_user("first")
    session.record_assistant("ok")
    hidden = session.record_user("look", attachments=(minted(session, ws, "a.txt", b"x\n"),))
    record_rewind(session, UndoLine(hidden.seq, False, True, ()), (), ())
    assert session.state.attachments == {}
    again = session.record_user("this one", attachments=(minted(session, ws, "b.txt", b"y\n"),))
    assert [a.id for a in again.attachments] == ["A-2"]
