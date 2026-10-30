"""The Request section names the files sent with the turn the work answers,
so the person approves them with it."""

from __future__ import annotations

from pathlib import Path

from files import png

from chickadee.context.attachment import AttachCaps, CheckedFile, check
from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.zeus.delegate import (
    attached_files,
    inserted,
    request_attachments,
    request_of,
    requested,
    with_attached,
)


def test_the_request_lists_the_turns_files(tmp_path: Path) -> None:
    session = Session.create(tmp_path / "sessions", "S-test")
    try:
        reader = Reader(
            SessionHome(session),
            HandleResolver(tmp_path, session.blob),
            vision=Vision("zeus", supports_images=True),
        )
        c = check("mockup.png", png(1440, 900), AttachCaps.defaults())
        assert isinstance(c, CheckedFile)
        minted = reader.attach(c, inline_bytes=1)
        session.record_user("make it like this", attachments=(minted,))
        state = session.state
        assert request_of(state) == "make it like this"
        assert (
            requested(state)
            == "make it like this\n\nAttached: A-1 mockup.png (image, 1440x900)"
        )
        (a,) = request_attachments(state)
        (f,) = attached_files((a,))
        assert (f.id, f.name, f.sha) == ("A-1", "mockup.png", a.handle.content_sha)
        assert "Attached: A-1 mockup.png" in inserted("# T\n\nbody\n", requested(state), ())
    finally:
        session.close()


def test_words_alone_are_the_request_as_before() -> None:
    assert with_attached("just words", ()) == "just words"
