"""The files sent with the turn Zeus hands over go with the work. The person
approves them with it, the run's request names them, and each box role's instructions list
them with a handle on its own reader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import SID, build, make_repo, zeus_delegates
from files import pdf, png
from zeus_turn_helpers import WAIT_S, state

from chickadee.box.compose import NOT_SHOWN, attachments_block
from chickadee.box.request import AttachedFile
from chickadee.client.request import text_of
from chickadee.context.attachment import AttachCaps
from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.engine import PacketEngine
from chickadee.journal.errors import BlobMissing

NOTES = b"use two, not one\n"


@pytest.fixture
def session(tmp_path: Path) -> Iterator[Session]:
    opened = Session.create(tmp_path / "sessions", "S-test")
    yield opened
    opened.close()


def reader_for(session: Session, ws: Path) -> Reader:
    return Reader(SessionHome(session), HandleResolver(ws, session.blob), vision=Vision("box"))


def test_a_handed_over_turns_files_reach_both_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo = make_repo(tmp_path / "repo")
    engine, model = build(monkeypatch, repo, raw, zeus_delegates())
    try:
        handle = engine.turn(SID, "edit it", attachments=[("notes.txt", NOTES)])
        handle.wait(timeout=WAIT_S)
        attached = state(engine, SID).attachments["A-1"]
        sent = [r for r in model.requests if any(m.role == "system" for m in r.messages)]
        systems = [text_of(r.messages[0]) for r in sent]
        heads = [s for s in systems if "Attachments the person sent with this work:" in s]
        assert heads, "a box role's instructions list the file"
        assert all("- A-1 notes.txt · text · 1 line · read(handle=" in h for h in heads)
        users = [text_of(m) for r in model.requests for m in r.messages if m.role == "user"]
        assert any("Attached: A-1 notes.txt (text, 1 line)" in u for u in users), (
            "the brief approved at stop 2 names the file"
        )
        assert attached.handle.kind == "attachment"
    finally:
        engine.close()


def _files(session: Session, *named: tuple[str, bytes]) -> tuple[AttachedFile, ...]:
    return tuple(
        AttachedFile(id=f"A-{n}", name=name, sha=session.put_blob(data))
        for n, (name, data) in enumerate(named, 1)
    )


def test_each_kind_is_offered_as_a_text_tool_can_read_it(
    session: Session, tmp_path: Path
) -> None:
    ws = tmp_path
    files = _files(
        session,
        ("shot.png", png(30, 20)),
        ("spec.pdf", pdf("section four says the settings page is blue, centred")),
        ("scan.pdf", pdf("")),
        ("notes.txt", NOTES),
    )
    reader = reader_for(session, ws)
    block = attachments_block(files, session.blob, AttachCaps.defaults(), reader)
    rows = block.strip().splitlines()
    assert rows[0] == "Attachments the person sent with this work:"
    assert rows[1] == f"- A-1 shot.png · image · 30x20 · {NOT_SHOWN}"
    assert rows[2].startswith(
        '- A-2 spec.pdf · pdf · 1 page · text layer only: read(handle="h_'
    )
    assert rows[3] == f"- A-3 scan.pdf · pdf · 1 page · no text layer: {NOT_SHOWN}"
    assert rows[4].startswith('- A-4 notes.txt · text · 1 line · read(handle="h_')
    text_id = rows[4].split('"')[1]
    shown = reader.read(text_id, lines="1-1")
    assert "use two, not one" in shown.render()  # type: ignore[union-attr]


def test_the_same_files_are_the_same_block_on_every_reader(
    session: Session, tmp_path: Path
) -> None:
    ws = tmp_path
    files = _files(session, ("notes.txt", NOTES))
    one = attachments_block(files, session.blob, None, reader_for(session, ws))
    two = attachments_block(files, session.blob, None, reader_for(session, ws))
    assert one == two


def test_a_file_the_run_cannot_reach_says_so(session: Session, tmp_path: Path) -> None:
    ws = tmp_path
    files = (AttachedFile(id="A-1", name="gone.txt", sha="f" * 64),)

    def missing(sha: str) -> bytes:
        raise BlobMissing(sha)

    reader = reader_for(session, ws)
    assert "its bytes are gone" in attachments_block(files, missing, None, reader)
    assert "it has no session" in attachments_block(files, None, None, reader)
    assert attachments_block((), None, None, reader) == ""


def test_an_engine_hands_nothing_over_when_nothing_was_attached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raw: dict[str, Any]
) -> None:
    repo = make_repo(tmp_path / "repo")
    engine: PacketEngine
    engine, model = build(monkeypatch, repo, raw, zeus_delegates())
    try:
        engine.turn(SID, "edit it").wait(timeout=WAIT_S)
        systems = [text_of(r.messages[0]) for r in model.requests if r.messages]
        assert not any("Attachments the person sent" in s for s in systems)
    finally:
        engine.close()
