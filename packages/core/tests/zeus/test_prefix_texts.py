"""Zeus's prompt is readable from the session log. Each part's text is a session blob
under the sha the `prefix` line already names, so the line plus the blobs are the prompt;
`system_text` rebuilds the system message from them byte for byte. A session begun before
D223 gains the texts at its next turn, without a second `prefix` line."""

from __future__ import annotations

import json
from pathlib import Path

from chickadee.context.session import OtherLine, Session
from chickadee.context.store import session_dir
from chickadee.zeus.prefix import PARTS, PartName, record_prefix, system_text

from .zeus_helpers import SESSION_ID, rig, zeus_prefix


def test_the_logged_parts_rebuild_the_system_message_exactly(tmp_path: Path) -> None:
    prefix = zeus_prefix(tmp_path / "ws")
    with Session.create(tmp_path, "S-texts") as session:
        record_prefix(session, prefix)
        texts: dict[PartName, str] = {
            p: session.blob(prefix.parts[p]).decode("utf-8") for p in PARTS
        }
    assert system_text(texts) == prefix.text
    listed = json.loads(texts["tools"])["tools"]
    assert [t["name"] for t in listed] == [t.name for t in prefix.tools]


def test_a_session_from_before_gains_the_texts_and_no_second_line(tmp_path: Path) -> None:
    r = rig(tmp_path)
    r.run("one")
    blobs = session_dir(tmp_path / "sessions", SESSION_ID) / "blobs"
    shas = set(r.loop._prefix.parts.values())  # pyright: ignore[reportPrivateUsage]
    for sha in shas:
        (blobs / sha).unlink()
    r.session.close()

    reopened = Session.open(tmp_path / "sessions", SESSION_ID)
    rig(tmp_path, session=reopened, workspace=r.workspace).run("two")
    assert all((blobs / sha).is_file() for sha in shas)
    lines = [
        rec
        for _, rec in reopened.state.lines()
        if isinstance(rec, OtherLine) and rec.kind == "prefix"
    ]
    assert len(lines) == 1
    reopened.close()
