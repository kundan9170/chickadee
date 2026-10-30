"""Pins of the session-log lines Zeus's own code writes, before the records theme gives each a
type beside its writer.

`T-3.56` moves the `frame` line (`loop._open`), the `prefix` line (`prefix.record_prefix`) and
the `scout` line (`navigate.SessionScoutSink`) onto a typed append. These pin what they write:
the exact bytes of a frame and of a scout record, and the shape of a prefix line, whose values
are template hashes that change with every prompt edit. The `bound` line's payload is already
pinned exactly by `test_bound.py`. A later task may change these tests' setup when the code
they call changes shape, never their assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

from chickadee.context.session import Session
from chickadee.context.store import session_dir
from chickadee.schema.journal import EventKind, ScoutStartP
from chickadee.zeus.prefix import record_prefix
from chickadee.zeus.tools.navigate import SessionScoutSink

from .zeus_helpers import HEAD, SESSION_ID, answer, rig, zeus_prefix


def _blobs(root: Path, sid: str, kind: str) -> list[str]:
    directory = session_dir(root, sid)
    out: list[str] = []
    for raw in (directory / "session.jsonl").read_text(encoding="utf-8").splitlines():
        line = json.loads(raw)
        if line["kind"] == kind:
            out.append((directory / "blobs" / line["payload_sha"]).read_bytes().decode())
    return out


def test_the_frame_line_is_written_as_before(tmp_path: Path) -> None:
    r = rig(tmp_path, [answer("ok")])
    r.run("hello")
    head, branch = HEAD
    assert _blobs(tmp_path / "sessions", SESSION_ID, "frame") == [
        '{"branch":"' + branch + '","date":"2026-09-27","head":"' + head + '",'
        '"text":"2026-09-27 · head ' + head[:12] + " on " + branch + '"}'
    ]


def test_a_scout_line_is_written_as_before(tmp_path: Path) -> None:
    with Session.create(tmp_path, "S-scout") as session:
        SessionScoutSink(session).record(
            EventKind.SCOUT_START,
            ScoutStartP(rung="local-mid", goal_sha="a" * 64, snapshot_sha="b" * 64),
        )
    assert _blobs(tmp_path, "S-scout", "scout") == [
        '{"event":"scout_start","payload":{"goal_sha":"' + "a" * 64 + '","rung":"local-mid",'
        '"snapshot_sha":"' + "b" * 64 + '"}}'
    ]


def test_a_prefix_line_has_the_shape_it_had(tmp_path: Path) -> None:
    prefix = zeus_prefix(tmp_path / "ws")
    with Session.create(tmp_path, "S-prefix") as session:
        record_prefix(session, prefix)
    (blob,) = _blobs(tmp_path, "S-prefix", "prefix")
    payload = json.loads(blob)
    assert sorted(payload) == ["model_id", "parts", "sha"]
    assert payload["sha"] == prefix.sha
    assert payload["model_id"] == prefix.model_id
    assert payload["parts"] == dict(prefix.parts)
    assert sorted(payload["parts"]) == sorted(PARTS)
    assert all(isinstance(v, str) and len(v) == 64 for v in payload["parts"].values())


PARTS = ["env", "skills", "template", "tools"]
"""The prefix's four parts by name, as `main` writes them."""
