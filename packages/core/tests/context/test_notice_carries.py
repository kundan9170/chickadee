"""A notice that carries a handle: a run handed over in the background reports at
its end as a trigger-4 notice, and the diff or journal its report names is read like a
result's handle. A notice with none keeps the bytes it had."""

from __future__ import annotations

import json
from pathlib import Path

from chickadee.context.handle import Handle, Shape
from chickadee.context.session import Session
from chickadee.context.standing import notice, paused_run_ended, run_reported
from chickadee.context.store import session_dir

RUN = "2026-10-27-0000000a"
DIFF = Handle(
    id="h_00aa11bb",
    kind="diff",
    shape=Shape(bytes=30, lines=3, symbols=0, kind="diff"),
    preview="+return 2",
    ops=("read",),
    content_sha="c" * 64,
    path=None,
    run_id=RUN,
    seqs=None,
)


def _turns(root: Path, sid: str) -> list[dict[str, object]]:
    directory = session_dir(root, sid)
    out: list[dict[str, object]] = []
    for raw in (directory / "session.jsonl").read_text(encoding="utf-8").splitlines():
        line = json.loads(raw)
        if line["kind"] == "turn":
            blob = (directory / "blobs" / line["payload_sha"]).read_bytes()
            out.append(json.loads(blob))
    return out


def test_the_handle_a_run_s_report_names_is_read_like_a_result_s(tmp_path: Path) -> None:
    with Session.create(tmp_path, "S-carry") as session:
        notice(session, run_reported(RUN, "COMPLETED · run …\ndiff: h_00aa11bb", DIFF))
        assert session.state.handles.get(DIFF.id) == DIFF
    with Session.open(tmp_path, "S-carry") as again:
        assert again.state.handles.get(DIFF.id) == DIFF, "replayed from the log"
        (told,) = [t.notice for t in again.state.turns() if t.notice is not None]
        assert told is not None and (told.handle, told.carries) == (RUN, DIFF)


def test_a_notice_carrying_nothing_is_written_as_before(tmp_path: Path) -> None:
    with Session.create(tmp_path, "S-plain") as session:
        notice(session, paused_run_ended(RUN, "run ended COMPLETED"))
    (turn,) = _turns(tmp_path, "S-plain")
    assert "carries" not in turn
