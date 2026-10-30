"""The open work's status rides in the turn frame: the frame carries it after its first line,
and a frame is due again only when the status
changed, so Zeus reads where the work stands before he decides, and an idle conversation pays
nothing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from chickadee.client.request import text_of
from chickadee.zeus.loop import frame_due, frame_text

from .zeus_helpers import HEAD, TODAY, rig

STATUS = (
    '<open_work key="work:docs/chickadee/x">\nplan    v1 approved · unchanged\n</open_work>'
)


def test_the_frame_carries_the_status_after_its_first_line() -> None:
    text = frame_text(date(2026, 10, 27), "abc", "main", STATUS)
    assert text == f"2026-10-27 · head abc on main\n{STATUS}"
    assert (
        frame_text(date(2026, 10, 27), "abc", "main", None) == "2026-10-27 · head abc on main"
    )


def test_a_frame_is_due_when_the_status_changes_and_only_then(tmp_path: Path) -> None:
    statuses = iter([STATUS, STATUS, STATUS.replace("v1", "v2"), STATUS.replace("v1", "v2")])
    r = rig(tmp_path, status=lambda: next(statuses))
    for n in range(4):
        r.run(f"turn {n}")
    frames = r.session.state.frames()
    assert [f.text.partition("\n")[2] for f in frames] == [STATUS, STATUS.replace("v1", "v2")]
    first = text_of(r.model.requests[0].messages[1])
    assert "<open_work" in first and HEAD[0][:12] in first


def test_a_closed_work_makes_one_frame_without_it(tmp_path: Path) -> None:
    r = rig(tmp_path, status=lambda: STATUS)
    r.run("first")
    state = r.session.state
    assert not frame_due(state, TODAY, HEAD[1], STATUS)
    assert frame_due(state, TODAY, HEAD[1], None), "the work closed: say so once"
