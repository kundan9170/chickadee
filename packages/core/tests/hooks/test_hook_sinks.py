"""The two `HookSink`s: a run's journal and Zeus's session log hold the same
`HookP`, and read back what they were given, the session's across a reopen.

What these pin: the run journal keeps a line's task and round in its envelope; the session log
keeps the whole payload, task-less; both read back in order; a `hook` session line survives the
session being closed and opened again, as resume reads it.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.context.session import Session
from chickadee.hooks.sinks import RunHookSink, SessionHookSink
from chickadee.journal.writer import Journal
from chickadee.schema.journal import HookP
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-10-24-0a1b2c3d"
LINE = HookP(event="before_stop", name="again", decision="continue", wall_ms=3, role="zeus")


def test_the_run_journal_keeps_task_and_round_in_the_envelope(tmp_path: Path) -> None:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    sink = RunHookSink(Journal.create(resolve_roots(tmp_path), RUN, {}))
    sink.hook(LINE, task_id="T1", round=2)
    (placed,) = sink.lines()
    assert (placed.payload, placed.task_id, placed.round) == (LINE, "T1", 2)
    assert sink.blob(b"why") and sink.lines()[0].payload.name == "again"


def test_the_session_log_reads_back_its_hook_lines_after_a_reopen(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session = Session.create(root, "s-hooks")
    sink = SessionHookSink(session)
    sink.hook(LINE, task_id=None, round=None)
    sink.hook(LINE.model_copy(update={"name": "other"}), task_id=None, round=None)
    text = sink.blob(b"the whole text")
    assert session.blob(text) == b"the whole text"
    session.close()
    again = Session.open(root, "s-hooks")
    try:
        assert [p.payload.name for p in SessionHookSink(again).lines()] == ["again", "other"]
    finally:
        again.close()
