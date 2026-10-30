"""Every record a client can be sent reads as its kind's type (`records.payload`): each event
of the recorded run (`serve/tests/recorded_run.py`) and each line of core's
recorded session log; a live session's served records are read in `test_wire_parity.py`.
Recorded by core's own writers, so a payload the generated types cannot read is a contract the
client and the server disagree about. The packet loop's recorded journals are v1, which no
reader takes since."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import RUN_RECORDS, SESSION_RECORDS
from chickadee_wire.records import is_run_record, payload

PACKAGES = Path(__file__).resolve().parents[2]
CORE_FIXTURES = PACKAGES / "core" / "tests" / "fixtures"
RUN_JOURNALS = [PACKAGES / "serve" / "tests" / "fixtures" / "run" / "journal.jsonl"]
SESSION_LINES = CORE_FIXTURES / "records" / "session_lines.jsonl"


def _run_frames(path: Path) -> list[RecordFrame]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        RecordFrame(
            scope=e["run_id"], seq=e["seq"], at=e["ts"], kind=e["kind"], run_id=e["run_id"],
            payload=e["payload"],
        )
        for e in events
    ]  # fmt: skip


@pytest.mark.parametrize("journal", RUN_JOURNALS, ids=lambda p: p.parent.name)
def test_every_recorded_run_event_reads_as_its_kind(journal: Path) -> None:
    for frame in _run_frames(journal):
        assert is_run_record(frame)
        typed = payload(frame)
        assert typed is not None, f"{frame.kind} has no type"
        assert type(typed) is type(RUN_RECORDS[frame.kind].validate_python(frame.payload))


TASK_LOOP_KINDS = {
    "run_start",
    "plan_written",
    "task_delegated",
    "model_call",
    "tool_call",
    "effect_intended",
    "effect_started",
    "effect_completed",
    "exec",
    "task_reported",
    "task_reviewed",
    "run_end",
}
"""What a task run's journal holds: every kind a frontend's task views read."""


def test_the_recorded_run_covers_the_task_loops_kinds() -> None:
    seen = {f.kind for j in RUN_JOURNALS for f in _run_frames(j)}
    assert seen >= TASK_LOOP_KINDS, sorted(TASK_LOOP_KINDS - seen)


def _session_frame(seq: int, line: dict[str, object]) -> RecordFrame:
    body = line["bytes"] if "bytes" in line else line.get("payload", {})
    raw = json.loads(body) if isinstance(body, str) else body
    assert isinstance(raw, dict)
    run_id = line.get("run_id")
    return RecordFrame(
        scope="S-pin", seq=seq, at="t", kind=str(line["kind"]),
        run_id=run_id if isinstance(run_id, str) else None, payload=raw,
    )  # fmt: skip


def test_every_recorded_session_line_reads_as_its_kind() -> None:
    lines = [json.loads(x) for x in SESSION_LINES.read_text(encoding="utf-8").splitlines()]
    typed = 0
    for seq, line in enumerate(lines, start=1):
        frame = _session_frame(seq, line)
        assert not is_run_record(frame)
        got = payload(frame)
        assert (got is None) == (frame.kind not in SESSION_RECORDS), frame.kind
        typed += got is not None
    assert typed >= 10
