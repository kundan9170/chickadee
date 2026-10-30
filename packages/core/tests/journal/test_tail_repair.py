"""Torn tail truncated; torn mid-file refused.

The asymmetry is the point: a crash can only lose the last line, so that is the only
legal repair. Anything we "fixed" mid-file would no longer be the source of truth.
"""

from __future__ import annotations

import pathlib

import pytest
from helpers import payload_for, scope_args

from chickadee.journal import Journal, JournalCorrupt, paths, read_events
from chickadee.schema import EventKind
from chickadee.state.roots import Roots


def tail_is_torn(path: pathlib.Path) -> bool:
    """Whether the file ends mid-line. `chickadee.journal` exported this until `T-3.50`, for
    these tests alone; the repair it reported moved to `durable.AppendLog.open`."""
    raw = path.read_bytes() if path.exists() else b""
    return bool(raw) and not raw.endswith(b"\n")


def _seed(repo: Roots, run_id: str, n: int = 5) -> pathlib.Path:
    with Journal.create(repo, run_id, {}) as j:
        for _ in range(n):
            j.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL))
    return paths.journal_path(repo, run_id)


def test_torn_trailing_line_is_ignored_by_the_reader(repo: Roots, run_id: str) -> None:
    path = _seed(repo, run_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "run_id": "2026-08-18-a1b2c3d4", "seq": 6, "ts": "2026')
    assert tail_is_torn(path)
    events = list(read_events(path))
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    # The reader is pure: it serves eval archives and other people's runs, so it reports
    # what it can parse and changes nothing.
    assert tail_is_torn(path)


def test_resume_truncates_the_torn_bytes_and_can_append(repo: Roots, run_id: str) -> None:
    """The bytes must actually leave the file.

    This is the scenario the previous test was named for but did not cover: reading a
    torn journal, then resuming and appending. While the repair was in-memory only, the
    next record landed straight after the partial bytes and concatenated into one
    malformed line — turning the single survivable crash into permanent corruption.
    """
    path = _seed(repo, run_id)
    intact = path.stat().st_size
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"v": 1, "run_id": "2026-08-18-a1b2c3d4", "seq": 6, "ts": "2026')
    assert path.stat().st_size > intact

    journal = Journal.open_for_resume(repo, run_id)
    state = journal.projection()
    assert path.stat().st_size == intact
    assert not tail_is_torn(path)
    assert state.last_seq == 5
    assert journal.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL)) == 6
    journal.close()

    assert [e.seq for e in read_events(path)] == [1, 2, 3, 4, 5, 6]


def test_resume_survives_a_tear_that_leaves_no_complete_line(repo: Roots, run_id: str) -> None:
    """A crash during the very first append leaves no newline anywhere."""
    Journal.create(repo, run_id, {}).close()
    path = paths.journal_path(repo, run_id)
    path.write_text('{"v": 1, "run_id": "2026-08', encoding="utf-8")
    journal = Journal.open_for_resume(repo, run_id)
    state = journal.projection()
    assert path.stat().st_size == 0
    assert state.last_seq == 0
    assert journal.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL)) == 1
    journal.close()


def test_torn_tail_refused_when_repair_disabled(repo: Roots, run_id: str) -> None:
    path = _seed(repo, run_id)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{partial")
    with pytest.raises(JournalCorrupt):
        list(read_events(path, repair_tail=False))


def test_mid_file_corruption_refuses(repo: Roots, run_id: str) -> None:
    path = _seed(repo, run_id)
    lines = path.read_text().split("\n")
    lines[2] = "{not json at all"
    path.write_text("\n".join(lines))
    with pytest.raises(JournalCorrupt) as exc:
        list(read_events(path))
    assert exc.value.seq == 3


def test_seq_gap_refuses(repo: Roots, run_id: str) -> None:
    path = _seed(repo, run_id)
    lines = [ln for ln in path.read_text().split("\n") if ln]
    del lines[2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(JournalCorrupt, match="seq gap"):
        list(read_events(path))


@pytest.mark.writes_worktree("empty.jsonl")
def test_empty_and_absent_journals_are_not_corrupt(
    repo: Roots, run_id: str, tmp_path: pathlib.Path
) -> None:
    assert list(read_events(tmp_path / "nope.jsonl")) == []
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert list(read_events(empty)) == []


def test_run_end_must_be_last(repo: Roots, run_id: str) -> None:
    from chickadee.journal import project

    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_END, payload_for(EventKind.RUN_END))
        j.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL))
    with pytest.raises(JournalCorrupt, match="follows run_end"):
        project(read_events(paths.journal_path(repo, run_id))).require_consistent()
    assert scope_args(EventKind.RUN_END) == {}


# A tail torn inside a multi-byte character (#276 B3). Payloads are raw UTF-8, so a goal or
# a reason in any script can sit at the tear. Decoding the whole file before dropping the
# partial line raised `UnicodeDecodeError` there instead of skipping it.
_TORN_IN_A_CHARACTER = '{"v": 1, "goal": "café'.encode()[:-1]


def test_a_tail_torn_inside_a_character_is_ignored_by_the_reader(
    repo: Roots, run_id: str
) -> None:
    path = _seed(repo, run_id)
    with path.open("ab") as fh:
        fh.write(_TORN_IN_A_CHARACTER)
    assert [e.seq for e in read_events(path)] == [1, 2, 3, 4, 5]
    assert tail_is_torn(path)


def test_resume_truncates_a_tail_torn_inside_a_character(repo: Roots, run_id: str) -> None:
    path = _seed(repo, run_id)
    intact = path.stat().st_size
    with path.open("ab") as fh:
        fh.write(_TORN_IN_A_CHARACTER)
    journal = Journal.open_for_resume(repo, run_id)
    state = journal.projection()
    assert path.stat().st_size == intact
    assert state.last_seq == 5
    journal.close()


def test_a_complete_line_that_is_not_utf8_is_corrupt(repo: Roots, run_id: str) -> None:
    path = _seed(repo, run_id)
    with path.open("ab") as fh:
        fh.write(b'{"v": 1, "seq": 6, "x": "\xff"}\n')
    with pytest.raises(JournalCorrupt) as exc:
        list(read_events(path))
    assert exc.value.seq == 6
