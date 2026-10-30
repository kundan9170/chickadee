"""A seek on a run's journal costs its own lines: `read_tail` stops at its
`limit` and names where it stopped, and `read_after` steps over the lines it is not asked for
as bytes, unparsed, yet still refuses a journal whose seqs disagree with its lines."""

from __future__ import annotations

import pathlib

import pytest
from helpers import payload_for

import chickadee.journal.reader as reader
from chickadee.journal import Journal, JournalCorrupt, paths
from chickadee.journal.reader import read_after, read_tail
from chickadee.schema import EventKind
from chickadee.scopes import Cursor, run_page
from chickadee.state.roots import Roots


def _journal(repo: Roots, run_id: str, n: int) -> pathlib.Path:
    with Journal.create(repo, run_id, {}) as j:
        for _ in range(n):
            j.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL))
    return paths.journal_path(repo, run_id)


def test_a_limited_read_names_where_it_stopped(repo: Roots, run_id: str) -> None:
    path = _journal(repo, run_id, 7)
    first = read_tail(path, limit=3)
    assert [e.seq for e in first.events] == [1, 2, 3]
    rest = read_tail(path, first.end, 4)
    assert [e.seq for e in rest.events] == [4, 5, 6, 7]
    assert rest.end == path.stat().st_size


def test_a_limited_read_parses_nothing_past_its_limit(
    repo: Roots, run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _journal(repo, run_id, 50)
    parsed: list[int] = []
    real = reader._event  # pyright: ignore[reportPrivateUsage]

    def counting(p: pathlib.Path, chunk: bytes, expected: int) -> object:
        parsed.append(expected)
        return real(p, chunk, expected)

    monkeypatch.setattr(reader, "_event", counting)
    read_after(path, 40, limit=3)
    assert parsed == [41, 42, 43], "the 40 lines before are stepped over, the rest unread"


def test_paging_by_offset_reads_every_line_once(repo: Roots, run_id: str) -> None:
    path = _journal(repo, run_id, 10)
    seen: list[int] = []
    after, offset = 0, None
    while True:
        tail = read_after(path, after, offset, limit=4)
        if not tail.events:
            break
        seen.extend(e.seq for e in tail.events)
        after, offset = tail.events[-1].seq, tail.end
    assert seen == list(range(1, 11))


def test_a_seek_past_the_end_is_empty(repo: Roots, run_id: str) -> None:
    path = _journal(repo, run_id, 3)
    assert read_after(path, 9).events == ()


def test_a_journal_whose_seqs_disagree_with_its_lines_still_refuses(
    repo: Roots, run_id: str
) -> None:
    """Stepping over lines unparsed trusts line number = seq; the first line read checks it."""
    path = _journal(repo, run_id, 4)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + lines[2] + lines[3])  # seq 2 is gone
    tail = read_after(path, 1)
    assert tail.events == ()
    assert isinstance(tail.damage, JournalCorrupt)


def test_a_runs_damage_is_served_after_the_records_before_it(repo: Roots, run_id: str) -> None:
    """`run_page`: the records before the damage come first; only a page that would hold
    nothing else raises, so a reader keeps everything the journal can still prove."""
    path = _journal(repo, run_id, 3)
    with path.open("ab") as fh:
        fh.write(b"not json\n")
    page = run_page(path, run_id, Cursor())
    assert [r.seq for r in page.records] == [1, 2, 3]
    assert page.cursor.offset is not None
    with pytest.raises(JournalCorrupt):
        run_page(path, run_id, page.cursor)
