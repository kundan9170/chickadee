"""`RunProjections` answers what a from-scratch read answers, after any growth.

Each test writes a recorded journal one prefix at a time, every byte length of it, so every
torn tail is seen, and asks one long-lived `RunProjections` and a fresh `project(read_run(...))`
the same question at each. The answers are equal, a refusal included. Damage, a replaced file
and a shorter file are prefixes too.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable

import pytest
from helpers import payload_for, scope_args

from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.paths import journal_path
from chickadee.journal.projection import RunProjection, project, project_readable
from chickadee.journal.reader import read_run, read_tail
from chickadee.journal.runs import RunProjections
from chickadee.journal.writer import Journal
from chickadee.schema import EventKind
from chickadee.state.roots import Roots

KINDS = (
    EventKind.RUN_START,
    EventKind.MODEL_CALL,
    EventKind.BUDGET,
    EventKind.MODEL_CALL,
    EventKind.RUN_END,
)


def recorded(repo: Roots, run_id: str) -> bytes:
    """A whole journal's bytes, written by the writer and then removed, so each test lays
    down its own prefixes."""
    with Journal.create(repo, run_id, {}) as j:
        for kind in KINDS:
            j.append(kind, payload_for(kind), **scope_args(kind))
    path = journal_path(repo, run_id)
    data = path.read_bytes()
    path.unlink()
    return data


Outcome = tuple[str, object]


def outcome(read: Callable[[], RunProjection]) -> Outcome:
    try:
        return "value", read()
    except (JournalCorrupt, FileNotFoundError) as exc:
        return type(exc).__name__, str(exc)


def assert_same(runs: RunProjections, repo: Roots, run_id: str, at: str) -> None:
    assert outcome(lambda: runs.projection(run_id)) == outcome(
        lambda: project(read_run(repo, run_id))
    ), at
    assert outcome(lambda: runs.readable(run_id)) == outcome(
        lambda: project_readable(read_run(repo, run_id))
    ), at


def lay(path: pathlib.Path, data: bytes) -> None:
    """Write `data` into the same file, as a writer's appends would leave it."""
    with path.open("r+b" if path.exists() else "wb") as fh:
        fh.seek(0)
        fh.write(data)
        fh.truncate()


def test_every_prefix_of_a_journal_reads_as_a_fresh_read(repo: Roots, run_id: str) -> None:
    data = recorded(repo, run_id)
    path = journal_path(repo, run_id)
    runs = RunProjections(repo)
    assert_same(runs, repo, run_id, "no journal yet")
    for n in range(len(data) + 1):
        lay(path, data[:n])
        assert_same(runs, repo, run_id, f"{n} bytes")


def test_damage_part_way_is_kept_and_never_read_again(repo: Roots, run_id: str) -> None:
    lines = recorded(repo, run_id).splitlines(keepends=True)
    data = b"".join([*lines[:2], b'{"seq": 3}\n', *lines[3:]])
    path = journal_path(repo, run_id)
    runs = RunProjections(repo)
    for n in range(len(data) + 1):
        lay(path, data[:n])
        assert_same(runs, repo, run_id, f"{n} bytes")
    with pytest.raises(JournalCorrupt, match="seq=3"):
        runs.projection(run_id)
    assert runs.readable(run_id).last_seq == 2


@pytest.mark.parametrize("how", ["replaced", "shorter"])
def test_a_journal_that_is_not_the_one_read_is_read_again(
    repo: Roots, run_id: str, how: str
) -> None:
    data = recorded(repo, run_id)
    path = journal_path(repo, run_id)
    runs = RunProjections(repo)
    lay(path, data)
    assert_same(runs, repo, run_id, "whole")
    first = b"".join(data.splitlines(keepends=True)[:2])
    if how == "replaced":
        other = path.with_name("next")
        other.write_bytes(first)
        os.replace(other, path)
    else:
        lay(path, first)
    assert_same(runs, repo, run_id, how)
    assert runs.projection(run_id).last_seq == 2


def test_a_journal_removed_is_not_found_again(repo: Roots, run_id: str) -> None:
    data = recorded(repo, run_id)
    path = journal_path(repo, run_id)
    runs = RunProjections(repo)
    lay(path, data)
    runs.projection(run_id)
    path.unlink()
    assert_same(runs, repo, run_id, "removed")
    lay(path, data[: len(data) // 2])
    assert_same(runs, repo, run_id, "written again")


def test_a_projection_handed_out_never_changes(repo: Roots, run_id: str) -> None:
    data = recorded(repo, run_id)
    path = journal_path(repo, run_id)
    runs = RunProjections(repo)
    handed: list[tuple[RunProjection, RunProjection]] = []
    for line in range(1, len(KINDS) + 1):
        lay(path, b"".join(data.splitlines(keepends=True)[:line]))
        handed.append((runs.projection(run_id), project(read_run(repo, run_id))))
    for got, then in handed:
        assert got == then


def test_read_tail_starts_where_it_left_off_and_stops_before_a_torn_line(
    repo: Roots, run_id: str
) -> None:
    data = recorded(repo, run_id)
    path = journal_path(repo, run_id)
    lay(path, data[:-5])
    first = read_tail(path)
    assert [e.seq for e in first.events] == [1, 2, 3, 4]
    assert first.damage is None
    assert data[first.end - 1 : first.end] == b"\n"
    lay(path, data)
    rest = read_tail(path, first.end, 5)
    assert [e.seq for e in rest.events] == [5]
    assert rest.end == len(data)
