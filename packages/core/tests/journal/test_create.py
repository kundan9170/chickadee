"""`Journal.create` starts one run, once.

It wrote the config snapshot before it took the run's LOCK and opened the journal with a seq
counter at 0 whatever the file held. So a second `create` on a live run was refused only
after it had replaced that run's `config.json`, and a `create` on a finished run appended seq
1 after the old run's last line, which made the whole journal unreadable.
"""

from __future__ import annotations

import json

import pytest
from helpers import payload_for

from chickadee.journal import Journal, JournalWriteError, ResumeRefused, paths, read_events
from chickadee.schema import EventKind
from chickadee.state.roots import Roots


def _config(repo: Roots, run_id: str) -> bytes:
    return (paths.run_dir(repo, run_id) / paths.CONFIG_NAME).read_bytes()


def _seqs(repo: Roots, run_id: str) -> list[int]:
    return [e.seq for e in read_events(paths.journal_path(repo, run_id))]


def test_a_second_create_on_a_live_run_leaves_its_config_alone(
    repo: Roots, run_id: str
) -> None:
    live = Journal.create(repo, run_id, {"model": "first"})
    before = _config(repo, run_id)
    with pytest.raises(ResumeRefused):
        Journal.create(repo, run_id, {"model": "second"})
    assert _config(repo, run_id) == before
    live.close()


def test_a_create_over_a_finished_run_is_refused_and_its_journal_still_reads(
    repo: Roots, run_id: str
) -> None:
    with Journal.create(repo, run_id, {"model": "first"}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
        j.append(EventKind.RUN_END, payload_for(EventKind.RUN_END))
    journal = paths.journal_path(repo, run_id).read_bytes()
    config = _config(repo, run_id)

    with pytest.raises(JournalWriteError, match="open_for_resume"):
        Journal.create(repo, run_id, {"model": "second"})

    assert paths.journal_path(repo, run_id).read_bytes() == journal
    assert _config(repo, run_id) == config
    assert _seqs(repo, run_id) == [1, 2]


def test_an_empty_journal_is_still_a_run(repo: Roots, run_id: str) -> None:
    """A `create` that died before its first line left the file: the id is taken all the
    same, and continuing it is a resume."""
    Journal.create(repo, run_id, {}).close()
    with pytest.raises(JournalWriteError, match="already exists"):
        Journal.create(repo, run_id, {})


def test_a_refused_create_holds_no_lock(repo: Roots, run_id: str) -> None:
    with Journal.create(repo, run_id, {}) as j:
        j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START))
    with pytest.raises(JournalWriteError):
        Journal.create(repo, run_id, {})
    resumed = Journal.open_for_resume(repo, run_id)
    state = resumed.projection()
    assert state.last_seq == 1
    resumed.close()


def test_a_fresh_create_writes_the_whole_config_and_appends_from_seq_1(
    repo: Roots, run_id: str
) -> None:
    with Journal.create(repo, run_id, {"model": "m", "cap": 2.5}) as j:
        assert j.append(EventKind.RUN_START, payload_for(EventKind.RUN_START)) == 1
    assert json.loads(_config(repo, run_id)) == {"cap": 2.5, "model": "m"}
    assert _seqs(repo, run_id) == [1]
