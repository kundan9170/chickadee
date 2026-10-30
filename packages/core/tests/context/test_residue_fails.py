"""A layer-3 timeout still commits the cut, with empty residue.

Layer 3 is the one layer that calls a model, and a compaction that cannot complete because
a model call failed is the market's failure reproduced from the other direction. So every
failure is recorded and none of them fails the cut: it commits with no residue section, and
an empty residue asserts nothing about what else was said.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import pytest
from context_helpers import FakeRecords, cut, exchange, run_end, run_start, work_unit

import chickadee.context.compaction.residue as residue_module
from chickadee.context.compaction.residue import L3_MAX_ITEMS, ResidueRequest, ResidueSource
from chickadee.context.lines import CutLine
from chickadee.context.session import Session

TIMEOUT_MS = 50


@pytest.fixture
def short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real bound is ten seconds; the behaviour at the bound is the same at fifty ms."""
    monkeypatch.setattr(residue_module, "L3_TIMEOUT_MS", TIMEOUT_MS)


@pytest.fixture
def released() -> Iterator[threading.Event]:
    """Lets a model stuck past its timeout finish once the test is done, so no thread is
    left waiting behind the suite."""
    gate = threading.Event()
    yield gate
    gate.set()


def prose(session: Session) -> None:
    """Old enough to cut, with words the record does not hold: a fallback cut's region."""
    # Long enough that the cut frees room: a cut that frees nothing is not taken.
    session.record_user(
        "the staging database is shared with the analytics team, "
        + "whose nightly jobs read from it and must not see it reset, " * 20
    )
    exchange(session, "c1", "shell", {"command": "ls"}, "a\nb\n", finding="two files")
    session.record_assistant("understood, I will not reset it")
    session.record_user("next")
    session.record_assistant("ok")


def residue_facts(line: CutLine) -> object:
    return line.facts["residue"]


def test_a_residue_model_that_times_out_still_commits_the_cut(
    session: Session, ws: Path, short_timeout: None, released: threading.Event
) -> None:
    prose(session)

    def stuck(_request: ResidueRequest) -> Sequence[str]:
        released.wait(5)
        return ["too late"]

    started = time.monotonic()
    done = cut(session, ws, layer3=ResidueSource(model=stuck))
    assert time.monotonic() - started < 3
    assert done.cut is not None
    assert session.state.cuts() == (done.cut,)
    assert residue_facts(done.cut) == {
        "job": "fallback",
        "fired": True,
        "items": 0,
        "failure": f"timed out after {TIMEOUT_MS} ms",
    }
    assert "[model recollection" not in done.cut.block
    assert "too late" not in done.cut.block


def test_a_residue_call_that_outlives_its_timeout_cannot_hold_the_process_open(
    session: Session, ws: Path, short_timeout: None, released: threading.Event
) -> None:
    """A running call cannot be cancelled; the thread it runs on must not be one interpreter
    exit waits for."""
    prose(session)
    running: list[threading.Thread] = []

    def stuck(_request: ResidueRequest) -> Sequence[str]:
        running.append(threading.current_thread())
        released.wait(5)
        return []

    assert cut(session, ws, layer3=ResidueSource(model=stuck)).cut is not None
    (thread,) = running
    assert thread.is_alive()
    assert thread.daemon


@pytest.mark.parametrize(
    ("answer", "failure"),
    [
        (RuntimeError("the endpoint is down"), "RuntimeError: the endpoint is down"),
        ("one string, not a list", "TypeError: a residue is a list of lines"),
        ([1, 2], "TypeError: a residue item is a line of text"),
    ],
)
def test_a_residue_model_that_fails_still_commits_the_cut(
    session: Session, ws: Path, answer: object, failure: str
) -> None:
    prose(session)

    def broken(_request: ResidueRequest) -> Sequence[str]:
        if isinstance(answer, Exception):
            raise answer
        return cast("Sequence[str]", answer)  # a model breaking its return type

    done = cut(session, ws, layer3=ResidueSource(model=broken))
    assert done.cut is not None
    assert residue_facts(done.cut) == {
        "job": "fallback",
        "fired": True,
        "items": 0,
        "failure": failure,
    }
    assert "[model recollection" not in done.cut.block


def test_no_residue_model_is_a_recorded_failure_not_a_skipped_cut(
    session: Session, ws: Path
) -> None:
    prose(session)
    done = cut(session, ws)
    assert done.cut is not None
    assert residue_facts(done.cut) == {
        "job": "fallback",
        "fired": False,
        "items": 0,
        "failure": "no residue model is configured",
    }


def test_an_empty_residue_asserts_nothing(session: Session, ws: Path) -> None:
    prose(session)
    done = cut(session, ws, layer3=ResidueSource(model=lambda _r: []))
    assert done.cut is not None
    assert residue_facts(done.cut) == {
        "job": "fallback",
        "fired": True,
        "items": 0,
        "failure": None,
    }
    assert "[model recollection" not in done.cut.block
    assert "nothing else" not in done.cut.block


def test_a_residue_is_marked_a_recollection_and_bounded(session: Session, ws: Path) -> None:
    prose(session)
    asked: list[ResidueRequest] = []

    def recalls(request: ResidueRequest) -> Sequence[str]:
        asked.append(request)
        return ["  the staging   database is shared  ", ""] + [f"item {i}" for i in range(20)]

    done = cut(session, ws, layer3=ResidueSource(model=recalls))
    assert done.cut is not None
    (request,) = asked
    assert request.job == "fallback"
    assert request.max_items == L3_MAX_ITEMS
    assert [t.seq for t in request.turns] == [1, 4, 5]
    assert "[model recollection · not a record · turns 1-5]" in done.cut.block
    assert "  - the staging database is shared" in done.cut.block
    assert f"item {L3_MAX_ITEMS - 1}" not in done.cut.block
    facts = residue_facts(done.cut)
    assert isinstance(facts, dict)
    assert facts["items"] == L3_MAX_ITEMS


def test_residue_reads_only_past_the_last_checkpoint_and_excludes_the_record(
    session: Session, ws: Path
) -> None:
    """X-36: turns a checkpoint already faced are not asked about again, and the record
    already made is an exclusion set, never material."""
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("thanks")
    session.record_assistant("welcome")
    asked: list[ResidueRequest] = []

    def recalls(request: ResidueRequest) -> Sequence[str]:
        asked.append(request)
        return []

    records = FakeRecords(
        {
            run: (run_start(run, f"goal of {name}"), run_end(run, 2))
            for name, run in (
                ("alpha", session.state.episodes()[0].run_id),
                ("beta", session.state.episodes()[1].run_id),
            )
        }
    )
    source = ResidueSource(model=recalls, last_checkpoint_seq=1, checkpoint_rows=("row-x",))
    done = cut(session, ws, records=records, layer3=source)
    assert done.cut is not None
    (request,) = asked
    assert request.job == "residue"
    assert all(t.seq > 1 for t in request.turns)
    assert request.exclusion[-1] == "row-x"
    assert '"goal of alpha"' in request.exclusion[0]
