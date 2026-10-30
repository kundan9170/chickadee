"""The `code_snapshot` line: written by
`zeus/snapshots.py`, typed in `context/rewind.py` where the fork reads it, checked as it
is written — it belongs to a user turn, a workspace has one `before` in a turn, and its `after`
follows that `before` once — and kept unread by the context fold, as every writer-owned kind
is."""

from __future__ import annotations

from typing import Literal

import pytest
from context_helpers import reopen
from pydantic import TypeAdapter

from chickadee.context.lines import PayloadError, canonical_json, render
from chickadee.context.rewind import CodeSnapshotLine, taken
from chickadee.context.session import OtherLine, Session
from chickadee.zeus.snapshots import record

READ = TypeAdapter(CodeSnapshotLine)


def line(
    turn: int, phase: Literal["before", "after"], key: str = "in-place"
) -> CodeSnapshotLine:
    return CodeSnapshotLine(
        turn=turn,
        phase=phase,
        key=key,
        mode="in_place",
        root="/repo",
        tree="a" * 40,
        ref=f"refs/chickadee/turns/S/{turn}/{key}/{phase}",
        head="b" * 40,
        skipped=("big.bin",),
    )


def test_the_line_reads_back_through_its_type() -> None:
    written = line(3, "before")
    assert READ.validate_json(render(written)) == written


def test_a_line_without_a_head_or_anything_skipped_reads_with_its_defaults() -> None:
    """An unborn head, and a snapshot that left nothing out."""
    raw = canonical_json(
        {
            "turn": 1,
            "phase": "after",
            "key": "R-1",
            "mode": "worktree",
            "root": "/wt",
            "tree": "t",
            "ref": "r",
        }
    )
    back = READ.validate_json(raw)
    assert (back.head, back.skipped) == (None, ())


def test_a_pair_folds_in_order_and_replays_to_it(session: Session) -> None:
    turn = session.record_user("change it").seq
    before, after = line(turn, "before"), line(turn, "after")
    record(session, before)
    record(session, after)
    assert taken(session) == (before, after)
    with reopen(session) as again:
        assert taken(again) == (before, after)
        kept = [r for e, r in again.state.lines() if e.kind == "code_snapshot"]
        assert all(isinstance(r, OtherLine) for r in kept), "the context fold keeps it unread"


def test_two_workspaces_in_one_turn_each_have_a_pair(session: Session) -> None:
    """A worktree run's own pair, beside another's."""
    turn = session.record_user("two runs").seq
    for key in ("R-1", "R-2"):
        record(session, line(turn, "before", key))
        record(session, line(turn, "after", key))
    assert len(taken(session)) == 4


def test_a_snapshot_names_a_user_turn(session: Session) -> None:
    session.record_user("hello")
    said = session.record_assistant("hi").seq
    with pytest.raises(PayloadError, match="is not a user turn"):
        record(session, line(said, "before"))
    with pytest.raises(PayloadError, match="is not a user turn"):
        record(session, line(999, "before"))


def test_one_before_per_workspace_in_a_turn(session: Session) -> None:
    turn = session.record_user("change it").seq
    record(session, line(turn, "before"))
    with pytest.raises(PayloadError, match="already has a before"):
        record(session, line(turn, "before"))


def test_an_after_follows_its_before_once(session: Session) -> None:
    turn = session.record_user("change it").seq
    with pytest.raises(PayloadError, match="has no before"):
        record(session, line(turn, "after"))
    record(session, line(turn, "before"))
    record(session, line(turn, "after"))
    with pytest.raises(PayloadError, match="already has a after"):
        record(session, line(turn, "after"))
