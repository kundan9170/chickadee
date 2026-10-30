"""One compaction rewrites the prefix exactly once, and the boundary does not move until the
next compaction fires.

A cut is an edit to the middle of the conversation and a prompt cache is a prefix cache, so
the layers are a priority order inside ONE cut and the cut is ONE durable line: boundary,
block and standing together. Between cuts the prefix only grows at its end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from context_helpers import (
    at_length,
    cut,
    exchange,
    no_cache_signal,
    over_threshold,
    reopen,
    work_unit,
)

from chickadee.context.compaction.boundary import Decline
from chickadee.context.compaction.cut import TurnTooLarge, compact
from chickadee.context.compaction.residue import ResidueRequest, ResidueSource
from chickadee.context.compaction.trigger import TriggerKind
from chickadee.context.request import Ctx, assemble
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.context.store import SessionLog


def two_units(session: Session) -> None:
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("thanks, what next?")
    session.record_assistant("let us see")


def kinds(session: Session) -> list[str]:
    return [e.kind for e in SessionLog.read(session.root, session.session_id).events()]


def test_one_compaction_is_one_line_holding_boundary_block_and_standing(
    session: Session, ws: Path
) -> None:
    two_units(session)
    before = kinds(session)
    done = cut(session, ws)
    assert done.cut is not None
    assert kinds(session) == [*before, "compaction"]
    line = done.cut
    assert line.seq == len(before) + 1
    assert line.boundary == 8
    assert line.previous == 0
    assert line.block.startswith("<harness_context>")
    assert line.standing.startswith("<harness_notice>")
    freed = line.facts["freed"]
    assert isinstance(freed, dict)
    assert set(freed) == {"l1", "l2", "l3", "dropped"}


def test_the_boundary_holds_while_the_conversation_grows(session: Session, ws: Path) -> None:
    """Sticky: the prefix a cut wrote is sent byte-identical by every later request
    until the next cut, and new lines only extend it."""
    two_units(session)
    done = cut(session, ws)
    assert done.cut is not None
    before = assemble(session.state, session.blob).messages
    for i in range(6):
        session.record_user(f"question {i}")
        exchange(session, f"q{i}", "shell", {"command": f"echo {i}"}, f"{i}\n")
    assert session.state.boundary == done.cut.boundary
    after = assemble(session.state, session.blob).messages
    assert after[: len(before)] == before
    assert len(after) > len(before)


def test_only_the_next_cut_moves_the_boundary(session: Session, ws: Path) -> None:
    two_units(session)
    first = cut(session, ws).cut
    assert first is not None
    work_unit(session, "gamma")
    session.record_user("and now?")
    session.record_assistant("one moment")
    second = cut(session, ws).cut
    assert second is not None
    assert second.previous == first.boundary
    assert second.boundary > first.boundary
    assert session.state.boundary == second.boundary
    assert [c.seq for c in session.state.cuts()] == [first.seq, second.seq]


def test_a_cut_the_log_did_not_take_did_not_happen(session: Session, ws: Path) -> None:
    """Commit, then emit: a failed append says nothing and moves nothing."""
    two_units(session)
    before = (session.state.last_seq, session.state.boundary)
    held = assemble(session.state, session.blob)
    session.close()
    done = cut(session, ws)
    assert done.cut is None
    assert done.aborted is not None
    assert done.aborted.startswith("SessionWriteError")
    assert (session.state.last_seq, session.state.boundary) == before
    assert assemble(session.state, session.blob) == held


def test_an_economic_cut_that_frees_nothing_is_not_taken(session: Session, ws: Path) -> None:
    """A region cheaper than the block that would replace it buys no room for its rebuild:
    nothing is appended, the prefix is untouched, and the residue's model call — whose
    answer could only add to the block — is never made."""
    session.record_user("go")
    exchange(session, "c1", "shell", {"command": "ls"}, "a\n", finding=None)
    session.record_user("next")
    session.record_assistant("ok")
    before = (session.state.last_seq, assemble(session.state, session.blob))
    asked: list[ResidueRequest] = []

    def model(request: ResidueRequest) -> list[str]:
        asked.append(request)
        return ["the user asked to go on"]

    done = cut(session, ws, layer3=ResidueSource(model=model))
    assert done.trigger is not None
    assert done.trigger.kind is TriggerKind.WARM
    assert (done.cut, done.aborted) == (None, None)
    assert (session.state.last_seq, assemble(session.state, session.blob)) == before
    assert asked == []
    # Skipped, never silent. The facts come back for the caller to journal.
    assert done.skipped is not None
    assert done.skipped["trigger"] == "warm"
    est = done.skipped["est_freed"]
    assert isinstance(est, int)
    assert est <= 0
    assert done.skipped["refused"] == 1, "the exchange with no finding could not collapse"


def test_a_fallback_cut_projects_the_finished_units_inside_it(
    session: Session, ws: Path
) -> None:
    """One finished run is too few for a normal boundary, so the cut falls back — and
    the run inside its region is still projected by layer 2, typed facts and all, rather
    than left to receipts and a model's recollection."""
    alpha = work_unit(session, "alpha")
    session.record_user(at_length("now something else entirely"))
    session.record_assistant("thinking about it at length " * 20)
    session.record_user("and one more thing")
    session.record_assistant("ok")
    done = cut(session, ws)
    assert done.cut is not None
    assert done.declined is Decline.NO_SAFE_BOUNDARY
    assert done.cut.facts["units"] == [alpha]
    assert f"▸ run {alpha}" in done.cut.block
    assert done.skipped is None


def test_replay_rebuilds_the_same_prefix_from_the_one_line(session: Session, ws: Path) -> None:
    two_units(session)
    assert cut(session, ws).cut is not None
    live = assemble(session.state, session.blob)
    with reopen(session) as again:
        assert again.state.boundary == 8
        assert assemble(again.state, again.blob) == live


def test_no_trigger_means_no_line(session: Session, ws: Path) -> None:
    two_units(session)
    before = kinds(session)
    held = assemble(session.state, session.blob).tokens
    roomy = Ctx(held, held * 10, "context_limit", 10)
    done = compact(
        session, roomy, no_cache_signal(), 0, resolver=HandleResolver(ws, session.blob)
    )
    assert done.trigger is None
    assert done.cut is None
    assert kinds(session) == before


def test_a_request_that_cannot_fit_with_nothing_to_cut_raises_and_writes_nothing(
    session: Session, ws: Path
) -> None:
    """MUST_FIT never declines, and never shortens what someone typed (invariant 19): with
    no line old enough to cut, the request is refused as too large, and nothing was
    written in the attempt."""
    session.record_user("x" * 4_000)
    before = kinds(session)
    held = assemble(session.state, session.blob).tokens
    tight = Ctx(held, held + 50, "context_limit", 10)
    with pytest.raises(TurnTooLarge) as raised:
        compact(
            session, tight, no_cache_signal(), 500, resolver=HandleResolver(ws, session.blob)
        )
    assert raised.value.cut is None
    assert kinds(session) == before


def test_a_must_fit_cut_that_still_does_not_fit_stays_committed(
    session: Session, ws: Path
) -> None:
    two_units(session)
    held = assemble(session.state, session.blob).tokens
    tight = Ctx(held, held + 20, "context_limit", 10)
    with pytest.raises(TurnTooLarge) as raised:
        compact(
            session, tight, no_cache_signal(), 50_000, resolver=HandleResolver(ws, session.blob)
        )
    committed = raised.value.cut
    assert committed is not None
    assert session.state.cuts() == (committed,)
    assert committed.facts["trigger"] == TriggerKind.MUST_FIT.value
    fit = committed.facts["fit"]
    assert isinstance(fit, dict)
    assert fit["fits_after"] is False


def test_a_cut_says_it_is_starting_and_no_cut_says_nothing(session: Session, ws: Path) -> None:
    """`on_cut` is told the tokens held once a cut is due, before it is made,
    so the status line can say `compacting…`; a request that needs no cut tells it nothing."""
    two_units(session)
    held = assemble(session.state, session.blob).tokens
    told: list[int] = []
    roomy = Ctx(held, held * 10, "context_limit", 10)
    compact(
        session,
        roomy,
        no_cache_signal(),
        0,
        resolver=HandleResolver(ws, session.blob),
        on_cut=told.append,
    )
    assert told == []
    done = compact(
        session,
        over_threshold(held),
        no_cache_signal(),
        0,
        resolver=HandleResolver(ws, session.blob),
        on_cut=told.append,
    )
    assert done.cut is not None
    assert told == [held]
