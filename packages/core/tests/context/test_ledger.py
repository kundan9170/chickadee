"""The constraint ledger.

A stated constraint is EXEMPT from every layer: it is a `constraint` line, never a message,
so no layer takes it as input, and the standing snapshot restates it after every cut.
Superseding one ends it and deletes nothing. Zeus words a constraint but cannot invent one:
it must cite a turn the user said, and only a later turn can change it. Session-scope
constraints bind without being cited.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from context_helpers import (
    FakeRecords,
    cut,
    exchange,
    reopen,
    run_end,
    run_start,
    work_unit,
)
from refusals import bare
from session_lines import raw

from chickadee.context.compaction.residue import ResidueRequest, ResidueSource
from chickadee.context.ledger import (
    binding_constraints,
    note_constraint,
    record_refusal_note,
)
from chickadee.context.lines import PayloadError
from chickadee.context.project import completed_units, project_unit
from chickadee.context.results import InvalidArgument
from chickadee.context.session import Session

SAID = "please don't change the public API of the fetcher"
WORDED = "fetch.py's public signatures are frozen"


def stated(session: Session) -> str:
    session.record_user(SAID)
    cid = note_constraint(session, WORDED, [1], "session")
    assert cid == "C-1"
    return "C-1"


# -- exempt from every layer ------------------------------------------------------


def test_a_constraint_is_never_an_input_to_collapse_projection_or_residue(
    session: Session, ws: Path
) -> None:
    stated(session)
    exchange(session, "c1", "shell", {"command": "ls"}, "fetch.py\n", finding="one file")
    work_unit(session, "alpha")
    work_unit(session, "beta")
    session.record_user("thanks")
    session.record_assistant("welcome")
    runs = [e.run_id for e in session.state.episodes()]
    records = FakeRecords({r: (run_start(r, "goal"), run_end(r, 2)) for r in runs})
    asked: list[ResidueRequest] = []

    def recalls(request: ResidueRequest) -> Sequence[str]:
        asked.append(request)
        return []

    done = cut(session, ws, records=records, layer3=ResidueSource(model=recalls))
    assert done.cut is not None
    assert WORDED not in done.cut.block
    for unit in completed_units(session.state):
        assert WORDED not in project_unit(session.state, unit, records)
    (request,) = asked
    assert not [t for t in request.turns if WORDED in t.text]
    assert not [x for x in request.exclusion if WORDED in x]
    assert f"C-1  {WORDED} · from turn 1" in done.cut.standing
    kept = session.state.ledger.get("C-1")
    assert kept is not None
    assert kept.live


def test_every_later_cut_restates_a_live_constraint(session: Session, ws: Path) -> None:
    stated(session)
    for name in ("alpha", "beta", "gamma"):
        work_unit(session, name)
    session.record_user("thanks")
    session.record_assistant("welcome")
    first = cut(session, ws).cut
    work_unit(session, "delta")
    session.record_user("and again")
    session.record_assistant("sure")
    second = cut(session, ws).cut
    assert first is not None
    assert second is not None
    assert WORDED in first.standing
    assert WORDED in second.standing


# -- supersession ends a row and deletes nothing ------------------------------------------


def test_superseding_sets_valid_until_and_keeps_the_row(session: Session) -> None:
    stated(session)
    later = session.record_user("actually, the retry parameter may be added")
    new = note_constraint(
        session, "fetch.py may gain a retry parameter", [later.seq], "session", supersedes="C-1"
    )
    assert new == "C-2"
    old = session.state.ledger.get("C-1")
    replacement = session.state.ledger.get("C-2")
    assert old is not None
    assert replacement is not None
    assert old.text == WORDED
    assert old.valid_until == replacement.valid_from
    assert old.superseded_by == "C-2"
    assert [c.id for c in session.state.ledger.constraints()] == ["C-1", "C-2"]
    assert [c.id for c in session.state.ledger.live_constraints()] == ["C-2"]
    with reopen(session) as again:
        assert again.state.ledger.get("C-1") == old
        assert [c.id for c in again.state.ledger.live_constraints()] == ["C-2"]


# -- INVALID_ARGUMENTS, and nothing written ------------------------------------------------


REFUSALS = [
    pytest.param(
        [],
        "session",
        None,
        "from",
        "a constraint must come from something the user said",
        id="no-turn",
    ),
    pytest.param(
        [2], "session", None, "from", "turn 2 was not said by the user", id="not-the-user"
    ),
    pytest.param([99], "session", None, "from", "turn 99 does not exist", id="no-such-turn"),
    pytest.param(
        [1],
        "packet_set",
        None,
        "scope",
        "a constraint Zeus writes is session or design scope",
        id="packet-scope",
    ),
    pytest.param(
        [1], "session", "C-9", "supersedes", "no such live constraint", id="no-such-row"
    ),
    pytest.param(
        [1],
        "session",
        "C-1",
        "supersedes",
        "only a later turn can change it",
        id="no-later-turn",
    ),
]


@pytest.mark.parametrize(("sources", "scope", "supersedes", "field", "why"), REFUSALS)
def test_a_constraint_without_a_user_turn_behind_it_is_refused_and_not_written(
    session: Session,
    sources: list[int],
    scope: str,
    supersedes: str | None,
    field: str,
    why: str,
) -> None:
    stated(session)
    before = session.state.last_seq
    refused = note_constraint(session, "anything", sources, scope, supersedes)
    assert bare(refused) == InvalidArgument(field, why)
    assert session.state.last_seq == before


@pytest.mark.parametrize("text", ["", "   ", "x" * 501])
def test_a_constraint_says_one_bounded_thing(session: Session, text: str) -> None:
    session.record_user(SAID)
    refused = note_constraint(session, text, [1], "session")
    assert isinstance(refused, InvalidArgument)
    assert refused.field == "text"


def test_an_ask_user_answer_is_the_users_words_and_citable(session: Session) -> None:
    session.record_user("fix the fetcher")
    exchange(session, "q1", "ask_user", {"question": "may I change the API?"}, "no, never")
    assert note_constraint(session, WORDED, [3], "design") == "C-1"


def test_the_fold_refuses_a_forged_line_before_it_is_written(session: Session) -> None:
    """§5.1 step 1: a line the fold would refuse is never written, because a durable line
    replay refuses is a session that never opens again."""
    session.record_user(SAID)
    session.record_assistant("noted")
    forged = {
        "id": "C-1",
        "text": WORDED,
        "sources": [2],
        "author": "zeus",
        "scope": "session",
        "scope_id": None,
        "valid_from": 3,
        "valid_until": None,
        "superseded_by": None,
        "supersedes": None,
    }
    with pytest.raises(PayloadError, match="not a user turn"):
        session.append(raw("constraint", forged))
    assert session.state.last_seq == 2
    with reopen(session) as again:
        assert again.state.last_seq == 2


# -- what binds ---------------------------------------------------------------------------


def test_session_constraints_bind_without_citation_and_design_ones_when_cited(
    session: Session,
) -> None:
    session.record_user("never touch the public API; the new module is async")
    assert note_constraint(session, WORDED, [1], "session") == "C-1"
    assert note_constraint(session, "the new module is async", [1], "design") == "C-2"
    uncited = binding_constraints(session.state, [])
    assert not isinstance(uncited, InvalidArgument)
    assert [c.id for c in uncited] == ["C-1"]
    cited = binding_constraints(session.state, ["C-2"])
    assert not isinstance(cited, InvalidArgument)
    assert [c.id for c in cited] == ["C-1", "C-2"]
    assert bare(binding_constraints(session.state, ["C-7"])) == InvalidArgument(
        "constraints", "C-7 is not a live constraint"
    )


def test_a_superseded_constraint_no_longer_binds(session: Session) -> None:
    stated(session)
    later = session.record_user("the API may change now")
    note_constraint(session, "the API may change", [later.seq], "session", supersedes="C-1")
    bound = binding_constraints(session.state, [])
    assert not isinstance(bound, InvalidArgument)
    assert [c.id for c in bound] == ["C-2"]
    assert bare(binding_constraints(session.state, ["C-1"])) == InvalidArgument(
        "constraints", "C-1 is not a live constraint"
    )


def test_a_refusal_note_is_stored_verbatim_in_the_users_words(session: Session) -> None:
    """The person typed it at the prompt, so it is stored as typed — no cap, no
    rewording — and marked as theirs wherever it is restated."""
    session.record_user("go ahead")
    exchange(session, "q1", "ask_user", {"question": "apply?"}, "skip")
    note = "  Not on Fridays.\nWe deploy then.  "
    cid = record_refusal_note(session, note, 3)
    assert cid == "C-1"
    row = session.state.ledger.get("C-1")
    assert row is not None
    assert row.text == note
    assert row.author == "user"
    assert record_refusal_note(session, "   ", 3) is None
