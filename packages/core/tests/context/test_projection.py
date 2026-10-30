"""project_unit renders identically across repeated runs and calls no model; an absent field
renders absent, never guessed (invariant 19).

> Render facts from typed fields. Quote humans verbatim. Generate nothing.

A finished run's journal never changes, so the same unit renders to the same bytes every
time. A field the record does not hold has no row.
"""

from __future__ import annotations

import ast
from pathlib import Path

from context_helpers import (
    FakeRecords,
    reopen,
    run_end,
    run_id,
    run_start,
)

import chickadee.context.project as project_module
from chickadee.context.concerns import note_concern
from chickadee.context.ledger import note_constraint, note_rejected
from chickadee.context.lines import EpisodeLine
from chickadee.context.project import completed_units, project_unit
from chickadee.context.session import Session
from chickadee.schema import JournalEvent, Tier, ToolCall

RUN = run_id("fetcher")


def full_journal(outcome: str = "COMPLETED") -> tuple[JournalEvent, ...]:
    """A run's start and end. The packet loop's rows between them (route, plan decision,
    patches, classifications, checkpoints, escalations) are parked."""
    return (run_start(RUN, "add a retry to the fetcher"), run_end(RUN, 2, outcome))


def one_unit(session: Session) -> None:
    """A user turn, the constraint it grounds, a rejection and a concern, then the dispatch,
    its episode and the run's result: one completed unit, seqs 1-9."""
    session.record_user("never touch the public API, and retries must back off")
    assert note_constraint(session, "the public API is frozen", [1], "session") == "C-1"
    assert (
        note_rejected(
            session,
            text="a global retry decorator",
            reason="it hides failures",
            by="user",
            at=1,
        )
        == "R-1"
    )
    assert note_concern(session, "retry storms under load", ["src/fetch.py"], "zeus") == "K-1"
    dispatch = session.record_assistant(
        "dispatching",
        [ToolCall(id="c-1", name="coder", arguments={"request": {"goal": "retry"}})],
    )
    session.record_episode(EpisodeLine(RUN, "planned", Tier.T1, None, ("C-1",), dispatch.seq))
    session.record_tool("c-1", f"run {RUN} COMPLETED", run_id=RUN)
    session.record_assistant("the retry is in", findings={"c-1": "solved"})


def render(session: Session, records: FakeRecords | None) -> str:
    (unit,) = completed_units(session.state)
    return project_unit(session.state, unit, records)


def test_a_superseded_constraint_keeps_its_words_in_the_frozen_block(session: Session) -> None:
    """Standing restates only what still binds. The unit's block names the constraint its
    run bound by its ledger row, so superseding it later does not leave an id whose words
    appear nowhere in context."""
    one_unit(session)
    later = session.record_user("actually, the public API may grow new methods")
    replaced = note_constraint(
        session, "the public API may only grow", [later.seq], "session", supersedes="C-1"
    )
    assert replaced == "C-2"
    rows = render(session, None).splitlines()
    assert (
        '    constraints C-1 (zeus, from turn 1): "the public API is frozen"'
        " · superseded by C-2"
    ) in rows


def test_a_unit_renders_every_typed_field_it_has(session: Session) -> None:
    """The packet loop's rows (route, decision, its note, packet, checkpoint,
    escalated) are parked, and the outcome row says the run's outcome and tasks; it said
    `SOLVED · 1 packet (1 solved, 0 escalated, 0 replanned) · $0.42 · 2.5s`."""
    one_unit(session)
    records = FakeRecords({RUN: full_journal()})
    assert render(session, records).splitlines() == [
        f"▸ run {RUN} · seq 1-9",
        '    goal        "add a retry to the fetcher"',
        '    constraints C-1 (zeus, from turn 1): "the public API is frozen"',
        '    rejected    R-1 at seq 1: "a global retry decorator" — "it hides failures"',
        "    raised      K-1",
        "    outcome     COMPLETED · 1 task (1 committed) · $0.42 · 2.5s",
    ]


def test_a_run_that_did_not_complete_says_why(session: Session) -> None:
    one_unit(session)
    rows = render(session, FakeRecords({RUN: full_journal("PARTIAL")})).splitlines()
    assert rows[-2:] == [
        "    outcome     PARTIAL · 1 task (0 committed) · $0.42 · 2.5s",
        '    why         "the task was not accepted"',
    ]


def test_the_same_unit_renders_to_the_same_bytes_every_time(session: Session) -> None:
    one_unit(session)
    records = FakeRecords({RUN: full_journal()})
    first = render(session, records)
    assert all(render(session, records) == first for _ in range(3))
    with reopen(session) as again:
        assert render(again, records) == first


def test_the_projection_imports_nothing_that_can_call_a_model() -> None:
    """No model call, checked where it would have to come from: the module's own imports,
    including the ones inside functions."""
    source = Path(project_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    model_side = ("chickadee.client", "chickadee.adapters", "chickadee.zeus", "openai", "httpx")
    assert not [m for m in imported if m.startswith(model_side)]


def test_absent_fields_render_absent(session: Session) -> None:
    one_unit(session)
    # A run with no end yet has no outcome row: no field, no row (it had a plan decision's
    # row here, parked with the packet loop's rows).
    sparse = (run_start(RUN, "add a retry to the fetcher"),)
    rows = render(session, FakeRecords({RUN: sparse})).splitlines()
    labels = [row[4:16].strip() for row in rows[1:]]
    assert labels == ["goal", "constraints", "rejected", "raised"]


def test_a_journal_that_was_not_read_is_said_not_filled_in(session: Session) -> None:
    one_unit(session)
    unconsulted = render(session, None).splitlines()
    missing = render(session, FakeRecords()).splitlines()
    assert unconsulted[1] == (
        "    journal     unavailable (its journal was not consulted) — its facts are not shown"
    )
    assert missing[1] == "    journal     unavailable (no journal) — its facts are not shown"
    for rows in (unconsulted, missing):
        assert not [r for r in rows if r.startswith(("    goal", "    outcome", "    packet"))]


def test_a_concern_raised_in_the_unit_renders_as_its_id_alone(session: Session) -> None:
    """The full text in a frozen block would be a permanent decision made on a
    temporary condition. The id is the pointer; `concerns(path=…)` has the words."""
    one_unit(session)
    text = render(session, FakeRecords({RUN: full_journal()}))
    assert "    raised      K-1" in text.splitlines()
    assert "retry storms" not in text
