"""An exchange missing `finding` is refused, never collapsed; collapse() over its own output
is a no-op. No intent is asked for.

A receipt replaces a tool exchange with which call it was and what it found. Without a
finding that is evidence destroyed and nothing put in its place, strictly worse than
spending the tokens, so the exchange stays verbatim. And a receipt has nothing left to
collapse: a second cut carries the first block exactly as it was committed.
"""

from __future__ import annotations

from pathlib import Path

from context_helpers import at_length, cut, exchange, png, reader_for, write_files
from tool_args import checked, zeus_answer

from chickadee.context.compaction.boundary import Decline
from chickadee.context.compaction.receipts import (
    CollapseRefused,
    Receipt,
    collapse,
    exchanges_in,
)
from chickadee.context.recovery import Recovery
from chickadee.context.request import assemble
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.schema import ToolCall

LISTING = "src/\ntests/\nREADME.md\n" * 10


def region_with(session: Session, *, finding: str | None) -> None:
    session.record_user(at_length("what is here?"))
    exchange(session, "c1", "shell", {"command": "ls"}, LISTING, finding=finding)
    session.record_user("next")
    session.record_assistant("ok")


def test_collapse_refuses_an_exchange_without_a_finding_and_asks_no_intent(
    session: Session,
) -> None:
    """D208, a moved answer: an exchange with a finding and no intent collapses (it was
    refused, `missing: ["intent"]`)."""
    session.record_user("go")
    exchange(session, "a", "shell", {"command": "ls"}, "x", finding=None)
    exchange(session, "b", "shell", {"command": "pwd"}, "y", finding="seen")
    found = {ex.call_id: ex for ex in exchanges_in(session.state, 1, session.state.last_seq)}
    assert collapse(found["a"], Recovery.NONE) == CollapseRefused("a")
    assert isinstance(collapse(found["b"], Recovery.NONE), Receipt)


def test_a_receipt_quotes_its_claims_and_states_its_facts_bare(session: Session) -> None:
    region_with(session, finding="three entries, src and tests")
    ex = exchanges_in(session.state, 1, 4)[0]
    receipt = collapse(ex, Recovery.NONE)
    assert isinstance(receipt, Receipt)
    rows = receipt.render().splitlines()
    assert rows[0] == "[evicted · shell ls"
    assert rows[1] == '   finding  "three entries, src and tests"'
    assert rows[2].startswith(f"   {ex.tokens:,} tok · 30 lines · rehydrated 0x")
    assert rows[3] == "   not recoverable - finding is the record ]"


def test_a_cut_keeps_an_exchange_without_a_finding_verbatim(session: Session, ws: Path) -> None:
    region_with(session, finding=None)
    done = cut(session, ws)
    assert done.cut is not None
    block = done.cut.block
    assert "[kept verbatim · shell ls · no finding was written" in block
    assert '   call     shell {"command":"ls"}' in block
    assert LISTING.strip() in block
    assert "[evicted" not in block
    assert done.cut.facts["refused"] == [{"call_id": "c1", "kept": "verbatim"}]


def test_a_refused_image_is_dropped_whole_and_says_so(session: Session, ws: Path) -> None:
    """Pixels cannot sit in a text block, and a receipt without a finding is refused: the
    image goes, marked, with how to get it back."""
    write_files(ws, {"shot.png": png(64, 48)})
    reader = reader_for(session, ws, vision=Vision("zeus", supports_images=True))
    session.record_user(at_length("look at the screenshot"))
    session.record_assistant(
        "",
        [ToolCall(id="i1", name="read", arguments={"path": "shot.png", "depth": "full"})],
    )
    line = zeus_answer(
        session, reader, "i1", "read", checked("read", {"path": "shot.png", "depth": "full"})
    )
    assert line.image is not None
    session.record_assistant("I see it")
    session.record_user("next")
    session.record_assistant("ok")
    done = cut(session, ws)
    assert done.cut is not None
    assert "[dropped · read shot.png · image 64x48 · no finding was written" in done.cut.block
    assert done.cut.facts["refused"] == [{"call_id": "i1", "kept": "dropped"}]
    assert all(
        isinstance(m.content, str) for m in assemble(session.state, session.blob).messages
    )


def test_a_second_cut_carries_the_first_block_exactly_as_committed(
    session: Session, ws: Path
) -> None:
    region_with(session, finding="three entries")
    first = cut(session, ws).cut
    assert first is not None
    session.record_user(at_length("where are we?"))
    exchange(session, "c2", "shell", {"command": "pwd"}, "/tmp/x\n", finding="in /tmp/x")
    session.record_user("and after that")
    session.record_assistant("done")
    second = cut(session, ws).cut
    assert second is not None
    assert second.previous == first.boundary
    messages = assemble(session.state, session.blob).messages
    assert [m.content for m in messages[:4]] == [
        first.block,
        first.standing,
        second.block,
        second.standing,
    ]
    assert "shell ls" not in second.block
    assert second.block.count("[evicted") == 1


def test_cutting_again_with_nothing_new_to_cut_changes_nothing(
    session: Session, ws: Path
) -> None:
    region_with(session, finding="three entries")
    assert cut(session, ws).cut is not None
    before = (session.state.last_seq, assemble(session.state, session.blob))
    again = cut(session, ws)
    assert again.cut is None
    assert again.declined is Decline.NO_SAFE_BOUNDARY
    assert (session.state.last_seq, assemble(session.state, session.blob)) == before
