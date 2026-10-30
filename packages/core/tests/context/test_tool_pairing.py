"""No cut ever falls between a tool_use and its tool_result.

A provider rejects a `tool_use` without its `tool_result` for the rest of the session, so
this is an assert and not a comment. It holds by construction: the fold refuses a user or
assistant turn while a call waits, so every boundary a cut can take is a line no pending
call spans, and assembly holds a notice that lands mid-exchange until the results are in.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import pytest
from context_helpers import cut, exchange, run_id, work_unit

from chickadee.client.request import Message
from chickadee.context.compaction.boundary import fallback_boundary
from chickadee.context.compaction.receipts import exchanges_in
from chickadee.context.lines import EpisodeLine, PayloadError
from chickadee.context.project import completed_units
from chickadee.context.request import assemble
from chickadee.context.session import Session
from chickadee.context.standing import Notice, NoticeTrigger
from chickadee.errors import HarnessBug
from chickadee.schema import Tier, ToolCall


def calls(*ids: str) -> list[ToolCall]:
    return [ToolCall(id=i, name="shell", arguments={"command": f"echo {i}"}) for i in ids]


def assert_paired(messages: Sequence[Message], pending: Sequence[str] = ()) -> None:
    """Every assistant message that calls tools is followed at once by exactly its results,
    and no result appears anywhere else."""
    i = 0
    while i < len(messages):
        m = messages[i]
        assert m.role != "tool", f"result for {m.tool_call_id} with no call right before it"
        i += 1
        if not m.tool_calls:
            continue
        answered: set[str | None] = set()
        while i < len(messages) and messages[i].role == "tool":
            answered.add(messages[i].tool_call_id)
            i += 1
        assert answered == {c.id for c in m.tool_calls} - set(pending)


def test_the_fold_refuses_a_turn_while_a_call_waits(session: Session) -> None:
    session.record_user("go")
    session.record_assistant("", calls("c1", "c2"))
    session.record_tool("c1", "one")
    with pytest.raises(PayloadError, match="call c2 is still waiting for its result"):
        session.record_user("are you done?")
    with pytest.raises(PayloadError, match="still waiting"):
        session.record_assistant("I am")
    assert session.state.last_seq == 3
    assert session.state.pending_calls() == ("c2",)


def test_a_notice_that_lands_mid_exchange_is_held_until_the_results_are_in(
    session: Session,
) -> None:
    session.record_user("go")
    session.record_assistant("", calls("c1", "c2"))
    session.record_tool("c1", "one")
    session.record_notice(Notice(NoticeTrigger.RUN_ENDED, "run r1 ended · not applied"))
    session.record_tool("c2", "two")
    messages = assemble(session.state, session.blob).messages
    assert [m.role for m in messages] == ["user", "assistant", "tool", "tool", "user"]
    assert str(messages[-1].content).startswith("<harness_notice>")
    assert_paired(messages)


def test_a_region_that_would_split_an_exchange_is_refused(session: Session) -> None:
    session.record_user("go")
    session.record_assistant("", calls("c1"))
    session.record_notice(Notice(NoticeTrigger.RUN_ENDED, "run r1 ended · not applied"))
    session.record_tool("c1", "one")
    with pytest.raises(HarnessBug, match="call c1 has no result inside seq 1-3"):
        exchanges_in(session.state, 1, 3)


def test_the_fallback_boundary_is_never_inside_an_exchange(session: Session) -> None:
    session.record_user("go")
    exchange(session, "c1", "shell", {"command": "ls"}, "a\n")
    session.record_user("more")
    session.record_assistant("", calls("c2"))
    boundary = fallback_boundary(session.state, 0)
    assert boundary == 5
    assert all(
        (r := session.state.result(c.call_id)) is None or r.seq < boundary or c.seq >= boundary
        for c in session.state.calls_upto(session.state.last_seq)
    )


def test_a_unit_ends_only_once_every_call_it_made_is_answered(
    session: Session, ws: Path
) -> None:
    """A `coder` result that lands before a sibling call's does not end its unit there: the
    end moves to the last answer, so the next unit's first line, where a normal cut puts
    its boundary, never has a call behind it still waiting."""
    run = run_id("alpha")
    session.record_user("do alpha and look around")
    dispatch = session.record_assistant(
        "",
        [
            ToolCall(id="c-alpha", name="coder", arguments={"request": {"goal": "alpha"}}),
            ToolCall(id="s1", name="shell", arguments={"command": "ls"}),
        ],
    )
    session.record_episode(EpisodeLine(run, "planned", Tier.T1, None, (), dispatch.seq))
    session.record_tool("c-alpha", f"run {run} SOLVED", run_id=run)
    assert completed_units(session.state) == ()
    sibling = session.record_tool("s1", "a.py\n" * 200)
    (unit,) = completed_units(session.state)
    assert unit.seq_end == sibling.seq
    session.record_assistant("done", findings={"c-alpha": "alpha solved", "s1": "one file"})
    work_unit(session, "beta")
    session.record_user("thanks")
    session.record_assistant("welcome")
    done = cut(session, ws)
    assert done.cut is not None
    assert done.cut.boundary == sibling.seq + 1
    assert_paired(assemble(session.state, session.blob).messages)


def converse(session: Session, rng: random.Random, tag: str) -> None:
    """Turns and multi-call exchanges, results in random order, notices landing between
    them."""
    for step in range(rng.randint(6, 18)):
        roll = rng.random()
        if roll < 0.3:
            session.record_user(f"user {tag}.{step}")
        elif roll < 0.45:
            session.record_assistant(f"assistant {tag}.{step}")
        else:
            ids = [f"{tag}.{step}.{k}" for k in range(rng.randint(1, 3))]
            session.record_assistant("", calls(*ids))
            for i in rng.sample(ids, len(ids)):
                if rng.random() < 0.3:
                    session.record_notice(Notice(NoticeTrigger.RUN_ENDED, f"run {i} ended"))
                session.record_tool(i, f"output of {i}\n" * (40 + rng.randint(1, 20)))
            found = {i: f"saw {i}" for i in ids if rng.random() < 0.8}
            session.record_assistant("noted", findings=found)


@pytest.mark.parametrize("seed", range(30))
def test_no_cut_separates_a_call_from_its_result(seed: int, session: Session, ws: Path) -> None:
    """Two cuts per conversation: the first while the newest call still waits for its
    result, the second after it is answered and the talk has moved on."""
    rng = random.Random(seed)
    for tag in ("a", "b"):
        converse(session, rng, tag)
        session.record_user(f"last question {tag}")
        waiting = f"{tag}.tail"
        session.record_assistant("", calls(waiting))
        pending = [waiting] if tag == "a" else []
        if not pending:
            session.record_tool(waiting, "done\n")
        assert_paired(assemble(session.state, session.blob).messages, pending)
        done = cut(session, ws)
        # The second cut may find too little new to cut to free anything, and then it is not
        # taken; the pairing holds either way, and the first cut always has room to free.
        assert done.cut is not None or tag == "b"
        messages = assemble(session.state, session.blob).messages
        assert_paired(messages, pending)
        sent = {c.id for m in messages for c in m.tool_calls}
        answered = {m.tool_call_id for m in messages if m.role == "tool"}
        assert answered == sent - set(pending)
        if pending:
            session.record_tool(waiting, "done\n")
