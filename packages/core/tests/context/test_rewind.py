"""A rewind is a session line: an `undo`
hides the conversation from its turn on, a `redo` shows it again until the next turn, what
cannot be taken back is read by one ranged reader, and a conversation-only rewind tells the
model what code it no longer remembers changing."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from context_helpers import (
    FakeRecords,
    cut,
    effect,
    exchange,
    reopen,
    run_end,
    run_id,
    run_start,
    work_unit,
)

from chickadee.context.lines import (
    SESSION_KINDS,
    ConstraintP,
    EpisodeLine,
    PayloadError,
    RedoLine,
    UndoLine,
)
from chickadee.context.project import completed_units
from chickadee.context.request import assemble
from chickadee.context.rewind import REWIND, CodeSnapshotLine, hidden_seqs, record_rewind
from chickadee.context.session import Session
from chickadee.context.standing import NoticeTrigger, journaled_effects
from chickadee.journal.effects import EffectKind
from chickadee.schema import Tier, ToolCall
from chickadee.schema.adapter import Usage
from chickadee.schema.journal import JournalEvent
from chickadee.zeus.snapshots import record as record_snapshot


def messages(session: Session) -> object:
    return assemble(session.state, session.blob).messages


def kept(session: Session) -> list[object]:
    """What a rewind never takes back: money and caps, read as `budget/family.py` reads them."""
    return [r for e, r in session.state.lines() if e.kind in ("spend", "cap")]


def charge(session: Session) -> None:
    session.record_spend("m", Usage(prompt_tokens=10, completion_tokens=5), 0.01, "zeus")


def redo(undo_seq: int, *, code: bool = False, conversation: bool = True) -> RedoLine:
    return RedoLine(undo_seq, code, conversation)


def undo(to_turn: int, *, code: bool = False, conversation: bool = True) -> UndoLine:
    return UndoLine(to_turn, code, conversation, ())


# -- the table ----------------------------------------------------------------------------


def test_every_session_kind_is_hidden_or_kept() -> None:
    """A kind added later must be placed: the table names exactly the kinds there are."""
    assert set(REWIND) == set(SESSION_KINDS)
    assert {k for k, v in REWIND.items() if v == "kept"} == {
        "spend",
        "cap",
        # The person's switch and the eval harness's pin.
        "model_switched",
        "pinned",
        # The session's name, the person's or the title role's.
        "title",
        # A handoff changed the repository, whatever is undone after.
        "handoff_intended",
        "handoff_completed",
        "limits",
        "episode",
        # A declared write happened, whatever is undone after.
        "shell_effect",
        # A call the box made happened, whatever is undone after.
        "box_call",
        "plan",
        "box_compaction",
        "workspace",
        # A work folder was opened or closed, whatever is undone after.
        "work",
        "prefix",
        "fork",
        "code_snapshot",
        "undo",
        "redo",
        "question_asked",
        "question_answered",
        # The person's decision at a design's gate binds the design, not the turn.
        "plan_decision",
        # The person's unload, and a crossing of the trust boundary that happened.
        "skill_unloaded",
        "untrusted_read",
        "command_grant",
        # An action ran under a grant, whatever is undone after.
        "action_granted",
        # A hook ran and decided, whatever is undone after.
        "hook",
        # A request left the machine, whatever is undone after.
        "external_call",
        # A tell reached the team, or waits for its next boundary.
        "tell",
    }


# -- hiding -------------------------------------------------------------------------------


def test_after_an_undo_the_context_holds_no_hidden_line_and_spend_reads_as_before(
    session: Session,
) -> None:
    session.record_user("first ask")
    session.record_assistant("first answer")
    charge(session)
    before_turn = messages(session)
    turn = session.record_user("second ask").seq
    exchange(session, "r1", "shell", {"command": "ls"}, "a.py\n")
    charge(session)
    session.record_cap(5.0)
    money = kept(session)
    hidden_from = session.next_seq - turn
    record_rewind(session, undo(turn), (), ())
    assert messages(session) == before_turn
    assert kept(session) == money
    assert session.state.turn(turn) is None and session.state.line(turn) is None
    assert session.state.result("r1") is None and session.state.call("r1") is None
    assert len(session.state.hidden) == hidden_from - 2, "every line but the spend and cap"
    with reopen(session) as again:
        assert messages(again) == before_turn, "replay hides exactly what the live fold did"
        assert kept(again) == money


def test_a_redo_before_the_next_turn_restores_the_context_byte_for_byte(
    session: Session,
) -> None:
    session.record_user("first ask")
    session.record_assistant("first answer")
    turn = session.record_user("second ask").seq
    exchange(session, "r1", "shell", {"command": "ls"}, "a.py\n")
    whole = messages(session)
    seq = record_rewind(session, undo(turn), (), ())
    session.append(redo(seq))
    assert messages(session) == whole
    assert session.state.result("r1") is not None
    with reopen(session) as again:
        assert messages(again) == whole


def test_a_conversation_redo_after_a_new_turn_is_refused_with_the_reason(
    session: Session,
) -> None:
    turn = session.record_user("first ask").seq
    session.record_assistant("first answer")
    seq = record_rewind(session, undo(turn), (), ())
    later = session.record_user("something else").seq
    with pytest.raises(
        PayloadError, match=f"seq {later} was written after the undo at seq {seq}"
    ):
        session.append(redo(seq))


def test_a_conversation_redo_after_any_conversation_line_is_refused(session: Session) -> None:
    """Not only a user turn: an assistant line after the undo moves the conversation on, and a
    redo would bring back call ids it may reuse (decision 7)."""
    turn = session.record_user("first ask").seq
    session.record_assistant("", [ToolCall(id="r1", name="shell", arguments={"command": "ls"})])
    session.record_tool("r1", "a.py")
    seq = record_rewind(session, undo(turn), (), ())
    said = session.record_assistant(
        "", [ToolCall(id="r1", name="shell", arguments={"command": "ls"})]
    ).seq
    session.record_tool("r1", "b.py")
    with pytest.raises(PayloadError, match=f"seq {said} was written after the undo"):
        session.append(redo(seq))


def test_the_kept_code_notice_does_not_end_the_redo(session: Session) -> None:
    turn = session.record_user("edit auth").seq
    whole = messages(session)
    seq = record_rewind(session, undo(turn), ("auth.py",), ("r7",))
    session.append(redo(seq))
    assert messages(session)[: len(whole)] == whole  # type: ignore[index]


def test_a_code_only_undo_hides_nothing_and_its_redo_is_not_bound_by_the_next_turn(
    session: Session,
) -> None:
    turn = session.record_user("change it").seq
    whole = messages(session)
    seq = record_rewind(session, undo(turn, code=True, conversation=False), (), ())
    assert messages(session) == whole
    session.record_user("and now")
    session.append(redo(seq, code=True, conversation=False))


@pytest.mark.parametrize(
    ("line", "why"),
    [
        (undo(1, conversation=False), "the code, the conversation, or both"),
        (undo(2), "not a user turn the person can see"),
        (undo(99), "not a user turn the person can see"),
    ],
    ids=["takes-nothing", "not-a-user-turn", "no-such-turn"],
)
def test_an_undo_names_a_visible_user_turn_and_takes_something(
    session: Session, line: UndoLine, why: str
) -> None:
    session.record_user("ask")
    session.record_assistant("answer")
    with pytest.raises(PayloadError, match=why):
        session.append(line)


def test_an_undo_cannot_name_a_turn_another_undo_hid(session: Session) -> None:
    session.record_user("one")
    turn = session.record_user("two").seq
    record_rewind(session, undo(turn), (), ())
    with pytest.raises(PayloadError, match="not a user turn the person can see"):
        session.append(undo(turn))


def test_an_undo_and_a_redo_are_written_at_a_safe_point(session: Session) -> None:
    turn = session.record_user("ask").seq
    session.record_assistant("", [ToolCall(id="r1", name="shell", arguments={"command": "ls"})])
    with pytest.raises(PayloadError, match="call r1 is still waiting"):
        session.append(undo(turn))


def test_a_redo_names_an_undo_once(session: Session) -> None:
    turn = session.record_user("ask").seq
    with pytest.raises(PayloadError, match=f"seq {turn} is not an undo"):
        session.append(redo(turn))
    seq = record_rewind(session, undo(turn), (), ())
    session.append(redo(seq))
    with pytest.raises(PayloadError, match="already has its conversation redone"):
        session.append(redo(seq))


def test_a_redo_takes_back_only_a_half_the_undo_took(session: Session) -> None:
    turn = session.record_user("ask").seq
    seq = record_rewind(session, undo(turn), (), ())
    with pytest.raises(PayloadError, match="did not take back the code"):
        session.append(redo(seq, code=True, conversation=False))
    with pytest.raises(PayloadError, match="the code, the conversation, or both"):
        session.append(redo(seq, code=False, conversation=False))


def test_after_the_conversation_moves_on_only_the_code_half_is_redone(session: Session) -> None:
    """The code half stays offered; the conversation stays hidden."""
    turn = session.record_user("change it").seq
    seq = record_rewind(session, undo(turn, code=True), (), ())
    session.record_user("something else")
    with pytest.raises(PayloadError, match="can no longer be redone"):
        session.append(redo(seq, code=True, conversation=True))
    session.append(redo(seq, code=True, conversation=False))
    assert session.state.turn(turn) is None, "a code redo shows nothing again"


def test_a_constraint_from_a_hidden_turn_no_longer_binds_and_cannot_be_cited(
    session: Session,
) -> None:
    session.record_user("one")
    turn = session.record_user("never use tabs").seq
    session.append(
        ConstraintP("C-1", "no tabs", (turn,), "user", "session", None, turn + 1, None, None)
    )
    assert session.state.ledger.live_constraints()
    record_rewind(session, undo(turn), (), ())
    assert not session.state.ledger.live_constraints()
    assert not session.state.is_user_words(turn)


def test_hidden_seqs_reads_only_standing_conversation_undos(session: Session) -> None:
    first = session.record_user("one").seq
    second = session.record_user("two").seq
    a = record_rewind(session, undo(second), (), ())
    session.append(redo(a))
    record_rewind(session, undo(first, code=True, conversation=False), (), ())
    assert hidden_seqs(list(session.state.lines())) == frozenset()


# -- compaction ---------------------------------------------------------------------------


def test_a_compaction_after_a_rewind_summarises_no_hidden_line(
    session: Session, ws: Path
) -> None:
    work_unit(session, "alpha")
    turn = session.record_user("the secret plan, never to be seen again").seq
    exchange(session, "r-secret", "shell", {"command": "ls secret"}, "hidden-result\n" * 30)
    session.record_assistant("acknowledged the secret plan")
    record_rewind(session, undo(turn), (), ())
    work_unit(session, "beta")
    session.record_user("thanks, what next?")
    session.record_assistant("let us see")
    done = cut(session, ws)
    assert done.cut is not None
    text = repr(messages(session))
    for word in ("secret", "hidden-result", "r-secret"):
        assert word not in done.cut.block and word not in done.cut.standing
        assert word not in text


def test_a_rewind_hides_a_compaction_and_its_boundary_returns(
    session: Session, ws: Path
) -> None:
    work_unit(session, "alpha")
    turn = session.record_user("go on").seq
    work_unit(session, "beta")
    session.record_user("thanks, what next?")
    session.record_assistant("let us see")
    assert cut(session, ws).cut is not None
    record_rewind(session, undo(turn), (), ())
    assert session.state.cuts() == () and session.state.boundary == 0


def test_a_rewound_coder_turn_opens_no_unit_and_a_later_cut_still_lands(
    session: Session, ws: Path
) -> None:
    """The episode is kept (its run happened) but its turn is hidden: it is no unit of the
    visible conversation, so it neither stalls the units after it nor reads as in flight."""
    work_unit(session, "alpha")
    turn = session.next_seq
    taken_back = work_unit(session, "secret")
    record_rewind(session, undo(turn), (), ())
    kept = work_unit(session, "beta")
    session.record_user("thanks, what next?")
    session.record_assistant("let us see")
    assert session.state.episode(taken_back) is not None
    ran = [u.run_id for u in completed_units(session.state)]
    assert taken_back not in ran and kept in ran
    done = cut(session, ws)
    assert done.cut is not None and "secret" not in done.cut.block


# -- the kept-code notice -------------------------------------------------------------------


def notices(session: Session) -> list[str]:
    return [
        t.text
        for t in session.state.turns()
        if t.notice is not None and t.notice.trigger is NoticeTrigger.FORK_EFFECTS
    ]


def test_a_conversation_only_rewind_names_the_kept_code_once(session: Session) -> None:
    turn = session.record_user("edit auth").seq
    record_rewind(session, undo(turn), ("auth.py", "retry.py"), ("r7",))
    assert notices(session) == [
        "code changed in turns you no longer see: auth.py, retry.py (run r7)"
    ]


@pytest.mark.parametrize(
    ("line", "paths"),
    [
        (undo(1, code=True, conversation=True), ("a.py",)),
        (undo(1, code=True, conversation=False), ("a.py",)),
        (undo(1), ()),
    ],
    ids=["code-restored", "code-only", "nothing-kept"],
)
def test_no_notice_when_no_kept_code_is_unremembered(
    session: Session, line: UndoLine, paths: tuple[str, ...]
) -> None:
    session.record_user("edit")
    record_rewind(session, line, paths, ("r1",))
    assert notices(session) == []


def test_the_notice_names_every_run(session: Session) -> None:
    turn = session.record_user("edit").seq
    record_rewind(session, undo(turn), ("a.py",), ("r1", "r2"))
    assert notices(session) == ["code changed in turns you no longer see: a.py (runs r1, r2)"]


# -- what a rewind cannot take back ------------------------------------------------------------


def shell(session: Session, cid: str, command: str) -> int:
    session.record_assistant(
        # A host list, as a session recorded in older versions wrote it: it still opened the
        # network.
        "",
        [ToolCall(id=cid, name="shell", arguments={"command": command, "network": ["x"]})],
    )
    return session.record_tool(cid, "ok").seq


def test_a_shell_call_whose_class_opened_the_network_is_listed(session: Session) -> None:
    """Review of #592: an approved push opens the network from its class with no `network`
    argument; its result header says so, and the listing reads that. At
    `sandbox none` every command had the network, said once in the run header, not per call."""
    for cid, header in (
        ("p1", "exit 0 · 3 ms · sandbox native · network on · repo read-only · t2 · op1"),
        ("p2", "exit 0 · 3 ms · sandbox native · network off · repo read-only · t0 · op2"),
        ("p3", "exit 0 · 3 ms · sandbox none · network on · repo not protected · t2 · op3"),
    ):
        session.record_assistant(
            "", [ToolCall(id=cid, name="shell", arguments={"command": f"git push {cid}"})]
        )
        hi = session.record_tool(cid, f"{header}\nok").seq
    listed = [(e.kind, e.detail) for e in journaled_effects(session.state, None, hi=hi)]
    assert listed == [("network", "git push p1")]


def test_journaled_effects_reads_a_range(session: Session) -> None:
    session.record_user("one")
    shell(session, "s1", "curl one")
    lo = session.record_user("two").seq
    shell(session, "s2", "curl two")
    hi = session.state.last_seq
    session.record_user("three")
    shell(session, "s3", "curl three")
    assert [e.detail for e in journaled_effects(session.state, None, hi=hi)] == [
        "curl one",
        "curl two",
    ]
    ranged = journaled_effects(session.state, None, lo=lo, hi=hi)
    assert [e.detail for e in ranged] == ["curl two"]


def _run(rid: str, *, worktree: bool) -> tuple[JournalEvent, ...]:
    """A task loop's run that committed its task: in place, or in a worktree. The
    recording this read before was the packet loop's, which no reader takes now."""
    isolation: Literal["worktree", "in_place"] = "worktree" if worktree else "in_place"
    return (
        run_start(rid, "build it", isolation=isolation),
        *effect(rid, 2, "op-commit", EffectKind.TASK_COMMIT),
        run_end(rid, 5),
    )


def _snapshot(session: Session, turn: int) -> None:
    for phase in ("before", "after"):
        record_snapshot(
            session,
            CodeSnapshotLine(
                turn=turn,
                phase=phase,  # type: ignore[arg-type]
                key="in-place",
                mode="in_place",
                root="/repo",
                tree="a" * 40,
                ref=f"refs/chickadee/turns/S/{turn}/in-place/{phase}",
                head=None,
                skipped=(),
            ),
        )


def _forked_kinds(tmp_path: Path, *, snapshot: bool, worktree: bool = False) -> list[str]:
    """One recorded run, in place (or in a worktree), dispatched in one turn; the parent forked
    after it. The kinds the child's fork notice lists."""
    name = "rejected"
    rid = run_id(name)
    root = tmp_path / "sessions"
    with Session.create(root, "P") as parent:
        turn = parent.record_user("build it").seq
        dispatch = parent.record_assistant(
            "", [ToolCall(id="c1", name="coder", arguments={"request": {"goal": name}})]
        )
        parent.record_episode(EpisodeLine(rid, "planned", Tier.T1, None, (), dispatch.seq))
        parent.record_tool("c1", "done", run_id=rid)
        shell(
            parent, "s1", "curl it"
        )  # what no rewind takes back, so the notice is always sent
        if snapshot:
            _snapshot(parent, turn)
        at = parent.state.last_seq
    records = FakeRecords(runs={rid: _run(rid, worktree=worktree)})
    with Session.fork(root, "P", at, "try again", "C", records=records, code="keep") as child:
        told = notices(child)
    if not told:
        return []
    return [row.split()[2] for row in told[0].splitlines()[2:-1]]


WORKTREE_KINDS = {"bootstrap", "task_commit"}


def test_in_place_a_snapshotted_turn_lists_no_worktree_kind(tmp_path: Path) -> None:
    listed = _forked_kinds(tmp_path, snapshot=True)
    assert "network" in listed
    assert not WORKTREE_KINDS & set(listed)


def test_in_place_a_turn_with_no_snapshot_lists_them(tmp_path: Path) -> None:
    assert "task_commit" in _forked_kinds(tmp_path, snapshot=False)


def test_a_worktree_run_never_lists_them(tmp_path: Path) -> None:
    assert not WORKTREE_KINDS & set(_forked_kinds(tmp_path, snapshot=False, worktree=True))
