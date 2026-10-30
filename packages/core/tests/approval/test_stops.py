"""The stops: `approval/stops.ask_stop`, the
D187 gate's rounds loop for one `ApprovalStop` (the gate retired with the work tier).

What these pin, the gate's behaviour carried over: approve binds the text last shown; an edit
is a new text with its own sha, shown again; an edit that is empty or changes nothing, and an
empty note, are refused and the same text shown again saying why; a note and a cancel end the
stop with the person's words verbatim; each showing has its own id; the rounds bound ends it
cancelled."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from chickadee.approval.stops import (
    STOP_ROUNDS,
    StopOutcome,
    StopWithdrawn,
    WorkspacePreview,
    ask_stop,
    stop_of,
)
from chickadee.errors import HarnessBug
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import ApprovalStop, StopAnswer, StopDecision, StopKind

PLAN = "# Plan: segment tree\n\n1. **Build and update.**\n\nFinal check: the suite.\n"
WORKSPACE = WorkspacePreview("worktree", "main")


class ScriptedStops:
    """A port that answers each showing in turn, keeping what it was shown."""

    def __init__(self, answers: Sequence[StopDecision]) -> None:
        self.answers = list(answers)
        self.shown: list[ApprovalStop] = []

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        self.shown.append(stop)
        return self.answers.pop(0)


def _plan() -> ApprovalStop:
    return stop_of(StopKind.PLAN, "athena", PLAN, WORKSPACE)


def test_a_stop_shows_the_text_exactly_and_binds_its_sha() -> None:
    stop = _plan()
    assert (stop.text, stop.sha, stop.round, stop.refused) == (PLAN, sha256_hex(PLAN), 1, None)
    assert (stop.kind, stop.author, stop.workspace, stop.source_ref) == (
        StopKind.PLAN,
        "athena",
        "worktree",
        "main",
    )
    assert stop.id.startswith(f"plan-{sha256_hex(PLAN)[:12]}-")


def test_approve_binds_the_text_shown() -> None:
    port = ScriptedStops([StopDecision(answer=StopAnswer.APPROVE)])
    outcome = ask_stop(_plan(), port)
    assert outcome == StopOutcome(StopAnswer.APPROVE, PLAN, sha256_hex(PLAN))
    assert len(port.shown) == 1


def test_an_edit_is_shown_again_and_approve_binds_the_last_shown() -> None:
    edited = PLAN + "2. **Range query.**\n"
    port = ScriptedStops(
        [
            StopDecision(answer=StopAnswer.EDIT, edited=edited),
            StopDecision(answer=StopAnswer.APPROVE),
        ]
    )
    outcome = ask_stop(_plan(), port)
    assert outcome == StopOutcome(StopAnswer.APPROVE, edited, sha256_hex(edited))
    first, second = port.shown
    assert (second.text, second.sha, second.round, second.refused) == (
        edited,
        sha256_hex(edited),
        2,
        None,
    )
    assert first.id != second.id, "each showing is answered once"


@pytest.mark.parametrize(
    ("said", "why"),
    [
        (
            StopDecision(answer=StopAnswer.EDIT, edited="  \n"),
            "the edit was not used: it was empty",
        ),
        (StopDecision(answer=StopAnswer.EDIT), "the edit was not used: it was empty"),
        (StopDecision(answer=StopAnswer.EDIT, edited=PLAN), "the edit changed nothing"),
        (
            StopDecision(answer=StopAnswer.NOTE, note=""),
            "a note says what to change: it was empty",
        ),
        (StopDecision(answer=StopAnswer.NOTE), "a note says what to change: it was empty"),
    ],
)
def test_an_answer_that_cannot_be_used_is_refused_and_the_same_text_shown_again(
    said: StopDecision, why: str
) -> None:
    port = ScriptedStops([said, StopDecision(answer=StopAnswer.APPROVE)])
    outcome = ask_stop(_plan(), port)
    assert outcome == StopOutcome(StopAnswer.APPROVE, PLAN, sha256_hex(PLAN))
    again = port.shown[1]
    assert (again.text, again.sha, again.round, again.refused) == (
        PLAN,
        sha256_hex(PLAN),
        2,
        why,
    )


def test_a_note_ends_the_stop_with_the_persons_words_verbatim() -> None:
    note = "  only do tasks 1-2 "
    port = ScriptedStops([StopDecision(answer=StopAnswer.NOTE, note=note)])
    assert ask_stop(_plan(), port) == StopOutcome(StopAnswer.NOTE, PLAN, sha256_hex(PLAN), note)


def test_cancel_ends_the_stop_with_the_reason() -> None:
    port = ScriptedStops([StopDecision(answer=StopAnswer.CANCEL, note="not now")])
    assert ask_stop(_plan(), port) == StopOutcome(
        StopAnswer.CANCEL, PLAN, sha256_hex(PLAN), "not now"
    )


def test_the_rounds_bound_ends_it_cancelled() -> None:
    port = ScriptedStops([StopDecision(answer=StopAnswer.NOTE)] * STOP_ROUNDS)
    outcome = ask_stop(_plan(), port)
    assert outcome.answer is StopAnswer.CANCEL
    # Moved: the harness's reason is `bound`, never the person's `note`, so
    # it is not recorded or reported as their words.
    assert outcome.note is None
    assert outcome.bound == f"the plan was shown {STOP_ROUNDS} times without an approval"
    assert [s.round for s in port.shown] == list(range(1, STOP_ROUNDS + 1))
    assert len({s.id for s in port.shown}) == STOP_ROUNDS
    with pytest.raises(HarnessBug, match="at least once"):
        ask_stop(_plan(), ScriptedStops([]), rounds=0)


def test_an_approval_with_words_is_shown_again_saying_where_they_go() -> None:
    """An approval binds the text shown and nothing else: words with it would reach nobody,
    so the stop is shown again, and a plain approval then binds the text."""
    port = ScriptedStops(
        [
            StopDecision(answer=StopAnswer.APPROVE, note="but skip the migrations"),
            StopDecision(answer=StopAnswer.APPROVE),
        ]
    )
    outcome = ask_stop(_plan(), port)
    assert (outcome.answer, outcome.note) == (StopAnswer.APPROVE, None)
    assert port.shown[1].refused is not None and "as a note" in port.shown[1].refused


def test_an_edit_on_the_last_round_is_not_what_the_bound_ends_on() -> None:
    """T-3.157 review: an edit sent at the last showing is never shown, so the bound ends on
    the text the person last saw and says the edit went unshown."""
    first = _plan()
    port = ScriptedStops(
        [
            StopDecision(answer=StopAnswer.NOTE),
            StopDecision(answer=StopAnswer.EDIT, edited="# B\n"),
        ]
    )
    outcome = ask_stop(first, port, rounds=2)
    assert (outcome.answer, outcome.text, outcome.sha) == (
        StopAnswer.CANCEL,
        first.text,
        first.sha,
    )
    assert outcome.bound is not None and "was not shown" in outcome.bound


def test_a_stop_whose_wait_was_ended_is_withdrawn_whatever_the_port_said() -> None:
    """D240 §2.9a: a stop has no timeout while the person is there, so its `cancel` token is
    what ends the wait, and an answer the port gives after it is never taken for the person's:
    as `confirm_action`'s rule has it."""
    cancel = threading.Event()

    class Ended:
        def confirm_stop(
            self, stop: ApprovalStop, *, cancel: threading.Event | None = None
        ) -> StopDecision:
            assert cancel is not None
            cancel.set()
            return StopDecision(answer=StopAnswer.APPROVE)

    with pytest.raises(StopWithdrawn):
        ask_stop(_plan(), Ended(), cancel=cancel)
