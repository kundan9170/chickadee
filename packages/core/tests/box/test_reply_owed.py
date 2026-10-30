"""`reply` is owed: Athena's answer to each message
from Zeus delivered to her, through `box/desk.TaskDesk` and `box/tells.Tells`.

What these pin: while a delivered tell has no reply, `delegate` and `plan` are refused naming
it; `reply` writes `tell_reply` and tells Zeus (`replied`), once per tell, a resumed call
answered from its line; a reply to a message she was never given, or one answered already, is
refused, as is an empty one; a tell with no worker round open reaches her before her next model
call; a run ending with one owed lists it `unanswered`, and Zeus reads that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from box_helpers import WORKER_RUN, EpisodeRig, athena_call, episode_rig
from engines import answer, calls
from repos import init_repo

from chickadee.box.report import CoderReport
from chickadee.box.tells import Tells, Told, unanswered
from chickadee.boxctx.athena import athena_head
from chickadee.context.results import InvalidArgument
from chickadee.journal.projection import RunSpend
from chickadee.schema.journal import EventKind, TellDeliveredP, TellOutcome, TellReplyP
from chickadee.zeus.delegate import render_report

APP = "def go() -> int:\n    return 1\n"
PLAN = "# Plan: go returns 2\n\n1. **Return 2.** Change `go()`.\n\nFinal check: the suite.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
SAID = "use argon2 instead of bcrypt"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


class _Told:
    """The session's `tell` lines for the run, and what Zeus was told of each reply."""

    def __init__(self) -> None:
        self.lines: list[Told] = []
        self.replies: list[tuple[str, TellOutcome, str]] = []
        self.tells: Tells | None = None

    def __call__(self, journal: Any) -> Tells:
        self.tells = Tells(
            WORKER_RUN,
            journal,
            lambda: tuple(self.lines),
            replied=lambda *said: self.replies.append(said),
        )
        return self.tells

    def tell(self, text: str = SAID) -> None:
        assert self.tells is not None
        self.lines.append(Told(f"tell-{len(self.lines) + 1}", text, (3,), WORKER_RUN))
        self.tells.wake()


def _delivered(root: Path, athena: list[object] | None = None) -> tuple[EpisodeRig, _Told]:
    """A plan approved and a tell delivered to her with no worker round open."""
    told = _Told()
    rig = episode_rig(root, athena or [], [], tells=told)
    assert rig.desk.plan(PLAN, call_id="a-plan").startswith("the person approved the plan")
    told.tell()
    assert told.tells is not None
    assert [t.id for t in told.tells.deliver()] == ["tell-1"]
    return rig, told


def _refused(got: object) -> str:
    assert isinstance(got, InvalidArgument), got
    assert got.field == "tell"
    return f"{got.why}. Do: {got.do}"


def test_delegate_and_plan_wait_for_the_reply(root: Path) -> None:
    rig, _ = _delivered(root)

    assert "reply to tell-1 first" in _refused(
        rig.desk.delegate("Return 2", BRIEF, call_id="d")
    )
    assert "reply to tell-1 first" in _refused(rig.desk.plan(PLAN, call_id="a-plan-2"))
    assert rig.desk.plan(PLAN, call_id="a-plan").startswith("the person approved the plan"), (
        "a plan call already written is answered from its line, owed or not (a resume)"
    )


def test_a_reply_is_journaled_and_tells_zeus_once(root: Path) -> None:
    rig, told = _delivered(root)

    got = rig.desk.reply("tell-1", "needs_design", "new behaviour", call_id="a-y1")

    assert got == "replied to tell-1: needs_design"
    [line] = rig.journal.projection().lines(TellReplyP)
    assert (line.payload.tell, line.payload.outcome, line.payload.call_id) == (
        "tell-1",
        "needs_design",
        "a-y1",
    )
    assert rig.journal.get_blob(line.payload.text_sha) == b"new behaviour"
    assert told.replies == [("tell-1", "needs_design", "new behaviour")]
    # The same call again (a resume): answered from its line, Zeus not told twice.
    assert rig.desk.reply("tell-1", "needs_design", "new behaviour", call_id="a-y1") == (
        "replied to tell-1: needs_design"
    )
    assert len(told.replies) == 1
    assert isinstance(rig.desk.delegate("Return 2", BRIEF, call_id="a-d1"), str)


def test_a_reply_to_no_message_of_hers_or_twice_or_empty_is_refused(root: Path) -> None:
    rig, told = _delivered(root)
    assert "tell-9 is no message of this run" in _refused(
        rig.desk.reply("tell-9", "applied", "done", call_id="a-y0")
    )
    empty = rig.desk.reply("tell-1", "applied", "  ", call_id="a-y1")
    assert isinstance(empty, InvalidArgument) and empty.field == "text"
    rig.desk.reply("tell-1", "applied", "T1 covers it", call_id="a-y2")
    assert "tell-1 was answered already" in _refused(
        rig.desk.reply("tell-1", "declined", "no", call_id="a-y3")
    )
    assert [r[1] for r in told.replies] == ["applied"]


def test_a_tell_with_no_round_open_reaches_her_before_her_next_model_call(root: Path) -> None:
    told = _Told()

    def reads_while_told(request: object) -> Any:
        told.tell()
        return calls(athena_call("read", "a1", path="app.py"))

    rig = episode_rig(
        root,
        [
            reads_while_told,
            calls(athena_call("reply", "a2", tell="tell-1", outcome="declined", text="no")),
            answer("nothing to do"),
        ],
        [],
        tells=told,
    )
    got = rig.athena.start(athena_head("You are Athena.", "P", "B", design="# Design"))

    assert got.report == "nothing to do"
    told_her = rig.athena_rung.requests[1].messages[-1]
    assert told_her.role == "user"
    assert "Message (tell-1), passed on from the user's turn 3:\n  " + SAID in told_her.content
    assert "answer each message (reply)" in told_her.content
    [delivered] = rig.journal.projection().lines(TellDeliveredP)
    assert (delivered.payload.task_id, delivered.payload.round) == (None, None)
    assert told.replies == [("tell-1", "declined", "no")]


def test_a_run_ending_with_a_reply_owed_lists_it_unanswered(root: Path) -> None:
    rig, _ = _delivered(root)
    rig.journal.append(
        EventKind.TELL_DELIVERED, TellDeliveredP(tell="tell-2", task_id=None, round=None)
    )
    rig.desk.reply("tell-1", "applied", "T1 covers it", call_id="a-y1")

    assert unanswered(rig.journal.projection()) == ("tell-2",)


def test_zeus_reads_the_unanswered_tells_in_the_report() -> None:
    report = CoderReport(
        outcome="PARTIAL", why="T2 is open", cost=RunSpend(), unanswered=("tell-2",)
    )
    assert "the team did not answer: tell-2" in str(render_report(report))


def test_a_reply_to_a_message_not_yet_delivered_says_so(root: Path) -> None:
    rig, told = _delivered(root)
    rig.desk.reply("tell-1", "applied", "T1 covers it", call_id="a-y1")
    told.tell("and use argon2id")  # written, not yet delivered
    assert "tell-2 has not reached you yet" in _refused(
        rig.desk.reply("tell-2", "applied", "done", call_id="a-y2")
    )
