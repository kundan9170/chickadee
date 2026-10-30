"""`review(task, "continue")`: the desk's answer to a
round held for a message from Zeus, through `box/desk.TaskDesk` with the run's `Tells`.

What these pin: a held round answers her `delegate` with the round so far and the message, each
delivered then (`tell_delivered`, naming the task and round); it fires no `round_ended` hook;
the task is `held`; `continue` opens the next round with no message of hers, on the
conversation as it stands, and is answered with that round's report; `continue` after a round
that was not held is refused; `feedback` after a held round is her words, as ever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from box_helpers import WORKER_RUN, EpisodeRig, episode_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.box.tasks import TaskBook
from chickadee.box.tells import Tells, Told
from chickadee.context.results import InvalidArgument
from chickadee.schema.journal import HookP, TaskReviewedP, TellDeliveredP
from chickadee.tools.loop import HELD_NOT_RUN

APP = "def go() -> int:\n    return 1\n"
PLAN = "# Plan: go returns 2\n\n1. **Return 2.** Change `go()`.\n\nFinal check: the suite.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
SAID = "use argon2 instead of bcrypt"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


class _Told:
    """The session's `tell` lines for the run; `tell` writes one and wakes the run."""

    def __init__(self) -> None:
        self.lines: list[Told] = []
        self.tells: Tells | None = None

    def __call__(self, journal: Any) -> Tells:
        self.tells = Tells(WORKER_RUN, journal, lambda: tuple(self.lines))
        return self.tells

    def tell(self, text: str = SAID) -> None:
        assert self.tells is not None
        self.lines.append(Told(f"tell-{len(self.lines) + 1}", text, (3,), WORKER_RUN))
        self.tells.wake()


def _held_rig(root: Path, after: list[object]) -> tuple[EpisodeRig, _Told, str]:
    """A plan approved, T1 delegated, and its first round held by a tell written while the
    worker's model answered; `after` is the worker's script from there. Returns the rig, the
    tells and her `delegate`'s answer."""
    told = _Told()

    def answers_while_told(request: object) -> Any:
        told.tell()
        return calls(call("edit", "w1", path="app.py", old="return 1", new="return 2"))

    rig = episode_rig(root, [], [answers_while_told, *after], tells=told)
    assert rig.desk.plan(PLAN, call_id="a-plan").startswith("the person approved the plan")
    got = rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    assert isinstance(got, str), got
    return rig, told, got


def test_a_held_round_answers_delegate_with_the_round_so_far_and_the_message(
    root: Path,
) -> None:
    rig, _, got = _held_rig(root, [])

    assert got.startswith("T1 'Return 2' held for a message (round 1, unfinished; the worker")
    assert "Message (tell-1), passed on from the user's turn 3:\n  " + SAID in got
    assert "answer each message (reply)" in got
    run = rig.journal.projection()
    [delivered] = [p.payload for p in run.lines(TellDeliveredP)]
    assert (delivered.tell, delivered.task_id, delivered.round) == ("tell-1", "T1", 1)
    entry = TaskBook(run).get("T1")
    assert entry is not None and entry.state == "held"
    assert not [p for p in run.lines(HookP) if p.payload.event == "round_ended"], (
        "a held round is no round's end to the hooks"
    )


def test_continue_opens_the_next_round_with_no_message_of_hers(root: Path) -> None:
    rig, _, _ = _held_rig(
        root,
        [
            calls(call("edit", "w2", path="app.py", old="return 1", new="return 2")),
            answer("go returns 2."),
        ],
    )

    got = rig.desk.review("T1", "continue", None, call_id="a-r1")

    assert isinstance(got, str) and "go returns 2." in got
    resumed = rig.worker_rung.requests[1].messages
    assert (resumed[-1].role, resumed[-1].content) == ("tool", HELD_NOT_RUN)
    [reviewed] = rig.journal.projection().lines(TaskReviewedP)
    assert (reviewed.payload.answer, reviewed.payload.call_id) == ("continue", "a-r1")
    entry = TaskBook(rig.journal.projection()).get("T1")
    assert entry is not None and (entry.state, entry.rounds) == ("reported", 2)


def test_feedback_after_a_held_round_is_her_words(root: Path) -> None:
    rig, _, _ = _held_rig(root, [answer("switched to argon2")])

    got = rig.desk.review("T1", "feedback", "Use argon2 in auth/hash.py.", call_id="a-r1")

    assert isinstance(got, str) and "switched to argon2" in got
    resumed = rig.worker_rung.requests[1].messages
    assert (resumed[-1].role, resumed[-1].content) == ("user", "Use argon2 in auth/hash.py.")


def test_continue_after_a_round_that_was_not_held_is_refused(root: Path) -> None:
    rig = episode_rig(
        root,
        [],
        [
            calls(call("edit", "w1", path="app.py", old="return 1", new="return 2")),
            answer("done"),
        ],
        tells=_Told(),
    )
    rig.desk.plan(PLAN, call_id="a-plan")
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")

    got = rig.desk.review("T1", "continue", None, call_id="a-r1")

    assert isinstance(got, InvalidArgument) and got.field == "answer"
    assert "continue resumes a round held for a message" in got.why
    assert rig.journal.projection().lines(TaskReviewedP) == ()
