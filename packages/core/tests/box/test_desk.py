"""The desk:
`box/desk.TaskDesk`, the harness side of Athena's `plan`, `delegate` and `review`.

What these pin: stop 2 is asked on her first plan and never again, a revision shown and not
asked; an edited plan is the plan now; a note answers her and writes nothing; a cancel ends the
episode. One writer: no delegation before a plan, none while a task is open, a review names
the open task. Each review answer: accept commits with the operation's trailer, feedback is the
worker's next round on the same conversation, take over makes the task hers, re-cut replaces it
and drop gives it up; what stays is `fresh`'s: every file or the listed ones put back
under their own step, or the changes kept; the re-cut depth is the harness's and capped. Her
edit tools are refused while she holds no task.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from box_helpers import APP, EpisodeRig, athena_call, episode_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import git, init_repo

from chickadee.approval.stops import StopOutcome
from chickadee.box import desk as desk_module
from chickadee.box.commit import Projector
from chickadee.box.desk import NO_TASK_HELD
from chickadee.box.tasks import TaskBook
from chickadee.box.worker import WorkerRun
from chickadee.boxctx.athena import athena_head
from chickadee.boxctx.conversation import message_of
from chickadee.context.results import InvalidArgument
from chickadee.executor.reconcile import TRAILER
from chickadee.journal.effects import EffectKind
from chickadee.roles import ATHENA_TASK
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import ApprovalStop, StopAnswer
from chickadee.schema.journal import (
    BoxMessageP,
    EventKind,
    PlanWrittenP,
    TaskDelegatedP,
    TaskReviewedP,
)

PLAN = "# Plan: go returns 2\n\n1. **Return 2.** Change `go()`.\n\nFinal check: the suite.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
EDIT = call("edit", "w1", path="app.py", old="return 1", new="return 2")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _rig(root: Path, worker: list[object], **kwargs: object) -> EpisodeRig:
    return episode_rig(root, [], worker, **kwargs)  # type: ignore[arg-type]


def _planned(rig: EpisodeRig) -> None:
    assert rig.desk.plan(PLAN, call_id="a-plan").startswith("the person approved the plan")


def _refusal(got: object, field: str) -> str:
    assert isinstance(got, InvalidArgument), got
    assert got.field == field
    return f"{got.why}. Do: {got.do}"


# -- the plan and stop 2 -----------------------------------------------------------------------


def test_stop_2_is_asked_on_the_first_plan_and_a_revision_is_shown_never_asked(
    root: Path,
) -> None:
    rig = _rig(root, [])
    _planned(rig)
    revised = rig.desk.plan(PLAN + "2. **And a test.**\n", call_id="a-plan-2")
    assert revised.startswith("plan revised (version 2): shown to the person, not asked")
    [stop] = rig.stops
    assert (stop.kind, stop.author, stop.text, stop.sha) == (
        "plan",
        "athena",
        PLAN,
        sha256_hex(PLAN),
    )
    lines = [p.payload for p in rig.journal.projection().lines(PlanWrittenP)]
    assert [(p.call_id, p.version, p.asked, p.by) for p in lines] == [
        ("a-plan", 1, True, "athena"),
        ("a-plan-2", 2, False, "athena"),
    ]
    assert rig.journal.get_blob(lines[0].sha).decode() == PLAN


def test_an_edited_plan_is_the_plan_now(root: Path) -> None:
    edited = PLAN.replace("Return 2", "Return two")

    def edit(stop: ApprovalStop) -> StopOutcome:
        return StopOutcome(StopAnswer.APPROVE, edited, sha256_hex(edited))

    rig = _rig(root, [], stop=edit)
    got = rig.desk.plan(PLAN, call_id="a-plan")
    assert got.startswith("the person approved the plan as they edited it")
    assert edited in got
    [line] = rig.journal.projection().lines(PlanWrittenP)
    assert line.payload.sha == sha256_hex(edited)


def test_a_note_answers_her_verbatim_and_writes_nothing(root: Path) -> None:
    answers = [
        StopOutcome(StopAnswer.NOTE, PLAN, sha256_hex(PLAN), "only task 1"),
        StopOutcome(StopAnswer.APPROVE, PLAN, sha256_hex(PLAN)),
    ]
    rig = _rig(root, [], stop=lambda _s: answers.pop(0))
    got = rig.desk.plan(PLAN, call_id="a-plan")
    assert "their note, verbatim:\nonly task 1" in got
    assert rig.journal.projection().lines(PlanWrittenP) == ()
    # Still her first plan: asked again.
    assert rig.desk.plan(PLAN, call_id="a-plan-2").startswith("the person approved the plan")
    assert len(rig.stops) == 2


def test_a_stop_2_the_rounds_bound_ended_is_not_the_persons_cancel(root: Path) -> None:
    """T-3.157 review: the harness's reason is journaled as `bound`, so the episode, resumed
    or not, says the stop ended unapproved and never that the person cancelled; and the line
    names the text last shown, stored, not Athena's first text."""
    shown = "# Plan: edited\n"
    out = StopOutcome(StopAnswer.CANCEL, shown, sha256_hex(shown), bound="shown 20 times")
    rig = _rig(root, [], stop=lambda _s: out)
    got = rig.desk.plan(PLAN, call_id="a-plan")
    assert got.startswith("the plan was not approved: shown 20 times")
    assert rig.desk.ended is not None and rig.desk.ended.why.startswith(
        "the plan was not approved"
    )
    assert rig.desk.ended.note is None
    [line] = rig.journal.projection().lines(PlanWrittenP)
    assert line.payload.sha == sha256_hex(shown)
    assert rig.journal.get_blob(line.payload.sha) == shown.encode("utf-8")


def test_a_cancel_at_stop_2_ends_the_episode(root: Path) -> None:
    out = StopOutcome(StopAnswer.CANCEL, PLAN, sha256_hex(PLAN), "not today")
    rig = _rig(root, [], stop=lambda _s: out)
    got = rig.desk.plan(PLAN, call_id="a-plan")
    assert "the user cancelled the plan: not today" in got
    assert rig.desk.ended is not None and rig.desk.ended.note == "not today"
    _refusal(rig.desk.delegate("Return 2", BRIEF, call_id="a-d1"), "plan")


# -- one writer --------------------------------------------------------------------------------


def test_no_delegation_before_a_plan(root: Path) -> None:
    rig = _rig(root, [])
    assert "no plan is approved" in _refusal(rig.desk.delegate("x", BRIEF, call_id="d"), "plan")


def test_one_task_at_a_time_and_a_review_names_the_open_task(root: Path) -> None:
    rig = _rig(root, [calls(EDIT), answer("Changed go to return 2.")])
    _planned(rig)
    got = rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    assert got.startswith("T1 · round 1\nREPORT (worker): Changed go to return 2.")
    assert "FACTS (harness):\n  stopped because: replied" in got
    assert "changed: app.py (edited, +1 -1)" in got
    why = _refusal(rig.desk.delegate("Again", BRIEF, call_id="a-d2"), "task")
    assert why.startswith("T1 is open; one task at a time. Do: review T1 first")
    assert "names the open task, T1, not T2" in _refusal(
        rig.desk.review("T2", "accept", None, call_id="r"), "task"
    )
    [line] = rig.journal.projection().lines(TaskDelegatedP)
    assert (line.payload.task_id, line.payload.call_id, line.payload.by) == (
        "T1",
        "a-d1",
        "athena",
    )
    assert rig.journal.get_blob(line.payload.brief_sha).decode() == BRIEF


# -- each review answer ------------------------------------------------------------------------


def test_accept_commits_the_task_under_its_trailer(root: Path) -> None:
    rig = _rig(root, [calls(EDIT), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    before = git(root, "rev-parse", "HEAD")
    got = rig.desk.review("T1", "accept", None, call_id="a-r1")
    sha = git(root, "rev-parse", "HEAD")
    assert sha != before and got == f"committed T1 as {sha[:7]}"
    assert git(root, "log", "-1", "--format=%s") == "chickadee: T1 Return 2"
    trailer = git(root, "log", "-1", f"--format=%(trailers:key={TRAILER},valueonly)")
    assert trailer.strip() == "task-commit:T1:1"
    assert git(root, "show", "--name-only", "--format=", "HEAD").split() == ["app.py"]
    entry = TaskBook(rig.journal.projection()).get("T1")
    assert entry is not None and (entry.state, entry.commit) == ("committed", sha)
    op = rig.journal.projection().operations["task-commit:T1:1"]
    assert op.kind == EffectKind.TASK_COMMIT.value and op.task_id == "T1"


def test_accept_in_place_commits_nothing(root: Path) -> None:
    rig = _rig(root, [calls(EDIT), answer("done")], in_place=True)
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    before = git(root, "rev-parse", "HEAD")
    assert rig.desk.review("T1", "accept", None, call_id="a-r1") == (
        "T1 accepted; in place nothing is committed (D39)"
    )
    assert git(root, "rev-parse", "HEAD") == before
    op = rig.journal.projection().operations["task-commit:T1:1"]
    assert op.kind == EffectKind.TASK_KNOWN_GOOD.value


def test_feedback_is_the_workers_next_round_on_the_same_conversation(root: Path) -> None:
    rig = _rig(root, [answer("first"), answer("second")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    got = rig.desk.review("T1", "feedback", "Now make it return 2.", call_id="a-r1")
    assert got.startswith("T1 · round 2\nREPORT (worker): second")
    last = rig.worker_rung.requests[-1].messages
    assert [(m.role, m.content) for m in last[-2:]] == [
        ("assistant", "first"),
        ("user", "Now make it return 2."),
    ]
    [reviewed] = rig.journal.projection().lines(TaskReviewedP)
    assert reviewed.payload.call_id == "a-r1"
    assert (
        rig.journal.get_blob(reviewed.payload.text_sha or "").decode()
        == "Now make it return 2."
    )
    why = _refusal(rig.desk.review("T1", "feedback", " ", call_id="a-r2"), "text")
    assert "feedback is the worker's next message" in why


def test_take_over_makes_the_task_hers(root: Path) -> None:
    rig = _rig(root, [answer("stuck")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    assert rig.desk.held is None
    got = rig.desk.review("T1", "take_over", "I will do it", call_id="a-r1")
    assert got == "you hold T1: your edit tools now work on it; then review T1 accept or drop"
    assert rig.desk.held == "T1"
    assert "already hold T1" in _refusal(
        rig.desk.review("T1", "take_over", None, call_id="a-r2"), "answer"
    )


@pytest.mark.parametrize(
    ("answer_", "fresh", "said"),
    [
        ("drop", None, "T1 dropped; the tree is as before it"),
        (
            "re_cut",
            True,
            "T1 replaced; delegate its pieces, which are named T1.1, T1.2, …\n\n"
            "Put back to the task's start: app.py.",
        ),
    ],
)
def test_drop_and_a_fresh_re_cut_put_the_tree_back_under_their_own_step(
    root: Path, answer_: str, fresh: bool | None, said: str
) -> None:
    """Pinned in older versions as `abandon` and `re_cut`, which always put every file back.
    Moved:
    `drop` does; a `re_cut` does with `fresh: true`, and says which files."""
    rig = _rig(root, [calls(EDIT), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    assert "return 2" in (root / "app.py").read_text()
    got = rig.desk.review("T1", answer_, "wrong approach", call_id="a-r1", fresh=fresh)  # type: ignore[arg-type]
    assert got == said
    assert (root / "app.py").read_text() == APP
    run = rig.journal.projection()
    entry = TaskBook(run).get("T1")
    assert entry is not None and entry.state == (
        "abandoned" if answer_ == "drop" else "replaced"
    )
    [restore] = [op for op in run.operations.values() if op.surface == ("review",)]
    assert restore.kind == EffectKind.WORKER_STEP.value and restore.task_id == "T1"
    assert (
        restore.completed is not None and restore.completed.post_state_sha == entry.start_tree
    )
    assert restore.completed.observed_surface == ("app.py",)
    # The tree is the task's start again, so the next task may be delegated.
    assert TaskBook(run).open() is None


TWO = call("create", "w2", path="b.py", content="b = 1\n")


def _two_files(root: Path, *worker: object) -> EpisodeRig:
    rig = _rig(root, [calls(EDIT), calls(TWO), answer("done"), *worker])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    assert "return 2" in (root / "app.py").read_text() and (root / "b.py").exists()
    return rig


def test_a_re_cut_without_fresh_keeps_the_changes(root: Path) -> None:
    rig = _two_files(root)
    got = rig.desk.review("T1", "re_cut", "too big", call_id="a-r1")
    assert got.endswith("what it changed and you kept stays")  # type: ignore[union-attr]
    assert "return 2" in (root / "app.py").read_text() and (root / "b.py").exists()
    run = rig.journal.projection()
    assert not [op for op in run.operations.values() if op.surface == ("review",)]
    assert TaskBook(run).get("T1").state == "replaced"  # type: ignore[union-attr]


def test_feedback_with_fresh_paths_puts_back_only_those_and_names_them(root: Path) -> None:
    rig = _two_files(root, answer("redone"))
    rig.desk.review("T1", "feedback", "keep b, redo app", call_id="a-r1", fresh=("app.py",))
    assert (root / "app.py").read_text() == APP
    assert (root / "b.py").exists(), "the part that was right is kept"
    run = rig.journal.projection()
    [line] = run.lines(TaskReviewedP)
    assert line.payload.fresh == ("app.py",)
    said = [
        str(message_of(rig.journal.get_blob(p.payload.sha)).content)
        for p in run.tasks["T1"]
        if isinstance(p.payload, BoxMessageP) and p.payload.side == "user"
    ]
    assert said[-1] == "keep b, redo app\n\nThe reviewer put back to the task's start: app.py."
    [restore] = [op for op in run.operations.values() if op.surface == ("review",)]
    assert restore.completed is not None and restore.completed.observed_surface == ("app.py",)


@pytest.mark.writes_worktree("b.py")
def test_take_over_with_fresh_true_starts_her_from_the_tasks_start(root: Path) -> None:
    rig = _two_files(root)
    got = rig.desk.review("T1", "take_over", "I will do it", call_id="a-r1", fresh=True)
    assert got == (
        "you hold T1: your edit tools now work on it; then review T1 accept or drop\n\n"
        "Put back to the task's start: app.py, b.py."
    )
    assert (root / "app.py").read_text() == APP and not (root / "b.py").exists()
    assert rig.desk.held == "T1"


def test_accept_with_fresh_is_refused_naming_what_to_send(root: Path) -> None:
    rig = _two_files(root)
    why = _refusal(rig.desk.review("T1", "accept", None, call_id="a-r1", fresh=True), "fresh")
    assert "send feedback, take_over or re_cut with fresh, or drop the task" in why
    assert rig.journal.projection().lines(TaskReviewedP) == ()


def test_fresh_naming_a_file_the_task_did_not_change_is_refused(root: Path) -> None:
    rig = _two_files(root)
    got = rig.desk.review("T1", "feedback", "x", call_id="a-r1", fresh=("tests/test_app.py",))
    assert _refusal(got, "fresh") == (
        "T1 did not change tests/test_app.py; it changed app.py, b.py. Do: send fresh again "
        "naming only the paths the task changed"
    )


@pytest.mark.writes_worktree("b.py")
def test_a_put_back_is_a_step_undo_brings_back(root: Path) -> None:
    rig = _two_files(root)
    rig.desk.review("T1", "re_cut", "too big", call_id="a-r1", fresh=("b.py",))
    run = rig.journal.projection()
    [restore] = [op for op in run.operations.values() if op.surface == ("review",)]
    assert restore.intended is not None and restore.intended.preimages_sha is not None
    assert not (root / "b.py").exists() and "return 2" in (root / "app.py").read_text()


def test_a_re_cuts_new_worker_begins_with_what_the_last_attempt_learned(root: Path) -> None:
    rig = _rig(root, [calls(EDIT), answer("done"), answer("again")])
    _planned(rig)
    rig.desk.delegate("Whole thing", BRIEF, call_id="a-d1")
    rig.desk.review("T1", "re_cut", "too big for one", call_id="a-r1", fresh=True)
    rig.desk.delegate("Half", BRIEF, call_id="a-d2")
    run = rig.journal.projection()
    head = next(
        str(message_of(rig.journal.get_blob(p.payload.sha)).content)
        for p in run.tasks["T1.1"]
        if isinstance(p.payload, BoxMessageP) and p.payload.side == "user"
    )
    assert head.startswith(
        "## What the last attempt learned\n"
        "T1 was re-cut: too big for one\n"
        "- it changed: app.py; all of it was put back\n"
        "- it ran no tests\n"
        "- it stopped because: replied"
    )
    assert head.endswith(BRIEF)


def test_a_review_line_from_an_older_version_still_folds(root: Path) -> None:
    """A line written with `abandon` and no `fresh` validates, and folds as `drop`."""
    line = TaskReviewedP.model_validate(
        {"task_id": "T1", "call_id": "a-r1", "answer": "abandon", "text_sha": None}
    )
    assert line.fresh is None and line.answer == "abandon"
    rig = _rig(root, [calls(EDIT), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    rig.journal.append(EventKind.TASK_REVIEWED, line, task_id="T1")
    assert TaskBook(rig.journal.projection()).get("T1").state == "abandoned"  # type: ignore[union-attr]
    # Resumed, the line's answer is done again as `drop`.
    assert rig.desk.review("T1", "drop", None, call_id="a-r1") == (
        "T1 dropped; the tree is as before it"
    )
    assert (root / "app.py").read_text() == APP


def test_a_re_cut_s_replacements_are_one_deeper_and_the_depth_is_capped(root: Path) -> None:
    rig = _rig(root, [answer("one"), answer("two"), answer("three")], recut_cap=1)
    _planned(rig)
    rig.desk.delegate("Whole thing", BRIEF, call_id="a-d1")
    rig.desk.review("T1", "re_cut", "too big", call_id="a-r1")
    rig.desk.delegate("Half", BRIEF, call_id="a-d2")
    book = TaskBook(rig.journal.projection())
    assert [(e.id, e.depth) for e in book.entries()] == [("T1", 0), ("T1.1", 1)]
    why = _refusal(rig.desk.review("T1.1", "re_cut", None, call_id="a-r2"), "answer")
    assert (
        why == "re-cut limit (1) reached for T1.1. Do: take it over, give feedback, or drop it"
    )
    rig.desk.review("T1.1", "accept", None, call_id="a-r3")
    rig.desk.delegate("Next", BRIEF, call_id="a-d3")
    # Moved by D241: the next piece after an accept is still T1's (`T1.2`, depth 1); once it was
    # a task of its own at depth 0, which a re-cut could take past the cap.
    second = TaskBook(rig.journal.projection()).get("T1.2")
    assert second is not None and second.depth == 1


# -- her edit tools while she holds no task ----------------------------------------------------


def test_her_edit_tools_are_refused_while_she_holds_no_task(root: Path) -> None:
    rig = episode_rig(
        root,
        [
            calls(athena_call("edit", "a1", path="app.py", old="return 1", new="return 2")),
            answer("I changed nothing."),
        ],
        [],
    )
    head = athena_head("You are Athena.", "PLAN TEMPLATE", "BRIEF TEMPLATE", design="# Design")
    got = rig.athena.start(head)
    assert (got.report, got.end) == ("I changed nothing.", "answered")
    assert (root / "app.py").read_text() == APP
    sent = rig.athena_rung.requests[-1].messages
    [result] = [m for m in sent if m.role == "tool"]
    assert NO_TASK_HELD.reason in str(result.content)
    # Her conversation is journaled whole, under no task.
    lines = rig.journal.projection().lines(BoxMessageP)
    assert {p.payload.role for p in lines} == {"athena"} and {p.task_id for p in lines} == {
        None
    }
    assert [p.payload.side for p in lines] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


# -- the review of #582: each pinned at the failure it found -----------------------------------


@pytest.mark.parametrize("landed", [True, False], ids=["landed", "not_landed"])
def test_a_commit_whose_git_raised_is_accepted_again_and_never_made_twice(
    root: Path, monkeypatch: pytest.MonkeyPatch, landed: bool
) -> None:
    # Its git raised: the attempt completes `ambiguous` and the task stays `accepted`. Another
    # accept is a new attempt under its own operation id that asks the first one's trailer.
    rig = _rig(root, [calls(EDIT), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    real = Projector.commit

    def raises(self: Projector, *args: object, **kwargs: object) -> str:
        if landed:
            real(self, *args, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("git failed part way")

    monkeypatch.setattr(Projector, "commit", raises)
    with pytest.raises(RuntimeError):
        rig.desk.review("T1", "accept", None, call_id="a-r1")
    monkeypatch.undo()
    first = rig.journal.projection().operations["task-commit:T1:1"]
    assert first.completed is not None and first.completed.outcome == "ambiguous"
    entry = TaskBook(rig.journal.projection()).get("T1")
    assert entry is not None and entry.state == "accepted"

    got = rig.desk.review("T1", "accept", None, call_id="a-r2")
    log = git(root, "log", f"--format=%H %(trailers:key={TRAILER},valueonly)").splitlines()
    [sha] = [line.split()[0] for line in log if " task-commit:T1:" in line]
    assert got == f"committed T1 as {sha[:7]}"
    second = rig.journal.projection().operations["task-commit:T1:2"]
    assert second.completed is not None
    assert (second.completed.outcome, second.completed.post_state_sha) == ("succeeded", sha)
    entry = TaskBook(rig.journal.projection()).get("T1")
    assert entry is not None and (entry.state, entry.commit) == ("committed", sha)


def test_a_restore_that_did_not_succeed_is_done_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _rig(root, [calls(EDIT), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    monkeypatch.setattr(desk_module, "restore_tree", _raises)
    with pytest.raises(RuntimeError):
        rig.desk.review("T1", "drop", None, call_id="a-r1")
    monkeypatch.undo()
    assert "return 2" in (root / "app.py").read_text()
    # The same call dispatched again (a resume) finds its line and restores under a new step.
    assert rig.desk.review("T1", "drop", None, call_id="a-r1") == (
        "T1 dropped; the tree is as before it"
    )
    assert (root / "app.py").read_text() == APP
    outcomes = [
        op.completed.outcome
        for op in rig.journal.projection().operations.values()
        if op.surface == ("review",) and op.completed is not None
    ]
    assert outcomes == ["ambiguous", "succeeded"]


@pytest.mark.writes_worktree("notes.txt")
def test_the_next_round_starts_from_the_tree_as_it_is_now(root: Path) -> None:
    # Between two rounds the tree moved outside the worker (her shell, here written by hand):
    # the next round's first step must not count that change as its own.
    rig = _rig(root, [answer("looked"), calls(EDIT), answer("returned 2")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    (root / "notes.txt").write_text("hers\n", encoding="utf-8")
    rig.desk.review("T1", "feedback", "now return 2", call_id="a-r1")
    [step] = [
        op for op in rig.journal.projection().operations.values() if op.surface == ("edit",)
    ]
    assert step.completed is not None and step.completed.observed_surface == ("app.py",)


def test_a_cancel_at_stop_2_is_journaled_and_holds_after_a_restart(root: Path) -> None:
    out = StopOutcome(StopAnswer.CANCEL, PLAN, sha256_hex(PLAN), "not today")
    rig = _rig(root, [], stop=lambda _s: out)
    rig.desk.plan(PLAN, call_id="a-plan")
    [line] = rig.journal.projection().lines(PlanWrittenP)
    assert (line.payload.outcome, line.payload.asked) == ("cancelled", True)
    rig.journal.close()
    again = episode_rig(root, [], [], resume=True)
    assert again.desk.ended is not None and again.desk.ended.note == "not today"
    assert again.desk.plans() == (), "a cancelled plan is not a plan"
    _refusal(again.desk.plan(PLAN, call_id="a-plan-2"), "plan")
    assert again.stops == [], "never asked again after the person cancelled"


def test_a_first_round_started_again_takes_the_symbol_table_then(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = object()
    seen: list[object] = []
    rig = episode_rig(root, [], [answer("done")], symbols=lambda: table)
    _planned(rig)
    real = WorkerRun.start
    calls_ = iter([True, False])

    def start(self: WorkerRun, *args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("before"))
        if next(calls_):
            raise RuntimeError("died before the round's first message")
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(WorkerRun, "start", start)
    with pytest.raises(RuntimeError):
        rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    assert seen == [table, table]


def test_her_policys_own_wrappers_stay_around_the_desks(root: Path) -> None:
    seen: list[str] = []

    class Spy:
        def around(self, pending: Any, run: Any, ctx: Any) -> Any:
            seen.append(pending.sent)
            return run()

    policy = replace(ATHENA_TASK, pipeline=replace(ATHENA_TASK.pipeline, around=(Spy(),)))
    rig = episode_rig(
        root,
        [calls(athena_call("read", "a1", path="app.py")), answer("read it")],
        [],
        policy=policy,
    )
    head = athena_head("You are Athena.", "P", "B", design="# Design")
    assert rig.athena.start(head).report == "read it"
    assert seen == ["read"]


def _raises(*_: object, **__: object) -> object:
    raise RuntimeError("restore failed part way")


def test_a_brief_without_the_template_sections_gives_no_goal_line(root: Path) -> None:
    """D204 decision 2 (was: the subject and the trailer alone, pinned before T-3.166 part 4):
    the body has what the journal holds, and a brief without `Goal` or `Done when` sections
    (this one's are `###`) gives neither line; nothing is invented."""
    rig = _rig(root, [calls(EDIT), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    rig.desk.review("T1", "accept", None, call_id="a-r1")
    assert git(root, "log", "-1", "--format=%B") == (
        "chickadee: T1 Return 2\n\nReviewed: accepted by Athena, round 1\nRan: nothing\n\n"
        f"{TRAILER}: task-commit:T1:1"
    )


def test_a_tasks_commit_body_copies_its_brief_and_its_facts(root: Path) -> None:
    """`Goal` and `Done when` from the brief's sections, `Ran` the commands
    the facts recorded with their exit codes, and the trailer still last, as recovery reads it.
    """
    brief = (
        "# T1: Return 2\n\n## Goal\n`go()` returns 2.\n\n## Approach\nEdit it.\n\n"
        "## Done when\n`pytest -q` passes.\n"
    )
    run = call("shell", "w2", command="pytest -q")
    rig = _rig(root, [calls(EDIT), calls(run), answer("done")])
    _planned(rig)
    rig.desk.delegate("Return 2", brief, call_id="a-d1")
    rig.desk.review("T1", "accept", None, call_id="a-r1")
    body = git(root, "log", "-1", "--format=%B")
    assert body.startswith(
        "chickadee: T1 Return 2\n\nGoal: `go()` returns 2.\nDone when: `pytest -q` passes.\n"
        "Reviewed: accepted by Athena, round 1\nRan: `pytest -q` (exit 0"
    )
    trailer = git(root, "log", "-1", f"--format=%(trailers:key={TRAILER},valueonly)")
    assert trailer.strip() == "task-commit:T1:1"


def test_a_task_delegated_again_after_drop_hears_what_the_last_attempt_learned(
    root: Path,
) -> None:
    """Pinned in older versions: the new worker's head was its brief and "done so far",
    with no word of the attempt it replaced. Moved with D214 decision 5: it begins with what
    that attempt learned, from its last report's facts."""
    rig = _rig(root, [calls(EDIT), answer("done"), answer("again")])
    _planned(rig)
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d1")
    rig.desk.review("T1", "drop", "wrong approach", call_id="a-r1")
    rig.desk.delegate("Return 2", BRIEF, call_id="a-d2")
    run = rig.journal.projection()
    head = [
        message_of(rig.journal.get_blob(p.payload.sha))
        for p in run.tasks["T1@2"]
        if isinstance(p.payload, BoxMessageP) and p.payload.side == "user"
    ]
    said = "\n".join(str(m.content) for m in head)
    assert BRIEF in said
    assert "T1 was dropped: wrong approach" in said
    assert "- it changed: app.py; all of it was put back" in said


def test_the_round_after_a_fresh_put_back_starts_from_the_tree_it_left(root: Path) -> None:
    """The worker's next step measures from the tree the put-back left, so the files put back
    are never counted as its own change."""
    rig = _two_files(
        root, calls(call("create", "w3", path="c.py", content="c = 1\n")), answer("ok")
    )
    rig.desk.review("T1", "feedback", "redo app", call_id="a-r1", fresh=("app.py",))
    run = rig.journal.projection()
    [step] = [
        op
        for op in run.operations.values()
        if op.kind == EffectKind.WORKER_STEP.value
        and op.surface == ("create",)
        and op.completed is not None
        and op.completed.observed_surface == ("c.py",)
    ]
    assert step.completed is not None
