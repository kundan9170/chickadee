"""The task book: `box/tasks.TaskBook`, the fold of a run's task lines.

What these pin: each task's state from its lines, in the order delegated; the rounds counted
from its reports; the ids the harness checks or names, never a model's own; one open
task at most, two a refusal; a
report for a task never delegated is not a task; each review answer's state, an
accept open until its commit effect succeeds, and the review calls kept for resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.box.tasks import TaskBook, lineage, next_folderless
from chickadee.ids import task_commit_id
from chickadee.journal.effects import EffectKind, Intent, on_journal, perform
from chickadee.journal.writer import Journal
from chickadee.schema.journal import (
    EventKind,
    ReviewAnswer,
    TaskDelegatedP,
    TaskFactsP,
    TaskReportedP,
    TaskReviewedP,
)
from chickadee.state.roots import roots as resolve_roots

RUN = "2026-10-15-0a1b2c3d"
TREE = "a" * 40


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    (tmp_path / ".git" / "info").mkdir(parents=True)
    return Journal.create(resolve_roots(tmp_path), RUN, {})


def _delegated(
    journal: Journal, task: str, *, depth: int = 0, plan_task: str | None = None
) -> None:
    journal.append(
        EventKind.TASK_DELEGATED,
        TaskDelegatedP(
            task_id=task,
            call_id=f"d-{task}",
            title=f"title of {task}",
            brief_sha=journal.put_blob(b"brief"),
            by="athena",
            start_tree=TREE,
            depth=depth,
            plan_task=plan_task,
        ),
        task_id=task,
    )


def _reported(journal: Journal, task: str, round_: int) -> None:
    journal.append(
        EventKind.TASK_REPORTED,
        TaskReportedP(
            task_id=task,
            round=round_,
            stopped="replied",
            facts=TaskFactsP(start_tree=TREE, end_tree=TREE),
        ),
        task_id=task,
    )


def test_an_empty_run_has_no_task_and_its_first_id_is_t1(journal: Journal) -> None:
    run = journal.projection()
    book = TaskBook(run)
    assert (book.entries(), book.open(), next_folderless(run, book)) == ((), None, "T1")
    assert book.next_key("T1") == "T1"


def test_a_delegated_task_is_open_with_no_round(journal: Journal) -> None:
    _delegated(journal, "T1", depth=1)
    book = TaskBook(journal.projection())
    entry = book.open()
    assert entry is not None
    assert (entry.id, entry.state, entry.rounds, entry.depth) == ("T1", "delegated", 0, 1)
    assert (entry.title, entry.by, entry.start_tree, entry.call_id) == (
        "title of T1",
        "athena",
        TREE,
        "d-T1",
    )
    assert next_folderless(journal.projection(), book) == "T2"
    assert (entry.name, entry.attempt, book.next_key("T1")) == ("T1", 1, "T1@2")


def test_each_report_counts_a_round(journal: Journal) -> None:
    _delegated(journal, "T1")
    _reported(journal, "T1", 1)
    _reported(journal, "T1", 2)
    entry = TaskBook(journal.projection()).get("T1")
    assert entry is not None and (entry.state, entry.rounds) == ("reported", 2)


def test_two_open_tasks_are_refused_one_writer(journal: Journal) -> None:
    _delegated(journal, "T1")
    _delegated(journal, "T2")
    book = TaskBook(journal.projection())
    assert [e.id for e in book.entries()] == ["T1", "T2"]
    with pytest.raises(ValueError, match="one writer"):
        book.open()


def test_a_report_for_a_task_never_delegated_is_not_a_task(journal: Journal) -> None:
    _reported(journal, "T1", 1)
    book = TaskBook(journal.projection())
    assert (book.entries(), book.get("T1"), next_folderless(journal.projection(), book)) == (
        (),
        None,
        "T1",
    )


def _reviewed(journal: Journal, task: str, answer: ReviewAnswer, call: str) -> None:
    journal.append(
        EventKind.TASK_REVIEWED,
        TaskReviewedP(task_id=task, call_id=call, answer=answer),
        task_id=task,
    )


def _committed(journal: Journal, task: str, *, ok: bool = True) -> None:
    op = task_commit_id(task)
    intent = Intent(op, EffectKind.TASK_COMMIT, "s" * 64, (), False, ("app.py",))
    with perform(on_journal(journal, task_id=task), intent) as fx:
        fx.complete("succeeded" if ok else "failed", post_state="c" * 40 if ok else None)


@pytest.mark.parametrize(
    ("answer", "state", "open_"),
    [
        ("accept", "accepted", True),
        ("feedback", "delegated", True),
        ("take_over", "taken_over", True),
        ("abandon", "abandoned", False),
        ("drop", "abandoned", False),
        ("re_cut", "replaced", False),
    ],
)
def test_each_review_answer_is_a_state(
    journal: Journal, answer: ReviewAnswer, state: str, open_: bool
) -> None:
    _delegated(journal, "T1")
    _reported(journal, "T1", 1)
    _reviewed(journal, "T1", answer, "r-1")
    book = TaskBook(journal.projection())
    entry = book.get("T1")
    assert entry is not None
    assert (entry.state, entry.answer, entry.reviews, entry.rounds) == (
        state,
        answer,
        ("r-1",),
        1,
    )
    assert (book.open() is not None) is open_


def test_an_accept_is_committed_once_its_commit_succeeds(journal: Journal) -> None:
    _delegated(journal, "T1")
    _reported(journal, "T1", 1)
    _reviewed(journal, "T1", "accept", "r-1")
    _committed(journal, "T1", ok=False)
    entry = TaskBook(journal.projection()).get("T1")
    assert entry is not None and (entry.state, entry.commit) == ("accepted", None)
    _delegated(journal, "T2")  # a second task's commit is not the first's
    _committed(journal, "T2")
    book = TaskBook(journal.projection())
    first, second = book.get("T1"), book.get("T2")
    assert first is not None and first.state == "accepted"
    assert second is not None and (second.state, second.commit) == ("committed", "c" * 40)


def test_feedback_then_a_report_counts_the_next_round(journal: Journal) -> None:
    _delegated(journal, "T1")
    _reported(journal, "T1", 1)
    _reviewed(journal, "T1", "feedback", "r-1")
    _reported(journal, "T1", 2)
    _reviewed(journal, "T1", "accept", "r-2")
    _committed(journal, "T1")
    book = TaskBook(journal.projection())
    entry = book.get("T1")
    assert entry is not None
    assert (entry.state, entry.rounds, entry.reviews) == ("committed", 2, ("r-1", "r-2"))
    assert (book.open(), next_folderless(journal.projection(), book)) == (None, "T2")


# -- lineage: which failed attempt a task replaces -----------------------


def _replaces(journal: Journal) -> dict[str, tuple[str, str]]:
    return {t: (r.task_id, r.answer) for t, r in lineage(journal.projection()).replaces.items()}


def test_a_re_cuts_pieces_are_its_children_whatever_came_between(journal: Journal) -> None:
    """Moved by D241: the lineage is read from the ids. Once, a re-cut's pieces were whatever
    was delegated before the next accept, so a second piece after an accepted first was a task
    of its own, at depth 0, which a further re-cut could take past the cap."""
    _delegated(journal, "T1")
    _reviewed(journal, "T1", "re_cut", "r-1")
    _delegated(journal, "T1.1", depth=1)
    _reviewed(journal, "T1.1", "accept", "r-2")
    _delegated(journal, "T1.2", depth=1)
    _delegated(journal, "T2")
    assert _replaces(journal) == {"T1.1": ("T1", "re_cut"), "T1.2": ("T1", "re_cut")}
    cut = lineage(journal.projection()).cut
    assert cut is not None and cut.task_id == "T1"


def test_a_drop_is_replaced_by_the_next_delegation_of_its_plan_task(journal: Journal) -> None:
    """The review of T-3.189: a drop's note went to whatever came next, an unrelated plan task
    too. It goes to the same plan task delegated again: its next attempt."""
    _delegated(journal, "T1", plan_task="T1")
    _reviewed(journal, "T1", "drop", "r-1")
    _delegated(journal, "T2", plan_task="T2")
    _reviewed(journal, "T2", "accept", "r-2")
    _delegated(journal, "T1@2", plan_task="T1")
    assert _replaces(journal) == {"T1@2": ("T1", "drop")}


def test_with_no_plan_tasks_a_drop_is_replaced_by_the_next_delegation(journal: Journal) -> None:
    """With no plan the harness names the next delegation after a drop as the dropped task's
    next attempt, so the lineage is its key."""
    _delegated(journal, "T1")
    _reviewed(journal, "T1", "abandon", "r-1")
    run = journal.projection()
    assert next_folderless(run, TaskBook(run)) == "T1@2"
    _delegated(journal, "T1@2")
    _reviewed(journal, "T1@2", "accept", "r-2")
    run = journal.projection()
    assert next_folderless(run, TaskBook(run)) == "T2"
    _delegated(journal, "T2")
    assert _replaces(journal) == {"T1@2": ("T1", "abandon")}, "once, and an old line reads too"


def test_with_no_plan_a_re_cuts_pieces_are_named_by_their_parent(journal: Journal) -> None:
    _delegated(journal, "T1")
    _reviewed(journal, "T1", "re_cut", "r-1")
    run = journal.projection()
    assert next_folderless(run, TaskBook(run)) == "T1.1"
    _delegated(journal, "T1.1", depth=1)
    _reviewed(journal, "T1.1", "accept", "r-2")
    run = journal.projection()
    assert next_folderless(run, TaskBook(run)) == "T1.2", "pieces go on until another re-cut"
