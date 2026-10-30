"""The desk names a task by its plan: `delegate("T1")` and `review("T1")` say the same
id, a retry is the task's next attempt with a conversation of its own, and a re-cut's pieces are
its children in the plan, one deeper. A journal written in older versions still folds."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from box_helpers import APP
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo
from test_plan_folder import PLAN, PLAN_README, TASK_FILE, WRITES, _rig, _write

from chickadee.box.carry import Carried, PriorTask
from chickadee.box.request import Continuation
from chickadee.box.tasks import TaskBook, TaskEntry
from chickadee.boxctx.conversation import message_of
from chickadee.context.results import InvalidArgument
from chickadee.journal.writer import Journal
from chickadee.schema.journal import BoxMessageP, EventKind, TaskDelegatedP
from chickadee.state.roots import roots as resolve_roots

EDIT = call("edit", "w1", path="app.py", old="return 1", new="return 2")
FILES = (f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _users(rig: object, key: str) -> list[str]:
    run = rig.journal.projection()  # type: ignore[attr-defined]
    return [
        str(message_of(rig.journal.get_blob(p.payload.sha)).content)  # type: ignore[attr-defined]
        for p in run.tasks[key]
        if isinstance(p.payload, BoxMessageP) and p.payload.side == "user"
    ]


@pytest.mark.writes_worktree(*WRITES, *FILES, "app.py")
def test_delegate_and_review_say_the_plans_id(root: Path) -> None:
    _write(root)
    rig = _rig(root, [calls(EDIT), answer("done")])
    rig.desk.plan(None, call_id="a-plan")
    rig.desk.delegate(task="T1", call_id="a-d1")
    assert rig.desk.review("T1", "accept", None, call_id="a-r1").startswith("committed T1 as ")
    [entry] = rig.desk.book().entries()
    assert (entry.id, entry.name, entry.plan_task, entry.state) == (
        "T1",
        "T1",
        "T1",
        "committed",
    )


@pytest.mark.writes_worktree(*WRITES, *FILES, "app.py")
def test_a_retry_is_the_next_attempt_with_a_conversation_of_its_own(root: Path) -> None:
    _write(root)
    rig = _rig(root, [calls(EDIT), answer("done"), answer("again")])
    rig.desk.plan(None, call_id="a-plan")
    rig.desk.delegate(task="T1", call_id="a-d1")
    rig.desk.review("T1", "drop", "wrong approach", call_id="a-r1")
    rig.desk.delegate(task="T1", call_id="a-d2")
    book = rig.desk.book()
    assert [(e.id, e.name, e.attempt) for e in book.entries()] == [
        ("T1", "T1", 1),
        ("T1@2", "T1", 2),
    ]
    first, second = _users(rig, "T1"), _users(rig, "T1@2")
    assert first and second and first != second, "each attempt is its own conversation"
    assert "T1 was dropped: wrong approach" in second[0]
    # The model names the task, never the attempt: `review("T1")` is the open attempt's.
    assert "dropped" in rig.desk.review("T1", "drop", None, call_id="a-r2")


@pytest.mark.writes_worktree(*WRITES, *FILES, f"{PLAN}/tasks/T1.1-first-half.md", "app.py")
def test_a_re_cuts_pieces_are_its_children_in_the_plan_and_one_deeper(root: Path) -> None:
    _write(root)
    rig = _rig(root, [calls(EDIT), answer("too big"), answer("half")], recut_cap=1)
    rig.desk.plan(None, call_id="a-plan")
    rig.desk.delegate(task="T1", call_id="a-d1")
    said = rig.desk.review("T1", "re_cut", "two halves", call_id="a-r1")
    assert "add its pieces to plan/ as T1.1, T1.2" in said
    # A piece with no parent before it is refused by the plan's shape.
    readme = PLAN_README.format(title="Return 2") + ""
    lines = readme.replace(
        "Depends on: none\n",
        "Depends on: none\n2. [T1.1: First half](tasks/T1.1-first-half.md): half. "
        "Depends on: none\n",
    )
    (root / PLAN / "README.md").write_text(lines)
    (root / PLAN / "tasks" / "T1.1-first-half.md").write_text(
        TASK_FILE.format(title="First half", goal="half of it").replace("# T1:", "# T1.1:")
    )
    assert rig.desk.plan(None, call_id="a-plan-2").startswith("plan revised (version 2)")
    rig.desk.delegate(task="T1.1", call_id="a-d2")
    piece = rig.desk.book().get("T1.1")
    assert piece is not None and piece.depth == 1
    refused = rig.desk.review("T1.1", "re_cut", None, call_id="a-r2")
    assert (
        isinstance(refused, InvalidArgument)
        and "re-cut limit (1) reached for T1.1" in refused.why
    )


@pytest.mark.writes_worktree(*WRITES, *FILES, f"{PLAN}/tasks/T1-other.md")
def test_an_approved_id_keeps_its_task_in_a_revision(root: Path) -> None:
    _write(root)
    rig = _rig(root)
    rig.desk.plan(None, call_id="a-plan")
    readme = (root / PLAN / "README.md").read_text().replace("T1-task.md", "T1-other.md")
    (root / PLAN / "README.md").write_text(readme)
    (root / PLAN / "tasks" / "T1-task.md").rename(root / PLAN / "tasks" / "T1-other.md")
    got = rig.desk.plan(None, call_id="a-plan-2")
    assert isinstance(got, InvalidArgument)
    assert "T1 was tasks/T1-task.md in an approved plan, and an id keeps its task" in got.why


@pytest.mark.writes_worktree(*WRITES, *FILES)
def test_a_task_an_earlier_run_re_cut_is_refused_naming_its_pieces(root: Path) -> None:
    """Within a run `book.of_plan` refuses a re-cut task; across runs `_done_before` does, or
    the re-cut task would run again beside its pieces."""
    _write(root)
    rig = _rig(root)
    earlier = "2026-10-27-0000000a"
    cut = TaskEntry("T1", "Return 2", "athena", 0, "replaced", 1, "a" * 40, plan_task="T1")
    rig.desk._parts = replace(  # pyright: ignore[reportPrivateUsage]
        rig.desk._parts,  # pyright: ignore[reportPrivateUsage]
        carried=Carried(
            Continuation(mode="CONTINUE", prior_runs=(earlier,)),
            _tasks={"T1": PriorTask(earlier, cut, 1)},
        ),
    )
    rig.desk.plan(None, call_id="a-plan")
    got = rig.desk.delegate(task="T1", call_id="a-d1")
    assert isinstance(got, InvalidArgument)
    assert f"T1 was re-cut in an earlier run of this work ({earlier})" in got.why
    assert rig.desk.book().entries() == ()


def test_a_journal_from_an_older_version_still_folds(tmp_path: Path) -> None:
    """Before D241 the harness counted delegations: a plan's T3 could run as T4."""
    (tmp_path / ".git" / "info").mkdir(parents=True)
    journal = Journal.create(resolve_roots(tmp_path), "2026-10-15-0a1b2c3d", {})
    journal.append(
        EventKind.TASK_DELEGATED,
        TaskDelegatedP(
            task_id="T4",
            call_id="d-1",
            title="Return 2 (plan T3)",
            brief_sha=journal.put_blob(b"brief"),
            by="athena",
            start_tree="a" * 40,
            plan_task="T3",
        ),
        task_id="T4",
    )
    book = TaskBook(journal.projection())
    entry = book.of_plan("T3")
    # Its key stays the journal's `T4`; its name is the plan's `T3`, as people read it.
    assert entry is not None and (entry.id, entry.name, entry.attempt) == ("T4", "T3", 1)
