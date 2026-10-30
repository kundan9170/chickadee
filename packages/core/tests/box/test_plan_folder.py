"""Athena's plan in its work folder (Amendment 1 A6): `plan()` checks
`plan/` and refuses naming what is wrong; the first plan that passes is asked at stop 2 with its
folder and pinned; `delegate(task=…)` starts a task of the pinned plan from its pinned file,
and refuses one whose file changed after the approval; a later `plan()` is a revision, pinned
again."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from box_helpers import APP, episode_rig
from repos import git, init_repo

from chickadee.approval.stops import StopOutcome
from chickadee.context.results import InvalidArgument
from chickadee.context.work_line import WorkLine
from chickadee.handoffs import SHIPPED
from chickadee.schema.interaction import StopAnswer
from chickadee.workspace import pin_of

PLAN_README = """# Plan: {title}

## Tasks
1. [T1: {title}](tasks/T1-task.md): the edit, with its test. Depends on: none

## Final check
The full suite.
"""

TASK_FILE = """# T1: {title}

## Goal
{goal}

## Where to look
- `app.py`: `go()`.

## Approach
Edit the return.

## Don't
- Edit a test to pass.

## Done when
`go()` returns what the goal says.

## Report
What changed.
"""

WORK = WorkLine(name="2026-10-18-return-2", dir="docs/chickadee", state="open")
PLAN = "docs/chickadee/2026-10-18-return-2/plan"
WRITES = ("docs", "docs/chickadee", "docs/chickadee/2026-10-18-return-2", PLAN, f"{PLAN}/tasks")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _write(root: Path, readme: str | None = None, task: str | None = None) -> None:
    folder = root / PLAN
    (folder / "tasks").mkdir(parents=True, exist_ok=True)
    (folder / "README.md").write_text(readme or PLAN_README.format(title="Return 2"))
    (folder / "tasks" / "T1-task.md").write_text(
        task or TASK_FILE.format(title="Return 2", goal="`go()` returns 2")
    )


def _rig(root: Path, worker: list[object] | None = None, **kwargs: Any) -> Any:
    rig = episode_rig(root, [], worker or [], work=WORK, **kwargs)
    rig.desk._parts = replace(  # pyright: ignore[reportPrivateUsage]
        rig.desk._parts,  # pyright: ignore[reportPrivateUsage]
        work=WORK,
        plan_template=(SHIPPED / "plan.md").read_text(),
        brief_template=(SHIPPED / "brief.md").read_text(),
    )
    return rig


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_a_plan_out_of_shape_is_refused_naming_what_is_wrong(root: Path) -> None:
    _write(
        root, readme=PLAN_README.format(title="Return 2").replace("## Final check", "## End")
    )
    rig = _rig(root)
    got = rig.desk.plan(None, call_id="a-plan")
    assert isinstance(got, InvalidArgument) and got.field == "plan"
    assert "the Final check section is missing" in got.why
    assert rig.stops == [], "never asked"


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_the_first_plan_is_asked_with_its_folder_and_pinned(root: Path) -> None:
    _write(root)
    rig = _rig(root)
    assert rig.desk.plan(None, call_id="a-plan").startswith("the person approved the plan")
    [stop] = rig.stops
    assert stop.folder == str(root / PLAN)
    assert stop.text == (root / PLAN / "README.md").read_text()
    assert pin_of(root, rig.journal.run_id, "plan-v1") is not None


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_a_task_is_delegated_by_id_from_its_pinned_file(root: Path) -> None:
    _write(root)
    rig = _rig(root)
    rig.desk.plan(None, call_id="a-plan")
    rig.desk.delegate(task="T1", call_id="a-d1")
    [line] = [p.payload for p in rig.journal.projection().lines(_delegated())]
    brief = rig.journal.get_blob(line.brief_sha).decode()
    assert brief.startswith("# T1: Return 2") and "## The work's documents" in brief
    assert "`docs/chickadee/2026-10-18-return-2/plan/tasks/T1-task.md`" in brief
    assert line.title == "Return 2"


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_a_task_changed_after_the_approval_cannot_be_delegated(root: Path) -> None:
    _write(root)
    rig = _rig(root)
    rig.desk.plan(None, call_id="a-plan")
    changed = TASK_FILE.format(title="Return 2", goal="`go()` returns 3")
    (root / PLAN / "tasks" / "T1-task.md").write_text(changed)
    got = rig.desk.delegate(task="T1", call_id="a-d1")
    assert isinstance(got, InvalidArgument) and "changed after the plan was approved" in got.why


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_a_task_not_in_the_plan_or_a_title_with_a_folder_is_refused(root: Path) -> None:
    _write(root)
    rig = _rig(root)
    rig.desk.plan(None, call_id="a-plan")
    got = rig.desk.delegate(task="T9", call_id="a-d1")
    assert isinstance(got, InvalidArgument) and got.why == "T9 is not in the approved plan: T1"
    got = rig.desk.delegate("Do it", "brief", call_id="a-d2")
    assert isinstance(got, InvalidArgument) and got.field == "task"
    got = rig.desk.plan("# Plan\n", call_id="a-p2")
    assert isinstance(got, InvalidArgument) and got.field == "text"


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_an_edit_at_the_stop_is_the_version_that_runs(root: Path) -> None:
    """The person edits the task's file at stop 2: what is pinned, and what
    the task is started from, is their version."""
    _write(root)
    theirs = TASK_FILE.format(title="Return 2", goal="`go()` returns 2, and is documented")

    def edits(stop: Any) -> Any:
        (root / PLAN / "tasks" / "T1-task.md").write_text(theirs)
        return StopOutcome(StopAnswer.APPROVE, stop.text, stop.sha)

    rig = _rig(root, stop=edits)
    rig.desk.plan(None, call_id="a-plan")
    rig.desk.delegate(task="T1", call_id="a-d1")
    [line] = [p.payload for p in rig.journal.projection().lines(_delegated())]
    assert "and is documented" in rig.journal.get_blob(line.brief_sha).decode()


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_a_revision_is_pinned_again_and_shown_not_asked(root: Path) -> None:
    _write(root)
    rig = _rig(root)
    rig.desk.plan(None, call_id="a-plan")
    (root / PLAN / "README.md").write_text(
        PLAN_README.format(title="Return 2").replace(
            "The full suite.", "The full suite, twice."
        )
    )
    got = rig.desk.plan(None, call_id="a-plan-2")
    assert got.startswith("plan revised (version 2)")
    assert len(rig.stops) == 1
    v1 = pin_of(root, rig.journal.run_id, "plan-v1")
    v2 = pin_of(root, rig.journal.run_id, "plan-v2")
    assert v1 and v2 and v1 != v2
    assert git(root, "cat-file", "-p", f"{v1}:README.md").endswith("The full suite.")


def _delegated() -> Any:
    from chickadee.schema.journal import TaskDelegatedP

    return TaskDelegatedP


def test_athenas_design_is_its_readme_then_each_file_by_its_path() -> None:
    """D204 decision 15a, A7: she is never told to find the design by name."""
    from chickadee.boxctx.athena import DESIGN_FILES, design_text

    got = design_text("# D\n", "docs/chickadee/w", (("api.md", "the endpoints"),))
    assert got == f"# D\n\n{DESIGN_FILES}\n\n- `docs/chickadee/w/design/api.md`: the endpoints"
    assert design_text("# D\n", "docs/chickadee/w", ()) == "# D"


@pytest.mark.writes_worktree(
    *WRITES,
    f"{PLAN}/README.md",
    f"{PLAN}/tasks/T1-task.md",
    "docs/chickadee/2026-10-18-return-2/README.md",
    "docs/chickadee/2026-10-18-return-2/report.md",
)
def test_with_work_commit_off_the_folder_is_never_committed_and_kept_privately(
    root: Path, tmp_path: Path
) -> None:
    """B7: the run is the same, its folder never committed: the approvals and the records are
    files, and a private copy holds them."""
    from chickadee.box.report import CoderReport, ReportHandles, stored_handle
    from chickadee.journal.projection import RunSpend

    _write(root)
    rig = _rig(root)
    private = tmp_path / "private"
    rig.desk._parts = replace(  # pyright: ignore[reportPrivateUsage]
        rig.desk._parts,  # pyright: ignore[reportPrivateUsage]
        design_tree="t",
        work_commit=False,
        private=private,
    )
    head = git(root, "rev-parse", "HEAD")
    rig.desk.record_design()
    rig.desk.plan(None, call_id="a-plan")
    report = CoderReport(
        outcome="COMPLETED",
        run_id=rig.journal.run_id,
        handles=ReportHandles(journal=stored_handle("journal", rig.journal.run_id, b"{}")),
        cost=RunSpend(),
    )
    assert rig.desk.record_report(report)
    assert git(root, "rev-parse", "HEAD") == head, "nothing of the folder is committed"
    copy = private / WORK.name
    assert (copy / "report.md").read_text().startswith("<!-- written by chickadee")
    assert (copy / "plan" / "tasks" / "T1-task.md").is_file()


# -- the second review of #607 ---------------------------------------------------------------


@pytest.mark.writes_worktree(*WRITES, f"{PLAN}/README.md", f"{PLAN}/tasks/T1-task.md")
def test_a_plan_task_runs_once_and_keeps_its_plan_id_across_attempts(root: Path) -> None:
    """A delegation records the plan task it runs. Dropped, it may run again,
    as its own next attempt under the plan's id (once the harness's next id, `T2`, titled
    "(plan T1)"); re-cut, never again."""
    _write(root)
    rig = _rig(root)
    rig.desk.plan(None, call_id="a-plan")
    rig.desk.delegate(task="T1", call_id="a-d1")
    rig.desk.review("T1", "drop", "start over", call_id="a-r1")
    rig.desk.delegate(task="T1", call_id="a-d2")
    lines = [p.payload for p in rig.journal.projection().lines(_delegated())]
    assert [(p.task_id, p.plan_task, p.title) for p in lines] == [
        ("T1", "T1", "Return 2"),
        ("T1@2", "T1", "Return 2"),
    ]
    rig.desk.review("T1", "re_cut", "two steps", call_id="a-r2")
    got = rig.desk.delegate(task="T1", call_id="a-d3")
    assert isinstance(got, InvalidArgument)
    assert "the plan's T1 was delegated, and is replaced: a plan task runs once" in got.why
