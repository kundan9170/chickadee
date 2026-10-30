"""When a one-task plan outgrows its task.

The pins of the answers part 5 moved, each now its new answer: an answered episode with no plan
is `PARTIAL` (was `COMPLETED`), and a one-task plan's fourth task is refused at the default cap
(its re-cut into four ran all four).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import athena_call, episode_rig
from box_helpers import worker_call as wcall
from engines import answer, calls
from repos import git, init_repo

from chickadee.box.loop import OUTGROWN_WHY, HeadParts, LoopResult, OneTaskPlan, TaskLoop
from chickadee.box.report import CoderReport
from chickadee.box.tasks import TaskBook, growth
from chickadee.schema import RetryParams
from chickadee.tools.tool import Said
from chickadee.zeus.delegate import render_report

STATS = "def total(xs):\n    return sum(xs)\n"
HEAD = HeadParts("You are Athena.", "PLAN TEMPLATE", "BRIEF TEMPLATE")
DESIGN = "# Totals\n\n## Request\n> make total handle None\n"
BRIEF = "## Task 1 of 1: totals\n\n### Goal\n`total` handles None.\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"stats.py": STATS})


def _loop(rig: object, **work: object) -> TaskLoop:
    return TaskLoop(
        rig.athena,  # type: ignore[attr-defined]
        rig.desk,  # type: ignore[attr-defined]
        rig.journal,  # type: ignore[attr-defined]
        HEAD,
        root=rig.root,  # type: ignore[attr-defined]
        params=RetryParams(temperature=0.2, max_tokens=8000),
        untracked_max_bytes=1 << 20,
        indexer=rig.indexer,  # type: ignore[attr-defined]
        **work,  # type: ignore[arg-type]
    )


def _a(tool: str, cid: str, **args: object) -> object:
    return calls(athena_call(tool, cid, **args))


def _w(tool: str, cid: str, **args: object) -> object:
    return calls(wcall(tool, cid, **args))


def _edit(n: int) -> object:
    return _w("create", f"w{n}", path=f"part{n}.py", content=f"PART = {n}\n")


def _one_task(rig: object) -> LoopResult:
    return _loop(rig, one_task=OneTaskPlan("Totals", BRIEF)).run()


def _said(rig: object, cid: str) -> str:
    """What Athena was answered for her call `cid`."""
    for request in rig.athena_rung.requests:  # type: ignore[attr-defined]
        for message in request.messages:
            if getattr(message, "tool_call_id", None) == cid:
                return str(message.content)
    raise AssertionError(f"no answer to {cid}")


# -- decision 20: no plan, or no accepted task, is never COMPLETED ----------------------------


def test_a_design_answered_with_no_plan_is_partial(root: Path) -> None:
    """Moved: it was `COMPLETED`."""
    rig = episode_rig(root, [answer("Nothing to do.")], [])
    result = _loop(rig, design=DESIGN).run()
    assert result == LoopResult("PARTIAL", "no plan was approved", "Nothing to do.")


@pytest.mark.writes_worktree("part1.py")
def test_a_one_task_plan_that_accepted_nothing_is_partial(root: Path) -> None:
    rig = episode_rig(
        root,
        [_a("review", "a1", task="T1", answer="drop"), answer("Gave up.")],
        [_edit(1), answer("T1: tried.")],
    )
    assert _one_task(rig) == LoopResult("PARTIAL", "no task was accepted", "Gave up.")


# -- decision 17: the cap ---------------------------------------------------------------------


@pytest.mark.writes_worktree("part1.py")
def test_the_fourth_task_is_refused_at_the_default_cap(root: Path) -> None:
    """Moved: a one-task plan re-cut into four tasks ran all four."""
    athena = [
        _a("review", "a1", task="T1", answer="re_cut", text="smaller steps"),
        _a("delegate", "a2", title="Step 1", brief="## Step 1"),
        _a("review", "a3", task="T1.1", answer="accept"),
        _a("delegate", "a4", title="Step 2", brief="## Step 2"),
        _a("review", "a5", task="T1.2", answer="accept"),
        _a("delegate", "a6", title="Step 3", brief="## Step 3"),
        answer("Stopped at the cap."),
    ]
    worker = [
        _edit(1), answer("T1: tried."),
        _edit(2), answer("T2: done."),
        _edit(3), answer("T3: done."),
    ]  # fmt: skip
    rig = episode_rig(root, athena, worker)
    result = _one_task(rig)
    assert [e.id for e in rig.desk.book().entries()] == ["T1", "T1.1", "T1.2"]
    refused = _said(rig, "a6")
    assert "athena.one_task_max_tasks = 3" in refused and "outgrown" in refused
    assert result.outcome == "COMPLETED", "two accepted tasks: it ended, with work committed"


@pytest.mark.writes_worktree("part1.py", "part2.py", "part3.py")
def test_a_dropped_task_counts(root: Path) -> None:
    athena = [
        _a("review", "a1", task="T1", answer="drop"),
        _a("delegate", "a2", title="Again", brief="## Again"),
        _a("review", "a3", task="T1", answer="drop"),
        _a("delegate", "a4", title="Once more", brief="## Once more"),
        _a("review", "a5", task="T1", answer="drop"),
        _a("delegate", "a6", title="And again", brief="## And again"),
        answer("Nothing landed."),
    ]
    worker = [
        _edit(1), answer("T1: tried."),
        _edit(2), answer("T2: tried."),
        _edit(3), answer("T3: tried."),
    ]  # fmt: skip
    rig = episode_rig(root, athena, worker)
    assert _one_task(rig) == LoopResult("PARTIAL", "no task was accepted", "Nothing landed.")
    assert "athena.one_task_max_tasks = 3" in _said(rig, "a6")
    assert [e.id for e in rig.desk.book().entries()] == ["T1", "T1@2", "T1@3"]


def test_a_design_is_not_capped(root: Path) -> None:
    plan = "# Plan\n\n1. **A.**\n2. **B.**\n3. **C.**\n4. **D.**\n\nFinal check: the suite.\n"
    athena = [_a("plan", "a0", text=plan)]
    worker: list[object] = []
    for n in range(1, 5):
        athena += [
            _a("delegate", f"d{n}", title=f"Step {n}", brief=f"## Step {n}"),
            _a("review", f"r{n}", task=f"T{n}", answer="accept"),
        ]
        worker += [_edit(n), answer(f"T{n}: done.")]
    rig = episode_rig(root, [*athena, answer("All four.")], worker)
    result = _loop(rig, design=DESIGN).run()
    assert result == LoopResult("COMPLETED", None, "All four.")
    assert len(rig.desk.book().entries()) == 4


# -- decision 16: OUTGROWN ---------------------------------------------------------------------


REPORT = (
    "Missing: whether None means zero or an error is a decision nobody made. "
    "Done: T1.1 makes total skip None, committed. Proposed: ask the user, then one task."
)


@pytest.mark.writes_worktree("part1.py")
def test_outgrown_ends_the_plan_and_reaches_zeus_with_her_report(root: Path) -> None:
    athena = [
        _a("review", "a1", task="T1", answer="re_cut", text="two steps"),
        _a("delegate", "a2", title="Skip None", brief="## Skip None"),
        _a("review", "a3", task="T1.1", answer="accept"),
        _a("outgrown", "a4"),
        _a("delegate", "a5", title="More", brief="## More"),
        answer(REPORT),
    ]
    worker = [_edit(1), answer("T1: tried."), _edit(2), answer("T2: done.")]
    rig = episode_rig(root, athena, worker)
    result = _one_task(rig)

    assert result == LoopResult("OUTGROWN", OUTGROWN_WHY, REPORT)
    assert "reply with your report" in _said(rig, "a4")
    assert "OUTGROWN" in _said(rig, "a5"), "nothing more is delegated after it"
    # The accepted work stays committed on the session's branch.
    subjects = git(root, "log", "--format=%s").splitlines()
    assert "chickadee: T1.1 Skip None" in subjects
    run = rig.journal.projection()
    assert (
        growth(run, TaskBook(run)) == "1 task approved, 2 delegated (T1.1 from a re-cut of T1)"
    )


def test_zeus_reads_outgrown_as_not_done() -> None:
    from chickadee.journal.projection import RunSpend

    report = CoderReport(
        outcome="OUTGROWN",
        why=OUTGROWN_WHY,
        growth="1 task approved, 2 delegated (T1.1 from a re-cut of T1)",
        athena=REPORT,
        cost=RunSpend(),
    )
    rendered = render_report(report)
    text = rendered.text if isinstance(rendered, Said) else str(rendered.content)
    assert text.startswith("OUTGROWN")
    assert "Do not call the work done" in text and "a new one-task plan" in text
    assert "growth: 1 task approved, 2 delegated (T1.1 from a re-cut of T1)" in text
    assert REPORT in text


def test_outgrown_is_refused_in_a_design(root: Path) -> None:
    rig = episode_rig(root, [_a("outgrown", "a1"), answer("No plan.")], [])
    result = _loop(rig, design=DESIGN).run()
    assert "runs a design" in _said(rig, "a1")
    assert result == LoopResult("PARTIAL", "no plan was approved", "No plan.")


def test_outgrown_is_refused_while_a_task_is_open(root: Path) -> None:
    athena = [
        _a("outgrown", "a1"),
        _a("review", "a2", task="T1", answer="accept"),
        answer("Done after all."),
    ]
    rig = episode_rig(root, athena, [_edit(1), answer("T1: done.")])
    result = _one_task(rig)
    assert "'T1 is open'}. Do: review T1 first" in _said(rig, "a1")
    assert result == LoopResult("COMPLETED", None, "Done after all.")


# -- decision 19: the growth is said ---------------------------------------------------------


@pytest.mark.writes_worktree("part1.py")
def test_each_commit_says_how_the_plan_grew(root: Path) -> None:
    athena = [
        _a("review", "a1", task="T1", answer="re_cut", text="two steps"),
        _a("delegate", "a2", title="Step 1", brief="## Step 1"),
        _a("review", "a3", task="T1.1", answer="accept"),
        answer("Done."),
    ]
    worker = [_edit(1), answer("T1: tried."), _edit(2), answer("T2: done.")]
    rig = episode_rig(root, athena, worker)
    _one_task(rig)
    body = git(root, "log", "-1", "--format=%b")
    assert "Growth: 1 task approved, 2 delegated (T1.1 from a re-cut of T1)" in body


def test_a_plan_that_did_not_grow_says_nothing(root: Path) -> None:
    rig = episode_rig(
        root,
        [_a("review", "a1", task="T1", answer="accept"), answer("Done.")],
        [_edit(1), answer("T1: done.")],
    )
    _one_task(rig)
    run = rig.journal.projection()
    assert growth(run, TaskBook(run)) is None
    assert "Growth:" not in git(root, "log", "-1", "--format=%b")


@pytest.mark.writes_worktree("part1.py", "part2.py")
def test_in_place_the_offered_body_says_the_growth_as_a_commit_does(root: Path) -> None:
    """The review of #607: one body assembly for the commit made and the one offered."""
    from chickadee.box.desk import known_good

    athena = [
        _a("review", "a1", task="T1", answer="re_cut", text="two steps"),
        _a("delegate", "a2", title="Step 1", brief="## Step 1"),
        _a("review", "a3", task="T1.1", answer="accept"),
        answer("Done."),
    ]
    worker = [_edit(1), answer("T1: tried."), _edit(2), answer("T2: done.")]
    rig = episode_rig(root, athena, worker, in_place=True)
    _one_task(rig)
    offered = known_good(rig.journal.projection(), rig.journal.get_blob)
    assert (
        offered
        and "Growth: 1 task approved, 2 delegated (T1.1 from a re-cut of T1)"
        in (offered[-1][1])
    )
