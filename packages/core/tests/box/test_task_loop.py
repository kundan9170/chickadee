"""The task loop end to end:
`box/loop.TaskLoop` with a scripted Athena and a scripted worker over a real repository.

What these pin: D196 the request runs as written: Athena reads, plans, stop 2 is asked once,
three tasks are delegated one at a time, each report reviewed (one sent back with feedback),
each accept a commit with the operation's trailer, the next worker's head saying what was done
so far, her final check run by her own shell, her reply the episode's report. The one-task
plan: Zeus's plan and delegation, the worker's first round before Athena, and Athena beginning
at review. A task she takes over carries her own edit into its commit. A cancel at stop 2 is
the person's stop; a reply with a task still open is `PARTIAL`. Stop 1 is the test's: the
design arrives approved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from box_helpers import EpisodeRig, athena_call, episode_rig
from box_helpers import worker_call as wcall
from engines import answer, calls
from repos import git, init_repo

from chickadee.approval.stops import StopOutcome
from chickadee.box.loop import HeadParts, LoopResult, OneTaskPlan, TaskLoop
from chickadee.box.tasks import TaskBook
from chickadee.boxctx.athena import ONE_TASK, TASK_DESIGN
from chickadee.executor.reconcile import TRAILER
from chickadee.journal.effects import EffectKind
from chickadee.schema import RetryParams
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import StopAnswer
from chickadee.schema.journal import (
    PlanWrittenP,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
)

STATS = (
    "def naive_sum(xs):\n"
    "    total = 0\n"
    "    for x in xs:\n"
    "        total += x\n"
    "    return total\n"
    "\n"
    "\n"
    "def total(xs):\n"
    "    return naive_sum(xs)\n"
)
LEGACY = "print('the old generator')\n"
DESIGN = (
    "# Segment tree with range-sum queries\n\n"
    "## Request\n> add a segment tree with range-sum queries and point updates, and use it "
    "in stats\n\n## What changes\n- New `segtree.py`.\n"
)
PLAN = (
    "# Plan: segment tree with range-sum queries\n\n"
    "1. **Build and update.** Create `segtree.py`.\n"
    "2. **Range query.** Add `query(lo, hi)`.\n"
    "3. **Use it in stats.** `stats.total()` uses it; delete `scripts/legacy/gen.py`.\n\n"
    "Final check: `tests_for` on every changed file, then the full suite.\n"
)
SEGTREE = (
    "class SegmentTree:\n"
    "    def __init__(self, values):\n"
    "        self.values = list(values)\n"
    "\n"
    "    def update(self, i, value):\n"
    "        self.values[i] = value\n"
)
QUERY = "\n    def query(self, lo, hi):\n        return sum(self.values[lo:hi])\n"
TEST_BUILD = "def test_build():\n    pass\n"
USE_IT = "    from segtree import SegmentTree\n\n    return SegmentTree(xs).query(0, len(xs))\n"
EMPTY = "    total = 0  # empty: 0\n"
REPORT = "Built the segment tree in three tasks; the full suite ran and printed 1 passed."
HEAD = HeadParts("You are Athena.", "PLAN TEMPLATE", "BRIEF TEMPLATE")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(
        tmp_path / "repo",
        {
            "stats.py": STATS,
            "scripts/legacy/gen.py": LEGACY,
            "tests/test_stats.py": "def test_total():\n    assert True\n",
        },
    )


def _loop(rig: EpisodeRig, **work: object) -> TaskLoop:
    return TaskLoop(
        rig.athena,
        rig.desk,
        rig.journal,
        HEAD,
        root=rig.root,
        params=RetryParams(temperature=0.2, max_tokens=8000),
        untracked_max_bytes=1 << 20,
        indexer=rig.indexer,
        **work,  # type: ignore[arg-type]
    )


def _a(tool: str, cid: str, **args: object) -> object:
    return calls(athena_call(tool, cid, **args))


def _w(tool: str, cid: str, **args: object) -> object:
    return calls(wcall(tool, cid, **args))


def _subjects(root: Path) -> list[str]:
    return git(root, "log", "--format=%s").splitlines()


# -- D196 §4, one request end to end -----------------------------------------------------------


def test_section_4_runs_end_to_end(root: Path) -> None:
    athena = [
        _a("read", "a1", path="stats.py"),
        _a("plan", "a2", text=PLAN),
        _a("delegate", "a3", title="Build and update", brief="## Task 1 of 3: build"),
        _a("review", "a4", task="T1", answer="accept"),
        _a("delegate", "a5", title="Range query", brief="## Task 2 of 3: range query"),
        _a("review", "a6", task="T2", answer="feedback", text="add a test for one element"),
        _a("review", "a7", task="T2", answer="accept"),
        _a("delegate", "a8", title="Use it in stats", brief="## Task 3 of 3: use it"),
        _a("review", "a9", task="T3", answer="accept"),
        _a("shell", "a10", command="pytest -q"),
        answer(REPORT),
    ]
    worker = [
        # T1
        _w("create", "w1", path="segtree.py", content=SEGTREE),
        _w("create", "w2", path="tests/test_segtree.py", content=TEST_BUILD),
        _w("shell", "w3", command="pytest tests/test_segtree.py -q"),
        answer("T1: created segtree.py with update(); 1 passed."),
        # T2, round 1, then round 2 after the feedback
        _w("edit", "w4", path="segtree.py", old="        self.values[i] = value\n",
           new="        self.values[i] = value\n" + QUERY),
        answer("T2: added query()."),
        _w("edit", "w5", path="tests/test_segtree.py", old="    pass\n",
           new="    pass\n\n\ndef test_one():\n    pass\n"),
        answer("T2: added a test for one element."),
        # T3
        _w("edit", "w6", path="stats.py", old="    return naive_sum(xs)\n", new=USE_IT),
        _w("delete", "w7", path="scripts/legacy/gen.py"),
        answer("T3: stats uses SegmentTree; the old generator is deleted."),
    ]  # fmt: skip
    rig = episode_rig(root, athena, worker)
    result = _loop(rig, design=DESIGN).run()

    assert result == LoopResult("COMPLETED", None, REPORT)
    # Stop 2 asked once, on her plan; stop 1 was the test's.
    [stop] = rig.stops
    assert (stop.kind, stop.text) == ("plan", PLAN)
    # Three commits, one per task, each under its operation's trailer.
    assert _subjects(root)[:3] == [
        "chickadee: T3 Use it in stats",
        "chickadee: T2 Range query",
        "chickadee: T1 Build and update",
    ]
    for i, task in enumerate(("T3", "T2", "T1")):
        trailer = git(
            root, "log", "-1", f"--skip={i}", f"--format=%(trailers:key={TRAILER},valueonly)"
        )
        assert trailer.strip() == f"task-commit:{task}:1"
    assert "def query" in (root / "segtree.py").read_text()
    assert not (root / "scripts" / "legacy" / "gen.py").exists()
    assert git(root, "status", "--porcelain") == ""

    run = rig.journal.projection()
    assert [p.payload.version for p in run.lines(PlanWrittenP)] == [1]
    assert [p.payload.task_id for p in run.lines(TaskDelegatedP)] == ["T1", "T2", "T3"]
    assert [(p.payload.task_id, p.payload.round) for p in run.lines(TaskReportedP)] == [
        ("T1", 1),
        ("T2", 1),
        ("T2", 2),
        ("T3", 1),
    ]
    assert [(p.payload.task_id, p.payload.answer) for p in run.lines(TaskReviewedP)] == [
        ("T1", "accept"),
        ("T2", "feedback"),
        ("T2", "accept"),
        ("T3", "accept"),
    ]
    assert [e.state for e in TaskBook(run).entries()] == ["committed"] * 3
    # T3's facts: stats.py edited, the generator deleted.
    t3 = [p.payload for p in run.lines(TaskReportedP) if p.payload.task_id == "T3"][-1]
    assert {(f.path, f.kind) for f in t3.facts.files} == {
        ("stats.py", "edited"),
        ("scripts/legacy/gen.py", "deleted"),
    }
    # Her final check is her own step, under no task.
    [check] = [
        op
        for op in run.operations.values()
        if op.kind == EffectKind.WORKER_STEP.value
        and op.operation_id.startswith("step:athena:0:")
    ]
    assert check.task_id is None and check.surface == ("shell",)

    # The second task's worker starts fresh, its head saying what the first one committed.
    heads = [r.messages[1].content for r in rig.worker_rung.requests if len(r.messages) == 2]
    assert len(heads) == 3
    assert "[T1] Build and update" in str(heads[1]) and "segtree.py" in str(heads[1])
    # Athena's head is her instructions and the templates, then the approved design.
    first = rig.athena_rung.requests[0].messages
    assert "PLAN TEMPLATE" in str(first[0].content) and "BRIEF TEMPLATE" in str(
        first[0].content
    )
    assert str(first[1].content).startswith(TASK_DESIGN) and DESIGN.strip() in str(
        first[1].content
    )
    # Each task's report reached her with its facts.
    sent = rig.athena_rung.requests[-1].messages
    reports = [
        str(m.content) for m in sent if m.role == "tool" and "REPORT (worker)" in str(m.content)
    ]
    assert len(reports) == 4 and all("FACTS (harness):" in r for r in reports)


# -- the one-task plan ------------------------------------------------------------


def test_a_one_task_plan_runs_the_worker_first_and_athena_begins_at_review(root: Path) -> None:
    brief = "## Task 1 of 1: total of an empty list\n\n### Goal\n`total([])` is 0.\n"
    rig = episode_rig(
        root,
        [_a("review", "a1", task="T1", answer="accept"), answer("Accepted T1; done.")],
        [
            _w("edit", "w1", path="stats.py", old="    total = 0\n", new=EMPTY),
            answer("T1: done."),
        ],
    )  # fmt: skip
    result = _loop(rig, one_task=OneTaskPlan("Empty total is 0", brief)).run()

    assert result == LoopResult("COMPLETED", None, "Accepted T1; done.")
    assert rig.stops == [], "stop 2 for a one-task plan is asked at dispatch, by Zeus's call"
    run = rig.journal.projection()
    [plan] = run.lines(PlanWrittenP)
    assert (plan.payload.by, plan.payload.call_id, plan.payload.asked) == ("zeus", None, True)
    assert plan.payload.sha == sha256_hex(brief)
    [delegated] = run.lines(TaskDelegatedP)
    assert (delegated.payload.by, delegated.payload.call_id) == ("zeus", None)
    assert _subjects(root)[0] == "chickadee: T1 Empty total is 0"
    # The worker ran before Athena was asked anything; her head holds the brief and its round.
    head = str(rig.athena_rung.requests[0].messages[1].content)
    assert head.startswith(ONE_TASK) and brief.strip() in head
    assert "T1 · round 1\nREPORT (worker): T1: done." in head


# -- she takes a task over ---------------------------------------------------------------------


def test_a_task_she_takes_over_carries_her_edit_into_its_commit(root: Path) -> None:
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Fix total", brief="## Task 1 of 1: fix it"),
            _a("review", "a3", task="T1", answer="take_over", text="I will finish it"),
            _a("edit", "a4", path="stats.py", old="    return naive_sum(xs)\n",
               new="    return sum(xs)\n"),
            _a("review", "a5", task="T1", answer="accept"),
            answer("Took T1 over and finished it."),
        ],
        [answer("I could not find it.")],
    )  # fmt: skip
    result = _loop(rig, design=DESIGN).run()

    assert result.outcome == "COMPLETED"
    assert git(root, "show", "HEAD:stats.py").endswith("    return sum(xs)")
    run = rig.journal.projection()
    [step] = [
        op
        for op in run.operations.values()
        if op.kind == EffectKind.WORKER_STEP.value and op.surface == ("edit",)
    ]
    assert step.task_id == "T1" and step.operation_id.startswith("step:T1:0:")


# -- how an episode ends otherwise -------------------------------------------------------------


def test_a_cancel_at_stop_2_is_the_persons_stop(root: Path) -> None:
    cancel = StopOutcome(StopAnswer.CANCEL, PLAN, sha256_hex(PLAN), "not now")
    rig = episode_rig(
        root,
        [_a("plan", "a1", text=PLAN), answer("Cancelled; nothing ran.")],
        [],
        stop=lambda _s: cancel,
    )
    result = _loop(rig, design=DESIGN).run()
    assert (result.outcome, result.why) == (
        "ABORTED",
        "the user cancelled the plan: not now",
    )
    assert rig.journal.projection().lines(TaskDelegatedP) == ()
    assert len(rig.athena_rung.requests) == 2, "her tool use ended: the next turn was her reply"


def test_a_reply_with_a_task_open_is_partial_after_one_reminder(root: Path) -> None:
    """T-3.159 (a REPL run): she replied with T1 reported and never reviewed. She is told once
    that nothing of it is committed; a second such reply ends the episode `PARTIAL`."""
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Fix total", brief="## Task 1 of 1: fix it"),
            answer("I stopped early."),
            answer("Still stopping."),
        ],
        [answer("done")],
    )
    result = _loop(rig, design=DESIGN).run()
    assert result.outcome == "PARTIAL"
    assert result.why == "T1 'Fix total' is open, uncommitted (reported)"
    assert result.athena == "Still stopping."
    told = [
        m.content
        for m in rig.athena_rung.requests[-1].messages
        if m.role == "user" and str(m.content).startswith("Not finished:")
    ]
    assert told == [
        "Not finished: T1 'Fix total' is still open and nothing of it is committed. Review it "
        "with `review` (accept, feedback, take_over, re_cut or drop), then reply with your "
        "report."
    ]


def test_reminded_she_reviews_and_the_task_is_committed(root: Path) -> None:
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Fix total", brief="## Task 1 of 1: fix it"),
            answer("Done, I think."),
            _a("review", "a3", task="T1", answer="accept"),
            answer(REPORT),
        ],
        [_w("edit", "w1", path="stats.py", old="    return naive_sum(xs)\n", new=USE_IT),
         answer("done")],
    )  # fmt: skip
    result = _loop(rig, design=DESIGN).run()
    assert (result.outcome, result.athena) == ("COMPLETED", REPORT)


def test_a_failed_request_after_the_reminder_keeps_her_first_reply(root: Path) -> None:
    """Review: the reminder's request replaced her first reply, so a client failure there
    ended the episode `BLOCKED` with no report. Her first reply stands: `PARTIAL`, reported."""
    from chickadee.client import TransportExhausted

    def fails(_request: object) -> object:
        raise TransportExhausted("reset by peer", 3)

    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Fix total", brief="## Task 1 of 1: fix it"),
            answer("I stopped early."),
            fails,
        ],
        [answer("done")],
    )
    result = _loop(rig, design=DESIGN).run()
    assert (result.outcome, result.athena) == ("PARTIAL", "I stopped early.")
