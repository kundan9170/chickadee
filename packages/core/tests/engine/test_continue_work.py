"""`coder(work)` continues a work where it stopped, through the engine and its real session log,
box, journal and git, the
models scripted.

What these pin: a work whose run ended with a task unfinished continues in a new run, no stop
asked, its approved plan inherited and never asked again, Athena told where it stands, a task
an earlier run committed refused, and the unfinished task's next attempt going on from its own
start; a work whose design changed is asked at stop 1 with what changed, its Request kept and
the new turn added; a work complete is said to be, and nothing runs; a suspended run resumes."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from coder_helpers import (
    DESIGN_README,
    SID,
    TASK_FILE,
    Approving,
    CoderModel,
    LeavesAsked,
    athena_accepts,
    box_call,
    build,
    make_repo,
    work_name,
    worker_edit,
    zeus_designs,
)
from repos import git
from test_session_worktree import owned
from zeus_turn_helpers import WAIT_S, answer, call, calls, state

import chickadee.delegation as delegation_module
from chickadee.box.episode import opened_with
from chickadee.box.tasks import TaskBook
from chickadee.engine import PacketEngine
from chickadee.errors import SessionError
from chickadee.executor.reconcile import TRAILER
from chickadee.journal.runs import RunProjections
from chickadee.schema.interaction import ApprovalStop, StopDecision
from chickadee.schema.journal import PlanWrittenP, TaskDelegatedP
from chickadee.state.roots import roots as resolve_roots

TITLE = "Return 2"
NAME = work_name(TITLE)
TWO_TASKS = """# Plan: Return 2

## Tasks
1. [T1: Return 2](tasks/T1-task.md): the edit, with its test. Depends on: none
2. [T2: Add b](tasks/T2-b.md): b.py. Depends on: T1

## Final check
The full suite.
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture
def raw(raw: dict[str, Any]) -> dict[str, Any]:
    raw["workspace"] = {"mode": "worktree"}
    raw["sandbox"] = {"require_sandbox": False, "allow_unsandboxed_shell": {"zeus": True}}
    return raw


class Stops(Approving):
    """`Approving`, which keeps every stop it was shown."""

    def __init__(self) -> None:
        super().__init__()
        self.stops: list[ApprovalStop] = []

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        self.stops.append(stop)
        return super().confirm_stop(stop)


def _plans_two() -> list[Any]:
    """Athena writes a two-task plan, has it approved, and delegates T1."""
    t2 = TASK_FILE.format(title="Add b", goal="`b.py` exists").replace("# T1:", "# T2:")
    return [
        calls(box_call("doc", "pd", op="write", file="README.md", content=TWO_TASKS)),
        calls(
            box_call(
                "doc",
                "pt1",
                op="write",
                file="tasks/T1-task.md",
                content=TASK_FILE.format(title=TITLE, goal="`go()` returns 2"),
            )
        ),
        calls(box_call("doc", "pt2", op="write", file="tasks/T2-b.md", content=t2)),
        calls(box_call("plan", "p1")),
        calls(box_call("delegate", "d1", task="T1")),
    ]


def _first_run() -> list[Any]:
    """T1 committed; T2 delegated, its worker half done, and Athena replying with it open:
    the run ends `PARTIAL`, T2 unfinished, its file on disk, uncommitted."""
    return [
        *zeus_designs(TITLE, "`go()` returns 2"),
        *_plans_two(),
        *worker_edit("return 1", "return 2"),
        calls(box_call("review", "a1", task="T1", answer="accept")),
        calls(box_call("delegate", "d2", task="T2")),
        calls(box_call("create", "w2", path="b.py", content="b = 1\n")),
        answer("T2: half done."),
        answer("Stopping with T2 open."),
        answer("Partly done."),
    ]


def _runs(engine: PacketEngine) -> list[str]:
    return [e.run_id for e in state(engine, SID).episodes()]


def _tree(engine: PacketEngine) -> Path:
    (ref, *_) = owned(engine)
    assert ref.worktree_path is not None
    return Path(ref.worktree_path)


def _projection(repo: Path, run_id: str) -> Any:
    return RunProjections(resolve_roots(repo)).readable(run_id)


def _user_texts(model: CoderModel) -> list[str]:
    return [
        str(next(m.content for m in r.messages if m.role == "user"))
        for r in model.requests
        if any(m.role == "user" for m in r.messages)
    ]


def test_a_work_left_partial_continues_in_a_new_run_from_where_it_stood(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    person = Stops()
    engine, model = build(monkeypatch, repo, raw, _first_run(), interaction=person)
    engines.append(engine)
    engine.turn(SID, "make go return 2, and add b").wait(timeout=WAIT_S)
    [first] = _runs(engine)
    assert _projection(repo, first).end.outcome == "PARTIAL"
    asked_before = len(person.stops)

    model.then(
        calls(call("coder", "c2", work=NAME)),
        # Athena, in the new run: T1 is refused, T2 goes on from what its attempt left.
        calls(box_call("delegate", "e1", task="T1")),
        calls(box_call("delegate", "e2", task="T2")),
        calls(box_call("create", "w3", path="c.py", content="c = 1\n")),
        answer("T2: done."),
        *athena_accepts("T2", n=9, report="T2 accepted."),
        answer("Done."),
    )
    engine.turn(SID, "continue").wait(timeout=WAIT_S)
    second = _runs(engine)[-1]
    assert second != first

    assert len(person.stops) == asked_before, "no stop: the design and plan were approved"
    request = opened_with(resolve_roots(repo), second)
    assert request.continues is not None
    assert (request.continues.mode, request.continues.prior_runs) == ("CONTINUE", (first,))
    run = _projection(repo, second)
    [seeded, *_] = [p.payload for p in run.lines(PlanWrittenP)]
    assert (seeded.inherited_from, seeded.asked, seeded.version) == (first, False, 1)

    head = next(t for t in _user_texts(model) if "## Where the work stands" in t)
    assert "Committed: T1" in head and "Not finished: T2" in head
    tools = "\n".join(
        str(m.content) for r in model.requests for m in r.messages if m.role == "tool"
    )
    assert "T1 was committed in an earlier run of this work" in tools
    texts = "\n".join(_user_texts(model))
    assert "## Where this task stands" in texts, "T2's worker is told it goes on"
    assert "uncommitted: b.py." in texts, "and what the earlier attempt left"

    entries = TaskBook(run).entries()
    assert [e.id for e in entries] == ["T2@2"], "T1 refused; T2's next attempt"
    # What a screen reads, the work's one fold across its runs: T1 is the first run's.
    work = engine.work(SID)
    assert work is not None and work.runs == (first, second)
    assert {t.id: (t.state, t.run) for t in work.tasks}["T1"] == ("committed", first)
    first_t2 = next(
        p.payload
        for p in _projection(repo, first).lines(TaskDelegatedP)
        if p.payload.task_id == "T2"
    )
    second_t2 = next(p.payload for p in run.lines(TaskDelegatedP))
    assert second_t2.start_tree != first_t2.start_tree, "it starts from the tree as it is"
    log = git(_tree(engine), "log", "--format=%H %s").splitlines()
    [t2] = [line.split(" ", 1)[0] for line in log if line.endswith("chickadee: T2 Add b")]
    files = git(_tree(engine), "show", "--name-only", "--format=", t2).split()
    assert {"b.py", "c.py"} <= set(files), "the earlier attempt's file is the task's"
    subjects = [line.split(" ", 1)[1] for line in log]
    assert subjects.count("chickadee: plan approved") == 1, "the plan was approved once"


def test_a_work_whose_design_changed_is_asked_with_what_changed(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    person = Stops()
    engine, model = build(monkeypatch, repo, raw, _first_run(), interaction=person)
    engines.append(engine)
    engine.turn(SID, "make go return 2, and add b").wait(timeout=WAIT_S)
    [first] = _runs(engine)
    amended = DESIGN_README.format(title=TITLE, goal="`go()` returns 2") + "\n## Also\nAnd c.\n"
    model.then(
        calls(call("doc", "z1", op="write", file="README.md", content=amended)),
        calls(call("coder", "c2", work=NAME)),
        # The plan stands for the amended design: no revision, so no stop 2.
        calls(box_call("plan", "p2")),
        answer("Nothing to add."),
        answer("Done."),
    )
    engine.turn(SID, "also c").wait(timeout=WAIT_S)
    second = _runs(engine)[-1]

    design_stops = [s for s in person.stops if s.kind.value == "design"]
    assert len(design_stops) == 2
    shown = design_stops[-1]
    assert (
        shown.notice is not None and "Changed since the design approved in run" in shown.notice
    )
    assert "## Request\nmake go return 2, and add b\n\nThen:\nalso c" in shown.text
    request = opened_with(resolve_roots(repo), second)
    assert request.continues is not None and request.continues.mode == "AMEND"
    assert request.continues.prior_runs == (first,)
    assert request.continues.amended_from is not None
    plan_stops = [s for s in person.stops if s.kind.value == "plan"]
    assert len(plan_stops) == 1, "the plan stood: never asked again"
    subjects = git(_tree(engine), "log", "--format=%s").splitlines()
    assert "chickadee: design amended" in subjects
    assert "chickadee: plan stands (version 2)" in subjects
    # The first run committed `approved:design` too: the amendment's commit is made all the
    # same, its trailer searched for only since this run began.
    amended = git(_tree(engine), "log", "-1", "--format=%B", "--grep=design amended")
    assert f"{TRAILER}: approved:design" in amended


def test_a_carried_task_dropped_puts_back_only_this_runs_changes(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """The unfinished T2 starts from the tree as it is, not from its earlier attempt's start,
    so dropping it never puts back the design amended since, nor T1's commit."""
    engine, model = build(monkeypatch, repo, raw, _first_run(), interaction=Stops())
    engines.append(engine)
    engine.turn(SID, "make go return 2, and add b").wait(timeout=WAIT_S)
    amended = DESIGN_README.format(title=TITLE, goal="`go()` returns 2") + "\n## Also\nAnd c.\n"
    model.then(
        calls(call("doc", "z1", op="write", file="README.md", content=amended)),
        calls(call("coder", "c2", work=NAME)),
        calls(box_call("plan", "p2")),
        calls(box_call("delegate", "e2", task="T2")),
        calls(box_call("create", "w3", path="c.py", content="c = 1\n")),
        answer("T2: tried."),
        calls(box_call("review", "a2", task="T2", answer="drop")),
        answer("Dropped T2."),
        answer("Done."),
    )
    engine.turn(SID, "also c").wait(timeout=WAIT_S)
    second = _runs(engine)[-1]
    assert [e.state for e in TaskBook(_projection(repo, second)).entries()] == ["abandoned"]

    tree = _tree(engine)
    readme = tree / "docs" / "chickadee" / NAME / "design" / "README.md"
    assert "## Also" in readme.read_text(encoding="utf-8"), "the amended design stays"
    assert "return 2" in (tree / "app.py").read_text(encoding="utf-8"), "T1's commit stays"
    assert not (tree / "c.py").exists(), "this run's attempt is put back"
    assert (tree / "b.py").exists(), "what the earlier run left waits for the next attempt"


def test_a_complete_work_is_said_to_be_and_nothing_runs(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    from coder_helpers import athena_plans_folder

    script = [
        *zeus_designs(TITLE, "`go()` returns 2"),
        *athena_plans_folder(),
        *worker_edit("return 1", "return 2"),
        *athena_accepts(),
        answer("done"),
        calls(call("coder", "c2", work=NAME)),
        answer("It is complete."),
    ]
    engine, _ = build(monkeypatch, repo, raw, script)
    engines.append(engine)
    engine.turn(SID, "make go return 2").wait(timeout=WAIT_S)
    engine.turn(SID, "continue").wait(timeout=WAIT_S)
    said = state(engine, SID).result("c2")
    assert said is not None and said.text.startswith(f"COMPLETE · work {NAME}")
    assert "nothing ran" in said.text
    assert len(_runs(engine)) == 2 and _projection_missing(repo, _runs(engine)[-1])


def _projection_missing(repo: Path, run_id: str) -> bool:
    try:
        _projection(repo, run_id)
    except FileNotFoundError:
        return True
    return False


def _left_suspended(monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]) -> str:
    """A run of the work left suspended at a question the person never answered: its id."""
    from coder_helpers import athena_plans_folder

    leaving = LeavesAsked()
    engine, _ = build(
        monkeypatch,
        repo,
        dict(raw),
        [
            *zeus_designs(TITLE, "`go()` returns 2"),
            *athena_plans_folder(),
            calls(box_call("shell", "w1", command="git checkout -- app.py")),
        ],
        interaction=leaving,
    )
    engine.turn(SID, "make go return 2")
    assert leaving.asked.wait(30), "the worker's command was put to the person"
    engine.close()
    [first] = _clone_runs(repo)
    assert _projection(repo, first).end is None, "left suspended"
    return first


def test_a_suspended_run_resumes_on_coder_and_no_stop_is_asked(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    first = _left_suspended(monkeypatch, repo, raw)
    back = Stops()
    engine, zeus = build(
        monkeypatch,
        repo,
        raw,
        [calls(call("coder", "c2", work=NAME)), answer("Resumed.")],
        interaction=back,
    )
    engines.append(engine)
    box = CoderModel(
        [
            calls(box_call("shell", "w2", command="true")),
            answer("T1: ran it."),
            *athena_accepts(),
        ]
    )
    monkeypatch.setattr(delegation_module, "client_for", lambda _endpoint: box)
    engine.turn(SID, "continue").wait(timeout=WAIT_S)
    said = state(engine, SID).result("c2")
    # Unattended, the call waits for the resumed run and answers its report (
    # as amended): moved from `RESUMED`, which said only that it went on.
    assert said is not None and said.text.startswith(f"COMPLETED · run {first}")
    assert back.stops == [], "nothing approved is asked again"
    deadline = time.monotonic() + 60
    while _projection(repo, first).end is None and time.monotonic() < deadline:
        time.sleep(0.2)
    assert _projection(repo, first).end is not None, "the resumed run ran to its end"
    del zeus


def test_a_suspended_run_that_cannot_resume_is_answered_not_raised(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """`/resume` refuses a run whose worktree was handed off with a `SessionError`; the same
    refusal reaches Zeus as the call's answer, never as an error ending his turn."""
    first = _left_suspended(monkeypatch, repo, raw)

    def handed_off(*_args: object) -> Any:
        raise SessionError(f"run {first} was paused and its worktree then handed off")

    monkeypatch.setattr(PacketEngine, "_continue_run", handed_off)
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [calls(call("coder", "c2", work=NAME)), answer("It cannot go on.")],
        interaction=Stops(),
    )
    engines.append(engine)
    engine.turn(SID, "continue").wait(timeout=WAIT_S)
    said = state(engine, SID).result("c2")
    assert said is not None and said.text.startswith("BLOCKED")
    assert "its worktree then handed off" in said.text
    assert _projection(repo, first).end is None, "left as it was"


def test_another_work_is_refused_while_a_run_is_paused(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    raw: dict[str, Any],
    engines: list[PacketEngine],
) -> None:
    """A `coder` call for another work while the session's run is paused is refused,
    naming the paused work and `/drop`: nothing is asked, and no run opens beside it."""
    first = _left_suspended(monkeypatch, repo, raw)
    back = Stops()
    engine, _ = build(
        monkeypatch,
        repo,
        raw,
        [
            calls(call("coder", "c2", title="Add b", brief="## Goal\n`b.py` exists.\n")),
            answer("It is paused."),
        ],
        interaction=back,
    )
    engines.append(engine)
    engine.turn(SID, "add b").wait(timeout=WAIT_S)
    said = state(engine, SID).result("c2")
    assert said is not None and said.text.startswith("BLOCKED")
    assert f"run {first} of work {NAME} is paused" in said.text
    assert f"coder(work={NAME!r})" in said.text and "/drop" in said.text
    assert back.stops == [], "nothing is asked"
    assert _clone_runs(repo) == [first], "no run opened beside it"
    assert _projection(repo, first).end is None, "left as it was"


def _clone_runs(repo: Path) -> list[str]:
    """Every run of the clone, by the journals' own directory."""
    from chickadee.journal.paths import runs_root

    return sorted(d.name for d in runs_root(resolve_roots(repo)).iterdir() if d.is_dir())
