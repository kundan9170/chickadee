"""The episode's resume (its remaining rows):
`box/loop.TaskLoop.resume`, each row made by killing the process where the row says, then
reopening the run's journal and resuming.

What these pin, row by row: Athena's step cut off is put back and answered `interrupted`, and
the call of hers that never ran `not run`; a cut-off command whose permission question was
still waiting is dispatched again, never answered for the person; a model call cut off is
asked again with the same messages; a `delegate` cut off mid-round resumes the worker's
round, and one cut off after the report answers the report, the task delegated once; a
feedback round cut off continues; an accept whose commit did not complete is found by its
trailer, or committed now, never twice; a committed task is not committed again; a drop
whose restore did not complete is restored again; a pause resumes; a one-task plan cut off in
its first round continues it; a request already answered ends with no model call.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from box_helpers import CrashingCommand, EpisodeRig, KilledMidCall, athena_call, episode_rig
from box_helpers import worker_call as wcall
from engines import answer, calls
from repos import git, init_repo

from chickadee.approval.stops import StopOutcome
from chickadee.box import commit as commit_module
from chickadee.box import desk as desk_module
from chickadee.box.desk import TaskDesk
from chickadee.box.loop import HeadParts, LoopResult, OneTaskPlan, TaskLoop
from chickadee.box.resume import INTERRUPTED, NOT_RUN
from chickadee.box.tasks import TaskBook
from chickadee.executor.reconcile import TRAILER
from chickadee.journal.effects import EffectKind
from chickadee.schema import RetryParams
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import StopAnswer
from chickadee.schema.journal import ExecP, TaskDelegatedP, TaskReportedP, ToolCallP

APP = "def go() -> int:\n    return 1\n"
DESIGN = "# go returns 2\n\n## What changes\n- `go()` returns 2.\n"
PLAN = "# Plan: go returns 2\n\n1. **Return 2.**\n\nFinal check: the suite.\n"
BRIEF = "## Task 1 of 1: return 2"
HEAD = HeadParts("You are Athena.", "PLAN TEMPLATE", "BRIEF TEMPLATE")
EDIT = wcall("edit", "w1", path="app.py", old="return 1", new="return 2")


def _crash(*_: object, **__: object) -> Any:
    raise KilledMidCall


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def _loop(rig: EpisodeRig, **work: Any) -> TaskLoop:
    if not work:
        work = {"design": DESIGN}
    return TaskLoop(
        rig.athena,
        rig.desk,
        rig.journal,
        HEAD,
        root=rig.root,
        params=RetryParams(temperature=0.2, max_tokens=8000),
        untracked_max_bytes=1 << 20,
        indexer=rig.indexer,
        **work,
    )


def _a(tool: str, cid: str, **args: object) -> object:
    return calls(athena_call(tool, cid, **args))


def _crashed(rig: EpisodeRig, **work: Any) -> None:
    """Run the episode until the process dies, and leave the journal as it died."""
    with pytest.raises(KilledMidCall):
        _loop(rig, **work).run()
    rig.journal.close()


def _resumed(
    root: Path,
    athena: list[object],
    worker: list[object],
    **work: Any,
) -> tuple[EpisodeRig, LoopResult]:
    rig = episode_rig(root, athena, worker, resume=True)
    return rig, _loop(rig, **work).resume()


def _last_tool(rig: EpisodeRig) -> list[tuple[str | None, object]]:
    sent = rig.athena_rung.requests[0].messages
    return [(m.tool_call_id, m.content) for m in sent if m.role == "tool"]


def _commits(root: Path, task: str) -> list[str]:
    log = git(root, "log", f"--format=%H %(trailers:key={TRAILER},valueonly)").splitlines()
    return [line.split()[0] for line in log if f" task-commit:{task}:" in line]


# -- Athena's own calls ------------------------------------------------------------------------


@pytest.mark.writes_worktree("app.py", "made.txt")
def test_her_step_cut_off_is_put_back_and_answered_and_a_call_never_run_is_said(
    root: Path,
) -> None:
    rig = episode_rig(
        root,
        [
            calls(
                athena_call("shell", "a1", command="make gen"),
                athena_call("read", "a2", path="app.py"),
            )
        ],
        [],
        supervisor=CrashingCommand(root),
    )
    _crashed(rig)
    assert (root / "app.py").read_text() == "half written\n"

    rig, result = _resumed(root, [answer("ok")], [])
    assert result == LoopResult("PARTIAL", "no plan was approved", "ok")
    assert (root / "app.py").read_text() == APP
    assert _last_tool(rig) == [("a1", INTERRUPTED), ("a2", NOT_RUN)]
    [op] = [o for o in rig.journal.projection().operations.values()]
    assert op.completed is not None and op.completed.outcome == "failed"
    assert op.task_id is None


class _Asks:
    """The person's answer to a shell's question: the process dies while it waits, or yes."""

    def __init__(self, *, dies: bool) -> None:
        self.dies = dies
        self.asked: list[str] = []

    def authorize(self, claim: Any, cancel: threading.Event | None = None) -> bool:
        self.asked.append(claim.operation_id)
        if self.dies:
            raise KilledMidCall
        return True


def test_a_command_whose_permission_question_was_waiting_is_dispatched_again(
    root: Path,
) -> None:
    waiting = _Asks(dies=True)
    rig = episode_rig(root, [_a("shell", "a1", command="uv sync")], [], authority=waiting)
    _crashed(rig)
    [cut] = waiting.asked
    assert rig.journal.projection().operations[cut].completed is None

    asks = _Asks(dies=False)
    rig = episode_rig(
        root,
        [answer("installed")],
        [],
        resume=True,
        authority=asks,
        asking=lambda op: op == cut,  # the session log: its question still waiting
    )
    result = _loop(rig).resume()
    assert (result.outcome, result.why) == ("PARTIAL", "no plan was approved")
    assert len(asks.asked) == 1, "the person is asked again, never answered for"
    [(cid, said)] = _last_tool(rig)
    assert cid == "a1" and said != INTERRUPTED and "1 passed" in str(said)
    run = rig.journal.projection()
    assert len(run.lines(ExecP)) == 1, "the command ran once, after the resume"
    assert all(o.completed is not None for o in run.operations.values())


def test_a_workers_command_whose_question_was_waiting_is_dispatched_again(root: Path) -> None:
    waiting = _Asks(dies=True)
    rig = episode_rig(
        root,
        [_a("plan", "a1", text=PLAN), _a("delegate", "a2", title="Install", brief=BRIEF)],
        [calls(wcall("shell", "w1", command="uv sync"))],
        authority=waiting,
    )
    _crashed(rig)
    [cut] = waiting.asked

    asks = _Asks(dies=False)
    rig = episode_rig(
        root,
        [_a("review", "a3", task="T1", answer="accept"), answer("installed")],
        [answer("synced")],
        resume=True,
        authority=asks,
        asking=lambda op: op == cut,
    )
    result = _loop(rig).resume()
    assert result.outcome == "COMPLETED"
    assert len(asks.asked) == 1, "the person is asked again, never answered for"
    said = [m for m in rig.worker_rung.requests[0].messages if m.role == "tool"]
    assert said[-1].tool_call_id == "w1" and "1 passed" in str(said[-1].content)


def test_a_model_call_cut_off_is_asked_again_with_the_same_messages(root: Path) -> None:
    rig = episode_rig(root, [_a("plan", "a1", text=PLAN), _crash], [])
    _crashed(rig)
    cut = rig.athena_rung.requests[-1].messages

    rig, result = _resumed(root, [answer("planned; nothing more")], [])
    assert result.athena == "planned; nothing more"
    assert rig.athena_rung.requests[0].messages == cut
    assert rig.stops == [], "the plan was approved before the crash: not asked again"


def test_a_request_already_answered_ends_with_no_model_call(root: Path) -> None:
    rig = episode_rig(root, [answer("nothing to do")], [])
    assert _loop(rig).run().outcome == "PARTIAL"  # no plan was approved
    rig.journal.close()
    rig, result = _resumed(root, [], [])
    assert result == LoopResult("PARTIAL", "no plan was approved", "nothing to do")
    assert rig.athena_rung.requests == []


# -- a task's rows -----------------------------------------------------------------------------


def test_a_delegate_cut_off_mid_round_resumes_the_workers_round(root: Path) -> None:
    rig = episode_rig(
        root,
        [_a("plan", "a1", text=PLAN), _a("delegate", "a2", title="Return 2", brief=BRIEF)],
        [calls(EDIT), _crash],
    )
    _crashed(rig)
    cut = rig.worker_rung.requests[-1].messages

    rig, result = _resumed(
        root,
        [_a("review", "a3", task="T1", answer="accept"), answer("done")],
        [answer("returned 2")],
    )
    assert result.outcome == "COMPLETED"
    assert rig.worker_rung.requests[0].messages == cut, "the round continued, not restarted"
    run = rig.journal.projection()
    assert len(run.lines(TaskDelegatedP)) == 1
    [delegated] = [m for m in _last_tool(rig) if m[0] == "a2"]
    assert "REPORT (worker): returned 2" in str(delegated[1])
    assert len(_commits(root, "T1")) == 1


def test_a_delegate_cut_off_after_its_report_answers_the_report(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = episode_rig(
        root,
        [_a("plan", "a1", text=PLAN), _a("delegate", "a2", title="Return 2", brief=BRIEF)],
        [calls(EDIT), answer("returned 2")],
    )
    monkeypatch.setattr(TaskDesk, "_render", _crash)
    _crashed(rig)
    monkeypatch.undo()
    assert len(rig.journal.projection().lines(TaskReportedP)) == 1

    rig, result = _resumed(
        root, [_a("review", "a3", task="T1", answer="accept"), answer("done")], []
    )
    assert result.outcome == "COMPLETED"
    assert rig.worker_rung.requests == [], "the worker is not asked again"
    [delegated] = [m for m in _last_tool(rig) if m[0] == "a2"]
    assert "REPORT (worker): returned 2" in str(delegated[1])


def test_a_feedback_round_cut_off_continues(root: Path) -> None:
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Return 2", brief=BRIEF),
            _a("review", "a3", task="T1", answer="feedback", text="also return 2"),
        ],
        [answer("looked"), _crash],
    )
    _crashed(rig)

    rig, result = _resumed(
        root,
        [_a("review", "a4", task="T1", answer="drop"), answer("dropped")],
        [answer("second round")],
    )
    assert (result.outcome, result.why) == ("PARTIAL", "no task was accepted")
    [reviewed] = [m for m in _last_tool(rig) if m[0] == "a3"]
    assert str(reviewed[1]).startswith("T1 · round 2\nREPORT (worker): second round")
    last = rig.worker_rung.requests[0].messages
    assert (last[-1].role, last[-1].content) == ("user", "also return 2")


@pytest.mark.parametrize("landed", [True, False], ids=["landed", "not_landed"])
def test_an_accept_whose_commit_did_not_complete_is_found_or_made_never_twice(
    root: Path, monkeypatch: pytest.MonkeyPatch, landed: bool
) -> None:
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Return 2", brief=BRIEF),
            _a("review", "a3", task="T1", answer="accept"),
        ],
        [calls(EDIT), answer("returned 2")],
    )
    real = commit_module.Projector.commit

    def dies(self: Any, *args: Any, **kwargs: Any) -> str:
        if landed:
            real(self, *args, **kwargs)
        raise KilledMidCall

    monkeypatch.setattr(commit_module.Projector, "commit", dies)
    _crashed(rig)
    monkeypatch.undo()
    assert len(_commits(root, "T1")) == (1 if landed else 0)

    rig, result = _resumed(root, [answer("done")], [])
    assert result.outcome == "COMPLETED"
    [sha] = _commits(root, "T1")
    op = rig.journal.projection().operations["task-commit:T1:1"]
    assert op.completed is not None
    assert (op.completed.outcome, op.completed.post_state_sha) == ("succeeded", sha)
    entry = TaskBook(rig.journal.projection()).get("T1")
    assert entry is not None and (entry.state, entry.commit) == ("committed", sha)


def test_a_committed_task_is_not_committed_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Return 2", brief=BRIEF),
            _a("review", "a3", task="T1", answer="accept"),
        ],
        [calls(EDIT), answer("returned 2")],
    )
    monkeypatch.setattr(TaskDesk, "_committed", _crash)
    _crashed(rig)
    monkeypatch.undo()

    rig, result = _resumed(root, [answer("done")], [])
    assert result.outcome == "COMPLETED"
    [sha] = _commits(root, "T1")
    [reviewed] = [m for m in _last_tool(rig) if m[0] == "a3"]
    assert reviewed[1] == f"committed T1 as {sha[:7]}"


def test_a_drop_whose_restore_did_not_complete_is_restored_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = episode_rig(
        root,
        [
            _a("plan", "a1", text=PLAN),
            _a("delegate", "a2", title="Return 2", brief=BRIEF),
            _a("review", "a3", task="T1", answer="drop"),
        ],
        [calls(EDIT), answer("returned 2")],
    )
    monkeypatch.setattr(desk_module, "restore_tree", _crash)
    _crashed(rig)
    monkeypatch.undo()
    assert "return 2" in (root / "app.py").read_text()

    rig, result = _resumed(root, [answer("dropped")], [])
    assert (result.outcome, result.why) == ("PARTIAL", "no task was accepted")
    assert (root / "app.py").read_text() == APP
    run = rig.journal.projection()
    [restore] = [op for op in run.operations.values() if op.surface == ("review",)]
    assert restore.kind == EffectKind.WORKER_STEP.value
    assert restore.completed is not None and restore.completed.outcome == "succeeded"
    assert TaskBook(run).get("T1").state == "abandoned"  # type: ignore[union-attr]


def test_a_cancel_at_a_stop_asked_again_on_resume_ends_her_tool_use(root: Path) -> None:
    # The crash came while stop 2 was put to the person; on resume her plan call is dispatched
    # again, they cancel, and every call she makes after is answered with the end's words.
    rig = episode_rig(root, [_a("plan", "a1", text=PLAN)], [], stop=_crash)
    _crashed(rig)

    cancel = StopOutcome(StopAnswer.CANCEL, PLAN, sha256_hex(PLAN), "not today")
    rig = episode_rig(
        root,
        [_a("read", "a2", path="app.py"), answer("cancelled; nothing ran")],
        [],
        resume=True,
        stop=lambda _s: cancel,
    )
    result = _loop(rig).resume()
    assert (result.outcome, result.why) == (
        "ABORTED",
        "the user cancelled the plan: not today",
    )
    after = rig.athena_rung.requests[-1].messages[-1]
    assert (after.tool_call_id, after.content) == (
        "a2",
        "your tool use has ended; reply with your report now",
    )
    tools = [p.payload.tool for p in rig.journal.projection().lines(ToolCallP)]
    assert "read" not in tools, "a call after the end is answered, never dispatched"

    # And after another restart the episode is still the person's stop.
    rig.journal.close()
    rig, again = _resumed(root, [], [])
    assert again.outcome == "ABORTED" and rig.athena_rung.requests == []


# -- a pause, and a one-task plan --------------------------------------------------------------


def test_a_pause_at_a_task_boundary_resumes(root: Path) -> None:
    pause = threading.Event()
    pause.set()
    rig = episode_rig(
        root,
        [_a("plan", "a1", text=PLAN), _a("delegate", "a2", title="Return 2", brief=BRIEF)],
        [],
        pause=pause,
    )
    result = _loop(rig).run()
    assert result == LoopResult(
        "PARTIAL", "paused by the user at a task boundary", None, paused=True
    )
    assert rig.journal.projection().lines(TaskDelegatedP) == ()
    rig.journal.close()

    rig, result = _resumed(
        root,
        [
            _a("delegate", "a3", title="Return 2", brief=BRIEF),
            _a("review", "a4", task="T1", answer="accept"),
            answer("done after the pause"),
        ],
        [calls(EDIT), answer("returned 2")],
    )
    assert result == LoopResult("COMPLETED", None, "done after the pause")
    assert len(_commits(root, "T1")) == 1


def test_a_one_task_plan_cut_off_in_its_first_round_continues_it(root: Path) -> None:
    work = {"one_task": OneTaskPlan("Return 2", BRIEF)}
    rig = episode_rig(root, [], [calls(EDIT), _crash])
    _crashed(rig, **work)
    cut = rig.worker_rung.requests[-1].messages

    rig, result = _resumed(
        root,
        [_a("review", "a1", task="T1", answer="accept"), answer("done")],
        [answer("returned 2")],
        **work,
    )
    assert result.outcome == "COMPLETED"
    assert rig.worker_rung.requests[0].messages == cut
    assert len(rig.journal.projection().lines(TaskDelegatedP)) == 1
    assert "REPORT (worker): returned 2" in str(rig.athena_rung.requests[0].messages[1].content)
