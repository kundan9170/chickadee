"""One episode through the box:
`Box.open` over a real repository, the task loop composed by `box/compose.compose_episode`, and
scripted Athena and worker models.

What these pin: a design runs as D196 §4 has it, its run lines are contracts v2's (the approved
work's sha and the templates at the start; the outcome, why and the counts at the end; never
`SOLVED`), and the report v2 says each task, Athena's words and what changed. A one-task plan
runs its worker before Athena. A cancel at stop 2 is the person's `ABORTED`. A pause is not a
report and resumes. A harness error is a `BLOCKED` report over a journal left for resume. A run
opened before contracts v2 is refused at resume, naming D196.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from box_helpers import BoxRig, design, lease_is_free, one_task, worker_call
from box_helpers import athena_call as acall
from engines import answer, calls
from repos import git

from chickadee.box.episode import Paused
from chickadee.box.loop import TaskLoop
from chickadee.box.report import CoderReport
from chickadee.engine import ScriptedInteraction
from chickadee.journal import paths
from chickadee.journal.errors import JournalCorrupt
from chickadee.journal.projection import RunProjection, project
from chickadee.journal.reader import read_events
from chickadee.roles import TASK_WORKER
from chickadee.schema import SCHEMA_VERSION, sha256_hex
from chickadee.schema.interaction import StopAnswer, StopDecision
from chickadee.schema.journal import PlanWrittenP, RunEndP, RunStartP, TaskDelegatedP
from chickadee.shell import ShellClaim
from chickadee.state.roots import roots as resolve_roots
from chickadee.workspace import Workspace

PLAN = "1. **Return 2.** `go()` in `app.py` returns 2.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
REPORT = "One task: go() now returns 2; the suite printed 1 passed."


def _athena(*review: object) -> list[object]:
    return [
        calls(acall("plan", "a1", text=PLAN)),
        calls(acall("delegate", "a2", title="Return 2", brief=BRIEF)),
        *(review or (calls(acall("review", "a3", task="T1", answer="accept")),)),
        calls(acall("shell", "a4", command="pytest -q")),
        answer(REPORT),
    ]


WORKER = [
    calls(worker_call("edit", "w1", path="app.py", old="return 1", new="return 2")),
    answer("T1: go() returns 2."),
]


def _run(ws: Workspace, run_id: str) -> RunProjection:
    return project(read_events(paths.journal_path(resolve_roots(ws.root), run_id)))


def test_a_design_runs_end_to_end_and_its_run_lines_are_contracts_v2(rig: BoxRig) -> None:
    rig.athena_script, rig.worker_script = _athena(), WORKER
    request = design()
    report = rig.box().open(request).report()

    assert isinstance(report, CoderReport)
    assert (report.outcome, report.why, report.athena) == ("COMPLETED", None, REPORT)
    [task] = report.tasks
    assert (task.id, task.title, task.state, task.answer) == (
        "T1",
        "Return 2",
        "committed",
        "accept",
    )
    assert task.commit is not None and task.facts is not None
    assert [(f.path, f.kind) for f in task.facts.files] == [("app.py", "edited")]
    assert report.change_summary.files == ("app.py",)
    ws = rig.workspaces[0]
    assert (ws.root / "app.py").read_text(
        encoding="utf-8"
    ) == "def go() -> int:\n    return 2\n"
    assert git(ws.root, "log", "-1", "--format=%s").strip() == "chickadee: T1 Return 2"
    assert lease_is_free(rig.source)

    run = _run(ws, report.run_id or "")
    start = run.first(RunStartP)
    assert start is not None
    assert (start.goal, start.work_sha) == ("Make go return 2", request.work_sha)
    assert start.schema_version == SCHEMA_VERSION == 2
    assert set(start.template_shas) == {"design", "plan", "brief"}
    assert start.template_shas["plan"] == sha256_hex("PLAN TEMPLATE")
    end = run.first(RunEndP)
    assert end is not None
    assert (end.outcome, end.why, end.tasks, end.committed) == ("COMPLETED", None, 1, 1)
    # The report the box returned is stored, and `run_end` names it.
    assert end.handoff_path is not None
    stored = json.loads(
        (paths.run_dir(resolve_roots(ws.root), run.run_id) / end.handoff_path).read_text()
    )
    assert stored["outcome"] == "COMPLETED" and stored["athena"] == REPORT
    # Every line is contracts v2's: no packet field, the envelope's version 2.
    lines = paths.journal_path(resolve_roots(ws.root), run.run_id).read_text().splitlines()
    assert all(json.loads(line)["v"] == 2 for line in lines)
    assert not any("packet_id" in line or "SOLVED" in line for line in lines)


def test_a_one_task_plan_runs_its_worker_before_athena(rig: BoxRig) -> None:
    rig.athena_script = [
        calls(acall("review", "a1", task="T1", answer="accept")),
        answer("Done."),
    ]
    rig.worker_script = WORKER
    request = one_task("Return 2", BRIEF)
    report = rig.box().open(request).report()

    assert isinstance(report, CoderReport) and report.outcome == "COMPLETED"
    run = _run(rig.workspaces[0], report.run_id or "")
    [plan] = run.lines(PlanWrittenP)
    assert (plan.payload.by, plan.payload.sha) == ("zeus", request.work_sha)
    [delegated] = run.lines(TaskDelegatedP)
    assert delegated.payload.by == "zeus"
    start = run.first(RunStartP)
    assert start is not None and start.template_shas["brief"] == request.task.template_sha  # type: ignore[union-attr]
    # Athena's first request already holds the worker's first round.
    athena = rig.rungs["athena"]
    assert "T1: go() returns 2." in str(athena.requests[0].messages[1].content)


def test_a_cancel_at_stop_2_is_the_persons_stop(rig: BoxRig) -> None:
    rig.interaction = ScriptedInteraction(
        stops=[StopDecision(answer=StopAnswer.CANCEL, note="not this week")]
    )
    rig.athena_script = [calls(acall("plan", "a1", text=PLAN)), answer("Stopped.")]
    report = rig.box().open(design()).report()

    assert isinstance(report, CoderReport) and report.outcome == "ABORTED"
    assert report.why is not None and "not this week" in report.why
    end = _run(rig.workspaces[0], report.run_id or "").first(RunEndP)
    assert end is not None and (end.outcome, end.tasks, end.committed) == ("ABORTED", 0, 0)


def test_a_pause_is_not_a_report_and_resumes_to_one(rig: BoxRig) -> None:
    rig.athena_script, rig.worker_script = _athena(), WORKER
    rig.cancel.set()
    paused = rig.box().open(design()).report()
    assert isinstance(paused, Paused)
    assert lease_is_free(rig.source)  # a pause gives the lease back
    first = rig.workspaces[0]
    assert _run(first, paused.run_id).first(RunEndP) is None  # not ended: resumable

    rig.cancel.clear()
    run = rig.box().resume(paused.run_id)
    report = run.report()
    assert isinstance(report, CoderReport)
    assert report.outcome == "COMPLETED", report.why
    assert report.run_id == paused.run_id and rig.workspaces[1].root == first.root
    assert report.change_summary.files == ("app.py",)
    assert lease_is_free(rig.source)


class _Leaving:
    """The person at a command's permission question: they leave while it is asked (the engine
    pauses the run), or, once back, approve it."""

    def __init__(self, rig: BoxRig) -> None:
        self.rig = rig
        self.leaves = True
        self.asked: list[str] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.asked.append(claim.operation_id)
        if not self.leaves:
            return True
        # As `BoxEpisode.suspend` stops it: the pause, then the hard stop the call waits on.
        self.rig.suspend.set()
        self.rig.abort.set()
        assert cancel is self.rig.abort, "the ask is handed the call's stop"
        return False


def test_leaving_at_a_permission_question_pauses_the_run_and_resume_asks_it_again(
    rig: BoxRig,
) -> None:
    """The run left at the question is not ended, its step is left as a kill leaves it,
    and resume puts the same question to the person again rather than answering it."""
    person = _Leaving(rig)
    rig.authority = person
    rig.athena_script = [calls(acall("shell", "a1", command="uv sync"))]
    paused = rig.box().open(design()).report()

    assert isinstance(paused, Paused)
    assert lease_is_free(rig.source)
    run = _run(rig.workspaces[0], paused.run_id)
    assert run.first(RunEndP) is None, "not ended: resumable"
    [step] = run.operations.values()
    assert step.started is not None and step.completed is None, "left as a kill leaves it"

    rig.suspend.clear()
    rig.abort.clear()
    person.leaves = False
    rig.asking = lambda op: op == step.operation_id  # nobody answered it
    rig.athena_script = [answer("Synced.")]
    report = rig.box().resume(paused.run_id).report()

    assert isinstance(report, CoderReport)
    assert (report.outcome, report.why) == ("PARTIAL", "no plan was approved")
    assert len(person.asked) == 2, "asked again, never answered for"
    assert "1 passed" in str(rig.rungs["athena"].requests[0].messages[-1].content)


def test_a_harness_error_is_a_blocked_report_over_a_journal_left_for_resume(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> TaskLoop:
        raise RuntimeError("the loop could not be composed")

    monkeypatch.setattr("chickadee.box.episode.compose_episode", broken)
    report = rig.box().open(design()).report()

    assert isinstance(report, CoderReport) and report.outcome == "BLOCKED"
    assert report.why is not None and "the loop could not be composed" in report.why
    assert report.handles is not None  # its journal is named
    run = _run(rig.workspaces[0], report.run_id or "")
    assert run.first(RunStartP) is not None and run.first(RunEndP) is None
    assert lease_is_free(rig.source)


def test_an_error_raised_as_the_pause_tears_the_run_down_is_a_pause(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The person left; the harness did not fail. Never a BLOCKED report to Zeus."""

    def torn(*args: object, **kwargs: object) -> TaskLoop:
        rig.suspend.set()
        raise OSError("the journal closed under the pause")

    monkeypatch.setattr("chickadee.box.episode.compose_episode", torn)
    paused = rig.box().open(design()).report()

    assert isinstance(paused, Paused)
    assert _run(rig.workspaces[0], paused.run_id).first(RunEndP) is None
    assert lease_is_free(rig.source)


def test_a_run_opened_before_contracts_v2_is_refused(rig: BoxRig) -> None:
    rig.athena_script, rig.worker_script = _athena(), WORKER
    rig.cancel.set()
    paused = rig.box().open(design()).report()
    assert isinstance(paused, Paused)
    roots = resolve_roots(rig.workspaces[0].root)
    # The packet loop's request, as its config snapshot held it.
    config = paths.run_dir(roots, paused.run_id) / paths.CONFIG_NAME
    config.write_text(json.dumps({"box": {"request": {"task": "t", "approval": {}}}}))
    rig.cancel.clear()
    with pytest.raises(JournalCorrupt, match="D196"):
        rig.box().resume(paused.run_id)
    assert lease_is_free(rig.source)


def test_a_damaged_v2_snapshot_is_said_as_damage_not_as_an_old_run(rig: BoxRig) -> None:
    """T-3.157 review: a v2 request that does not validate is damage, named with its error,
    never sent to the `pre-d196` tag, which cannot resume it."""
    rig.athena_script, rig.worker_script = _athena(), WORKER
    rig.cancel.set()
    paused = rig.box().open(design()).report()
    assert isinstance(paused, Paused)
    roots = resolve_roots(rig.workspaces[0].root)
    config = paths.run_dir(roots, paused.run_id) / paths.CONFIG_NAME
    body = json.loads(config.read_text())
    body["box"]["request"]["design"]["sha"] = "truncated"
    config.write_text(json.dumps(body))
    rig.cancel.clear()
    with pytest.raises(JournalCorrupt, match="does not validate") as exc:
        rig.box().resume(paused.run_id)
    assert "pre-d196" not in str(exc.value)
    assert lease_is_free(rig.source)


def test_a_journal_line_before_contracts_v2_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps(
            {
                "v": 1,
                "run_id": "2026-10-15-0a1b2c3d",
                "seq": 1,
                "ts": "2026-10-15T00:00:00Z",
                "kind": "budget",
                "payload": {},
            }
        )
        + "\n"
    )
    with pytest.raises(JournalCorrupt, match="D196"):
        list(read_events(path))


def test_a_configured_line_tool_and_grammar_reach_the_worker(rig: BoxRig) -> None:
    """A config that offers `replace_lines` gives the worker's edit port its anchor style, and
    one that constrains `apply_patch` reaches the tool's context: before the
    composition passed either, the first line edit raised."""
    rig.worker_policy = replace(
        TASK_WORKER,
        enabled=(*TASK_WORKER.enabled, "replace_lines", "apply_patch"),
        constrain=("apply_patch",),
    )
    rig.athena_script = _athena()
    diff = "--- a/app.py\n+++ b/app.py\n@@ -2 +2 @@\n-    return 1\n+    return 3\n"
    rig.worker_script = [
        calls(worker_call("apply_patch", "w0", patch=diff)),
        calls(
            worker_call(
                "replace_lines", "w1", path="app.py", start=2, end=2, content="    return 2"
            )
        ),
        answer("T1: go() returns 2."),
    ]
    report = rig.box().open(design()).report()

    assert isinstance(report, CoderReport)
    assert report.outcome == "COMPLETED", report.why
    assert (rig.workspaces[0].root / "app.py").read_text(encoding="utf-8") == (
        "def go() -> int:\n    return 2\n"
    )
    # Constrained to its grammar, `apply_patch` takes V4A alone: the unified diff is refused.
    worker = rig.rungs["worker"]
    assert "no V4A section" in str(worker.requests[1].messages[-1].content)


def test_a_one_task_plan_paused_before_it_starts_pauses_and_resumes(rig: BoxRig) -> None:
    """A pause before the one task is delegated is taken at once, as a design's first
    `delegate` takes it; before T-3.157 the run went on to complete."""
    rig.athena_script = [
        calls(acall("review", "a1", task="T1", answer="accept")),
        answer("Done."),
    ]
    rig.worker_script = WORKER
    rig.cancel.set()
    paused = rig.box().open(one_task("Return 2", BRIEF)).report()
    assert isinstance(paused, Paused)
    assert _run(rig.workspaces[0], paused.run_id).lines(TaskDelegatedP) == ()
    rig.cancel.clear()
    report = rig.box().resume(paused.run_id).report()
    assert isinstance(report, CoderReport)
    assert report.outcome == "COMPLETED", report.why


def test_a_worker_whose_policy_offers_no_shell_is_told_nothing_runs(rig: BoxRig) -> None:
    """Review: the prompt's shell lines were chosen by whether a sandbox exists, not by what
    the worker's policy offers, so a config without `shell` told it to run tests by shell."""
    rig.worker_policy = replace(
        TASK_WORKER, enabled=tuple(n for n in TASK_WORKER.enabled if n != "shell")
    )
    rig.athena_script = _athena()
    rig.worker_script = [answer("T1: nothing to run.")]
    rig.box().open(design()).report()
    system = str(rig.rungs["worker"].requests[0].messages[0].content)
    assert "No command can run" in system and "`shell`" not in system
