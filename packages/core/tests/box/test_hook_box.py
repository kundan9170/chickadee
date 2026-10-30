"""The box roles' points, end to end: a hook configured as `[[hooks]]`,
bound when the run starts, and what it says reaching the role that reads next.

Each test registers a built-in of its own for the length of the test, so the whole path runs:
the config, `bind`, the episode's bound set, the fire site, the line, and the next request.

What these pin: a note at a worker's call and at its step rides on that call's answer; a
`Continue` at the worker's stop gives its round another turn, at most `hooks.max_continues`; a
`RetryTask` at a round's end is Athena's own `feedback` line written by the harness, and the
report she reads is the retried round's, at most `hooks.max_retries`; a note at a review and at
a commit is in her answer; and a note at the episode's end is in the report Zeus reads.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from box_helpers import BoxRig, design, with_hooks, worker_call
from box_helpers import athena_call as acall
from engines import answer, calls

from chickadee.box.report import CoderReport
from chickadee.hooks import builtins
from chickadee.hooks.builtins import Builtin
from chickadee.hooks.fire import HookLimits
from chickadee.hooks.payloads import HookEvent
from chickadee.hooks.points import HookPoint, spec_of
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Continue, HookVerdict, Note, Proceed, RetryTask
from chickadee.journal import paths
from chickadee.journal.projection import project
from chickadee.journal.reader import read_events
from chickadee.schema.journal import TaskReviewedP
from chickadee.state.roots import roots as resolve_roots
from chickadee.zeus.delegate import render_report

PLAN = "1. **Return 2.** `go()` in `app.py` returns 2.\n"
BRIEF = "## Task 1 of 1: return 2\n\n### Goal\n`go()` returns 2.\n"
EDIT = calls(worker_call("edit", "w1", path="app.py", old="return 1", new="return 2"))


def _athena() -> list[object]:
    return [
        calls(acall("plan", "a1", text=PLAN)),
        calls(acall("delegate", "a2", title="Return 2", brief=BRIEF)),
        calls(acall("review", "a3", task="T1", answer="accept")),
        answer("One task: go() now returns 2."),
    ]


def _hook(
    monkeypatch: pytest.MonkeyPatch,
    rig: BoxRig,
    point: HookPoint,
    run: Callable[[HookEvent], HookVerdict],
    *,
    role: str | None = None,
    limits: HookLimits | None = None,
) -> None:
    """`run` as the built-in `probe` at `point`, configured on `rig`'s episodes."""
    returns = frozenset({Proceed, Note, Continue, RetryTask}) & spec_of(point).accepts
    probe = Builtin("probe", frozenset({point}), returns, lambda params: run)
    monkeypatch.setitem(builtins.BUILTINS, "probe", probe)  # type: ignore[arg-type]
    fields: dict[str, object] = {"event": point, "name": "probe", "use": "probe"}
    if role is not None:
        fields["role"] = role
    with_hooks(rig, HookSpec.model_validate(fields), limits=limits)


def _asked(rig: BoxRig, role: str, n: int) -> str:
    return str(rig.rungs[role].requests[n].messages)


def _report(rig: BoxRig) -> CoderReport:
    report = rig.box().open(design()).report()
    assert isinstance(report, CoderReport)
    assert report.outcome == "COMPLETED", report.why
    return report


def test_a_note_at_a_workers_call_rides_on_its_answer(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hook(monkeypatch, rig, HookPoint.AFTER_TOOL, lambda e: Note("lint: clean"), role="worker")
    rig.athena_script, rig.worker_script = _athena(), [EDIT, answer("T1: returns 2.")]
    _report(rig)
    assert "lint: clean" in _asked(rig, "worker", 1)


def test_a_note_at_a_step_rides_on_the_answer_of_the_call_that_made_it(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def changed(event: HookEvent) -> HookVerdict:
        return Note(f"changed: {', '.join(event.payload.paths())}")

    _hook(monkeypatch, rig, HookPoint.AFTER_STEP, changed, role="worker")
    rig.athena_script, rig.worker_script = _athena(), [EDIT, answer("T1: returns 2.")]
    _report(rig)
    assert "changed: app.py" in _asked(rig, "worker", 1)


def test_a_continue_at_the_workers_stop_is_one_more_turn_and_bounded(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hook(
        monkeypatch,
        rig,
        HookPoint.BEFORE_STOP,
        lambda e: Continue("run the suite before you stop"),
        role="worker",
    )
    rig.athena_script = _athena()
    rig.worker_script = [EDIT, answer("done"), answer("ran it: 1 passed"), answer("never")]
    _report(rig)
    worker = rig.rungs["worker"].requests
    assert len(worker) == 3, "one continue, then the bound: the round stops"
    assert "run the suite before you stop" in _asked(rig, "worker", 2)
    assert "ran it: 1 passed" in _asked(rig, "athena", 2), "she reads the last reply"


def test_a_retry_at_a_rounds_end_is_her_feedback_line_and_bounded(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hook(
        monkeypatch,
        rig,
        HookPoint.ROUND_ENDED,
        lambda e: RetryTask("again, and say what you ran", fresh=False),
    )
    rig.athena_script = _athena()
    rig.worker_script = [EDIT, answer("T1 round 1"), answer("T1 round 2: ran pytest")]
    report = _report(rig)
    assert "again, and say what you ran" in _asked(rig, "worker", 2)
    assert "T1 round 2: ran pytest" in _asked(rig, "athena", 2)
    assert report.tasks[0].rounds == 2, "one retry: `hooks.max_retries` is 1"
    run_id = report.run_id or ""
    run = project(
        read_events(paths.journal_path(resolve_roots(rig.workspaces[0].root), run_id))
    )
    [retry, accepted] = [p.payload for p in run.lines(TaskReviewedP)]
    assert (retry.answer, retry.by) == ("feedback", "hook:probe")
    assert (accepted.answer, accepted.by) == ("accept", None)


@pytest.mark.parametrize(
    ("point", "words"),
    [
        (HookPoint.TASK_REVIEWED, "reviewed: accept"),
        (HookPoint.TASK_COMMITTED, "committed: T1"),
    ],
)
def test_a_note_at_a_review_or_a_commit_is_in_her_answer(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch, point: HookPoint, words: str
) -> None:
    def said(event: HookEvent) -> HookVerdict:
        payload = event.payload
        detail = getattr(payload, "answer", None) or getattr(payload, "task", "")
        return Note(f"{words.split(':')[0]}: {detail}")

    _hook(monkeypatch, rig, point, said)
    rig.athena_script, rig.worker_script = _athena(), [EDIT, answer("T1: returns 2.")]
    _report(rig)
    assert words in _asked(rig, "athena", 3)


def test_a_note_at_the_episodes_end_is_in_the_report_zeus_reads(
    rig: BoxRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hook(monkeypatch, rig, HookPoint.EPISODE_ENDED, lambda e: Note("open a PR next"))
    rig.athena_script, rig.worker_script = _athena(), [EDIT, answer("T1: returns 2.")]
    report = _report(rig)
    assert report.harness == ("open a PR next",)
    assert "open a PR next" in str(render_report(report))
