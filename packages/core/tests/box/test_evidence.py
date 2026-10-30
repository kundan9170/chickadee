"""Evidence at handoff, stage A, in a worker's round.

What these pin: a worker that edits and says "done" with no test run is sent back once with the
gap named, runs the tests, and stops; Athena's line says the tests ran after the last change; a
round the progress watchdog asked to stop and report is never sent back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from box_helpers import delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from judging import Judging
from repos import init_repo

from chickadee.graph.diagnostics import Diagnostic
from chickadee.graph.indexer import Indexer
from chickadee.graph.queries import Graph
from chickadee.hooks.builtins import bind_builtin
from chickadee.hooks.fire import HookSet, RoleHooks
from chickadee.hooks.payloads import RoundEnded
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.steer import harness_notes, round_ended

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


def _spec(event: str, use: str, role: str, with_: dict[str, object] | None = None) -> HookSpec:
    return HookSpec.model_validate(
        {"event": event, "name": use, "use": use, "role": role, "with": with_ or {}}
    )


def _evidence(*watchdog: str) -> HookSet:
    bound = [
        bind_builtin(_spec("before_stop", "evidence_check", "worker")),
        bind_builtin(_spec("round_ended", "evidence_report", "athena")),
    ]
    for use, event in zip(watchdog, ("after_batch", "round_ended"), strict=False):
        role = "worker" if event == "after_batch" else "athena"
        bound.append(bind_builtin(_spec(event, use, role, {"looping": 4})))
    return HookSet(tuple(bound))


def _user_texts(request: Any) -> list[str]:
    return [
        m.content for m in request.messages if m.role == "user" and isinstance(m.content, str)
    ]


def test_done_with_no_test_run_is_sent_back_once_then_stops(root: Path) -> None:
    worker, journal, rung = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
            answer("Done."),
            calls(call("shell", "c2", command="pytest -q")),
            answer("Done; 1 passed."),
        ],
        stdout=b"1 passed in 0.1s\n",
        hooks=_evidence(),
    )
    report = worker.start("T1", "Return 2.", delegate(journal, root))

    assert report.report == "Done; 1 passed."
    [sent] = [t for t in _user_texts(rung.requests[2]) if t.startswith("[evidence]")]
    assert "- you changed app.py and ran no tests in this task" in sent
    athena = RoleHooks(worker.hooks.hooks, "athena", run_id=journal.run_id)
    ended = RoundEnded(
        task="T1",
        round=1,
        stopped=report.stopped,
        report=report.report,
        facts=report.facts,
        progress=worker.progress("T1", 1),
    )
    assert round_ended(athena, ended, room=True) is None
    [line] = harness_notes(athena, task="T1", round_=1)
    assert line.startswith(
        "Evidence (harness): tests after the last change: yes, the last run passed: 1 test(s) "
        "passed"
    )


def _stuck(root: Path, hooks: HookSet) -> tuple[str | None, list[str]]:
    """An edit, eight refused calls, then `finish`: what the worker reported, and every
    evidence send-back it was given."""
    nope = [calls(call("nope", f"n{i}", why=str(i))) for i in range(1, 9)]
    worker, journal, rung = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
            *nope,
            calls(call("finish", "f", report="stuck")),
            answer("stuck, still"),
            answer("stuck, still"),
        ],
        hooks=hooks,
    )
    report = worker.start("T1", "Return 2.", delegate(journal, root))
    sent = [t for r in rung.requests for t in _user_texts(r) if t.startswith("[evidence]")]
    return report.report, sent


def test_a_round_the_watchdog_asked_to_report_is_never_sent_back(root: Path) -> None:
    reported, sent = _stuck(root, _evidence("progress_watchdog", "progress_report"))
    assert (reported, sent) == ("stuck", [])


def test_the_same_round_unwatched_is_sent_back(tmp_path: Path) -> None:
    """The control: without the watchdog's request, the same stop is sent back."""
    reported, sent = _stuck(init_repo(tmp_path / "other", {"app.py": APP}), _evidence())
    assert len(sent) == 1 and reported == "stuck, still"


def _problem(line: int, message: str) -> Diagnostic:
    return Diagnostic(line, 1, "error", "reportCallIssue", message, "fake")


def test_an_error_the_change_brought_in_sends_it_back_and_one_already_there_is_apart(
    root: Path,
) -> None:
    """D237 B1 in a round: the index keeps the task's start before its first edit is judged,
    and the stop reads the error the edit brought in apart from the one already there."""
    judge = Judging(root, {"app.py": (_problem(1, "already there"),)})
    indexer = Indexer(root, judge)
    indexer.full_index()
    judge.problems["app.py"] = (_problem(1, "already there"), _problem(2, "brought in"))
    worker, journal, rung = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
            calls(call("shell", "c2", command="pytest -q")),
            answer("Done."),
            answer("Done; the error at line 2 is not mine."),
        ],
        stdout=b"1 passed in 0.1s\n",
        graph=Graph(root, indexer.conn),
        indexer=indexer,
        hooks=_evidence(),
    )
    report = worker.start("T1", "Return 2.", delegate(journal, root))

    assert report.report == "Done; the error at line 2 is not mine."
    [sent] = [t for t in _user_texts(rung.requests[3]) if t.startswith("[evidence]")]
    assert "- errors your changes brought in: app.py:2 fake reportCallIssue brought in" in sent
    assert "never counted against it: fix them only if you can.\n- app.py:1 fake" in sent
    progress = worker.progress("T1", 1)
    assert progress is not None and progress.diagnostics is not None
    assert progress.diagnostics.already == ("app.py:1 fake reportCallIssue already there",)
    indexer.close()
