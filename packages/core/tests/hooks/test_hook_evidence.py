"""Evidence at handoff, stage A: pure, over a stop's payload.

What these pin: a worker that changed files after its last test run, ran none in the task, or
whose last run showed nothing is sent back once, its last test command named; so is one whose
tests the task broke or left failing; never when no runner is there, its tools are closed, the
watchdog asked it to report, or the record backs it; each check can be switched off; Athena is
given one line a round; the report must share the check's `with`.
"""

from __future__ import annotations

import pytest

from chickadee.errors import ConfigError
from chickadee.hooks.builtins import bind_builtin, check_shared
from chickadee.hooks.evidence import EvidenceParams, evidence_check, evidence_report
from chickadee.hooks.payloads import (
    BeforeStop,
    DiagnosticsSeen,
    HookEvent,
    ProgressCall,
    RoundEnded,
    TestRunSeen,
    TestsSeen,
    WorkerProgress,
)
from chickadee.hooks.points import HookPoint
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import Continue, Note, Proceed
from chickadee.schema.journal import CommandFactP, FileFactP, TaskFactsP

PASSED = TestRunSeen(outcome="passed", how="summary", passed=12)


def _edit(response: int, file: str = "app.py") -> ProgressCall:
    return ProgressCall(
        response=response,
        tool="edit",
        tag="changed",
        key=f"e{response}",
        about="edit",
        changed=(file,),
    )


def _run(response: int, run: TestRunSeen = PASSED) -> ProgressCall:
    return ProgressCall(
        response=response,
        tool="shell",
        tag="learned",
        key=f"t{response}",
        about="shell `pytest -q`",
        run=run,
    )


def _facts(*, tested: bool = True) -> TaskFactsP:
    command = CommandFactP(
        operation_id="op",
        command="pytest -q",
        exit_code=0,
        timed_out=False,
        tail="",
        test_outcome="passed" if tested else None,
    )
    return TaskFactsP(
        start_tree="a",
        end_tree="b",
        files=(FileFactP(path="app.py", kind="edited", added=1, removed=1),),
        commands=(command,),
        tests_edited=("tests/test_app.py",),
    )


def _stop(
    *calls: ProgressCall,
    runnable: bool = True,
    tests: TestsSeen | None = None,
    facts: TaskFactsP | None = None,
    diagnostics: DiagnosticsSeen | None = None,
    **kw: object,
) -> HookEvent:
    progress = WorkerProgress(
        calls=calls, runnable=runnable, tests=tests, diagnostics=diagnostics
    )
    stop = BeforeStop.model_validate(
        {"reply": "Done.", "progress": progress, "facts": facts or _facts(), **kw}
    )
    return HookEvent(HookPoint.BEFORE_STOP, "worker", stop)


CHECK = evidence_check(EvidenceParams())


def _said(event: HookEvent) -> str:
    verdict = CHECK(event)
    assert isinstance(verdict, Continue)
    return verdict.text


def test_changes_after_the_last_test_run_send_the_worker_back() -> None:
    text = _said(_stop(_run(1), _edit(2), _edit(3, "db.py")))
    assert text.startswith("[evidence] Not finished yet")
    assert "- you changed app.py, db.py after your last test run" in text
    assert "Your last test run was shell `pytest -q`." in text
    assert "say why in your report and finish" in text


def test_no_test_run_in_the_task_sends_it_back() -> None:
    text = _said(_stop(_edit(1), facts=_facts(tested=False)))
    assert "- you changed app.py and ran no tests in this task" in text
    assert "Your last test run" not in text


def test_a_last_run_that_showed_nothing_sends_it_back() -> None:
    partial = TestRunSeen(outcome="partial", how="partial")
    text = _said(_stop(_edit(1), _run(2, partial)))
    assert "your last test run's result was not shown" in text


def test_the_task_s_failing_tests_send_it_back_and_old_failures_do_not() -> None:
    tests = TestsSeen(broken=("t::route",), failing=("t::seg",), already_failing=("t::db",))
    text = _said(_stop(_edit(1), _run(2), tests=tests))
    assert "- tests that passed earlier in the task now fail: t::route" in text
    assert "- the task's tests were last seen failing: t::seg" in text
    assert "t::db" not in text
    only_old = TestsSeen(already_failing=("t::db",))
    assert isinstance(CHECK(_stop(_edit(1), _run(2), tests=only_old)), Proceed)


@pytest.mark.parametrize(
    "event",
    [
        _stop(_edit(1), _run(2)),
        _stop(_run(1), _edit(2), runnable=False),
        _stop(_run(1), _edit(2), tools_open=False),
        _stop(_run(1), _edit(2), handed_up=True),
        _stop(),
    ],
    ids=["backed", "no-runner", "tools-closed", "handed-up", "nothing-done"],
)
def test_it_proceeds_where_the_record_backs_the_stop_or_a_send_back_means_nothing(
    event: HookEvent,
) -> None:
    assert isinstance(CHECK(event), Proceed)


def test_each_check_can_be_switched_off() -> None:
    tests = TestsSeen(failing=("t::seg",))
    untested = evidence_check(EvidenceParams(untested=False))
    assert isinstance(untested(_stop(_run(1), _edit(2))), Proceed)
    failing = evidence_check(EvidenceParams(failing=False))
    assert isinstance(failing(_stop(_edit(1), _run(2), tests=tests)), Proceed)


def _ended(progress: WorkerProgress) -> HookEvent:
    ended = RoundEnded(
        task="T1", round=1, stopped="replied", report="done", facts=_facts(), progress=progress
    )
    return HookEvent(HookPoint.ROUND_ENDED, "athena", ended)


def test_athena_is_given_one_evidence_line_a_round() -> None:
    report = evidence_report(EvidenceParams())
    tests = TestsSeen(broken=("t::route",))
    said = report(_ended(WorkerProgress(calls=(_edit(1), _run(2)), runnable=True, tests=tests)))
    assert isinstance(said, Note)
    assert said.text == (
        "Evidence (harness): tests after the last change: yes, the last run passed: 12 test(s) "
        "passed, 0 failed (read from its summary); tests the task broke: t::route; the task's "
        "tests last seen failing: none; test files the task changed: tests/test_app.py."
    )
    stale = report(_ended(WorkerProgress(calls=(_run(1), _edit(2)), runnable=True)))
    assert isinstance(stale, Note)
    assert "tests after the last change: no, app.py changed since the last run" in stale.text


def _spec(event: str, use: str, with_: dict[str, object]) -> HookSpec:
    return HookSpec.model_validate(
        {"event": event, "name": "evidence", "use": use, "role": "worker", "with": with_}
    )


def test_the_report_shares_the_check_s_with() -> None:
    check = bind_builtin(_spec("before_stop", "evidence_check", {"failing": False}))
    report = bind_builtin(_spec("round_ended", "evidence_report", {}))
    with pytest.raises(ConfigError, match="give both the same `with`"):
        check_shared((check, report))
    with pytest.raises(ConfigError, match="fires at before_stop"):
        bind_builtin(_spec("after_batch", "evidence_check", {}))


# -- B1: errors since the task began -----------------------------------------------------------

SEEN = DiagnosticsSeen(
    new_errors=("app.py:7 pyright reportCallIssue Argument missing",),
    new_warnings=("app.py:2 pyright reportUnusedImport unused",),
    already=("db.py:3 pyright reportAttributeAccessIssue no attribute",),
    unchecked=("main.py",),
)


def test_new_errors_send_it_back_with_those_already_there_apart() -> None:
    text = _said(_stop(_edit(1), _run(2), diagnostics=SEEN))
    assert "- errors your changes brought in: app.py:7 pyright reportCallIssue" in text
    assert text.endswith(
        "Already there before your task began, never counted against it: fix them only if "
        "you can.\n- db.py:3 pyright reportAttributeAccessIssue no attribute"
    )


def test_errors_already_there_alone_never_send_it_back() -> None:
    only = DiagnosticsSeen(already=SEEN.already, new_warnings=SEEN.new_warnings)
    assert isinstance(CHECK(_stop(_edit(1), _run(2), diagnostics=only)), Proceed)
    off = evidence_check(EvidenceParams(errors=False))
    assert isinstance(off(_stop(_edit(1), _run(2), diagnostics=SEEN)), Proceed)


def test_the_line_counts_what_the_stop_measured() -> None:
    progress = WorkerProgress(calls=(_edit(1), _run(2)), runnable=True, diagnostics=SEEN)
    said = evidence_report(EvidenceParams())(_ended(progress))
    assert isinstance(said, Note)
    assert said.text.endswith(
        "new errors since the task began: 1; new warnings: 1; errors already there: 1; not "
        "re-checked: main.py."
    )
