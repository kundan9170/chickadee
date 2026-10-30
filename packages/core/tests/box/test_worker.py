"""The task worker's rounds: `box/worker.WorkerRun`.

What these pin: a round runs the worker's calls over the worktree, every writing call one
`worker_step`, and ends with its reply as the report, kept whole; its `task_reported` line
carries why it stopped and the facts measured from the trees; the reviewer's feedback is the
next user message of the same conversation; the call cap and the deadline end a round as
`limit`, a repetition as `looped`; a person's stop raises, the task left open.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from box_helpers import WORKER_LIMITS, delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo

from chickadee.box.worker import WorkerStopped
from chickadee.journal.effects import EffectKind
from chickadee.schema.journal import (
    BoxMessageP,
    TaskReportedP,
)

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP, "tests/test_app.py": "x = 1\n"})


def test_a_round_edits_runs_and_reports_in_its_own_words(root: Path) -> None:
    worker, journal, _ = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
            calls(call("create", "c2", path="tests/test_two.py", content="y = 2\n")),
            calls(call("shell", "c3", command="pytest -q")),
            answer("Changed go to return 2 and added a test.\nNOT: nothing parsed"),
        ],
    )
    start = delegate(journal, root)
    report = worker.start("T1", "Make go return 2.", start)

    assert (report.stopped, report.round) == ("replied", 1)
    assert report.report == "Changed go to return 2 and added a test.\nNOT: nothing parsed"
    assert "return 2" in (root / "app.py").read_text(encoding="utf-8")
    facts = report.facts
    assert (facts.start_tree, facts.end_tree != start) == (start, True)
    assert {(f.path, f.kind) for f in facts.files} == {
        ("app.py", "edited"),
        ("tests/test_two.py", "created"),
    }
    assert facts.tests_added == ("tests/test_two.py",)
    [command] = facts.commands
    assert (command.command, command.exit_code) == ("pytest -q", 0)
    assert command.test_outcome is not None

    run = journal.projection()
    steps = [
        op.intended
        for op in run.operations.values()
        if op.kind == EffectKind.WORKER_STEP.value and op.intended is not None
    ]
    assert all(op.task_id == "T1" for op in run.operations.values())
    assert [s.declared_surface for s in steps] == [("edit",), ("create",), ("shell",)]
    [reported] = run.lines(TaskReportedP)
    assert reported.task_id == "T1" and reported.payload.facts == facts
    assert journal.get_blob(reported.payload.report_sha or "") == report.report.encode()  # type: ignore[union-attr]
    assert worker.task is None


def test_the_head_is_the_instructions_then_the_brief_and_each_line_is_the_tasks(
    root: Path,
) -> None:
    worker, journal, rung = worker_rig(root, [answer("nothing to do")])
    worker.start("T1", "Make go return 2.", delegate(journal, root))
    [sent] = rung.requests
    system, user = sent.messages[:2]
    assert (system.role, system.content) == ("system", "You are the worker.")
    assert "Make go return 2." in str(user.content)
    names = {t.name for t in sent.tools}
    assert {
        "edit",
        "create",
        "delete",
        "move",
        "shell",
        "read",
        "tests_for",
        "grep",
        "glob",
    } <= names
    lines = journal.projection().lines(BoxMessageP)
    assert [p.payload.side for p in lines] == ["system", "user", "assistant"]
    assert all((p.task_id, p.round) == ("T1", 1) for p in lines)


def test_the_worker_lists_files_by_name_with_glob(root: Path) -> None:
    """Without a shell the worker had no way to list files. `glob` is its."""
    worker, journal, _ = worker_rig(
        root,
        [calls(call("glob", "c1", pattern="**/*.py")), answer("listed")],
    )
    worker.start("T1", "List the code.", delegate(journal, root))
    [line] = [p for p in journal.projection().lines(BoxMessageP) if p.payload.side == "tool"]
    shown = journal.get_blob(line.payload.sha).decode()
    assert "app.py" in shown and "tests/test_app.py" in shown


def test_feedback_is_the_next_message_of_the_same_conversation(root: Path) -> None:
    worker, journal, rung = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
            answer("done"),
            answer("added the test"),
        ],
    )
    first = worker.start("T1", "Make go return 2.", delegate(journal, root))
    second = worker.cont("T1", "Add a test for it.", first.round)
    assert (second.round, second.stopped, second.report) == (2, "replied", "added the test")
    first_round, last = rung.requests[1].messages, rung.requests[-1].messages
    # The same conversation, grown: the first round's messages, its reply, then the feedback.
    assert last[: len(first_round)] == first_round
    assert [(m.role, m.content) for m in last[len(first_round) :]] == [
        ("assistant", "done"),
        ("user", "Add a test for it."),
    ]
    # The facts are the task's, measured from its start: the first round's edit still counts.
    assert [f.path for f in second.facts.files] == ["app.py"]


def test_the_call_cap_ends_a_round_as_limit(root: Path) -> None:
    worker, journal, _ = worker_rig(
        root,
        [calls(call("read", f"c{i}", path="app.py", start=i + 1)) for i in range(3)],
        limits={**WORKER_LIMITS, "worker.max_tool_calls": 2},
    )
    report = worker.start("T1", "Look.", delegate(journal, root))
    assert report.stopped == "limit"


def test_the_same_call_repeated_ends_a_round_as_looped(root: Path) -> None:
    worker, journal, _ = worker_rig(
        root, [calls(call("read", f"c{i}", path="app.py")) for i in range(4)]
    )
    report = worker.start("T1", "Look.", delegate(journal, root))
    assert report.stopped == "looped"


def test_a_command_that_only_prints_is_answered_and_the_next_ends_the_round(
    root: Path,
) -> None:
    """T-3.159 (the owner's REPL run): the worker echoed "task completed" in a loop instead of
    replying. The first is answered, not run, saying how a task ends; the next ends the round
    as `looped`, and nothing printed ran."""
    worker, journal, rung = worker_rig(
        root,
        [
            calls(call("shell", "c1", command='echo "task completed"')),
            calls(call("shell", "c2", command="echo 'all done'")),
            answer("never asked"),
        ],
    )
    report = worker.start("T1", "Do it.", delegate(journal, root))
    assert report.stopped == "looped"
    told = str(rung.requests[1].messages[-1].content)
    assert told.startswith("not run: 'echo \"task completed\"' only prints")
    assert "call `finish` with your report" in told
    assert report.facts.commands == (), "nothing that only prints ran"


def test_a_round_past_its_deadline_ends_as_limit(root: Path) -> None:
    worker, journal, rung = worker_rig(root, [answer("late")], task_seconds=-1)
    report = worker.start("T1", "Look.", delegate(journal, root))
    assert (report.stopped, report.report, rung.requests) == ("limit", None, [])


def test_a_persons_stop_leaves_the_task_open(root: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    worker, journal, _ = worker_rig(root, [answer("never")], cancel=cancel)
    with pytest.raises(WorkerStopped):
        worker.start("T1", "Look.", delegate(journal, root))
    assert journal.projection().lines(TaskReportedP) == ()
    assert worker.task is None


def _every_call_answered(messages: object) -> bool:
    """Each tool call in a request has its result before the next turn: a provider refuses
    one without."""
    seen: set[str] = set()
    owed: set[str] = set()
    for m in messages:  # type: ignore[attr-defined]
        if m.role == "assistant":
            if owed - seen:
                return False
            owed |= {c.id for c in m.tool_calls}
        elif m.role == "tool":
            seen.add(m.tool_call_id)
        elif m.role == "user" and owed - seen:
            return False
    return not (owed - seen)


def test_feedback_after_calls_left_unanswered_answers_them_first(root: Path) -> None:
    worker, journal, rung = worker_rig(
        root,
        [
            calls(call("read", "c1", path="app.py")),
            calls(call("read", "c2", path="app.py", start=2)),
            calls(call("read", "c3", path="app.py", start=3)),
            calls(call("read", "c4", path="app.py", start=4)),
            answer("now done"),
        ],
        limits={**WORKER_LIMITS, "worker.repeat": 9, "worker.max_tool_calls": 1},
    )
    first = worker.start("T1", "Look.", delegate(journal, root))
    assert first.stopped == "limit" and first.report is None
    second = worker.cont("T1", "Go on.", first.round)
    assert second.report == "now done"
    last = rung.requests[-1].messages
    assert _every_call_answered(last)
    assert (last[-1].role, last[-1].content) == ("user", "Go on.")


def test_feedback_continues_a_task_this_process_did_not_start(root: Path) -> None:
    worker, journal, _ = worker_rig(
        root,
        [
            calls(call("edit", "c1", path="app.py", old="return 1", new="return 2")),
            answer("done"),
        ],
    )
    worker.start("T1", "Make go return 2.", delegate(journal, root))
    journal.close()
    later, journal, rung = worker_rig(root, [answer("tested")], resume=True)
    report = later.cont("T1", "Add a test.", 1)
    assert (report.round, report.report) == (2, "tested")
    assert [f.path for f in report.facts.files] == ["app.py"]
    assert rung.requests[0].messages[-2].content == "done"


def test_the_deadline_bounds_the_task_not_each_round(root: Path) -> None:
    def slow(request: object) -> object:
        del request
        time.sleep(1.2)
        return answer("slow")

    worker, journal, rung = worker_rig(root, [slow, answer("never")], task_seconds=1)
    first = worker.start("T1", "Look.", delegate(journal, root))
    assert first.stopped == "replied"
    second = worker.cont("T1", "Again.", first.round)
    assert (second.stopped, len(rung.requests)) == ("limit", 1)


def test_the_report_is_written_before_the_reply_joins_the_conversation(root: Path) -> None:
    worker, journal, _ = worker_rig(root, [answer("done")])
    worker.start("T1", "Look.", delegate(journal, root))
    lines = journal.projection().tasks["T1"]
    kinds = [type(p.payload).__name__ for p in lines]
    assert kinds[-2:] == ["TaskReportedP", "BoxMessageP"]


def test_finish_ends_the_task_with_its_report(root: Path) -> None:
    """T-3.159 (the owner's REPL run): a model that ends its turns only by calling a tool
    looped instead of replying. `finish` ends the round, its report the worker's reply, and a
    call after it in the same turn is answered, never run."""
    worker, journal, rung = worker_rig(
        root,
        [
            calls(
                call("finish", "c1", report="Changed nothing: the task was already done."),
                call("shell", "c2", command="pytest -q"),
            ),
            answer("never asked"),
        ],
    )
    report = worker.start("T1", "Do it.", delegate(journal, root))
    assert (report.stopped, report.report) == (
        "replied",
        "Changed nothing: the task was already done.",
    )
    assert len(rung.requests) == 1, "the task ended on the call"
    assert report.facts.commands == (), "the call after it never ran"


def test_a_finish_without_a_report_is_refused_and_the_task_goes_on(root: Path) -> None:
    worker, journal, rung = worker_rig(
        root, [calls(call("finish", "c1")), answer("Done: nothing to change.")]
    )
    report = worker.start("T1", "Do it.", delegate(journal, root))
    assert (report.stopped, report.report) == ("replied", "Done: nothing to change.")
    assert str(rung.requests[1].messages[-1].content).startswith("INVALID_ARGUMENTS")


def test_the_worker_may_not_create_a_file_in_the_work_folder(root: Path) -> None:
    """D204 decision 10 (was: written, pinned before T-3.166 part 2): the folder holds the
    design and the plan, read-only to the worker, and its refusal says so."""
    worker, journal, _ = worker_rig(
        root,
        [
            calls(
                call("create", "c1", path="docs/chickadee/w/plan/README.md", content="# p\n")
            ),
            answer("made"),
        ],
    )
    worker.start("T1", "Write the plan.", delegate(journal, root))
    assert not (root / "docs/chickadee").exists()
    [line] = [p for p in journal.projection().lines(BoxMessageP) if p.payload.side == "tool"]
    shown = journal.get_blob(line.payload.sha).decode()
    assert "DENIED" in shown and "the work folder is read-only to you" in shown


def test_a_harness_error_names_its_type_and_logs_its_traceback(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Was: `the harness failed: ` and nothing else, for an `AssertionError` raised with no
    message (the OpenAI SDK's, on a stream it could not assemble), and no traceback anywhere."""

    def broken(_request: object) -> object:
        raise AssertionError

    worker, journal, _ = worker_rig(root, [broken])
    with caplog.at_level("ERROR", logger="chickadee.box.worker"):
        report = worker.start("T1", "Make go return 2.", delegate(journal, root))
    assert (report.stopped, report.report) == (
        "harness_error",
        "the harness failed: AssertionError: ",
    )
    [record] = caplog.records
    assert record.exc_info is not None and record.exc_info[0] is AssertionError


def test_a_forced_worker_asks_every_turn_but_the_wrap_up_to_be_a_tool_call(root: Path) -> None:
    """With `force_tool_call`, each of the worker's requests asks for a tool call, but the
    turn after the call cap, which asks for its report in text."""
    from dataclasses import replace

    from chickadee.roles import TASK_WORKER

    worker, journal, rung = worker_rig(
        root,
        [
            *(calls(call("read", f"c{i}", path="app.py", start=i + 1)) for i in range(3)),
            answer("read app.py twice"),
        ],
        limits={**WORKER_LIMITS, "worker.max_tool_calls": 2},
        policy=replace(TASK_WORKER, force_tool_call=True),
    )
    report = worker.start("T1", "Look.", delegate(journal, root))
    assert (report.stopped, report.report) == ("limit", "read app.py twice")
    assert [r.force_tool for r in rung.requests] == [True, True, True, False]


def test_an_unforced_worker_never_asks(root: Path) -> None:
    worker, journal, rung = worker_rig(
        root, [calls(call("read", "c1", path="app.py")), answer("done")]
    )
    worker.start("T1", "Look.", delegate(journal, root))
    assert [r.force_tool for r in rung.requests] == [False, False]
