"""The progress watchdog in a worker's round: its calls tagged from what the harness
recorded, the hooks told the round's progress, and the ladder's words where the worker reads
them.

What these pin: an edit, its undo and its redo are a change and then nothing; a read already
held is nothing; a test run's by-products are never a change, and cache directories never count;
a loop of refused calls is nudged on the batch's last answer, told to report, and ends with the
worker's report and what the harness measured; ignoring the report ends the round `halted`, the
reason shown; the worker's `Repetition` gives way while the watchdog is bound and is unchanged
when it is not; a record rebuilt from the journal tags the round as it was tagged live.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from box_helpers import delegate, worker_rig
from box_helpers import worker_call as call
from engines import answer, calls
from repos import init_repo
from sandboxes import Scripted

from chickadee.box.desk import render_round
from chickadee.box.progress import StepSeen, TaskProgress, counts, fingerprint, named_files
from chickadee.box.worker import WorkerStopped
from chickadee.hooks.builtins import bind_builtin
from chickadee.hooks.fire import BoundHook, HookSet, RoleHooks
from chickadee.hooks.payloads import AfterBatch, HookEvent, RoundEnded, WorkerProgress
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.steer import harness_notes, round_ended
from chickadee.hooks.verdicts import HookVerdict, Proceed
from chickadee.process import Completed
from chickadee.profiles import load_profile
from chickadee.roles import TASK_WORKER
from chickadee.schema.journal import HookP
from chickadee.workspace import current_tree

APP = "def go() -> int:\n    return 1\n"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return init_repo(tmp_path / "repo", {"app.py": APP})


LADDER: dict[str, object] = {"looping": 4}
"""The threshold these tests' call numbers are written for (D231's looping): the defaults are
pinned in `tests/hooks/test_hook_watchdog.py`."""


def _spec(event: str, use: str, role: str) -> HookSpec:
    return HookSpec.model_validate(
        {"event": event, "name": "progress", "use": use, "role": role, "with": LADDER}
    )


def _watched(*extra: BoundHook) -> HookSet:
    """The watchdog on the worker; the report at Athena's desk, as the hooks contract binds
    it."""
    return HookSet(
        (
            bind_builtin(_spec("after_batch", "progress_watchdog", "worker")),
            bind_builtin(_spec("round_ended", "progress_report", "athena")),
            *extra,
        )
    )


def _recording(seen: list[WorkerProgress]) -> BoundHook:
    """A hook that keeps what each `after_batch` was told."""

    def keep(event: HookEvent) -> HookVerdict:
        if isinstance(event.payload, AfterBatch) and event.payload.progress is not None:
            seen.append(event.payload.progress)
        return Proceed()

    spec = HookSpec.model_validate({"event": "after_batch", "name": "seen", "use": "seen"})
    return BoundHook(spec, keep, "seen")


def _tags(progress: WorkerProgress) -> list[str]:
    return [c.tag for c in progress.calls]


def _rebuilt(journal: Any, root: Path) -> TaskProgress:
    return TaskProgress.rebuilt(
        journal.projection(),
        "T1",
        journal.get_blob,
        root,
        profiles=(load_profile("python"),),
        registry_name=TASK_WORKER.registry_name,
    )


def test_the_noise_of_a_run_is_masked_and_cache_paths_never_count() -> None:
    one = "exit 1 · 340 ms · step:T1:1:call_ab12:9f8e7d6c\n1 failed in 0.31s at 0x7f3a2b"
    two = "exit 1 · 1203 ms · step:T1:1:call_zz99:01ab23cd\n1 failed in 1.02s at 0x7f3a99"
    assert fingerprint(one) == fingerprint(two)
    assert fingerprint(one) != fingerprint(one.replace("1 failed", "2 failed"))
    assert not counts("pkg/__pycache__/app.cpython-312.pyc")
    assert not counts(".pytest_cache/v/cache/lastfailed")
    assert not counts(".tox/py312/log/1.log") and not counts(".venv/bin/python")
    assert not counts("src/app.egg-info/PKG-INFO") and not counts("htmlcov/index.html")
    # The repetition rule's note counts the calls; the answer itself is the same one.
    again = "exit 1 · 30 ms\n1 failed\nrepetition{count: 2}. Do: you sent this exact call 2"
    assert fingerprint(again) == fingerprint(again.replace("count: 2", "count: 3"))
    assert counts("src/build_steps.py") and counts("README.md")


def test_only_repository_files_are_named_from_a_failure(root: Path) -> None:
    text = (
        f'File "{root}/app.py", line 2, in go\n'
        'File "/usr/lib/python3.12/unittest/case.py", line 58\n'
        "see ./app.py and app.py again"
    )
    assert named_files(text, {"app.py", "tests/test_app.py"}, root) == ("app.py",)


@pytest.mark.writes_worktree("app.py")
def test_an_edit_its_undo_and_its_redo_are_a_change_then_nothing(root: Path) -> None:
    seen: list[WorkerProgress] = []
    script = [
        calls(call("edit", "e1", path="app.py", old="return 1", new="return 2")),
        calls(call("edit", "e2", path="app.py", old="return 2", new="return 1")),
        calls(call("edit", "e3", path="app.py", old="return 1", new="return 2")),
        calls(call("read", "r1", path="app.py")),
        calls(call("read", "r2", path="app.py")),
        calls(call("nope", "n1")),
        answer("done"),
    ]
    worker, journal, _ = worker_rig(root, script, hooks=HookSet((_recording(seen),)))
    worker.start("T1", "Return 2.", delegate(journal, root))

    live = seen[-1]
    assert _tags(live) == ["changed", "nothing", "nothing", "learned", "nothing", "nothing"]
    assert [c.response for c in live.calls] == [1, 2, 3, 4, 5, 6]
    # A process that did not run the round tags it as it was tagged.
    assert _rebuilt(journal, root).now().calls == live.calls


class _Writing(Scripted):
    """A supervisor whose command writes the files its line names in `writes`."""

    def __init__(self, root: Path, writes: Mapping[str, tuple[str, ...]]) -> None:
        super().__init__(b"ok\n", 0)
        self._root = root
        self._writes = writes

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Completed:
        line = " ".join(argv)
        for command, paths in self._writes.items():
            if command in line:
                for p in paths:
                    target = self._root / p
                    if target.parent != self._root:
                        target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(f"{command}\n", encoding="utf-8")
        return super().run(argv, **kwargs)


@pytest.mark.writes_worktree(
    ".pytest_cache",
    ".pytest_cache/v",
    ".pytest_cache/v/lastfailed",
    "__pycache__",
    "__pycache__/app.cpython-312.pyc",
    "report.xml",
    "out.txt",
)
def test_a_test_runs_by_products_are_never_a_change(root: Path) -> None:
    seen: list[WorkerProgress] = []
    writes = {
        "pytest -q": (".pytest_cache/v/lastfailed", "__pycache__/app.cpython-312.pyc"),
        "pytest -x": ("report.xml",),
        "python gen.py": ("out.txt",),
    }
    script = [
        calls(call("shell", "s1", command="pytest -q")),
        calls(call("shell", "s2", command="pytest -x")),
        calls(call("shell", "s3", command="python gen.py")),
        calls(call("shell", "s4", command="pytest -q")),
        answer("done"),
    ]
    worker, journal, _ = worker_rig(
        root,
        script,
        hooks=HookSet((_recording(seen),)),
        supervisor=_Writing(root, writes),
    )
    worker.start("T1", "Generate it.", delegate(journal, root))

    # The cache never counts. `report.xml` moves the state, but a test run made it, and its
    # output masks to the first run's: nothing. The generator's file is a change; the first
    # run again shows nothing new.
    assert _tags(seen[-1]) == ["learned", "nothing", "changed", "nothing"]


def _nope(n: int, start: int = 1) -> list[Any]:
    return [calls(call("nope", f"n{i}", why=str(i))) for i in range(start, start + n)]


def _tool_text(request: Any, call_id: str) -> str:
    [found] = [m for m in request.messages if m.role == "tool" and m.tool_call_id == call_id]
    assert isinstance(found.content, str)
    return found.content


def test_a_loop_is_nudged_then_reported_with_what_the_harness_measured(root: Path) -> None:
    script = [*_nope(8), calls(call("finish", "f", report="stuck on a tool I lack")), answer()]
    worker, journal, rung = worker_rig(root, script, hooks=_watched())
    report = worker.start("T1", "Do it.", delegate(journal, root))

    assert (report.stopped, report.report) == ("replied", "stuck on a tool I lack")
    nudge = _tool_text(rung.requests[4], "n4")
    assert "[progress watchdog] Your last 4 calls have not moved the task" in nudge
    assert "Stop repeating these calls." in nudge
    assert "call `finish`" in _tool_text(rung.requests[8], "n8")
    # The round at Athena's desk, as `desk._steered` asks it: her hooks told its progress.
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
    [told] = harness_notes(athena, task="T1", round_=1)
    assert told.startswith("Progress (harness): 0 change(s)")
    assert "the worker was asked to stop and report (looping)" in told
    # What Athena reads of the round: the worker's report, the facts, then the harness's
    # findings under their own heading (`desk.render_round`).
    shown = render_round(report, notes=(told,))
    assert shown.index("stuck on a tool I lack") < shown.index("Measured by the harness")
    # A process that did not run the round rebuilds the same progress from the journal.
    again, _, _ = worker_rig(root, [], resume=True, journal=journal)
    assert again.progress("T1", 1) == worker.progress("T1", 1)


def test_a_worker_that_ignores_the_report_is_halted_and_told_why(root: Path) -> None:
    script = [*_nope(11), answer("I was stuck: no such tool")]
    worker, journal, rung = worker_rig(root, script, hooks=_watched())
    report = worker.start("T1", "Do it.", delegate(journal, root))

    assert (report.stopped, report.report) == ("halted", "I was stuck: no such tool")
    assert "Reply now, in plain text" in _tool_text(rung.requests[11], "n11")
    assert len(rung.requests) == 12


def _same(n: int) -> list[Any]:
    return [calls(call("nope", f"s{i}")) for i in range(n)]


def test_repetition_gives_way_to_the_watchdog(root: Path) -> None:
    script = [*_same(8), calls(call("finish", "f", report="stuck")), answer()]
    worker, journal, rung = worker_rig(root, script, hooks=_watched())
    assert worker.start("T1", "Do it.", delegate(journal, root)).stopped == "replied"
    assert "[progress watchdog]" in _tool_text(rung.requests[4], "s3")


def test_repetition_is_unchanged_without_the_watchdog(root: Path) -> None:
    worker, journal, _ = worker_rig(root, [*_same(6), answer()])
    assert worker.start("T1", "Do it.", delegate(journal, root)).stopped == "looped"


def test_a_round_resumed_in_another_process_keeps_its_ladder(root: Path) -> None:
    """The watchdog replays the round from the journal: a nudge given before the crash is not
    given again, and the report comes when it would have."""
    stopping = threading.Event()

    def stop_after(request: Any) -> Any:
        stopping.set()
        return calls(call("nope", "n6", why="6"))

    script = [*_nope(5), stop_after]
    worker, journal, _ = worker_rig(root, script, hooks=_watched(), cancel=stopping)
    with pytest.raises(WorkerStopped):
        worker.start("T1", "Do it.", delegate(journal, root))
    journal.close()

    resumed, journal, _ = worker_rig(
        root,
        [
            *_nope(3, start=7),
            calls(call("finish", "f", report="r")),
            answer(),
        ],
        resume=True,
        hooks=_watched(),
    )
    report = resumed.resume("T1")

    assert report is not None and report.stopped == "replied"
    said = [
        p.payload for p in journal.projection().lines(HookP) if p.payload.event == "after_batch"
    ]
    assert [h.decision for h in said] == ["note", "note"]


@pytest.mark.writes_worktree("cmd/build/main.py")
def test_a_tracked_file_under_a_build_directory_is_a_change(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo", {"cmd/build/main.py": "x = 1\n"})
    seen: list[WorkerProgress] = []
    script = [
        calls(call("edit", "e1", path="cmd/build/main.py", old="x = 1", new="x = 2")),
        answer("done"),
    ]
    worker, journal, _ = worker_rig(repo, script, hooks=HookSet((_recording(seen),)))
    worker.start("T1", "Set x.", delegate(journal, repo))
    assert _tags(seen[-1]) == ["changed"]


def _tree(root: Path) -> str:
    return current_tree(root, untracked_max_bytes=None)[0]


@pytest.mark.writes_worktree("app.py", "gen.txt")
def test_only_the_calls_own_move_of_the_files_is_its_change(root: Path) -> None:
    progress = TaskProgress(root, _tree(root))
    progress.begin_round(1, None)
    progress.respond()
    # Written by someone else (Athena between rounds, a job): the call's step changed nothing.
    (root / "app.py").write_text("def go() -> int:\n    return 2\n", encoding="utf-8")
    other = progress.observe(
        call("shell", "s1", command="ls"), "a", refused=False, step=StepSeen(_tree(root), ())
    )
    # Beside a running job, a command's changes may be the job's.
    (root / "gen.txt").write_text("made\n", encoding="utf-8")
    beside = StepSeen(_tree(root), ("gen.txt",), beside=True)
    job = progress.observe(call("shell", "s2", command="make"), "b", refused=False, step=beside)
    assert (other.tag, job.tag) == ("learned", "learned")
    # An edit knows what it wrote: a job running beside it does not take its change away.
    (root / "app.py").write_text("def go() -> int:\n    return 3\n", encoding="utf-8")
    edited = StepSeen(_tree(root), ("app.py",), beside=True)
    edit = progress.observe(
        call("edit", "e1", path="app.py", old="2", new="3"), "c", refused=False, step=edited
    )
    assert edit.tag == "changed"


class _Runs(Scripted):
    """Each command answers the next of `outputs`, as (stdout, exit code)."""

    def __init__(self, outputs: list[tuple[bytes, int]]) -> None:
        super().__init__()
        self._outputs = list(outputs)

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Completed:
        self.spawned += 1
        stdout, code = self._outputs.pop(0)
        return Completed(stdout, b"", code, False)


ROUTE = "tests/test_app.py::test_route"


def test_a_rebuilt_record_reads_each_test_run_from_its_whole_output(root: Path) -> None:
    """The record reads a run from its `exec` line, not from the answer the
    model read, whose head is all a long output keeps; a test failing, then passing under the
    same command after an edit, is fixed, and Athena is told so."""
    padding = b"".join(b"line %d of a long log\n" % i for i in range(4000))
    failing = padding + f"FAILED {ROUTE} - assert 1 == 2\n1 failed, 3 passed in 0.2s\n".encode()
    worker, journal, _ = worker_rig(
        root,
        [
            calls(call("shell", "c1", command="pytest -q")),
            calls(call("edit", "c2", path="app.py", old="return 1", new="return 2")),
            calls(call("shell", "c3", command="pytest -q")),
            answer("Fixed."),
        ],
        supervisor=_Runs([(failing, 1), (b"4 passed in 0.2s\n", 0)]),
        hooks=_watched(),
    )
    worker.start("T1", "Fix test_route.", delegate(journal, root))
    live = worker.progress("T1", 1)
    progress = TaskProgress.rebuilt(
        journal.projection(), "T1", journal.get_blob, root, profiles=(load_profile("python"),)
    ).now()
    runs = [c.run for c in progress.calls if c.run is not None]
    assert [(r.outcome, r.how, r.failing, r.fixed) for r in runs] == [
        ("failed", "summary", (ROUTE,), ()),
        ("passed", "summary", (), (ROUTE,)),
    ]
    assert progress.tests is not None and progress.tests.fixed == (ROUTE,)
    assert progress.runnable
    edit = next(c for c in progress.calls if c.tool == "edit")
    assert edit.changed == ("app.py",)
    # The live record read each run as a process that did not run the round rebuilds it.
    again, _, _ = worker_rig(root, [], resume=True, journal=journal, hooks=_watched())
    assert live is not None and live.tests == progress.tests
    assert again.progress("T1", 1) == live
