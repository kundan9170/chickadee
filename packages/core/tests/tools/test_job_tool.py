"""`job`: list, tail, wait and kill over a real job table, Zeus's shell
spawning the jobs it reads."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from repos import init_repo
from tool_args import checked

from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import CallCancelled, ToolResult, UnknownJob
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.isolation.none import Passthrough
from chickadee.shell import Shell
from chickadee.tools.builtin import job
from chickadee.tools.context import ToolContext
from chickadee.tools.tool import Invocation, Said
from chickadee.zeus import jobs as jobs_module
from chickadee.zeus.jobs import JobLimits, JobTable


class JobToolRig:
    def __init__(self, tmp_path: Path) -> None:
        root = init_repo(tmp_path / "repo", {"a.txt": "a\n"})
        self.session = Session.create(tmp_path / "sessions", "S-tool")
        self.reader = Reader(
            SessionHome(self.session),
            HandleResolver(workspace=root, session_blob=self.session.blob),
            vision=Vision("z"),
        )
        self.table = JobTable(
            self.session,
            limits=JobLimits(yield_s=1, inline_bytes=64),
        )
        self.shell = Shell(
            Executor(Passthrough()),
            root,
            lambda data: self.reader.store("output", data),
            allow_unsandboxed_shell=True,
            jobs=self.table,
        )

    def start(self, command: str, op: str = "S:c1") -> None:
        self.shell.run(
            checked("shell", {"command": command, "background": True}), operation_id=op
        )

    def call(self, cancel: threading.Event | None = None, **args: object) -> object:
        ctx = ToolContext(role="zeus", jobs=self.table, handles=self.reader, cancel=cancel)
        return job.run(Invocation("c", "S:t1", 0), checked("job", args), ctx)


@pytest.fixture
def rig(tmp_path: Path) -> JobToolRig:
    return JobToolRig(tmp_path)


def test_list_names_every_job_running_or_ended(rig: JobToolRig) -> None:
    rig.start("sleep 30")
    rig.start("sleep 0.5", op="S:c2")  # outlives the yield, so it is a job
    deadline = time.monotonic() + 10
    while "j2" in rig.table.running() and time.monotonic() < deadline:
        time.sleep(0.05)
    listed = rig.call(action="list")
    assert isinstance(listed, Said)
    text = listed.render()
    assert "j1 · task · running" in text and "j2 · task · exited 0" in text
    rig.table.close()


def test_tail_is_capped_with_the_whole_output_behind_a_handle(rig: JobToolRig) -> None:
    rig.start("for i in $(seq 1 40); do echo line-$i; done; sleep 30")
    time.sleep(1)
    tailed = rig.call(action="tail", id="j1", lines=20)
    assert isinstance(tailed, ToolResult)
    assert tailed.truncation is not None, "64 bytes cannot hold 20 lines"
    assert "line-21" in tailed.render() and "line-1\n" not in (tailed.content or "")
    assert rig.session.blob(tailed.handle.content_sha).decode().count("line-") == 40
    rig.table.close()


def test_wait_returns_at_the_exit_before_its_timeout(rig: JobToolRig) -> None:
    rig.start("sleep 1; exit 2")
    waited = rig.call(action="wait", id="j1", timeout_s=30)
    assert isinstance(waited, Said) and waited.render() == "j1 exited 2 after 1s"


def test_wait_answers_cancelled_on_a_hard_stop_and_the_job_runs_on(rig: JobToolRig) -> None:
    rig.start("sleep 30")
    token = threading.Event()
    threading.Timer(0.3, token.set).start()
    stopped = rig.call(cancel=token, action="wait", id="j1", timeout_s=30)
    assert isinstance(stopped, CallCancelled)
    assert rig.table.running() == ("j1",)
    rig.table.close()


def test_kill_ends_it_and_an_unknown_id_is_unresolved(rig: JobToolRig) -> None:
    rig.start("sleep 30")
    killed = rig.call(action="kill", id="j1")
    assert isinstance(killed, Said) and killed.render() == "j1 killed"
    unknown = rig.call(action="tail", id="j9")
    assert isinstance(unknown, UnknownJob) and "job(action='list')" in unknown.render()
    rig.table.close()


def test_tail_reads_a_long_log_from_its_end_and_says_so(
    rig: JobToolRig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs_module, "TAIL_READ_BYTES", 200)
    rig.start("for i in $(seq 1 400); do echo line-$i; done; sleep 30")
    time.sleep(1)
    tailed = rig.call(action="tail", id="j1", lines=5)
    assert isinstance(tailed, ToolResult)
    assert "bytes before this unread" in tailed.render().splitlines()[0]
    kept = rig.session.blob(tailed.handle.content_sha).decode()
    assert len(kept) <= 200 and kept.startswith("line-") and "line-400" in kept
    rig.table.close()
