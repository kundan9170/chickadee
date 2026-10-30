"""`ProcessSupervisor.start` and `Job`: a child that runs
on past its call, its output in files, its exit seen once by one waiter."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from chickadee.process.supervisor import (
    CANCEL_POLL_MS,
    Cancelled,
    Job,
    JobExit,
    ProcessSupervisor,
    process_started,
    scrub_env,
)

MIB = 1 << 20


def _start(
    tmp_path: Path,
    code: str,
    *,
    max_runtime_s: int | None = 30,
    max_log_bytes: int = 64 * MIB,
    exits: list[JobExit] | None = None,
) -> Job:
    def on_exit(_job: Job, exit_: JobExit) -> None:
        if exits is not None:
            exits.append(exit_)

    return ProcessSupervisor().start(
        (sys.executable, "-c", code),
        cwd=tmp_path,
        env=scrub_env(),
        out=tmp_path / "job.out",
        err=tmp_path / "job.err",
        max_runtime_s=max_runtime_s,
        max_log_bytes=max_log_bytes,
        on_exit=on_exit,
    )


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_stdout_and_stderr_reach_their_own_files_and_stdin_is_dev_null(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "print('out'); print('err', file=sys.stderr)\n"
        "print('stdin=' + repr(sys.stdin.read()))\n"
    )
    job = _start(tmp_path, code)
    done = job.wait(10)
    assert done is not None and done.exit_code == 0 and done.ended == "exited"
    assert (tmp_path / "job.out").read_text() == "out\nstdin=''\n"
    assert (tmp_path / "job.err").read_text() == "err\n"


def test_wait_answers_none_at_its_timeout_and_the_job_runs_on(tmp_path: Path) -> None:
    job = _start(tmp_path, "import time; time.sleep(30)")
    assert job.wait(0.2) is None
    assert job.alive()
    job.kill("test")
    job.join(10)


def test_wait_raises_cancelled_on_the_token_and_leaves_the_job_running(tmp_path: Path) -> None:
    job = _start(tmp_path, "import time; time.sleep(30)")
    token = threading.Event()
    threading.Timer(0.2, token.set).start()
    begun = time.monotonic()
    with pytest.raises(Cancelled):
        job.wait(30, token)
    assert time.monotonic() - begun < 0.2 + 3 * CANCEL_POLL_MS / 1000
    assert job.alive()
    job.kill("test")
    job.join(10)


@pytest.mark.parametrize(
    ("how", "ended"),
    [("exit", "exited"), ("timeout", "timed_out"), ("kill", "killed")],
)
def test_on_exit_fires_exactly_once(tmp_path: Path, how: str, ended: str) -> None:
    exits: list[JobExit] = []
    code = "pass" if how == "exit" else "import time; time.sleep(30)"
    job = _start(tmp_path, code, max_runtime_s=1 if how == "timeout" else 30, exits=exits)
    if how == "kill":
        job.kill("person")
    job.join(15)
    assert [e.ended for e in exits] == [ended]
    assert exits[0].by == ("person" if how == "kill" else None)


def test_a_kill_names_who_asked_and_a_late_kill_changes_nothing(tmp_path: Path) -> None:
    job = _start(tmp_path, "pass")
    job.join(10)
    job.kill("close")
    outcome = job.outcome
    assert outcome is not None and outcome.ended == "exited" and outcome.by is None


def test_a_kill_during_the_reap_after_an_exit_is_not_its_cause(tmp_path: Path) -> None:
    # The leader exits 3 on its own and leaves a child that ignores SIGTERM, so the waiter's
    # reap of the group takes the whole grace. A kill sent then came after the exit.
    pidfile = tmp_path / "child.pid"
    stubborn = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    code = (
        "import subprocess, sys\n"
        f"p = subprocess.Popen([sys.executable, '-c', {stubborn!r}])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "sys.exit(3)\n"
    )
    job = _start(tmp_path, code)
    deadline = time.monotonic() + 10
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)  # the leader is gone; the waiter is inside its reap's grace
    assert job.outcome is None, "still reaping the child"
    assert job.kill("zeus") is False, "the process had already ended"
    job.join(15)
    outcome = job.outcome
    assert outcome is not None
    assert (outcome.ended, outcome.by, outcome.exit_code) == ("exited", None, 3)


def test_a_leader_that_exits_takes_the_child_it_left_behind(tmp_path: Path) -> None:
    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
    )
    job = _start(tmp_path, code)
    job.join(15)
    child = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while not _gone(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child)


def test_the_runtime_bound_reaps_a_tasks_grandchildren_and_a_server_has_none(
    tmp_path: Path,
) -> None:
    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    task = _start(tmp_path, code, max_runtime_s=1)
    task.join(15)
    outcome = task.outcome
    assert outcome is not None and outcome.ended == "timed_out"
    assert _gone(int(pidfile.read_text()))

    server = _start(tmp_path, "import time; time.sleep(30)", max_runtime_s=None)
    assert server.wait(2.5) is None
    server.kill("test")
    server.join(10)


def test_the_log_ceiling_kills_within_a_second(tmp_path: Path) -> None:
    code = "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n"
    job = _start(tmp_path, code, max_log_bytes=MIB)
    job.join(15)
    outcome = job.outcome
    assert outcome is not None and outcome.ended == "ceiling" and outcome.by is None


def test_an_exit_a_waiter_was_blocked_on_is_seen(tmp_path: Path) -> None:
    job = _start(tmp_path, "import time; time.sleep(0.5)")
    seen = job.wait(10, observe=True)
    assert seen is not None and seen.seen


def test_an_exit_nobody_waited_on_is_not_seen(tmp_path: Path) -> None:
    exits: list[JobExit] = []
    job = _start(tmp_path, "pass", exits=exits)
    job.join(10)
    assert exits and not exits[0].seen
    # A wait after the exit returns it, and changes nothing about how it was recorded.
    assert job.wait(1, observe=True) == exits[0]


def test_process_started_is_stable_for_a_live_pid_and_differs_for_a_new_one(
    tmp_path: Path,
) -> None:
    first = _start(tmp_path, "import time; time.sleep(30)")
    started = process_started(first.pid)
    assert started is not None and started == first.started == process_started(first.pid)
    time.sleep(0.05)
    second = _start(tmp_path, "import time; time.sleep(30)")
    # Clock ticks are 10 ms: two processes started 50 ms apart never share one.
    assert second.started is not None and second.started != started
    for job in (first, second):
        job.kill("test")
        job.join(10)
    assert process_started(first.pid) is None
