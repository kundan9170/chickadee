"""A cancellation token reaches the process.

A `shell` call used to block in one `communicate(timeout=)` and end only at its timeout, so a
stop pressed during a build came back when the build finished. The supervisor now waits in
slices and looks at the token between them. A batch run is reaped on the token; a session's
blocking call raises and leaves its process running, because stopping a turn is not stopping
a dev server.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from chickadee.process.supervisor import (
    CANCEL_POLL_MS,
    GRACE_S,
    Cancelled,
    ProcessSupervisor,
    scrub_env,
)

SLACK_S = 1.0
"""Scheduling noise on a loaded CI runner. The bound under test is `CANCEL_POLL_MS` plus the
reap's grace; a regression to waiting out the timeout overshoots it by thirty seconds."""


def _set_after(event: threading.Event, delay_s: float) -> threading.Thread:
    timer = threading.Timer(delay_s, event.set)
    timer.start()
    return timer


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _group_alive(pgid: int, within_s: float = 1.0) -> bool:
    """Whether any process of the group is still there after `within_s`. Polled, because a
    killed grandchild is reparented and reaped by init a moment after its group's leader."""
    deadline = time.monotonic() + within_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.05)


def _read_pid(path: Path) -> int:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.02)
    raise AssertionError(f"{path} never received a pid")


def test_a_set_token_reaps_a_running_command_within_the_poll(tmp_path: Path) -> None:
    """The child records its pid, which is its group's id (`start_new_session`), and starts a
    grandchild beside itself: the whole group has to be gone, not just the shell."""
    pidfile = tmp_path / "pid"
    token = threading.Event()
    timer = _set_after(token, 0.5)
    started = time.monotonic()
    done = ProcessSupervisor().run(
        ("sh", "-c", f"echo $$ > {pidfile}; sleep 30 & sleep 30"),
        cwd=tmp_path,
        env=scrub_env(),
        timeout_s=20,
        cancel=token,
    )
    returned = time.monotonic()
    timer.join()
    assert done.cancelled is True
    assert done.timed_out is False
    assert returned - started < 0.5 + CANCEL_POLL_MS / 1000 + GRACE_S + SLACK_S
    assert not _group_alive(_read_pid(pidfile)), "no process of the group survives the reap"


def test_partial_output_is_kept(tmp_path: Path) -> None:
    """What the command printed before the stop is evidence, and the result keeps it."""
    token = threading.Event()
    timer = _set_after(token, 0.5)
    done = ProcessSupervisor().run(
        ("sh", "-c", "echo started; sleep 30"),
        cwd=tmp_path,
        env=scrub_env(),
        timeout_s=20,
        cancel=token,
    )
    timer.join()
    assert done.cancelled
    assert b"started" in done.stdout


def test_a_token_never_set_runs_to_the_timeout(tmp_path: Path) -> None:
    """The limit is still the harness's and still reads as one: `timed_out`, not `cancelled`."""
    done = ProcessSupervisor().run(
        ("sleep", "30"),
        cwd=tmp_path,
        env=scrub_env(),
        timeout_s=1,
        cancel=threading.Event(),
    )
    assert done.timed_out is True
    assert done.cancelled is False


def test_a_command_that_finishes_is_not_cancelled(tmp_path: Path) -> None:
    """Slicing the wait changes nothing about a run the token never touched."""
    done = ProcessSupervisor().run(
        ("sh", "-c", "echo hi"),
        cwd=tmp_path,
        env=scrub_env(),
        timeout_s=10,
        cancel=threading.Event(),
    )
    assert done.exit_code == 0 and done.stdout == b"hi\n"
    assert not done.cancelled and not done.timed_out


def test_stdin_is_written_once_across_the_slices(tmp_path: Path) -> None:
    """`communicate` refuses new input on a retry, so the input goes in on the first slice
    and the rest are waits. A child that is slow to read still gets all of it, once."""
    done = ProcessSupervisor().run(
        ("sh", "-c", "sleep 0.5; cat"),
        cwd=tmp_path,
        env=scrub_env(),
        timeout_s=10,
        stdin=b"payload\n",
        cancel=threading.Event(),
    )
    assert done.stdout == b"payload\n"
    assert not done.cancelled


def test_a_session_read_raises_on_the_token_and_the_child_lives_on(tmp_path: Path) -> None:
    """A dev server survives a turn interrupt: the blocked read returns, typed, and the
    process is reaped only when the session closes."""
    pidfile = tmp_path / "pid"
    token = threading.Event()
    supervisor = ProcessSupervisor()
    with supervisor.session(
        ("sh", "-c", f"echo $$ > {pidfile}; exec sleep 30"),
        cwd=tmp_path,
        timeout_s=20,
        cancel=token,
    ) as sess:
        pid = _read_pid(pidfile)
        timer = _set_after(token, 0.3)
        started = time.monotonic()
        with pytest.raises(Cancelled):
            sess.read(1)
        timer.join()
        assert time.monotonic() - started < 0.3 + CANCEL_POLL_MS / 1000 + SLACK_S
        assert _alive(pid), "the token stops the call, not the process"
        with pytest.raises(Cancelled):
            sess.write(b"x\n")
    assert not _group_alive(pid), "close() still reaps it"


def test_a_session_with_no_token_still_times_out(tmp_path: Path) -> None:
    with (
        pytest.raises(TimeoutError),
        ProcessSupervisor().session(("sleep", "30"), cwd=tmp_path, timeout_s=1) as sess,
    ):
        sess.read(1)
