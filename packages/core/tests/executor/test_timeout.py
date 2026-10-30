"""Timeout kills the process group and reaps — no orphans."""

from __future__ import annotations

import os
import shlex
import time
from pathlib import Path

from executor_helpers import request_for

from chickadee.executor.executor import Executor


def test_a_timeout_is_reported_not_hidden(unsandboxed: Executor, work: Path) -> None:
    result = unsandboxed.run(
        request_for(work, ("sleep", "30"), timeout_s=1), operation_id="op-1"
    )
    assert result.timed_out
    assert not result.ok


def test_a_timeout_reaps_the_whole_process_group(unsandboxed: Executor, work: Path) -> None:
    """Killing only the direct child leaves grandchildren holding the very files the effect
    bracket is about to measure."""
    marker = work / "grandchild.pid"
    quoted = shlex.quote(str(marker))
    script = f"sh -c 'echo $$ > {quoted}; sleep 30' & sleep 30"
    result = unsandboxed.run(
        request_for(work, ("sh", "-c", script), timeout_s=2), operation_id="op-1"
    )
    assert result.timed_out

    time.sleep(0.5)
    pid = int(marker.read_text(encoding="utf-8").strip())
    assert not _alive(pid), f"grandchild {pid} survived the timeout"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but not ours
        return True
    return True


def test_output_before_a_timeout_survives(unsandboxed: Executor, work: Path) -> None:
    """The tail of a hung run is often the only clue about why it hung."""
    result = unsandboxed.run(
        request_for(work, ("sh", "-c", "echo starting; sleep 30"), timeout_s=2),
        operation_id="op-1",
    )
    assert result.timed_out
    assert "starting" in result.stdout


def test_a_background_child_does_not_outlive_a_normal_exit(
    unsandboxed: Executor, work: Path
) -> None:
    """The direct child exiting does not mean the group is finished.

    `cmd &` leaves a process holding the worktree, free to write *after* the post-run
    snapshot — so the effect diff reports nothing and the process is still running.
    """
    marker = work / "late.txt"
    # Streams redirected, which is what makes this the real hazard: with the pipes still
    # attached, `communicate()` waits for the background child anyway and the effect
    # snapshot sees its write. Detached, the call returns at once and the child outlives it.
    quoted = shlex.quote(str(marker))
    script = f"(sleep 3; echo late > {quoted}) >/dev/null 2>&1 & echo done"
    result = unsandboxed.run(
        request_for(work, ("sh", "-c", script), timeout_s=20), operation_id="op-1"
    )
    assert result.exit_code == 0
    assert not result.timed_out

    time.sleep(4)
    assert not marker.exists(), "a background child wrote after the run was declared finished"


def test_a_child_ignoring_sigterm_is_killed(unsandboxed: Executor, work: Path) -> None:
    """SIGTERM is a request; a child may decline it.

    Without escalation such a child keeps running after `run()` returns, still holding the
    worktree the effect bracket has already measured.
    """
    marker = work / "stubborn.txt"
    quoted = shlex.quote(str(marker))
    script = (
        f"trap '' TERM; (sleep 6; echo late > {quoted}) >/dev/null 2>&1 & "
        "trap '' TERM; echo done"
    )
    result = unsandboxed.run(
        request_for(work, ("sh", "-c", script), timeout_s=20), operation_id="op-1"
    )
    assert result.exit_code == 0

    time.sleep(7)
    assert not marker.exists(), "a SIGTERM-ignoring child outlived the run"


def test_the_grace_period_is_bounded(unsandboxed: Executor, work: Path) -> None:
    """Escalation must not make a normal run wait: nothing to kill means no delay."""
    started = time.monotonic()
    unsandboxed.run(request_for(work, ("echo", "hi")), operation_id="op-1")
    assert time.monotonic() - started < Executor.GRACE_S
