"""D95 at the executor: a cancelled run is a result a caller can read, and never a verdict.

`ExecResult.cancelled` carries the supervisor's fact, `ok` is false, the partial output is kept
as evidence, and a cancelled test run reads as `cancelled`: not `passed`, because it did
not finish, and not `errored` or `failed`, because the nonzero exit is the reap's.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from executor_helpers import request_for

from chickadee.executor.executor import Executor
from chickadee.executor.tests import TestOutcome, read_test_run
from chickadee.executor.types import ExecResult, PolicyClass, SandboxLevel
from chickadee.isolation.none import Passthrough
from chickadee.process.supervisor import Completed, ProcessSupervisor
from chickadee.profiles import load_profile


class _Cancelled(ProcessSupervisor):
    """Returns what a reaped run returns, and records the token it was handed."""

    def __init__(self, stdout: bytes = b"") -> None:
        self.stdout = stdout
        self.tokens: list[threading.Event | None] = []

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_s: int,
        stdin: bytes | None = None,
        cancel: threading.Event | None = None,
    ) -> Completed:
        del argv, cwd, env, timeout_s, stdin
        self.tokens.append(cancel)
        return Completed(self.stdout, b"", -15, False, cancelled=True)


def test_the_result_carries_the_supervisors_cancellation(work: Path) -> None:
    supervisor = _Cancelled(stdout=b"collected 12 items\n")
    token = threading.Event()
    result = Executor(backend=Passthrough(), supervisor=supervisor).run(
        request_for(work, ("pytest", "-q")), operation_id="op-1", cancel=token
    )
    assert supervisor.tokens == [token], "the token reaches the supervisor unchanged"
    assert result.cancelled is True
    assert result.timed_out is False
    assert result.ok is False
    assert result.stdout_sha is not None, "partial output is kept as evidence"


def test_no_token_is_the_old_call(work: Path) -> None:
    """Callers with no human pass nothing, and the supervisor sees `None`."""
    supervisor = _Cancelled()
    Executor(backend=Passthrough(), supervisor=supervisor).run(
        request_for(work, ("true",)), operation_id="op-1"
    )
    assert supervisor.tokens == [None]


def test_a_real_run_stops_on_the_token(work: Path) -> None:
    token = threading.Event()
    timer = threading.Timer(0.5, token.set)
    timer.start()
    result = Executor(backend=Passthrough()).run(
        request_for(work, ("sh", "-c", "echo started; sleep 30"), timeout_s=20),
        operation_id="op-1",
        cancel=token,
    )
    timer.join()
    assert result.cancelled and not result.ok
    assert result.stdout == "started\n"
    assert result.stdout_sha is not None


def test_a_cancelled_test_run_is_neither_passed_nor_errored(work: Path) -> None:
    """Assert on the outcome, not the code. The partial output here reads like a failure; a
    classifier that trusted it would report a test failing that never finished."""
    supervisor = _Cancelled(stdout=b"FAILED tests/test_a.py::test_b\n")
    result = Executor(backend=Passthrough(), supervisor=supervisor).run(
        request_for(work, ("pytest", "-q"), PolicyClass.WORKTREE),
        operation_id="op-1",
        cancel=threading.Event(),
    )
    ran = read_test_run("pytest -q", result, (load_profile("python"),))
    assert ran is not None
    assert ran.outcome is TestOutcome.CANCELLED
    assert ran.outcome not in (TestOutcome.PASSED, TestOutcome.ERRORED, TestOutcome.FAILED)
    assert ran.behavioral is False
    assert ran.exec is result, "the raw run survives the classification"


def test_a_cancelled_run_is_not_ok_even_at_exit_zero() -> None:
    """A child that traps SIGTERM and exits 0 is still a run the user stopped. `ok` reads
    the flag, not only the code."""
    result = ExecResult(
        exit_code=0,
        stdout_sha=None,
        stderr_sha=None,
        wall_ms=5,
        timed_out=False,
        sandbox=SandboxLevel.NONE,
        cancelled=True,
    )
    assert result.ok is False
