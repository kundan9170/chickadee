"""`Executor.start` and `finish`: `run`'s checks through one `_admit`, and
for a command that ends inside its window, `run`'s answer field for field."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from executor_helpers import request_for

from chickadee.errors import InfraError
from chickadee.executor.executor import Executor, Started
from chickadee.executor.types import ExecRequest, ExecResult, PolicyClass, SandboxLevel
from chickadee.isolation.none import Passthrough

MIB = 1 << 20


def _start(executor: Executor, request: ExecRequest, tmp: Path) -> Started | ExecResult:
    return executor.start(
        request, out=tmp / "o.out", err=tmp / "o.err", max_runtime_s=30, max_log_bytes=MIB
    )


def _finished(executor: Executor, request: ExecRequest, tmp: Path) -> ExecResult:
    started = _start(executor, request, tmp)
    assert isinstance(started, Started)
    exit_ = started.job.wait(10)
    assert exit_ is not None
    return executor.finish(started, exit_)


@pytest.mark.parametrize(
    ("command", "policy"),
    [
        ("echo out; echo err >&2", PolicyClass.WORKTREE),
        ("echo changed > existing.txt; exit 3", PolicyClass.ACTION),
        ("true", PolicyClass.ACTION),
    ],
)
def test_start_then_finish_answers_as_run_does(
    work: Path, tmp_path: Path, command: str, policy: PolicyClass
) -> None:
    executor = Executor(backend=Passthrough(), put_blob=lambda b: f"blob:{len(b)}")
    request = request_for(work, ("sh", "-c", command), policy)
    (work / "existing.txt").write_text("original\n", encoding="utf-8")
    ran = executor.run(request, operation_id="op-1")
    (work / "existing.txt").write_text("original\n", encoding="utf-8")
    finished = _finished(executor, request, tmp_path)
    assert replace(finished, wall_ms=0) == replace(ran, wall_ms=0)
    assert Executor.exec_event(request, replace(finished, wall_ms=0), "op-1") == (
        Executor.exec_event(request, replace(ran, wall_ms=0), "op-1")
    )


def test_a_refusal_from_start_is_runs(work: Path, tmp_path: Path) -> None:
    executor = Executor(backend=Passthrough())
    request = request_for(work, ("cat", str(Path.home() / ".ssh" / "id_rsa")))
    refused = _start(executor, request, tmp_path)
    assert isinstance(refused, ExecResult)
    assert refused == executor.run(request, operation_id="op-1")
    assert refused.exit_code == 126 and not (tmp_path / "o.out").exists()


def test_start_refuses_without_a_sandbox_where_one_is_required(
    work: Path, tmp_path: Path
) -> None:
    executor = Executor(backend=Passthrough(), require_sandbox=True)
    request = request_for(work, ("true",))
    with pytest.raises(InfraError, match="require_sandbox"):
        _start(executor, request, tmp_path)
    with pytest.raises(InfraError, match="require_sandbox"):
        executor.run(request, operation_id="op-1")


def test_an_approved_request_starts_unwrapped_with_the_persons_environment(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHICKADEE_TEST_PERSON_VAR", "theirs")
    executor = Executor(backend=Passthrough(), require_sandbox=True)
    request = replace(
        request_for(work, ("sh", "-c", "echo $CHICKADEE_TEST_PERSON_VAR")), unsandboxed=True
    )
    finished = _finished(executor, request, tmp_path)
    assert finished.sandbox is SandboxLevel.NONE
    assert finished.stdout == "theirs\n"
    assert finished.network_enforced == "on"


def test_a_timeout_and_a_cancel_are_the_callers_to_say(work: Path, tmp_path: Path) -> None:
    executor = Executor(backend=Passthrough())
    started = _start(executor, request_for(work, ("sleep", "30")), tmp_path)
    assert isinstance(started, Started)
    started.job.kill("timeout")
    started.job.join(10)
    exit_ = started.job.outcome
    assert exit_ is not None
    timed = executor.finish(started, exit_, timed_out=True)
    assert timed.timed_out and not timed.cancelled and timed.exit_code != 0
    cancelled = executor.finish(started, exit_, cancelled=True)
    assert cancelled.cancelled and not cancelled.timed_out
