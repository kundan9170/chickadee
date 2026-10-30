"""§7 — one run produces exactly one `exec` event, with every required field."""

from __future__ import annotations

from pathlib import Path

from executor_helpers import request_for

from chickadee.executor.executor import Executor
from chickadee.executor.types import NetworkPolicy, PolicyClass
from chickadee.schema.journal import ExecP


def test_a_run_fills_every_required_field(unsandboxed: Executor, work: Path) -> None:
    """Validated against the real contract, not a hand-written field list — a list would
    drift from `ExecP` the moment either changed."""
    request = request_for(work, ("echo", "hi"), label="say hi")
    result = unsandboxed.run(request, operation_id="op-1")
    event = ExecP(**Executor.exec_event(request, result, "op-1"))  # type: ignore[arg-type]

    assert event.operation_id == "op-1"
    assert event.policy == "worktree"
    assert event.label == "say hi"
    assert event.exit_code == 0
    assert event.sandbox == "none"
    assert event.network is False
    assert event.violations == ()


def test_the_command_is_hashed_not_recorded(unsandboxed: Executor, work: Path) -> None:
    """A command line can carry a token, and the journal is durable."""
    secret = "deploy --token=hunter2"
    request = request_for(work, ("echo", secret))
    payload = Executor.exec_event(
        request, unsandboxed.run(request, operation_id="op-1"), "op-1"
    )
    assert "hunter2" not in repr(payload)
    assert len(str(payload["argv_sha"])) == 64


def test_network_use_is_recorded_as_a_fact(unsandboxed: Executor, work: Path) -> None:
    """Whether a command *wanted* the network is a fact about the request, so the event
    records it even though this backend refuses to run an allowlist it cannot enforce."""
    wants_network = request_for(work, ("echo", "hi"), network=NetworkPolicy(open=True))
    ran = unsandboxed.run(request_for(work, ("echo", "hi")), operation_id="op-1")
    payload = Executor.exec_event(wants_network, ran, "op-1")
    assert payload["network"] is True


def test_violations_travel_with_the_event(unsandboxed: Executor, work: Path) -> None:
    request = request_for(work, ("sh", "-c", "echo x > stray.txt"), PolicyClass.ACTION)
    result = unsandboxed.run(request, operation_id="op-1")
    payload = Executor.exec_event(request, result, "op-1")
    assert payload["violations"] == ("stray.txt",)


def test_a_timeout_is_visible_in_the_event(unsandboxed: Executor, work: Path) -> None:
    request = request_for(work, ("sleep", "30"), timeout_s=1)
    payload = Executor.exec_event(
        request, unsandboxed.run(request, operation_id="op-1"), "op-1"
    )
    assert payload["timed_out"] is True
