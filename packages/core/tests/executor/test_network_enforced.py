"""D199, `13-executor.md` §5.2: the network is a fact with two values.

`off` when the run was sealed, `on` when the whole network was open: because the call asked for
it (`network: true`), which reaches the executor only approved, or because nothing here could
close it (`sandbox=none`). Every backend is all-or-nothing, so there is no host to enforce and
nothing refuses an approved network after the fact, whatever `require_sandbox` says.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from executor_helpers import request_for

from chickadee.executor.executor import Executor
from chickadee.executor.types import NetworkPolicy, PolicyClass
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.process.supervisor import Completed, ProcessSupervisor

OPEN = NetworkPolicy(open=True)


class _Native:
    """A native-level backend that runs nothing itself: the recording supervisor below is
    what would spawn, so a test can say whether anything did. It records each wrap, so a test
    can say whether the network was closed."""

    def __init__(self) -> None:
        self.wraps: list[Isolation] = []

    def available(self) -> bool:
        return True

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE

    def wrap(self, isolation: Isolation) -> tuple[str, ...]:
        self.wraps.append(isolation)
        return isolation.argv


class _SpawnRecorder(ProcessSupervisor):
    def __init__(self) -> None:
        self.spawned: list[tuple[str, ...]] = []

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
        del cwd, env, timeout_s, stdin
        self.spawned.append(argv)
        return Completed(b"", b"", 0, False)


def _executor(
    backend: _Native, *, require_sandbox: bool = False
) -> tuple[Executor, _SpawnRecorder]:
    supervisor = _SpawnRecorder()
    executor = Executor(backend=backend, supervisor=supervisor, require_sandbox=require_sandbox)
    return executor, supervisor


def test_an_undeclared_run_is_off(work: Path) -> None:
    backend = _Native()
    executor, supervisor = _executor(backend)
    result = executor.run(request_for(work, ("uv", "sync")), operation_id="op-1")
    assert result.network_enforced == "off"
    assert [w.network_off for w in backend.wraps] == [True]
    assert len(supervisor.spawned) == 1


def test_an_open_network_runs_on_and_the_wrap_opens_it(work: Path) -> None:
    """Before D199 a declared host list ran at `all`; now the call says yes or no, and the
    result says the network was on."""
    backend = _Native()
    executor, supervisor = _executor(backend)
    result = executor.run(
        request_for(
            work, ("uv", "sync"), PolicyClass.ACTION, network=OPEN, effects=("uv.lock",)
        ),
        operation_id="op-1",
    )
    assert result.network_enforced == "on"
    assert result.ok
    assert [w.network_off for w in backend.wraps] == [False]
    assert len(supervisor.spawned) == 1


def test_require_sandbox_no_longer_refuses_an_open_network(work: Path) -> None:
    """D199 moved this answer: under `require_sandbox` (the default, and always the box's
    until D199) an approved `network: true` was refused before spawn, `NETWORK_DENIED{heuristic:
    false}`. A sandbox exists here, so the command runs, with the network open."""
    executor, supervisor = _executor(_Native(), require_sandbox=True)
    result = executor.run(request_for(work, ("uv", "sync"), network=OPEN), operation_id="op-1")
    assert result.network_enforced == "on"
    assert len(supervisor.spawned) == 1


def test_the_prefilters_refusal_carries_the_field_too(work: Path) -> None:
    """A refusal before spawn is still a result a caller reads, and every result says how its
    network was, or would have been."""
    executor, supervisor = _executor(_Native())
    result = executor.run(request_for(work, ("cat", ".env"), network=OPEN), operation_id="op-1")
    assert result.exit_code == 126
    assert result.network_enforced == "on"
    assert supervisor.spawned == []
