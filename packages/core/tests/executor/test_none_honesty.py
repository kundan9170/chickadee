"""EX-2 — `sandbox=none` is reported, never silent; `require_sandbox` aborts instead."""

from __future__ import annotations

from pathlib import Path

import pytest
from executor_helpers import request_for

from chickadee.errors import InfraError
from chickadee.executor.executor import Executor
from chickadee.executor.policy import isolation_for
from chickadee.executor.types import NetworkPolicy
from chickadee.isolation.base import select
from chickadee.isolation.none import Passthrough, reads_unmasked
from chickadee.isolation.types import SandboxLevel


def test_an_unsandboxed_run_says_so_in_the_result(unsandboxed: Executor, work: Path) -> None:
    """Invariant 6 applied to isolation: an un-sandboxed run is a fact about the result."""
    result = unsandboxed.run(request_for(work, ("echo", "hi")), operation_id="op-1")
    assert result.sandbox is SandboxLevel.NONE


def test_require_sandbox_aborts_rather_than_degrading(work: Path) -> None:
    """The eval default (D22 sealing). A number produced without the isolation it claims is
    worse than no number."""
    executor = Executor(backend=Passthrough(), require_sandbox=True)
    with pytest.raises(InfraError, match="require_sandbox"):
        executor.run(request_for(work, ("echo", "hi")), operation_id="op-1")


def test_availability_is_not_application(work: Path) -> None:
    """`select()` reports what the machine could enforce; the result reports what a run got.
    Conflating them is how "the sandbox held" becomes a claim nobody checked."""
    backend = select()
    assert backend.available()
    assert backend.level() in set(SandboxLevel)


def test_the_passthrough_is_always_available() -> None:
    """There is always somewhere honest to fall back to, so the fallback never has to lie."""
    assert Passthrough().available()
    assert Passthrough().level() is SandboxLevel.NONE


def test_at_none_an_open_network_is_on_and_reads_are_unmasked(work: Path) -> None:
    """EX-2's honesty, both facts at once: nothing wrapped the run, so the network was open
    and the mask did not hold. The result carries both — `network_enforced`,
    and `sandbox=none`, which is what `reads_unmasked` reads."""
    secret = work / ".env"
    secret.write_text("KEY=1\n")
    executor = Executor(backend=Passthrough(), deny_read=(secret,))
    request = request_for(work, ("echo", "hi"), network=NetworkPolicy(open=True))
    result = executor.run(request, operation_id="op-1")
    assert result.network_enforced == "on"
    assert result.sandbox is SandboxLevel.NONE
    isolation = isolation_for(request, deny_read=executor.deny_read, denylist=executor.denylist)
    assert reads_unmasked(result.sandbox, isolation)


def test_at_none_even_an_undeclared_network_is_on(unsandboxed: Executor, work: Path) -> None:
    """`off` says no socket opens, and at `sandbox=none` nothing closed one: an undeclared
    request is not a sealed run when there is no seal."""
    result = unsandboxed.run(request_for(work, ("echo", "hi")), operation_id="op-1")
    assert result.network_enforced == "on"


def test_an_explicit_none_preference_is_honoured() -> None:
    """`none` must mean "run unwrapped and say so", not "fall through to whatever this
    machine has" — which is the opposite of what was asked."""
    assert select(preferred="none").level() is SandboxLevel.NONE


def test_an_unrestricted_network_is_allowed_through(unsandboxed: Executor, work: Path) -> None:
    """Nothing refuses a run for its network at `none`: there is nothing to enforce."""
    result = unsandboxed.run(request_for(work, ("echo", "hi")), operation_id="op-1")
    assert result.ok
