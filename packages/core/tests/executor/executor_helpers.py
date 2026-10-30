"""Plain helpers for the executor suite.

Deliberately **not** in `conftest.py`. Every suite has a module by that name,
Python caches modules by name, and pytest puts each test directory on `sys.path` —
so `from conftest import...` resolves to whichever loaded first and naming two
suites in one invocation collected the wrong one.
"""

from __future__ import annotations

from pathlib import Path

from chickadee.executor.types import ExecRequest, NetworkPolicy, PolicyClass


def request_for(
    cwd: Path,
    command: str | tuple[str, ...],
    policy: PolicyClass = PolicyClass.WORKTREE,
    **kw: object,
) -> ExecRequest:
    """Build an `ExecRequest` with test-friendly defaults.

    Unknown keywords **raise**. They used to be discarded, so `timeot_s=1` built the request
    with the default timeout and the test asserted against a configuration it never asked
    for — silently, and while reading as though it had.
    """
    timeout_s = int(kw.pop("timeout_s", 30))  # type: ignore[arg-type]
    network = kw.pop("network", NetworkPolicy())
    effects = kw.pop("effects", ())
    idempotent = bool(kw.pop("idempotent", False))
    label = str(kw.pop("label", ""))
    if kw:
        raise TypeError(f"request_for got unexpected keyword(s): {sorted(kw)}")
    return ExecRequest(
        command=command,
        cwd=cwd,
        policy=policy,
        timeout_s=timeout_s,
        network=network,  # type: ignore[arg-type]
        effects=effects,  # type: ignore[arg-type]
        idempotent=idempotent,
        label=label,
    )
