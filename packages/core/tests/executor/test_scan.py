"""Fix 3: a command whose class does not enforce effects takes no stat scan.
`Executor.run` discarded that scan's result for `VERIFICATION` and `HUMAN` already, so the
result is the same and a large tree is not walked twice per command. A class that enforces
its effects (`ACTION`, `DISCOVERY`) is scanned as before."""

from __future__ import annotations

from pathlib import Path

import pytest
from executor_helpers import request_for

from chickadee.executor import effects as effect_mod
from chickadee.executor.executor import Executor
from chickadee.executor.types import PolicyClass

WRITE = ("sh", "-c", "echo x > undeclared.txt")


@pytest.mark.parametrize("policy", [PolicyClass.WORKTREE, PolicyClass.HUMAN])
def test_an_unenforced_class_reports_no_violation(
    unsandboxed: Executor, work: Path, policy: PolicyClass
) -> None:
    result = unsandboxed.run(request_for(work, WRITE, policy), operation_id="op-1")
    assert result.exit_code == 0
    assert result.violations == ()
    assert (work / "undeclared.txt").is_file()


def test_an_enforced_class_still_reports_its_undeclared_write(
    unsandboxed: Executor, work: Path
) -> None:
    result = unsandboxed.run(request_for(work, WRITE, PolicyClass.ACTION), operation_id="op-2")
    assert [(v.path, v.reason) for v in result.violations] == [("undeclared.txt", "created")]


@pytest.mark.parametrize(
    ("policy", "scans"),
    [
        (PolicyClass.WORKTREE, 0),
        (PolicyClass.HUMAN, 0),
        (PolicyClass.ACTION, 2),
    ],
)
def test_only_an_enforced_class_walks_the_tree(
    unsandboxed: Executor,
    work: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: PolicyClass,
    scans: int,
) -> None:
    taken: list[Path] = []
    real = effect_mod.snapshot_paths

    def counted(root: Path) -> dict[str, effect_mod.Fingerprint]:
        taken.append(root)
        return real(root)

    monkeypatch.setattr(effect_mod, "snapshot_paths", counted)
    unsandboxed.run(request_for(work, ("true",), policy), operation_id="op-3")
    assert len(taken) == scans
