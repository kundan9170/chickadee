"""Pins before standing preferences are wired into the shell: the tier
the shell gives each command class with no preferences, under a sandbox that seals.

What these pin: a command of each class `06-approval.md` §6.3 names is classified as that class
by the shipped table, runs at the class's floor, and is put to the person exactly when that
floor is above T0. Item 8 moves some of these answers on purpose (a class set to `ask` or
`deny`, `lockfile_install` allowed); with no preferences none of them moves.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from repos import init_repo
from sandboxes import Enforcing, Scripted
from tool_args import checked

from chickadee.approval.commands import CLASS_TIERS, CommandClass, shipped
from chickadee.context.homes import Scratch
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import ToolResult
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.schema.authorization import Tier
from chickadee.shell import Shell, ShellClaim

LINES: dict[CommandClass, str] = {
    CommandClass.READ_ONLY: "cat a.py",
    CommandClass.WORKTREE_WRITE: "cp a.py b.py",
    CommandClass.TESTS: "pytest -q",
    CommandClass.BUILD: "make",
    CommandClass.DEV_SERVER: "webpack serve",
    CommandClass.LOCKFILE_INSTALL: "npm ci",
    CommandClass.EXTERNAL_FETCH: "curl https://example.com",
    CommandClass.PUSH: "git push origin main",
    CommandClass.DESTRUCTIVE: "rm -rf build",
    CommandClass.CREDENTIAL: "env",
    CommandClass.SCOPE_EXPANDING: "chmod +x run.sh",
}


class _Yes:
    def __init__(self) -> None:
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return True


def test_every_class_has_a_line_here() -> None:
    assert set(LINES) == set(CommandClass)


@pytest.mark.parametrize("cls", list(CommandClass), ids=lambda c: c.value)
def test_with_no_preferences_each_class_runs_at_its_floor(
    tmp_path: Path, cls: CommandClass
) -> None:
    line = LINES[cls]
    found = shipped().classify_line(line)
    assert found is not None and found.cls is cls, f"{line!r} is not {cls.value}"
    floor, _ = CLASS_TIERS[cls]
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    asked = _Yes()
    own = Scratch()
    reader = Reader(own, HandleResolver(root, own.blob), vision=Vision("z"))
    shell = Shell(
        Executor(Enforcing(), supervisor=Scripted(b"ok\n")),
        root,
        lambda data: reader.store("output", data),
        authority=asked,
    )
    done = shell.run(checked("shell", {"command": line}), operation_id="op-1", cancel=None)
    assert isinstance(done, ToolResult)
    assert f" · {floor.value} · op-1" in done.header
    if floor is Tier.T0:
        assert asked.claims == [], "a T0 command is never put to the person"
    else:
        [claim] = asked.claims
        assert (claim.tier, claim.classification) == (floor, cls.value)
