"""What an approved command runs under.

Pinned in older versions: an approved command was wrapped exactly as an unasked one, the network
its
only difference, and the question said nothing about the sandbox. Moved with T-3.178, listed in
its Log: approved, it runs outside the sandbox, unwrapped, with the person's environment less
the harness's keys; the question says so and what it reaches; the header says what is not
recorded.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

import pytest
from box_helpers import FakeRunJournal
from repos import init_repo
from sandboxes import Enforcing, Scripted
from tool_args import checked

from chickadee.box.report import stored_handle
from chickadee.box.shell import BoxShell
from chickadee.context.results import ToolResult
from chickadee.executor.executor import Executor
from chickadee.executor.types import ExecRequest, NetworkPolicy, PolicyClass
from chickadee.isolation.none import Passthrough
from chickadee.journal.port import JournalPort
from chickadee.schema.authorization import Tier
from chickadee.shell import AGENT, OUTSIDE, RefusedAboveT0, Shell, ShellClaim
from chickadee.zeus.tools.talk import REACHES, approval_of


class _PersonSays:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return self.answer


def _box(
    root: Path, *, answer: bool = True, withheld: frozenset[str] = frozenset()
) -> tuple[BoxShell, Enforcing, Scripted, _PersonSays]:
    backend, supervisor, authority = Enforcing(), Scripted(b"ok\n"), _PersonSays(answer)
    shell = BoxShell(
        Executor(backend, supervisor=supervisor, withheld_env=withheld),
        root,
        cast(JournalPort, FakeRunJournal()),
        authority,
        asker="athena",
        task=lambda: "T1",
    )
    return shell, backend, supervisor, authority


def _run(shell: BoxShell, command: str, **extra: object) -> Any:
    args = checked("shell", {"command": command, **extra})
    return shell.run(args, operation_id="op-1", cancel=None)


def test_an_approved_install_runs_outside_the_sandbox_with_the_persons_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("GITHUB_TOKEN", "the person's own")
    monkeypatch.setenv("FRONTIER_API_KEY", "the harness's")
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell, backend, supervisor, authority = _box(root, withheld=frozenset({"FRONTIER_API_KEY"}))
    done = _run(shell, "pip install pygame")
    assert isinstance(done, ToolResult)
    assert [c.tier for c in authority.claims] == [Tier.T1]
    assert backend.seen == [], "nothing wraps it"
    assert supervisor.env["HTTPS_PROXY"] == "http://proxy.local:3128"
    assert supervisor.env["GITHUB_TOKEN"] == "the person's own"
    assert "FRONTIER_API_KEY" not in supervisor.env
    assert "sandbox none · network on" in done.header
    assert "outside the sandbox, approved; effects outside the worktree are not recorded" in (
        done.header
    )


def test_a_command_nobody_is_asked_about_stays_in_the_sandbox(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell, backend, _, authority = _box(root)
    done = _run(shell, "pytest -q")
    assert isinstance(done, ToolResult)
    assert authority.claims == []
    [wrap] = backend.seen
    assert wrap.network_off and "sandbox native" in done.header


def test_sandbox_false_asks_at_t2_and_runs_outside_once_approved(tmp_path: Path) -> None:
    """A command nobody would be asked about, which hit the sandbox."""
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell, backend, _, authority = _box(root)
    done = _run(shell, "make setup", sandbox=False)
    assert isinstance(done, ToolResult)
    [claim] = authority.claims
    assert (claim.tier, claim.classification) == (Tier.T2, OUTSIDE)
    assert backend.seen == []
    assert shell.opens_network("make setup", False, True), "resume never runs it again"
    assert not shell.opens_network("make test", False), "a T0 command ran sealed"


def test_declined_nothing_runs(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell, backend, supervisor, _ = _box(root, answer=False)
    done = _run(shell, "pip install pygame")
    assert isinstance(done, RefusedAboveT0)
    assert backend.seen == [] and supervisor.spawned == 0


def test_a_profile_with_no_authority_refuses_to_leave(tmp_path: Path) -> None:
    """Was the packet worker's `WORKER`, deleted with nothing left using it (the second review
    of #640); `AGENT`, a review agent's, is the profile with no authority now."""
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell = Shell(
        Executor(Enforcing(), supervisor=Scripted()),
        root,
        lambda data: stored_handle("output", "run-1", data),
        profile=AGENT,
    )
    args = checked("shell", {"command": "make", "sandbox": False})
    refused = shell.run(args, operation_id="op-1")
    assert isinstance(refused, RefusedAboveT0) and refused.classification == OUTSIDE


def test_require_sandbox_never_refuses_an_approved_run(tmp_path: Path) -> None:
    """EX-2 is about what runs unasked: this one was asked."""
    supervisor = Scripted(b"ok\n")
    executor = Executor(Passthrough(), require_sandbox=True, supervisor=supervisor)
    request = ExecRequest(
        command="pip install pygame",
        cwd=tmp_path,
        policy=PolicyClass.WORKTREE,
        timeout_s=30,
        network=NetworkPolicy(True),
        unsandboxed=True,
    )
    result = executor.run(request, operation_id="op-1")
    assert (result.exit_code, result.sandbox.value, result.network_enforced) == (
        0,
        "none",
        "on",
    )
    assert supervisor.spawned == 1


def test_a_secret_named_in_an_approved_command_is_still_refused(tmp_path: Path) -> None:
    supervisor = Scripted()
    executor = Executor(Passthrough(), supervisor=supervisor)
    request = ExecRequest(
        command="cat .env",
        cwd=tmp_path,
        policy=PolicyClass.WORKTREE,
        timeout_s=30,
        unsandboxed=True,
    )
    result = executor.run(request, operation_id="op-1")
    assert result.exit_code == 126 and supervisor.spawned == 0


def test_the_question_says_it_runs_outside_and_what_that_reaches() -> None:
    claim = ShellClaim(
        "pip install pygame",
        ".",
        Tier.T1,
        "lockfile_install",
        True,
        operation_id="op-1",
        idempotent=False,
    )
    asked = approval_of(claim)
    assert asked.display == (
        "pip install pygame  (in . · outside the sandbox · network on; lockfile_install at t1)"
    )
    assert asked.effect_surface == (REACHES,)
    assert REACHES.startswith("anything you can")


class _PlantsAHook(Scripted):
    """A command that, run outside the sandbox, writes git's own config: what an approved
    install's postinstall script could do."""

    def __init__(self, root: Path) -> None:
        super().__init__(b"installed\n")
        self.root = root

    def run(self, argv: tuple[str, ...], **kwargs: Any) -> Any:
        (self.root / ".git" / "config").write_text("[core]\n\tfsmonitor = ./evil\n")
        return super().run(argv, **kwargs)


def test_git_files_an_approved_command_changed_are_put_back(tmp_path: Path) -> None:
    """The second review of T-3.178: `.git` was read-only only inside the sandbox."""
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    config = (root / ".git" / "config").read_text()
    authority = _PersonSays()
    shell = BoxShell(
        Executor(Enforcing(), supervisor=_PlantsAHook(root)),
        root,
        cast(JournalPort, FakeRunJournal()),
        authority,
        asker="worker",
        task=lambda: "T1",
    )
    done = _run(shell, "npm install")
    assert isinstance(done, ToolResult)
    assert (root / ".git" / "config").read_text() == config
    said = done.content or ""
    assert "effect check: changed" in said and "put back" in said


def test_an_unrecorded_shell_says_nothing_it_changed_is_recorded(tmp_path: Path) -> None:
    """The second review: a shell with nowhere to record writes never implies a record."""
    shell = Shell(
        Executor(Enforcing(), supervisor=Scripted(b"ok\n")),
        tmp_path,
        lambda data: stored_handle("output", "run-1", data),
        authority=_PersonSays(),
    )
    args = checked("shell", {"command": "pip install pygame"})
    done = shell.run(args, operation_id="op-1")
    assert isinstance(done, ToolResult)
    assert "outside the sandbox, approved; nothing it changed is recorded" in done.header
