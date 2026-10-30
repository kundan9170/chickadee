"""Invariant: no model can write `.git/config`, `.git/hooks` or `.git/info` (T-3.158 F).

Every command grant reads state its line does not show. A granted `git push origin main`
pushes wherever `.git/config` says `origin` is (`remote.*.url`, `url.*.insteadOf`, a configured
mirror, `core.sshCommand`) and runs `.git/hooks/pre-push` with the network on; `.git/info`
holds the excludes and attributes git applies without being asked. A grant binds a remote's
name and the declared hosts, not where the name points or what a hook runs, so it is safe
only while no command a model proposes can change those files. This test states that once,
for every way a model's command runs:

- **every shell profile** `chickadee.shell` defines, found from the module rather than listed
  here, so a profile added later (the rebuild's `BOX`) is covered the day it lands. Each runs
  a command the way it runs every command: sealed, as a recorded write with the worktree
  writable (Zeus's), or in the box's writable worktree. None reaches git's
  own files, including a command that writes a worktree file and git's in one line. Declared
  writes (`effects=`), which this test also covered, are retired;
- **a job**: a command that runs on past its call, after its step's guard has looked.
  Sandboxed, git's settings are read-only to it; approved, it runs outside the
  sandbox, and its exit puts back what it changed (decision 2);
- **the scout's escapes**, which run under `DISCOVERY`.

Each attempt must leave the three paths byte-identical. It runs where a sandbox backend
exists; elsewhere every command is T2 and confirmed, which `zeus/test_no_write.py` pins.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from repos import git, init_repo
from tool_args import checked

import chickadee.shell as shell_module
from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.executor.types import ExecRequest, PolicyClass
from chickadee.isolation.base import has
from chickadee.isolation.linux import BINARY
from chickadee.shell import Shell, ShellClaim, ShellProfile
from chickadee.zeus.jobs import JobLimits, JobReport, JobTable
from chickadee.zeus.writes import ShellWrites

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")

ATTEMPTS: Final = (
    "echo '[core] sshCommand = touch pwned' >> .git/config",
    "printf '#!/bin/sh\\n' > .git/hooks/pre-push",
    "touch .git/hooks/pre-push",
    "git config remote.origin.url https://elsewhere.example/x",
    "echo '*.py' >> .git/info/exclude",
    # A command that writes the worktree, as a writing one may, and git's file in the same line.
    "echo note > notes.txt; echo '[core] sshCommand = touch pwned' >> .git/config",
    # What a tool outside the sandbox runs from the worktree: the hooks
    # folder `core.hooksPath` names, and an editor's tasks.
    "echo 'touch pwned' >> .husky/pre-commit",
    "touch .husky/post-checkout",
    "echo '{\"tasks\": []}' > .vscode/tasks.json",
    "touch .idea/runConfigurations.xml",
)

_TOOL_DIRS: Final = (".husky", ".vscode", ".idea")

PROFILES: Final = tuple(
    sorted(
        (v for v in vars(shell_module).values() if isinstance(v, ShellProfile)),
        key=lambda p: p.name,
    )
)


class _Approves:
    """The person, saying yes to everything: the wall, not the question, is under test."""

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        return True


def _state(repo: Path) -> dict[str, bytes]:
    git_dir = repo / ".git"
    out: dict[str, bytes] = {"config": (git_dir / "config").read_bytes()}
    for top in (git_dir / "hooks", git_dir / "info", *(repo / d for d in _TOOL_DIRS)):
        for path in sorted(top.rglob("*")):
            out[str(path.relative_to(repo))] = path.read_bytes() if path.is_file() else b"<dir>"
    return out


@pytest.fixture
def repo() -> Iterator[Path]:
    """A repository outside `/tmp`, which the sealed wrap replaces with its own."""
    base = Path("/var/tmp") if Path("/var/tmp").is_dir() else Path(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="git-dir-ro-", dir=base))
    try:
        init_repo(
            root / "repo",
            {
                "app.py": "x = 1\n",
                ".husky/pre-commit": "npm test\n",
                ".vscode/settings.json": "{}\n",
                ".idea/workspace.xml": "<project/>\n",
            },
        )
        # The person uses husky: git runs `.husky/`'s scripts on their commits.
        git(root / "repo", "config", "core.hooksPath", ".husky")
        (root / "repo" / ".git" / "info").mkdir(exist_ok=True)
        (root / "repo" / ".git" / "info" / "exclude").write_text("")
        yield root / "repo"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _shell(tmp_path: Path, repo: Path, profile: ShellProfile) -> tuple[Shell, Session]:
    session = Session.create(tmp_path / "sessions", "S-gitdir")
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=repo, session_blob=session.blob),
        vision=Vision("z"),
    )
    built = Shell(
        Executor(),
        repo,
        lambda data: reader.store("output", data),
        profile=profile,
        authority=_Approves() if profile.may_ask else None,
        # Zeus's profile records each writing command as a step; the others
        # are never given a place to record one.
        writes=ShellWrites(session, repo, commits=True)
        if profile.declarations == "honoured"
        else None,
    )
    return built, session


def _job_shell(tmp_path: Path, repo: Path) -> tuple[Shell, JobTable, Session]:
    """Zeus's shell with its job table, as the engine composes it."""
    session = Session.create(tmp_path / "sessions", "S-gitdir-jobs")
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=repo, session_blob=session.blob),
        vision=Vision("z"),
    )
    table = JobTable(session, limits=JobLimits(yield_s=1))
    built = Shell(
        Executor(),
        repo,
        lambda data: reader.store("output", data),
        authority=_Approves(),
        writes=ShellWrites(session, repo, commits=True, beside=table),
        jobs=table,
    )
    return built, table, session


def _ended(table: JobTable, job_id: str) -> JobReport:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        view = next((v for v in table.views() if v.job_id == job_id), None)
        if view is not None and view.report is not None:
            entry = table.entry(job_id)
            if entry is None or entry.recorded.wait(10):
                return view.report
        time.sleep(0.05)
    raise AssertionError(f"{job_id} never ended")


def test_the_profiles_are_found_from_the_module() -> None:
    """`worker` was `shell.WORKER`'s, deleted (the second review of #640); `agent` is the
    profile with no authority now."""
    names = {p.name for p in PROFILES}
    assert {"zeus", "box", "agent"} <= names, names


@bwrap_only
# A command that also tampered with git's files fails, and its step puts back every path it
# changed, `notes.txt` too: the harness's own write, not the model's (`git_guard`).
@pytest.mark.writes_worktree("*")
@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.name)
@pytest.mark.parametrize("command", ATTEMPTS)
def test_no_command_can_write_git_internals(
    tmp_path: Path, repo: Path, profile: ShellProfile, command: str
) -> None:
    sh, session = _shell(tmp_path, repo, profile)
    before = _state(repo)
    sh.run(checked("shell", {"command": command}), operation_id="S:c1")
    session.close()
    assert _state(repo) == before, f"{profile.name} wrote git's own files with {command!r}"


@bwrap_only
@pytest.mark.writes_worktree("*")
@pytest.mark.parametrize("command", ATTEMPTS)
def test_a_job_cannot_write_git_internals_after_its_yield(
    tmp_path: Path, repo: Path, command: str
) -> None:
    """The write comes after the yield, when the step's guard has already looked: the wall
    holds it, so there is nothing for the exit to put back."""
    sh, table, session = _job_shell(tmp_path, repo)
    before = _state(repo)
    sh.run(
        checked("shell", {"command": f"sleep 1; {command}", "background": True}),
        operation_id="S:c1",
    )
    report = _ended(table, "j1")
    table.close()
    session.close()
    assert _state(repo) == before, f"a job wrote git's own files with {command!r}"
    assert report.git_restored == (), "held by the wall, not put back after"


@pytest.mark.writes_worktree("*")
def test_an_approved_job_outside_the_sandbox_has_git_s_settings_put_back_at_its_exit(
    tmp_path: Path, repo: Path
) -> None:
    """No wall holds a command run outside the sandbox: its exit is the backstop."""
    sh, table, session = _job_shell(tmp_path, repo)
    before = _state(repo)
    line = "sleep 1; echo '[core] sshCommand = touch pwned' >> .git/config"
    sh.run(
        checked("shell", {"command": line, "background": True, "sandbox": False}),
        operation_id="S:c1",
    )
    report = _ended(table, "j1")
    table.close()
    told = [t.text for t in session.state.turns() if t.notice is not None]
    session.close()
    assert _state(repo) == before
    assert report.git_restored and report.git_restored[0].endswith("config")
    assert any("changed git's own settings" in t for t in told)


@bwrap_only
def test_a_walled_job_leaves_the_person_s_own_change_alone(tmp_path: Path, repo: Path) -> None:
    """A wall held it, so it changed nothing of git's: what changed while it ran is the
    person's, and its exit puts nothing back."""
    sh, table, session = _job_shell(tmp_path, repo)
    sh.run(checked("shell", {"command": "sleep 2", "background": True}), operation_id="S:c1")
    git(repo, "remote", "add", "upstream", "https://example.com/up.git")  # in their terminal
    report = _ended(table, "j1")
    table.close()
    session.close()
    assert report.git_restored == ()
    assert git(repo, "config", "remote.upstream.url") == "https://example.com/up.git"


@bwrap_only
@pytest.mark.parametrize("command", ATTEMPTS)
def test_the_scouts_escapes_cannot_write_git_internals(repo: Path, command: str) -> None:
    before = _state(repo)
    Executor().run(
        ExecRequest(command=command, cwd=repo, policy=PolicyClass.DISCOVERY, timeout_s=30),
        operation_id="scout-1",
    )
    assert _state(repo) == before, f"a DISCOVERY run wrote git's own files with {command!r}"
