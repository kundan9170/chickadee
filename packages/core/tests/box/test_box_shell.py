"""The box shell: `Shell` under `BOX` for the task worker and Athena.

What these pin: the worktree is writable from any `cwd` and its `.git` is not; `network: true`
is T2 and is put to the session's authority, naming who asks and for which task; unattended, a
call above T0 is refused; `effects=` is refused; every command is one `exec` line under the
verification class, and its output a run blob; no shell where nothing enforces a seal.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from box_helpers import FakeRunJournal
from repos import init_repo
from sandboxes import Enforcing, Scripted
from tool_args import checked

from chickadee.approval.grants import GrantingPort
from chickadee.approval.prefs import merge, parse_level
from chickadee.approval.tier import Preferences
from chickadee.box.shell import BoxShell, RunSuspended, shell_for
from chickadee.config import ShellLift
from chickadee.context.questions import RecordingPort
from chickadee.context.results import ToolResult
from chickadee.context.session import Session
from chickadee.errors import HarnessBug
from chickadee.executor.executor import Executor
from chickadee.isolation.base import has
from chickadee.isolation.linux import BINARY, Bubblewrap
from chickadee.isolation.none import Passthrough
from chickadee.journal.port import JournalPort
from chickadee.process.supervisor import Completed
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    ApprovalDecision,
    ApprovalStop,
    Question,
    StopDecision,
)
from chickadee.schema.journal import EventKind, ExecP
from chickadee.shell import BOX, PreferenceDenied, RefusedAboveT0, ShellClaim, ShellCommand
from chickadee.state.roots import Roots
from chickadee.tools.spec import spec
from chickadee.tools.validate import InvalidArguments, validate
from chickadee.zeus.tools.talk import PersonAuthority

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")


class _Authority:
    """An authority that records being asked and answers `grants`."""

    def __init__(self, grants: bool) -> None:
        self.grants = grants
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return self.grants


def _shell(
    root: Path,
    executor: Executor,
    authority: _Authority | None = None,
    *,
    prefs: Callable[[], Preferences] | None = None,
    lifted: bool = False,
) -> tuple[BoxShell, FakeRunJournal]:
    journal = FakeRunJournal()
    shell = BoxShell(
        executor,
        root,
        cast(JournalPort, journal),
        authority,
        asker="worker",
        task=lambda: "T1",
        prefs=prefs,
        allow_unsandboxed_shell=lifted,
    )
    return shell, journal


def _unsealed() -> Executor:
    """No sandbox, and the person's `require_sandbox` off: the box shell runs unsealed."""
    return Executor(Passthrough(), supervisor=Scripted(b"ok\n"))


def _scripted(journal: FakeRunJournal | None = None) -> Executor:
    put = journal.put_blob if journal is not None else None
    return Executor(Enforcing(), put_blob=put, supervisor=Scripted(b"ok\n"))


def _args(command: str, **extra: object) -> Any:
    return checked("shell", {"command": command, **extra})


def test_the_box_profile_writes_the_worktree_honours_the_network_and_asks() -> None:
    assert (BOX.declarations, BOX.may_ask, BOX.writes) == ("network_only", True, "worktree")


def test_every_command_is_one_exec_line_under_the_verification_class(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell, journal = _shell(root, _scripted())
    done = shell.run(_args("make test"), operation_id="op-1", cancel=None)
    assert isinstance(done, ToolResult)
    assert journal.kinds() == [EventKind.EXEC]
    line = journal.lines[0][1]
    assert isinstance(line, ExecP)
    assert (line.operation_id, line.policy, line.network) == ("op-1", "worktree", False)


class _TimesOut(Scripted):
    def run(self, argv: Any, **kwargs: Any) -> Completed:
        del argv, kwargs
        return Completed(b"partial\n", b"", -9, True)


class _LeftMidRun(Scripted):
    """The person leaves while the command runs: the pause, then the hard stop that reaps it."""

    def __init__(self, suspend: threading.Event, cancel: threading.Event) -> None:
        super().__init__()
        self.suspend, self.cancel = suspend, cancel

    def run(self, argv: Any, **kwargs: Any) -> Completed:
        del argv, kwargs
        self.suspend.set()
        self.cancel.set()
        return Completed(b"half\n", b"", -9, False)


def test_a_command_reaped_by_the_pause_leaves_its_call_unanswered(tmp_path: Path) -> None:
    """Whatever the reaped command returned, the call is cut off as a kill cuts it, so
    its step stays open for resume; it is never answered "cancelled" as if it had ended."""
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    suspend, cancel = threading.Event(), threading.Event()
    journal = FakeRunJournal()
    executor = Executor(
        Enforcing(), put_blob=journal.put_blob, supervisor=_LeftMidRun(suspend, cancel)
    )
    shell = BoxShell(
        executor,
        root,
        cast(JournalPort, journal),
        None,
        asker="worker",
        task=lambda: "T1",
        suspend=suspend,
    )
    with pytest.raises(RunSuspended):
        shell.run(_args("make test"), operation_id="op-1", cancel=cancel)
    assert journal.kinds() == [EventKind.EXEC], "it ran, and its exec line says so"


def test_a_timed_out_commands_first_line_is_still_its_header_and_handle(
    tmp_path: Path,
) -> None:
    """An evicted result keeps its first line (`evict.live_receipt`), so the box's header,
    naming the output's handle, stays first when a command times out."""
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    executor = Executor(Enforcing(), put_blob=None, supervisor=_TimesOut())
    shell, _ = _shell(root, executor)
    done = shell.run(_args("pytest -q"), operation_id="op-1", cancel=None)
    assert isinstance(done, ToolResult)
    first, second = done.render().split("\n")[:2]
    assert first.startswith("exit ") and " · output " in first
    assert second.startswith("TIMED_OUT")


def test_the_root_is_writable_from_any_cwd_and_git_is_kept(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"sub/a.py": "a = 1\n"})
    shell, _ = _shell(root, _scripted())
    request = shell._shell.request(  # pyright: ignore[reportPrivateUsage]
        shell._shell.parse(_args("make", cwd="sub"))  # type: ignore[arg-type] # pyright: ignore[reportPrivateUsage, reportArgumentType]
    )
    assert request.also_writable == (root.resolve(),)
    assert request.read_only == (root.resolve() / ".git",)


def test_effects_are_refused_as_retired() -> None:
    """No shell takes `effects` now, the box's as before."""
    found = spec("shell")
    assert found is not None
    got = validate(found, {"command": "make", "effects": ["out.txt"]})
    assert isinstance(got, InvalidArguments)
    assert [v.field for v in got.violations] == ["effects"]


def test_a_declared_network_is_t2_and_names_the_asker_and_the_task(tmp_path: Path) -> None:
    asked = _Authority(grants=True)
    shell, journal = _shell(tmp_path, _scripted(), asked)
    done = shell.run(_args("pip download x", network=True), operation_id="op-3", cancel=None)
    assert isinstance(done, ToolResult)
    [claim] = asked.claims
    assert (claim.tier, claim.asker, claim.task_id) == (Tier.T2, "worker", "T1")
    assert journal.kinds() == [EventKind.EXEC]


def test_an_approved_install_gets_the_network_from_its_class(tmp_path: Path) -> None:
    """D199 item 12, the box's shell alike: the worker need not set `network: true`. Moved
    with T-3.178: approved, it is no longer wrapped, so nothing seals it."""
    backend = Enforcing()
    asked = _Authority(grants=True)
    shell, _ = _shell(tmp_path, Executor(backend, supervisor=Scripted(b"ok\n")), asked)
    done = shell.run(_args("npm ci"), operation_id="op-1", cancel=None)
    assert isinstance(done, ToolResult) and "network on" in done.header
    [claim] = asked.claims
    assert claim.network and claim.asker == "worker"
    assert backend.seen == [], "approved, it runs outside the sandbox"
    assert "outside the sandbox, approved" in done.header


def test_unattended_a_call_above_t0_is_refused_and_nothing_runs(tmp_path: Path) -> None:
    asked = _Authority(grants=False)
    shell, journal = _shell(tmp_path, _scripted(), asked)
    refused = shell.run(
        _args("curl https://example.com", network=True),
        operation_id="op-4",
        cancel=None,
    )
    assert isinstance(refused, RefusedAboveT0)
    assert journal.lines == []


# -- where no sandbox runs --------------------------------------------------


def test_withheld_where_nothing_seals_and_require_sandbox_is_set(tmp_path: Path) -> None:
    """The person's EX-2 opt-out is the switch. With `require_sandbox` set there
    is no shell, and a box shell built anyway is a harness bug."""
    assert shell_for(tmp_path, Passthrough(), lambda _j: _unsealed(), None) is None
    with pytest.raises(HarnessBug):
        BoxShell(
            Executor(Passthrough(), require_sandbox=True),
            tmp_path,
            cast(JournalPort, FakeRunJournal()),
            None,
            asker="worker",
            task=lambda: None,
        )


def test_offered_unsealed_on_the_persons_opt_out(tmp_path: Path) -> None:
    """D199 moved this answer: where nothing seals, the worker and Athena had no shell at all.
    With `require_sandbox` off they are offered one, as Zeus is."""
    built = shell_for(
        tmp_path, Passthrough(), lambda _j: _unsealed(), _Authority(True), require_sandbox=False
    )
    assert built is not None
    assert shell_for(tmp_path, Enforcing(), lambda _j: _scripted(), None) is not None


def test_unsealed_every_command_is_asked_naming_who_and_runs_once_approved(
    tmp_path: Path,
) -> None:
    """D199 item 7, D137's floor: an undeclared `pytest` is T2 `unsandboxed`, put to the
    person with the role and the task, and the result says nothing protected the repo. Moved
    with T-3.178: it says the command ran outside, what of it is not recorded."""
    asked = _Authority(grants=True)
    shell, journal = _shell(tmp_path, _unsealed(), asked)
    done = shell.run(_args("pytest -q"), operation_id="op-1", cancel=None)
    [claim] = asked.claims
    assert (claim.tier, claim.classification) == (Tier.T2, "unsandboxed")
    assert (claim.asker, claim.task_id) == ("worker", "T1")
    assert isinstance(done, ToolResult)
    assert "sandbox none" in done.header
    assert "outside the sandbox, approved; effects outside the worktree are not recorded" in (
        done.header
    )
    assert journal.kinds() == [EventKind.EXEC]


def test_unsealed_and_unattended_nothing_asked_runs(tmp_path: Path) -> None:
    """With nobody to ask (the authority answers no), the command is refused and
    nothing spawns."""
    shell, journal = _shell(tmp_path, _unsealed(), _Authority(grants=False))
    refused = shell.run(_args("pytest -q"), operation_id="op-1", cancel=None)
    assert isinstance(refused, RefusedAboveT0)
    assert (refused.tier, refused.classification) == (Tier.T2, "unsandboxed")
    assert journal.lines == []


def test_unsealed_a_denied_class_is_refused_before_anyone_is_asked(tmp_path: Path) -> None:
    """The person's `deny` holds unsealed as sealed, and nobody is asked."""
    asked = _Authority(grants=True)
    denied = merge([parse_level({"approval": {"commands": {"tests": "deny"}}}, "user")])
    shell, journal = _shell(tmp_path, _unsealed(), asked, prefs=lambda: denied)
    refused = shell.run(_args("pytest -q"), operation_id="op-1", cancel=None)
    assert isinstance(refused, PreferenceDenied)
    assert (asked.claims, journal.lines) == ([], [])


def test_the_lift_is_per_role(tmp_path: Path) -> None:
    """`sandbox.allow_unsandboxed_shell.worker` lifts the worker's floor and not
    Athena's: her command is still asked."""
    asked = _Authority(grants=False)
    build = shell_for(
        tmp_path,
        Passthrough(),
        lambda _j: _unsealed(),
        asked,
        require_sandbox=False,
        lifts=ShellLift(worker=True),
    )
    assert build is not None
    journal = cast(JournalPort, FakeRunJournal())
    worker = build(journal, "worker", lambda: "T1")
    athena = build(journal, "athena", lambda: "T1")
    ran = worker.run(_args("pytest -q"), operation_id="op-1", cancel=None)
    assert isinstance(ran, ToolResult) and "· t0 ·" in ran.header
    assert asked.claims == []
    refused = athena.run(_args("pytest -q"), operation_id="op-2", cancel=None)
    assert isinstance(refused, RefusedAboveT0)
    assert [c.asker for c in asked.claims] == ["athena"]


class _Always:
    """The person at the session's port, answering *always this session* to every command."""

    def __init__(self) -> None:
        self.asked: list[ActionApproval] = []

    def ask(self, q: Question, *, cancel: object = None) -> Answer:
        raise AssertionError("not asked here")

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        raise AssertionError("not asked here")

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self.asked.append(approval)
        return ApprovalDecision(
            action_id=approval.action_id, verdict="approve", source="human", scope="session"
        )


def test_unsealed_a_session_grant_covers_the_roles_command_as_it_does_zeuss(
    tmp_path: Path,
) -> None:
    """Unsealed, the worker's commands reach the person through the session's
    grants (`GrantingPort`), so *always this session* on `pytest -q` stops its questions."""
    repo = tmp_path / "repo"
    roots = Roots(
        project=repo / ".chickadee",
        repo_local=repo / ".git" / "chickadee",
        checkout_local=repo / ".git" / "chickadee",
        user=tmp_path / "user",
    )
    session = Session.create(tmp_path / "sessions", "s-box")
    person = _Always()
    port = GrantingPort(RecordingPort(person, session), session, roots)
    authority = PersonAuthority(port.for_run("run-1"), lambda: None)
    shell, journal = _shell(tmp_path, _unsealed(), cast(Any, authority))
    for n in (1, 2):
        done = shell.run(_args("pytest -q"), operation_id=f"op{n}", cancel=None)
        assert isinstance(done, ToolResult)
    assert [(a.asker, a.task_id) for a in person.asked] == [("worker", "T1")]
    assert journal.kinds() == [EventKind.EXEC, EventKind.EXEC]
    session.close()


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ("git commit -m x", "git state"),
        ("git update-ref refs/heads/main HEAD~1", "git state"),
        ("git config core.hooksPath /tmp/h", "git state"),
        ("touch /tmp/x", "outside the repository"),
        ("touch ../x", "outside the repository"),
        ("touch ~/x", "outside the repository"),
        # The review's four: a redirect, a write's operand, a `cd`, a `.git` path.
        ("echo x > ~/.bashrc", "outside the repository"),
        ("sed -i s/a/b/ /etc/hosts", "outside the repository"),
        ("cd .. && touch x", "outside the repository"),
        ("cp a .git/hooks/pre-commit", "git state"),
        ("printf x >> .git/config", "git state"),
        # The second review: a build or test run writes where its operands say, a
        # shell variable or a glued flag value names a path too.
        ("make -C /elsewhere install", "outside the repository"),
        ("pytest --basetemp=/etc/foo", "outside the repository"),
        ('cp a "$HOME/.bashrc"', "outside the repository"),
        ("cc -o/usr/local/bin/x a.c", "outside the repository"),
    ],
)
def test_the_lift_never_covers_git_state_or_a_path_outside_the_repository(
    tmp_path: Path, line: str, why: str
) -> None:
    """Unsealed, nothing stops either, so each stays asked under the lift."""
    asked = _Authority(grants=False)
    shell, journal = _shell(tmp_path, _unsealed(), asked, lifted=True)
    refused = shell.run(_args(line), operation_id="op-1", cancel=None)
    assert isinstance(refused, RefusedAboveT0), refused
    assert (refused.tier, refused.classification) == (Tier.T2, f"unsandboxed, {why}")
    assert journal.lines == []


@pytest.mark.parametrize(
    "line",
    [
        "pytest -q",
        "touch build/x",
        "git status",
        "git log -1",
        "sed -i s/a/b/ src/x.py",
        "make 2>&1 | tail -5",
        "echo done > out.txt",
        "cat /etc/hosts",
        "touch {root}/build/y",
        'printf %s "$VALUE" > out.txt',
    ],
)
def test_the_lift_covers_what_stays_in_the_worktree(tmp_path: Path, line: str) -> None:
    """A read anywhere and a write the line keeps in the worktree, an absolute path into the
    workspace included, stay under the lift."""
    asked = _Authority(grants=False)
    shell, _ = _shell(tmp_path, _unsealed(), asked, lifted=True)
    line = line.format(root=tmp_path.resolve())
    ran = shell.run(_args(line), operation_id="op-1", cancel=None)
    assert isinstance(ran, ToolResult) and "· t0 ·" in ran.header, ran
    assert asked.claims == []


def test_the_lift_never_lowers_the_network(tmp_path: Path) -> None:
    asked = _Authority(grants=False)
    shell, _ = _shell(tmp_path, _unsealed(), asked, lifted=True)
    refused = shell.run(_args("make deps", network=True), operation_id="op-1", cancel=None)
    assert isinstance(refused, RefusedAboveT0)
    assert (refused.tier, refused.classification) == (Tier.T2, "network")


@bwrap_only
def test_bwrap_writes_the_worktree_from_a_subdirectory_and_never_git(
    tmp_path: Path,
) -> None:  # pragma: no cover - platform
    root = init_repo(tmp_path / "repo", {"sub/a.py": "a = 1\n"})
    config = (root / ".git" / "config").read_text(encoding="utf-8")
    journal = FakeRunJournal()
    executor = Executor(Bubblewrap(), require_sandbox=True, put_blob=journal.put_blob)
    shell = BoxShell(
        executor, root, cast(JournalPort, journal), None, asker="worker", task=lambda: "T1"
    )
    wrote = shell.run(
        _args("echo y > ../top.txt && echo z > new.txt", cwd="sub"),
        operation_id="op-5",
        cancel=None,
    )
    assert isinstance(wrote, ToolResult)
    assert (root / "top.txt").read_text(encoding="utf-8") == "y\n"
    assert (root / "sub" / "new.txt").read_text(encoding="utf-8") == "z\n"
    kept = shell.run(_args("echo x >> .git/config"), operation_id="op-6", cancel=None)
    assert isinstance(kept, ToolResult)
    assert (root / ".git" / "config").read_text(encoding="utf-8") == config
    exits = [line.exit_code for kind, line, _, _ in journal.lines if kind is EventKind.EXEC]
    assert exits[0] == 0 and exits[1] != 0


def test_the_shell_takes_a_command_and_no_purpose_line(tmp_path: Path) -> None:
    """D208, moved: the shell refused a blank `why` and kept one in its record;
    it takes neither now."""
    root = init_repo(tmp_path / "repo", {"a.py": "a = 1\n"})
    shell, _ = _shell(root, _scripted())
    parsed = shell._shell.parse(  # pyright: ignore[reportPrivateUsage]
        checked("shell", {"command": "make"})
    )
    assert isinstance(parsed, ShellCommand) and parsed.command == "make"
    assert not hasattr(parsed, "intent")
    # The run record names it by its command (it was the purpose line), so the stream says
    # `ran make: exit N`, never `ran: exit N`.
    assert shell._shell.request(parsed).label == "make"  # pyright: ignore[reportPrivateUsage]
