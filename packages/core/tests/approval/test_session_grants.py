"""*Always for this session* and *always for this repo*: one dispatch check, `GrantingPort`, for
both scopes.

A session grant spares the second prompt for a matching call in the same session and in the
same session resumed, and not in a fork or a new session. A grant binds the class, prefix and
destination and every declaration the call made, so a call reaching further asks again. A repo
grant is a committed line, inert on a clone until its owner accepts the list."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from repos import clone_roots

from chickadee.approval.commands import shipped
from chickadee.approval.grants import (
    ACTION_GRANTED,
    COMMAND_GRANT,
    NEVER_ASKED,
    ActionGrantedLine,
    GrantingPort,
    _line,  # pyright: ignore[reportPrivateUsage]
    grantable,
    offer,
    record_repo_grant,
    repo_grants,
    session_grants,
)
from chickadee.approval.prefs import PreferenceError
from chickadee.context.questions import RecordingPort
from chickadee.context.session import Session
from chickadee.engine import NonInteractive
from chickadee.executor.executor import Executor
from chickadee.isolation.none import Passthrough
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    CommandBinding,
    GrantScope,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee.shell import Shell, ShellClaim
from chickadee.shell_line import exact_words
from chickadee.state.roots import Roots
from chickadee.state.trust import TRUST_FILE, trusted_digests
from chickadee.zeus.tools.talk import PersonAuthority, approval_of

SESSION = "s-grants"
_GRANTED = TypeAdapter(ActionGrantedLine)


def granted_lines(session: Session) -> list[ActionGrantedLine]:
    with session.lock:
        return [
            _GRANTED.validate_json(session.blob(e.payload_sha))
            for e, _ in session.state.lines()
            if e.kind == ACTION_GRANTED
        ]


class _Answering:
    """An interactive port that answers every confirmation the same way, and counts them."""

    def __init__(self, verdict: str = "approve", scope: GrantScope = "once") -> None:
        self.verdict = verdict
        self.scope: GrantScope = scope
        self.asked: list[ActionApproval] = []

    def ask(self, q: Question, *, cancel: object = None) -> Answer:
        raise AssertionError("not asked here")

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        self.asked.append(approval)
        return ApprovalDecision.model_validate(
            {
                "action_id": approval.action_id,
                "verdict": self.verdict,
                "source": "human",
                "scope": self.scope,
            }
        )


def _binding(
    destination: str | None = "origin",
    *,
    network: bool = False,
    effects: tuple[str, ...] = (),
    credentials: tuple[str, ...] = (),
) -> CommandBinding:
    return CommandBinding(
        cls="push",
        prefix=("git", "push"),
        destination=destination,
        network=network,
        effects=effects,
        credentials=credentials,
    )


def _approval(binding: CommandBinding | None, n: int = 1) -> ActionApproval:
    return ActionApproval(
        action_id=f"a{n}",
        operation_id=f"op{n}",
        kind="command",
        display="git push origin main",
        spec_sha=f"{n:064d}",
        network=True,
        idempotent=False,
        reversible=False,
        tier=Tier.T2,
        grant=binding,
    )


def _port(session: Session, roots: Roots, person: _Answering) -> GrantingPort:
    """As the engine builds it (`engine._port`): the grant check over the recorder."""
    return GrantingPort(RecordingPort(person, session), session, roots)


# -- what *always* binds --------------------------------------------------------------------


def test_always_is_offered_for_one_binding_and_binds_what_the_call_declared() -> None:
    table = shipped()
    push = grantable(table.classify_line("git push origin main"), network=True)
    assert push == _binding(network=True)
    assert grantable(table.classify_line("frobnicate --all")) is None, "unrecognised: once"
    two = table.classify_line("git push origin main && rm -rf build")
    assert grantable(two) is None, "no single grant covers two bindings"
    declared = grantable(
        table.classify_line("git push origin main"),
        network=True,
        effects=["x"],
        credentials=["TOKEN"],
    )
    assert declared is not None
    assert (declared.network, declared.effects, declared.credentials) == (
        True,
        ("x",),
        ("TOKEN",),
    )


def test_a_destructive_command_is_never_granted(tmp_path: Path) -> None:
    """A hard floor nothing could have granted. Offered *once* only, and a committed
    line naming the class refuses the load."""
    table = shipped()
    assert grantable(table.classify_line("rm -rf build")) is None
    assert grantable(table.classify_line("git push --force origin main")) is None
    roots = clone_roots(tmp_path)
    roots.project.mkdir(parents=True)
    (roots.project / "approval.toml").write_text(
        '[[approval.grants]]\neffect = "command"\nclass = "destructive"\nprefix = ["rm"]\n',
        encoding="utf-8",
    )
    with pytest.raises(PreferenceError, match="never granted"):
        repo_grants(roots)


def test_a_grant_binds_whether_the_call_opened_the_network(tmp_path: Path) -> None:
    """An *always* given with the network covers later calls with it; one given without
    covers none with it. A repository grant is written back as `network = true`."""
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering(scope="session")
    port = _port(session, roots, person)
    port.confirm_action(_approval(_binding(), 1))
    person.scope = "once"
    asked = port.confirm_action(_approval(_binding(network=True), 2))
    assert (asked.source, len(person.asked)) == ("human", 2), "without covers none with it"
    person.scope = "repo"
    port.confirm_action(_approval(_binding(network=True), 3))
    text = (roots.project / "approval.toml").read_text(encoding="utf-8")
    assert "network = true" in text and "hosts" not in text
    assert repo_grants(roots).honoured == (_binding(network=True),)
    session.close()


def test_a_grant_written_by_an_older_version_reads_its_hosts_as_network(tmp_path: Path) -> None:
    """A row or a `command_grant` line naming hosts opened the network, so it reads as
    `network = true`, binding no more than it did; one naming none reads as `false`."""
    roots = clone_roots(tmp_path)
    roots.project.mkdir(parents=True)
    (roots.project / "approval.toml").write_text(
        '[[approval.grants]]\neffect = "command"\nclass = "push"\nprefix = ["git", "push"]\n'
        'destination = "origin"\nhosts = ["github.com"]\n'
        '\n[[approval.grants]]\neffect = "command"\nclass = "push"\nprefix = ["git", "push"]\n'
        'destination = "upstream"\nhosts = []\n',
        encoding="utf-8",
    )
    assert repo_grants(roots).bindings == (
        _binding(network=True),
        _binding("upstream"),
    )
    old = {
        "cls": "push",
        "prefix": ["git", "push"],
        "destination": "origin",
        "hosts": ["github.com"],
        "effects": [],
        "credentials": [],
        "action_id": "a0",
    }
    line = _line(json.dumps(old).encode())
    assert (line.binding(), line.exact) == (_binding(network=True), False)
    assert _line(json.dumps({**old, "hosts": []}).encode()).binding() == _binding()


def test_a_record_naming_both_hosts_and_network_is_refused_wherever_it_is_read() -> None:
    """One reading of the old shape (`network_of_hosts`), so a line and a binding refuse alike
    what a repository row refuses: which of the two it meant is unknown."""
    both = {"cls": "push", "prefix": ["git", "push"], "network": False, "hosts": ["github.com"]}
    with pytest.raises(ValueError, match="give network only"):
        CommandBinding.model_validate(both)
    line = {**both, "destination": None, "effects": [], "credentials": [], "action_id": "a"}
    with pytest.raises(ValueError, match="give network only"):
        _line(json.dumps(line).encode())


def test_a_row_naming_both_hosts_and_network_is_refused(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    roots.project.mkdir(parents=True)
    (roots.project / "approval.toml").write_text(
        '[[approval.grants]]\neffect = "command"\nclass = "push"\nprefix = ["git", "push"]\n'
        'network = true\nhosts = ["github.com"]\n',
        encoding="utf-8",
    )
    with pytest.raises(PreferenceError, match="give network only"):
        repo_grants(roots)


# -- a session grant -------------------------------------------------------------------------


def test_a_session_grant_spares_the_second_prompt_and_is_on_the_log(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering(scope="session")
    port = _port(session, roots, person)
    first = port.confirm_action(_approval(_binding(), 1))
    assert (first.verdict, first.scope, first.source) == ("approve", "session", "human")
    assert [g.binding() for g in session_grants(session)] == [_binding()]
    again = port.confirm_action(_approval(_binding(), 2))
    assert (again.verdict, again.source, again.scope) == ("approve", "grant", "session")
    assert [a.action_id for a in person.asked] == ["a1"], "the second was not asked"
    kinds = [e.kind for e, _ in session.state.lines()]
    assert kinds.count(COMMAND_GRANT) == 1
    # The person's question is on the log; the grant's answer is one line and never a
    # question, so nothing shows a prompt that is already answered.
    assert kinds.count("question_asked") == 1 and kinds.count("question_answered") == 1
    assert kinds.count(ACTION_GRANTED) == 1
    session.close()


def test_a_call_reaching_further_asks_again(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering(scope="session")
    port = _port(session, roots, person)
    port.confirm_action(_approval(_binding(), 1))
    person.scope = "once"
    for n, other in enumerate(
        (
            _binding(destination="evil-remote"),
            _binding(network=True),
            _binding(credentials=("GH_TOKEN",)),
            CommandBinding(cls="destructive", prefix=("git", "push"), destination="origin"),
        ),
        start=2,
    ):
        port.confirm_action(_approval(other, n))
    assert [a.action_id for a in person.asked] == ["a1", "a2", "a3", "a4", "a5"]
    session.close()


def test_the_resumed_session_keeps_it(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    root = tmp_path / "sessions"
    session = Session.create(root, SESSION)
    _port(session, roots, _Answering(scope="session")).confirm_action(_approval(_binding(), 1))
    session.close()
    resumed = Session.open(root, SESSION)
    person = _Answering()
    got = _port(resumed, roots, person).confirm_action(_approval(_binding(), 2))
    assert (got.source, person.asked) == ("grant", [])
    resumed.close()


def test_a_fork_and_a_new_session_do_not_have_it(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    root = tmp_path / "sessions"
    session = Session.create(root, SESSION)
    _port(session, roots, _Answering(scope="session")).confirm_action(_approval(_binding(), 1))
    at = session.state.last_seq
    session.close()
    child = Session.fork(root, SESSION, at, "try another way", "s-child", code="keep")
    assert session_grants(child) == ()
    person = _Answering()
    _port(child, roots, person).confirm_action(_approval(_binding(), 2))
    assert [a.action_id for a in person.asked] == ["a2"], "the fork asks again"
    child.close()
    fresh = Session.create(root, "s-new")
    person = _Answering()
    _port(fresh, roots, person).confirm_action(_approval(_binding(), 3))
    assert [a.action_id for a in person.asked] == ["a3"], "a new session asks again"
    fresh.close()


def test_only_an_approve_with_an_offered_grant_issues_one(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    denied = _port(session, roots, _Answering("deny", "session")).confirm_action(
        _approval(_binding(), 1)
    )
    assert denied.verdict == "deny" and session_grants(session) == ()
    # A question offering *once* only (an unrecognised command) cannot be answered *always*.
    once = _port(session, roots, _Answering(scope="session")).confirm_action(_approval(None, 2))
    assert (once.verdict, once.scope) == ("approve", "once")
    assert session_grants(session) == ()
    # Unattended, nothing is approved, so nothing is granted.
    unattended = GrantingPort(RecordingPort(NonInteractive(), session), session, roots)
    assert unattended.confirm_action(_approval(_binding(), 3)).verdict == "deny"
    assert session_grants(session) == ()
    session.close()


# -- a repo grant ----------------------------------------------------------------------------


def test_a_repo_grant_covers_every_session_of_this_clone(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    root = tmp_path / "sessions"
    session = Session.create(root, SESSION)
    first = _port(session, roots, _Answering(scope="repo")).confirm_action(
        _approval(_binding(), 1)
    )
    assert first.note == f"granted for this repository in {roots.project / 'approval.toml'}"
    assert session_grants(session) == (), "a repo grant is not a session line"
    session.close()
    text = (roots.project / "approval.toml").read_text(encoding="utf-8")
    assert '[[approval.grants]]\neffect = "command"\nclass = "push"' in text
    grants = repo_grants(roots)
    assert grants.trusted and grants.honoured == (_binding(),)
    other = Session.create(root, "s-other")
    person = _Answering()
    got = _port(other, roots, person).confirm_action(_approval(_binding(), 2))
    assert (got.source, got.scope, person.asked) == ("grant", "repo", [])
    # #553: the granted action opened no question; it is one line naming its grant.
    assert [e.kind for e, _ in other.state.lines()] == [ACTION_GRANTED]
    (granted,) = granted_lines(other)
    assert (granted.action_id, granted.scope, granted.prefix) == ("a2", "repo", ("git", "push"))
    other.close()


def test_a_cloned_grant_is_inert_until_accepted_and_an_answer_never_accepts_it(
    tmp_path: Path,
) -> None:
    roots = clone_roots(tmp_path)
    roots.project.mkdir(parents=True)
    # A teammate's committed line, never accepted on this clone.
    (roots.project / "approval.toml").write_text(
        '[[approval.grants]]\neffect = "command"\nclass = "push"\n'
        'prefix = ["git", "push"]\ndestination = "upstream"\n',
        encoding="utf-8",
    )
    assert repo_grants(roots).honoured == ()
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering(scope="repo")
    port = _port(session, roots, person)
    upstream = _binding(destination="upstream")
    port.confirm_action(_approval(upstream, 1))
    answered = port.confirm_action(_approval(_binding(), 2))
    assert answered.note is not None and "inert on this clone" in answered.note
    assert trusted_digests(roots.repo_local) == set(), "the answer accepted no one's line"
    person.scope = "once"
    port.confirm_action(_approval(_binding(), 3))
    assert [a.action_id for a in person.asked] == ["a1", "a2", "a3"]
    session.close()


def test_a_grant_file_that_cannot_be_read_refuses_and_is_left_alone(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    roots.project.mkdir(parents=True)
    path = roots.project / "approval.toml"
    path.write_text(
        '[[approval.grants]]\neffect = "command"\nclass = "pushh"\nprefix = ["git"]\n',
        encoding="utf-8",
    )
    with pytest.raises(PreferenceError, match="not a command class"):
        repo_grants(roots)
    # A file this would break is not rewritten: an inline list cannot take a table after it.
    path.write_text("[approval]\ngrants = []\n", encoding="utf-8")
    with pytest.raises(PreferenceError, match="could not be added"):
        record_repo_grant(roots, _binding())
    assert path.read_text(encoding="utf-8") == "[approval]\ngrants = []\n"
    assert not (roots.repo_local / TRUST_FILE).exists()
    # Asked through the port, the person's yes still runs this call, once, and says why.
    session = Session.create(tmp_path / "sessions", SESSION)
    got = _port(session, roots, _Answering(scope="repo")).confirm_action(
        _approval(_binding(), 1)
    )
    assert (got.verdict, got.scope) == ("approve", "once")
    assert got.note is not None and got.note.startswith("approved once; not granted:")
    session.close()


# -- the two callers -------------------------------------------------------------------------


def test_zeus_shell_asks_once_for_a_command_granted_for_the_session(tmp_path: Path) -> None:
    """With no sandbox every command is T2 and confirmed; a session grant for it spares
    the second confirmation, end to end through `PersonAuthority`."""
    from tool_args import checked

    from chickadee.context.homes import SessionHome
    from chickadee.context.reader import Reader
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.terms import Vision

    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=tmp_path, session_blob=session.blob),
        vision=Vision("z"),
    )
    person = _Answering(scope="session")
    shell = Shell(
        Executor(backend=Passthrough()),
        tmp_path,
        lambda data: reader.store("output", data),
        authority=PersonAuthority(_port(session, roots, person), lambda: None),
    )
    for n in (1, 2):
        shell.run(checked("shell", {"command": "git status"}), operation_id=f"op{n}")
    assert len(person.asked) == 1
    asked = person.asked[0]
    assert asked.grant == CommandBinding(cls="read_only", prefix=("git", "status"))
    session.close()


# -- never ask ----------------------------------------------------------------


def test_never_ask_approves_an_ungranted_command_with_one_line_and_nobody_asked(
    tmp_path: Path,
) -> None:
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering(verdict="deny")
    port = GrantingPort(RecordingPort(person, session), session, roots, never_ask=True)
    got = port.for_run("R1").confirm_action(_approval(_binding(), 1))
    assert (got.verdict, got.source, got.scope) == ("approve", "preference", "once")
    assert got.note == "run unasked: sandbox.never_ask"
    assert person.asked == []
    (line,) = granted_lines(session)
    assert (line.scope, line.cls, line.prefix) == ("never_ask", "push", ("git", "push"))
    # A command no binding reads is run too, and named as unbound.
    port.confirm_action(_approval(None, 2))
    assert granted_lines(session)[-1].cls == NEVER_ASKED
    assert [e.kind for e, _ in session.state.lines()] == [ACTION_GRANTED, ACTION_GRANTED]
    session.close()


def test_never_ask_leaves_every_other_approval_asked(tmp_path: Path) -> None:
    """It covers shell commands only: a mutating MCP call is still put to the person."""
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering()
    port = GrantingPort(RecordingPort(person, session), session, roots, never_ask=True)
    mcp = _approval(None, 1).model_copy(update={"kind": "mcp_tool"})
    assert port.confirm_action(mcp).source == "human"
    assert [a.action_id for a in person.asked] == ["a1"]
    session.close()


def test_under_never_ask_a_denied_class_still_refuses(tmp_path: Path) -> None:
    """The shell's `deny` is checked before the port is reached (OpenCode's `deny` surviving
    `--auto`), so `never_ask` never runs it."""
    from tool_args import checked

    from chickadee.approval.prefs import merge, parse_level
    from chickadee.context.homes import SessionHome
    from chickadee.context.reader import Reader
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.terms import Vision
    from chickadee.shell import PreferenceDenied

    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=tmp_path, session_blob=session.blob),
        vision=Vision("z"),
    )
    port = GrantingPort(RecordingPort(_Answering(), session), session, roots, never_ask=True)
    denied = merge([parse_level({"approval": {"commands": {"push": "deny"}}}, "user")])
    shell = Shell(
        Executor(backend=Passthrough()),
        tmp_path,
        lambda data: reader.store("output", data),
        authority=PersonAuthority(port, lambda: None),
        prefs=lambda: denied,
    )
    got = shell.run(
        checked("shell", {"command": "git push origin main"}),
        operation_id="op1",
    )
    assert isinstance(got, PreferenceDenied)
    assert granted_lines(session) == []
    session.close()


def test_the_question_carries_the_binding_from_the_shell() -> None:
    claim = ShellClaim(
        "git push origin main",
        ".",
        Tier.T2,
        "push",
        True,
        operation_id="op",
        idempotent=False,
        grant=_binding(network=True),
    )
    assert approval_of(claim).grant == _binding(network=True)


def test_a_granted_action_in_a_run_is_tagged_with_the_run(tmp_path: Path) -> None:
    """The run-scoped port's grant line names the run, as the question it replaces did."""
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    port = _port(session, roots, _Answering(scope="session"))
    port.confirm_action(_approval(_binding(), 1))
    in_run = port.for_run("2026-10-13-0a1b2c3d")
    assert in_run.confirm_action(_approval(_binding(), 2)).source == "grant"
    (event,) = [e for e, _ in session.state.lines() if e.kind == ACTION_GRANTED]
    assert event.run_id == "2026-10-13-0a1b2c3d"
    session.close()


# -- T-3.158 C: a grant covers only what was approved -----------------------------------------

_PUSH_GRANT_NEVER_COVERS = (
    # The shell reader's holes.
    "git push origin main && { git push --force origin main; }",
    "git push origin main; if true; then git push --force origin main; fi",
    "git push origin main; f() { git push --force origin main; }; f",
    "git push origin main; eval 'git push --force origin main'",
    "git push origin main && ./deploy.sh",
    "git push origin {main,--force}",
    "git push origin --for\\ce main",
    "git push origin main > /dev/null",
    "git push origin main 2>&1",
    # The table reader's holes.
    "git push origin main --force",
    "git push -uf origin main",
    "git push origin +main",
    "git push origin :feature",
    "git push origin --delete feature",
    "git push --mirror origin",
    "git -c remote.origin.mirror=true push origin",
    "git push -o origin upstream main",
    "git push origin main --exec=/tmp/x",
    "git -C ../other-repo push origin main",
    "git --git-dir=../x/.git push origin main",
    "cd ../other-repo && git push origin main",
    "git push origin main --no-verify",
)


def _zeus_shell(tmp_path: Path, person: _Answering) -> tuple[Shell, Session, list[str]]:
    """Zeus's shell over `GrantingPort`, as the engine builds it, with a spawn that records
    instead of running: a granted `curl … | sh` must not run in a test."""
    from chickadee.context.homes import SessionHome
    from chickadee.context.reader import Reader
    from chickadee.context.resolver import HandleResolver
    from chickadee.context.terms import Vision
    from chickadee.executor.types import ExecRequest, ExecResult
    from chickadee.schema.execution import SandboxLevel

    ran: list[str] = []

    def spawn(request: ExecRequest, operation_id: str, cancel: object) -> ExecResult:
        ran.append(operation_id)
        return ExecResult(
            exit_code=0,
            stdout_sha=None,
            stderr_sha=None,
            wall_ms=0,
            timed_out=False,
            sandbox=SandboxLevel.NONE,
        )

    session = Session.create(tmp_path / "sessions", SESSION)
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=tmp_path, session_blob=session.blob),
        vision=Vision("z"),
    )
    shell = Shell(
        Executor(backend=Passthrough()),
        tmp_path,
        lambda data: reader.store("output", data),
        authority=PersonAuthority(_port(session, clone_roots(tmp_path), person), lambda: None),
        spawn=spawn,
    )
    return shell, session, ran


def _run(shell: Shell, line: str, n: int) -> None:
    from tool_args import checked

    shell.run(checked("shell", {"command": line}), operation_id=f"op{n}")


@pytest.mark.parametrize(
    ("scope", "covered"),
    [
        # *This session* binds the exact line approved.
        ("session", ("git push origin main",)),
        # *This repo* binds `git push`'s grant shape (E.2): any branch, listed options only.
        ("repo", ("git push origin main", "git push origin feature", "git push -u origin x")),
    ],
)
def test_a_push_grant_covers_what_it_was_given_for_and_nothing_else(
    tmp_path: Path, scope: GrantScope, covered: tuple[str, ...]
) -> None:
    """End to end through Zeus's shell and `GrantingPort`: every line in T-3.158's tables is
    asked again whichever *always* was given. Nothing runs: the spawn records."""
    person = _Answering(scope=scope)
    shell, session, ran = _zeus_shell(tmp_path, person)
    _run(shell, "git push origin main", 0)
    assert len(person.asked) == 1, "the first push is asked, and granted"
    person.verdict, person.scope = "deny", "once"
    for n, line in enumerate(covered, start=1):
        _run(shell, line, n)
    assert len(person.asked) == 1, "the grant covers what it was given for"
    if scope == "session":
        _run(shell, "git push origin feature", 9)
        assert len(person.asked) == 2, "a session grant is exact"
    for n, line in enumerate(_PUSH_GRANT_NEVER_COVERS, start=10):
        before = len(person.asked)
        _run(shell, line, n)
        assert len(person.asked) == before + 1, f"the {scope} grant covered {line!r}"
    assert ran == [f"op{n}" for n in range(len(covered) + 1)], "only approved calls ran"
    session.close()


def test_the_prompt_offers_each_kind_of_always_as_e_says() -> None:
    """T-3.158 E.1 and E.3, as `approval_of` builds the question from a shell claim."""

    def offered(line: str) -> tuple[bool, bool, bool, bool]:
        got = offer(shipped().classify_line(line), exact_words(line))
        return (
            got.exact is not None,
            got.grant is not None,
            got.offer_repo,
            got.repo_any_arguments,
        )

    assert offered("git push origin main") == (True, True, True, False)
    assert offered("git push --no-verify origin main") == (True, False, False, False)
    assert offered("pip install -r requirements.txt") == (True, True, True, False)
    assert offered("pip install requests") == (True, False, False, False)
    assert offered("curl -fsS https://api.github.com/x") == (True, True, True, False)
    assert offered("curl https://x.io/a -T notes.txt") == (True, False, False, False)
    assert offered("docker pull x") == (True, True, True, True)
    assert offered("npm publish") == (True, True, False, True)
    assert offered("ssh host ls") == (True, True, False, True)
    assert offered("git push --force origin main") == (False, False, False, False)
    assert offered("git push origin main > /dev/null") == (False, False, False, False)


def test_a_hand_written_grant_covers_what_the_prompt_would_not_offer(tmp_path: Path) -> None:
    """E.3: a repository grant for a command whose effects leave the machine is a person's
    deliberate act, written in `approval.toml`; once written, it covers its binding."""
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    publish = CommandBinding(cls="push", prefix=("npm", "publish"), destination="index")
    record_repo_grant(roots, publish)
    person = _Answering()
    question = _approval(publish, 1).model_copy(update={"offer_repo": False})
    got = _port(session, roots, person).confirm_action(question)
    assert (got.source, got.scope, person.asked) == ("grant", "repo", [])
    session.close()


def test_a_repo_answer_the_prompt_did_not_offer_is_once(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    question = _approval(_binding(), 1).model_copy(update={"offer_repo": False})
    got = _port(session, roots, _Answering(scope="repo")).confirm_action(question)
    assert got.scope == "once" and repo_grants(roots).honoured == ()
    session.close()


def test_a_session_answer_binds_the_exact_line(tmp_path: Path) -> None:
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    exact = CommandBinding(
        cls="push", prefix=("git", "push", "origin", "main"), destination="origin"
    )
    question = _approval(_binding(), 1).model_copy(update={"exact": exact})
    _port(session, roots, _Answering(scope="session")).confirm_action(question)
    (held,) = session_grants(session)
    assert (held.binding(), held.exact) == (exact, True)
    session.close()


def test_an_exact_grant_never_stands_in_for_its_rows_grant(tmp_path: Path) -> None:
    """The review of #578: `make`'s exact line is its row's own words, so the two bindings are
    equal. An exact grant for `make` must not cover `make deploy`, whose row binding is it."""
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    table = shipped()

    def question(line: str, n: int) -> ActionApproval:
        got = offer(table.classify_line(line), exact_words(line))
        return _approval(got.grant, n).model_copy(update={"display": line, "exact": got.exact})

    make, deploy = question("make", 1), question("make deploy", 2)
    assert make.exact == make.grant == deploy.grant, "the case: equal bindings"
    person = _Answering(scope="session")
    port = _port(session, roots, person)
    port.confirm_action(make)
    person.scope = "once"
    again = port.confirm_action(question("make", 3))
    assert again.source == "grant", "the exact line itself is covered"
    port.confirm_action(deploy)
    assert [a.display for a in person.asked] == ["make", "make deploy"]
    session.close()


def test_a_row_grant_covers_its_row_and_not_an_exact_line(tmp_path: Path) -> None:
    """A session answer to a question with no exact binding is a row grant, as before
    It covers a later question's row binding, never its exact one."""
    roots = clone_roots(tmp_path)
    session = Session.create(tmp_path / "sessions", SESSION)
    person = _Answering(scope="session")
    port = _port(session, roots, person)
    port.confirm_action(_approval(_binding(), 1))
    person.scope = "once"
    covered = port.confirm_action(_approval(_binding(), 2))
    exact_only = _approval(None, 3).model_copy(update={"exact": _binding()})
    port.confirm_action(exact_only)
    assert covered.source == "grant" and len(person.asked) == 2
    session.close()


@pytest.mark.parametrize(
    "grant",
    [None, CommandBinding(cls="build", prefix=("make", "deploy"))],
    ids=["none", "other"],
)
def test_the_prompt_offers_once_only_when_the_shells_table_binds_otherwise(
    grant: CommandBinding | None,
) -> None:
    """A shell given another table binds a line otherwise than the shipped one: the question
    then offers *once* only, so no answer covers more than its prompt could state."""
    claim = ShellClaim(
        "make deploy",
        ".",
        Tier.T2,
        "build",
        False,
        operation_id="op",
        idempotent=False,
        grant=grant,
    )
    got = approval_of(claim)
    assert (got.grant, got.exact, got.offer_repo) == (None, None, False)
