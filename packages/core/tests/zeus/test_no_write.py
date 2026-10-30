"""Zeus never writes source (invariant 12): there is no write or edit
tool, AND a `sed -i` against tracked source fails at the sandbox. Both, because the second is
the one that holds when a prompt is ignored.

The sandbox half needs bubblewrap. Its workspace is under `/var/tmp`, not pytest's `tmp_path`:
the wrap mounts a private tmpfs over `/tmp`, so a workspace there would not exist inside it,
the command would fail to start, and the test would pass without the write ever being tried.
A control read in the same directory proves the command ran.

Where there is no sandbox, a person stands in for it: every command is T2 classified
`unsandboxed`, and runs only once confirmed, unless the person set `allow_unsandboxed_shell`.
Where the configuration requires a sandbox and there is none, every call is `SANDBOX_REQUIRED`
before anybody is asked."""

from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from repos import init_repo
from tool_args import checked

from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import InvalidArgument, ToolResult
from chickadee.context.session import Session
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.executor.policy import isolation_for
from chickadee.executor.types import ExecRequest, ExecResult
from chickadee.isolation.base import SandboxBackend, has
from chickadee.isolation.linux import BINARY, Bubblewrap
from chickadee.isolation.none import Passthrough
from chickadee.schema import sha256_hex
from chickadee.schema.authorization import Tier
from chickadee.schema.execution import SandboxLevel
from chickadee.shell import (
    RefusedAboveT0,
    SandboxRequired,
    Shell,
    ShellAuthority,
    ShellClaim,
    ShellCommand,
)
from chickadee.tools.spec import SURFACE, spec
from chickadee.tools.validate import InvalidArguments, validate

from .zeus_helpers import SESSION_ID, FakeBox, answer, call, calls, rig, zeus_prefix

bwrap_only = pytest.mark.skipif(not has(BINARY), reason="bubblewrap unavailable")

SOURCE = "def answer():\n    return 41\n"


@pytest.fixture
def repo() -> Iterator[Path]:
    """A git repository with one tracked file, outside `/tmp` (see the module docstring)."""
    base = Path("/var/tmp") if Path("/var/tmp").is_dir() else Path(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="zeus-no-write-", dir=base))
    try:
        (root / "src").mkdir()
        (root / "src" / "a.py").write_text(SOURCE, encoding="utf-8")
        init_repo(root, commit="one")
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _shell(
    tmp_path: Path,
    workspace: Path,
    executor: Executor,
    *,
    authority: ShellAuthority | None = None,
    allow_unsandboxed_shell: bool = False,
) -> tuple[Shell, Session]:
    session = Session.create(tmp_path / "sessions", SESSION_ID)
    reader = Reader(
        SessionHome(session),
        HandleResolver(workspace=workspace, session_blob=session.blob),
        vision=Vision("z"),
    )
    shell = Shell(
        executor,
        workspace,
        lambda data: reader.store("output", data),
        authority=authority,
        allow_unsandboxed_shell=allow_unsandboxed_shell,
    )
    return shell, session


# -- half one: nothing on the surface writes -----------------------------------------------


def test_no_write_or_edit_tool_exists(tmp_path: Path) -> None:
    names = {s.name for s in SURFACE}
    for verb in (
        "write",
        "edit",
        "apply",
        "patch",
        "create",
        "replace",
        "str_replace",
        "delete",
    ):
        assert not any(verb in n for n in names), f"a tool named like {verb!r} exists"
    writers = [s.name for s in SURFACE if s.mutates == "source"]
    # `doc` writes the work folder's `design/` alone, never a source path: its port
    # resolves every file inside that folder (`tests/tools/test_doc.py`). `worktree` lands or
    # discards what the box committed, asked of the person, and authors nothing.
    assert writers == ["coder", "doc", "worktree"], "only the box writes source"
    offered = {t.name for t in zeus_prefix(tmp_path).tools}
    assert offered == names, "the model is offered SURFACE and nothing else"


def test_every_request_the_shell_builds_leaves_the_workspace_read_only(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    shell, session = _shell(tmp_path, ws, Executor())
    variants: list[dict[str, object]] = [
        {"command": "sed -i s/1/2/ a.py"},
        {"command": "uv sync", "network": True},
        {"command": "make", "idempotent": True, "cwd": "sub"},
        {"command": "echo x > a.py"},
    ]
    for args in variants:
        parsed = shell.parse(checked("shell", args))
        assert isinstance(parsed, ShellCommand), args
        wrap = isolation_for(shell.request(parsed))
        assert wrap.repo_writable is False, args
    session.close()


def test_declaring_an_effect_is_refused_not_made_writable(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    """`effects=` is retired; a declaration is refused before anything is built,
    and nothing is made writable by it."""
    found = spec("shell")
    assert found is not None
    got = validate(found, {"command": "touch src/a.py", "effects": ["src/a.py"]})
    assert isinstance(got, InvalidArguments)
    assert [v.field for v in got.violations] == ["effects"]
    assert not (ws / "src" / "a.py").exists()


def test_a_cwd_outside_the_workspace_is_refused(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    shell, session = _shell(tmp_path, ws, Executor())
    for cwd in ("..", "/", "../ws/../.."):
        got = shell.run(checked("shell", {"command": "ls", "cwd": cwd}), operation_id="op")
        assert isinstance(got, InvalidArgument) and got.field == "cwd", cwd
    session.close()


# -- half two: the sandbox refuses the write anyway ----------------------------------------


@bwrap_only
def test_sed_i_against_tracked_source_fails_at_the_sandbox(tmp_path: Path, repo: Path) -> None:
    shell, session = _shell(tmp_path, repo, Executor(backend=Bubblewrap()))
    control = shell.run(checked("shell", {"command": "cat src/a.py"}), operation_id="op-1")
    assert isinstance(control, ToolResult)
    assert control.header.startswith("exit 0") and "return 41" in control.render()
    for command in ("sed -i 's/41/42/' src/a.py", "echo 'x = 1' > src/a.py"):
        got = shell.run(
            checked("shell", {"command": command}),
            operation_id="op-2",
        )
        assert isinstance(got, ToolResult), command
        assert not got.header.startswith("exit 0"), command
        assert "sandbox native" in got.header and "repo read-only" in got.header
        assert (repo / "src" / "a.py").read_text(encoding="utf-8") == SOURCE, command
        assert "read-only file system" in got.render().lower(), command
    assert "WRITE_DENIED" in got.render()
    session.close()


@bwrap_only
def test_through_the_loop_the_write_fails_and_zeus_is_told(tmp_path: Path, repo: Path) -> None:
    sed = call("shell", "s1", command="sed -i 's/41/42/' src/a.py")
    wrapped = Executor(backend=Bubblewrap())
    r = rig(tmp_path, [calls(sed)], workspace=repo, shell=True, executor=wrapped)
    r.run("change answer() to return 42")
    assert (repo / "src" / "a.py").read_text(encoding="utf-8") == SOURCE
    line = r.session.state.result("s1")
    assert line is not None and line.ran
    assert "WRITE_DENIED" in line.text and "repo read-only" in line.text
    assert "WRITE_DENIED" in r.model.wire(1), "the refusal reached the model"


def test_with_no_sandbox_to_require_the_call_is_sandbox_required(tmp_path: Path) -> None:
    """`require_sandbox` with no backend: nothing can run here, and Zeus is told so in
    its own failure, not `INTERRUPTED`, which a model may retry. The turn goes on, and
    nothing is left waiting for the next turn's repair to call what it did unknown."""
    required = Executor(backend=Passthrough(), require_sandbox=True)
    script = [calls(call("shell", "s1", command="ls")), answer("no shell, then")]
    r = rig(tmp_path, script, shell=True, executor=required)
    events = r.run("what is in the repo?")
    line = r.session.state.result("s1")
    assert line is not None
    assert line.text.startswith("SANDBOX_REQUIRED{level: none, reason: 'require_sandbox is set")
    assert "SANDBOX_REQUIRED" in r.model.wire(1), "the failure reached the model"
    assert [e.payload for e in events if e.kind == "answer"] == ["no shell, then"]
    assert r.session.state.pending_calls() == ()


class _Granted:
    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        return True


class _AllOrNothing(Passthrough):
    """A sandbox that cannot filter by host: the whole network, or none of it."""

    def level(self) -> SandboxLevel:
        return SandboxLevel.NATIVE


FETCH = {"command": "true", "network": True}


class _Person:
    """An authority that records what it was asked, and answers `answer`."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.claims: list[ShellClaim] = []

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool:
        self.claims.append(claim)
        return self.answer


class _Counting(Executor):
    """An executor that records the command of each request it is handed, so a test can say
    none was."""

    def __init__(self, backend: SandboxBackend, *, require_sandbox: bool = False) -> None:
        super().__init__(backend, require_sandbox=require_sandbox)
        self.ran: list[str] = []

    def run(
        self,
        request: ExecRequest,
        *,
        operation_id: str,
        cancel: threading.Event | None = None,
    ) -> ExecResult:
        command = request.command
        self.ran.append(command if isinstance(command, str) else " ".join(command))
        return super().run(request, operation_id=operation_id, cancel=cancel)


@pytest.mark.parametrize("args", [FETCH, {"command": "ls"}], ids=["declared", "not"])
def test_under_require_sandbox_every_call_is_sandbox_required_and_nobody_asked(
    tmp_path: Path, args: dict[str, object]
) -> None:
    """Declared or not, the answer is the same, and it is given before the
    tier and the authority. It is not `NETWORK_DENIED`, which would have Zeus ask for the
    network it asked for."""
    required = _Counting(backend=Passthrough(), require_sandbox=True)
    asked = _Person(True)
    shell, session = _shell(tmp_path, tmp_path, required, authority=asked)
    got = shell.run(checked("shell", args), operation_id="op-1")
    assert isinstance(got, SandboxRequired)
    assert got.level is SandboxLevel.NONE
    assert "require_sandbox" in got.why
    assert (required.ran, asked.claims) == ([], [])
    session.close()


def test_an_approved_network_runs_with_the_network_on_under_require_sandbox(
    tmp_path: Path,
) -> None:
    """D199 moved this answer: under `require_sandbox` (the default) the person was asked,
    approved, and the call answered `NETWORK_DENIED{heuristic: false}`, since no backend
    filters by host. Now `network: true` is asked once and, approved, runs with the network
    on."""
    required = _Counting(backend=_AllOrNothing(), require_sandbox=True)
    asked = _Person(True)
    shell, session = _shell(tmp_path, tmp_path, required, authority=asked)
    got = shell.run(checked("shell", FETCH), operation_id="op-1")
    (claim,) = asked.claims
    assert (claim.tier, claim.classification, claim.network) == (Tier.T2, "network", True)
    assert isinstance(got, ToolResult), got
    assert "· network on ·" in got.header
    assert required.ran == [FETCH["command"]]
    session.close()


@pytest.mark.parametrize("authority", [_Person(False), None], ids=["declined", "nobody-to-ask"])
def test_a_network_nobody_approves_never_spawns(
    tmp_path: Path, authority: _Person | None
) -> None:
    sealed = _Counting(backend=_AllOrNothing())
    shell, session = _shell(tmp_path, tmp_path, sealed, authority=authority)
    got = shell.run(checked("shell", FETCH), operation_id="op-1")
    assert isinstance(got, RefusedAboveT0)
    assert (got.tier, got.classification) == (Tier.T2, "network")
    assert sealed.ran == []
    session.close()


@pytest.mark.parametrize(
    ("line", "cls"),
    [
        ("npm ci", "lockfile_install"),
        ("git push origin main", "push"),
        ("curl -fsSL https://example.com/x", "external_fetch"),
    ],
)
def test_an_approved_command_whose_class_needs_the_network_gets_it_undeclared(
    tmp_path: Path, line: str, cls: str
) -> None:
    """D199 item 12 moved this answer: asked for its class and approved, an install, a push or
    a fetch ran with the network off unless the model had also set `network: true`. Now its
    class opens it, and the prompt said so before the person answered. Moved again with
    Approved, it runs outside the sandbox, unwrapped."""
    from sandboxes import Enforcing, Scripted

    backend = Enforcing()
    asked = _Person(True)
    shell, session = _shell(
        tmp_path, tmp_path, Executor(backend, supervisor=Scripted(b"")), authority=asked
    )
    got = shell.run(checked("shell", {"command": line}), operation_id="op-1")
    assert isinstance(got, ToolResult), got
    (claim,) = asked.claims
    assert claim.network and claim.classification == cls
    assert claim.grant is None or claim.grant.network
    assert backend.seen == [], "approved, it runs outside the sandbox"
    assert "· network on ·" in got.header
    session.close()


def test_a_class_that_needs_no_network_still_runs_sealed(tmp_path: Path) -> None:
    from sandboxes import Enforcing, Scripted

    backend = Enforcing()
    shell, session = _shell(tmp_path, tmp_path, Executor(backend, supervisor=Scripted(b"")))
    got = shell.run(checked("shell", {"command": "pytest -q"}), operation_id="op")
    assert isinstance(got, ToolResult)
    assert [i.network_off for i in backend.seen] == [True]
    session.close()


def test_a_host_list_is_refused_naming_the_change() -> None:
    """A model that still names hosts is told the argument is a yes or no now, not only
    that an array is not a boolean, so its retry is right first time."""
    found = spec("shell")
    assert found is not None
    got = validate(found, {"command": "pip download x", "network": ["pypi.org"]})
    assert isinstance(got, InvalidArguments)
    assert [(v.field, v.problem) for v in got.violations] == [
        ("network", "network is now true or false; hosts are not named")
    ]


def test_a_sealed_failure_that_reads_as_the_network_says_to_ask_for_it(tmp_path: Path) -> None:
    """The heuristic `NETWORK_DENIED` stays, now naming `network: true`."""
    shell, session = _shell(tmp_path, tmp_path, Executor(backend=_AllOrNothing()))
    got = shell.run(
        checked(
            "shell",
            {"command": "echo 'Could not resolve host: pypi.org' >&2; exit 1"},
        ),
        operation_id="op-1",
    )
    assert isinstance(got, ToolResult)
    assert "· network off ·" in got.header
    [note] = [n for n in got.notes if n.startswith("NETWORK_DENIED")]
    assert note.startswith("NETWORK_DENIED{heuristic: true")
    assert "network: true" in note
    session.close()


# -- where there is no sandbox, a person stands in for it ----------------------------

LS = {"command": "ls"}


def test_with_no_sandbox_an_undeclared_command_is_t2_unsandboxed_and_runs_once_confirmed(
    tmp_path: Path,
) -> None:
    unwrapped = _Counting(backend=Passthrough())
    asked = _Person(True)
    shell, session = _shell(tmp_path, tmp_path, unwrapped, authority=asked)
    got = shell.run(checked("shell", LS), operation_id="op-1")
    (claim,) = asked.claims
    assert (claim.command, claim.tier, claim.classification) == ("ls", Tier.T2, "unsandboxed")
    assert isinstance(got, ToolResult) and got.header.startswith("exit 0")
    # Moved with T-3.178: confirmed, the header says it ran outside the sandbox.
    assert "outside the sandbox, approved" in got.header and "· t2 ·" in got.header
    assert unwrapped.ran == ["ls"]
    session.close()


@pytest.mark.parametrize("authority", [_Person(False), None], ids=["refused", "nobody-to-ask"])
def test_with_no_sandbox_a_command_nobody_confirms_never_spawns(
    tmp_path: Path, authority: _Person | None
) -> None:
    """Refused, or unattended with no `ShellAuthority` at all (the engine's today, until
    `T-4.5`): `REFUSED_ABOVE_T0`, and the executor is never handed the command."""
    unwrapped = _Counting(backend=Passthrough())
    shell, session = _shell(tmp_path, tmp_path, unwrapped, authority=authority)
    got = shell.run(checked("shell", LS), operation_id="op-1")
    assert isinstance(got, RefusedAboveT0)
    assert (got.tier, got.classification) == (Tier.T2, "unsandboxed")
    assert unwrapped.ran == []
    session.close()


def test_with_no_sandbox_a_row_already_at_t2_keeps_its_own_name(tmp_path: Path) -> None:
    """The floor names the tier only where it raised it: the person confirming `rm -rf` is
    told what it is, not only that nothing wraps it."""
    unwrapped = _Counting(backend=Passthrough())
    asked = _Person(False)
    shell, session = _shell(tmp_path, tmp_path, unwrapped, authority=asked)
    got = shell.run(checked("shell", {"command": "rm -rf build"}), operation_id="op")
    assert isinstance(got, RefusedAboveT0)
    (claim,) = asked.claims
    assert claim.tier is Tier.T2 and claim.classification not in ("unsandboxed", "sealed")
    assert unwrapped.ran == []
    session.close()


def test_allow_unsandboxed_shell_lifts_the_floor_and_nothing_else(tmp_path: Path) -> None:
    """The undeclared `ls` is T0 again and runs unasked; `network: true`
    still raises its call to T2, and with nobody to ask it does not run."""
    unwrapped = _Counting(backend=Passthrough())
    shell, session = _shell(tmp_path, tmp_path, unwrapped, allow_unsandboxed_shell=True)
    ran = shell.run(checked("shell", LS), operation_id="op-1")
    assert isinstance(ran, ToolResult) and "· t0 ·" in ran.header
    declared = {**LS, "network": True}
    fetch = shell.run(checked("shell", declared), operation_id="op-2")
    assert isinstance(fetch, RefusedAboveT0)
    assert (fetch.tier, fetch.classification) == (Tier.T2, "network")
    assert unwrapped.ran == ["ls"]
    session.close()


def test_with_an_enforcing_sandbox_an_undeclared_command_is_t0_as_before(
    tmp_path: Path,
) -> None:
    """The floor is the sandbox's absence, so a backend that enforces leaves
    D93's construction as it was."""
    enforcing = _Counting(backend=_AllOrNothing())
    shell, session = _shell(tmp_path, tmp_path, enforcing)
    got = shell.run(checked("shell", LS), operation_id="op-1")
    assert isinstance(got, ToolResult) and "· t0 ·" in got.header
    assert enforcing.ran == ["ls"]
    session.close()


# -- coder is the only writer: what a delegation carries ------------------------------------


@pytest.mark.parametrize(
    ("args", "field"),
    [
        ({}, "work"),
        ({"work": "2026-10-18-x", "title": "also this", "brief": "and this"}, "work"),
        ({"title": "fix it"}, "brief"),
        ({"brief": "fix it"}, "title"),
        ({"work": "2026-10-18-x"}, "work"),
    ],
    ids=["no-work", "both", "no-brief", "no-title", "no-work-open"],
)
def test_a_malformed_delegation_is_refused_by_field(
    tmp_path: Path, args: dict[str, str], field: str
) -> None:
    r = rig(tmp_path, [calls(call("coder", "c1", **args))])
    r.run("delegate")
    text = r.session.state.result("c1").text  # type: ignore[union-attr]
    assert text.startswith(f"INVALID_ARGUMENTS{{field: {field}")
    assert r.box.started == [] and r.session.state.episodes() == ()


def test_a_design_by_its_text_is_refused_naming_work(tmp_path: Path) -> None:
    """A design is written with doc and delegated by name."""
    r = rig(tmp_path, [calls(call("coder", "c1", design="# Split the reader\n"))])
    r.run("delegate")
    text = r.session.state.result("c1").text  # type: ignore[union-attr]
    assert "- design: design is retired" in text and "coder(work=" in text
    assert r.box.started == []


def test_the_episode_is_written_at_dispatch_and_carries_the_work_sha(tmp_path: Path) -> None:
    """Each episode line names the work its stop approved (invariant 15): here the
    one-task plan's brief. A design's, its pinned README with the harness's two sections, is
    `engine/test_work_folder.py`'s, since the folder is the engine's."""
    brief = "## Goal\nx is 2\n"
    r = rig(tmp_path, [calls(call("coder", "c2", title="Set x", brief=brief))])
    events = r.run("do it")
    state = r.session.state
    (task,) = state.episodes()
    assert (task.path, task.design_sha, task.tier) == ("planned", sha256_hex(brief), None)
    assert task.seq < state.result("c2").seq, "written before the result, at dispatch"  # type: ignore[union-attr]
    assert state.result("c2").run_id == task.run_id  # type: ignore[union-attr]
    delegations = [e for e in events if e.kind == "delegation"]
    assert [e.seq for e in delegations] == [task.seq]


def test_the_box_runs_only_after_its_episode_line_is_in_the_log(tmp_path: Path) -> None:
    """D123's order, as a contract: a crash anywhere in a run leaves a session that names it,
    because nothing of the run happens until its line is written."""
    from .zeus_helpers import FakeBox

    box = FakeBox()
    r = rig(tmp_path, [calls(call("coder", "c1", title="migrate", brief="migrate"))], box=box)
    seen: list[tuple[str, ...]] = []
    box.on_run = lambda run_id: seen.append(tuple(e.run_id for e in r.session.state.episodes()))
    r.run("migrate")
    assert box.ran == ["R-1"]
    assert seen == [("R-1",)], "the episode line was in the log when the run began its work"


def test_the_episode_line_is_synced_before_the_run_does_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's journal fsyncs its effects, so the line naming the run must be as durable, or
    a power cut leaves a waiting call with no run to reconcile it from."""
    box = FakeBox()
    r = rig(tmp_path, [calls(call("coder", "c1", title="fix it", brief="fix it"))], box=box)
    synced: list[int] = []
    real = r.session.sync

    def sync() -> None:
        real()
        synced.append(r.session.state.last_seq)

    monkeypatch.setattr(r.session, "sync", sync)
    covered: list[bool] = []

    def on_run(run_id: str) -> None:
        (episode,) = r.session.state.episodes()
        covered.append(bool(synced) and synced[-1] >= episode.seq)

    box.on_run = on_run
    r.run("fix it")
    assert covered == [True]
