"""`shell` — the one shell, which never keeps a source edit, under its profiles.

**One implementation, its profiles.** Zeus's shell and the box roles' are this class;
`ShellProfile` is everything that differs. `ZEUS` asks its `ShellAuthority` above T0, `BOX`
does too, naming the role and the task. `AGENT` (a review agent's) has no authority, ever,
and reads a declaration as a request above T0: above T0 is `REFUSED_ABOVE_T0`, never a prompt
with nobody to answer it.

**Zeus's shell writes the worktree, every command recorded.** A command that
is not a pure read (`commands.reads_only`) runs with the worktree writable and git's objects,
refs and index for a git command, through `RecordedWrites` (`zeus.writes.ShellWrites`): the tree
before it parked as an undo unit, its lifecycle on the session log, a tracked source file it
changed put back unless the line is git's own operation, git's `config` and `hooks` guarded,
and in the session's worktree a commit of its own. A pure read runs sealed, with no snapshot. A
shell built with nowhere to record a write runs everything sealed. Every request is checked
against its own wrap before it runs (`Shell.request`): writable only where it is recorded.
Nothing is declared: `effects=` is retired, refused at the schema naming why.

**The tier is the command's, and approval opens what it needs.** The
tier is `max(the hint, the command table's row through the person's rules, the unsandboxed
floor)` and nothing on this path lowers it; above T0 the call needs authority (`ShellAuthority`,
the dispatch-time confirmation) and without it is `REFUSED_ABOVE_T0`,
never run. Approved, a command whose class needs the network (`NETWORK_CLASSES`: a push, a
fetch, an install) runs with it on; `network: true` is the model's hint for one no row reads as
networked, and is T2. Nothing refuses an approved network after: every backend is
all-or-nothing and no host is named.

**T0 is what the sandbox grants.** Where no backend enforces anything, nothing makes a
command sealed, so every command is T2 and classified `unsandboxed`: a person confirms it, or it
does not run. `allow_unsandboxed_shell`, the person's `sandbox` key for this shell's role, lifts
that floor and nothing else, and never for a command that writes git's own state
or names a path outside the repository. Where the configuration requires a sandbox
and there is none, nothing can run at all, and the call is `SANDBOX_REQUIRED` before anything is
built or anybody asked.

**Cap and tell.** Output under `shell_inline_cap` returns untouched. Over it, the head
that fits and a `Truncation` naming the total; the whole output is behind a handle either way,
so `continue_with` loses nothing.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from chickadee.approval.commands import (
    Classification,
    CommandTable,
    GitLine,
    needs_network,
    reads_only,
    runs_git,
    serves,
)
from chickadee.approval.grants import grantable
from chickadee.approval.prefs import (
    class_floor,
    command_denied,
    command_floor,
    command_rule,
)
from chickadee.approval.tier import Preferences, higher
from chickadee.context.handle import Handle
from chickadee.context.result_data import Completed
from chickadee.context.results import (
    CallCancelled,
    InvalidArgument,
    JobsFull,
    SecretPath,
    ToolResult,
    instructed,
)
from chickadee.errors import HarnessBug, RepoLeaseBusy
from chickadee.executor.policy import isolation_for, secret_path_denial
from chickadee.executor.types import (
    ExecRequest,
    ExecResult,
    NetworkEnforced,
    NetworkPolicy,
    PolicyClass,
    SandboxLevel,
    joined_output,
)
from chickadee.git_guard import guarded
from chickadee.isolation.base import level_of
from chickadee.paths import within
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import CommandBinding
from chickadee.tools.catalogue import capped
from chickadee.tools.spec import SHELL_INLINE_BYTES_DEFAULT

if TYPE_CHECKING:
    from chickadee.executor.executor import Executor
    from chickadee.tools.validate import Arguments

__all__ = [
    "AGENT",
    "SHELL_TIMEOUT_S",
    "ZEUS",
    "AskCut",
    "JobKind",
    "JobSpawn",
    "NetworkDenied",
    "PreferenceDenied",
    "RefusedAboveT0",
    "SandboxRequired",
    "Shell",
    "ShellAuthority",
    "ShellClaim",
    "ShellCommand",
    "ShellFailure",
    "ShellJobs",
    "ShellProfile",
    "Spawn",
    "TimedOut",
    "WriteDenied",
    "Yielded",
]

SHELL_TIMEOUT_S: Final = 600
"""Per command. A starting value: a test suite or a lockfile install is minutes, and a hard
interrupt reaps a command sooner than this whenever the person wants it gone."""

_NETWORK_SHAPE: Final = re.compile(
    r"could not resolve host|temporary failure in name resolution|name or service not known"
    r"|nodename nor servname|network is unreachable|getaddrinfo|eai_again|enetunreach"
    r"|failed to establish a new connection|connection refused",
    re.IGNORECASE,
)
_PORT: Final = r"(?:server_bind|\bbind\b|\blisten\b)"
_REFUSED: Final = r"(?:operation not permitted|permission denied|\beperm\b|\beacces\b)"
_BIND_SHAPE: Final = re.compile(
    rf"{_PORT}[^\n]*?{_REFUSED}|{_REFUSED}[^\n]*?{_PORT}"
    # Python's traceback puts the `.bind(` frame and the error on lines of their own.
    r"|(?<=\.bind\()[\s\S]{0,400}?\bPermissionError: \[Errno (?:1|13)\][^\n]*",
    re.IGNORECASE,
)
"""A server whose port the sandbox refused: Seatbelt's `(deny network*)` fails
`bind` with EPERM, which Node (`listen EPERM`), Go (`bind: operation not permitted`) and Python
(a `PermissionError` under a `.bind(` frame) each word their own way."""
_READ_ONLY_SHAPE: Final = re.compile(
    r"(?:'([^']+)'|\"([^\"]+)\"|(\S+))?:?\s*read-only file system", re.IGNORECASE
)


# -- the failures ---------------------------------------------------------


@dataclass(frozen=True)
class TimedOut:
    """`TIMED_OUT{command, limit_s}` — killed at the deadline; its partial output, if any, is
    the handle the result carries."""

    command: str
    limit_s: int
    failure: Literal["TIMED_OUT"] = "TIMED_OUT"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{command: {self.command!r}, limit_s: {self.limit_s}}}",
            "do not run it again as it is: read its partial output, then run a narrower "
            "command (one test file, fewer cases)",
        )


@dataclass(frozen=True)
class RefusedAboveT0:
    """`REFUSED_ABOVE_T0{tier, classification, missing}` — obtain authority; never retry the
    same call."""

    tier: Tier
    classification: str
    missing: str
    failure: Literal["REFUSED_ABOVE_T0"] = "REFUSED_ABOVE_T0"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{tier: {self.tier.value}, classification: {self.classification}, "
            f"missing: {self.missing!r}}}",
            f"do not send this command again: it needs {self.missing}. Do the work with a "
            "command that needs none, or report that it needs the person",
        )


@dataclass(frozen=True)
class NetworkDenied:
    """`NETWORK_DENIED{heuristic: true, reason}`: a sealed run whose output looks
    like a network failure. Always a shape read from the output, so always marked as one:
    nothing refuses the network after approval, so there is no
    case where it is a fact."""

    why: str
    do: str = (
        "the command looks to have needed the network: do not run it again as it is; work "
        "without the network, or report that it needs it"
    )
    failure: Literal["NETWORK_DENIED"] = "NETWORK_DENIED"

    def render(self) -> str:
        return instructed(f"{self.failure}{{heuristic: true, reason: {self.why!r}}}", self.do)


@dataclass(frozen=True)
class WriteDenied:
    """`WRITE_DENIED{path, heuristic}`: the sandbox refused a write outside what this
    shell may write. Read from the command's own error output, so marked as a shape."""

    path: str
    why: str
    heuristic: bool = True
    do: str | None = None
    """What to do instead, where it is not the sandbox's own refusal's."""
    failure: Literal["WRITE_DENIED"] = "WRITE_DENIED"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{path: {self.path}, heuristic: {str(self.heuristic).lower()}, "
            f"reason: {self.why!r}}}",
            self.do
            or f"do not write {self.path} with a command again: write only inside the "
            "worktree, with the edit tools",
        )


@dataclass(frozen=True)
class SandboxRequired:
    """`SANDBOX_REQUIRED{level, reason}`: the configuration requires an enforcing sandbox
    and the detected level is `level`, so no command can run here. Never retried: the fix is
    the person's, a sandbox or `require_sandbox` off."""

    level: SandboxLevel
    why: str
    failure: Literal["SANDBOX_REQUIRED"] = "SANDBOX_REQUIRED"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{level: {self.level.value}, reason: {self.why!r}}}",
            "no command can run here until the person sets up a sandbox: do not run commands; "
            "go on without them and say so in your answer",
        )


@dataclass(frozen=True)
class PreferenceDenied:
    """`PREFERENCE_DENIED{what, rule}`: the person's approval.toml forbids it, a
    command of a class they deny or a declared write to a path they deny. Refused before the
    tier, so nothing is asked and no grant covers it, attended or not. Never retried."""

    what: str
    """The command line, or the declared path."""
    rule: str
    """The rule that forbids it, with its file (`prefs.command_rule`, `prefs.path_rule`)."""
    failure: Literal["PREFERENCE_DENIED"] = "PREFERENCE_DENIED"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{what: {self.what!r}, rule: {self.rule!r}}}",
            "the person forbids this: do not send it again or work around the rule; do it "
            "another way, or say that it is forbidden",
        )


ShellFailure = (
    TimedOut
    | RefusedAboveT0
    | NetworkDenied
    | WriteDenied
    | SecretPath
    | SandboxRequired
    | PreferenceDenied
    | JobsFull
)


# -- authority ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Approved:
    """The person, a grant or `never_ask` approved a call that asked: it runs outside the
    sandbox. A value of its own, so "approved" is never read as "nobody asked"."""


APPROVED: Final = Approved()

OUTSIDE: Final = "outside the sandbox"
"""The classification of a call that asked to leave the sandbox (`sandbox: false`)."""

OUTSIDE_RECORD: Final = (
    "outside the sandbox, approved; effects outside the worktree are not recorded"
)
"""What a header says of a command that ran outside: what it did in the worktree
is a step, and anything else it did is not, and the record never claims otherwise."""

OUTSIDE_UNRECORDED: Final = "outside the sandbox, approved; nothing it changed is recorded"
"""The same, from a shell that records no writes at all: no worktree half to name."""

_SERVER_RAN_OUTSIDE: Final = (
    "[a server: it was asked, and runs outside the sandbox, where it can be reached. Send "
    "sandbox: false with every server; one the harness does not know as a server runs sealed, "
    "where nothing can reach it]"
)
"""The note on a server a row asked about, sent without `sandbox: false`."""


def _serve_outside(jobs: bool) -> str:
    """What a refused port tells the model to do: `background` only where this shell keeps
    jobs, so the box roles are never told an argument they lack."""
    also = " and background: true" if jobs else ""
    return (
        f"run it again with sandbox: false{also}, which asks the person and runs the server "
        "outside the sandbox; do not run it again as it is"
    )


@dataclass(frozen=True)
class ShellClaim:
    """What a command above T0 asks authority for. `operation_id`
    is the call's own, so the confirmation names the call it confirms."""

    command: str
    cwd: str
    tier: Tier
    classification: str
    network: bool
    """Whether the call asked for the network: the whole network, for this command."""
    operation_id: str
    idempotent: bool
    where: str = ""
    """Where the command may write, when it runs with the worktree writable:
    the session's worktree, or the person's checkout; empty for a sealed command."""
    grant: CommandBinding | None = None
    """What *always* would bind for this call (`approval/grants.py`), or `None` for *once*
    only. The router's, from the argv and the command table; never the model's."""
    asker: str = "zeus"
    """Who asks: the shell's role (`zeus`, `athena`, `worker`), so the person is told whose
    command it is."""
    task_id: str | None = None
    """The box task the command belongs to, when a box role asks inside one."""
    found: Classification | None = None
    """What the shell's table read the command as: the prompt offers *always* from
    this, so the grant it offers is the one this table would bind."""


class ShellAuthority(Protocol):
    """The dispatch-time confirmation, and the grants that spare a second one. `True` is
    authority for this call; the engine wires it to the router and the interaction port.
    `cancel` is the call's stop: set while the person is asked, the ask returns at once and
    is no authority."""

    def authorize(self, claim: ShellClaim, cancel: threading.Event | None = None) -> bool: ...


@runtime_checkable
class AskCut(Protocol):
    """A call's stop that may end the person's permission question where it would not end the
    command: a box round's token (`box/tells.RoundToken`), which a message from Zeus sets while
    a question waits, since nothing has run yet, though a networked command
    already running finishes. `asking` spans the question."""

    def asking(self) -> AbstractContextManager[None]: ...


# -- the profiles ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellProfile:
    """What differs between the shells; the rest is one code."""

    name: Literal["zeus", "worker", "box", "agent"]
    declarations: Literal["honoured", "above_t0", "network_only"]
    """`honoured`: `network: true` opens the network for the command and raises its tier.
    `above_t0`: any declaration is a request for more than the profile gives, so a
    refusal. `network_only`: `network: true` is honoured; `effects=` is refused, since the
    profile writes the workspace already."""
    may_ask: bool
    """Whether a call above T0 may be put to a `ShellAuthority`. The packet worker's may not:
    there is no person in its box, so a prompt there is a hang, not a question."""
    writes: Literal["none", "worktree"] = "none"
    """`worktree`: every command runs with the workspace writable (`VERIFICATION`, its `.git`
    read-only), each recorded as a step by the box around it. `none`: the
    workspace is writable only beside a declared write."""


ZEUS: Final = ShellProfile("zeus", declarations="honoured", may_ask=True)
BOX: Final = ShellProfile("box", declarations="network_only", may_ask=True, writes="worktree")
"""The task loop's shell, the worker's and Athena's."""
AGENT: Final = ShellProfile("agent", declarations="above_t0", may_ask=False)
"""A review agent's shell: T0 only, any declaration refused as a
request for more, never asked, since no person is in an agent's loop. The packet worker's
`WORKER` was this profile; it retired with the packet loop."""


JobKind = Literal["task", "server"]
"""A job's kind: a task notifies on every exit, a server only on one nobody
asked for, and only a task has a runtime bound."""


@dataclass(frozen=True)
class Yielded:
    """A command still running at its yield: it runs on as a job, and its
    call answers with what it printed so far."""

    job_id: str
    kind: JobKind
    stdout: str
    stderr: str
    wall_ms: int
    sandbox: SandboxLevel
    network_enforced: NetworkEnforced


Spawn = Callable[[ExecRequest, str, "threading.Event | None"], "ExecResult | Yielded"]
"""How a checked request runs: `Executor.run` itself, the worker's, which runs it inside the
effect lifecycle the run journal holds (`box/worker_shell.py`), or Zeus's job table, whose
command may still be running when it answers (`zeus/jobs.py`)."""


class JobSpawn(Protocol):
    """One call's spawn through Zeus's job table (`ShellJobs.spawn`)."""

    @property
    def held(self) -> tuple[str, ...]:
        """Once it has run: the jobs whose count (`zeus.max_jobs`) kept the command from
        becoming a job at its yield, so it ran to its end in its call; empty when none did."""
        ...

    @property
    def ceiling(self) -> bool:
        """Once it has run: it was killed in its call for passing `zeus.job_max_log_bytes`."""
        ...

    def __call__(
        self, request: ExecRequest, operation_id: str, cancel: threading.Event | None, /
    ) -> ExecResult | Yielded: ...


class ShellJobs(Protocol):
    """Where Zeus's shell runs a command that may become a job (`zeus/jobs.JobTable`)."""

    def full(self) -> tuple[str, ...]:
        """The running jobs' ids when no other may start (`zeus.max_jobs`), else empty."""
        ...

    def spawn(
        self,
        *,
        executor: Executor,
        root: Path,
        store: Callable[[bytes], Handle],
        background: bool,
        kind: JobKind,
        line: str,
    ) -> JobSpawn:
        """The spawn for one call, through this shell's executor in its workspace: it answers
        `Yielded` once the command becomes a job."""
        ...


class RecordedWrites(Protocol):
    """Where a command that may write runs (`zeus.writes.ShellWrites`):
    its snapshot, its lifecycle on the session log, the source and denied paths it changed put
    back, git's own files guarded and, in the session's worktree, its commit. A shell built with
    none runs every command sealed, having nowhere to record a write."""

    @property
    def where(self) -> str: ...

    def git_dirs(self) -> tuple[Path, ...]: ...

    def run(
        self,
        request: ExecRequest,
        operation_id: str,
        cancel: threading.Event | None,
        spawn: Spawn,
        *,
        git: GitLine,
    ) -> tuple[ExecResult | Yielded, tuple[str, ...]]:
        """The run's result, and the notes its result carries: the commit, a path put back.
        `git`: how much of the line's source changes git's own history accounts for, and so
        keeps (`CommandTable.git_line`). A command that yielded ends its step at the yield.
        """
        ...


# -- the shell ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellCommand:
    """A call's arguments, checked: what the schema admits and the meaning also does."""

    command: str
    cwd: Path
    cwd_text: str
    network: bool
    """`network: true`: the call's own hint that the command needs the network."""
    idempotent: bool
    needs_network: bool = False
    """The command's class needs the network (`commands.NETWORK_CLASSES`)."""
    writable: bool = False
    """It runs with the worktree writable, recorded as a step: a shell with
    somewhere to record writes, and a line that is not a pure read."""
    found: Classification | None = None
    """What the shell's table read the line as, once, for every check after the parse."""
    outside: bool = False
    """`sandbox: false`: the call asks to run outside the sandbox, which makes it
    T2, so the person is asked."""
    background: bool = False
    """`background: true`: it becomes a job at once."""
    kind: JobKind = "task"
    """The job it may become: the call's, or `server` for a `dev_server` row (decision 2)."""

    @property
    def opens_network(self) -> bool:
        """Approved, the command runs with the network on: hinted, or its class needs it."""
        return self.network or self.needs_network


class Shell:
    """`shell(command, cwd, network=, idempotent=)`, under `profile`.

    `store` keeps a command's output in the session's blob store and returns its handle
    (`Reader.store("output", …)`), so every result — including a timed-out or cancelled one —
    carries the whole output behind a handle. `commands` is the `[[commands]]` table every
    tier is read from; `None` is its executor's. `prefs` gives the person's
    approval.toml, read per call so an edit applies at the next one; `None` is no
    preferences, for a caller with no config. `allow_unsandboxed_shell` is the
    person's `sandbox.allow_unsandboxed_shell`, read by whoever builds the shell."""

    def __init__(
        self,
        executor: Executor,
        workspace: Path,
        store: Callable[[bytes], Handle],
        *,
        profile: ShellProfile = ZEUS,
        authority: ShellAuthority | None = None,
        spawn: Spawn | None = None,
        commands: CommandTable | None = None,
        timeout_s: int = SHELL_TIMEOUT_S,
        inline_cap: int = SHELL_INLINE_BYTES_DEFAULT,
        allow_unsandboxed_shell: bool = False,
        writes: RecordedWrites | None = None,
        asker: str | None = None,
        task: Callable[[], str | None] | None = None,
        prefs: Callable[[], Preferences] | None = None,
        jobs: ShellJobs | None = None,
        team: Callable[[], str | None] | None = None,
    ) -> None:
        if authority is not None and not profile.may_ask:
            raise HarnessBug(f"the {profile.name} shell is never given an authority (D74)")
        self._profile = profile
        self._executor = executor
        self._workspace = workspace.resolve()
        self._store = store
        self._authority = authority
        self._team = team
        """The session's run working in the background now, or None: Zeus's shell
        may not write beside it. Another session's run is the repository lease's to refuse."""
        self._spawn: Spawn = spawn if spawn is not None else self._run_unrecorded
        # The executor's table is the engine's: its pre-filter and this tier read one.
        self._commands = commands if commands is not None else executor.commands
        self._timeout_s = timeout_s
        self._cap = inline_cap
        self._allow_unsandboxed = allow_unsandboxed_shell
        if writes is not None and not profile.may_ask:
            raise HarnessBug(f"the {profile.name} shell never writes the workspace (D74)")
        self._writes = writes
        self._asker = asker if asker is not None else profile.name
        self._task = task
        self._prefs = prefs
        self._jobs = jobs
        """Zeus's job table; `None` for every shell whose commands never run on."""

    def run(
        self,
        args: Arguments,
        *,
        operation_id: str,
        cancel: threading.Event | None = None,
    ) -> ToolResult | ShellFailure | InvalidArgument | CallCancelled:
        """One command. `args` are the tool's own arguments, already valid against its schema;
        what the schema cannot say is checked here."""
        prefs = self._preferences()
        call = self.parse(args, prefs)
        if isinstance(call, InvalidArgument | RefusedAboveT0 | PreferenceDenied):
            return call
        working = self._team() if self._team is not None and call.writable else None
        if working is not None:
            # The team writes this workspace while its run is live, so a command that
            # may write is refused before anything is asked or spawned; a pure read still runs.
            return WriteDenied(
                str(self._workspace),
                f"{working} is working in this workspace",
                heuristic=False,
                do="pause the team first, or wait for it to end; reading is fine meanwhile",
            )
        full = self._jobs.full() if self._jobs is not None else ()
        if call.background and full:
            # Refused before anything is asked or spawned.
            return JobsFull(full)
        if self._unsandboxed():
            # Asked before the tier and the authority, so a call whose answer can only
            # be no asks nobody first.
            return SandboxRequired(
                SandboxLevel.NONE,
                "require_sandbox is set and no sandbox backend is available here",
            )
        request = self.request(call)
        verdict = self._refusal(request, call, operation_id, prefs, cancel)
        refused = None if isinstance(verdict, Approved) else verdict
        if isinstance(verdict, Approved):
            # Approved, the command leaves the sandbox, the network on. Where this
            # shell records writes, it runs through them, so what it changes in the worktree is
            # a step that can be undone, whatever its class.
            call = replace(call, writable=call.writable or self._writes is not None)
            request = replace(self.request(call), unsandboxed=True, network=NetworkPolicy(True))
        stopped = cancel is not None and cancel.is_set()
        if isinstance(refused, RefusedAboveT0) and stopped:
            # A person who aborted at the confirmation stopped the call, which is not the
            # same answer as declining it.
            return CallCancelled("shell")
        if refused is not None:
            return refused
        if stopped:
            return CallCancelled("shell")
        notes: tuple[str, ...] = ()
        if isinstance(verdict, Approved) and call.kind == "server" and not call.outside:
            # Asked by its row, so it ran as it should; said so the model sends the flag
            # for a server no row names, which nothing would ask about.
            notes = (_SERVER_RAN_OUTSIDE,)
        job = (
            None
            if self._jobs is None
            else self._jobs.spawn(
                executor=self._executor,
                root=self._workspace,
                store=self._store,
                background=call.background,
                kind=call.kind,
                line=call.command,
            )
        )
        spawn: Spawn = self._spawn if job is None else job
        try:
            if call.writable and self._writes is not None:
                result, written = self._writes.run(
                    request,
                    operation_id,
                    cancel,
                    spawn,
                    git=self._commands.git_line(call.command),
                )
                notes = (*notes, *written)
            else:
                result = spawn(request, operation_id, cancel)
        except RepoLeaseBusy as busy:
            # A run holds the repository; a write now would land between its packets.
            return WriteDenied(".", f"{busy}; run it again once the run ends")
        if isinstance(result, Yielded):
            return self._yielded(call, request, result, operation_id, prefs, notes)
        if job is not None and job.ceiling:
            notes = (
                *notes,
                "[killed by the harness: its output passed zeus.job_max_log_bytes; a command "
                "that prints this much belongs in a file]",
            )
        held = job.held if job is not None else ()
        if held:
            notes = (
                *notes,
                f"[{len(held)} jobs were running ({', '.join(held)}), the most zeus.max_jobs "
                "allows, so this command could not become a job and ran to its end]",
            )
        return self._result(call, request, result, operation_id, prefs, notes)

    def _preferences(self) -> Preferences:
        """The person's preferences for this call, read now."""
        return self._prefs() if self._prefs is not None else Preferences()

    def _run_unrecorded(
        self, request: ExecRequest, operation_id: str, cancel: threading.Event | None
    ) -> ExecResult:
        """Zeus's spawn: its call is recorded by the session's tool line, not by a run's
        effect lifecycle."""
        return self._executor.run(request, operation_id=operation_id, cancel=cancel)

    def _unsandboxed(self) -> bool:
        """The executor's refusal, which it makes before it looks at the network."""
        return self._executor.require_sandbox and self._level() is SandboxLevel.NONE

    def _level(self) -> SandboxLevel:
        """The level a command runs at, read as `Executor.run` reads it."""
        return level_of(self._executor.backend)

    # -- before anything spawns -------------------------------------------------------------

    def request(self, call: ShellCommand) -> ExecRequest:
        """The request a call becomes, checked against the wrap it will get: a request whose
        wrap would leave the workspace writable is a harness bug, raised before any spawn."""
        worktree = self._profile.writes == "worktree"
        if worktree:
            # Every backend binds only `cwd` writable, so the root is named whatever the `cwd`;
            # the worktree's `.git` stays read-only over it, hooks and config untouchable.
            also: tuple[Path, ...] = (self._workspace,)
            # And what a tool outside the sandbox runs from the worktree,
            # but for what lies in `.git`, read-only whole already.
            git = self._workspace / ".git"
            whole = (git,) if git.exists() else ()
            kept = (
                *whole,
                *(p for p in guarded(self._workspace) if within(git, p) is None),
            )
        elif call.writable and self._writes is not None:
            # The worktree writable whatever the `cwd`, and git's objects, refs
            # and index for a git command. Its settings, and what a tool outside the sandbox
            # runs, stay read-only: a command may still be running as a job long after
            # its step's guard has looked.
            git = self._writes.git_dirs() if runs_git(call.command) else ()
            also = (self._workspace, *git)
            kept = guarded(self._workspace)
        else:
            also, kept = (), ()
        request = ExecRequest(
            command=call.command,
            cwd=call.cwd,
            policy=PolicyClass.WORKTREE if worktree or call.writable else PolicyClass.ACTION,
            timeout_s=self._timeout_s,
            network=NetworkPolicy(call.opens_network),
            also_writable=also,
            read_only=kept,
            idempotent=call.idempotent,
            # What the run record names it by: the command, now that no call carries a purpose
            # line; the stream shows `ran <label>: exit N`.
            label=call.command,
        )
        wrap = isolation_for(
            request, deny_read=self._executor.deny_read, denylist=self._executor.denylist
        )
        # Writable only where each call is recorded: the box's steps, or Zeus's `RecordedWrites`
        # anything else writable is a harness bug.
        if wrap.repo_writable and not call.writable and not worktree:
            raise HarnessBug(
                f"the {self._profile.name} shell would run with the workspace writable "
                "(invariant 12)"
            )
        return request

    def parse(
        self, args: Arguments, prefs: Preferences | None = None
    ) -> ShellCommand | InvalidArgument | RefusedAboveT0 | PreferenceDenied:
        """`prefs` is the call's preferences, read by `run`; None reads them now."""
        command = args.text("command")
        if not command.strip():
            return InvalidArgument(
                "command",
                "the command is empty",
                do="send command with the command line to run",
            )
        declared = self._declared(args)
        if declared is not None:
            return declared
        refused = self._commands.git_refusal(command) if self._writes is not None else None
        if refused is not None:
            # What this git would put in git's history is past the reach of the
            # put-back after the command, which restores files. Refused before anyone is asked.
            return InvalidArgument("command", refused.why, do=refused.do)
        network = args.boolean("network", False)
        given = args.text_or_none("cwd")
        cwd_text = "." if given is None else given
        cwd = within(self._workspace, cwd_text)
        if cwd is None:
            return InvalidArgument(
                "cwd",
                f"{cwd_text!r} is outside the workspace",
                do="leave cwd out to run at the repository root, or send a folder inside it",
            )
        if not cwd.is_dir():
            return InvalidArgument(
                "cwd",
                f"{cwd_text!r} is not a directory in the workspace",
                do="leave cwd out, or send a folder glob lists, from the repository root",
            )
        # Whether recovery may run the command again.
        idempotent = args.boolean("idempotent", False)
        sandbox = args.boolean("sandbox", None)
        given_background = args.boolean("background", None)
        background = bool(given_background)
        given_kind = args.text_or_none("kind")
        if self._jobs is None and (background or given_kind is not None):
            # Only Zeus's shell keeps jobs; the box roles are never shown
            # these arguments (`RolePolicy.withheld`), so this is a harness without a table.
            return InvalidArgument(
                "background" if background else "kind",
                "this shell runs no jobs (D201)",
                do="send the command again without background and kind; it runs to its end",
            )
        if given_kind not in (None, "task", "server"):
            return InvalidArgument(
                "kind",
                f"{given_kind!r}: a job is a task or a server",
                do='send kind as "task" or "server", or leave it out',
            )
        found = self._commands.classify_line(command)
        kind: JobKind = (
            "server"
            if given_kind == "server" or (given_kind is None and serves(found))
            else "task"
        )
        # A call that says it starts a server says it needs to be reached, which
        # only outside the sandbox it can be; `sandbox: true` sent with it is kept.
        outside = sandbox is False or (sandbox is None and given_kind == "server")
        if given_background is None and kind == "server" and self._jobs is not None:
            # A server never ends, so it is a job at once rather than after the yield
            # window; `background: false` still waits it out.
            background = True
        return ShellCommand(
            command,
            cwd,
            cwd_text,
            network,
            idempotent,
            needs_network=needs_network(found),
            writable=self._writes is not None and not reads_only(command, found),
            found=found,
            outside=outside,
            background=background,
            kind=kind,
        )

    def opens_network(self, command: str, hint: bool) -> bool:
        """Whether `command`, approved, runs with the network on: hinted, or its
        class needs it, as this shell's table reads it. The one answer for the step around a
        box command, which resume reads (`box/steps.py`)."""
        return hint or needs_network(self._commands.classify_line(command))

    def asks(self, command: str, hint: bool, outside: bool = False) -> bool:
        """Whether `command` is put to the person (or a grant), and so runs outside the
        sandbox once approved: the one answer for the step around a box command,
        which resume reads (`box/steps.py`) and never runs such a step again."""
        found = self._commands.classify_line(command)
        call = ShellCommand(
            command,
            self._workspace,
            ".",
            hint,
            False,
            needs_network=needs_network(found),
            found=found,
            outside=outside,
        )
        tier, _ = self._tier(call, self._preferences())
        return tier is not Tier.T0

    def _declared(self, args: Arguments) -> RefusedAboveT0 | None:
        """Under a profile that reads a declaration as a request above T0 (the worker's), the
        declaration is the refusal: network is T2."""
        if self._profile.declarations != "above_t0":
            return None
        if args.boolean("network", False):
            return RefusedAboveT0(
                Tier.T2, "declared network", "network, which this profile is off"
            )
        if not args.boolean("sandbox", True):
            return RefusedAboveT0(
                Tier.T2, OUTSIDE, "a run outside the sandbox, which this profile never asks"
            )
        return None

    def _refusal(
        self,
        request: ExecRequest,
        call: ShellCommand,
        operation_id: str,
        prefs: Preferences,
        cancel: threading.Event | None,
    ) -> SecretPath | PreferenceDenied | RefusedAboveT0 | Approved | None:
        """Why the call cannot run, `APPROVED` when it was asked and the person (or a grant,
        or `never_ask`) said yes, or None for a T0 call nobody is asked about."""
        # The executor's pre-filter refuses a deny-listed path too, with a string; asked here
        # first so the refusal is typed and nothing is built for a call that cannot run.
        secret = secret_path_denial(request, self._executor.denylist)
        if secret is not None:
            return SecretPath(secret.denial.path, secret.denial.pattern)
        found = call.found
        forbidden = None if found is None else command_denied(prefs, found)
        if forbidden is not None:
            # Before the tier, so no grant covers it and nobody is asked.
            return PreferenceDenied(call.command, command_rule(prefs, forbidden) or "")
        tier, classification = self._tier(call, prefs)
        if tier is Tier.T0:
            return None
        claim = ShellClaim(
            call.command,
            call.cwd_text,
            tier,
            classification,
            call.opens_network,
            operation_id=operation_id,
            idempotent=call.idempotent,
            where=self._writes.where if call.writable and self._writes is not None else "",
            # A grant binds whether the call opened the network, so one given
            # without it never covers a call with it.
            grant=grantable(found, network=call.opens_network),
            asker=self._asker,
            task_id=None if self._task is None else self._task(),
            found=found,
        )
        authority = self._authority if self._profile.may_ask else None
        with cancel.asking() if isinstance(cancel, AskCut) else nullcontext():
            granted = authority is not None and authority.authorize(claim, cancel)
        if granted:
            return APPROVED
        return RefusedAboveT0(
            tier, classification, f"authority for {classification} at {tier.value}"
        )

    def _tier(self, call: ShellCommand, prefs: Preferences) -> tuple[Tier, str]:
        """`max(declared_floor, prefilter_floor, unsandboxed_floor)`, where the pre-filter is
        the command table: every simple command in the line, and the table's
        pattern rows. Each term can only raise. The classification names what set the tier:
        the class of the row that did, or the declaration when it is higher; an unrecognised,
        undeclared command is T0 and `sealed`. With no sandbox the floor is T2, and it
        names the tier only where it raised it: a row or a declaration already at T2 keeps its
        own name, so the person confirming `rm -rf` is told it is destructive."""
        # The hint is T2, and so is `sandbox: false`; a class that needs the
        # network has its own floor.
        declared = Tier.T2 if call.network or call.outside else Tier.T0
        label = OUTSIDE if call.outside else "network" if call.network else "sealed"
        found = call.found
        tier = declared
        if found is not None:
            # The table's term through the person's class rules: `allow` lowers a
            # lowerable class, `ask` raises a lower one; the declaration stays as it is.
            floor = command_floor(prefs, found)
            tier = higher(declared, floor)
            if tier is floor:
                # The table's own choice among equal floors (`Classification.top`: a
                # destructive command names the tier), unless a rule moved another class.
                top = found.top
                setter = (
                    top
                    if class_floor(prefs, top) is floor
                    else next(m for m in found.matches if class_floor(prefs, m) is floor)
                )
                rule = command_rule(prefs, setter.cls)
                label = setter.cls.value if rule is None else f"{setter.cls.value} ({rule})"
        unsealed = self._level() is SandboxLevel.NONE
        # The lift never covers git's own state or a path outside the
        # repository, which nothing stops unsealed; the person's grant still may.
        beyond = (
            self._commands.beyond_the_worktree(call.command, self._workspace)
            if unsealed and self._allow_unsandboxed
            else None
        )
        floored = unsealed and (not self._allow_unsandboxed or beyond is not None)
        if floored and higher(tier, Tier.T2) is not tier:
            return Tier.T2, "unsandboxed" if beyond is None else f"unsandboxed, {beyond}"
        return tier, label

    # -- after it ran ---------------------------------------------------------------------

    def _result(
        self,
        call: ShellCommand,
        request: ExecRequest,
        result: ExecResult,
        operation_id: str,
        prefs: Preferences,
        written: tuple[str, ...] = (),
    ) -> ToolResult:
        output = joined_output(result.stdout, result.stderr)
        handle = self._store(output.encode("utf-8"))
        content, truncation = capped(output, handle, limit=self._cap, unit="bytes")
        header = self._header(call, request, result, operation_id, handle.id, prefs)
        notes: list[str] = []
        stopped = (
            TimedOut(call.command, request.timeout_s).render()
            if result.timed_out
            else CallCancelled("shell", handle.id).render()
            if result.cancelled
            else None
        )
        if stopped is not None:
            # The box's header stays the first line, which an evicted result keeps (its
            # output's handle, `boxctx/evict.live_receipt`); Zeus's reads the stop first.
            header = (
                f"{header}\n{stopped}"
                if self._profile.writes == "worktree"
                else f"{stopped}\n{header}"
            )
        if stopped is None and result.exit_code != 0 and not request.unsandboxed:
            # Not after a run outside: nothing sandboxed it, so no shape is the sandbox's.
            notes.extend(self._shapes(call, result))
        if result.violations and not call.writable:
            paths = ", ".join(v.path for v in result.violations)
            notes.append(
                f"[effect check: wrote {paths}; this shell is read-only and the sandbox "
                f"({result.sandbox.value}) did not hold it]"
            )
        notes.extend(written)
        if not output:
            notes.append("[no output]")
        # Named as `subprocess` names them; the output only when it was
        # shown whole, so the data never holds more than the text.
        whole = truncation is None
        completed = Completed(
            result.exit_code,
            result.stdout if whole else None,
            result.stderr if whole else None,
        )
        return ToolResult(
            handle=handle,
            content=content,
            header=header,
            truncation=truncation,
            notes=tuple(notes),
            data=completed,
        )

    def _yielded(
        self,
        call: ShellCommand,
        request: ExecRequest,
        yielded: Yielded,
        operation_id: str,
        prefs: Preferences,
        notes: tuple[str, ...],
    ) -> ToolResult:
        """A command that runs on as a job: what it printed so far, capped
        and behind a handle as any result is, its step's notes, and what it now is. The
        `job` tool reads the rest of its output."""
        output = joined_output(yielded.stdout, yielded.stderr)
        handle = self._store(output.encode("utf-8"))
        content, truncation = capped(output, handle, limit=self._cap, unit="bytes")
        tier, _ = self._tier(call, prefs)
        # Its step recorded what it wrote inside the window; nothing records the rest.
        repo = _protection(call, request, yielded.sandbox, recorded=False)
        header = (
            f"running as {yielded.job_id} ({yielded.kind}) · {yielded.wall_ms} ms so far · "
            f"sandbox {yielded.sandbox.value} · network {yielded.network_enforced} · {repo} · "
            f"{tier.value} · {operation_id}"
        )
        said = (
            f"[still running as {yielded.job_id}; you will be told when it exits, so do not "
            "poll. What it writes from now on is not recorded and cannot be undone]"
        )
        return ToolResult(
            handle=handle,
            content=content,
            header=header,
            truncation=truncation,
            notes=(*notes, said),
        )

    def _header(
        self,
        call: ShellCommand,
        request: ExecRequest,
        result: ExecResult,
        operation_id: str,
        output: str,
        prefs: Preferences,
    ) -> str:
        """The profile the command ran under, as facts, so *sealed, and failed* is always
        something Zeus has rather than something it infers."""
        tier, _ = self._tier(call, prefs)
        repo = _protection(call, request, result.sandbox, recorded=True)
        head = (
            f"exit {result.exit_code} · {result.wall_ms} ms · sandbox {result.sandbox.value} · "
            f"network {result.network_enforced} · {repo} · {tier.value} · {operation_id}"
        )
        if self._profile.writes == "worktree":
            # The box shell writes the worktree, every call a step, and its result may be
            # evicted to this line, so the output's handle is named here.
            repo = (
                OUTSIDE_RECORD
                if request.unsandboxed
                else "worktree writable, .git read-only"
                if result.sandbox is not SandboxLevel.NONE
                else "repo not protected"
            )
            head = (
                f"exit {result.exit_code} · {result.wall_ms} ms · sandbox "
                f"{result.sandbox.value} · network {result.network_enforced} · {repo} · "
                f"{tier.value} · {operation_id} · output {output}"
            )
        return head

    def _shapes(self, call: ShellCommand, result: ExecResult) -> list[str]:
        """The sealed-failure shapes, each marked for what it is: read from the
        output. A refused port is read before the network's shape, which it is a case of."""
        text = f"{result.stdout}\n{result.stderr}"
        out: list[str] = []
        if result.network_enforced == "off":
            refused = _BIND_SHAPE.search(text)
            if refused is not None:
                # The last line: a traceback's match runs from its `.bind(` frame to the error.
                said = refused.group(0).splitlines()[-1].strip()
                why = (
                    f"sealed run, output reads {said!r}: the sandbox refused the server its "
                    "port. A server the person opens, or a check reaches, runs outside the "
                    "sandbox"
                )
                do = _serve_outside(self._jobs is not None)
                out.append(NetworkDenied(why, do=do).render())
            hit = None if refused is not None else _NETWORK_SHAPE.search(text)
            if hit is not None:
                why = (
                    f"sealed run, output reads {hit.group(0)!r}; run it again with "
                    "network: true, which asks the person and runs it outside the sandbox"
                )
                out.append(NetworkDenied(why).render())
        hit = _READ_ONLY_SHAPE.search(result.stderr)
        if hit is not None:
            path = next((g for g in hit.groups() if g), call.cwd_text).rstrip(":")
            why = (
                "read-only file system: the sandbox writes only the worktree; to write "
                f"{path} run it again with sandbox: false, which asks the person"
            )
            out.append(WriteDenied(path, why).render())
        return out


def _protection(
    call: ShellCommand, request: ExecRequest, sandbox: SandboxLevel, *, recorded: bool
) -> str:
    """What the wrap protected, as Zeus's header says it. Read-only is what the wrap enforces;
    with no sandbox nothing does, and a header that said so anyway would overstate the
    protection the run had. `recorded`: whether its writes are, as a job's after its
    yield are not."""
    if request.unsandboxed:
        return OUTSIDE_RECORD if call.writable and recorded else OUTSIDE_UNRECORDED
    if sandbox is SandboxLevel.NONE:
        return "repo not protected"
    if call.writable:
        return "worktree writable, recorded" if recorded else "worktree writable, not recorded"
    return "repo read-only"
