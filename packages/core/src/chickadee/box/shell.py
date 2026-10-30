"""The box shell: the `exec` port the task worker's and Athena's `shell` tool
runs on.

**One shell, the box profile.** The command is `chickadee.shell.Shell`'s, under `BOX`: the
worktree writable and its `.git` read-only, the network off unless `network: true` asks for it
(T2, so it asks), credentials stripped, and every command above T0 put to the session's
authority, naming who asks and for which task. What is this module's is what the box adds
around it:

- **Offered where a seal holds, or where the person opted out of one.**
  Where the backend enforces nothing, `shell_for` answers `None` while `require_sandbox` is set,
  so a role is never shown a tool that could not run; with it off, the shell is offered
  unsealed, and the floor puts every command to the person (`Shell._tier`), unless the
  person lifted it for that role (`sandbox.allow_unsandboxed_shell.<role>`).
- **Every command is one `exec` line** in the run journal (`Executor.exec_event`), written when
  it ran, under the task it belongs to. Its writes are the step's: the `worker_step` effect
  around the call records the trees before and after (`box/steps.py`), so this module keeps no
  lifecycle of its own.
- **Its output is the run's** (invariant 14): stored through the run journal's `put_blob`, and
  named by the handle the box report would give the same bytes.

Composed by the engine's episode (`delegation._Composed`) through `shell_for`.

**Concurrency.** A `BoxShell` belongs to one episode, which serves one tool call at a time on
its thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from chickadee.approval.tier import Preferences
from chickadee.box.report import stored_handle
from chickadee.config import ShellLift
from chickadee.context.handle import Handle
from chickadee.context.results import CallCancelled, InvalidArgument, ToolResult
from chickadee.errors import HarnessBug
from chickadee.executor.executor import Executor
from chickadee.executor.types import ExecRequest, ExecResult
from chickadee.git_guard import held, put_back
from chickadee.isolation.base import SandboxBackend
from chickadee.journal.port import JournalPort
from chickadee.schema.execution import SandboxLevel
from chickadee.schema.journal import BoxRoleName, EventKind, ExecP
from chickadee.shell import BOX, Shell, ShellAuthority, ShellFailure
from chickadee.tools.validate import Arguments

__all__ = ["BoxShell", "BoxShellFor", "RunSuspended", "sealed_level", "shell_for"]


class RunSuspended(BaseException):
    """The run is pausing because the process is leaving, and a command it ran or was
    asking about returned while it did. Not an `Exception`, so it leaves as a kill would: the
    step's effect is left started (`journal.effects.perform`), nothing answers the call, and
    resume continues it asking the person again where nobody
    answered. The episode reads it as `Paused` (`box/episode.BoxRun`)."""


def sealed_level(backend: SandboxBackend) -> SandboxLevel | None:
    """The level `backend` enforces, or `None` when it enforces nothing: where a shell is
    offered at all (moved from `box/worker_shell.py`, which retires with the packet loop)."""
    if not backend.available():
        return None
    level = backend.level()
    return None if level is SandboxLevel.NONE else level


@dataclass(frozen=True)
class BoxShellFor:
    """Builds a role's box shell over the run's journal, naming the role and reading the task it
    holds when a command asks. `sealed` says whether its backend seals what the shells run: the
    one answer the composition's prompts read."""

    sealed: bool
    build: Callable[[JournalPort, BoxRoleName, Callable[[], str | None]], BoxShell]

    def __call__(
        self, journal: JournalPort, asker: BoxRoleName, task: Callable[[], str | None]
    ) -> BoxShell:
        return self.build(journal, asker, task)


def shell_for(
    workspace: Path,
    backend: SandboxBackend,
    executor_for: Callable[[JournalPort], Executor],
    authority: ShellAuthority | None,
    prefs: Callable[[], Preferences] | None = None,
    *,
    require_sandbox: bool = True,
    lifts: ShellLift | None = None,
    suspend: threading.Event | None = None,
) -> BoxShellFor | None:
    """The box shell for episodes in `workspace`, or `None` where `backend` enforces nothing
    and `require_sandbox` is set: a role is refused a shell rather than given one that could
    not run. `executor_for` builds the run's executor over its journal, as the
    run's own is built, on the same backend; `prefs` gives the person's approval.toml, read
    per command; `lifts` is the person's `sandbox.allow_unsandboxed_shell`, each
    role's shell given its own; `suspend` is the episode's pause."""
    sealed = sealed_level(backend) is not None
    if not sealed and require_sandbox:
        return None
    lifted = lifts if lifts is not None else ShellLift()

    def build(
        journal: JournalPort, asker: BoxRoleName, task: Callable[[], str | None]
    ) -> BoxShell:
        return BoxShell(
            executor_for(journal),
            workspace,
            journal,
            authority,
            asker=asker,
            task=task,
            prefs=prefs,
            allow_unsandboxed_shell=lifted.of(asker),
            suspend=suspend,
        )

    return BoxShellFor(sealed, build)


class BoxShell:
    """`ExecPort` for the task worker and Athena: `Shell` under `BOX`, each command one `exec`
    line, its output a run blob, its asker and task named on every claim."""

    def __init__(
        self,
        executor: Executor,
        workspace: Path,
        journal: JournalPort,
        authority: ShellAuthority | None,
        *,
        asker: BoxRoleName,
        task: Callable[[], str | None],
        prefs: Callable[[], Preferences] | None = None,
        allow_unsandboxed_shell: bool = False,
        suspend: threading.Event | None = None,
    ) -> None:
        """`allow_unsandboxed_shell` is this role's lift of the floor;
        `suspend`, set with the hard stop when the run pauses."""
        if sealed_level(executor.backend) is None and executor.require_sandbox:
            # `shell_for` refuses such a machine before an episode is composed; reaching here
            # is a composition that skipped it.
            raise HarnessBug(
                "the box shell was built on a backend that enforces nothing, under "
                "require_sandbox"
            )
        self._executor = executor
        self._workspace = workspace
        self._journal = journal
        self._task = task
        self._suspend = suspend
        self._shell = Shell(
            executor,
            workspace,
            self._store,
            profile=BOX,
            authority=authority,
            spawn=self._spawn,
            asker=asker,
            task=task,
            prefs=prefs,
            allow_unsandboxed_shell=allow_unsandboxed_shell,
        )

    @property
    def sealed(self) -> bool:
        """Whether a sandbox seals what this shell runs. Unsealed, every command may reach the
        network and write outside the worktree, so its step is never run again."""
        return sealed_level(self._executor.backend) is not None

    def opens_network(self, command: str, hint: bool, outside: bool = False) -> bool:
        """Whether a command runs with the network on: unsealed, every command does; sealed,
        as its class or its hint says (`Shell.opens_network`), and every command that asks,
        which approved runs outside the sandbox (`Shell.asks`)."""
        return (
            not self.sealed
            or self._shell.opens_network(command, hint)
            or self._shell.asks(command, hint, outside)
        )

    def run(
        self, args: Arguments, *, operation_id: str, cancel: threading.Event | None
    ) -> ToolResult | ShellFailure | InvalidArgument | CallCancelled:
        result = self._shell.run(args, operation_id=operation_id, cancel=cancel)
        if self._suspend is not None and self._suspend.is_set():
            # Whatever came back: a refused ask (`CallCancelled`), or a command reaped mid-run,
            # whose result says only that it was cut off.
            raise RunSuspended(operation_id)
        return result

    def _store(self, data: bytes) -> Handle:
        self._journal.put_blob(data)
        return stored_handle("output", self._journal.run_id, data)

    def _spawn(
        self, request: ExecRequest, operation_id: str, cancel: threading.Event | None
    ) -> ExecResult:
        # Approved, the command runs outside the sandbox that kept `.git`
        # read-only, so git's own files are held across it and put back if it changed them:
        # a hook or a config line it planted would run in the harness's next git call.
        guarded = held(self._workspace) if request.unsandboxed else None
        result = self._executor.run(request, operation_id=operation_id, cancel=cancel)
        if guarded is not None:
            tampered = put_back(guarded, held(self._workspace))
            if tampered:
                note = (
                    f"[effect check: changed {', '.join(tampered)}, which git runs outside any "
                    "sandbox; each was put back as it was]"
                )
                result = replace(result, stderr=f"{result.stderr}\n{note}".lstrip("\n"))
        self._journal.append(
            EventKind.EXEC,
            ExecP.model_validate(Executor.exec_event(request, result, operation_id)),
            task_id=self._task(),
        )
        return result
