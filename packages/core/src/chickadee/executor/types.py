"""Executor value types.

Policy-front shapes. Spawn and wrap live in `chickadee.process` and `chickadee.isolation`.
`SandboxLevel` is re-exported so `ExecResult.sandbox` keeps its public name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from chickadee.isolation.types import SandboxLevel as SandboxLevel

NetworkEnforced = Literal["off", "on"]
"""D199, §5.2: how a run's network actually was, whatever the request declared. `off`: sealed,
no socket opens. `on`: the whole network was open for the command's duration, because the call
asked for it and was approved, or because nothing here could close it (`sandbox=none`)."""


class PolicyClass(StrEnum):
    DISCOVERY = "discovery"
    """Read-only, scout/planner. Read-only is self-securing, so no approval."""
    ACTION = "action"
    """Mutating, planned and approved."""
    WORKTREE = "worktree"
    """The box shell's: writes the task's worktree, held to no declaration.
    Named `verification` until the gates it ran for retired."""
    HUMAN = "human"
    """The user's own `!cmd`."""


@dataclass(frozen=True)
class NetworkPolicy:
    """Off unless the call asked for the network. Every backend is all-or-nothing, so
    there is nothing finer to say: no host is named, because none could be enforced."""

    open: bool = False

    @property
    def off(self) -> bool:
        return not self.open

    @staticmethod
    def off_policy() -> NetworkPolicy:
        return NetworkPolicy()


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str


@dataclass(frozen=True)
class ExecRequest:
    command: str | tuple[str, ...]
    """A tuple is argv and is preferred; a string is a shell line. EX-1 constrains by
    sandbox rather than by vocabulary, so both are permitted and the kernel is what
    actually enforces the boundary."""
    cwd: Path
    policy: PolicyClass
    timeout_s: int
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    effects: tuple[str, ...] = ()
    """Path globs this may modify. `()` means read-only is expected."""
    also_writable: tuple[Path, ...] = ()
    """Directories outside `cwd` the wrap binds writable when the workspace is writable: a
    linked worktree's common git directory beside a declared write (`Isolation.writable`), and
    the box shell's worktree root whatever its `cwd`."""
    read_only: tuple[Path, ...] = ()
    """Paths the wrap binds read-only over a writable workspace (`Isolation.read_only`): the
    box shell's worktree `.git`."""
    idempotent: bool = False
    label: str = ""
    unsandboxed: bool = False
    """The person approved this command, so it runs outside the sandbox:
    unwrapped, the network on, their environment. Set only by a shell after an approval, a
    grant or `never_ask`; never by a model, whose `sandbox: false` only makes the call ask."""

    @property
    def argv(self) -> tuple[str, ...]:
        return self.command if isinstance(self.command, tuple) else ()

    @property
    def shell_line(self) -> str | None:
        return self.command if isinstance(self.command, str) else None


def joined_output(stdout: str, stderr: str) -> str:
    """A command's two streams as one text, the way every result shows them: stdout, then
    `[stderr]` and stderr on a line of their own, or stdout alone when stderr is empty."""
    if not stderr:
        return stdout
    sep = "" if not stdout or stdout.endswith("\n") else "\n"
    return f"{stdout}{sep}[stderr]\n{stderr}"


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout_sha: str | None
    stderr_sha: str | None
    wall_ms: int
    timed_out: bool
    sandbox: SandboxLevel
    violations: tuple[Violation, ...] = ()
    stdout: str = ""
    stderr: str = ""
    """Kept alongside the blob refs so a caller can classify without a store round-trip.
    Classification needs the text to tell an import error from a failed assertion."""
    network_enforced: NetworkEnforced = "off"
    """D199, §5.2. `on` covers two cases: the call asked for the network and was approved, or
    `sandbox=none` could not close it for anything, so an undeclared run there is never said to
    be sealed. Derivable from the request's `network` and the `sandbox` for replay, so the
    `exec` event does not change; the default keeps every existing constructor unchanged."""
    cancelled: bool = False
    """The session's token stopped the run, and its process group was reaped. Distinct
    from `timed_out`: the user's stop, not the harness's limit. Not a field of the `exec`
    event (`ExecP` is frozen); replay derives it from the `interrupt` event and the next
    signal-ended `exec`."""

    @property
    def ok(self) -> bool:
        return (
            self.exit_code == 0
            and not self.violations
            and not self.timed_out
            and not self.cancelled
        )


@dataclass(frozen=True)
class ActionSpec:
    """A real shell command plus the metadata that makes it approvable.

    There is no capability catalogue — EX-1 constrains by sandbox, not vocabulary. The
    metadata exists so a human can decide, and so the effect surface can be checked after.
    """

    command: str | tuple[str, ...]
    effects: tuple[str, ...] = ()
    network: NetworkPolicy = field(default_factory=NetworkPolicy)
    idempotent: bool = False
    why: str = ""
    done_when: str = ""
