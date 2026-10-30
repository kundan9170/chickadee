"""A script hook's run: the harness running a program the person
configured, so its own spawn (`box/spawn.run_argv`: Isolation and the supervisor), never
`Executor`, which judges commands a model proposed.

- **Sandboxed, reading only.** The repository read-only, the network off, the deny list masked,
  the supervisor's scrubbed environment and three variables beside it. Where nothing seals,
  `bind` refuses the entry before it is ever run.
- **An argv, never a shell string.** `~` and `$CHICKADEE_PROJECT_DIR` are resolved by the
  harness, in any element.
- **Bounded.** Its `timeout_s`, stdin written within it; on timeout the supervisor reaps the
  process group (SIGTERM, then SIGKILL). stdout and stderr are each kept to `OUTPUT_MAX`.
- **Checked again before each run.** The digest the set was bound with (its entry and every
  repository file its argv names, `state/trust.script_digest`) is read again first, so a
  script changed since it was accepted does not run (Codex re-verifies inside the deadline).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Final, Protocol

from chickadee.executor.types import ExecResult
from chickadee.hooks.bind import ScriptRunner
from chickadee.hooks.fire import HookError, HookFn
from chickadee.hooks.payloads import HookEvent
from chickadee.hooks.protocol import from_output, to_input
from chickadee.hooks.spec import HookSpec
from chickadee.hooks.verdicts import HookVerdict
from chickadee.isolation.base import SandboxBackend, level_of
from chickadee.isolation.types import SandboxLevel
from chickadee.state.trust import script_argv, script_digest

__all__ = ["OUTPUT_MAX", "ScriptSpawn", "argv_of", "script_runner", "sealed_runner", "seals"]

OUTPUT_MAX: Final = 1024 * 1024
"""Bytes of stdout and of stderr a script's answer is read from (Cline's cap); past it, cut."""


def argv_of(spec: HookSpec, root: Path) -> tuple[str, ...]:
    """`spec`'s argv as it runs: `~` and the project directory resolved, nothing else."""
    if spec.command is None:
        raise HookError(f"hook {spec.name!r} is a built-in, not a script")
    return script_argv(spec.command, root)


class ScriptSpawn(Protocol):
    """How the harness runs a process of its own (`box/spawn.run_argv` over the run's spawn,
    bound by whoever composes the run), so the framework never imports the box."""

    def __call__(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        *,
        timeout_s: int,
        stdin: bytes | None = None,
        repo_writable: bool = True,
        extra_env: Mapping[str, str] | None = None,
    ) -> ExecResult: ...


def seals(backend: SandboxBackend) -> bool:
    """Whether `backend` seals a process, as a script needs: the one answer `bind`'s
    callers and `chickadee hooks` ask."""
    return level_of(backend) is not SandboxLevel.NONE


def sealed_runner(
    backend: SandboxBackend, spawn: Callable[[], ScriptSpawn], root: Path
) -> ScriptRunner | None:
    """`script_runner` where `backend` seals a process, else None: then `bind` refuses a
    script entry. Every caller that binds a set asks this, never the backend. `spawn`
    is made when the first script is bound, so a set with none never resolves the deny list
    a spawn masks."""
    if not seals(backend):
        return None
    made = cache(spawn)
    return lambda spec: script_runner(made(), root)(spec)


def script_runner(spawn: ScriptSpawn, root: Path) -> ScriptRunner:
    """`bind`'s way to run a script entry over the worktree at `root`."""

    def runner(spec: HookSpec) -> tuple[HookFn, str]:
        argv = argv_of(spec, root)
        digest = script_digest(spec.model_dump_json(by_alias=True), argv, root)

        def run(event: HookEvent) -> HookVerdict:
            if script_digest(spec.model_dump_json(by_alias=True), argv, root) != digest:
                raise HookError(
                    "its script changed since the set was bound; it is not run until it is "
                    "accepted again (`chickadee trust`)"
                )
            done = spawn(
                argv,
                root,
                timeout_s=spec.timeout_s,
                stdin=to_input(event, root),
                repo_writable=False,
                extra_env={
                    "CHICKADEE_PROJECT_DIR": str(root),
                    "CHICKADEE_HOOK_EVENT": event.point.value,
                    # Gemini CLI sets it for the same reason: a Claude Code script ports.
                    "CLAUDE_PROJECT_DIR": str(root),
                },
            )
            if done.timed_out:
                raise HookError(f"timed out after {spec.timeout_s}s", timed_out=True)
            return from_output(
                event.point, done.exit_code, _capped(done.stdout), _capped(done.stderr)
            )

        return run, digest

    return runner


def _capped(text: str | None) -> str:
    if not text:
        return ""
    data = text.encode("utf-8", "replace")
    if len(data) <= OUTPUT_MAX:
        return text
    return data[:OUTPUT_MAX].decode("utf-8", "ignore") + "\n[cut: past 1 MiB]"
