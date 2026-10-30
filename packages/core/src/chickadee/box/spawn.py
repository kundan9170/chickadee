"""The harness's own spawn: a process the harness runs for itself, never one
a model proposed, so it goes through Isolation and the supervisor and never through
`Executor`, which judges proposed commands. Its callers are the syntax check
(`box/syntax.SyntaxChecker`), which runs a language's check command over a written file, and
the index's language servers (`graph/lsp.CheckerResolver`), wrapped by `harness_wrap`.

Moved whole from `verifier/infra.py` when the verifier retired, `GateSpawn`
renamed `HarnessSpawn` since no gate runs it now. A required tool that is missing raises
(`HarnessInfraError`): a missing interpreter is infrastructure, never a failing check.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from chickadee.durable import BlobPut
from chickadee.errors import InfraError
from chickadee.executor.types import ExecResult
from chickadee.graph.lsp import CheckerResolver
from chickadee.graph.servers import EpisodeServers
from chickadee.isolation.base import SandboxBackend, level_of
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.process.supervisor import ProcessSupervisor, scrub_env
from chickadee.profiles.loader import ProfileSet
from chickadee.schema import sha256_hex

__all__ = [
    "HarnessInfraError",
    "HarnessSpawn",
    "harness_isolation",
    "harness_wrap",
    "index_resolver",
    "require",
    "run_argv",
    "which",
]


class HarnessInfraError(InfraError):
    """A required tool is missing, or a sealed spawn was asked for where nothing seals."""


@dataclass(frozen=True)
class HarnessSpawn:
    """Supervisor + wrap backend: what the harness spawns its own processes through."""

    backend: SandboxBackend
    supervisor: ProcessSupervisor
    put_blob: BlobPut | None = None
    require_sandbox: bool = False
    """Checked here, not on Executor. Default False so a direct HarnessSpawn in
    a unit test can inject Passthrough; the engine passes its config's (True by default)."""
    deny_read: tuple[Path, ...] = ()
    """The resolved deny-list paths every gate and checkpoint runs masked from.
    Resolved once by the caller (`secrets/denylist.resolve`); `()` keeps every existing
    construction unchanged."""


def which(name: str, path: str | None = None) -> Path | None:
    """Resolve an executable. `None` means the optional-gate skip, not a fail."""
    import shutil

    found = shutil.which(name, path=path)
    return Path(found) if found else None


def require(name: str) -> Path:
    found = which(name)
    if found is None:
        raise HarnessInfraError(
            f"required tool {name!r} is not on PATH; this is infrastructure, not a gate fail"
        )
    return found


def harness_isolation(
    argv: tuple[str, ...],
    cwd: Path,
    deny_read: tuple[Path, ...] = (),
    *,
    repo_writable: bool = True,
) -> Isolation:
    """The harness's wrap: network off, repo writable unless the caller only reads it, and
    masked, so a check that reads `.env` errors rather than carrying the value into a
    digest a model reads. Not a PolicyClass."""
    return Isolation(
        argv=argv, cwd=cwd, network_off=True, repo_writable=repo_writable, deny_read=deny_read
    )


def harness_wrap(
    spawn: HarnessSpawn, argv: tuple[str, ...], cwd: Path, *, repo_writable: bool = True
) -> tuple[str, ...]:
    """`argv` wrapped as the harness runs its own processes: the backend's wrap, or none when
    no backend is available, which is `sandbox=none`, never a silent wrap that still claims
    NATIVE. Raises `HarnessInfraError` when `require_sandbox` is set and nothing seals: the
    one home of that rule, for a run and a session alike."""
    if spawn.require_sandbox and level_of(spawn.backend) is SandboxLevel.NONE:
        raise HarnessInfraError(
            "require_sandbox is set and no sandbox backend is available; refusing to "
            "run unwrapped rather than reporting a result that overstates its isolation"
        )
    backend = spawn.backend if spawn.backend.available() else Passthrough()
    isolation = harness_isolation(argv, cwd, spawn.deny_read, repo_writable=repo_writable)
    return backend.wrap(isolation)


def index_resolver(
    root: Path,
    spawn: HarnessSpawn,
    profiles: ProfileSet,
    servers: EpisodeServers | None = None,
) -> CheckerResolver:
    """The code graph's resolver for an index of `root`: each language's
    server from `profiles`, run as the harness runs its own processes, reading the repository
    and writing none of it. Where `spawn` may not run unwrapped and nothing seals, no server
    starts and the edges fall back. With an episode's `servers`, each is kept between passes;
    without, each pass starts its own."""
    return CheckerResolver(
        root,
        spawn.supervisor,
        profiles,
        wrap=partial(harness_wrap, spawn, repo_writable=False),
        servers=servers,
    )


def run_argv(
    spawn: HarnessSpawn,
    argv: tuple[str, ...],
    cwd: Path,
    *,
    timeout_s: int,
    label: str = "",
    operation_id: str = "",
    stdin: bytes | None = None,
    repo_writable: bool = True,
    extra_env: Mapping[str, str] | None = None,
) -> ExecResult:
    """The only way the harness asks for a process of its own. Isolation + supervisor, not
    Executor. `stdin` is its input; `repo_writable=False` for one that only reads (a hook's
    script); `extra_env` overlays the scrubbed environment, never replaces it."""
    del label, operation_id
    # Same cut as Executor.run: availability is not application (`harness_wrap`).
    level = level_of(spawn.backend)
    wrapped = harness_wrap(spawn, argv, cwd, repo_writable=repo_writable)
    started = time.perf_counter()
    completed = spawn.supervisor.run(
        wrapped,
        cwd=cwd,
        env={**scrub_env(), **(extra_env or {})},
        timeout_s=timeout_s,
        stdin=stdin,
    )
    wall_ms = int((time.perf_counter() - started) * 1000)
    stdout = completed.stdout.decode("utf-8", "replace")
    stderr = completed.stderr.decode("utf-8", "replace")
    return ExecResult(
        exit_code=completed.exit_code,
        stdout_sha=_blob(spawn, stdout),
        stderr_sha=_blob(spawn, stderr),
        wall_ms=wall_ms,
        timed_out=completed.timed_out,
        sandbox=level,
        stdout=stdout,
        stderr=stderr,
    )


def _blob(spawn: HarnessSpawn, text: str) -> str | None:
    if not text:
        return None
    data = text.encode("utf-8", "replace")
    return spawn.put_blob(data) if spawn.put_blob else sha256_hex(data)
