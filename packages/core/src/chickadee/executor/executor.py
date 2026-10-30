"""Policy front for proposed commands. Spawn lives in `chickadee.process`.

The Executor does **not decide what to run** — callers do. It decides whether the caller
may run it, under what isolation, and whether the result is admissible. Harness-owned
argv (gates, checkpoints, MCP stdio, graph LSP) never comes through `run`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from chickadee.approval.commands import CommandTable, shipped
from chickadee.durable import BlobPut
from chickadee.errors import InfraError
from chickadee.executor import effects as effect_mod
from chickadee.executor.policy import command_argv, isolation_for, prefilter, rules_for
from chickadee.executor.types import (
    ExecRequest,
    ExecResult,
    NetworkEnforced,
    PolicyClass,
    SandboxLevel,
    Violation,
)
from chickadee.isolation.base import SandboxBackend, level_of, select
from chickadee.isolation.types import session_tmp
from chickadee.process.supervisor import (
    Job,
    JobExit,
    ProcessSupervisor,
    bound_session,
    person_env,
    scrub_env,
)
from chickadee.schema import sha256_hex
from chickadee.secrets.denylist import Denylist


@dataclass(frozen=True)
class _Admitted:
    """A request that passed every check, and how it will run: what `run` and `start` share."""

    level: SandboxLevel
    network_enforced: NetworkEnforced
    argv: tuple[str, ...]
    env: Mapping[str, str]
    before: dict[str, effect_mod.Fingerprint] | None
    """The effect bracket's before-scan, for a class that enforces its effects."""


@dataclass(frozen=True)
class Started:
    """A command `start` spawned: its job, and what `finish` needs to answer for it as `run`
    would have."""

    job: Job
    request: ExecRequest
    admitted: _Admitted

    @property
    def sandbox(self) -> SandboxLevel:
        return self.admitted.level

    @property
    def network_enforced(self) -> NetworkEnforced:
        return self.admitted.network_enforced


class Executor:
    GRACE_S = ProcessSupervisor.GRACE_S

    def __init__(
        self,
        backend: SandboxBackend | None = None,
        require_sandbox: bool = False,
        put_blob: BlobPut | None = None,
        supervisor: ProcessSupervisor | None = None,
        *,
        denylist: Denylist | None = None,
        deny_read: tuple[Path, ...] = (),
        commands: CommandTable | None = None,
        tool_cache: Path | None = None,
        withheld_env: frozenset[str] = frozenset(),
    ) -> None:
        """`denylist` is the list the run resolved: `None` is the built-in list.
        `deny_read` is that list's `resolve()` over the workspace and `$HOME`, done once by
        the caller, as `HarnessSpawn.deny_read` is. Both reach every spawn.

        `commands` is the engine's `[[commands]]` table (`EngineConfig.command_table`),
        built once and given to each executor beside the deny list, so every reader of what a
        command is (this pre-filter, `Shell`, `Escapes`) reads one table. `None` is the
        shipped table, for a caller with no config.

        `tool_cache` is the repository's tool cache, writable in
        every wrap and named to the package managers; `withheld_env` the harness's own model
        endpoints' keys (`EngineConfig.endpoint_keys`), never in an approved command's
        environment.
        """
        self.backend = backend or select()
        self.require_sandbox = require_sandbox
        self.denylist = denylist if denylist is not None else Denylist()
        self.deny_read = deny_read
        self.commands = commands if commands is not None else shipped()
        self.tool_cache = tool_cache
        self.withheld_env = withheld_env
        self._put_blob = put_blob
        self._supervisor = supervisor or ProcessSupervisor()

    # -- the one entry point ------------------------------------------------

    def run(
        self,
        request: ExecRequest,
        *,
        operation_id: str,
        cancel: threading.Event | None = None,
    ) -> ExecResult:
        """Spawn under policy, measure what it touched, and report honestly.

        `cancel` is the session's token, handed through to the supervisor, which
        looks at it between waits and reaps the group when it is set. `None` for callers
        with no human — the worker's shell, harness argv — which keeps every existing call
        site unchanged. The `exec` event is the same one either way; replay derives the
        cancellation from the `interrupt` event beside it.
        """
        admitted = self._admit(request)
        if isinstance(admitted, ExecResult):
            return admitted
        started = time.perf_counter()
        completed = self._supervisor.run(
            admitted.argv,
            cwd=request.cwd,
            env=admitted.env,
            timeout_s=request.timeout_s,
            cancel=cancel,
        )
        wall_ms = int((time.perf_counter() - started) * 1000)
        return self._result(
            request,
            admitted,
            completed.stdout,
            completed.stderr,
            exit_code=completed.exit_code,
            wall_ms=wall_ms,
            timed_out=completed.timed_out,
            cancelled=completed.cancelled,
        )

    def start(
        self,
        request: ExecRequest,
        *,
        out: Path,
        err: Path,
        max_runtime_s: int | None,
        max_log_bytes: int,
        on_exit: Callable[[Job, JobExit], None] | None = None,
    ) -> Started | ExecResult:
        """`run`'s non-blocking twin: the same checks, through `_admit`, then
        a job whose output goes to `out` and `err`. A refusal is the `ExecResult` `run` would
        have returned. The caller waits on the job, and answers a command that ended inside
        its window with `finish`."""
        admitted = self._admit(request)
        if isinstance(admitted, ExecResult):
            return admitted
        job = self._supervisor.start(
            admitted.argv,
            cwd=request.cwd,
            env=admitted.env,
            out=out,
            err=err,
            max_runtime_s=max_runtime_s,
            max_log_bytes=max_log_bytes,
            on_exit=on_exit,
        )
        return Started(job, request, admitted)

    def finish(
        self,
        started: Started,
        exit_: JobExit,
        *,
        timed_out: bool = False,
        cancelled: bool = False,
    ) -> ExecResult:
        """The `ExecResult` `run` builds, field for field, for a command `start` spawned that
        ended inside its window: its two files as the blobs, the effect bracket closed against
        the before-scan. `timed_out` and `cancelled` say why the caller ended it, as
        `ProcessSupervisor.run` says why it did; a timeout's code is `124` when the child's own
        is 0, as there."""
        code = exit_.exit_code
        return self._result(
            started.request,
            started.admitted,
            started.job.out.read_bytes(),
            started.job.err.read_bytes(),
            exit_code=(code or 124) if timed_out else code,
            wall_ms=exit_.wall_ms,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    def _admit(self, request: ExecRequest) -> _Admitted | ExecResult:
        """Every check in front of a spawn, once, for `run` and `start` alike: the sandbox
        requirement (raised), the network's answer, the secret pre-filter (a refusal returned
        as the `ExecResult` the caller answers with), the effect bracket's before-scan, and
        the wrap and environment the command runs with."""
        # Approved, a command runs outside the sandbox, at level `none` with the
        # network on, and `require_sandbox` does not refuse it: EX-2 is about what runs unasked.
        outside = request.unsandboxed
        level = SandboxLevel.NONE if outside else level_of(self.backend)
        if not outside and self.require_sandbox and level is SandboxLevel.NONE:
            # Eval configs abort rather than degrade. A number produced without the
            # isolation it claims is worse than no number.
            raise InfraError(
                "require_sandbox is set and no sandbox backend is available; refusing to "
                "run unwrapped rather than reporting a result that overstates its isolation"
            )

        network_enforced = "on" if outside else self._network_enforced(request, level)

        # The secret pre-filter refuses a command naming a deny-listed path, outside too.
        denial = prefilter(request, self.denylist, self.commands)
        if denial is not None:
            return ExecResult(
                exit_code=126,
                stdout_sha=None,
                stderr_sha=None,
                wall_ms=0,
                timed_out=False,
                sandbox=level,
                violations=(Violation(str(request.cwd), denial),),
                stderr=denial,
                network_enforced=network_enforced,
            )

        # Fix 3: a class that does not enforce its effects discards the scan's
        # result, so it takes none: a test run in a large tree is not walked twice.
        enforced = rules_for(request.policy).effects_enforced
        before = effect_mod.snapshot_paths(request.cwd) if enforced else None
        if outside:
            argv = command_argv(request)
            env = person_env(withheld=self.withheld_env, workspace=request.cwd)
        else:
            tmp = self._session_tmp()
            isolation = isolation_for(
                request,
                deny_read=self.deny_read,
                denylist=self.denylist,
                scratch=() if self.tool_cache is None else (self.tool_cache,),
                tmp=tmp,
            )
            argv = self.backend.wrap(isolation)
            env = self._sandboxed_env(request, tmp)
        return _Admitted(level, network_enforced, tuple(argv), env, before)

    def _result(
        self,
        request: ExecRequest,
        admitted: _Admitted,
        out: bytes,
        err: bytes,
        *,
        exit_code: int,
        wall_ms: int,
        timed_out: bool,
        cancelled: bool,
    ) -> ExecResult:
        stdout = out.decode("utf-8", "replace")
        stderr = err.decode("utf-8", "replace")
        before = admitted.before
        # Advisory for VERIFICATION and HUMAN: a test run touching `__pycache__` is not a
        # policy breach, and failing on it would produce constant false failures.
        violations = (
            ()
            if before is None
            else effect_mod.diff_against(
                before, effect_mod.snapshot_paths(request.cwd), request.effects
            )
        )
        return ExecResult(
            exit_code=exit_code,
            stdout_sha=self._blob(stdout),
            stderr_sha=self._blob(stderr),
            wall_ms=wall_ms,
            timed_out=timed_out,
            sandbox=admitted.level,
            violations=violations,
            stdout=stdout,
            stderr=stderr,
            network_enforced=admitted.network_enforced,
            cancelled=cancelled,
        )

    def _session_tmp(self) -> Path | None:
        """The session's temporary directory, when this context runs one."""
        session = bound_session()
        return session_tmp(session) if session else None

    def _sandboxed_env(self, request: ExecRequest, tmp: Path | None) -> dict[str, str]:
        """A sandboxed command's environment: the keep-list, the network's variables when its
        network is open, and the tool cache and temporary directory named to
        what reads them (items 10, 11), set last so a person's own cache variable, read-only
        inside the wrap, does not win."""
        env = scrub_env(
            human=request.policy is PolicyClass.HUMAN,
            workspace=request.cwd,
            network=request.network.open,
        )
        if self.tool_cache is not None:
            env.update(cache_env(self.tool_cache))
        if tmp is not None:
            env["TMPDIR"] = str(tmp)
        return env

    def _network_enforced(self, request: ExecRequest, level: SandboxLevel) -> NetworkEnforced:
        """How the network will actually be: open where the call asked for it, which
        reached here only approved, and wherever nothing closes it (`sandbox=none`), so an
        undeclared run there is never recorded as sealed. Nothing is refused after approval:
        every backend is all-or-nothing, and the request names no host to enforce."""
        if request.network.open or level is SandboxLevel.NONE:
            return "on"
        return "off"

    @staticmethod
    def exec_event(
        request: ExecRequest, result: ExecResult, operation_id: str
    ) -> dict[str, object]:
        """Everything `ExecP` requires, from one run.

        Built here rather than at the call site so the driver cannot accidentally journal a
        run with a field it invented. `argv_sha` rather than the argv itself: a command line
        can carry a token, and the journal is durable.
        """
        argv = request.shell_line or " ".join(request.argv)
        return {
            "operation_id": operation_id,
            "policy": request.policy.value,
            "label": request.label,
            "argv_sha": sha256_hex(argv.encode()),
            "cwd": str(request.cwd),
            "exit_code": result.exit_code,
            "wall_ms": result.wall_ms,
            "timed_out": result.timed_out,
            "sandbox": result.sandbox.value,
            "network": not request.network.off,
            "stdout_sha": result.stdout_sha or "",
            "stderr_sha": result.stderr_sha or "",
            "violations": tuple(v.path for v in result.violations),
        }

    def _blob(self, text: str) -> str | None:
        if not text:
            return None
        data = text.encode("utf-8", "replace")
        return self._put_blob(data) if self._put_blob else sha256_hex(data)


def cache_env(cache: Path) -> dict[str, str]:
    """Where each package manager keeps its downloads, inside the repository's tool cache:
    pip, uv, npm, yarn, pnpm, Go, and everything that reads XDG."""
    return {
        "XDG_CACHE_HOME": str(cache),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "npm_config_cache": str(cache / "npm"),
        "NPM_CONFIG_CACHE": str(cache / "npm"),
        "YARN_CACHE_FOLDER": str(cache / "yarn"),
        "PNPM_STORE_DIR": str(cache / "pnpm-store"),
        "GOMODCACHE": str(cache / "go-mod"),
        "GOCACHE": str(cache / "go-build"),
    }
