"""Harness-owned argv through ProcessSupervisor.

`ProcessSupervisor.run` is spawn. This helper is the shared *policy* around it for
work the harness authored: scrubbed env, optional extra keys (snapshot
`GIT_INDEX_FILE`), and a `Completed` the caller interprets.

It does not wrap Isolation and does not call `Executor.run`. It does not raise on
timeout — a hung `git log` is `UNKNOWN` for reconcile, not `InfraError`, so callers
read `timed_out`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

from chickadee.errors import InfraError
from chickadee.process.supervisor import Completed, ProcessSupervisor, scrub_env

GIT_TIMEOUT_S = 30
"""Git probes that hold a lease or a tree baseline. Same band as handoff / applier /
trailer reconcile."""

VERSION_TIMEOUT_S = 10
"""`binary --version` at init. A hang is "present (version unknown)", not a failed init."""


def run_harness(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    timeout_s: int,
    extra_env: Mapping[str, str] | None = None,
    supervisor: ProcessSupervisor | None = None,
    stdin: bytes | None = None,
) -> Completed:
    """One argv through the supervisor, `stdin` its input when given. `extra_env` overlays
    `scrub_env()`, it does not replace it — snapshot must keep `GIT_INDEX_FILE` without
    restoring ambient secrets.

    `OSError` (including `FileNotFoundError` from a missing binary) becomes `InfraError`
    here so workspace/detect cannot mistake a missing `git` for "not a repository" or
    leak a raw spawn failure. The default supervisor already maps this; the catch is for
    injected supervisors and any `OSError` that escapes `run`.
    """
    env = scrub_env()
    if extra_env:
        env = {**env, **dict(extra_env)}
    sup = supervisor or ProcessSupervisor()
    try:
        return sup.run(argv, cwd=cwd, env=env, timeout_s=timeout_s, stdin=stdin)
    except OSError as exc:
        raise InfraError(f"could not run {argv[0]!r}: {exc}") from exc


_REPOSITORY_PROGRAMS_OFF: Final = (
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
)
"""`run_git`'s prefix: `fsmonitor` always off, then the hooks, unless the
call is the person's (`hooks=True`), which takes only the first two."""

DIFF_FLAGS: Final = ("--no-color", "--no-ext-diff")
"""Every `git diff` the harness reads: plain text, and never the repository's
`diff.external` program, which a command could have planted in its config."""


def run_git(
    cwd: Path,
    *args: str,
    extra_env: Mapping[str, str] | None = None,
    supervisor: ProcessSupervisor | None = None,
    timeout_s: int = GIT_TIMEOUT_S,
    stdin: bytes | None = None,
    hooks: bool = False,
) -> Completed:
    """Git with `cwd` as the repository. No `-C`.

    Git resolves `-C` relative to the child's cwd. Passing the same (possibly relative)
    path as both `cwd` and `-C` joins it onto itself — `Path("..")` would run in the
    parent of the intended repo. The graph indexer already uses cwd-only; this is that
    pattern for the rest of core.

    The repository's own programs never run in the harness's git: no
    `core.fsmonitor`, and no hooks. A command could have planted either, and this call may
    come before a guard has put it back. `hooks=True` is for a call made on the person's
    behalf onto their branch (an undo's commit there, the handoff's merges), which runs their
    hooks as their own `git` would.
    """
    return run_harness(
        ("git", *_REPOSITORY_PROGRAMS_OFF[: 2 if hooks else 4], *args),
        cwd=cwd,
        timeout_s=timeout_s,
        extra_env=extra_env,
        supervisor=supervisor,
        stdin=stdin,
    )


def git(
    cwd: Path,
    *args: str,
    supervisor: ProcessSupervisor | None = None,
    extra_env: Mapping[str, str] | None = None,
    on_error: Callable[[str], Exception] | None = None,
    hooks: bool = False,
) -> str:
    """`run_git`, then the two things every caller did afterwards: a timeout is an
    `InfraError` naming the subcommand; a non-zero exit raises `on_error(detail)` — or an
    `InfraError` when the caller has no more specific type — where `detail` is
    `git <args> failed in <cwd>: <stderr | stdout | exit N>`. Returns stripped stdout.

    Three modules had grown their own copy of this (workspace, handoff, init.detect),
    differing only in the exception they raised; the difference is now the parameter.
    Callers that want the exit code *as the answer* keep using `run_git`.
    """
    done = run_git(cwd, *args, extra_env=extra_env, supervisor=supervisor, hooks=hooks)
    if done.timed_out:
        raise InfraError(f"git {args[0]} timed out after {GIT_TIMEOUT_S}s in {cwd}")
    text = done.stdout.decode("utf-8", "replace").strip()
    if done.exit_code != 0:
        why = done.stderr.decode("utf-8", "replace").strip() or text or f"exit {done.exit_code}"
        detail = f"git {' '.join(args)} failed in {cwd}: {why}"
        raise on_error(detail) if on_error else InfraError(detail)
    return text
