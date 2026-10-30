"""The episode's first operation: a worktree's environment, prepared once.

A worktree is a checkout of tracked files and nothing else: no `.venv/`, no `node_modules/`.
The profile's `[toolchain].bootstrap` runs once at episode start, journaled as an effect
and run through Isolation and the ProcessSupervisor. A bootstrap that fails means
the episode does not start; it is never a packet's failure.

Moved out of the facade unchanged: `engine.py` re-exports `bootstrap_worktree` and
`BootstrapReport`, where hosts import them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chickadee import shell_line
from chickadee.errors import InfraError
from chickadee.ids import bootstrap_operation
from chickadee.isolation.base import SandboxBackend, level_of
from chickadee.isolation.base import select as _select_backend
from chickadee.isolation.none import Passthrough
from chickadee.isolation.types import Isolation, SandboxLevel
from chickadee.journal.effects import EffectKind, Intent, on_journal, perform
from chickadee.journal.writer import Journal
from chickadee.process.supervisor import ProcessSupervisor, scrub_env
from chickadee.profiles.model import LanguageProfile
from chickadee.run_header import IN_PLACE_NO_BOOTSTRAP, NO_BOOTSTRAP
from chickadee.schema import sha256_hex
from chickadee.workspace import Workspace

__all__ = [
    "BOOTSTRAP_TIMEOUT_S",
    "BootstrapReport",
    "bootstrap_env",
    "bootstrap_worktree",
]


BOOTSTRAP_TIMEOUT_S = 600
"""A lockfile install with a cold cache is minutes, not seconds; the first session pays
it once. Past this the worktree could not be prepared, which is the same InfraError a
non-zero exit is."""


@dataclass(frozen=True)
class BootstrapReport:
    """What episode start did about the worktree's environment.

    `header` is the one line the episode header carries, so a reader of the run sees
    whether a bootstrap ran without deriving it from the operation list."""

    ran: bool
    header: str
    operation_id: str | None = None
    argv: tuple[str, ...] = ()
    output_sha: str | None = None
    """The journal blob holding the command's output — the handle a failure names."""
    sandbox: SandboxLevel | None = None


def bootstrap_env() -> dict[str, str]:
    """The environment the bootstrap runs under: `scrub_env()`'s keep-list, plus whichever of
    `NETWORK_ENV` the operator's shell has (proxies, CA bundles, private indexes), each still
    judged by the credential scrub. The bootstrap is the one harness spawn that needs the
    network, and behind a proxy or a private index `uv sync --frozen` fails without them."""
    return scrub_env(network=True)


def bootstrap_worktree(
    workspace: Workspace,
    profile: LanguageProfile,
    journal: Journal,
    *,
    supervisor: ProcessSupervisor | None = None,
    backend: SandboxBackend | None = None,
    require_sandbox: bool = False,
    deny_read: tuple[Path, ...] = (),
    timeout_s: int = BOOTSTRAP_TIMEOUT_S,
) -> BootstrapReport:
    """Run the profile's `[toolchain].bootstrap` once, as the episode's first operation.

    A worktree is a checkout of tracked files: no `.venv/`, no `node_modules/`. Without
    this the first checkpoint either fails loudly on a missing module or — the case D109
    exists for — imports the package from the *source checkout* through an editable
    install and reports on the unpatched tree. So the episode does not start on a worktree
    that could not be prepared: a non-zero exit or a timeout is an `InfraError` naming the
    command and the output handle, never a packet's failure.

    The run is journaled like any effect: `effect_intended` → `effect_started` →
    `effect_completed` under one `operation_id`, so a resume finds it and a reader of the
    run sees what was installed. Through Isolation + ProcessSupervisor with the
    worktree writable and the network **on** — its class is `lockfile_install` with an
    `index_host` destination (T1, lowerable), and the classification and approval
    of that class land with the executor's command table; here the network is simply not
    shut, which is what the class allows. The deny-list mask still applies.

    `in_place` runs nothing: the checkout has its own environment. A profile without a
    bootstrap runs nothing either, and the header says so rather than leaving the
    absence to be inferred.
    """
    if not workspace.isolated:
        return BootstrapReport(ran=False, header=IN_PLACE_NO_BOOTSTRAP)
    template = profile.toolchain.bootstrap
    if template is None:
        return BootstrapReport(ran=False, header=NO_BOOTSTRAP)

    argv = template.argv
    root = workspace.root
    operation_id = bootstrap_operation(journal.run_id)
    shown = shell_line.join(argv)
    # Probed before the intent: the probe writes nothing, so everything after the start is
    # inside the effect, which `perform` closes however it ends.
    chosen = backend or _select_backend()
    level = level_of(chosen)
    intent = Intent(
        operation_id,
        EffectKind.BOOTSTRAP,
        # A shipped profile's bootstrap needs no acceptance; what stands in for the
        # approval is the command itself, so a resume can tell this run's bootstrap from one
        # a later profile edit would have run.
        approved_sha=sha256_hex(shown.encode("utf-8")),
        preconditions=(f"worktree {root} exists",),
        idempotent=True,
        surface=(str(root),),
    )
    # An exception may come after the command ran and changed the worktree, so it is closed
    # `ambiguous` (the default). A kill writes nothing, because it cannot.
    with perform(on_journal(journal), intent) as fx:
        if require_sandbox and level is SandboxLevel.NONE:
            fx.complete("failed", observed=(str(root),))
            raise InfraError(
                f"bootstrap {shown!r} needs a sandbox and none is available; refusing to "
                "run it unwrapped rather than preparing the worktree with more reach than "
                "the run was promised"
            )
        wrap_with = chosen if chosen.available() else Passthrough()
        wrapped = wrap_with.wrap(
            Isolation(
                argv=argv, cwd=root, network_off=False, repo_writable=True, deny_read=deny_read
            )
        )
        completed = (supervisor or ProcessSupervisor()).run(
            wrapped, cwd=root, env=bootstrap_env(), timeout_s=timeout_s
        )
        output = completed.stdout
        if completed.stderr:
            output += b"\n--- stderr ---\n" + completed.stderr
        output_sha = journal.put_blob(output)
        ok = completed.exit_code == 0 and not completed.timed_out
        fx.complete("succeeded" if ok else "failed", observed=(str(root),), result=output_sha)
    if not ok:
        what = (
            f"timed out after {timeout_s}s"
            if completed.timed_out
            else f"exited {completed.exit_code}"
        )
        raise InfraError(
            f"bootstrap {shown!r} {what} in {root}; the episode does not start on a "
            f"worktree that could not be prepared (D109). Output handle: {output_sha}"
        )
    return BootstrapReport(
        ran=True,
        header=f"bootstrap: {shown} (operation {operation_id})",
        operation_id=operation_id,
        argv=argv,
        output_sha=output_sha,
        sandbox=level,
    )
