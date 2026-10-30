"""A host's configuration for one repository, read once, the one way.

Every host that builds a `PacketEngine` for a repository needs the same three answers: where
its state lives (`roots`), what its layers resolve to (`resolve(load_layers(...))`), and
whether this clone accepted the repository's executable subset (`Resolved.trust`, with the
project profile's overrides in the digest). `chickadee serve` is the first host; the REPL
and the wire's in-process host are the next. Each composing those three
itself is how one of them comes to digest a different subset than the acceptance surface did,
and an accepted config then reads as unaccepted (one concept, one home).

It reads and decides, and builds no engine. `executable` is what `chickadee trust` shows and
accepts: the subset `trust` digests, through the same mapping. `repository_trust` is the whole
of what that command shows, the subset and the repository's grants, and `accept` its one write.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from chickadee.approval.grants import RepoGrants, accept_for_clone, repo_grants
from chickadee.config import EngineConfig
from chickadee.errors import ConfigError
from chickadee.profiles.loader import executable_overrides
from chickadee.state.layers import LayerReadError, load_layers
from chickadee.state.merge import MergeError
from chickadee.state.resolve import Resolved, TrustDecision, resolve
from chickadee.state.roots import Roots, roots
from chickadee.state.trust import Executable, executable_subset

__all__ = [
    "HostConfig",
    "RepositoryTrust",
    "accept",
    "executable",
    "load_host_config",
    "repository_trust",
]


@dataclasses.dataclass(frozen=True)
class HostConfig:
    """What a host hands `PacketEngine(repo, engine, trust=trust)`."""

    repo: Path
    """The working tree's root: the directory that holds `.chickadee/`."""
    roots: Roots
    resolved: Resolved
    trust: TrustDecision
    """Taken for `resolved.engine`, so the engine accepts it."""

    @property
    def engine(self) -> EngineConfig:
        return self.resolved.engine


def load_host_config(cwd: Path, *, environ: Mapping[str, str] | None = None) -> HostConfig:
    """The repository containing `cwd`, its layers resolved and its trust decided.

    Raises what its parts raise, unchanged: `NotARepository` outside a git repository,
    `LayerReadError` for a file that cannot be read, `MergeError` (an unknown or misplaced key)
    and `ConfigError` for a merge that does not validate or a `trusted.toml` that cannot be
    read. A host says each in its own words; none is a reason to guess."""
    scopes = roots(cwd)
    repo = scopes.project.parent
    resolved = resolve(load_layers(scopes, environ=environ))
    trust = resolved.trust(scopes.repo_local, overrides=executable_overrides(repo), root=repo)
    return HostConfig(repo=repo, roots=scopes, resolved=resolved, trust=trust)


def executable(host: HostConfig) -> Executable:
    """The repository's executable subset and its digest, exactly as `host.trust` was taken
    over it (`Resolved.declared`, with the project profile's overrides): what `chickadee
    trust` shows itemised and records on the person's yes."""
    return executable_subset(host.resolved.declared(executable_overrides(host.repo)), host.repo)


@dataclasses.dataclass(frozen=True)
class RepositoryTrust:
    """What `chickadee trust` shows: the repository's executable
    config subset and its `[[approval.grants]]`, each with its digest and whether this clone
    accepted it. Both are recorded in the one `trusted.toml` on the same yes."""

    roots: Roots
    executable: Executable
    executable_accepted: bool
    grants: RepoGrants

    @property
    def pending(self) -> tuple[str, ...]:
        """The digests a yes would record: each non-empty part not yet accepted. Empty when
        there is nothing to ask about."""
        found: list[str] = []
        if not self.executable.is_empty and not self.executable_accepted:
            found.append(self.executable.digest)
        if not self.grants.trusted:
            found.append(self.grants.digest)
        return tuple(found)


def repository_trust(cwd: Path, *, environ: Mapping[str, str] | None = None) -> RepositoryTrust:
    """The repository containing `cwd`, read as `load_host_config` reads it, so the digest
    shown is the digest the engine checks. `NotARepository` outside one; anything else that
    stops the read is a `ValueError` naming it (`ConfigError`, or `PreferenceError` for an
    approval file that does not parse), so a surface has one refusal to say."""
    try:
        host = load_host_config(cwd, environ=environ)
    except (LayerReadError, MergeError) as exc:
        raise ConfigError(str(exc)) from exc
    return RepositoryTrust(
        roots=host.roots,
        executable=executable(host),
        executable_accepted=host.trust.accepted,
        grants=repo_grants(host.roots),
    )


def accept(shown: RepositoryTrust) -> Path | None:
    """Record the digests `shown` showed, on the person's yes; `None` when there was nothing
    to accept. The digests are those of what was shown, never re-read: a file changed since
    has another digest, which stays unaccepted until it is shown in turn."""
    if not shown.pending:
        return None
    return accept_for_clone(shown.roots, shown.pending)
