"""Each language's servers on this machine, and their install (`chickadee setup`).

A profile names its language server (`[lsp].server`) and its linter (`[lint].server`); where one
is missing, its `[lsp].install` is the one command that installs it into Chickadee's own
servers directory (`profiles.tools.servers_dir`), where `find_tool` looks for the harness's own
servers. The install is the person's: `setup` lists it, asks, and runs only what they said yes
to. It reaches the network, so it never runs inside a run's sandbox, and never on its own.

Nothing here decides for a repository: a server found in the repository's `.venv` or on `PATH`
is used first, and a profile's override can name another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chickadee.errors import InfraError
from chickadee.process import ProcessSupervisor, scrub_env
from chickadee.profiles.loader import ProfileSet
from chickadee.profiles.model import LanguageProfile, Template
from chickadee.profiles.tools import find_tool, servers_dir

__all__ = ["INSTALL_TIMEOUT_S", "Installed", "ServerNeed", "install", "server_needs"]

INSTALL_TIMEOUT_S: Final = 900
"""One install: a toolchain fetching and building a server (gopls) takes minutes."""


@dataclass(frozen=True)
class ServerNeed:
    """One server a language's profile names, and where it stands on this machine."""

    languages: tuple[str, ...]
    """The profiles that name it (`clangd` serves C and C++)."""
    server: str
    """Its executable, as the profile names it."""
    found: Path | None
    install: tuple[str, ...] | None
    """The command that installs it into Chickadee's servers directory; None where the
    profile has none, and the person installs it themselves."""
    needs: str | None
    """The tool the install runs (`go`, `npm`), when it is not on this machine."""

    @property
    def installable(self) -> bool:
        return self.found is None and self.install is not None and self.needs is None


@dataclass(frozen=True)
class Installed:
    ok: bool
    said: str
    """The install's last lines, or why it could not run."""


def server_needs(profiles: ProfileSet | None = None) -> tuple[ServerNeed, ...]:
    """Every server the profiles name, once each, in the profiles' order."""
    found: dict[str, ServerNeed] = {}
    for profile in (profiles or ProfileSet.load()).all():
        for spec, install in (
            (profile.lsp.server, profile.lsp.install),
            (profile.lint.server, None),
        ):
            if spec is None:
                continue
            name = spec.argv[0]
            seen = found.get(name)
            if seen is not None:
                found[name] = ServerNeed(
                    (*seen.languages, profile.name),
                    name,
                    seen.found,
                    seen.install,
                    seen.needs,
                )
                continue
            found[name] = _need(profile, name, install)
    return tuple(found.values())


def _need(profile: LanguageProfile, name: str, install: Template | None) -> ServerNeed:
    where = find_tool(name, own=True)
    argv = None if install is None else install.render(dir=str(servers_dir()))
    runs = None if argv is None else _runner(argv)
    needs = runs if runs is not None and find_tool(runs) is None else None
    return ServerNeed((profile.name,), name, where, argv, needs)


def _runner(argv: tuple[str, ...]) -> str | None:
    """The tool an install command runs: past `env` and its `NAME=value` settings."""
    rest = list(argv)
    if rest and Path(rest[0]).name == "env":
        rest = rest[1:]
    while rest and "=" in rest[0] and not rest[0].startswith("-"):
        rest = rest[1:]
    return rest[0] if rest else None


def install(need: ServerNeed, supervisor: ProcessSupervisor | None = None) -> Installed:
    """Run `need`'s install, as the person asked: the network open (its proxies kept), the
    servers directory its working directory. Whether it worked is whether the server is then
    found where `find_tool` looks, not the installer's exit code alone."""
    if need.install is None:
        return Installed(False, f"{need.server} has no installer here; install it yourself")
    if need.needs is not None:
        return Installed(False, f"installing {need.server} needs {need.needs}, not found")
    where = servers_dir()
    (where / "bin").mkdir(parents=True, exist_ok=True)
    try:
        done = (supervisor or ProcessSupervisor()).run(
            need.install,
            cwd=where,
            env=scrub_env(network=True),
            timeout_s=INSTALL_TIMEOUT_S,
        )
    except (OSError, InfraError) as exc:
        return Installed(False, f"could not run {need.install[0]}: {exc}")
    tail = (done.stderr or done.stdout).decode("utf-8", "replace").strip().splitlines()[-3:]
    if done.timed_out:
        return Installed(False, f"timed out after {INSTALL_TIMEOUT_S}s")
    if done.exit_code != 0:
        return Installed(False, f"exit {done.exit_code}: " + " / ".join(tail))
    landed = find_tool(need.server, own=True)
    if landed is None:
        return Installed(False, f"the install ran, but {need.server} is not found after it")
    return Installed(True, f"installed at {landed}")
