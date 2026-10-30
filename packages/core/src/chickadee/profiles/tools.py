"""Where an executable a profile names is found: the one home of that lookup.

`PATH` alone misses two places a tool is commonly installed. A repository's own virtualenv is
on `PATH` only while it is activated; Chickadee's own environment is never on it when Chickadee
was installed as a tool (`uv tool install`, `pipx`), which puts only `chickadee` itself there.
So a tool is looked for, in order:

1. the repository's `.venv`: the versions it pins, so the team's verdicts match its CI, but
   only one git tracks nothing of. A clone can commit a `.venv/bin/ruff`, and a server starts
   as soon as the repository is opened, sandboxed or not: an environment the person made
   (untracked, as one always is) is theirs; one the repository brought is not;
2. `PATH`;
3. with `own`, Chickadee's own environment: the servers it ships (`basedpyright`, `ruff`,
   `packages/core/pyproject.toml`), then its servers directory (`servers_dir`), where
   `chickadee setup` installs the others the person said yes to. Only for the harness's own
   processes, a language server or a linter, never a model's command: that runs the
   project's tools, not the harness's.

In each place, a name is looked for and then its alternates (`ALTERNATES`): `python` is
`python3` on a system that has only that (macOS, Debian and Ubuntu without `python-is-python3`).
Place by place, so a repository's own `python` wins over a `python3` on `PATH`.

The path found is absolute, so a spawn runs that binary whatever `PATH` its environment has.
"""

from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path
from typing import Final

from chickadee.errors import InfraError
from chickadee.isolation.types import harness_cache
from chickadee.process import run_git

__all__ = ["ALTERNATES", "find_tool", "servers_dir"]

_SCRIPTS = "Scripts" if os.name == "nt" else "bin"

ALTERNATES: Final[dict[str, tuple[str, ...]]] = {"python": ("python3",)}
"""A name a profile runs, and the names the same program has where that one is missing."""


def servers_dir() -> Path:
    """Where `chickadee setup` installs a language's server: one per machine, its executables
    in `bin/`. Under the harness's cache, never `/tmp`, which a sandbox replaces."""
    return harness_cache() / "servers"


def find_tool(name: str, root: Path | None = None, *, own: bool = False) -> Path | None:
    """`name`'s executable, absolute; None when none of the places holds it. A `name` that is
    a path is that file, or None."""
    places: list[str | None] = []
    if root is not None and _made_here(root / ".venv"):
        places.append(str(root / ".venv" / _SCRIPTS))
    places.append(None)  # `PATH`
    if own:
        places.append(sysconfig.get_path("scripts"))
        places.append(str(servers_dir() / _SCRIPTS))
    for place in places:
        for candidate in (name, *ALTERNATES.get(name, ())):
            found = shutil.which(candidate, path=place)
            if found is not None:
                return Path(found).absolute()
    return None


def _made_here(venv: Path) -> bool:
    """Whether `venv` is an environment made in this checkout: there, and nothing in it
    tracked by git. Outside a repository, there is no clone to have brought it; where git
    cannot be asked, it is not looked in."""
    if not venv.is_dir():
        return False
    try:
        done = run_git(venv.parent, "ls-files", "-z", "--", venv.name)
    except (OSError, InfraError):
        return False
    return done.exit_code != 0 or not done.stdout
