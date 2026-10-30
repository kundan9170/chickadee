"""`chickadee hooks`: the points, the bounds, and what this
repository's config binds at each, read and printed, nothing run. What each entry would do is
core's (`hooks.show`, through the engine's facade), from the same config a run binds; this file
prints.

Exit codes: `0` shown; `1` the config could not be read; `4` not a repository.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from chickadee.engine import NotARepository, hooks_shown
from chickadee_cli.surface import DONE, PRECONDITION, USAGE

__all__ = ["run"]


def run(cwd: Path, *, say: Callable[[str], None]) -> int:
    try:
        said = hooks_shown(cwd)
    except NotARepository as exc:
        say(f"hooks: not a repository: {exc}")
        return PRECONDITION
    except ValueError as exc:
        say(f"hooks: {exc}")
        return USAGE
    say(said)
    return DONE
