"""`chickadee trust`: what this repository declares
that can run, shown itemised and verbatim with its digest, and recorded on the person's yes.

Two things a clone does not run until its person accepts them: the config's executable subset
(hooks, MCP servers, `[[commands]]` rows, a deny-list removal, a bootstrap override) and the
repository's approval file's `[[approval.grants]]` and `allow` rules. Each has its own digest;
a yes records the pending ones in the one `trusted.toml`, and the prompt names them. Reading
and deciding are core's (`state/load.repository_trust`, the digests the engine checks); this
file prints, asks and calls `accept_trust`.

Exit codes: `0` shown and, where anything was pending, accepted or declined; `1` the
config or approval file could not be read; `4` not a repository; `6` not a terminal, so
nothing pending was accepted: a script cannot say yes on a person's behalf.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from chickadee.engine import NotARepository, RepositoryTrust, accept_trust, repository_trust
from chickadee_cli.surface import DONE, NOT_A_TTY, PRECONDITION, USAGE

__all__ = ["prompt", "render", "run"]

EXECUTABLE = "the executable config"
APPROVAL = "the approval file's grants and allow rules"


def run(
    cwd: Path,
    *,
    ask: Callable[[str], str] | None,
    say: Callable[[str], None],
) -> int:
    """Show what `cwd`'s repository declares and, when `ask` is given (a terminal) and anything
    is pending, record it on a yes. `ask` takes the prompt and returns the line typed."""
    try:
        shown = repository_trust(cwd)
    except NotARepository as exc:
        say(f"trust: not a repository: {exc}")
        return PRECONDITION
    except ValueError as exc:
        say(f"trust: {exc}")
        return USAGE
    say(render(shown))
    if not shown.pending:
        return DONE
    if ask is None:
        say("trust: not a terminal, so nothing was accepted; run `chickadee trust` in one")
        return NOT_A_TTY
    if ask(prompt(shown)).strip().lower() not in {"y", "yes"}:
        say("nothing accepted; what is listed stays inert")
        return DONE
    written = accept_trust(shown)
    say(f"accepted, recorded in {written}")
    return DONE


def prompt(shown: RepositoryTrust) -> str:
    """The question, naming exactly the parts a yes records: those not yet accepted."""
    parts = [
        name
        for name, waiting in (
            (EXECUTABLE, not shown.executable.is_empty and not shown.executable_accepted),
            (APPROVAL, not shown.grants.trusted),
        )
        if waiting
    ]
    return f"accept {' and '.join(parts)} for this clone? [y/N] "


def render(shown: RepositoryTrust) -> str:
    """Each part, itemised: every executable key with its value exactly as declared, every
    grant table as written and every class the file allows, each part's digest and whether
    this clone accepted it."""
    lines = ["executable config"]
    if shown.executable.is_empty:
        lines.append("  none declared")
    else:
        state = _state(shown.executable_accepted)
        lines.append(f"  digest  {shown.executable.digest}  {state}")
        lines.extend(_items(shown.executable.subset))
    lines.append("approval file: grants and allow rules")
    if shown.grants.empty:
        lines.append("  none declared")
    else:
        lines.append(f"  digest  {shown.grants.digest}  {_state(shown.grants.trusted)}")
        for n, table in enumerate(shown.grants.declared):
            lines.append(f"  [{n}] {_verbatim(table)}")
        # An `allow` lowers a class for everyone who clones the repository (`prefs.py`).
        lines.extend(f'  [approval.commands] {c} = "allow"' for c in shown.grants.lowerings)
    return "\n".join(lines)


def _state(accepted: bool) -> str:
    return "accepted" if accepted else "NOT accepted: inert"


def _items(subset: Mapping[str, Any]) -> list[str]:
    return [f"  {key} = {_verbatim(value)}" for key, value in sorted(subset.items())]


def _verbatim(value: object) -> str:
    # JSON is exact for every TOML value a config holds and shows strings quoted, so a command
    # with a trailing space or an embedded newline reads as it is.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
