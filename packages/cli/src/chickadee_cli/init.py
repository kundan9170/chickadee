"""`chickadee init` — the command.

Thin by construction: it calls the facade and renders what comes back. The ordering,
idempotency and `--force` semantics are core's, because a second surface implementing them
would be a second answer to "is this repository ready".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from chickadee.engine import (
    InitError,
    InitReport,
    LeaseBusy,
    NotAGitRepository,
    in_repository,
    init_repo,
    make_repository,
)
from chickadee_cli import doctor
from chickadee_cli.surface import DONE, HELD, PRECONDITION


def run(
    root: Path, *, force: bool = False, say: Callable[[str], None] = lambda _line: None
) -> tuple[int, str]:
    """The exit code (`0`, `4` not initialisable, `5` the lease is held) and
    what to print. A folder that is not a git repository is made one first, each step told
    to `say` as it happens."""
    try:
        report = init_repo(root, force=force, say=say)
    except (NotAGitRepository, InitError) as exc:
        return PRECONDITION, f"chickadee init: {exc}"
    except LeaseBusy as exc:
        return HELD, f"chickadee init: {exc}"
    return DONE, render(report)


def prepare(here: Path, *, say: Callable[[str], None]) -> tuple[int, str] | None:
    """What an engine needs of `here` before it starts: a git repository, made
    where there is none, each step told to `say`. The index is not built here: the engine
    builds it in the background (`PacketEngine.index_in_background`), so the screen opens at
    once. None when ready, else the code and what to print."""
    if in_repository(here):
        return None
    try:
        make_repository(here, say=say)
    except InitError as exc:
        return PRECONDITION, f"chickadee: {exc}"
    return None


def render(report: InitReport) -> str:
    cap = report.hard_cost_cap_usd
    lines = [
        f"initialised {report.root}",
        f"  repository  {report.repository_id}",
        f"  generation  {report.generation}",
        # The cap is the engine config's. With none set, the note below says where
        # it goes; a `$None` here would read as a number to anyone skimming.
        f"  hard cap    {'none set (see note)' if cap is None else f'${cap}'}",
    ]
    for evidence in report.profiles:
        lines.append(f"  profile     {evidence.name} — {evidence.evidence}")
    index = report.index
    lines.append(
        f"  index       {index.files_indexed} files, {index.symbols} symbols, "
        f"{len(index.gaps)} not indexed"
        # A capped index is a prefix of the repository, said so rather than implied whole.
        + (" (truncated at the file cap)" if index.truncated else "")
    )
    if report.gaps:
        # Gaps are printed, never counted away: a file the harness cannot parse is a hole
        # in every later answer about the repository (invariant 7).
        shown = ", ".join(report.gaps[:5])
        more = f" (+{len(report.gaps) - 5} more)" if len(report.gaps) > 5 else ""
        lines.append(f"  gaps        {len(report.gaps)}: {shown}{more}")
    for created in report.created:
        lines.append(f"  created     {created}")
    for warning in report.warnings:
        lines.append(f"  note        {warning}")
    probe = doctor.render(report.toolchains, report.degraded)
    return "\n".join(lines) + ("\n\n" + probe if probe else "")
