"""`chickadee replay <run>`: the run's journal folded,
and whether the fold is clean.

Host-side, like `init`: it reads a journal in this process and needs no server, so it is one of
the CLI's modules allowed the engine (`test_repl_confined.py`).

**What "replays" means here** is the fold: every line of the journal read and folded
into the projection, in seq order. A clean fold is identical; a `Defect` (a seq gap, a line
after `run_end`, an operation id reused,...) is a divergence. The cassetted kinds are folded
from their recorded results and never re-executed, and the report says so, so a clean replay is
never read as a re-run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from chickadee.engine import JournalError, NotARepository, replay_run
from chickadee_cli.surface import DONE, JOURNAL, STOPPED, USAGE

__all__ = ["DIVERGED", "IDENTICAL", "NO_SUCH_RUN", "UNREADABLE", "run"]

IDENTICAL: Final = DONE
NO_SUCH_RUN: Final = USAGE
DIVERGED: Final = STOPPED
UNREADABLE: Final = JOURNAL
"""The `replay` row, in its own words; the codes are `surface`'s."""


def run(cwd: Path, run_id: str) -> tuple[int, str]:
    """The exit code and the report for run `run_id` of the repository containing `cwd`."""
    try:
        folded = replay_run(cwd, run_id)
    except NotARepository as exc:
        return NO_SUCH_RUN, f"not a repository: {exc}"
    except FileNotFoundError as exc:
        return NO_SUCH_RUN, f"no such run: {exc}"
    except (JournalError, OSError, ValueError) as exc:
        return UNREADABLE, f"run {run_id}: the journal cannot be read: {exc}"
    outcome = folded.end.outcome if folded.end is not None else "none (no run_end)"
    lines = [f"run {run_id}: folded, never re-executed; outcome {outcome}"]
    if folded.cassetted_kinds:
        lines.append(
            f"  read from recorded results: {', '.join(sorted(folded.cassetted_kinds))}"
        )
    if not folded.defects:
        lines.append("  identical: every line folds without a defect")
        return IDENTICAL, "\n".join(lines)
    lines.append(f"  diverged: {len(folded.defects)} defect(s)")
    lines += [f"    seq {d.seq}: {d.why}" for d in folded.defects]
    return DIVERGED, "\n".join(lines)
