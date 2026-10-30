"""`chickadee gc`: the bulk blobs of old,
finished runs removed, one `gc.log` line per run.

Host-side, like `init`: `journal.gc` reads the runs on disk in this process and needs no
server. It runs only when a person types it; nothing here schedules it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from chickadee.engine import BULK, BlobClass, GcPlan, JournalError, NotARepository, gc_runs
from chickadee_cli.surface import DONE, USAGE

__all__ = ["CLASSES", "RETAIN_DAYS", "run"]

RETAIN_DAYS: Final = 30
"""`--older-than`'s default: `runs.retain_days`'s default, which no config
loader reads yet."""

CLASSES: Final = tuple(sorted(c.value for c in BULK))
"""What `--class` may name: the bulk classes. The record is never pruned."""


def run(
    cwd: Path,
    *,
    older_than_days: int,
    classes: Sequence[str],
    run_id: str | None,
    dry_run: bool,
) -> tuple[int, str]:
    """The exit code and the report: `0` with the per-run rows, `1` refused before anything
    was removed."""
    unknown = sorted(set(classes) - set(CLASSES))
    if unknown:
        return USAGE, (
            f"gc: --class may name {', '.join(CLASSES)}, not {', '.join(unknown)}; "
            "the record is never pruned (J-6)"
        )
    try:
        chosen = frozenset(BlobClass(c) for c in classes) if classes else BULK
        planned, done = gc_runs(
            cwd,
            older_than=dt.timedelta(days=older_than_days),
            classes=chosen,
            run=run_id,
            dry_run=dry_run,
        )
    except NotARepository as exc:
        return USAGE, f"gc: not a repository: {exc}"
    except JournalError as exc:
        return USAGE, f"gc: {exc}"
    if done is None:
        return DONE, _report(planned, "would remove")
    return DONE, _report(planned, "removed") + f"\n  logged to {done.log}"


def _report(planned: GcPlan, verb: str) -> str:
    cutoff = planned.cutoff.isoformat(timespec="seconds")
    lines = [f"gc: runs last written before {cutoff}"]
    lines += [
        f"  {row.run_id}: {verb} {len(row.shas)} blob(s), {row.bytes} bytes "
        f"({', '.join(sorted(c.value for c in row.classes))})"
        for row in planned.rows
    ]
    lines += [f"  {run_id}: skipped, {why}" for run_id, why in planned.skipped]
    if not planned.rows:
        lines.append("  nothing to remove")
    lines.append(f"  total: {planned.shas} blob(s), {planned.bytes} bytes")
    return "\n".join(lines)
