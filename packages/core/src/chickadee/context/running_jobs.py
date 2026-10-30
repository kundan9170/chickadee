"""The jobs running now, folded from the session log's `shell_effect` lines.

A job is an open `JOB` effect: its intent is written at the yield, its completion by its waiter,
by the session's close, or by resume. The session knows its running jobs from its log alone,
so the safe point, liveness and `zeus.max_jobs` read one answer, in this process or another.
"""

from __future__ import annotations

from chickadee.context.kept_lines import EffectRecord
from chickadee.ids import job_of
from chickadee.journal.effects import EffectKind
from chickadee.schema.journal import EffectCompletedP, EffectIntendedP

__all__ = ["RunningJobs"]


class RunningJobs:
    """Open `JOB` effects, operation id to job id, in the order they started."""

    def __init__(self) -> None:
        self._open: dict[str, str] = {}

    def apply(self, record: EffectRecord) -> None:
        payload = record.line.payload if record.line is not None else None
        if isinstance(payload, EffectIntendedP) and payload.effect_kind == EffectKind.JOB.value:
            self._open[payload.operation_id] = job_of(payload.preconditions) or "?"
        elif isinstance(payload, EffectCompletedP):
            self._open.pop(payload.operation_id, None)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._open.values())
