"""The lines the session's fold keeps unread, or reads only in part.

Its own module because `session.py` is held at its size by the context package's guard. These
are the records of lines another package owns: `OtherLine` for one the fold keeps in
order and never reads; `EffectRecord` for Zeus's shell's D40 records, which the fold reads for
the jobs they open and close; `AgentRecord` for a spawned agent's lines, kept
in order for `agents/` to read.
"""

from __future__ import annotations

from dataclasses import dataclass

from chickadee.context.record_lines import AgentLine, ShellEffectLine
from chickadee.context.vocabulary import SessionKind

__all__ = ["AgentRecord", "EffectRecord", "OtherLine"]


@dataclass(frozen=True)
class OtherLine:
    """A line another spec owns. Kept in order; its payload is read by its
    owner, and by `recall`."""

    seq: int
    kind: SessionKind
    run_id: str | None


@dataclass(frozen=True)
class AgentRecord(OtherLine):
    """An `agent` line: its owner reads the payload (`agents/`); the fold keeps it in order."""

    line: AgentLine | None = None


@dataclass(frozen=True)
class EffectRecord(OtherLine):
    """A `shell_effect` line: its owner reads the payload (`zeus/writes.shell_effects`); the
    fold reads only whether it opens or closes a job."""

    line: ShellEffectLine | None = None
