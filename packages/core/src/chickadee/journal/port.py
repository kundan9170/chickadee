"""The run journal as its writers and readers hold it (part 09): `JournalPort`.

`journal.writer.Journal` satisfies it. The box's roles, steps and desk append through it and
read the run's projection back from it; a test gives a fake with the same methods. Moved here
from `orchestrator/ports.py` when the orchestrator retired: the journal's
own package is its one home.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from chickadee.journal.projection import RunProjection
from chickadee.schema.journal import EventKind, JournalEvent

__all__ = ["JournalPort"]


class JournalPort(Protocol):
    """One run's journal: append a line, store and read a blob, read the lines back."""

    @property
    def run_id(self) -> str: ...

    def append(
        self,
        kind: EventKind,
        payload: Any,
        *,
        task_id: str | None = None,
        round: int | None = None,
    ) -> int: ...

    def put_blob(self, content: bytes) -> str: ...

    def get_blob(self, sha: str) -> bytes: ...

    def events(self) -> Iterator[JournalEvent]: ...

    def projection(self) -> RunProjection:
        """The run as of the last durable append, folded once: what every reader
        of the run's own journal reads."""
        ...
