"""A graph indexer that records what it was asked to reindex (`box/commit.Reindexer`), for any
suite that checks exactly which paths a commit or a step reindexed. One home: the
orchestrator's run box and the task commit's tests both read it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RecordingIndexer:
    calls: list[tuple[Path, ...]] = field(default_factory=list[tuple[Path, ...]])

    def reindex(self, changed: Sequence[Path] | None = None) -> object:
        self.calls.append(tuple(changed or ()))
        return None
