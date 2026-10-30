"""The append-only source of truth (invariant 4).

Everything else — the task book, budget spend, the operations — is a cache
reconstructed by folding these events, once (`projection`).
"""

from __future__ import annotations

from chickadee.journal.errors import (
    BlobMissing,
    JournalCorrupt,
    JournalError,
    JournalWriteError,
    ResumeRefused,
)
from chickadee.journal.metrics import RunMetrics, derive_metrics
from chickadee.journal.paths import (
    journal_path,
    run_dir,
    runs_root,
)
from chickadee.journal.projection import (
    Defect,
    Fold,
    Operation,
    Placed,
    RunProjection,
    RunSpend,
    project,
    project_before,
    project_readable,
)
from chickadee.journal.reader import read_events
from chickadee.journal.writer import Journal

__all__ = [
    "BlobMissing",
    "Defect",
    "Fold",
    "Journal",
    "JournalCorrupt",
    "JournalError",
    "JournalWriteError",
    "Operation",
    "Placed",
    "ResumeRefused",
    "RunMetrics",
    "RunProjection",
    "RunSpend",
    "derive_metrics",
    "journal_path",
    "project",
    "project_before",
    "project_readable",
    "read_events",
    "run_dir",
    "runs_root",
]
