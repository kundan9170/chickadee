"""A handle back to its bytes: rehydration.

A file handle resolves from disk, by its path; anything stored resolves from a blob, in the
session's store or a run's. Re-reading is what makes recovery exact. It sits above `handle`
and `results`, since it returns `HandleExpired` for a handle whose bytes are gone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from chickadee.context.files import read_file
from chickadee.context.handle import Handle
from chickadee.context.results import HandleExpired
from chickadee.context.spans import Span
from chickadee.journal.errors import BlobMissing

__all__ = ["HandleResolver"]


@dataclass(frozen=True)
class HandleResolver:
    """Where a handle's bytes are. Rehydration is re-reading them — from disk for a file,
    from a blob for everything stored — which is why recovery can be exact.

    `workspace` resolves a relative path; an absolute one is read as given (the
    workspace is not a read boundary for Zeus). `run_blob` reads a run's blob store and
    raises `journal.errors.BlobMissing` on a miss, whose `why` becomes `HANDLE_EXPIRED`'s.
    `journal_text` renders a run's journal, or a seq range of it, for a journal handle.
    """

    workspace: Path
    session_blob: Callable[[str], bytes]
    run_blob: Callable[[str, str], bytes] | None = None
    journal_text: Callable[[str, Span | None], bytes] | None = None

    def file_path(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.workspace / candidate

    def disk_bytes(self, handle: Handle) -> bytes | None:
        """What the file holds now, or None when it is gone. Only a file has a disk.

        Not checked against the deny list, so these bytes are for comparing hashes, never
        for showing: a model sees a file only through the Reader, which asks the list on
        every open (`opener.py`)."""
        if handle.path is None:
            return None
        return read_file(self.file_path(handle.path))

    def stored_bytes(self, handle: Handle) -> bytes | HandleExpired:
        """The bytes a stored handle names, or the typed reason they are gone."""
        try:
            if handle.kind == "journal":
                if self.journal_text is None or handle.run_id is None:
                    return HandleExpired(handle.id, "absent", run_id=handle.run_id)
                return self.journal_text(handle.run_id, handle.seqs)
            if handle.run_id is not None:
                if self.run_blob is None:
                    return HandleExpired(handle.id, "absent", run_id=handle.run_id)
                return self.run_blob(handle.run_id, handle.content_sha)
            return self.session_blob(handle.content_sha)
        except BlobMissing as miss:
            return HandleExpired(handle.id, miss.why, run_id=miss.run_id, gc_at=miss.gc_at)
