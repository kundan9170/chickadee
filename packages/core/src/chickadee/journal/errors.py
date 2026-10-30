"""Journal failure modes.

Each is a distinct recovery path, so they are distinct types. Collapsing them would make
"refuse to resume" and "degrade and continue" the same decision at the call site.
"""

from __future__ import annotations

from typing import Literal


class JournalError(Exception):
    """Base for every journal failure."""


class JournalWriteError(JournalError):
    """Append failed (IOError, disk full). The run aborts — we cannot journal a journal
    failure, so the handoff goes to stderr and the process exits 3."""


class JournalCorrupt(JournalError):
    """Mid-file corruption: a seq gap, unparseable line, or an event that cannot validate.

    Refuses resume. Non-tail content is never "repaired" — a journal we edited is no longer
    the source of truth, and invariant 4 is the whole basis for recovery.
    """

    def __init__(self, seq: int | None, path: str, detail: str = "") -> None:
        self.seq = seq
        self.path = path
        self.detail = detail
        super().__init__(f"journal corrupt at seq={seq} in {path}: {detail}")


class ResumeRefused(JournalError):
    """LOCK is held by a live process. There is never more than one writer."""

    def __init__(self, pid: int, host: str) -> None:
        self.pid = pid
        self.host = host
        super().__init__(f"run is locked by pid {pid} on {host}")


class BlobMissing(JournalError):
    """A referenced digest is not in the blob store.

    Consumers degrade honestly — the classifier sees "digest unavailable" and the handoff
    says so. Nothing fabricates a substitute.

    `why` says which kind of missing: `"pruned"` names the `gc.log` line that removed
    it — its `run_id`, `gc_at` and `user`, looked up on the miss by `journal.gc.blob_missing`,
    never stored beside the blob — and the handle layer reports it as
    `HANDLE_EXPIRED{reason: pruned}`. `"absent"` is everything else: never written,
    or lost to something that was not a gc.
    """

    def __init__(
        self,
        sha: str,
        *,
        why: Literal["pruned", "absent"] = "absent",
        run_id: str | None = None,
        gc_at: str | None = None,
        user: str | None = None,
    ) -> None:
        self.sha = sha
        self.why: Literal["pruned", "absent"] = why
        self.run_id = run_id
        self.gc_at = gc_at
        self.user = user
        if why == "pruned":
            super().__init__(
                f"blob {sha} was pruned from run {run_id} by gc at {gc_at} ({user}); see gc.log"
            )
        else:
            super().__init__(f"blob {sha} is missing")
