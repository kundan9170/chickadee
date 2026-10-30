"""A content-addressed, write-once blob directory.

The run journal's `blobs/`, the session store's and the applier's pre-images were three
stores with three rules, and one of them trusted a torn file as a dedupe hit (#276 B1). This
is the one rule: a blob reaches its name whole, a file of the right size is a hit only once
its directory is known synced, and a file of another size is a torn write and is replaced.

Addressed by sha-256 hex, the digest `chickadee.schema.sha256_hex` names for the project; it is
computed here with `hashlib` because this package is a leaf, and `tests/durable/test_blobs.py`
pins that the two agree.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

from chickadee.durable.files import sync_dir, write_all

__all__ = ["BlobStore", "BlobWriteFailed"]


class BlobWriteFailed(OSError):
    """A blob did not reach its name durably. `synced` is False when the rename happened and
    the directory sync did not: the blob was then unlinked (best effort), and the next `put`
    syncs the directory before it trusts any existing name."""

    def __init__(self, message: str, *, sha: str, synced: bool) -> None:
        super().__init__(message)
        self.sha = sha
        self.synced = synced


BlobPut = Callable[[bytes], str]
"""Stores bytes and returns their sha: `BlobStore.put`, or a journal's `put_blob` over one.
What a component that writes output into a run's blobs is handed (the executor, a gate)."""


class BlobStore:
    """A flat directory of `<sha256 hex>` files, write-once. One writer per instance: the
    owed-sync flag is this instance's, so two writers over one directory each keep their own,
    which is safe because every write is whole and idempotent."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._unsynced = False
        """A directory sync failed after a rename and none has succeeded since."""

    def path(self, sha: str) -> Path:
        return self.directory / sha

    def has(self, sha: str) -> bool:
        return self.path(sha).is_file()

    def read(self, sha: str) -> bytes:
        """The blob's bytes. `FileNotFoundError` on a miss: what a miss means (pruned,
        absent, damage) is the caller's to say."""
        return self.path(sha).read_bytes()

    def put(self, content: bytes) -> str:
        """Store `content` and return its sha. An existing blob of the right size is a
        dedupe hit; one of another size is replaced."""
        sha = hashlib.sha256(content).hexdigest()
        target = self.path(sha)
        with contextlib.suppress(OSError):
            if target.stat().st_size == len(content):
                self._sync_owed(sha)
                return sha
        tmp = self.directory / f".{sha}.{os.getpid()}.tmp"
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                write_all(fd, content)
                os.fsync(fd)
            finally:
                os.close(fd)
            # Written aside and renamed: a crash leaves a stray temp file, never a torn blob
            # under the name a dedupe hit would trust.
            os.replace(tmp, target)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise BlobWriteFailed(
                f"blob write failed for {sha}: {exc}", sha=sha, synced=True
            ) from exc
        try:
            # The rename is a directory entry. Without this, a crash could keep the record
            # that names the blob (synced after) and lose the blob it names.
            sync_dir(self.directory)
        except OSError as exc:
            # The blob is whole but its name may not survive a crash. Left in place, a retry
            # would take it as a dedupe hit and never sync; removed, the retry writes and
            # syncs it again. The unlink is best effort, so the flag is what a retry cannot
            # get past.
            self._unsynced = True
            with contextlib.suppress(OSError):
                target.unlink()
            raise BlobWriteFailed(
                f"blob directory sync failed for {sha}: {exc}", sha=sha, synced=False
            ) from exc
        self._unsynced = False
        return sha

    def _sync_owed(self, sha: str) -> None:
        """Before a dedupe hit is trusted after a failed directory sync: the file is there,
        but whether its name is durable is not known until the directory syncs."""
        if not self._unsynced:
            return
        try:
            sync_dir(self.directory)
        except OSError as exc:
            raise BlobWriteFailed(
                f"blob directory sync failed for {sha}: {exc}", sha=sha, synced=False
            ) from exc
        self._unsynced = False
