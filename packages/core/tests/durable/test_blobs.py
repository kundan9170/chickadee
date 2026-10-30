"""A blob reaches its name whole, and a dedupe hit is trusted only when it can be.


The first three are `T-3.46`'s journal blob tests (#276 B1), written against the one store
every caller now uses.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from chickadee.durable import BlobStore, BlobWriteFailed
from chickadee.schema import sha256_hex


def test_a_torn_blob_under_its_name_is_replaced(tmp_path: Path) -> None:
    # A crash mid-write under the old rule left fewer bytes under the blob's name. Taking
    # that file as a dedupe hit would feed every later reader the torn bytes.
    content = b"plan bytes, whole"
    store = BlobStore(tmp_path)
    store.path(sha256_hex(content)).write_bytes(content[:4])
    sha = store.put(content)
    assert store.read(sha) == content


def test_a_crash_before_the_rename_leaves_no_blob_under_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"packet bytes"
    sha = sha256_hex(content)
    store = BlobStore(tmp_path)

    def crash(src: object, dst: object) -> None:
        raise OSError(errno.EIO, "crash before the rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(BlobWriteFailed, match=sha) as failed:
        store.put(content)
    assert failed.value.synced, "nothing was named, so no directory sync is owed"
    assert list(tmp_path.iterdir()) == []

    monkeypatch.undo()
    assert store.put(content) == sha
    assert store.read(sha) == content


def test_a_failed_directory_sync_leaves_the_blob_for_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The blob is whole but its name may not survive a crash. A retry must not take it as a
    # dedupe hit and skip the sync, or the record naming it could outlive it.
    content = b"bundle bytes"
    sha = sha256_hex(content)
    store = BlobStore(tmp_path)
    real_fsync = os.fsync

    def failing(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory sync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", failing)
    with pytest.raises(BlobWriteFailed, match=sha) as failed:
        store.put(content)
    assert not failed.value.synced
    monkeypatch.undo()

    synced: list[int] = []

    def counting(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", counting)
    assert store.put(content) == sha
    assert synced, "the retry must sync the directory, not trust the unsynced name"
    assert store.read(sha) == content


def test_a_dedupe_hit_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = BlobStore(tmp_path)
    sha = store.put(b"same bytes")
    before = store.path(sha).stat().st_mtime_ns
    monkeypatch.setattr(os, "replace", lambda *a: pytest.fail("a hit must not rewrite"))
    assert store.put(b"same bytes") == sha
    assert store.path(sha).stat().st_mtime_ns == before
    assert [p.name for p in tmp_path.iterdir()] == [sha]


def test_the_address_is_the_projects_digest(tmp_path: Path) -> None:
    """The store computes sha-256 with `hashlib` to stay a leaf; it must be the same digest
    every record names a blob by."""
    for content in (b"", b"x", "café".encode(), bytes(range(256)) * 9):
        assert BlobStore(tmp_path).put(content) == sha256_hex(content)


def test_a_miss_is_file_not_found_and_has_says_no(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    assert not store.has("f" * 64)
    with pytest.raises(FileNotFoundError):
        store.read("f" * 64)
