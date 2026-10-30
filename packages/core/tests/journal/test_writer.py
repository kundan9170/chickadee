"""Append/read round-trip, seq monotonicity, and the blob store: dedupe, and a blob only
ever reaching its name whole."""

from __future__ import annotations

import errno
import os
import stat

import pytest
from helpers import payload_for, scope_args
from pydantic import ValidationError

from chickadee.journal import Journal, JournalWriteError, paths, read_events
from chickadee.schema import EventKind, sha256_hex
from chickadee.state.roots import Roots


def test_append_read_roundtrip(repo: Roots, run_id: str) -> None:
    with Journal.create(repo, run_id, {"cfg": 1}) as j:
        for kind in (EventKind.RUN_START, EventKind.SCOUT_START, EventKind.RUN_END):
            j.append(kind, payload_for(kind), **scope_args(kind))
    events = list(read_events(paths.journal_path(repo, run_id)))
    assert [e.kind for e in events] == [
        EventKind.RUN_START,
        EventKind.SCOUT_START,
        EventKind.RUN_END,
    ]
    assert [e.seq for e in events] == [1, 2, 3]


def test_seq_is_strictly_monotonic(repo: Roots, run_id: str) -> None:
    with Journal.create(repo, run_id, {}) as j:
        seqs = [
            j.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL)) for _ in range(25)
        ]
    assert seqs == list(range(1, 26))


def test_config_snapshot_is_written(repo: Roots, run_id: str) -> None:
    """Archive discipline: a result nobody can reproduce the config of is not
    a result."""
    with Journal.create(repo, run_id, {"rungs": {"coder": "qwen"}}):
        pass
    snap = paths.run_dir(repo, run_id) / paths.CONFIG_NAME
    assert "qwen" in snap.read_text()


def test_blob_dedupe_and_write_once(repo: Roots, run_id: str) -> None:
    with Journal.create(repo, run_id, {}) as j:
        a = j.put_blob(b"prompt bytes")
        b = j.put_blob(b"prompt bytes")
        assert a == b, "identical content must dedupe to one blob"
        assert j.get_blob(a) == b"prompt bytes"
        blobs = list(paths.blobs_dir(repo, run_id).iterdir())
        assert len(blobs) == 1


def test_a_torn_blob_under_its_name_is_replaced(repo: Roots, run_id: str) -> None:
    # A crash mid-write under the old rule left fewer bytes under the blob's name. Taking
    # that file as a dedupe hit would feed resume the torn bytes from then on.
    content = b"plan bytes, whole"
    with Journal.create(repo, run_id, {}) as j:
        target = paths.blobs_dir(repo, run_id) / sha256_hex(content)
        target.write_bytes(content[:4])
        assert j.put_blob(content) == target.name
        assert j.get_blob(target.name) == content


def test_a_crash_before_the_rename_leaves_no_blob_under_its_name(
    repo: Roots, run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"packet bytes"
    sha = sha256_hex(content)
    with Journal.create(repo, run_id, {}) as j:
        target = paths.blobs_dir(repo, run_id) / sha

        def crash(src: object, dst: object) -> None:
            raise OSError(errno.EIO, "crash before the rename")

        monkeypatch.setattr(os, "replace", crash)
        with pytest.raises(JournalWriteError, match=sha):
            j.put_blob(content)
        assert not target.exists(), "only a whole blob may ever reach its name"
        assert list(paths.blobs_dir(repo, run_id).iterdir()) == []

        monkeypatch.undo()
        assert j.put_blob(content) == sha
        assert target.read_bytes() == content


def test_a_failed_directory_sync_leaves_the_blob_for_a_retry(
    repo: Roots, run_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The blob is whole but its name may not survive a crash. A retry must not take it as a
    # dedupe hit and skip the sync, or the event naming it could outlive it.
    content = b"bundle bytes"
    sha = sha256_hex(content)
    real_fsync = os.fsync
    synced_dirs: list[int] = []

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced_dirs.append(fd)
            raise OSError(errno.EIO, "directory sync failed")
        real_fsync(fd)

    with Journal.create(repo, run_id, {}) as j:
        monkeypatch.setattr(os, "fsync", fsync)
        with pytest.raises(JournalWriteError, match=sha):
            j.put_blob(content)
        assert synced_dirs, "the rename must be followed by a directory sync"
        monkeypatch.undo()

        calls: list[int] = []

        def counting(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                calls.append(fd)
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", counting)
        assert j.put_blob(content) == sha
        assert calls, "the retry must sync the directory, not trust the unsynced name"
        assert j.get_blob(sha) == content


def test_missing_blob_raises_rather_than_fabricating(repo: Roots, run_id: str) -> None:
    from chickadee.journal import BlobMissing

    with Journal.create(repo, run_id, {}) as j, pytest.raises(BlobMissing):
        j.get_blob("f" * 64)


def test_append_after_close_refuses(repo: Roots, run_id: str) -> None:
    j = Journal.create(repo, run_id, {})
    j.close()
    with pytest.raises(JournalWriteError):
        j.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL))


def test_malformed_payload_raises_before_writing(repo: Roots, run_id: str) -> None:
    """Contracts §8: our-code-produced structures always raise. A bad payload must never
    become a line someone has to explain later.

    It must not consume a sequence number either. This asserted `== 2` and passed while
    the rejected append silently burned seq 1, because it never read the journal back —
    and a journal whose first line is seq 2 is one `read_events` away from unreadable.
    """
    with Journal.create(repo, run_id, {}) as j:
        with pytest.raises(ValidationError):
            j.append(EventKind.RUN_START, {"goal": "incomplete"})
        assert j.append(EventKind.SCOUT_TOOL, payload_for(EventKind.SCOUT_TOOL)) == 1
    assert [e.seq for e in read_events(paths.journal_path(repo, run_id))] == [1]
