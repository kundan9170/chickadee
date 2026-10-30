"""`Hints.save` writes the whole record or none of it.

The record was written to a fixed sibling `.tmp` name and renamed without an fsync: a crash
before the rename left the temp file behind, a crash after it could leave a name whose bytes
never reached the disk, and two processes saving at once shared one temp file. It now goes
through `durable.write_atomic`.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from chickadee.scout.hints import HINTS_FILE, Hints


def _hints(path: Path) -> Hints:
    return Hints(path, now=lambda: "2026-10-02T00:00:00+00:00")


def test_a_crash_before_the_rename_keeps_the_old_record_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state" / HINTS_FILE
    first = _hints(path)
    first.record_run([("rg -n TODO", None)])
    first.save()
    before = path.read_bytes()

    second = _hints(path)
    second.record_run([("git log -1", None)])

    def crash(src: object, dst: object) -> None:
        raise OSError(errno.EIO, "crash before the rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="crash before the rename"):
        second.save()
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [HINTS_FILE]


def test_save_syncs_the_file_before_it_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / HINTS_FILE
    synced: list[int] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    hints = _hints(path)
    hints.record_run([("rg -n TODO", None)])
    hints.save()
    assert len(synced) >= 2, "the file and its directory must both be synced"
    assert _hints(path).get("rg -n TODO") is not None
