"""A-5, DU-1 to DU-3: a crash leaves a file old XOR new, never a hybrid.

Moved from `tests/applier/test_atomicity.py` by `T-3.48` with its assertions unchanged:
the primitive it tests now lives in `chickadee.durable`, which the applier calls."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from chickadee.durable import (
    NEW_FILE_MODE,
    Fault,
    make_dir,
    rename_durably,
    unlink_durably,
    write_all,
    write_atomic,
    write_new,
)
from chickadee.durable import files as durable_files


class Crash(RuntimeError):
    """Stands in for the process dying. Raised from the injection seam."""


@pytest.mark.parametrize("point", list(Fault))
def test_a_crash_at_any_point_leaves_old_xor_new(tmp_path, point: Fault) -> None:
    """The property, tested by actually interrupting rather than by assertion.

    An atomicity claim nobody has interrupted is a claim about code that has never been
    interrupted.
    """
    target = tmp_path / "f.txt"
    old, new = b"old contents\n", b"new contents\n"
    target.write_bytes(old)

    def fault(hit: Fault) -> None:
        if hit is point:
            raise Crash(point)

    with pytest.raises(Crash):
        write_atomic(target, new, fault=fault)

    got = target.read_bytes()
    assert got in (old, new), f"hybrid state after crash at {point}: {got!r}"


@pytest.mark.parametrize("point", list(Fault))
def test_no_temp_files_survive_a_crash(tmp_path, point: Fault) -> None:
    target = tmp_path / "f.txt"
    target.write_bytes(b"old\n")

    def fault(hit: Fault) -> None:
        if hit is point:
            raise Crash(point)

    with pytest.raises(Crash):
        write_atomic(target, b"new\n", fault=fault)

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "f.txt"]
    assert leftovers == [], f"temp file survived a crash at {point}: {leftovers}"


def test_a_create_that_crashes_leaves_no_partial_file(tmp_path) -> None:
    target = tmp_path / "new.txt"

    def fault(hit: Fault) -> None:
        if hit is Fault.AFTER_WRITE:
            raise Crash(hit)

    with pytest.raises(Crash):
        write_atomic(target, b"partial", fault=fault)
    assert not target.exists()


def test_the_happy_path_writes_the_bytes(tmp_path) -> None:
    target = tmp_path / "a" / "b" / "f.txt"
    write_atomic(target, b"hello\n")
    assert target.read_bytes() == b"hello\n"


def test_permission_bits_survive_replacement(tmp_path) -> None:
    """`mkstemp` creates 0600 and `replace` carries the temp mode onto the target, so an
    executable hook would keep its bytes and stop running."""
    import stat

    target = tmp_path / "hook.sh"
    target.write_text("#!/bin/sh\necho old\n", encoding="utf-8")
    target.chmod(0o755)
    write_atomic(target, b"#!/bin/sh\necho new\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_a_new_file_is_not_owner_only(tmp_path) -> None:
    import stat

    target = tmp_path / "fresh.py"
    write_atomic(target, b"x = 1\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


# -- added by T-3.48: every primitive, at every fault, and the failures it must report ------


def _crash_at(point: Fault):
    def fault(hit: Fault) -> None:
        if hit is point:
            raise Crash(point)

    return fault


def _failing_dir_sync(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Fail the next directory fsync; file fsyncs pass."""
    real = os.fsync
    failed: list[int] = []

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode) and not failed:
            failed.append(fd)
            raise OSError(errno.EIO, "directory sync failed")
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    return failed


@pytest.mark.parametrize("point", [Fault.AFTER_FSYNC, Fault.AFTER_RENAME])
def test_a_crash_while_renaming_a_file_leaves_it_at_one_name(tmp_path, point: Fault) -> None:
    src, dst = tmp_path / "a" / "f.txt", tmp_path / "b" / "f.txt"
    src.parent.mkdir()
    dst.parent.mkdir()
    src.write_bytes(b"bytes\n")
    with pytest.raises(Crash):
        rename_durably(src, dst, fault=_crash_at(point))
    at = [p for p in (src, dst) if p.exists()]
    assert len(at) == 1 and at[0].read_bytes() == b"bytes\n"


def test_a_directory_renamed_across_parents_syncs_both(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging, generations = tmp_path / ".init" / "g1", tmp_path / "generations"
    staging.mkdir(parents=True)
    (staging / "config.toml").write_bytes(b"x = 1\n")
    generations.mkdir()
    synced: list[Path] = []
    real = durable_files.sync_dir
    monkeypatch.setattr(durable_files, "sync_dir", lambda d: (synced.append(d), real(d)))
    rename_durably(staging, generations / "g1")
    assert (generations / "g1" / "config.toml").read_bytes() == b"x = 1\n"
    assert set(synced) == {generations, tmp_path / ".init"}


def test_unlink_durably_removes_and_tolerates_a_missing_file(tmp_path) -> None:
    target = tmp_path / "f.txt"
    target.write_bytes(b"old\n")
    unlink_durably(target)
    assert not target.exists()
    unlink_durably(target)  # already gone: nothing to do, and no error


@pytest.mark.parametrize(
    "act",
    [
        lambda p: write_atomic(p / "f.txt", b"new\n"),
        lambda p: unlink_durably(p / "old.txt"),
        lambda p: rename_durably(p / "old.txt", p / "moved.txt"),
        lambda p: make_dir(p / "x" / "y"),
    ],
    ids=["write_atomic", "unlink_durably", "rename_durably", "make_dir"],
)
def test_a_failed_directory_sync_raises(tmp_path, monkeypatch: pytest.MonkeyPatch, act) -> None:
    """A rename, create or unlink the caller is told is durable must be."""
    (tmp_path / "old.txt").write_bytes(b"old\n")
    failed = _failing_dir_sync(monkeypatch)
    with pytest.raises(OSError, match="directory sync failed"):
        act(tmp_path)
    assert failed


def test_make_dir_creates_every_level_and_syncs_each_new_entry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[Path] = []
    real = durable_files.sync_dir
    monkeypatch.setattr(durable_files, "sync_dir", lambda d: (synced.append(d), real(d)))
    make_dir(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b").is_dir()
    assert synced == [tmp_path, tmp_path / "a"]
    synced.clear()
    make_dir(tmp_path / "a" / "b")  # already there: nothing new to sync
    assert synced == []


def test_write_all_writes_every_byte_across_short_writes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: real(fd, bytes(data)[:3]))
    target = tmp_path / "f.bin"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        write_all(fd, b"0123456789")
    finally:
        os.close(fd)
    assert target.read_bytes() == b"0123456789"


def test_the_mode_is_set_before_the_fsync(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set after the fsync, a crash could bring the file back with mkstemp's 0600."""
    real = os.fsync
    modes: list[int] = []

    def fsync(fd: int) -> None:
        st = os.fstat(fd)
        if stat.S_ISREG(st.st_mode):
            modes.append(stat.S_IMODE(st.st_mode))
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    write_atomic(tmp_path / "f.txt", b"x\n")
    assert modes == [0o644]


# -- write_new: create, never replace ---------------------------------------------


def test_write_new_creates_a_file_that_is_not_there(tmp_path: Path) -> None:
    target = tmp_path / "a" / "config.toml"
    assert write_new(target, b"seed\n")
    assert target.read_bytes() == b"seed\n"
    assert stat.S_IMODE(target.stat().st_mode) == NEW_FILE_MODE
    assert [p.name for p in target.parent.iterdir()] == ["config.toml"]


def test_write_new_never_replaces_an_existing_file(tmp_path: Path) -> None:
    """The check and the create are one step: a file that appears between them is kept, and
    the call says it wrote nothing."""
    target = tmp_path / "config.toml"
    target.write_bytes(b"the person's own\n")
    assert not write_new(target, b"seed\n")
    assert target.read_bytes() == b"the person's own\n"
    assert [p.name for p in tmp_path.iterdir()] == ["config.toml"], "no temp file is left"
