"""The snapshot/diff bracket: what a command actually touched."""

from __future__ import annotations

from pathlib import Path

from chickadee.executor.effects import diff_against, snapshot_paths


def test_an_undeclared_creation_is_a_violation(work: Path) -> None:
    before = snapshot_paths(work)
    (work / "new.txt").write_text("x\n", encoding="utf-8")
    violations = diff_against(before, snapshot_paths(work), ())
    assert [(v.path, v.reason) for v in violations] == [("new.txt", "created")]


def test_an_undeclared_modification_is_a_violation(work: Path) -> None:
    before = snapshot_paths(work)
    (work / "existing.txt").write_text("changed and longer\n", encoding="utf-8")
    violations = diff_against(before, snapshot_paths(work), ())
    assert [(v.path, v.reason) for v in violations] == [("existing.txt", "modified")]


def test_a_deletion_is_a_violation(work: Path) -> None:
    before = snapshot_paths(work)
    (work / "existing.txt").unlink()
    violations = diff_against(before, snapshot_paths(work), ())
    assert [(v.path, v.reason) for v in violations] == [("existing.txt", "deleted")]


def test_a_declared_glob_covers_its_paths(work: Path) -> None:
    before = snapshot_paths(work)
    (work / "build").mkdir()
    (work / "build" / "out.o").write_text("x\n", encoding="utf-8")
    assert diff_against(before, snapshot_paths(work), ("build/**",)) == ()


def test_an_exact_declaration_covers_one_path(work: Path) -> None:
    before = snapshot_paths(work)
    (work / "existing.txt").write_text("declared change\n", encoding="utf-8")
    assert diff_against(before, snapshot_paths(work), ("existing.txt",)) == ()


def test_noise_directories_are_not_measured(work: Path) -> None:
    """`.git` and `__pycache__` churn on their own; measuring them would make every run
    look like a violation."""
    before = snapshot_paths(work)
    (work / "__pycache__").mkdir()
    (work / "__pycache__" / "m.pyc").write_text("bytes\n", encoding="utf-8")
    assert diff_against(before, snapshot_paths(work), ()) == ()


def test_violations_are_sorted(work: Path) -> None:
    """A stable list, so a digest built from it does not change run to run."""
    before = snapshot_paths(work)
    for name in ("c.txt", "a.txt", "b.txt"):
        (work / name).write_text("x\n", encoding="utf-8")
    assert [v.path for v in diff_against(before, snapshot_paths(work), ())] == [
        "a.txt",
        "b.txt",
        "c.txt",
    ]


def test_an_undeclared_empty_directory_is_a_violation(work: Path) -> None:
    """A file-only snapshot sees nothing when an action creates or removes an empty dir."""
    before = snapshot_paths(work)
    (work / "sneaky").mkdir()
    violations = diff_against(before, snapshot_paths(work), ())
    assert [v.path for v in violations] == ["sneaky"]


def test_a_declared_directory_is_not_a_violation(work: Path) -> None:
    before = snapshot_paths(work)
    (work / "build").mkdir()
    assert diff_against(before, snapshot_paths(work), ("build",)) == ()


def test_removing_an_empty_directory_is_a_violation(work: Path) -> None:
    (work / "gone").mkdir()
    before = snapshot_paths(work)
    (work / "gone").rmdir()
    assert [v.reason for v in diff_against(before, snapshot_paths(work), ())] == ["deleted"]


def test_a_same_size_rewrite_with_a_restored_mtime_is_still_caught(work: Path) -> None:
    """The evasion mtime+size allowed, verified before the fix: rewrite a file to the same
    length and put its timestamp back, and the diff reported nothing.

    `ctime` closes it because it changes on any inode modification and cannot be set from
    userland — which is the whole reason it is in the fingerprint.
    """
    import os
    import time

    target = work / "existing.txt"
    target.write_text("AAAA", encoding="utf-8")
    original = target.stat()
    before = snapshot_paths(work)

    time.sleep(0.01)
    target.write_text("BBBB", encoding="utf-8")
    # Nanoseconds, not float seconds. A float round-trip shifts `st_mtime_ns` by ~90ns, so
    # the earlier version of this test passed because *mtime* caught the rewrite — the
    # opposite of what it claims to prove. The assertion below pins that.
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))

    restored = target.stat()
    assert restored.st_mtime_ns == original.st_mtime_ns, "mtime must be exactly restored"
    assert restored.st_size == original.st_size, "size must be unchanged"
    assert restored.st_ctime_ns != original.st_ctime_ns, "ctime is the only signal left"

    violations = diff_against(before, snapshot_paths(work), ())
    assert [v.path for v in violations] == ["existing.txt"]


def test_a_directory_made_to_hold_a_declared_change_is_part_of_it(work: Path) -> None:
    """`build/**` names what is inside `build`; making `build` is how it is
    written, so the directory is covered by the file it holds."""
    before = snapshot_paths(work)
    (work / "build" / "obj").mkdir(parents=True)
    (work / "build" / "obj" / "out.o").write_text("x\n", encoding="utf-8")
    assert diff_against(before, snapshot_paths(work), ("build/**",)) == ()


def test_an_empty_directory_is_judged_by_its_own_path(work: Path) -> None:
    """Nothing declared is inside it, so `build/**` does not cover it; `build`
    would. Before D168 `build/**` matched `build` itself."""
    before = snapshot_paths(work)
    (work / "build").mkdir()
    after = snapshot_paths(work)
    assert [v.path for v in diff_against(before, after, ("build/**",))] == ["build"]
    assert diff_against(before, after, ("build",)) == ()
