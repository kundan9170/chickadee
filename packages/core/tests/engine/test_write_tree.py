"""Fix 1: a snapshot starts its throwaway index from a copy of the
worktree's own, so git re-reads only what changed, and writes the same tree an empty index
did. Pinned before the change against the tree an empty index writes, built here with plain
git, in the main checkout and in a linked worktree, whose index is not `.git/index`.

A copied index carries the person's assume-unchanged and skip-worktree flags, which hide a
file's change from `add -A`; the snapshot clears each on a file that exists on disk.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.workspace import current_tree

FILES = {
    "a.py": "a = 1\n",
    "b.py": "b = 1\n",
    "gone.py": "g = 1\n",
    "assumed.py": "x = 1\n",
    "skipped.py": "y = 1\n",
    "both.py": "z = 1\n",
    "pkg/deep.py": "d = 1\n",
}


def _empty_index_tree(root: Path) -> str:
    """The tree an empty throwaway index writes: every file on disk as `add -A` stages it,
    with no flag from the person's index. What a snapshot wrote before fix 1."""
    with tempfile.TemporaryDirectory() as tmp:
        index = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(root, "add", "-A", env=index)
        return git(root, "write-tree", env=index)


@pytest.fixture(params=["main", "linked"])
def checkout(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    main = init_repo(tmp_path / "repo", FILES)
    if request.param == "main":
        return main
    linked = tmp_path / "linked"
    git(main, "worktree", "add", "-q", str(linked), "-b", "side")
    return linked


def _flag(root: Path, *args: str) -> None:
    git(root, "update-index", *args)


def test_an_edit_a_delete_and_an_untracked_file_write_the_same_tree(checkout: Path) -> None:
    (checkout / "a.py").write_text("a = 2\n", encoding="utf-8")
    (checkout / "gone.py").unlink()
    (checkout / "new.py").write_text("n = 1\n", encoding="utf-8")
    (checkout / "pkg" / "deep.py").write_text("d = 2\n", encoding="utf-8")
    tree, skipped = current_tree(checkout, untracked_max_bytes=None)
    assert tree == _empty_index_tree(checkout)
    assert skipped == ()


def test_a_flagged_file_changed_on_disk_is_in_the_tree(checkout: Path) -> None:
    """A copy that kept the flags would hide these three changes and keep their old text."""
    _flag(checkout, "--assume-unchanged", "assumed.py")
    _flag(checkout, "--skip-worktree", "skipped.py")
    _flag(checkout, "--assume-unchanged", "both.py")
    _flag(checkout, "--skip-worktree", "both.py")
    for name in ("assumed.py", "skipped.py", "both.py"):
        (checkout / name).write_text(f"# changed\n{name[0]} = 2\n", encoding="utf-8")
    tree, _ = current_tree(checkout, untracked_max_bytes=None)
    assert tree == _empty_index_tree(checkout)


def test_the_persons_index_and_its_flags_are_untouched(checkout: Path) -> None:
    _flag(checkout, "--assume-unchanged", "assumed.py")
    _flag(checkout, "--skip-worktree", "skipped.py")
    (checkout / "a.py").write_text("a = 3\n", encoding="utf-8")
    before = git(checkout, "ls-files", "-v", "-s")
    staged = git(checkout, "diff", "--cached", "--name-only")
    current_tree(checkout, untracked_max_bytes=None)
    assert git(checkout, "ls-files", "-v", "-s") == before
    assert git(checkout, "diff", "--cached", "--name-only") == staged


def test_an_oversized_untracked_file_is_named_and_left_out(checkout: Path) -> None:
    (checkout / "big.bin").write_bytes(b"x" * 64)
    (checkout / "small.txt").write_text("s\n", encoding="utf-8")
    tree, skipped = current_tree(checkout, untracked_max_bytes=16)
    assert skipped == ("big.bin",)
    (checkout / "big.bin").unlink()
    assert tree == _empty_index_tree(checkout)


def test_a_sparse_checkouts_absent_file_keeps_its_content(checkout: Path) -> None:
    """The one answer that moves: a skip-worktree file the checkout leaves absent is
    held at its index content, where an empty index read it as deleted, so a snapshot compared
    with a commit no longer names it a deletion."""
    _flag(checkout, "--skip-worktree", "skipped.py")
    (checkout / "skipped.py").unlink()
    tree, _ = current_tree(checkout, untracked_max_bytes=None)
    assert git(checkout, "rev-parse", f"{tree}:skipped.py") == git(
        checkout, "rev-parse", "HEAD:skipped.py"
    )
    assert tree == git(checkout, "rev-parse", "HEAD^{tree}")


@pytest.mark.writes_worktree("a.py")
def test_a_same_size_edit_in_the_index_s_second_is_seen_under_minimal_stat(
    checkout: Path,
) -> None:
    """Racy git: under `core.checkStat=minimal` an edit of the same size within the second
    the index was written matches its stat, and only the index's own mtime tells git to read
    the file again. The copy keeps that mtime; a copy stamped now hid the edit."""
    git(checkout, "config", "core.checkStat", "minimal")
    git(checkout, "config", "core.trustctime", "false")
    path = checkout / "a.py"
    # An entry recorded clean, not smudged: its file's mtime well before the index's write.
    old = path.stat().st_mtime_ns - 10 * 10**9
    os.utime(path, ns=(old, old))
    git(checkout, "update-index", "--refresh")
    # The same size, the same mtime: the stat matches the entry. Only the index's mtime, no
    # later than the entry's, marks the entry racy, so git reads the file again.
    path.write_text("a = 9\n", encoding="utf-8")
    os.utime(path, ns=(old, old))
    index = Path(git(checkout, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    os.utime(index, ns=(old, old))
    tree, _ = current_tree(checkout, untracked_max_bytes=None)
    assert tree == _empty_index_tree(checkout)
    assert git(checkout, "cat-file", "-p", f"{tree}:a.py") == "a = 9"


def test_untracked_caches_and_environments_are_left_out(checkout: Path) -> None:
    """T-3.159 (a REPL run): `python -m py_compile` wrote `__pycache__/*.pyc`, and the task's
    facts counted it as a change. An untracked file under a cache or environment directory is
    in no snapshot; a tracked one there is, and `dist` and `build` are never skipped."""
    for name in (
        "__pycache__/a.cpython-312.pyc",
        "pkg/__pycache__/deep.cpython-312.pyc",
        ".venv/lib/site.py",
        "node_modules/x/index.js",
        ".pytest_cache/v",
    ):
        (checkout / name).parent.mkdir(parents=True, exist_ok=True)
        (checkout / name).write_text("cache\n", encoding="utf-8")
    (checkout / "build").mkdir()
    (checkout / "build" / "gen.py").write_text("g = 1\n", encoding="utf-8")
    (checkout / "new.py").write_text("n = 1\n", encoding="utf-8")
    tree, skipped = current_tree(checkout, untracked_max_bytes=None)
    held = set(git(checkout, "ls-tree", "-r", "--name-only", tree).splitlines())
    assert {"new.py", "build/gen.py", "pkg/deep.py"} <= held
    assert not {
        p for p in held if "__pycache__" in p or p.startswith((".venv", "node_modules"))
    }
    assert ".pytest_cache/v" not in held and skipped == ()


def test_a_tracked_file_under_a_skipped_directory_is_kept(checkout: Path) -> None:
    vendored = checkout / "node_modules" / "kept.js"
    vendored.parent.mkdir()
    vendored.write_text("v = 1\n", encoding="utf-8")
    git(checkout, "add", "-f", "node_modules/kept.js")
    git(checkout, "commit", "-qm", "vendor")
    vendored.write_text("v = 2\n", encoding="utf-8")
    (checkout / "node_modules" / "untracked.js").write_text("u\n", encoding="utf-8")
    tree, _ = current_tree(checkout, untracked_max_bytes=None)
    assert git(checkout, "show", f"{tree}:node_modules/kept.js") == "v = 2"
    held = git(checkout, "ls-tree", "-r", "--name-only", tree).splitlines()
    assert "node_modules/untracked.js" not in held


def test_an_oversized_cache_file_is_named_once_and_left_out(checkout: Path) -> None:
    big = checkout / ".venv" / "big.bin"
    big.parent.mkdir()
    big.write_bytes(b"x" * 2048)
    tree, skipped = current_tree(checkout, untracked_max_bytes=1024)
    assert skipped == (".venv/big.bin",)
    assert ".venv/big.bin" not in git(checkout, "ls-tree", "-r", "--name-only", tree)


def test_a_left_out_name_is_never_read_as_a_pattern(checkout: Path) -> None:
    """Review: `node_modules/pkg/[id].js` as a pathspec also matched `i.js` and `d.js`, so a
    tracked file beside it fell out of the tree."""
    pkg = checkout / "node_modules" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "i.js").write_text("tracked\n", encoding="utf-8")
    git(checkout, "add", "-f", "node_modules/pkg/i.js")
    git(checkout, "commit", "-qm", "vendor")
    (pkg / "[id].js").write_text("untracked\n", encoding="utf-8")
    tree, _ = current_tree(checkout, untracked_max_bytes=None)
    held = git(checkout, "ls-tree", "-r", "--name-only", tree).splitlines()
    assert "node_modules/pkg/i.js" in held and "node_modules/pkg/[id].js" not in held
