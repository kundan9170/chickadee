"""Index cap keeps git-recent files, not whichever path sorts first."""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import git, init_repo

from chickadee.graph import indexer as indexer_mod
from chickadee.graph.indexer import Indexer


def test_over_cap_keeps_the_newer_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "r"
    root.mkdir()
    init_repo(root, commit=None)
    (root / "aaa.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    git(root, "add", "aaa.py")
    git(root, "commit", "-m", "old")
    (root / "zzz.py").write_text("def new():\n    return 1\n", encoding="utf-8")
    git(root, "add", "zzz.py")
    git(root, "commit", "-m", "new")

    monkeypatch.setattr(indexer_mod, "MAX_FILES", 1)
    ix = Indexer(root)
    report = ix.full_index()
    assert report.truncated
    paths = {r[0] for r in ix.conn.execute("SELECT path FROM files")}
    assert paths == {"zzz.py"}, "alphabetical cap would have kept aaa.py"


def test_over_cap_does_not_spend_slots_on_unprofiled_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untracked files all rank equally; path-sort would keep `aaa.md` and drop source."""
    root = tmp_path / "r"
    root.mkdir()
    for i in range(3):
        (root / f"aaa{i}.md").write_text("# notes\n", encoding="utf-8")
    (root / "zzz.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
    (root / "yyy.py").write_text("def also():\n    return 1\n", encoding="utf-8")

    monkeypatch.setattr(indexer_mod, "MAX_FILES", 1)
    ix = Indexer(root)
    report = ix.full_index()
    assert report.truncated
    paths = {r[0] for r in ix.conn.execute("SELECT path FROM files")}
    assert paths <= {"yyy.py", "zzz.py"}
    assert len(paths) == 1
    assert not any(p.endswith(".md") for p in paths)


def test_over_cap_does_not_prefer_files_outside_the_recency_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git log -n 200` omits older tracked files; those must rank oldest, not -1."""
    root = tmp_path / "r"
    root.mkdir()
    init_repo(root, commit=None)
    (root / "aaa.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    git(root, "add", "aaa.py")
    git(root, "commit", "-m", "old")
    (root / "zzz.py").write_text("def new():\n    return 1\n", encoding="utf-8")
    git(root, "add", "zzz.py")
    git(root, "commit", "-m", "new")

    monkeypatch.setattr(indexer_mod, "git_recency", lambda _repo: {"zzz.py": 0})
    monkeypatch.setattr(indexer_mod, "MAX_FILES", 1)
    ix = Indexer(root)
    report = ix.full_index()
    assert report.truncated
    paths = {r[0] for r in ix.conn.execute("SELECT path FROM files")}
    assert paths == {"zzz.py"}


def test_over_cap_keeps_a_staged_edit_outside_the_recency_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ls-files -m` misses staged-only edits; those are still the working set."""
    root = tmp_path / "r"
    root.mkdir()
    init_repo(root, commit=None)
    (root / "aaa.py").write_text("def old():\n    return 1\n", encoding="utf-8")
    git(root, "add", "aaa.py")
    git(root, "commit", "-m", "old")
    (root / "zzz.py").write_text("def new():\n    return 1\n", encoding="utf-8")
    git(root, "add", "zzz.py")
    git(root, "commit", "-m", "new")
    (root / "aaa.py").write_text("def old():\n    return 2\n", encoding="utf-8")
    git(root, "add", "aaa.py")

    monkeypatch.setattr(indexer_mod, "git_recency", lambda _repo: {"zzz.py": 0})
    monkeypatch.setattr(indexer_mod, "MAX_FILES", 1)
    ix = Indexer(root)
    report = ix.full_index()
    assert report.truncated
    paths = {r[0] for r in ix.conn.execute("SELECT path FROM files")}
    assert paths == {"aaa.py"}
