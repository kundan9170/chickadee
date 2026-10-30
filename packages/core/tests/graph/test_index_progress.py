"""How far an index pass has got (`graph.indexer.IndexProgress`): what the TUI draws
as a bar while a repository's index is built beside the person.

What these pin: a pass reports `reading` 0 of the files it will take, once per file taken (a
gap too: the file was looked at), then `linking` once every file is read and before the
edges; the total is the capped list's; an indexer with no callback reports nothing and builds
the same index.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from repos import init_repo

from chickadee.graph import indexer as indexer_mod
from chickadee.graph.indexer import Indexer, IndexProgress

FILES = {
    "a.py": "def a() -> int:\n    return 1\n",
    "b.py": "from a import a\n\n\ndef b() -> int:\n    return a()\n",
    "notes.xyz": "not a language\n",
}


def _pass(root: Path, *, progress: list[IndexProgress] | None) -> int:
    told = progress.append if progress is not None else None
    indexer = Indexer(root, progress=told)
    try:
        return indexer.full_index().edges
    finally:
        indexer.close()


def test_a_pass_reports_each_file_then_linking(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", FILES)
    seen: list[IndexProgress] = []
    _pass(root, progress=seen)
    total = seen[-1].total
    assert total == len(seen) - 1 >= 2
    assert seen[:-1] == [IndexProgress("reading", n, total) for n in range(total)]
    assert seen[-1] == IndexProgress("linking", total, total)


def test_the_total_is_the_capped_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = init_repo(tmp_path / "repo", FILES)
    monkeypatch.setattr(indexer_mod, "MAX_FILES", 1)
    seen: list[IndexProgress] = []
    _pass(root, progress=seen)
    assert seen == [IndexProgress("reading", 0, 1), IndexProgress("linking", 1, 1)]


def test_no_callback_reports_nothing_and_builds_the_same_index(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", FILES)
    assert _pass(root, progress=None) == _pass(root, progress=[])
