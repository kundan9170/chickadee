"""Phase 2 DoD: a 50-file fixture indexes in under 2 s, with golden counts."""

from __future__ import annotations

import time
from pathlib import Path

from chickadee.graph.indexer import Indexer

MODULE = '''\
"""Module {n}."""


class Thing{n}:
    """Thing {n}."""

    def method_{n}(self, x: int) -> str:
        return helper_{n}(x)


def helper_{n}(x: int) -> str:
    """Helper {n}."""
    return str(x)
'''


def test_a_fifty_file_fixture_indexes_under_two_seconds(tmp_path: Path) -> None:
    root = tmp_path / "big"
    root.mkdir()
    for n in range(50):
        (root / f"mod_{n}.py").write_text(MODULE.format(n=n), encoding="utf-8")

    started = time.perf_counter()
    report = Indexer(root).full_index()
    elapsed = time.perf_counter() - started

    assert report.files_indexed == 50
    # 3 symbols per module: the class, its method, and the free function.
    assert report.symbols == 150
    assert report.gaps == ()
    assert not report.truncated
    assert elapsed < 2.0, f"indexing 50 files took {elapsed:.2f}s (Phase 2 DoD is < 2s)"


def test_the_report_counts_edge_classes(tmp_path: Path) -> None:
    root = tmp_path / "small"
    root.mkdir()
    (root / "a.py").write_text(MODULE.format(n=0), encoding="utf-8")
    report = Indexer(root).full_index()
    assert sum(report.edge_classes.values()) == report.edges
    assert set(report.edge_classes) <= {"resolved", "fallback", "unknown"}
