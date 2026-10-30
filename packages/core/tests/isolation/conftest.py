"""Isolation fixtures. Real processes — wrap is about what the kernel actually denies."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def work(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "existing.txt").write_text("original\n", encoding="utf-8")
    return root
