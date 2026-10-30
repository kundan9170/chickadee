"""Executor fixtures. Real processes and a real filesystem — the whole component is about
what actually happens when something runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.executor.executor import Executor
from chickadee.isolation.none import Passthrough


@pytest.fixture
def work(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "existing.txt").write_text("original\n", encoding="utf-8")
    return root


@pytest.fixture
def unsandboxed() -> Executor:
    """Most behaviour under test is policy and lifecycle, not isolation. Pinning the
    passthrough keeps those tests from passing or failing on which backend a machine has."""
    return Executor(backend=Passthrough())
