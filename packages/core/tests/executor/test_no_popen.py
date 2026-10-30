"""Executor is policy only: no spawn, no wrap package."""

from __future__ import annotations

from pathlib import Path

from chickadee.executor import executor as executor_mod


def test_executor_policy_modules_do_not_popen() -> None:
    root = Path(executor_mod.__file__).parent
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "Popen" not in text, path.name
        assert "subprocess" not in text, path.name


def test_executor_package_has_no_sandbox_or_supervisor() -> None:
    root = Path(executor_mod.__file__).parent
    assert not (root / "supervisor.py").exists()
    assert not (root / "sandbox").exists()
