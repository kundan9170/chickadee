"""The limits guard fails a second lookup of a resolved limit (T-3.85's review).

Run over fixture trees, never the live one: the live tree is what the guard checks in CI.
Loaded by file path, as the other guard tests load the runner.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDS = REPO_ROOT / "scripts" / "guards"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_limits_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
limits: Any = importlib.import_module("guards.limits")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(limits, "REPO_ROOT", root)
    monkeypatch.setattr(limits, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0]) for v in limits.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(limits.GUARD, find=limits.find, baseline={})


@pytest.mark.parametrize(
    "body",
    [
        "{r.name: r.tokens_or_value for r in resolved if r.role == 'zeus'}",
        "next(r.tokens_or_value for r in resolved if r.name == 'zeus.repeat')",
    ],
)
def test_a_row_read_in_place_fails(tree: Path, body: str) -> None:
    _write(tree, "zeus/loop.py", f"def f(resolved):\n    return {body}\n")
    assert _found(tree) == [("zeus/loop.py", 2, "L1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}zeus/loop.py", 1, 0),)


def test_a_read_through_the_home_passes(tree: Path) -> None:
    source = "def f(resolved):\n    return resolved_value(resolved, 'zeus', 'zeus.repeat')\n"
    _write(tree, "zeus/loop.py", source)
    assert _found(tree) == []


def test_the_home_reads_the_field(tree: Path) -> None:
    source = (
        "def resolved_value(resolved, role, name):\n"
        "    for rl in resolved:\n"
        "        if rl.role == role and rl.name == name:\n"
        "            return rl.tokens_or_value\n"
    )
    _write(tree, "limits.py", source)
    assert _found(tree) == []


def test_an_exemption_that_exempts_nothing_fails(tree: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(limits, "EXEMPT", {(f"{CORE}packer/types.py", "f"): "a reason"})
    _write(tree, "packer/types.py", "def f():\n    return 1\n")
    assert _found(tree) == [("packer/types.py", 0, "stale exemption")]


def test_the_live_baseline_and_exemptions_are_empty() -> None:
    assert limits.BASELINE == {}
    assert limits.EXEMPT == {}
