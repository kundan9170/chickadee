"""The module-size guard holds each `context/` module to 600 lines, as a ratchet.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI, and
a test pinned to today's sizes would fail whenever the theme legitimately shrinks a file.
Loaded by file path, as `test_guard_runner.py` loads the runner.
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
    spec = importlib.util.spec_from_file_location("_module_size_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
module_size: Any = importlib.import_module("guards.module_size")

SRC = "packages/core/src/chickadee/context"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with `context/`'s layout and the guard pointed at it."""
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(module_size, "REPO_ROOT", root)
    return root


def _write(root: Path, rel: str, lines: int) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"x{i} = {i}\n" for i in range(lines)), encoding="utf-8")


def _found(root: Path) -> dict[str, list[int]]:
    files = sorted((root / SRC).rglob("*.py"))
    out: dict[str, list[int]] = {}
    for v in module_size.find(files):
        out.setdefault(v.path.removeprefix(f"{SRC}/"), []).append(v.line)
    return out


def _guard(baseline: dict[str, int] | None = None) -> Any:
    return dataclasses.replace(module_size.GUARD, baseline=baseline or {})


def test_a_module_at_the_budget_is_clean(tree: Path) -> None:
    _write(tree, "at.py", 600)
    assert _found(tree) == {}


def test_each_line_past_the_budget_is_one_violation(tree: Path) -> None:
    _write(tree, "over.py", 603)
    assert _found(tree) == {"over.py": [601, 602, 603]}
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{SRC}/over.py", 3, 0),)


def test_a_module_in_a_subpackage_is_counted(tree: Path) -> None:
    """`compaction/` is a package inside the corpus."""
    _write(tree, "compaction/cut.py", 601)
    assert _found(tree) == {"compaction/cut.py": [601]}


def test_a_file_over_budget_may_not_grow_and_must_record_shrinking(tree: Path) -> None:
    """The ratchet: held where it is. One line more fails; one line fewer fails too, until the
    baseline says what is true, since a loose baseline would let it grow back unseen."""
    rel = f"{SRC}/big.py"
    _write(tree, "big.py", 700)
    assert not runner.evaluate(_guard({rel: 100}), tree).failed
    _write(tree, "big.py", 701)
    assert runner.evaluate(_guard({rel: 100}), tree).over == ((rel, 101, 100),)
    _write(tree, "big.py", 699)
    assert runner.evaluate(_guard({rel: 100}), tree).slack == ((rel, 99, 100),)


def test_the_real_guard_is_discovered_with_its_budget() -> None:
    guards = {g.name: g for g in runner.discover(GUARDS)}
    assert "module_size" in guards
    assert module_size.BUDGET == 600
