"""The tokens guard fails a second token estimate.

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
    spec = importlib.util.spec_from_file_location("_tokens_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
tokens: Any = importlib.import_module("guards.tokens")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(tokens, "REPO_ROOT", root)
    monkeypatch.setattr(tokens, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0]) for v in tokens.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(tokens.GUARD, find=tokens.find, baseline={})


@pytest.mark.parametrize("line", ["CHARS_PER_TOKEN = 4", "CHARS_PER_TOKEN: float = 3.5"])
def test_a_second_constant_fails(tree: Path, line: str) -> None:
    _write(tree, "packer/types.py", f"{line}\n")
    assert _found(tree) == [("packer/types.py", 1, "T1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}packer/types.py", 1, 0),)


@pytest.mark.parametrize(
    "expr", ["len(text) // 4", "len(full) / 3.5", "len(s) // 4.0", "math.ceil(len(t) / 3.5)"]
)
def test_an_inline_estimate_fails(tree: Path, expr: str) -> None:
    _write(tree, "orchestrator/machine.py", f"def f(text, full, s, t):\n    return {expr}\n")
    assert _found(tree) == [("orchestrator/machine.py", 2, "T2")]


@pytest.mark.parametrize(
    "expr", ["len(data) // 100", "count / 4", "len(a) * 4", "tokens_est(text)", "n // 3.5"]
)
def test_what_is_not_an_estimate_passes(tree: Path, expr: str) -> None:
    _write(tree, "context/outline.py", f"def f(data, count, a, text, n):\n    return {expr}\n")
    assert _found(tree) == []


def test_the_home_holds_the_constant_and_the_rule(tree: Path) -> None:
    source = (
        "CHARS_PER_TOKEN = 3.5\ndef tokens_est(text):\n    return math.ceil(len(text) / 3.5)\n"
    )
    _write(tree, "limits.py", source)
    assert _found(tree) == []


def test_an_exemption_that_exempts_nothing_fails(tree: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(tokens, "EXEMPT", {(f"{CORE}packer/tools.py", "truncate"): "a reason"})
    _write(tree, "packer/tools.py", "def truncate(text):\n    return text\n")
    assert _found(tree) == [("packer/tools.py", 0, "stale exemption")]


def test_the_live_baseline_and_exemptions_are_empty() -> None:
    assert tokens.BASELINE == {}
    assert tokens.EXEMPT == {}
