"""The languages guard fails a second extension-to-language table.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI.
"""

from __future__ import annotations

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
    spec = importlib.util.spec_from_file_location("_languages_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
languages: Any = importlib.import_module("guards.languages")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(languages, "REPO_ROOT", root)
    monkeypatch.setattr(languages, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [(v.path.removeprefix(CORE), v.line) for v in languages.find(files)]


# The two tables T-3.164 removed, as they were written: the guard fails each, and the next.
_OLD_TABLES = {
    "graph/parse.py": (
        'PYTHON = "python"\n'
        'JAVASCRIPT = "javascript"\n'
        "EXTENSIONS = {\n"
        '    ".py": PYTHON,\n'
        '    ".pyi": PYTHON,\n'
        '    ".js": JAVASCRIPT,\n'
        "}\n"
    ),
    "context/outline.py": (
        "from typing import Final, Mapping\n"
        'LANGUAGES: Final[Mapping[str, str]] = {".py": "python", ".go": "go", ".rs": "rust"}\n'
    ),
}


def test_the_guard_finds_the_tables_removed(tree: Path) -> None:
    for rel, body in _OLD_TABLES.items():
        _write(tree, rel, body)
    assert _found(tree) == [("context/outline.py", 2), ("graph/parse.py", 3)]


def test_a_table_inside_a_function_fails(tree: Path) -> None:
    _write(
        tree,
        "lsp.py",
        'def language_id(ext):\n    return {".ts": "typescript", ".tsx": "x"}[ext]\n',
    )
    assert _found(tree) == [("lsp.py", 2)]


def test_what_is_not_a_language_table_passes(tree: Path) -> None:
    _write(
        tree,
        "outline.py",
        # A format table: extension to a format and its label, not a language's name.
        '_TEXT_FORMATS = {".md": ("markdown", "markdown"), ".json": ("json", "json")}\n'
        # One extension is a special case, not a table.
        'MINIFIED = {".min.js": "javascript"}\n'
        # Extension keys to numbers, and names to names.
        'LIMITS = {".py": 1, ".js": 2}\n'
        'ALIASES = {"py": "python", "js": "javascript"}\n',
    )
    assert _found(tree) == []


def test_the_home_passes(tree: Path) -> None:
    _write(tree, "profiles/model.py", 'DEFAULTS = {".py": "python", ".js": "javascript"}\n')
    assert _found(tree) == []


def test_a_stale_exemption_fails(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(languages, "EXEMPT", {(f"{SRC}/gone.py", "<module>"): "was a copy"})
    _write(tree, "ok.py", "x = 1\n")
    files = sorted((tree / SRC).rglob("*.py"))
    assert [v.text for v in languages.find(files)] == ["stale exemption: <module>"]


def test_the_runner_finds_it_with_a_zero_baseline() -> None:
    """The task's terms: no debt tolerated, from the day the guard lands."""
    (guard,) = [g for g in runner.discover(GUARDS) if g.name == "languages"]
    assert guard.baseline == {}
