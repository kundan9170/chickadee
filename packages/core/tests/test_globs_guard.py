"""The globs guard fails a second path-glob matcher.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI.
Loaded by file path, as the other guard tests load the runner.
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
    spec = importlib.util.spec_from_file_location("_globs_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
globs: Any = importlib.import_module("guards.globs")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(globs, "REPO_ROOT", root)
    monkeypatch.setattr(globs, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [(v.path.removeprefix(CORE), v.line) for v in globs.find(files)]


def test_fnmatch_outside_paths_fails(tree: Path) -> None:
    _write(
        tree,
        "executor/effects.py",
        "import fnmatch\n"
        "import glob\n"
        "from fnmatch import fnmatchcase\n"
        "def f(path, g):\n"
        "    return fnmatch.fnmatch(path, g) or glob.translate(g)\n",
    )
    assert _found(tree) == [
        ("executor/effects.py", 3),
        ("executor/effects.py", 5),
        ("executor/effects.py", 5),
    ]


def test_a_glob_translated_to_a_regex_fails_but_prose_passes(tree: Path) -> None:
    _write(
        tree,
        "scout/search.py",
        '"""`*` becomes `[^/]*`."""\n'
        "def _glob_regex(c):\n"
        '    """`?` becomes `[^/]`."""\n'
        '    return "[^/]*" if c == "*" else "x"\n'
        'ANY = "[^/]"\n'
        '"""What `?` becomes: `[^/]`."""\n',
    )
    assert _found(tree) == [("scout/search.py", 4), ("scout/search.py", 5)]


def test_the_home_passes(tree: Path) -> None:
    _write(tree, "paths.py", 'import fnmatch\nX = fnmatch.fnmatchcase("a", "*")\nY = "[^/]*"\n')
    assert _found(tree) == []


def test_a_stale_exemption_fails(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(globs, "EXEMPT", {(f"{SRC}/gone.py", "f"): "was a copy"})
    _write(tree, "ok.py", "x = 1\n")
    files = sorted((tree / SRC).rglob("*.py"))
    assert [v.text for v in globs.find(files)] == ["stale exemption: f"]


# The matchers D168 removed, as they were: each is a G1 or G2 site, so the guard would have
# failed them, and fails the next.
_OLD_MATCHERS = {
    "scout/search.py": (
        "def _glob_regex(pattern):\n"
        "    out = []\n"
        "    for c in pattern:\n"
        '        out.append("[^/]*" if c == "*" else "[^/]" if c == "?" else c)\n'
        '    return "".join(out)\n'
    ),
    "graph/ignore.py": (
        "def _glob_to_regex(pattern):\n"
        "    parts = []\n"
        "    for ch in pattern:\n"
        '        parts.append("[^/]*" if ch == "*" else ch)\n'
        '    return "".join(parts)\n'
    ),
    "executor/effects.py": (
        "import fnmatch\n"
        "def _declared(path, effects):\n"
        "    return any(fnmatch.fnmatch(path, g) for g in effects)\n"
    ),
    "state/preferences.py": (
        "import fnmatch\n"
        "def _matches(patterns, subject):\n"
        "    return any(fnmatch.fnmatch(subject, p) for p in patterns)\n"
    ),
}


def test_the_guard_finds_the_matchers_removed(tree: Path) -> None:
    for rel, body in _OLD_MATCHERS.items():
        _write(tree, rel, body)
    found = _found(tree)
    assert {rel for rel, _ in found} == set(_OLD_MATCHERS)
