"""The file-access guard fails a second read rule (F1) and a second containment answer
(F2): D154 decision 4.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI, and
a test pinned to today's count would fail whenever a theme legitimately moves it. Loaded by
file path, as `test_guard_runner.py` loads the runner: `scripts/` is not a package core
imports.
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
    spec = importlib.util.spec_from_file_location("_file_access_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
file_access: Any = importlib.import_module("guards.file_access")

SRC = "packages/core/src/chickadee"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with core's layout and the guard pointed at it."""
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(file_access, "REPO_ROOT", root)
    return root


def _write(root: Path, rel: str, source: str) -> Path:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _found(root: Path) -> list[tuple[str, int]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [(v.path.removeprefix(f"{SRC}/"), v.line) for v in file_access.find(files)]


def test_a_new_containment_check_outside_paths_fails(tree: Path) -> None:
    _write(tree, "scout/hints.py", "def f(p, root):\n    return p.is_relative_to(root)\n")
    assert _found(tree) == [("scout/hints.py", 2)]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{SRC}/scout/hints.py", 1, 0),)


def test_the_home_is_not_counted(tree: Path) -> None:
    _write(tree, "paths.py", "def within(root, p):\n    return p.is_relative_to(root)\n")
    assert _found(tree) == []


def test_a_new_deny_list_question_outside_secrets_fails(tree: Path) -> None:
    _write(tree, "scout/hints.py", "def f(denylist, p):\n    return denylist.readable(p)\n")
    assert _found(tree) == [("scout/hints.py", 2)]


def test_each_rule_has_its_own_home(tree: Path) -> None:
    """`secrets/` asks the list (the rule, `resolve`, `write_denied`); `paths.py` answers
    containment. Neither home excuses the other rule."""
    _write(tree, "secrets/denylist.py", "def f(denylist, p):\n    return denylist.match(p)\n")
    _write(tree, "paths.py", "def g(denylist, p):\n    return denylist.match(p)\n")
    _write(tree, "secrets/walk.py", "def h(p, root):\n    return p.is_relative_to(root)\n")
    assert _found(tree) == [("paths.py", 2), ("secrets/walk.py", 2)]


@pytest.mark.parametrize(
    ("source", "counted"),
    [
        ("p.is_relative_to(root)", 1),
        ("ok = not p.resolve().is_relative_to(root.resolve())", 1),
        ("ok = root in p.parents", 1),
        ("ok = root not in p.parents", 1),
        ("ok = p == root or root in p.resolve().parents", 1),
        ("ok = 0 < n and root in p.parents", 1),
        ("import os\nok = str(p).startswith(str(root) + os.sep)", 1),
        ("import os\nok = s.startswith(root + os.path.sep)", 1),
        ('import os\nok = s.startswith(f"{root}{os.sep}")', 1),
        ("import os as o\nok = s.startswith(root + o.sep)", 1),
        ("from os import sep\nok = s.startswith(root + sep)", 1),
        # A walk over the parents and other look-alikes are not membership tests.
        ("for parent in p.parents:\n    pass", 0),
        ("names = [q.name for q in p.parents]", 0),
        ("ok = root == p.parent", 0),
        ("ok = 'x' in p.parts", 0),
        ("ok = s.startswith('#')", 0),
        ("ok = s.startswith(prefix + '/')", 0),
        ("import os\nok = s.endswith(root + os.sep)", 0),
        ("rel = p.relative_to(root)", 0),
        # F1: a deny list asked directly.
        ("ok = self.denylist.readable(p)", 1),
        ("d = deny.match(p)", 1),
        ("d = listed.match(p, is_dir=True)", 1),
        ("d = Denylist().match(p)", 1),
        ("d = Denylist.configure(add=('x',)).readable(p)", 1),
        ("d = secrets.Denylist.configure().match(p)", 1),
        ("d = config.configure().match(p)", 0),
        ("d = (denylist if denylist is not None else Denylist()).match(p)", 1),
        ("d = (denylist or Denylist()).readable(p)", 1),
        ("d = self._deny.match(p)", 1),
        # A regex, a glob or the rule itself is not the list asked directly.
        ("import re\nm = re.match('a', s)", 0),
        ("m = pattern.match(s)", 0),
        ("import fnmatch\nok = fnmatch.fnmatch(p, '*.pem')", 0),
        ("d = name_denied(p, denylist)", 0),
        ("d = read_denied(p, at, denylist, workspace=root)", 0),
    ],
)
def test_what_counts_as_a_site(tree: Path, source: str, counted: int) -> None:
    _write(tree, "anywhere.py", source + "\n")
    assert len(_found(tree)) == counted


def test_a_file_below_its_baseline_fails_until_the_baseline_is_lowered(tree: Path) -> None:
    """The ratchet: a site removed leaves the recorded count loose, and a loose count would
    let the next copy back in unseen. So it fails until the baseline says what is true."""
    _write(tree, "scout/hints.py", "ok = root in p.parents\n")
    rel = f"{SRC}/scout/hints.py"
    outcome = runner.evaluate(_guard(baseline={rel: 2}), tree)
    assert outcome.failed and outcome.slack == ((rel, 1, 2),)
    assert not runner.evaluate(_guard(baseline={rel: 1}), tree).failed


def test_the_real_guard_is_well_formed() -> None:
    """Discovered by the real runner, holding no tolerated debt."""
    guards = {g.name: g for g in runner.discover(GUARDS)}
    assert "file_access" in guards
    assert dict(guards["file_access"].baseline) == {}


def _guard(baseline: dict[str, int] | None = None) -> Any:
    """The real guard with `baseline`. It reads `file_access.REPO_ROOT`, which `tree` points
    at the fixture."""
    return dataclasses.replace(file_access.GUARD, baseline=baseline or {})
