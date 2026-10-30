"""The storage guard fails a second copy of the durable primitives.

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
    spec = importlib.util.spec_from_file_location("_storage_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
storage: Any = importlib.import_module("guards.storage")

SRC = "packages/core/src/chickadee"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with core's layout, no exemptions, and the guard pointed at it."""
    root = tmp_path.resolve()
    (root / SRC / "durable").mkdir(parents=True)
    monkeypatch.setattr(storage, "REPO_ROOT", root)
    monkeypatch.setattr(storage, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> Path:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _found(root: Path) -> list[tuple[str, int]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [(v.path.removeprefix(f"{SRC}/"), v.line) for v in storage.find(files)]


def test_a_new_fsync_outside_durable_fails(tree: Path) -> None:
    _write(tree, "journal/writer.py", "import os\n\nos.fsync(3)\n")
    assert _found(tree) == [("journal/writer.py", 3)]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == (
        ("packages/core/src/chickadee/journal/writer.py", 1, 0),
    )


def test_the_home_is_not_counted(tree: Path) -> None:
    _write(tree, "durable/files.py", "import os\n\nos.fsync(3)\nos.replace('a', 'b')\n")
    assert _found(tree) == []


@pytest.mark.parametrize(
    ("source", "counted"),
    [
        ("import os as fs\nfs.fsync(3)", 1),
        ("from os import replace as move\nmove('a', 'b')", 1),
        ("import os\nfs = os\nfs.rename('a', 'b')", 1),
        ("import fcntl\nfcntl.flock(3, fcntl.LOCK_EX)", 1),
        ("import tempfile\ntempfile.mkstemp()", 1),
        ("import os\nflags = os.O_WRONLY | os.O_APPEND", 1),
        ("from os import O_EXCL\nflags = O_EXCL", 1),
        ("import pathlib\npathlib.Path('p').write_text('x')", 1),
        ("import pathlib\npathlib.Path('a').replace('b')", 1),
        ("import pathlib\npathlib.Path('p').open('a')", 1),
        ("open('p', 'w')", 1),
        ("open('p', mode='x')", 1),
        # Reads and look-alikes stay invisible, or the guard reports the whole repository.
        ("text = 'a'.replace('a', 'b')", 0),
        ("import datetime\ndatetime.datetime.now().replace(microsecond=0)", 0),
        ("open('p')", 0),
        ("open('p', 'r', -1, 'ascii')", 0),
        ("archive = object()\narchive.open('data.txt')", 0),
        ("import os\nos.open('p', os.O_RDONLY)", 0),
        # One call is one site, however many rules it matches.
        ("import os\nos.rename('a', 'b')", 1),
    ],
)
def test_what_counts_as_a_site(tree: Path, source: str, counted: int) -> None:
    _write(tree, "anywhere.py", source + "\n")
    assert len(_found(tree)) == counted


def test_a_file_below_its_baseline_fails_until_the_baseline_is_lowered(tree: Path) -> None:
    """The ratchet: a site removed leaves the recorded count loose, and a loose count would
    let the next copy back in unseen. So it fails until the baseline says what is true."""
    _write(tree, "scout/hints.py", "import os\n\nos.fsync(3)\n")
    rel = f"{SRC}/scout/hints.py"
    outcome = runner.evaluate(_guard(baseline={rel: 2}), tree)
    assert outcome.failed and outcome.slack == ((rel, 1, 2),)
    assert not runner.evaluate(_guard(baseline={rel: 1}), tree).failed


def test_an_exemption_is_not_scanned_but_a_stale_one_fails(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tree, "schema/export.py", "import os\n\nos.replace('a', 'b')\n")
    _write(tree, "workspace.py", "x = 1\n")
    exempt = {f"{SRC}/schema/export.py": "frozen", f"{SRC}/workspace.py": "git's"}
    monkeypatch.setattr(storage, "EXEMPT", exempt)
    assert _found(tree) == [("workspace.py", 0)]


def test_the_real_guard_is_well_formed() -> None:
    """Discovered by the real runner, with its exemptions each naming a file that exists."""
    guards = {g.name: g for g in runner.discover(GUARDS)}
    assert "storage" in guards
    assert all((REPO_ROOT / rel).is_file() for rel in storage.EXEMPT)


def _guard(baseline: dict[str, int] | None = None) -> Any:
    """The real guard with `baseline`. It reads `storage.REPO_ROOT`, which `tree` points at
    the fixture."""
    return dataclasses.replace(storage.GUARD, baseline=baseline or {})
