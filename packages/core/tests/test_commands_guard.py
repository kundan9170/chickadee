"""The commands guard fails a second command classifier.

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
    spec = importlib.util.spec_from_file_location("_commands_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
commands: Any = importlib.import_module("guards.commands")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(commands, "REPO_ROOT", root)
    monkeypatch.setattr(commands, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [(v.path.removeprefix(CORE), v.line) for v in commands.find(files)]


def test_a_command_regex_outside_the_table_fails(tree: Path) -> None:
    _write(
        tree,
        "scout/escapes.py",
        "import re\n"
        '_FETCH = re.compile(r"\\b(?:curl|wget)\\b")\n'
        "def f(line):\n"
        '    return re.search(r"\\bgit\\s+push", line)\n'
        'X = re.compile(r"\\b(rm|mv)\\b" + r"|>")\n',
    )
    assert _found(tree) == [
        ("scout/escapes.py", 2),
        ("scout/escapes.py", 4),
        ("scout/escapes.py", 5),
    ]


def test_the_table_and_other_regexes_pass(tree: Path) -> None:
    _write(tree, "approval/commands.py", 'import re\nP = re.compile(r"\\bgit\\b")\n')
    _write(
        tree,
        "shell.py",
        "import re\n"
        '_SHAPE = re.compile(r"Read-only file system|could not resolve host")\n'
        '_ID = re.compile(r"^[a-z]+-\\d+$")\n'
        '_WORD = re.compile(r"gitignore|remake|pip_cache")\n',
    )
    assert _found(tree) == []


def test_a_second_splitter_fails(tree: Path) -> None:
    _write(
        tree,
        "verifier/checkpoint.py",
        "import shlex\n"
        "argv = tuple(shlex.split(cmd))\n"
        "shown = shlex.join(argv)\n"
        "quoted = shlex.quote(cmd)\n",
    )
    _write(tree, "shell_line.py", "import shlex\nwords = shlex.split(line)\n")
    assert _found(tree) == [("verifier/checkpoint.py", 2), ("verifier/checkpoint.py", 3)]


def test_a_stale_exemption_fails(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(commands, "EXEMPT", {(f"{SRC}/gone.py", "f"): "was a copy"})
    _write(tree, "ok.py", "x = 1\n")
    files = sorted((tree / SRC).rglob("*.py"))
    assert [v.text for v in commands.find(files)] == ["stale exemption: f"]


# The classifiers `T-3.122` removed, as `main` had them: each is a C1 site, so the guard would
# have failed them, and fails the next.
_OLD_CLASSIFIERS = {
    "executor/approval.py": (
        "DESTRUCTIVE = re.compile(\n"
        '    r"\\b(rm|rmdir|shred|dd|mkfs|truncate|drop\\s+table|drop\\s+database)\\b|"\n'
        '    r"\\bgit\\s+(push|reset\\s+--hard|clean|filter-branch)\\b",\n'
        "    re.I,\n"
        ")\n"
    ),
    "executor/policy.py": (
        "_MUTATING = re.compile(\n"
        '    r"\\b(rm|mv|cp|dd|chmod|chown|truncate|tee|mkfs|shred)\\b|>>?\\s*\\S"\n'
        ")\n"
        "_EXTERNAL_FETCH = re.compile(\n"
        '    r"\\b(?:curl|wget)\\b.*?\\bhttps?://", re.IGNORECASE\n'
        ")\n"
    ),
    "scout/escapes.py": (
        '_GIT_MUTATION = re.compile(r"\\bgit\\s+(?:-C\\s+\\S+\\s+)?(?:commit|push)\\b")\n'
        '_FETCH = re.compile(r"\\b(?:curl|wget|ssh|scp|rsync|nc)\\b")\n'
        '_IN_PLACE = re.compile(r"\\b(?:sed|perl)\\s+(?:-\\w+\\s+)*-\\w*i")\n'
    ),
}


def test_the_guard_finds_the_classifiers_main_had_before_it_was_removed(tree: Path) -> None:
    for rel, body in _OLD_CLASSIFIERS.items():
        _write(tree, rel, "import re\n" + body)
    found = _found(tree)
    assert {rel for rel, _ in found} == set(_OLD_CLASSIFIERS)
    assert len(found) == 6
