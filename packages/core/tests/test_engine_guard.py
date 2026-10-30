"""The engine guard fails domain code in the facade, and a re-export that would be a cycle.


Run over fixture trees, never the live one, which is what the guard checks in CI. Loaded by
file path, as `test_guard_runner.py` loads the runner: `scripts/` is not a package core imports.
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
    spec = importlib.util.spec_from_file_location("_engine_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
guard: Any = importlib.import_module("guards.engine")

ENGINE = "packages/core/src/chickadee/engine.py"
BUDGET_INIT = "packages/core/src/chickadee/budget/__init__.py"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with an engine whose allowlist is `PacketEngine` and `__all__`, and a
    `budget/__init__.py` that imports nothing of `family`."""
    root = tmp_path.resolve()
    (root / ENGINE).parent.mkdir(parents=True)
    (root / BUDGET_INIT).parent.mkdir(parents=True)
    (root / BUDGET_INIT).write_text("from chickadee.budget.spend import Spend\n")
    monkeypatch.setattr(guard, "REPO_ROOT", root)
    monkeypatch.setattr(guard, "ALLOWED", {"PacketEngine": "the facade", "__all__": "names"})
    return root


ALLOWED_ONLY = '__all__ = ["PacketEngine"]\n\n\nclass PacketEngine:\n    pass\n'


def _engine(root: Path, extra: str = "") -> None:
    (root / ENGINE).write_text(ALLOWED_ONLY + extra, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int]]:
    files = [root / ENGINE, root / BUDGET_INIT]
    return [(v.path, v.line) for v in guard.find(files)]


def test_the_facade_and_its_allowlist_pass(tree: Path) -> None:
    _engine(tree)
    assert _found(tree) == []


@pytest.mark.parametrize(
    "extra",
    [
        "\n\ndef _spend(x):\n    return x\n",
        "\n\nclass _Projection:\n    pass\n",
        "\n\n_RATE = 0.5\n",
        "\n\nTOTAL: float = 0.0\n",
        # Wrapping a definition in a module-level block does not make it the facade's.
        "\n\nif True:\n    def _hidden():\n        pass\n",
        "\n\ntry:\n    _cache = {}\nexcept Exception:\n    pass\n",
    ],
    ids=["function", "class", "assignment", "annotated", "inside-if", "inside-try"],
)
def test_a_definition_the_allowlist_lacks_fails(tree: Path, extra: str) -> None:
    _engine(tree, extra)
    found = _found(tree)
    assert len(found) == 1 and found[0][0] == ENGINE and found[0][1] > 0


def test_a_re_export_is_not_a_definition(tree: Path) -> None:
    """How the facade keeps a moved name where hosts import it."""
    _engine(
        tree, "\nfrom chickadee.bootstrap import bootstrap_worktree as bootstrap_worktree\n"
    )
    assert _found(tree) == []


def test_a_stale_allowlist_entry_fails(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _engine(tree)
    monkeypatch.setattr(guard, "ALLOWED", {**guard.ALLOWED, "_Gone": "it moved out"})
    assert _found(tree) == [(ENGINE, 0)]


@pytest.mark.parametrize(
    "line",
    [
        "from chickadee.budget.family import SessionBudget",
        "import chickadee.budget.family",
        "from chickadee.budget import family",
        "from . import family",
        "from .family import family_budget",
    ],
)
def test_budget_init_importing_family_fails(tree: Path, line: str) -> None:
    """`context.session` imports `budget.spend`, which runs this file first,
    and `family.py` imports `context.session`."""
    _engine(tree)
    (tree / BUDGET_INIT).write_text(f"{line}\n", encoding="utf-8")
    assert _found(tree) == [(BUDGET_INIT, 1)]


def test_the_ledgers_are_the_baseline_and_a_new_helper_goes_over_it(tree: Path) -> None:
    """The ratchet: two definitions `T-3.74` removes are tolerated, a third is not."""
    _engine(tree, "\n\nclass _SessionLedger:\n    pass\n\n\nclass _RungLedger:\n    pass\n")
    real: Any = guard.GUARD
    baselined = type(real)(
        name=real.name,
        rule=real.rule,
        corpus=real.corpus,
        find=real.find,
        baseline={ENGINE: 2},
    )
    assert not runner.evaluate(baselined, tree).failed
    _engine(
        tree,
        "\n\nclass _SessionLedger:\n    pass\n\n\nclass _RungLedger:\n    pass\n"
        "\n\ndef _spend(x):\n    return x\n",
    )
    outcome = runner.evaluate(baselined, tree)
    assert outcome.failed and outcome.over == ((ENGINE, 3, 2),)


def test_the_real_guard_is_discovered() -> None:
    assert "engine" in {g.name for g in runner.discover(GUARDS)}
