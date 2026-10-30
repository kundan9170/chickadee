"""The budget guard fails a second ledger or a second ledger interface.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI, and
a test pinned to today's count would fail whenever a theme legitimately moves it. Loaded by
file path, as `test_records_guard.py` loads the runner: `scripts/` is not a package core
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
    spec = importlib.util.spec_from_file_location("_budget_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
budget: Any = importlib.import_module("guards.budget")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"

CHARGE = "    def charge_usage(self, scope_packet, usage, model):\n        return 0.0\n"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(budget, "REPO_ROOT", root)
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0]) for v in budget.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(budget.GUARD, find=budget.find, baseline={})


@pytest.mark.parametrize("base", ["CostLedger", "manager.CostLedger"])
def test_a_subclass_of_the_ledger_fails(tree: Path, base: str) -> None:
    _write(tree, "engine.py", f"class _SessionLedger({base}):\n    pass\n")
    assert _found(tree) == [("engine.py", 1, "B1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}engine.py", 1, 0),)


def test_a_subclass_in_the_ledgers_own_file_fails_too(tree: Path) -> None:
    _write(
        tree,
        "budget/manager.py",
        "class CostLedger:\n    pass\nclass Two(CostLedger):\n    pass\n",
    )
    assert _found(tree) == [("budget/manager.py", 3, "B1")]


@pytest.mark.parametrize(
    ("rel", "source"),
    [
        ("scout/loop.py", f"class Ledger(Protocol):\n{CHARGE}"),
        ("engine.py", f"class _RungLedger:\n{CHARGE}"),
        ("orchestrator/ports.py", f"class BudgetPort(Protocol):\n{CHARGE}"),
    ],
    ids=["a-second-protocol", "a-second-view", "a-port-that-charges"],
)
def test_charge_usage_defined_outside_the_ledgers_home_fails(
    tree: Path, rel: str, source: str
) -> None:
    _write(tree, rel, source)
    assert _found(tree) == [(rel, 2, "B2")]


def test_the_ledgers_home_defines_the_interface_and_its_view(tree: Path) -> None:
    source = f"class Ledger(Protocol):\n{CHARGE}class RungLedger:\n{CHARGE}"
    _write(tree, "budget/manager.py", source)
    assert _found(tree) == []


def test_calling_charge_usage_is_not_defining_it(tree: Path) -> None:
    _write(
        tree,
        "zeus/loop.py",
        "class ZeusLoop:\n    def charge(self, ledger, usage):\n"
        "        return ledger.charge_usage('', usage, 'm')\n",
    )
    assert _found(tree) == []


# -- B3, spend read once ---------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "    if isinstance(e.payload, CoderDoneP):\n        return e.payload.cost_usd\n",
        "    if isinstance(r, SpendLine | CapLine):\n        return r.cost_usd\n",
    ],
    ids=["a-coder-done", "a-spend-line"],
)
def test_a_cost_read_off_a_narrowed_record_fails(tree: Path, body: str) -> None:
    _write(tree, "context/session.py", f"def spent(e, r):\n{body}")
    assert _found(tree) == [("context/session.py", 3, "B3")]


def test_a_cost_read_off_a_record_parameter_fails(tree: Path) -> None:
    _write(tree, "orchestrator/run.py", "def cost(p: CoderDoneP):\n    return p.cost_usd\n")
    assert _found(tree) == [("orchestrator/run.py", 2, "B3")]


@pytest.mark.parametrize(
    "body",
    [
        "    return sum(c.cost_usd for c in lines)\n",
        "    total = 0.0\n    for c in lines:\n        total += c.cost_usd\n    return total\n",
    ],
    ids=["a-sum", "a-running-total"],
)
def test_a_sum_of_costs_fails_outside_the_two_readers(tree: Path, body: str) -> None:
    _write(tree, "context/session.py", f"def spent(lines):\n{body}")
    assert [rule for _p, _l, rule in _found(tree)] == ["B3"]


@pytest.mark.parametrize("home", ["budget/family.py", "journal/projection.py"])
def test_the_two_readers_read_and_sum_costs(tree: Path, home: str) -> None:
    body = (
        "def spending(lines):\n    cost = 0.0\n    for r in lines:\n"
        "        if isinstance(r, SpendLine):\n            cost += r.cost_usd\n"
        "    return cost\n"
    )
    _write(tree, home, body)
    assert _found(tree) == []


def test_reading_a_totals_cost_is_not_a_second_reader(tree: Path) -> None:
    """A `Spend`, a `RunSpend` or a `run_end` already holds a total; showing it is not
    summing records."""
    body = "def show(spend, end):\n    return spend.cost_usd, end.cost_usd\n"
    _write(tree, "context/recall.py", body)
    assert _found(tree) == []


def test_the_live_baseline_is_empty() -> None:
    # `T-3.74` deleted every second ledger and interface and `T-3.75` every
    # second spend reader, so the guard tolerates none.
    assert budget.BASELINE == {}
