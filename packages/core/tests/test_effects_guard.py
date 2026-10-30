"""The effects guard fails a second way to record an effect.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI.
Loaded by file path, as the other guard tests load the runner: `scripts/` is not a package
core imports.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chickadee.journal.effects import EffectKind

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARDS = REPO_ROOT / "scripts" / "guards"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_effects_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
effects: Any = importlib.import_module("guards.effects")
scope_of: Any = importlib.import_module("guards._scope")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with core's layout, no exemptions, and the guard pointed at it."""
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(effects, "REPO_ROOT", root)
    monkeypatch.setattr(effects, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0])
        for v in effects.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(effects.GUARD, find=effects.find, baseline={})


# -- E1 --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        'EffectIntendedP(operation_id="x", effect_kind="k", approved_sha="", preconditions=(), '
        "idempotent=True, declared_surface=())",
        'EffectStartedP(operation_id="x")',
        'journal.EffectCompletedP(operation_id="x", outcome="failed", observed_surface=())',
    ],
)
def test_building_an_effect_payload_outside_the_recorder_fails(tree: Path, call: str) -> None:
    _write(tree, "orchestrator/driver.py", f"def write(journal):\n    return {call}\n")
    assert _found(tree) == [("orchestrator/driver.py", 2, "E1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}orchestrator/driver.py", 1, 0),)


def test_reading_a_payload_back_is_not_e1(tree: Path) -> None:
    source = (
        "def closed(run, p: EffectCompletedP) -> bool:\n"
        "    lines = run.lines(EffectIntendedP)\n"
        "    return isinstance(p, EffectCompletedP) and bool(lines)\n"
    )
    _write(tree, "context/standing.py", source)
    assert _found(tree) == []


def test_the_recorder_builds_payloads_and_spells_kinds(tree: Path) -> None:
    source = (
        'KIND = "patch_apply"\n'
        "def perform(log):\n"
        '    log.record(EventKind.EFFECT_STARTED, EffectStartedP(operation_id="x"))\n'
    )
    _write(tree, "journal/effects.py", source)
    assert _found(tree) == []


# -- E2 --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase",
    [
        "EventKind.EFFECT_INTENDED",
        "schema.EventKind.EFFECT_STARTED",
        "EventKind.EFFECT_COMPLETED",
    ],
)
def test_naming_an_effect_phase_outside_the_recorder_fails(tree: Path, phase: str) -> None:
    _write(tree, "scout/sink.py", f"DURABLE = frozenset({{{phase}}})\n")
    assert _found(tree) == [("scout/sink.py", 1, "E2")]


def test_other_event_kinds_are_not_e2(tree: Path) -> None:
    _write(
        tree, "scout/sink.py", "KINDS = frozenset({EventKind.EXEC, EventKind.SCOUT_START})\n"
    )
    assert _found(tree) == []


def test_the_schema_defines_the_phases(tree: Path) -> None:
    source = (
        'class EventKind:\n    SCOUT_ESCAPE = "scout_escape"\nX = EventKind.EFFECT_STARTED\n'
    )
    _write(tree, "schema/journal.py", source)
    assert _found(tree) == []


# -- E3 --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(e.value for e in EffectKind))
def test_an_effect_kind_spelled_as_a_string_fails(tree: Path, kind: str) -> None:
    _write(tree, "run_header.py", f"def booted(op):\n    return op.kind == {kind!r}\n")
    assert _found(tree) == [("run_header.py", 2, "E3")]


def test_a_docstring_naming_a_kind_is_not_e3(tree: Path) -> None:
    source = 'def f():\n    """bootstrap"""\n    return 1\n'
    _write(tree, "bootstrap.py", source)
    assert _found(tree) == []


def test_the_guards_kinds_are_the_enums() -> None:
    # A guard imports only the standard library, so it restates the values; this is what
    # keeps the restatement true.
    assert frozenset(e.value for e in EffectKind) == effects.KINDS


# -- exemptions, baseline, the live tree -----------------------------------------------------


def test_exemptions_are_by_scope_and_only_for_e3(tree: Path, monkeypatch: Any) -> None:
    rel = f"{CORE}profiles/loader.py"
    monkeypatch.setattr(effects, "EXEMPT", {(rel, "executable_overrides"): "a TOML key"})
    source = (
        "def executable_overrides(t):\n"
        '    return "bootstrap" in t\n'
        "def other(t):\n"
        '    return "bootstrap" in t\n'
        "def executable_overrides_too(t):\n"
        '    return EffectStartedP(operation_id="x")\n'
    )
    _write(tree, "profiles/loader.py", source)
    assert _found(tree) == [("profiles/loader.py", 4, "E3"), ("profiles/loader.py", 6, "E1")]


def test_an_exemption_that_exempts_nothing_fails(tree: Path, monkeypatch: Any) -> None:
    rel = f"{CORE}run_header.py"
    monkeypatch.setattr(effects, "EXEMPT", {(rel, "_bootstrap"): "a header kind"})
    _write(tree, "run_header.py", "def _bootstrap(facts):\n    return []\n")
    assert _found(tree) == [("run_header.py", 0, "stale exemption")]


def test_a_count_within_its_baseline_passes(tree: Path) -> None:
    _write(tree, "bootstrap.py", 'def run(j):\n    return EffectStartedP(operation_id="x")\n')
    guard = dataclasses.replace(_guard(), baseline={f"{CORE}bootstrap.py": 1})
    assert not runner.evaluate(guard, tree).failed


def test_the_live_baseline_is_empty() -> None:
    # T-3.66 moved every caller in the task that landed the guard.
    assert effects.BASELINE == {}


def test_every_live_exemption_names_a_scope_that_exists() -> None:
    for (rel, scope), reason in effects.EXEMPT.items():
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"))
        scopes = {name for name, _ in scope_of.functions(tree)} | {scope_of.MODULE}
        assert scope in scopes, f"{rel}: no scope {scope}"
        assert reason
