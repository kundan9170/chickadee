"""The records guard fails a record read as a dict.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI, and
a test pinned to today's count would fail whenever a theme legitimately moves it. Loaded by
file path, as `test_storage_guard.py` loads the runner: `scripts/` is not a package core
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
    spec = importlib.util.spec_from_file_location("_records_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
records: Any = importlib.import_module("guards.records")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"

SCHEMA = """
class BaseFixtureP: ...
class RanFixtureP(BaseFixtureP): ...
class SaidFixtureP(BaseFixtureP): ...

KIND_TABLE: dict[str, tuple[type[BaseFixtureP], str]] = {
    "exec": (RanFixtureP, "packet"),
    "classify": (SaidFixtureP, "packet"),
}
"""


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository root with core's layout, a two-kind journal schema, no exemptions, and
    the guard pointed at it."""
    root = tmp_path.resolve()
    schema = root / SRC / "schema" / "journal.py"
    schema.parent.mkdir(parents=True)
    schema.write_text(SCHEMA, encoding="utf-8")
    monkeypatch.setattr(records, "REPO_ROOT", root)
    monkeypatch.setattr(records, "SCHEMA", schema)
    monkeypatch.setattr(records, "R1_EXEMPT", {})
    monkeypatch.setattr(records, "R3_EXEMPT", {})
    monkeypatch.setattr(records, "R3_EXEMPT_FILES", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0])
        for v in records.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(records.GUARD, find=records.find, baseline={})


# -- R1 --------------------------------------------------------------------------------------


def test_a_model_dump_read_fails_anywhere_in_core(tree: Path) -> None:
    _write(tree, "budget/spend.py", "def spent(ev):\n    return ev.payload.model_dump()['x']\n")
    assert _found(tree) == [("budget/spend.py", 2, "R1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}budget/spend.py", 1, 0),)


def test_a_serializer_is_exempt_by_its_function(tree: Path, monkeypatch: Any) -> None:
    rel = f"{CORE}journal/writer.py"
    monkeypatch.setattr(records, "R1_EXEMPT", {(rel, "Journal.append"): "the writer"})
    source = (
        "class Journal:\n"
        "    def append(self, p):\n"
        "        return p.model_dump()\n"
        "    def read(self, p):\n"
        "        return p.model_dump()\n"
    )
    _write(tree, "journal/writer.py", source)
    assert _found(tree) == [("journal/writer.py", 5, "R1")]


def test_model_dump_json_is_not_a_read(tree: Path) -> None:
    _write(tree, "packer/pack.py", "def sha(p):\n    return p.model_dump_json()\n")
    assert _found(tree) == []


# -- R2 --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cast", ['cast("RanFixtureP", e.payload)', "cast(RanFixtureP, e.payload)"]
)
def test_a_cast_to_a_payload_class_fails(tree: Path, cast: str) -> None:
    _write(tree, "boxctx/records.py", f"def code(e):\n    return {cast}.exit_code\n")
    assert _found(tree) == [("boxctx/records.py", 2, "R2")]


def test_a_cast_to_any_other_class_is_not_r2(tree: Path) -> None:
    _write(tree, "graph/store.py", 'def rows(x):\n    return cast("Row", x)\n')
    assert _found(tree) == []


# -- R3 --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    ["Payload", "Mapping[str, object]", "Mapping[str, Any]", "dict[str, Any]", '"Payload"'],
)
def test_a_key_read_off_a_decoded_object_fails_in_the_corpus(
    tree: Path, annotation: str
) -> None:
    source = f"def why(p: {annotation}) -> object:\n    return p['reason'] or p.get('detail')\n"
    _write(tree, "journal/metrics.py", source)
    assert _found(tree) == [("journal/metrics.py", 2, "R3"), ("journal/metrics.py", 2, "R3")]


def test_a_cast_to_a_decoded_object_fails_in_the_corpus(tree: Path) -> None:
    _write(
        tree, "context/session.py", 'def read(x):\n    return cast("dict[str, object]", x)\n'
    )
    assert _found(tree) == [("context/session.py", 2, "R3")]


def test_r3_reads_only_the_records_corpus(tree: Path) -> None:
    source = "def why(p: Payload) -> object:\n    return p['reason']\n"
    _write(tree, "eval_helpers/rows.py", source)
    assert _found(tree) == []


def test_a_typed_parameter_is_not_a_decoded_object(tree: Path) -> None:
    source = (
        "def why(p: RanFixtureP, q: Mapping[str, int]) -> object:\n    return q['n'], p.x\n"
    )
    _write(tree, "journal/metrics.py", source)
    assert _found(tree) == []


def test_r3_exemptions_by_function_and_by_file(tree: Path, monkeypatch: Any) -> None:
    run, validate = f"{CORE}zeus/loop.py", f"{CORE}tools/validate.py"
    monkeypatch.setattr(records, "R3_EXEMPT", {(run, "_digest"): "as sent"})
    monkeypatch.setattr(records, "R3_EXEMPT_FILES", {validate: "schema walkers"})
    _write(
        tree, "zeus/loop.py", "def _digest(own: Mapping[str, object]):\n    return own['x']\n"
    )
    _write(
        tree,
        "tools/validate.py",
        "def walk(s: Mapping[str, object]):\n    return s['type']\n",
    )
    assert _found(tree) == []


# -- exemptions --------------------------------------------------------------------------------


def test_an_exemption_that_exempts_nothing_fails(tree: Path, monkeypatch: Any) -> None:
    gone = f"{CORE}orchestrator/action.py"
    monkeypatch.setattr(records, "R1_EXEMPT", {(gone, "effect_sha"): "the effect sha"})
    monkeypatch.setattr(records, "R3_EXEMPT_FILES", {f"{CORE}zeus/old.py": "gone"})
    _write(tree, "orchestrator/action.py", "def effect_sha(spec):\n    return 1\n")
    assert _found(tree) == [
        ("orchestrator/action.py", 0, "stale exemption"),
        ("zeus/old.py", 0, "stale exemption"),
    ]


def test_the_payload_classes_are_read_from_the_kind_table(tree: Path) -> None:
    assert records.payload_classes(records.SCHEMA) == {"RanFixtureP", "SaidFixtureP"}


def test_the_live_kind_table_is_read() -> None:
    """Against the real schema: the guard knows every payload class, not a stale list."""
    from chickadee.schema.journal import KIND_TABLE

    schema = REPO_ROOT / SRC / "schema" / "journal.py"
    assert records.payload_classes(schema) == {m.__name__ for m, _ in KIND_TABLE.values()}
