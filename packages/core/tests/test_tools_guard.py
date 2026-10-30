"""The tools guard fails a second dispatcher, a built-in reaching the world, a second list of
the tools, and `tools/` importing a role; and a role's default
rung read outside its two homes (T6).

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
    spec = importlib.util.spec_from_file_location("_tools_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
tools: Any = importlib.import_module("guards.tools")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"

_SPECS = (
    "SURFACE = (\n"
    '    ToolSpec(name="read", family="read"),\n'
    '    ToolSpec(name="find", family="navigate"),\n'
    '    ToolSpec(name="shell", family="act"),\n'
    '    ToolSpec(name="ask_user", family="ask"),\n'
    ")\n"
)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC / "tools" / "builtin").mkdir(parents=True)
    (root / SRC / "tools" / "spec.py").write_text(_SPECS, encoding="utf-8")
    (root / SRC / "tools" / "builtin" / "search.py").write_text(
        'SPEC = ToolSpec(name="search", family="read")\n', encoding="utf-8"
    )
    monkeypatch.setattr(tools, "REPO_ROOT", root)
    monkeypatch.setattr(tools, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(" ", 1)[0]) for v in tools.find(files)
    ]


def test_the_tool_names_are_read_from_the_specs(tree: Path) -> None:
    assert tools.tool_names(tree) == {"read", "find", "shell", "ask_user", "search"}


def test_t1_a_name_compared_or_matched_outside_tools_fails(tree: Path) -> None:
    _write(
        tree,
        "zeus/loop.py",
        "def f(call, name, request):\n"
        '    if call.name == "read":\n'
        "        pass\n"
        '    if name in {"find", "shell"}:\n'
        "        pass\n"
        '    if request.kind != "search":\n'
        "        pass\n"
        "    match call.tool:\n"
        '        case "shell":\n'
        "            pass\n",
    )
    assert _found(tree) == [
        ("zeus/loop.py", 2, "T1"),
        ("zeus/loop.py", 4, "T1"),
        ("zeus/loop.py", 6, "T1"),
        ("zeus/loop.py", 8, "T1"),
    ]


def test_t1_passes_inside_tools_and_for_what_is_not_a_tool_s_name(tree: Path) -> None:
    _write(tree, "tools/dispatch.py", 'def f(call):\n    return call.name == "read"\n')
    _write(
        tree,
        "context/files.py",
        "def f(mode, record, name):\n"
        '    a = mode == "read"\n'  # not a name: a file's access mode
        '    b = record.kind == "prefix"\n'  # a name, but not a tool's
        '    c = name == "README"\n'
        "    return a, b, c\n",
    )
    assert _found(tree) == []


def test_t2_a_built_in_reaching_the_world_fails(tree: Path) -> None:
    _write(
        tree,
        "tools/builtin/grep.py",
        "import subprocess\n"
        "from chickadee.process.supervisor import ProcessSupervisor\n"
        "from chickadee.graph.store import GraphStore\n"
        "def run(path):\n"
        "    return open(path).read() + path.read_text()\n",
    )
    assert _found(tree) == [
        ("tools/builtin/grep.py", 1, "T2"),
        ("tools/builtin/grep.py", 2, "T2"),
        ("tools/builtin/grep.py", 3, "T2"),
        ("tools/builtin/grep.py", 5, "T2"),
        ("tools/builtin/grep.py", 5, "T2"),
    ]


def test_t2_passes_a_port_s_own_open(tree: Path) -> None:
    """A port is where a built-in is meant to reach: `Opener.open` is one (T-3.119 review)."""
    _write(
        tree,
        "tools/builtin/peek.py",
        "import os\n"
        "def run(ctx, path):\n"
        "    ctx.files.open(path)\n"
        "    return os.open(path, 0)\n",
    )
    assert _found(tree) == [("tools/builtin/peek.py", 4, "T2")]


def test_t2_passes_outside_the_built_ins(tree: Path) -> None:
    _write(tree, "context/reader.py", "def f(path):\n    return path.read_text()\n")
    assert _found(tree) == []


def test_t3_a_closed_literal_of_tool_names_outside_schema_fails(tree: Path) -> None:
    _write(
        tree,
        "zeus/families.py",
        "from typing import Literal\n"
        'Reads = Literal["read", "find"]\n'
        'Mode = Literal["read", "write"]\n'  # not every member is a tool
        'One = Literal["shell"]\n',  # one member names one tool, not a list of them
    )
    _write(
        tree, "schema/adapter.py", 'from typing import Literal\nT = Literal["read", "find"]\n'
    )
    assert _found(tree) == [("zeus/families.py", 2, "T3")]


def test_t4_tools_importing_a_role_fails(tree: Path) -> None:
    _write(
        tree,
        "tools/catalogue.py",
        "from chickadee.zeus.loop import ZeusLoop\n"
        "import chickadee.scout.tools\n"
        "from chickadee.boxctx.worker import WorkerContext\n"
        "from chickadee.schema import ToolCall\n"
        "from chickadee.tools.spec import SURFACE\n",
    )
    assert _found(tree) == [
        ("tools/catalogue.py", 1, "T4"),
        ("tools/catalogue.py", 2, "T4"),
        ("tools/catalogue.py", 3, "T4"),
    ]


def test_t1b_a_model_name_compared_outside_config_fails(tree: Path) -> None:
    """Per-model choice is configuration; code does not branch on the model."""
    _write(
        tree,
        "client/openai_client.py",
        "def f(rung, served_model, cfg, model_name):\n"
        '    if rung.model_id == "qwen2.5-coder-7b":\n'
        "        pass\n"
        '    if served_model.lower() in {"gpt-5", "o3"}:\n'
        "        pass\n"
        '    if cfg["model"] != "local":\n'
        "        pass\n"
        '    if model_name.startswith("claude"):\n'
        "        pass\n"
        "    match rung.model:\n"
        '        case "gpt-5":\n'
        "            pass\n",
    )
    _write(tree, "tools/policy.py", 'def f(model):\n    return model == "gpt-5"\n')
    assert _found(tree) == [
        ("client/openai_client.py", 2, "T1b"),
        ("client/openai_client.py", 4, "T1b"),
        ("client/openai_client.py", 6, "T1b"),
        ("client/openai_client.py", 8, "T1b"),
        ("client/openai_client.py", 10, "T1b"),
        ("tools/policy.py", 2, "T1b"),
    ]


def test_t1b_passes_config_and_words_that_only_begin_with_model(tree: Path) -> None:
    _write(tree, "config.py", 'def resolve(model_id):\n    return model_id == "gpt-5"\n')
    _write(
        tree,
        "zeus/calls.py",
        "def f(spec, payload, role, model_id):\n"
        '    a = spec.model_calls == "box"\n'  # a spec field (zeus/calls.py today), not a model
        '    b = "model" in payload\n'  # a key looked up, not a model compared
        '    c = role == "model"\n'
        "    d = model_id == other_id\n"  # two models compared, no name written in code
        "    return a, b, c, d\n",
    )
    assert _found(tree) == []


def test_t5_a_tier_made_under_tools_fails(tree: Path) -> None:
    """The router and the classifier decide a tier; a rule reads it."""
    _write(
        tree,
        "tools/catalogue.py",
        "def tier_of(spec, risky):\n"
        "    if risky:\n"
        "        return Tier.T2\n"
        '    return Tier("t1")\n',
    )
    assert _found(tree) == [
        ("tools/catalogue.py", 3, "T5"),
        ("tools/catalogue.py", 4, "T5"),
    ]


def test_t5_passes_a_tier_read_declared_or_outside_tools(tree: Path) -> None:
    _write(
        tree,
        "tools/rules.py",
        "_T0 = Tier.T0\n"  # a spec's declared tier
        "class Ask:\n"
        "    tier: Tier = Tier.T1\n"  # a default, declared
        "    def check(self, pending, floor=Tier.T0):\n"  # a parameter's default, declared
        "        tier = self.router(pending)\n"
        "        if tier is Tier.T0 or tier == Tier.T1:\n"  # the router's answer, compared
        "            return None\n"
        "        return tier\n",
    )
    _write(tree, "approval/router.py", "def tier_of(spec):\n    return Tier.T2\n")
    assert _found(tree) == []


def test_t6_a_default_rung_read_outside_its_homes_fails(tree: Path) -> None:
    """A role's rung is `EngineConfig.assignment`; the cast's default is read
    where the assignment is computed, and nowhere else."""
    _write(
        tree,
        "orchestrator/driver.py",
        "def _on(self, name):\n    return self.budget.for_rung(role(name).default_rung)\n",
    )
    _write(
        tree,
        "config.py",
        "def assign(r, roles):\n    return roles.get(r.name, r.default_rung)\n",
    )
    _write(tree, "roles.py", "def rows(r):\n    return r.default_rung\n")
    assert _found(tree) == [("orchestrator/driver.py", 2, "T6")]


def test_a_stale_exemption_fails(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "EXEMPT", {(f"{SRC}/gone.py", "f"): "was a dispatcher"})
    _write(tree, "ok.py", "x = 1\n")
    files = sorted((tree / SRC).rglob("*.py"))
    assert [v.text for v in tools.find(files)] == ["stale exemption: f"]


# The dispatchers D164 replaced, as `main` had them before `T-3.114`: each chose what a call
# does by its name, so the guard would have failed every one, and fails the next.
_OLD_DISPATCHERS = {
    "zeus/loop.py": (
        "def _dispatch(self, name, call):\n"
        '    if name == "ask_user":\n'
        "        return self._ask(call)\n"
        '    if name == "shell":\n'
        "        return self._shell(call)\n"
    ),
    "scout/tools.py": (
        "def _shared(self, name, args):\n"
        '    if name == "read":\n'
        "        return self.reader.read(args)\n"
        '    if name == "find":\n'
        "        return self.navigate.find(args)\n"
    ),
    "packer/tools.py": (
        "def run_query(request):\n"
        '    if request.kind == "read":\n'
        "        return _read(request)\n"
        '    if request.kind == "search":\n'
        "        return _search(request)\n"
    ),
}


def test_the_guard_finds_the_dispatchers_replaced(tree: Path) -> None:
    for rel, body in _OLD_DISPATCHERS.items():
        _write(tree, rel, body)
    found = _found(tree)
    assert {rel for rel, _, _ in found} == set(_OLD_DISPATCHERS)
    assert len(found) == 6 and {rule for _, _, rule in found} == {"T1"}
