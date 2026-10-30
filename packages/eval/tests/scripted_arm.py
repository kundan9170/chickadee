"""The Chickadee arm over a real engine with scripted models, for the suites that drive it
(`test_chickadee_arm.py`, `test_cli.py`): the engine is real (Zeus, the box, the leased
worktree, the handoff); only the models are scripted and the sandbox passed through, as the core
suites that delegate run it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from engines import answer, call, calls, config_for, raw_config, scripted

import chickadee.engine as engine_module
from chickadee.engine import PacketEngine
from chickadee.isolation.none import Passthrough
from chickadee.schema import ToolCall
from chickadee.state.roots import roots as resolve_roots
from chickadee_eval.manifest import Language
from chickadee_eval.problems import polyglot

STUB = "def f(x):\n    raise NotImplementedError\n"
TEST = (
    "from lib import f\n\n\n"
    "def test_f_doubles_its_argument():\n"
    '    assert f(2) == 4, "doubling two has to give four"\n'
)
FAILURE = (
    "    def test_f_doubles_its_argument():\n"
    '>       assert f(2) == 4, "doubling two has to give four"\n'
    "E       NotImplementedError\n"
)


def write_problem(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "instructions.md").write_text("Double it.\n", encoding="utf-8")
    (d / "lib.py").write_text(STUB, encoding="utf-8")
    (d / "lib_test.py").write_text(TEST, encoding="utf-8")


def make_problem(root: Path, pid: str = "p") -> polyglot.LoadedProblem:
    write_problem(root / pid)
    return polyglot.load(root, pid, Language.PYTHON)


def workspace(tmp_path: Path) -> Path:
    """The problem as `fresh_worktree` gives it: the stub, no tests."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "instructions.md").write_text("Double it.\n", encoding="utf-8")
    (ws / "lib.py").write_text(STUB, encoding="utf-8")
    return ws


def box_call(tool: str, cid: str, /, **args: object) -> ToolCall:
    return ToolCall(id=cid, name=tool, arguments=dict(args))


def opener(
    monkeypatch: pytest.MonkeyPatch, script: list[Any]
) -> Callable[[Path], PacketEngine]:
    """What `open_host_engine` builds, with the scripted model behind every role."""
    scripted(monkeypatch, script)
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    raw = raw_config()
    raw["sandbox"] = {"require_sandbox": False}
    raw["prices"] = {
        raw["rungs"]["high-param"]: {"price_in": 1.0, "price_out": 2.0},
        raw["rungs"]["low-param"]: {"price_in": 0.5, "price_out": 1.0},
    }
    config = config_for(raw, priced=False)

    def open_engine(ws: Path) -> PacketEngine:
        roots = resolve_roots(ws)
        return PacketEngine(ws, config.engine, trust=config.trust(roots.repo_local, root=ws))

    return open_engine


def doubled(_problem: polyglot.LoadedProblem, ws: Path) -> tuple[bool, str]:
    """The tests' verdict, without running them: whether `f` doubles."""
    passed = "return x * 2" in (ws / "lib.py").read_text(encoding="utf-8")
    return passed, "" if passed else FAILURE


def delegates_and_doubles() -> list[Any]:
    """Zeus's one-task `coder` call, the worker's edit that makes `f` double, Athena's accept,
    and Zeus's reply."""
    work = {"title": "Double it", "brief": "## Goal\n`f` doubles.\n", "workspace": "worktree"}
    return [
        calls(call("coder", "c1", **work)),
        calls(
            box_call(
                "edit", "w1", path="lib.py", old="raise NotImplementedError", new="return x * 2"
            )
        ),
        answer("T1: edited lib.py."),
        calls(box_call("review", "a1", task="T1", answer="accept")),
        answer("T1 accepted: f doubles."),
        answer("done"),
    ]
