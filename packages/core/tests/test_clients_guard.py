"""The clients guard fails a second client protocol.

Run over fixture trees, never the live one: the live tree is what the guard checks in CI.
Loaded by file path, as the other guard tests load the runner.
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
    spec = importlib.util.spec_from_file_location("_clients_guard_runner", GUARDS / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()
clients: Any = importlib.import_module("guards.clients")

SRC = "packages/core/src/chickadee"
CORE = f"{SRC}/"


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path.resolve()
    (root / SRC).mkdir(parents=True)
    monkeypatch.setattr(clients, "REPO_ROOT", root)
    monkeypatch.setattr(clients, "EXEMPT", {})
    return root


def _write(root: Path, rel: str, source: str) -> None:
    path = root / SRC / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _found(root: Path) -> list[tuple[str, int, str]]:
    files = sorted((root / SRC).rglob("*.py"))
    return [
        (v.path.removeprefix(CORE), v.line, v.text.split(":", 1)[0])
        for v in clients.find(files)
    ]


def _guard() -> Any:
    return dataclasses.replace(clients.GUARD, find=clients.find, baseline={})


# -- C1 --------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", ["Protocol", "typing.Protocol", "Protocol[T]"])
def test_a_second_protocol_with_send_fails(tree: Path, base: str) -> None:
    source = f"class ChatClient({base}):\n    def send(self, request): ...\n"
    _write(tree, "zeus/loop.py", source)
    assert _found(tree) == [("zeus/loop.py", 1, "C1")]
    outcome = runner.evaluate(_guard(), tree)
    assert outcome.failed and outcome.over == ((f"{CORE}zeus/loop.py", 1, 0),)


def test_the_home_declares_the_protocol(tree: Path) -> None:
    source = (
        "class ModelClient(Protocol):\n    def send(self, r): ...\n    def close(self): ...\n"
    )
    _write(tree, "client/request.py", source)
    assert _found(tree) == []


def test_a_class_that_sends_without_being_a_protocol_is_not_c1(tree: Path) -> None:
    _write(tree, "client/openai_client.py", "class OpenAIClient:\n    def send(self, r): ...\n")
    _write(tree, "context/port.py", "class Port(Protocol):\n    def read(self): ...\n")
    assert _found(tree) == []


# -- C2, C3 ----------------------------------------------------------------------------------


def test_closing_through_getattr_fails(tree: Path) -> None:
    source = 'def stop(z):\n    closer = getattr(z.client, "close", None)\n    closer()\n'
    _write(tree, "engine.py", source)
    assert _found(tree) == [("engine.py", 2, "C2")]


def test_other_getattrs_are_not_c2(tree: Path) -> None:
    _write(tree, "engine.py", 'def f(z):\n    return getattr(z, "name", None)\n')
    assert _found(tree) == []


@pytest.mark.parametrize("call", ["client.complete(p, q)", "self.client.complete(p, q)"])
def test_the_retired_complete_on_a_client_fails(tree: Path, call: str) -> None:
    _write(tree, "arm.py", f"def go(self, client, p, q):\n    return {call}\n")
    assert _found(tree) == [("arm.py", 2, "C3")]


def test_an_effects_complete_is_not_c3(tree: Path) -> None:
    _write(
        tree,
        "driver.py",
        'def go(fx, effect):\n    fx.complete("x")\n    effect.complete("y")\n',
    )
    assert _found(tree) == []


# -- exemptions, baseline --------------------------------------------------------------------


def test_an_exemption_that_exempts_nothing_fails(tree: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(clients, "EXEMPT", {(f"{CORE}engine.py", "stop"): "a reason"})
    _write(tree, "engine.py", "def stop(z):\n    z.client.close()\n")
    assert _found(tree) == [("engine.py", 0, "stale exemption")]


def test_the_live_baseline_and_exemptions_are_empty() -> None:
    assert clients.BASELINE == {}
    assert clients.EXEMPT == {}
