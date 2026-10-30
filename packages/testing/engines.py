"""An engine with a scripted model behind it, made one way (one concept, one home).

The core suites that drive `PacketEngine.turn()` and the serve suites that drive it over a
socket both need a model that answers from a script, and the minimal config that names it. It
lives here, beside `repos.py`, so neither package keeps its own copy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import chickadee.engine as engine_module
from chickadee.client.request import ChatRequest
from chickadee.context.session import Session
from chickadee.context.store import sessions_root
from chickadee.engine import PacketEngine
from chickadee.roles import DEFAULT_RUNGS
from chickadee.schema import StopReason, ToolCall
from chickadee.schema.adapter import Completion, Usage
from chickadee.state.layers import Layer
from chickadee.state.resolve import Resolved, resolve
from chickadee.state.roots import roots as resolve_roots

MODEL: dict[str, Any] = {
    "base_url": "http://localhost:8000/v1",
    "model_id": "m",
    # Large enough for every role to fit one request: the worker needs
    # 0.40 of the window, one 8,000-token result and a 16,000-token output floor.
    "context_window": 65_536,
    "max_output_tokens": 16_384,
}


def raw_config() -> dict[str, Any]:
    """One model per rung, each named unlike its rung, so a test can change one rung's
    endpoint without moving the others' — and so a price keyed by the rung, or a window read
    off the wrong entry, fails instead of passing by coincidence. A test that edits one
    rung's endpoint edits `raw["models"][raw["rungs"][rung]]`."""
    return {
        "models": {f"{name}-model": dict(MODEL) for name in DEFAULT_RUNGS},
        "rungs": {name: f"{name}-model" for name in DEFAULT_RUNGS},
        "spend": {"hard_cost_usd": 5.0},
    }


def config_toml(raw: dict[str, Any] | None = None) -> str:
    """`raw` (by default `raw_config()`) as the engine config file a person writes: for a
    test that runs a host in its own process, which reads the file, not a dict."""
    tables: dict[str, Any] = raw if raw is not None else raw_config()
    out: list[str] = []

    def table(name: str, body: dict[str, Any]) -> None:
        scalars = {k: v for k, v in body.items() if not isinstance(v, dict)}
        out.append(f"[{name}]")
        out.extend(f"{key} = {_toml_value(value)}" for key, value in scalars.items())
        out.append("")
        for key, value in body.items():
            if isinstance(value, dict):
                table(f"{name}.{_toml_key(key)}", cast("dict[str, Any]", value))

    for name, body in tables.items():
        table(_toml_key(name), body)
    return "\n".join(out)


def _toml_key(key: str) -> str:
    return key if key.replace("-", "").replace("_", "").isalnum() else f'"{key}"'


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in cast("list[object]", value)) + "]"
    return repr(value)


# -- the scripted rung ------------------------------------------------------------------------


Step = Completion | Callable[[ChatRequest], Completion]


def calls(*items: ToolCall) -> Completion:
    return Completion(
        text="",
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=items,
    )


def answer(text: str = "done") -> Completion:
    return Completion(
        text=text,
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        latency_ms=1,
        stop_reason=StopReason.END_TURN,
    )


def call(tool: str, cid: str, /, **args: object) -> ToolCall:
    return ToolCall(id=cid, name=tool, arguments=dict(args))


class ScriptedRung:
    """The zeus rung: the listed completions in order, then `done`."""

    def __init__(self, script: Sequence[Step] = ()) -> None:
        self._script = list(script)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        if not self._script:
            return answer()
        step = self._script.pop(0)
        return step(request) if callable(step) else step

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing to release."""


# -- the engine -------------------------------------------------------------------------------


def with_prices(raw: dict[str, Any]) -> dict[str, Any]:
    """`raw` with a price for the rung a charge is billed to: an unpriced charge is refused.
    Changes `raw` and returns it."""
    raw["prices"] = {raw["rungs"]["high-param"]: {"price_in": 1.0, "price_out": 2.0}}
    return raw


def config_for(
    raw: dict[str, Any], *, project: dict[str, Any] | None = None, priced: bool = True
) -> Resolved:
    if priced:
        with_prices(raw)
    # A session's name is a model call that would take a scripted answer meant for the
    # turn under test; a test that names a session turns it back on.
    raw.setdefault("interaction", {}).setdefault("auto_title", False)
    layers = [Layer("user", None, raw)]
    if project:
        layers.append(Layer("project", Path("/repo/.chickadee/config.toml"), project))
    return resolve(layers)


def with_sessions(repo: Path, *sids: str) -> Path:
    """`repo` with the sessions a suite turns into, created as `create_session()` creates
    them: a turn opens a session and never makes one."""
    root = sessions_root(resolve_roots(repo))
    for sid in sids:
        Session.create(root, sid).close()
    return repo


def scripted(monkeypatch: pytest.MonkeyPatch, script: Sequence[Step] = ()) -> ScriptedRung:
    """Every engine built from here on calls `script`'s model, whoever builds it: a test, or
    a host in this process (`hosting.py`). An episode makes its roles' clients through
    `delegation.client_for`, so that one is patched too, or a `coder` call reaches a real
    client."""
    from chickadee import delegation

    model = ScriptedRung(script)
    monkeypatch.setattr(engine_module, "client_for", lambda _endpoint: model)
    monkeypatch.setattr(delegation, "client_for", lambda _endpoint: model)
    return model


def start(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    config: Resolved,
    script: Sequence[Step] = (),
) -> tuple[PacketEngine, ScriptedRung]:
    model = scripted(monkeypatch, script)
    engine = PacketEngine(
        repo, config.engine, trust=config.trust(resolve_roots(repo).repo_local, root=repo)
    )
    return engine, model
