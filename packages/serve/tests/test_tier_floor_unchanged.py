"""A T2 fact asks over the socket exactly as in-process: serving adds a
transport, never a tier. An unprotected shell command is T2; its approval reaches the
session log with the tier the router computed, and only an interactive answer lets it run: an
automation token's `approve` is the deny an unattended engine gives."""

from __future__ import annotations

from pathlib import Path

import pytest
from engines import Step, call, calls, config_for, raw_config, start, with_sessions
from repos import init_repo
from serve_helpers import served_engine

import chickadee.engine as engine_module
from chickadee.engine import PacketEngine
from chickadee.isolation.none import Passthrough
from chickadee_serve.auth import TokenClass

SID = "S-tier"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


def _engine(monkeypatch: pytest.MonkeyPatch, repo: Path) -> PacketEngine:
    """No sandbox and the floor on: every shell command is T2 and confirmed."""
    monkeypatch.setattr(engine_module, "_select_backend", lambda _asked: Passthrough())
    raw = raw_config()
    raw["sandbox"] = {"require_sandbox": False}
    script: list[Step] = [calls(call("shell", "s1", command="touch ran"))]
    engine, _ = start(monkeypatch, repo, config_for(raw), script)
    return engine


@pytest.mark.parametrize(
    ("token_class", "runs", "source"),
    [(TokenClass.INTERACTIVE, True, "human"), (TokenClass.AUTOMATION, False, "default")],
)
def test_a_t2_command_is_asked_over_the_socket_at_t2(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    token_class: TokenClass,
    runs: bool,
    source: str,
) -> None:
    engine = _engine(monkeypatch, repo)
    try:
        with served_engine(engine) as h:
            h.client.post(f"/session/{SID}/turn", json={"text": "go"}, headers=h.headers())
            asked = h.until(SID, "question_asked")
            approval = asked.payload["approval"]
            assert isinstance(approval, dict)
            assert (approval["kind"], approval["tier"]) == ("command", "t2")
            assert approval["operation_id"] == f"{SID}:s1"
            qid = str(approval["action_id"])
            r = h.client.post(
                f"/session/{SID}/answer/{qid}",
                json={"choice": "approve"},
                headers=h.headers(token_class),
            )
            assert r.status_code == 200, r.text
            answered = h.until(SID, "question_answered")
            h.until(SID, "tool")
    finally:
        engine.close()
    assert answered.payload["source"] == source
    assert answered.payload["choice"] == ("approve" if runs else "deny")
    assert (repo / "ran").exists() is runs
