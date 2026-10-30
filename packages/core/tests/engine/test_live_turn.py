"""A reply streams to the turn's events, through a real engine: what Zeus's
model streams reaches `TurnHandle.events()` as `text` events labelled `zeus`, before the
`answer` event, and none of it is on the session log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from repos import init_repo
from zeus_turn_helpers import SID, WAIT_S, answer, config_for, start, state, with_sessions

from chickadee.client.live import LiveEvent, Text, TextUpdate
from chickadee.client.request import ChatRequest
from chickadee.engine import PacketEngine
from chickadee.schema.adapter import Completion


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return with_sessions(init_repo(tmp_path / "repo", commit=None), SID)


def _streams(request: ChatRequest) -> Completion:
    assert request.live is not None, "the engine gives Zeus's request a live view"
    for word in ("all ", "good"):
        request.live(Text(word))
    return answer("all good")


def test_his_reply_streams_to_the_turns_events(
    monkeypatch: pytest.MonkeyPatch, repo: Path, raw: dict[str, Any]
) -> None:
    engine: PacketEngine
    engine, _model = start(monkeypatch, repo, config_for(raw), [_streams])
    try:
        handle = engine.turn(SID, "hi")
        handle.wait(timeout=WAIT_S)
        events = list(handle.events())
        live = [e for e in events if e.kind == "text"]
        assert live, "the reply streamed"
        assert all(isinstance(e.payload, LiveEvent) and e.payload.role == "zeus" for e in live)
        text = "".join(
            e.payload.update.piece
            for e in live
            if isinstance(e.payload, LiveEvent) and isinstance(e.payload.update, TextUpdate)
        )
        assert text == "all good"
        kinds = [e.kind for e in events]
        assert kinds.index("text") < kinds.index("answer")
        logged = {event.kind for event, _ in state(engine).lines()}
        assert not logged & {"text", "thought", "tool_started"}, "never journaled"
    finally:
        engine.close()
