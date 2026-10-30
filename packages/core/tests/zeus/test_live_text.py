"""Zeus's replies shown as they stream: each request carries a live view
labelled `zeus`, a fresh one per request; the pieces never reach the session log, which records
the finished reply as before."""

from __future__ import annotations

from pathlib import Path

from chickadee.client.live import ContextUpdate, LiveEvent, Text, TextUpdate
from chickadee.client.request import ChatRequest
from chickadee.schema.adapter import Completion

from .zeus_helpers import answer, payloads, rig


def _streams(text: str) -> object:
    def step(request: ChatRequest) -> Completion:
        assert request.live is not None
        for word in text.split(" "):
            request.live(Text(word + " "))
        return answer(text)

    return step


def test_his_reply_streams_labelled_zeus_and_the_log_keeps_the_finished_reply(
    tmp_path: Path,
) -> None:
    shown: list[LiveEvent] = []
    r = rig(tmp_path, [_streams("the readme is fine")], live=shown.append)  # type: ignore[list-item]
    r.run("is the readme fine?")
    assert {(e.role, e.task) for e in shown} == {("zeus", None)}
    assert "".join(e.update.piece for e in shown if isinstance(e.update, TextUpdate)) == (
        "the readme is fine "
    )
    turns = [p for p in payloads(r.session, "turn") if p.get("role") == "assistant"]
    assert [t["text"] for t in turns] == ["the readme is fine"]


def test_without_a_sink_no_live_is_sent(tmp_path: Path) -> None:
    r = rig(tmp_path, [answer("ok")])
    r.run("hi")
    assert r.model.requests[0].live is None


def test_his_context_is_reported_after_each_answer(tmp_path: Path) -> None:
    """The tokens his last call used, against his window, for the status
    line; reported only where the window is known."""
    shown: list[LiveEvent] = []
    r = rig(tmp_path, [answer("ok")], live=shown.append, window=1000)
    r.run("hi")
    context = [e.update for e in shown if isinstance(e.update, ContextUpdate)]
    assert context == [ContextUpdate(110, 1000, "frontier")]
