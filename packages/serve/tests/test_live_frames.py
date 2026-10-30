"""D212's three live progress frames: a role's reply as it streams, described with its
role, task and reply id, and only the piece it streamed; a new attempt as `restart`."""

from __future__ import annotations

from chickadee.client.live import (
    ContextUpdate,
    LiveEvent,
    Restarted,
    TextUpdate,
    ThoughtUpdate,
    ToolUpdate,
    Update,
)
from chickadee.limits import tokens_for_chars
from chickadee.zeus.loop import live_event
from chickadee_serve.frames import PROGRESS_KINDS, decode_frame, encode_frame
from chickadee_serve.progress import describe


def _frame(update: Update) -> tuple[str, dict[str, object]]:
    frame = describe("S", live_event(LiveEvent("worker", "T1", "worker-1", update)))
    again = decode_frame(encode_frame(frame))
    assert again == frame, "round-trips as NDJSON"
    return frame.kind, dict(frame.payload)


def test_each_update_is_its_kind_with_who_said_it() -> None:
    who = {"role": "worker", "task": "T1", "reply": "worker-1"}
    assert _frame(TextUpdate(4, "more")) == (
        "text",
        {**who, "offset": 4, "piece": "more", "tokens": tokens_for_chars(8)},
    )
    assert _frame(Restarted()) == ("text", {**who, "restart": True})
    assert _frame(ThoughtUpdate(2.5, 0, "hmm")) == (
        "thought",
        {**who, "elapsed_s": 2.5, "offset": 0, "piece": "hmm", "tokens": tokens_for_chars(3)},
    )
    assert _frame(ToolUpdate(0, "create", "core/ai.py", 4100)) == (
        "tool_started",
        {
            **who,
            "index": 0,
            "tool": "create",
            "path": "core/ai.py",
            "size": 4100,
            "tokens": tokens_for_chars(4100),
        },
    )
    assert _frame(ContextUpdate(58_000, 128_000, "qwen3", compacting=True)) == (
        "context",
        {"used": 58_000, "window": 128_000, "model": "qwen3", "compacting": True},
    )


def test_the_live_kinds_are_on_the_wire() -> None:
    assert {"text", "thought", "tool_started", "context"} <= set(PROGRESS_KINDS)
