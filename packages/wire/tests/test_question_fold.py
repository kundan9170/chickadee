"""The open-question fold (`chickadee_wire.questions`) over each kind a
`question_asked` record holds; here the stop: it opens under its id, holds
the markdown as shown, and its `question_answered` closes it."""

from __future__ import annotations

from chickadee_serve.frames import JsonValue, RecordFrame
from chickadee_wire.questions import Questions

STOP: dict[str, JsonValue] = {
    "id": "stop-1",
    "kind": "design",
    "author": "zeus",
    "text": "# Design\n\nMake go return 2.\n",
    "sha": "a" * 64,
    "workspace": "worktree",
}


def _frame(seq: int, kind: str, payload: dict[str, JsonValue]) -> RecordFrame:
    return RecordFrame(
        scope="S", seq=seq, at="2026-10-15T10:00:00Z", kind=kind, payload=payload
    )


def test_a_stop_opens_with_its_markdown_and_its_answer_closes_it() -> None:
    fold = Questions()
    fold.feed(_frame(1, "question_asked", {"stop": STOP}))
    [asked] = fold.open
    assert asked.id == "stop-1" and asked.stop is not None
    assert asked.stop.text == STOP["text"]
    assert (asked.question, asked.approval, asked.gate) == (None, None, None)
    fold.feed(
        _frame(
            2,
            "question_answered",
            {"id": "stop-1", "choice": "approve", "source": "human", "elapsed_ms": 1},
        )
    )
    assert fold.open == ()
