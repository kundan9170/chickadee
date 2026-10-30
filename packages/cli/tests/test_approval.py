"""The approval-stop rate at a terminal, and a design gate an older
log left open. The gate itself retired with the switch; its stop's prompt is
`test_stop_prompt.py`'s. A gate in an older log is still read: its id pairs with its answer,
so the rate counts it, and the REPL closes it with any line."""

from __future__ import annotations

from typing import Any

from chickadee_cli.approval import (
    STOP_RATE_CEILING,
    StopRate,
    stop_rate,
    stop_rate_line,
)
from chickadee_cli.interaction import RETIRED_GATE, Interaction
from chickadee_cli.render.style import Style
from chickadee_serve.frames import RecordFrame

PIPE = Style(tty=False)

GATE: dict[str, Any] = {
    # A design gate as a log written in older versions holds it.
    "id": "design-abc-1",
    "design": {
        "goal": "add a session refresh endpoint",
        "components": ["auth"],
        "contracts_changing": ["auth.refresh_session"],
        "contracts_protected": ["api.*"],
        "effects": [],
        "removes": [
            {"path": "auth/session.py", "symbol": "legacy_refresh_v1"},
            {"path": "auth/keys.py", "symbol": "_v1_salt"},
        ],
    },
    "design_sha": "a" * 64,
    "plan": {
        "v": 1,
        "run_goal": "add a session refresh endpoint",
        "stubs": [
            {
                "id": "t1",
                "deps": [],
                "goal": "g",
                "task": "scaffold",
                "target_hint": "auth/session.py",
                "kind": "code",
            },
            {
                "id": "t2",
                "deps": ["t1"],
                "goal": "g",
                "task": "refresh_session",
                "target_hint": "auth/session.py",
                "kind": "code",
            },
        ],
        "checkpoints": [],
    },
    "tier": "t1",
    "hits": [{"row": 4, "evidence": "changes a test", "floor": "t1", "lowerable": True}],
    "estimate": {"tokens": 12400, "cost_usd": 0.04, "basis": "dag"},
    "workspace": {"mode": "worktree", "source_ref": "main"},
    "constraints": [
        {
            "id": "C-3",
            "text": "no file under auth/ is renamed",
            "sources": [4],
            "author": "zeus",
            "scope": "session",
            "scope_id": None,
            "valid_from": 5,
            "valid_until": None,
            "superseded_by": None,
        }
    ],
    "not_applied": [
        {
            "id": "C-9",
            "text": "the migration is reversible",
            "sources": [40],
            "author": "zeus",
            "scope": "design",
            "scope_id": None,
            "valid_from": 41,
            "valid_until": None,
            "superseded_by": None,
        }
    ],
    "round": 1,
}


def no_editor(_text: str) -> str | None:
    raise AssertionError("no editor was asked for")


class _Wire:
    """The client's answer route, as the REPL reaches it."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def answer(
        self, sid: str, qid: str, choice: str, note: str | None = None, **kw: Any
    ) -> Any:
        self.sent.append({"qid": qid, "choice": choice, "note": note, **kw})


def test_an_old_logs_open_gate_is_named_and_closed_by_any_line() -> None:
    """A design gate a log written before the stops left open is
    not shown as a gate: it is named, and the next line closes it as a cancel."""
    wire = _Wire()
    out: list[str] = []
    questions = Interaction(wire, "S-1", PIPE, out.append, edit=no_editor)  # type: ignore[arg-type]
    questions.feed(
        RecordFrame(
            scope="S-1",
            seq=3,
            at="2026-10-12T10:00:00Z",
            kind="question_asked",
            payload={"gate": GATE},
        )
    )
    assert " ".join(out) == RETIRED_GATE, "named as it opens, not drawn"
    assert questions.answer("a") == "design-abc-1"
    assert wire.sent == [{"qid": "design-abc-1", "choice": "cancel", "note": "a", "text": None}]


# -- the approval-stop rate -------------------------------------------------------------


def _frame(
    seq: int, kind: str, payload: dict[str, Any], run_id: str | None = None
) -> RecordFrame:
    return RecordFrame(
        scope="S-1",
        seq=seq,
        at="2026-10-12T10:00:00Z",
        kind=kind,
        payload=payload,
        run_id=run_id,
    )


def _episode(seq: int, run: str) -> RecordFrame:
    return _frame(
        seq,
        "episode",
        {
            "run_id": run,
            "path": "planned",
            "tier": "t1",
            "design_sha": None,
            "constraints": [],
            "opened_at_seq": 1,
        },
    )


def _answered(seq: int, qid: str, source: str, run_id: str | None = None) -> RecordFrame:
    return _frame(
        seq,
        "question_answered",
        {"id": qid, "choice": "approve", "source": source, "elapsed_ms": 1},
        run_id,
    )


def test_the_stop_rate_counts_delegations_a_person_approved() -> None:
    records = [
        # A gate a person approved, then its run.
        _frame(1, "question_asked", {"gate": GATE}),
        _answered(2, "design-abc-1", "human"),
        _frame(
            3,
            "plan_decision",
            {
                "id": "design-abc-1",
                "decision": "approve",
                "design_sha": "a" * 64,
                "source": "human",
            },
        ),
        _episode(4, "R-1"),
        # A run nobody was asked about.
        _episode(5, "R-2"),
        # A question that is not an approval, answered by a person: not a stop.
        _frame(
            6,
            "question_asked",
            {
                "question": {
                    "id": "q1",
                    "kind": "clarify_goal",
                    "text": "which?",
                    "default": "a",
                }
            },
        ),
        _answered(7, "q1", "human"),
        _episode(8, "R-3"),
        # Unattended: the default approved it. Not a stop.
        _frame(9, "question_asked", {"gate": {**GATE, "id": "design-abc-2"}}),
        _answered(10, "design-abc-2", "default"),
        _episode(11, "R-4"),
        _episode(12, "R-5"),
    ]
    rate = stop_rate(records)
    assert (rate.stopped, rate.delegations) == (1, 5)
    text, _tone = stop_rate_line(rate)
    assert text == "1 of 5 delegations stopped for a person (20%)", "a fifth is not above it"


def test_a_cancelled_gate_is_one_delegation_and_a_stop() -> None:
    """The cancelled call still has its `episode` record, and is counted once."""
    records = [
        _frame(1, "question_asked", {"gate": GATE}),
        _answered(2, "design-abc-1", "human"),
        _frame(
            3,
            "plan_decision",
            {
                "id": "design-abc-1",
                "decision": "cancel",
                "design_sha": "a" * 64,
                "source": "human",
            },
        ),
        _episode(4, "R-1"),
    ]
    rate = stop_rate(records)
    assert (rate.stopped, rate.delegations) == (1, 1)


def test_above_a_fifth_the_line_says_the_floor_asks_too_often() -> None:
    assert StopRate(1, 5).too_high is False and StopRate(2, 5).too_high is True
    assert STOP_RATE_CEILING == 0.2
    assert "asks too often" in stop_rate_line(StopRate(2, 5))[0]
    assert stop_rate_line(StopRate(0, 0))[0] == "no delegation yet"
