"""Frames for the TUI's tests: the recorded run as the server frames it (`recordings`, the one
home of the recordings a client's tests read), and a session's records built by hand.

The run is `serve/tests/recorded_run.py`'s recording: Athena plans two
tasks, T1 fails its tests in round 1, passes in round 2 and is committed, T2 ends its round at
its limit and is abandoned, and the run ends `PARTIAL`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

from recordings import recorded_run_id, run_frames
from textual.pilot import Pilot

from chickadee_serve.frames import JsonValue, ProgressFrame, ProgressKind, RecordFrame

__all__ = [
    "RUN_ID",
    "SID",
    "WAIT_S",
    "answered",
    "approval_asked",
    "assistant",
    "episode",
    "progress",
    "run_frames",
    "run_record",
    "session",
    "spend",
    "stop_asked",
    "until",
    "user",
    "workspace",
]

RUN_ID: Final = recorded_run_id()
SID: Final = "S-tui"


def session(kind: str, payload: dict[str, JsonValue], seq: int) -> RecordFrame:
    """One session record."""
    return RecordFrame(
        scope=SID, seq=seq, at="2026-10-10T10:00:00Z", kind=kind, payload=payload
    )


def progress(kind: ProgressKind, **payload: JsonValue) -> ProgressFrame:
    return ProgressFrame(scope=SID, kind=kind, payload=dict(payload))


def user(text: str, seq: int) -> RecordFrame:
    return session("turn", {"role": "user", "text": text}, seq)


def assistant(text: str, seq: int) -> RecordFrame:
    return session("turn", {"role": "assistant", "text": text, "calls": []}, seq)


def episode(seq: int, run_id: str = RUN_ID) -> RecordFrame:
    return session("episode", {"run_id": run_id, "path": "planned"}, seq)


def stop_asked(stop_id: str, kind: str, seq: int) -> RecordFrame:
    """Stop 1 (`design`, Zeus's) or stop 2 (`plan`, Athena's), asked."""
    author = "zeus" if kind == "design" else "athena"
    stop: dict[str, JsonValue] = {
        "id": stop_id, "kind": kind, "author": author, "sha": "s" * 64,
        "text": f"# the {kind}", "workspace": "worktree",
    }  # fmt: skip
    return session("question_asked", {"stop": stop}, seq)


def approval_asked(action_id: str, seq: int, **fields: JsonValue) -> RecordFrame:
    """A command's confirmation, asked (`ActionApproval`): the worker's, on T2, at T2, with no
    *always* to give unless `fields` adds a `grant` or an `exact` binding."""
    approval: dict[str, JsonValue] = {
        "action_id": action_id, "operation_id": f"{SID}:{action_id}", "asker": "worker",
        "task_id": "T2", "kind": "command", "tier": "t2", "network": False,
        "idempotent": False, "reversible": False, "spec_sha": "p" * 64,
        "display": "uv add numpy  (in .; package install at t2) — the tests need numpy",
    }  # fmt: skip
    approval.update(fields)
    return session("question_asked", {"approval": approval}, seq)


def answered(qid: str, seq: int, choice: str = "approve") -> RecordFrame:
    return session(
        "question_answered",
        {"id": qid, "choice": choice, "source": "human", "elapsed_ms": 10},
        seq,
    )


def workspace(handoff: str, seq: int) -> RecordFrame:
    return session(
        "workspace",
        {
            "handoff": handoff, "lease_id": "L1", "mode": "worktree", "source_head": "a" * 40,
            "source_ref": "refs/heads/main", "branch": "ccr-1", "worktree_path": "/wt/s-1",
        },
        seq,
    )  # fmt: skip


def spend(cost: float, seq: int, model: str = "served-model") -> RecordFrame:
    return session(
        "spend",
        {
            "cost_usd": cost, "cache_writes": 0, "cached_in": 0, "estimated": False,
            "model": model, "rung": "high-param", "tokens_in": 10, "tokens_out": 5,
        },
        seq,
    )  # fmt: skip


def run_record(
    kind: str, payload: dict[str, JsonValue], seq: int, **envelope: str
) -> RecordFrame:
    """One record of the recorded run's scope, built by hand."""
    return RecordFrame(
        scope=RUN_ID, run_id=RUN_ID, seq=seq, at="2026-10-10T10:02:00Z", kind=kind,
        payload=payload, task_id=envelope.get("task_id"),
    )  # fmt: skip


WAIT_S: Final = 20.0


async def until(pilot: Pilot[None], test: Callable[[], object], what: str) -> None:
    """Wait on the app's loop until `test()` holds; fail naming `what` after `WAIT_S`."""
    deadline = time.monotonic() + WAIT_S
    while not test():
        assert time.monotonic() < deadline, f"never: {what}"
        await pilot.pause(0.05)
