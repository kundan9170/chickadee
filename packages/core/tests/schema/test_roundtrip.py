"""Every contract survives model → json → model unchanged (§9 DoD).

A wire format that does not round-trip is one where the journal cannot reconstruct state,
which is invariant 4 failing quietly rather than loudly.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from chickadee.schema import (
    ActionAuthorization,
    Completion,
    Confirmation,
    ExecRequest,
    Findings,
    IsolationSpec,
    JournalEvent,
    Pointer,
    PolicyClass,
    Question,
    QuestionKind,
    StopReason,
    Tier,
    Usage,
)
from chickadee.schema.journal import EventKind, RunStartP


def _instances() -> list[BaseModel]:
    return [
        JournalEvent(
            run_id="2026-08-18-a1b2c3d4",
            seq=1,
            ts="2026-08-18T09:00:00",
            kind=EventKind.RUN_START,
            payload=RunStartP(
                goal="g",
                work_sha="a" * 64,
                config_sha="b" * 64,
                repo_head="c" * 40,
                engine_version="0.0.0",
                schema_version=2,
                isolation="worktree",
                hard_cost_cap_usd=5.0,
            ),
        ),
        Completion(
            text="done",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            latency_ms=120,
            stop_reason=StopReason.END_TURN,
        ),
        Question(id="q1", kind=QuestionKind.PLAN_APPROVAL, text="ok?", default="approve"),
        ExecRequest(command="pytest -q", policy=PolicyClass.WORKTREE, timeout_s=120),
        Findings(targets=(Pointer(path="src/a.py", line_start=1, line_end=9),)),
        ActionAuthorization(
            operation_id="op-2",
            approved_hash="f" * 64,
            tier=Tier.T2,
            granted_at="2026-08-18T09:00:00",
            confirmations=(
                # Both name the action they authorize; the fixture previously pointed
                # them at a different hash, which D40 says is not consent at all.
                Confirmation(at="2026-08-18T09:00:00", source="plan", scope_sha="f" * 64),
                Confirmation(at="2026-08-18T09:00:05", source="dispatch", scope_sha="f" * 64),
            ),
        ),
        IsolationSpec(
            mode="worktree", lease_id="L1", branch="chickadee/run", worktree_path="/tmp/wt"
        ),
    ]


def test_roundtrip() -> None:
    for original in _instances():
        revived = type(original).model_validate_json(original.model_dump_json())
        assert revived == original, type(original).__name__


@pytest.mark.parametrize(
    "model",
    [
        JournalEvent,
        Completion,
        Question,
        ExecRequest,
        Findings,
        ActionAuthorization,
        IsolationSpec,
    ],
)
def test_wire_models_are_strict_and_frozen(model: type[BaseModel]) -> None:
    """Strict + immutable on every wire model, not merely on the ones we remembered."""
    assert model.model_config.get("extra") == "forbid"
    assert model.model_config.get("frozen") is True


def test_usage_cache_write_tokens_roundtrip() -> None:
    """Cache creation is its own counter, and None is not 0."""
    reported = Usage(prompt_tokens=10, completion_tokens=5, cache_write_tokens=7)
    assert Usage.model_validate_json(reported.model_dump_json()) == reported
    assert Usage.model_validate_json(reported.model_dump_json()).cache_write_tokens == 7

    unreported = Usage(prompt_tokens=10, completion_tokens=5)
    revived = Usage.model_validate_json(unreported.model_dump_json())
    assert revived == unreported
    assert revived.cache_write_tokens is None

    # None (not reported) and 0 (reported, nothing created) must survive as different values:
    # collapsing them is the D18 cache-metric corruption the docstring names.
    zero = Usage(prompt_tokens=10, completion_tokens=5, cache_write_tokens=0)
    assert Usage.model_validate_json(zero.model_dump_json()).cache_write_tokens == 0
    assert Usage.model_validate_json(zero.model_dump_json()) != revived
