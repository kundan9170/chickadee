"""Every EventKind: valid payload accepted, missing field rejected, unknown kind rejected.

§4.2 is normative and is the only list of kinds. This suite is what stops a payload model
from drifting away from it, and what stops a kind from being added in code without a row.
"""

from __future__ import annotations

import pytest
from event_builders import RUN_ID, SHA, TS, envelope, sample_payload
from pydantic import ValidationError

from chickadee.schema import ENVELOPE_VERSION, EventKind, JournalEvent
from chickadee.schema.journal import KIND_TABLE


@pytest.mark.parametrize("kind", list(EventKind), ids=lambda k: k.value)
def test_every_kind_accepts_its_payload(kind: EventKind) -> None:
    model, _ = KIND_TABLE[kind]
    event = JournalEvent.model_validate(envelope(kind, sample_payload(model)))
    assert event.kind is kind
    assert type(event.payload) is model


@pytest.mark.parametrize("kind", list(EventKind), ids=lambda k: k.value)
def test_every_kind_rejects_a_missing_required_field(kind: EventKind) -> None:
    model, _ = KIND_TABLE[kind]
    payload = sample_payload(model)
    if not payload:
        pytest.skip(f"{kind} has no required payload fields")
    payload.pop(next(iter(payload)))
    with pytest.raises(ValidationError):
        JournalEvent.model_validate(envelope(kind, payload))


@pytest.mark.parametrize("kind", list(EventKind), ids=lambda k: k.value)
def test_every_kind_rejects_an_extra_field(kind: EventKind) -> None:
    model, _ = KIND_TABLE[kind]
    payload = {**sample_payload(model), "surprise": 1}
    with pytest.raises(ValidationError):
        JournalEvent.model_validate(envelope(kind, payload))


@pytest.mark.parametrize("kind", list(EventKind), ids=lambda k: k.value)
def test_every_kind_roundtrips(kind: EventKind) -> None:
    model, _ = KIND_TABLE[kind]
    event = JournalEvent.model_validate(envelope(kind, sample_payload(model)))
    assert JournalEvent.model_validate_json(event.model_dump_json()) == event


def test_kind_table_matches_the_enum() -> None:
    """A kind not in the table does not exist — in both directions."""
    assert set(KIND_TABLE) == set(EventKind)


def test_unknown_kind_rejects() -> None:
    with pytest.raises(ValidationError, match=r"unknown event kind|kind"):
        JournalEvent.model_validate(
            {
                "v": ENVELOPE_VERSION,
                "run_id": RUN_ID,
                "seq": 1,
                "ts": TS,
                "kind": "invented_kind",
                "payload": {},
            }
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("run_id", "nope"),
        ("run_id", "2026-8-18-a1b2c3d4"),
        ("ts", "2026-08-18 09:00:00"),
        ("ts", "2026-08-18T09:00:00+00:00"),
        ("seq", 0),
    ],
)
def test_envelope_format_enforced(field: str, value: object) -> None:
    """seq starts at 1; ts is second-precision, zone-free or UTC `Z` and no other zone
    (`test_utc_ts.py` covers the `Z`)."""
    env = envelope(
        EventKind.SCOUT_START, {"rung": "scout", "goal_sha": SHA, "snapshot_sha": SHA}
    )
    env[field] = value
    with pytest.raises(ValidationError):
        JournalEvent.model_validate(env)


def test_scope_is_enforced() -> None:
    """A run-scoped kind carrying task_id would make replay's grouping meaningless."""
    payload = {"rung": "scout", "goal_sha": SHA, "snapshot_sha": SHA}
    env = envelope(EventKind.SCOUT_START, payload)
    with pytest.raises(ValidationError, match="must not carry"):
        JournalEvent.model_validate({**env, "task_id": "T1"})

    reported = sample_payload(KIND_TABLE[EventKind.TASK_REPORTED][0])
    task_env = envelope(EventKind.TASK_REPORTED, reported)
    task_env.pop("task_id")
    with pytest.raises(ValidationError, match="requires task_id"):
        JournalEvent.model_validate(task_env)


def test_a_line_of_another_version_is_refused() -> None:
    """Contracts v2: a packet-loop journal is not read or resumed, and the refusal says where
    it can be read instead rather than failing on its first retired kind."""
    env = {
        **envelope(
            EventKind.SCOUT_START, {"rung": "scout", "goal_sha": SHA, "snapshot_sha": SHA}
        ),
        "v": 1,
    }
    with pytest.raises(ValidationError, match="D196"):
        JournalEvent.model_validate(env)


def test_payload_type_must_match_kind() -> None:
    from chickadee.schema.journal import RunEndP, RunStartP

    good = RunStartP(
        goal="g",
        work_sha=SHA,
        config_sha=SHA,
        repo_head="c" * 40,
        engine_version="0",
        schema_version=2,
        isolation="worktree",
        hard_cost_cap_usd=1.0,
    )
    JournalEvent(run_id=RUN_ID, seq=1, ts=TS, kind=EventKind.RUN_START, payload=good)
    with pytest.raises(ValidationError, match="requires payload"):
        JournalEvent(
            run_id=RUN_ID,
            seq=1,
            ts=TS,
            kind=EventKind.RUN_START,
            payload=RunEndP(
                outcome="COMPLETED",
                tasks=1,
                committed=1,
                tokens=1,
                cost_usd=0.0,
                wall_ms=1,
                handoff_path="h.md",
            ),
        )
