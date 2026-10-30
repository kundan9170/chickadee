"""Event builders shared by the contract suite.

Kept out of conftest because these are called as plain functions, not injected as
fixtures — the same split the journal suite uses. Named distinctly rather than
`helpers.py` because the test directories are not packages: two `helpers` modules on
sys.path resolve to whichever pytest imported first, silently.

Payloads are derived from each model's own required fields rather than hand-written for
every kind, so a newly required field is exercised the moment it is added instead of when
someone remembers to update a fixture.
"""

from __future__ import annotations

import enum
from typing import Any, Literal, get_args, get_origin

from chickadee.schema import ENVELOPE_VERSION, EventKind, JournalEvent
from chickadee.schema.journal import KIND_TABLE, Scope

RUN_ID = "2026-08-18-a1b2c3d4"
TS = "2026-08-18T09:00:00"
SHA = "a" * 64


def value_for(ann: Any, name: str) -> Any:
    if name == "agent":
        return "A1"  # an `AgentId`
    if name == "tell":
        return "tell-1"  # a `TellId`
    if get_origin(ann) is Literal:
        return get_args(ann)[0]
    if ann is bool:
        return True
    if ann is int:
        return 1
    if ann is float:
        return 1.0
    if ann is str or name == "task_id":
        if name == "task_id":
            return "T1"
        return SHA if name.endswith("_sha") else "x"
    if get_origin(ann) is tuple:
        return []
    if get_origin(ann) is dict:
        return {}
    if isinstance(ann, type) and hasattr(ann, "model_fields"):
        return sample_payload(ann)
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return next(iter(ann.__members__.values())).value
    return "x"


def sample_payload(model: type) -> dict[str, Any]:
    """A minimal valid payload built from the model's own required fields."""
    out: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if not field.is_required():
            continue
        out[field.alias or name] = value_for(field.annotation, name)
    return out


def envelope(kind: EventKind, payload: dict[str, Any]) -> dict[str, Any]:
    _, scope = KIND_TABLE[kind]
    env: dict[str, Any] = {
        "v": ENVELOPE_VERSION,
        "run_id": RUN_ID,
        "seq": 1,
        "ts": TS,
        "kind": kind.value,
        "payload": payload,
    }
    if scope is Scope.TASK:
        env["task_id"] = "T1"
    return env


def sample_event(kind: EventKind) -> JournalEvent:
    model, _ = KIND_TABLE[kind]
    return JournalEvent.model_validate(envelope(kind, sample_payload(model)))
