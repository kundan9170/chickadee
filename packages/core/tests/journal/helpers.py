"""Event builders shared by the journal suite.

Kept out of conftest because these are called as plain functions, not injected as
fixtures; pytest puts this directory on sys.path for non-package test dirs.
"""

from __future__ import annotations

from typing import Any

from chickadee.schema import EventKind
from chickadee.schema.journal import KIND_TABLE, Scope

SHA = "a" * 64


def payload_for(kind: EventKind, **over: Any) -> Any:
    """A minimal valid payload for a kind, built from the model's own required fields."""
    model, _ = KIND_TABLE[kind]
    data: dict[str, Any] = {}
    for name, f in model.model_fields.items():
        if not f.is_required():
            continue
        data[f.alias or name] = _sample(f.annotation, name)
    data.update(over)
    return model.model_validate(data)


def _sample(ann: Any, name: str) -> Any:
    from typing import Literal, get_args, get_origin

    if get_origin(ann) is Literal:
        return get_args(ann)[0]
    if ann is bool:
        return True
    if ann is int:
        return 1
    if ann is float:
        return 1.0
    if name == "task_id":
        return "T1"
    if ann is str:
        return SHA if name.endswith("_sha") else "x"
    if get_origin(ann) is tuple:
        return []
    if get_origin(ann) is dict:
        return {}
    if isinstance(ann, type) and hasattr(ann, "model_fields"):
        return {
            (f.alias or n): _sample(f.annotation, n)
            for n, f in ann.model_fields.items()
            if f.is_required()
        }
    return "x"


def scope_args(kind: EventKind) -> dict[str, Any]:
    _, scope = KIND_TABLE[kind]
    if scope is Scope.TASK:
        return {"task_id": "T1"}
    return {}
