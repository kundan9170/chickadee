"""The additive tool contract: the `tool_call` line and its fold
over the scout's older kind.

`test_journal_payloads.py` already holds every kind to "accepts its payload, refuses a missing
or extra field, round-trips". This file holds what is particular to `tool_call`: the scopes it
is written at, the fields it refuses, and that `journal.projection.tool_call` reads both kinds
as one shape, so no reader learns two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from event_builders import RUN_ID, SHA, TS
from pydantic import ValidationError

from chickadee.journal.projection import tool_call
from chickadee.schema import EventKind, JournalEvent
from chickadee.schema.journal import KIND_TABLE, Scope, ScoutToolP, ToolCallP

GOLDEN = Path(__file__).resolve().parents[2] / "schema" / "golden" / "journal_event.schema.json"


def _call(**fields: Any) -> dict[str, Any]:
    return {
        "role": "worker",
        "tool": "grep_in",
        "args_sha": SHA,
        "outcome": "served",
        "result_tokens": 12,
        **fields,
    }


def _event(payload: dict[str, Any], **scope: Any) -> JournalEvent:
    return JournalEvent.model_validate(
        {"run_id": RUN_ID, "seq": 1, "ts": TS, "kind": "tool_call", "payload": payload, **scope}
    )


def test_tool_call_is_flexible_scoped() -> None:
    """A worker's call sits in its task's round; a role outside a task writes at run scope."""
    assert KIND_TABLE[EventKind.TOOL_CALL] == (ToolCallP, Scope.FLEXIBLE)


@pytest.mark.parametrize(
    "scope",
    [{}, {"task_id": "T1", "round": 1}],
    ids=["run", "task"],
)
def test_a_tool_call_line_round_trips_at_either_scope(scope: dict[str, Any]) -> None:
    event = _event(_call(result_sha="r" * 64), **scope)
    assert isinstance(event.payload, ToolCallP)
    assert JournalEvent.model_validate_json(event.model_dump_json()) == event


def test_the_defaults_say_served_and_unrefused() -> None:
    p = ToolCallP.model_validate(_call())
    assert (p.denied_reason, p.result_sha, p.dedup_hit) == (None, None, False)


@pytest.mark.parametrize(
    "bad",
    [
        {"role": ""},
        {"tool": ""},
        {"result_tokens": -1},
        {"outcome": "denied"},
        {"allowed": True},
    ],
    ids=["empty-role", "empty-tool", "negative-tokens", "unknown-outcome", "extra-field"],
)
def test_a_malformed_line_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ToolCallP.model_validate(_call(**bad))


def test_any_registered_name_is_carried() -> None:
    """The point of the kind: a name, not a closed list of tools."""
    assert ToolCallP.model_validate(_call(tool="grep_in")).tool == "grep_in"


def test_the_golden_carries_the_kind_and_its_rule() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert "ToolCallP" in golden["$defs"]
    assert "tool_call" in golden["$defs"]["EventKind"]["enum"]
    rules = [r for r in golden.get("allOf", []) if _kind_of(r) == "tool_call"]
    assert len(rules) == 1
    assert rules[0]["then"]["properties"]["payload"]["$ref"] == "#/$defs/ToolCallP"


def _kind_of(rule: dict[str, Any]) -> object:
    return rule.get("if", {}).get("properties", {}).get("kind", {}).get("const")


# -- the fold -----------------------------------------------------------------------------


def test_a_tool_call_line_folds_to_itself() -> None:
    p = ToolCallP.model_validate(_call())
    assert tool_call(p) is p


@pytest.mark.parametrize("dedup", [False, True])
def test_a_scout_tool_line_folds_to_the_locate_role_s_call(dedup: bool) -> None:
    folded = tool_call(
        ScoutToolP(tool="search", args_sha=SHA, result_tokens=7, dedup_hit=dedup)
    )
    assert folded == ToolCallP(
        role="locate",
        tool="search",
        args_sha=SHA,
        outcome="served",
        result_tokens=7,
        dedup_hit=dedup,
    )
