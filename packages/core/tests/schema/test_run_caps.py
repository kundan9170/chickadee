"""`RunStartP.caps` and a `rung` on priced records (Issue #247).

Additive only: every field is optional, so a journal written before them still validates, in
the model and in the golden alike. A run in a family whose total cap is lifted can say
so, and every priced record can name the rung that made the call.
"""

from __future__ import annotations

import json
import math
from typing import Any

import jsonschema
import pytest
from event_builders import envelope, sample_payload
from pydantic import ValidationError

from chickadee.schema import EventKind
from chickadee.schema.export import GOLDEN_DIR
from chickadee.schema.journal import (
    KIND_TABLE,
    BudgetP,
    JournalEvent,
    ModelCallP,
    ReadAnsweredP,
    RunCapsP,
    RunStartP,
    ScoutReportP,
)

RUNG_PAYLOADS = {
    # The box's priced call: the line the run's spend is counted from.
    EventKind.MODEL_CALL: ModelCallP,
    EventKind.SCOUT_REPORT: ScoutReportP,
    EventKind.READ_ANSWERED: ReadAnsweredP,
    EventKind.BUDGET: BudgetP,
}


def golden() -> dict[str, Any]:
    return json.loads((GOLDEN_DIR / "journal_event.schema.json").read_text(encoding="utf-8"))


def written(kind: EventKind, **fields: Any) -> dict[str, Any]:
    """An event as the journal writes its line, with `fields` over the minimal payload."""
    model, _ = KIND_TABLE[kind]
    event = JournalEvent.model_validate(envelope(kind, {**sample_payload(model), **fields}))
    return json.loads(event.model_dump_json(by_alias=True, exclude_none=True))


def run_start(**fields: Any) -> RunStartP:
    return RunStartP.model_validate({**sample_payload(RunStartP), **fields})


# -- run_start -------------------------------------------------------------------------------


def test_a_run_start_written_before_caps_existed_still_validates() -> None:
    line = written(EventKind.RUN_START)
    assert "caps" not in line["payload"]
    jsonschema.validate(line, golden())
    assert run_start().caps is None


def test_caps_survive_the_journals_own_serialization_a_lifted_total_included() -> None:
    """The journal writes with `exclude_none`. A lifted total written as `null` would vanish on
    the way to disk, which is why lifting is said in `total_lifted`, never left as a `null`."""
    caps = {"total_lifted": True, "rungs": {"zeus": 5.0}, "lifted_rungs": ["coder"]}

    line = written(EventKind.RUN_START, hard_cost_cap_usd=20.0, caps=caps)
    jsonschema.validate(line, golden())
    back = RunStartP.model_validate(line["payload"])

    assert back.hard_cost_cap_usd == 20.0, "the config's total, unchanged"
    assert back.caps == RunCapsP(
        total_lifted=True, rungs={"zeus": 5.0}, lifted_rungs=("coder",)
    )


def test_a_capped_total_round_trips_and_no_rung_caps_is_empty() -> None:
    line = written(EventKind.RUN_START, caps={"total": 20.0})
    back = RunStartP.model_validate(line["payload"])
    assert back.caps == RunCapsP(total=20.0)
    assert back.caps is not None
    assert (back.caps.rungs, back.caps.lifted_rungs, back.caps.total_lifted) == ({}, (), False)


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, math.inf, math.nan])
def test_a_cap_no_config_could_hold_is_refused(bad: float) -> None:
    with pytest.raises(ValidationError):
        RunCapsP(total=bad)
    with pytest.raises(ValidationError):
        RunCapsP(total=1.0, rungs={"coder": bad})


@pytest.mark.parametrize(
    "bad",
    [
        {},  # neither a total nor a lift
        {"total": 1.0, "total_lifted": True},  # both
        {"total": 1.0, "rungs": {"coder": 1.0}, "lifted_rungs": ["coder"]},
        {"total": 1.0, "lifted_rungs": ["coder", "coder"]},
        {"total": 1.0, "rungs": {"": 1.0}},
        {"total": 1.0, "lifted_rungs": [""]},
        {"total": 1.0, "rungs": {"coder": None}},  # a lift is said, not a null
    ],
)
def test_caps_that_contradict_themselves_are_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunCapsP.model_validate(bad)


def test_the_golden_refuses_the_shapes_it_can_express() -> None:
    for caps in (
        {"total": 0},
        {"total": 1.0, "rungs": {"coder": -1.0}},
        {"total": 1.0, "rungs": {"coder": None}},
        {"total": 1.0, "lifted_rungs": [""]},
    ):
        line = written(EventKind.RUN_START)
        line["payload"]["caps"] = caps
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(line, golden())


def test_caps_still_refuse_an_unknown_field() -> None:
    with pytest.raises(ValidationError):
        run_start(caps={"total": 1.0, "per_model": {}})
    with pytest.raises(ValidationError):
        run_start(caps={"total": 1.0}, cap_lifted=True)


# -- rung on priced records -------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(RUNG_PAYLOADS), ids=lambda k: k.value)
def test_each_priced_record_validates_with_and_without_a_rung(kind: EventKind) -> None:
    model = RUNG_PAYLOADS[kind]
    before = written(kind)
    assert "rung" not in before["payload"]
    jsonschema.validate(before, golden())

    named = written(kind, rung="coder")
    assert named["payload"]["rung"] == "coder"
    jsonschema.validate(named, golden())
    assert model.model_validate(named["payload"]).rung == "coder"


@pytest.mark.parametrize("kind", list(RUNG_PAYLOADS), ids=lambda k: k.value)
def test_an_empty_rung_is_refused(kind: EventKind) -> None:
    model = RUNG_PAYLOADS[kind]
    with pytest.raises(ValidationError):
        model.model_validate({**sample_payload(model), "rung": ""})
    line = written(kind)
    line["payload"]["rung"] = ""
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(line, golden())
