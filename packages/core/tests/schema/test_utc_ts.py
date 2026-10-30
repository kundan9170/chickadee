"""The journal's `ts` takes an optional trailing `Z`.

Additive only: a zone-free `ts`, which every journal before this carries, still validates; a
UTC one with its `Z` validates and survives the journal's own serialization. Anything else
that names a zone, or a precision other than seconds, is refused, in the model and in the
golden alike.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
import pytest
from event_builders import envelope, sample_payload
from pydantic import ValidationError

from chickadee.schema import EventKind, JournalEvent
from chickadee.schema.export import GOLDEN_DIR
from chickadee.schema.journal import RunEndP

GOLDEN = json.loads((GOLDEN_DIR / "journal_event.schema.json").read_text(encoding="utf-8"))


def _event(ts: str) -> dict[str, Any]:
    raw = envelope(EventKind.RUN_END, sample_payload(RunEndP))
    raw["ts"] = ts
    return raw


@pytest.mark.parametrize("ts", ["2026-10-04T09:30:00", "2026-10-04T09:30:00Z"])
def test_a_zone_free_or_utc_ts_validates(ts: str) -> None:
    event = JournalEvent.model_validate(_event(ts))
    assert event.ts == ts
    jsonschema.validate(json.loads(event.model_dump_json(by_alias=True)), GOLDEN)


def test_a_utc_ts_round_trips_through_the_journal_s_serialization() -> None:
    """`Journal.append` writes `model_dump_json(by_alias=True, exclude_none=True)`, and the
    reader validates it back: the `Z` survives both."""
    event = JournalEvent.model_validate(_event("2026-10-04T09:30:00Z"))
    line = event.model_dump_json(by_alias=True, exclude_none=True)
    assert JournalEvent.model_validate_json(line).ts == "2026-10-04T09:30:00Z"


@pytest.mark.parametrize(
    "ts",
    [
        "2026-10-04T09:30:00+00:00",
        "2026-10-04T09:30:00+05:30",
        "2026-10-04T09:30:00.123Z",
        "2026-10-04T09:30:00.123",
        "2026-10-04T09:30:00z",
        "2026-10-04T09:30:00ZZ",
    ],
)
def test_any_other_zone_or_precision_is_refused(ts: str) -> None:
    with pytest.raises(ValidationError):
        JournalEvent.model_validate(_event(ts))
    raw = _event("2026-10-04T09:30:00")
    raw["payload"] = json.loads(RunEndP.model_validate(raw["payload"]).model_dump_json())
    raw["ts"] = ts
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, GOLDEN)
