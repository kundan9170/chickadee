"""`export_all` output byte-equals the committed goldens.

This is what makes "frozen at G0" enforceable. A field added to a wire model without a
regenerated golden fails here, in CI, rather than reaching six agents who each build
against a different shape.
"""

from __future__ import annotations

import json
import pathlib

import jsonschema
import pytest
from event_builders import sample_event

from chickadee.schema import EventKind
from chickadee.schema.export import CONTRACTS, GOLDEN_DIR, export_all, render
from chickadee.schema.journal import KIND_TABLE


@pytest.mark.parametrize("name", sorted(CONTRACTS), ids=sorted(CONTRACTS))
def test_golden_matches_regeneration(name: str) -> None:
    golden = GOLDEN_DIR / f"{name}.schema.json"
    assert golden.exists(), f"missing golden {golden}; run `python -m chickadee.schema.export`"
    assert golden.read_text(encoding="utf-8") == render(name), (
        f"{name} schema drifted from its golden. If the change is intended, regenerate "
        "with `python -m chickadee.schema.export` and review the diff as a contract change."
    )


def test_goldens_are_valid_json_schema_2020_12() -> None:
    for name in CONTRACTS:
        doc = json.loads((GOLDEN_DIR / f"{name}.schema.json").read_text())
        assert doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_export_writes_every_contract(tmp_path: pathlib.Path) -> None:
    written = export_all(tmp_path)
    assert {p.name for p in written} == {f"{n}.schema.json" for n in CONTRACTS}


def _golden(name: str) -> dict[str, object]:
    return json.loads((GOLDEN_DIR / f"{name}.schema.json").read_text(encoding="utf-8"))


def _event(kind: EventKind) -> dict[str, object]:
    """One real event, serialized exactly as the journal writes its lines."""
    line = sample_event(kind).model_dump_json(by_alias=True, exclude_none=True)
    return json.loads(line)


@pytest.mark.parametrize("kind", list(EventKind), ids=lambda k: k.value)
def test_real_events_validate_against_the_golden(kind: EventKind) -> None:
    """The golden must accept what the journal actually writes.

    Byte-equality with a regenerated export only proves the file is self-consistent. It
    passed happily while `payload` resolved to the empty base `P` and the schema rejected
    all 52 kinds — a frozen contract describing none of its own traffic. This is the
    check that has to hold at the `contracts-v1` tag.
    """
    jsonschema.validate(_event(kind), _golden("journal_event"))


def test_golden_rejects_a_payload_from_the_wrong_kind() -> None:
    """Accepting everything would be the same defect wearing the opposite sign."""
    kind, other = EventKind.RUN_START, EventKind.SCOUT_TOOL
    assert KIND_TABLE[kind][0] is not KIND_TABLE[other][0]
    mismatched = {**_event(kind), "payload": _event(other)["payload"]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(mismatched, _golden("journal_event"))


def test_no_retired_fields_survive() -> None:
    """G0 condition 5: no stale `verify_tier`, held-out example, or command-hook field."""
    blob = "".join((GOLDEN_DIR / f"{n}.schema.json").read_text() for n in CONTRACTS)
    for retired in ("verify_tier", "examples_shown", "command_hook", "BUDGET_EXHAUSTED"):
        assert retired not in blob, f"retired field {retired!r} is still in the wire format"
