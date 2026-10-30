"""JSON Schema export.

Draft 2020-12, one file per contract, committed under `schema/golden/`. CI fails if
regeneration differs from the goldens — that is what makes "frozen at G0" mean something
rather than being a statement of intent. A field added to a wire model without a golden
update is caught by the diff, not by a reviewer noticing.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from pydantic import BaseModel

from chickadee.schema.journal import KIND_TABLE, JournalEvent

# Of the five contracts, the journal is the one that outlived the packet loop: `Packet`,
# `PlanDAG`, `PatchSet` and `ApplyResult` retired with it, their goldens with
# them. Contracts 6-11 are frozen too, but their JSON Schemas are exported with the parts that
# own their semantics, exactly as the graph types are.
CONTRACTS: dict[str, type[BaseModel]] = {
    "journal_event": JournalEvent,
}

GOLDEN_DIR = pathlib.Path(__file__).resolve().parents[3] / "schema" / "golden"


def _journal_payload_rules() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Per-kind payload defs and the `if kind then payload` rules that select them.

    `JournalEvent.payload` is annotated as the base `P` so that `SerializeAsAny` writes
    the concrete payload in full. That is right for Python and useless for JSON Schema:
    the generator sees only `P`, which has no properties and forbids extras, so the
    exported schema rejects every event the harness actually writes.

    The kind-to-payload mapping lives on the envelope rather than inside the payload, so
    it cannot be a pydantic discriminated union. JSON Schema expresses it directly as a
    conditional per kind, generated from the one table that already defines it.
    """
    defs: dict[str, Any] = {}
    rules: list[dict[str, Any]] = []
    for kind, (model, _scope) in KIND_TABLE.items():
        js = model.model_json_schema(mode="serialization", ref_template="#/$defs/{model}")
        defs.update(js.pop("$defs", {}))
        defs[model.__name__] = js
        rules.append(
            {
                "if": {"properties": {"kind": {"const": kind.value}}, "required": ["kind"]},
                "then": {"properties": {"payload": {"$ref": f"#/$defs/{model.__name__}"}}},
            }
        )
    return defs, rules


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    doc = model.model_json_schema(mode="serialization")
    if model is JournalEvent:
        defs, rules = _journal_payload_rules()
        # The envelope only requires an object here; the conditionals below decide which
        # one. Leaving the `$ref` to `P` in place would forbid every real payload field.
        doc["properties"]["payload"] = {"type": "object"}
        doc.setdefault("$defs", {}).pop("P", None)
        doc["$defs"].update(defs)
        doc["allOf"] = rules
    doc["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return doc


def render(name: str) -> str:
    """One contract's schema as the exact bytes that belong in its golden file."""
    return json.dumps(schema_for(CONTRACTS[name]), indent=2, sort_keys=True) + "\n"


def export_all(out_dir: pathlib.Path | str | None = None) -> list[pathlib.Path]:
    target = pathlib.Path(out_dir) if out_dir is not None else GOLDEN_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for name in CONTRACTS:
        path = target / f"{name}.schema.json"
        path.write_text(render(name), encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    for p in export_all():
        print(p)
