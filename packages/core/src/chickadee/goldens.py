"""JSON Schema goldens for a package's own contracts.

A contract a part owns freezes at its gate and is additive-only from then. Its golden is
the reviewable record of that: a committed schema the tests compare the live type against.
Every part that owns contracts declares one `GoldenSet`; the rendering and the write live
here once, so two parts cannot drift in how a golden is spelled.

`schema/export.py` is not a caller: it is frozen with `contracts-v1`, and its journal
goldens carry rules of their own.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel

from chickadee.durable import write_atomic

DIALECT = "https://json-schema.org/draft/2020-12/schema"


def schema_for(model: type[BaseModel]) -> dict[str, object]:
    doc = model.model_json_schema(mode="serialization")
    doc["$schema"] = DIALECT
    return doc


@dataclass(frozen=True)
class GoldenSet:
    """One part's contracts, by golden name, and the directory their goldens live in."""

    contracts: Mapping[str, type[BaseModel]]
    directory: pathlib.Path

    def render(self, name: str) -> str:
        return json.dumps(schema_for(self.contracts[name]), indent=2, sort_keys=True) + "\n"

    def path(self, name: str) -> pathlib.Path:
        return self.directory / f"{name}.schema.json"

    def export_all(self, out_dir: pathlib.Path | str | None = None) -> list[pathlib.Path]:
        target = pathlib.Path(out_dir) if out_dir is not None else self.directory
        target.mkdir(parents=True, exist_ok=True)
        written: list[pathlib.Path] = []
        for name in self.contracts:
            path = target / f"{name}.schema.json"
            write_atomic(path, self.render(name).encode())
            written.append(path)
        return written
