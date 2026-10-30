"""Writes `chickadee_wire/models.py` from the wire golden: the facade types the routes
answer with, the ones a client sends inside a body, and the payload of every session record
kind a client may read.

The golden is the contract. A hand-written copy of these types would drift from the
facade without anything noticing, so the client's types are generated from it, committed, and
held to it by `test_models_generated.py`, which runs this into a temporary file and compares.

Each response is named as the golden names it; each session record is `<Kind>Record` after its
kind, because three kinds are unions with no title of their own. Their `$defs` merge into one
document, so a type two of them share (`Tier`, `FactHit`) is one class; two different
definitions under one name stop the generator rather than one silently winning.

    uv run python packages/wire/tools/generate_models.py # rewrites models.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[3]
GOLDEN: Final = ROOT / "packages" / "serve" / "schema" / "golden" / "wire.schema.json"
JOURNAL: Final = ROOT / "packages" / "core" / "schema" / "golden" / "journal_event.schema.json"
"""A run's records are journal events; their payload per kind is this frozen contract (the
wire golden's `run_records` points here)."""
MODELS: Final = ROOT / "packages" / "wire" / "src" / "chickadee_wire" / "models.py"

HEADER: Final = (
    (
        "# A nested model's default is written as its JSON (`= {{...}}`) and validated into\n"
        "# the model on construction (`validate_default=True`); pyright reads the literal,\n"
        "# not the validation.\n"
        "# pyright: reportAssignmentType=false\n"
    )
    + '''"""The wire's response and session-record types. GENERATED: do not edit.

`packages/wire/tools/generate_models.py` writes this from the golden,
`packages/serve/schema/golden/wire.schema.json` at `WIRE_VERSION` {wire}, and
`test_models_generated.py` fails when the two differ. Every class is a `WireModel`: frozen,
and ignoring a field it does not declare, so a server a minor version ahead still reads.
`SESSION_RECORDS` maps each readable session kind to its payload's type; a kind it lacks is
opaque, its writer's alone (`golden.py`).
"""

'''
)


def record_name(kind: str) -> str:
    return "".join(part.capitalize() for part in kind.split("_")) + "Record"


def _nullable(node: Any) -> Any:
    """`anyOf: [<constrained scalar>, null]`, as pydantic writes `str | None` with a
    constraint, rewritten as `type: [<scalar>, "null"]`. The generator turns the first form
    into a `RootModel` whose default is None, a type error; the second it reads as meant. An
    enum keeps `null` among its values too, or the generator reads `type` alone and writes a
    `Literal` that refuses the `None` the field allows."""
    if isinstance(node, list):
        return [_nullable(v) for v in node]
    if not isinstance(node, dict):
        return node
    node = {k: _nullable(v) for k, v in node.items()}
    options = node.get("anyOf")
    if isinstance(options, list) and len(options) == 2 and {"type": "null"} in options:
        (other,) = [o for o in options if o != {"type": "null"}]
        scalar = other.get("type") in ("string", "integer", "number")
        if scalar and len(other) > 1 and "$ref" not in other:
            rest = {k: v for k, v in node.items() if k != "anyOf"}
            merged = {**rest, **other, "type": [other["type"], "null"]}
            if "enum" in other:
                merged["enum"] = [*other["enum"], None]
            return merged
    return node


def _merge(defs: dict[str, Any], name: str, schema: dict[str, Any]) -> None:
    if name in defs and defs[name] != schema:
        raise SystemExit(f"two definitions named {name!r} differ; name one of them apart")
    defs[name] = schema


@dataclass(frozen=True)
class Union:
    """A record kind whose payload is one of several types, told apart by one field."""

    name: str
    members: tuple[str, ...]
    discriminator: str


@dataclass(frozen=True)
class Document:
    schema: dict[str, Any]
    """One JSON Schema whose properties name every type to generate."""
    session_kinds: dict[str, str]
    """Each readable session kind's payload type."""
    run_kinds: dict[str, str]
    """Each run event kind's payload type."""
    unions: list[Union]
    """The record kinds that are discriminated unions: the generator would inline those into
    a property and lose their names, so their members are generated and the union itself is
    written here."""


def run_payloads(journal: dict[str, Any]) -> dict[str, str]:
    """Each event kind's payload type, from the journal golden's `if kind then payload`."""
    return {
        branch["if"]["properties"]["kind"]["const"]: branch["then"]["properties"]["payload"][
            "$ref"
        ].removeprefix("#/$defs/")
        for branch in journal["allOf"]
    }


def document(golden: dict[str, Any], journal: dict[str, Any]) -> Document:
    defs: dict[str, Any] = {}
    named: dict[str, Any] = {}
    kinds: dict[str, str] = {}
    unions: list[Union] = []
    sources = [*golden["responses"].items(), *golden["requests"].items()]
    for kind, schema in golden["session_records"].items():
        if schema.get("x-opaque"):
            continue
        kinds[kind] = record_name(kind)
        sources.append((record_name(kind), schema))
    for name, schema in sources:
        body = dict(schema)
        for inner, sub in body.pop("$defs", {}).items():
            _merge(defs, inner, sub)
        if "oneOf" in body:
            members = tuple(ref["$ref"].removeprefix("#/$defs/") for ref in body["oneOf"])
            unions.append(Union(name, members, body["discriminator"]["propertyName"]))
            for member in members:
                named[member] = {"$ref": f"#/$defs/{member}"}
            continue
        body["title"] = name
        _merge(defs, name, body)
        named[name] = {"$ref": f"#/$defs/{name}"}
    for inner, sub in journal["$defs"].items():
        _merge(defs, inner, sub)
    run_kinds = run_payloads(journal)
    for name in run_kinds.values():
        named[name] = {"$ref": f"#/$defs/{name}"}
    doc = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Wire",
        "type": "object",
        "properties": named,
        "$defs": defs,
    }
    return Document(doc, kinds, run_kinds, unions)


def generate(out: Path) -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    journal = json.loads(JOURNAL.read_text(encoding="utf-8"))
    built = document(golden, journal)
    doc = _nullable(built.schema)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "wire.json"
        source.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        raw = Path(tmp) / "models.py"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "datamodel_code_generator",
                "--input",
                str(source),
                "--input-file-type",
                "jsonschema",
                "--output",
                str(raw),
                "--output-model-type",
                "pydantic_v2.BaseModel",
                "--base-class",
                "chickadee_wire._base.WireModel",
                "--extra-fields",
                "ignore",
                "--target-python-version",
                "3.12",
                "--use-annotated",
                "--use-standard-collections",
                "--use-union-operator",
                "--use-schema-description",
                "--use-field-description",
                "--enum-field-as-literal",
                "all",
                "--collapse-root-models",
                "--strict-nullable",
                "--disable-timestamp",
                "--formatters",
                "isort",
            ],
            check=True,
        )
        body = raw.read_text(encoding="utf-8")
    body = _without_wrapper(body)
    aliases = "".join(
        f"\n\n{u.name} = Annotated[\n    {' | '.join(u.members)},\n"
        f"    Field(discriminator={u.discriminator!r}),\n]"
        for u in built.unions
    )
    footer = (
        f"{aliases}\n\n\n"
        + _table("SESSION_RECORDS", built.session_kinds)
        + '"""Each session kind a client may read: its payload\'s type, as an adapter because\n'
        'three kinds are unions."""\n\n\n'
        + _table("RUN_RECORDS", built.run_kinds)
        + '"""Each run event kind: its payload\'s type, from the journal golden."""\n'
    )
    out.write_text(HEADER.format(wire=golden["wire"]) + body + footer, encoding="utf-8")
    # The repository's ruff settings wherever `out` is, so a copy written to a temporary
    # directory (`test_models_generated.py`) is formatted as the committed one is.
    config = ["--config", str(ROOT / "pyproject.toml"), "-q", str(out)]
    imports = ["--select", "I,F401,UP", "--fix"]
    subprocess.run([sys.executable, "-m", "ruff", "check", *imports, *config], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", *config], check=True)


def _table(name: str, kinds: dict[str, str]) -> str:
    rows = "\n".join(
        f'    "{kind}": TypeAdapter[WireModel]({cls}),' for kind, cls in sorted(kinds.items())
    )
    return f"{name}: Final[dict[str, TypeAdapter[WireModel]]] = {{\n{rows}\n}}\n"


_CONSTR: Final = re.compile(r"\bconstr\(([^()]*)\)")


def _without_wrapper(body: str) -> str:
    """The generator's own banner and the `Wire` class that only names the others: neither is
    a type of the wire. `Final` joins the imports for the record tables. A constrained dict
    key, which the generator writes as a `constr(...)` call no type checker accepts in a type,
    is written as the `Annotated[str, StringConstraints(...)]` it means."""
    lines = body.splitlines()
    lines = [ln for ln in lines if not ln.startswith("# generated by datamodel-codegen")]
    lines = [ln for ln in lines if not ln.startswith("#   filename:")]
    text = "\n".join(lines).strip("\n") + "\n"
    start = text.find("\nclass Wire(")
    if start != -1:
        end = text.find("\nclass ", start + 1)
        text = text[:start] + ("\n" + text[end:] if end != -1 else "\n")
    text = text.replace("from typing import ", "from typing import Final, ", 1)
    text, keys = _CONSTR.subn(r"Annotated[str, StringConstraints(\1)]", text)
    extra = "StringConstraints, TypeAdapter, " if keys else "TypeAdapter, "
    return text.replace("from pydantic import ", f"from pydantic import {extra}", 1)


if __name__ == "__main__":
    generate(MODELS)
    print(f"wrote {MODELS.relative_to(ROOT)}")
