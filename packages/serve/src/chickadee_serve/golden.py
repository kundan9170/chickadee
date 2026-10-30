"""The wire's contract as one JSON Schema document, pinned to `WIRE_VERSION`.

A client compiles against more than the frames: the stream carries the session log's
payloads as stored, and the routes answer with the facade's own types. All of it is exported
here, built from the types themselves and never restated:

- `frames`: every type in `frames.py`, both streams' frames, the bodies, the small answers;
- `routes`: `routes.ROUTES`, as a client's method table;
- `session_records`: each session kind's payload, from `context.lines`. A kind no type of that
  module parses is `{"x-opaque": true}`: its writer owns it, and a client must not read it;
- `run_records`: a pointer to the run journal's golden, which is a frozen contract already;
- `responses`: the facade types the routes answer with;
- `requests`: the facade types a client sends inside a body (`CoderBody.request`), as the
  server validates them. The bodies carry them as JSON so
  `frames.py` stays core-free; the golden is how a client still sends a typed one.

**The version moves with the contract.** `versions.json` maps every `WIRE_VERSION` ever
released to the sha of the contract it described. `test_wire_goldens.py` refuses a contract
whose sha differs from its version's: change a payload, and the test says to bump the version,
minor for an addition and major for a removal or a change. A released entry is never
rewritten.

`python -m chickadee_serve.golden` regenerates both files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, TypeAdapter

from chickadee.approval.grants import ACTION_GRANTED, ActionGrantedLine
from chickadee.box.request import CoderRequest
from chickadee.context.attachment import AttachmentPreview
from chickadee.context.lines import PARSED
from chickadee.context.questions import LINES as QUESTION_LINES
from chickadee.context.vocabulary import SESSION_KINDS
from chickadee.engine import (
    FileText,
    Forked,
    JobView,
    Liveness,
    Restored,
    RunBlob,
    RunDiff,
    RunHeader,
    SessionBudget,
    SessionInfo,
    Undoable,
    UndoReport,
    WorkProgress,
)
from chickadee.handoff import HandoffResult
from chickadee.handoff_summary import HandoffSummary
from chickadee.journal.metrics import RunMetrics
from chickadee.open_works import OpenWorkInfo
from chickadee.schema.findings import Findings, Unresolved
from chickadee.sessions import SessionTitle
from chickadee.wiring import ModelSwitch, ModelWiring
from chickadee_serve import frames
from chickadee_serve.routes import ROUTES

__all__ = ["GOLDEN", "VERSIONS", "content_sha", "contract", "render", "write"]

SCHEMA_DIR: Final = Path(__file__).resolve().parents[2] / "schema"
GOLDEN: Final = SCHEMA_DIR / "golden" / "wire.schema.json"
VERSIONS: Final = SCHEMA_DIR / "versions.json"

_FRAMES: Final[tuple[type[BaseModel], ...]] = (
    frames.RecordFrame,
    frames.ProgressFrame,
    frames.Hello,
    frames.ErrorBody,
    frames.CreateSessionBody,
    frames.TurnBody,
    frames.TurnAccepted,
    frames.InterruptBody,
    frames.CapBody,
    frames.AnswerBody,
    frames.AnswerTaken,
    frames.UndoBody,
    frames.ForkBody,
    frames.RungBody,
    frames.CoderBody,
    frames.CoderStarted,
    frames.Resumed,
    frames.LocateBody,
    frames.FileBody,
    frames.RenameBody,
    frames.FinalizeBody,
    frames.Hidden,
    frames.TokenFile,
)

_RESPONSES: Final[dict[str, object]] = {
    "SessionInfo": SessionInfo,
    "OpenWorkInfo": OpenWorkInfo,
    "WorkProgress": WorkProgress,
    "SessionBudget": SessionBudget,
    "RunHeader": RunHeader,
    "Undoable": Undoable,
    "UndoReport": UndoReport,
    "Restored": Restored,
    "Forked": Forked,
    "Liveness": Liveness,
    "RunMetrics": RunMetrics,
    "HandoffSummary": HandoffSummary,
    "HandoffResult": HandoffResult,
    "Findings": Findings,
    "Unresolved": Unresolved,
    "ModelWiring": ModelWiring,
    "ModelSwitch": ModelSwitch,
    "JobView": JobView,
    "RunBlob": RunBlob,
    "RunDiff": RunDiff,
    "FileText": FileText,
    "AttachmentPreview": AttachmentPreview,
}
"""The facade types a route answers with. One joins as the facade method returning it lands."""


_REQUESTS: Final[dict[str, type[BaseModel]]] = {
    "CoderRequest": CoderRequest,
}
"""The facade types a body carries as JSON, which the server validates on arrival."""


_TYPED: Final[dict[str, object]] = {
    **PARSED,
    **QUESTION_LINES,
    ACTION_GRANTED: ActionGrantedLine,
    ModelSwitch.KIND: ModelSwitch,
    SessionTitle.KIND: SessionTitle,
}
"""Every session kind a client may read, by its type: the fold's, the question kinds', an
action a grant approved, and a rung's model switched (so a client sees which model a rung calls
from when), and the session's name.
"""


def _schema(of: object) -> dict[str, Any]:
    return TypeAdapter(of).json_schema(mode="serialization")


def contract() -> dict[str, Any]:
    """The whole wire contract at this `WIRE_VERSION`."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "wire": frames.WIRE_VERSION,
        "frames": {m.__name__: m.model_json_schema(mode="serialization") for m in _FRAMES},
        "routes": [
            {
                "method": r.method,
                "path": r.path,
                "capability": r.capability,
                "mutating": r.mutating,
                "stream": r.stream,
            }
            for r in ROUTES
        ],
        "session_records": {
            kind: _schema(_TYPED[kind]) if kind in _TYPED else {"x-opaque": True}
            for kind in SESSION_KINDS
        },
        "run_records": {
            "$comment": "a run's records are journal events: their payload per kind is in "
            "packages/core/schema/golden/journal_event.schema.json, a frozen contract"
        },
        "responses": {name: _schema(of) for name, of in _RESPONSES.items()},
        "requests": {
            name: of.model_json_schema(mode="validation") for name, of in _REQUESTS.items()
        },
    }


def render(doc: dict[str, Any] | None = None) -> str:
    return json.dumps(doc if doc is not None else contract(), indent=2, sort_keys=True) + "\n"


def content_sha(doc: dict[str, Any]) -> str:
    """The contract's sha without its version: what a version must not change under it."""
    body = {k: v for k, v in doc.items() if k != "wire"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def write() -> str:
    """Regenerate the golden, and record the version's sha. Refuses to move a released
    version's sha: a changed contract is a new version."""
    doc = contract()
    sha = content_sha(doc)
    versions: dict[str, str] = (
        json.loads(VERSIONS.read_text(encoding="utf-8")) if VERSIONS.exists() else {}
    )
    known = versions.get(frames.WIRE_VERSION)
    if known is not None and known != sha:
        raise SystemExit(
            f"the wire contract changed under WIRE_VERSION {frames.WIRE_VERSION}: bump it in "
            f"frames.py (minor for an addition, major for a removal or change), then run again"
        )
    versions[frames.WIRE_VERSION] = sha
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(render(doc), encoding="utf-8")
    VERSIONS.write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha


if __name__ == "__main__":
    print(f"wire {frames.WIRE_VERSION}: {write()}")
