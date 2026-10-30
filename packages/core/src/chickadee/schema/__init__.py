"""The frozen seams every other part is written against.

This is the only import surface other packages use. Per the module map's DoD, it exports
exactly the names the design docs reference — nothing else — so that what is public is a
decision rather than an accident of what happened to be importable.

Frozen: only additive changes are allowed.

"""

from __future__ import annotations

from chickadee.schema.adapter import (
    CallForm,
    Completion,
    ParseError,
    Prompt,
    ReasoningSpec,
    RetryParams,
    StopReason,
    ToolCall,
    Usage,
)
from chickadee.schema.authorization import (
    TIER_ORDER,
    ActionAuthorization,
    Confirmation,
    FactHit,
    RiskClass,
    Tier,
)
from chickadee.schema.execution import (
    ExecRequest,
    ExecResult,
    PolicyClass,
    SandboxLevel,
)
from chickadee.schema.findings import Findings, Pointer, Unresolved
from chickadee.schema.ids import (
    RUN_ID_RE,
    SHA256_RE,
    TASK_ID_RE,
    TASK_KEY_RE,
    TS_RE,
    new_run_id,
    sha256_hex,
)
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    CommandBinding,
    GrantScope,
    InteractionPort,
    Option,
    Question,
    QuestionKind,
    StopAnswer,
    StopDecision,
    StopKind,
)
from chickadee.schema.isolation import GitHandoffChoice, IsolationSpec
from chickadee.schema.journal import (
    ENVELOPE_VERSION,
    KIND_TABLE,
    EventKind,
    JournalEvent,
    Scope,
)

SCHEMA_VERSION = 2
"""Contracts v2: what `run_start.schema_version` records."""

# Grouped by contract rather than sorted: a reader checking what contract 4 exposes
# wants those names together, and this list is the public surface's definition.
__all__ = [  # noqa: RUF022
    "ENVELOPE_VERSION",
    "KIND_TABLE",
    "RUN_ID_RE",
    "SCHEMA_VERSION",
    "SHA256_RE",
    "TASK_ID_RE",
    "TASK_KEY_RE",
    "TS_RE",
    # contracts 1, 2 and 4 (Packet, PlanDAG, Patch) retired with the packet loop
    # contract 3 — JournalEvent
    "EventKind",
    "JournalEvent",
    "Scope",
    # contract 5 — the model call (ModelClient is client/request.py's)
    "Completion",
    "CallForm",
    "ParseError",
    "Prompt",
    "ReasoningSpec",
    "RetryParams",
    "StopReason",
    "ToolCall",
    "Usage",
    # contract 6 — Interaction
    "ActionApproval",
    "Answer",
    "AnswerSource",
    "ApprovalDecision",
    "CommandBinding",
    "GrantScope",
    "InteractionPort",
    "Option",
    "Question",
    "QuestionKind",
    "ApprovalStop",
    "StopAnswer",
    "StopDecision",
    "StopKind",
    # contract 7 — Exec
    "ExecRequest",
    "ExecResult",
    "PolicyClass",
    "SandboxLevel",
    # contract 8 — Findings
    "Findings",
    "Pointer",
    "Unresolved",
    # contract 9 — Routing, retired with the packet loop
    # contract 10 — Authorization
    "ActionAuthorization",
    "Confirmation",
    "FactHit",
    "RiskClass",  # retired; nothing constructs one
    "Tier",
    "TIER_ORDER",
    # contract 11 — Isolation
    "GitHandoffChoice",
    "IsolationSpec",
    # ids
    "new_run_id",
    "sha256_hex",
]
