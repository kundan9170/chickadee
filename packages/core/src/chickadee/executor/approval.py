"""Concrete-spec hash, D40 risk classification, authorization validity.

The asymmetry in §6.3 is what drives all of this: git reverts `pyproject.toml`, but nothing
un-installs a package, un-runs a migration, or un-writes a database. Action packets
therefore carry weaker rollback guarantees than code packets, and the design says so rather
than pretending — which is why some of them have to be confirmed before they happen, since
afterwards is too late.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from chickadee.approval.commands import CommandClass, shipped
from chickadee.executor.types import ActionSpec
from chickadee.schema import sha256_hex


class Risk(StrEnum):
    LOW = "low"
    HIGH = "high"
    """Destructive, networked, credential-bearing, or scope-expanding."""


def spec_text(spec: ActionSpec) -> str:
    return spec.command if isinstance(spec.command, str) else " ".join(spec.command)


def concrete_hash(spec: ActionSpec) -> str:
    """A stable digest of everything a user would have been agreeing to.

    Hashes the *effect surface and network* alongside the command, because the whole point
    is to detect an elaboration that changed what the action does. A digest over the command
    alone would let `effects` widen from `pyproject.toml` to `**` without invalidating the
    approval anyone gave.
    """
    payload = json.dumps(
        {
            # The structured value, not the joined text. `("printf", "%s", "a b")` and
            # `("printf", "%s a", "b")` join to the same string and execute differently —
            # verified colliding — so a text digest let an approval cover a command nobody
            # approved. The `kind` tag keeps a one-element tuple distinct from a shell line.
            "command_kind": "argv" if isinstance(spec.command, tuple) else "shell",
            "command": list(spec.command) if isinstance(spec.command, tuple) else spec.command,
            "effects": sorted(spec.effects),
            "network": spec.network.open,
            "idempotent": spec.idempotent,
        },
        sort_keys=True,
    )
    return sha256_hex(payload.encode())


_RISK_REASONS: dict[CommandClass, str] = {
    CommandClass.DESTRUCTIVE: "destructive command",
    CommandClass.PUSH: "publishes to a remote",
    CommandClass.CREDENTIAL: "references a credential",
    CommandClass.SCOPE_EXPANDING: "expands privileges or scope",
}


def classify_risk(spec: ActionSpec) -> tuple[Risk, tuple[str, ...]]:
    """`(risk, reasons)`. Reasons are returned so a prompt can say *why* it is asking. What
    the command is comes from the command table, the one classifier."""
    table = shipped()
    found = (
        table.classify_line(spec.command)
        if isinstance(spec.command, str)
        else table.classify(spec.command)
    )
    classes = found.classes if found is not None else frozenset[CommandClass]()
    reasons = [why for cls, why in _RISK_REASONS.items() if cls in classes]
    if not spec.network.off:
        reasons.append("network access")
    if any(glob in ("**", "*", "/**") for glob in spec.effects):
        reasons.append("undeclared effect surface")
    return (Risk.HIGH if reasons else Risk.LOW), tuple(reasons)


@dataclass(frozen=True)
class Authorization:
    approved_hash: str
    risk: Risk
    confirmed: bool = False


def still_valid(auth: Authorization, spec: ActionSpec) -> tuple[bool, str]:
    """Whether a prior approval still covers this concrete spec.

    A material change at dispatch time invalidates it. "Material" is defined by the hash, so
    it cannot drift by argument: if any of command, effects, network or idempotency moved,
    the thing being approved is not the thing that was approved.
    """
    current = concrete_hash(spec)
    if current != auth.approved_hash:
        return False, "the concrete specification changed since approval (D40)"
    risk, reasons = classify_risk(spec)
    if risk is Risk.HIGH and not auth.confirmed:
        return False, f"high-risk action needs a second confirmation: {'; '.join(reasons)}"
    return True, ""
