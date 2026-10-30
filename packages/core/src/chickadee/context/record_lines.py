"""A session-log line that carries one frozen journal record.

Some of what a session does is a journal record by nature: the scout's records when `locate`
runs from Zeus, and a declared Zeus write's D40 lifecycle. Neither belongs to a
run, so each is written to the session log, one line per record, as `{"event": <kind>,
"payload": {...}}`, the payload the same frozen model the run journal would hold. `RecordLine`
is that shape and its reading, once; each kind is a subclass naming its `KIND`.

A call that left the machine on a model's behalf is one too:
`ExternalCallLine`, written by `external.policy` and read back by the citation check, never
folded here.

`BoxCallLine`, Athena's calls before her run had a journal, retired with the design gate that
made them: she makes none outside her run now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, cast

from pydantic import SerializeAsAny, model_validator, with_config

from chickadee.context.vocabulary import LINE, SessionKind
from chickadee.schema.journal import KIND_TABLE, EventKind, P

__all__ = [
    "AGENT",
    "EXTERNAL_CALL",
    "SHELL_EFFECT",
    "AgentLine",
    "ExternalCallLine",
    "RecordLine",
    "ShellEffectLine",
]

SHELL_EFFECT: Final[SessionKind] = "shell_effect"
AGENT: Final[SessionKind] = "agent"
EXTERNAL_CALL: Final[SessionKind] = "external_call"


@with_config(LINE)
@dataclass(frozen=True)
class RecordLine:
    """One journal record on the session log, read as the journal reads it: its kind, and the
    payload `KIND_TABLE` types it by."""

    event: EventKind
    payload: SerializeAsAny[P]

    @model_validator(mode="before")
    @classmethod
    def _typed(cls, raw: object) -> object:
        if not isinstance(raw, dict):
            return raw
        line = cast("dict[str, object]", raw)
        if not isinstance(line.get("payload"), dict):
            return line
        try:
            model, _ = KIND_TABLE[EventKind(line.get("event"))]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"unknown event kind {line.get('event')!r}") from exc
        return {**line, "payload": model.model_validate(line["payload"])}


@with_config(LINE)
@dataclass(frozen=True)
class ShellEffectLine(RecordLine):
    """One D40 record of Zeus's shell on the session log (`zeus/writes.SessionEffects`): a
    step's or a job's intent, start or completion. Folded here, so the session knows which jobs
    are running from its log alone (a running job is not a safe point)."""

    KIND: ClassVar[SessionKind] = SHELL_EFFECT


@with_config(LINE)
@dataclass(frozen=True)
class AgentLine(RecordLine):
    """One record of a spawned agent's round on the session log (`agents/spawn.py`): its
    `agent_start`, each `agent_message`, its `agent_report`. Folded here, so the session knows
    its agents, and what each was told and answered, from its log alone."""

    KIND: ClassVar[SessionKind] = AGENT


@with_config(LINE)
@dataclass(frozen=True)
class ExternalCallLine(RecordLine):
    """One call through the external-call policy (`external/policy.py`): an
    `EventKind.EXTERNAL_CALL` record, its result a blob of the same store. Its writer's kind,
    not folded: what reads it back is the citation check, by the blob it names."""

    KIND: ClassVar[SessionKind] = EXTERNAL_CALL
