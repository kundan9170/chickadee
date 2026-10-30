"""What every session-log line is written in: the kinds, the
config each line type validates with, the one encoding, and the field types several line types
share. A leaf: it imports nothing from `context/`, so any line type, in this package or beside
its writer, can be written in it. `lines.py` re-exports every name here."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, Protocol, get_args

from pydantic import AfterValidator, ConfigDict, Field, JsonValue, PlainSerializer

__all__ = [
    "AUTHORS",
    "LINE",
    "RAISERS",
    "SESSION_KINDS",
    "Author",
    "Cost",
    "Count",
    "Envelope",
    "Frozen",
    "Line",
    "PayloadError",
    "Raiser",
    "Scope",
    "SessionKind",
    "canonical_json",
]


SessionKind = Literal[
    "turn",
    "tool",
    "constraint",
    "rejected",
    # §5.13.7 writes concern lines as `"kind":"concern"`, and the list of kinds omits it.
    # The example is the one that states what is stored; the list is the one that forgot.
    "concern",
    "fork",
    "episode",
    "compaction",
    "workspace",
    "work",  # D204 decision 4a: the session's open work folder, opened and closed
    # D121/D122 — the box's records and the resolved limits (
    # 35-limits.md §4). Session-log lines, never run-journal events: those kinds are frozen.
    "packet_record",
    "design_record",
    "summary",
    "note",
    "plan",  # D186: a design's plan, by its blob, so a design is decomposed once
    "limits",
    # A box compaction's trace. Retired: it moved to the run journal, so
    # nothing writes one; an older log's lines are kept, never read.
    "box_compaction",
    # D97-D99 — Zeus's loop: the prefix's sha, the turn frame, and a
    # turn's bound. The loop's facts are session-log lines; the run journal has no kind.
    "prefix",
    "frame",
    "bound",
    # D132 — a `locate` from Zeus records the scout's frozen journal payloads here, one line
    # each, as `{"event": <kind>, "payload": {...}}` (the `Locator`'s sink).
    "scout",
    # D188 decision 6 — a declared Zeus write's D40 lifecycle, one record a line, in
    # the `scout` line's shape (`context/record_lines.py`); its type is its writer's.
    "shell_effect",
    # D213 — a spawned agent's round: its start, each message of its conversation
    # and its report, in the `scout` line's shape (`context/record_lines.py`). Folded, so the
    # session knows its agents from its log alone, and a follow-up after a resume rebuilds one.
    "agent",
    # D191 — one of Athena's calls before the run had a journal (the design gate's
    # plan). Retired with that gate: nothing writes one; kept, never read.
    "box_call",
    # D141 — one Zeus model call's charge. The session's spend is a projection of these and
    # each episode's journal, stored nowhere as a total.
    "spend",
    # D143, D144 — the person's cap for the session family, or `null` when they lifted it.
    # The family's latest line is its cap; with none, the engine config's.
    "cap",
    # D128, D185 — the person moved a rung to another catalog model, and a session the eval
    # harness pinned, which refuses that and a cap. Session-log lines, as `cap` is.
    "model_switched",
    "pinned",
    # D239 — the session's name: the `title` role's, written once after its first turn, or
    # the person's (`/rename`). Its type is `sessions.SessionTitle`, beside its writer.
    "title",
    "code_snapshot",  # D162, D179: a turn's snapshot of a workspace; its type is its writer's
    # D162 — a rewind is a line, never a deletion: `undo` takes turns back, `redo` reverses
    # an `undo` (`rewind.py` says what the fold hides).
    "undo",
    "redo",
    # CL-9, `30-serve.md` §6 — a question put to the person, written before the wait, and its
    # answer with who gave it. The run journal carries the box's own pair as well; these carry
    # the whole frozen `Question` or `ActionApproval`, so a frontend can render what was asked.
    "question_asked",
    "question_answered",
    # What the person decided at a design's gate. Retired with the gate:
    # nothing writes one; an older log's lines are kept, never read.
    "plan_decision",
    # D183 — the knowledge layer: a skill body loaded and pinned, a pin the
    # person cleared, and a third-party project skill crossing the trust boundary. Their
    # types are `skills/pin.py`'s; the pinned set is folded from them at every cut.
    "skill_loaded",
    "skill_unloaded",
    "untrusted_read",
    # D188 decision 3 — the session's handoff, write-ahead: the intent before any git change,
    # its completion after. Their types are `session_handoff.py`'s, beside their writers.
    "handoff_intended",
    "handoff_completed",
    # D165 §3 — *always for this session*: one grant the person gave, folded by
    # `approval/grants.py` and dropped at a fork.
    "command_grant",
    # D192 — an action a standing grant approved: one line, never a question.
    "action_granted",
    # D227 decision 15 — a hook's verdict, error or bound, and the set a session bound; its
    # type is `hooks/sinks.HookLine`, beside its writer.
    "hook",
    # D232 decision 1 — a call that left the machine on a model's behalf (a web search or
    # fetch), one record a line in the `scout` line's shape; its type is
    # `external/policy.ExternalCallLine`, beside its writer.
    "external_call",
    # D240 §2.6 — a message from Zeus to the team about the session's live or suspended run,
    # citing the person's turns it came from; the run delivers it.
    "tell",
]
SESSION_KINDS: Final[tuple[SessionKind, ...]] = get_args(SessionKind)


class PayloadError(ValueError):
    """A payload that does not have the shape its kind requires. Raised while folding; the
    fold turns it into `SessionCorrupt` naming the seq."""


def canonical_json(payload: Mapping[str, object]) -> bytes:
    """The one encoding of a payload. Sorted keys and no whitespace, so equal payloads share
    a blob and a replayed fold sees the same bytes the live one did."""
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


LINE: Final = ConfigDict(
    strict=True, extra="ignore", allow_inf_nan=False, ser_json_inf_nan="constants"
)
"""Every line type's config. Nested dataclasses (`Handle`, `Shape`) take it from the line.
A key the log has always been allowed to omit has its default in `Field(default=...)`, which
pydantic reads and the dataclass's own `__init__` does not, so a writer still names it.
A non-finite number renders as itself, for the reparse to refuse: pydantic's default renders
it `null`, and a `null` cap is a lift."""


def _frozen(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _as_written(value: float) -> float:
    """A number goes out as the writer gave it: `1` stays `1`, which the log has always
    held, rather than becoming `1.0`."""
    return value


Frozen = Annotated[
    Mapping[str, JsonValue],
    AfterValidator(_frozen),
    PlainSerializer(dict, return_type=dict[str, JsonValue]),
]
"""A JSON object a record holds read-only, as `MappingProxyType`."""
Count = Annotated[int, Field(ge=0)]
Cost = Annotated[float, Field(ge=0), PlainSerializer(_as_written, return_type=int | float)]
Envelope = Annotated[int, Field(exclude=True)]
"""Supplied by the line's envelope (its `seq`), never by the payload."""

Scope = Literal["session", "design", "packet_set"]
"""ONE type, three users: a constraint, a checkpoint's `ConstraintProposal` and a
rejected alternative. They scope to the same three things because they are said in
the same three places."""

Author = Literal["zeus", "user"]
AUTHORS: Final[tuple[Author, ...]] = get_args(Author)

Raiser = Literal["zeus", "coder", "critic", "user"]
RAISERS: Final[tuple[Raiser, ...]] = get_args(Raiser)


class Line(Protocol):
    """A line any package writes: a dataclass validated with `LINE`, naming its kind."""

    KIND: ClassVar[SessionKind]
