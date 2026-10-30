"""The pinned set: which loaded skills stay in context.

**Folded from the log, never held.** A load writes `skill_loaded` with the `call_id` of the
`load_skill` call whose tool line carries the body; the person's unload writes
`skill_unloaded`. `pinned` reads both, so a resumed session pins what it pinned, a rewind past a
load unpins it with the tool line it hid, and nothing in memory can disagree with the record.

**Compaction asks at every cut** (`pinned_seqs`), and lifts those tool lines intact into the
standing region. That is the whole of unload: a skill whose pin
is cleared is not in the set at the next cut, so its bytes are simply not re-emitted. The slot
frees at once; the bytes leave at the next cut; no rebuild is paid for.

**Unload is the person's, and only theirs.** `unload_skill` is called by the CLI's
`/unload` and by nothing a model can reach: no tool names it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic import TypeAdapter, with_config

from chickadee.context.lines import LINE, SessionKind
from chickadee.context.session import Session, ToolLine
from chickadee.skills.discover import Origin, Scope

__all__ = [
    "SKILL_LOADED",
    "SKILL_PINNED_MAX",
    "SKILL_UNLOADED",
    "Pin",
    "SkillLoaded",
    "SkillUnloaded",
    "pinned",
    "pinned_seqs",
    "unload_skill",
]

SKILL_PINNED_MAX = 4
"""The real always-on budget: past it a load is refused, never one evicted."""

SKILL_LOADED: SessionKind = "skill_loaded"
SKILL_UNLOADED: SessionKind = "skill_unloaded"


@with_config(LINE)
@dataclass(frozen=True)
class SkillLoaded:
    """A skill body entered context and is pinned."""

    KIND: ClassVar[SessionKind] = SKILL_LOADED
    call_id: str
    """The `load_skill` call whose tool line carries the body: what a cut lifts."""
    name: str
    scope: Scope
    origin: Origin
    path_sha: str
    content_sha: str
    shadowed: bool
    """Whether a skill of the same name in a lower scope was passed over."""


@with_config(LINE)
@dataclass(frozen=True)
class SkillUnloaded:
    """The person cleared a pin (`/unload`)."""

    KIND: ClassVar[SessionKind] = SKILL_UNLOADED
    name: str


@dataclass(frozen=True)
class Pin:
    name: str
    call_id: str
    content_sha: str
    tool_seq: int | None
    """The tool line carrying the body, or None while the load's call is still running."""


_LOADED = TypeAdapter(SkillLoaded)
_UNLOADED = TypeAdapter(SkillUnloaded)


def pinned(session: Session) -> tuple[Pin, ...]:
    """The skills pinned now, in the order they were loaded.

    A load whose call did not run pins nothing: the session ended or the person stopped the
    turn after the load was recorded and before its body was, so the call's line says
    `INTERRUPTED` or `CANCELLED` (`ran: false`). Counting it would lift that text as the
    skill and answer every reload "already loaded", so the body could never come back."""
    pins: dict[str, Pin] = {}
    tools: dict[str, ToolLine] = {}
    for event, record in session.state.lines():
        if isinstance(record, ToolLine):
            tools[record.call_id] = record
        elif event.kind == SKILL_LOADED:
            line = _LOADED.validate_json(session.blob(event.payload_sha))
            pins[line.name] = Pin(line.name, line.call_id, line.content_sha, None)
        elif event.kind == SKILL_UNLOADED:
            pins.pop(_UNLOADED.validate_json(session.blob(event.payload_sha)).name, None)
    held: list[Pin] = []
    for p in pins.values():
        answered = tools.get(p.call_id)
        if answered is None:
            held.append(p)  # the load's call is still running: its slot is taken
        elif answered.ran:
            held.append(Pin(p.name, p.call_id, p.content_sha, answered.seq))
    return tuple(held)


def pinned_seqs(session: Session) -> tuple[int, ...]:
    """The tool lines a cut lifts (`compact(..., pinned=)`), re-read at every cut."""
    return tuple(p.tool_seq for p in pinned(session) if p.tool_seq is not None)


def unload_skill(session: Session, name: str) -> str:
    """Clear `name`'s pin: the whole state change. Returns what the person is told."""
    with session.lock:
        if name not in {p.name for p in pinned(session)}:
            return f"{name} is not loaded"
        session.append(SkillUnloaded(name))
    return f"{name} unloaded · leaves context at the next compaction"
