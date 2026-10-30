"""Every session-log line, as its type.

The store hands out a line's bytes; the line's type parses them. Pydantic validates the
records as they are, frozen stdlib dataclasses, in strict JSON mode (`LINE`): a bool is never
an int, a float never an int, a string becomes an enum or a `Literal` only when it is a
member, and a cost is finite. An unknown key is ignored, as the hand readers this replaced
ignored it, so a line a newer writer adds a field to still reads.

**One class per shape.** Where a record is its payload plus what the envelope supplies (the
line's `seq`, and for a concern the run it was raised in), the record is the type, and those
fields are never read from the payload or written to it. Where the record is derived from
the payload (a turn makes `Call`s and a `Notice`, a tool result takes its run from the
envelope, a fork's origin is nested, a constraint line carries `supersedes` beside its
constraint), a wire class here names exactly the payload, and the fold builds the record.

**The writer checks what replay will read** (`Session.append`): a line is rendered to its
canonical bytes and parsed back through the same type before it is checked and written, so a
line the fold would refuse is never written.

A kind another package writes (`prefix`, `bound`, `scout`, the box's kinds) has its type beside
its writer; this module checks only that it is a JSON object, which the fold keeps unread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Final, Literal, cast, get_args

from pydantic import (
    BeforeValidator,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    with_config,
)

from chickadee.context import standing
from chickadee.context.handle import Handle
from chickadee.context.record_lines import AgentLine, ShellEffectLine
from chickadee.context.results import CoderOutcome, ImageRef
from chickadee.context.spans import SpanWire
from chickadee.context.vocabulary import (
    AUTHORS,
    LINE,
    RAISERS,
    SESSION_KINDS,
    Author,
    Cost,
    Count,
    Envelope,
    Frozen,
    Line,
    PayloadError,
    Raiser,
    Scope,
    SessionKind,
    canonical_json,
)
from chickadee.context.work_line import WorkLine
from chickadee.context.workspace_line import WorkspaceRef
from chickadee.schema import Tier, ToolCall
from chickadee.schema.isolation import WorkspaceMode
from chickadee.schema.journal import RungName, TellId

__all__ = [
    "AUTHORS",
    "LINE",
    "PAYLOAD_CLASSES",
    "RAISERS",
    "SESSION_KINDS",
    "AssistantTurnP",
    "Attachment",
    "AttachmentKind",
    "Author",
    "Call",
    "CapLine",
    "Concern",
    "ConcernLine",
    "ConstraintP",
    "CutLine",
    "Delivery",
    "Dismissed",
    "EpisodeLine",
    "ForkOrigin",
    "ForkP",
    "FrameLine",
    "HarnessTurnP",
    "Line",
    "PayloadError",
    "Promoted",
    "Raiser",
    "Reconsideration",
    "RedoLine",
    "RejectedAlt",
    "RewoundCode",
    "Scope",
    "SessionKind",
    "SpendLine",
    "ToolP",
    "Touched",
    "TurnP",
    "UndoLine",
    "UserTurnP",
    "attachment_size",
    "canonical_json",
    "parse",
    "render",
]


def _given(value: object) -> object:
    """An optional field the log omits when unset. A writer never wrote it as `null`, so a
    `null` is a shape no writer produced, refused as the hand reader refused it."""
    if value is None:
        raise ValueError("omitted when unset, never null")
    return value


RoleName = Annotated[str, Field(min_length=1)]
"""A cast role's name, as a spend line names whose call it was (`zeus`, `locate`, `residue`)."""


# -- turns --------------------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class Call:
    """One tool call an assistant turn made. Its arguments are kept exactly as the model
    sent them: no layer ever rewrites a call (invariant 19)."""

    call_id: Annotated[str, Field(alias="id")]
    tool: Annotated[str, Field(alias="name")]
    args: Annotated[Frozen, Field(alias="arguments")]
    seq: Envelope = 0
    """The assistant turn that made it."""

    def tool_call(self) -> ToolCall:
        return ToolCall(id=self.call_id, name=self.tool, arguments=dict(self.args))


AttachmentKind = Literal["image", "pdf", "text"]
"""What the person attached, decided from its bytes (`context/attachment.py`)."""

Delivery = Literal["native", "text", "outline", "none"]
"""How Zeus's model was sent an attachment's content when the turn was recorded: `native`, the
image or the PDF itself; `text`, a text file's lines or a PDF's text
layer, whole; `outline`, that text's outline, too long to send whole, read on by its handle;
`none`, the `[attached …]` line alone. Decided once, against his rung's sight and the inline
cap, and kept, so the turn's bytes never change under a cached prefix."""

_ATTACHMENT_ID_RE: Final = r"^A-[1-9][0-9]*$"


def _unset(v: object) -> bool:
    return v is None


def attachment_size(
    kind: AttachmentKind,
    width: int | None,
    height: int | None,
    pages: int | None,
    lines: int | None,
) -> str:
    """Pixels, pages or lines: what a line, a chip and a head show of an attachment."""
    if kind == "image":
        return f"{width}x{height}"
    if kind == "pdf":
        return f"{pages} page{'' if pages == 1 else 's'}"
    return f"{lines} line{'' if lines == 1 else 's'}"


@with_config(LINE)
@dataclass(frozen=True)
class Attachment:
    """A file the person sent with a turn. The
    bytes are a session blob, which `handle` names; a PDF's text layer is a second blob, which
    `text_handle` names. Numbered per session, `A-1` on, never reused."""

    id: Annotated[str, Field(pattern=_ATTACHMENT_ID_RE)]
    kind: AttachmentKind
    name: Annotated[str, Field(min_length=1, max_length=200)]
    media_type: str
    bytes: Annotated[int, Field(ge=1)]
    est_tokens: Annotated[int, Field(ge=0)]
    """What its content cost to send, as `delivery` sent it."""
    delivery: Delivery
    handle: Handle
    """Kind `attachment`, in the session's own store (`run_id` None)."""
    width: Annotated[int | None, Field(exclude_if=_unset)] = None
    height: Annotated[int | None, Field(exclude_if=_unset)] = None
    pages: Annotated[int | None, Field(exclude_if=_unset)] = None
    lines: Annotated[int | None, Field(exclude_if=_unset)] = None
    text_handle: Annotated[Handle | None, Field(exclude_if=_unset)] = None
    """A PDF's text layer; None when it has none, and for any other kind."""

    @property
    def size(self) -> str:
        return attachment_size(self.kind, self.width, self.height, self.pages, self.lines)

    @property
    def label(self) -> str:
        """`A-1 shot.png · image · 1440x900`: how every reader names it."""
        return f"{self.id} {self.name} · {self.kind} · {self.size}"


@with_config(LINE)
@dataclass(frozen=True)
class UserTurnP:
    KIND: ClassVar[SessionKind] = "turn"
    text: str
    role: Literal["user"] = "user"
    workspace: Annotated[WorkspaceMode | None, Field(exclude_if=lambda v: v is None)] = None
    attachments: Annotated[tuple[Attachment, ...], Field(exclude_if=lambda v: not v)] = ()
    """What the person sent with their words. Absent when none, so a turn without
    attachments keeps the bytes it had before the field existed."""


@with_config(LINE)
@dataclass(frozen=True)
class AssistantTurnP:
    KIND: ClassVar[SessionKind] = "turn"
    text: str
    calls: tuple[Call, ...] = ()
    findings: Frozen = MappingProxyType({})
    """`call_id → finding`. Each finding's text is the fold's check, which says which one is
    empty."""
    role: Literal["assistant"] = "assistant"


@with_config(LINE)
@dataclass(frozen=True)
class HarnessTurnP:
    KIND: ClassVar[SessionKind] = "turn"
    trigger: int
    text: str
    handle: str | None = None
    carries: Annotated[Handle | None, Field(exclude_if=lambda v: v is None)] = None
    """The handle the notice names for Zeus to read (`Notice.carries`). Absent,
    it is not written, so every other notice keeps the bytes it had."""
    role: Literal["harness"] = "harness"


TurnP = Annotated[UserTurnP | AssistantTurnP | HarnessTurnP, Field(discriminator="role")]


# -- tool results ----------------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class ToolP:
    """A `tool` line's payload. Its run is the envelope's, so the record (`ToolLine`) is
    built from this and the line."""

    KIND: ClassVar[SessionKind] = "tool"
    call_id: str
    text: str
    ran: bool
    handle: Handle | None = None
    view: tuple[SpanWire, ...] = ()
    supersedes: tuple[str, ...] = ()
    image: ImageRef | None = None
    outcome: Annotated[CoderOutcome | None, Field(exclude_if=lambda v: v is None)] = None
    """A `coder` result's terminal and why. Absent, it is not written, so every other
    tool's line keeps the bytes it had before the field existed."""


# -- episodes, the workspace, cuts, forks, frames ---------------------------------------------


_Seq = Annotated[int, Field(ge=1)]


@with_config(LINE)
@dataclass(frozen=True)
class TellLine:
    """A `tell` line: Zeus's message to the team about run `run_id`, in his words,
    the person its authority. `sources` are the person's turns it came from, each a user turn
    or an `ask_user` answer. Its text is a session blob; the run
    delivers it at its next boundary and journals `tell_delivered`."""

    KIND: ClassVar[SessionKind] = "tell"
    id: TellId
    """`tell-N`, numbered in the session from 1."""
    run_id: str
    text_sha: str
    sources: Annotated[tuple[_Seq, ...], Field(alias="from", min_length=1)]
    key: str | None = None
    """The work it is about, as `WorkProgress` keys it (`work:<dir>/<name>`, `task:<run>`)."""
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class EpisodeLine:
    """Payload of an `episode` line — one per `coder` call, written at dispatch."""

    KIND: ClassVar[SessionKind] = "episode"
    run_id: str
    path: Literal["exact", "planned"]
    """`planned` since the rebuild: a design, or a one-task plan."""
    tier: Annotated[Tier | None, Field(default=None)]
    """The work tier the router applied, on a line written in older versions §12.3 removed it;
    None
    on every line since: the work is approved at its stop."""
    design_sha: Annotated[str | None, Field(default=None)]
    """The work the person approved at its stop: the design's sha, or the one-task brief's.
    None on the exact path, on a T0 task before the rebuild, and for a call nothing was approved
    for."""
    constraints: Annotated[tuple[str, ...], Field(default=())]
    """The bound constraint ids."""
    opened_at_seq: Annotated[int | None, Field(default=None)]
    """The turn that called `coder`; None for a run a host started."""
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class CutLine:
    """One committed compaction: the whole cut in one line, so a crash can
    never leave a block without its standing, or a boundary without its block."""

    KIND: ClassVar[SessionKind] = "compaction"
    boundary: int
    """The first seq kept verbatim. Sticky until the next cut."""
    previous: int
    """The boundary before this one; it bounds this cut's region from below."""
    block: str
    """The assembled region's text, exactly as sent — replayed, never re-rendered, since
    a layer-3 residue cannot be regenerated."""
    standing: str
    """The standing snapshot appended after the block, stamped as of `seq`."""
    lifted: Annotated[tuple[int, ...], Field(default=())]
    """Seqs of the tool lines whose pinned bodies are re-emitted, intact, after the block."""
    facts: Frozen
    """What §5.12 step 12 journals beside the boundary: trigger, basis, per-layer freed.
    Recorded for the reader of the log; assembly never reads it."""
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class ForkOrigin:
    session_id: str
    at_seq: Annotated[int, Field(ge=0)]


@with_config(LINE)
@dataclass(frozen=True)
class ForkP:
    """A child's first line, written by the store when it creates the fork."""

    KIND: ClassVar[SessionKind] = "fork"
    forked_from: ForkOrigin
    reason: str
    at: str | None = None
    """When the fork was made, for people; ordering is `seq`. Nothing reads it, so a line
    without it reads, as it always did; the store always writes it."""


@with_config(LINE)
@dataclass(frozen=True)
class FrameLine:
    """The turn frame: one harness line before the session's first
    user turn, and before the first one after the date changes. The date and the git head
    are here, not in the system prefix, where either would be a cache miss on the largest
    region for the cheapest fact."""

    KIND: ClassVar[SessionKind] = "frame"
    date: str
    """`YYYY-MM-DD`, which is what says whether the next turn needs a frame."""
    text: str
    """The line as the model reads it."""
    head: str | None = None
    """None outside git, and the line then may omit it."""
    branch: str | None = None
    status: Annotated[
        str | None,
        BeforeValidator(_given),
        Field(exclude_if=lambda v: v is None),
    ] = None
    """The open work's status the text carries after its first line
    (`work_progress.status_text`), kept apart so a frame is due again exactly when it changes.
    None while the session has no open work, and on a line written before it: the line then
    has no `status` key, its bytes as before."""
    seq: Envelope = 0

    def framed(self) -> str:
        return f"<harness_notice>\n{standing.escape_frames(self.text)}\n</harness_notice>"


# -- spend and caps ---------------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class SpendLine:
    """One Zeus model call's charge: what the ledger charged,
    as the session log keeps it, so a session's spend outlives the process that spent it.
    `model` is the catalog name that served the call, never the rung; `rung` is the
    rung that made it, named at composition because two rungs may share a model."""

    KIND: ClassVar[SessionKind] = "spend"
    model: str
    tokens_in: Count
    tokens_out: Count
    cached_in: Count
    cache_writes: Count
    cost_usd: Cost
    estimated: bool
    rung: RungName = "zeus"
    """A line written in older versions has none, and only Zeus wrote them then: the former rung
    `zeus`, read through `roles.current_rung`. Every writer since names its rung."""
    role: Annotated[
        RoleName | None,
        BeforeValidator(_given),
        Field(exclude_if=lambda v: v is None),
    ] = None
    """The role whose call it was (`zeus`, `locate`, `residue`): two roles may share a
    rung, so the rung does not say whose prompt this was. None on a line written before, or by
    a view handed to no role, and the line then has no `role` key: its bytes are as before."""
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class CapLine:
    """The person's cap for the session family, or `None` when they lifted it.
    The family's latest line is its cap; with none, the engine config's `hard_cost_usd`."""

    KIND: ClassVar[SessionKind] = "cap"
    hard_cost_usd: Cost | None
    rung: Annotated[
        RungName | None,
        BeforeValidator(_given),
        Field(exclude_if=lambda v: v is None),
    ] = None
    """The rung whose cap this is, beneath the total; `None` is the total itself, and
    the line then has no `rung` key."""
    seq: Envelope = 0


# -- the ledger --------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class ConstraintP:
    """A `constraint` line's payload: the constraint, and the id it supersedes. The ledger
    keeps the constraint (`ledger.Constraint`); `supersedes` is the line's alone."""

    KIND: ClassVar[SessionKind] = "constraint"
    id: str
    text: str
    sources: tuple[Annotated[int, Field(ge=1)], ...]
    author: Author
    scope: Scope
    scope_id: Annotated[str | None, Field(default=None)]
    valid_from: int
    valid_until: Annotated[int | None, Field(default=None)]
    superseded_by: Annotated[str | None, Field(default=None)]
    supersedes: str | None = None


@with_config(LINE)
@dataclass(frozen=True)
class RejectedAlt:
    """A historical fact, never a live rule. There is no field saying whether the
    rejection was right, no anchor — a rejection is about a design, not a path — and no
    status: state is the lines."""

    KIND: ClassVar[SessionKind] = "rejected"
    id: str
    at: int
    """The session-log seq it came from. REQUIRED, and always rendered —
    the age is part of the fact."""
    by: Author
    text: str
    """What was considered, in the proposer's words."""
    reason: str
    """Why it was declined, in the decliner's words. REQUIRED (rule 1)."""
    scope: Scope
    scope_id: Annotated[str | None, Field(default=None)]
    recorded_by: Literal["explicit", "checkpoint"]
    event: Literal["noted"] = "noted"


@with_config(LINE)
@dataclass(frozen=True)
class Reconsideration:
    """`rejected_reconsidered`: we went back to it. The original line is untouched — both are
    true, we rejected it then and reopened it later."""

    KIND: ClassVar[SessionKind] = "rejected"
    rejected_id: Annotated[str, Field(alias="id")]
    reason: str
    by: Author
    event: Literal["reconsidered"] = "reconsidered"
    seq: Envelope = 0


RejectedLine = Annotated[RejectedAlt | Reconsideration, Field(discriminator="event")]


# -- concerns ---------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class Concern:
    """`concern_raised`: the utterance itself. The evidence class is always `claim` — a
    concern is never a fact, and has no field that could make it look like one."""

    KIND: ClassVar[SessionKind] = "concern"
    id: str
    raised_by: Raiser
    text: str
    """VERBATIM, in the raiser's words. Never re-worded."""
    anchor: Annotated[tuple[str, ...], Field(default=())]
    """Paths. Never empty: the fold's check says so, with its own words."""
    event: Literal["raised"] = "raised"
    raised_at: Envelope = 0
    """Session-log seq."""
    run_id: Annotated[str | None, Field(exclude=True)] = None
    """The run it was noticed during, when the box raised it; None at session level. The
    line's envelope carries it."""


@with_config(LINE)
@dataclass(frozen=True)
class Touched:
    """`concern_touched`: a run modified an anchored path. Records that something happened
    NEAR the concern and decides nothing about it."""

    KIND: ClassVar[SessionKind] = "concern"
    concern_id: Annotated[str, Field(alias="id")]
    path: str
    run_id: str | None = None
    event: Literal["touched"] = "touched"
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class Promoted:
    """`concern_promoted`: a registered check now watches it. Watched, not resolved."""

    KIND: ClassVar[SessionKind] = "concern"
    concern_id: Annotated[str, Field(alias="id")]
    check_id: str
    event: Literal["promoted"] = "promoted"
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class Dismissed:
    """`concern_dismissed`: the user, and only the user, said stop displaying it."""

    KIND: ClassVar[SessionKind] = "concern"
    concern_id: Annotated[str, Field(alias="id")]
    event: Literal["dismissed"] = "dismissed"
    seq: Envelope = 0


ConcernLine = Annotated[Concern | Touched | Promoted | Dismissed, Field(discriminator="event")]


# -- rewinds ------------------------------------------------------------------------


@with_config(LINE)
@dataclass(frozen=True)
class RewoundCode:
    """What an undo did to one workspace's code: enough for a redo to
    put it back, and for a reader to know where it happened."""

    key: str
    """The workspace within a turn: a run's id for a worktree, `in-place` for the checkout."""
    mode: WorkspaceMode
    root: str
    """Where the files were restored: the worktree before the handoff, the checkout after it
    or in place."""
    target: str
    """The tree returned to: the first undone turn's `before`."""
    redo_tree: str
    """The tree as it stood before the undo, parked at `redo_ref`, which a redo puts back."""
    redo_ref: str
    branch_head: Annotated[str | None, Field(default=None)]
    """Before the handoff, the worktree branch's commit before the undo moved it back, which a
    redo moves it to again; None elsewhere."""
    restored: Annotated[tuple[str, ...], Field(default=())]
    conflicts_kept: Annotated[tuple[str, ...], Field(default=())]
    """Paths the harness wrote that the person changed after, and kept."""
    committed: Annotated[str | None, Field(default=None)]
    """After the handoff under `undo.on_your_branch = "commit"`, the undo's commit."""


@with_config(LINE)
@dataclass(frozen=True)
class UndoLine:
    """A person took back turns `to_turn` onward: the line is appended, and
    nothing is deleted. With `conversation`, the fold hides what `rewind.REWIND` says a rewind
    hides, from `to_turn` up to this line. With `code`, `workspaces` says what was restored in
    each."""

    KIND: ClassVar[SessionKind] = "undo"
    to_turn: Annotated[int, Field(ge=1)]
    """The seq of the first user turn taken back."""
    code: bool
    conversation: bool
    workspaces: Annotated[tuple[RewoundCode, ...], Field(default=())]
    seq: Envelope = 0


@with_config(LINE)
@dataclass(frozen=True)
class RedoLine:
    """A person took an undo back, or one half of it: after the conversation
    has moved on, only the code half is offered, and the line says which."""

    KIND: ClassVar[SessionKind] = "redo"
    undo_seq: Annotated[int, Field(ge=1)]
    code: bool
    conversation: bool
    seq: Envelope = 0


# -- parse and render ----------------------------------------------------------------------

PARSED: Final[Mapping[SessionKind, object]] = {
    "turn": TurnP,
    "tool": ToolP,
    "episode": EpisodeLine,
    "workspace": WorkspaceRef,
    "work": WorkLine,
    "compaction": CutLine,
    "fork": ForkP,
    "constraint": ConstraintP,
    "rejected": RejectedLine,
    "concern": ConcernLine,
    "frame": FrameLine,
    "spend": SpendLine,
    "cap": CapLine,
    "undo": UndoLine,
    "redo": RedoLine,
    "shell_effect": ShellEffectLine,
    "agent": AgentLine,
    "tell": TellLine,
}
"""The kinds this package folds, each by its type. Every other kind is its writer's."""

PAYLOAD_CLASSES: Final = {
    kind: get_args(inner) or (cast(type, inner),)
    for kind, of in PARSED.items()
    for inner in [get_args(of)[0] if get_args(of) else of]
}
"""The classes each kind's type admits: a class, or an annotated union's (`TurnP`)."""

_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@cache
def _adapter(of: object) -> TypeAdapter[Any]:
    return TypeAdapter(of)


def _say(exc: ValidationError) -> str:
    first = exc.errors(include_url=False)[0]
    where = ".".join(str(p) for p in first["loc"]) or "<payload>"
    return f"{where}: {first['msg']}"


def parse(kind: SessionKind, raw: bytes) -> object | None:
    """A line's payload as its kind's type, or `None` for a kind another package writes,
    which only has to be a JSON object here. A shape the type refuses is `PayloadError`."""
    of = PARSED.get(kind)
    try:
        if of is None:
            _OBJECT.validate_json(raw)
            return None
        return _adapter(of).validate_json(raw)
    except ValidationError as exc:
        raise PayloadError(_say(exc)) from None


def render(line: Line) -> bytes:
    """A line's canonical bytes, by its own type: what the store writes and replay reads."""
    dumped = _adapter(type(line)).dump_python(line, mode="json", by_alias=True)
    return canonical_json(dumped)


def reparse(line: Line, raw: bytes) -> object | None:
    """`raw` back through the type that rendered it: the writer's check of what a reader of
    its kind will read. A context kind comes back as `parse` reads it; another package's is
    checked by its own type and comes back `None`, as `parse` returns it."""
    if line.KIND in PARSED:
        return parse(line.KIND, raw)
    try:
        _adapter(type(line)).validate_json(raw)
    except ValidationError as exc:
        raise PayloadError(_say(exc)) from None
    return None
