"""The wire's types: what crosses, in both directions.

**Core-free by rule.** This module and `routes.py` import pydantic and the standard library,
never `chickadee.*` (`test_wire_is_core_free.py`). They are the contract every client compiles
against, the server included, so a TUI can take them without the engine, and they can move to
a distribution of their own without an edit.

**Two stream frames, told apart by `type`, never guessed**. A `RecordFrame` is a durable
line of a scope, seekable by its `seq`. A `ProgressFrame` is a hint about work in flight
that a correct client may drop entirely: it carries no seq, because asking for progress
from a cursor is a category error.

**`WIRE_VERSION` moves with the contract**. A minor step adds a route, a frame field,
a body field or a record kind; a major step removes or changes one. `golden.py` exports the
whole contract, record payloads included, and `test_wire_goldens.py` refuses a contract that
changed without the version moving (`schema/versions.json`).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

__all__ = [
    "APPROVAL_CHOICES",
    "GRANT_CHOICES",
    "KEEPALIVE_S",
    "PROGRESS_KINDS",
    "STOP_CHOICES",
    "TOKEN_FILE",
    "WIRE_VERSION",
    "AnswerBody",
    "AnswerTaken",
    "AttachLimits",
    "CapBody",
    "CoderBody",
    "CoderStarted",
    "CreateSessionBody",
    "ErrorBody",
    "ErrorCode",
    "FileBody",
    "FinalizeBody",
    "ForkBody",
    "Hello",
    "Hidden",
    "InterruptBody",
    "LocateBody",
    "ProgressFrame",
    "ProgressKind",
    "RecordFrame",
    "RenameBody",
    "Resumed",
    "RungBody",
    "SecretBody",
    "StreamFrame",
    "TokenClass",
    "TokenFile",
    "TurnAccepted",
    "TurnAttachment",
    "TurnBody",
    "TurnWorkspace",
    "UndoBody",
    "UndoOperationBody",
    "UndoTaskBody",
    "approval_choices",
    "decode_frame",
    "encode_frame",
    "major",
    "read_token",
]

WIRE_VERSION: Final = "4.22"
"""`<major>.<minor>`. A client refuses a server whose major differs (`WIRE_MISMATCH`)."""


def major(version: str) -> int:
    """The major part of a wire version, which is all a client compares."""
    head, _, _ = version.partition(".")
    return int(head)


_WIRE = ConfigDict(frozen=True, extra="forbid")
"""Every wire type refuses a field it does not declare: a misspelt key is a 422 the client
sees, never a value silently ignored."""


# -- the stream ----------------------------------------------------------------------------


class RecordFrame(BaseModel):
    """One durable line of a scope, as stored. `scope` is the session or run it is
    numbered in; a client's cursor for that scope is the highest `seq` it has seen."""

    model_config = _WIRE
    type: Literal["record"] = "record"
    scope: str
    seq: Annotated[int, Field(ge=1)]
    at: str
    """When it was written, for people. Ordering is `seq`."""
    kind: str
    """A session kind (`turn`, `tool`, `question_asked` …) or a run's event kind. The goldens
    give each kind's payload; a kind they mark opaque is not for a client to interpret."""
    run_id: str | None = None
    payload: dict[str, JsonValue]
    task_id: str | None = None
    """A run event's task, when it names one."""
    round: int | None = None


ProgressKind = Literal[
    "alive",
    "thinking",
    "tool_call",
    "tool_result",
    "delegation",
    "answer",
    "error",
    "text",
    "thought",
    "tool_started",
    "context",
]
"""`alive` is the keep-alive every stream sends (`KEEPALIVE_S`). The next six are Zeus's turn
events, each a hint that the line it reflects is coming or has committed.

The next three are a reply of any role as it streams, shaped as ACP's
updates. Each names its `role`, the `task` a worker holds, and `reply`, the model call's id,
says `tokens`, what the reply has streamed so far at the harness's estimating rate, and
`calls: "python"` when the role calls its tools by Python blocks:

- `text`: `offset` and `piece`, the reply's text from that offset (ACP's
  `agent_message_chunk`); or `restart`, a new attempt that voids the text shown so far;
- `thought`: `elapsed_s` the model has been thinking, and its reasoning as `offset` and
  `piece` (ACP's `agent_thought_chunk`);
- `tool_started`: `index` in the reply, `tool`, `path` once one can be read, and `size`, the
  bytes of its arguments so far (ACP's `tool_call`, pending then in progress).

`context` is Zeus's context for the status line (decision 8): `used` tokens of `window` on
`model`, the catalog model that answered him, after his last answer, and `compacting` while a
cut runs.

None is journaled: the finished record is always sent, and a client that lost a frame sees a
gap in the offsets, which the record fills (decision 5)."""

PROGRESS_KINDS: Final[tuple[ProgressKind, ...]] = (
    "alive",
    "thinking",
    "tool_call",
    "tool_result",
    "delegation",
    "answer",
    "error",
    "text",
    "thought",
    "tool_started",
    "context",
)


KEEPALIVE_S: Final = 15.0
"""How often an idle stream sends `progress{kind: alive}`, so a client notices a server that
died without closing its connection: a client that hears nothing for several of these
may treat the stream as gone. Droppable like every progress frame."""


class ProgressFrame(BaseModel):
    """Work in flight: droppable, unseekable, never the only place a fact appears.
    `scope` is the session it concerns; `payload` is a short description for display."""

    model_config = _WIRE
    type: Literal["progress"] = "progress"
    scope: str
    kind: ProgressKind
    payload: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


StreamFrame = Annotated[RecordFrame | ProgressFrame, Field(discriminator="type")]
_STREAM: Final = TypeAdapter[RecordFrame | ProgressFrame](StreamFrame)


def encode_frame(frame: RecordFrame | ProgressFrame) -> bytes:
    """One frame as one NDJSON line, newline included."""
    return frame.model_dump_json().encode() + b"\n"


def decode_frame(line: bytes | str) -> RecordFrame | ProgressFrame:
    """One NDJSON line back into its frame, by its `type`."""
    return _STREAM.validate_json(line)


# -- the handshake and errors ---------------------------------------------------------


class AttachLimits(BaseModel):
    """The limits register's `turn.*` rows, as this server sets them, and the edge an image is
    downscaled to. A client checks against these before it
    sends, so a refusal says why before the bytes travel. The wire's copy of core's
    `AttachCaps`, field for field, because this module stays core-free; `test_turn_attachments`
    holds the two to the same fields."""

    model_config = _WIRE
    image_max_edge: int
    attachments_max: int
    attachment_bytes_max: int
    attachment_file_max_bytes: int
    pdf_file_max_bytes: int
    pdf_pages_max: int
    pdf_text_min_chars: int
    text_file_max_bytes: int
    text_inline_bytes: int


class Hello(BaseModel):
    """`GET /`: the one call a client makes before it knows anything, needing no token."""

    model_config = _WIRE
    wire: str
    engine: str
    capabilities: tuple[str, ...]
    """The capability of every route that answers here (`routes.Route.capability`). A route
    whose facade method is not built is absent, and appears when the engine gains it."""
    replay: bool = False
    """The server serves a recorded session (`--replay`): mutating routes refuse."""
    question_timeout_s: float | None = None
    """How long this server waits for an answer before a question resolves to its default
    (CL-5's `question_timeout_s`): a question's deadline is its record's time
    plus this. None from a server that does not say."""
    attach: AttachLimits | None = None
    """What a turn may carry: None from a server that takes no attachments (a replay
    server, or one before wire 4.18), and a client then does not offer attaching."""


# -- the token file ----------------------------------------------------------------


class TokenClass(StrEnum):
    """What a token's answers are recorded as (`auth.source_for`): `interactive` is a person's,
    `automation` the default a script agreed with."""

    INTERACTIVE = "interactive"
    AUTOMATION = "automation"


TOKEN_FILE: Final = "serve.token"
"""Beside the socket, owner-readable only, removed when the server stops (`auth.TokenStore`)."""


class TokenFile(BaseModel):
    """`serve.token`: one token per class. Part of the wire because a client reads it."""

    model_config = _WIRE
    interactive: str
    automation: str

    def token(self, token_class: TokenClass) -> str:
        return self.interactive if token_class is TokenClass.INTERACTIVE else self.automation


def read_token(path: Path, token_class: TokenClass) -> str:
    """A client's half: the token of `token_class` from a `serve.token` file."""
    return TokenFile.model_validate_json(path.read_bytes()).token(token_class)


class ErrorCode(StrEnum):
    BAD_REQUEST = "BAD_REQUEST"
    """The body or a parameter did not validate. 422."""
    UNAUTHORIZED = "UNAUTHORIZED"
    """No token, or one this server did not issue. 401."""
    NOT_FOUND = "NOT_FOUND"
    """No such session, run or pending question. 404."""
    CONFLICT = "CONFLICT"
    """The session cannot do that now: open elsewhere, a turn running, refused by its state.
    409."""
    ANSWERED = "ANSWERED"
    """The question was already answered, by another client or by its timeout. 409."""
    BUSY = "BUSY"
    """An episode is already in flight on this engine (`EngineBusy`). 409."""
    CONFIG = "CONFIG"
    """The engine refused a value as a config load would (`ConfigError`). 422."""
    NOT_BUILT = "NOT_BUILT"
    """The facade method behind the route does not exist yet, or raised
    `NotImplementedError`. 501. `GET /` omits such a route's capability."""
    REPLAY_READ_ONLY = "REPLAY_READ_ONLY"
    """A mutating route on a `--replay` server. 405."""
    INTERNAL = "INTERNAL"
    """A harness bug. 500. The message names it; the server log has the rest."""


class ErrorBody(BaseModel):
    model_config = _WIRE
    code: ErrorCode
    message: str


# -- request bodies ----------------------------------------------------------------------------


class CreateSessionBody(BaseModel):
    model_config = _WIRE
    session_id: str | None = None
    """None mints one."""
    adopt: str | None = None
    """An open work's key (`GET /works`): the new session continues it."""


TurnWorkspace = Literal["worktree", "in_place"]
"""Where the person said a turn's work runs (`--worktree`, `--in-place`)."""


class TurnAttachment(BaseModel):
    """A file the person sent with a turn: its name and its bytes, base64. The server
    decides what it is from the bytes, never from the name."""

    model_config = _WIRE
    name: Annotated[str, Field(min_length=1, max_length=200)]
    data: Annotated[str, Field(min_length=1)]


class TurnBody(BaseModel):
    """The person's words to the session."""

    model_config = _WIRE
    text: Annotated[str, Field(min_length=1)]
    workspace: TurnWorkspace | None = None
    """Where the person said this turn's work runs (`--worktree`, `--in-place`), or None for
    the configured default."""
    attachments: Annotated[list[TurnAttachment], Field(exclude_if=lambda v: not v)] = []
    """What the person sent with the words: at most the hello's `attachments_max`,
    each checked again by the engine before anything is recorded. Not sent when empty, so a
    turn without files is the body a server before 4.18 accepts."""


class TurnAccepted(BaseModel):
    """`POST /turn`'s answer, at once: the turn itself is read on the session's stream."""

    model_config = _WIRE
    queued: bool
    """The words wait behind a running turn and are delivered at its boundary."""


class InterruptBody(BaseModel):
    model_config = _WIRE
    hard: bool = False


class CapBody(BaseModel):
    """The family's cap, or one rung's; `null` lifts it."""

    model_config = _WIRE
    hard_cost_usd: float | None
    rung: str | None = None


STOP_CHOICES: Final = ("approve", "edit", "note", "cancel")
"""What an answer to a stop may choose (`StopAnswer`): `edit` carries the person's
markdown as `text`, `note` the person's words for the author to revise by, and `cancel` the
reason, in `note`."""

APPROVAL_CHOICES: Final = ("approve", "deny", "abort")
"""What an answer to an action approval may choose (`ApprovalDecision.verdict`): a client
offers these, and the server reads the choice as one. An automation token's `approve`
is recorded as a deny."""

GRANT_CHOICES: Final = ("approve", "session", "repo", "deny", "abort")
"""An approval offering both *always* answers: `approve` is *once*,
`session` and `repo` approve and grant what the question names for the rest of this session
or in this repository. A question offering neither offers `APPROVAL_CHOICES`; one
offering only *this session, exact* leaves out `repo`."""


def approval_choices(*, grant: bool, exact: bool, offer_repo: bool) -> tuple[str, ...]:
    """What an approval offers: the one home for it, read by the server's slot, its replay and
    every client. `session` when the question carries an exact binding or a grant (an exact one
    binds the line itself); `repo` when it carries a grant and offers it (E.3)."""
    session = exact or grant
    repo = grant and offer_repo
    if not session and not repo:
        return APPROVAL_CHOICES
    return (
        "approve",
        *(("session",) if session else ()),
        *(("repo",) if repo else ()),
        "deny",
        "abort",
    )


class AnswerBody(BaseModel):
    """The other half of a `question_asked`. Carries no source: who answered is the
    token's class, never anything a request says."""

    model_config = _WIRE
    choice: Annotated[str, Field(min_length=1)]
    note: str | None = None
    text: str | None = None
    """With `edit` at a stop, the person's markdown, shown again as the next stop.
    The design gate's `design` and `subset` retired with it."""

    @model_validator(mode="after")
    def _text_with_its_choice(self) -> AnswerBody:
        if self.text is not None and self.choice != "edit":
            raise ValueError("a stop's text travels only with the choice edit")
        return self


class AnswerTaken(BaseModel):
    model_config = _WIRE
    question_id: str
    source: str
    """The source the answer was recorded with, from the token's class."""


class UndoBody(BaseModel):
    """Take back turns `to_turn` onward."""

    model_config = _WIRE
    to_turn: Annotated[int, Field(ge=1)]
    code: bool
    conversation: bool
    commit: bool | None = None


class UndoTaskBody(BaseModel):
    """`/undo task`: a worktree run moved back to before one task's
    commit."""

    model_config = _WIRE
    run_id: Annotated[str, Field(min_length=1)]
    task_id: Annotated[str, Field(min_length=1)]


class UndoOperationBody(BaseModel):
    """`/undo <operation> [<run>]`: one operation's declared paths, from its snapshot
    ref. `run_id` names the run when more than one journaled the id (wire 1.9)."""

    model_config = _WIRE
    operation_id: Annotated[str, Field(min_length=1)]
    run_id: Annotated[str | None, Field(min_length=1)] = None


class ForkBody(BaseModel):
    """A child session from `at_seq`. `code` restores the workspace to the fork
    point or keeps it as it is; absent, the engine asks the person (`question_asked`)."""

    model_config = _WIRE
    at_seq: Annotated[int, Field(ge=1)]
    reason: str
    code: Literal["restore", "keep"] | None = None


class RungBody(BaseModel):
    """Move one rung to another catalog model, from its next call."""

    model_config = _WIRE
    rung: str
    model: str


class CoderBody(BaseModel):
    """A `CoderRequest` (`box/request.py`): work already approved at its stop. Kept as a JSON
    object here so the wire stays core-free; the server validates it."""

    model_config = _WIRE
    request: dict[str, JsonValue]


class CoderStarted(BaseModel):
    """`POST /coder`'s answer: the episode is named and holds the lease; it is read on the run's
    stream (`/run/{run_id}/events`) and its `episode` record on the session's."""

    model_config = _WIRE
    run_id: str


class Resumed(BaseModel):
    """`POST /resume`'s answer: the runs the session's journal shows unfinished,
    oldest first, and the one continued, if any."""

    model_config = _WIRE
    session_id: str
    reconciled: tuple[str, ...]
    in_flight: str | None


class LocateBody(BaseModel):
    model_config = _WIRE
    question: Annotated[str, Field(min_length=1)]


class RenameBody(BaseModel):
    """`POST /session/{sid}/title`: the person's name for the session, with words."""

    model_config = _WIRE
    text: Annotated[str, Field(pattern=r"\S")]


class FileBody(BaseModel):
    """`POST /session/{sid}/file`: the one file to read, a path `/files` lists. In a
    body, as `/locate`'s question is, so a path is never a URL segment to quote."""

    model_config = _WIRE
    path: Annotated[str, Field(min_length=1)]


class SecretBody(BaseModel):
    """A line the REPL would keep in its history: the server answers whether
    it holds a secret value (`secrets.redact`), and the client keeps it nowhere if it does."""

    model_config = _WIRE
    text: str


class FinalizeBody(BaseModel):
    model_config = _WIRE
    choice: Literal["apply", "squash", "keep", "discard", "bring"]
    """A `GitHandoffChoice`; any other word is refused at the
    wire, `422`."""


class Hidden(BaseModel):
    """What the session's standing rewinds hide now: a client re-asks after each
    `undo` or `redo` record rather than re-implementing the rule."""

    model_config = _WIRE
    seqs: tuple[int, ...]
