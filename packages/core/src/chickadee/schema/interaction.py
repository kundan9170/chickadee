"""Contract 6 — Interaction: Question / Answer / the stops (v2).

The engine requests approval and acceptance through
this port and never touches a terminal.
"""

from __future__ import annotations

import threading
from collections.abc import Sized
from enum import StrEnum
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator

from chickadee.schema.authorization import FactHit, Tier
from chickadee.schema.isolation import WorkspaceMode

WIRE = ConfigDict(extra="forbid", frozen=True)


class QuestionKind(StrEnum):
    """The closed ask-point list. Another member is a decision, never a commit."""

    CLARIFY_GOAL = "clarify_goal"
    PLAN_APPROVAL = "plan_approval"
    ROUTE_AMBIGUOUS = "route_ambiguous"
    DESIGN_FORK = "design_fork"
    ESCALATION = "escalation"
    REPLAN_PREMISE = "replan_premise"
    SCOPE_EXPANSION = "scope_expansion"
    UNDO_CONFLICT = "undo_conflict"
    """An undo or redo would restore a path the person edited too. Asked by a command
    the person started, never by a run; the default keeps their version."""
    FORK_CODE = "fork_code"
    """A fork asks whether the code goes back to the fork point or stays as it is.
    Asked by a command the person started; the default keeps it."""
    WORK_FOLDER = "work_folder"
    """The work folder cannot be used here (a file at its path, or git
    ignores it), so the harness offers a free folder for this repository. Asked by the
    harness, never a model; the default, `no`, changes nothing. Appended: every member above
    keeps its meaning."""


class AnswerSource(StrEnum):
    HUMAN = "human"
    DEFAULT = "default"
    SESSION_REUSE = "session_reuse"
    TIMEOUT_DEFAULT = "timeout_default"


class Option(BaseModel):
    model_config = WIRE

    key: str
    label: str


class Question(BaseModel):
    model_config = WIRE

    id: str
    kind: QuestionKind
    text: str
    options: tuple[Option, ...] = ()
    default: str
    """Required, with no fallback. Constructing a Question without a default raises,
    because non-interactive mode must always be able to resolve it to today's designed
    behavior — and a question that cannot be resolved unattended would hang the run."""
    evidence: tuple[str, ...] = ()
    blocking: bool = True

    @model_validator(mode="after")
    def _default_is_resolvable(self) -> Question:
        if not self.default:
            raise ValueError(f"question {self.id!r} has no default")
        if self.options and self.default not in {o.key for o in self.options}:
            raise ValueError(
                f"question {self.id!r} default {self.default!r} is not one of its options"
            )
        return self


class Answer(BaseModel):
    model_config = WIRE

    question_id: str
    choice: str
    source: AnswerSource
    note: str | None = None
    """The person's words, verbatim — never paraphrased. On a refusing choice the
    harness appends a Constraint{source: human} to the ledger before the typed
    refusal returns; on an approving choice it is recorded on the answer and binds nothing.
    `""` is preserved, not collapsed to None: a person who typed nothing and one who typed
    nothing on purpose are told apart by the field."""


# --- the stops -----------------------------------
# Contract 6, v2: these types are additive; `confirm_stop` replaced
# `confirm_plan` and its `PlanDecision` on `InteractionPort` with the switch.


class StopKind(StrEnum):
    DESIGN = "design"
    """Stop 1: Zeus's design."""
    PLAN = "plan"
    """Stop 2: Athena's plan, or Zeus's one-task plan, its brief alone."""


class ApprovalStop(BaseModel):
    """What the person is shown at a stop: the markdown, exactly. Approving binds `sha`, the
    text last shown (invariant 15); an edit or a note makes a new text, shown again. Not
    `Stop`, the spec's first name for it: `tools/rules.Stop` is the dispatcher's verdict."""

    model_config = WIRE

    id: str
    """The question's id: one per showing, since each is answered once."""
    kind: StopKind
    author: Literal["zeus", "athena"]
    text: str
    """The markdown shown, exactly. Nothing parses it."""
    sha: str
    """`sha256_hex` of `text`, which approval binds."""
    round: int = 1
    """Which showing this is: 2 after the first edit."""
    refused: str | None = None
    """Why the last answer was not used (an empty note, an edit that changed nothing): the
    same text, shown again, saying so."""
    workspace: WorkspaceMode
    source_ref: str | None = None
    """The branch the work starts from, the commit when the checkout is detached, or None
    when it has no commit yet."""
    folder: str | None = None
    """The absolute path of the work folder's part this stop approves
    (`design/`, `plan/`), whose README is `text`; an edit is the person's, of these files, and
    the harness re-reads the folder. None for a text with no folder (a one-task plan)."""
    notice: str | None = None
    """A line shown once with the stop, never part of what is approved."""


class StopAnswer(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    NOTE = "note"
    CANCEL = "cancel"


class StopDecision(BaseModel):
    model_config = WIRE

    answer: StopAnswer
    edited: str | None = None
    """The person's text, on `edit`: what is shown next, and approved if they approve it."""
    note: str | None = None
    """Verbatim: on `note`, what the author revises by; on `cancel`, the reason."""
    source: AnswerSource | None = None
    """Who decided: a person, the default, or the timeout's default.
    Only the port knows: `RemoteInteraction` reads the answering token's class. Absent means
    not recorded, never a person."""


GrantScope = Literal["once", "session", "repo"]
"""How far a person's `approve` reaches: this call;
every matching call for the rest of this session; every matching call in this repository."""


class CommandBinding(BaseModel):
    """What an *always* answer would grant, and what a later call must equal for the grant to
    cover it: the command's class, prefix and extracted destination, and what the
    call declared beyond them, so a grant never covers a call reaching further than the one
    approved. Computed by the router from the argv, the
    command table and the call's declarations, never typed."""

    model_config = WIRE

    cls: str
    """A command class, by its value."""
    prefix: tuple[str, ...]
    destination: str | None = None
    network: bool = False
    """Whether the call opened the network: a grant given with it covers later calls
    with it, and one given without it covers none that open it. Was `hosts`, a list no backend
    could enforce; a record written before reads as `network_of_hosts` says."""
    effects: tuple[str, ...] = ()
    """The write surface the call declared, sorted."""
    credentials: tuple[str, ...] = ()
    """Credential NAMES the call declared, sorted; never values."""

    @model_validator(mode="before")
    @classmethod
    def _hosts_read_as_network(cls, data: object) -> object:
        return network_of_hosts(data)


def network_opened(value: object) -> bool:
    """Whether a `network` value opened the network: `true`, or a non-empty host list,
    as a `shell` call or a grant written in older versions named one. The one reading of the old
    shape, for a call's arguments (`context/standing`, `box/steps`) and a grant's binding
    (`network_of_hosts`)."""
    if isinstance(value, bool):
        return value
    return isinstance(value, list | tuple) and len(cast("Sized", value)) > 0


def network_of_hosts(data: object) -> object:
    """A grant's fields with an older `hosts` read as `network` (`network_opened`): a grant
    the person gave survives the change, binding no more than it did. The one reading for the
    binding here, a session's `command_grant` line and a repository's `[[approval.grants]]` row
    (`approval/grants.py`). A record naming both is refused: which one it meant is unknown."""
    if not isinstance(data, dict):
        return data
    raw = cast("dict[str, object]", data)
    if "hosts" not in raw:
        return raw
    if "network" in raw:
        raise ValueError("hosts is network since D199; give network only")
    old = {k: v for k, v in raw.items() if k != "hosts"}
    old["network"] = network_opened(raw["hosts"])
    return old


class ActionApproval(BaseModel):
    """One concrete effect, at dispatch time — the
    only ask point where the argv is real rather than declared.

    **This is the question, not the record.** The durable record is contract 10's
    `ActionAuthorization`, and an `approve` here is what a `Confirmation` on it attests.
    Keeping them apart lets the question carry what a record should not — a rendered
    `display` line, the router's evidence — without touching a frozen row.

    Every field is an observation of the action about to run, never a claim about it.
    Added at G4."""

    model_config = WIRE

    action_id: str
    operation_id: str
    """The operation this confirms: the `ActionAuthorization` it would be recorded on."""
    task_id: str | None = None
    """The box task the action belongs to, when the worker or Athena asks inside one; `None`
    for Zeus's, and for Athena's own outside a task. Contract 6 v2."""
    asker: Literal["zeus", "athena", "worker"] = "zeus"
    """Who asks, so the person is told whose command it is. Contract 6 v2."""
    kind: Literal["command", "mcp_tool", "git"]
    display: str
    """The argv or tool call, rendered for a person to read."""
    spec_sha: str
    """What approval binds to; a changed action is a new question."""
    effect_surface: tuple[str, ...] = ()
    network: bool
    credentials: tuple[str, ...] = ()
    """Credential NAMES, never values."""
    idempotent: bool | None
    """`None` is unknown, which is not `False` (invariant 20)."""
    reversible: bool
    """The question invariant 18 says approval is actually about."""
    tier: Tier
    """Recomputed on the concrete argv."""
    hits: tuple[FactHit, ...] = ()
    """Why: the router's rows that fired, each with its evidence."""
    grant: CommandBinding | None = None
    """What *always for this repo* would bind, and what a held grant is compared with. `None`
    offers no repository answer: an unrecognised command, a line two bindings make up, a call
    declaring more than a grant binds, or one outside its row's grant shape.
    """
    exact: CommandBinding | None = None
    """What *always for this session* binds: the line's every word, so it covers that line and
    nothing else. `None` with a `grant` is a question that binds its `grant` for
    the session instead, as before. Additive."""
    offer_repo: bool = True
    """Whether the question offers *always for this repo* for its `grant`: false
    for a command whose effects leave the machine and whose row has no grant shape, which a
    person grants for the repository only by hand. Additive."""
    repo_any_arguments: bool = False
    """Whether the repository answer covers any arguments after the command's words, because
    its row has no grant shape, so the prompt says so. Additive."""
    command: str | None = None
    """A command's line as it runs, which `display` composes for a person
    with where and how it runs; None for any other action, and on a line written before it.
    Additive."""
    cwd: str | None = None
    """The directory the command runs in, as `display` names it. Additive."""
    sandboxed: bool | None = None
    """Whether the action runs inside the sandbox once approved. False on every command a
    shell asks about: an approved command runs outside it. None where nothing
    said it: another kind of action, and a line written before it. Additive."""


class ApprovalDecision(BaseModel):
    """The answer to an `ActionApproval`. `remember` is the only path from an ask to a
    standing preference: a line the person accepted for `.chickadee/approval.toml`
    — a rule is offered and confirmed, never inferred behind their back."""

    model_config = WIRE

    action_id: str
    verdict: Literal["approve", "deny", "abort"]
    source: Literal["human", "preference", "default", "explicit_authorization", "grant"]
    """`grant`: a standing grant matched the action, and nobody was asked."""
    scope: GrantScope = "once"
    """How far an `approve` reaches. Anything but `once` is read only where the question
    offered a `grant`; it issues a grant, never a preference."""
    remember: str | None = None
    note: str | None = None
    """Free text, journaled beside the decision."""


@runtime_checkable
class InteractionPort(Protocol):
    """The ask seam The engine never touches a terminal.

    Two methods, not four. earlier drafts disagreed about this
    Protocol's shape, and the fuller version carries `Estimate`, `WorkspacePreview`,
    `ActionApproval` and `ApprovalDecision` — Phase 4/5 types describing behaviour that is
    not built. Freezing the pair the engine needs at G0 keeps `PacketEngine`'s constructor
    honest without inventing four contracts to support it; `confirm_action`
    and `notify` are added at their own gate. No implementer
    exists yet, so extending later costs nothing that extending now would save.

    `NonInteractive` is the default: every ask point resolves to its declared
    default, which is why `Question.default` is required rather than optional.

    Added at G4: `confirm_action`, the dispatch-time confirmation, and `ask`'s
    `cancel`, the session's cancellation token. When `cancel` is set the port returns promptly,
    and
    the caller answers the question cancelled whatever was returned, so a hard stop — and
    `PacketEngine.close()` — never waits on a person.

    Also `confirm_action`'s `cancel`, the same token with the same rule, so a hard
    stop or a pause never waits on a person at a command's permission either.

    Also `confirm_stop`'s `cancel`, the same token with the same rule. A stop the
    person is there to answer has no timeout, so this token is what ends its wait:
    the run dropped, or the process leaving, never a clock.
    """

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer: ...

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        """A stop (contract 6 v2): the markdown shown, exactly, and the person's
        answer. Never budgeted or reused: every stop is asked; an edit or a note is a new
        showing (`approval/stops.ask_stop`). Unattended, a stop is approved, as a design once
        was, and an action it leads to above T0 is still confirmed at dispatch.
        Attended, it waits until answered or `cancel` is set."""
        ...

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision: ...
