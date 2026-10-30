# A nested model's default is written as its JSON (`= {...}`) and validated into
# the model on construction (`validate_default=True`); pyright reads the literal,
# not the validation.
# pyright: reportAssignmentType=false
"""The wire's response and session-record types. GENERATED: do not edit.

`packages/wire/tools/generate_models.py` writes this from the golden,
`packages/serve/schema/golden/wire.schema.json` at `WIRE_VERSION` 4.22, and
`test_models_generated.py` fails when the two differ. Every class is a `WireModel`: frozen,
and ignoring a field it does not declare, so a server a minor version ahead still reads.
`SESSION_RECORDS` maps each readable session kind to its payload's type; a kind it lacks is
opaque, its writer's alone (`golden.py`).
"""

from __future__ import annotations

from typing import Annotated, Any, Final, Literal

from pydantic import ConfigDict, Field, RootModel, StringConstraints, TypeAdapter

from chickadee_wire._base import WireModel


class ActionGrantedRecord(WireModel):
    """
    An action a standing grant approved, so no one was asked: one line, never a
    question. What ran, under which grant, and what that grant covers.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    action_id: Annotated[str, Field(title="Action Id")]
    cls: Annotated[str, Field(title="Cls")]
    display: Annotated[str, Field(title="Display")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    prefix: Annotated[list[str], Field(title="Prefix")]
    scope: Annotated[
        Literal["once", "session", "repo"] | Literal["never_ask"], Field(title="Scope")
    ]


class AgentMessageP(WireModel):
    """
    One message of an agent's conversation, journaled before it joins it, so a follow-up
    after a resume rebuilds it exactly, as `box_message` is.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    agent: Annotated[str, Field(pattern="^A[1-9][0-9]*$", title="Agent")]
    round: Annotated[int, Field(ge=0, title="Round")]
    sha: Annotated[str, Field(title="Sha")]
    side: Annotated[Literal["system", "user", "assistant", "tool"], Field(title="Side")]


class AgentReportP(WireModel):
    """
    A round's end and its report: the report itself is a blob, and its
    checked counts are here.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    agent: Annotated[str, Field(pattern="^A[1-9][0-9]*$", title="Agent")]
    cost_usd: Annotated[float, Field(ge=0.0, title="Cost Usd")]
    findings: Annotated[int, Field(ge=0, title="Findings")]
    flagged: Annotated[int, Field(ge=0, title="Flagged")]
    report_sha: Annotated[str, Field(title="Report Sha")]
    round: Annotated[int, Field(ge=0, title="Round")]
    status: Annotated[
        Literal["complete", "partial", "failed", "cancelled"], Field(title="Status")
    ]
    tokens: Annotated[int, Field(ge=0, title="Tokens")]
    tool_calls: Annotated[int, Field(ge=0, title="Tool Calls")]


class AgentStartP(WireModel):
    """
    A round of an agent begins: before anything of it runs, so a round
    with this line and no `agent_report` is one a crash cut. Round 0 is the spawn.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    agent: Annotated[str, Field(pattern="^A[1-9][0-9]*$", title="Agent")]
    call_id: Annotated[str, Field(title="Call Id")]
    handles: Annotated[list[str], Field(title="Handles")] = []
    model: Annotated[str, Field(min_length=1, title="Model")]
    profile: Annotated[str, Field(min_length=1, title="Profile")]
    round: Annotated[int, Field(ge=0, title="Round")]
    rung: Annotated[str, Field(min_length=1, title="Rung")]
    skill: Annotated[str | None, Field(title="Skill")] = None
    source: Annotated[Literal["built-in", "user", "project", "external"], Field(title="Source")]
    task_sha: Annotated[str, Field(title="Task Sha")]


class AttachedFile(WireModel):
    """
    A file the person sent with the turn the work answers: named as the
    session numbered it, its bytes the session blob `sha` names. The Request section lists it,
    so it was approved with the work.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    id: Annotated[str, Field(pattern="^A-[1-9][0-9]*$", title="Id")]
    name: Annotated[str, Field(max_length=200, min_length=1, title="Name")]
    sha: Annotated[str, Field(min_length=1, title="Sha")]


class AttachmentPreview(WireModel):
    """
    What a file would be on this session's turn, before it is sent:
    what the check found, how Zeus's rung would be sent it and what that costs, and whether
    its text holds a secret. What a client's chip shows; nothing is recorded to make it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    bytes: Annotated[int, Field(title="Bytes")]
    delivery: Annotated[Literal["native", "text", "outline", "none"], Field(title="Delivery")]
    est_tokens: Annotated[int, Field(title="Est Tokens")]
    kind: Annotated[Literal["image", "pdf", "text"], Field(title="Kind")]
    media_type: Annotated[str, Field(title="Media Type")]
    name: Annotated[str, Field(title="Name")]
    secret: Annotated[bool, Field(title="Secret")]
    size: Annotated[str, Field(title="Size")]


class BoxCompactionP(WireModel):
    """
    A box compaction (`box_compaction`, moved to the run journal): what became
    receipts, folded, demoted or collapsed, and the tokens on both sides. The session-log kind
    retired with the packet loop; the ladder's other steps stay empty until the
    parked compaction returns.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    collapse: Annotated[list[str], Field(title="Collapse")] = []
    demote: Annotated[list[str], Field(title="Demote")] = []
    evict: Annotated[list[str], Field(title="Evict")] = []
    fold: Annotated[list[str], Field(title="Fold")] = []
    role: Annotated[Literal["athena", "worker"], Field(title="Role")]
    steps: Annotated[
        list[Literal["evict", "fold", "demote_history", "collapse_unrelated", "reset"]],
        Field(title="Steps"),
    ]
    tokens_after: Annotated[int, Field(ge=0, title="Tokens After")] = 0
    tokens_before: Annotated[int, Field(ge=0, title="Tokens Before")] = 0


class BoxMessageP(WireModel):
    """
    One message of a box role's conversation, journaled before it joins it, so resume
    rebuilds the conversation exactly.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    role: Annotated[Literal["athena", "worker"], Field(title="Role")]
    sha: Annotated[str, Field(title="Sha")]
    side: Annotated[Literal["system", "user", "assistant", "tool"], Field(title="Side")]


class BudgetP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    cap_usd: Annotated[float, Field(title="Cap Usd")]
    kind: Annotated[Literal["warning", "hard_cap"], Field(title="Kind")]
    rung: Annotated[str | None, Field(min_length=1, title="Rung")] = None
    scope: Annotated[Literal["run", "packet"], Field(title="Scope")]
    spent_usd: Annotated[float, Field(title="Spent Usd")]
    terminal: Annotated[bool, Field(title="Terminal")]


class CallbackErrorP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    error_sha: Annotated[str, Field(title="Error Sha")]
    event: Annotated[str, Field(title="Event")]
    name: Annotated[str, Field(title="Name")]


class CapRecord(WireModel):
    """
    The person's cap for the session family, or `None` when they lifted it.
    The family's latest line is its cap; with none, the engine config's `hard_cost_usd`.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    hard_cost_usd: Annotated[int | float | None, Field(title="Hard Cost Usd")]
    rung: Annotated[str | None, Field(min_length=1, title="Rung")] = None


class CatalogModel(WireModel):
    """
    What a catalog model is, as a person choosing a rung's model reads it:
    its windows, who serves it, and the name its provider knows it by. Never its `base_url` or
    key: an endpoint can carry a credential.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    context_window: Annotated[int, Field(title="Context Window")]
    max_output_tokens: Annotated[int, Field(title="Max Output Tokens")]
    model_id: Annotated[str | None, Field(title="Model Id")] = None
    provider: Annotated[str, Field(title="Provider")]


class CitationGateP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    checked: Annotated[int, Field(title="Checked")]
    coverage: Annotated[float, Field(title="Coverage")]
    pass_: Annotated[bool, Field(alias="pass", title="Pass")]


class CodeOutcome(WireModel):
    """
    What an undo or redo did to one workspace's code.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    committed: Annotated[str | None, Field(title="Committed")] = None
    kept: Annotated[list[str], Field(title="Kept")] = []
    kept_by_default: Annotated[list[str], Field(title="Kept By Default")] = []
    key: Annotated[str, Field(title="Key")]
    note: Annotated[str | None, Field(title="Note")] = None
    restored: Annotated[list[str], Field(title="Restored")] = []
    root: Annotated[str, Field(title="Root")]
    skipped: Annotated[list[str], Field(title="Skipped")] = []
    where: Annotated[Literal["worktree", "checkout", "in_place"], Field(title="Where")]


class CoderOutcome(WireModel):
    """
    A `coder` result's terminal and why, typed, so a reader shows the reason without
    reading it out of the text. `terminal` is the result's head, the episode's outcome
    (`COMPLETED`, `PARTIAL`, `BLOCKED`, …), or a packet-loop terminal on an older log;
    `why` the lines that say why, as the result states them.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    terminal: Annotated[str, Field(title="Terminal")]
    why: Annotated[list[str], Field(title="Why")] = []


class CommandBinding(WireModel):
    """
    What an *always* answer would grant, and what a later call must equal for the grant to
    cover it: the command's class, prefix and extracted destination, and what the
    call declared beyond them, so a grant never covers a call reaching further than the one
    approved. Computed by the router from the argv, the
    command table and the call's declarations, never typed.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    cls: Annotated[str, Field(title="Cls")]
    credentials: Annotated[list[str], Field(title="Credentials")] = []
    destination: Annotated[str | None, Field(title="Destination")] = None
    effects: Annotated[list[str], Field(title="Effects")] = []
    network: Annotated[bool, Field(title="Network")] = False
    prefix: Annotated[list[str], Field(title="Prefix")]


class CommandFactP(WireModel):
    """
    One command a task ran, from its step and its `exec` line.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    command: Annotated[str, Field(title="Command")]
    exit_code: Annotated[int | None, Field(title="Exit Code")] = None
    failing: Annotated[list[str], Field(title="Failing")] = []
    operation_id: Annotated[str, Field(title="Operation Id")]
    output_sha: Annotated[str | None, Field(title="Output Sha")] = None
    tail: Annotated[str, Field(title="Tail")]
    test_outcome: Annotated[str | None, Field(title="Test Outcome")] = None
    timed_out: Annotated[bool, Field(title="Timed Out")]


class Concern(WireModel):
    """
    `concern_raised`: the utterance itself. The evidence class is always `claim` — a
    concern is never a fact, and has no field that could make it look like one.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    anchor: Annotated[list[str], Field(title="Anchor")] = []
    event: Annotated[Literal["raised"], Field(title="Event")] = "raised"
    id: Annotated[str, Field(title="Id")]
    raised_by: Annotated[Literal["zeus", "coder", "critic", "user"], Field(title="Raised By")]
    text: Annotated[str, Field(title="Text")]


class Source(RootModel[int]):
    root: Annotated[int, Field(ge=1)]


class ConstraintRecord(WireModel):
    """
    A `constraint` line's payload: the constraint, and the id it supersedes. The ledger
    keeps the constraint (`ledger.Constraint`); `supersedes` is the line's alone.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    author: Annotated[Literal["zeus", "user"], Field(title="Author")]
    id: Annotated[str, Field(title="Id")]
    scope: Annotated[Literal["session", "design", "packet_set"], Field(title="Scope")]
    scope_id: Annotated[str | None, Field(title="Scope Id")] = None
    sources: Annotated[list[Source], Field(title="Sources")]
    superseded_by: Annotated[str | None, Field(title="Superseded By")] = None
    supersedes: Annotated[str | None, Field(title="Supersedes")] = None
    text: Annotated[str, Field(title="Text")]
    valid_from: Annotated[int, Field(title="Valid From")]
    valid_until: Annotated[int | None, Field(title="Valid Until")] = None


class ContextTokensP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    compacted: Annotated[int, Field(title="Compacted")]
    layer1: Annotated[int, Field(title="Layer1")]
    raw_window: Annotated[int, Field(title="Raw Window")]


class Continuation(WireModel):
    """
    A run that continues a work an ended or superseded run left: what it inherits, read back
    on resume from the run's config
    snapshot, so the seed never depends on a record that may have changed since.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    amended_from: Annotated[str | None, Field(title="Amended From")] = None
    mode: Annotated[Literal["CONTINUE", "AMEND"], Field(title="Mode")]
    plan_bound_to: Annotated[str | None, Field(title="Plan Bound To")] = None
    plan_run: Annotated[str | None, Field(title="Plan Run")] = None
    plan_sha: Annotated[str | None, Field(title="Plan Sha")] = None
    plan_version: Annotated[int | None, Field(ge=1, title="Plan Version")] = None
    prior_runs: Annotated[list[str], Field(min_length=1, title="Prior Runs")]


class Dismissed(WireModel):
    """
    `concern_dismissed`: the user, and only the user, said stop displaying it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    event: Annotated[Literal["dismissed"], Field(title="Event")] = "dismissed"
    id: Annotated[str, Field(title="Id")]


class EffectCompletedP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    observed_surface: Annotated[list[str], Field(title="Observed Surface")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    outcome: Annotated[Literal["succeeded", "failed", "ambiguous"], Field(title="Outcome")]
    post_state_sha: Annotated[str | None, Field(title="Post State Sha")] = None
    result_sha: Annotated[str | None, Field(title="Result Sha")] = None


class EffectIntendedP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    approved_sha: Annotated[str, Field(title="Approved Sha")]
    declared_surface: Annotated[list[str], Field(title="Declared Surface")]
    effect_kind: Annotated[str, Field(title="Effect Kind")]
    idempotent: Annotated[bool, Field(title="Idempotent")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    preconditions: Annotated[list[str], Field(title="Preconditions")]
    preimages_sha: Annotated[str | None, Field(title="Preimages Sha")] = None


class EffectStartedP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    operation_id: Annotated[str, Field(title="Operation Id")]


class ExecP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    argv_sha: Annotated[str, Field(title="Argv Sha")]
    cwd: Annotated[str, Field(title="Cwd")]
    escalation_id: Annotated[str | None, Field(title="Escalation Id")] = None
    exit_code: Annotated[int, Field(title="Exit Code")]
    label: Annotated[str, Field(title="Label")]
    network: Annotated[bool, Field(title="Network")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    policy: Annotated[str, Field(title="Policy")]
    sandbox: Annotated[str, Field(title="Sandbox")]
    stderr_sha: Annotated[str, Field(title="Stderr Sha")]
    stdout_sha: Annotated[str, Field(title="Stdout Sha")]
    timed_out: Annotated[bool, Field(title="Timed Out")]
    violations: Annotated[list[str], Field(title="Violations")]
    wall_ms: Annotated[int, Field(title="Wall Ms")]


class ExternalCallP(WireModel):
    """
    One call that left the machine on a model's behalf, through the external-call policy
    (`chickadee.external`): a web search or fetch now, an MCP tool's call
    later, which writes this in place of `McpCallP`. Additive under the owner's grant.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    args_sha: Annotated[str, Field(title="Args Sha")]
    error: Annotated[str | None, Field(title="Error")] = None
    ok: Annotated[bool, Field(title="Ok")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    policy: Annotated[Literal["discovery", "action"], Field(title="Policy")]
    provider: Annotated[str, Field(min_length=1, title="Provider")]
    result_bytes: Annotated[int, Field(ge=0, title="Result Bytes")] = 0
    result_sha: Annotated[str | None, Field(title="Result Sha")] = None
    tool: Annotated[str, Field(min_length=1, title="Tool")]
    truncated: Annotated[bool, Field(title="Truncated")] = False
    wall_ms: Annotated[int, Field(ge=0, title="Wall Ms")]


class FileFactP(WireModel):
    """
    One file a task changed, measured between its start tree and its end tree.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    added: Annotated[int, Field(ge=0, title="Added")]
    kind: Annotated[Literal["created", "edited", "deleted"], Field(title="Kind")]
    path: Annotated[str, Field(title="Path")]
    removed: Annotated[int, Field(ge=0, title="Removed")]


class FileText(WireModel):
    """
    A workspace file as a frontend reads it: its text, or why there is none.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    path: Annotated[str, Field(title="Path")]
    size: Annotated[int, Field(title="Size")]
    text: Annotated[str | None, Field(title="Text")]
    withheld: Annotated[str | None, Field(title="Withheld")] = None


class ForkEffect(WireModel):
    """
    An effect a fork cannot undo: something that already happened in the world.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    detail: Annotated[str, Field(title="Detail")]
    kind: Annotated[str, Field(title="Kind")]
    seq: Annotated[int, Field(title="Seq")]


class ForkOrigin(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    at_seq: Annotated[int, Field(ge=0, title="At Seq")]
    session_id: Annotated[str, Field(title="Session Id")]


class ForkRecord(WireModel):
    """
    A child's first line, written by the store when it creates the fork.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    at: Annotated[str | None, Field(title="At")] = None
    forked_from: ForkOrigin
    reason: Annotated[str, Field(title="Reason")]


class Forked(WireModel):
    """
    What a fork made: the child, what it was told on arrival, and its code.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    child_id: Annotated[str, Field(title="Child Id")]
    code: Annotated[Literal["restore", "keep"], Field(title="Code")]
    notices: Annotated[list[str], Field(title="Notices")]
    workspaces: Annotated[
        list[CodeOutcome], Field(title="Workspaces", validate_default=True)
    ] = []
    worktree: Annotated[str | None, Field(title="Worktree")] = None


class FrameRecord(WireModel):
    """
    The turn frame: one harness line before the session's first
    user turn, and before the first one after the date changes. The date and the git head
    are here, not in the system prefix, where either would be a cache miss on the largest
    region for the cheapest fact.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    branch: Annotated[str | None, Field(title="Branch")] = None
    date: Annotated[str, Field(title="Date")]
    head: Annotated[str | None, Field(title="Head")] = None
    status: Annotated[str | None, Field(title="Status")] = None
    text: Annotated[str, Field(title="Text")]


class HandoffPreview(WireModel):
    """
    Everything `01-orchestrator` §3.5.1 requires the handoff to identify, plus whether
    the source has moved underneath the run.

    Read-only and non-raising on drift: the review surface has to *show* the user that the
    branch moved, which it cannot do if learning that fact is an exception.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    base: Annotated[str, Field(title="Base")]
    branch: Annotated[str | None, Field(title="Branch")]
    commits: Annotated[list[str], Field(title="Commits")]
    drift: Annotated[str | None, Field(title="Drift")]
    lease_id: Annotated[str, Field(title="Lease Id")]
    mode: Annotated[Literal["worktree", "in_place"], Field(title="Mode")]
    overlapping: Annotated[list[str], Field(title="Overlapping")] = []
    seed: Annotated[str | None, Field(title="Seed")] = None
    source_branch: Annotated[str | None, Field(title="Source Branch")]
    source_clean: Annotated[bool, Field(title="Source Clean")]
    source_head: Annotated[str | None, Field(title="Source Head")]
    tip: Annotated[str | None, Field(title="Tip")]
    uncommitted: Annotated[list[str], Field(title="Uncommitted")] = []
    worktree_path: Annotated[str | None, Field(title="Worktree Path")]


class HandoffResult(WireModel):
    """
    What a completed handoff did. `pushed` is a field rather than a docstring claim
    because a host that has to trust prose is a host that cannot assert on it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    branch: Annotated[str | None, Field(title="Branch")]
    choice: Annotated[
        Literal["apply", "squash", "keep", "discard", "bring"], Field(title="GitHandoffChoice")
    ]
    """
    What the user does with the run's worktree branch.

    `APPLY` and `SQUASH` are separate because they are different git operations, not two
    spellings of one. Each packet's commit is a rollback unit, and that property
    only survives `APPLY` — after `SQUASH` the packets are welded into a single commit and
    undoing one means undoing all of them. Collapsing them into a single `apply_squash`
    member would put that choice in the engine, where the user cannot reach it.
    """
    commits_applied: Annotated[list[str], Field(title="Commits Applied")]
    detail: Annotated[str, Field(title="Detail")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    pushed: Annotated[Literal[False], Field(title="Pushed")] = False
    source_branch: Annotated[str | None, Field(title="Source Branch")]
    source_head: Annotated[str | None, Field(title="Source Head")]
    worktree_path: Annotated[str | None, Field(title="Worktree Path")]


class HeaderLine(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    kind: Annotated[
        Literal[
            "workspace",
            "isolation",
            "sandbox",
            "deny_list",
            "commands",
            "bootstrap",
            "mcp",
            "discovery",
            "secrets",
            "storage",
            "models",
            "spend",
        ],
        Field(title="Kind"),
    ]
    source: Annotated[str, Field(title="Source")]
    text: Annotated[str, Field(title="Text")]


class HookP(WireModel):
    """
    One hook's verdict, error or bound reached, or, as `event = "bound"`,
    the set an episode or a session bound (decision 14). A silent `Pass` writes none. Fields
    after `timed_out` are additive.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    bounded: Annotated[bool, Field(title="Bounded")] = False
    call_id: Annotated[str | None, Field(title="Call Id")] = None
    decision: Annotated[str, Field(title="Decision")]
    error: Annotated[str | None, Field(title="Error")] = None
    event: Annotated[str, Field(title="Event")]
    input_sha: Annotated[str | None, Field(title="Input Sha")] = None
    message: Annotated[str | None, Field(title="Message")] = None
    name: Annotated[str, Field(title="Name")]
    role: Annotated[str | None, Field(title="Role")] = None
    text_sha: Annotated[str | None, Field(title="Text Sha")] = None
    timed_out: Annotated[bool, Field(title="Timed Out")] = False
    wall_ms: Annotated[int, Field(title="Wall Ms")]


class ImageRef(WireModel):
    """
    The pixels a result sent: the downscaled bytes, in the session's blob store,
    so a resumed session sends the same bytes the live one did. A PDF sent whole is one too,
    by its media type.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    media_type: Annotated[str, Field(title="Media Type")]
    sha: Annotated[str, Field(title="Sha")]


class IndexProgress(WireModel):
    """
    How far a pass has got: what a person watching a build is shown.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    done: Annotated[int, Field(title="Done")] = 0
    stage: Annotated[Literal["preparing", "reading", "linking"], Field(title="Stage")]
    total: Annotated[int, Field(title="Total")] = 0


class InterruptP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    action: Annotated[str, Field(title="Action")]
    level: Annotated[str, Field(title="Level")]


class JobReport(WireModel):
    """
    A job's own facts, the blob its `JOB` effect's completion names.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    alongside: Annotated[list[str], Field(title="Alongside")]
    by: Annotated[str | None, Field(title="By")]
    dropped: Annotated[int, Field(title="Dropped")]
    ended: Annotated[
        Literal["exited", "timed_out", "killed", "ceiling", "lost"], Field(title="Ended")
    ]
    err_bytes: Annotated[int, Field(title="Err Bytes")]
    exit_code: Annotated[int | None, Field(title="Exit Code")]
    git_restored: Annotated[list[str], Field(title="Git Restored")]
    job_id: Annotated[str, Field(title="Job Id")]
    kept: Annotated[str | None, Field(title="Kept")]
    kind: Annotated[Literal["task", "server"], Field(title="Kind")]
    line: Annotated[str, Field(title="Line")]
    moved: Annotated[bool, Field(title="Moved")]
    notify: Annotated[bool, Field(title="Notify")]
    out_bytes: Annotated[int, Field(title="Out Bytes")]
    output: Annotated[str | None, Field(title="Output")]
    overwrote: Annotated[list[str], Field(title="Overwrote")]
    wall_ms: Annotated[int | None, Field(title="Wall Ms")]
    wrote: Annotated[list[str], Field(title="Wrote")]


class JobView(WireModel):
    """
    One job as `job list` and `/jobs` show it: from its effect, and its report once ended.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    job_id: Annotated[str, Field(title="Job Id")]
    kind: Annotated[Literal["task", "server"], Field(title="Kind")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    report: JobReport | None
    row: Annotated[str, Field(title="Row")]
    running: Annotated[bool, Field(title="Running")]
    state: Annotated[str, Field(title="State")]
    step: Annotated[str, Field(title="Step")]


class McpCallP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    args_sha: Annotated[str, Field(title="Args Sha")]
    error: Annotated[str | None, Field(title="Error")] = None
    ok: Annotated[bool, Field(title="Ok")]
    operation_id: Annotated[str, Field(title="Operation Id")]
    policy: Annotated[str, Field(title="Policy")]
    result_tokens: Annotated[int, Field(title="Result Tokens")]
    server: Annotated[str, Field(title="Server")]
    tool: Annotated[str, Field(title="Tool")]
    truncated: Annotated[bool, Field(title="Truncated")]
    wall_ms: Annotated[int, Field(title="Wall Ms")]


class McpConnectP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    hints: Annotated[dict[str, Any], Field(title="Hints")] = {}
    ok: Annotated[bool, Field(title="Ok")]
    server: Annotated[str, Field(title="Server")]
    tools: Annotated[list[str], Field(title="Tools")]
    transport: Annotated[str, Field(title="Transport")]


class ModelCallP(WireModel):
    """
    One model call of a box role (Athena's, the worker's), so a reader sees what each
    call was and what it said. Written once the call is answered and charged..
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    cost_usd: Annotated[float, Field(ge=0.0, title="Cost Usd")]
    model: Annotated[str | None, Field(title="Model")] = None
    refusal: Annotated[str | None, Field(title="Refusal")] = None
    request: Annotated[str, Field(min_length=1, title="Request")]
    role: Annotated[str, Field(min_length=1, title="Role")]
    rung: Annotated[str | None, Field(min_length=1, title="Rung")] = None
    sent_sha: Annotated[str | None, Field(title="Sent Sha")] = None
    stop_reason: Annotated[str, Field(title="Stop Reason")]
    text_preview: Annotated[str | None, Field(title="Text Preview")] = None
    tokens_in: Annotated[int, Field(ge=0, title="Tokens In")]
    tokens_out: Annotated[int, Field(ge=0, title="Tokens Out")]
    tools: Annotated[list[str], Field(title="Tools")] = []


class ModelSwitch(WireModel):
    """
    What `switch_model` changed: the payload `model_switched` journals.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    from_model: Annotated[str, Field(title="From Model")]
    model_id: Annotated[str, Field(title="Model Id")]
    rung: Annotated[str, Field(title="Rung")]
    rungs_sha: Annotated[str, Field(title="Rungs Sha")]
    to_model: Annotated[str, Field(title="To Model")]


class ModelSwitchedRecord(WireModel):
    """
    What `switch_model` changed: the payload `model_switched` journals.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    from_model: Annotated[str, Field(title="From Model")]
    model_id: Annotated[str, Field(title="Model Id")]
    rung: Annotated[str, Field(title="Rung")]
    rungs_sha: Annotated[str, Field(title="Rungs Sha")]
    to_model: Annotated[str, Field(title="To Model")]


class ModelWiring(WireModel):
    """
    A session's wiring, derived from the engine config and its `model_switched` lines.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    catalog: Annotated[list[str], Field(title="Catalog")]
    models: Annotated[dict[str, CatalogModel] | None, Field(title="Models")] = None
    roles: Annotated[dict[str, str] | None, Field(title="Roles")] = None
    rungs: Annotated[dict[str, str], Field(title="Rungs")]
    switches: Annotated[list[ModelSwitch], Field(title="Switches")]


class OneTask(WireModel):
    """
    Zeus's one-task plan as approved at stop 2.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    brief: Annotated[str, Field(min_length=1, title="Brief")]
    sha: Annotated[str, Field(title="Sha")]
    template_sha: Annotated[str, Field(title="Template Sha")]
    title: Annotated[str, Field(min_length=1, title="Title")]


class OpenWorkInfo(WireModel):
    """
    One piece of open work as a frontend reads it: what the start header
    says once, and what the person picks from when several are open.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    at: Annotated[str, Field(title="At")]
    key: Annotated[str, Field(title="Key")]
    mode: Annotated[Literal["worktree", "in_place"], Field(title="Mode")]
    session_id: Annotated[str, Field(title="Session Id")]
    title: Annotated[str | None, Field(title="Title")]
    work: Annotated[str | None, Field(title="Work")]
    worktree_path: Annotated[str | None, Field(title="Worktree Path")]


class Option(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    key: Annotated[str, Field(title="Key")]
    label: Annotated[str, Field(title="Label")]


class P(WireModel):
    """
    Base for every payload: strict, frozen, no extra fields.
    """

    model_config = ConfigDict(
        extra="ignore",
    )


class PartState(WireModel):
    """
    The design's or the plan's approval against what is on disk now.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    approved: Annotated[str | None, Field(title="Approved")]
    bound_to: Annotated[str | None, Field(title="Bound To")] = None
    current: Annotated[str | None, Field(title="Current")]
    version: Annotated[int, Field(title="Version")]


class PathDiff(WireModel):
    """
    One path's change between two trees, as `diff_stats` counts it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    added: Annotated[int, Field(title="Added")]
    kind: Annotated[Literal["created", "edited", "deleted"], Field(title="Kind")]
    path: Annotated[str, Field(title="Path")]
    removed: Annotated[int, Field(title="Removed")]


class PlanOutgrownP(WireModel):
    """
    A one-task plan ended `OUTGROWN` by Athena: its finishing
    needs a decision, a problem or a scope its brief does not hold. Journaled once, by her
    `outgrown` call, so a resume ends the episode as she did; her report is her reply after it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    call_id: Annotated[str, Field(title="Call Id")]


class PlanTaskP(WireModel):
    """
    One task an approved plan lists: its `Tasks` line's id and title, as
    `work.plan_tasks` reads them, the reading `delegate(task)` resolves an id through.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    id: Annotated[str, Field(title="Id")]
    title: Annotated[str, Field(title="Title")]


class PlanWrittenP(WireModel):
    """
    A plan written: Athena's, by her `plan` call,
    or Zeus's one-task plan, its brief alone. The first is asked at stop 2; every
    later one is a revision, shown and never asked.

    The spec's row named a `text_sha` beside `sha`: blobs are content-addressed by the sha256
    of their bytes, so the text's blob is `sha` itself, and one field holds both.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    asked: Annotated[bool, Field(title="Asked")]
    bound: Annotated[str | None, Field(title="Bound")] = None
    by: Annotated[Literal["zeus", "athena"], Field(title="By")]
    call_id: Annotated[str | None, Field(title="Call Id")] = None
    inherited_from: Annotated[str | None, Field(title="Inherited From")] = None
    note_sha: Annotated[str | None, Field(title="Note Sha")] = None
    outcome: Annotated[Literal["approved", "cancelled"], Field(title="Outcome")] = "approved"
    sha: Annotated[str, Field(title="Sha")]
    tasks: Annotated[list[PlanTaskP], Field(title="Tasks", validate_default=True)] = []
    version: Annotated[int, Field(ge=1, title="Version")]


class Pointer(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    edge_class: Annotated[str | None, Field(title="Edge Class")] = None
    line_end: Annotated[int, Field(title="Line End")]
    line_start: Annotated[int, Field(title="Line Start")]
    path: Annotated[str, Field(title="Path")]
    symbol: Annotated[str | None, Field(title="Symbol")] = None


class Promoted(WireModel):
    """
    `concern_promoted`: a registered check now watches it. Watched, not resolved.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    check_id: Annotated[str, Field(title="Check Id")]
    event: Annotated[Literal["promoted"], Field(title="Event")] = "promoted"
    id: Annotated[str, Field(title="Id")]


class QuestionAnsweredP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    choice: Annotated[str, Field(title="Choice")]
    elapsed_ms: Annotated[int, Field(title="Elapsed Ms")]
    id: Annotated[str, Field(title="Id")]
    source: Annotated[str, Field(title="Source")]


class QuestionAnsweredRecord(WireModel):
    """
    The choice, who gave it, and how long the wait was. `source` is an `AnswerSource` for a
    question and an `ApprovalDecision.source` for an approval, as given.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    choice: Annotated[str, Field(title="Choice")]
    elapsed_ms: Annotated[int, Field(ge=0, title="Elapsed Ms")]
    id: Annotated[str, Field(title="Id")]
    note: Annotated[str | None, Field(title="Note")] = None
    scope: Annotated[str, Field(title="Scope")] = "once"
    source: Annotated[str, Field(title="Source")]


class QuestionAskedP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    blocking: Annotated[bool, Field(title="Blocking")]
    evidence: Annotated[list[str], Field(title="Evidence")]
    id: Annotated[str, Field(title="Id")]
    kind: Annotated[str, Field(title="Kind")]
    options: Annotated[list[str], Field(title="Options")]


class ReadAnsweredP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    cost_usd: Annotated[float, Field(title="Cost Usd")]
    low_confidence: Annotated[bool, Field(title="Low Confidence")]
    rung: Annotated[str | None, Field(min_length=1, title="Rung")] = None
    tier: Annotated[str, Field(title="Tier")]
    tokens: Annotated[int, Field(title="Tokens")]


class ReadTierP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    reason: Annotated[str, Field(title="Reason")]
    tier: Annotated[str, Field(title="Tier")]


class Reconsideration(WireModel):
    """
    `rejected_reconsidered`: we went back to it. The original line is untouched — both are
    true, we rejected it then and reopened it later.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    by: Annotated[Literal["zeus", "user"], Field(title="By")]
    event: Annotated[Literal["reconsidered"], Field(title="Event")] = "reconsidered"
    id: Annotated[str, Field(title="Id")]
    reason: Annotated[str, Field(title="Reason")]


class RedactionP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    count: Annotated[int, Field(title="Count")]
    kinds: Annotated[list[str], Field(title="Kinds")]
    target_blob: Annotated[str, Field(title="Target Blob")]


class RedoRecord(WireModel):
    """
    A person took an undo back, or one half of it: after the conversation
    has moved on, only the code half is offered, and the line says which.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    code: Annotated[bool, Field(title="Code")]
    conversation: Annotated[bool, Field(title="Conversation")]
    undo_seq: Annotated[int, Field(ge=1, title="Undo Seq")]


class RejectedAlt(WireModel):
    """
    A historical fact, never a live rule. There is no field saying whether the
    rejection was right, no anchor — a rejection is about a design, not a path — and no
    status: state is the lines.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    at: Annotated[int, Field(title="At")]
    by: Annotated[Literal["zeus", "user"], Field(title="By")]
    event: Annotated[Literal["noted"], Field(title="Event")] = "noted"
    id: Annotated[str, Field(title="Id")]
    reason: Annotated[str, Field(title="Reason")]
    recorded_by: Annotated[Literal["explicit", "checkpoint"], Field(title="Recorded By")]
    scope: Annotated[Literal["session", "design", "packet_set"], Field(title="Scope")]
    scope_id: Annotated[str | None, Field(title="Scope Id")] = None
    text: Annotated[str, Field(title="Text")]


class Restored(WireModel):
    """
    What a task or operation undo restored, in which workspace.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    beside: Annotated[list[str], Field(title="Beside")] = []
    kept: Annotated[list[str], Field(title="Kept")] = []
    kept_by_default: Annotated[list[str], Field(title="Kept By Default")] = []
    restored: Annotated[list[str], Field(title="Restored")]
    root: Annotated[str, Field(title="Root")]


class RetiredGate(WireModel):
    """
    A design gate's question as a log written before the stops holds it: its id, so
    its answer still pairs with it. Nothing asks one since the stops replaced it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    id: Annotated[str, Field(title="Id")]


class RewoundCode(WireModel):
    """
    What an undo did to one workspace's code: enough for a redo to
    put it back, and for a reader to know where it happened.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    branch_head: Annotated[str | None, Field(title="Branch Head")] = None
    committed: Annotated[str | None, Field(title="Committed")] = None
    conflicts_kept: Annotated[list[str], Field(title="Conflicts Kept")] = []
    key: Annotated[str, Field(title="Key")]
    mode: Annotated[Literal["worktree", "in_place"], Field(title="Mode")]
    redo_ref: Annotated[str, Field(title="Redo Ref")]
    redo_tree: Annotated[str, Field(title="Redo Tree")]
    restored: Annotated[list[str], Field(title="Restored")] = []
    root: Annotated[str, Field(title="Root")]
    target: Annotated[str, Field(title="Target")]


class RouteP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    confidence: Annotated[float, Field(title="Confidence")]
    mode: Annotated[str, Field(title="Mode")]
    read_type: Annotated[str, Field(title="Read Type")]
    signals: Annotated[list[str], Field(title="Signals")]


class RunAt(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    at_task: Annotated[str | None, Field(title="At Task")]
    outcome: Annotated[str | None, Field(title="Outcome")]
    run_id: Annotated[str, Field(title="Run Id")]
    state: Annotated[Literal["live", "suspended", "ended"], Field(title="State")]


class RunBlob(WireModel):
    """
    A run blob as a frontend reads it: its text, or why there is none.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    field: Annotated[str, Field(title="Field")]
    run_id: Annotated[str, Field(title="Run Id")]
    sha: Annotated[str, Field(title="Sha")]
    size: Annotated[int | None, Field(title="Size")]
    text: Annotated[str | None, Field(title="Text")]
    withheld: Annotated[str | None, Field(title="Withheld")] = None


class LiftedRung(RootModel[str]):
    root: Annotated[str, Field(min_length=1)]


type RungsAdditionalProperty = Annotated[float, Field(gt=0.0)]


class RunCapsP(WireModel):
    """
    Every cap a run started under, and every one the person lifted.

    "Lifted" is said, never left as a `null`: the journal writes with `exclude_none`, so a
    `null` total would vanish on the way to disk and read back as a writer that forgot it.
    A rung with no cap configured is in neither `rungs` nor `lifted_rungs`.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    lifted_rungs: Annotated[
        list[LiftedRung], Field(title="Lifted Rungs", validate_default=True)
    ] = []
    rungs: Annotated[
        dict[Annotated[str, StringConstraints(min_length=1)], RungsAdditionalProperty] | None,
        Field(title="Rungs"),
    ] = None
    total: Annotated[float | None, Field(gt=0.0, title="Total")] = None
    total_lifted: Annotated[bool, Field(title="Total Lifted")] = False


class RunDiff(WireModel):
    """
    A task's or one step's change, between two trees the journal names.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    base: Annotated[str, Field(title="Base")]
    files: Annotated[list[PathDiff], Field(title="Files")]
    operation_id: Annotated[str | None, Field(title="Operation Id")]
    run_id: Annotated[str, Field(title="Run Id")]
    task_id: Annotated[str | None, Field(title="Task Id")]
    text: Annotated[str, Field(title="Text")]
    tip: Annotated[str | None, Field(title="Tip")]


class RunEndP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    committed: Annotated[int, Field(ge=0, title="Committed")]
    cost_usd: Annotated[float, Field(title="Cost Usd")]
    handoff_path: Annotated[str | None, Field(title="Handoff Path")] = None
    outcome: Annotated[
        Literal["COMPLETED", "PARTIAL", "BLOCKED", "ASK", "BUDGET_CAP", "ABORTED", "OUTGROWN"],
        Field(title="Outcome"),
    ]
    tasks: Annotated[int, Field(ge=0, title="Tasks")]
    tokens: Annotated[int, Field(title="Tokens")]
    wall_ms: Annotated[int, Field(title="Wall Ms")]
    why: Annotated[str | None, Field(title="Why")] = None


class RunHeader(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    lines: Annotated[list[HeaderLine], Field(title="Lines")]
    sandbox: Annotated[str, Field(title="Sandbox")] = ""
    session_id: Annotated[str, Field(title="Session Id")]
    uncommitted: Annotated[list[str], Field(title="Uncommitted")] = []


class RunMetrics(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    budget_hard_caps: Annotated[int, Field(title="Budget Hard Caps")] = 0
    budget_warnings: Annotated[int, Field(title="Budget Warnings")] = 0
    cost_usd: Annotated[float, Field(title="Cost Usd")] = 0.0
    isolation: Annotated[str, Field(title="Isolation")] = ""
    model_calls: Annotated[int, Field(title="Model Calls")] = 0
    operations: Annotated[int, Field(title="Operations")] = 0
    operations_ambiguous: Annotated[int, Field(title="Operations Ambiguous")] = 0
    operations_incomplete: Annotated[int, Field(title="Operations Incomplete")] = 0
    outcome: Annotated[str | None, Field(title="Outcome")] = None
    plans: Annotated[int, Field(title="Plans")] = 0
    reviews: Annotated[dict[str, int] | None, Field(title="Reviews")] = None
    round_ends: Annotated[dict[str, int] | None, Field(title="Round Ends")] = None
    rounds: Annotated[int, Field(title="Rounds")] = 0
    run_id: Annotated[str, Field(title="Run Id")] = ""
    tasks_accepted: Annotated[int, Field(title="Tasks Accepted")] = 0
    tasks_committed: Annotated[int, Field(title="Tasks Committed")] = 0
    tasks_total: Annotated[int, Field(title="Tasks Total")] = 0
    tokens_in: Annotated[int, Field(title="Tokens In")] = 0
    tokens_out: Annotated[int, Field(title="Tokens Out")] = 0
    tool_calls: Annotated[int, Field(title="Tool Calls")] = 0
    tool_refusals: Annotated[int, Field(title="Tool Refusals")] = 0
    wall_ms: Annotated[int, Field(title="Wall Ms")] = 0


class RunStartP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    caps: RunCapsP | None = None
    config_sha: Annotated[str, Field(title="Config Sha")]
    engine_version: Annotated[str, Field(title="Engine Version")]
    goal: Annotated[str, Field(title="Goal")]
    hard_cost_cap_usd: Annotated[float, Field(title="Hard Cost Cap Usd")]
    isolation: Annotated[Literal["worktree", "in_place"], Field(title="Isolation")]
    repo_head: Annotated[str, Field(title="Repo Head")]
    schema_version: Annotated[int, Field(title="Schema Version")]
    template_shas: Annotated[dict[str, str] | None, Field(title="Template Shas")] = None
    work_sha: Annotated[str, Field(title="Work Sha")]


class RungBudget(WireModel):
    """
    One rung's share of the family's spend, and the rung's own cap beneath the
    total. An engine value, like `SessionBudget`.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    cap_seq: Annotated[int | None, Field(title="Cap Seq")]
    cap_source: Annotated[Literal["config", "session", "none"], Field(title="Cap Source")]
    cap_usd: Annotated[float | None, Field(title="Cap Usd")]
    spent_usd: Annotated[float, Field(title="Spent Usd")]


class ScoutEscalatedP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    dropped_pointers: Annotated[int, Field(title="Dropped Pointers")]
    from_rung: Annotated[str, Field(title="From Rung")]
    reason: Annotated[str, Field(title="Reason")]
    to_rung: Annotated[str, Field(title="To Rung")]
    truncated: Annotated[bool, Field(title="Truncated")]


class ScoutEscapeP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    command: Annotated[str, Field(title="Command")]
    exec_seq: Annotated[int, Field(title="Exec Seq")]
    granted: Annotated[bool, Field(title="Granted")]
    reason: Annotated[str, Field(title="Reason")]


class ScoutReportP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    cost_usd: Annotated[float, Field(title="Cost Usd")]
    facts: Annotated[int, Field(title="Facts")]
    rung: Annotated[str | None, Field(min_length=1, title="Rung")] = None
    targets: Annotated[int, Field(title="Targets")]
    tokens: Annotated[int, Field(title="Tokens")]
    truncated: Annotated[bool, Field(title="Truncated")]
    unresolved: Annotated[int, Field(title="Unresolved")]


class ScoutStartP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    goal_sha: Annotated[str, Field(title="Goal Sha")]
    rung: Annotated[str, Field(title="Rung")]
    snapshot_sha: Annotated[str, Field(title="Snapshot Sha")]


class ScoutToolP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    args_sha: Annotated[str, Field(title="Args Sha")]
    dedup_hit: Annotated[bool, Field(title="Dedup Hit")]
    result_tokens: Annotated[int, Field(title="Result Tokens")]
    tool: Annotated[str, Field(title="Tool")]


class SecretDeniedP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    path: Annotated[str, Field(title="Path")]
    rung: Annotated[str, Field(title="Rung")]
    surface: Annotated[str, Field(title="Surface")]


class SecretSuspectedP(WireModel):
    """
    Advisory only: the content was read; nothing was withheld.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    detector: Annotated[str, Field(title="Detector")]
    line: Annotated[int, Field(title="Line")]
    path: Annotated[str, Field(title="Path")]


class SessionBudget(WireModel):
    """
    A session family's spend against its cap — what `/budget` shows. An
    engine value, computed per call and stored nowhere; not a schema contract
    (`BudgetStatus` stays retired).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    cap_seq: Annotated[int | None, Field(title="Cap Seq")]
    cap_source: Annotated[Literal["config", "session"], Field(title="Cap Source")]
    cap_usd: Annotated[float | None, Field(title="Cap Usd")]
    roles: Annotated[dict[str, float] | None, Field(title="Roles")] = None
    rungs: Annotated[dict[str, RungBudget] | None, Field(title="Rungs")] = None
    spent_usd: Annotated[float, Field(title="Spent Usd")]
    warning_fractions: Annotated[list[float], Field(title="Warning Fractions")] = []


class SessionEndP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    cost_usd: Annotated[float, Field(title="Cost Usd")]
    turns: Annotated[int, Field(title="Turns")]
    wall_ms: Annotated[int, Field(title="Wall Ms")]


class SessionInfo(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    continuing: OpenWorkInfo | None = None
    created_at: Annotated[str, Field(title="Created At")] = ""
    last_run_id: Annotated[str | None, Field(title="Last Run Id")] = None
    last_seq: Annotated[int, Field(title="Last Seq")]
    session_id: Annotated[str, Field(title="Session Id")]
    title: Annotated[str | None, Field(title="Title")] = None
    updated_at: Annotated[str, Field(title="Updated At")] = ""


class SessionStartP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    profile: Annotated[str, Field(title="Profile")]
    repo: Annotated[str, Field(title="Repo")]
    rungs_sha: Annotated[str, Field(title="Rungs Sha")]
    session_id: Annotated[str, Field(title="Session Id")]


class Shape(WireModel):
    """
    Enough to decide whether to fetch, without fetching.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    bytes: Annotated[int, Field(title="Bytes")]
    est_tokens: Annotated[int | None, Field(title="Est Tokens")] = None
    height: Annotated[int | None, Field(title="Height")] = None
    kind: Annotated[str, Field(title="Kind")]
    lines: Annotated[int | None, Field(title="Lines")] = None
    pages: Annotated[int | None, Field(title="Pages")] = None
    symbols: Annotated[int | None, Field(title="Symbols")] = None
    width: Annotated[int | None, Field(title="Width")] = None


class ShellEffectRecord(WireModel):
    """
    One D40 record of Zeus's shell on the session log (`zeus/writes.SessionEffects`): a
    step's or a job's intent, start or completion. Folded here, so the session knows which jobs
    are running from its log alone (a running job is not a safe point).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    event: Annotated[
        Literal[
            "run_start",
            "effect_intended",
            "effect_started",
            "effect_completed",
            "budget",
            "exec",
            "route",
            "read_tier",
            "citation_gate",
            "read_answered",
            "scout_start",
            "scout_tool",
            "scout_escape",
            "scout_escalated",
            "scout_report",
            "session_start",
            "turn",
            "interrupt",
            "session_end",
            "question_asked",
            "question_answered",
            "context_tokens",
            "untrusted_read",
            "mcp_connect",
            "mcp_call",
            "external_call",
            "secret_denied",
            "secret_suspected",
            "redaction",
            "hook",
            "callback_error",
            "run_end",
            "tool_call",
            "model_call",
            "task_delegated",
            "task_reported",
            "box_message",
            "box_compaction",
            "task_reviewed",
            "plan_written",
            "plan_outgrown",
            "agent_start",
            "agent_message",
            "agent_report",
            "tell_delivered",
            "tell_reply",
        ],
        Field(title="EventKind"),
    ]
    payload: P


class Spend(WireModel):
    """
    A read-only snapshot for the stream and the handoff.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    cache_writes: Annotated[int, Field(title="Cache Writes")] = 0
    cached_in: Annotated[int, Field(title="Cached In")]
    cost_usd: Annotated[float, Field(title="Cost Usd")]
    estimated_calls: Annotated[int, Field(title="Estimated Calls")] = 0
    scope: Annotated[Literal["session", "run", "packet"], Field(title="Scope")]
    tokens_in: Annotated[int, Field(title="Tokens In")]
    tokens_out: Annotated[int, Field(title="Tokens Out")]
    wall_ms: Annotated[int, Field(title="Wall Ms")]


class SpendRecord(WireModel):
    """
    One Zeus model call's charge: what the ledger charged,
    as the session log keeps it, so a session's spend outlives the process that spent it.
    `model` is the catalog name that served the call, never the rung; `rung` is the
    rung that made it, named at composition because two rungs may share a model.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    cache_writes: Annotated[int, Field(ge=0, title="Cache Writes")]
    cached_in: Annotated[int, Field(ge=0, title="Cached In")]
    cost_usd: Annotated[int | float, Field(title="Cost Usd")]
    estimated: Annotated[bool, Field(title="Estimated")]
    model: Annotated[str, Field(title="Model")]
    role: Annotated[str | None, Field(min_length=1, title="Role")] = None
    rung: Annotated[str, Field(min_length=1, title="Rung")] = "zeus"
    tokens_in: Annotated[int, Field(ge=0, title="Tokens In")]
    tokens_out: Annotated[int, Field(ge=0, title="Tokens Out")]


class SyntaxFactP(WireModel):
    """
    One changed file's syntax, judged by its own language's profile over the file as the
    task left it: the same check the edit tool reported.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    depth: Annotated[Literal["compiled", "command", "parsed"] | None, Field(title="Depth")] = (
        None
    )
    detail: Annotated[str | None, Field(title="Detail")] = None
    ok: Annotated[bool, Field(title="Ok")]
    path: Annotated[str, Field(title="Path")]


class TaskDelegatedP(WireModel):
    """
    A task given to the worker: by Athena's `delegate` call, or by Zeus as a
    one-task plan. Its id is the harness's, never the model's.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    brief_sha: Annotated[str, Field(title="Brief Sha")]
    by: Annotated[Literal["zeus", "athena"], Field(title="By")]
    call_id: Annotated[str | None, Field(title="Call Id")] = None
    depth: Annotated[int, Field(ge=0, title="Depth")] = 0
    plan_task: Annotated[str | None, Field(title="Plan Task")] = None
    start_commit: Annotated[str | None, Field(title="Start Commit")] = None
    start_tree: Annotated[str, Field(title="Start Tree")]
    task_id: Annotated[
        str,
        Field(
            pattern="^T[1-9][0-9]{0,5}(\\.[1-9][0-9]{0,3}){0,4}(@([2-9]|[1-9][0-9]{1,3}))?$",
            title="Task Id",
        ),
    ]
    template_sha: Annotated[str | None, Field(title="Template Sha")] = None
    title: Annotated[str, Field(title="Title")]


class TaskFactsP(WireModel):
    """
    What a task did, measured from the trees and the journal, never from the
    report.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    commands: Annotated[list[CommandFactP], Field(title="Commands", validate_default=True)] = []
    end_tree: Annotated[str, Field(title="End Tree")]
    files: Annotated[list[FileFactP], Field(title="Files", validate_default=True)] = []
    skipped: Annotated[list[str], Field(title="Skipped")] = []
    start_tree: Annotated[str, Field(title="Start Tree")]
    symbols_added: Annotated[list[str], Field(title="Symbols Added")] = []
    symbols_known: Annotated[bool, Field(title="Symbols Known")] = True
    symbols_removed: Annotated[list[str], Field(title="Symbols Removed")] = []
    syntax: Annotated[list[SyntaxFactP], Field(title="Syntax", validate_default=True)] = []
    tests_added: Annotated[list[str], Field(title="Tests Added")] = []
    tests_edited: Annotated[list[str], Field(title="Tests Edited")] = []


class TaskReportedP(WireModel):
    """
    One worker round's end: why it stopped, its words, and the facts measured.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    facts: TaskFactsP
    report_sha: Annotated[str | None, Field(title="Report Sha")] = None
    round: Annotated[int, Field(ge=1, title="Round")]
    stopped: Annotated[
        Literal[
            "replied",
            "limit",
            "looped",
            "context_full",
            "model_error",
            "harness_error",
            "halted",
            "held",
        ],
        Field(title="Stopped"),
    ]
    task_id: Annotated[
        str,
        Field(
            pattern="^T[1-9][0-9]{0,5}(\\.[1-9][0-9]{0,3}){0,4}(@([2-9]|[1-9][0-9]{1,3}))?$",
            title="Task Id",
        ),
    ]


class TaskReviewedP(WireModel):
    """
    Athena's answer to a task's report, by her
    `review` call. On `feedback` the text is the worker's next message; on `re_cut` the tasks
    she delegates next are its replacements.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    answer: Annotated[
        Literal["accept", "feedback", "continue", "take_over", "abandon", "re_cut", "drop"],
        Field(title="Answer"),
    ]
    by: Annotated[str | None, Field(title="By")] = None
    call_id: Annotated[str, Field(title="Call Id")]
    fresh: Annotated[bool | list[str] | None, Field(title="Fresh")] = None
    task_id: Annotated[
        str,
        Field(
            pattern="^T[1-9][0-9]{0,5}(\\.[1-9][0-9]{0,3}){0,4}(@([2-9]|[1-9][0-9]{1,3}))?$",
            title="Task Id",
        ),
    ]
    text_sha: Annotated[str | None, Field(title="Text Sha")] = None


class TaskStatus(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    commit: Annotated[str | None, Field(title="Commit")] = None
    id: Annotated[str, Field(title="Id")]
    key: Annotated[str | None, Field(title="Key")] = None
    run: Annotated[str | None, Field(title="Run")] = None
    state: Annotated[
        Literal[
            "pending", "open", "held", "committed", "unchecked", "dropped", "replaced", "lost"
        ],
        Field(title="State"),
    ]
    title: Annotated[str, Field(title="Title")]


class TellDeliveredP(WireModel):
    """
    A message from Zeus (a session's `tell` line) delivered to Athena in this run: at the
    boundary of the worker's round it held, or before her next model call with no
    round open.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    round: Annotated[int | None, Field(ge=1, title="Round")] = None
    task_id: Annotated[str | None, Field(title="Task Id")] = None
    tell: Annotated[str, Field(pattern="^tell-[1-9][0-9]*$", title="Tell")]


class FromItem(RootModel[int]):
    root: Annotated[int, Field(ge=1)]


class TellRecord(WireModel):
    """
    A `tell` line: Zeus's message to the team about run `run_id`, in his words,
    the person its authority. `sources` are the person's turns it came from, each a user turn
    or an `ask_user` answer. Its text is a session blob; the run
    delivers it at its next boundary and journals `tell_delivered`.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    from_: Annotated[list[FromItem], Field(alias="from", min_length=1, title="From")]
    id: Annotated[str, Field(pattern="^tell-[1-9][0-9]*$", title="Id")]
    key: Annotated[str | None, Field(title="Key")] = None
    run_id: Annotated[str, Field(title="Run Id")]
    text_sha: Annotated[str, Field(title="Text Sha")]


class TellReplyP(WireModel):
    """
    Athena's reply to a delivered tell, by her `reply` call: owed for each one.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    call_id: Annotated[str, Field(title="Call Id")]
    outcome: Annotated[
        Literal["applied", "planned", "needs_design", "declined"], Field(title="Outcome")
    ]
    tell: Annotated[str, Field(pattern="^tell-[1-9][0-9]*$", title="Tell")]
    text_sha: Annotated[str, Field(title="Text Sha")]


class TellStatus(WireModel):
    """
    A tell to the work's team: `sent` until a run delivers it, `delivered` until
    Athena replies, `replied` with her outcome; `unanswered` when the run it was delivered in
    ended with no reply.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    id: Annotated[str, Field(title="Id")]
    outcome: Annotated[
        Literal["applied", "planned", "needs_design", "declined"] | None, Field(title="Outcome")
    ] = None
    state: Annotated[
        Literal["sent", "delivered", "replied", "unanswered"], Field(title="State")
    ]


class TitleRecord(WireModel):
    """
    The session's name: the `title` role's, or the person's (`/rename`).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    by: Annotated[Literal["model", "person"], Field(title="By")]
    text: Annotated[str, Field(title="Text")]


class ToolCallP(WireModel):
    """
    One tool call by any role, through the one dispatcher.
    `scout_tool` stays, and `journal.projection.tool_call` reads both as this. A call was
    refused exactly when `denied_reason` is set.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    args_preview: Annotated[str | None, Field(title="Args Preview")] = None
    args_sha: Annotated[str, Field(title="Args Sha")]
    dedup_hit: Annotated[bool, Field(title="Dedup Hit")] = False
    denied_reason: Annotated[str | None, Field(title="Denied Reason")] = None
    form: Annotated[
        Literal["native", "tool_line", "need", "json", "json_fenced"] | None,
        Field(title="Form"),
    ] = None
    outcome: Annotated[Literal["served", "repack"], Field(title="Outcome")]
    result_preview: Annotated[str | None, Field(title="Result Preview")] = None
    result_sha: Annotated[str | None, Field(title="Result Sha")] = None
    result_tokens: Annotated[int, Field(ge=0, title="Result Tokens")]
    role: Annotated[str, Field(min_length=1, title="Role")]
    tool: Annotated[str, Field(min_length=1, title="Tool")]


class Touched(WireModel):
    """
    `concern_touched`: a run modified an anchored path. Records that something happened
    NEAR the concern and decides nothing about it.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    event: Annotated[Literal["touched"], Field(title="Event")] = "touched"
    id: Annotated[str, Field(title="Id")]
    path: Annotated[str, Field(title="Path")]
    run_id: Annotated[str | None, Field(title="Run Id")] = None


class TurnP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    artifact_id: Annotated[str, Field(title="Artifact Id")]
    input_sha: Annotated[str, Field(title="Input Sha")]
    mode: Annotated[str, Field(title="Mode")]


class UndoRecord(WireModel):
    """
    A person took back turns `to_turn` onward: the line is appended, and
    nothing is deleted. With `conversation`, the fold hides what `rewind.REWIND` says a rewind
    hides, from `to_turn` up to this line. With `code`, `workspaces` says what was restored in
    each.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    code: Annotated[bool, Field(title="Code")]
    conversation: Annotated[bool, Field(title="Conversation")]
    to_turn: Annotated[int, Field(ge=1, title="To Turn")]
    workspaces: Annotated[
        list[RewoundCode], Field(title="Workspaces", validate_default=True)
    ] = []


class UndoReport(WireModel):
    """
    What an undo or a redo did, and what it could not do (decision 3).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    code: Annotated[bool, Field(title="Code")]
    conversation: Annotated[bool, Field(title="Conversation")]
    not_undone: Annotated[
        list[ForkEffect], Field(title="Not Undone", validate_default=True)
    ] = []
    prompt: Annotated[str | None, Field(title="Prompt")] = None
    seq: Annotated[int, Field(title="Seq")]
    to_turn: Annotated[int, Field(title="To Turn")]
    withheld: Annotated[list[str], Field(title="Withheld")] = []
    workspaces: Annotated[
        list[CodeOutcome], Field(title="Workspaces", validate_default=True)
    ] = []


class Undoable(WireModel):
    """
    A turn a person can take back, and how (decision 1).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    no_code: Annotated[str | None, Field(title="No Code")] = None
    options: Annotated[list[Literal["both", "conversation", "code"]], Field(title="Options")]
    prompt: Annotated[str, Field(title="Prompt")]
    turn: Annotated[int, Field(title="Turn")]


class Unresolved(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    candidates: Annotated[list[str], Field(title="Candidates")] = []
    question: Annotated[str, Field(title="Question")]


class UntrustedReadP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    bytes: Annotated[int, Field(title="Bytes")]
    path: Annotated[str, Field(title="Path")]
    rung: Annotated[str, Field(title="Rung")]
    span: Annotated[str, Field(title="Span")]


class WorkKey(WireModel):
    """
    Which work: a design's folder (`dir`, `name`), or a one-task plan, named by the run that
    delegated it first.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    dir: Annotated[str | None, Field(title="Dir")] = None
    first_run: Annotated[str | None, Field(title="First Run")] = None
    name: Annotated[str | None, Field(title="Name")] = None


class WorkPin(WireModel):
    """
    The work folder a design was approved in: its name, `work.dir`
    then, and the `design/` tree stop 1 pinned, which every later read of it is answered from
    (`workspace.pinned_text`).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    dir: Annotated[str, Field(min_length=1, title="Dir")]
    name: Annotated[str, Field(min_length=1, title="Name")]
    tree: Annotated[str, Field(min_length=1, title="Tree")]


class WorkProgress(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    design: PartState
    key: WorkKey
    plan: PartState
    plan_run: Annotated[str | None, Field(title="Plan Run")]
    plan_sha: Annotated[str | None, Field(title="Plan Sha")]
    plan_version: Annotated[int | None, Field(title="Plan Version")]
    run: RunAt | None
    runs: Annotated[list[str], Field(title="Runs")]
    tasks: Annotated[list[TaskStatus], Field(title="Tasks")]
    tells: Annotated[list[TellStatus], Field(title="Tells", validate_default=True)] = []


class WorkRecord(WireModel):
    """
    The session's work folder: opened by Zeus's first `doc` call, closed
    by the handoff. At most one is open: the latest line, while its state is `open`.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    dir: Annotated[str, Field(title="Dir")]
    name: Annotated[str, Field(title="Name")]
    state: Annotated[Literal["open", "landed", "kept", "dropped"], Field(title="State")]


class WorkspaceRecord(WireModel):
    """
    The durable pointer a turn keeps to the workspace it ran in.

    A reference, deliberately: the worktree may be gone by the time anyone reads the turn
    back, and recording ids rather than a live object is what lets a resumed session say so
    instead of failing to open something it assumed was there.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    branch: Annotated[str | None, Field(title="Branch")] = None
    handoff: Annotated[
        Literal["pending", "apply", "squash", "keep", "discard", "bring"],
        Field(title="Handoff"),
    ]
    lease_id: Annotated[str, Field(title="Lease Id")]
    mode: Annotated[Literal["worktree", "in_place"], Field(title="Mode")]
    source_head: Annotated[str, Field(title="Source Head")]
    source_ref: Annotated[str, Field(title="Source Ref")]
    worktree_path: Annotated[str | None, Field(title="Worktree Path")] = None


class WorkspaceRef(WireModel):
    """
    The durable pointer a turn keeps to the workspace it ran in.

    A reference, deliberately: the worktree may be gone by the time anyone reads the turn
    back, and recording ids rather than a live object is what lets a resumed session say so
    instead of failing to open something it assumed was there.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    branch: Annotated[str | None, Field(title="Branch")] = None
    handoff: Annotated[
        Literal["pending", "apply", "squash", "keep", "discard", "bring"],
        Field(title="Handoff"),
    ]
    lease_id: Annotated[str, Field(title="Lease Id")]
    mode: Annotated[Literal["worktree", "in_place"], Field(title="Mode")]
    source_head: Annotated[str, Field(title="Source Head")]
    source_ref: Annotated[str, Field(title="Source Ref")]
    worktree_path: Annotated[str | None, Field(title="Worktree Path")] = None


class AgentRecord(WireModel):
    """
    One record of a spawned agent's round on the session log (`agents/spawn.py`): its
    `agent_start`, each `agent_message`, its `agent_report`. Folded here, so the session knows
    its agents, and what each was told and answered, from its log alone.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    event: Annotated[
        Literal[
            "run_start",
            "effect_intended",
            "effect_started",
            "effect_completed",
            "budget",
            "exec",
            "route",
            "read_tier",
            "citation_gate",
            "read_answered",
            "scout_start",
            "scout_tool",
            "scout_escape",
            "scout_escalated",
            "scout_report",
            "session_start",
            "turn",
            "interrupt",
            "session_end",
            "question_asked",
            "question_answered",
            "context_tokens",
            "untrusted_read",
            "mcp_connect",
            "mcp_call",
            "external_call",
            "secret_denied",
            "secret_suspected",
            "redaction",
            "hook",
            "callback_error",
            "run_end",
            "tool_call",
            "model_call",
            "task_delegated",
            "task_reported",
            "box_message",
            "box_compaction",
            "task_reviewed",
            "plan_written",
            "plan_outgrown",
            "agent_start",
            "agent_message",
            "agent_report",
            "tell_delivered",
            "tell_reply",
        ],
        Field(title="EventKind"),
    ]
    payload: P


class ApprovalStop(WireModel):
    """
    What the person is shown at a stop: the markdown, exactly. Approving binds `sha`, the
    text last shown (invariant 15); an edit or a note makes a new text, shown again. Not
    `Stop`, the spec's first name for it: `tools/rules.Stop` is the dispatcher's verdict.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    author: Annotated[Literal["zeus", "athena"], Field(title="Author")]
    folder: Annotated[str | None, Field(title="Folder")] = None
    id: Annotated[str, Field(title="Id")]
    kind: Annotated[Literal["design", "plan"], Field(title="StopKind")]
    notice: Annotated[str | None, Field(title="Notice")] = None
    refused: Annotated[str | None, Field(title="Refused")] = None
    round: Annotated[int, Field(title="Round")] = 1
    sha: Annotated[str, Field(title="Sha")]
    source_ref: Annotated[str | None, Field(title="Source Ref")] = None
    text: Annotated[str, Field(title="Text")]
    workspace: Annotated[Literal["worktree", "in_place"], Field(title="Workspace")]


class ApprovedDesign(WireModel):
    """
    A design as approved at stop 1.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    sha: Annotated[str, Field(title="Sha")]
    template_sha: Annotated[str, Field(title="Template Sha")]
    text: Annotated[str, Field(min_length=1, title="Text")]
    work: WorkPin | None = None


class Call(WireModel):
    """
    One tool call an assistant turn made. Its arguments are kept exactly as the model
    sent them: no layer ever rewrites a call (invariant 19).
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    arguments: Annotated[dict[str, Any], Field(title="Arguments")]
    id: Annotated[str, Field(title="Id")]
    name: Annotated[str, Field(title="Name")]


class CoderRequest(WireModel):
    """
    `design` or `task`, exactly one, and where the work runs.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    attachments: Annotated[
        list[AttachedFile], Field(title="Attachments", validate_default=True)
    ] = []
    continues: Continuation | None = None
    design: ApprovedDesign | None = None
    task: OneTask | None = None
    workspace: Annotated[Literal["worktree", "in_place"], Field(title="Workspace")] = "worktree"


class CompactionRecord(WireModel):
    """
    One committed compaction: the whole cut in one line, so a crash can
    never leave a block without its standing, or a boundary without its block.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    block: Annotated[str, Field(title="Block")]
    boundary: Annotated[int, Field(title="Boundary")]
    facts: Annotated[dict[str, Any], Field(title="Facts")]
    lifted: Annotated[list[int], Field(title="Lifted")] = []
    previous: Annotated[int, Field(title="Previous")]
    standing: Annotated[str, Field(title="Standing")]


class EpisodeRecord(WireModel):
    """
    Payload of an `episode` line — one per `coder` call, written at dispatch.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    constraints: Annotated[list[str], Field(title="Constraints")] = []
    design_sha: Annotated[str | None, Field(title="Design Sha")] = None
    opened_at_seq: Annotated[int | None, Field(title="Opened At Seq")] = None
    path: Annotated[Literal["exact", "planned"], Field(title="Path")]
    run_id: Annotated[str, Field(title="Run Id")]
    tier: Literal["t0", "t1", "t2"] | None = None


class FactHit(WireModel):
    """
    One fact row that fired, with the evidence that fired it.

    Here rather than in `chickadee/approval/` for `Tier`'s reason: a person's
    dispatch-time confirmation carries the rows that raised the tier (`ActionApproval.hits`,
    contract 6), and `schema/` cannot import from the package that computes them.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    command_class: Annotated[str | None, Field(title="Command Class")] = None
    evidence: Annotated[str, Field(title="Evidence")]
    floor: Annotated[Literal["t0", "t1", "t2"], Field(title="Tier")]
    """
    How much the user is consulted before a change happens.

    Ordered — `TIER_ORDER` is the order, and `max()` over it is what makes "never
    de-escalate" (invariant 17) structurally impossible to violate rather than a policy
    someone must remember to enforce.

    It lives here rather than in `chickadee/approval/` because a G0 contract carries it and
    `schema/` cannot import from a package built three phases later. That is also the right
    home on the merits: a tier is written down and read back — by the journal, the handoff,
    the visibility surface — and what crosses that boundary is a wire type.
    """
    lowerable: Annotated[bool, Field(title="Lowerable")]
    path: Annotated[str | None, Field(title="Path")] = None
    row: Annotated[int, Field(title="Row")]


class Findings(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    callers: Annotated[list[Pointer], Field(title="Callers", validate_default=True)] = []
    evidence: Annotated[list[Pointer], Field(title="Evidence", validate_default=True)] = []
    facts: Annotated[list[str], Field(title="Facts")] = []
    targets: Annotated[list[Pointer], Field(title="Targets", validate_default=True)] = []
    truncated: Annotated[bool, Field(title="Truncated")] = False
    unresolved: Annotated[
        list[Unresolved], Field(title="Unresolved", validate_default=True)
    ] = []


class Handle(WireModel):
    """
    Stable within the session; rehydratable.

    The first six fields are the spec's. The next three say where the bytes are, which the
    spec leaves to "resolves via": a file handle names its `path`; a stored handle names the
    run whose blob store holds it, or None for the session's own (a checkpoint's output
    lives in its run's store); a journal handle names its run and, when it is narrower than
    the whole journal, its seq range. `name` is a stored file's own name, when it had one.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    content_sha: Annotated[str, Field(title="Content Sha")]
    id: Annotated[str, Field(title="Id")]
    kind: Annotated[
        Literal["file", "diff", "log", "journal", "findings", "output", "attachment"],
        Field(title="Kind"),
    ]
    name: Annotated[str | None, Field(title="Name")] = None
    ops: Annotated[list[str], Field(title="Ops")] = []
    path: Annotated[str | None, Field(title="Path")] = None
    preview: Annotated[str, Field(title="Preview")]
    run_id: Annotated[str | None, Field(title="Run Id")] = None
    seqs: Annotated[list[int] | None, Field(title="Seqs")] = None
    shape: Shape


class HandoffSummary(WireModel):
    """
    What a session leaves behind, and which completion choices are live right now. The
    git half is `HandoffPreview` carried whole, whose `refusal(choice)` `finalize()` itself
    asks, so a summary never offers a choice `finalize` then refuses.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    diff: Annotated[str, Field(title="Diff")] = ""
    episodes: Annotated[list[str], Field(title="Episodes")]
    git: HandoffPreview | None
    last_outcome: Annotated[
        Literal["COMPLETED", "PARTIAL", "BLOCKED", "ASK", "BUDGET_CAP", "ABORTED", "OUTGROWN"]
        | None,
        Field(title="Last Outcome"),
    ]
    message: Annotated[str, Field(title="Message")] = ""
    notes: Annotated[list[str], Field(title="Notes")] = []
    offers: Annotated[dict[str, str] | None, Field(title="Offers")] = None
    refused: Annotated[dict[str, str] | None, Field(title="Refused")] = None
    session_id: Annotated[str, Field(title="Session Id")]
    spend: Spend
    unverified: Annotated[list[str], Field(title="Unverified")]
    workspace: WorkspaceRef | None = None


class HarnessTurnP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    carries: Handle | None = None
    handle: Annotated[str | None, Field(title="Handle")] = None
    role: Annotated[Literal["harness"], Field(title="Role")] = "harness"
    text: Annotated[str, Field(title="Text")]
    trigger: Annotated[int, Field(title="Trigger")]


class Liveness(WireModel):
    """
    Computed on every read, stored nowhere. It carries why: `abandoned` with no `last_seq`
    would be an accusation rather than a finding.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    index_progress: IndexProgress | None = None
    indexing: Annotated[str | None, Field(title="Indexing")] = None
    kind: Annotated[
        Literal["idle", "awaiting", "working", "abandoned", "unknown"],
        Field(title="LivenessKind"),
    ]
    last_seq: Annotated[int, Field(title="Last Seq")]
    outcome: Annotated[str | None, Field(title="Outcome")] = None
    owner_host: Annotated[str | None, Field(title="Owner Host")] = None
    paused_run: Annotated[str | None, Field(title="Paused Run")] = None
    question_id: Annotated[str | None, Field(title="Question Id")] = None
    team_run: Annotated[str | None, Field(title="Team Run")] = None
    why: Annotated[str | None, Field(title="Why")] = None


class Question(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    blocking: Annotated[bool, Field(title="Blocking")] = True
    default: Annotated[str, Field(title="Default")]
    evidence: Annotated[list[str], Field(title="Evidence")] = []
    id: Annotated[str, Field(title="Id")]
    kind: Annotated[
        Literal[
            "clarify_goal",
            "plan_approval",
            "route_ambiguous",
            "design_fork",
            "escalation",
            "replan_premise",
            "scope_expansion",
            "undo_conflict",
            "fork_code",
            "work_folder",
        ],
        Field(title="QuestionKind"),
    ]
    """
    The closed ask-point list. Another member is a decision, never a commit.
    """
    options: Annotated[list[Option], Field(title="Options", validate_default=True)] = []
    text: Annotated[str, Field(title="Text")]


class ToolRecord(WireModel):
    """
    A `tool` line's payload. Its run is the envelope's, so the record (`ToolLine`) is
    built from this and the line.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    call_id: Annotated[str, Field(title="Call Id")]
    handle: Handle | None = None
    image: ImageRef | None = None
    outcome: CoderOutcome | None = None
    ran: Annotated[bool, Field(title="Ran")]
    supersedes: Annotated[list[str], Field(title="Supersedes")] = []
    text: Annotated[str, Field(title="Text")]
    view: Annotated[list[list[int]], Field(title="View")] = []


class ActionApproval(WireModel):
    """
    One concrete effect, at dispatch time — the
    only ask point where the argv is real rather than declared.

    **This is the question, not the record.** The durable record is contract 10's
    `ActionAuthorization`, and an `approve` here is what a `Confirmation` on it attests.
    Keeping them apart lets the question carry what a record should not — a rendered
    `display` line, the router's evidence — without touching a frozen row.

    Every field is an observation of the action about to run, never a claim about it.
    Added at G4.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    action_id: Annotated[str, Field(title="Action Id")]
    asker: Annotated[Literal["zeus", "athena", "worker"], Field(title="Asker")] = "zeus"
    command: Annotated[str | None, Field(title="Command")] = None
    credentials: Annotated[list[str], Field(title="Credentials")] = []
    cwd: Annotated[str | None, Field(title="Cwd")] = None
    display: Annotated[str, Field(title="Display")]
    effect_surface: Annotated[list[str], Field(title="Effect Surface")] = []
    exact: CommandBinding | None = None
    grant: CommandBinding | None = None
    hits: Annotated[list[FactHit], Field(title="Hits", validate_default=True)] = []
    idempotent: Annotated[bool | None, Field(title="Idempotent")]
    kind: Annotated[Literal["command", "mcp_tool", "git"], Field(title="Kind")]
    network: Annotated[bool, Field(title="Network")]
    offer_repo: Annotated[bool, Field(title="Offer Repo")] = True
    operation_id: Annotated[str, Field(title="Operation Id")]
    repo_any_arguments: Annotated[bool, Field(title="Repo Any Arguments")] = False
    reversible: Annotated[bool, Field(title="Reversible")]
    sandboxed: Annotated[bool | None, Field(title="Sandboxed")] = None
    spec_sha: Annotated[str, Field(title="Spec Sha")]
    task_id: Annotated[str | None, Field(title="Task Id")] = None
    tier: Annotated[Literal["t0", "t1", "t2"], Field(title="Tier")]
    """
    How much the user is consulted before a change happens.

    Ordered — `TIER_ORDER` is the order, and `max()` over it is what makes "never
    de-escalate" (invariant 17) structurally impossible to violate rather than a policy
    someone must remember to enforce.

    It lives here rather than in `chickadee/approval/` because a G0 contract carries it and
    `schema/` cannot import from a package built three phases later. That is also the right
    home on the merits: a tier is written down and read back — by the journal, the handoff,
    the visibility surface — and what crosses that boundary is a wire type.
    """


class AssistantTurnP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    calls: Annotated[list[Call], Field(title="Calls", validate_default=True)] = []
    findings: Annotated[dict[str, Any], Field(title="Findings")] = {}
    role: Annotated[Literal["assistant"], Field(title="Role")] = "assistant"
    text: Annotated[str, Field(title="Text")]


class Attachment(WireModel):
    """
    A file the person sent with a turn. The
    bytes are a session blob, which `handle` names; a PDF's text layer is a second blob, which
    `text_handle` names. Numbered per session, `A-1` on, never reused.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    bytes: Annotated[int, Field(ge=1, title="Bytes")]
    delivery: Annotated[Literal["native", "text", "outline", "none"], Field(title="Delivery")]
    est_tokens: Annotated[int, Field(ge=0, title="Est Tokens")]
    handle: Handle
    height: Annotated[int | None, Field(title="Height")] = None
    id: Annotated[str, Field(pattern="^A-[1-9][0-9]*$", title="Id")]
    kind: Annotated[Literal["image", "pdf", "text"], Field(title="Kind")]
    lines: Annotated[int | None, Field(title="Lines")] = None
    media_type: Annotated[str, Field(title="Media Type")]
    name: Annotated[str, Field(max_length=200, min_length=1, title="Name")]
    pages: Annotated[int | None, Field(title="Pages")] = None
    text_handle: Handle | None = None
    width: Annotated[int | None, Field(title="Width")] = None


class QuestionAskedRecord(WireModel):
    """
    Exactly one of `question` (an `ask`), `approval` (a `confirm_action`), `stop` (a
    `confirm_stop`), or, on an older log, `gate`, each whole: its text, options and
    default, the concrete action and its tier, or the markdown shown at the stop.
    """

    model_config = ConfigDict(
        extra="ignore",
    )
    approval: ActionApproval | None = None
    gate: RetiredGate | None = None
    question: Question | None = None
    stop: ApprovalStop | None = None


class UserTurnP(WireModel):
    model_config = ConfigDict(
        extra="ignore",
    )
    attachments: Annotated[
        list[Attachment], Field(title="Attachments", validate_default=True)
    ] = []
    role: Annotated[Literal["user"], Field(title="Role")] = "user"
    text: Annotated[str, Field(title="Text")]
    workspace: Annotated[Literal["worktree", "in_place"] | None, Field(title="Workspace")] = (
        None
    )


ConcernRecord = Annotated[
    Concern | Touched | Promoted | Dismissed,
    Field(discriminator="event"),
]

RejectedRecord = Annotated[
    RejectedAlt | Reconsideration,
    Field(discriminator="event"),
]

TurnRecord = Annotated[
    UserTurnP | AssistantTurnP | HarnessTurnP,
    Field(discriminator="role"),
]


SESSION_RECORDS: Final[dict[str, TypeAdapter[WireModel]]] = {
    "action_granted": TypeAdapter[WireModel](ActionGrantedRecord),
    "agent": TypeAdapter[WireModel](AgentRecord),
    "cap": TypeAdapter[WireModel](CapRecord),
    "compaction": TypeAdapter[WireModel](CompactionRecord),
    "concern": TypeAdapter[WireModel](ConcernRecord),
    "constraint": TypeAdapter[WireModel](ConstraintRecord),
    "episode": TypeAdapter[WireModel](EpisodeRecord),
    "fork": TypeAdapter[WireModel](ForkRecord),
    "frame": TypeAdapter[WireModel](FrameRecord),
    "model_switched": TypeAdapter[WireModel](ModelSwitchedRecord),
    "question_answered": TypeAdapter[WireModel](QuestionAnsweredRecord),
    "question_asked": TypeAdapter[WireModel](QuestionAskedRecord),
    "redo": TypeAdapter[WireModel](RedoRecord),
    "rejected": TypeAdapter[WireModel](RejectedRecord),
    "shell_effect": TypeAdapter[WireModel](ShellEffectRecord),
    "spend": TypeAdapter[WireModel](SpendRecord),
    "tell": TypeAdapter[WireModel](TellRecord),
    "title": TypeAdapter[WireModel](TitleRecord),
    "tool": TypeAdapter[WireModel](ToolRecord),
    "turn": TypeAdapter[WireModel](TurnRecord),
    "undo": TypeAdapter[WireModel](UndoRecord),
    "work": TypeAdapter[WireModel](WorkRecord),
    "workspace": TypeAdapter[WireModel](WorkspaceRecord),
}
"""Each session kind a client may read: its payload's type, as an adapter because
three kinds are unions."""


RUN_RECORDS: Final[dict[str, TypeAdapter[WireModel]]] = {
    "agent_message": TypeAdapter[WireModel](AgentMessageP),
    "agent_report": TypeAdapter[WireModel](AgentReportP),
    "agent_start": TypeAdapter[WireModel](AgentStartP),
    "box_compaction": TypeAdapter[WireModel](BoxCompactionP),
    "box_message": TypeAdapter[WireModel](BoxMessageP),
    "budget": TypeAdapter[WireModel](BudgetP),
    "callback_error": TypeAdapter[WireModel](CallbackErrorP),
    "citation_gate": TypeAdapter[WireModel](CitationGateP),
    "context_tokens": TypeAdapter[WireModel](ContextTokensP),
    "effect_completed": TypeAdapter[WireModel](EffectCompletedP),
    "effect_intended": TypeAdapter[WireModel](EffectIntendedP),
    "effect_started": TypeAdapter[WireModel](EffectStartedP),
    "exec": TypeAdapter[WireModel](ExecP),
    "external_call": TypeAdapter[WireModel](ExternalCallP),
    "hook": TypeAdapter[WireModel](HookP),
    "interrupt": TypeAdapter[WireModel](InterruptP),
    "mcp_call": TypeAdapter[WireModel](McpCallP),
    "mcp_connect": TypeAdapter[WireModel](McpConnectP),
    "model_call": TypeAdapter[WireModel](ModelCallP),
    "plan_outgrown": TypeAdapter[WireModel](PlanOutgrownP),
    "plan_written": TypeAdapter[WireModel](PlanWrittenP),
    "question_answered": TypeAdapter[WireModel](QuestionAnsweredP),
    "question_asked": TypeAdapter[WireModel](QuestionAskedP),
    "read_answered": TypeAdapter[WireModel](ReadAnsweredP),
    "read_tier": TypeAdapter[WireModel](ReadTierP),
    "redaction": TypeAdapter[WireModel](RedactionP),
    "route": TypeAdapter[WireModel](RouteP),
    "run_end": TypeAdapter[WireModel](RunEndP),
    "run_start": TypeAdapter[WireModel](RunStartP),
    "scout_escalated": TypeAdapter[WireModel](ScoutEscalatedP),
    "scout_escape": TypeAdapter[WireModel](ScoutEscapeP),
    "scout_report": TypeAdapter[WireModel](ScoutReportP),
    "scout_start": TypeAdapter[WireModel](ScoutStartP),
    "scout_tool": TypeAdapter[WireModel](ScoutToolP),
    "secret_denied": TypeAdapter[WireModel](SecretDeniedP),
    "secret_suspected": TypeAdapter[WireModel](SecretSuspectedP),
    "session_end": TypeAdapter[WireModel](SessionEndP),
    "session_start": TypeAdapter[WireModel](SessionStartP),
    "task_delegated": TypeAdapter[WireModel](TaskDelegatedP),
    "task_reported": TypeAdapter[WireModel](TaskReportedP),
    "task_reviewed": TypeAdapter[WireModel](TaskReviewedP),
    "tell_delivered": TypeAdapter[WireModel](TellDeliveredP),
    "tell_reply": TypeAdapter[WireModel](TellReplyP),
    "tool_call": TypeAdapter[WireModel](ToolCallP),
    "turn": TypeAdapter[WireModel](TurnP),
    "untrusted_read": TypeAdapter[WireModel](UntrustedReadP),
}
"""Each run event kind: its payload's type, from the journal golden."""
