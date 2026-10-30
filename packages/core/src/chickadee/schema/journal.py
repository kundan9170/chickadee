"""Contract 3 — JournalEvent v2.

The single source of truth (invariant 4). Every transition and effect phase is durable
here, and replay reconstructs all state from it, so a payload that omits a field is state
that cannot be recovered after a crash.

the table is normative and is the *only* list of event kinds — "a kind that is not in
this table does not exist". This module is that table as code; `scripts/audit_docs.py`
check 1 keeps the two from drifting.

**v2**: the packet loop's kinds, and the envelope's `packet_id` and `attempt`,
retired with it; the run lines name the approved work and the episode's outcome. A line of an
earlier version is refused: those journals are not resumed, and the
tag `pre-d196` reads them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from chickadee.schema.adapter import CallForm
from chickadee.schema.ids import RUN_ID_RE, TASK_KEY_RE, TS_RE
from chickadee.schema.isolation import WorkspaceMode

WIRE = ConfigDict(extra="forbid", frozen=True)

ENVELOPE_VERSION = 2
"""Contracts v2. Every line is written with it, and a line naming another is refused."""


class Scope(StrEnum):
    """Which envelope fields a kind may carry (the scope column)."""

    RUN = "run"
    OPERATION = "operation"
    ASK = "ask"
    SESSION = "session"
    FLEXIBLE = "flexible"
    """"task or run" rows: a task is optional, not free-form."""
    TASK = "task"
    """A task loop line: names its task."""


class EventKind(StrEnum):
    RUN_START = "run_start"
    EFFECT_INTENDED = "effect_intended"
    EFFECT_STARTED = "effect_started"
    EFFECT_COMPLETED = "effect_completed"
    BUDGET = "budget"
    EXEC = "exec"
    ROUTE = "route"
    READ_TIER = "read_tier"
    CITATION_GATE = "citation_gate"
    READ_ANSWERED = "read_answered"
    SCOUT_START = "scout_start"
    SCOUT_TOOL = "scout_tool"
    SCOUT_ESCAPE = "scout_escape"
    SCOUT_ESCALATED = "scout_escalated"
    SCOUT_REPORT = "scout_report"
    SESSION_START = "session_start"
    TURN = "turn"
    INTERRUPT = "interrupt"
    SESSION_END = "session_end"
    QUESTION_ASKED = "question_asked"
    QUESTION_ANSWERED = "question_answered"
    CONTEXT_TOKENS = "context_tokens"
    UNTRUSTED_READ = "untrusted_read"
    MCP_CONNECT = "mcp_connect"
    MCP_CALL = "mcp_call"
    EXTERNAL_CALL = "external_call"
    SECRET_DENIED = "secret_denied"
    SECRET_SUSPECTED = "secret_suspected"
    REDACTION = "redaction"
    HOOK = "hook"
    CALLBACK_ERROR = "callback_error"
    RUN_END = "run_end"
    TOOL_CALL = "tool_call"
    MODEL_CALL = "model_call"
    TASK_DELEGATED = "task_delegated"
    TASK_REPORTED = "task_reported"
    BOX_MESSAGE = "box_message"
    BOX_COMPACTION = "box_compaction"
    TASK_REVIEWED = "task_reviewed"
    PLAN_WRITTEN = "plan_written"
    PLAN_OUTGROWN = "plan_outgrown"
    AGENT_START = "agent_start"
    AGENT_MESSAGE = "agent_message"
    AGENT_REPORT = "agent_report"
    TELL_DELIVERED = "tell_delivered"
    TELL_REPLY = "tell_reply"


class P(BaseModel):
    """Base for every payload: strict, frozen, no extra fields."""

    model_config = WIRE


CapUsd = Annotated[float, Field(gt=0, allow_inf_nan=False)]
"""A cap as config loads one: finite and above zero. `None` beside it means lifted."""

RungName = Annotated[str, Field(min_length=1)]
"""The rung that made a priced call. A name, not `roles.RUNG_NAMES`: a leaf contract must
not move when the cast does."""


class RunCapsP(P):
    """Every cap a run started under, and every one the person lifted.

    "Lifted" is said, never left as a `null`: the journal writes with `exclude_none`, so a
    `null` total would vanish on the way to disk and read back as a writer that forgot it.
    A rung with no cap configured is in neither `rungs` nor `lifted_rungs`."""

    total: CapUsd | None = None
    """The family's total cap in force; absent exactly when `total_lifted`."""
    total_lifted: bool = False
    rungs: dict[RungName, CapUsd] = Field(default_factory=dict)
    """Each rung's cap in force."""
    lifted_rungs: tuple[RungName, ...] = ()
    """Each rung whose cap the person lifted for the family."""

    @model_validator(mode="after")
    def _lifted_is_said_once(self) -> RunCapsP:
        if (self.total is None) != self.total_lifted:
            raise ValueError("the total is either a cap in force or lifted, exactly one")
        if len(set(self.lifted_rungs)) != len(self.lifted_rungs):
            raise ValueError("a lifted rung is named once")
        both = set(self.rungs) & set(self.lifted_rungs)
        if both:
            raise ValueError(f"a rung is capped or lifted, not both: {sorted(both)}")
        return self


# --- run lifecycle ---------------------------------------------------------


Outcome = Literal["COMPLETED", "PARTIAL", "BLOCKED", "ASK", "BUDGET_CAP", "ABORTED", "OUTGROWN"]
"""How an episode ended (`box/outcome.py` names it for the box). Never `SOLVED`: that claimed a
verification no gate gives a task run (invariant 7). `ABORTED` is the person's stop (owner,
2026-10-05): a run they stopped is not `PARTIAL`, which would hide who stopped it. `OUTGROWN`
is Athena's: a one-task plan whose finishing needs a decision, a problem or a scope its brief
does not hold; Zeus decides what follows."""


class RunStartP(P):
    goal: str
    """One line naming the work: the one-task plan's title, or the design's first line."""
    work_sha: str
    """The approved work's sha: the design's at stop 1, or the one-task brief's at
    stop 2, which is what runs (invariant 15)."""
    template_shas: dict[str, str] = Field(default_factory=dict)
    """The handoff templates in force, by kind (`handoffs.resolve`): which ones the work was
    written and planned to."""
    config_sha: str
    repo_head: str
    engine_version: str
    schema_version: int
    isolation: WorkspaceMode
    hard_cost_cap_usd: float
    """Required. A config without a finite cap is rejected at load, so a run that
    started has one by construction. It is the config's total; `caps`, when present, is
    what was enforced."""
    caps: RunCapsP | None = None
    """The caps in force when the run started, a lifted total included. Absent on
    every journal written before it existed."""


class RunEndP(P):
    outcome: Outcome
    why: str | None = None
    """One line: why the episode ended as it did; None for `COMPLETED`."""
    tasks: int = Field(ge=0)
    """Tasks delegated."""
    committed: int = Field(ge=0)
    """Tasks accepted and committed."""
    tokens: int
    cost_usd: float
    wall_ms: int
    handoff_path: str | None = None
    """`blobs/<sha>`: the report the box returned, when one was stored."""

    @model_validator(mode="after")
    def _why_unless_completed(self) -> RunEndP:
        if (self.outcome == "COMPLETED") == (self.why is not None):
            raise ValueError("why is set exactly when the outcome is not COMPLETED")
        return self


# --- calls ---------------------------------------------------------------------------


class ToolCallP(P):
    """One tool call by any role, through the one dispatcher.
    `scout_tool` stays, and `journal.projection.tool_call` reads both as this. A call was
    refused exactly when `denied_reason` is set."""

    role: Annotated[str, Field(min_length=1)]
    """The role that made the call (`worker`, `locate`): a name, not `roles`' list, so the
    contract does not move when the cast does (as `RungName`)."""
    tool: Annotated[str, Field(min_length=1)]
    """Any registered tool's name."""
    args_sha: str
    outcome: Literal["served", "repack"]
    result_tokens: int = Field(ge=0)
    result_sha: str | None = None
    denied_reason: str | None = None
    dedup_hit: bool = False
    form: CallForm | None = None
    """How the call was written; None for a role whose calls are only native,
    or a line written before it."""
    args_preview: str | None = None
    """The call's arguments, redacted and bounded (`journal.previews`); None on a line
    written before it."""
    result_preview: str | None = None
    """What the model was given back, or the refusal, redacted and bounded."""


class ModelCallP(P):
    """One model call of a box role (Athena's, the worker's), so a reader sees what each
    call was and what it said. Written once the call is answered and charged.."""

    role: Annotated[str, Field(min_length=1)]
    request: Annotated[str, Field(min_length=1)]
    """What the call served: the role's turn (`box/sink.BoxSink`)."""
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost_usd: float = Field(ge=0)
    stop_reason: str
    """How the call ended, as the provider said."""
    tools: tuple[str, ...] = ()
    """The tools the call asked for, by name, in order."""
    text_preview: str | None = None
    """The call's text, redacted and bounded (`journal.previews`)."""
    refusal: str | None = None
    """Why the answer was refused and asked again, or why the request ended there, as
    `<kind>: <hint>`, redacted and bounded the same way. `None` for an answer taken, and for
    a turn that asked for tools unless it was cut at `max_tokens` (`truncated`: its calls are
    never dispatched)."""
    rung: RungName | None = None
    """The rung that made the call, which the run's spend is counted by
    (`journal.projection.RunSpend.by_rung`).
    became the run's one account of its model spend."""
    sent_sha: str | None = None
    """The run blob indexing the request exactly as the transport got it: its messages
    (each a blob in `message_bytes`' form), its tool schemas, its settings
    (`boxctx.sent.record_sent`). None on a line written before it."""
    model: str | None = None
    """The catalog name of the model that served the call, as Zeus's
    `spend` line records it: the rung's model when it was called, a `/rung` switch included.
    None on a line written before it. Additive."""


# --- effects ---------------------------------------------------------


class EffectIntendedP(P):
    operation_id: str
    effect_kind: str
    approved_sha: str
    preconditions: tuple[str, ...]
    idempotent: bool
    declared_surface: tuple[str, ...]
    preimages_sha: str | None = None


class EffectStartedP(P):
    operation_id: str


class EffectCompletedP(P):
    operation_id: str
    outcome: Literal["succeeded", "failed", "ambiguous"]
    observed_surface: tuple[str, ...]
    post_state_sha: str | None = None
    result_sha: str | None = None


# --- budget, execution ----------------------------------------------------------------


class BudgetP(P):
    scope: Literal["run", "packet"]
    kind: Literal["warning", "hard_cap"]
    spent_usd: float
    cap_usd: float
    terminal: bool
    rung: RungName | None = None
    """The rung that made the call; on a rung's hard stop, the rung."""


class ExecP(P):
    operation_id: str
    policy: str
    label: str
    argv_sha: str
    cwd: str
    exit_code: int
    wall_ms: int
    timed_out: bool
    sandbox: str
    network: bool
    stdout_sha: str
    stderr_sha: str
    violations: tuple[str, ...]
    escalation_id: str | None = None


# --- routing and reads -----------------------------------------------------


class RouteP(P):
    mode: str
    read_type: str
    confidence: float
    signals: tuple[str, ...]


class ReadTierP(P):
    tier: str
    reason: str


class CitationGateP(P):
    passed: bool = Field(alias="pass")
    checked: int
    coverage: float

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ReadAnsweredP(P):
    tier: str
    tokens: int
    cost_usd: float
    low_confidence: bool
    rung: RungName | None = None
    """The rung that made the call."""


# --- scout -----------------------------------------------------------------


class ScoutStartP(P):
    rung: str
    goal_sha: str
    snapshot_sha: str


class ScoutToolP(P):
    tool: str
    args_sha: str
    result_tokens: int
    dedup_hit: bool


class ScoutEscapeP(P):
    command: str
    granted: bool
    reason: str
    exec_seq: int


class ScoutEscalatedP(P):
    reason: str
    from_rung: str
    to_rung: str
    dropped_pointers: int
    truncated: bool


class ScoutReportP(P):
    targets: int
    facts: int
    unresolved: int
    truncated: bool
    tokens: int
    cost_usd: float
    rung: RungName | None = None
    """The rung that made the call."""


# --- session and interaction -----------------------------------------------


class SessionStartP(P):
    session_id: str
    repo: str
    profile: str
    rungs_sha: str


class TurnP(P):
    input_sha: str
    mode: str
    artifact_id: str


class InterruptP(P):
    level: str
    action: str


class SessionEndP(P):
    turns: int
    cost_usd: float
    wall_ms: int


class QuestionAskedP(P):
    id: str
    kind: str
    options: tuple[str, ...]
    evidence: tuple[str, ...]
    blocking: bool


class QuestionAnsweredP(P):
    id: str
    choice: str
    source: str
    elapsed_ms: int


# --- context, trust, mcp, secrets, hooks -----------------------------------


class ContextTokensP(P):
    layer1: int
    compacted: int
    raw_window: int


class UntrustedReadP(P):
    path: str
    span: str
    rung: str
    bytes: int


class McpConnectP(P):
    server: str
    transport: str
    tools: tuple[str, ...]
    ok: bool
    hints: dict[str, object] = {}


class McpCallP(P):
    operation_id: str
    server: str
    tool: str
    args_sha: str
    policy: str
    ok: bool
    result_tokens: int
    wall_ms: int
    truncated: bool
    error: str | None = None


class ExternalCallP(P):
    """One call that left the machine on a model's behalf, through the external-call policy
    (`chickadee.external`): a web search or fetch now, an MCP tool's call
    later, which writes this in place of `McpCallP`. Additive under the owner's grant.
    """

    operation_id: str
    """The tool call it served."""
    provider: str = Field(min_length=1)
    """Who answered it: `web.fetch.http`, `web.search.github`; later `mcp:<server>`."""
    tool: str = Field(min_length=1)
    args_sha: str
    """Of its arguments, redacted, canonical: what was asked, never a credential."""
    policy: Literal["discovery", "action"]
    """`executor.types.PolicyClass`: a read, or a mutation under the effect lifecycle."""
    ok: bool
    result_sha: str | None = None
    """What came back, as a blob: what the model was shown, read back by the citation check."""
    result_bytes: int = Field(default=0, ge=0)
    wall_ms: int = Field(ge=0)
    truncated: bool = False
    """Whether the result was cut at its bound."""
    error: str | None = None
    """The failure's name, when it did not succeed (`URL_REFUSED`, `FETCH_FAILED`,...)."""


class SecretDeniedP(P):
    path: str
    rung: str
    surface: str


class SecretSuspectedP(P):
    """Advisory only: the content was read; nothing was withheld."""

    path: str
    line: int
    detector: str


class RedactionP(P):
    target_blob: str
    count: int
    kinds: tuple[str, ...]


class HookP(P):
    """One hook's verdict, error or bound reached, or, as `event = "bound"`,
    the set an episode or a session bound (decision 14). A silent `Pass` writes none. Fields
    after `timed_out` are additive."""

    event: str
    """The point, or `bound`."""
    name: str
    decision: str
    """The verdict (`hooks/verdicts.kind_of`), `error`, or `bound`."""
    wall_ms: int
    message: str | None = None
    """A preview of the verdict's text, or the error."""
    timed_out: bool = False
    role: str | None = None
    """The role the event was about: Athena's and the worker's bounds are counted by it."""
    call_id: str | None = None
    """The tool call a tool point's verdict was about."""
    text_sha: str | None = None
    """The verdict's whole text, or the bound set, as a run blob."""
    input_sha: str | None = None
    """The digest of what the hook was told."""
    bounded: bool = False
    """The verdict was not taken: its bound was reached (decision 8)."""
    error: str | None = None
    """Why the hook failed: it raised, timed out, or answered what its point does not take."""


class CallbackErrorP(P):
    name: str
    event: str
    error_sha: str


# --- the task loop (the additive half) -------------------------

RoundEnd = Literal[
    "replied",
    "limit",
    "looped",
    "context_full",
    "model_error",
    "harness_error",
    "halted",
    "held",
]
"""Why a worker's round ended, read from the loop and the dispatcher, never from
the report. Not `StopReason`: `schema/adapter.py` names that for why a model's turn ended.
`held`: a message from Zeus held it at its next tool boundary; the worker wrote no report, and
the round is not ended to any hook."""

ReviewAnswer = Literal[
    "accept", "feedback", "continue", "take_over", "abandon", "re_cut", "drop"
]
"""Athena's answer to a task's report. `drop` is
`abandon` renamed: `abandon` stays, so a run
written in older versions still reads, and is read as `drop`. `continue` opens the next round of
a
held one with no message."""

TellOutcome = Literal["applied", "planned", "needs_design", "declined"]
"""Athena's reply to a tell: done within the task, planned as a task, a change
the approved design does not hold, or not done. The last two wake Zeus."""

Fresh = bool | tuple[str, ...] | None
"""What a review puts back to the task's start before it acts: `True` every
file, paths only those, None (or `False`) nothing."""

BoxRoleName = Literal["athena", "worker"]
"""The box roles whose conversations the journal holds (`box_message`, `box_compaction`)."""

TaskId = Annotated[str, Field(pattern=TASK_KEY_RE.pattern)]
"""A task's attempt as the journal keys it: `T3`, `T3.1`, then `T3@2`. Every
id written in older versions is one."""


class FileFactP(P):
    """One file a task changed, measured between its start tree and its end tree."""

    path: str
    added: int = Field(ge=0)
    removed: int = Field(ge=0)
    kind: Literal["created", "edited", "deleted"]


class CommandFactP(P):
    """One command a task ran, from its step and its `exec` line."""

    operation_id: str
    """The step's, and its `exec` line's."""
    command: str
    """Previewed from the call, as its `tool_call` line holds it."""
    exit_code: int | None = None
    """None: it never ran (refused, or asked and denied)."""
    timed_out: bool
    tail: str
    """The last lines of its output; the whole is the blob."""
    output_sha: str | None = None
    test_outcome: str | None = None
    """`executor/tests.TestOutcome`'s value when the command is its profile's own test runner;
    None for any other command, so a run that collected nothing never reads as a pass."""
    failing: tuple[str, ...] = ()
    """The failing tests that run names, from the same reading."""


SyntaxDepth = Literal["compiled", "command", "parsed"]
"""How far a syntax check went: the language's own compiler in process
(`compiled`, Python's `compile()`), the profile's command (`command`, `node --check`), or a
tree-sitter grammar's parse (`parsed`), the weakest, which says nothing of names or types."""


class SyntaxFactP(P):
    """One changed file's syntax, judged by its own language's profile over the file as the
    task left it: the same check the edit tool reported."""

    path: str
    ok: bool
    detail: str | None = None
    """What the check said when it failed: the first error, where."""
    depth: SyntaxDepth | None = None
    """How far the check went, so a pass is never read as more than it was.
    None in a fact recorded before it was carried."""


class TaskFactsP(P):
    """What a task did, measured from the trees and the journal, never from the
    report."""

    start_tree: str
    end_tree: str
    files: tuple[FileFactP, ...] = ()
    tests_added: tuple[str, ...] = ()
    tests_edited: tuple[str, ...] = ()
    """An existing test file the task changed."""
    symbols_added: tuple[str, ...] = ()
    symbols_removed: tuple[str, ...] = ()
    symbols_known: bool = True
    """False after a resume, when the symbols before are not in memory (invariant 20)."""
    commands: tuple[CommandFactP, ...] = ()
    skipped: tuple[str, ...] = ()
    """Untracked files over the snapshot cap, named and not stored."""
    syntax: tuple[SyntaxFactP, ...] = ()
    """Each changed, still-existing file a profile judges; a file none judges is
    absent, never `ok`."""


class TaskDelegatedP(P):
    """A task given to the worker: by Athena's `delegate` call, or by Zeus as a
    one-task plan. Its id is the harness's, never the model's."""

    task_id: TaskId
    call_id: str | None = None
    """The `delegate` call that made it, so a resume continues it rather than delegating it
    twice; None for Zeus's one-task plan."""
    title: str
    brief_sha: str
    by: Literal["zeus", "athena"]
    start_tree: str
    start_commit: str | None = None
    depth: int = Field(default=0, ge=0)
    """The re-cut lineage's depth."""
    template_sha: str | None = None
    plan_task: str | None = None
    """The approved plan's task this delegation runs, when delegated by its id: a plan task's
    own id, which the harness's `task_id` need not equal once a re-cut has
    added tasks. None for a brief Athena wrote, and for a one-task plan."""


class TaskReportedP(P):
    """One worker round's end: why it stopped, its words, and the facts measured."""

    task_id: TaskId
    round: int = Field(ge=1)
    stopped: RoundEnd
    report_sha: str | None = None
    """The worker's words as a blob; None when it did not reply."""
    facts: TaskFactsP


class BoxMessageP(P):
    """One message of a box role's conversation, journaled before it joins it, so resume
    rebuilds the conversation exactly."""

    role: BoxRoleName
    side: Literal["system", "user", "assistant", "tool"]
    sha: str
    """The message's canonical JSON, as a run blob."""


class BoxCompactionP(P):
    """A box compaction (`box_compaction`, moved to the run journal): what became
    receipts, folded, demoted or collapsed, and the tokens on both sides. The session-log kind
    retired with the packet loop; the ladder's other steps stay empty until the
    parked compaction returns."""

    role: BoxRoleName
    steps: tuple[Literal["evict", "fold", "demote_history", "collapse_unrelated", "reset"], ...]
    evict: tuple[str, ...] = ()
    fold: tuple[str, ...] = ()
    demote: tuple[str, ...] = ()
    collapse: tuple[str, ...] = ()
    tokens_before: int = Field(default=0, ge=0)
    tokens_after: int = Field(default=0, ge=0)


class TaskReviewedP(P):
    """Athena's answer to a task's report, by her
    `review` call. On `feedback` the text is the worker's next message; on `re_cut` the tasks
    she delegates next are its replacements."""

    task_id: TaskId
    call_id: str
    """The `review` call that made it: a resume continues from this line for that call rather
    than reviewing twice."""
    answer: ReviewAnswer
    text_sha: str | None = None
    """Her text as a run blob, when she gave one."""
    fresh: Fresh = None
    """What is put back to the task's start:
    `True` every file, paths only those; None keeps the changes. Set only on `feedback`,
    `take_over` and `re_cut`; a `drop` always puts every file back."""
    by: str | None = None
    """`hook:<name>` for a hook's `Retry`, which is her `feedback` written by the harness;
    None, as every line before it, for her own review ."""


class PlanTaskP(P):
    """One task an approved plan lists: its `Tasks` line's id and title, as
    `work.plan_tasks` reads them, the reading `delegate(task)` resolves an id through."""

    id: str
    title: str


class PlanWrittenP(P):
    """A plan written: Athena's, by her `plan` call,
    or Zeus's one-task plan, its brief alone. The first is asked at stop 2; every
    later one is a revision, shown and never asked.

    The spec's row named a `text_sha` beside `sha`: blobs are content-addressed by the sha256
    of their bytes, so the text's blob is `sha` itself, and one field holds both."""

    call_id: str | None = None
    """The `plan` call that wrote it; None for Zeus's one-task plan."""
    sha: str
    """`sha256_hex` of the plan's text, which stop 2 binds, and the run blob holding it."""
    by: Literal["zeus", "athena"]
    version: int = Field(ge=1)
    """1 for the first plan, then one more per revision."""
    asked: bool
    """Whether it was put to the person at stop 2: the first plan, attended or not."""
    outcome: Literal["approved", "cancelled"] = "approved"
    """`cancelled`: the person cancelled at stop 2, so the episode ends `ABORTED` and nothing
    runs; journaled so a resume keeps their answer and never asks again. Its `sha` is the plan
    shown, and no version follows it."""
    note_sha: str | None = None
    """The person's reason for a cancel, verbatim, as a run blob."""
    bound: str | None = None
    """Why the harness ended stop 2 itself, when no answer of the person's did: the rounds
    bound (`StopOutcome.bound`). Set, the cancel is not the person's and is never reported as
    theirs."""
    tasks: tuple[PlanTaskP, ...] = ()
    """The tasks the plan's text lists, in order, so a reader counts them
    without parsing the plan. Empty when its text lists none: a plan given as text, and Zeus's
    one-task plan, one task (`by` is `zeus`); and on a line written before it.
    Additive."""
    inherited_from: str | None = None
    """The run whose approved plan this line carries into a run that continues the work:
    written by the harness before Athena's first call, never asked again.
    None for a plan written in this run. Additive."""


class PlanOutgrownP(P):
    """A one-task plan ended `OUTGROWN` by Athena: its finishing
    needs a decision, a problem or a scope its brief does not hold. Journaled once, by her
    `outgrown` call, so a resume ends the episode as she did; her report is her reply after it.
    """

    call_id: str
    """The `outgrown` call that wrote it."""


TELL_ID = r"^tell-[1-9][0-9]*$"
TellId = Annotated[str, Field(pattern=TELL_ID)]
"""A tell's id: `tell-N`, numbered in its session from 1, the session's `tell`
line's (`context/lines.TellLine`)."""


class TellDeliveredP(P):
    """A message from Zeus (a session's `tell` line) delivered to Athena in this run: at the
    boundary of the worker's round it held, or before her next model call with no
    round open."""

    tell: TellId
    task_id: str | None = None
    """The task whose round it held; None when no worker round was open."""
    round: int | None = Field(default=None, ge=1)
    """That round, with `task_id`."""


class TellReplyP(P):
    """Athena's reply to a delivered tell, by her `reply` call: owed for each one."""

    tell: TellId
    outcome: TellOutcome
    text_sha: str
    """Her words, as a run blob."""
    call_id: str
    """The `reply` call that wrote it: a resume answers it again from this line."""


# --- the table -------------------------------------------------------------

# --- agents (additive under the owner's grant) ---------------------------------

AGENT_ID = r"^A[1-9][0-9]*$"
"""An agent's id: `A1`, `A2`,... in the order a session spawned them. `follow_up`'s schema
names it too (`tools/spec.py`)."""
AgentId = Annotated[str, Field(pattern=AGENT_ID)]
AgentStatus = Literal["complete", "partial", "failed", "cancelled"]
"""`complete`: it answered within its limits. `partial`: a limit ended its tool use and it
reported what it had. `failed`: its report never parsed, or its model could not be reached.
`cancelled`: the person's stop."""
AgentSource = Literal["built-in", "user", "project", "external"]
"""Where an agent's skill came from, by whose tree it is in (`Skill.origin`), whatever format
it is written in: `project` is what a repository's trust admits (decision 6), `user` the
person's own, `built-in` a profile run with no skill. `external` is read, never written now:
this task's first build wrote it for another tool's file, and a line it wrote still
validates (the schema grant is additive only)."""


class AgentStartP(P):
    """A round of an agent begins: before anything of it runs, so a round
    with this line and no `agent_report` is one a crash cut. Round 0 is the spawn."""

    agent: AgentId
    round: int = Field(ge=0)
    skill: str | None = None
    """The skill it runs, or None for a profile run with none."""
    source: AgentSource
    profile: str = Field(min_length=1)
    rung: RungName
    model: str = Field(min_length=1)
    task_sha: str
    handles: tuple[str, ...] = ()
    call_id: str
    """Zeus's call that spawned it."""


class AgentMessageP(P):
    """One message of an agent's conversation, journaled before it joins it, so a follow-up
    after a resume rebuilds it exactly, as `box_message` is."""

    agent: AgentId
    round: int = Field(ge=0)
    side: Literal["system", "user", "assistant", "tool"]
    sha: str
    """The message's canonical JSON, as a session blob."""


class AgentReportP(P):
    """A round's end and its report: the report itself is a blob, and its
    checked counts are here."""

    agent: AgentId
    round: int = Field(ge=0)
    status: AgentStatus
    findings: int = Field(ge=0)
    flagged: int = Field(ge=0)
    """Citations the gate did not pass (`context/citations.py`)."""
    tool_calls: int = Field(ge=0)
    tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0, allow_inf_nan=False)
    report_sha: str


KIND_TABLE: dict[EventKind, tuple[type[P], Scope]] = {
    EventKind.RUN_START: (RunStartP, Scope.RUN),
    EventKind.EFFECT_INTENDED: (EffectIntendedP, Scope.OPERATION),
    EventKind.EFFECT_STARTED: (EffectStartedP, Scope.OPERATION),
    EventKind.EFFECT_COMPLETED: (EffectCompletedP, Scope.OPERATION),
    EventKind.BUDGET: (BudgetP, Scope.FLEXIBLE),
    EventKind.EXEC: (ExecP, Scope.FLEXIBLE),
    EventKind.ROUTE: (RouteP, Scope.ASK),
    EventKind.READ_TIER: (ReadTierP, Scope.ASK),
    EventKind.CITATION_GATE: (CitationGateP, Scope.ASK),
    EventKind.READ_ANSWERED: (ReadAnsweredP, Scope.ASK),
    EventKind.SCOUT_START: (ScoutStartP, Scope.RUN),
    EventKind.SCOUT_TOOL: (ScoutToolP, Scope.RUN),
    EventKind.SCOUT_ESCAPE: (ScoutEscapeP, Scope.RUN),
    EventKind.SCOUT_ESCALATED: (ScoutEscalatedP, Scope.RUN),
    EventKind.SCOUT_REPORT: (ScoutReportP, Scope.RUN),
    EventKind.SESSION_START: (SessionStartP, Scope.SESSION),
    EventKind.TURN: (TurnP, Scope.SESSION),
    EventKind.INTERRUPT: (InterruptP, Scope.SESSION),
    EventKind.SESSION_END: (SessionEndP, Scope.SESSION),
    EventKind.QUESTION_ASKED: (QuestionAskedP, Scope.FLEXIBLE),
    EventKind.QUESTION_ANSWERED: (QuestionAnsweredP, Scope.FLEXIBLE),
    EventKind.CONTEXT_TOKENS: (ContextTokensP, Scope.RUN),
    EventKind.UNTRUSTED_READ: (UntrustedReadP, Scope.RUN),
    EventKind.MCP_CONNECT: (McpConnectP, Scope.RUN),
    EventKind.MCP_CALL: (McpCallP, Scope.FLEXIBLE),
    EventKind.EXTERNAL_CALL: (ExternalCallP, Scope.FLEXIBLE),
    EventKind.SECRET_DENIED: (SecretDeniedP, Scope.RUN),
    EventKind.SECRET_SUSPECTED: (SecretSuspectedP, Scope.RUN),
    EventKind.REDACTION: (RedactionP, Scope.RUN),
    EventKind.HOOK: (HookP, Scope.FLEXIBLE),
    EventKind.CALLBACK_ERROR: (CallbackErrorP, Scope.FLEXIBLE),
    EventKind.RUN_END: (RunEndP, Scope.RUN),
    EventKind.TOOL_CALL: (ToolCallP, Scope.FLEXIBLE),
    EventKind.MODEL_CALL: (ModelCallP, Scope.FLEXIBLE),
    EventKind.TASK_DELEGATED: (TaskDelegatedP, Scope.TASK),
    EventKind.TASK_REPORTED: (TaskReportedP, Scope.TASK),
    EventKind.BOX_MESSAGE: (BoxMessageP, Scope.FLEXIBLE),
    EventKind.BOX_COMPACTION: (BoxCompactionP, Scope.FLEXIBLE),
    EventKind.TASK_REVIEWED: (TaskReviewedP, Scope.TASK),
    EventKind.PLAN_WRITTEN: (PlanWrittenP, Scope.RUN),
    EventKind.PLAN_OUTGROWN: (PlanOutgrownP, Scope.RUN),
    EventKind.AGENT_START: (AgentStartP, Scope.SESSION),
    EventKind.AGENT_MESSAGE: (AgentMessageP, Scope.SESSION),
    EventKind.AGENT_REPORT: (AgentReportP, Scope.SESSION),
    EventKind.TELL_DELIVERED: (TellDeliveredP, Scope.RUN),
    EventKind.TELL_REPLY: (TellReplyP, Scope.RUN),
}


class JournalEvent(BaseModel):
    """The envelope. Identical for every kind; `payload` is kind-discriminated."""

    model_config = WIRE

    v: int = ENVELOPE_VERSION
    run_id: Annotated[str, Field(pattern=RUN_ID_RE.pattern)]
    seq: int = Field(ge=1)
    """Strictly monotonic per run, no gaps. A gap on read is a corrupt journal (part 09)."""
    ts: Annotated[str, Field(pattern=TS_RE.pattern)]
    kind: EventKind
    payload: SerializeAsAny[P]
    """`SerializeAsAny` so the concrete payload is written in full. Annotated as the
    base `P`, pydantic would serialize only the base's fields — which is none of
    them — and every event would round-trip as an empty payload."""
    task_id: TaskId | None = None
    """The task the line is about."""
    round: int | None = Field(default=None, ge=1)
    """The worker's round within `task_id`; only with a task."""

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, data: Any) -> Any:
        """Build the payload as its kind's model when handed a plain dict.

        Deserialization has to go through this: a bare dict would otherwise validate
        against the `P` base, which forbids extras but requires nothing, so every payload
        would round-trip as empty.
        """
        if not isinstance(data, dict):
            return data
        raw = cast("dict[str, Any]", data)
        if raw.get("v", ENVELOPE_VERSION) != ENVELOPE_VERSION:
            raise ValueError(
                f"a journal line of version {raw.get('v')!r}, before contracts v2: journals "
                "of the packet loop are not read or resumed (D196); the tag pre-d196 reads them"
            )
        kind = raw.get("kind")
        payload = raw.get("payload")
        if kind is None or not isinstance(payload, dict):
            return raw
        try:
            model, _ = KIND_TABLE[EventKind(kind)]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"unknown event kind {kind!r}; §4.2 is the only list") from exc
        return {**raw, "payload": model.model_validate(payload)}

    @model_validator(mode="after")
    def _payload_and_scope_match_kind(self) -> JournalEvent:
        model, scope = KIND_TABLE[self.kind]
        if type(self.payload) is not model:
            raise ValueError(
                f"kind={self.kind} requires payload {model.__name__}, "
                f"got {type(self.payload).__name__}"
            )

        if self.v != ENVELOPE_VERSION:
            raise ValueError(
                f"envelope version {self.v}; contracts v2 writes {ENVELOPE_VERSION}"
            )
        has_task = self.task_id is not None
        if self.round is not None and not has_task:
            raise ValueError("a round belongs to a task: `round` requires `task_id`")
        if scope is Scope.TASK and not has_task:
            raise ValueError(f"kind={self.kind} is task-scoped: it requires task_id")
        if has_task and scope not in (Scope.TASK, Scope.FLEXIBLE, Scope.OPERATION):
            raise ValueError(f"kind={self.kind} is {scope}-scoped and must not carry task_id")
        return self
