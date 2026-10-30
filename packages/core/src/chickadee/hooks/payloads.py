"""What a hook is told (D227 decision 11's input): one frozen, typed model per point, and the
event every fire site builds around it. No point is told a `Mapping[str, object]`: a record is
typed end to end.

A built-in reads these as they are; a script reads them as JSON, redacted (`protocol.py`), with
JSON Schemas exported from these same models, so a script's author has a contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from chickadee.errors import HarnessBug
from chickadee.hooks.points import HookPoint
from chickadee.schema.journal import Fresh, ReviewAnswer, RoundEnd, TaskFactsP

__all__ = [
    "PAYLOADS",
    "AfterBatch",
    "AfterStep",
    "AfterTool",
    "BeforeCompact",
    "BeforeRequest",
    "BeforeStop",
    "BeforeTool",
    "DiagnosticsSeen",
    "EpisodeEnded",
    "HookEvent",
    "HookPayload",
    "HookRole",
    "JobExited",
    "OpenTask",
    "PermissionAsked",
    "ProgressCall",
    "ProgressTag",
    "PromptSubmitted",
    "Regression",
    "RoundEnded",
    "SessionEnd",
    "SessionStart",
    "TaskCommitted",
    "TaskReviewed",
    "TestRunSeen",
    "TestsSeen",
    "WorkerProgress",
]

HookRole = Literal["zeus", "athena", "worker", "locate"]


class HookPayload(BaseModel):
    """A point's facts. Frozen: a hook reads them and cannot change what the harness does next
    except through its verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def paths(self) -> tuple[str, ...]:
        """The repository paths this event is about, for a hook's `paths` matcher."""
        return ()


class SessionStart(HookPayload):
    source: Literal["open", "resume", "fork"]

    @classmethod
    def of(cls, *, turns: bool, forked: bool) -> SessionStart:
        """How a session opened: with turns of its own, resumed; a fork's first open,
        forked; else new."""
        return cls(source="resume" if turns else "fork" if forked else "open")


class PromptSubmitted(HookPayload):
    prompt: str
    wake: bool = False
    """A wake turn, not words the person typed."""


class BeforeRequest(HookPayload):
    messages: int
    tokens_est: int
    """The request's size by P-2's estimate, never a count."""


class BeforeTool(HookPayload):
    tool: str
    """The registry name, whatever the model wrote."""
    tool_sent: str
    """The name as the model wrote it."""
    input: dict[str, JsonValue]
    resolved: tuple[str, ...] = ()
    """The repository paths the call names, as the role's file view resolves them."""
    programs: tuple[str, ...] = ()
    """A shell call's programs, each simple command's first word by its base name, as the
    command table's grammar reads the line (`shell_line.read_line`)."""

    def paths(self) -> tuple[str, ...]:
        return self.resolved


class AfterTool(BeforeTool):
    output: str
    """The answer as the model will read it, before any hook's note."""
    failed: bool


class AfterStep(HookPayload):
    tool: str
    changed: tuple[str, ...]
    """The paths the step changed, whatever wrote them: an edit or a command (`sed -i`)."""

    def paths(self) -> tuple[str, ...]:
        return self.changed


ProgressTag = Literal["changed", "learned", "nothing"]
"""What one worker call did, from recorded facts: it left the files in a state
the task never had, it showed something new, or neither."""


class TestRunSeen(BaseModel):
    """A test run as the progress record read it: how it was read, what it
    counted, and what it changed in the record of the task's tests."""

    __test__ = False
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: str
    """`executor/tests.TestOutcome`'s value."""
    how: Literal["summary", "exit_code", "partial", "stopped"]
    passed: int = 0
    failed: int = 0
    errors: int = 0
    unit: Literal["test", "package"] = "test"
    failing: tuple[str, ...] = ()
    """The failing tests it named."""
    fixed: tuple[str, ...] = ()
    """Failed before, passed in this run."""
    broken: tuple[str, ...] = ()
    """Passed before, failed in this run."""


class Regression(BaseModel):
    """A test broken and still failing after more changes: the note that
    rides on the run that confirmed it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test: str
    passed_at: str
    """Where it last passed: `call 9 of round 1`."""
    changed: tuple[str, ...] = ()
    """The files the task changed since."""


class TestsSeen(BaseModel):
    """The task's tests as its runs left them, for the reader of a report."""

    __test__ = False
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixed: tuple[str, ...] = ()
    """Seen failing in the task, last seen passing."""
    broken: tuple[str, ...] = ()
    """Seen passing in the task, last seen failing."""
    failing: tuple[str, ...] = ()
    """Last seen failing, and the task's: its brief names it, the task changed its file or a
    file its name names, or it passed earlier in the task (`box/progress.relevant`)."""
    already_failing: tuple[str, ...] = ()
    """Failing the first time a run saw it, and not the task's: never counted against it."""
    edited: tuple[str, ...] = ()
    """Of `fixed`, the tests in a file the task changed."""


class DiagnosticsSeen(BaseModel):
    """The task's files at the stop against their kept start, each problem as
    `file:line source code message`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    new_errors: tuple[str, ...] = ()
    new_warnings: tuple[str, ...] = ()
    already: tuple[str, ...] = ()
    """Errors there at the start and still there: never the task's."""
    no_record: tuple[str, ...] = ()
    """Files with no verdict at the start: their problems cannot be told new."""
    unchecked: tuple[str, ...] = ()
    """Files with no fresh verdict at the stop, the wait over: never clean."""


class ProgressCall(BaseModel):
    """One worker call as the progress record tagged it (`box/progress.py`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: int
    """The model response of the round it was in, from 1: one response's calls share it."""
    tool: str
    """The registry name."""
    tag: ProgressTag
    key: str
    """The tool and its arguments as one short sha: two calls alike have one key."""
    about: str
    """The call in a few words, for what the watchdog quotes back."""
    file: str | None = None
    """The file a read-family call read, as it named it; None for any other call."""
    weight: int = 1
    """The tool calls it made: more than one for a Python block, which the task's call
    cap counts call by call."""
    changed: tuple[str, ...] = ()
    """The files its own step changed, when it is tagged `changed`."""
    run: TestRunSeen | None = None
    """Set when it ran tests, read by what they printed."""
    regressed: tuple[Regression, ...] = ()
    """The tests this run confirmed broken."""


class WorkerProgress(BaseModel):
    """The worker's progress this round, as the progress record holds it:
    every call in order, with what a hook needs to say what was ignored. A hook replays it, so
    it holds no state of its own."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calls: tuple[ProgressCall, ...] = ()
    """This round's calls, in order."""
    brief_files: tuple[str, ...] = ()
    """The repository files the task's brief names."""
    opened: tuple[str, ...] = ()
    """The files the task's reads have asked for, every round."""
    last_failure: str | None = None
    """The last lines of the task's last refused call or failing command, as shown."""
    failure_call: int | None = None
    """Where in `calls` that failure was, from 1; None when it came before this round. A hook
    quotes it only from within the stall, never one a later change may have answered."""
    failure_files: tuple[str, ...] = ()
    """The repository files `last_failure` names."""
    calls_left: int | None = None
    """The task's calls (`worker.max_tool_calls`) left when the round began; None where
    nothing bounds them."""
    runnable: bool = False
    """A test runner exists for a changed file's language and the worker may run commands:
    whether "no test run since the last change" means anything."""
    tests: TestsSeen | None = None
    """The task's tests; None when no complete run was read."""
    diagnostics: DiagnosticsSeen | None = None
    """The task's files against their start, as its last stop read them; None
    where nothing measured them."""


class AfterBatch(HookPayload):
    tools: tuple[str, ...]
    """The registry names of one response's calls, in order."""
    progress: WorkerProgress | None = None
    """The worker's, where a hook at this point is bound for it."""


class OpenTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    state: str


class BeforeStop(HookPayload):
    reply: str | None
    """What the role is about to answer with: a script's `last_assistant_message`."""
    via: Literal["reply", "finish"] = "reply"
    tools_open: bool = True
    """Whether the role may still call tools; a `Continue` is offered only while it may."""
    continues_used: int = 0
    continues_left: int = 0
    facts: TaskFactsP | None = None
    """The worker's: its task's facts so far."""
    open_task: OpenTask | None = None
    """Athena's: the task she holds, if one is open."""
    ended: bool = False
    """Athena's: her plan was ended (`outgrown`), so nothing more runs."""
    outcomes: tuple[str, ...] = ()
    """Zeus's: the outcome of each episode his turn ran."""
    unattended: bool = False
    """Zeus's: no person is there to answer."""
    progress: WorkerProgress | None = None
    """The worker's: its round's progress."""
    handed_up: bool = False
    """The worker's: the progress watchdog asked it this round to stop and report, so
    nothing sends it back to work more."""

    def paths(self) -> tuple[str, ...]:
        return () if self.facts is None else tuple(f.path for f in self.facts.files)


class PermissionAsked(HookPayload):
    question: str
    tier: str
    tool: str | None = None


class RoundEnded(HookPayload):
    task: str
    round: int
    stopped: RoundEnd
    report: str | None
    facts: TaskFactsP
    retries_used: int = 0
    retries_left: int = 0
    calls_left: int | None = None
    """The task's tool calls left for another round; None where nothing bounds them."""
    progress: WorkerProgress | None = None
    """The round's progress as the worker's record holds it, where a hook at this point
    is bound."""

    def paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.facts.files)


class TaskReviewed(HookPayload):
    task: str
    answer: ReviewAnswer
    text: str | None = None
    fresh: Fresh = None


class TaskCommitted(HookPayload):
    task: str
    sha: str | None
    files: tuple[str, ...] = ()

    def paths(self) -> tuple[str, ...]:
        return self.files


class EpisodeEnded(HookPayload):
    outcome: str
    why: str
    tasks: int
    committed: int


class BeforeCompact(HookPayload):
    prefix_tokens: int


class JobExited(HookPayload):
    job: str
    command: str
    exit_code: int | None


class SessionEnd(HookPayload):
    turns: int


PAYLOADS: Final[Mapping[HookPoint, type[HookPayload]]] = {
    HookPoint.SESSION_START: SessionStart,
    HookPoint.PROMPT_SUBMITTED: PromptSubmitted,
    HookPoint.BEFORE_REQUEST: BeforeRequest,
    HookPoint.BEFORE_TOOL: BeforeTool,
    HookPoint.AFTER_TOOL: AfterTool,
    HookPoint.AFTER_STEP: AfterStep,
    HookPoint.AFTER_BATCH: AfterBatch,
    HookPoint.BEFORE_STOP: BeforeStop,
    HookPoint.PERMISSION_ASKED: PermissionAsked,
    HookPoint.ROUND_ENDED: RoundEnded,
    HookPoint.TASK_REVIEWED: TaskReviewed,
    HookPoint.TASK_COMMITTED: TaskCommitted,
    HookPoint.EPISODE_ENDED: EpisodeEnded,
    HookPoint.BEFORE_COMPACT: BeforeCompact,
    HookPoint.JOB_EXITED: JobExited,
    HookPoint.SESSION_END: SessionEnd,
}
"""Each point's one payload model."""


@dataclass(frozen=True)
class HookEvent:
    """One firing: the point, who it is about, and its payload. `model` is the role's model,
    for a hook's `model` matcher; the ids place it in the journal."""

    point: HookPoint
    role: HookRole
    payload: HookPayload
    model: str = ""
    run_id: str | None = None
    session: str | None = None
    task_id: str | None = None
    round: int | None = None
    call_id: str | None = None

    def __post_init__(self) -> None:
        # The exact model, not a subclass: `AfterTool` extends `BeforeTool`, and one point
        # handed the other's facts is a fire site's bug.
        want = PAYLOADS[self.point]
        if type(self.payload) is not want:
            raise HarnessBug(
                f"{self.point} carries a {want.__name__}, got {type(self.payload).__name__}"
            )
