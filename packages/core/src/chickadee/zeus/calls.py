"""Zeus's calls on the one dispatcher: the ports its
tools are given, its own admit rules, and its sink.

Everything that is the same for every role is `chickadee/tools/`'s: the tools themselves
(`tools/builtin/`), the shared rules and their order (`ROLES`' Zeus pipeline), the stages.
What is Zeus's own is here:

- **`ZeusTools`**, the ports: the session's `Reader` as `files` and
  `handles`, the graph's navigator, the scout as `locate`'s locator, `recall`'s journal, the
  catalog the `tools` tool lists, the person, the notes, the shell.
- **Its three slots** in the pipeline: `IntentGiven` and `FindingsGiven` read the two meta
  arguments every call carries, and `ExcessCoder` refuses a second
  `coder` split into its own exchange. Repetition is the shared rule with Zeus's
  identity and note, both carrying the workspace's generation.
- **`ZeusSink`**, the session-log tool line every call ends in (commit, then emit).

**What a response tells the rules** is `ResponseFacts`: written by the loop before it admits
a response, read by the slots while it does. One loop's, on one thread, as its dispatcher is.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from chickadee.context.lines import canonical_json
from chickadee.context.request import est_text
from chickadee.context.results import InvalidArgument, ToolResult
from chickadee.errors import HarnessBug
from chickadee.hooks.fire import RoleHooks
from chickadee.schema import sha256_hex
from chickadee.skills.load import NoSkills
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import Repetition, refused
from chickadee.tools.context import (
    AgentsPort,
    DocsPort,
    ExecPort,
    FileView,
    HandleStore,
    JobsPort,
    Locator,
    Navigator,
    SearchPort,
    SkillsPort,
    TeamPort,
    ToolContext,
    WorktreePort,
)
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.meta import FINDINGS, split_meta
from chickadee.tools.policy import RolePolicy
from chickadee.tools.rules import (
    BoundReached,
    CallOutcome,
    Level,
    Pass,
    Pending,
    Refuse,
    ScopeState,
    Verdict,
    repeated,
)
from chickadee.tools.sink import Counters, RoleExecutors
from chickadee.tools.spec import INLINE_BYTES_DEFAULT, RESULT_LINES_DEFAULT, SURFACE, ToolSpec
from chickadee.tools.tool import Defaulted, Invocation, Said
from chickadee.zeus.delegate import BoxPort, work_digest
from chickadee.zeus.tools.act import NoShell
from chickadee.zeus.tools.navigate import NoIndex, NoLocator, SessionJournal, ZeusCatalog
from chickadee.zeus.tools.record import SessionNotes
from chickadee.zeus.tools.talk import ZeusPerson

if TYPE_CHECKING:
    import threading

    from chickadee.context.recall import Recall
    from chickadee.context.session import Session, SessionState, ToolLine
    from chickadee.schema.interaction import InteractionPort

__all__ = [
    "ExcessCoder",
    "FindingsGiven",
    "RepetitionKey",
    "ResponseFacts",
    "ZeusSink",
    "ZeusTools",
    "repetition_key",
    "zeus_dispatcher",
]

RepetitionKey = tuple[str, str, int]
"""D98's key: the tool as sent, its own arguments' digest, the workspace's generation."""


# -- the ports ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeusTools:
    """The ports Zeus's tools are given (`25-zeus.md` the `tools/`). A port
    with nothing behind it answers its own failure (`NoIndex`, `NoLocator`), so no tool leaves
    the offered set because an index or a scout is missing."""

    session: Session
    files: FileView
    """The one way source text enters Zeus's context (invariant 13): the session's `Reader`."""
    handles: HandleStore
    """The blob store every navigation answer is kept in: the same `Reader`."""
    recall: Recall
    interaction: InteractionPort
    navigator: Navigator | None = None
    locator: Locator | None = None
    shell: ExecPort | None = None
    box: BoxPort | None = None
    """The box `coder` delegates to; the loop runs it (`act.Delegations`), since it streams."""
    skills: SkillsPort | None = None
    """The session's skills; none indexed answers `NoSkills`."""
    search: SearchPort | None = None
    """`grep` and `glob` over the session's root; None, and neither is offered."""
    docs: DocsPort | None = None
    """His design's documents (`zeus/docs.DesignDocs`); None, and `doc` is not offered."""
    jobs: JobsPort | None = None
    """The session's jobs (`zeus/jobs.JobTable`); None, and `job` is not offered."""
    worktree: WorktreePort | None = None
    """The session's moves between the checkout and a worktree (`zeus/worktree.WorktreeMoves`);
    None, and `worktree` is not offered."""
    team: TeamPort | None = None
    """The session's run, handed over in the background; None, and `pause` is
    not offered."""
    agents: AgentsPort | None = None
    """The agents he spawns (`agents/spawn.Agents`); None, and `spawn_agent` is not
    offered."""
    surface: tuple[ToolSpec, ...] = SURFACE
    descriptions: Mapping[str, str] | None = None

    def context(
        self,
        cancel: threading.Event | None,
        *,
        inline_bytes: int = INLINE_BYTES_DEFAULT,
        result_lines: int = RESULT_LINES_DEFAULT,
        hooks: RoleHooks | None = None,
    ) -> ToolContext:
        return ToolContext(
            role="zeus",
            files=self.files,
            navigator=self.navigator if self.navigator is not None else NoIndex(),
            handles=self.handles,
            exec=self.shell if self.shell is not None else NoShell(),
            locator=self.locator if self.locator is not None else NoLocator(),
            journal=SessionJournal(
                self.session, self.recall, self.handles, inline_bytes=inline_bytes
            ),
            notes=SessionNotes(self.session),
            catalog=ZeusCatalog(self.surface, self.descriptions, entry_cap=result_lines),
            user=ZeusPerson(self.interaction),
            skills=self.skills if self.skills is not None else NoSkills(),
            search=self.search,
            docs=self.docs,
            jobs=self.jobs,
            worktree=self.worktree,
            team=self.team,
            agents=self.agents,
            cancel=cancel,
            hooks=hooks,
        )


# -- Zeus's rules ---------------------------------------------------------------------------


@dataclass
class ResponseFacts:
    """What the loop knows of the response being admitted that one call cannot show: whether
    it is a second `coder` split from the first, and why each call's findings were
    refused (they are checked across the response, before it is recorded)."""

    excess_coder: bool = False
    bad_findings: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class FindingsGiven:
    """The findings a call writes, refused as the response's check found them."""

    facts: ResponseFacts
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        why = self.facts.bad_findings.get(pending.call.id)
        if why is None:
            return Pass()
        do = f"send the call again with {FINDINGS} fixed as the reason says, or without it"
        return Refuse(InvalidArgument(FINDINGS, why, do=do))


@dataclass(frozen=True)
class ExcessCoder:
    """33 §7: one `coder` delegation at a time. A second one in a response is its own
    exchange, and refused there; it does not count."""

    facts: ResponseFacts
    stage: Literal["admit", "dispatch"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        spec = pending.spec
        if spec is None or spec.parallel != "exclusive" or not self.facts.excess_coder:
            return Pass()
        return Refuse(BoundReached("coder", 1, 1, "response"), counts=False)


def repetition_key(pending: Pending, generation: int) -> RepetitionKey:
    """D98's `(tool, args_digest, generation)`. The digest is of the tool's own arguments, so
    a call cannot leave the count by rewording its `findings`; for a delegation to the box it is
    the work as Zeus wrote it, so delegating one design again is the same key however the call
    around it was worded."""
    own, _ = split_meta(pending.raw)
    digest = None
    if pending.spec is not None and pending.spec.model_calls == "box":
        digest = work_digest(own)
    if digest is None:
        digest = sha256_hex(canonical_json(dict(own)))[:16]
    return (pending.sent, digest, generation)


def _repetition_note(count: int, key: Hashable) -> str:
    since = cast("RepetitionKey", key)[2]
    return repeated(f"repetition{{count: {count}, since_generation: {since}}}", count)


# -- the sink -------------------------------------------------------------------------------


class ZeusSink:
    """The session-log tool line: the result as the model will read
    it, with the dispatcher's notes after it, and whether the call ran — false for every
    refusal, every failure but a search that found nothing (`UNRESOLVED` is its result),
    and a default nobody gave. A `coder` result names the run
    its episode line opened."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        """The read family holds the session's lock from the read's merge decision to its
        line: two reads of one file answered in parallel could otherwise
        both absorb the same view. The reader reads and records nothing."""
        if spec.family == "read":
            return self._session.lock
        return nullcontext()

    def render(self, outcome: CallOutcome) -> str | ToolResult:
        notes = outcome.pending.notes
        decided = outcome.pending.decided
        if isinstance(decided, Refuse) and isinstance(decided.failure, BoundReached):
            # A bound refused it, so it was never served: a note admit gave it before a bound
            # refused it at dispatch (a repetition notice) does not go with the refusal.
            notes = ()
        answer = outcome.answer
        if isinstance(answer, ToolResult):
            return answer.with_notes(*notes)
        if answer is not None:
            text = answer.render()
        else:
            if not isinstance(decided, Refuse):
                raise HarnessBug(f"Zeus's pipeline answered {decided!r}, which it never makes")
            failure = decided.failure
            text = failure.render()
        return "\n".join((text, *notes)) if notes else text

    def record(self, call: Invocation, outcome: CallOutcome) -> ToolLine:
        answer = outcome.answer
        # A search that ran and found nothing ran: one rule with Athena's and the
        # worker's records (`refused`).
        ran = not refused(answer) and not isinstance(answer, Defaulted)
        run_id = None
        spec = outcome.pending.spec
        if ran and spec is not None and spec.model_calls == "box":
            run_id = self._run_of(call.at)
        told = answer.outcome if isinstance(answer, ToolResult | Said) else None
        return self._session.record_tool(
            call.id, self.render(outcome), ran=ran, run_id=run_id, outcome=told
        )

    def replay(self, level: Level) -> Counters:
        """A turn starts from zero: a resumed session's turn counters are not resumed."""
        del level
        return Counters()

    def measure(self, outcome: CallOutcome) -> int:
        """What a call took in: `est_text` of the very text its line records."""
        shown = self.render(outcome)
        return est_text(shown if isinstance(shown, str) else shown.render())

    def _run_of(self, at: int) -> str | None:
        return next(
            (e.run_id for e in self._session.state.episodes() if e.opened_at_seq == at), None
        )


# -- the composition ------------------------------------------------------------------------


def zeus_dispatcher(
    tools: ZeusTools,
    sink: ZeusSink,
    facts: ResponseFacts,
    *,
    policy: RolePolicy,
    offered: Sequence[str],
    limits: Mapping[str, int],
    generation: Callable[[SessionState], int],
    cancel: threading.Event | None,
    executors: RoleExecutors,
    hooks: RoleHooks | None = None,
) -> Dispatcher[ToolLine]:
    """Zeus's dispatcher for one turn: its resolved `policy` (`EngineConfig.tool_policy`,
    `[tools.zeus]` over `ROLES`' default), offering what the prefix lists (`offered`, the
    names the model is shown, so the provider prefix and the dispatcher agree), its slots
    filled, its result rules measuring what the sink records, his `hooks` at his calls'
    points. The turn's scope is open."""
    enabled = tuple(policy.registry_name(n) for n in offered)
    unknown = [n for n, e in zip(offered, enabled, strict=True) if e not in BUILTINS]
    if unknown:
        raise HarnessBug(f"the prefix offers tools the policy does not enable: {unknown}")
    session = tools.session

    def identity(pending: Pending) -> Hashable:
        return repetition_key(pending, generation(session.state))

    pipeline = policy.pipeline.fill(
        {
            "findings": FindingsGiven(facts),
            "excess_coder": ExcessCoder(facts),
            "repetition": Repetition(
                "turn", "zeus.repeat", identity=identity, note=_repetition_note, by_span=False
            ),
        }
    ).measured_by(sink.measure)
    policy = dataclasses.replace(
        policy, enabled=tuple(e for e in enabled if e is not None), pipeline=pipeline
    )
    ctx = tools.context(
        cancel,
        inline_bytes=limits["zeus.inline_bytes"],
        result_lines=limits["zeus.result_lines"],
        hooks=hooks,
    )
    dispatcher = Dispatcher(
        BUILTINS,
        policy,
        ctx,
        sink,
        limits=limits,
        executors=executors,
        operation_id=lambda call_id: f"{session.session_id}:{call_id}",
    )
    dispatcher.open_scope("turn")
    return dispatcher
