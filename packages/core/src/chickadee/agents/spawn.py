"""`spawn_agent` and `follow_up`: an agent's rounds and their reports (Amendment 1).

**A follow-up** (Amendment 1 decision 2) is one more round of an agent that has ended: its
conversation, read back from the log (`resume.py`), then Zeus's message as its next user turn.
It runs on the agent's profile and tools, with a round's full limits, and answers with a new
report. It is refused for an id the session does not have, an agent still running, one past
`agents.max_follow_ups`, and a conversation that would no longer fit its window, each saying
what to do instead.

**What an agent starts from.** Its profile's prompt (`adapters/shared/templates/<role>.md`), the
skill's body when a skill was named, and the report contract; then one user message holding the
task and the text of the handles it was passed. Never Zeus's conversation (decision 4).

**What it may do.** Its profile's tools (`profiles.py`), narrowed by the skill's
`allowed-tools`, through one dispatcher over its role's pipeline: the role's call cap and result
budget are that pipeline's rules. A reader of its own (`Reader.scratch`), so what
it read never reaches Zeus's session. The profile's role is never wider than Zeus's ceiling:
refused if it is.

**How a round ends**, each said in the report:
- it answers, `complete`;
- a limit ends its tool use (calls, result tokens, `seconds`, `max_cost_cents`): the next turn
  is its report, `partial` (Amendment 1 decision 3);
- its report does not parse after the re-asks, it keeps calling tools after they ended, its
  model fails, or its conversation stops fitting: `failed`, with no report made up in its place;
- the person's stop: `cancelled`;
- the session's spend cap: `failed`, the cap named.

**What is recorded**, on the session log, as `agent` lines: `agent_start` before anything runs,
each message as `agent_message` before it joins (`conversation.py`), and `agent_report` last,
with the report's blob. The round's spend is charged to the session's ledger as it happens, so
it counts toward the session's cap (decision 4).

**Concurrency, stated where the state is** (Amendment 1 decision 1). Up to
`agents.max_parallel` rounds run at once, each on its own dispatcher thread
(`Dispatcher.execute_group`). What a round holds is its own: its dispatcher, sink, transcript,
`AgentRun`, client, and ports, whose graph connection it opened on its thread and closes when
the round ends (`AgentPorts.close`). What rounds share is guarded where it is declared:
- the session log and its blobs: `Session.lock`. An agent's id is counted from the log and its
  `agent_start` written under that one hold, so two rounds never take one id;
- spend: the rung's `Ledger`, under the budget's lock; a crossed cap is read before each model
  call (`may_start`), so every running round stops at its next one;
- the stop: the one cancel event of the response, set by Esc and read by every round.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from chickadee.adapters.base import (
    SHARED_TEMPLATES,
    first_attempt_params,
    load_template,
    select_blocks,
    tool_descriptions,
)
from chickadee.adapters.frontier import TEMPLATES
from chickadee.agents.conversation import AgentTranscript, Overflow
from chickadee.agents.profiles import Profile
from chickadee.agents.report import (
    BASE_SECTIONS,
    AgentReport,
    ReportDraft,
    check,
    contract,
    parse,
)
from chickadee.agents.resume import (
    conversation,
    histories,
    next_agent_id,
    pages,
    read_path,
    reads,
)
from chickadee.boxctx.conversation import message_bytes
from chickadee.client.request import Message, ModelClient, ToolSchema
from chickadee.config import RungEndpoint
from chickadee.context.citations import LineSource, seen_in_reads
from chickadee.context.record_lines import AgentLine
from chickadee.context.results import InvalidArgument, ToolFailure
from chickadee.limits import limit, window
from chickadee.schema import sha256_hex
from chickadee.schema.adapter import ParseError, ToolCall, Usage
from chickadee.schema.journal import (
    AgentMessageP,
    AgentReportP,
    AgentSource,
    AgentStartP,
    AgentStatus,
    EventKind,
)
from chickadee.skills.discover import Skill, SkillCatalog
from chickadee.skills.frame import frame_body, is_untrusted
from chickadee.skills.load import read_again
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import answer_text
from chickadee.tools.context import (
    ExecPort,
    FileView,
    Navigator,
    SearchPort,
    ToolContext,
    WebPort,
)
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.loop import AskResult, Reply, ask, kept
from chickadee.tools.policy import Ceiling, RolePolicy, within
from chickadee.tools.rules import CallOutcome, Level
from chickadee.tools.sink import Counters, no_hold
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Invocation
from chickadee.trust.events import UntrustedRead

if TYPE_CHECKING:
    from chickadee.budget.manager import Ledger
    from chickadee.context.session import Session

__all__ = ["AgentPorts", "AgentRun", "Agents", "Spawned"]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentPorts:
    """What an agent's tools are given: a role's ports, each built for this one agent, on the
    thread its round runs on. `files` is a reader of its own and `navigator` a graph connection
    of its own; `search` is Zeus's, which holds no state; `exec` the reviewer's shell; `web` the
    researcher's, one `external/web.Web` for the round. Each is None for a profile
    without it: a researcher has the web and nothing else. `close` releases what they opened,
    when the round ends."""

    files: FileView | None = None
    navigator: Navigator | None = None
    search: SearchPort | None = None
    exec: ExecPort | None = None
    web: WebPort | None = None
    close: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class Spawned:
    """What `spawn_agent` resolved before anything ran, which `agent_start` records."""

    profile: Profile
    skill: Skill | None
    """The skill as its file says now (`load.read_again`), never as the index has it."""
    tools: tuple[str, ...]
    notes: tuple[str, ...]
    """What narrowed nothing and was said so: a tool name the profile does not have."""
    body: str | None = None
    """The skill's body, read with it."""


class _AgentSink:
    """The agent's dispatcher sink: the bytes the model sees, nothing recorded per call (each
    message is, by the transcript), and every read's text kept for the citation check."""

    def __init__(self, reads: Mapping[str, list[str]] | None = None) -> None:
        self.reads: dict[str, list[str]] = {p: list(t) for p, t in (reads or {}).items()}
        """By path, what each `read` showed: a follow-up's start with its earlier rounds'
        (`resume.reads`), since they are in its conversation still."""
        self.ended = False

    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        return no_hold(spec)

    def replay(self, level: Level) -> Counters:
        return Counters()

    def render(self, outcome: CallOutcome) -> str:
        return answer_text(outcome)

    def record(self, call: Invocation, outcome: CallOutcome) -> tuple[str, bool]:
        pending = outcome.pending
        ended = outcome.ended_by is not None
        text = answer_text(outcome)
        if ended:
            self.ended = True
        path = read_path(pending.tool or "", pending.raw)
        if path is not None and not ended:
            self.reads.setdefault(path, []).append(text)
        return text, ended


@dataclass(frozen=True)
class _Opening:
    """How a round's conversation opens. A spawn's: its prompt and task, none of them on the
    log yet. A follow-up's: the conversation the log holds, every message of it recorded, and
    `extra`, the follow-up's message, added once the whole is known to fit."""

    round: int
    messages: tuple[Message, ...]
    task_sha: str
    recorded: int = 0
    extra: Message | None = None
    handles: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Setup:
    """A round's dispatcher, its sink, the tools it offers as the model sees them, and the
    endpoint its model is on."""

    dispatcher: Dispatcher[tuple[str, bool]]
    lines: _AgentSink
    tools: tuple[ToolSchema, ...]
    endpoint: RungEndpoint
    web: WebPort | None = None
    """The round's web, whose pages a web citation is checked against."""


@dataclass
class AgentRun:
    """One round's account, written as the round goes."""

    agent: str
    round: int
    started: float = field(default_factory=time.monotonic)
    cost_usd: float = 0.0
    limit_hit: str = ""
    """The limit that ended its tool use, as the report says it."""


@dataclass
class _Researching:
    """The agent's half of the shared loop (`tools.loop.Role`): its calls go through its
    dispatcher, the round's time and spend are checked before each, and its answer is a
    report."""

    dispatcher: Dispatcher[tuple[str, bool]]
    lines: _AgentSink
    run: AgentRun
    profile: Profile
    required: tuple[str, ...]

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        limits = self.profile.limits
        if time.monotonic() - self.run.started >= limits.seconds:
            return self._ended(f"its {limits.seconds} seconds")
        if self.run.cost_usd * 100 >= limits.max_cost_cents:
            return self._ended(f"its {limits.max_cost_cents} cents of spend")
        if self.lines.ended:
            return Reply(self.after_end(), ends=True)
        text, ended = self.dispatcher.call(call)
        if ended and not self.run.limit_hit:
            self.run.limit_hit = "its tool calls or the results it may read"
        return Reply(text, ends=ended)

    def _ended(self, what: str) -> Reply:
        if not self.run.limit_hit:
            self.run.limit_hit = what
        self.lines.ended = True
        return Reply(f"BOUND_REACHED{{{what}}} — not run. Write your report now.", ends=True)

    def after_end(self) -> str:
        return "not run: your tools have ended. Write your report now, with no tool call."

    def parse(self, text: str) -> ReportDraft | ParseError:
        return parse(text, self.required)

    def reask(self, error: ParseError) -> str:
        return error.hint

    def cut(self) -> ParseError:
        return ParseError(
            kind="truncated",
            hint="your report hit the output limit; send it again, shorter, every section in",
        )


class Agents:
    """The `agents` port over one session: it resolves, refuses, runs and records.

    `endpoint` names a rung's model and its endpoint, or why it cannot run (no price row);
    `ledger` is a rung's ledger, so a round's spend counts toward the session's cap;
    `client_for` makes a model client for a rung's endpoint; `ports` builds one agent's ports
    for a role; `handle_text` is a handle of Zeus's as text for the task message, opened as
    his `read` opens it (`Reader.text_of`), or why it cannot be given; `read` and the
    workspace's lines are what the citation check reads (`citations.py`); `trusted` is whether
    this repository's executable config is accepted, which admits its agent skills (decision
    6)."""

    def __init__(
        self,
        session: Session,
        catalog: SkillCatalog,
        profiles: Mapping[str, Profile],
        *,
        ceiling: Ceiling,
        ceilings: Callable[[str], Ceiling],
        policy: Callable[[str], RolePolicy],
        endpoint: Callable[[str], tuple[str, RungEndpoint] | str],
        client_for: Callable[[RungEndpoint], ModelClient],
        ports: Callable[[Profile], AgentPorts],
        handle_text: Callable[[str], str | ToolFailure],
        read: LineSource,
        ledger: Callable[[str], Ledger] | None = None,
        trusted: bool = True,
        third_party: bool = False,
        max_follow_ups: int = int(limit("agents.max_follow_ups").value),
    ) -> None:
        self._session = session
        self._catalog = catalog
        self._profiles = profiles
        self._ceiling = ceiling
        self._ceilings = ceilings
        self._policy = policy
        self._endpoint = endpoint
        self._client_for = client_for
        self._ports = ports
        self._handle_text = handle_text
        self._read = read
        self._ledger = ledger
        self._trusted = trusted
        self._third_party = third_party
        self._max_follow_ups = max_follow_ups
        self._lock = threading.Lock()
        self._running: set[str] = set()
        """The agents with a round running in this process, so one is never continued while
        it runs. Guarded by `_lock`; a spawn adds its agent while it holds the session's lock
        too (`_begin`), and nothing takes the session's lock while holding this one."""

    # -- resolving -----------------------------------------------------------------------

    def resolve(self, skill: str | None, profile: str | None) -> Spawned | InvalidArgument:
        """What would run, the skill read from its file now, or why nothing will: an unknown
        name with the nearest ones, a file that no longer reads, a knowledge skill, one
        that cannot run, an untrusted repository's, or a profile wider than the spawner."""
        if skill is not None and profile is not None:
            return InvalidArgument(
                "profile",
                "a skill and a profile were both named, and a skill names its own profile",
                do="send skill alone, or profile alone",
            )
        found: Skill | None = None
        body: str | None = None
        if skill is not None:
            indexed = self._catalog.get(skill)
            if indexed is None:
                near = self._catalog.near(skill)
                return InvalidArgument(
                    "skill",
                    f"no skill is named {skill!r}",
                    do=f"send one of: {', '.join(near)}"
                    if near
                    else "call find_skill with what the job is, and send a name it gives",
                )
            # Read again, as `load_skill` reads it: the file may have changed or gone since
            # the index was built, and what runs, its profile and tools as well as its body,
            # is what is on disk now.
            again = read_again(indexed)
            if isinstance(again, InvalidArgument):
                return InvalidArgument(
                    "skill",
                    again.why,
                    do="call find_skill to see the skills as they are now, and send one",
                )
            found, body = again
            if found.kind != "agent":
                return InvalidArgument(
                    "skill",
                    f"{skill} is a knowledge skill, not an agent",
                    do=f"load it with load_skill({skill!r}) instead",
                )
            if not found.runnable:
                return InvalidArgument(
                    "skill",
                    f"{skill} cannot run here: {found.note}",
                    do="send profile reader, reviewer or researcher, with the job as the task, "
                    "instead",
                )
            if is_untrusted(found, self._third_party) and not self._trusted:
                return InvalidArgument(
                    "skill",
                    f"{skill} is this repository's agent skill, and the person has not "
                    "trusted the repository (`chickadee trust`), which admits its agents",
                    do="send a profile instead of the skill, or ask the person to trust the "
                    "repository",
                )
            profile = found.profile
        if profile is None:
            return InvalidArgument(
                "skill",
                "neither a skill nor a profile was named",
                do="send skill, or profile (reader, reviewer or researcher) to run with no "
                "skill",
            )
        chosen = self._profiles.get(profile)
        if chosen is None:
            known = ", ".join(sorted(self._profiles))
            return InvalidArgument(
                "profile", f"no profile is named {profile!r}", do=f"send one of: {known}"
            )
        refusal = self._wider(chosen)
        if refusal is not None:
            return InvalidArgument(
                "profile",
                refusal,
                do="send a profile within yours: reader, reviewer or researcher",
            )
        tools, foreign = chosen.narrowed(found.tools if found is not None else None)
        notes = (f"not this profile's tools, so they narrow nothing: {', '.join(foreign)}",)
        return Spawned(chosen, found, tools, notes if foreign else (), body)

    def _wider(self, profile: Profile) -> str | None:
        """Why `profile`'s role would reach past the spawner's ceiling, or None (decision 4)."""
        theirs = self._ceilings(profile.role)
        if (
            not within(theirs.tier, self._ceiling.tier)
            or not theirs.ports <= self._ceiling.ports
        ):
            return f"the {profile.name} profile reaches past what the spawner may do"
        return None

    # -- running -------------------------------------------------------------------------

    def spawn(
        self,
        skill: str | None,
        profile: str | None,
        task: str,
        handles: Sequence[str],
        *,
        call_id: str,
        cancel: threading.Event | None = None,
    ) -> str | InvalidArgument:
        """Resolve, run one round, and return the report as Zeus reads it."""
        got = self.run(skill, profile, task, handles, call_id=call_id, cancel=cancel)
        return got if isinstance(got, InvalidArgument) else got.render()

    def run(
        self,
        skill: str | None,
        profile: str | None,
        task: str,
        handles: Sequence[str],
        *,
        call_id: str,
        cancel: threading.Event | None = None,
    ) -> AgentReport | InvalidArgument:
        """`spawn`'s round, its report as a value: what a measurement reads (the researcher's
        evaluation), where Zeus reads it rendered."""
        spawned = self.resolve(skill, profile)
        if isinstance(spawned, InvalidArgument):
            return spawned
        if not task.strip():
            return InvalidArgument(
                "task",
                "the task is empty",
                do="send task: what to find out, what to return, where to look, and what is "
                "out of scope",
            )
        reachable = self._endpoint(spawned.profile.rung)
        if isinstance(reachable, str):
            return InvalidArgument(
                "profile",
                f"the {spawned.profile.name} profile is unavailable: {reachable}",
                do="send another profile, or do the job yourself",
            )
        if handles and "files" not in self._ceilings(spawned.profile.role).ports:
            # An agent with no files is given none of the session's, so
            # nothing of the repository rides out in what it sends to the web.
            return InvalidArgument(
                "handles",
                f"the {spawned.profile.name} profile reads no files, so it is given none",
                do="put what it must know in the task, in your own words, and send no handles",
            )
        given: list[str] = []
        for handle in handles:
            text = self._handle_text(handle)
            if not isinstance(text, str):
                return InvalidArgument(
                    "handles",
                    f"{handle} cannot be given: {text.render()}",
                    do="send only handles of this session's text results, or none",
                )
            given.append(f"### {handle}\n{text}")
        model, endpoint = reachable
        skill_text: str | None = None
        if spawned.skill is not None and spawned.body is not None:
            skill_text = self._framed(spawned.skill, spawned.body)
        required = contract(spawned.skill.front.report_sections if spawned.skill else ())
        web = "web" in self._ceilings(spawned.profile.role).ports
        opening = _Opening(
            round=0,
            messages=(
                Message(role="system", content=_system(spawned, required, skill_text, web)),
                Message(role="user", content=_task(task, tuple(given))),
            ),
            task_sha=sha256_hex(task),
            handles=tuple(handles),
        )
        return self._round(
            spawned, model, endpoint, opening, None, call_id=call_id, cancel=cancel
        )

    def _framed(self, skill: Skill, body: str) -> str:
        """An agent skill's body as `load_skill` frames it: untrusted under a third-party
        repository, with its read journaled as `untrusted_read`, since that agent may
        hold a shell."""
        untrusted = is_untrusted(skill, self._third_party)
        if untrusted:
            self._session.append(
                UntrustedRead(path=str(skill.path), bytes=len(body.encode("utf-8")))
            )
        return frame_body(skill, body, sha256_hex(body.encode("utf-8")), untrusted=untrusted)

    def follow_up(
        self,
        agent: str,
        message: str,
        *,
        call_id: str,
        cancel: threading.Event | None = None,
    ) -> str | InvalidArgument:
        """Continue `agent`'s conversation with `message`: one more round, a new report
        (Amendment 1 decision 2). The conversation is the log's (`resume.py`)."""
        if not message.strip():
            return InvalidArgument(
                "message",
                "the message is empty",
                do="send message: what you want from the agent now",
            )
        with self._lock:
            if agent in self._running:
                return InvalidArgument(
                    "agent",
                    f"{agent} is still running",
                    do="follow it up once its report is back",
                )
            self._running.add(agent)
        try:
            return self._continue(agent, message, call_id=call_id, cancel=cancel)
        finally:
            with self._lock:
                self._running.discard(agent)

    def _continue(
        self, agent: str, message: str, *, call_id: str, cancel: threading.Event | None
    ) -> str | InvalidArgument:
        with self._session.lock:
            known = histories(self._session.state)
        history = known.get(agent)
        if history is None:
            ids = ", ".join(known) or "none yet"
            return InvalidArgument(
                "agent",
                f"no agent {agent} in this session; its agents are: {ids}",
                do="send an id a report of this session opened with, or spawn_agent a new one",
            )
        fresh = f"spawn a fresh agent, with what it needs from {agent}'s report in its task"
        done = history.rounds - 1
        if done >= self._max_follow_ups:
            return InvalidArgument(
                "agent",
                f"{agent} has had its {done} follow-ups (`agents.max_follow_ups`)",
                do=fresh,
            )
        start = history.start
        spawned = self.resolve(start.skill, None if start.skill is not None else start.profile)
        if isinstance(spawned, InvalidArgument):
            return InvalidArgument(
                "agent", f"{agent} cannot be continued: {spawned.why}", do=fresh
            )
        reachable = self._endpoint(spawned.profile.rung)
        if isinstance(reachable, str):
            return InvalidArgument(
                "agent", f"{agent} cannot be continued: {reachable}", do=fresh
            )
        model, endpoint = reachable
        before = conversation(history, self._session.blob)
        opening = _Opening(
            round=history.rounds,
            messages=tuple(before),
            recorded=len(before),
            extra=Message(role="user", content=message),
            task_sha=sha256_hex(message),
        )
        got = self._round(
            spawned,
            model,
            endpoint,
            opening,
            agent,
            call_id=call_id,
            cancel=cancel,
            fresh=fresh,
        )
        return got if isinstance(got, InvalidArgument) else got.render()

    def _round(
        self,
        spawned: Spawned,
        model: str,
        endpoint: RungEndpoint,
        opening: _Opening,
        agent: str | None,
        *,
        call_id: str,
        cancel: threading.Event | None,
        fresh: str = "",
    ) -> AgentReport | InvalidArgument:
        """One round over ports of its own, opened on this thread and closed when it ends. A
        follow-up's message is added only once the conversation is known to fit."""
        ports = self._ports(spawned.profile)
        try:
            setup = self._setup(spawned, endpoint, ports, opening, cancel)
            transcript = AgentTranscript(
                list(opening.messages),
                first_attempt_params(endpoint),
                budget=window(endpoint) - endpoint.max_output_tokens,
                tools=setup.tools,
                recorded=opening.recorded,
            )
            extra = opening.extra
            if extra is not None and not transcript.fits(str(extra.content)):
                return InvalidArgument(
                    "agent", f"{agent}'s conversation would no longer fit its window", do=fresh
                )
            agent = self._begin(spawned, model, opening, agent, call_id)
            try:
                round_ = opening.round
                transcript.record = lambda m: self._message(agent, round_, m)
                if extra is not None:
                    transcript.add(extra)
                run = AgentRun(agent, round_)
                try:
                    report = self._research(spawned, run, model, setup, transcript, cancel)
                except Exception as exc:  # the harness's own failure ends the round, said
                    # Named by its type and logged whole, as the worker's is; the round is
                    # then reported `failed`, never left looking like one a crash cut.
                    _log.error(
                        "agent %s round %d stopped on a harness error",
                        agent,
                        round_,
                        exc_info=exc,
                    )
                    report = AgentReport(
                        agent=agent,
                        round=round_,
                        skill=spawned.skill.name if spawned.skill is not None else None,
                        profile=spawned.profile.name,
                        status="failed",
                        tool_calls=setup.dispatcher.state.calls("call"),
                        cost_usd=run.cost_usd,
                        why=f"the harness failed: {type(exc).__name__}: {exc}",
                    )
            finally:
                with self._lock:
                    self._running.discard(agent)
        finally:
            ports.close()
        self._finish(report)
        return report

    def _begin(
        self,
        spawned: Spawned,
        model: str,
        opening: _Opening,
        agent: str | None,
        call_id: str,
    ) -> str:
        """The round's `agent_start`, and a spawn's id, under one hold of the session's lock:
        two rounds never take one id, and a spawn is running from its first line."""
        profile = spawned.profile
        skill = spawned.skill
        with self._session.lock:
            if agent is None:
                agent = next_agent_id(self._session.state)
                with self._lock:
                    self._running.add(agent)
            self._line(
                EventKind.AGENT_START,
                AgentStartP(
                    agent=agent,
                    round=opening.round,
                    skill=skill.name if skill is not None else None,
                    source=_source(skill),
                    profile=profile.name,
                    rung=profile.rung,
                    model=model,
                    task_sha=opening.task_sha,
                    handles=opening.handles,
                    call_id=call_id,
                ),
            )
        return agent

    def _setup(
        self,
        spawned: Spawned,
        endpoint: RungEndpoint,
        ports: AgentPorts,
        opening: _Opening,
        cancel: threading.Event | None,
    ) -> _Setup:
        """The round's dispatcher over `ports`, and the tools it offers, as the model sees
        them."""
        profile = spawned.profile
        policy = replace(self._policy(profile.role), enabled=spawned.tools)
        lines = _AgentSink(reads(opening.messages))
        ctx = ToolContext(
            role=profile.role,
            files=ports.files,
            navigator=ports.navigator,
            search=ports.search,
            exec=ports.exec,
            web=ports.web,
            cancel=cancel,
        )
        if ports.web is not None:
            # Guard 1 of `web_fetch`: what the round was given may be
            # fetched: the person's and Zeus's words, and what a follow-up's earlier rounds'
            # tools answered. Never the profile's own prompt, and never what the agent wrote:
            # a URL a page talked it into writing was given to nobody.
            for message in opening.messages:
                if message.role in ("user", "tool") and isinstance(message.content, str):
                    ports.web.saw(message.content)
            if opening.extra is not None and isinstance(opening.extra.content, str):
                ports.web.saw(opening.extra.content)
            for url, shown in pages(opening.messages).items():
                ports.web.carry(url, shown)
        dispatcher = Dispatcher[tuple[str, bool]](
            BUILTINS,
            policy,
            ctx,
            lines,
            limits=_limits(profile),
        )
        dispatcher.open_scope("call")
        offered = dispatcher.offered()
        described = tool_descriptions(offered, TEMPLATES)
        tools = tuple(
            ToolSchema(name=s.name, description=described[s.description], schema=s.schema)
            for s in offered
        )
        return _Setup(dispatcher, lines, tools, endpoint, ports.web)

    def _research(
        self,
        spawned: Spawned,
        run: AgentRun,
        model: str,
        setup: _Setup,
        transcript: AgentTranscript,
        cancel: threading.Event | None,
    ) -> AgentReport:
        """The round itself: the shared loop over the conversation, and its report. `run` is
        kept by the caller, so a harness error still reports what the round spent."""
        profile = spawned.profile
        skill = spawned.skill
        ledger = self._ledger(profile.rung) if self._ledger is not None else None
        required = contract(skill.front.report_sections if skill is not None else ())
        researching = _Researching(setup.dispatcher, setup.lines, run, profile, required)

        def charge(usage: Usage) -> float:
            cost = 0.0 if ledger is None else ledger.charge_usage("", usage, model)
            run.cost_usd += cost
            return cost

        asked: AskResult[ReportDraft]
        try:
            asked = ask(
                self._client_for(setup.endpoint),
                transcript,
                researching,
                reasks=profile.limits.report_reasks,
                charge=charge,
                charge_failed=charge,
                may_start=None if ledger is None else ledger.may_start,
                cancel=cancel,
            )
        except Overflow as exc:
            asked = AskResult(end="failed")
            run.limit_hit = f"its conversation no longer fits its window: {exc}"
        reply = kept(asked)
        if reply is not None:
            # The loop keeps no final answer (`tools.loop.kept`); its conversation does, so a
            # follow-up continues from its report.
            transcript.add(reply)
        return self._report(spawned, run, asked, setup)

    def _report(
        self,
        spawned: Spawned,
        run: AgentRun,
        asked: AskResult[ReportDraft],
        setup: _Setup,
    ) -> AgentReport:
        calls = setup.dispatcher.state.calls("call")
        base = AgentReport(
            agent=run.agent,
            round=run.round,
            skill=spawned.skill.name if spawned.skill is not None else None,
            profile=spawned.profile.name,
            status="failed",
            tool_calls=calls,
            tokens=asked.tokens,
            cost_usd=run.cost_usd,
        )
        if asked.end == "answered" and asked.answer is not None:
            draft = asked.answer
            findings, result = check(
                draft,
                read=self._read,
                seen=seen_in_reads(setup.lines.reads),
                fetched=None if setup.web is None else setup.web.fetched,
            )
            status: AgentStatus = (
                "partial" if asked.tools_ended or run.limit_hit else "complete"
            )
            extra = {k: v for k, v in draft.sections.items() if k not in BASE_SECTIONS}
            why = f"ended by {run.limit_hit}" if status == "partial" and run.limit_hit else ""
            return replace(
                base,
                status=status,
                summary=draft.summary,
                findings=findings,
                sections=extra,
                not_done=draft.not_done,
                checked=result,
                why=why,
            )
        why = {
            "unparsed": "its report never had every section, after the re-asks",
            "kept_calling": "it kept calling tools after they had ended, and never reported",
            "cancelled": "stopped by the person",
            "spend_cap": "the session's spend cap is crossed; no further call started",
        }.get(asked.end, "")
        if asked.end == "failed":
            why = run.limit_hit or (
                f"its model failed: {asked.failure}" if asked.failure else "its model failed"
            )
        status = "cancelled" if asked.end == "cancelled" else "failed"
        return replace(base, status=status, why=why)

    # -- recording -----------------------------------------------------------------------

    def _line(
        self, kind: EventKind, payload: AgentStartP | AgentMessageP | AgentReportP
    ) -> None:
        self._session.append(AgentLine(kind, payload))

    def _message(self, agent: str, round_: int, message: Message) -> None:
        sha = self._session.put_blob(message_bytes(message))
        self._line(
            EventKind.AGENT_MESSAGE,
            AgentMessageP(agent=agent, round=round_, side=message.role, sha=sha),
        )

    def _finish(self, report: AgentReport) -> None:
        rendered = report.render()
        sha = self._session.put_blob(rendered.encode("utf-8"))
        flagged = sum(len(f.flagged) for f in report.findings)
        self._line(
            EventKind.AGENT_REPORT,
            AgentReportP(
                agent=report.agent,
                round=report.round,
                status=report.status,
                findings=len(report.findings),
                flagged=flagged,
                tool_calls=report.tool_calls,
                tokens=report.tokens,
                cost_usd=report.cost_usd,
                report_sha=sha,
            ),
        )
        self._session.sync()


def _limits(profile: Profile) -> dict[str, int]:
    """The role's pipeline's keys, at this profile's values."""
    role, lim = profile.role, profile.limits
    return {
        f"{role}.max_tool_calls": lim.max_tool_calls,
        f"{role}.max_cumulative_result_tokens": lim.max_result_tokens,
    }


def _source(skill: Skill | None) -> AgentSource:
    """By whose tree the skill is in, as trust decides (`frame.is_untrusted`): a repository's
    `.claude/agents/` file is `project`, as its `.chickadee/skills/` one is."""
    return "built-in" if skill is None else skill.origin


def _system(
    spawned: Spawned, required: tuple[str, ...], skill: str | None, web: bool = False
) -> str:
    """The profile's prompt, the skill's body framed as its author's instructions
    (`Agents._framed`), then the report contract: fixed for the round, so its bytes are
    cached. `web` is a profile whose findings cite pages, not lines."""
    parts = [load_template(SHARED_TEMPLATES, f"{spawned.profile.role}.md").rstrip()]
    if skill is not None:
        parts.append(skill)
    sections = "\n".join(f"## {s}" for s in required)
    report = select_blocks(
        load_template(SHARED_TEMPLATES, "agent_report.md"), ("web",) if web else ()
    )
    parts.append(report.replace("{{sections}}", sections).rstrip())
    if spawned.notes:
        parts.append("\n".join(f"Note: {n}." for n in spawned.notes))
    return "\n\n".join(parts) + "\n"


def _task(task: str, handles: tuple[str, ...]) -> str:
    """The task, then what Zeus passed: what the agent is given, and all it is given."""
    out = ["## Task", task.strip()]
    if handles:
        out += ["", "## What you were given", *handles]
    return "\n".join(out)
