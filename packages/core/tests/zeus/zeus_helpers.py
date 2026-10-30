"""Plain helpers for the Zeus loop's suite — a uniquely named module, never `conftest`.

A scripted model stands in for the frontier rung: it returns the completions a test lists,
in order, and keeps every request it was sent, so a test reads exactly what the loop put on
the wire. The ports — the graph, the box, the person — are small fakes; the session, the
Reader, the prefix and the dispatcher are the real ones.
"""

from __future__ import annotations

import itertools
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from chickadee.budget.manager import CostLedger
from chickadee.client.live import LiveSink
from chickadee.client.request import ChatRequest, text_of
from chickadee.context.homes import SessionHome
from chickadee.context.reader import Reader
from chickadee.context.recall import Recall
from chickadee.context.replay import ReplayCache
from chickadee.context.resolver import HandleResolver
from chickadee.context.results import ToolFailure
from chickadee.context.session import Session, SessionState
from chickadee.context.store import SessionLog
from chickadee.context.terms import Vision
from chickadee.executor.executor import Executor
from chickadee.hooks.fire import RoleHooks
from chickadee.isolation.none import Passthrough
from chickadee.roles import role
from chickadee.schema import StopReason, ToolCall, sha256_hex
from chickadee.schema.adapter import Completion, RetryParams, Usage
from chickadee.schema.execution import SandboxLevel
from chickadee.schema.interaction import (
    ActionApproval,
    Answer,
    AnswerSource,
    ApprovalDecision,
    ApprovalStop,
    Question,
    StopAnswer,
    StopDecision,
)
from chickadee.shell import Shell
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import Locator
from chickadee.tools.policy import RolePolicy
from chickadee.tools.spec import SURFACE
from chickadee.tools.tool import Said
from chickadee.work import DEFAULT_DIR
from chickadee.zeus.calls import ZeusTools
from chickadee.zeus.delegate import Delegation
from chickadee.zeus.interrupt import Interrupts
from chickadee.zeus.loop import Event, LoopConfig, ZeusLoop, workspace_generation
from chickadee.zeus.prefix import Env, Prefix, assemble

SESSION_ID = "S-zeus"
TODAY = date(2026, 9, 27)
HEAD = ("0123456789abcdef0123456789abcdef01234567", "feature/frame-branch")
PARAMS = RetryParams(temperature=0.0, max_tokens=2048)


def env(workspace: Path, *, model: str = "frontier-model-a") -> Env:
    return Env(
        workspace=str(workspace),
        platform="linux",
        sandbox=SandboxLevel.NATIVE,
        reads_masked=True,
        model=model,
        repo_trust="first_party",
    )


def zeus_prefix(
    workspace: Path,
    *,
    skills: str = "",
    model: str = "frontier-model-a",
    policy: RolePolicy | None = None,
) -> Prefix:
    surface = policy.surface(BUILTINS) if policy is not None else SURFACE
    return assemble(
        role("zeus"), surface, skills, env(workspace, model=model), work_dir=DEFAULT_DIR
    )


_ids = itertools.count(1)


def call(tool: str, cid: str | None = None, /, **args: object) -> ToolCall:
    """A call as a model writes it: the tool's own arguments, no purpose line.
    Positional only, so a tool's own `name` argument (`symbol`, `tools`) is an argument like
    any other."""
    return ToolCall(id=cid or f"c{next(_ids)}", name=tool, arguments=dict(args))


def calls(*items: ToolCall, text: str = "") -> Completion:
    return Completion(
        text=text,
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        latency_ms=1,
        stop_reason=StopReason.TOOL_USE,
        tool_calls=items,
    )


def answer(text: str = "done") -> Completion:
    return Completion(
        text=text,
        usage=Usage(prompt_tokens=100, completion_tokens=10),
        latency_ms=1,
        stop_reason=StopReason.END_TURN,
    )


Step = Completion | Callable[[ChatRequest], Completion]


def payloads(session: Session, kind: str) -> list[dict[str, object]]:
    """Every payload of one kind, read back from the log as replay would read it."""
    log = SessionLog.read(session.root, session.session_id)
    return [json.loads(log.payload(e)) for e in log.events() if e.kind == kind]


class ScriptedModel:
    """The frontier rung, scripted. Past the script's end it answers `done`, so a test that
    only cares about the first responses need not script the last."""

    def __init__(self, script: Sequence[Step] = ()) -> None:
        self._script = list(script)
        self.requests: list[ChatRequest] = []

    def send(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        if not self._script:
            return answer()
        step = self._script.pop(0)
        return step(request) if callable(step) else step

    def close(self) -> None:
        """The protocol's; a scripted client holds nothing to release."""

    def wire(self, n: int = -1) -> str:
        """Every message of request `n` as text — its content and the arguments of every call
        it carries — for asserting what reached the model."""
        parts: list[str] = []
        for m in self.requests[n].messages:
            parts.append(text_of(m))
            parts.extend(json.dumps(c.arguments, ensure_ascii=False) for c in m.tool_calls)
        return "\n".join(parts)


class ScriptedGraph:
    """The graph's answers, rendered. Records what it was asked."""

    def __init__(self, big: str | None = None) -> None:
        self.asked: list[tuple[str, object]] = []
        self._big = big

    def _say(self, tool: str, arg: object, text: str) -> str:
        self.asked.append((tool, arg))
        return self._big if self._big is not None else text

    def atlas(self) -> str | ToolFailure:
        return self._say("atlas", None, "components: chickadee (python) · edges: 3")

    def search(
        self,
        query: str,
        *,
        kind: Literal["text", "path"] = "text",
        paths: Sequence[str] = (),
        cancel: object = None,
    ) -> str:
        """The `search` port, for `grep` and `glob`."""
        if kind == "path":
            return self._say("glob", query, "src/a.py")
        return self._say("grep", (query, tuple(paths)), f"src/a.py:1: {query}")

    def symbol(self, name: str) -> str | ToolFailure:
        return self._say("symbol", name, f"{name} · def at src/a.py:1 · callers: 0")

    def neighbors(self, target: str, depth: int) -> str | ToolFailure:
        return self._say("neighbors", (target, depth), f"{target}: calls nothing")

    def impact(self, symbols: Sequence[str]) -> str | ToolFailure:
        return self._say("impact", tuple(symbols), "no dependents")

    def tests_for(self, paths: Sequence[str]) -> str | ToolFailure:
        return self._say("tests_for", tuple(paths), "no test file")


@dataclass
class FakeRun:
    run_id: str
    text: str
    box: FakeBox
    design_sha: str | None = None
    opened: bool = True

    def report(self) -> Said:
        # The run's work happens here, and only here: `open` named it and did nothing.
        if self.box.on_run is not None:
            self.box.on_run(self.run_id)
        self.box.ran.append(self.run_id)
        return Said(self.text)

    def abandon(self) -> None:
        self.box.abandoned.append(self.run_id)


@dataclass
class FakeBox:
    """The box: each open names a run; its work, and its report, come at `report()`."""

    started: list[Delegation] = field(default_factory=list[Delegation])
    """Every delegation opened."""
    ran: list[str] = field(default_factory=list[str])
    abandoned: list[str] = field(default_factory=list[str])
    """Runs handed back unreported: what `open` took is released."""
    on_run: Callable[[str], None] | None = None
    """Called with the run id as the run starts its work, to look at the world it runs in."""

    def open(
        self, delegation: Delegation, *, session_id: str, cancel: threading.Event | None
    ) -> FakeRun:
        self.started.append(delegation)
        n = len(self.started)
        # A fake stop approves the work as written: the sha the run names is Zeus's text's.
        text = f"COMPLETED · run R-{n} · T1 committed"
        return FakeRun(f"R-{n}", text, self, sha256_hex(delegation.text))


class ScriptedPerson:
    """The interaction port: answers each question with the next scripted words, or with
    the question's default when there are none — which is what an unattended session does."""

    def __init__(self, *words: str) -> None:
        self._words = list(words)
        self.asked: list[Question] = []

    def ask(self, q: Question, *, cancel: threading.Event | None = None) -> Answer:
        self.asked.append(q)
        if self._words:
            return Answer(
                question_id=q.id, choice=self._words.pop(0), source=AnswerSource.HUMAN
            )
        return Answer(question_id=q.id, choice=q.default, source=AnswerSource.DEFAULT)

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision:
        return StopDecision(answer=StopAnswer.APPROVE, source=AnswerSource.DEFAULT)

    def confirm_action(
        self, approval: ActionApproval, *, cancel: threading.Event | None = None
    ) -> ApprovalDecision:
        raise AssertionError("the rig's shell is given no authority")


@dataclass
class Rig:
    loop: ZeusLoop
    session: Session
    model: ScriptedModel
    graph: ScriptedGraph
    box: FakeBox
    person: ScriptedPerson
    workspace: Path

    def run(self, text: str) -> list[Event]:
        return list(self.loop.turn(text))


def rig(
    tmp_path: Path,
    script: Sequence[Step] = (),
    *,
    config: LoopConfig | None = None,
    workspace: Path | None = None,
    shell: bool = False,
    graph: ScriptedGraph | None = None,
    locator: Locator | None = None,
    box: FakeBox | None = None,
    person: ScriptedPerson | None = None,
    session: Session | None = None,
    interrupts: Interrupts | None = None,
    ledger: CostLedger | None = None,
    charged_as: str | None = "frontier",
    executor: Executor | None = None,
    generation: Callable[[SessionState], int] | None = None,
    reconcile: Callable[[str], str] | None = None,
    today: Callable[[], date] = lambda: TODAY,
    policy: RolePolicy | None = None,
    live: LiveSink | None = None,
    window: int | None = None,
    hooks: RoleHooks | None = None,
    status: Callable[[], str | None] = lambda: None,
) -> Rig:
    ws = workspace or tmp_path / "ws"
    if not ws.is_dir():
        ws.mkdir(parents=True)
    opened = session or Session.create(tmp_path / "sessions", SESSION_ID)
    reader = Reader(
        SessionHome(opened),
        HandleResolver(workspace=ws, session_blob=opened.blob),
        vision=Vision("zeus"),
    )
    graph = graph or ScriptedGraph()
    box = box or FakeBox()
    person = person or ScriptedPerson()
    # The dispatcher's tests run their commands unwrapped: bubblewrap mounts a private tmpfs
    # over `/tmp`, where `tmp_path` lives, so a wrapped command could not even start there.
    # The sandbox's own test passes a wrapping executor and a workspace outside `/tmp`.
    # Unwrapped on purpose, so with the person's key that says so: without it every
    # command here would wait for a confirmation the rig has nobody to give. The floor's own
    # tests are in `test_no_write.py`.
    runner = executor or Executor(backend=Passthrough())
    zeus_shell = (
        Shell(
            runner,
            ws,
            lambda data: reader.store("output", data),
            allow_unsandboxed_shell=True,
        )
        if shell
        else None
    )
    tools = ZeusTools(
        opened,
        reader,
        reader,
        Recall(opened.root, ReplayCache(opened.root)),
        person,
        navigator=graph,
        locator=locator,
        shell=zeus_shell,
        box=box,
        search=graph.search,
    )
    model = ScriptedModel(script)
    loop = ZeusLoop(
        opened,
        zeus_prefix(ws, policy=policy),
        model,
        PARAMS,
        tools,
        config or LoopConfig(max_result_tokens=1_000_000),
        interrupts=interrupts,
        ledger=ledger.for_rung("high-param") if ledger is not None else None,
        model=charged_as,
        today=today,
        head=lambda: HEAD,
        status=status,
        reconcile=reconcile,
        generation=generation or workspace_generation,
        policy=policy,
        live=live,
        window=window,
        hooks=hooks,
    )
    return Rig(loop, opened, model, graph, box, person, ws)
