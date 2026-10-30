"""Doubles for the tools package's own tests: a spec, a tool, a sink that keeps what it is
handed, ports over a dict, the rest of the ports as fakes, and the task worker's edit tools
over a real worktree."""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path

from chickadee.box.edits import WorktreeEdits
from chickadee.box.syntax import SyntaxChecker
from chickadee.context.handle import Handle, HandleKind, Shape, content_sha, handle_id
from chickadee.context.results import CallCancelled, SecretPath, ToolFailure, ToolResult
from chickadee.edits.draft import EditCheck
from chickadee.limits import limit
from chickadee.paths import Place
from chickadee.profiles.loader import load_profile
from chickadee.roles import TASK_WORKER
from chickadee.schema.adapter import ParseError, ToolCall
from chickadee.schema.authorization import Tier
from chickadee.schema.interaction import Answer as InteractionAnswer
from chickadee.schema.interaction import AnswerSource, Question
from chickadee.secrets.denylist import Denylist
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.catalogue import Valid, answer_text
from chickadee.tools.context import EditsPort, ToolContext
from chickadee.tools.dispatch import Dispatcher
from chickadee.tools.loop import Reply
from chickadee.tools.policy import FileRule, RolePolicy
from chickadee.tools.registry import Registry
from chickadee.tools.rules import CallOutcome, Level, Pipeline, Refuse
from chickadee.tools.sink import Counters
from chickadee.tools.spec import INVALID_ARGUMENTS, SURFACE, Bound, Parallel, ToolSpec
from chickadee.tools.tool import Answer, Failure, Invocation, Port, TextAnswer, Tool
from chickadee.tools.validate import Arguments


def make_handle(kind: HandleKind, name: str, data: bytes = b"") -> Handle:
    sha = content_sha(data)
    path = name if kind == "file" else None
    return Handle(
        id=handle_id(kind, path=path, run_id=None, sha=sha),
        kind=kind,
        shape=Shape(bytes=len(data), lines=None, symbols=None, kind="text"),
        preview="",
        ops=(),
        content_sha=sha,
        path=path,
    )


def make_spec(
    name: str = "echo",
    *,
    parallel: Parallel = "any",
    mutates: str = "nothing",
    effects: str = "none",
    idempotent: bool = True,
    bound: Bound | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        family="navigate",
        parallel=parallel,
        mutates=mutates,  # type: ignore[arg-type]
        tier=Tier.T0,
        model_calls="none",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
            "additionalProperties": False,
        },
        description=f"{name}.md",
        bound=bound or Bound("inline_bytes", 4096),
        failures=frozenset({INVALID_ARGUMENTS}),
        effects=effects,  # type: ignore[arg-type]
        idempotent=idempotent,
    )


def echo_tool(
    name: str = "echo",
    *,
    needs: frozenset[Port] = frozenset(),
    runs: list[str] | None = None,
    **spec: object,
) -> Tool:
    seen = runs if runs is not None else []

    def run(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
        seen.append(call.id)
        return TextAnswer(f"{name}: {args.text('text')}", "bytes")

    return Tool(make_spec(name, **spec), needs, run)  # type: ignore[arg-type]


@dataclass
class KeepSink:
    """A `ToolSink` returning the outcome, the bytes it would show, and its replay."""

    replays: Mapping[str, Counters] = field(default_factory=dict[str, Counters])
    records: list[CallOutcome] = field(default_factory=list[CallOutcome])

    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        return nullcontext()

    def render(self, outcome: CallOutcome) -> str | ToolResult:
        return answer_text(outcome)

    def record(self, call: Invocation, outcome: CallOutcome) -> CallOutcome:
        self.records.append(outcome)
        return outcome

    def replay(self, level: Level) -> Counters:
        return self.replays.get(level, Counters())


def policy(
    enabled: Sequence[str] = ("echo",),
    *,
    pipeline: Pipeline | None = None,
    shown_as: Mapping[str, str] | None = None,
) -> RolePolicy:
    return RolePolicy(
        role="test",
        enabled=tuple(enabled),
        shown_as=dict(shown_as or {}),
        files=FileRule("repository", "visible"),
        graph_depth=None,
        pipeline=pipeline or Pipeline(admit=(Valid(),), counts_at=("turn",)),
    )


def dispatcher(
    tools: Sequence[Tool],
    *,
    pipeline: Pipeline | None = None,
    limits: Mapping[str, int] | None = None,
    ctx: ToolContext | None = None,
    sink: KeepSink | None = None,
    shown_as: Mapping[str, str] | None = None,
    enabled: Sequence[str] | None = None,
    executors: Mapping[str, Callable[..., Answer]] | None = None,
    level: Level = "turn",
) -> Dispatcher[CallOutcome]:
    names = enabled if enabled is not None else [t.spec.name for t in tools]
    d = Dispatcher(
        Registry(tools),
        policy(names, pipeline=pipeline, shown_as=shown_as),
        ctx or ToolContext(role="test"),
        sink or KeepSink(),
        limits=limits or {},
        executors=executors or {},
    )
    d.open_scope(level)
    return d


def call(name: str = "echo", text: str = "hi", cid: str = "c1", **extra: object) -> ToolCall:
    return ToolCall(id=cid, name=name, arguments={"text": text, **extra} if text else extra)


@dataclass
class DictFiles:
    """A `FileView` over a dict, refusing the paths `denied` names, as the Reader does."""

    files: Mapping[str, str]
    denied: frozenset[str] = frozenset()
    stored: dict[str, bytes] = field(default_factory=dict[str, bytes])

    def read(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
        max_bytes: int | None = None,
        call_id: str | None = None,
    ) -> ToolResult | ToolFailure:
        if target in self.denied:
            return SecretPath(target, "deny-listed")
        text = self.files[target]
        return ToolResult(make_handle("file", target, text.encode()), content=text)

    def repeat_key(
        self,
        target: str,
        *,
        lines: str | None = None,
        at: str | None = None,
        depth: str | None = None,
    ) -> Hashable | None:
        # A dict's file has no bytes to key on: a read counts by its words.
        return None

    def grep_in(
        self, handle: str, pattern: str, *, context: int = 2
    ) -> ToolResult | ToolFailure:
        hits = [line for line in self.files.get(handle, "").splitlines() if pattern in line]
        return ToolResult(make_handle("file", handle), content="\n".join(hits))

    def place(self, sent: str) -> Place | None:
        return Place(sent) if sent in self.files else None

    def store(self, kind: HandleKind, data: bytes) -> Handle:
        handle = make_handle(kind, f"blob{len(self.stored)}", data)
        self.stored[handle.id] = data
        return handle


@dataclass
class DictNavigator:
    answers: Mapping[str, str]

    def atlas(self) -> str | ToolFailure:
        return self.answers["atlas"]

    def symbol(self, name: str) -> str | ToolFailure:
        return self.answers["symbol"]

    def neighbors(self, target: str, depth: int) -> str | ToolFailure:
        return self.answers["neighbors"]

    def impact(self, symbols: Sequence[str]) -> str | ToolFailure:
        return self.answers["impact"]

    def tests_for(self, paths: Sequence[str]) -> str | ToolFailure:
        return self.answers.get("tests_for", f"tests for {', '.join(paths)}")


@dataclass
class FakePorts:
    """Every other port a built-in needs, answering as its owner would: `recall`'s journal,
    `locate`'s locator, the `tools` catalog, the notes, the person and the shell."""

    asked: list[str] = field(default_factory=list[str])

    def recall(self, query: str) -> ToolResult | ToolFailure:
        self.asked.append(f"recall {query}")
        return ToolResult(make_handle("findings", "recall"), content=f"recalled {query}")

    def locate(
        self, question: str, *, cancel: threading.Event | None
    ) -> str | ToolFailure | CallCancelled:
        self.asked.append(f"locate {question}")
        return f"src/a.py:1 ({question})"

    def offered(self) -> tuple[ToolSpec, ...]:
        return SURFACE

    def descriptions(self) -> Mapping[str, str]:
        return {s.name: f"{s.name}: a stub\n" for s in SURFACE}

    def shown_schema(self, spec: ToolSpec) -> Mapping[str, object]:
        return spec.schema

    @property
    def entry_cap(self) -> int:
        return 20

    def note_concern(self, text: str, anchor: Sequence[str]) -> str | ToolFailure:
        self.asked.append(f"concern {text}")
        return "concern K-1"

    def note_rejected(self, text: str, reason: str, *, at: int) -> str | ToolFailure:
        self.asked.append(f"rejected {text}")
        return "rejection R-1"

    def note(self, text: str) -> str | ToolFailure:
        self.asked.append(f"note {text}")
        return "note N-1"

    def ask(
        self, question: Question, *, cancel: threading.Event | None = None
    ) -> InteractionAnswer:
        self.asked.append(f"ask {question.text}")
        return InteractionAnswer(
            question_id=question.id, choice=question.default, source=AnswerSource.DEFAULT
        )

    def confirm(self, question: str, tier: Tier) -> bool:
        return False

    def run(
        self, args: Arguments, *, operation_id: str, cancel: threading.Event | None
    ) -> ToolResult | Failure:
        self.asked.append(f"shell {operation_id}")
        return ToolResult(make_handle("output", operation_id), content="ok")


# -- the task worker's edit tools --------------------------------------------------------------


def worktree_edits(
    root: Path, *, edit_check: EditCheck = "off", untracked_max_bytes: int = 1024
) -> WorktreeEdits:
    """The box's edit port over the git worktree at `root`, Python's syntax its judge
    (`box/compose.py` builds it so) and JavaScript's files read by name too, as in a checkout
    holding both. `off` by default: a tool test pins the anchor and the answer, and each file's
    syntax line is `tests/box/test_edits.py`'s subject."""
    checker = SyntaxChecker(root, (load_profile("python"), load_profile("javascript")))
    return WorktreeEdits(
        root,
        Denylist(),
        checker,
        edit_check=edit_check,
        untracked_max_bytes=untracked_max_bytes,
    )


WORKER_LIMITS: Mapping[str, int] = {
    name: int(limit(name).value)
    for name in (
        "worker.repeat",
        "worker.print_only",
        "worker.max_tool_calls",
        "worker.result_max",
    )
}
"""The register's defaults for what the task worker's pipeline reads."""


class EditWorker:
    """The task worker's tools as the box serves them: `roles.TASK_WORKER` offering `names`,
    in `line_anchors`' style, decoding `constrain` under their grammars (the policy's
    `constrain`, the context's `constrained`), on one dispatcher over `edits`, its task scope
    open."""

    def __init__(
        self,
        edits: EditsPort,
        names: Sequence[str],
        *,
        line_anchors: str = "number",
        constrain: Sequence[str] = (),
    ) -> None:
        tools = [t for n in names if (t := BUILTINS.get(n)) is not None]
        policy = replace(
            TASK_WORKER,
            enabled=tuple(names),
            line_anchors=line_anchors,
            constrain=tuple(constrain),
        )
        ctx = ToolContext(role="worker", edits=edits, constrained=frozenset(constrain))
        self.dispatcher = Dispatcher(
            Registry(tools), policy, ctx, KeepSink(), limits=WORKER_LIMITS
        )
        self.dispatcher.open_scope("task")
        self.calls = 0

    def call(self, name: str, **arguments: object) -> CallOutcome:
        self.calls += 1
        return self.dispatcher.call(
            ToolCall(id=f"c{self.calls}", name=name, arguments=dict(arguments))
        )


def said(outcome: CallOutcome) -> str:
    """What the model reads of an answered call; asserts it was not refused."""
    assert failure_of(outcome) is None, answer_text(outcome)
    return answer_text(outcome)


def failure_of(outcome: CallOutcome) -> str | None:
    """The failure's name a call was answered with, or None when it ran and stood."""
    if outcome.answer is not None:
        return getattr(outcome.answer, "failure", None)
    decided = outcome.pending.decided
    return decided.failure.failure if isinstance(decided, Refuse) else None


def refused(outcome: CallOutcome, failure: str) -> str:
    """The refusal's text, asserting it is `failure`'s."""
    text = answer_text(outcome)
    assert failure_of(outcome) == failure and text.startswith(f"{failure}: "), text
    return text


@dataclass
class ScriptedRole:
    """A role for `tools.loop.ask`: answers `ok:<x>` as `<x>`, and a call named `stop` ends tool
    use. Every call it ran is in `ran`; `on_call` is set at each."""

    ran: list[str] = field(default_factory=list[str])
    on_call: threading.Event | None = None

    def call(self, call: ToolCall, cancel: threading.Event | None) -> Reply:
        self.ran.append(call.name)
        if self.on_call is not None:
            self.on_call.set()
        return Reply(f"ran {call.name}", ends=call.name == "stop")

    def after_end(self) -> str:
        return "ENDED"

    def parse(self, text: str) -> str | ParseError:
        if text.startswith("ok:"):
            return text[3:]
        return ParseError(kind="bad_format", hint="start with ok:")

    def reask(self, error: ParseError) -> str:
        return f"again: {error.hint}"

    def cut(self) -> ParseError:
        return ParseError(kind="truncated", hint="shorter")
