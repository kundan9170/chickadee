"""A tool is a spec bound to a run function.

`run` is thin: it reads its arguments through `Arguments` and calls the ports its tool
needs. It never opens a file, a database, a socket or a process itself; that is what the
ports are for, and what keeps data policy out of a tool's reach (decision 3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Protocol, cast, get_args

from chickadee.context.result_data import ResultData, Structured
from chickadee.context.results import CoderOutcome, ToolFailure, ToolResult, instructed
from chickadee.errors import HarnessBug
from chickadee.tools.spec import ToolSpec
from chickadee.tools.validate import Arguments

if TYPE_CHECKING:
    from chickadee.tools.context import ToolContext

__all__ = [
    "PORTS",
    "Answer",
    "CallTimedOut",
    "Defaulted",
    "Failure",
    "Invocation",
    "Port",
    "Said",
    "TextAnswer",
    "Tool",
    "as_text",
    "noted",
    "streams_only",
    "unit_of",
]

Port = Literal[
    "files",
    "navigator",
    "graph",
    "handles",
    "exec",
    "exec_request",
    "mcp",
    "mcp_request",
    "journal",
    "notes",
    "user",
    "box",
    "locator",
    "search",
    "catalog",
    "approval",
    "edits",
    "skills",
    "design_notes",
    "desk",
    "finish",
    "docs",
    "jobs",
    "worktree",
    "agents",
    "web",
    "team",
]
"""What a tool may touch. A new port is a decision, because a port is a trust surface;
`approval` is D165's, `skills` D183's, `edits` D196's (the one write path
every edit tool uses, the box's worktree; D169's `packet` retired with the packet loop), `desk`
D196's too (Athena's plan, delegations and reviews, `box/desk.TaskDesk`); `finish`
T-3.159's, on the owner's word: where the worker's `finish` hands over its report, so a model
that ends its turns only by calling a tool can end its task; `docs` D204's (decision 9): the
edit port scoped to the caller's part of the session's work folder, for `doc`; `jobs` D216's:
Zeus's jobs, which the `job` tool lists, tails, waits on and kills;
`worktree` D230's (decision 8): Zeus moving the session between the person's checkout and a
worktree, and landing or discarding it, each landing or discard asked of the person; `agents`
D213's: the agents Zeus spawns, each on a profile no wider than his own ceiling; `web` D232's:
pages and searches through the external-call policy, the researcher's alone; `team` D240's:
Zeus pausing the session's run, which goes on in the background."""

PORTS: frozenset[Port] = frozenset(cast("tuple[Port, ...]", get_args(Port)))


class Failure(Protocol):
    """A typed failure: a name from the vocabulary and its fields, which
    `render()` writes as the one line the model reads. Never prose. How a role shows one is
    its sink's; `render()` is the default."""

    @property
    def failure(self) -> str: ...

    def render(self) -> str: ...


@dataclass(frozen=True)
class Invocation:
    """What a role knows about one call."""

    id: str
    """The provider's call id, or the role's own."""
    operation_id: str
    """D153's stable id, for a tool whose run is an effect."""
    at: int
    """The role's sequence position (Zeus's record tools)."""


@dataclass(frozen=True)
class TextAnswer:
    """A raw answer, bounded by the role's result rules (`Bound`), not by its producer: a
    navigation answer is text Zeus caps and stores and the scout renders whole."""

    text: str
    unit: Literal["bytes", "lines"]
    data: ResultData | None = None
    """What the text was made from, for the worker's Python blocks. Any
    result rule that changes the text leaves it as it was; the dispatcher then narrows it
    (`dispatch._data_as_shown`), so it never holds more than the text shows. An answer built
    outside `Dispatcher` must do the same."""

    def render(self) -> str:
        return self.text


@dataclass(frozen=True)
class CallTimedOut:
    """`TIMED_OUT{tool, seconds}` (33 §5): the `Timeout` wrapper's clock ran out and the
    tool, told through the cancel event, stopped."""

    tool: str
    seconds: float
    failure: Literal["TIMED_OUT"] = "TIMED_OUT"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{tool: {self.tool}, seconds: {self.seconds:g}}}",
            "do not send the same call again: ask for less (a narrower path, query or range)",
        )


@dataclass(frozen=True)
class Said:
    """A short answer its producer already sized, shown as it is and never bounded again: a
    record tool's receipt, a person's own words, a delegation's report."""

    text: str
    outcome: CoderOutcome | None = None
    """A delegation's terminal and why, for its `tool` line."""

    def render(self) -> str:
        return self.text


@dataclass(frozen=True)
class Defaulted:
    """An answer nobody gave at this point: a question's stated default, or an earlier answer
    reused. Shown as it is, and recorded as not run, so it is never taken for
    something the person said."""

    text: str

    def render(self) -> str:
        return self.text


Answer = ToolResult | TextAnswer | Said | Defaulted | Failure
"""What a run returns. A `ToolResult` is already bounded by whoever produced it (the Reader's
reads); a `TextAnswer` is bounded by the role's rules; a `Said` or `Defaulted` was sized by
its producer; a failure is typed, never prose."""


def noted(answer: Answer, note: str) -> Answer:
    """`answer` with `note` after it; a failure is left as it is, since what the call did is
    said by its record (a step's `observed`, a write's completion)."""
    if isinstance(answer, ToolResult):
        return replace(answer, notes=(*answer.notes, note))
    if isinstance(answer, TextAnswer | Said | Defaulted):
        return replace(answer, text=f"{answer.text}\n{note}")
    return answer


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    needs: frozenset[Port]
    """The context ports `run` uses. A role that cannot be given one cannot enable the tool,
    and a scope that lacks one does not offer it (decision 9)."""
    run: Callable[[Invocation, Arguments, ToolContext], Answer]
    streams: bool = False
    """It yields before it finishes (Zeus's `coder`: the delegation, then the report), so
    only its role's executor runs it (`RoleExecutors`), and a dispatcher
    with none for it does not offer it. `run` is then never called."""

    def __post_init__(self) -> None:
        unknown = self.needs - PORTS
        if unknown:
            raise ValueError(f"tool {self.spec.name!r} needs unknown ports {sorted(unknown)}")


def unit_of(spec: ToolSpec) -> Literal["bytes", "lines"]:
    """The unit a `TextAnswer` answer of `spec` is bounded in: lines for a line-bounded tool
    (`find`, `grep_in`, `tools`), bytes for every other."""
    return "lines" if spec.bound.kind == "inline_lines" else "bytes"


def as_text(spec: ToolSpec, answer: str | ToolFailure) -> TextAnswer | ToolFailure:
    """A raw answer as `TextAnswer` in `spec`'s unit, bounded later by the role's result rules;
    a failure as it is."""
    if isinstance(answer, Structured):
        return TextAnswer(str(answer), unit_of(spec), answer.data)
    return TextAnswer(answer, unit_of(spec)) if isinstance(answer, str) else answer


def streams_only(call: Invocation, args: Arguments, ctx: ToolContext) -> Answer:
    """The `run` of a streaming tool: never called, because its role's executor runs it."""
    raise HarnessBug(f"call {call.id} reached a streaming tool's run; its role runs it")
