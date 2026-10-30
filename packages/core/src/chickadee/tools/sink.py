"""How a role shows, records and replays a call.

A sink keeps each role's own bytes and records: Zeus's session-log tool line, the scout's
`scout_tool`, the worker's `coder_tool`. One typed failure can therefore keep each role's
rendering (`UNKNOWN_TOOL{name, nearest}` for Zeus, `{name, offered}` for the scout), and the
dispatcher stays generic over what a loop gets back (`R`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol

from chickadee.context.results import ToolResult
from chickadee.tools.rules import CallOutcome, Level, Pending
from chickadee.tools.spec import ToolSpec
from chickadee.tools.tool import Answer, Invocation

__all__ = ["NO_EXECUTORS", "Counters", "RoleExecutors", "ToolSink", "no_hold"]


@dataclass(frozen=True)
class Counters:
    """Where a reopened scope's counters start: a projection of the
    role's records, because the journal is the single source of truth."""

    calls: int = 0
    result_tokens: int = 0


class ToolSink[R](Protocol):
    """A role's half of a call: what it holds while one runs, what the model sees, what it
    records, and what a resumed scope starts from."""

    def executing(self, spec: ToolSpec) -> AbstractContextManager[object]:
        """What the role holds while a tool runs and is recorded: Zeus's session lock for its
        read family; nothing for every other."""
        ...

    def render(self, outcome: CallOutcome) -> str | ToolResult:
        """The bytes the model sees."""
        ...

    def record(self, call: Invocation, outcome: CallOutcome) -> R:
        """The role's own record, written once; what the loop gets back."""
        ...

    def replay(self, level: Level) -> Counters:
        """The counters a reopened `level` scope starts from. A role that restarts at zero
        today returns `Counters()`, and its pin records it."""
        ...


def no_hold(spec: ToolSpec) -> AbstractContextManager[object]:
    """`executing` for a role that holds nothing while a tool runs."""
    del spec
    return nullcontext()


RoleExecutors = Mapping[str, Callable[[Pending], Answer]]
"""A streaming tool's executor, by registry name: Zeus's `coder` opens an
episode, yields a delegation, and finishes later, so its loop runs it between `admit` and
the sink's `record`. Any future tool that streams takes the same seam."""

NO_EXECUTORS: RoleExecutors = {}
