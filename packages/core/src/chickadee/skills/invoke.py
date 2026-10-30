"""Invocation: the person loads a skill by name, with arguments.

**Exactly loading plus an argument.** `invoke` calls `Skills.load`, so an invoked skill passes
the same size refusal, the same pin budget and the same trust gate as one the model selects. A
second path into context would come to disagree with the first about one of them.

**The arguments are the person's words**: framed apart from the body, which
is its author's, never merged into one block.

The `/<skill> [args]` command that calls this is the CLI's.
"""

from __future__ import annotations

from dataclasses import dataclass

from chickadee.context.results import InvalidArgument
from chickadee.skills.frame import frame_arguments
from chickadee.skills.load import Skills

__all__ = ["Invoked", "invoke"]


@dataclass(frozen=True)
class Invoked:
    body: str
    """The load's answer: the framed body, or that it was already pinned."""
    arguments: str | None
    """The person's arguments, framed as theirs; None when they gave none."""

    def render(self) -> str:
        return self.body if self.arguments is None else f"{self.body}\n\n{self.arguments}"


def invoke(
    skills: Skills, name: str, arguments: str, *, call_id: str
) -> Invoked | InvalidArgument:
    loaded = skills.load(name, call_id)
    if isinstance(loaded, InvalidArgument):
        return loaded
    given = arguments.strip()
    return Invoked(loaded, frame_arguments(given) if given else None)
