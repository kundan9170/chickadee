"""What a hook's refusal and halt say to the model (D221's form): the
two failures a hook adds, beside the dispatcher's own. Their own module, so a role that says
one (Zeus, the worker) and the rules that make one (`hooks/rule.py`, `hooks/fire.py`) share
the words without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chickadee.context.results import instructed

__all__ = ["HookDenied", "HookHalted"]


@dataclass(frozen=True)
class HookDenied:
    """`HOOK_DENIED{hook, reason}`: a hook refused the call; it never ran."""

    hook: str
    reason: str
    do: str | None = None
    failure: Literal["HOOK_DENIED"] = "HOOK_DENIED"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{hook: {self.hook}, reason: {self.reason!r}}}",
            self.do or "do not send this call again; go on another way",
        )


@dataclass(frozen=True)
class HookHalted:
    """`HOOK_HALTED{hook, reason}`: a hook ended the role's tool use."""

    hook: str
    reason: str
    failure: Literal["HOOK_HALTED"] = "HOOK_HALTED"

    def render(self) -> str:
        return instructed(
            f"{self.failure}{{hook: {self.hook}, reason: {self.reason!r}}}",
            "stop calling tools: give your answer or your report now, from what you have",
        )
