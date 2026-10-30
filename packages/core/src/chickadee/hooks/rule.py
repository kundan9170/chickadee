"""The tool points as dispatcher rules: `before_tool` at admit, `after_tool`
at the result stage, before the record. Never a second dispatcher: a hook's verdict
becomes the rule verdict that already means it.

    Deny → Refuse, answered `HOOK_DENIED{hook, reason}. Do: …`
    Ask → AskPerson, through the role's `user` port; a role with none is refused unasked
    Note → Noted, beside the call's answer
    Halt → Stop: the role's tool use ends, and its next turn is its answer

The rules are static members of every role's pipeline (`roles.py`) and read the role's hooks
from the dispatcher's `ScopeState`, which hands them from the context as it hands the file
view's `place`. A role given none, as every test that builds a dispatcher
without them, passes every call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

from pydantic import JsonValue

from chickadee.hooks.failures import HookDenied, HookHalted
from chickadee.hooks.payloads import AfterTool, BeforeTool
from chickadee.hooks.points import HookPoint
from chickadee.hooks.verdicts import Ask, Combined, Deny, Halt
from chickadee.schema.authorization import Tier
from chickadee.shell_line import read_line
from chickadee.tools.catalogue import answer_text, failed
from chickadee.tools.rules import (
    AskPerson,
    CallOutcome,
    Noted,
    Pass,
    Pending,
    Refuse,
    ScopeState,
    Stop,
    Verdict,
)

if TYPE_CHECKING:
    from chickadee.tools.context import ToolContext

__all__ = ["HookAdmit", "HookResult", "tool_payload"]


def tool_payload(pending: Pending, state: ScopeState) -> BeforeTool:
    """What `before_tool` is told of a call, as the dispatcher holds it: each file its tool's
    spec says an argument names (`ToolSpec.paths`), resolved, and a shell line's programs.
    `HookAdmit` keeps it for the call's `after_tool` (`ScopeState.admitted`)."""
    raw = pending.raw
    named: list[str] = []
    for key in pending.spec.paths if pending.spec is not None else ():
        value = raw.get(key)
        if isinstance(value, str):
            named.append(value)
        elif isinstance(value, list):
            named.extend(v for v in cast("list[object]", value) if isinstance(v, str))
    resolved = tuple(
        place.path
        for name in named
        if state.place is not None and (place := state.place(name)) is not None
    )
    command = raw.get("command")
    return BeforeTool(
        tool=pending.tool or pending.sent,
        tool_sent=pending.sent,
        input=cast("dict[str, JsonValue]", dict(raw)),
        resolved=resolved,
        programs=read_line(command).programs if isinstance(command, str) else (),
    )


@dataclass(frozen=True)
class HookAdmit:
    """`before_tool`: the last admit rule; a hook sees only a call that would otherwise run."""

    stage: Literal["admit"] = "admit"

    def check(self, pending: Pending, state: ScopeState) -> Verdict:
        hooks = state.hooks
        if hooks is None or pending.tool is None or not hooks.has(HookPoint.BEFORE_TOOL):
            return Pass()
        payload = tool_payload(pending, state)
        state.admitted[pending.call.id] = payload
        said = hooks.fire(HookPoint.BEFORE_TOOL, payload, call_id=pending.call.id)
        return _admitted(said, pending)


def _admitted(said: Combined, pending: Pending) -> Verdict:
    by = said.by or "hook"
    match said.decided:
        case Halt(reason=reason):
            return Stop(reason, HookHalted(by, reason))
        case Deny(reason=reason, do=do):
            return Refuse(HookDenied(by, reason, do))
        case Ask(question=question):
            tier = pending.spec.tier if pending.spec is not None else Tier.T1
            return AskPerson(question, tier if isinstance(tier, Tier) else Tier.T1)
        case _:
            note = said.note
            return Noted(note) if note is not None else Pass()


@dataclass(frozen=True)
class HookResult:
    """`after_tool`: after the answer is cut to what the role reads, before it is accounted and
    recorded, so a note is both."""

    def apply(self, outcome: CallOutcome, state: ScopeState, ctx: ToolContext) -> CallOutcome:
        del ctx
        hooks = state.hooks
        pending = outcome.pending
        answer = outcome.answer
        admitted = state.admitted.pop(pending.call.id, None)
        if hooks is None or answer is None or pending.tool is None:
            return outcome
        if not hooks.has(HookPoint.AFTER_TOOL):
            return outcome
        before = admitted or tool_payload(pending, state)
        payload = AfterTool(
            tool=before.tool,
            tool_sent=before.tool_sent,
            input=before.input,
            resolved=before.resolved,
            programs=before.programs,
            output=answer_text(outcome),
            failed=failed(answer),
        )
        said = hooks.fire(HookPoint.AFTER_TOOL, payload, call_id=pending.call.id)
        if isinstance(said.decided, Halt):
            reason = said.decided.reason
            outcome = replace(outcome, stop=Stop(reason, HookHalted(said.by or "hook", reason)))
        note = said.note
        if note is not None:
            outcome = replace(outcome, pending=pending.with_verdict(Noted(note)))
        return outcome
