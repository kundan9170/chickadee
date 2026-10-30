"""A script's input and output: Claude Code's subset, so a script written for
it ports, with what Chickadee adds beside it.

**Input**, JSON on stdin: the envelope (`ScriptEnvelope`: Claude Code's names where it has one,
then ours) and the point's payload fields beside it, every string redacted, as the
callbacks' frozen copy was. One model per point (`input_model`), exported as a JSON Schema
golden (`hooks/export.py`), so a script's author has a contract.

**Output**: the exit code, then JSON on stdout, read through `ScriptOutput`, which forbids what
it does not name, so a typo is a hook error rather than a silence:

    0, no output Proceed
    2 the point's blocking verdict, stderr its words (`PointSpec.blocking`)
    anything else a hook error, handled by its `on_error`

    {"continue": false, "stopReason": …} Halt
    {"decision": "block", "reason": …} exit 2
    {"hookSpecificOutput": {"permissionDecision":
        "deny" | "ask", "permissionDecisionReason": …}} Deny, Ask; "allow" is Proceed, never
                                                        an approval
    {"hookSpecificOutput": {"additionalContext": …}} Note
    {"systemMessage": …} Warn
    {"retry_task": {"text": …, "fresh": …}} RetryTask (ours, at round_ended)
    {"do": …} the next action beside a deny (ours)

One output is one verdict, the strongest it names. `updatedInput` and `updatedToolOutput` are
refused by name: a call is never corrected and run, and an answer never replaced.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, create_model

from chickadee.hooks.fire import HookError
from chickadee.hooks.payloads import (
    PAYLOADS,
    AfterTool,
    BeforeStop,
    BeforeTool,
    HookEvent,
    HookPayload,
)
from chickadee.hooks.points import HookPoint, blocking, spec_of
from chickadee.hooks.verdicts import (
    Ask,
    Deny,
    Halt,
    HookVerdict,
    Note,
    Proceed,
    RetryTask,
    Warn,
)
from chickadee.schema.journal import Fresh
from chickadee.secrets.redact import redact_text

__all__ = ["ScriptEnvelope", "ScriptOutput", "from_output", "input_model", "to_input"]

REFUSED: Final[Mapping[str, str]] = {
    "updatedInput": "a call is never corrected and run (D101): deny it and say what to send",
    "updatedToolOutput": "an answer is never replaced (D50): add a note beside it instead",
}


class ScriptEnvelope(BaseModel):
    """What every point's input carries. `hook_event_name` is Claude Code's name for the point
    where it has one; `chickadee_event` is always ours."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hook_event_name: str
    chickadee_event: HookPoint
    session_id: str | None
    cwd: str
    role: str
    model: str
    run_id: str | None
    task_id: str | None
    round: int | None
    tool_name: str | None = None
    tool_input: dict[str, JsonValue] | None = None
    tool_response: str | None = None
    tool_name_sent: str | None = None
    stop_hook_active: bool | None = None
    last_assistant_message: str | None = None


@cache
def input_model(point: HookPoint) -> type[BaseModel]:
    """`point`'s input: the envelope, and its payload's fields beside it."""
    payload = PAYLOADS[point]
    name = "".join(w.capitalize() for w in point.value.split("_")) + "Input"
    return create_model(name, __base__=(ScriptEnvelope, payload))


def to_input(event: HookEvent, cwd: Path) -> bytes:
    """`event` as a script reads it on stdin, every string redacted."""
    payload = event.payload
    claude = _claude_fields(payload)
    body: dict[str, JsonValue] = {
        "hook_event_name": spec_of(event.point).claude or event.point.value,
        "chickadee_event": event.point.value,
        "session_id": event.session or event.run_id,
        "cwd": str(cwd),
        "role": event.role,
        "model": event.model,
        "run_id": event.run_id,
        "task_id": event.task_id,
        "round": event.round,
        **claude,
        **json.loads(payload.model_dump_json()),
    }
    return json.dumps(_redacted(body), sort_keys=True).encode("utf-8")


def _claude_fields(payload: HookPayload) -> dict[str, JsonValue]:
    if isinstance(payload, AfterTool):
        return {
            "tool_name": payload.tool,
            "tool_input": payload.input,
            "tool_name_sent": payload.tool_sent,
            "tool_response": payload.output,
        }
    if isinstance(payload, BeforeTool):
        return {
            "tool_name": payload.tool,
            "tool_input": payload.input,
            "tool_name_sent": payload.tool_sent,
        }
    if isinstance(payload, BeforeStop):
        return {
            "stop_hook_active": payload.continues_used > 0,
            "last_assistant_message": payload.reply,
        }
    return {}


def _redacted(value: JsonValue) -> JsonValue:
    """A secret never reaches a script: every string, keys included, redacted."""
    if isinstance(value, str):
        return redact_text(value)[0]
    if isinstance(value, list):
        return [_redacted(v) for v in value]
    if isinstance(value, dict):
        return {redact_text(k)[0]: _redacted(v) for k, v in value.items()}
    return value


class SpecificOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hookEventName: str | None = None
    permissionDecision: Literal["allow", "deny", "ask"] | None = None
    permissionDecisionReason: str | None = None
    additionalContext: str | None = None


class RetryOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    fresh: Fresh = True


class ScriptOutput(BaseModel):
    """A script's JSON answer, every key it may write."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    continue_: bool = Field(default=True, alias="continue")
    stopReason: str | None = None
    decision: Literal["block", "approve"] | None = None
    reason: str | None = None
    hookSpecificOutput: SpecificOutput | None = None
    systemMessage: str | None = None
    suppressOutput: bool | None = None
    do: str | None = None
    retry_task: RetryOutput | None = None


def from_output(point: HookPoint, exit_code: int, stdout: str, stderr: str) -> HookVerdict:
    """A script's verdict, or `HookError` saying why it cannot be read."""
    if exit_code == 2:
        words = stderr.strip() or stdout.strip()
        if not words:
            raise HookError("exited 2 with nothing on stderr to say why")
        return blocking(point, words, None)
    if exit_code != 0:
        tail = (stderr.strip() or stdout.strip())[-500:]
        raise HookError(f"exited {exit_code}" + (f": {tail}" if tail else ""))
    text = stdout.strip()
    if not text.startswith("{"):
        return Proceed()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HookError(f"its stdout starts as JSON and does not parse: {exc}") from None
    if not isinstance(raw, dict):
        raise HookError("its JSON answer is not an object")
    found = cast("dict[str, object]", raw)
    specific = found.get("hookSpecificOutput")
    for key, why in REFUSED.items():
        if key in found or (isinstance(specific, dict) and key in specific):
            raise HookError(f"`{key}` is not taken: {why}")
    try:
        out = ScriptOutput.model_validate(found)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        where = ".".join(str(p) for p in first["loc"])
        raise HookError(f"its JSON answer: {where}: {first['msg']}") from None
    return _verdict(point, out)


def _verdict(point: HookPoint, out: ScriptOutput) -> HookVerdict:
    specific = out.hookSpecificOutput or SpecificOutput()
    if not out.continue_:
        return Halt(out.stopReason or out.reason or "a hook halted")
    if out.decision == "block":
        return blocking(point, out.reason or "a hook refused", out.do)
    if specific.permissionDecision == "deny":
        reason = specific.permissionDecisionReason or out.reason or "a hook refused"
        return Deny(reason, out.do)
    if specific.permissionDecision == "ask":
        return Ask(specific.permissionDecisionReason or out.reason or "a hook asks first")
    if out.retry_task is not None:
        return RetryTask(out.retry_task.text, out.retry_task.fresh)
    if specific.additionalContext:
        return Note(specific.additionalContext)
    if out.systemMessage:
        return Warn(out.systemMessage)
    return Proceed()
