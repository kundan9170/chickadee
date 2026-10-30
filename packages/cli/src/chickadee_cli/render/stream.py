"""A session's frames as lines: one function from a frame to what the
person sees, by verbosity.

**Every record is read as its kind's type** (`chickadee_wire.records.payload`), never as a
dict, so a field the renderer names is one the contract has. A payload that is not its kind's
type is printed as unreadable rather than crashing the REPL: the server and this client
disagree, and the person should see that they do.

**Verbosity** (`/verbose`):

- `q`: the person's and the assistant's words. Questions are always shown, by
  `interaction.py`, whatever the level.
- `normal`: adds a line per tool call and result, episodes, answers, caps, undo, redo and fork.
- `v`: adds progress frames, spend, and every other record by kind, opaque ones named.

**A run's records** (a coder episode's journal, followed by `runs.py`) read by task: at
`normal` the run's start, each plan, each task delegated, reported and reviewed, each step that
changed something, each command, and the end; at `v` the rest. Its questions are left to the
session stream, which carries them too.

**Every call the box makes**, at `normal`: each tool call of Athena's or a worker's, as
a line naming who made it (and the task, for Athena's), the tool and its arguments, and a line
with the first line of what came back; each model turn, with its tokens, cost, stop reason and
the first line of its text. At `v` each preview is shown whole. A record written before
previews existed reads with what it has.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Final

from pydantic import ValidationError

from chickadee_cli.attach import attachment_line
from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import ProgressFrame, RecordFrame
from chickadee_wire.events import Frame
from chickadee_wire.models import (
    ActionGrantedRecord,
    AssistantTurnP,
    Call,
    CapRecord,
    EffectCompletedP,
    EpisodeRecord,
    ExecP,
    ForkRecord,
    HarnessTurnP,
    ModelCallP,
    PlanWrittenP,
    QuestionAnsweredRecord,
    RedoRecord,
    RunEndP,
    RunStartP,
    SpendRecord,
    TaskDelegatedP,
    TaskReportedP,
    TaskReviewedP,
    ToolCallP,
    ToolRecord,
    UndoRecord,
    UserTurnP,
    WireModel,
)
from chickadee_wire.records import is_run_record, payload

__all__ = [
    "LIVE_PROGRESS",
    "RESULT_PREVIEW",
    "SHOWN_APART",
    "TURN_ERROR",
    "Verbosity",
    "call_lines",
    "call_text",
    "clip",
    "frame_lines",
    "preview",
    "user_turn_text",
    "who",
]

RESULT_PREVIEW = 120
"""A tool result's first line, cut here: the record holds the rest, and `/journal` shows it."""
BODY_LINES: Final = 3
"""How many lines of a Zeus tool result's body the default stream shows under its header."""

# `tools/meta.FINDINGS`, spelled out: a surface imports only the facade, and a REPL module loads
# no engine (`test_no_internal_imports`, `test_repl_confined`).
SHOWN_APART: Final = frozenset({"findings"})
"""The argument a call line does not show among the tool's own: Zeus's `findings`, shown in the
turn's. No call carries a purpose line."""
WHY_LINES: Final = 4
"""A coder result's reason lines shown under it when it did not complete; `-v` shows
them all."""
SETTLED_WELL: Final = frozenset({"COMPLETED"})
"""Outcomes with nothing to explain: their result stays one line."""


class Verbosity(StrEnum):
    QUIET = "q"
    NORMAL = "normal"
    VERBOSE = "v"


def user_turn(frame: RecordFrame) -> UserTurnP | None:
    """The session's record of a user turn, when `frame` is one; else None. A payload this
    client cannot read is never the person's turn (`frame_lines` says it is unreadable)."""
    if frame.kind != "turn":
        return None
    try:
        typed = payload(frame)
    except ValidationError:
        return None
    return typed if isinstance(typed, UserTurnP) else None


def user_turn_text(frame: RecordFrame) -> str | None:
    """The person's words, when `frame` is the session's record of a user turn; else None."""
    turn = user_turn(frame)
    return turn.text if turn is not None else None


def attachment_lines(turn: UserTurnP, style: Style) -> list[str]:
    """A line for each file sent with a turn: what its record numbered it."""
    return [
        line
        for a in turn.attachments
        for line in style.lines(attachment_line(a), Tone.DIM, indent="  attached ")
    ]


def frame_lines(frame: Frame, verbosity: Verbosity, style: Style) -> list[str]:
    """What `frame` prints at `verbosity`: nothing, or its lines."""
    if isinstance(frame, ProgressFrame):
        return _progress(frame, verbosity, style)
    return _record(frame, verbosity, style)


TURN_ERROR: Final = "the turn ended on an error"
"""How an `error` frame reads, its reason after it (`chickadee_serve.progress.describe`). A
turn that fails before replying writes no record, so this is said at every verbosity, to a
pipe too: it may be the only sign the turn has ended."""


def _progress(frame: ProgressFrame, verbosity: Verbosity, style: Style) -> list[str]:
    if frame.kind == "error":
        text = frame.payload.get("text")
        return style.lines(f"{TURN_ERROR}: {text}" if text else TURN_ERROR, Tone.BAD)
    # Progress is a hint: shown only when asked for, never the keep-alive, and
    # never to a pipe, where a hint interleaved with records would read as one. A live frame
    # is the live view's to draw (`render/live.py`), never a line of its own here.
    if verbosity is not Verbosity.VERBOSE or frame.kind in _QUIET_PROGRESS or not style.tty:
        return []
    tool = frame.payload.get("tool")
    return style.lines(f"… {frame.kind}" + (f" {tool}" if tool else ""), Tone.DIM, indent="  ")


LIVE_PROGRESS: Final = frozenset({"text", "thought", "tool_started"})
"""A reply as it streams, of any role: what the live view draws (`render/live.py`)."""
_QUIET_PROGRESS: Final = frozenset({"alive", "context", *LIVE_PROGRESS})
"""Progress no line is printed for: the keep-alive, what the live view draws, and Zeus's
context, which the status line draws."""

JOB_EXITED: Final = 8
"""The trigger for a job's exit, as the wire carries it."""
RUN_ENDED: Final = 4
"""The trigger for a run's end with no `coder` call waiting on it: a
run handed over in the background, or resumed after a pause. Its `handle` names the run."""


def _record(frame: RecordFrame, verbosity: Verbosity, style: Style) -> list[str]:
    if frame.kind == "question_asked":
        return []  # `interaction.py` asks it, at every verbosity
    try:
        typed = payload(frame)
    except ValidationError:
        return style.lines(f"[{frame.kind}: a payload this client cannot read]", Tone.BAD)
    if is_run_record(frame):
        return _run(frame, typed, verbosity, style)
    return _session(frame, typed, verbosity, style)


def _user(turn: UserTurnP, style: Style) -> list[str]:
    """The person's words, and a line for each file sent with them."""
    return [*style.lines(turn.text, Tone.PERSON, indent="> "), *attachment_lines(turn, style)]


def _session(
    frame: RecordFrame, typed: WireModel | None, verbosity: Verbosity, style: Style
) -> list[str]:
    normal = verbosity is not Verbosity.QUIET
    match typed:
        case UserTurnP():
            return _user(typed, style)
        case AssistantTurnP():
            return _assistant(typed, normal, style, whole=verbosity is Verbosity.VERBOSE)
        case HarnessTurnP() if typed.trigger == JOB_EXITED:
            # A job's exit, at every verbosity: a reply after it with no
            # words of the person's between is the turn it woke.
            return style.lines(typed.text, Tone.ASK, indent="  [job] ")
        case HarnessTurnP() if normal:
            return style.lines(typed.text, Tone.DIM, indent="  [harness] ")
        case ToolRecord() if normal:
            return _tool(typed, style, every=verbosity is Verbosity.VERBOSE)
        case QuestionAnsweredRecord() if normal:
            return _answered(typed, style)
        case EpisodeRecord() if normal:
            return style.lines(f"episode {typed.run_id}", Tone.DIM, indent="  ")
        case CapRecord() if normal:
            return style.lines(_cap(typed), Tone.DIM, indent="  ")
        case UndoRecord() if normal:
            what = _halves(typed.code, typed.conversation)
            return style.lines(
                f"undone from turn {typed.to_turn}: {what}", Tone.DIM, indent="  "
            )
        case RedoRecord() if normal:
            what = _halves(typed.code, typed.conversation)
            return style.lines(f"redone: {what}", Tone.DIM, indent="  ")
        case ForkRecord() if normal:
            return style.lines(f"forked: {typed.reason}", Tone.DIM, indent="  ")
        case ActionGrantedRecord() if normal:
            # Approved by a standing grant, so never asked and never a prompt;
            # or by the person's `sandbox.never_ask`, which is no grant.
            text = (
                f"{typed.display} — run unasked (sandbox.never_ask)"
                if typed.scope == "never_ask"
                else f"{typed.display} — approved by your {typed.scope} grant"
            )
            return style.lines(text, Tone.DIM, indent="  ")
        case SpendRecord() if verbosity is Verbosity.VERBOSE:
            text = (
                f"spend ${typed.cost_usd:.4f} on {typed.rung} ({typed.model}): "
                f"{typed.tokens_in} in, {typed.tokens_out} out"
                + (", estimated" if typed.estimated else "")
            )
            return style.lines(text, Tone.DIM, indent="  ")
        case _:
            if verbosity is Verbosity.VERBOSE:
                return style.lines(f"[{frame.kind}]", Tone.DIM, indent="  ")
            return []


def _run(
    frame: RecordFrame, typed: WireModel | None, verbosity: Verbosity, style: Style
) -> list[str]:
    """One record of a run's journal, as a line by task."""
    if frame.kind == "question_answered" or verbosity is Verbosity.QUIET:
        return []
    box = _box(typed, frame.task_id, verbosity, style)
    if box is not None:
        return box
    task = _task_lines(typed, verbosity, style)
    if task is not None:
        return task
    line = _run_line(frame, typed)
    if line is None:
        return (
            style.lines(f"[{frame.kind}]", Tone.DIM, indent="    ")
            if (verbosity is Verbosity.VERBOSE)
            else []
        )
    text, tone = line
    return style.lines(text, tone)


def _run_line(frame: RecordFrame, typed: WireModel | None) -> tuple[str, Tone] | None:
    match typed:
        case RunStartP():
            cap = f"cap ${typed.hard_cost_cap_usd:.2f}"
            return f"run {frame.scope}: {typed.goal} · {typed.isolation} · {cap}", Tone.PLAIN
        case ExecP():
            late = " (timed out)" if typed.timed_out else ""
            return f"    ran {typed.label}: exit {typed.exit_code}{late}", Tone.DIM
        case RunEndP():
            why = f" · {typed.why}" if typed.why else ""
            return (
                f"run {frame.scope} ended {typed.outcome} · {typed.committed}/{typed.tasks} "
                f"task(s) committed · ${typed.cost_usd:.4f} · {typed.wall_ms / 1000:.1f}s{why}",
                Tone.GOOD if typed.outcome == "COMPLETED" else Tone.BAD,
            )
        case _:
            return None


# -- the task loop ---------------------------------------------------------

STEP_PREFIX: Final = "step:"
"""How a box step's operation id begins (`chickadee.ids.step_id`): an edit or a command of the
worker's or Athena's, recorded and undoable."""


def _task_lines(
    typed: WireModel | None, verbosity: Verbosity, style: Style
) -> list[str] | None:
    """A task loop's record (a plan written, a task delegated, reported or reviewed, a step),
    as lines; None for any other record."""
    line: tuple[str, Tone] | None
    match typed:
        case PlanWrittenP():
            line = _plan(typed)
        case TaskDelegatedP():
            again = f" · re-cut, depth {typed.depth}" if typed.depth else ""
            line = (
                f"> {typed.task_id} delegated by {typed.by}: {typed.title}{again}",
                Tone.PLAIN,
            )
        case TaskReportedP():
            line = _reported(typed)
        case TaskReviewedP():
            tone = Tone.BAD if typed.answer in ("drop", "abandon", "re_cut") else Tone.PLAIN
            line = f"  {typed.task_id} reviewed: {typed.answer}", tone
        case EffectCompletedP() if typed.operation_id.startswith(STEP_PREFIX):
            line = _step(typed, verbose=verbosity is Verbosity.VERBOSE)
        case _:
            return None
    return [] if line is None else style.lines(*line)


def _plan(typed: PlanWrittenP) -> tuple[str, Tone]:
    if typed.outcome == "cancelled":
        return f"plan v{typed.version} by {typed.by} cancelled at stop 2", Tone.BAD
    asked = "approved at stop 2" if typed.asked else "a revision, shown and not asked"
    return f"plan v{typed.version} by {typed.by} · {asked}", Tone.PLAIN


def _reported(typed: TaskReportedP) -> tuple[str, Tone]:
    facts = typed.facts
    ran = len(facts.commands)
    parts = [f"{len(facts.files)} file(s)", *([f"{ran} command(s)"] if ran else [])]
    tests = [c.test_outcome for c in facts.commands if c.test_outcome is not None]
    if tests:
        parts.append(f"tests {tests[-1]}")
    tone = Tone.GOOD if typed.stopped == "replied" else Tone.BAD
    return (
        f"  {typed.task_id} round {typed.round} {typed.stopped} · {' · '.join(parts)}",
        tone,
    )


def _step(typed: EffectCompletedP, *, verbose: bool) -> tuple[str, Tone] | None:
    """A step completed: what it changed. One that changed nothing is shown only verbose."""
    changed = typed.observed_surface
    if not changed and not verbose:
        return None
    what = ", ".join(changed) if changed else "nothing changed"
    tone = Tone.DIM if typed.outcome == "succeeded" else Tone.BAD
    return f"    step {typed.outcome}: {what}", tone


# -- the box's calls ----------------------------------------------------------------

_CALL: Final = "    "
_RESULT: Final = "        "


def who(role: str, task_id: str | None) -> str:
    """Who made a call of the box's, or is speaking live: `worker (T1)`, `athena (T2)`,
    `zeus`. One label, so a live line and its record name the same speaker."""
    return f"{role} ({task_id})" if task_id else role


def _box(
    typed: WireModel | None, task_id: str | None, verbosity: Verbosity, style: Style
) -> list[str] | None:
    """A call the box made, as lines; None for any other record."""
    whole = verbosity is Verbosity.VERBOSE
    match typed:
        case ToolCallP():
            return _call(
                style,
                who(typed.role, task_id),
                typed.tool,
                typed.args_preview,
                typed.result_preview,
                typed.denied_reason,
                typed.result_tokens,
                whole=whole,
            )
        case ModelCallP():
            asked = f" → {', '.join(typed.tools)}" if typed.tools else ""
            head = (
                f"{who(typed.role, task_id)} · {typed.request} · "
                f"{_tokens(typed.tokens_in, typed.tokens_out)} · ${typed.cost_usd:.4f} · "
                f"{typed.stop_reason}{asked}"
            )
            lines = _turn(style, head, typed.text_preview, whole=whole)
            if typed.refusal:
                # Why her answer was asked again, or why the request ended.
                said = preview(typed.refusal, whole=whole)
                lines += style.lines(f"✗ re-asked: {said}", Tone.BAD, indent=_RESULT)
            return lines
        case _:
            return None


def _call(
    style: Style,
    who: str,
    tool: str,
    args: str | None,
    result: str | None,
    denied: str | None,
    result_tokens: int,
    *,
    whole: bool,
) -> list[str]:
    asked = f" {preview(args, whole=whole)}" if args else ""
    lines = style.lines(f"{who} → {tool}{asked}", Tone.PLAIN, indent=_CALL)
    if denied is not None:
        return lines + style.lines(
            f"✗ refused: {preview(result or denied, whole=whole)}", Tone.BAD, indent=_RESULT
        )
    if result is None:
        return lines + style.lines(f"← {result_tokens} tokens", Tone.DIM, indent=_RESULT)
    return lines + style.lines(f"← {preview(result, whole=whole)}", Tone.DIM, indent=_RESULT)


def _turn(style: Style, head: str, text: str | None, *, whole: bool) -> list[str]:
    lines = style.lines(head, Tone.DIM, indent=_CALL)
    if text:
        lines += style.lines(f"“{preview(text, whole=whole)}”", Tone.DIM, indent=_RESULT)
    return lines


def preview(text: str, *, whole: bool) -> str:
    """A preview whole, or collapsed to one line and then clipped: a JSON answer or a numbered
    read reads as what it holds, not as its first line (`{`). Public so every client
    surface clips a preview as the REPL does (the TUI)."""
    return text if whole else clip(" ".join(text.split()))


def clip(text: str) -> str:
    """One line cut at `RESULT_PREVIEW`, its cut marked with `…` (invariant 16)."""
    return text if len(text) <= RESULT_PREVIEW else text[: RESULT_PREVIEW - 1] + "…"


def _tokens(tokens_in: int, tokens_out: int) -> str:
    return f"{tokens_in:,} in / {tokens_out:,} out"


def _assistant(
    turn: AssistantTurnP, normal: bool, style: Style, *, whole: bool = False
) -> list[str]:
    lines = style.lines(turn.text) if turn.text else []
    if normal:
        lines += call_lines(turn, style, whole=whole)
    return lines


def call_text(call: Call, *, whole: bool = False) -> str:
    """One of Zeus's calls as asked: its name, then the arguments as sent, as the
    session line holds them, less the findings shown apart. Public so every client surface
    names a call as the REPL does (the TUI)."""
    own = {k: v for k, v in call.arguments.items() if k not in SHOWN_APART}
    text = json.dumps(own, sort_keys=True, ensure_ascii=False, default=str)
    return f"{call.name} {preview(text, whole=whole)}" if own else call.name


def call_lines(turn: AssistantTurnP, style: Style, *, whole: bool = False) -> list[str]:
    """A turn's calls, one line each: what the stream shows at `normal` under the turn's
    words."""
    return [
        line
        for call in turn.calls
        for line in style.lines(f"→ {call_text(call, whole=whole)}", Tone.DIM, indent="  ")
    ]


def _tool(record: ToolRecord, style: Style, *, every: bool = False) -> list[str]:
    head, *body = record.text.splitlines() if record.text else [""]
    ran = "" if record.ran else " (not run)"
    lines = style.lines(f"← {clip(head)}{ran}", Tone.DIM, indent="  ")
    # The header says what came back; the first lines of the body show it. `-v` shows
    # the body whole, as it shows every preview whole. A `coder` result's body is its
    # report, which its reasons below already stand for.
    body = body if record.outcome is None else []
    while body and not body[-1].strip():
        body.pop()
    shown = body if every else [clip(line) for line in body[:BODY_LINES]]
    for line in shown:
        lines += style.lines(line, Tone.DIM, indent="     ")
    if len(body) > len(shown):
        lines += style.lines(f"… {len(body) - len(shown)} more lines", Tone.DIM, indent="     ")
    told = record.outcome
    if told is None or told.terminal in SETTLED_WELL or not told.why:
        return lines
    # A coder result that did not complete says why, from its typed fields: the
    # person sees the reason Zeus read, rather than a terminal they must guess behind.
    # `-v` shows every line, as it shows every preview whole.
    shown = told.why if every else told.why[:WHY_LINES]
    for why in shown:
        lines += style.lines(why, Tone.DIM, indent="     ")
    if len(told.why) > len(shown):
        more = len(told.why) - WHY_LINES
        lines += style.lines(
            f"… {more} more; /status shows the board", Tone.DIM, indent="     "
        )
    return lines


def _answered(record: QuestionAnsweredRecord, style: Style) -> list[str]:
    if record.source == "withdrawn":
        why = f": {record.note}" if record.note else ""
        return style.lines(f"question {record.id} withdrawn{why}", Tone.DIM, indent="  ")
    note = f" ({record.note})" if record.note else ""
    text = f"answered {record.id}: {record.choice}{note}, by {record.source}"
    return style.lines(text, Tone.DIM, indent="  ")


def _cap(record: CapRecord) -> str:
    whose = f"{record.rung}'s cap" if record.rung else "the cap"
    if record.hard_cost_usd is None:
        return f"{whose} lifted for this session family"
    return f"{whose} set to ${record.hard_cost_usd}"


def _halves(code: bool, conversation: bool) -> str:
    if code and conversation:
        return "code and conversation"
    return "code only" if code else "conversation only"
