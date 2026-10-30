"""The card every pane draws a call in (the box
roles included): a rounded box, its title naming what ran and its right how it ended; its body
the
calls, a command whole and wrapped, never cut; its foot, under a rule, the result.

- **Zeus's calls** (`turn_cards`): a card per command or per run of other calls, its right the
  approval's tick.
- **The Planner's and the Worker's** (`box_cards`): a card per call with the step and the
  commands it made; an edit is its files and the lines each gained and lost (the server's
  step diff), never its text; a review is its verdict.

**The tick says how it ended, never who** (`tick`): `✔ approved once`, `✖ denied`,
`waiting for you` while the modal asks. An approval of another role's (the worker's, Athena's)
names no call of Zeus's and has a card of its own where its answer lands (`approval_card`).

What a card says is a plain function of the records (`turn_cards`, `approval_card`), so it is
tested without a terminal; `Card` only places it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.events import Resize
from textual.widgets import Static

from chickadee_cli.render.stream import SHOWN_APART, Verbosity, clip, frame_lines, preview
from chickadee_cli.render.style import Style
from chickadee_cli.tui.model import (
    Approval,
    Asked,
    AthenaEntry,
    WorkerEntry,
    ZeusEntry,
    ZeusPane,
)
from chickadee_cli.tui.panes import border_title
from chickadee_cli.tui.theme import UNICODE
from chickadee_wire.models import (
    AssistantTurnP,
    Call,
    EffectCompletedP,
    ExecP,
    HookP,
    PlanTaskP,
    RunDiff,
    TaskReviewedP,
    ToolCallP,
    ToolRecord,
)

__all__ = [
    "Card",
    "CardText",
    "approval_card",
    "box_cards",
    "failed",
    "hook_card",
    "hook_tick",
    "show_cards",
    "tick",
    "turn_cards",
]

_TERMINAL: Final = Style(tty=True)
_COMMAND: Final = "command"
"""The argument that makes a call a command: drawn whole, on a card of its own."""
_EXIT: Final = re.compile(r"exit (-?\d+)\b")
"""The shell's result header leads with `exit N` (`shell._header`)."""


@dataclass(frozen=True)
class CardText:
    title: Content
    tick: Content
    """The title's right: how the approval ended, or nothing when none was asked."""
    body: Content
    foot: Content | None
    """Each call's result under a rule; None for a card with no result to say."""
    live: bool = False
    """A step under way, which no call has closed yet: its border is lit."""
    marks: frozenset[int] = frozenset()
    """The records it shows, by seq: a pane's chosen row marks the card that holds it."""


def tick(approval: Approval | None, colours: dict[str, str]) -> Content:
    """How an approval ended, as its card's title says it; nothing when none was asked."""
    if approval is None:
        return Content("")
    ok, bad, dim = colours["worker"], colours["bad"], colours["dim"]
    match approval.choice, approval.source:
        case None, _:
            return Content.styled("waiting for you", colours["zeus"])
        case _, "withdrawn":
            return Content.styled("withdrawn", dim)
        case "approve", "grant":
            return Content.styled("✔ allowed by a grant", ok)
        case "approve", "never_ask":
            return Content.styled("✔ never asked", ok)
        case "approve", "default":
            return Content.styled("✔ approved by default", ok)
        case "approve", "preference":
            return Content.styled("✔ approved by preference", ok)
        case "approve", _ if approval.scope == "session":
            return Content.styled("✔ approved for this session", ok)
        case "approve", _ if approval.scope == "repo":
            return Content.styled("✔ approved for this repo", ok)
        case "approve", _:
            return Content.styled("✔ approved once", ok)
        case "deny", "default":
            return Content.styled("✖ denied · no answer", bad)
        case "deny", _:
            return Content.styled("✖ denied", bad)
        case "abort", _:
            return Content.styled("✖ stopped", bad)
        case choice, _:
            return Content.styled(choice, dim)


def failed(result: ToolRecord) -> bool:
    """A result that did not run, or whose header says a command exited other than 0."""
    if not result.ran:
        return True
    head = result.text.split("\n", 1)[0]
    found = _EXIT.match(head)
    return found is not None and found.group(1) != "0"


def _value(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _arguments(call: Call) -> str:
    """A call's arguments as words: its one argument's value, or each `key value`, apart by
    ` · ` (a preview folds runs of spaces)."""
    own = {k: v for k, v in call.arguments.items() if k not in SHOWN_APART}
    if len(own) == 1:
        return _value(next(iter(own.values())))
    return " · ".join(f"{k} {_value(v)}" for k, v in sorted(own.items()))


def _call_line(call: Call, colours: dict[str, str], *, whole: bool, named: bool) -> Content:
    """A call in its card: a command whole; else its arguments, after its name unless the
    card's title already names it (`named`)."""
    command = call.arguments.get(_COMMAND)
    if isinstance(command, str):
        return Content.styled(command, f"bold {colours['fg']}")
    said = _arguments(call)
    if not said:
        return Content.styled(call.name, colours["fg"])
    shown = preview(said, whole=whole)
    if named:
        return Content.styled(shown, colours["fg"])
    return Content.assemble((f"{call.name}  ", colours["dim"]), (shown, colours["fg"]))


def _question_card(asked: Asked, colours: dict[str, str]) -> CardText:
    """A question his call asked: the question, then how it was answered. Never the REPL's
    prose around it (its hint, `answered q1: …, by human`): the owner's call, 2026-10-26."""
    dim = colours["dim"]
    body = Content.styled(asked.text, colours["fg"])
    if asked.choice is None:
        tick = Content.styled("waiting for you", colours["zeus"])
        foot = Content.styled("Enter for the default", dim)
    elif asked.source == "withdrawn":
        tick, foot = Content.styled("withdrawn", dim), None
    elif asked.defaulted:
        tick = Content.styled("no answer", dim)
        foot = Content.styled("proceeding on the default", dim)
    else:
        tick = Content.styled("✔ answered", colours["worker"])
        foot = Content.styled(asked.note or asked.choice, colours["fg"])
    return CardText(Content.styled("question", dim), tick, body, foot)


def _result_lines(
    entry: ZeusEntry | None, colours: dict[str, str], *, expanded: bool, verbosity: Verbosity
) -> list[Content]:
    """One call's result: `✔` or `✖` and its first line, clipped as the REPL clips it; the
    REPL's own body lines under it expanded; `running…` until it lands."""
    if entry is None or not isinstance(entry.record, ToolRecord):
        return [Content.styled("running…", colours["dim"])]
    result = entry.record
    head = result.text.split("\n", 1)[0] if result.text else ""
    bad = failed(result)
    mark = ("✖ ", colours["bad"]) if bad else ("✔ ", colours["worker"])
    said = clip(head) if result.ran else f"not run{': ' + clip(head) if head else ''}"
    lines = [Content.assemble(mark, (said, colours["dim"]))]
    if expanded and entry.frame is not None:
        # The REPL's own lines of the result, less the first, which the line above says, and
        # less the indent that sets them under it in a stream: the card does that here.
        for line in frame_lines(entry.frame, verbosity, _TERMINAL)[1:]:
            text = Text.from_ansi(line)
            lead = len(text.plain) - len(text.plain.lstrip(" "))
            lines.append(Content.from_rich_text(text[lead:]))
    return lines


def _own(call: Call, approved: dict[str, Approval], asked: dict[str, Asked]) -> bool:
    return (
        isinstance(call.arguments.get(_COMMAND), str) or call.id in approved or call.id in asked
    )


def turn_cards(
    turn: AssistantTurnP,
    zeus: ZeusPane,
    colours: dict[str, str],
    *,
    expanded: bool,
    verbosity: Verbosity,
) -> list[CardText]:
    """The cards under one of his turns: a command, a call an approval names, or a question,
    on its own; the calls between them, in a row, together."""
    approved = {a.call_id: a for a in zeus.approvals.values() if a.call_id is not None}
    asked = zeus.questions
    groups: list[list[Call]] = []
    for call in turn.calls:
        if _own(call, approved, asked) or not groups or _own(groups[-1][0], approved, asked):
            groups.append([call])
        else:
            groups[-1].append(call)
    whole = verbosity is Verbosity.VERBOSE
    cards: list[CardText] = []
    for group in groups:
        first = group[0]
        question = asked.get(first.id)
        if question is not None:
            cards.append(_question_card(question, colours))
            continue
        if isinstance(first.arguments.get(_COMMAND), str):
            title = f"$ {first.name}"
        else:
            title = first.name if len(group) == 1 else f"{len(group)} calls"
        foot: list[Content] = []
        for call in group:
            foot += _result_lines(
                zeus.results.get(call.id), colours, expanded=expanded, verbosity=verbosity
            )
        cards.append(
            CardText(
                title=Content.styled(title, colours["dim"]),
                tick=tick(approved.get(first.id) if len(group) == 1 else None, colours),
                body=Content("\n").join(
                    _call_line(c, colours, whole=whole, named=len(group) == 1) for c in group
                ),
                foot=Content("\n").join(foot),
            )
        )
    return cards


def approval_card(approval: Approval, colours: dict[str, str]) -> CardText:
    """An approval no call of Zeus's makes (the worker's, Athena's): a card of its own."""
    who = approval.asker or "command"
    task = f" · {approval.task_id}" if approval.task_id else ""
    return CardText(
        title=Content.styled(f"$ {who}{task}", colours["dim"]),
        tick=tick(approval, colours),
        body=Content.styled(approval.command, f"bold {colours['fg']}"),
        foot=None,
    )


# -- the box roles' calls and steps ------------------------------------------------

type BoxEntry = AthenaEntry | WorkerEntry
_SAID_BY_RECORD: Final = frozenset({"review"})
"""Calls a record of their own already says, when it happens: her `review` call's verdict is
its `task_reviewed` record, which lands before the call does (after the next round)."""
_BAD_VERDICTS: Final = frozenset({"feedback", "drop", "abandon", "re_cut"})
"""A review's answers drawn as `✖`: the task is not accepted as it stands."""


def _preview_arguments(call: ToolCallP) -> dict[str, object] | None:
    """A box call's arguments as its record previews them, when the preview is whole JSON."""
    if not call.args_preview:
        return None
    try:
        parsed = json.loads(call.args_preview)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _said(call: ToolCallP) -> str:
    """A call's arguments as words: its one argument's value, or each `key value`."""
    parsed = _preview_arguments(call)
    if parsed is None:
        return preview(call.args_preview or "", whole=False)
    if len(parsed) == 1:
        return _value(next(iter(parsed.values())))
    return " · ".join(f"{k} {_value(v)}" for k, v in sorted(parsed.items()))


def _result_line(call: ToolCallP, *, after_exit: bool) -> str | None:
    """The result's first line worth reading: past a leading `exit N` for a command."""
    lines = [line for line in (call.result_preview or "").splitlines() if line.strip()]
    if after_exit and lines and _EXIT.match(lines[0]):
        lines = lines[1:]
    return clip(lines[0]) if lines else None


def _files(
    step: EffectCompletedP, step_diffs: dict[str, RunDiff], colours: dict[str, str]
) -> list[Content]:
    """`✎ path +a -r` per file the step changed, the counts the server's step diff; the paths
    alone until it is read, or where it is not served."""
    diff = step_diffs.get(step.operation_id)
    if diff is not None and diff.files:
        return [
            Content.assemble(
                ("✎ ", colours["worker"]),
                (f"{f.path}  ", colours["fg"]),
                (f"+{f.added} ", colours["worker"]),
                (f"-{f.removed}", colours["bad"]),
            )
            for f in diff.files
        ]
    return [
        Content.assemble(("✎ ", colours["worker"]), (path, colours["fg"]))
        for path in step.observed_surface
    ]


def _step_tick(
    call: ToolCallP | None,
    step: EffectCompletedP | None,
    runs: list[ExecP],
    colours: dict[str, str],
) -> Content:
    ok, bad = colours["worker"], colours["bad"]
    if call is not None and call.denied_reason is not None:
        return Content.styled("✖ refused", bad)
    late = next((r for r in runs if r.timed_out), None)
    if late is not None:
        return Content.styled("✖ timed out", bad)
    exited = next((r for r in runs if r.exit_code != 0), None)
    if exited is not None:
        return Content.styled(f"✖ exit {exited.exit_code}", bad)
    if step is not None and step.outcome != "succeeded":
        return Content.styled(f"✖ {step.outcome}", bad)
    if call is None:
        return Content.styled("running…", colours["zeus"])
    return Content.styled("✔", ok)


def _call_card(
    call: ToolCallP | None,
    before: list[BoxEntry],
    *,
    step_diffs: dict[str, RunDiff],
    plan: Sequence[PlanTaskP],
    colours: dict[str, str],
) -> CardText:
    """One call and the step and commands it made (`before`); a step no call has closed yet
    when `call` is None."""
    step_entry = next(
        (e for e in reversed(before) if isinstance(e.record, EffectCompletedP)), None
    )
    step = step_entry.record if step_entry is not None else None
    assert step is None or isinstance(step, EffectCompletedP)
    runs = [e.record for e in before if isinstance(e.record, ExecP)]
    marks = frozenset(e.seq for e in before)
    dim, fg = colours["dim"], colours["fg"]
    arguments = _preview_arguments(call) if call is not None else None
    command = arguments.get("command") if arguments is not None else None
    if step is not None or runs or call is None:
        number = step_entry.step if step_entry is not None else None
        body = _files(step, step_diffs, colours) if step is not None else []
        if isinstance(command, str):
            body.append(Content.assemble(("$ ", colours["shell"]), (command, f"bold {fg}")))
        elif runs and not body:
            body += [Content.assemble(("$ ", colours["shell"]), (r.label, fg)) for r in runs]
        if not body:
            body = [Content.styled("nothing changed", dim)]
        said = _result_line(call, after_exit=True) if call is not None else None
        foot = Content.styled(said, dim) if said and (command or runs) else None
        return CardText(
            title=Content.styled(f"step {number}" if number else "step", dim),
            tick=_step_tick(call, step, runs, colours),
            body=Content("\n").join(body),
            foot=foot,
            live=call is None,
            marks=marks,
        )
    assert call is not None
    if call.tool == "plan" and plan:
        right = Content.styled(f"✔ {len(plan)} tasks", colours["worker"])
        lines = [Content.assemble((f"{p.id} ", dim), (p.title, fg)) for p in plan]
        return CardText(Content.styled("plan", dim), right, Content("\n").join(lines), None)
    said = _result_line(call, after_exit=False)
    if call.denied_reason is not None:
        right = Content.styled("✖ refused", colours["bad"])
        said = clip(call.denied_reason)
    else:
        right = Content("")
    body = Content.styled(_said(call), fg) if call.args_preview else Content("")
    return CardText(
        title=Content.styled(call.tool, dim),
        tick=right,
        body=body,
        foot=Content.styled(said, dim) if said else None,
        marks=marks,
    )


def _review_card(
    review: TaskReviewedP, titles: dict[str, str], colours: dict[str, str]
) -> CardText:
    bad = review.answer in _BAD_VERDICTS
    mark = "✖" if bad else "✔"
    title = titles.get(review.task_id, "")
    return CardText(
        title=Content.styled("review", colours["dim"]),
        tick=Content.styled(
            f"{mark} {review.answer.replace('_', ' ')}", colours["bad" if bad else "worker"]
        ),
        body=Content.assemble((f"{review.task_id} ", colours["dim"]), (title, colours["fg"])),
        foot=None,
    )


_HOOK_BAD: Final = frozenset({"deny", "halt", "error"})
"""A hook's decisions that stopped something, or a hook that failed: said in red."""
_HOOK_STEERS: Final = frozenset({"continue", "retry_task", "ask"})
"""A hook's decisions that change the role's next move: said in amber."""
HOOK_MARK: Final = UNICODE.hook
"""A hook card's mark: the glyph table's, drawn as a card draws `✔`, whole."""


def hook_tick(hook: HookP, colours: dict[str, str]) -> Content:
    """What a hook decided, on its card's right and in its line: `✖` and red for one
    that stopped something, failed or ran out of time; amber for one that changed the role's
    next move; dim for a note, a warning or no opinion."""
    word = "timed out" if hook.timed_out else hook.decision.replace("_", " ")
    if hook.timed_out or hook.decision in _HOOK_BAD:
        return Content.styled(f"✖ {word}", colours["bad"])
    tone = colours["zeus"] if hook.decision in _HOOK_STEERS else colours["dim"]
    return Content.styled(word, tone)


def hook_card(hook: HookP, seq: int, colours: dict[str, str]) -> CardText:
    """A hook that fired: what it decided on the right of a title kept short, so
    a narrow pane never cuts the verdict; then its name, where it fired and its time, what it
    said as the record previews it, and its bound when that was reached."""
    dim, fg = colours["dim"], colours["fg"]
    where = Content.assemble(
        (hook.name, fg), (" at ", dim), (hook.event, fg), (f" · {hook.wall_ms} ms", dim)
    )
    lines = [where]
    if hook.message:
        tone = colours["bad"] if hook.error is not None else fg
        lines.append(Content.styled(clip(hook.message), tone))
    if hook.bounded:
        lines.append(Content.styled("its bound was reached: it decides nothing more", dim))
    return CardText(
        title=Content.assemble((f"{HOOK_MARK} ", colours["zeus"]), ("hook", dim)),
        tick=hook_tick(hook, colours),
        body=Content("\n").join(lines),
        foot=None,
        marks=frozenset({seq}),
    )


def box_cards(
    entries: Sequence[BoxEntry],
    *,
    step_diffs: dict[str, RunDiff],
    colours: dict[str, str],
    plan: Sequence[PlanTaskP] = (),
    titles: dict[str, str] | None = None,
) -> list[CardText]:
    """A box role's records as cards, in order: each call with the steps and commands since
    the one before (a run writes the step, its commands, then the call that made them), a
    review as its verdict, a hook as what it decided where it fired, and the step
    under way, which no call has closed yet, last."""
    cards: list[CardText] = []
    before: list[BoxEntry] = []
    for entry in entries:
        match entry.record:
            case EffectCompletedP() | ExecP():
                before.append(entry)
            case TaskReviewedP() as review:
                cards.append(_review_card(review, titles or {}, colours))
            case HookP() as hook:
                cards.append(hook_card(hook, entry.seq, colours))
            case ToolCallP() as call if call.tool in _SAID_BY_RECORD and not before:
                continue
            case ToolCallP() as call:
                card = _call_card(
                    call, before, step_diffs=step_diffs, plan=plan, colours=colours
                )
                cards.append(replace(card, marks=card.marks | {entry.seq}))
                before = []
            case None if entry.text is not None:
                # A record this client cannot read is said, never dropped.
                cards.append(
                    CardText(
                        title=Content.styled("unreadable record", colours["dim"]),
                        tick=Content.styled("✖", colours["bad"]),
                        body=Content.styled(
                            f"{entry.text}: a record this client cannot read", colours["bad"]
                        ),
                        foot=None,
                        marks=frozenset({entry.seq}),
                    )
                )
            case _:
                continue
    if before:
        cards.append(
            _call_card(None, before, step_diffs=step_diffs, plan=plan, colours=colours)
        )
    return cards


class Card(Vertical):
    """One card: a rounded box, its title left and its tick right on the top edge
    (`border_title`, as a pane's title and label are), the body, and the foot under a rule."""

    DEFAULT_CSS = """
    Card {
        height: auto; border: round $faint; padding: 0 1; margin: 1 0 0 0;
        border-title-color: $dim;
    }
    Card.-live { border: round $zeus; }
    Card.-chosen { border: double $shell; }
    Card > .card-body { height: auto; width: 1fr; }
    Card > .card-foot { height: auto; width: 1fr; border-top: solid $faint; }
    """
    # Its parts are `card-body` and `card-foot`, never a pane's `body`: a pane's rule for its
    # own body (the worker's no-wrap) must not reach a card inside it.

    def __init__(self, text: CardText, *, classes: str = "") -> None:
        super().__init__(classes=classes)
        self.body = Static("", classes="card-body")
        self.foot = Static("", classes="card-foot")
        self.text = text

    def compose(self) -> ComposeResult:
        yield self.body
        yield self.foot

    def on_mount(self) -> None:
        self.show(self.text)

    def show(self, text: CardText) -> None:
        self.text = text
        self.body.update(text.body)
        self.foot.update(text.foot or "")
        self.foot.display = text.foot is not None
        self.set_class(text.live, "-live")
        self._retitle()

    def on_resize(self, _event: Resize) -> None:
        self._retitle()

    def _retitle(self) -> None:
        self.border_title = border_title(
            self.text.title, self.text.tick, self.outer_size.width, double=False
        )


def show_cards(host: Vertical, texts: Sequence[CardText], chosen: int | None = None) -> None:
    """`host`'s cards made `texts`: each one there shown again in place, the new ones mounted
    after them and the spare ones removed, so a pane's growing list never rebuilds. The card
    whose `marks` hold `chosen` is marked."""
    cards = list(host.query_children(Card))
    for card, text in zip(cards, texts, strict=False):
        card.show(text)
    for card in cards[len(texts) :]:
        card.remove()
    if len(texts) > len(cards):
        host.mount_all(Card(text) for text in texts[len(cards) :])
    for card in host.query_children(Card):
        card.set_class(chosen is not None and chosen in card.text.marks, "-chosen")
