"""A stop at a terminal, and the approval-stop rate.

**A stop**: the markdown shown exactly as the engine asks it (invariant 15: what is shown is
what is approved), then `a`, `e` (the person's markdown, from the editor; an edit that changed
nothing is not sent), `n <words>` (the author revises by them, verbatim) or `c [reason]`.
**Enter does not approve.** The stop's default, approve, is what an unattended run gets;
a person at the stop types `a`. The design gate it replaced retired with the switch.


**The approval-stop rate** is read from the session's records: of the
delegations, how many waited on a person's approval. Above a fifth, the floor is asking too
often, and `/status` says so.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import RecordFrame
from chickadee_wire.models import (
    ApprovalStop,
    EpisodeRecord,
    QuestionAnsweredRecord,
    QuestionAskedRecord,
)

__all__ = [
    "STOP_HINT",
    "STOP_RATE_CEILING",
    "Invalid",
    "StopRate",
    "StopReply",
    "parse_stop",
    "stop_head",
    "stop_lines",
    "stop_rate",
    "stop_rate_line",
    "stop_where",
]

STOP_RATE_CEILING: Final = 0.2
"""Above roughly a fifth of delegations stopping for approval, the floor is
wrong."""

Editor = Callable[[str], str | None]
"""Opens text for the person to edit, and returns what they saved, or None when they left it
unchanged or no editor could be opened."""

FolderEditor = Callable[[Path], bool]
"""Opens every file of a stop's folder for the person to edit, and says
whether an editor opened; the files are saved where they are, and the engine re-reads them."""


@dataclass(frozen=True)
class Invalid:
    """A line that answers nothing, and why: the gate stays open."""

    why: str


# -- the approval-stop rate -------------------------------------------------------------


@dataclass(frozen=True)
class StopRate:
    stopped: int
    delegations: int

    @property
    def share(self) -> float:
        return self.stopped / self.delegations if self.delegations else 0.0

    @property
    def too_high(self) -> bool:
        return self.share > STOP_RATE_CEILING


_PERSON: Final = "human"


def stop_rate(records: Iterable[RecordFrame]) -> StopRate:
    """Of the session's delegations, how many stopped for a person's permission.

    A delegation is a `coder` call, and each has one `episode` record, whether its run went
    ahead or was declined or cancelled before it started. It stopped when a person
    answered a command's permission question for it, inside its run. the approval floor's rate
    measures how often
    the tier floor asks, and now the floor applies only at a command: the two
    stops are asked of every attended delegation by design, so counting them would read 100%
    whatever the floor did. Questions that are not approvals (Zeus asking what was meant) are
    not counted either. A session log from before the rebuild may hold a design gate or a
    plan-approval question: those were the work tier's, and count as they did."""
    approvals: set[str] = set()
    pending = False
    stopped_runs: set[str] = set()
    runs: list[str] = []
    for frame in records:
        if frame.kind == "question_asked":
            asked = QuestionAskedRecord.model_validate(frame.payload)
            if asked.gate is not None:
                approvals.add(asked.gate.id)
            elif asked.approval is not None:
                approvals.add(asked.approval.action_id)
            elif asked.question is not None and asked.question.kind == "plan_approval":
                approvals.add(asked.question.id)
        elif frame.kind == "question_answered":
            answer = QuestionAnsweredRecord.model_validate(frame.payload)
            if answer.id in approvals and answer.source == _PERSON:
                if frame.run_id is not None:
                    stopped_runs.add(frame.run_id)
                else:
                    pending = True
        elif frame.kind == "episode":
            run_id = EpisodeRecord.model_validate(frame.payload).run_id
            runs.append(run_id)
            if pending:
                stopped_runs.add(run_id)
                pending = False
    return StopRate(sum(1 for r in runs if r in stopped_runs), len(runs))


def stop_rate_line(rate: StopRate) -> tuple[str, Tone]:
    """`/status`'s line for it."""
    if not rate.delegations:
        return "no delegation yet", Tone.DIM
    text = (
        f"{rate.stopped} of {rate.delegations} delegations stopped for a person "
        f"({rate.share:.0%})"
    )
    if rate.too_high:
        return text + " — above a fifth: the approval floor asks too often (R-8)", Tone.BAD
    return text, Tone.DIM


# -- a stop -----------------------------------------------------------------------

STOP_HINT: Final = "[a]pprove  [e]dit  [n]ote <words>  [c]ancel [reason]"

_TITLES: Final = {"design": "the design", "plan": "the plan"}


@dataclass(frozen=True)
class StopReply:
    """An answer to send: the choice, and what travels with it."""

    choice: Literal["approve", "edit", "note", "cancel"]
    note: str | None = None
    text: str | None = None
    """The person's markdown, with `edit`."""


def stop_head(stop: ApprovalStop) -> str:
    """The stop's first line: who asks you to approve what."""
    kind = stop.kind
    round_ = f" (round {stop.round})" if stop.round > 1 else ""
    # Zeus's plan is a one-task plan; his design is planned by the team.
    title = (
        "the one-task plan"
        if (stop.author, kind) == ("zeus", "plan")
        else _TITLES.get(kind, kind)
    )
    return f"{stop.author} asks you to approve {title}{round_}"


def stop_where(stop: ApprovalStop) -> str:
    """Where the approved work runs: `the work runs in a new worktree from main`."""
    where = "a new worktree" if stop.workspace == "worktree" else "your checkout, in place"
    source = f" from {stop.source_ref}" if stop.source_ref else ""
    return f"the work runs in {where}{source}"


def stop_lines(stop: ApprovalStop, style: Style) -> list[str]:
    """The stop: who wrote what, the markdown as shown, why the last answer was not used, and
    the commands."""
    kind = stop.kind
    lines = style.lines(stop_head(stop), Tone.ASK, indent="? ")
    if stop.author == "zeus":
        other = "a design the team plans" if kind == "plan" else "one task"
        lines += style.lines(
            f"too big or too small? n <words> asks for {other} instead", Tone.DIM, indent="    "
        )
    lines += style.lines(stop_where(stop), Tone.DIM, indent="    ")
    lines.append("")
    if stop.folder is not None:
        # Never wrapped: the path is one the person copies into their own editor.
        lines.append(f"    its files are in {stop.folder}; e opens them all")
    if stop.notice is not None:
        lines += style.lines(stop.notice, Tone.ASK, indent="    ")
    lines.append("")
    lines += stop.text.rstrip("\n").split("\n")
    lines.append("")
    if stop.refused:
        lines += style.lines(stop.refused, Tone.BAD, indent="    ")
    lines += style.lines(STOP_HINT, Tone.DIM, indent="    ")
    return lines


def parse_stop(
    stop: ApprovalStop, line: str, edit: Editor, edit_folder: FolderEditor | None = None
) -> StopReply | Invalid:
    """`line` as an answer to `stop`, or why it answers nothing (the stop stays open). A stop
    with a `folder` is edited as its files: they are opened, saved in place, and the
    README as saved is sent, for the engine reads the folder again."""
    head, _, rest = line.strip().partition(" ")
    words = rest.strip() or None
    match head.lower():
        case "a" | "approve":
            if words is not None:
                # An approval binds the text shown; words with it would reach nobody.
                return Invalid("an approval takes no words: n <words> sends a note, e edits")
            return StopReply("approve")
        case "e" | "edit" if stop.folder is not None:
            folder = Path(stop.folder)
            if edit_folder is None or not folder.is_dir() or not edit_folder(folder):
                return Invalid(f"could not open the files of {stop.folder} in an editor")
            readme = folder / "README.md"
            saved = readme.read_text(encoding="utf-8") if readme.is_file() else ""
            return StopReply("edit", text=saved or stop.text)
        case "e" | "edit":
            saved = edit(stop.text)
            if saved is None or saved == stop.text:
                return Invalid("the text is unchanged; nothing was sent")
            return StopReply("edit", text=saved)
        case "n" | "note":
            if words is None:
                return Invalid("a note needs words: n <what to change>")
            return StopReply("note", note=words)
        case "c" | "cancel":
            return StopReply("cancel", note=words)
        case _:
            return Invalid(f"not an answer: {line.strip() or '(empty)'}; {STOP_HINT}")
