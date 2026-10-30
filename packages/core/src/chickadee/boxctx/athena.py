"""Athena's context: her conversation's head and her
notes.

Her episode is one request, whose head is her instructions and the templates in force, then
the approved design or Zeus's one-task plan (`athena_head`); the rest is her conversation
(`conversation.BoxConversation`), which fits by eviction alone. Her notes are `note` lines on
the session log, durable before they are shown (`write_note`).

Her planning session of the packet loop (one session per design, its requests, the earlier
designs' history, its compaction ladder and its summaries) is parked as it stood.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

from pydantic import with_config

from chickadee.client.request import Message
from chickadee.context.lines import LINE, Line, SessionKind
from chickadee.context.request import est_text
from chickadee.context.results import InvalidArgument
from chickadee.errors import HarnessBug
from chickadee.limits import chars_for_tokens

if TYPE_CHECKING:
    from chickadee.context.store import SessionEvent

__all__ = [
    "NOTE",
    "AthenaNote",
    "NumberedLog",
    "athena_head",
    "design_text",
    "write_note",
]

NOTE: Final[SessionKind] = "note"


class NumberedLog(Protocol):
    """The session log as a writer that names its line by its seq needs it (`Session`): the
    seq the next write takes is valid under `lock` until that write lands. Moved here from
    `summary.py` when it was parked: her notes are its one writer now."""

    lock: threading.RLock

    @property
    def next_seq(self) -> int: ...

    def append(self, line: Line, *, run_id: str | None = None) -> SessionEvent: ...


@with_config(LINE)
@dataclass(frozen=True)
class AthenaNote:
    """A note she wrote with `note()`: her claim, never a fact. The `note` line's
    type."""

    KIND: ClassVar[SessionKind] = "note"

    id: str
    """`N-<seq>`."""
    design_sha: str
    text: str
    """At most `athena.note_max` tokens."""


def write_note(
    log: NumberedLog, design_sha: str, run_id: str, cap: int, text: str
) -> AthenaNote | InvalidArgument:
    """The `note` line, durable before it is shown, for her notes port
    (`box/athena.Notes`). Over `cap` (`athena.note_max`) tokens is refused, and nothing is
    written."""
    if (tokens := est_text(text)) > cap:
        return _too_long(text, tokens, cap)
    with log.lock:
        written = AthenaNote(f"N-{log.next_seq}", design_sha, text)
        log.append(written, run_id=run_id)
    return written


def _too_long(text: str, tokens: int, cap: int) -> InvalidArgument:
    """The refusal of a note over `cap`, measured in characters, which she can count:
    her tokenizer is not P-2, so "200 tokens" alone left her trimming a few words at a time."""
    most = chars_for_tokens(cap)
    cut = len(text) - most
    share = -(-cut * 100 // len(text))  # rounded up: a cut that falls short is refused again
    return InvalidArgument(
        "text",
        f"this note is about {tokens} tokens ({len(text)} characters); a note is at most {cap} "
        f"tokens (athena.note_max), about {most} characters",
        do=f"send note again with text of at most {most} characters: cut at least {cut} "
        f"characters ({share}% of it) in one go, not a few words at a time, or split it into "
        "one note per decision",
    )


# -- her head ------------------------------

PLAN_TEMPLATE: Final = "## THE PLAN TEMPLATE (write your plan this way)"
BRIEF_TEMPLATE: Final = "## THE BRIEF TEMPLATE (write each task's brief this way)"
TASK_DESIGN: Final = "## THE DESIGN (approved by the user; it does not change)"
ONE_TASK: Final = "## THE TASK (a one-task plan the user approved: delegated as written, as T1)"
DESIGN_FILES: Final = (
    "## THE DESIGN'S OTHER FILES (approved with it; read each with `read(path)` as you need it)"
)
FIRST_ROUND: Final = "## ITS FIRST ROUND"
REVIEW_FIRST: Final = (
    "Review T1 with `review`; after it, make the final check and reply with your report."
)


def design_text(readme: str, folder: str, files: Sequence[tuple[str, str]]) -> str:
    """What Athena works from for a design in a work folder (A7): its pinned
    README, then each other file of `design/` by its full path and the line the README gives
    it, to read as she needs. Built from the pinned version, so the bytes never move."""
    text = readme.strip()
    if not files:
        return text
    rows = [f"- `{folder}/design/{name}`: {line}" for name, line in files]
    return "\n\n".join((text, DESIGN_FILES, "\n".join(rows)))


def athena_head(
    fixed: str,
    plan_template: str,
    brief_template: str,
    *,
    design: str | None = None,
    one_task: tuple[str, str] | None = None,
    standing: str | None = None,
) -> tuple[Message, Message]:
    """Her task-loop conversation's head: her instructions and the templates in force (the
    same bytes for every episode under them, so the provider's cache holds them), then what she
    works from. From a design: the approved design, its Request and Constraints sections
    already in it. From a one-task plan: Zeus's brief, delegated as T1, and its first
    round as she reads every round (`box/desk.render_round`), so she begins at `review`.
    Exactly one of `design` and `one_task` (the brief, then the round's text). `standing`, in a
    run that continues a design's work, says where it stands (`TaskDesk.standing`)."""
    if (design is None) == (one_task is None):
        raise HarnessBug("athena_head takes a design or a one-task plan, exactly one")
    system = "\n\n".join(
        (
            fixed.strip(),
            PLAN_TEMPLATE,
            plan_template.strip(),
            BRIEF_TEMPLATE,
            brief_template.strip(),
        )
    )
    if design is not None:
        user = f"{TASK_DESIGN}\n\n{design.strip()}"
        if standing:
            # A run that continues the work: where it stands, after the design.
            user = f"{user}\n\n{standing.strip()}"
    else:
        assert one_task is not None
        brief, round_ = one_task
        user = f"{ONE_TASK}\n\n{brief.strip()}\n\n{FIRST_ROUND}\n\n{round_}\n\n{REVIEW_FIRST}"
    return Message(role="system", content=system), Message(role="user", content=user)
