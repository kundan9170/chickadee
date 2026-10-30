"""The two stops: the design (stop 1) and the plan
(stop 2), asked by one loop.

`ask_stop` is the old gate's rounds loop (`delegation.gate`), moved and made one for both
stops: what is approved is what was last shown (invariant 15), an edit is a new text shown
again, a note ends the stop so the author can revise by it, and the rounds are bounded
(invariant 10), so a client that answers `edit` forever cannot hold a run without end. The
markdown is never parsed: an edit is refused only when it is empty or changes nothing, and a
note only when it says nothing.

`Estimate` and `WorkspacePreview`, the approval display's types, moved here from
`approval/gate.py`, which is retired.

Stop 1 is asked at dispatch (`delegation.stop_at_dispatch`); stop 2 by the desk, through a
`StopAsker` the episode composes over the interaction port's `confirm_stop`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from chickadee.errors import HarnessBug
from chickadee.ids import stop_id
from chickadee.schema.ids import sha256_hex
from chickadee.schema.interaction import ApprovalStop, StopAnswer, StopDecision, StopKind
from chickadee.schema.isolation import WorkspaceMode

__all__ = [
    "STOP_ROUNDS",
    "Estimate",
    "Reread",
    "StopAsker",
    "StopOutcome",
    "StopPort",
    "StopWithdrawn",
    "WorkspacePreview",
    "ask_stop",
    "stop_of",
]


@dataclass(frozen=True)
class Estimate:
    tokens: int
    cost_usd: float
    basis: Literal["dag", "history"]
    """`dag`: counted from the plan itself. An exact edit calls no model, so it is zero."""


@dataclass(frozen=True)
class WorkspacePreview:
    mode: WorkspaceMode
    source_ref: str | None
    """The branch the work starts from, the commit when the checkout is detached, or None
    when it has no commit yet."""


STOP_ROUNDS: Final = 20
"""How many times one stop is shown before it is given up as cancelled: the old gate's
bound, carried over."""


class StopPort(Protocol):
    """What `ask_stop` asks: `InteractionPort.confirm_stop`, which the port gains later. Until
    then a stop's
    asker is any value with this one method."""

    def confirm_stop(
        self, stop: ApprovalStop, *, cancel: threading.Event | None = None
    ) -> StopDecision: ...


class StopWithdrawn(Exception):
    """The stop's wait was ended by its `cancel` token, not by an answer: the run
    was dropped, or the process is leaving, or the turn that asked it was stopped hard. Nothing
    was approved; the caller ends or suspends what asked it, and a resumed run asks it again."""


@dataclass(frozen=True)
class Reread:
    """A folder stop's part as it stands now: its README, the folder's tree,
    and what its shape check found wrong. Read again at every answer, since the person edits
    the files themselves."""

    text: str
    version: str
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StopOutcome:
    """How a stop ended. `answer` is `approve`, `note` or `cancel`: an edit never ends one, it
    is shown again. `text` and `sha` are what was approved, or what was last shown."""

    answer: StopAnswer
    text: str
    sha: str
    note: str | None = None
    """The person's words, verbatim: what to revise by, or why they cancelled. Only
    ever theirs: never on an approval, which carries no words, and never the harness's."""
    bound: str | None = None
    """Why the harness ended the stop itself, when it was not the person's answer: the rounds
    bound. Kept apart from `note` so a reason the harness wrote is never recorded or reported
    as the person's."""
    version: str | None = None
    """A folder stop's approved tree: what the caller pins. None for a text
    stop."""


class StopAsker(Protocol):
    """How the desk asks stop 2: `ask_stop` over the session's port, bound by the composition;
    `reread` for a plan in its work folder."""

    def __call__(
        self, stop: ApprovalStop, *, reread: Callable[[], Reread] | None = None
    ) -> StopOutcome: ...


def stop_of(
    kind: StopKind,
    author: Literal["zeus", "athena"],
    text: str,
    workspace: WorkspacePreview,
    *,
    folder: str | None = None,
    notice: str | None = None,
) -> ApprovalStop:
    """The first showing of `text` at a stop: a part's README with its `folder`, or a
    text with none."""
    sha = sha256_hex(text)
    return ApprovalStop(
        id=stop_id(kind.value, sha),
        kind=kind,
        author=author,
        text=text,
        sha=sha,
        workspace=workspace.mode,
        source_ref=workspace.source_ref,
        folder=folder,
        notice=notice,
    )


def ask_stop(
    first: ApprovalStop,
    port: StopPort,
    *,
    rounds: int = STOP_ROUNDS,
    reread: Callable[[], Reread] | None = None,
    cancel: threading.Event | None = None,
) -> StopOutcome:
    """Ask the person about `first` until they approve it, send a note, or cancel.

    *Approve* binds the text last shown. *Edit* makes the person's text the one shown next,
    with its own sha; an empty edit, or one that changes nothing, is refused and the same text
    shown again, saying why. *Note* ends the stop with the note, verbatim; an empty one is
    refused and shown again, since a revision needs to know what to change. An approval with
    words is refused and shown again: it binds the text and nothing else. *Cancel* ends it with
    the person's reason. Past `rounds` showings it ends cancelled, the bound in `bound`, never
    in `note`.

    A folder stop (`reread`) reads its part again at every answer: an edit is
    the person's, saved in the folder's files, so it is what the folder now holds, whatever
    text came with it; a folder whose shape is wrong is shown again naming what is wrong; and
    an approval of a folder changed since it was shown, or of one whose shape is wrong, is
    refused and the folder shown again. Approved, `version` is the tree approved.

    `cancel` ends the wait: `StopWithdrawn`, never an answer."""
    if rounds < 1:
        raise HarnessBug(f"a stop is shown at least once, not {rounds} times")
    if reread is not None:
        return _ask_folder(first, port, rounds, reread, cancel)
    shown = first
    last = first
    """The showing the person last saw: what a bound ends on, never an edit no stop showed."""
    for round_ in range(1, rounds + 1):
        if round_ > 1:
            shown = shown.model_copy(update={"id": stop_id(shown.kind.value, shown.sha)})
        last = shown
        said = _asked(port, shown, cancel)
        refused: str | None = None
        match said.answer:
            case StopAnswer.APPROVE if said.note is not None and said.note.strip():
                # An approval binds the text shown and nothing else: words with it
                # would be dropped unseen, so the stop is shown again saying where they go.
                refused = (
                    "an approval carries no words: it binds the text shown. Send them as a "
                    "note, or edit the text"
                )
            case StopAnswer.APPROVE:
                return StopOutcome(StopAnswer.APPROVE, shown.text, shown.sha)
            case StopAnswer.CANCEL:
                return StopOutcome(StopAnswer.CANCEL, shown.text, shown.sha, said.note)
            case StopAnswer.NOTE:
                if said.note is not None and said.note.strip():
                    return StopOutcome(StopAnswer.NOTE, shown.text, shown.sha, said.note)
                refused = "a note says what to change: it was empty"
            case StopAnswer.EDIT:
                edited = said.edited
                if edited is None or not edited.strip():
                    refused = "the edit was not used: it was empty"
                elif edited == shown.text:
                    refused = "the edit changed nothing"
                else:
                    shown = shown.model_copy(
                        update={"text": edited, "sha": sha256_hex(edited), "refused": None}
                    )
        if refused is not None:
            shown = shown.model_copy(update={"refused": refused})
        shown = shown.model_copy(update={"round": round_ + 1})
    why = f"the {first.kind.value} was shown {rounds} times without an approval"
    if shown.sha != last.sha:
        why += "; the last edit came after the last showing and was not shown"
    return StopOutcome(StopAnswer.CANCEL, last.text, last.sha, bound=why)


def _asked(port: StopPort, shown: ApprovalStop, cancel: threading.Event | None) -> StopDecision:
    """One showing's answer; `StopWithdrawn` when `cancel` ended the wait, whatever the port
    returned, as `confirm_action`'s rule has it."""
    said = port.confirm_stop(shown, cancel=cancel)
    if cancel is not None and cancel.is_set():
        raise StopWithdrawn(f"stop {shown.id} was withdrawn before it was answered")
    return said


def _ask_folder(
    first: ApprovalStop,
    port: StopPort,
    rounds: int,
    reread: Callable[[], Reread],
    cancel: threading.Event | None,
) -> StopOutcome:
    """`ask_stop` for a folder: what is shown, and approved, is always the folder as it is."""
    now = reread()
    shown = _showing(first, now)
    seen = now.version
    for round_ in range(1, rounds + 1):
        if round_ > 1:
            shown = shown.model_copy(update={"id": stop_id(shown.kind.value, shown.sha)})
        said = _asked(port, shown, cancel)
        now = reread()
        refused: str | None = None
        match said.answer:
            case StopAnswer.APPROVE if said.note is not None and said.note.strip():
                refused = (
                    "an approval carries no words: it binds the folder shown. Send them as a "
                    "note, or edit the files"
                )
            case StopAnswer.APPROVE if now.version != seen:
                refused = "the folder changed after it was shown: this is it now"
            case StopAnswer.APPROVE if now.errors:
                refused = _wrong(now)
            case StopAnswer.APPROVE:
                return StopOutcome(StopAnswer.APPROVE, shown.text, shown.sha, version=seen)
            case StopAnswer.CANCEL:
                return StopOutcome(StopAnswer.CANCEL, shown.text, shown.sha, said.note)
            case StopAnswer.NOTE:
                if said.note is not None and said.note.strip():
                    return StopOutcome(StopAnswer.NOTE, shown.text, shown.sha, said.note)
                refused = "a note says what to change: it was empty"
            case StopAnswer.EDIT:
                if now.version == seen:
                    refused = "the edit changed nothing in the folder"
                elif now.errors:
                    refused = _wrong(now)
        shown = _showing(shown, now).model_copy(
            update={"refused": refused, "round": round_ + 1}
        )
        seen = now.version
    why = f"the {first.kind.value} was shown {rounds} times without an approval"
    return StopOutcome(StopAnswer.CANCEL, shown.text, shown.sha, bound=why)


def _showing(stop: ApprovalStop, now: Reread) -> ApprovalStop:
    return stop.model_copy(update={"text": now.text, "sha": sha256_hex(now.text)})


def _wrong(now: Reread) -> str:
    return "the folder is not in shape: " + "; ".join(now.errors)
