"""What a hook may answer, and how several answers at one point become one.

Each point takes a subset (`points.POINTS`); a verdict outside it is refused when a built-in is
bound, and is a hook error when a script returns it. `combine` is the one merge, for every
point: Gemini CLI's aggregator, with one precedence instead of one rule per event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from chickadee.schema.journal import Fresh

__all__ = [
    "KINDS",
    "NOTHING",
    "PRECEDENCE",
    "Ask",
    "Combined",
    "Continue",
    "Deny",
    "Halt",
    "HookVerdict",
    "Note",
    "Proceed",
    "RetryTask",
    "Warn",
    "combine",
    "kind_of",
    "label",
    "labelled",
]


@dataclass(frozen=True)
class Proceed:
    """No opinion. Never approves anything."""


@dataclass(frozen=True)
class Deny:
    """The call, the prompt or the question does not go ahead. `do` is the next action, as
    every refusal a model reads ends with one."""

    reason: str
    do: str | None = None


@dataclass(frozen=True)
class Ask:
    """Ask the person first. It only tightens: no answer, or an unattended run, is a no."""

    question: str


@dataclass(frozen=True)
class Note:
    """Something the model reads, on the next message it was going to read (decision 4)."""

    text: str


@dataclass(frozen=True)
class Continue:
    """At `before_stop` only: the role does not stop yet, and reads `text` next."""

    text: str


@dataclass(frozen=True)
class RetryTask:
    """At `round_ended` only: the task gets another round before Athena reviews it, as her own
    `feedback` would give it (decision 5). `fresh` puts the tree back first, as hers does."""

    text: str
    fresh: Fresh = True


@dataclass(frozen=True)
class Halt:
    """End this role's loop cleanly: its tool use ends and it answers with what it has; before a
    model call, no call is made (decision 7)."""

    reason: str


@dataclass(frozen=True)
class Warn:
    """Something the person reads; no model does."""

    text: str


HookVerdict = Proceed | Deny | Ask | Note | Continue | RetryTask | Halt | Warn

PRECEDENCE: Final[tuple[type[HookVerdict], ...]] = (Halt, Deny, Ask, RetryTask, Continue)
"""The deciding verdicts, strongest first. `Note`, `Warn` and `Proceed` decide nothing: every
note and warning is delivered beside whatever decided."""

KINDS: Final[Mapping[type[HookVerdict], str]] = {
    Proceed: "proceed",
    Deny: "deny",
    Ask: "ask",
    Note: "note",
    Continue: "continue",
    RetryTask: "retry_task",
    Halt: "halt",
    Warn: "warn",
}
"""Each verdict's name, as a `hook` line's `decision` and a script's output spell it."""


def kind_of(verdict: HookVerdict) -> str:
    """A verdict's name as a `hook` line's `decision` and a script's output spell it."""
    return KINDS[type(verdict)]


@dataclass(frozen=True)
class Combined:
    """One point's answer: what decided, if anything did, and every note and warning."""

    decided: Deny | Ask | RetryTask | Continue | Halt | None = None
    notes: tuple[str, ...] = ()
    warns: tuple[str, ...] = ()
    by: str | None = None
    """The hook whose verdict decided, as a refusal names it (`HOOK_DENIED{hook, …}`); `hooks`
    for two joined `Continue`s."""

    @property
    def note(self) -> str | None:
        """The notes as one block, or None."""
        return "\n".join(self.notes) if self.notes else None


NOTHING: Final = Combined()


LABEL_HEAD: Final = "[hook "
"""How a script's words begin (`label`)."""


def label(name: str, text: str) -> str:
    """`text` as a model reads a script's words: named, so it is never taken for a tool's
    output, Athena's words or the worker's report."""
    return f"{LABEL_HEAD}{name}] {text}"


def labelled(name: str, verdict: HookVerdict) -> HookVerdict:
    """`verdict` with its words labelled (`label`). A script's are; a built-in's are the
    harness's own, written as such in its template, and are not."""
    match verdict:
        case Deny(reason=reason, do=do):
            return Deny(label(name, reason), do)
        case Ask(question=question):
            return Ask(label(name, question))
        case Halt(reason=reason):
            return Halt(label(name, reason))
        case RetryTask(text=text, fresh=fresh):
            return RetryTask(label(name, text), fresh)
        case Note(text=text):
            return Note(label(name, text))
        case Continue(text=text):
            return Continue(label(name, text))
        case Warn(text=text):
            return Warn(label(name, text))
        case Proceed():
            return verdict


def combine(said: Sequence[tuple[str, HookVerdict]]) -> Combined:
    """The answers at one point, in configuration order, as one (decision 3): the strongest
    deciding verdict wins; two `Continue`s are joined; of two `RetryTask`s the first is kept.
    Each verdict's words are as its hook gave them (`labelled` is the fire site's)."""
    decided: Deny | Ask | RetryTask | Continue | Halt | None = None
    by: str | None = None
    continues: list[str] = []
    continuers: list[str] = []
    notes: list[str] = []
    warns: list[str] = []
    for name, verdict in said:
        match verdict:
            case Note(text=text):
                notes.append(text)
            case Warn(text=text):
                warns.append(text)
            case Continue(text=text):
                continues.append(text)
                continuers.append(name)
            case Proceed():
                pass
            case Deny() | Ask() | RetryTask() | Halt():
                if decided is None or _rank(verdict) < _rank(decided):
                    decided, by = verdict, name
    if continues and (decided is None or _rank(Continue("")) < _rank(decided)):
        decided = Continue("\n\n".join(continues))
        by = continuers[0] if len(continuers) == 1 else "hooks"
    return Combined(decided, tuple(notes), tuple(warns), by)


def _rank(verdict: HookVerdict) -> int:
    return PRECEDENCE.index(type(verdict))
