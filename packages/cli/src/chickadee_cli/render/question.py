"""A question as the person sees it: its text, its evidence, its options
numbered, and the default marked. An action approval shows what would run and where it
reaches, and offers `approval_choices`: *once*, *always for this session* (this exact command,
when the question binds the line itself) and *always for this repo* where it is offered, and
what each *always* would cover, saying so when that is any arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from chickadee_cli.render.style import Style, Tone
from chickadee_serve.frames import approval_choices
from chickadee_wire.models import ActionApproval, CommandBinding
from chickadee_wire.questions import OpenQuestion

__all__ = [
    "APPROVAL_DEFAULT",
    "Choice",
    "approval_head",
    "asker",
    "choices",
    "covers",
    "default_of",
    "grant_breadth",
    "question_lines",
]

APPROVAL_DEFAULT = "deny"
"""What Enter answers to an approval: an action runs only on a person's word."""


@dataclass(frozen=True)
class Choice:
    key: str
    label: str


_LABELS = {
    "approve": "once",
    "session": "always for this session",
    "repo": "always for this repo",
}
"""How an approval's choices read when *always* is offered; without it, each is its key."""


def choices(asked: OpenQuestion) -> tuple[Choice, ...]:
    if asked.question is not None:
        return tuple(Choice(o.key, o.label) for o in asked.question.options)
    approval = asked.approval
    offered = approval_choices(
        grant=approval is not None and approval.grant is not None,
        exact=approval is not None and approval.exact is not None,
        offer_repo=approval is None or approval.offer_repo,
    )
    labels = dict(_LABELS)
    if approval is not None and approval.exact is not None:
        labels["session"] = "always for this session, this exact command"
    always = "session" in offered or "repo" in offered
    return tuple(Choice(c, labels.get(c, c) if always else c) for c in offered)


def default_of(asked: OpenQuestion) -> str:
    return asked.question.default if asked.question is not None else APPROVAL_DEFAULT


def covers(binding: CommandBinding) -> str:
    """What *always* would cover, as the person granting it reads it: the class,
    the prefix and destination, and whatever the call declared beyond them."""
    parts = [f"{binding.cls} `{' '.join(binding.prefix)}`"]
    if binding.destination is not None:
        parts.append(f"to {binding.destination}")
    if binding.network:
        parts.append("network on")
    if binding.effects:
        parts.append(f"writing {', '.join(binding.effects)}")
    if binding.credentials:
        parts.append(f"with {', '.join(binding.credentials)}")
    return " ".join(parts)


def grant_breadth(approval: ActionApproval) -> str:
    """How far the repository's *always* reaches past the prefix: any arguments
    when its row has no grant shape, so the person is told before granting it."""
    if approval.repo_any_arguments:
        return "with any arguments, without asking"
    return "within its allowed options"


def asker(approval: ActionApproval) -> str:
    """Who asks, and for which task: `zeus`, `athena`, or `worker on T2`."""
    who = approval.asker
    return f"{who} on {approval.task_id}" if approval.task_id else who


def approval_head(approval: ActionApproval) -> str:
    """An approval's first line: who asks to run what."""
    return f"{asker(approval)} asks to run {approval.kind}: {approval.display}"


def question_lines(asked: OpenQuestion, style: Style) -> list[str]:
    lines: list[str] = []
    if asked.question is not None:
        lines += style.lines(asked.question.text, Tone.ASK, indent="? ")
        for line in asked.question.evidence:
            lines += style.lines(line, Tone.DIM, indent="    ")
    elif asked.approval is not None:
        approval = asked.approval
        lines += style.lines(approval_head(approval), Tone.ASK, indent="! ")
        if approval.effect_surface:
            lines += style.lines(
                ", ".join(approval.effect_surface), Tone.DIM, indent="    reaches "
            )
        if approval.network:
            lines += style.lines("uses the network", Tone.DIM, indent="    ")
        if approval.credentials:
            names = ", ".join(approval.credentials)
            lines += style.lines(names, Tone.DIM, indent="    credentials ")
        if approval.grant is not None and approval.offer_repo:
            lines += style.lines(covers(approval.grant), Tone.DIM, indent="    always covers ")
            lines += style.lines(grant_breadth(approval), Tone.DIM, indent="    ")
    default = default_of(asked)
    options = choices(asked)
    if not options:
        hint = (
            f"answer in words; Enter for the default ({default})"
            if default
            else ("answer in words")
        )
        return lines + style.lines(hint, Tone.DIM, indent="    ")
    for n, choice in enumerate(options, start=1):
        mark = "  (default)" if choice.key == default else ""
        label = "" if choice.label == choice.key else f" — {choice.label}"
        lines += style.lines(f"[{n}] {choice.key}{label}{mark}", indent="    ")
    lines += style.lines(
        "answer by key or number; Enter for the default", Tone.DIM, indent="    "
    )
    return lines
