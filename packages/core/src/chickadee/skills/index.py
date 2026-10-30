"""The skills index: the only resident part of the layer.

One row per skill, `name · scope · description`, assembled once per session into the stable
prefix's `skills` part (`zeus/prefix.assemble`). Never a body. Its cost is bounded by
`max_rows * desc_chars`, the limits register's `skills.index_max_rows` and
`skills.index_desc_chars`, and by nothing else.

**Every cut is visible (invariant 16).** A description past `desc_chars` ends in `…`; past
`max_rows` rows, the block says how many were withheld, by scope. A model shown 24 of 30
skills that is not told so believes it saw them all, and its "there is no skill for that" is a
negative answer that was never a fact (invariant 20). A shadowed skill is listed, marked with
its winner, and an external skill's ignored fields are named in one note. An agent
skill says so, with the profile it runs as and the index's note on how it runs.

**Under `repo_trust = third_party`** a project skill's row says where it came from and its
description is framed as untrusted: resident text an attacker could choose.

**One row, wherever a skill is shown (`skill_row`).** `find_skill` answers in the index's rows
(`load.Skills.find`), so a skill reads the same however Zeus came to it: cut at the same
length, with the same note on how it runs, framed the same way.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from chickadee.limits import limit
from chickadee.skills.discover import Skill, SkillCatalog
from chickadee.skills.frame import frame_description

__all__ = ["HOW", "render_index", "skill_row"]

RepoTrust = Literal["first_party", "third_party"]

_HEAD = (
    "Skills: knowledge you may load whole with `load_skill(name)`. Only these descriptions "
    "are resident; a loaded body stays for the session."
)


_AGENTS = (
    " An agent skill runs as a fresh agent with `spawn_agent(skill=…, task=…)` and returns "
    "its report."
)

HOW = "A knowledge skill loads whole with `load_skill(name)`." + _AGENTS
"""How either kind is used, for a list of rows without the index's head (`find_skill`'s)."""

_SKIPPED_SHOWN = 3
"""Skipped external files named one by one; past this many, the rest are counted."""


def render_index(
    catalog: SkillCatalog,
    *,
    repo_trust: RepoTrust = "first_party",
    max_rows: int | None = None,
    desc_chars: int | None = None,
) -> str:
    """The index block, or `""` when there are no skills and nothing was skipped, as the
    prefix had before. `max_rows` and `desc_chars` are the config's rows; None is the
    register's default."""
    if not catalog.skills and not catalog.skipped:
        return ""
    if max_rows is None:
        max_rows = int(limit("skills.index_max_rows").value)
    if desc_chars is None:
        desc_chars = int(limit("skills.index_desc_chars").value)
    shown = catalog.skills[:max_rows]
    rows = [_HEAD + (_AGENTS if any(s.kind == "agent" for s in catalog.skills) else "")]
    third_party = repo_trust == "third_party"
    rows.extend(skill_row(s, third_party=third_party, desc_chars=desc_chars) for s in shown)
    for shadow in catalog.shadowed:
        if shadow.by in shown:
            rows.append(
                f"- {shadow.skill.name} · {_where(shadow.skill)} · shadowed by the "
                f"{_where(shadow.by)} skill of the same name; not loadable"
            )
    withheld = Counter(_where(s) for s in catalog.skills[max_rows:])
    if withheld:
        by_scope = ", ".join(f"{n} {scope}" for scope, n in sorted(withheld.items()))
        total = sum(withheld.values())
        rows.append(
            f"[{len(shown)} of {len(catalog.skills)} skills shown; {total} withheld "
            f"({by_scope}). find_skill finds them by what they are for.]"
        )
    for skip in catalog.skipped[:_SKIPPED_SHOWN]:
        rows.append(f"[skipped, not a skill we can read: {skip.path} ({skip.why})]")
    if len(catalog.skipped) > _SKIPPED_SHOWN:
        rows.append(
            f"[and {len(catalog.skipped) - _SKIPPED_SHOWN} more external files skipped]"
        )
    return "\n".join(rows)


def skill_row(
    skill: Skill, *, third_party: bool, desc_chars: int, cost: str | None = None
) -> str:
    """One skill's row: name, scope, how an agent skill runs (and `cost`, what one round of
    its profile may spend, when given), its description cut at `desc_chars` and framed, what
    to pass, and the notes."""
    text = frame_description(
        skill, _desc(skill.front.description, desc_chars), third_party=third_party
    )
    kind = ""
    if skill.kind == "agent":
        kind = f"agent skill, runs as {skill.profile} · " + (f"{cost} · " if cost else "")
    row = f"- {skill.name} · {_where(skill)} · {kind}{text}"
    if skill.front.argument_hint:
        row += f" · args: {skill.front.argument_hint}"
    if skill.note:
        row += f" · ({skill.note})"
    if skill.ignored:
        fields = ", ".join(skill.ignored)
        row += f" · (ignored from another tool's format: {fields})"
    return row


def _where(skill: Skill) -> str:
    return skill.scope if skill.scope != "external" else f"external {skill.origin}"


def _desc(text: str, chars: int) -> str:
    if len(text) <= chars:
        return text
    return text[: chars - 1].rstrip() + "…"
