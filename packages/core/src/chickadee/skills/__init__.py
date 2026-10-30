"""The knowledge layer.

A skill is a Markdown file with declared front matter: its description is resident in Zeus's
prefix (`index.py`), its body is loaded whole and pinned on demand (`load.py`, `pin.py`), and
its bundled resources are read or executed by the ordinary paths. It instructs the model and
never the harness, and it is never evidence.
"""

from __future__ import annotations

from chickadee.skills.discover import RESERVED_NAMES, Skill, SkillCatalog, SkillRoots, discover
from chickadee.skills.index import render_index
from chickadee.skills.invoke import Invoked, invoke
from chickadee.skills.load import SKILL_BODY_MAX, Skills
from chickadee.skills.pin import SKILL_PINNED_MAX, Pin, pinned, pinned_seqs, unload_skill

__all__ = [
    "RESERVED_NAMES",
    "SKILL_BODY_MAX",
    "SKILL_PINNED_MAX",
    "Invoked",
    "Pin",
    "Skill",
    "SkillCatalog",
    "SkillRoots",
    "Skills",
    "discover",
    "invoke",
    "pinned",
    "pinned_seqs",
    "render_index",
    "unload_skill",
]
