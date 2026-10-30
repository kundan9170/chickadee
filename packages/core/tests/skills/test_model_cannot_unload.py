"""No tool, no skill body and no packet can reach unload. At the cap the model refuses
and names the pinned set; it never frees a slot for itself."""

from __future__ import annotations

import inspect
from pathlib import Path

import chickadee
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import SkillsPort, ToolContext
from chickadee.tools.spec import SURFACE

SRC = Path(chickadee.__file__).parent


def test_no_tool_unloads() -> None:
    assert not [n for n in BUILTINS.names() if "unload" in n]
    assert not [s.name for s in SURFACE if "unload" in s.name]


def test_the_skills_port_can_only_load() -> None:
    """It loads and, since, finds; it never unloads."""
    methods = {n for n, _ in inspect.getmembers(SkillsPort) if not n.startswith("_")}
    assert methods == {"load", "find"}
    assert "skills" in {f for f in ToolContext.__dataclass_fields__}


def test_only_the_pin_module_writes_an_unload_and_no_tool_or_box_code_names_it() -> None:
    """The module boundary is the enforcement: nothing a model's call runs imports it."""
    callers = [
        p.relative_to(SRC).as_posix()
        for p in SRC.rglob("*.py")
        if "unload_skill" in p.read_text(encoding="utf-8")
    ]
    assert sorted(callers) == ["skills/__init__.py", "skills/pin.py"]
