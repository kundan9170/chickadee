"""The coder box cannot enumerate or load skills. Its roles, the task worker and Athena,
are offered no skill tool, their ceilings give no `skills` port, and the edits port they write
through has no way to name one. (The packet, whose DECLARED skills were to arrive in its
manifest, retired with the packet loop: D196, `T-3.157`.)"""

from __future__ import annotations

from chickadee.roles import ATHENA_CEILING, ATHENA_TASK, TASK_WORKER, WORKER_CEILING
from chickadee.tools.builtin import BUILTINS
from chickadee.tools.context import DraftCall, EditsPort


def test_the_box_roles_are_offered_no_skill_tool() -> None:
    skill_tools = {"load_skill", "find_skill"}  # `find_skill` since
    for policy in (TASK_WORKER, ATHENA_TASK):
        offered = {s.name for s in policy.surface(BUILTINS)}
        assert not offered & skill_tools, policy.role
    for name in BUILTINS.names():
        tool = BUILTINS.get(name)
        assert tool is not None
        if "skills" in tool.needs:
            assert name in skill_tools


def test_the_box_ceilings_give_no_skills_port() -> None:
    assert "skills" not in WORKER_CEILING.ports and "skills" not in ATHENA_CEILING.ports


def test_the_edits_port_names_no_skill() -> None:
    for port in (EditsPort, DraftCall):
        assert not [m for m in dir(port) if "skill" in m], port
