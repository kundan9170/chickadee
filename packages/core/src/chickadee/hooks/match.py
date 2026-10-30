"""Whether a configured hook is about an event: every matcher it sets must
hold, and one it leaves out matches anything. Each glob is read by the one path-glob matcher,
`paths.glob_match`; a command is matched by the programs its line runs, never by a
regex over the line.
"""

from __future__ import annotations

from chickadee.hooks.payloads import AfterStep, BeforeTool, HookEvent, PermissionAsked
from chickadee.hooks.spec import HookSpec
from chickadee.paths import glob_match

__all__ = ["matches"]


def matches(spec: HookSpec, event: HookEvent) -> bool:
    if not spec.enabled or spec.event is not event.point:
        return False
    if spec.role is not None and spec.role != event.role:
        return False
    if spec.model is not None and glob_match(event.model, (spec.model,)) is None:
        return False
    payload = event.payload
    named = isinstance(payload, BeforeTool | AfterStep | PermissionAsked)
    tool = payload.tool if named else None
    if spec.tool is not None and tool != spec.tool:
        return False
    if spec.program:
        programs = payload.programs if isinstance(payload, BeforeTool) else ()
        if not any(glob_match(p, spec.program) is not None for p in programs):
            return False
    return not spec.paths or any(glob_match(p, spec.paths) is not None for p in payload.paths())
