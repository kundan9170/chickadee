"""The points: where in a run a hook fires, at which level, and what it may
answer there, as data.

Level 1 guides one role inside its loop; level 2 steers the episode. `before_stop` is both: for
the worker and the scout it ends a round or a request, for Athena the episode's planning, for
Zeus the session's turn. Each point names the verdicts it takes (OpenCode types a failure
channel per hook for the same reason), what a script's exit code 2 means there (decision 11),
and Claude Code's name for it where one exists, so a script written for Claude Code ports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from chickadee.hooks.verdicts import (
    Ask,
    Continue,
    Deny,
    Halt,
    HookVerdict,
    Note,
    Proceed,
    RetryTask,
    Warn,
)

__all__ = ["POINTS", "HookPoint", "Level", "PointSpec", "blocking", "spec_of"]


class HookPoint(StrEnum):
    SESSION_START = "session_start"
    PROMPT_SUBMITTED = "prompt_submitted"
    BEFORE_REQUEST = "before_request"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    AFTER_STEP = "after_step"
    AFTER_BATCH = "after_batch"
    BEFORE_STOP = "before_stop"
    PERMISSION_ASKED = "permission_asked"
    ROUND_ENDED = "round_ended"
    TASK_REVIEWED = "task_reviewed"
    TASK_COMMITTED = "task_committed"
    EPISODE_ENDED = "episode_ended"
    BEFORE_COMPACT = "before_compact"
    JOB_EXITED = "job_exited"
    SESSION_END = "session_end"


Level = Literal[1, 2]


@dataclass(frozen=True)
class PointSpec:
    level: Level
    accepts: frozenset[type[HookVerdict]]
    """Every verdict a hook may answer here; `Proceed` always."""
    blocking: type[HookVerdict]
    """What a script's exit code 2 is here, its stderr the text."""
    claude: str | None = None
    """Claude Code's event name, a script's `hook_event_name` where there is one."""


def _takes(*kinds: type[HookVerdict]) -> frozenset[type[HookVerdict]]:
    return frozenset({Proceed, Warn, *kinds})


POINTS: Final[Mapping[HookPoint, PointSpec]] = {
    HookPoint.SESSION_START: PointSpec(1, _takes(Note), Warn, "SessionStart"),
    HookPoint.PROMPT_SUBMITTED: PointSpec(1, _takes(Deny, Note), Deny, "UserPromptSubmit"),
    # No `Note`: a request has no unsent message to carry one, and adding one would move the
    # prefix the provider caches.
    HookPoint.BEFORE_REQUEST: PointSpec(1, _takes(Halt), Warn),
    HookPoint.BEFORE_TOOL: PointSpec(1, _takes(Halt, Deny, Ask, Note), Deny, "PreToolUse"),
    HookPoint.AFTER_TOOL: PointSpec(1, _takes(Halt, Note), Note, "PostToolUse"),
    # A step's note rides on the answer of the call that made it; a hook that would end the
    # role there answers at `after_tool`, which sees the same call.
    HookPoint.AFTER_STEP: PointSpec(1, _takes(Note), Note),
    HookPoint.AFTER_BATCH: PointSpec(1, _takes(Halt, Note), Note, "PostToolBatch"),
    HookPoint.BEFORE_STOP: PointSpec(1, _takes(Continue, Note), Continue, "Stop"),
    HookPoint.PERMISSION_ASKED: PointSpec(1, _takes(Deny), Deny, "PermissionRequest"),
    HookPoint.ROUND_ENDED: PointSpec(2, _takes(Note, RetryTask), Note),
    HookPoint.TASK_REVIEWED: PointSpec(2, _takes(Note), Warn),
    HookPoint.TASK_COMMITTED: PointSpec(2, _takes(Note), Warn),
    HookPoint.EPISODE_ENDED: PointSpec(2, _takes(Note), Warn),
    HookPoint.BEFORE_COMPACT: PointSpec(1, _takes(), Warn, "PreCompact"),
    HookPoint.JOB_EXITED: PointSpec(1, _takes(Note), Warn),
    HookPoint.SESSION_END: PointSpec(1, _takes(), Warn, "SessionEnd"),
}


def spec_of(point: HookPoint) -> PointSpec:
    return POINTS[point]


def blocking(point: HookPoint, words: str, do: str | None = None) -> HookVerdict:
    """`words` as `point`'s blocking verdict: what a script's exit code 2 is there, and how a
    hook set to fail closed fails."""
    kind = spec_of(point).blocking
    if kind is Deny:
        return Deny(words, do)
    if kind is Continue:
        return Continue(words)
    if kind is Note:
        return Note(words)
    return Warn(words)
