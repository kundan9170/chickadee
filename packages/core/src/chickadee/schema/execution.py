"""Contract 7 — Exec.

Invariant 2: the Executor owns process spawning. Nothing else in the system may call
`subprocess`, so this is the only shape a command can take.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

WIRE = ConfigDict(extra="forbid", frozen=True)


class PolicyClass(StrEnum):
    DISCOVERY = "discovery"
    ACTION = "action"
    WORKTREE = "worktree"
    """`verification` until contracts v2: the box shell's class."""
    HUMAN = "human"


class SandboxLevel(StrEnum):
    """What isolation a command actually ran under.

    Defined here and re-exported by `isolation` and `executor` rather than the other way
    round: `ExecResult.sandbox` carries it, that row is frozen at G0, and a frozen contract
    cannot depend on a package frozen later.

    It was defined twice until 2026-09-12 — here and in `isolation/types.py`, with identical
    members. `StrEnum` made that almost invisible: the two compared `==` True and `is` False,
    against a codebase whose house style is `is EnumMember` in ninety-odd places. The check
    it would have broken first is `applied is SandboxLevel.NONE` — the refusal to run
    un-sandboxed when one was required, which would have silently evaluated False.
    """

    NATIVE = "native"
    CONTAINER = "container"
    NONE = "none"
    """**Reported, never silent**. An un-sandboxed run is a fact about the result,
    and `require_sandbox=true` aborts rather than assuming one."""


class ExecRequest(BaseModel):
    model_config = WIRE

    command: str
    cwd: str = "."
    policy: PolicyClass
    timeout_s: int
    network: bool = False
    effects: tuple[str, ...] = ()
    idempotent: bool = False
    label: str = ""

    @model_validator(mode="after")
    def _effects_match_policy(self) -> ExecRequest:
        # A discovery command that declares effects is not discovery; an action that
        # declares none has nothing for the effect-surface diff to check against.
        if self.policy is PolicyClass.DISCOVERY and self.effects:
            raise ValueError(
                "policy=discovery is read-only by definition; effects must be empty"
            )
        if self.policy is PolicyClass.ACTION and not self.effects:
            raise ValueError("policy=action requires a declared effect surface")
        return self


class ExecResult(BaseModel):
    model_config = WIRE

    exit_code: int
    stdout_sha: str
    stderr_sha: str
    wall_ms: int
    timed_out: bool
    sandbox: SandboxLevel
    violations: tuple[str, ...] = ()
    ok: bool
