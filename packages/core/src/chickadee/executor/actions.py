"""`ActionSpec` validation, lint rules and DONE WHEN wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chickadee import shell_line
from chickadee.executor.approval import spec_text
from chickadee.executor.types import ActionSpec, ExecRequest, PolicyClass


@dataclass(frozen=True)
class LintIssue:
    field: str
    message: str


def lint(spec: ActionSpec) -> tuple[LintIssue, ...]:
    """Everything checkable before dispatch, so a bad action fails at LINT rather than
    halfway through an effect nobody can undo."""
    issues: list[LintIssue] = []
    text = spec_text(spec)
    if not text.strip():
        issues.append(LintIssue("command", "an action needs a command"))
    else:
        try:
            if isinstance(spec.command, str) and not shell_line.argv(spec.command):
                issues.append(LintIssue("command", "command parses to nothing"))
        except ValueError as exc:
            issues.append(LintIssue("command", f"command cannot be split: {exc}"))

    if not spec.effects:
        # An action with no declared surface cannot be effect-checked at all, and §6.2's
        # second check would be vacuous — the generalised scope gate with nothing to compare.
        issues.append(LintIssue("effects", "an action must declare its effect surface"))
    for glob in spec.effects:
        if glob.startswith("/") or ".." in glob:
            issues.append(LintIssue("effects", f"effect glob escapes the worktree: {glob}"))

    if not spec.done_when:
        issues.append(
            LintIssue(
                "done_when",
                "an action must declare DONE WHEN — exit 0 proves the command ran, not "
                "that it achieved its purpose (§6.2)",
            )
        )
    if not spec.why:
        issues.append(LintIssue("why", "an action must say why, so a human can decide"))
    return tuple(issues)


def to_request(spec: ActionSpec, cwd: Path, timeout_s: int = 300) -> ExecRequest:
    """The approved action as something the Executor can run."""
    return ExecRequest(
        command=spec.command,
        cwd=cwd,
        policy=PolicyClass.ACTION,
        timeout_s=timeout_s,
        network=spec.network,
        effects=spec.effects,
        idempotent=spec.idempotent,
        label=spec.why or spec_text(spec),
    )
