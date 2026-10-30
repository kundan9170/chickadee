"""Evidence at handoff, stage A (D237 and its Amendment 1): whether the harness's own record
backs the worker's "done", measured from the stop's payload, never from the worker's words.

- **Tests after the last change**: the watchdog's measure (`watchdog.unchecked`), files changed
  since the round's last test run; a task whose facts show changes and no test run at all; a
  last run read as `partial`, which shows nothing.
- **The task's tests**: a test it broke, or one of its own last seen failing.
  Tests failing before the task touched them never count.

`evidence_check` sends the worker back once at `before_stop` when the record does not back it,
and only where that means something: a runner exists for a changed file and the worker may run
commands, its tools are open, and the watchdog has not asked it to stop and report.
`evidence_report` gives Athena one line a round at `round_ended`. Both are pure functions of the
payload, as the watchdog's are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict

from chickadee.hooks.fire import HookFn
from chickadee.hooks.payloads import (
    BeforeStop,
    DiagnosticsSeen,
    HookEvent,
    RoundEnded,
    TestRunSeen,
    TestsSeen,
    WorkerProgress,
)
from chickadee.hooks.verdicts import Continue, HookVerdict, Note, Proceed
from chickadee.hooks.watchdog import Unchecked, fill, listed, unchecked
from chickadee.schema.journal import TaskFactsP

__all__ = [
    "HEAD",
    "Evidence",
    "EvidenceParams",
    "evidence_check",
    "evidence_line",
    "evidence_of",
    "evidence_report",
]

HEAD: Final = "[evidence]"


class EvidenceParams(BaseModel):
    """What sends the worker back; give `evidence_report` the same `with`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    untested: bool = True
    """Files changed since the last test run, none ran in the task, or the last run's result
    was not shown."""
    failing: bool = True
    """A test the task broke, or one of its own last seen failing."""
    errors: bool = True
    """Errors in the task's files that were not there when it began."""


@dataclass(frozen=True)
class Evidence:
    """What the record holds of a round's "done"."""

    runnable: bool
    """A runner exists for a changed file's language and the worker may run commands."""
    since: Unchecked
    """Changes since the round's last test run."""
    never_tested: bool
    """The task changed files and ran no test command at all."""
    last_run: TestRunSeen | None
    """The round's last test run."""
    last_command: str | None
    """That run's call in a few words, as the record quotes it."""
    tests: TestsSeen | None
    tests_changed: tuple[str, ...]
    """The test files the task added or edited."""
    diagnostics: DiagnosticsSeen | None = None
    """The task's files against their start, where the stop measured them."""

    @property
    def unread(self) -> bool:
        """The last run came after the last change and its result was not shown."""
        return (
            self.last_run is not None
            and self.last_run.how == "partial"
            and self.since.responses == 0
        )

    @property
    def untested(self) -> bool:
        return self.runnable and (self.since.responses > 0 or self.never_tested or self.unread)

    @property
    def failing(self) -> tuple[str, ...]:
        """The task's tests the record last saw failing: those it broke first."""
        if self.tests is None:
            return ()
        return (*self.tests.broken, *self.tests.failing)


def evidence_of(progress: WorkerProgress, facts: TaskFactsP | None) -> Evidence:
    """The round's evidence, from its progress record and the task's facts."""
    runs = [c for c in progress.calls if c.run is not None]
    last = runs[-1] if runs else None
    tested = any(c.test_outcome is not None for c in facts.commands) if facts else bool(runs)
    changed = bool(facts.files) if facts is not None else False
    return Evidence(
        runnable=progress.runnable,
        since=unchecked(progress.calls),
        never_tested=changed and not tested and not runs,
        last_run=last.run if last is not None else None,
        last_command=last.about if last is not None else None,
        tests=progress.tests,
        tests_changed=((*facts.tests_added, *facts.tests_edited) if facts is not None else ()),
        diagnostics=progress.diagnostics,
    )


def evidence_check(params: EvidenceParams) -> HookFn:
    """At the worker's `before_stop`: sent back once when the record does not back its stop."""

    def run(event: HookEvent) -> HookVerdict:
        stop = event.payload
        if not isinstance(stop, BeforeStop) or stop.progress is None:
            return Proceed()
        if not stop.tools_open or stop.handed_up:
            return Proceed()
        found = evidence_of(stop.progress, stop.facts)
        gaps = _gaps(found, params)
        if not gaps:
            return Proceed()
        command = (
            f"\nYour last test run was {found.last_command}." if found.last_command else ""
        )
        return Continue(
            fill(
                "evidence_check.md",
                gaps="\n".join(f"- {g}" for g in gaps),
                last=command,
                already=_already(found),
            )
        )

    return run


def _gaps(found: Evidence, params: EvidenceParams) -> list[str]:
    gaps: list[str] = []
    if params.untested and found.untested:
        changed = listed(found.since.files) or "files"
        if found.never_tested:
            gaps.append(f"you changed {changed} and ran no tests in this task")
        elif found.since.responses > 0:
            gaps.append(f"you changed {changed} after your last test run")
        else:
            gaps.append(
                "your last test run's result was not shown: its output was cut, or its exit "
                "code was another command's (`| tail`, `; echo`)"
            )
    seen = found.diagnostics
    if params.errors and seen is not None and seen.new_errors:
        shown = "; ".join(seen.new_errors[:_SHOWN])
        more = len(seen.new_errors) - _SHOWN
        gaps.append(
            f"errors your changes brought in: {shown}"
            + (f" and {more} more" if more > 0 else "")
        )
    if params.failing and found.tests is not None and found.failing:
        if found.tests.broken:
            gaps.append(
                f"tests that passed earlier in the task now fail: {listed(found.tests.broken)}"
            )
        if found.tests.failing:
            gaps.append(
                f"the task's tests were last seen failing: {listed(found.tests.failing)}"
            )
    return gaps


_SHOWN: Final = 6


def _already(found: Evidence) -> str:
    """The errors there before the task began, apart and optional."""
    seen = found.diagnostics
    if seen is None or not seen.already:
        return ""
    shown = "\n".join(f"- {e}" for e in seen.already[:_SHOWN])
    more = len(seen.already) - _SHOWN
    tail = f"\n- and {more} more" if more > 0 else ""
    return (
        "\nAlready there before your task began, never counted against it: fix them only if "
        f"you can.\n{shown}{tail}"
    )


def evidence_report(params: EvidenceParams) -> HookFn:
    """At `round_ended`, once a round, however it ended: the evidence line for the report's
    reader. `params` is `evidence_check`'s, read for nothing yet: the line says it all."""
    del params

    def run(event: HookEvent) -> HookVerdict:
        ended = event.payload
        if not isinstance(ended, RoundEnded) or ended.progress is None:
            return Proceed()
        return Note(evidence_line(evidence_of(ended.progress, ended.facts)))

    return run


def evidence_line(found: Evidence) -> str:
    """One line: what the record backs of the round, for Athena."""
    if not found.runnable:
        after = "no test runner for the changed files, or nothing changed"
    elif found.never_tested:
        after = "no, no test ran in the task"
    elif found.since.responses > 0:
        after = f"no, {listed(found.since.files) or 'files'} changed since the last run"
    elif found.last_run is None:
        after = "nothing changed this round"
    else:
        after = f"yes, {_run(found.last_run)}"
    tests = found.tests
    line = fill(
        "evidence_report.md",
        after=after,
        broken=listed(tests.broken) if tests and tests.broken else "none",
        failing=listed(tests.failing) if tests and tests.failing else "none",
        changed=listed(found.tests_changed) or "none",
    )
    seen = found.diagnostics
    if seen is None:
        return line
    unchecked = listed(seen.unchecked) or "none"
    return (
        f"{line.removesuffix('.')}; new errors since the task began: {len(seen.new_errors)}; "
        f"new warnings: {len(seen.new_warnings)}; errors already there: {len(seen.already)}; "
        f"not re-checked: {unchecked}."
    )


def _run(run: TestRunSeen) -> str:
    if run.how == "partial":
        return "but its result was not shown"
    if run.how != "summary":
        return f"the last run {run.outcome} (read from its exit code: no summary was printed)"
    unit = "package(s)" if run.unit == "package" else "test(s)"
    counts = f"{run.passed} {unit} passed, {run.failed} failed"
    if run.errors:
        counts += f", {run.errors} errors"
    return f"the last run {run.outcome}: {counts} (read from its summary)"
