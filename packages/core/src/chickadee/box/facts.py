"""A task's facts: what it did, measured, never read from its report.

Between the task's start tree and the tree its last step reached: each file created, edited
or deleted and its line counts (`workspace.diff_stats`); the test files it added and the
existing ones it changed, each by its own language's profile; the symbols added and removed
against the table taken at delegation; each changed file's syntax, by the edit tool's own
check (`box/syntax.SyntaxChecker`); and every command it ran, from its step and its
`exec` line, a test run read by what it printed (`executor/tests.read_test_run`),
so a run that collected nothing, or whose output was cut, never reads as a pass.

`task` None is Athena's own steps since the last commit: her final check.

`render_facts` is the facts as Athena reads them under each report (step 6): what
the harness measured, said as such. `last_attempt` is the short form a new conversation that
replaces a failed attempt begins with, so nothing it learned is lost silently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from chickadee.box.steps import reached, task_steps
from chickadee.box.syntax import SyntaxChecker
from chickadee.boxctx.records import SymbolGraph, SymbolTable, symbol_table
from chickadee.edits.lines import split_lines
from chickadee.executor.tests import read_test_run
from chickadee.executor.types import ExecResult
from chickadee.journal.projection import Operation, Placed, RunProjection
from chickadee.profiles.model import LanguageProfile
from chickadee.schema.execution import SandboxLevel
from chickadee.schema.journal import (
    CommandFactP,
    ExecP,
    FileFactP,
    SyntaxFactP,
    TaskDelegatedP,
    TaskFactsP,
    ToolCallP,
)
from chickadee.workspace import diff_stats

__all__ = [
    "LAST_ATTEMPT",
    "TAIL_LINES",
    "exec_result",
    "last_attempt",
    "measure",
    "ran_of",
    "render_facts",
]

TAIL_LINES = 20
"""How much of a command's output its fact keeps; the whole is its blob."""

_COMMAND = re.compile(r'"command":\s*"((?:[^"\\]|\\.)*)"')


def measure(
    root: Path,
    run: RunProjection,
    task: str | None,
    profiles: Sequence[LanguageProfile],
    graph: SymbolGraph | None,
    before: SymbolTable | None,
    blob: Callable[[str], bytes],
    *,
    start: str | None = None,
    syntax: SyntaxChecker | None = None,
) -> TaskFactsP:
    """The facts of `task` as the journal and the trees hold them now. `before` is the symbol
    table taken at delegation, kept in memory; None after a resume, and the symbols are then
    unknown rather than diffed against the wrong tree (invariant 20). `start` is the tree the
    measure starts from when the task's own line does not say (Athena's, since a commit).
    `syntax` judges each changed file on disk, which must be the task's end tree: None, and no
    syntax is measured."""
    lines = run.tasks.get(task, ()) if task is not None else ()
    first = start if start is not None else _start(lines)
    steps = task_steps(run, task)
    end = reached(steps, first)
    files = diff_stats(root, first, end) if end != first else ()
    tests = [f for f in files if _is_test(f.path, profiles)]
    known = before is not None and graph is not None
    added, removed = _symbols(before, graph, [f.path for f in files]) if known else ((), ())
    return TaskFactsP(
        start_tree=first,
        end_tree=end,
        files=tuple(
            FileFactP(path=f.path, added=f.added, removed=f.removed, kind=f.kind) for f in files
        ),
        tests_added=tuple(f.path for f in tests if f.kind == "created"),
        tests_edited=tuple(f.path for f in tests if f.kind == "edited"),
        symbols_added=added,
        symbols_removed=removed,
        symbols_known=known,
        commands=_commands(run, steps, profiles, blob),
        syntax=_syntax(root, [f.path for f in files if f.kind != "deleted"], syntax),
    )


_DETAIL_MAX = 200
"""Characters of a failed check's first line a fact keeps: where, and what."""


def _syntax(
    root: Path, paths: Sequence[str], checker: SyntaxChecker | None
) -> tuple[SyntaxFactP, ...]:
    """Each path's verdict over its bytes on disk; a file no profile judges is left out."""
    if checker is None:
        return ()
    out: list[SyntaxFactP] = []
    for path in paths:
        try:
            content = (root / path).read_bytes()
        except OSError:
            continue
        verdict = checker.check(path, content)
        if verdict is None:
            continue
        said = (verdict.detail or "").strip()
        detail = said.splitlines()[0][:_DETAIL_MAX] if said else None
        out.append(SyntaxFactP(path=path, ok=verdict.ok, detail=detail, depth=verdict.depth))
    return tuple(out)


def _start(lines: Sequence[Placed[object]]) -> str:  # type: ignore[type-var]
    for line in lines:
        if isinstance(line.payload, TaskDelegatedP):
            return line.payload.start_tree
    raise ValueError("a task's facts are measured after it was delegated")


def _is_test(path: str, profiles: Sequence[LanguageProfile]) -> bool:
    return any(p.tests.matches(path) for p in profiles)


def _symbols(
    before: SymbolTable | None, graph: SymbolGraph | None, paths: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if before is None or graph is None:
        return (), ()
    after = symbol_table(graph, paths)
    was = {s.qualified_name for p in paths for s in before.symbols.get(p, ())}
    now = {s.qualified_name for p in paths for s in after.symbols.get(p, ())}
    return tuple(sorted(now - was)), tuple(sorted(was - now))


def _commands(
    run: RunProjection,
    steps: Sequence[Operation],
    profiles: Sequence[LanguageProfile],
    blob: Callable[[str], bytes],
) -> tuple[CommandFactP, ...]:
    """Each `shell` step's command: its words from its `tool_call` line (matched by the sha of
    its arguments), its result from its `exec` line, none when it never ran."""
    execs = {str(p.payload.operation_id): p.payload for p in run.lines(ExecP)}
    calls = {str(p.payload.args_sha): p.payload for p in run.lines(ToolCallP)}
    out: list[CommandFactP] = []
    for op in steps:
        intent = op.intended
        if intent is None or intent.declared_surface != ("shell",):
            continue
        said = calls.get(str(intent.approved_sha))
        command = _command(said)
        ran = execs.get(str(intent.operation_id))
        if ran is None:
            out.append(
                CommandFactP(
                    operation_id=str(intent.operation_id),
                    command=command,
                    exit_code=None,
                    timed_out=False,
                    tail="",
                    output_sha=None,
                )
            )
            continue
        stdout = _text(blob, ran.stdout_sha)
        stderr = _text(blob, ran.stderr_sha)
        outcome, failing = _tests(command, ran, stdout, stderr, profiles)
        out.append(
            CommandFactP(
                operation_id=str(intent.operation_id),
                command=command,
                exit_code=int(ran.exit_code),
                timed_out=bool(ran.timed_out),
                tail="\n".join(split_lines(stdout + stderr)[-TAIL_LINES:]),
                output_sha=ran.stdout_sha or None,
                test_outcome=outcome,
                failing=failing,
            )
        )
    return tuple(out)


def _command(call: ToolCallP | None) -> str:
    """The command line from its call's argument preview (key-sorted JSON, redacted, possibly
    cut): its `command`, read even from a cut preview."""
    if call is None or call.args_preview is None:
        return ""
    preview = call.args_preview
    try:
        args: object = json.loads(preview)
    except ValueError:
        found = _COMMAND.search(preview)
        return str(json.loads(f'"{found.group(1)}"')) if found else ""
    if not isinstance(args, dict):
        return ""
    command = cast("dict[str, object]", args).get("command")
    return command if isinstance(command, str) else ""


def _text(blob: Callable[[str], bytes], sha: str) -> str:
    if not sha:
        return ""
    try:
        return blob(sha).decode("utf-8", "replace")
    except (KeyError, OSError, ValueError):
        return ""


def _tests(
    command: str,
    ran: ExecP,
    stdout: str,
    stderr: str,
    profiles: Sequence[LanguageProfile],
) -> tuple[str | None, tuple[str, ...]]:
    """`read_test_run`'s outcome and failing tests when the run is a test run;
    None and nothing for any other command."""
    read = read_test_run(command, exec_result(ran, stdout, stderr), profiles)
    return (None, ()) if read is None else (read.outcome.value, read.failing)


def exec_result(ran: ExecP, stdout: str, stderr: str) -> ExecResult:
    """An `exec` line and its output as the run it records: what a test run is read from, by
    the facts and the progress record alike."""
    return ExecResult(
        exit_code=int(ran.exit_code),
        stdout_sha=ran.stdout_sha or None,
        stderr_sha=ran.stderr_sha or None,
        wall_ms=int(ran.wall_ms),
        timed_out=bool(ran.timed_out),
        sandbox=SandboxLevel(ran.sandbox),
        stdout=stdout,
        stderr=stderr,
    )


def ran_of(ran: ExecP, blob: Callable[[str], bytes]) -> ExecResult:
    """An `exec` line's run, its output read from its blobs."""
    return exec_result(ran, _text(blob, ran.stdout_sha), _text(blob, ran.stderr_sha))


def render_facts(facts: TaskFactsP, stopped: str) -> str:
    """`FACTS (harness):` and one row per kind of fact, in its fixed order. A row with
    nothing to say says so, so a missing row is never read as "none"."""
    rows = ["FACTS (harness):", f"  stopped because: {stopped}"]
    changed = ", ".join(f"{f.path} ({f.kind}, +{f.added} -{f.removed})" for f in facts.files)
    rows.append(f"  changed: {changed or 'nothing'}")
    if facts.tests_added or facts.tests_edited:
        added = ", ".join(facts.tests_added) or "none"
        edited = ", ".join(facts.tests_edited) or "none"
        rows.append(f"  test files: added {added}; existing ones edited {edited}")
    else:
        rows.append("  test files: none changed")
    if facts.symbols_known:
        added = ", ".join(facts.symbols_added) or "none"
        removed = ", ".join(facts.symbols_removed) or "none"
        rows.append(f"  symbols: added {added}; removed {removed}")
    else:
        rows.append("  symbols: not known (the table before the task was lost with a restart)")
    rows.append(f"  syntax: {_judged(facts)}")
    if not facts.commands:
        rows.append("  commands: none ran")
    for i, c in enumerate(facts.commands):
        label = "  commands: " if i == 0 else "            "
        rows.append(f"{label}{c.command} -> {_ran(c)}")
    if facts.skipped:
        rows.append(f"  not snapshotted (over the size cap): {', '.join(facts.skipped)}")
    return "\n".join(rows)


def _judged(facts: TaskFactsP) -> str:
    broken = [f"{s.path} ({s.detail or 'no detail'})" for s in facts.syntax if not s.ok]
    # A pass that was a grammar's parse says so, apart from a compile's.
    ok = [s.path for s in facts.syntax if s.ok and s.depth != "parsed"]
    parsed = [s.path for s in facts.syntax if s.ok and s.depth == "parsed"]
    if not broken and not ok and not parsed:
        return "no changed file was checked"
    parts = [f"broken in {'; '.join(broken)}"] if broken else []
    if ok:
        parts.append(f"ok in {', '.join(ok)}")
    if parsed:
        parts.append(f"parsed in {', '.join(parsed)}")
    return "; ".join(parts)


LAST_ATTEMPT = "## What the last attempt learned"


def last_attempt(
    task: str,
    gone: str,
    why: str | None,
    stopped: str,
    facts: TaskFactsP,
    put_back: bool | Sequence[str],
) -> str:
    """What a failed attempt learned, from its last report's facts: what it
    changed and whether that is still in the tree, which tests failed, and why it stopped.
    `gone` is how it ended (`dropped`, `re-cut`), `why` the reviewer's reason, `put_back` every
    file (`True`), those named, or none."""
    head = f"{task} was {gone}" + (f": {why.strip()}" if why and why.strip() else ".")
    changed = ", ".join(f.path for f in facts.files) or "nothing"
    if put_back is True:
        state = "all of it was put back"
    elif put_back:
        state = f"{', '.join(put_back)} put back, the rest kept in the tree"
    else:
        state = "it is still in the tree"
    failing = sorted({t for c in facts.commands for t in c.failing})
    tested = [c for c in facts.commands if c.test_outcome is not None]
    if failing:
        tests = f"failing tests: {', '.join(failing)}"
    elif tested:
        tests = f"its last test run: {tested[-1].test_outcome}"
    else:
        tests = "it ran no tests"
    rows = [
        LAST_ATTEMPT,
        head,
        f"- it changed: {changed}; {state}",
        f"- {tests}",
        f"- it stopped because: {stopped}",
    ]
    return "\n".join(rows)


def _ran(c: CommandFactP) -> str:
    if c.exit_code is None:
        return "did not run"
    said = f"exit {c.exit_code}" + (", timed out" if c.timed_out else "")
    if c.test_outcome is not None:
        said += f" (tests {c.test_outcome}"
        said += f": {', '.join(c.failing)})" if c.failing else ")"
    return said
