"""Task records: what each committed task changed, as history for the next task's
"done so far"; and the symbol table the facts diff a task's symbols against.

Built by the harness from a task's facts, which were measured from the trees and the graph; no
model writes any field. A record says what happened, which stays true; a view of a file stops
being true at the next commit. So a record, never a view, is what one task hands the next.

The packet records and their session-log lines are parked as they stood:
the task loop's records are a projection of the run journal's task lines, never a line of
their own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from chickadee.schema.journal import FileFactP, TaskFactsP

if TYPE_CHECKING:
    from chickadee.graph.queries import SymbolInfo

__all__ = [
    "SymbolGraph",
    "SymbolTable",
    "TaskRecord",
    "field_row",
    "flat",
    "render_task_full",
    "render_task_line",
    "symbol_table",
    "task_record",
]

_SHORT_COMMIT: Final = 7


# -- the symbol table --------------------------------------------------------------------------


class SymbolGraph(Protocol):
    """The graph reads a symbol table takes. Positional-only: an
    implementation's parameter names are not the contract."""

    def symbols_in(self, path: str, /) -> tuple[SymbolInfo, ...]: ...
    def public_contracts(self, paths: Sequence[str], /) -> tuple[SymbolInfo, ...]: ...
    def file_known(self, path: str, /) -> bool: ...


@dataclass(frozen=True)
class SymbolTable:
    """Each path's declarations at one moment, and which of them were public then.

    Captured at a task's delegation and again when its round reports: the one fact about a task
    that is not already journaled, and that cannot be taken later — later, the graph describes
    a newer tree."""

    symbols: Mapping[str, tuple[SymbolInfo, ...]]
    """Indexed paths only: a path absent here was not indexed when the table was taken."""
    public: frozenset[str]


def symbol_table(graph: SymbolGraph, paths: Sequence[str]) -> SymbolTable:
    indexed = [p for p in dict.fromkeys(paths) if graph.file_known(p)]
    return SymbolTable(
        symbols={p: graph.symbols_in(p) for p in indexed},
        public=frozenset(s.qualified_name for s in graph.public_contracts(indexed)),
    )


@dataclass(frozen=True)
class TaskRecord:
    """What a committed task changed, for the next task's "done so far": from its facts, which
    were measured, never from its report. History, so it stays true however far the tree has
    moved."""

    id: str
    title: str
    files: tuple[FileFactP, ...] = ()
    tests_added: tuple[str, ...] = ()
    symbols_added: tuple[str, ...] = ()
    symbols_removed: tuple[str, ...] = ()
    commit: str | None = None


def task_record(
    task: str, title: str, facts: TaskFactsP | None, commit: str | None
) -> TaskRecord:
    """A task's record from its last round's facts; a task with none changed nothing known."""
    if facts is None:
        return TaskRecord(task, title, commit=commit)
    return TaskRecord(
        id=task,
        title=title,
        files=facts.files,
        tests_added=facts.tests_added,
        symbols_added=facts.symbols_added,
        symbols_removed=facts.symbols_removed,
        commit=commit,
    )


def render_task_full(record: TaskRecord) -> str:
    """The task's title, then what it changed: files with their counts, tests added, symbols."""
    head = f"[{record.id}] {flat(record.title)}"
    if record.commit is not None:
        head += f" · commit {record.commit[:_SHORT_COMMIT]}"
    rows = [head]
    if record.files:
        shown = "  ".join(
            f"{flat(f.path)} {f.kind} +{f.added} -{f.removed}" for f in record.files
        )
        rows.append(field_row("files", shown))
    if record.tests_added:
        rows.append(field_row("tests", " ".join(flat(t) for t in record.tests_added)))
    if record.symbols_added:
        rows.append(field_row("added", " ".join(record.symbols_added)))
    if record.symbols_removed:
        rows.append(field_row("removed", " ".join(record.symbols_removed)))
    return "\n".join(rows)


def render_task_line(record: TaskRecord) -> str:
    """One line: the task, and the files it changed."""
    files = " ".join(flat(f.path) for f in record.files)
    return f"[{record.id}] ✓ {flat(record.title)}" + (f" · {files}" if files else "")


def field_row(label: str, text: str) -> str:
    return f"  {label:<9} {text}".rstrip()


def flat(text: str) -> str:
    return " ".join(text.split())
