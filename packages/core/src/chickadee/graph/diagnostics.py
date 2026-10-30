"""What a language server said about each file: the one home of the `diagnostics` and `analyzed`
tables, their writes and the
read every reader is given.

A verdict is about bytes and goes stale when they change, so every row carries the sha256 it
judged, and a verdict a dependency's change has outdated carries that file (`stale_by`). The
read never contacts a server: it is one indexed SELECT. Coverage
defaults to `UNAVAILABLE`, because the dangerous mistake runs one way: no row is not "clean".
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

__all__ = [
    "DiagCoverage",
    "DiagFinding",
    "Diagnostic",
    "DiagnosticSet",
    "Severity",
    "SinceStart",
    "Verdict",
    "clear",
    "diagnostics_in",
    "forget",
    "judged_any",
    "keep",
    "keep_found",
    "mark_stale",
    "record",
    "record_rechecked",
    "since_start",
    "unkept",
]

Severity = Literal["error", "warning", "hint"]


class DiagCoverage(StrEnum):
    TOTAL = "total"
    """A server finished judging these bytes, and nothing they depend on has changed since."""
    STALE = "stale"
    """It judged older bytes, or a file these depend on has changed since."""
    UNAVAILABLE = "unavailable"
    """No server judged the file: none for its language, it failed, or it never finished."""


@dataclass(frozen=True)
class Diagnostic:
    line: int
    """1-based, as the file's lines are numbered everywhere else."""
    col: int | None
    """1-based, in the server's UTF-16 units; None when it named no column."""
    severity: Severity
    code: str | None
    message: str
    source: str


@dataclass(frozen=True)
class DiagnosticSet:
    """`context/diagnostics.DiagnosticSetLike`: what `diagnostics_in` returns."""

    in_span: tuple[Diagnostic, ...] = ()
    out_of_span: tuple[Diagnostic, ...] = ()
    coverage: DiagCoverage = DiagCoverage.UNAVAILABLE
    analyzed_sha: str | None = None
    stale_by: str | None = None
    """The file whose change made this verdict stale, when one did."""


@dataclass(frozen=True)
class Verdict:
    """One file as a server finished judging it: the bytes it was sent, by their sha256, and
    every problem it published for them, none when it found none."""

    path: str
    sha256: str
    source: str
    diagnostics: tuple[Diagnostic, ...] = ()
    local: bool = False
    """From a source that judges the file alone (a linter): no other file's change
    outdates it."""


def record(conn: sqlite3.Connection, verdicts: Iterable[Verdict]) -> None:
    """The capture: each file's previous verdict replaced whole, then its `analyzed` row,
    which is what tells "found nothing" from "never judged". A fresh verdict clears its stale
    mark. Inside the caller's transaction."""
    for verdict in verdicts:
        # Its own source's rows only: a linter's verdict never erases a language server's.
        conn.execute(
            "DELETE FROM diagnostics WHERE path = ? AND server = ?",
            (verdict.path, verdict.source),
        )
        conn.executemany(
            "INSERT INTO diagnostics(path, line, col, severity, code, message, source, sha256, "
            "server) VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    verdict.path,
                    d.line,
                    d.col,
                    d.severity,
                    d.code,
                    d.message,
                    d.source,
                    verdict.sha256,
                    verdict.source,
                )
                for d in verdict.diagnostics
            ],
        )
        conn.execute(
            "INSERT INTO analyzed(path, sha256, source, stale_by, stale_gen, local) "
            "VALUES(?,?,?,NULL,NULL,?) "
            "ON CONFLICT(path, source) DO UPDATE SET sha256=excluded.sha256, "
            "stale_by=NULL, stale_gen=NULL, local=excluded.local",
            (verdict.path, verdict.sha256, verdict.source, int(verdict.local)),
        )


def mark_stale(conn: sqlite3.Connection, by: Mapping[str, str], generation: int) -> list[str]:
    """Each judged file in `by` marked stale by the file it maps to, at the
    index's `generation`: what a re-check asked earlier cannot clear (A2). A file already
    marked keeps the change that marked it first, still true and the earlier one, and takes the
    newer generation. Returns the files marked, in `by`'s order (nearest first)."""
    judged = {
        str(r[0])
        for r in conn.execute(
            "SELECT DISTINCT a.path FROM analyzed a JOIN json_each(?) j ON j.value = a.path "
            "WHERE a.local = 0",
            (json.dumps(list(by)),),
        )
    }
    marked = [path for path in by if path in judged]
    # A linter's verdict is the file's own: another file's change never outdates it.
    conn.executemany(
        "UPDATE analyzed SET stale_by = COALESCE(stale_by, ?), stale_gen = ? "
        "WHERE path = ? AND local = 0",
        [(by[path], generation, path) for path in marked],
    )
    return marked


def record_rechecked(conn: sqlite3.Connection, verdict: Verdict, asked_at: int) -> bool:
    """A re-check's verdict (A2), asked at generation `asked_at`, written in its own
    transaction unless a change at a later generation has marked the file since: that mark
    stands until a verdict asked after it. Whether it was written."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT stale_gen FROM analyzed WHERE path = ? AND source = ?",
            (verdict.path, verdict.source),
        ).fetchone()
        later = row is not None and row[0] is not None and int(row[0]) > asked_at
        if not later:
            record(conn, (verdict,))
        conn.commit()
        return not later
    except BaseException:
        conn.rollback()
        raise


def forget(conn: sqlite3.Connection, paths: Iterable[str]) -> None:
    """The verdicts of files gone from the index (deleted, or denied): nothing reads them."""
    gone = [(path,) for path in paths]
    conn.executemany("DELETE FROM diagnostics WHERE path = ?", gone)
    conn.executemany("DELETE FROM analyzed WHERE path = ?", gone)


def clear(conn: sqlite3.Connection) -> None:
    """Every verdict, for a full index: it judges the repository afresh."""
    conn.execute("DELETE FROM diagnostics")
    conn.execute("DELETE FROM analyzed")


def judged_any(conn: sqlite3.Connection) -> bool:
    """Whether any file has a verdict: with none, there is nothing a change could outdate."""
    return conn.execute("SELECT 1 FROM analyzed LIMIT 1").fetchone() is not None


def diagnostics_in(
    conn: sqlite3.Connection, path: str, lines: range | None = None
) -> DiagnosticSet:
    """The read. `lines` partitions rather than filters: a problem elsewhere in the file is
    counted, never dropped. With no `analyzed` row the answer is `UNAVAILABLE`, whatever
    `diagnostics` holds; with one, `TOTAL` only when it judged the file's indexed bytes and no
    dependency has changed since."""
    judged = conn.execute(
        "SELECT a.sha256 AS sha, a.stale_by AS stale_by, f.sha256 AS current FROM analyzed a "
        "LEFT JOIN files f ON f.path = a.path WHERE a.path = ? ORDER BY a.source",
        (path,),
    ).fetchall()
    if not judged:
        return DiagnosticSet()
    # Every source's rows; the file's coverage is its weakest source's.
    stale = [r for r in judged if r["current"] != str(r["sha"]) or r["stale_by"] is not None]
    shown = stale[0] if stale else judged[0]
    analyzed_sha, stale_by = str(shown["sha"]), shown["stale_by"]
    fresh = not stale
    rows = conn.execute(
        "SELECT line, col, severity, code, message, source FROM diagnostics "
        "WHERE path = ? ORDER BY line, col",
        (path,),
    ).fetchall()
    found = [_diagnostic(r) for r in rows]
    return DiagnosticSet(
        in_span=tuple(d for d in found if lines is None or d.line in lines),
        out_of_span=tuple(d for d in found if lines is not None and d.line not in lines),
        coverage=DiagCoverage.TOTAL if fresh else DiagCoverage.STALE,
        analyzed_sha=analyzed_sha,
        stale_by=None if stale_by is None else str(stale_by),
    )


def _diagnostic(row: sqlite3.Row) -> Diagnostic:
    return Diagnostic(
        line=int(row["line"]),
        col=None if row["col"] is None else int(row["col"]),
        severity=_severity(str(row["severity"])),
        code=None if row["code"] is None else str(row["code"]),
        message=str(row["message"]),
        source=str(row["source"]),
    )


def _severity(value: str) -> Severity:
    if value == "error" or value == "warning":
        return value
    return "hint"


# -- a task's start, kept ----------------------------------------------------------


def keep(conn: sqlite3.Connection, task: str, expected: Mapping[str, str | None]) -> None:
    """Each file of `expected` as it stood before the task's pass replaces its verdict: its rows
    copied aside the first time, `judged` only when its verdict was fresh for the bytes it had
    (`expected` maps it to their sha256, None when it had none). A file kept before keeps its
    first. Inside the caller's transaction, before the pass writes."""
    for path, sha in expected.items():
        if conn.execute(
            "SELECT 1 FROM baseline_files WHERE task = ? AND path = ?", (task, path)
        ).fetchone():
            continue
        judged = sha is not None and _fresh(conn, path, sha)
        conn.execute(
            "INSERT INTO baseline_files(task, path, judged) VALUES(?,?,?)",
            (task, path, int(judged)),
        )
        if judged:
            conn.execute(
                "INSERT INTO baseline_diags(task, path, line, severity, code, message, source) "
                "SELECT ?, path, line, severity, code, message, source FROM diagnostics "
                "WHERE path = ?",
                (task, path),
            )


def unkept(
    conn: sqlite3.Connection, task: str, previous: Mapping[str, str | None]
) -> list[str]:
    """The files of `previous` not yet kept for `task` whose verdict was not fresh for the
    bytes they had (each mapped to their sha256): those whose start must be asked."""
    out: list[str] = []
    for path, sha in previous.items():
        if conn.execute(
            "SELECT 1 FROM baseline_files WHERE task = ? AND path = ?", (task, path)
        ).fetchone():
            continue
        if sha is None or not _fresh(conn, path, sha):
            out.append(path)
    return out


def keep_found(
    conn: sqlite3.Connection, task: str, path: str, found: Iterable[Diagnostic]
) -> None:
    """A file's start as a server judged it apart from the index (`lsp.CheckerResolver.judge`),
    or found empty: a file the task created had nothing at its start."""
    conn.execute(
        "INSERT OR IGNORE INTO baseline_files(task, path, judged) VALUES(?,?,1)", (task, path)
    )
    conn.executemany(
        "INSERT INTO baseline_diags(task, path, line, severity, code, message, source) "
        "VALUES(?,?,?,?,?,?,?)",
        [(task, path, d.line, d.severity, d.code, d.message, d.source) for d in found],
    )


def _fresh(conn: sqlite3.Connection, path: str, sha: str) -> bool:
    """Whether every source that judged `path` judged these bytes and is not outdated, and
    one did."""
    rows = conn.execute(
        "SELECT sha256, stale_by FROM analyzed WHERE path = ?", (path,)
    ).fetchall()
    return bool(rows) and all(str(r[0]) == sha and r[1] is None for r in rows)


@dataclass(frozen=True)
class DiagFinding:
    """One problem, as the stop reads it."""

    path: str
    line: int
    severity: Severity
    code: str | None
    message: str
    source: str

    def __str__(self) -> str:
        code = f" {self.code}" if self.code else ""
        return f"{self.path}:{self.line} {self.source}{code} {self.message}"


@dataclass(frozen=True)
class SinceStart:
    """A task's files now against their kept start, compared by `(source, code, message)`, so
    a problem that moved a line is not new."""

    new_errors: tuple[DiagFinding, ...] = ()
    new_warnings: tuple[DiagFinding, ...] = ()
    already: tuple[DiagFinding, ...] = ()
    """Errors that were there at the start and still are: never the task's."""
    no_record: tuple[str, ...] = ()
    """Files with no fresh verdict at the start: their problems cannot be told new."""
    unchecked: tuple[str, ...] = ()
    """Files with no fresh verdict now: stale, or never judged since the change."""
    files: tuple[str, ...] = ()
    """Every file kept for the task."""


def since_start(conn: sqlite3.Connection, task: str) -> SinceStart:
    """`task`'s kept files compared now with their start."""
    kept = [
        (str(r[0]), bool(r[1]))
        for r in conn.execute(
            "SELECT path, judged FROM baseline_files WHERE task = ? ORDER BY path", (task,)
        )
    ]
    new_errors: list[DiagFinding] = []
    new_warnings: list[DiagFinding] = []
    already: list[DiagFinding] = []
    no_record: list[str] = []
    unchecked: list[str] = []
    for path, judged in kept:
        now = diagnostics_in(conn, path)
        if now.coverage is not DiagCoverage.TOTAL:
            unchecked.append(path)
            continue
        if not judged:
            no_record.append(path)
            continue
        before = Counter(
            (str(r[0]), r[1], str(r[2]))
            for r in conn.execute(
                "SELECT source, code, message FROM baseline_diags WHERE task = ? AND path = ?",
                (task, path),
            )
        )
        for d in now.in_span:
            found = DiagFinding(path, d.line, d.severity, d.code, d.message, d.source)
            key = (d.source, d.code, d.message)
            if before[key] > 0:
                before[key] -= 1
                if d.severity == "error":
                    already.append(found)
            elif d.severity == "error":
                new_errors.append(found)
            elif d.severity == "warning":
                new_warnings.append(found)
    return SinceStart(
        tuple(new_errors),
        tuple(new_warnings),
        tuple(already),
        tuple(no_record),
        tuple(unchecked),
        tuple(p for p, _ in kept),
    )
