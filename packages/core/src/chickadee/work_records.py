"""The work folder's harness-owned files (Amendment 1 A8; Amendment 2
B1-B4): the work's `README.md` and `report.md`, projected from the run's report, and the index
`<work.dir>/README.md`, one row appended per landed work.

Nothing here is written by a model, and nothing is read from a clock: each projection is a
function of what the journal holds, so rebuilding it from the same journal gives the same bytes.
Status lives only in these files (A8); the design's and the plan's own files are the models'
and the person's.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from chickadee.box.commit import ran_line
from chickadee.box.report import CoderReport, ReportedTask
from chickadee.durable.files import unlink_durably, write_atomic
from chickadee.schema.ids import TASK_KEY_RE
from chickadee.work import MARKER, marked, sections, title_of

__all__ = [
    "INDEX",
    "INDEX_HEAD",
    "REPORT",
    "goal_of",
    "index_row",
    "indexed",
    "keep_private",
    "missing_rows",
    "records",
    "report_text",
    "task_count",
    "with_rows",
    "work_readme_text",
]

INDEX: Final = "README.md"
"""The index's name under the work root."""
REPORT: Final = "report.md"

INDEX_HEAD: Final = (
    f"{MARKER}\n# Work\n\nEach piece of work that landed, one row each, appended and never "
    "edited.\n\n"
)
"""The index's first lines when the harness makes it (B4: marked as chickadee's)."""

_ROW: Final = re.compile(r"^- \S+ · \[[^\]]*\]\(([^)/]+)/\) · ")
_TASK: Final = re.compile(rf"^- {TASK_KEY_RE.pattern[1:-1]}: ")
"""A task's row (`_task_row`): its key, as the journal's grammar has it."""


def _task_row(task: ReportedTask) -> str:
    commit = f" · {task.commit[:7]}" if task.commit else ""
    return f"- {task.id}: {task.title} · {task.state}{commit}"


def _task_rows(report: CoderReport) -> list[str]:
    return [_task_row(t) for t in report.tasks] or ["- none delegated"]


def work_readme_text(name: str, title: str, report: CoderReport) -> str:
    """The work's own README (decision 5): its title, how its run ended, each task's state and
    commit, and links to its parts. A projection of `report`, never edited by a model."""
    return "\n".join(
        (
            MARKER,
            f"# {title}",
            "",
            f"`{name}` · {report.outcome} · run {report.run_id}",
            "",
            "## Tasks",
            *_task_rows(report),
            "",
            "## Parts",
            "- [design](design/README.md)",
            "- [plan](plan/README.md)",
            f"- [report]({REPORT})",
            "",
        )
    )


def report_text(report: CoderReport) -> str:
    """`report.md` (decision 5): how the run ended and why, each task as it was left with what
    it ran, and the team's own report verbatim under its own heading, so it is never read as
    the harness's."""
    lines = [MARKER, "# Report", "", f"Outcome: {report.outcome}"]
    if report.why:
        lines.append(f"Why: {report.why}")
    if report.growth:
        lines.append(f"Growth: {report.growth}")
    lines += ["", "## Tasks", *_task_rows(report)]
    for task in report.tasks:
        commands = task.facts.commands if task.facts else ()
        ran = [c for c in commands if c.exit_code is not None]
        if ran:
            lines += ["", f"### {task.id} ran", *(f"- {ran_line(c)}" for c in ran)]
    if report.athena:
        lines += ["", "## The team's report", "", report.athena.strip()]
    return "\n".join(lines) + "\n"


def records(root: Path, folder: str, name: str, report: CoderReport) -> dict[str, str]:
    """The work's README and `report.md` under `folder` of `root`, by path from `root`, as its
    run ends; the caller writes them through the edit port. The title is kept from the README
    as it stands: Zeus named the work there."""
    readme = root / folder / "README.md"
    title = title_of(readme.read_text(encoding="utf-8") if readme.is_file() else "") or name
    return {
        f"{folder}/README.md": work_readme_text(name, title, report),
        f"{folder}/{REPORT}": report_text(report),
    }


def keep_private(folder: Path, into: Path) -> None:
    """B7, `work.commit` off: the work folder's files copied under `into`, replacing what an
    earlier run of the same work left there: a file the folder no longer has is removed."""
    kept = {p.relative_to(folder) for p in folder.rglob("*") if p.is_file()}
    if into.is_dir():
        for path in sorted(into.rglob("*")):
            if path.is_file() and path.relative_to(into) not in kept:
                unlink_durably(path)
    for name in sorted(kept):
        write_atomic(into / name, (folder / name).read_bytes())


# -- the index (B1-B3) ----------------------------------------------------------------------


def index_row(name: str, goal: str, tasks: int, commit: str) -> str:
    """One landed work's row (B2): facts only. The goal is the design's own `Goal`, never a
    model's words about the work; the date is the folder's."""
    one = " ".join(goal.split()) or name
    count = f"{tasks} task{'' if tasks == 1 else 's'}"
    return f"- {name[:10]} · [{name}]({name}/) · {one} · {count} · landed {commit[:7]}"


def indexed(index: str) -> frozenset[str]:
    """The works the index has a row for, by folder name."""
    return frozenset(m.group(1) for line in index.splitlines() if (m := _ROW.match(line)))


def with_rows(index: str | None, rows: Sequence[str]) -> str:
    """The index with `rows` appended (B1): made, marked, when there is none. A work already in
    it gets no second row, and no row is ever edited."""
    text = INDEX_HEAD if index is None else index
    have = set(indexed(text))
    for row in rows:
        found = _ROW.match(row)
        if found is None or found.group(1) in have:
            continue
        have.add(found.group(1))
        text = (text if text.endswith("\n") else text + "\n") + row + "\n"
    return text


def missing_rows(
    work_root: Path, index: str | None, landed: Callable[[str], str | None]
) -> list[str]:
    """A row for each landed work under `work_root` the index lacks, oldest first (B1, B3).
    `landed(name)` is the commit that brought the work's `report.md` onto the branch, or None
    while it has none (in place, uncommitted): not landed. The task count is the work README's
    own list. Only a folder the harness made, its README marked (B4): a folder of the
    person's beside the works is never given a row."""
    if not work_root.is_dir():
        return []
    have = indexed(index or "")
    rows: list[str] = []
    for folder in sorted(p for p in work_root.iterdir() if p.is_dir()):
        if folder.name in have or not (folder / REPORT).is_file():
            continue
        if not marked(folder / "README.md"):
            continue
        commit = landed(folder.name)
        if not commit:
            continue
        goal = _goal_at(folder / "design" / "README.md")
        rows.append(index_row(folder.name, goal, _task_count(folder / "README.md"), commit))
    return rows


def goal_of(design_readme: str) -> str:
    """The design's `Goal`, flattened to one line: what a row says of the work (B2)."""
    return " ".join(sections(design_readme).get("goal", "").split())


def task_count(readme: str) -> int:
    """How many tasks the work's README lists."""
    listed = sections(readme).get("tasks", "")
    return sum(1 for line in listed.splitlines() if _TASK.match(line))


def _goal_at(path: Path) -> str:
    return goal_of(path.read_text(encoding="utf-8")) if path.is_file() else ""


def _task_count(path: Path) -> int:
    return task_count(path.read_text(encoding="utf-8")) if path.is_file() else 0
