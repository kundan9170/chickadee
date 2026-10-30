"""The plan's ids are unique and stable, not sequential: a piece is listed
after its parent, an approved id keeps its file, and a delegated id never leaves the plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from chickadee.handoffs import SHIPPED
from chickadee.work import plan_errors, plan_tasks

TASK = """# {id}: {title}

## Goal
Do {title}.

## Where to look
- `app.py`.

## Approach
Edit it.

## Don't
- Edit a test to pass.

## Done when
It is done.

## Report
What changed.
"""


def _plan(folder: Path, tasks: list[tuple[str, str, str]]) -> None:
    """`tasks` is `(id, slug, depends)`, in the README's order."""
    (folder / "tasks").mkdir(parents=True, exist_ok=True)
    lines = [
        f"{n}. [{task}: {slug}](tasks/{task}-{slug}.md): the {slug}. Depends on: {depends}"
        for n, (task, slug, depends) in enumerate(tasks, start=1)
    ]
    (folder / "README.md").write_text(
        "# Plan: it\n\n## Tasks\n" + "\n".join(lines) + "\n\n## Final check\nThe suite.\n"
    )
    for task, slug, _ in tasks:
        (folder / "tasks" / f"{task}-{slug}.md").write_text(TASK.format(id=task, title=slug))


def _errors(folder: Path, **history: object) -> list[str]:
    return plan_errors(
        folder,
        (SHIPPED / "plan.md").read_text(),
        (SHIPPED / "brief.md").read_text(),
        **history,  # type: ignore[arg-type]
    )


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    return tmp_path / "plan"


def test_pieces_listed_after_their_parent_pass_and_are_read(folder: Path) -> None:
    _plan(
        folder,
        [
            ("T1", "a", "none"),
            ("T2", "b", "T1"),
            ("T2.1", "c", "T1"),
            ("T2.2", "d", "T2.1"),
            ("T3", "e", "T2.2"),
        ],
    )
    assert _errors(folder) == []
    readme = (folder / "README.md").read_text()
    assert [t for t, _, _ in plan_tasks(readme)] == ["T1", "T2", "T2.1", "T2.2", "T3"]


def test_ids_need_not_be_sequential(folder: Path) -> None:
    """Moved by D241: `T1, T3` was refused as out of order; an id removed while never
    delegated leaves a gap, and the ids after it keep theirs."""
    _plan(folder, [("T1", "a", "none"), ("T3", "c", "T1")])
    assert _errors(folder) == []


def test_a_piece_before_its_parent_or_without_one_is_refused(folder: Path) -> None:
    _plan(folder, [("T1", "a", "none"), ("T2.1", "b", "T1")])
    assert "T2.1 is a piece of T2, which is not listed before it" in "\n".join(_errors(folder))


def test_an_id_listed_twice_is_refused(folder: Path) -> None:
    _plan(folder, [("T1", "a", "none"), ("T1", "b", "none")])
    assert "T1 is listed twice" in "\n".join(_errors(folder))


def test_an_approved_id_keeps_its_file(folder: Path) -> None:
    _plan(folder, [("T1", "renamed", "none")])
    got = "\n".join(_errors(folder, approved={"T1": "tasks/T1-first.md"}))
    assert "T1 was tasks/T1-first.md in an approved plan" in got
    _plan(folder, [("T1", "first", "none")])
    assert _errors(folder, approved={"T1": "tasks/T1-first.md"}) == [
        "plan/tasks/T1-renamed.md is not listed under Tasks in plan/README.md"
    ]


def test_a_delegated_id_never_leaves_the_plan(folder: Path) -> None:
    _plan(folder, [("T2", "b", "none")])
    got = _errors(folder, approved={"T1": "tasks/T1-a.md"}, delegated={"T1"})
    assert "plan/README.md Tasks: T1 was delegated, so it stays in the plan" in got


def test_an_approved_id_never_delegated_may_leave(folder: Path) -> None:
    _plan(folder, [("T2", "b", "none")])
    assert _errors(folder, approved={"T1": "tasks/T1-a.md", "T2": "tasks/T2-b.md"}) == []


def test_lines_are_still_numbered_in_order(folder: Path) -> None:
    _plan(folder, [("T1", "a", "none"), ("T2", "b", "T1")])
    readme = folder / "README.md"
    readme.write_text(readme.read_text().replace("2. [T2", "3. [T2"))
    assert "T2's line is numbered 3, not 2" in "\n".join(_errors(folder))


@pytest.mark.parametrize("bad", ["T0", "T1.0", "T01", "T1.1.1.1.1.1"])
def test_an_id_the_journal_would_refuse_is_refused_by_the_plan(folder: Path, bad: str) -> None:
    """The line's pattern takes any digits; an id `delegate` cannot journal is refused at the
    plan, not after it was approved."""
    _plan(
        folder, [("T1", "a", "none"), (bad, "b", "T1")] if bad != "T0" else [(bad, "a", "none")]
    )
    assert any(f"{bad} is not a task id" in e for e in _errors(folder))
