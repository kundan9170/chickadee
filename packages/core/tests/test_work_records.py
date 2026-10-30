"""The work folder's harness-owned records (B1-B4): the work's README and
`report.md` are projections, the same bytes from the same report; the index on the person's
branch gets one row per landed work, appended and never edited; two works landed from two
branches leave both rows; a row is written once; a work whose report is not committed is not
landed. Nothing writes the index as a session starts: `record_works` runs at
landing alone."""

from __future__ import annotations

from pathlib import Path

from repos import git, init_repo

from chickadee.box.report import CoderReport, ReportedTask, ReportHandles, stored_handle
from chickadee.handoff import record_works
from chickadee.journal.projection import RunSpend
from chickadee.schema.journal import CommandFactP, TaskFactsP
from chickadee.work import MARKER
from chickadee.work_records import (
    INDEX_HEAD,
    index_row,
    keep_private,
    records,
    report_text,
    with_rows,
    work_readme_text,
)

DIR = "docs/chickadee"


def _report() -> CoderReport:
    ran = CommandFactP(
        operation_id="s1", command="pytest -q", exit_code=0, timed_out=False, tail=""
    )
    never = CommandFactP(operation_id="s2", command="rm -rf /", timed_out=False, tail="")
    task = ReportedTask(
        id="T1",
        title="Return 2",
        by="worker",
        state="committed",
        rounds=1,
        commit="9b54b67aa0",
        facts=TaskFactsP(start_tree="a", end_tree="b", commands=(ran, never)),
    )
    return CoderReport(
        outcome="COMPLETED",
        run_id="r1",
        tasks=(task,),
        athena="Done: `go()` returns 2.",
        handles=ReportHandles(journal=stored_handle("journal", "r1", b"{}")),
        cost=RunSpend(),
    )


def test_the_work_readme_and_report_are_projections_of_the_report() -> None:
    report = _report()
    assert work_readme_text("2026-10-18-return-2", "Return 2", report) == (
        f"{MARKER}\n# Return 2\n\n`2026-10-18-return-2` · COMPLETED · run r1\n\n## Tasks\n"
        "- T1: Return 2 · committed · 9b54b67\n\n## Parts\n- [design](design/README.md)\n"
        "- [plan](plan/README.md)\n- [report](report.md)\n"
    )
    assert report_text(report) == (
        f"{MARKER}\n# Report\n\nOutcome: COMPLETED\n\n## Tasks\n"
        "- T1: Return 2 · committed · 9b54b67\n\n### T1 ran\n- `pytest -q` (exit 0)\n\n"
        "## The team's report\n\nDone: `go()` returns 2.\n"
    ), "a command that never ran is not listed"


def test_rebuilt_from_the_same_report_the_records_are_byte_identical(
    tmp_path: Path,
) -> None:
    folder = f"{DIR}/2026-10-18-return-2"
    (tmp_path / folder).mkdir(parents=True)
    (tmp_path / folder / "README.md").write_text(f"{MARKER}\n# Return 2\n\nopen\n")
    first = records(tmp_path, folder, "2026-10-18-return-2", _report())
    for path, text in first.items():
        (tmp_path / path).write_text(text)
    again = records(tmp_path, folder, "2026-10-18-return-2", _report())
    assert first == again, "rebuilt over its own output, the same bytes"
    assert "# Return 2\n" in first[f"{folder}/README.md"], "the title Zeus gave is kept"


def test_a_row_is_appended_once_and_never_edited() -> None:
    row = index_row("2026-10-18-return-2", "`go()`\nreturns 2", 1, "9b54b67aa0")
    assert row == (
        "- 2026-10-18 · [2026-10-18-return-2](2026-10-18-return-2/) · `go()` returns 2 · "
        "1 task · landed 9b54b67"
    )
    once = with_rows(None, [row])
    assert once == INDEX_HEAD + row + "\n" and once.startswith(MARKER)
    other = index_row("2026-10-18-return-2", "something else", 3, "0000000")
    assert with_rows(once, [other]) == once


DESIGN = "# Design\n\n## Goal\n{goal}\n"


def _work(root: Path, name: str, goal: str, *, report: bool = True) -> None:
    folder = root / DIR / name
    (folder / "design").mkdir(parents=True)
    (folder / "design" / "README.md").write_text(DESIGN.format(goal=goal))
    (folder / "README.md").write_text(
        f"{MARKER}\n# {name}\n\n## Tasks\n- T1: one · committed\n- T2: two · committed\n"
    )
    if report:
        (folder / "report.md").write_text(f"{MARKER}\n# Report\n")


def _commit(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD").strip()


def _index(root: Path) -> str:
    return (root / DIR / "README.md").read_text()


def test_two_works_landed_from_two_branches_leave_both_rows(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    git(root, "checkout", "-q", "-b", "a")
    _work(root, "2026-10-18-a", "the first")
    a = _commit(root, "work a")
    git(root, "checkout", "-q", "main")
    git(root, "checkout", "-q", "-b", "b")
    _work(root, "2026-10-18-b", "the second")
    b = _commit(root, "work b")
    git(root, "checkout", "-q", "main")

    git(root, "merge", "-q", "--ff-only", "a")
    assert record_works(root, DIR) == ("2026-10-18-a",)
    assert git(root, "log", "-1", "--format=%s").strip() == "chickadee: record 2026-10-18-a"
    first = _index(root)
    git(root, "merge", "-q", "--no-edit", "b")
    assert record_works(root, DIR) == ("2026-10-18-b",)
    both = _index(root)
    assert both.startswith(first), "the first row is never rewritten"
    assert f"the first · 2 tasks · landed {a[:7]}" in both
    assert f"the second · 2 tasks · landed {b[:7]}" in both
    assert git(root, "status", "--porcelain").strip() == ""


def test_a_landed_works_row_is_written_once(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    _work(root, "2026-10-18-a", "the first")
    _commit(root, "merged by hand")
    assert record_works(root, DIR) == ("2026-10-18-a",)
    assert record_works(root, DIR) == (), "a row is written once"


def test_a_work_whose_report_is_not_committed_is_not_landed(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    _work(root, "2026-10-18-a", "the first", report=False)
    _commit(root, "the design only")
    (root / DIR / "2026-10-18-a" / "report.md").write_text("in place, not committed\n")
    assert record_works(root, DIR) == ()
    assert not (root / DIR / "README.md").exists()


def test_a_private_copy_drops_a_file_the_folder_no_longer_has(tmp_path: Path) -> None:
    """The review of #607: a later run of the same work replaces the copy, never adds to it."""
    folder, into = tmp_path / "work", tmp_path / "private"
    (folder / "plan").mkdir(parents=True)
    (folder / "plan" / "old.md").write_text("old\n")
    (folder / "README.md").write_text("one\n")
    keep_private(folder, into)
    (folder / "plan" / "old.md").unlink()
    (folder / "README.md").write_text("two\n")
    keep_private(folder, into)
    assert sorted(p.relative_to(into).as_posix() for p in into.rglob("*") if p.is_file()) == [
        "README.md"
    ]
    assert (into / "README.md").read_text() == "two\n"


def test_a_row_names_the_commit_that_added_the_report_not_a_later_edit(tmp_path: Path) -> None:
    root = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    _work(root, "2026-10-18-a", "the first")
    added = _commit(root, "work a")
    (root / DIR / "2026-10-18-a" / "report.md").write_text(f"{MARKER}\n# Report, fixed\n")
    _commit(root, "a typo")
    assert record_works(root, DIR) == ("2026-10-18-a",)
    assert _index(root).endswith(f"landed {added[:7]}\n")


def test_the_persons_own_index_is_never_written_over(tmp_path: Path) -> None:
    """A `README.md` of the person's in the work folder, unmarked, is
    theirs. A landed work gets no row in it, and the file is left as it was."""
    root = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    (root / DIR).mkdir(parents=True)
    (root / DIR / "README.md").write_text("# My own notes on these docs\n")
    _work(root, "2026-10-18-a", "the first")
    head = _commit(root, "landed, beside the person's README")
    assert record_works(root, DIR) == ()
    assert _index(root) == "# My own notes on these docs\n"
    assert git(root, "rev-parse", "HEAD").strip() == head, "nothing committed"


def test_a_folder_of_the_persons_with_a_report_gets_no_row(tmp_path: Path) -> None:
    """Only a folder the harness made, its README marked, is a work."""
    root = init_repo(tmp_path / "repo", {"app.py": "x = 1\n"})
    mine = root / DIR / "my-old-notes"
    mine.mkdir(parents=True)
    (mine / "README.md").write_text("# my notes\n")
    (mine / "report.md").write_text("# my report\n")
    _commit(root, "the person's folder")
    assert record_works(root, DIR) == ()
    assert not (root / DIR / "README.md").exists()
